"""Z-axis chopper tuning using synchronized coordinated G1 moves.

Upstream chopper-autotune leaves stepper_z alone because FORCE_MOVE moves a single motor
at a time (which would rack a multi-motor gantry) and disables motors during standstill.
This module tunes Z safely on multi-motor or single-motor Z printers by:
- NEVER using FORCE_MOVE on Z (always coordinated G1 Z moves across all Z steppers)
- NEVER disabling Z motors during or after a run (gantry never drops)
- Requiring XYZ homing before motion and oscillating strictly inside a safe mid-height band
- Applying identical chopper registers to all Z drivers simultaneously and restoring every
  driver, input shaper, velocity limit, and bed mesh state on every exit path.
"""
from __future__ import annotations

import math
import os
import re
import statistics
from dataclasses import dataclass, replace
from datetime import datetime

from . import tmc
from .collect import (MIN_STEADY_SAMPLES, OVERHEAD_STREAM_SEC, Hardware, PrinterBusy, RunStopped,
                      Screen, ThermalGuard, autotune_advice, autotune_goal, autotune_tag,
                      capture_stream, confirmed, current_registers, driver_config, driver_of,
                      enter_spreadcycle, eta_text, exit_spreadcycle, fit_measure_time,
                      measure_baseline, motor_label, rail_snippet, rail_steppers,
                      refuse_after_shutdown, refuse_autotune_save, refuse_corexz,
                      refuse_if_printing, refuse_nothing_to_try, resolve_accel_chip, round_order,
                      run_restore, set_rail_fields, steady_window, travel_for, untouched_autotune,
                      z_driver_sections, accel_command_chip)
from .current import live_limits
from .dataset import load_json, save_json
from .find_speed import MIN_CRUISE_SEC, cruise_for, find_peaks, recommend, smooth
from .klippy import Klippy, KlippyError, find_socket
from .metrics import transients, vibration_score, window
from .search import descent_budget, heard_score, multi_start_descent, penalized_score

STATE = os.path.expanduser('~/printer_data/config/chopper-autotune/z_axis.json')

DEFAULT_MIN_SPEED = 5
DEFAULT_MAX_SPEED = 30
DEFAULT_STEP = 1
DEFAULT_MEASURE_TIME = 1.5
DEFAULT_SCAN_MEASURE_TIME = 1.0
DEFAULT_TRIM = 0.1
MIN_EDGE_MM = 30.0
EDGE_SHARE = 0.2
MAX_STROKE_MM = 100.0
APPROACH_SPEED = 15.0
VALIDATE_TOP = 3
VALIDATE_EXTRA_ITERATIONS = 2
DEMO_FALLBACK_SPEED = 15.0

STEP_OPTIONS = ('rotation_distance', 'gear_ratio', 'full_steps_per_rotation', 'microsteps')
COMPARE_OPTIONS = ('run_current', 'hold_current', 'stealthchop_threshold', 'coolstep_threshold',
                   'interpolate')


@dataclass(frozen=True)
class ZContext:
    driver: tmc.Driver
    driver_name: str
    steppers: 'list[str]'
    kinematics: str
    max_z_velocity: float
    max_z_accel: float
    max_accel: float
    z_min: float
    z_max: float
    safe_lo: float
    safe_hi: float
    z_center: float
    z_limit: float
    center_xy: 'tuple[float, float]'
    autotune: 'str | None'
    notes: 'tuple[str, ...]' = ()


def save_winner_state(driver_name: str, steppers: 'list[str]', winner: tmc.Chopper,
                      speed: 'float | None' = None, autotune: 'str | None' = None,
                      hearing: tmc.Hearing = tmc.Hearing(), kept: bool = False,
                      improvement: 'float | None' = None):
    """Persist the latest Z winner for CHOPPER_Z SAVE_LAST=1 and CHOPPER_SAVE MOTOR=Z."""
    state = {
        'driver': driver_name,
        'steppers': list(steppers),
        'fields': winner.fields(),
        'kept': bool(kept),
        **hearing.manifest_fields(),
    }
    if speed is not None:
        state['speed'] = speed
    if autotune:
        state['autotune'] = autotune
    if improvement is not None:
        state['improvement'] = improvement
    save_json(STATE, state)


def load_winner_state() -> 'dict | None':
    return load_json(STATE) or None


def _stepper_option(settings: dict, stepper: str, option: str):
    from .rail import step_option
    return step_option(settings, stepper, option)


def _xy_center(settings: dict) -> 'tuple[float, float]':
    centers = []
    for axis in 'xy':
        rail = settings.get('stepper_' + axis) or {}
        if rail.get('position_max') is None:
            centers.append(0.0)
        else:
            lo = float(rail.get('position_min', 0.0))
            hi = float(rail['position_max'])
            centers.append((lo + hi) / 2.0)
    return centers[0], centers[1]


def z_context(settings: dict) -> ZContext:
    """Validate Z hardware geometry, multi-motor uniformity, and safe vertical range."""
    refuse_corexz(settings)
    z_rail = settings.get('stepper_z')
    if z_rail is None or z_rail.get('position_max') is None:
        raise SystemExit('unsupported config: no stepper_z with position_max. Nothing was moved')

    steppers = rail_steppers(settings, 'z')
    models = {stepper: driver_of(settings, stepper) for stepper in steppers}
    missing = [stepper for stepper, model in models.items() if model is None]
    if missing:
        raise SystemExit('%s: no supported TMC section found (%s); every Z driver must match. '
                         'Nothing was moved'
                         % (', '.join(missing), ', '.join('tmc' + name for name in sorted(tmc.DRIVERS))))
    if len(set(models.values())) > 1:
        raise SystemExit('use one TMC driver model on Z (%s): one register set runs different '
                         'choppers on different drivers. Nothing was moved'
                         % ', '.join('%s TMC%s' % item for item in models.items()))

    driver_name = models['stepper_z']
    driver = tmc.DRIVERS[driver_name]

    managed = [stepper for stepper in steppers if autotune_goal(settings, stepper) is not None]
    if managed and len(managed) < len(steppers):
        raise SystemExit('put klipper_tmc_autotune on all Z steppers (%s) or on none: it manages '
                         '%s only, leaving Z on mixed choppers. Nothing was moved'
                         % (', '.join(steppers), ', '.join(managed)))

    for option in STEP_OPTIONS:
        values = {stepper: _stepper_option(settings, stepper, option) for stepper in steppers}
        if len(set(values.values())) > 1:
            raise SystemExit('match %s across all Z steppers (%s): mismatched gearing or steps '
                             'will rack the gantry. Nothing was moved'
                             % (option, ', '.join('%s %s' % (
                                 s, 'unset' if v is None else '%g' % v) for s, v in values.items())))

    notes = []
    sections = {stepper: settings['tmc%s %s' % (driver_name, stepper)] for stepper in steppers}
    for option in COMPARE_OPTIONS:
        vals = {s: str(sec.get(option)) for s, sec in sections.items()}
        if len(set(vals.values())) > 1:
            notes.append('WARNING: Z drivers differ in %s (%s)'
                         % (option, ', '.join('%s=%s' % pair for pair in vals.items())))

    if not managed:
        baselines = {s: tmc.baseline_chopper(driver_config(settings, s)['baseline'],
                                             default=driver.default).label()
                     for s in steppers}
        if len(set(baselines.values())) > 1:
            notes.append('WARNING: Z drivers currently have different config registers (%s)'
                         % ', '.join('%s=%s' % pair for pair in baselines.items()))

    printer = settings.get('printer') or {}
    kinematics = str(printer.get('kinematics') or '')
    max_accel = float(printer.get('max_accel') or 500.0)
    max_z_velocity = float(printer.get('max_z_velocity') or printer.get('max_velocity') or 30.0)
    max_z_accel = float(printer.get('max_z_accel') or max_accel)

    z_min = float(z_rail.get('position_min', 0.0))
    z_max = float(z_rail['position_max'])
    z_span = z_max - z_min
    edge = max(MIN_EDGE_MM, EDGE_SHARE * z_span)
    safe_lo = max(z_min + edge, 40.0 if z_max >= 120.0 else z_min + 15.0)
    safe_hi = z_max - edge
    if safe_hi - safe_lo < 20.0:
        raise SystemExit('not enough safe Z travel (%.1f..%.1f mm inside %.1f..%.1f mm). '
                         'Nothing was moved' % (safe_lo, safe_hi, z_min, z_max))

    z_center = (safe_lo + safe_hi) / 2.0
    z_limit = min(safe_hi - safe_lo, MAX_STROKE_MM)

    return ZContext(
        driver=driver,
        driver_name=driver_name,
        steppers=steppers,
        kinematics=kinematics,
        max_z_velocity=max_z_velocity,
        max_z_accel=max_z_accel,
        max_accel=max_accel,
        z_min=z_min,
        z_max=z_max,
        safe_lo=safe_lo,
        safe_hi=safe_hi,
        z_center=z_center,
        z_limit=z_limit,
        center_xy=_xy_center(settings),
        autotune=autotune_goal(settings, 'stepper_z'),
        notes=tuple(notes),
    )


def build_z_hardware(kl: Klippy, ctx: ZContext, accel: bool = True) -> Hardware:
    settings = kl.settings()
    chip = resolve_accel_chip(settings, 'z', lambda: kl.config_sections()) if accel else ''
    own = driver_config(settings, ctx.steppers[0])
    hw = Hardware(
        kl=kl,
        accel_chip=chip,
        kinematics=ctx.kinematics,
        axis_span=ctx.z_limit,
        center=ctx.center_xy,
        max_accel=ctx.max_z_accel,
        display='display_status' in kl.object_list(),
        measure_chip=accel_command_chip(settings, chip) if chip else '',
        **own,
    )
    hw.twins = [replace(hw, twins=[], **driver_config(settings, twin))
                for twin in ctx.steppers[1:]]
    return hw


def _restore_drive_chopper(kl: Klippy, drive: Hardware):
    if untouched_autotune(drive):
        return
    fields = dict(drive.driver.default.fields(), **drive.baseline)
    kl.gcode(tmc.set_fields_script(drive.stepper, fields))


class ZMotion:
    """Coordinated synchronized Z oscillation without disabling motors or calling FORCE_MOVE."""

    def __init__(self, kl: Klippy, hw: Hardware, ctx: ZContext, z_accel: float):
        self.kl = kl
        self.hw = hw
        self.ctx = ctx
        self.z_accel = z_accel
        self.settings = kl.settings()
        self.limits = {}
        self.guard = None
        self.origin_z = 0.0
        self.start_z = ctx.z_center
        self.moved = False
        self.restores = []
        self.reloads = []

    def _query_z_origin(self) -> float:
        status = self.kl.request('objects/query', {'objects': {'gcode_move': ['homing_origin']}})
        return float(status['status']['gcode_move']['homing_origin'][2])

    def _query_z_shaper(self) -> 'str | None':
        report = '\n'.join(self.kl.gcode_output('SET_INPUT_SHAPER'))
        match = re.search(r'shaper_freq_z:(\S+)', report)
        return match.group(1) if match else None

    def prepare(self):
        refuse_if_printing(self.kl)
        status = self.kl.request('objects/query', {'objects': {
            'toolhead': ['homed_axes', 'position'],
            'gcode_move': ['homing_origin'],
            'bed_mesh': ['profile_name', 'profiles'],
        }})['status']

        homed = str(status['toolhead']['homed_axes'])
        if not set('xyz') <= set(homed):
            raise SystemExit('home all axes (G28), then retry: Z tuning requires homed XYZ so '
                             'bounds stay safe (currently homed: %s). Nothing was moved'
                             % (homed or 'none'))

        if 'quad_gantry_level' in self.settings:
            try:
                qgl = self.kl.request('objects/query', {'objects': {'quad_gantry_level': ['applied']}})
                if not qgl['status'].get('quad_gantry_level', {}).get('applied'):
                    print('note: [quad_gantry_level] is not applied — run QUAD_GANTRY_LEVEL before '
                          'long Z runs for truest motion')
            except KlippyError:
                pass

        self.start_z = float(status['toolhead']['position'][2])
        self.origin_z = float(status['gcode_move']['homing_origin'][2])

        self.guard = ThermalGuard(self.kl, self.settings,
                                  sections=z_driver_sections(self.settings),
                                  release_on_hot=False)
        self.guard.preflight()

        self.limits = live_limits(self.kl)
        self.restores.append(lambda: self.kl.gcode('M204 S%s' % self.limits['max_accel']))
        self.restores.append(lambda: self.kl.gcode(
            'SET_VELOCITY_LIMIT MINIMUM_CRUISE_RATIO=%s' % self.limits['minimum_cruise_ratio']))
        self.kl.gcode('SET_VELOCITY_LIMIT MINIMUM_CRUISE_RATIO=0')

        if self.limits['speed_factor'] != 1:
            self.restores.append(lambda: self.kl.gcode('M220 S%g' % (self.limits['speed_factor'] * 100)))
            self.kl.gcode('M220 S100')

        if 'input_shaper' in self.settings:
            z_freq = self._query_z_shaper()
            if z_freq is not None and float(z_freq) > 0:
                self.restores.append(lambda val=z_freq: self.kl.gcode(
                    'SET_INPUT_SHAPER SHAPER_FREQ_Z=%s' % val))
                self.kl.gcode('SET_INPUT_SHAPER SHAPER_FREQ_Z=0')

        mesh = status.get('bed_mesh') or {}
        mesh_name = mesh.get('profile_name')
        if mesh_name:
            if mesh_name in (mesh.get('profiles') or {}):
                self.reloads.append(lambda name=mesh_name: self.kl.gcode(
                    'BED_MESH_PROFILE LOAD="%s"' % name))
            else:
                print('note: active bed_mesh %s is not a saved profile: cleared for the Z run'
                      % mesh_name)
            self.kl.gcode('BED_MESH_CLEAR')

        approach_feed = min(APPROACH_SPEED, self.ctx.max_z_velocity) * 60.0
        target_gcode = self.ctx.z_center - self.origin_z
        self.kl.gcode('G90\nM204 S%.3f\nG1 Z%.3f F%.1f\nM400'
                      % (self.z_accel, target_gcode, approach_feed))
        self.moved = True

    def move_bounds(self, travel: float, direction: int) -> 'tuple[float, float]':
        """Physical safety check + G-code Z endpoints around z_center."""
        half = travel / 2.0
        phys_start = self.ctx.z_center - direction * half
        phys_end = self.ctx.z_center + direction * half
        if (phys_start < self.ctx.safe_lo - 1e-6 or phys_start > self.ctx.safe_hi + 1e-6
                or phys_end < self.ctx.safe_lo - 1e-6 or phys_end > self.ctx.safe_hi + 1e-6):
            raise RunStopped('Z move %.1f..%.1f mm exceeds safe window %.1f..%.1f mm; stopping'
                             % (min(phys_start, phys_end), max(phys_start, phys_end),
                                self.ctx.safe_lo, self.ctx.safe_hi))
        self.origin_z = self._query_z_origin()
        return phys_start - self.origin_z, phys_end - self.origin_z

    def measure_stroke(self, speed: float, cruise: float, trim: float,
                       direction: int) -> 'tuple[float, int]':
        travel = travel_for(speed, self.z_accel, cruise)
        duration = travel / speed + speed / self.z_accel
        approach_feed = min(APPROACH_SPEED, self.ctx.max_z_velocity) * 60.0

        last_err = None
        for attempt in (1, 2):
            try:
                self.guard.check()
                z_start, z_end = self.move_bounds(travel, direction)
                self.kl.gcode('G90\nM204 S%.3f\nG1 Z%.3f F%.1f\nM400'
                              % (self.z_accel, z_start, approach_feed))
                move_script = 'G90\nM204 S%.3f\nG1 Z%.3f F%.3f' % (
                    self.z_accel, z_end, speed * 60.0)
                t_end, data = capture_stream(self.hw, move_script, duration)
                steady = steady_window(t_end, speed, self.z_accel, cruise, trim)
                sliced = vibration_score(window(data, *steady), 0.0)
                if sliced['samples'] < MIN_STEADY_SAMPLES:
                    raise ValueError('only %d samples in steady window' % sliced['samples'])
                clicks = transients(data)['clicks']
                return sliced['median_magnitude'], clicks
            except (KlippyError, TimeoutError, ValueError, OSError) as err:
                refuse_after_shutdown(err)
                last_err = err
                if attempt == 2:
                    raise
        raise last_err

    def measure_pair(self, speed: float, cruise: float, trim: float,
                     iterations: int = 1) -> 'tuple[list[float], int]':
        mags, total_clicks = [], 0
        for _ in range(iterations):
            for direction in (1, -1):
                mag, clicks = self.measure_stroke(speed, cruise, trim, direction)
                mags.append(mag)
                total_clicks += clicks
        return mags, total_clicks

    def _return_z(self):
        if not self.moved:
            return
        if 'z' not in self.kl.homed_axes():
            return
        safe_park = min(self.ctx.safe_hi, max(self.start_z, max(self.ctx.safe_lo, 20.0)))
        # Prefer original height when inside a safe clearance (>= 20 mm above bed)
        if 20.0 <= self.start_z <= self.ctx.safe_hi:
            safe_park = self.start_z
        origin = self._query_z_origin()
        feed = min(APPROACH_SPEED, self.ctx.max_z_velocity) * 60.0
        self.kl.gcode('G90\nM204 S%.3f\nG1 Z%.3f F%.1f\nM400'
                      % (self.z_accel, safe_park - origin, feed))

    def restore(self):
        """Restore every Z driver's registers/mode and printer settings; never disable Z."""
        kl, drives = self.kl, self.hw.rail
        run_restore(
            *[lambda drive=drive: _restore_drive_chopper(kl, drive) for drive in drives],
            *[lambda drive=drive: exit_spreadcycle(kl, drive) for drive in drives],
            *self.restores,
            self._return_z,
            *self.reloads,
        )


def _baseline_description(hw: Hardware) -> str:
    if hw.autotune is not None:
        return 'klipper_tmc_autotune (%s, read live)' % hw.autotune
    regs = current_registers(hw, hw.driver.default)
    return regs.label() if regs is not None else 'mixed config registers'


def _block_rivals(hw: Hardware, space: tmc.Space, hearing: tmc.Hearing,
                  sweep_tpfd: bool) -> 'tuple[list[tmc.Chopper], str]':
    stock = tmc.stock_chopper(hw.driver, sweep_tpfd)
    current = current_registers(hw, stock)
    why = {None: 'the Z drivers run different registers'}
    for combo in (stock, current):
        if combo is None:
            continue
        if tmc.validate(combo, hw.driver) is not None:
            why[combo] = 'past the tmc%s limits' % hw.driver.name
        elif space.outside(combo):
            why[combo] = space.outside(combo)
        elif hearing.skips(combo, hw.driver):
            why[combo] = 'below %g kHz, SKIP_AUDIBLE' % (hearing.limit_hz / 1000)
    rivals = [combo for combo in dict.fromkeys((stock, current)) if combo not in why]
    return rivals, why.get(current, '')


def _quieter_every_round(rounds: dict, combo: tmc.Chopper, rival: tmc.Chopper,
                         by: str = 'score') -> bool:
    return all(mine[by] < theirs[by] for mine, theirs in zip(rounds[combo], rounds[rival]))


def _round_mean(rounds: dict, combo: tmc.Chopper, key: str = 'magnitude') -> float:
    return statistics.mean(entry[key] for entry in rounds[combo])


def _resolve_z_accel(args, ctx: ZContext) -> float:
    ceiling = min(ctx.max_z_accel, ctx.max_accel)
    if args.accel is None:
        return ceiling
    accel = float(args.accel)
    if accel <= 0:
        raise SystemExit('ACCEL must be positive, got %g. Nothing was moved' % accel)
    if accel > ceiling + 1e-6:
        raise SystemExit('ACCEL=%g exceeds Klipper\'s Z acceleration ceiling (%g mm/s2). '
                         'Lower ACCEL or raise max_z_accel. Nothing was moved' % (accel, ceiling))
    return accel


def _build_scan_plan(args, ctx: ZContext, z_accel: float) -> 'list[tuple[int, float]]':
    cap_speed = min(args.max_speed, int(math.floor(ctx.max_z_velocity + 1e-6)))
    if args.min_speed <= 0:
        raise SystemExit('MIN_SPEED must be positive, got %s. Nothing was moved' % args.min_speed)
    if args.step <= 0:
        raise SystemExit('STEP must be positive, got %s. Nothing was moved' % args.step)
    if args.min_speed > cap_speed:
        raise SystemExit('MIN_SPEED=%d exceeds max_z_velocity %g mm/s. Nothing was moved'
                         % (args.min_speed, ctx.max_z_velocity))
    plan = []
    for v in range(args.min_speed, cap_speed + 1, args.step):
        cruise = cruise_for(float(v), z_accel, ctx.z_limit, args.scan_measure_time)
        if cruise >= MIN_CRUISE_SEC:
            plan.append((v, cruise))
    if not plan:
        raise SystemExit('no Z scan speeds fit inside %.1f mm travel at accel %.0f mm/s2. '
                         'Nothing was moved' % (ctx.z_limit, z_accel))
    return plan


def _demo_registers(hw: Hardware) -> tmc.Chopper:
    tuned = current_registers(hw, hw.driver.default)
    if tuned is None:
        raise SystemExit('give all Z drivers (%s) the same registers before CHOPPER_Z DEMO=1. '
                         'Nothing was moved' % ', '.join(drive.stepper for drive in hw.rail))
    return tuned


def z_show(kl: Klippy, args, ctx: ZContext, z_accel: float) -> int:
    from .demo import gain, round_result
    rounds_count = max(2, int(args.rounds))
    if rounds_count % 2:
        print('note: using even ROUNDS=%d instead of %d so thermal drift cancels'
              % (rounds_count + 1, rounds_count))
        rounds_count += 1

    stored = load_winner_state() or {}
    speed = float(args.speed if args.speed is not None else stored.get('speed') or DEMO_FALLBACK_SPEED)
    if speed <= 0 or speed > ctx.max_z_velocity + 1e-6:
        raise SystemExit('SPEED=%g exceeds Z limits (0..%g mm/s). Nothing was moved'
                         % (speed, ctx.max_z_velocity))

    hw = build_z_hardware(kl, ctx)
    cruise = fit_measure_time([int(round(speed))], z_accel, ctx.z_limit, args.measure_time,
                              keep_limit=True)
    travel = travel_for(speed, z_accel, cruise)
    print('Z demo: Klipper defaults vs live/saved Z registers across %s' % ', '.join(ctx.steppers))
    print('  safe Z band %.1f..%.1f mm (center %.1f), stroke %.1f mm at %g mm/s, accel %.0f mm/s2'
          % (ctx.safe_lo, ctx.safe_hi, ctx.z_center, travel, speed, z_accel))
    print('  %d rounds (defaults vs tuned, UP+DOWN each)' % rounds_count)
    if args.dry_run:
        return 0
    if not confirmed(args):
        return 1

    refuse_if_printing(kl)
    motion = ZMotion(kl, hw, ctx, z_accel)
    kl.subscribe_accel(hw.accel_chip)
    screen = Screen(kl, hw.display)
    motion.prepare()
    try:
        for drive in hw.rail:
            enter_spreadcycle(kl, drive)
        tuned = _demo_registers(hw)
        stock = hw.driver.default
        if tuned == stock:
            raise SystemExit('Z drivers are already on Klipper defaults %s — tune and save Z first. '
                             'Nothing to compare' % stock.label())
        mags = {'defaults': [], 'tuned': []}
        sets = [('defaults', stock.fields()), ('tuned', tuned.fields())]
        for round_no in range(1, rounds_count + 1):
            for label, fields in round_order(sets, round_no - 1):
                set_rail_fields(kl, hw, fields)
                screen.update('Chopper Z %d/%d: %s' % (round_no, rounds_count, label.upper()),
                              force=True, short='Z %d/%d %s' % (round_no, rounds_count, label.upper()))
                pair_mags, _ = motion.measure_pair(speed, cruise, args.trim, iterations=1)
                mean_mag = statistics.mean(pair_mags)
                mags[label].append(mean_mag)
                print('  round %d %s: %.1f' % (round_no, label, mean_mag))
        d = statistics.mean(mags['defaults'])
        t = statistics.mean(mags['tuned'])
        screen.final(round_result('Motor Z', d, t), 'Z %s' % gain(d, t, short=True))
    finally:
        motion.restore()
    return 0


def run_z(args) -> int:
    kl = Klippy(find_socket(args.socket)).connect()
    try:
        return z_tune(kl, args)
    finally:
        kl.close()


def z_tune(kl: Klippy, args) -> int:
    if args.save_last:
        from .analyze import run_save_latest
        return run_save_latest(args, 'Z')

    settings = kl.settings()
    ctx = z_context(settings)
    if args.save:
        refuse_autotune_save(settings, ctx.driver_name, *ctx.steppers)

    hearing = tmc.Hearing.of(args)
    space = tmc.Space.of(args, ctx.driver)
    refuse_nothing_to_try(ctx.driver, space, hearing)
    z_accel = _resolve_z_accel(args, ctx)

    if args.demo:
        return z_show(kl, args, ctx, z_accel)

    hw = build_z_hardware(kl, ctx)
    sweep_tpfd = space.tpfd is not None and ctx.driver.has_tpfd
    stock = tmc.stock_chopper(ctx.driver, sweep_tpfd)
    rivals, left_out = _block_rivals(hw, space, hearing, sweep_tpfd)

    if args.speed is not None:
        speed = float(args.speed)
        if speed <= 0 or speed > ctx.max_z_velocity + 1e-6:
            raise SystemExit('SPEED=%g must be within 0 < SPEED <= max_z_velocity (%g mm/s). '
                             'Nothing was moved' % (speed, ctx.max_z_velocity))
        scan_plan = []
        descent_cruise = fit_measure_time([int(math.ceil(speed))], z_accel, ctx.z_limit,
                                          args.measure_time, keep_limit=True)
    else:
        speed = None
        scan_plan = _build_scan_plan(args, ctx, z_accel)
        descent_cruise = fit_measure_time([scan_plan[-1][0]], z_accel, ctx.z_limit,
                                          args.measure_time, keep_limit=True)

    rounds_count = max(2, int(args.rounds))
    if rounds_count % 2:
        rounds_count += 1

    budget = descent_budget(ctx.driver, space)
    est_scan_sec = sum(2 * (c + 2 * v / z_accel + OVERHEAD_STREAM_SEC) for v, c in scan_plan)
    est_speed = speed if speed is not None else scan_plan[-1][0]
    est_move_sec = descent_cruise + 2 * est_speed / z_accel + OVERHEAD_STREAM_SEC
    est_search_sec = (budget * args.iterations + VALIDATE_TOP * VALIDATE_EXTRA_ITERATIONS
                      + rounds_count * (1 + len(rivals))) * 2 * est_move_sec

    print('Z chopper tune: %d TMC%s drivers (%s), accel chip %s'
          % (len(ctx.steppers), ctx.driver_name, ', '.join(ctx.steppers), hw.accel_chip))
    print('  current config: %s; Z limits max_z_velocity %g mm/s, accel %.0f mm/s2'
          % (_baseline_description(hw), ctx.max_z_velocity, z_accel))
    print('  safe Z band %.1f..%.1f mm (inside %.1f..%.1f mm), center %.1f mm, max stroke %.1f mm'
          % (ctx.safe_lo, ctx.safe_hi, ctx.z_min, ctx.z_max, ctx.z_center, ctx.z_limit))
    print('  coordinated G1 Z moves only (never FORCE_MOVE; Z motors stay enabled throughout)')
    if scan_plan:
        print('  speed scan: %d speeds (%d..%d mm/s, step %d) on Klipper defaults, ~%s'
              % (len(scan_plan), scan_plan[0][0], scan_plan[-1][0], args.step,
                 eta_text(est_scan_sec)))
    else:
        print('  fixed speed: %g mm/s (scan skipped, SPEED given)' % speed)
    print('  register search: %s (up to %d candidates, cruise %.2fs, %d back-to-back rounds), '
          'ETA under %s' % (space.text(), budget, descent_cruise, rounds_count,
                            eta_text(est_scan_sec + est_search_sec)))
    for note in ctx.notes:
        print('  ' + note)

    if args.dry_run:
        return 0
    if not confirmed(args):
        return 1

    refuse_if_printing(kl)
    kl.subscribe_accel(hw.accel_chip)
    screen = Screen(kl, hw.display)
    motion = ZMotion(kl, hw, ctx, z_accel)
    motion.prepare()

    try:
        # Standstill noise floor with Z motors holding
        motion.guard.check()
        _, floor = capture_stream(hw, 'G4 P1000', 1.0, motion.guard.check)
        floor_mag = vibration_score(floor, 0.0)['median_magnitude']
        print('Baseline standstill noise (Z motors holding): %.1f' % floor_mag)

        for drive in hw.rail:
            enter_spreadcycle(kl, drive)

        # Recompute rivals after autotune live registers are populated in hw.rail
        rivals, left_out = _block_rivals(hw, space, hearing, sweep_tpfd)
        current = current_registers(hw, stock)

        if speed is None:
            print('Scanning Z speeds on Klipper-default registers %s...' % stock.label())
            set_rail_fields(kl, hw, stock.fields())
            curve = []
            for idx, (v, scruise) in enumerate(scan_plan, 1):
                pair_mags, _ = motion.measure_pair(float(v), scruise, args.trim, iterations=1)
                med = statistics.mean(pair_mags)
                curve.append((v, med))
                print('  [%d/%d] %2d mm/s: mean %.1f (up %.1f, down %.1f)'
                      % (idx, len(scan_plan), v, med, pair_mags[0], pair_mags[1]))
                screen.update('Chopper Z scan %d/%d (%d mm/s)' % (idx, len(scan_plan), v),
                              short='Z scan %d mm/s' % v)
            peaks = find_peaks(smooth([m for _, m in curve]))
            peak = recommend(curve, peaks)
            loudest = max(curve, key=lambda p: p[1])
            if peak is not None:
                speed = float(peak)
                print('Selected resonant Z speed: %g mm/s (peaks: %s)'
                      % (speed, ', '.join('%d mm/s (%.0f)' % curve[i] for i in peaks)))
            else:
                speed = float(loudest[0])
                print('No isolated Z resonance peak; tuning at loudest Z speed %g mm/s (%.0f)'
                      % (speed, loudest[1]))

        descent_cruise = fit_measure_time([int(math.ceil(speed))], z_accel, ctx.z_limit,
                                          args.measure_time, keep_limit=True)
        print('Multi-start Z descent at %g mm/s (cruise %.2fs, stroke %.1f mm)...'
              % (speed, descent_cruise, travel_for(speed, z_accel, descent_cruise)))

        cache = {}
        history = {}
        clicks_map = {}

        def evaluate(combo: tmc.Chopper) -> float:
            if combo in cache:
                return cache[combo]
            if hearing.skips(combo, ctx.driver):
                cache[combo] = float('inf')
                return cache[combo]
            set_rail_fields(kl, hw, combo.fields())
            mags, combo_clicks = motion.measure_pair(speed, descent_cruise, args.trim,
                                                     iterations=args.iterations)
            history[combo] = list(mags)
            clicks_map[combo] = combo_clicks
            score = penalized_score(combo, mags, ctx.driver, hearing, combo_clicks / len(mags))
            cache[combo] = score
            note = (' audible' if hearing.audible(combo, ctx.driver) else '') \
                + (' clicks %d!' % combo_clicks if combo_clicks else '')
            print('  cand %2d %s -> %.1f%s' % (len(history), combo.label(), score, note))
            screen.update('Chopper Z cand %d of max %d: %.0f' % (len(history), budget, score),
                          short='Z %d/%d %.0f' % (len(history), budget, score))
            return score

        raw_start = current if current is not None else stock
        start = space.nearest(replace(raw_start, tpfd=None if stock.tpfd is None else (
            raw_start.tpfd if raw_start.tpfd is not None else stock.tpfd)), ctx.driver)
        multi_start_descent(ctx.driver, space, start, evaluate, rounds=2)

        top = sorted((c for c in history if cache[c] != float('inf')), key=cache.get)[:VALIDATE_TOP]
        if not top:
            raise SystemExit('no valid Z chopper candidates were measured')

        print('Validating top %d Z candidates (%d extra UP+DOWN iterations each)...'
              % (len(top), VALIDATE_EXTRA_ITERATIONS))
        rescored = {}
        for combo in top:
            set_rail_fields(kl, hw, combo.fields())
            mags, combo_clicks = motion.measure_pair(speed, descent_cruise, args.trim,
                                                     iterations=VALIDATE_EXTRA_ITERATIONS)
            history[combo].extend(mags)
            clicks_map[combo] += combo_clicks
            rescored[combo] = penalized_score(combo, history[combo], ctx.driver, hearing,
                                              clicks_map[combo] / len(history[combo]))
            print('  validate %s: score %.1f (mean %.1f, n=%d)'
                  % (combo.label(), rescored[combo], statistics.mean(history[combo]),
                     len(history[combo])))
        best = min(rescored, key=rescored.get)

        # Fair back-to-back verification against Klipper defaults and current/autotune registers
        combos = list(dict.fromkeys([best] + rivals))
        rounds = {}
        if len(combos) > 1:
            if left_out and current != best:
                print('Current Z registers not in back-to-back (%s)' % left_out)
            print('Measuring %s back to back (%d alternating rounds)...'
                  % (' / '.join(c.label() for c in combos), rounds_count))
            rounds = {c: [] for c in combos}
            for round_idx in range(rounds_count):
                for combo in round_order(combos, round_idx):
                    set_rail_fields(kl, hw, combo.fields())
                    mags, combo_clicks = motion.measure_pair(speed, descent_cruise, args.trim,
                                                             iterations=1)
                    rounds[combo].append({
                        'magnitude': statistics.mean(mags),
                        'clicks': combo_clicks,
                        'score': heard_score(combo, mags, ctx.driver, hearing,
                                             combo_clicks / len(mags)),
                    })

        chosen = current if current in rounds else best
        if rounds:
            beaten = None
            for rival in (best, stock):
                if rival not in rounds or rival in (chosen, beaten):
                    continue
                if _quieter_every_round(rounds, rival, chosen):
                    chosen, beaten = rival, chosen
        kept = current is not None and chosen == current != best and bool(rounds)

        from .demo import gain
        improvement = None
        if stock in rounds and chosen != stock:
            ref = _round_mean(rounds, stock)
            meas = _round_mean(rounds, chosen)
            shown = round(ref / meas, 2)
            if _quieter_every_round(rounds, chosen, stock, by='magnitude') and shown > 1:
                improvement = shown
                print('\nvs Klipper defaults %s: %s across all %d rounds (%.1f -> %.1f)'
                      % (stock.label(), gain(ref, meas), rounds_count, ref, meas))
            else:
                print('\nvs Klipper defaults %s: mean %.1f -> %.1f (not lower in every round)'
                      % (stock.label(), ref, meas))

        if current in rounds and current != chosen:
            ref_cur = _round_mean(rounds, current)
            meas = _round_mean(rounds, chosen)
            print('vs current/autotune %s: %s (%.1f -> %.1f across %d rounds)'
                  % (current.label(), gain(ref_cur, meas), ref_cur, meas, rounds_count))
        elif kept:
            print('\nKeep current Z registers %s: %s did not beat them in all %d rounds '
                  '(%.1f -> %.1f)' % (current.label(), best.label(), rounds_count,
                                      _round_mean(rounds, current), _round_mean(rounds, best)))

        tag = autotune_tag(ctx.driver_name, ctx.autotune)
        save_winner_state(ctx.driver_name, ctx.steppers, chosen, speed=speed, autotune=tag,
                          hearing=hearing, kept=kept, improvement=improvement)

        print('\n=== Z Winner (%g mm/s) ===' % speed)
        print('%s  (f_chop %.1f kHz, h_eff %d)'
              % (chosen.label(), tmc.chopper_freq_hz(chosen, ctx.driver) / 1000.0,
                 tmc.effective_hysteresis(chosen)))
        print(rail_snippet(ctx.driver, ctx.steppers, chosen))

        pct = round((1 - 1 / improvement) * 100) if improvement else 0
        pct_note = ' (-%d%%)' % pct if pct >= 1 else ''
        screen.final('Chopper Z: %s%s%s' % ('keep ' if kept else '', chosen.label(), pct_note),
                     'Z %s%s' % ('keep' if kept else chosen.compact(), pct_note))
    finally:
        print('Restoring Z driver registers and printer settings')
        motion.restore()

    if args.save:
        from .analyze import _persist, register_options, save_refusal
        from .moonraker import Moonraker
        why = save_refusal(settings, ctx.driver_name, *ctx.steppers, fields=chosen.fields(),
                           autotune=tag)
        if why:
            raise SystemExit(why)
        writes = [('tmc%s %s' % (ctx.driver_name, stepper), register_options(chosen.fields()))
                  for stepper in ctx.steppers]
        _persist(Moonraker(args.url), writes, 'the Z registers')
    elif ctx.autotune is not None:
        print('\n' + autotune_advice(settings, ctx.driver_name, *ctx.steppers))
    else:
        print('\nTo save without re-tuning: CHOPPER_Z SAVE_LAST=1 (or CHOPPER_SAVE MOTOR=Z)')
    return 0
