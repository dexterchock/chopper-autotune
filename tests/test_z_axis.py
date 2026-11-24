import os
import re
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from chopper_autotune import analyze, cli, tmc, z_axis
from chopper_autotune.collect import DriverTooHot, RunStopped


def voron24_settings(autotune_z=False, slope=True):
    settings = {
        'printer': {
            'kinematics': 'corexy',
            'max_velocity': '390',
            'max_accel': '4400',
            'max_z_velocity': '30',
            'max_z_accel': '350',
        },
        'stepper_x': {'position_min': '0', 'position_max': '350', 'rotation_distance': '40',
                      'microsteps': '32', 'full_steps_per_rotation': '200'},
        'stepper_y': {'position_min': '0', 'position_max': '350', 'rotation_distance': '40',
                      'microsteps': '32', 'full_steps_per_rotation': '200'},
        'resonance_tester': {'accel_chip': 'adxl345', 'accel_chip_z': 'adxl345'},
        'adxl345': {},
        'input_shaper': {'shaper_type_z': '2hump_ei', 'shaper_freq_z': '76.6'},
        'quad_gantry_level': {},
    }
    for suffix in ('', '1', '2', '3'):
        stepper = 'stepper_z' + suffix
        step_cfg = {
            'rotation_distance': '40',
            'gear_ratio': [[80, 16]],
            'microsteps': '32',
            'full_steps_per_rotation': '200',
        }
        if suffix == '':
            step_cfg.update(position_min='-2', position_max='330')
        settings[stepper] = step_cfg
        tmc_cfg = {'run_current': '0.86'}
        if slope:
            tmc_cfg['driver_slope_control'] = '3'
        settings['tmc2240 ' + stepper] = tmc_cfg
        if autotune_z:
            settings['autotune_tmc ' + stepper] = {'motor': 'leisai-42cm06d', 'tuning_goal': 'auto'}
    return settings


class FakeZKl:
    def __init__(self, settings, homed='xyz', z_pos=20.0, origin_z=0.0, z_shaper='76.600',
                 mesh_name='default', autotune_live=None, fail_on_move=None, hot_on_move=False,
                 printing=False):
        self.path = '/tmp/klippy_uds'
        self._settings = settings
        self.homed = homed
        self.position = [175.0, 175.0, z_pos]
        self.origin_z = origin_z
        self.z_shaper = z_shaper
        self.mesh_name = mesh_name
        self.profiles = {'default': {}} if mesh_name else {}
        self.autotune_live = autotune_live or {'tbl': 2, 'toff': 3, 'hstrt': 5, 'hend': 2, 'tpfd': 4}
        self.fail_on_move = fail_on_move
        self.hot_on_move = hot_on_move
        self.printing = printing
        self.overflows = 0
        self.scripts = []
        self.subscribed_accel = None
        self.subscribed_status = None
        self.live_fields = {
            'stepper_z' + s: dict(self.autotune_live) for s in ('', '1', '2', '3')
        }
        self.gconf = {'stepper_z' + s: 1 for s in ('', '1', '2', '3')}
        self._t = 100.0
        self._samples = []
        self.move_count = 0
        self.active_mcr = 0.5
        self.active_accel = 350.0
        self.speed_factor = 1.0

    def close(self):
        pass

    def settings(self):
        return self._settings

    def config_sections(self):
        return list(self._settings.keys())

    def object_list(self):
        return ['display_status', 'gcode', 'toolhead']

    def is_printing(self):
        return self.printing

    def info(self):
        return {'state': 'ready', 'software_version': 'v0.13.0'}

    def homed_axes(self):
        return self.homed

    def subscribe_accel(self, chip):
        self.subscribed_accel = chip

    def subscribe_status(self, objects):
        self.subscribed_status = objects

    def status(self):
        out = {}
        for sec in (self.subscribed_status or {}):
            temp = 105.0 if (self.hot_on_move and self.move_count >= 1) else 45.0
            out[sec] = {'drv_status': {}, 'temperature': temp}
        return out

    def request(self, method, params=None):
        if method == 'objects/query':
            objs = (params or {}).get('objects', {})
            res = {}
            if 'toolhead' in objs:
                res['toolhead'] = {
                    'homed_axes': self.homed,
                    'position': list(self.position),
                    'max_velocity': 390.0,
                    'max_accel': self.active_accel,
                    'minimum_cruise_ratio': self.active_mcr,
                }
            if 'gcode_move' in objs:
                res['gcode_move'] = {
                    'homing_origin': [0.0, 0.0, self.origin_z],
                    'speed_factor': self.speed_factor,
                }
            if 'bed_mesh' in objs:
                res['bed_mesh'] = {
                    'profile_name': self.mesh_name,
                    'profiles': dict(self.profiles),
                }
            if 'quad_gantry_level' in objs:
                res['quad_gantry_level'] = {'applied': True}
            if 'gcode' in objs:
                res['gcode'] = {'commands': {'M117': {}, 'M118': {}, 'RESPOND': {}}}
            return {'status': res}
        raise AssertionError('unexpected request %r' % method)

    def gcode_output(self, script):
        self.scripts.append(script)
        if script == 'SET_INPUT_SHAPER':
            if self.z_shaper is None:
                return ['// shaper_type_x:mzv shaper_freq_x:56.000']
            return ['// shaper_type_z:2hump_ei shaper_freq_z:%s' % self.z_shaper]
        m = re.match(r'DUMP_TMC STEPPER=(\S+) REGISTER=(\S+)', script)
        if m:
            stepper, reg = m.group(1), m.group(2)
            if reg == 'GCONF':
                return ['// GCONF: 00000000 en_pwm_mode=%d' % self.gconf[stepper]]
            if reg == 'CHOPCONF':
                f = self.live_fields[stepper]
                return ['// CHOPCONF: 00000000 toff=%d hstrt=%d hend=%d tbl=%d tpfd=%d'
                        % (f['toff'], f['hstrt'], f['hend'], f['tbl'], f['tpfd'])]
        return []

    def gcode(self, script):
        self.scripts.append(script)
        for line in script.splitlines():
            for match in re.finditer(r'SET_TMC_FIELD STEPPER=(\S+) FIELD=(\w+) VALUE=(\d+)', line):
                stepper, field, val = match.group(1), match.group(2).lower(), int(match.group(3))
                if field == 'en_pwm_mode':
                    self.gconf[stepper] = val
                elif field in ('tbl', 'toff', 'hstrt', 'hend', 'tpfd'):
                    self.live_fields[stepper][field] = val
            m_shaper = re.search(r'SHAPER_FREQ_Z=(\S+)', line)
            if m_shaper:
                self.z_shaper = m_shaper.group(1)
            if 'BED_MESH_CLEAR' in line:
                self.mesh_name = ''
            m_mesh = re.search(r'BED_MESH_PROFILE LOAD="([^"]+)"', line)
            if m_mesh:
                self.mesh_name = m_mesh.group(1)
            m_mcr = re.search(r'MINIMUM_CRUISE_RATIO=(\S+)', line)
            if m_mcr:
                self.active_mcr = float(m_mcr.group(1))
            m_f = re.search(r'M220 S(\S+)', line)
            if m_f:
                self.speed_factor = float(m_f.group(1)) / 100.0
            m_a = re.search(r'M204 S(\S+)', line)
            if m_a:
                self.active_accel = float(m_a.group(1))
            m_z = re.search(r'G1 Z(\S+) F(\S+)', line)
            if m_z:
                self.move_count += 1
                if self.fail_on_move and self.move_count == self.fail_on_move:
                    raise RuntimeError('injected mid-run Z move failure')
                self.position[2] = float(m_z.group(1)) + self.origin_z
                feed_mm_s = float(m_z.group(2)) / 60.0
                # Synthetic measurement window ending at new print time
                step = 1.0 / 3200.0
                dur = 2.6
                n = int(dur / step)
                f = self.live_fields['stepper_z']
                # Distinct acoustic signature with clear resonance at 15 mm/s and optimal tbl1/toff5/hstrt4/hend6/tpfd2
                v = abs(round(feed_mm_s))
                res_gain = 3.0 if v == 15 else (1.5 if abs(v - 15) <= 2 else 1.0)
                amp = res_gain * (120.0 + 40.0 * abs(f['tbl'] - 1) + 35.0 * abs(f['toff'] - 5)
                                  + 20.0 * abs(f['hstrt'] - 4) + 15.0 * abs(f['hend'] - 6)
                                  + 8.0 * abs(f['tpfd'] - 2))
                times = np.linspace(self._t, self._t + dur, n, endpoint=False)
                xs = np.zeros(n)
                ys = np.zeros(n)
                zs = amp * np.sin(2.0 * np.pi * 80.0 * times)
                self._samples = list(zip(times, xs, ys, zs))
                self._t += dur
            elif line.startswith('G4 P'):
                dur = int(line.split('P')[1]) / 1000.0
                n = max(64, int(dur * 1600))
                times = np.linspace(self._t, self._t + dur, n, endpoint=False)
                self._samples = [(float(t), 1.0, 1.0, 2.0) for t in times]
                self._t += dur

    def print_time(self):
        return self._t

    def wait_for_sample(self, t_end):
        pass

    def samples_between(self, t0, t1):
        return [s for s in self._samples if t0 <= s[0] <= t1]


def test_z_context_voron24_4motor():
    ctx = z_axis.z_context(voron24_settings())
    assert ctx.steppers == ['stepper_z', 'stepper_z1', 'stepper_z2', 'stepper_z3']
    assert ctx.driver_name == '2240'
    assert ctx.max_z_velocity == 30.0
    assert ctx.max_z_accel == 350.0
    assert ctx.safe_lo == pytest.approx(64.4)
    assert ctx.safe_hi == pytest.approx(263.6)
    assert ctx.z_center == pytest.approx(164.0)
    assert ctx.z_limit == pytest.approx(100.0)


def test_z_context_refusals():
    cfg = voron24_settings()
    cfg['printer']['kinematics'] = 'corexz'
    with pytest.raises(SystemExit, match='corexz'):
        z_axis.z_context(cfg)

    cfg = voron24_settings()
    cfg['stepper_z2']['gear_ratio'] = [[40, 16]]
    with pytest.raises(SystemExit, match='gear_ratio'):
        z_axis.z_context(cfg)

    cfg = voron24_settings()
    cfg['stepper_z1']['microsteps'] = '16'
    with pytest.raises(SystemExit, match='microsteps'):
        z_axis.z_context(cfg)

    cfg = voron24_settings()
    del cfg['tmc2240 stepper_z3']
    cfg['tmc2209 stepper_z3'] = {'run_current': '0.86'}
    with pytest.raises(SystemExit, match='one TMC driver model'):
        z_axis.z_context(cfg)

    cfg = voron24_settings()
    cfg['autotune_tmc stepper_z'] = {'motor': 'leisai-42cm06d'}
    with pytest.raises(SystemExit, match='all Z steppers'):
        z_axis.z_context(cfg)


def test_z_safety_guards_and_refusals(tmp_path, monkeypatch):
    monkeypatch.setattr(z_axis, 'STATE', str(tmp_path / 'z_axis.json'))
    parser = cli.build_parser()

    # SAVE=1 under autotune_tmc refuses before moving
    kl = FakeZKl(voron24_settings(autotune_z=True))
    args = parser.parse_args(['z', '--save', '--yes'])
    with pytest.raises(SystemExit, match='autotune'):
        z_axis.z_tune(kl, args)
    assert kl.scripts == []

    # SPEED above max_z_velocity refuses before moving
    kl = FakeZKl(voron24_settings())
    args = parser.parse_args(['z', '--speed', '40', '--yes'])
    with pytest.raises(SystemExit, match='max_z_velocity'):
        z_axis.z_tune(kl, args)
    assert kl.scripts == []

    # ACCEL above max_z_accel refuses before moving
    args = parser.parse_args(['z', '--accel', '500', '--yes'])
    with pytest.raises(SystemExit, match='acceleration ceiling'):
        z_axis.z_tune(kl, args)
    assert kl.scripts == []

    # Unhomed Z refuses before moving
    kl = FakeZKl(voron24_settings(), homed='xy')
    args = parser.parse_args(['z', '--speed', '15', '--yes'])
    with pytest.raises(SystemExit, match='G28'):
        z_axis.z_tune(kl, args)
    assert not any('G1 ' in s or 'FORCE_MOVE' in s for s in kl.scripts)


def test_z_safety_invariants_bounds_offset_and_error_recovery(tmp_path, monkeypatch):
    monkeypatch.setattr(z_axis, 'STATE', str(tmp_path / 'z_axis.json'))
    cfg = voron24_settings(autotune_z=True)
    ctx = z_axis.z_context(cfg)

    # Bound enforcement raises RunStopped before emitting out-of-range G1
    kl = FakeZKl(cfg, origin_z=-0.5)
    hw = z_axis.build_z_hardware(kl, ctx)
    motion = z_axis.ZMotion(kl, hw, ctx, 350.0)
    with pytest.raises(RunStopped, match='safe window'):
        motion.move_bounds(travel=300.0, direction=1)

    # Check homing_origin subtraction: physical 164.0 with origin -0.5 -> G1 Z164.500
    zs, ze = motion.move_bounds(travel=20.0, direction=1)
    assert zs == pytest.approx(154.5)
    assert ze == pytest.approx(174.5)

    # Injected failure mid-run: restores registers, mode, shaper_z, MCR, mesh, and NEVER disables Z
    kl_err = FakeZKl(cfg, origin_z=-0.5, fail_on_move=3)
    parser = cli.build_parser()
    args = parser.parse_args(['z', '--speed', '15', '--yes', '--tbl', '1:2', '--toff', '4:5',
                              '--hstrt', '3:4', '--hend', '5:6', '--tpfd', '2:3'])
    with pytest.raises(RuntimeError, match='injected'):
        z_axis.z_tune(kl_err, args)
    joined = '\n'.join(kl_err.scripts)
    assert 'FORCE_MOVE' not in joined
    assert 'ENABLE=0' not in joined
    assert kl_err.z_shaper == '76.600'
    assert kl_err.mesh_name == 'default'
    assert kl_err.active_mcr == 0.5
    assert all(kl_err.gconf[s] == 1 for s in ctx.steppers)
    assert all(kl_err.live_fields[s] == kl_err.autotune_live for s in ctx.steppers)

    # DriverTooHot aborts cleanly WITHOUT disabling any Z motor
    kl_hot = FakeZKl(voron24_settings(), hot_on_move=True)
    with pytest.raises(DriverTooHot):
        z_axis.z_tune(kl_hot, args)
    joined_hot = '\n'.join(kl_hot.scripts)
    assert 'FORCE_MOVE' not in joined_hot
    assert 'ENABLE=0' not in joined_hot


def test_z_shaper_absent_on_vanilla_klipper(tmp_path, monkeypatch):
    monkeypatch.setattr(z_axis, 'STATE', str(tmp_path / 'z_axis.json'))
    kl = FakeZKl(voron24_settings(), z_shaper=None)
    ctx = z_axis.z_context(kl.settings())
    hw = z_axis.build_z_hardware(kl, ctx)
    motion = z_axis.ZMotion(kl, hw, ctx, 350.0)
    motion.prepare()
    motion.restore()
    assert not any('SHAPER_FREQ_Z' in s for s in kl.scripts)
    assert 'FORCE_MOVE' not in '\n'.join(kl.scripts)
    assert 'ENABLE=0' not in '\n'.join(kl.scripts)


def test_z_tune_end_to_end_and_save_restore(tmp_path, monkeypatch):
    monkeypatch.setattr(z_axis, 'STATE', str(tmp_path / 'z_axis.json'))
    monkeypatch.setattr(analyze, 'RESULTS_HOME', tmp_path)

    cfg = voron24_settings(autotune_z=False)
    kl = FakeZKl(cfg, origin_z=-0.4)
    parser = cli.build_parser()
    args = parser.parse_args([
        'z', '--min-speed', '10', '--max-speed', '20', '--step', '5',
        '--tbl', '1:2', '--toff', '3:5', '--hstrt', '4:5', '--hend', '2:6', '--tpfd', '2:4',
        '--yes',
    ])
    assert z_axis.z_tune(kl, args) == 0

    joined = '\n'.join(kl.scripts)
    assert 'FORCE_MOVE' not in joined
    assert 'ENABLE=0' not in joined
    # Physical stroke never leaves safe_lo..safe_hi
    for val in re.findall(r'G1 Z(\S+)', joined):
        phys = float(val) + kl.origin_z
        assert 20.0 <= phys <= 263.6 + 1e-6

    state = z_axis.load_winner_state()
    assert state is not None
    assert state['driver'] == '2240'
    assert state['steppers'] == ['stepper_z', 'stepper_z1', 'stepper_z2', 'stepper_z3']
    assert state['speed'] == 15.0
    assert state['fields']['tbl'] == 1
    assert state['fields']['toff'] == 5
    assert state['fields']['tpfd'] == 2
    assert state['improvement'] > 1.0

    # Save into all 4 [tmc2240 stepper_z*] sections via CHOPPER_SAVE MOTOR=Z
    from tests.test_save import FakeMoonraker
    cfg_text = '\n'.join([
        '[tmc2240 stepper_x]\ncs_pin: PE7\nrun_current: 0.86\n',
        '[tmc2240 stepper_z]\ncs_pin: PD10\nrun_current: 0.86\n',
        '[tmc2240 stepper_z1]\ncs_pin: PD7\nrun_current: 0.86\n',
        '[tmc2240 stepper_z2]\ncs_pin: PC14\nrun_current: 0.86\n',
        '[tmc2240 stepper_z3]\ncs_pin: PC15\nrun_current: 0.86\n',
    ])
    mk = FakeMoonraker({'printer.cfg': cfg_text}, settings=cfg)
    monkeypatch.setattr(analyze, 'Moonraker', lambda url: mk)
    analyze.run_save_latest(parser.parse_args(['save', '--motor', 'z']))
    assert mk.scripts[-1] == 'RESTART'
    for suffix in ('', '1', '2', '3'):
        block = mk.files['printer.cfg'].split('[tmc2240 stepper_z%s]' % suffix)[1].split('[')[0]
        assert 'driver_TBL: 1' in block and 'driver_TOFF: 5' in block and 'driver_TPFD: 2' in block

    # Rollback BACKUP=1 right after first save removes driver_* lines that were absent before
    analyze.run_restore_config(SimpleNamespace(defaults=False, backup=True, motor=None, url=''))
    for suffix in ('', '1', '2', '3'):
        block = mk.files['printer.cfg'].split('[tmc2240 stepper_z%s]' % suffix)[1].split('[')[0]
        assert 'driver_TBL' not in block and 'driver_TOFF' not in block and 'driver_TPFD' not in block

    # Rollback refuses to auto-re-save an older run, while direct run_save + DEFAULTS=1 works
    assert mk.files['printer.cfg'] == cfg_text
    analyze.run_save(mk, [], lines=[], z_state=state)
    analyze.run_restore_config(SimpleNamespace(defaults=True, backup=False, motor='Z', url=''))
    for suffix in ('', '1', '2', '3'):
        block = mk.files['printer.cfg'].split('[tmc2240 stepper_z%s]' % suffix)[1].split('[')[0]
        assert 'driver_TBL: 2' in block and 'driver_HEND: 2' in block and 'driver_TPFD: 4' in block

    # Set config registers in kl_demo settings and run DEMO=1 comparing defaults vs tuned
    kl_demo = FakeZKl(cfg)
    kl_demo.autotune_live = dict(state['fields'])
    for s in ctx_steppers():
        kl_demo.live_fields[s] = dict(state['fields'])
    for suffix in ('', '1', '2', '3'):
        sec = cfg['tmc2240 stepper_z' + suffix]
        for k, v in state['fields'].items():
            sec['driver_' + k] = str(v)
    assert z_axis.z_tune(kl_demo, parser.parse_args(['z', '--demo', '--yes'])) == 0
    assert 'FORCE_MOVE' not in '\n'.join(kl_demo.scripts)
    assert 'ENABLE=0' not in '\n'.join(kl_demo.scripts)


def ctx_steppers():
    return ['stepper_z', 'stepper_z1', 'stepper_z2', 'stepper_z3']


def test_z_cli_macro_params_no_numpy_and_selfcheck():
    parser = cli.build_parser()
    flags = cli.boolean_flags(parser)
    args = parser.parse_args(cli._gcode_args(
        ['z', 'MIN_SPEED=5', 'MAX_SPEED=30', 'STEP=1', 'SPEED=15', 'TPFD=0:15',
         'SAVE=1', 'SAVE_LAST=1', 'DEMO=1', 'DRY_RUN=1'], flags))
    assert (args.min_speed == 5 and args.max_speed == 30 and args.step == 1
            and args.speed == 15.0 and args.tpfd == tmc.Range(0, 15)
            and args.save and args.save_last and args.demo and args.dry_run)

    args_save = parser.parse_args(cli._gcode_args(['save', 'MOTOR=ABEZ'], flags))
    assert args_save.motor == 'ABEZ'

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    done = subprocess.run(
        [sys.executable, '-c', 'import sys, chopper_autotune.cli; print("numpy" in sys.modules)'],
        cwd=repo, capture_output=True, text=True)
    assert done.stdout.strip() == 'False', done.stderr
