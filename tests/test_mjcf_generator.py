"""Integration tests: every generated scene compiles and behaves physically."""

import os

import mujoco
import numpy as np
import pytest

from config.composite_ball_params import (
    BB8Params,
    BumpSpec,
    ControlMode,
    HeightfieldSpec,
    KiwiDriveParams,
    OmniwheelMode,
    PendulumParams,
    PlanarBalancerParams,
    SimulationParams,
    SingleOmniwheelParams,
)
from models.mjcf_generator import (
    MeshImportSpec,
    PrimitiveSpec,
    SceneExtras,
    build_bb8,
    build_example,
    build_kiwi_drive,
    build_single_omniwheel,
    list_examples,
)
from models.runtime import SceneStepper, contact_forces_between

MODES = list(OmniwheelMode)


def _settle(bundle, seconds):
    m, d, st = bundle.make()
    st.step(int(seconds / m.opt.timestep))
    return m, d, st


@pytest.mark.parametrize("num", [1, 2, 3, 4, 5])
def test_examples_compile_and_run(num):
    bundle = build_example(num)
    m, d, st = _settle(bundle, 0.5)
    assert np.all(np.isfinite(d.qpos))
    assert m.nu == len(bundle.meta.actuators) > 0
    for s in bundle.meta.sensors:
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, s) >= 0
    assert len(list_examples()) == 5


@pytest.mark.parametrize("variant", ["cartpole", "spherical"])
def test_pendulum_variants(variant):
    bundle = build_example("inverted_pendulum", PendulumParams(variant=variant))
    m, d, st = _settle(bundle, 1.0)
    pitch = d.qpos[m.jnt_qposadr[m.joint("pole_pitch").id]]
    assert abs(pitch) > 0.1  # unactuated pendulum falls from its 3 deg start


def test_ball_inertia_injected_exactly():
    p = BB8Params()
    m = build_bb8(p).compile()
    bid = m.body("composite_ball").id
    assert m.body_mass[bid] == pytest.approx(p.ball.mass, rel=1e-8)
    assert np.allclose(m.body_inertia[bid], p.ball.inertia, rtol=1e-8)


@pytest.mark.parametrize("mode", MODES)
def test_kiwi_drive_tracks_mixer(mode):
    p = KiwiDriveParams()
    p.wheel.mode = mode
    bundle = build_kiwi_drive(p)
    m, d, st = bundle.make()
    M = np.array(bundle.meta.kinematics["mixing_matrix"])
    for cmd in ([0.3, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 1.0]):
        bundle.reset(m, d)
        d.ctrl[:] = M @ np.array(cmd)
        st.step(int(2.0 / m.opt.timestep))
        v = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, m.body("chassis").id, v, 1)
        assert np.allclose([v[3], v[4], v[2]], cmd, atol=0.05)


@pytest.mark.parametrize("mode", MODES)
def test_single_omniwheel_anisotropy(mode):
    p = SingleOmniwheelParams()
    p.wheel.mode = mode
    bundle = build_single_omniwheel(p)
    m, d, st = bundle.make()
    st.push("chassis", [0, 5, 0], 0.5)          # along the axle: rollers spin freely
    st.step(int(0.5 / m.opt.timestep))
    lateral = d.qvel[1]
    bundle.reset(m, d)
    st.reset_state()
    st.push("chassis", [5, 0, 0], 0.5)          # along the rolling direction: wheel servo resists
    st.step(int(0.5 / m.opt.timestep))
    forward = d.qvel[0]
    assert lateral > 1.0
    assert abs(forward) < 0.2 * lateral


@pytest.mark.parametrize("mode", MODES)
def test_bb8_wheel_jacobian(mode):
    p = BB8Params()
    p.head.wheel.mode = mode
    p.head.motor.control_mode = ControlMode.VELOCITY
    p.head.motor.kv = 2.0
    p.head.magnet.enabled = True
    bundle = build_bb8(p)
    m, d, st = bundle.make()
    m.body_inertia[m.body("head").id] *= 1e4     # keep the head attitude (nearly) fixed
    J = np.array(bundle.meta.kinematics["ball_to_wheel_jacobian"])
    omega = np.array([0.0, 2.0, 0.0])
    d.ctrl[:] = J @ omega
    st.step(int(4.0 / m.opt.timestep))
    w_ball = d.cvel[m.body("composite_ball").id][:3]
    assert np.allclose(w_ball, omega, atol=0.12)


@pytest.mark.parametrize("backend", ["applied", "passive_callback"])
def test_magnetic_preload_adds_normal_force(backend):
    p = BB8Params()
    p.head.magnet.enabled = True
    p.head.magnet.force = 20.0
    bundle = build_bb8(p)
    m = bundle.compile()
    d = mujoco.MjData(m)
    bundle.reset(m, d)
    st = SceneStepper(m, d, bundle.meta, magnet_backend=backend)
    st.step(int(0.5 / m.opt.timestep))
    wheel_geoms = [m.geom(g).id for w in bundle.meta.wheels for g in w.collision_geoms]
    _, total = contact_forces_between(m, d, wheel_geoms, [m.geom("ball_geom").id])
    weight = p.head.total_mass * 9.81
    assert total[2] == pytest.approx(weight + 20.0, rel=0.02)
    assert not d.qfrc_applied.any()            # magnet share never leaks into user inputs
    # compiling another model afterwards still works (callback is scoped)
    build_example(1).compile()


def test_fallen_head_lands_on_wheels():
    p = BB8Params(initial_tilt_deg=(2.0, 3.0))
    m, d, st = _settle(build_bb8(p), 3.0)
    assert d.xpos[m.body("head").id][2] > 0.03


def test_planar_balancer_variants():
    for variant in ("1d", "2d"):
        bundle = build_example(4, PlanarBalancerParams(variant=variant))
        m, d, st = _settle(bundle, 0.3)
        assert np.all(np.isfinite(d.qpos))


def test_extras_terrain_meshes_and_files(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    box = trimesh.creation.box(extents=(100, 60, 40))
    box.apply_translation((30, 0, 0))
    stl = tmp_path / "bracket.stl"
    box.export(stl)
    sim = SimulationParams()
    sim.terrain.floor_slope_deg = 5.0
    sim.terrain.bumps = [BumpSpec()]
    sim.terrain.heightfields = [HeightfieldSpec(profile="random", pos=(3, 0, 0))]
    extras = SceneExtras(
        primitives=[PrimitiveSpec("crate", size=(0.1, 0.1, 0.1), pos=(1, 0, 0.1), dynamic=True, density=300)],
        meshes=[MeshImportSpec(str(stl), scale=0.001, dynamic=True, pos=(0.6, 0.6, 0.3)),
                MeshImportSpec(str(stl), name="head_bracket", scale=0.001, parent="head", pos=(0, 0, 0.15))],
    )
    p = BB8Params()
    p.head.counterweight.enabled = True
    p.head.magnet.enabled = True
    bundle = build_bb8(p, sim, extras)
    m, d, st = _settle(bundle, 0.5)
    bid = m.body("bracket").id
    assert m.body_mass[bid] == pytest.approx(1240 * 0.1 * 0.06 * 0.04, rel=1e-6)
    assert np.allclose(m.body_ipos[bid], [0.03, 0, 0], atol=1e-6)
    assert np.all(np.isfinite(d.qpos))
    path = bundle.write(str(tmp_path / "scene"))
    assert os.path.isfile(path)
    assert mujoco.MjModel.from_xml_path(path).nbody == m.nbody
    assert bundle.to_spec().compile().nbody == m.nbody
