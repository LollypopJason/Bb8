"""Physics-calculator tests for config.composite_ball_params."""

import math

import numpy as np
import pytest

from config.composite_ball_params import (
    BB8Params,
    CompositeBallParams,
    HeadUnitParams,
    KiwiDriveParams,
    params_from_dict,
    params_to_dict,
    thick_shell_inertia,
    thick_shell_mass,
)


def test_solid_sphere_limit():
    m = thick_shell_mass(1000.0, 0.0, 0.1)
    assert m == pytest.approx(4 / 3 * math.pi * 1000 * 0.1**3)
    assert thick_shell_inertia(m, 0.0, 0.1) == pytest.approx(0.4 * m * 0.1**2)


def test_thin_shell_limit():
    m = 2.0
    assert thick_shell_inertia(m, 0.1, 0.1) == pytest.approx(2 / 3 * m * 0.1**2)
    # nearly thin shell converges continuously to the limit
    assert thick_shell_inertia(m, 0.1 - 1e-7, 0.1) == pytest.approx(2 / 3 * m * 0.01, rel=1e-5)


def test_layers_sum_equals_difference_of_solid_spheres():
    ball = CompositeBallParams(inner_radius=0.2, foam_thickness=0.03, epoxy_thickness=0.004,
                               foam_density=35, epoxy_density=950)
    def solid(rho, r):
        m = thick_shell_mass(rho, 0, r)
        return m, 0.4 * m * r * r
    m2, i2 = solid(950, ball.r2)
    m1e, i1e = solid(950, ball.r1)
    m1f, i1f = solid(35, ball.r1)
    m0, i0 = solid(35, ball.r0)
    assert ball.mass == pytest.approx((m2 - m1e) + (m1f - m0))
    assert ball.inertia == pytest.approx((i2 - i1e) + (i1f - i0))
    assert 0.4 < ball.inertia_ratio < 2 / 3


def test_from_radii_and_inertial_attrs():
    ball = CompositeBallParams.from_radii(0.2, 0.23, 0.234)
    assert ball.r2 == pytest.approx(0.234)
    attrs = ball.mjcf_inertial_attrs()
    vals = [float(v) for v in attrs["diaginertia"].split()]
    assert vals == pytest.approx([ball.inertia] * 3)
    assert float(attrs["mass"]) == pytest.approx(ball.mass)


def test_scaled_domain_randomisation():
    ball = CompositeBallParams()
    heavy = ball.scaled(epoxy_mass_scale=1.15)
    epoxy = [p for p in ball.layer_properties() if p.name == "epoxy_sawdust"][0]
    assert heavy.mass - ball.mass == pytest.approx(0.15 * epoxy.mass)
    assert ball.epoxy_density == 950.0  # original untouched


def test_validation_errors():
    with pytest.raises(ValueError):
        CompositeBallParams.from_radii(0.2, 0.1, 0.3)
    head = HeadUnitParams(zenith_angle_deg=70)
    with pytest.raises(ValueError):
        head.validate()


def test_wheel_geometry_frames_and_jacobian():
    head = HeadUnitParams()
    R = 0.234
    for g in head.wheel_geometry(R):
        n, t, a = map(np.array, (g.normal, g.rolling_dir, g.axle))
        assert np.allclose(np.cross(n, t), a)
        # contact point lies on the sphere (head origin is the ball top)
        assert np.linalg.norm(np.array(g.contact_point) + [0, 0, R]) == pytest.approx(R)
        assert math.degrees(math.acos(n[2])) == pytest.approx(head.zenith_angle_deg)
    J = head.ball_to_wheel_jacobian(R)
    assert J.shape == (3, 3)
    assert np.linalg.matrix_rank(J) == 3  # any ball rotation is reachable


def test_kiwi_mixing_matrix_matches_formula():
    p = KiwiDriveParams()
    M = p.mixing_matrix()
    th = p.wheel_angles[1]
    assert M[1] == pytest.approx(np.array([-math.sin(th), math.cos(th), p.chassis_radius]) / p.wheel.radius)


def test_dict_roundtrip():
    p = BB8Params()
    p.head.magnet.enabled = True
    q = params_from_dict(BB8Params, params_to_dict(p))
    assert params_to_dict(q) == params_to_dict(p)
    assert q.head.wheel.mode == p.head.wheel.mode
