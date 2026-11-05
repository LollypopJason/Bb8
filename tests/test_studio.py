"""Module 6 tests: studio session, REST API and WebSocket stream."""

import io
import os
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main_studio import StudioSession, create_app
from config.composite_ball_params import CompositeBallParams
from training.train_rl import TrainingConfig, TrainingManager


@pytest.fixture()
def client(tmp_path):
    sess = StudioSession(runs_dir=str(tmp_path / "runs"))
    app = create_app(sess, autostart=False)
    with TestClient(app) as c:
        c.session = sess
        yield c


def test_static_frontend_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "studio.js" in r.text
    for path in ("/static/studio.js", "/static/studio.css", "/static/vendor/three/three.module.min.js",
                 "/static/vendor/three/OrbitControls.js"):
        assert client.get(path).status_code == 200


@pytest.mark.parametrize("example", ["inverted_pendulum", "single_omniwheel", "kiwi_drive", "planar_balancer", "bb8"])
def test_scene_switch_descriptor_and_stepping(client, example):
    r = client.post("/api/scene", json={"example": example})
    assert r.status_code == 200, r.text
    st = r.json()
    assert st["example"] == example and st["mode"] in st["modes"]
    desc = client.get("/api/scene").json()
    m = client.session.model
    assert len(desc["bodies"]) == m.nbody
    assert len(desc["geoms"]) == m.ngeom
    assert client.post("/api/step", json={"n": 50}).status_code == 200
    rows = client.session.log_rows()
    assert len(rows) == 50 and np.isfinite(rows[-1]["t"])


def test_controller_modes_and_errors(client):
    for mode in ("lqr", "manual", "off", "pid"):
        r = client.post("/api/controller", json={"mode": mode})
        assert r.status_code == 200 and r.json()["mode"] == mode
        client.post("/api/step", json={"n": 20})
    r = client.post("/api/controller", json={"mode": "rl"})
    assert r.status_code == 400 and client.get("/api/status").json()["mode"] == "pid"
    r = client.post("/api/controller", json={"mode": "pid", "pid": {"kp": 120.0, "outer_vel": 0.3}})
    assert r.json()["pid"]["kp"] == 120.0 and r.json()["pid"]["outer_vel"] == 0.3
    r = client.post("/api/controller", json={"mode": "lqr", "lqr": {"ex": 100.0}})
    assert r.json()["lqr"]["spectral_radius"] < 1.0


def test_balancing_push_and_teleop(client):
    sess = client.session
    client.post("/api/step", json={"n": 100})
    assert client.post("/api/push", json={"body": "head", "force": [6, 0, 0], "duration": 0.1}).status_code == 200
    client.post("/api/step", json={"n": 30})
    peak = max(max(abs(r["pitch_deg"]), abs(r["roll_deg"])) for r in sess.log_rows()[-30:])
    assert peak > 0.3                                             # the push tilts the head
    client.post("/api/step", json={"n": 400})
    assert not sess.fallen
    assert client.post("/api/push", json={"body": "nope", "force": [1, 0, 0]}).status_code == 400
    client.post("/api/teleop", json={"keys": ["w"], "speed": 0.3})
    client.post("/api/step", json={"n": 400})
    row = sess.log_rows()[-1]
    assert row["target_vx"] == pytest.approx(0.3) and row["ball_vx"] == pytest.approx(0.3, abs=0.08)


def test_ball_preview_and_apply(client):
    body = {"inner_radius": 0.18, "foam_thickness": 0.04, "epoxy_thickness": 0.005, "epoxy_density": 900}
    prev = client.post("/api/ball/preview", json=body).json()
    ref = CompositeBallParams(inner_radius=0.18, foam_thickness=0.04, epoxy_thickness=0.005, epoxy_density=900)
    assert prev["total_mass_kg"] == pytest.approx(ref.mass)
    assert client.post("/api/ball/preview", json={"foam_thickness": -1}).status_code == 400
    client.post("/api/ball/apply", json=body)
    m = client.session.model
    assert m.body_mass[m.body("composite_ball").id] == pytest.approx(ref.mass)
    assert client.get("/api/status").json()["ball_summary"]["inertia_kgm2"] == pytest.approx(ref.inertia)


def test_params_roundtrip_and_hardware(client):
    st = client.get("/api/status").json()
    params = st["params"]
    params["head"]["num_wheels"] = 4
    params["head"]["magnet"]["enabled"] = True
    sim = st["sim"]
    sim["terrain"]["floor_slope_deg"] = 3.0
    r = client.post("/api/params", json={"params": params, "sim": sim})
    assert r.status_code == 200, r.text
    assert len(client.session.bundle.meta.wheels) == 4 and client.session.bundle.meta.magnets
    bad = dict(params)
    bad["head"] = dict(params["head"], zenith_angle_deg=80)
    assert client.post("/api/params", json={"params": bad}).status_code == 400
    assert len(client.session.bundle.meta.wheels) == 4                          # unchanged after the error
    r = client.post("/api/hardware", json={"latency_ms": 24, "noise": False})
    assert r.json()["hardware"]["latency"]["delay"] == pytest.approx(0.024)
    assert client.session.env.actuators.delay.delay_steps == 12
    assert client.post("/api/hardware", json={"latency_ms": 80}).status_code == 400


def test_logging_csv_json(client):
    client.post("/api/sim", json={"recording": True})
    client.post("/api/step", json={"n": 30})
    csv_text = client.get("/api/log.csv").text
    header = csv_text.splitlines()[0]
    assert "roll_deg" in header and "reward_terms.tilt" in header and "wheel_torque[0]" in header
    assert len(csv_text.splitlines()) == 31
    assert len(client.get("/api/log.json").json()) == 30


def test_mesh_upload_and_primitives(client):
    trimesh = pytest.importorskip("trimesh")
    buf = io.BytesIO()
    trimesh.creation.box(extents=(80, 50, 30)).export(buf, file_type="stl")
    r = client.post("/api/mesh", files={"file": ("bracket.stl", buf.getvalue(), "application/octet-stream")},
                    data={"scale": "0.001", "parent": "head", "z": "0.15"})
    assert r.status_code == 200, r.text
    assert r.json()["mass"] == pytest.approx(1240 * 0.08 * 0.05 * 0.03, rel=1e-6)
    desc = client.get("/api/scene").json()
    assert desc["meshes"] and any(g["type"] == "mesh" for g in desc["geoms"])
    assert client.post("/api/mesh", files={"file": ("x.exe", b"MZ", "application/octet-stream")}).status_code == 400
    r = client.post("/api/primitive", json={"name": "crate", "type": "box", "size": [0.05, 0.05, 0.05],
                                            "pos": [0.8, 0, 0.1], "dynamic": True})
    assert r.status_code == 200
    assert client.session.model.body("crate").id > 0
    client.post("/api/step", json={"n": 20})
    client.post("/api/extras/clear", json={})
    assert not client.get("/api/scene").json()["meshes"]


def test_websocket_stream(client):
    with client.websocket_connect("/ws") as ws:
        first = ws.receive_json()
        assert first["type"] == "scene" and first["scene"]["geoms"]
        client.post("/api/step", json={"n": 5})
        frame = ws.receive_json()
        assert frame["type"] == "frame"
        assert len(frame["xpos"]) == 3 * client.session.model.nbody
        assert len(frame["xquat"]) == 4 * client.session.model.nbody
        ws.send_json({"type": "keys", "keys": ["w", "x"]})
        deadline = time.time() + 5
        while client.session.keys != {"w"} and time.time() < deadline:
            ws.receive_json()
        assert client.session.keys == {"w"}


def test_policy_hot_load_and_rl_mode(client, tmp_path):
    runs = client.session.runs_dir
    cfg = TrainingConfig(total_timesteps=1024, n_envs=1, vec_env="dummy", out_dir=runs, checkpoint_every=0)
    cfg.ppo.n_steps = 512
    mgr = TrainingManager(cfg)
    mgr.start(background=False)
    assert mgr.state == "finished", mgr.error
    names = [p["name"] for p in client.get("/api/policies").json()]
    assert any(n.endswith(".onnx") for n in names)
    r = client.post("/api/policy/load", json={"path": mgr.latest_onnx})
    assert r.status_code == 200, r.text
    assert client.get("/api/status").json()["mode"] == "rl"
    assert client.post("/api/step", json={"n": 20}).status_code == 200
    outside = tmp_path / "evil.onnx"
    outside.write_bytes(b"x")
    assert client.post("/api/policy/load", json={"path": str(outside)}).status_code == 400


def test_training_via_api(client):
    r = client.post("/api/train/start", json={"algo": "PPO", "steps": 600, "n_envs": 1, "vec_env": "dummy"})
    assert r.status_code == 200, r.text
    assert client.post("/api/train/start", json={"steps": 600, "vec_env": "dummy"}).status_code == 400
    trainer = client.session.trainer
    assert trainer.wait(timeout=300) and trainer.state == "finished", trainer.error
    status = client.get("/api/train/status").json()
    assert status["status"]["state"] == "finished"
    assert any(e["event"]["type"] == "finished" for e in client.session.events_since(0))
    assert client.post("/api/policy/load", json={"path": "latest"}).status_code == 200
