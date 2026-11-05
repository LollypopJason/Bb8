"""Runtime physics helpers that accompany generated scenes.

Generated MJCF is plain MuJoCo, but two features of the studio need a little
help at step time:

1. **Anisotropic omniwheel contacts (fast contact-pair mode).** MuJoCo builds
   the contact tangent frame from the contact normal only, so the two tangent
   friction coefficients of a ``<pair>`` would point in arbitrary directions.
   :class:`SceneStepper` re-orients the frame of every wheel contact so that
   tangent 1 is the wheel rolling direction (``mu_tangent``) and tangent 2 the
   axle direction (``mu_lateral``), and rebuilds the constraints before the
   velocity stage. This is done by splitting ``mj_step`` into its documented
   stages (``mj_fwdPosition`` ... ``mj_step2``).

2. **Magnetic preload.** A ``mjcb_passive`` callback adds an equal-and-opposite
   attraction between the head magnet site and the ball centre (or internal
   counterweight). The callback is global in MuJoCo, so a small registry
   dispatches by ``MjData`` address to support many environments per process.

It also provides contact-force readout and timed push disturbances.
"""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Tuple

import mujoco
import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from models.mjcf_generator import SceneMetadata

__all__ = [
    "AlignedContact",
    "MagnetLink",
    "PassiveForceRegistry",
    "SceneStepper",
    "PushDisturbance",
    "contact_forces_between",
    "DelayLine",
]


@dataclass(frozen=True)
class AlignedContact:
    """Contacts between ``geom`` and ``target_geom`` get their tangent frame from ``site``.

    ``site``'s local x axis is the rolling direction, its y axis the axle.
    """

    geom: str
    target_geom: str
    site: str
    friction: Tuple[float, float, float, float, float]


@dataclass(frozen=True)
class MagnetLink:
    """Attraction from ``magnet_site`` (on ``source_body``) to ``target_site`` (on ``target_body``)."""

    source_body: str
    magnet_site: str
    target_body: str
    target_site: str
    force: float
    model: str = "constant"            # constant | dipole
    max_force: float = 60.0
    nominal_distance: float = 0.25


# --------------------------------------------------------------------------- #
# Global passive-force dispatch
# --------------------------------------------------------------------------- #
class PassiveForceRegistry:
    """Scoped installation of the global ``mjcb_passive`` hook.

    MuJoCo has a single process-wide passive callback, and the Python bindings
    cannot compile *any* model while a Python callback is installed (MuJoCo
    invokes it on compiler-internal data that has no Python wrapper). The
    registry therefore installs the hook only while a stepper is inside
    :meth:`SceneStepper.step` and dispatches by ``MjData`` address so several
    steppers can share it. Compiling models concurrently in another thread
    while a callback-backed stepper is stepping is not supported; use the
    default ``magnet_backend="applied"`` in that case.
    """

    _lock = threading.RLock()
    _handlers: Dict[int, Callable[[mujoco.MjModel, mujoco.MjData], None]] = {}
    _depth = 0

    @classmethod
    def _dispatch(cls, m: mujoco.MjModel, d: mujoco.MjData) -> None:
        handler = cls._handlers.get(d._address)
        if handler is not None:
            handler(m, d)

    @classmethod
    @contextlib.contextmanager
    def active(cls, data: mujoco.MjData, handler: Callable[[mujoco.MjModel, mujoco.MjData], None]):
        with cls._lock:
            cls._handlers[data._address] = handler
            if cls._depth == 0:
                mujoco.set_mjcb_passive(cls._dispatch)
            cls._depth += 1
            try:
                yield
            finally:
                cls._depth -= 1
                cls._handlers.pop(data._address, None)
                if cls._depth == 0:
                    mujoco.set_mjcb_passive(None)


# --------------------------------------------------------------------------- #
# Disturbances
# --------------------------------------------------------------------------- #
@dataclass
class PushDisturbance:
    """A constant world-frame force (and optional torque) on a body for a duration."""

    body_id: int
    force: np.ndarray
    torque: np.ndarray
    remaining: float
    point: Optional[np.ndarray] = None   # world point; None = body CoM


def contact_forces_between(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geoms_a: Sequence[int],
    geoms_b: Sequence[int],
) -> Tuple[float, np.ndarray]:
    """Total normal force magnitude and world-frame total contact force between two geom sets."""
    set_a, set_b = set(int(g) for g in geoms_a), set(int(g) for g in geoms_b)
    normal_sum = 0.0
    total = np.zeros(3)
    buf = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1 in set_a and g2 in set_b) or (g2 in set_a and g1 in set_b):
            mujoco.mj_contactForce(model, data, i, buf)
            frame = c.frame.reshape(3, 3)
            normal_sum += float(buf[0])
            total += frame.T @ buf[:3]
    return normal_sum, total


# --------------------------------------------------------------------------- #
# Stepper
# --------------------------------------------------------------------------- #
class SceneStepper:
    """Steps a generated scene, applying contact-frame alignment, magnets and pushes.

    Use it in place of ``mujoco.mj_step``::

        stepper = SceneStepper(model, data, bundle.meta)
        data.ctrl[:] = u
        stepper.step(n=frame_skip)
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        meta: Optional["SceneMetadata"] = None,
        aligned: Optional[Sequence[AlignedContact]] = None,
        magnets: Optional[Sequence[MagnetLink]] = None,
        magnet_backend: str = "applied",
    ) -> None:
        """``magnet_backend``: ``"applied"`` (default) adds the magnet force to
        ``qfrc_applied`` right after the position stage of every step;
        ``"passive_callback"`` adds it to ``qfrc_passive`` from a scoped
        ``mjcb_passive`` hook. Both give identical dynamics."""
        if magnet_backend not in ("applied", "passive_callback"):
            raise ValueError("magnet_backend must be 'applied' or 'passive_callback'")
        self.model = model
        self.data = data
        aligned = list(aligned if aligned is not None else (meta.aligned_contacts if meta else []))
        magnets = list(magnets if magnets is not None else (meta.magnets if meta else []))
        self._aligned_pairs: Dict[Tuple[int, int], Tuple[int, np.ndarray]] = {}
        for ac in aligned:
            g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, ac.geom)
            tg = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, ac.target_geom)
            s = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ac.site)
            if min(g, tg, s) < 0:
                raise KeyError(f"aligned contact refers to unknown names: {ac}")
            # Friction is read live from model.pair_friction (so domain randomisation of the
            # model reaches the aligned contacts); the metadata value is the fallback.
            pid = -1
            for k in range(model.npair):
                if {int(model.pair_geom1[k]), int(model.pair_geom2[k])} == {g, tg}:
                    pid = k
                    break
            fr = model.pair_friction[pid] if pid >= 0 else np.asarray(ac.friction, dtype=float)
            self._aligned_pairs[(g, tg)] = (s, fr)
            self._aligned_pairs[(tg, g)] = (s, fr)
        self._magnets: List[Tuple[int, int, int, int, MagnetLink]] = []
        for mg in magnets:
            ids = (
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mg.source_body),
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, mg.magnet_site),
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mg.target_body),
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, mg.target_site),
            )
            if min(ids) < 0:
                raise KeyError(f"magnet link refers to unknown names: {mg}")
            self._magnets.append((*ids, mg))
        self.magnets_enabled = True
        self.magnet_scale = 1.0
        self.pushes: List[PushDisturbance] = []
        self.last_magnet_force = np.zeros(3)
        self.magnet_backend = magnet_backend
        self._use_callback = magnet_backend == "passive_callback" and bool(self._magnets)
        self._magnet_qfrc = np.zeros(model.nv)      # magnet share added to qfrc_applied during a step
        self._split = bool(self._aligned_pairs) or (bool(self._magnets) and not self._use_callback)

    # ------------------------------------------------------------- lifecycle
    def close(self) -> None:
        """Release resources. ``qfrc_applied`` only ever holds user values between calls."""
        self._remove_magnet_share()
        self.pushes.clear()

    @property
    def has_aligned_contacts(self) -> bool:
        return bool(self._aligned_pairs)

    # ------------------------------------------------------------- magnets
    def _magnet_forces(self, m: mujoco.MjModel, d: mujoco.MjData, qfrc_out: np.ndarray) -> None:
        if not self.magnets_enabled or self.magnet_scale == 0.0:
            self.last_magnet_force[:] = 0.0
            return
        zero = np.zeros(3)
        for src_body, src_site, tgt_body, tgt_site, link in self._magnets:
            p_src = d.site_xpos[src_site]
            p_tgt = d.site_xpos[tgt_site]
            delta = p_tgt - p_src
            dist = float(np.linalg.norm(delta))
            if dist < 1e-9:
                continue
            if link.model == "dipole":
                mag = min(link.force * (link.nominal_distance / dist) ** 4, link.max_force)
            else:
                mag = link.force
            f = (self.magnet_scale * mag / dist) * delta
            self.last_magnet_force[:] = f
            mujoco.mj_applyFT(m, d, f, zero, p_src, src_body, qfrc_out)
            mujoco.mj_applyFT(m, d, -f, zero, p_tgt.copy(), tgt_body, qfrc_out)

    def _passive_callback(self, m: mujoco.MjModel, d: mujoco.MjData) -> None:
        self._magnet_forces(m, d, d.qfrc_passive)

    # ------------------------------------------------------------- pushes
    def push(
        self,
        body: str,
        force: Sequence[float],
        duration: float = 0.1,
        torque: Sequence[float] = (0.0, 0.0, 0.0),
        point: Optional[Sequence[float]] = None,
    ) -> None:
        """Apply a world-frame force [N] (and torque [N m]) on ``body`` for ``duration`` seconds."""
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        if bid < 0:
            raise KeyError(f"unknown body {body!r}")
        self.pushes.append(
            PushDisturbance(
                body_id=bid,
                force=np.asarray(force, dtype=float),
                torque=np.asarray(torque, dtype=float),
                remaining=float(duration),
                point=None if point is None else np.asarray(point, dtype=float),
            )
        )

    def impulse(self, body: str, impulse: Sequence[float], duration: float = 0.02) -> None:
        """Deliver a linear impulse [N s] spread over ``duration`` seconds."""
        imp = np.asarray(impulse, dtype=float)
        self.push(body, imp / max(duration, self.model.opt.timestep), duration)

    def _apply_pushes(self) -> None:
        d, m = self.data, self.model
        d.xfrc_applied[:] = 0.0
        if not self.pushes:
            return
        dt = m.opt.timestep
        alive: List[PushDisturbance] = []
        for p in self.pushes:
            if p.point is None:
                d.xfrc_applied[p.body_id, :3] += p.force
                d.xfrc_applied[p.body_id, 3:] += p.torque
            else:
                # force at an arbitrary world point = force at CoM + moment
                arm = p.point - d.xipos[p.body_id]
                d.xfrc_applied[p.body_id, :3] += p.force
                d.xfrc_applied[p.body_id, 3:] += p.torque + np.cross(arm, p.force)
            p.remaining -= dt
            if p.remaining > 1e-12:
                alive.append(p)
        self.pushes = alive

    # ------------------------------------------------------------- stepping
    def _build_alignment_lut(self) -> None:
        """Dense (geom, geom) lookup tables so alignment is vectorised over all contacts."""
        ng = self.model.ngeom
        self._lut_site = np.full((ng, ng), -1, dtype=np.int64)
        self._lut_pair = np.full((ng, ng), -1, dtype=np.int64)
        self._lut_fb = np.zeros((ng, ng), dtype=np.int64)
        self._fallback = []
        for (g, tg), (site, fr) in self._aligned_pairs.items():
            self._lut_site[g, tg] = site
            pid = -1
            for k in range(self.model.npair):
                if {int(self.model.pair_geom1[k]), int(self.model.pair_geom2[k])} == {g, tg}:
                    pid = k
                    break
            self._lut_pair[g, tg] = pid
            self._lut_fb[g, tg] = len(self._fallback)
            self._fallback.append(np.array(fr, dtype=float))
        self._fallback_arr = np.array(self._fallback) if self._fallback else np.zeros((0, 5))

    def _align_contact_frames(self) -> None:
        d = self.data
        if d.ncon == 0:
            return
        if not hasattr(self, "_lut_site"):
            self._build_alignment_lut()
        con = d.contact
        g1, g2 = con.geom1, con.geom2
        sites = self._lut_site[g1, g2]
        idx = np.nonzero(sites >= 0)[0]
        if idx.size == 0:
            return
        frame = con.frame
        n = frame[idx, 0:3]
        t = d.site_xmat[sites[idx]].reshape(-1, 3, 3)[:, :, 0]
        t = t - n * np.einsum("ij,ij->i", t, n)[:, None]
        norm = np.sqrt(np.einsum("ij,ij->i", t, t))
        ok = norm > 1e-9
        idx, n, t = idx[ok], n[ok], t[ok] / norm[ok, None]
        b = np.empty_like(t)                      # n x t without np.cross overhead
        b[:, 0] = n[:, 1] * t[:, 2] - n[:, 2] * t[:, 1]
        b[:, 1] = n[:, 2] * t[:, 0] - n[:, 0] * t[:, 2]
        b[:, 2] = n[:, 0] * t[:, 1] - n[:, 1] * t[:, 0]
        frame[idx, 3:6] = t
        frame[idx, 6:9] = b
        pids = self._lut_pair[g1[idx], g2[idx]]
        fr = self._fallback_arr[self._lut_fb[g1[idx], g2[idx]]]
        live = pids >= 0
        if np.any(live):
            fr[live] = self.model.pair_friction[pids[live]]
        con.friction[idx] = fr

    def _apply_magnets_to_qfrc_applied(self) -> None:
        """Temporarily add the magnet forces to ``qfrc_applied`` (removed again after the step)."""
        self._magnet_qfrc[:] = 0.0
        self._magnet_forces(self.model, self.data, self._magnet_qfrc)
        self.data.qfrc_applied[:] += self._magnet_qfrc

    def _remove_magnet_share(self) -> None:
        if self._magnet_qfrc.any():
            self.data.qfrc_applied[:] -= self._magnet_qfrc
            self._magnet_qfrc[:] = 0.0

    def _position_stage(self) -> None:
        m, d = self.model, self.data
        mujoco.mj_fwdPosition(m, d)
        if self._aligned_pairs:
            self._align_contact_frames()
            mujoco.mj_makeConstraint(m, d)
            mujoco.mj_projectConstraint(m, d)
        if self._magnets and not self._use_callback:
            self._apply_magnets_to_qfrc_applied()

    def _step_once(self) -> None:
        m, d = self.model, self.data
        self._apply_pushes()
        if not self._split:
            mujoco.mj_step(m, d)
            return
        # mj_step == mj_step1 + mj_step2; mj_step1 is expanded so contact frames
        # and position-dependent forces can be set between its stages.
        mujoco.mj_checkPos(m, d)
        mujoco.mj_checkVel(m, d)
        self._position_stage()
        mujoco.mj_sensorPos(m, d)
        mujoco.mj_energyPos(m, d)
        mujoco.mj_fwdVelocity(m, d)
        mujoco.mj_sensorVel(m, d)
        mujoco.mj_energyVel(m, d)
        mujoco.mj_step2(m, d)
        self._remove_magnet_share()

    def step(self, n: int = 1) -> None:
        """Advance ``n`` physics steps (``mj_step`` semantics; RK4 falls back to Euler when split)."""
        if self._use_callback:
            with PassiveForceRegistry.active(self.data, self._passive_callback):
                for _ in range(int(n)):
                    self._step_once()
        else:
            for _ in range(int(n)):
                self._step_once()

    def forward(self) -> None:
        """``mj_forward`` including aligned contact frames and magnet forces."""
        m, d = self.model, self.data
        if not self._split and not self._use_callback:
            mujoco.mj_forward(m, d)
            return

        def _run() -> None:
            if not self._split:
                mujoco.mj_forward(m, d)
                return
            self._position_stage()
            mujoco.mj_sensorPos(m, d)
            mujoco.mj_energyPos(m, d)
            mujoco.mj_forwardSkip(m, d, mujoco.mjtStage.mjSTAGE_POS, 0)
            self._remove_magnet_share()

        if self._use_callback:
            with PassiveForceRegistry.active(d, self._passive_callback):
                _run()
        else:
            _run()

    def reset_state(self) -> None:
        """Clear pending pushes (call after ``mj_resetData``)."""
        self.pushes.clear()
        self.data.xfrc_applied[:] = 0.0


class DelayLine:
    """Ring-buffer pipeline delay at physics-step resolution.

    A command submitted at physics step ``k`` becomes active at step
    ``k + delay_steps`` and stays active until a newer one activates.
    """

    def __init__(self, size: int, delay_steps: int, initial: Optional[Sequence[float]] = None) -> None:
        if delay_steps < 0:
            raise ValueError("delay_steps must be >= 0")
        self.size = int(size)
        self.delay_steps = int(delay_steps)
        self._initial = np.zeros(self.size) if initial is None else np.asarray(initial, dtype=float).copy()
        self.reset()

    def reset(self, initial: Optional[Sequence[float]] = None) -> None:
        if initial is not None:
            self._initial = np.asarray(initial, dtype=float).copy()
        self._queue: List[Tuple[int, np.ndarray]] = []
        self._active = self._initial.copy()

    def submit(self, value: Sequence[float], now_step: int) -> None:
        self._queue.append((int(now_step) + self.delay_steps, np.asarray(value, dtype=float).copy()))

    def current(self, now_step: int) -> np.ndarray:
        while self._queue and self._queue[0][0] <= now_step:
            self._active = self._queue.pop(0)[1]
        return self._active
