"""Read-only SO-101 scene loading and independent front-camera rendering."""

from __future__ import annotations
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, cast
import numpy as np
from numpy.typing import NDArray

_F64 = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
_I32 = np.ndarray[tuple[int, ...], np.dtype[np.int32]]
_U8 = np.ndarray[tuple[int, ...], np.dtype[np.uint8]]


class _Opt(Protocol):
    timestep: float


class _Geom(Protocol):
    name: str | None


class _Model(Protocol):
    nq: int
    nv: int
    opt: _Opt
    jnt_range: _F64
    jnt_qposadr: _I32
    actuator_ctrlrange: _F64

    @classmethod
    def from_xml_string(cls, xml: str, assets: dict[str, bytes] | None = None) -> _Model: ...
    def geom(self, id: int) -> _Geom: ...


class _Contact(Protocol):
    geom1: int
    geom2: int


class _Data(Protocol):
    def __init__(self, model: _Model) -> None: ...

    qpos: _F64
    qvel: _F64
    mocap_pos: _F64
    site_xpos: _F64
    xpos: _F64
    ctrl: _F64
    time: float
    ncon: int
    contact: Sequence[_Contact]


class _Renderer(Protocol):
    def __init__(self, model: _Model, height: int = 480, width: int = 640) -> None: ...
    def update_scene(
        self,
        data: _Data,
        camera: str | int | None = None,
        scene_option: object | None = None,
    ) -> None: ...
    def render(self) -> _U8: ...
    def close(self) -> None: ...
    @property
    def height(self) -> int: ...
    @property
    def width(self) -> int: ...


class _Obj(Protocol):
    mjOBJ_BODY: int
    mjOBJ_CAMERA: int
    mjOBJ_JOINT: int
    mjOBJ_SITE: int


class _Option(Protocol):
    geomgroup: NDArray[np.int8]


class _RendererFactory(Protocol):
    def __call__(self, model: _Model, height: int = 480, width: int = 640) -> _Renderer: ...


class _MuJoCo(Protocol):
    __version__: str
    MjModel: type[_Model]
    MjData: Callable[[_Model], _Data]
    Renderer: _RendererFactory
    MjvOption: Callable[[], _Option]
    mjtObj: type[_Obj]

    def mj_name2id(self, model: _Model, type: int, name: str) -> int: ...
    def mj_forward(self, model: _Model, data: _Data) -> None: ...
    def mj_step(self, model: _Model, data: _Data) -> None: ...
    def mj_resetData(self, model: _Model, data: _Data) -> None: ...
    def mj_jacSite(self, model: _Model, data: _Data, jacp: _F64, jacr: _F64, site: int) -> None: ...
    def mj_contactForce(self, model: _Model, data: _Data, id: int, result: _F64) -> None: ...


mujoco = cast(_MuJoCo, __import__("mujoco"))

PACKAGE = Path(__file__).resolve().parents[3]
PROJECT_ROOT = PACKAGE.parents[1]
OVERLAY = PACKAGE / "assets/mujoco/so101_pusht_overlay.xml"
UPSTREAM = (
    PROJECT_ROOT
    / "04_experiments/so101_pusht_benchmark/cache/upstream/so101/Simulation/SO101/so101_new_calib.xml"
)


class CameraNotFoundError(ValueError):
    pass


class SceneError(RuntimeError):
    pass


class Scene:
    mujoco: _MuJoCo
    model: _Model

    def __init__(self) -> None:
        if not OVERLAY.is_file() or not UPSTREAM.is_file():
            raise SceneError("owned overlay or pinned SO-101 model is missing")
        xml = OVERLAY.read_text(encoding="utf-8").replace(
            'file="so101_new_calib.xml"', f'file="{UPSTREAM}"'
        )
        assets = {
            path.name: path.read_bytes() for path in (UPSTREAM.parent / "assets").glob("*.stl")
        }
        try:
            self.model = mujoco.MjModel.from_xml_string(xml, assets=assets)
        except ValueError as exc:
            raise SceneError("cannot load SO-101 overlay") from exc
        self.data: _Data = mujoco.MjData(self.model)
        self.renderer: _Renderer = mujoco.Renderer(self.model, height=96, width=96)
        self.mujoco = mujoco

    def render(
        self,
        camera: str = "front",
        size: int = 96,
        *,
        hide_robot: bool | None = None,
    ) -> NDArray[np.uint8]:
        """Render one camera view.

        ``size`` is the square output resolution; the renderer is resized on
        demand. ``hide_robot`` excludes the SO-101 robot geom groups (2/3)
        from the view: the topdown camera defaults to hiding them so the
        paper-style policy image shows only table, T block, target, and
        pusher, matching the canonical PushT topdown semantics. The front
        observer camera keeps the robot visible by default.
        """
        cam_type = int(self.mujoco.mjtObj.mjOBJ_CAMERA)  # type: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
        cam_id = int(self.mujoco.mj_name2id(self.model, cam_type, camera))  # type: ignore[reportUnknownArgumentType]
        if cam_id == -1:
            raise CameraNotFoundError(f"Camera {camera!r} not found in model")
        if size != self.renderer.height or size != self.renderer.width:
            self.renderer.close()
            self.renderer = mujoco.Renderer(self.model, height=size, width=size)
        if hide_robot is None:
            hide_robot = camera == "topdown"
        if hide_robot:
            option = self.mujoco.MjvOption()
            option.geomgroup[2] = 0
            option.geomgroup[3] = 0
            self.renderer.update_scene(self.data, camera=camera, scene_option=option)
        else:
            self.renderer.update_scene(self.data, camera=camera)
        out = self.renderer.render()
        return np.array(out, dtype=np.uint8)

    def close(self) -> None:
        self.renderer.close()
