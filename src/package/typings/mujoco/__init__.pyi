from collections.abc import Sequence
from enum import IntEnum
from typing import ClassVar, Literal, TypeAlias
import numpy as np
from numpy.typing import NDArray

__version__: str

_F64: TypeAlias = NDArray[np.float64]
_U8: TypeAlias = NDArray[np.uint8]

class MjtObj(IntEnum):
    mjOBJ_BODY: ClassVar[MjtObj]
    mjOBJ_JOINT: ClassVar[MjtObj]
    mjOBJ_SITE: ClassVar[MjtObj]

def __getattr__(name: Literal["mjtObj"]) -> type[MjtObj]: ...

class _Opt:
    timestep: float

class _Geom:
    name: str | None

class MjModel:
    nq: int
    nv: int
    opt: _Opt
    jnt_range: _F64
    actuator_ctrlrange: _F64

    @classmethod
    def from_xml_string(cls, xml: str, assets: dict[str, bytes] | None = None) -> MjModel: ...
    def geom(self, id: int) -> _Geom: ...

class MjContact:
    geom1: int
    geom2: int

class MjData:
    def __init__(self, model: MjModel) -> None: ...
    qpos: _F64
    mocap_pos: _F64
    site_xpos: _F64
    ctrl: _F64
    time: float
    ncon: int
    contact: Sequence[MjContact]

class Renderer:
    model: MjModel
    height: int
    width: int
    def __init__(self, model: MjModel, height: int = 480, width: int = 640) -> None: ...
    def update_scene(self, data: MjData, camera: str | int | None = None) -> None: ...
    def render(self) -> _U8: ...
    def close(self) -> None: ...

def mj_name2id(model: MjModel, type: MjtObj, name: str) -> int: ...
def mj_jacSite(model: MjModel, data: MjData, jacp: _F64, jacr: _F64, site: int) -> None: ...
def mj_forward(model: MjModel, data: MjData) -> None: ...
def mj_step(model: MjModel, data: MjData) -> None: ...
def mj_resetData(model: MjModel, data: MjData) -> None: ...
def mj_contactForce(model: MjModel, data: MjData, id: int, result: _F64) -> None: ...
