# SO-100 Push-T native bridge contract

## Authority

The sole governing plan is
`.omo/plans/pusht-so100-four-model-clean-restart.md`. The active machine route
is `configs/workspace_status.yaml`; the dedicated native collection/evaluation
lock is `environments/sim-runtime.lock`.

## Frozen source contract

The runtime source is `05_references/external_repos/pushT-so100` at commit
`f4d6d1311bc0b43ce65458a9edd856f3c7e0a520`, with the approved F710 patch in
`src/env_human_ee.py`. The source dataset feature dictionary binds:

- `observation.images.cam_top` -> `cam_top:uint8[224,224,3]`
- `observation.images.cam_side` -> `cam_side:uint8[224,224,3]`
- `observation.state` -> `agent_pos:float32[5]`
- `action` -> `action:float32[2]`
- metadata FPS -> 10

`agent_pos` order is exactly `Rotation, Pitch, Elbow, Wrist_Pitch,
Wrist_Roll`. Action is absolute mocap XY in `[-1,1]^2`. Persisted time is
`frame_index / 10`; recorder action deduplication means original wall-clock
gaps are not recoverable and must not be claimed.

## Bridge rule

The active bridge is identity-preserving: retain both camera arrays at native
resolution and dtype, measured five-joint state, planar action, frame order,
and episode boundaries. Rename only the LeRobot source keys into the canonical
policy namespace. No resize, selected camera, Z synthesis, finite-difference
velocity, FK/end-effector derivation, sixth joint, clipping, or silent repair
is permitted.

## Input contract

The custom Logitech F710 mapping remains axes 0/1 XY, buttons 4/0 Z, axis 3
rotation, button 3 reset, button 1 record toggle, and button 7 exit. This input
mapping does not widen the persisted planar action.

## Historical material

The existing paper-view v1, `[15]` state, `[3]` XYZ action, custom `PushTEnv`,
mouse/keyboard, schema-3, 96x96, and selected-view configs/modules/tests are
historical only. They remain stored for provenance but are absent from active
config routes and CLI defaults. There is no compatibility fallback.
