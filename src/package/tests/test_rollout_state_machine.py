from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import copy
from dataclasses import replace
from threading import Barrier

import pytest

from so101_pusht_benchmark.sim_to_real import rollout_authority
from so101_pusht_benchmark.sim_to_real.rollout_authority import (
    ProcessTransitionCoordinator,
    TransitionCoordinator,
    request_transition_coordinator,
)
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation
from so101_pusht_benchmark.sim_to_real.rollout_state_machine import (
    RolloutSnapshot,
    RolloutState,
    advance,
)

COMMAND_ID = "command-1"


def _coordinator(name: str) -> ProcessTransitionCoordinator:
    return request_transition_coordinator(name)


def _armed(
    coordinator: TransitionCoordinator | None = None,
    *,
    rollout_id: str = "rollout-1",
) -> RolloutSnapshot:
    snapshot = RolloutSnapshot.received(rollout_id, coordinator=coordinator)
    snapshot = advance(snapshot, RolloutState.VALIDATING)
    return advance(snapshot, RolloutState.ARMED)


def _dispatching(
    coordinator: TransitionCoordinator,
    *,
    rollout_id: str = "rollout-1",
    command_id: str = COMMAND_ID,
) -> RolloutSnapshot:
    return advance(
        _armed(coordinator, rollout_id=rollout_id),
        RolloutState.DISPATCHING,
        command_id=command_id,
    )


def _complete(coordinator: TransitionCoordinator) -> RolloutSnapshot:
    snapshot = _dispatching(coordinator)
    snapshot = advance(snapshot, RolloutState.ACK_WAIT)
    snapshot = advance(snapshot, RolloutState.OBSERVING)
    return advance(snapshot, RolloutState.COMPLETE)


def test_nominal_one_way_transitions() -> None:
    dispatching = _dispatching(_coordinator("nominal-domain"))
    ack_wait = advance(dispatching, RolloutState.ACK_WAIT)
    complete = advance(advance(ack_wait, RolloutState.OBSERVING), RolloutState.COMPLETE)

    assert dispatching.state is RolloutState.DISPATCHING
    assert complete.state is RolloutState.COMPLETE
    assert complete.dispatched_command_ids == frozenset({COMMAND_ID})
    assert not hasattr(dispatching, "writer_capability")


def test_forbidden_skips_reversals_and_self_transitions_fail_closed() -> None:
    coordinator = _coordinator("forbidden-domain")
    received = RolloutSnapshot.received("rollout-1", coordinator=coordinator)
    validating = advance(received, RolloutState.VALIDATING)
    armed = advance(validating, RolloutState.ARMED)
    dispatching = advance(armed, RolloutState.DISPATCHING, command_id=COMMAND_ID)
    ack_wait = advance(dispatching, RolloutState.ACK_WAIT)
    observing = advance(ack_wait, RolloutState.OBSERVING)
    probes = (
        (received, RolloutState.ARMED),
        (validating, RolloutState.RECEIVED),
        (armed, RolloutState.COMPLETE),
        (dispatching, RolloutState.ARMED),
        (ack_wait, RolloutState.DISPATCHING),
        (observing, RolloutState.ACK_WAIT),
        *((snapshot, snapshot.state) for snapshot in (received, validating, armed, observing)),
    )

    for snapshot, target in probes:
        with pytest.raises(RolloutViolation) as caught:
            advance(snapshot, target, command_id="unused-command")
        assert caught.value.code is RolloutCode.R_INVALID_TRANSITION


def test_terminal_state_cannot_rearm() -> None:
    complete = _complete(_coordinator("terminal-domain"))

    with pytest.raises(RolloutViolation) as caught:
        advance(complete, RolloutState.ARMED)
    assert caught.value.code is RolloutCode.R_TERMINAL_STATE


def test_command_id_cannot_dispatch_twice() -> None:
    coordinator = _coordinator("same-command-domain")
    first_machine = _armed(coordinator, rollout_id="rollout-1")
    independent_machine = _armed(coordinator, rollout_id="rollout-2")

    first = advance(first_machine, RolloutState.DISPATCHING, command_id=COMMAND_ID)
    with pytest.raises(RolloutViolation) as caught:
        advance(independent_machine, RolloutState.DISPATCHING, command_id=COMMAND_ID)

    assert first.state is RolloutState.DISPATCHING
    assert caught.value.code is RolloutCode.R_DUPLICATE_DISPATCH


def test_three_coordinator_requests_share_one_atomic_backing() -> None:
    coordinators = tuple(
        request_transition_coordinator("factory-shared-domain") for _index in range(3)
    )
    machines = tuple(
        _armed(coordinator, rollout_id=f"factory-rollout-{index}")
        for index, coordinator in enumerate(coordinators)
    )
    advance(machines[0], RolloutState.DISPATCHING, command_id="shared-command")
    codes: list[RolloutCode] = []
    for machine in machines[1:]:
        with pytest.raises(RolloutViolation) as caught:
            advance(machine, RolloutState.DISPATCHING, command_id="shared-command")
        codes.append(caught.value.code)

    assert coordinators[0] is coordinators[1] is coordinators[2]
    assert codes == [RolloutCode.R_DUPLICATE_DISPATCH] * 2


def test_caller_created_coordinators_are_state_only_not_authority() -> None:
    first = ProcessTransitionCoordinator("caller-created")
    second = ProcessTransitionCoordinator("caller-created")
    snapshots = (
        advance(_armed(first, rollout_id="caller-1"), RolloutState.DISPATCHING, command_id="same"),
        advance(_armed(second, rollout_id="caller-2"), RolloutState.DISPATCHING, command_id="same"),
    )

    assert all(snapshot.state is RolloutState.DISPATCHING for snapshot in snapshots)
    assert all(not hasattr(snapshot, "writer_capability") for snapshot in snapshots)
    assert all(not hasattr(coordinator, "writer_capability") for coordinator in (first, second))


def test_stale_armed_revision_consumed_independently_of_command_id() -> None:
    stale_armed = _armed(_coordinator("alternate-command-domain"))
    advance(stale_armed, RolloutState.DISPATCHING, command_id="alternate-1")
    codes: list[RolloutCode] = []
    for command_id in ("alternate-2", "alternate-3"):
        with pytest.raises(RolloutViolation) as caught:
            advance(stale_armed, RolloutState.DISPATCHING, command_id=command_id)
        codes.append(caught.value.code)

    assert codes == [RolloutCode.R_STALE_TRANSITION] * 2


def test_forged_higher_and_lower_revisions_cannot_advance() -> None:
    stale_armed = _armed(_coordinator("forged-revision-domain"))
    first = advance(stale_armed, RolloutState.DISPATCHING, command_id="revision-first")
    forged_snapshots = (
        replace(stale_armed, revision=stale_armed.revision - 1),
        replace(stale_armed, revision=stale_armed.revision + 1),
    )
    codes: list[RolloutCode] = []
    for index, forged in enumerate(forged_snapshots):
        with pytest.raises(RolloutViolation) as caught:
            advance(forged, RolloutState.DISPATCHING, command_id=f"revision-forged-{index}")
        codes.append(caught.value.code)

    assert first.state is RolloutState.DISPATCHING
    assert codes == [RolloutCode.R_STALE_TRANSITION] * 2


@pytest.mark.parametrize(
    ("case_id", "command_id"),
    [
        ("empty", ""),
        ("spaces", "   "),
        ("leading", " padded"),
        ("trailing", "padded "),
        ("tab", "tab\tcommand"),
        ("fullwidth", "\uff26\uff35\uff2c\uff2c\uff37\uff29\uff24\uff34\uff28"),
        ("decomposed", "e\u0301"),
        ("uppercase", "UPPERCASE"),
        ("separator", "double--separator"),
    ],
)
def test_ambiguous_command_ids_reject_without_consuming_source_revision(
    case_id: str,
    command_id: str,
) -> None:
    armed = _armed(
        _coordinator("identifier-domain"),
        rollout_id=f"identifier-{case_id}",
    )

    with pytest.raises(RolloutViolation) as caught:
        advance(armed, RolloutState.DISPATCHING, command_id=command_id)
    assert caught.value.code is RolloutCode.R_MISSING


def test_construction_without_coordinator_has_no_writer_surface() -> None:
    armed = _armed()

    assert not hasattr(armed, "writer_capability")
    with pytest.raises(RolloutViolation) as caught:
        advance(armed, RolloutState.DISPATCHING, command_id=COMMAND_ID)
    assert caught.value.code is RolloutCode.R_POLICY_UNAUTHORIZED


@pytest.mark.parametrize(
    "state",
    [RolloutState.DISPATCHING, RolloutState.ACK_WAIT, RolloutState.OBSERVING],
)
def test_forged_inflight_reconstruction_is_rejected(state: RolloutState) -> None:
    forged = {
        "rollout_id": "forged-rollout",
        "state": state.value,
        "command_id": "forged-command",
        "dispatched_command_ids": "",
        "terminal_code": None,
    }

    with pytest.raises(RolloutViolation) as caught:
        RolloutSnapshot.from_dict(forged)
    assert caught.value.code is RolloutCode.R_INVALID_TRANSITION


def test_safe_resume_has_no_writer_surface() -> None:
    armed = _armed(_coordinator("resume-domain"))
    resumed = RolloutSnapshot.from_dict(armed.to_dict())

    assert resumed.state is RolloutState.ARMED
    assert not hasattr(resumed, "writer_capability")
    with pytest.raises(RolloutViolation) as caught:
        advance(resumed, RolloutState.DISPATCHING, command_id=COMMAND_ID)
    assert caught.value.code is RolloutCode.R_POLICY_UNAUTHORIZED


def test_ambiguous_dispatch_is_terminal_and_replay_consumed_once() -> None:
    stale_dispatching = _dispatching(_coordinator("ambiguous-domain"))
    fault = advance(
        stale_dispatching,
        RolloutState.FAULT,
        code=RolloutCode.F_AMBIGUOUS_DISPATCH,
    )
    rejection_codes: list[RolloutCode] = []
    for _attempt in range(2):
        with pytest.raises(RolloutViolation) as caught:
            advance(
                stale_dispatching,
                RolloutState.FAULT,
                code=RolloutCode.F_AMBIGUOUS_DISPATCH,
            )
        rejection_codes.append(caught.value.code)

    assert fault.state is RolloutState.FAULT
    assert rejection_codes == [RolloutCode.R_DUPLICATE_DISPATCH] * 2


def test_runtime_forgery_and_snapshot_copies_have_no_writer_capability_surface() -> None:
    issued = _coordinator("no-capability-issued")
    caller_created = ProcessTransitionCoordinator("no-capability-issued")
    structural = ProcessTransitionCoordinator("structural-coordinator")
    snapshots = tuple(
        advance(
            _armed(coordinator, rollout_id=f"no-capability-{index}"),
            RolloutState.DISPATCHING,
            command_id=f"no-capability-command-{index}",
        )
        for index, coordinator in enumerate((issued, caller_created, structural))
    )
    values = (
        *snapshots,
        copy(snapshots[0]),
        replace(snapshots[0]),
        issued,
        caller_created,
        structural,
    )

    assert not hasattr(rollout_authority, "_IssueToken")
    for value in values:
        assert not hasattr(value, "writer_capability")
        assert not hasattr(value, "writer_token")
        assert not hasattr(value, "authorization_token")


def test_same_source_thread_race_accepts_exactly_one_transition() -> None:
    stale_armed = _armed(_coordinator("thread-race-domain"))
    barrier = Barrier(20)

    def attempt(index: int) -> str:
        barrier.wait()
        try:
            advance(stale_armed, RolloutState.DISPATCHING, command_id=f"race-{index}")
        except RolloutViolation as error:
            return error.code.value
        return "ACCEPTED"

    with ThreadPoolExecutor(max_workers=20) as pool:
        results = tuple(pool.map(attempt, range(20)))

    assert results.count("ACCEPTED") == 1
    assert results.count(RolloutCode.R_STALE_TRANSITION.value) == 19
