"""Runtime identity checks for Stanford's unchanged LSTM-GMM path.

Original symbols: TrainRobomimicImageWorkspace at Stanford commit
5ba07ac6661db573af695b419a7947ecb704690f and BC_RNN_GMM plus
RNNGMMActorNetwork at robomimic commit 62ed2de905caeb9133136e4d14d810a8b6baa96c.
"""

from __future__ import annotations

from typing import Protocol, cast

from torch import nn

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.policy.robomimic_image_policy import RobomimicImagePolicy
from diffusion_policy.workspace.train_robomimic_image_workspace import (
    TrainRobomimicImageWorkspace,
)
from robomimic.algo.bc import BC_RNN_GMM
from robomimic.models.policy_nets import RNNGMMActorNetwork


class _RnnConfig(Protocol):
    enabled: bool
    rnn_type: str


class _GmmConfig(Protocol):
    enabled: bool


class _AlgoConfig(Protocol):
    rnn: _RnnConfig
    gmm: _GmmConfig


class _RobomimicConfig(Protocol):
    algo: _AlgoConfig


class _Actor(Protocol):
    nets: nn.ModuleDict


def assert_lstm_gmm_runtime(policy: BaseImagePolicy) -> None:
    """Prove the wrapper resolved the locked recurrent mixture implementation."""
    if not isinstance(policy, RobomimicImagePolicy):
        raise TypeError("LSTM-GMM must use Stanford RobomimicImagePolicy")
    model = policy.model
    actor_object: object = policy.nets["policy"]
    config = cast("_RobomimicConfig", policy.config)
    actor = cast("_Actor", actor_object)
    rnn = cast("_RnnConfig", actor.nets["rnn"])
    if (
        type(model) is not BC_RNN_GMM
        or type(actor_object) is not RNNGMMActorNetwork
        or config.algo.rnn.enabled is not True
        or config.algo.rnn.rnn_type != "LSTM"
        or config.algo.gmm.enabled is not True
        or rnn.rnn_type != "LSTM"
    ):
        raise RuntimeError("robomimic runtime is not BC_RNN_GMM/RNNGMMActorNetwork/LSTM/GMM")
    expected = "diffusion_policy.workspace.train_robomimic_image_workspace"
    if TrainRobomimicImageWorkspace.__module__ != expected:
        raise RuntimeError("unexpected robomimic workspace origin")
