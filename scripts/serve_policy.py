import dataclasses
import enum
import gc
import logging
import socket

import tyro
from openpi_client import base_policy as _base_policy

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    # If true, an already-connected client can swap the served checkpoint at
    # runtime (see ReloadablePolicy below) instead of this process only ever
    # serving the one checkpoint it started with. Opt-in: this is a real
    # capability escalation (any client that can reach this port can replace
    # the running model), so it's off unless explicitly requested. Built for
    # zeyu-Pi0.5Viewer's "Select Model" button (see its backend/sim_worker.py
    # and backend/app.py for the client side).
    enable_checkpoint_reload: bool = False


class ReloadablePolicy(_base_policy.BasePolicy):
    """Wraps a Policy so its checkpoint can be swapped in-process at runtime.

    A reserved `_reload_checkpoint` key in the observation dict (instead of
    the usual real observation) triggers a reload rather than an inference
    call: `{"_reload_checkpoint": {"config": "<train config name>", "dir":
    "<checkpoint dir>"}}`, returning `{"reloaded": True, ...}` on success or
    `{"reload_error": "<message>"}` on failure -- never raises, so a bad
    request just gets reported back to the client instead of tearing down the
    whole server/connection.
    """

    def __init__(
        self, policy: _policy.Policy, *, default_prompt: str | None, checkpoint: "Checkpoint | None" = None
    ) -> None:
        self._policy = policy
        self._default_prompt = default_prompt
        # The checkpoint being served, so a failed reload can restore it (None: started from a Default policy).
        self._current = (checkpoint.config, checkpoint.dir) if checkpoint is not None else None

    @property
    def metadata(self) -> dict:
        return self._policy.metadata

    def infer(self, obs: dict) -> dict:
        reload_request = obs.get("_reload_checkpoint")
        if reload_request is not None:
            return self._reload(reload_request)
        if self._policy is None:
            raise RuntimeError("No checkpoint loaded (the last reload failed); load one with Select Model")
        return self._policy.infer(obs)

    def reset(self) -> None:
        self._policy.reset()

    def _load(self, config_name: str, checkpoint_dir: str) -> _policy.Policy:
        return _policy_config.create_trained_policy(
            _config.get_config(config_name), checkpoint_dir, default_prompt=self._default_prompt
        )

    def _reload(self, request: dict) -> dict:
        config_name, checkpoint_dir = request["config"], request["dir"]
        logging.info("Reloading checkpoint: config=%s dir=%s", config_name, checkpoint_dir)
        # Free the old checkpoint BEFORE loading the new one: its params are JAX device arrays, and the policy
        # GPU usually has room for only one checkpoint (e.g. a 24 GB card shared with Isaac Sim at
        # XLA_PYTHON_CLIENT_MEM_FRACTION=0.30). Loading first and freeing after needs two at once -> OOM.
        self._policy = None
        gc.collect()
        try:
            self._policy = self._load(config_name, checkpoint_dir)
        except Exception as exc:  # noqa: BLE001 -- reported to the client, not raised
            logging.exception("Checkpoint reload failed")
            self._policy = None
            gc.collect()
            if self._current is not None:
                # Put the previous checkpoint back so the server keeps serving something.
                try:
                    self._policy = self._load(*self._current)
                    logging.info("Restored previous checkpoint: config=%s dir=%s", *self._current)
                except Exception:  # noqa: BLE001
                    logging.exception("Restoring the previous checkpoint failed too; no policy loaded")
            return {"reload_error": str(exc)}
        self._current = (config_name, checkpoint_dir)
        logging.info("Checkpoint reload complete: config=%s dir=%s", config_name, checkpoint_dir)
        return {"reloaded": True, "config": config_name, "dir": checkpoint_dir}


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    if args.enable_checkpoint_reload:
        policy = ReloadablePolicy(
            policy,
            default_prompt=args.default_prompt,
            checkpoint=args.policy if isinstance(args.policy, Checkpoint) else None,
        )

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
