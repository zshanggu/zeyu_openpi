"""Like scripts/serve_policy.py, but serves an `AttentionCapturingPolicy` instead
of the standard `Policy`, so the client also gets back the action-expert's
attention over the language prompt (see
src/openpi/policies/attention_policy.py and
examples/libero/visualize_attention.py).

JAX Pi0/Pi0.5 checkpoints only (not PyTorch, not pi0-FAST). Everything else --
the websocket protocol, checkpoint loading, default checkpoints -- is identical
to serve_policy.py; only the Policy class served differs.

Usage:
    uv run scripts/serve_policy_with_attention.py policy:checkpoint \
        --policy.config=pi05_libero \
        --policy.dir=gs://openpi-assets/checkpoints/pi05_libero

To override --num-denoising-steps (or --port/--default-prompt), put it *before*
the `policy:checkpoint` subcommand -- tyro scopes flags after a subcommand to
that subcommand's own fields (Checkpoint's `config`/`dir`), not Args':
    uv run scripts/serve_policy_with_attention.py --num-denoising-steps 20 \
        policy:checkpoint --policy.config=pi05_libero --policy.dir=...
"""

import dataclasses
import logging
import socket

import tyro

from openpi.policies import attention_policy as _attention_policy
from openpi.policies import policy as _policy
from openpi.serving import websocket_policy_server

# Sibling-module import: running this file directly (`uv run
# scripts/serve_policy_with_attention.py`) puts `scripts/` itself on sys.path,
# not the repo root, so `scripts` isn't importable as a package here -- only
# `serve_policy` (its sibling file) is.
from serve_policy import Args as _ServePolicyArgs
from serve_policy import create_policy as _create_base_policy


@dataclasses.dataclass
class Args(_ServePolicyArgs):
    # Number of flow-matching denoising steps to run per action chunk. Must be a
    # concrete int (unlike the base Policy's sample_kwargs, which may also accept
    # a traced array) -- see Pi0.sample_actions_with_attention's docstring.
    num_denoising_steps: int = 10


def create_policy(args: Args) -> _policy.Policy:
    base_policy = _create_base_policy(args)
    return _attention_policy.AttentionCapturingPolicy(
        base_policy._model,  # noqa: SLF001
        transforms=base_policy._input_transform.transforms,  # noqa: SLF001
        output_transforms=base_policy._output_transform.transforms,  # noqa: SLF001
        sample_kwargs=base_policy._sample_kwargs,  # noqa: SLF001
        metadata=base_policy.metadata,
        num_denoising_steps=args.num_denoising_steps,
    )


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

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
