"""A `Policy` variant that also returns the action-expert's attention over the
language prompt, for visualization (see examples/libero/visualize_attention.py).

This exists as a *server-side-only* addition: LIBERO's environment and the full
openpi/JAX model package have hard-conflicting pinned dependencies (LIBERO needs
Python 3.8 + torch==1.11.0+cu113; openpi needs Python >=3.11 + torch==2.7.1) and
cannot be installed in the same environment, which is why the model is normally
served over a websocket to a separate LIBERO-side client (see
scripts/serve_policy.py / examples/libero/main.py). This class runs alongside
that same server process; only the *client* (examples/libero/visualize_attention.py)
differs from the standard eval client.
"""

from __future__ import annotations

import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi.models import model as _model
from openpi.policies import policy as _policy


class AttentionCapturingPolicy(_policy.Policy):
    """Like `Policy`, but `infer()` also returns `attn` and `prefix_len`.

    Requires a JAX Pi0/Pi0.5 model (the model's `sample_actions_with_attention`,
    currently only implemented on `openpi.models.pi0.Pi0`). `actions`/`state` in
    the result are computed the same way as the base `Policy` (same transform
    pipeline, same rng draw), just via `sample_actions_with_attention` instead of
    `sample_actions` -- diagnostic-only, doesn't change training or the standard
    eval path at all.

    Extra result keys:
      - attn: attention from the last flow-matching denoising step, averaged
        over attention heads only (*not* layers -- different layers attend to
        very different things, e.g. early layers doing lower-level pattern
        matching vs. later layers doing higher-level semantic combination, so
        blending them together washes out whichever pattern is actually
        informative), shape [depth, suffix_len, prefix_len + suffix_len]
        (float32). `suffix_len` is the action-expert's query rows (state token
        if present, then each of the action_horizon predicted steps); the key
        axis spans the full prefix (images + language) followed by the suffix
        itself. The client picks which layer(s) to look at.
      - prefix_len: int, number of columns in `attn`'s last axis that belong to
        the prefix. The language block is its last `lang_len` columns (see
        below) -- use that field, not a client-side guess, to slice it out.
      - camera_names: list[str], the camera keys in `observation.images` in the
        same order `embed_prefix` concatenates their patch tokens into the
        prefix -- i.e. `attn`'s columns `[0:patches_per_camera]` belong to
        `camera_names[0]`, `[patches_per_camera:2*patches_per_camera]` to
        `camera_names[1]`, and so on, followed by the language columns.
      - patches_per_camera: int, number of visual patch tokens contributed by
        each camera (all cameras share one resolution/encoder, so this is the
        same for all of them).
      - lang_len: int, width of the language block (the model's configured
        max_token_len -- 48 for pi0, 200 for pi0.5), i.e. `attn`'s columns
        `[prefix_len - lang_len : prefix_len]`. This is padded (right-padded
        with masked-out positions past the real token count), not just the
        real token count, so the client must use this exact value for its
        slice rather than re-deriving one from its own re-tokenization of the
        prompt -- getting it wrong doesn't crash, it silently slices into the
        wrong columns (e.g. padding, which reads back as a hard zero, not a
        small-but-real value).
    """

    def __init__(self, *args: Any, num_denoising_steps: int = 10, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if self._is_pytorch_model:
            raise NotImplementedError("AttentionCapturingPolicy requires a JAX Pi0/Pi0.5 model.")
        if not hasattr(self._model, "sample_actions_with_attention"):
            raise NotImplementedError(
                "AttentionCapturingPolicy requires a model with sample_actions_with_attention "
                "(currently only openpi.models.pi0.Pi0 -- not pi0-FAST)."
            )
        self._num_denoising_steps = num_denoising_steps

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        self._rng, sample_rng = jax.random.split(self._rng)

        sample_kwargs = dict(self._sample_kwargs)
        sample_kwargs["num_steps"] = self._num_denoising_steps
        if noise is not None:
            noise = jnp.asarray(noise)
            if noise.ndim == 2:
                noise = noise[None, ...]
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        actions, attn, prefix_len = self._model.sample_actions_with_attention(sample_rng, observation, **sample_kwargs)
        model_time = time.monotonic() - start_time

        outputs = {"state": inputs["state"], "actions": actions}
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        # Run the normal output transforms (unnormalization, LiberoOutputs' action
        # truncation, ...) on just {"state", "actions"} first -- LiberoOutputs
        # reconstructs its return dict as exactly {"actions": ...}, so any extra
        # keys added before this point would silently be dropped.
        outputs = self._output_transform(outputs)

        # attn: [depth, b=1, num_kv_heads, num_query_heads_per_kv_head, suffix_len, S]
        attn_np = np.asarray(attn[:, 0].astype(jnp.float32))
        outputs["attn"] = attn_np.mean(axis=(1, 2))  # -> [depth, suffix_len, S] -- heads averaged, layers kept
        outputs["prefix_len"] = int(prefix_len)

        # Derived without any extra model calls: all cameras share one resolution
        # and encoder, so (prefix_len - lang_len) visual-patch columns split
        # evenly across them, in observation.images' iteration order (the same
        # order embed_prefix concatenates them in).
        camera_names = list(observation.images.keys())
        lang_len = int(inputs["tokenized_prompt"].shape[-1])
        outputs["camera_names"] = camera_names
        outputs["patches_per_camera"] = (int(prefix_len) - lang_len) // len(camera_names)
        # The client re-tokenizes the prompt itself and needs to know exactly
        # how many of attn's trailing prefix columns are language columns to
        # slice them out correctly -- this is the model's configured
        # max_token_len (48 for pi0, 200 for pi0.5), which the client has no
        # way to know on its own. Getting this wrong doesn't crash anything;
        # it silently slices into the wrong columns (e.g. padding, which is
        # masked to a hard zero), so this must not be left for the client to
        # guess via its own --args.max-token-len default.
        outputs["lang_len"] = lang_len
        # The client can't otherwise tell whether attn's suffix_len includes a
        # leading state-token row (pi0) or not (pi0.5, self._model.pi05) -- it
        # has no access to the model config to infer this itself.
        outputs["has_state_token"] = not self._model.pi05
        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        return outputs
