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
from openpi.shared import nnx_utils


class AttentionCapturingPolicy(_policy.Policy):
    """Like `Policy`, but `infer()` also returns `attn` and `prefix_len`, and --
    for a model with `config.aux_signal_mode=="classification_head"` -- `phase_prob`/`done_prob`.

    Requires a JAX Pi0/Pi0.5 model (the model's `sample_actions_with_attention`,
    currently only implemented on `openpi.models.pi0.Pi0`). `actions`/`state` in
    the result are computed the same way as the base `Policy` (same transform
    pipeline, same rng draw), just via `sample_actions_with_attention`/
    `sample_actions_with_aux_heads` instead of `sample_actions` -- diagnostic-only,
    doesn't change training or the standard eval path at all.

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
        itself. The client picks which layer(s) to look at. Only present when
        `_compute_attention` is truthy (see below).
      - prefix_len: int, number of columns in `attn`'s last axis that belong to
        the prefix. The language block is its last `lang_len` columns (see
        below) -- use that field, not a client-side guess, to slice it out.
        Only present alongside `attn`.
      - camera_names: list[str], the camera keys in `observation.images` in the
        same order `embed_prefix` concatenates their patch tokens into the
        prefix -- i.e. `attn`'s columns `[0:patches_per_camera]` belong to
        `camera_names[0]`, `[patches_per_camera:2*patches_per_camera]` to
        `camera_names[1]`, and so on, followed by the language columns. Only
        present alongside `attn`.
      - patches_per_camera: int, number of visual patch tokens contributed by
        each camera (all cameras share one resolution/encoder, so this is the
        same for all of them). Only present alongside `attn`.
      - lang_len: int, width of the language block (the model's configured
        max_token_len -- 48 for pi0, 200 for pi0.5), i.e. `attn`'s columns
        `[prefix_len - lang_len : prefix_len]`. This is padded (right-padded
        with masked-out positions past the real token count), not just the
        real token count, so the client must use this exact value for its
        slice rather than re-deriving one from its own re-tokenization of the
        prompt -- getting it wrong doesn't crash, it silently slices into the
        wrong columns (e.g. padding, which reads back as a hard zero, not a
        small-but-real value). Only present alongside `attn`.
      - has_state_token: bool. Only present alongside `attn`.
      - phase_prob, done_prob: float in [0, 1], the phase/sub-skill-done
        auxiliary heads' predicted probabilities (see
        `pi0.Pi0.sample_actions_with_aux_heads`) -- only present when the served
        model has `config.aux_signal_mode=="classification_head"` AND `_compute_aux_heads` is
        truthy (see below). Absent (not just False/None -- the keys don't exist
        at all) for any model trained without these heads, so a client should
        use `.get(...)`/`in` to tell "no aux heads on this checkpoint" apart
        from "heads exist but predicted exactly 0".
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
        # JIT-compiled (cached) the same way Policy.__init__ wraps the ordinary
        # sample_actions -- unlike sample_actions_with_attention (left uncompiled; its
        # cost is accepted as the price of the opt-in, off-by-default attention
        # heatmap), this is the path taken on EVERY call when the served checkpoint has
        # aux heads and attention capture is off (the default), so leaving it uncompiled
        # made aux-heads checkpoints dramatically slower than plain ones by default --
        # not the two extra linear heads themselves, which cost nothing.
        self._sample_actions_with_aux_heads = (
            nnx_utils.module_jit(self._model.sample_actions_with_aux_heads)
            if getattr(self._model, "has_aux_heads", False)
            else None
        )

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        inputs = dict(obs)
        # Reserved client-side control keys, popped before anything else touches them: let a
        # caller skip this class's extra (non-jitted, more expensive) inference paths and fall
        # back to the base Policy's ordinary infer() for this call. Both default to True so an
        # unmodified caller (e.g. examples/libero/visualize_attention.py) keeps getting
        # attention exactly as before; see zeyu-Pi0.5Viewer's "Mode" menu's "Calculate
        # attention" toggle (only current caller that ever sets _compute_attention to False)
        # and its "Show predicted phase/done labels" toggle (_compute_aux_heads).
        # `_compute_aux_heads` is further gated on the model actually having aux heads at all --
        # a plain Pi0/Pi0.5 checkpoint has nothing for it to compute regardless of the toggle.
        compute_attention = inputs.pop("_compute_attention", True)
        compute_aux = bool(inputs.pop("_compute_aux_heads", True)) and bool(getattr(self._model, "has_aux_heads", False))
        if not compute_attention and not compute_aux:
            return super().infer(inputs, noise=noise)

        inputs = jax.tree.map(lambda x: x, inputs)
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
        if compute_attention:
            actions, attn, prefix_len, phase_prob, done_prob = self._model.sample_actions_with_attention(
                sample_rng, observation, compute_aux=compute_aux, **sample_kwargs
            )
        else:
            # Attention off, aux heads on: sample_actions_with_aux_heads skips the
            # attention-capturing twin module entirely (cheaper than the branch above),
            # and -- unlike that branch -- is JIT-compiled/cached (see __init__), so this
            # is the fast path, not just the cheaper-of-two-slow-paths one.
            actions, phase_prob, done_prob = self._sample_actions_with_aux_heads(sample_rng, observation, **sample_kwargs)
        model_time = time.monotonic() - start_time

        outputs = {"state": inputs["state"], "actions": actions}
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        # Run the normal output transforms (unnormalization, LiberoOutputs' action
        # truncation, ...) on just {"state", "actions"} first -- LiberoOutputs
        # reconstructs its return dict as exactly {"actions": ...}, so any extra
        # keys added before this point would silently be dropped.
        outputs = self._output_transform(outputs)

        if compute_attention:
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

        if compute_aux:
            outputs["phase_prob"] = float(np.asarray(phase_prob)[0])
            outputs["done_prob"] = float(np.asarray(done_prob)[0])

        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        return outputs
