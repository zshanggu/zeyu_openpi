import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
from flax.nnx.bridge import variables as _bridge_variables
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    def _velocity(
        self,
        observation: _model.Observation,
        x_t: at.Float[at.Array, "b ah ad"],
        time: at.Float[at.Array, ""],
        *,
        prefix_tokens: at.Float[at.Array, "b p emb"],
        prefix_mask: at.Bool[at.Array, "b p"],
        kv_cache,
        batch_size: int,
    ) -> at.Float[at.Array, "b ah ad"]:
        """Computes v_theta(x_t, t | obs) -- the flow-matching velocity field at one
        (x_t, t) pair, given a prefix already encoded into `kv_cache`. Factored out
        of `sample_actions`'s inner loop so both the forward (noise -> action) and
        reverse (action -> noise, for flow-reversal steering) integration loops can
        share the exact same computation -- this function's body is unchanged from
        what used to be inlined in `sample_actions`'s `step` closure.
        """
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size)
        )
        # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
        # other
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
        # prefix tokens
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
        # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        assert full_attn_mask.shape == (
            batch_size,
            suffix_tokens.shape[1],
            prefix_tokens.shape[1] + suffix_tokens.shape[1],
        )
        # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        assert prefix_out is None
        return self.action_out_proj(suffix_out[:, -self.action_horizon :])

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            v_t = self._velocity(
                observation,
                x_t,
                time,
                prefix_tokens=prefix_tokens,
                prefix_mask=prefix_mask,
                kv_cache=kv_cache,
                batch_size=batch_size,
            )
            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def sample_actions_steered(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        reference_action: at.Float[at.Array, "b rh ad"],
        pin_steps: int,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> _model.Actions:
        """Flow Reversal Steering (FRS): https://arxiv.org/abs/2606.13675 Sec. 4.1.

        Takes a coarse, "reasonable-but-imprecise" `reference_action` (e.g. a
        scripted Cartesian nudge, already normalized/padded the same way a real
        training action would be -- see `Policy.infer`'s handling of an optional
        `"actions"` observation key) and refines it into an in-distribution action
        via this same model's own velocity field, biased toward the reference:

          1. Tile `reference_action` (its time axis may be shorter than
             `self.action_horizon`, e.g. matching a short scripted nudge) out to
             the full action horizon.
          2. Integrate the *same* flow ODE `_velocity` computes, but in reverse
             (t=0, the tiled reference, up to t=1, an estimated noise) -- the
             literal mirror of `sample_actions`'s forward loop: same `dt`
             magnitude and step count, opposite sign, opposite start/end.
          3. "Noise-space in-painting": keep the reverse-estimated noise only for
             the first `pin_steps` timesteps (the part that actually came from a
             real reference), and resample everything after that as fresh
             `N(0,I)` -- so only the near-term nudge is actually pinned to the
             reference; the rest is left for the policy to fill in based on the
             observation, exactly like ordinary sampling would.
          4. Forward-denoise that partially-pinned noise through the *unmodified*
             `sample_actions` (which already accepts a `noise` override) to get
             the final steered action.

        Never used by training or the standard eval path -- `sample_actions`
        itself is completely untouched by this method's existence.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]

        ref_len = reference_action.shape[1]
        if ref_len < self.action_horizon:
            reps = -(-self.action_horizon // ref_len)  # ceil division
            reference_action = jnp.tile(reference_action, (1, reps, 1))[:, : self.action_horizon]
        elif ref_len > self.action_horizon:
            reference_action = reference_action[:, : self.action_horizon]

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def reverse_step(carry):
            x_t, time = carry
            v_t = self._velocity(
                observation,
                x_t,
                time,
                prefix_tokens=prefix_tokens,
                prefix_mask=prefix_mask,
                kv_cache=kv_cache,
                batch_size=batch_size,
            )
            # Mirror image of sample_actions's forward step (x_t + dt * v_t, time + dt):
            # same magnitude, opposite sign, since we're integrating the same ODE the
            # other way (t=0, the reference action, up to t=1, noise).
            return x_t - dt * v_t, time - dt

        def reverse_cond(carry):
            _x_t, time = carry
            return time <= 1 + dt / 2

        estimated_noise, _ = jax.lax.while_loop(reverse_cond, reverse_step, (reference_action, 0.0))

        resample_rng, denoise_rng = jax.random.split(rng)
        fresh_noise = jax.random.normal(resample_rng, estimated_noise.shape)
        step_idx = jnp.arange(self.action_horizon)[None, :, None]
        pinned_noise = jnp.where(step_idx < pin_steps, estimated_noise, fresh_noise)

        return self.sample_actions(denoise_rng, observation, num_steps=num_steps, noise=pinned_noise)

    def _llm_with_attn(self):
        """Builds a "twin" of self.PaliGemma.llm's underlying Linen module with
        capture_attn=True baked in, sharing the exact same (already-trained)
        params, so it can be called via `.apply(..., method="forward_with_attn")`
        to also get attention weights out. See `gemma.Module.capture_attn`.

        This never touches `self.PaliGemma.llm` itself, so the normal
        `compute_loss`/`sample_actions` call paths are completely unaffected.
        """
        original = self.PaliGemma.llm.module
        twin = dataclasses.replace(original, capture_attn=True)
        nnx_attrs = {name: getattr(self.PaliGemma.llm, name) for name in self.PaliGemma.llm.linen_attributes}
        variables = _bridge_variables.nnx_attrs_to_linen_vars(nnx_attrs)
        return twin, variables

    def sample_actions_with_attention(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> tuple[_model.Actions, at.Array, int]:
        """Like `sample_actions`, but also returns the action-expert's attention
        over the prefix (image + language tokens) from the *last* flow-matching
        denoising step.

        Diagnostic/visualization use only -- not used by training or the
        standard eval path. Returns:
          - actions: same as `sample_actions`.
          - attn: attention weights from the final denoising step, shape
            [depth, b, num_kv_heads, num_query_heads_per_kv_head, suffix_len,
            prefix_len + suffix_len]. Averaging over the leading two head axes
            and slicing key-columns [prefix_len - language_len : prefix_len]
            gives per-language-token attention for each of the `suffix_len`
            query rows (state token if present, then the action_horizon steps).
          - prefix_len: number of key/value columns in `attn` that belong to
            the prefix (images + language) -- language occupies the last
            `observation.tokenized_prompt.shape[1]` of those columns.

        `num_steps` must be a concrete Python int here (unlike `sample_actions`,
        which also accepts a traced array): capturing every denoising step
        would require rewriting the flow-matching loop from `lax.while_loop` to
        `lax.scan` (while_loop has no mechanism to stack per-iteration outputs,
        only to carry a fixed-shape value to the next iteration -- discovered
        empirically while implementing this), so instead this carries the
        latest step's attention forward and returns whichever step happened to
        be last, which requires a statically-known number of iterations.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        llm_twin, llm_variables = self._llm_with_attn()

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_len = prefix_tokens.shape[1]
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache, _ = llm_twin.apply(
            llm_variables, [prefix_tokens, None], positions, prefix_attn_mask, method="forward_with_attn"
        )

        # Precompute the exact shape/dtype of one step's attention array analytically
        # (rather than via an extra warmup forward pass), so it can seed the
        # while_loop's carry -- lax.while_loop requires the carry's pytree
        # structure/shapes to be fixed across iterations.
        suffix_tokens_probe, _, _, _ = self.embed_suffix(observation, noise, jnp.ones((batch_size,)))
        suffix_len = suffix_tokens_probe.shape[1]
        num_kv_heads = llm_twin.configs[0].num_kv_heads
        num_query_heads = llm_twin.configs[0].num_heads
        depth = llm_twin.configs[0].depth
        attn_shape = (depth, batch_size, num_kv_heads, num_query_heads // num_kv_heads, suffix_len, prefix_len + suffix_len)
        # probs is cast to the model's compute dtype inside Attention.__call__ (e.g.
        # bfloat16), not necessarily float32 -- match it so the while_loop carry's
        # dtype is consistent across iterations.
        initial_attn = jnp.zeros(attn_shape, dtype=jnp.dtype(llm_twin.embed_dtype))

        def step(carry):
            x_t, time, _last_attn = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_b = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_b, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _, attn = llm_twin.apply(
                llm_variables,
                [None, suffix_tokens],
                positions,
                full_attn_mask,
                [None, adarms_cond],
                kv_cache=kv_cache,
                method="forward_with_attn",
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt, attn

        def cond(carry):
            _x_t, time, _attn = carry
            return time >= -dt / 2

        x_0, _, last_attn = jax.lax.while_loop(cond, step, (noise, 1.0, initial_attn))
        return x_0, last_attn, prefix_len
