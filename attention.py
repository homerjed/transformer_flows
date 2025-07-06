import functools as ft
import math
import warnings
from typing import Callable, Literal, Optional, Tuple, Union, Dict

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
from equinox.nn import Dropout, Linear, State, StateIndex
from jaxtyping import Array, Bool, Int, Float, PRNGKeyArray, PyTree, jaxtyped
from beartype import beartype as typechecker

typecheck = jaxtyped(typechecker=typechecker)


@typecheck
def standard_attention(
    query_heads: Float[Array, "q_seq num_heads q_size"],
    key_heads: Float[Array, "kv_seq num_heads k_size"],
    value_heads: Float[Array, "kv_seq num_heads v_size"],
    num_heads: int,
    mask: Optional[Bool[Array, "q_seq kv_seq"]] = None,
    dropout: Optional[Dropout] = None,
    inference: Optional[bool] = None,
    scale_factor: Optional[float] = None,
    attn_bias: Optional[Float[Array, "#q_seq kv_seq"]] = None,
    *,
    keys: Optional[PRNGKeyArray] = None,
):
    attn_fn = ft.partial(
        dot_product_attention, 
        dropout=dropout, 
        inference=inference, 
        scale_factor=scale_factor,
        attn_bias=attn_bias
    )

    in_axes = (
        1, 1, 1, 0 if mask is not None and mask.ndim == 3 else None
    )
    attn = jax.vmap(
        attn_fn, 
        in_axes=in_axes, 
        out_axes=1, 
        axis_size=num_heads
    )(
        query_heads, key_heads, value_heads, mask, key=keys
    )

    return attn


@typecheck
def dot_product_attention_weights(
    query: Float[Array, "q_seq qk_size"],
    key: Float[Array, "kv_seq qk_size"],
    mask: Optional[Bool[Array, "q_seq kv_seq"]] = None,
    scale_factor: Optional[float] = None,
    attn_bias: Optional[Float[Array, "1 kv_seq"]] = None,
) -> Float[Array, "q_seq kv_seq"]:

    if scale_factor is not None:
        query = query / scale_factor 
        key = key / scale_factor
    else:
        query = query / math.sqrt(query.shape[-1]) 
        key = key / math.sqrt(key.shape[-1]) 

    logits = jnp.einsum("sd, Sd -> sS", query, key) # QK^T

    if attn_bias is not None:
        attn_bias = jnp.broadcast_to(
            attn_bias, (query.shape[0], attn_bias.shape[-1]) 
        )
        logits = logits + attn_bias # NOTE: must mask out bias too...

    if mask is not None:
        if mask.shape != logits.shape:
            raise ValueError(
                f"mask must have shape (query_seq_length, "
                f"kv_seq_length)=({query.shape[0]}, "
                f"{key.shape[0]}). Got {mask.shape}."
            )

        logits = jnp.where(mask, logits, jnp.finfo(logits.dtype).min)
        assert isinstance(logits, Array)

    weights = jax.nn.softmax(
        (logits - jnp.max(logits)).astype(jnp.float32), axis=-1
    ).astype(query.dtype)
    
    return weights


@typecheck
def dot_product_attention(
    query: Float[Array, "q_seq qk_size"],
    key_: Float[Array, "kv_seq qk_size"],
    value: Float[Array, "kv_seq v_size"],
    mask: Optional[Bool[Array, "q_seq kv_seq"]] = None,
    dropout: Optional[Dropout] = None,
    scale_factor: Optional[float] = None,
    attn_bias: Optional[Float[Array, "1 kv_seq"]] = None,
    *,
    key: Optional[PRNGKeyArray] = None,
    inference: Optional[bool] = None,
) -> Float[Array, "q_seq v_size"]:

    weights = dot_product_attention_weights(
        query, key_, mask, scale_factor, attn_bias=attn_bias
    )

    if dropout is not None:
        weights = dropout(weights, key=key, inference=inference)

    attn = jnp.einsum("sS, Sd -> sd", weights, value)

    return attn # sigma[QK^T/s].V


def vmapped_attention(
    query_heads: Float[Array, "seq_length query_multihead_dim qk_size"],
    key_heads: Float[Array, "seq_length qk_size"],
    value_heads: Float[Array, "seq_length v_size"],
    dropout: Optional[Dropout] = None,
    inference: Optional[bool] = None,
    mask: Optional[Float[Array, "q_seq kv_seq"]] = None,
    keys: Optional[PRNGKeyArray] = None,
):

    attn_fn = ft.partial(
        dot_product_attention, 
        dropout=dropout, 
        inference=inference, 
        key=keys, 
        mask=mask
    )

    dpa = jax.vmap(
        lambda q, k, v: attn_fn(q, k, v),
        in_axes=(1, None, None),
        out_axes=1,
    )(query_heads, key_heads, value_heads)

    return dpa


class MultiheadAttention(eqx.Module):
    query_proj: Linear
    key_proj: Linear
    value_proj: Linear
    output_proj: Linear
    dropout: Dropout

    autoregressive_index: StateIndex[
        Dict[
            str,
            Tuple[
                Float[Array, "S H QK"] | Float[Array, "S QK"], 
                Float[Array, "S H VO"] | Float[Array, "S VO"], 
                Int[Array, ""],
            ]
        ]
    ]

    num_heads: int = eqx.field(static=True)
    query_size: int = eqx.field(static=True)
    key_size: int = eqx.field(static=True)
    value_size: int = eqx.field(static=True)
    output_size: int = eqx.field(static=True)

    state_length: Optional[int] = eqx.field(static=True)
    qk_size: int = eqx.field(static=True)
    vo_size: int = eqx.field(static=True)
    use_query_bias: bool = eqx.field(static=True)
    use_key_bias: bool = eqx.field(static=True)
    use_value_bias: bool = eqx.field(static=True)
    use_output_bias: bool = eqx.field(static=True)

    query_multihead_dim: int = eqx.field(static=True)
    kv_multihead_dim: int = eqx.field(static=True)

    kv_interpolation_mode: Literal["average", "repeat"] = eqx.field(static=True)
    scale_factor: Optional[float] = eqx.field(static=True)

    attn_bias: Optional[Float[Array, "1 q"]]

    @typecheck
    def __init__(
        self,
        num_heads: int,
        query_size: int,
        *,
        key_size: Optional[int] = None,
        value_size: Optional[int] = None,
        output_size: Optional[int] = None,
        query_multihead_dim: Optional[int] = None,
        kv_multihead_dim: Optional[int] = None,
        state_length: Optional[int] = None,
        qk_size: Optional[int] = None,
        vo_size: Optional[int] = None,
        use_query_bias: bool = False,
        use_key_bias: bool = False,
        use_value_bias: bool = False,
        use_output_bias: bool = False,
        dropout_p: float = 0.0,
        inference: bool = False,
        kv_interpolation_mode: Literal["average", "repeat"] = "average",
        scale_factor: Optional[float] = None,
        attn_weight_bias: bool = False,
        key: PRNGKeyArray,
        **kwargs,
    ):
        qkey, kkey, vkey, okey = jr.split(key, 4)

        if key_size is None:
            key_size = query_size
        if value_size is None:
            value_size = query_size
        if qk_size is None:
            qk_size = query_size // num_heads
        if vo_size is None:
            vo_size = query_size // num_heads
        if output_size is None:
            output_size = query_size

        def _make_autoregressive_cache(**_):
            if state_length is None:
                raise ValueError(
                    "Cannot use autoregressive decoding without specifying "
                    "`MultiheadAttention(..., state_length=...)`."
                )

            if kv_multihead_dim: 
                key_shape = (state_length, num_heads, qk_size) 
                value_shape = (state_length, num_heads, vo_size)
            else:
                key_shape = (state_length, qk_size)
                value_shape = (state_length, vo_size)

            if jax.config.jax_enable_x64:  # pyright: ignore
                _int = jnp.int64
            else:
                _int = jnp.int32

            # return jnp.empty(key_shape), jnp.empty(value_shape), jnp.zeros((), _int)
            initial_cache = (jnp.empty(key_shape), jnp.empty(value_shape), jnp.zeros((), _int))
            return dict(uncond=initial_cache, cond=initial_cache)

        query_proj_out_size = qk_size
        key_proj_out_size = qk_size
        value_proj_out_size = vo_size

        query_multihead_dim = (
            num_heads if query_multihead_dim is None else query_multihead_dim
        )
        kv_multihead_dim = num_heads if kv_multihead_dim is None else kv_multihead_dim

        query_proj_out_size = query_proj_out_size * query_multihead_dim
        key_proj_out_size = key_proj_out_size * kv_multihead_dim
        value_proj_out_size = value_proj_out_size * kv_multihead_dim

        self.query_proj = Linear(
            query_size, query_proj_out_size, use_bias=use_query_bias, key=qkey
        )
        self.key_proj = Linear(
            key_size, key_proj_out_size, use_bias=use_key_bias, key=kkey
        )
        self.value_proj = Linear(
            value_size, value_proj_out_size, use_bias=use_value_bias, key=vkey
        )

        self.output_proj = Linear(
            vo_size * num_heads, output_size, use_bias=use_output_bias, key=okey
        )
        self.dropout = Dropout(dropout_p, inference=inference)
        self.autoregressive_index = StateIndex(_make_autoregressive_cache())

        self.num_heads = num_heads
        self.query_size = query_size

        self.key_size = key_size
        self.value_size = value_size
        self.output_size = output_size
        self.qk_size = qk_size
        self.vo_size = vo_size
        self.use_query_bias = use_query_bias
        self.use_key_bias = use_key_bias
        self.use_value_bias = use_value_bias
        self.use_output_bias = use_output_bias
        self.state_length = state_length
        self.kv_multihead_dim = kv_multihead_dim
        self.query_multihead_dim = query_multihead_dim
        self.kv_interpolation_mode = kv_interpolation_mode
        self.scale_factor = scale_factor

        if attn_weight_bias:
            self.attn_bias = jnp.zeros((1, self.state_length))
        else:
            self.attn_bias = None

    @typecheck
    def __call__(
        self,
        query: Float[Array, "q_seq q_size"],
        key_: Float[Array, "kv_seq k_size"],
        value: Float[Array, "kv_seq v_size"],
        mask: Union[
            None,
            Bool[Array, "q_seq kv_seq"],
            Bool[Array, "num_heads q_seq kv_seq"],
            Literal["causal"],
        ] = None,
        state: Optional[State] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
        temperature: Optional[float] = 1.,
        which_cache: Literal["cond", "uncond"],
        inference: Optional[bool] = None,
        deterministic: Optional[bool] = None,
        process_heads: Optional[
            Callable[
                [
                    Float[Array, "seq_length num_heads qk_size"],
                    Float[Array, "seq_length num_heads qk_size"],
                    Float[Array, "seq_length num_heads vo_size"],
                ],
                Tuple[
                    Float[Array, "seq_length num_heads qk_size"],
                    Float[Array, "seq_length num_heads qk_size"],
                    Float[Array, "seq_length num_heads vo_size"],
                ],
            ]
        ] = None,
    ) -> Union[
        Float[Array, "q_seq o_size"], Tuple[Float[Array, "q_seq o_size"], State]
    ]:
        if deterministic is not None:
            inference = deterministic
            warnings.warn(
                "MultiheadAttention()(deterministic=...) is deprecated "
                "in favour of MultiheadAttention()(inference=...)"
            )

        query_seq_length, _ = query.shape
        kv_seq_length, _ = key_.shape
        kv_seq_length2, _ = value.shape
        if kv_seq_length != kv_seq_length2:
            # Query length can be different
            raise ValueError("key and value must both be sequences of equal length.")
        del kv_seq_length2

        query_heads = self._project(self.query_proj, self.query_multihead_dim, query)
        key_heads = self._project(self.key_proj, self.kv_multihead_dim, key_)
        value_heads = self._project(self.value_proj, self.kv_multihead_dim, value)

        if process_heads is not None:
            q_shape, k_shape, v_shape = (
                query_heads.shape,
                key_heads.shape,
                value_heads.shape
            )
            query_heads, key_heads, value_heads = process_heads(
                query_heads, key_heads, value_heads
            )

            if (
                query_heads.shape != q_shape
                or key_heads.shape != k_shape
                or value_heads.shape != v_shape
            ):
                raise ValueError(
                    "process_heads must not change the shape of the heads."
                )

        if state is None:
            causal_mask_offset = 0
        else:
            key_state, value_state, index = state.get(self.autoregressive_index)[which_cache]

            # If the index is larger than state length, it will wrap around and start from zero
            key_state = lax.dynamic_update_slice_in_dim(
                key_state, key_heads, index, axis=0 
            )
            value_state = lax.dynamic_update_slice_in_dim(
                value_state, value_heads, index, axis=0
            )
            
            causal_mask_offset = index # Offset shifts attention lower-tril
            index = index + kv_seq_length # i -> i + 1, nudging autoregression

            other_cache = "cond" if which_cache == "uncond" else "uncond"
            empty_cache = jax.tree.map(
                lambda x: jnp.zeros_like(x), (key_state, value_state, index)
            )
            state = state.set(
                self.autoregressive_index, 
                {which_cache : (key_state, value_state, index), other_cache : empty_cache}
            )

            # if sample:
            #     state = state.set(
            #         self.autoregressive_index, (key_state, value_state, index)
            #     )

            # The keys and values stack the preceeding keys and values, 
            # key-value sequence length updated; masking adopts this
            key_heads = key_state
            value_heads = value_state
            kv_seq_length = self.state_length # Re

        # Default to lower-tril mask matrix if no state
        if mask == "causal":
            query_indices = jnp.arange(query_seq_length)[:, jnp.newaxis]
            kv_indices = jnp.arange(kv_seq_length)[jnp.newaxis, :]
            mask = kv_indices <= query_indices + causal_mask_offset

        if state is not None:
            # Also mask out the latter parts of the state we haven't written into yet.
            unwritten_mask = jnp.arange(self.state_length) < index  # pyright: ignore
            if mask is None:
                mask = jnp.broadcast_to(
                    unwritten_mask, (query_seq_length, self.state_length) 
                )
            else:
                mask = mask & unwritten_mask # Use index to mask out where we haven't used yet (autoregression)

        keys = None if key is None else jr.split(key, self.num_heads)

        # If using default multi-head attention (these m-head dims == n_heads if not specified)
        # Normal multi-head attention
        attn = standard_attention(
            query_heads,
            key_heads,
            value_heads,
            self.num_heads,
            mask,
            self.dropout,
            inference,
            attn_bias=self.attn_bias, 
            scale_factor=self.scale_factor if temperature is None else temperature,
            keys=keys,
        )

        # Out is query_seq_length (1 or N) x output_size (=query_size if not specified)
        attn = attn.reshape(query_seq_length, self.num_heads * self.vo_size)
        out = jax.vmap(self.output_proj)(attn)

        if state is None:
            return out
        else:
            return out, state

    @typecheck
    def _project(self, proj: PyTree, multihead: int | None, x: Array) -> Array:
        seq_length, _ = x.shape
        projection = jax.vmap(proj)(x)

        if multihead is not None:
            _, projection_size = projection.shape
            size_per_head = projection_size // multihead
            projection = projection.reshape(seq_length, multihead, size_per_head)

        return projection


def self_attention(
    num_heads: int,
    size: int,
    *,
    multiquery: bool = False,
    state_length: Optional[int] = None,
    scale_factor: Optional[float] = None,
    attn_weight_bias: bool = False,
    key: PRNGKeyArray,
) -> MultiheadAttention:
    """Multi-head or multi-query attention. Also supports autoregressive decoding.

    This function is just a convenience wrapper for creating
    [`equinox.nn.MultiheadAttention`][] instances, as the full API has a great many
    options.

    **Arguments:**

    - `num_heads`: Number of parallel attention heads.
    - `size`: Number of input channels in the key, value, and query, and the number of
        channels in the output.
    - `multiquery`: if `True`, then compute multi-query rather than full multi-head
        attention. (Keyword only argument.)
    - `state_length`: Used when autoregressively decoding. This is the size of the
        key and value buffers that are updated each time the module is called. (Keyword
        only argument.)
    - `key`: A `jax.random.PRNGKeyArray` used to provide randomness for parameter
        initialisation. (Keyword only argument.)

    **Returns:**

    An [`equinox.nn.MultiheadAttention`][] instance.
    """
    return MultiheadAttention(
        num_heads=num_heads,
        query_size=size,
        state_length=state_length,
        key_multihead=not multiquery,
        value_multihead=not multiquery,
        scale_factor=scale_factor,
        attn_weight_bias=attn_weight_bias,
        key=key
    )