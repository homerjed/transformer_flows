import math
import dataclasses 
from pathlib import Path
from shutil import rmtree
from copy import deepcopy
from typing import Tuple, List, Optional, Callable, Literal, Generator, Union
from functools import partial 

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.sharding import NamedSharding, PositionalSharding, Mesh, PartitionSpec
from jax.experimental import mesh_utils
import equinox as eqx
import optax
from jaxtyping import Array, Key, Float, Int, Bool, DTypeLike, PyTree, jaxtyped
from beartype import beartype as typechecker

from einops import rearrange
from ml_collections import ConfigDict
import matplotlib.pyplot as plt
from tqdm.auto import trange
import torchvision as tv

from attention import MultiheadAttention, self_attention


typecheck = jaxtyped(typechecker=typechecker)

KeyType = Key[jnp.ndarray, "..."]

MetricsDict = dict[
    str, Union[Float[Array, ""], Float[Array, "..."]]
]

ScalarArray = Float[Array, ""]

Leaves = List[Array]

ConditioningType = Optional[
    Literal["layernorm", "embed", "layernorm and embed"]
]

DatasetName = Literal["MNIST", "CIFAR10"]

NoiseType = Union[Literal["gaussian", "uniform"], None]

MaskArray = Union[
    Float[Array, "s s"], Int[Array, "s s"], Bool[Array, "s s"]
]

ArbitraryConditioning = Optional[
    Union[Float[Array, "..."], Int[Array, "..."]] # Flattened regardless
]


@dataclasses.dataclass(frozen=True)
class StaticLossScale:
    """ Scales and unscales by a fixed constant. """

    loss_scale: Float[Array, ""]

    def scale(self, tree: PyTree) -> PyTree:
        return jax.tree.map(lambda x: x * self.loss_scale, tree)

    def unscale(self, tree: PyTree) -> PyTree:
        return jax.tree.map(lambda x: x / self.loss_scale, tree)

    def adjust(self, grads_finite: Array):
        del grads_finite
        return self


def _cast_floating_to(tree: PyTree, dtype: DTypeLike) -> PyTree:
    def conditional_cast(x):
        # Cast only floating point arrays
        if (
            isinstance(x, jnp.ndarray) 
            and
            jnp.issubdtype(x.dtype, jnp.floating)
        ):
            x = x.astype(dtype)
        return x

    tree = jax.tree.map(conditional_cast, tree)
    return tree


@dataclasses.dataclass(frozen=True)
class Policy:
    param_dtype: Optional[DTypeLike] = None
    compute_dtype: Optional[DTypeLike] = None
    output_dtype: Optional[DTypeLike] = None

    def cast_to_param(self, x: PyTree) -> PyTree:
        if self.param_dtype is not None:
            x = _cast_floating_to(x, self.param_dtype)
        return x

    def cast_to_compute(self, x: PyTree) -> PyTree:
        if self.compute_dtype is not None:
            x = _cast_floating_to(x, self.compute_dtype) 
        return x

    def cast_to_output(self, x: PyTree) -> PyTree:
        if self.output_dtype is not None:
            x = _cast_floating_to(x, self.output_dtype)
        return x 

    def with_output_dtype(self, output_dtype: DTypeLike) -> "Policy":
        return dataclasses.replace(self, output_dtype=output_dtype)


def default(v, d):
    return v if v is not None else d


def clear_and_get_results_dir(dataset_name: str) -> Path:
    # Image save directories
    imgs_dir = Path("imgs/{}/".format(dataset_name.lower()))
    rmtree(str(imgs_dir), ignore_errors=True) # Clear old ones
    if not imgs_dir.exists():
        imgs_dir.mkdir(exist_ok=True)
        for _dir in ["samples/", "warps/", "latents/"]:
            (imgs_dir / _dir).mkdir(exist_ok=True)
    return imgs_dir


def count_parameters(model: eqx.Module) -> int:
    n_parameters = sum(
        x.size 
        for x in 
        jax.tree.leaves(
            eqx.filter(model, eqx.is_inexact_array)
        )
    )
    return n_parameters


def clip_grad_norm(grads: PyTree, max_norm: float) -> PyTree:
    leaves = jax.tree.leaves(
        jax.tree.map(jnp.linalg.norm, grads)
    )
    norm = jnp.linalg.norm(jnp.asarray(leaves))
    factor = jnp.minimum(max_norm, max_norm / (norm + 1e-6))
    grads = jax.tree.map(lambda x: x * factor, grads)
    return grads 


def add_spacing(imgs: Array, img_size: int, cols_only: bool = False) -> Array:
    h, w, c = imgs.shape # Assuming channels last
    idx = jnp.arange(img_size, h, img_size)
    if not cols_only:
        imgs  = jnp.insert(imgs, idx, jnp.nan, axis=1)
    imgs  = jnp.insert(imgs, idx, jnp.nan, axis=0)
    return imgs


def get_shardings() -> Tuple[NamedSharding, PositionalSharding]:
    devices = jax.local_devices()
    n_devices = jax.local_device_count()

    print(f"Running on {n_devices} local devices: \n\t{devices}")

    if n_devices > 1:
        mesh = Mesh(devices, ("x",))
        sharding = NamedSharding(mesh, PartitionSpec("x"))

        devices = mesh_utils.create_device_mesh((n_devices, 1))
        replicated = PositionalSharding(devices).replicate()
    else:
        sharding = replicated = None

    return sharding, replicated


def shard_batch(
    batch: Union[
        Tuple[Float[Array, "n ..."], Float[Array, "n ..."]],
        Float[Array, "n ..."]
    ], 
    sharding: Optional[NamedSharding] = None
) -> Union[
    Tuple[Float[Array, "n ..."], Float[Array, "n ..."]],
    Float[Array, "n ..."]
]:
    if sharding:
        batch = eqx.filter_shard(batch, sharding)
    return batch


def shard_model(
    model: eqx.Module,
    opt_state: Optional[optax.OptState] = None,
    sharding: Optional[PositionalSharding] = None
) -> Union[eqx.Module, Tuple[eqx.Module, optax.OptState]]:
    if sharding:
        model = eqx.filter_shard(model, sharding)
        if opt_state:
            opt_state = eqx.filter_shard(opt_state, sharding)
            return model, opt_state
        return model
    else:
        if opt_state:
            return model, opt_state
        return model


def apply_ema(
    ema_model: eqx.Module, 
    model: eqx.Module, 
    ema_rate: float,
    policy: Optional[Policy] = None
) -> eqx.Module:
    if policy is not None:
        model = policy.cast_to_param(model)
    ema_fn = lambda p_ema, p: p_ema * ema_rate + p * (1. - ema_rate)
    m_, _m = eqx.partition(model, eqx.is_inexact_array) # Current model params
    e_, _e = eqx.partition(ema_model, eqx.is_inexact_array) # Old EMA params
    e_ = jax.tree.map(ema_fn, e_, m_) # New EMA params
    return eqx.combine(e_, _m)


def precision_cast(fn, x, *args, **kwargs):
    return fn(x.astype(jnp.float32), *args, **kwargs).astype(x.dtype)


def maybe_stop_grad(a: Array, stop: bool = True) -> Array:
    return jax.lax.stop_gradient(a) if stop else a


def use_adalayernorm(
    conditioning_type: ConditioningType, 
    y_dim: Optional[int]
) -> bool:
    if conditioning_type is not None:
        if "layernorm" in conditioning_type:
            return True and (y_dim is not None)
    else:
        return False


class Linear(eqx.Module):
    weight: Float[Array, "o i"]
    bias: Float[Array, "o"]

    @typecheck
    def __init__(
        self,
        in_size: int, 
        out_size: int, 
        *, 
        zero_init_weight: bool = False, 
        key: KeyType
    ):
        key_weight, key_bias = jr.split(key)
        l = math.sqrt(1. / in_size)

        if zero_init_weight:
            self.weight = jnp.zeros((out_size, in_size))
        else:
            self.weight = jr.uniform(
                key_weight, shape=(out_size, in_size), minval=-1., maxval=1.
            ) * l

        self.bias = jr.uniform(
            key_bias, shape=(out_size,), minval=-1., maxval=1.
        ) * l

    @typecheck
    def __call__(
        self, 
        x: Float[Array, "i"], 
        key: Optional[KeyType] = None
    ) -> Float[Array, "o"]:
        return self.weight @ x + self.bias


class AdaLayerNorm(eqx.Module):
    x_dim: int
    y_dim: int

    eps: float = eqx.field(static=True)

    gamma_beta: Linear

    @typecheck
    def __init__(
        self, 
        x_dim: int, 
        y_dim: int, 
        eps: float = 1e-5,
        *, 
        key: KeyType
    ):
        self.x_dim = x_dim 
        self.y_dim = y_dim 
        self.eps = eps

        # Zero-initialised gamma and beta parameters
        self.gamma_beta = Linear(
            y_dim, x_dim * 2, zero_init_weight=True, key=key
        )

    @typecheck
    def __call__(
        self, 
        x: Float[Array, "{self.x_dim}"], 
        y: Union[Float[Array, "{self.y_dim}"], Int[Array, "{self.y_dim}"]]
    ) -> Float[Array, "{self.x_dim}"]:
        params = self.gamma_beta(y.astype(jnp.float32))  
        gamma, beta = jnp.split(params, 2, axis=-1)  

        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_normalized = (x - mean) / precision_cast(jnp.sqrt, var + self.eps)
        
        out = precision_cast(jnp.exp, gamma) * x_normalized + beta
        return out


class Attention(eqx.Module):
    patch_size: int
    n_patches: int

    n_heads: int
    head_channels: int

    norm: eqx.nn.LayerNorm | AdaLayerNorm
    attention: MultiheadAttention

    y_dim: int
    conditioning_type: ConditioningType

    @typecheck
    def __init__(
        self, 
        in_channels: int, 
        head_channels: int, 
        patch_size: int,
        n_patches: int,
        y_dim: Optional[int],
        conditioning_type: ConditioningType,
        attn_weight_bias: bool = True,
        *, 
        key: KeyType
    ):
        assert in_channels % head_channels == 0

        keys = jr.split(key)

        self.patch_size = patch_size
        self.n_patches = n_patches

        self.n_heads = int(in_channels / head_channels)
        self.head_channels = head_channels 

        self.norm = (
            AdaLayerNorm(in_channels, y_dim=y_dim, key=keys[0])
            if use_adalayernorm(conditioning_type, y_dim) 
            else eqx.nn.LayerNorm(in_channels)
        )

        self.attention = self_attention(
            self.n_heads,
            size=in_channels,
            state_length=n_patches,
            scale_factor=head_channels ** 0.5, 
            attn_weight_bias=attn_weight_bias,
            key=keys[1]
        )

        self.y_dim = y_dim
        self.conditioning_type = conditioning_type

    @typecheck
    def __call__(
        self, 
        x: Float[Array, "#s q"], # Autoregression
        y: Optional[
            Union[Float[Array, "{self.y_dim}"], Int[Array, "{self.y_dim}"]]
        ],
        mask: Optional[Union[MaskArray, Literal["causal"]]], 
        state: Optional[eqx.nn.State] 
    ) -> Tuple[
        Float[Array, "#s q"], Optional[eqx.nn.State] # Autoregression
    ]:
        if use_adalayernorm(self.conditioning_type, self.y_dim):
            _norm = partial(self.norm, y=y)
        else:
            _norm = self.norm

        x = precision_cast(jax.vmap(_norm), x) 

        a = self.attention(x, x, x, mask=mask, state=state)

        # Return updated state if it was given
        if state is None:
            x = a
        else:
            x, state = a

        return x, state 


class MLP(eqx.Module):
    y_dim: int
    conditioning_type: ConditioningType

    norm: eqx.nn.LayerNorm | AdaLayerNorm
    net: eqx.nn.Sequential

    @typecheck
    def __init__(
        self, 
        channels: int, 
        expansion: int, 
        y_dim: Optional[int],
        conditioning_type: ConditioningType,
        *, 
        key: KeyType
    ):
        keys = jr.split(key, 3)
        self.y_dim = y_dim
        self.conditioning_type = conditioning_type

        self.norm = (
            AdaLayerNorm(channels, y_dim, key=keys[0])
            if use_adalayernorm(self.conditioning_type, self.y_dim) 
            else eqx.nn.LayerNorm(channels)
        )
        self.net = eqx.nn.Sequential(
            [
                Linear(channels, channels * expansion, key=keys[1]),
                eqx.nn.Lambda(jax.nn.gelu), # NOTE: possible precision cast?
                Linear(channels * expansion, channels, key=keys[2]),
            ]
        )

    @typecheck
    def __call__(
        self, 
        x: Float[Array, "c"], 
        y: Optional[
            Union[Float[Array, "{self.y_dim}"], Int[Array, "{self.y_dim}"]]
        ]
    ) -> Float[Array, "c"]:
        if use_adalayernorm(self.conditioning_type, self.y_dim):
            x = precision_cast(self.norm, x, y)
        else: 
            x = precision_cast(self.norm, x)
        return self.net(x)


class AttentionBlock(eqx.Module):
    attention: Attention
    mlp: MLP

    n_patches: int
    sequence_dim: int
    y_dim: int

    @typecheck
    def __init__(
        self, 
        channels: int, 
        head_channels: int, 
        expansion: int, 
        patch_size: int,
        n_patches: int,
        y_dim: Optional[int] = None,
        conditioning_type: ConditioningType = None,
        *, 
        key: KeyType
    ):
        keys = jr.split(key)

        self.attention = Attention(
            channels, 
            head_channels, 
            patch_size=patch_size,
            n_patches=n_patches,
            y_dim=y_dim,
            conditioning_type=conditioning_type,
            key=keys[0]
        )

        self.mlp = MLP(
            channels, 
            expansion, 
            y_dim=y_dim, 
            conditioning_type=conditioning_type,
            key=keys[1]
        )

        self.n_patches = n_patches
        self.sequence_dim = channels
        self.y_dim = y_dim

    @typecheck
    def __call__(
        self, 
        x: Float[Array, "#{self.n_patches} {self.sequence_dim}"], # 1 patch in autoregression step
        y: Optional[
            Union[Float[Array, "{self.y_dim}"], Int[Array, "{self.y_dim}"]]
        ] = None,
        attn_mask: Optional[
            Union[
                Float[Array, "{self.n_patches} {self.n_patches}"],
                Int[Array, "{self.n_patches} {self.n_patches}"],
                Bool[Array, "{self.n_patches} {self.n_patches}"],
                Literal["causal"]
            ]
        ] = None, 
        state: Optional[eqx.nn.State] = None # No state during forward pass
    ) -> Union[
        Float[Array, "#{self.n_patches} {self.sequence_dim}"],
        Tuple[
            Float[Array, "#{self.n_patches} {self.sequence_dim}"], 
            eqx.nn.State
        ]
    ]:
        a, state = self.attention(x, y, mask=attn_mask, state=state) 

        x = x + a         
        x = x + jax.vmap(partial(self.mlp, y=y))(x)

        if state is not None:
            return x, state
        else:
            return x


class Permutation(eqx.Module):
    permute: Bool[Array, ""] 
    sequence_length: int

    @typecheck
    def __init__(
        self, 
        permute: Bool[Array, ""], 
        sequence_length: int
    ):
        self.permute = permute # Flip if true else pass
        self.sequence_length = sequence_length

    @property
    def permute_idx(self):
        return jax.lax.stop_gradient(self.permute)
    
    @typecheck
    def forward(
        self, 
        x: Float[Array, "{self.sequence_length} q"], 
        axis: int = 0
    ) -> Float[Array, "{self.sequence_length} q"]:
        x = jax.lax.select(self.permute_idx, jnp.flip(x, axis=axis), x)
        return x 
    
    @typecheck
    def reverse(
        self, 
        x: Float[Array, "{self.sequence_length} q"], 
        axis: int = 0
    ) -> Float[Array, "{self.sequence_length} q"]:
        x = jax.lax.select(self.permute_idx, jnp.flip(x, axis=axis), x)
        return x


class CausalTransformerBlock(eqx.Module):
    proj_in: Linear
    pos_embed: Float[Array, "s q"]
    class_embed: Optional[Float[Array, "c 1 q"]]
    attn_blocks: List[AttentionBlock]
    proj_out: Linear
    permutation: Permutation

    channels: int
    n_layers: int
    n_patches: int
    patch_size: int
    sequence_dim: int
    head_dim: int

    y_dim: Optional[int]

    @typecheck
    def __init__(
        self,
        in_channels: int,
        channels: int,
        n_patches: int,
        permutation: Permutation,
        n_layers: int,
        patch_size: int,
        head_dim: int,
        expansion: int,
        y_dim: Optional[int] = None,
        n_classes: Optional[int] = None,
        conditioning_type: ConditioningType = None,
        *,
        key: KeyType
    ):
        keys = jr.split(key, 5)

        self.proj_in = Linear(in_channels, channels, key=keys[0])

        self.pos_embed = jr.normal(keys[1], (n_patches, channels)) * 1e-2

        if n_classes and ("embed" in conditioning_type):
            self.class_embed = jr.normal(keys[2], (n_classes, 1, channels)) * 1e-2
        else:
            self.class_embed = None

        block_keys = jr.split(keys[3], n_layers)

        def _get_attention_block(key: KeyType) -> AttentionBlock:
            block = AttentionBlock(
                channels, 
                head_dim, 
                expansion, 
                patch_size=patch_size,
                n_patches=n_patches,
                y_dim=y_dim,
                conditioning_type=conditioning_type,
                key=key
            ) 
            return block

        self.attn_blocks = eqx.filter_vmap(_get_attention_block)(block_keys)
 
        self.proj_out = Linear(
            channels, 
            in_channels * 2, 
            zero_init_weight=True, # Initial identity mapping
            key=keys[4]
        ) 

        self.channels = channels
        self.n_layers = n_layers
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.sequence_dim = in_channels 
        self.head_dim = head_dim
        self.y_dim = y_dim
    
        self.permutation = permutation

    @typecheck
    def forward(
        self, 
        x: Float[Array, "{self.n_patches} {self.sequence_dim}"], 
        y: Optional[
            Union[Float[Array, "{self.y_dim}"], Int[Array, "{self.y_dim}"]]
        ]
    ) -> Tuple[
        Float[Array, "{self.n_patches} {self.sequence_dim}"], ScalarArray
    ]: 
        all_params, struct = eqx.partition(self.attn_blocks, eqx.is_array)

        def _block_step(x, params):
            block = eqx.combine(params, struct)
            x = block(x, y, attn_mask="causal")
            return x, None

        x_in = x.copy()

        # Permute position embedding and input together
        x = self.permutation.forward(x)
        pos_embed = self.permutation.forward(self.pos_embed) 

        # Encode each key and add positional information
        x = jax.vmap(self.proj_in)(x) + pos_embed 

        if self.class_embed is not None:
            assert y.ndim == 1 and y.dtype == jnp.int32, (
                "Class embedding defined only for scalar classing."
                "y had shape {} and type {}".format(y.shape, y.dtype)
            )
            if y is not None:
                x = x + self.class_embed[jnp.squeeze(y)]
            else:
                x = x + self.class_embed.mean(axis=0)

        x, _ = jax.lax.scan(_block_step, x, all_params)

        # Project to input channels dimension, from hidden dimension
        x = jax.vmap(self.proj_out)(x)

        # Propagate no scaling to x_0
        x = jnp.concatenate([jnp.zeros_like(x[:1]), x[:-1]], axis=0) 

        # NVP scale and shift along token dimension 
        x_a, x_b = jnp.split(x, 2, axis=-1) 

        # Shift and scale all tokens in sequence; except first and last
        u = (x_in - x_b) * precision_cast(jnp.exp, -x_a)

        u = self.permutation.reverse(u)

        return u, -x_a.mean() # Jacobian of transform on sequence

    @typecheck
    def reverse_step(
        self,
        x: Float[Array, "{self.n_patches} {self.sequence_dim}"],
        y: Optional[
            Union[Float[Array, "{self.y_dim}"], Int[Array, "{self.y_dim}"]]
        ],
        pos_embed: Float[Array, "{self.n_patches} {self.channels}"],
        s: Int[Array, ""],
        state: eqx.nn.State
    ) -> Tuple[
        Float[Array, "1 {self.sequence_dim}"], 
        Float[Array, "1 {self.sequence_dim}"], # Autoregression
        eqx.nn.State
    ]:

        all_params, struct = eqx.partition(self.attn_blocks, eqx.is_array)

        def _block_step(x, params__state):
            params, state = params__state

            block = eqx.combine(params, struct)

            x, state = block(x, y, attn_mask="causal", state=state)

            return x, state 

        # Autoregressive generation, start with s-th patch in sequence
        x_in = x[s].copy() 

        # Embed positional information to this patch
        x = (self.proj_in(x_in) + pos_embed[s])[jnp.newaxis, :] # Sequence dimension

        if self.class_embed is not None:
            assert y.ndim == 1 and y.dtype == jnp.int32, (
                "Class embedding defined only for scalar integer conditioning."
                "y had shape {} and type {}".format(y.shape, y.dtype)
            )
            if y is not None:
                x = x + self.class_embed[jnp.squeeze(y)]
            else:
                x = x + self.class_embed.mean(axis=0)

        x, state = jax.lax.scan(_block_step, x, (all_params, state)) 

        # Project to input channels dimension, from hidden dimension
        x = jax.vmap(self.proj_out)(x)

        x_a, x_b = jnp.split(x, 2, axis=-1) 

        # Shift and scale for i-th token, state with updated k/v
        return x_a, x_b, state 

    @typecheck
    def reverse(
        self,
        x: Float[Array, "{self.n_patches} {self.sequence_dim}"], 
        y: Optional[
            Union[Float[Array, "{self.y_dim}"], Int[Array, "{self.y_dim}"]]
        ],
        state: eqx.nn.State 
    ) -> Tuple[
        Float[Array, "{self.n_patches} {self.sequence_dim}"], 
        eqx.nn.State
    ]:

        def _autoregression_step(
            _x_embed_state: Tuple[
                Float[Array, "{self.n_patches} {self.sequence_dim}"], 
                eqx.nn.State
            ], 
            s: Int[Array, ""]
        ) -> Tuple[
            Tuple[
                Float[Array, "{self.n_patches} {self.sequence_dim}"],
                eqx.nn.State
            ], 
            Int[Array, ""]
        ]:
            _x, pos_embed, state = _x_embed_state

            z_a, z_b, state = self.reverse_step(
                _x, y, pos_embed=pos_embed, s=s, state=state
            )

            scale = precision_cast(jnp.exp, z_a[0])
            _x = _x.at[s + 1].set(_x[s + 1] * scale + z_b[0])

            return (_x, pos_embed, state), s

        x = self.permutation.forward(x)
        pos_embed = self.permutation.forward(self.pos_embed) 

        S = x.shape[0] 
        (x, _, state), _ = jax.lax.scan(
            _autoregression_step, 
            init=(x, pos_embed, state), 
            xs=jnp.arange(S - 1), 
            length=S - 1
        )

        x = self.permutation.reverse(x)

        return x, state 


class TransformerFlow(eqx.Module):
    blocks: List[CausalTransformerBlock]

    img_size: int
    n_channels: int

    patch_size: int
    n_patches: int
    sequence_dim: int
    n_blocks: int

    y_dim: Optional[int]
    n_classes: Optional[int]
    conditioning_type: ConditioningType

    eps_sigma: Optional[float] 

    @typecheck
    def __init__(
        self,
        in_channels: int,
        img_size: int,
        patch_size: int,
        channels: int,
        n_blocks: int,
        layers_per_block: int,
        head_dim: int = 64,
        expansion: int = 4,
        eps_sigma: Optional[float] = 0.05,
        y_dim: Optional[int] = None,
        n_classes: Optional[int] = None,
        conditioning_type: ConditioningType = None,
        *,
        key: KeyType
    ):
        self.img_size = img_size
        self.n_channels = in_channels

        self.patch_size = patch_size
        self.n_patches = int(img_size / patch_size) ** 2
        self.sequence_dim = in_channels * patch_size ** 2
        self.n_blocks = n_blocks

        self.y_dim = y_dim
        self.n_classes = n_classes
        self.conditioning_type = conditioning_type

        def _make_block(permute: bool, key: KeyType) -> CausalTransformerBlock:
            block = CausalTransformerBlock(
                self.sequence_dim,
                channels,
                n_patches=self.n_patches,
                permutation=Permutation(
                    permute=permute, sequence_length=self.n_patches
                ), 
                n_layers=layers_per_block,
                patch_size=patch_size,
                head_dim=head_dim,
                expansion=expansion,
                y_dim=y_dim,
                n_classes=n_classes,
                conditioning_type=conditioning_type,
                key=key
            )
            return block 

        block_keys = jr.split(key, n_blocks)
        permutes = (jnp.arange(n_blocks) % 2).astype(jnp.bool) # Alternate permutations
        self.blocks = eqx.filter_vmap(_make_block)(permutes, block_keys)

        self.eps_sigma = eps_sigma

        if self.eps_sigma is not None:
            assert self.eps_sigma >= 0., (
                "Noise sigma must be positive or zero."
            )

    @typecheck
    def flatten(self, return_treedef: bool = False) -> Union[Tuple[Leaves, PyTree], Leaves]:
        leaves, treedef = jax.tree.flatten(self)
        return (leaves, treedef) if return_treedef else leaves

    @typecheck
    def unflatten(self, leaves: Leaves) -> PyTree:
        treedef = self.flatten(return_treedef=True)[1]
        return jax.tree.unflatten(treedef, leaves)

    @typecheck
    def sample_prior(
        self, 
        key: KeyType, 
        n_samples: int
    ) -> Float[Array, "#n {self.n_patches} {self.sequence_dim}"]:
        return jr.normal(key, (n_samples, self.n_patches, self.sequence_dim))

    @typecheck
    def get_loss(
        self, 
        z: Float[Array, "{self.n_patches} {self.sequence_dim}"], 
        logdet: ScalarArray
    ) -> ScalarArray:
        return 0.5 * jnp.mean(jnp.square(z)) - logdet

    @typecheck
    def log_prob(
        self, 
        x: Float[Array, "{self.n_channels} {self.img_size} {self.img_size}"], 
        y: ArbitraryConditioning # Arbitrary shape conditioning is flattened
    ) -> ScalarArray:
        z, _, logdet = self.forward(x, y)
        log_prob = -self.get_loss(z, logdet)
        return log_prob
    
    @typecheck
    def denoise(
        self, 
        x: Float[Array, "{self.n_channels} {self.img_size} {self.img_size}"], 
        y: ArbitraryConditioning # Arbitrary shape conditioning is flattened
    ) -> Float[Array, "{self.n_channels} {self.img_size} {self.img_size}"]:
        score = precision_cast(jax.jacfwd(self.log_prob), x, y)
        x = x + jnp.square(self.eps_sigma) * score
        return x

    @typecheck
    def sample_model(
        self, 
        key: KeyType,
        y: ArbitraryConditioning, # Arbitrary shape conditioning is flattened
        state: eqx.nn.State,
        *,
        denoise: bool = False,
        return_sequence: bool = False,
    ) -> Union[
        Float[Array, "n _ _ _"], Float[Array, "n s _ _ _"]
    ]:
        z = self.sample_prior(key, n=1)[0] # Remove batch axis
        x = sample_model(
            self, z, y, state=state, return_sequence=return_sequence
        )
        if denoise:
            if return_sequence:
                dx = self.denoise(x[-1], y)
                x = jnp.concatenate([x, dx], axis=1)
            else:
                x = self.denoise(x, y)
        return x

    @typecheck
    def patchify(
        self, 
        x: Float[Array, "{self.n_channels} {self.img_size} {self.img_size}"]
    ) -> Float[Array, "{self.n_patches} {self.sequence_dim}"]:
        h = w = int(self.img_size / self.patch_size)
        ph = pw = self.patch_size
        u = rearrange(
            x, "c (h ph) (w pw) -> (h w) (c ph pw)", h=h, w=w, ph=ph, pw=pw
        )
        return u

    @typecheck
    def unpatchify(
        self, 
        x: Float[Array, "{self.n_patches} {self.sequence_dim}"]
    ) -> Float[Array, "{self.n_channels} {self.img_size} {self.img_size}"]:
        h = w = int(self.img_size / self.patch_size)
        ph = pw = self.patch_size
        u = rearrange(
            x, "(h w) (c ph pw) -> c (h ph) (w pw)", h=h, w=w, ph=ph, pw=pw
        ) 
        return u

    @typecheck
    def forward(
        self, 
        x: Float[Array, "{self.n_channels} {self.img_size} {self.img_size}"], 
        y: ArbitraryConditioning # Arbitrary shape conditioning is flattened
    ) -> Tuple[
        Float[Array, "{self.n_patches} {self.sequence_dim}"], 
        Float[Array, "{self.n_blocks} {self.n_patches} {self.sequence_dim}"],
        ScalarArray
    ]:
        if y is not None:
            y = y.flatten()

        all_params, struct = eqx.partition(self.blocks, eqx.is_array)

        def _block_step(x_logdet_s_sequence, params):
            x, logdet, s, sequence = x_logdet_s_sequence
            block = eqx.combine(params, struct)

            x, block_logdet = block.forward(x, y)
            logdet = logdet + block_logdet

            sequence = sequence.at[s].set(x)

            return (x, logdet, s + 1, sequence), None

        x = self.patchify(x)
        logdet = jnp.zeros((), dtype=x.dtype)
        sequence = jnp.zeros((self.n_blocks, self.n_patches, self.sequence_dim), dtype=x.dtype)

        (z, logdet, _, sequence), _ = jax.lax.scan(
            _block_step, (x, logdet, 0, sequence), all_params
        )

        return z, sequence, logdet

    @typecheck
    def reverse(
        self, 
        z: Float[Array, "{self.n_patches} {self.sequence_dim}"], 
        y: ArbitraryConditioning, # Arbitrary shape conditioning is flattened
        state: eqx.nn.State,
        return_sequence: bool = False 
    ) -> Tuple[
        Union[
            Float[Array, "{self.n_channels} {self.img_size} {self.img_size}"],
            Float[Array, "n {self.n_channels} {self.img_size} {self.img_size}"] 
        ],
        eqx.nn.State # State used in sampling
    ]:
        if y is not None:
            y = y.flatten()

        all_params, struct = eqx.partition(self.blocks, eqx.is_array)

        def _block_step(z_s_sequence, params__state):
            z, s, sequence = z_s_sequence 
            params, state = params__state
            block = eqx.combine(params, struct)

            z, state = block.reverse(z, y, state=state)

            sequence = sequence.at[s].set(self.unpatchify(z))

            return (z, s + 1, sequence), None

        sequence = jnp.zeros((self.n_blocks + 1, self.n_channels, self.img_size, self.img_size), dtype=z.dtype)
        sequence = sequence.at[0].set(self.unpatchify(z))

        (z, _, sequence), _ = jax.lax.scan(
            _block_step, (z, 1, sequence), (all_params, state), reverse=True 
        )

        x = self.unpatchify(z)

        return sequence if return_sequence else x, state


@typecheck
def single_loss_fn(
    model: TransformerFlow, 
    key: KeyType, 
    x: Float[Array, "_ _ _"], 
    y: ArbitraryConditioning, 
    policy: Optional[Policy] = None
) -> Tuple[ScalarArray, MetricsDict]:
    if policy is not None:
        x, y = policy.cast_to_compute((x, y))
        model = policy.cast_to_compute(model)

    z, _, logdet = model.forward(x, y)
    loss = model.get_loss(z, logdet)
    metrics = dict(z=jnp.mean(jnp.square(z)), latent=z, logdets=logdet)

    if policy is not None:
        loss, metrics = policy.cast_to_output((loss, metrics))
    return loss, metrics


@typecheck
def batch_loss_fn(
    model: TransformerFlow, 
    key: KeyType, 
    X: Float[Array, "n _ _ _"], 
    Y: Optional[Union[Float[Array, "n ..."], Int[Array, "n ..."]]] = None,
    policy: Optional[Policy] = None
) -> Tuple[ScalarArray, MetricsDict]:
    keys = jr.split(key, X.shape[0])
    _fn = partial(single_loss_fn, model, policy=policy)
    loss, metrics = eqx.filter_vmap(_fn)(keys, X, Y)
    metrics = jax.tree.map(
        lambda m: jnp.mean(m) if m.ndim == 1 else m, metrics
    ) 
    return jnp.mean(loss), metrics


@typecheck
@eqx.filter_jit(donate="all-except-first")
def evaluate(
    model: TransformerFlow, 
    key: KeyType, 
    x: Float[Array, "n _ _ _"], 
    y: Optional[Union[Float[Array, "n ..."], Int[Array, "n ..."]]] = None,
    *,
    policy: Optional[Policy] = None,
    sharding: Optional[NamedSharding] = None,
    replicated_sharding: Optional[PositionalSharding] = None
) -> Tuple[ScalarArray, MetricsDict]:
    model = shard_model(model, sharding=replicated_sharding)
    x, y = shard_batch((x, y), sharding=sharding)
    loss, metrics = batch_loss_fn(model, key, x, y, policy=policy)
    return loss, metrics


@typecheck
def accumulate_gradients_scan(
    model: eqx.Module,
    key: KeyType,
    x: Float[Array, "n _ _ _"], 
    y: ArbitraryConditioning,
    n_minibatches: int,
    *,
    grad_fn: Callable[
        [
            eqx.Module, 
            KeyType,
            Float[Array, "n _ _ _"],
            Optional[Float[Array, "n ..."]]
        ],
        Tuple[ScalarArray, MetricsDict]
    ]
) -> Tuple[Tuple[ScalarArray, MetricsDict], PyTree]:

    batch_size = x.shape[0]
    minibatch_size = int(batch_size / n_minibatches)

    keys = jr.split(key, n_minibatches)

    def _minibatch_step(minibatch_idx):
        # Gradients and metrics for a single minibatch
        slicer = lambda x: jax.lax.dynamic_slice_in_dim(  
            x, 
            start_index=minibatch_idx * minibatch_size, 
            slice_size=minibatch_size, 
            axis=0
        )
        _x, _y = jax.tree.map(slicer, (x, y))

        (step_L, step_metrics), step_grads = grad_fn(
            model, keys[minibatch_idx], _x, _y
        )
        return step_grads, step_L, step_metrics

    def _scan_step(carry, minibatch_idx):
        # Scan step function for looping over minibatches
        step_grads, step_L, step_metrics = _minibatch_step(minibatch_idx)
        carry = jax.tree.map(jnp.add, carry, (step_grads, step_L, step_metrics))
        return carry, None

    def _get_grads_loss_metrics_shapes():
        # Determine initial shapes for gradients and metrics.
        grads_shapes, L_shape, metrics_shape = jax.eval_shape(_minibatch_step, 0)
        grads = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), grads_shapes)
        L = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), L_shape)
        metrics = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), metrics_shape)
        return grads, L, metrics

    grads, L, metrics = _get_grads_loss_metrics_shapes()
        
    (grads, L, metrics), _ = jax.lax.scan(
        _scan_step, 
        init=(grads, L, metrics), 
        xs=jnp.arange(n_minibatches), 
        length=n_minibatches
    )

    grads = jax.tree.map(lambda g: g / n_minibatches, grads)
    metrics = jax.tree.map(lambda m: m / n_minibatches, metrics)

    return (L / n_minibatches, metrics), grads # Same signature as unaccumulated 


@typecheck
@eqx.filter_jit(donate="all")
def make_step(
    model: TransformerFlow, 
    x: Float[Array, "n _ _ _"], 
    y: Optional[Union[Float[Array, "n ..."], Int[Array, "n ..."]]], # Arbitrary conditioning shape is flattened
    key: KeyType, 
    opt_state: optax.OptState, 
    opt: optax.GradientTransformation,
    *,
    n_minibatches: Optional[int] = 4,
    accumulate_gradients: Optional[bool] = False,
    policy: Optional[Policy] = None,
    sharding: Optional[NamedSharding] = None,
    replicated_sharding: Optional[PositionalSharding] = None
) -> Tuple[
    ScalarArray, MetricsDict, TransformerFlow, optax.OptState
]:
    model, opt_state = shard_model(model, opt_state, replicated_sharding)
    x, y = shard_batch((x, y), sharding)

    grad_fn = eqx.filter_value_and_grad(
        partial(batch_loss_fn, policy=policy), has_aux=True
    )

    if policy is not None:
        model = policy.cast_to_compute(model)

    if accumulate_gradients and n_minibatches:
        (loss, metrics), grads = accumulate_gradients_scan(
            model, 
            key, 
            x, 
            y, 
            n_minibatches=n_minibatches, 
            grad_fn=grad_fn
        ) 
    else:
        (loss, metrics), grads = grad_fn(model, key, x, y)

    if policy is not None:
        grads = policy.cast_to_param(grads)
        model = policy.cast_to_param(model)

    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    return loss, metrics, model, opt_state


def get_sample_state(config: ConfigDict, key: Key) -> eqx.nn.State:
    return eqx.nn.make_with_state(TransformerFlow)(**config.model, key=key)[1]


@typecheck
@eqx.filter_jit
def sample_model(
    model: TransformerFlow, 
    z: Float[Array, "#n s q"], 
    y: Optional[Union[Float[Array, "#n ..."], Int[Array, "#n ..."]]], 
    state: eqx.nn.State,
    *,
    return_sequence: bool = False,
    denoise_samples: bool = False,
    sharding: Optional[NamedSharding] = None,
    replicated_sharding: Optional[PositionalSharding] = None
) -> Union[
    Float[Array, "#n c h w"], Float[Array, "#n t c h w"]
]:
    model = shard_model(model, sharding=replicated_sharding)
    z, y = shard_batch((z, y), sharding=sharding)

    # Sample
    sample_fn = lambda z, y: model.reverse(
        z, y, state=state, return_sequence=return_sequence
    )
    samples, state = eqx.filter_vmap(sample_fn)(z, y)

    # Denoising
    if denoise_samples:
        if return_sequence:
            denoised = jax.vmap(model.denoise)(samples[:, -1], y)
            samples = jnp.concatenate(
                [samples, denoised[:, jnp.newaxis]], axis=1
            )
        else:
            samples = jax.vmap(model.denoise)(samples, y)

    return samples


@typecheck
def loader(
    data: Float[Array, "n _ _ _"], 
    targets: Optional[Union[Float[Array, "n ..."], Int[Array, "n ..."]]], 
    batch_size: int, 
    *, 
    key: KeyType
) -> Generator[
    Union[
        Tuple[
            Float[Array, "b _ _ _"], 
            Optional[
                Union[Float[Array, "b ..."], Int[Array, "b ..."]]
            ]
        ],
    ], 
    None, 
    None
]:
    def _get_batch(perm, x, y):
        batch = (x[perm], y[perm] if y is not None else None)
        return batch

    dataset_size = data.shape[0]
    indices = jnp.arange(dataset_size)
    while True:
        key, subkey = jr.split(key, 2)
        perm = jr.permutation(subkey, indices)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield _get_batch(batch_perm, data, targets) 
            start = end
            end = start + batch_size


@typecheck
def get_data(
    key: KeyType,
    dataset_name: DatasetName,
    img_size: int, 
    n_channels: int,
    split: float = 0.9,
    use_y: bool = False,
    use_integer_labels: bool = True,
    *,
    dataset_path: Optional[str | Path] 
) -> Tuple[
    Union[
        Tuple[
            Float[Array, "t _ _ _"], 
            Optional[Union[Float[Array, "t ..."], Int[Array, "t ..."]]]
        ], 
    ],
    Union[
        Tuple[
            Float[Array, "v _ _ _"], 
            Optional[Union[Float[Array, "v ..."], Int[Array, "v ..."]]]
        ], 
    ],
    Callable[[KeyType, int], Float[Array, "_ ..."]], 
    Callable[[Float[Array, "_ _ _"]], Float[Array, "_ _ _"]]
]:
    dataset_path = default(dataset_path, "./")
    dataset_path = Path(dataset_path) if not isinstance(dataset_path, Path) else dataset_path
    dataset_path = dataset_path / "{}/".format(dataset_name.lower())

    # Get dataset, note that dataset name must match torchvision name
    dataset = getattr(tv.datasets, dataset_name) 
    dataset = dataset(dataset_path, download=True)

    target_type = jnp.int32 if use_integer_labels else jnp.float32

    if dataset_name == "MNIST":
        data = jnp.asarray(dataset.data.numpy(), dtype=jnp.uint8) 
        targets = jnp.asarray(dataset.targets.float().numpy(), dtype=target_type)
        targets = targets[:, jnp.newaxis]

        data = data / 255.
        data = data[:, jnp.newaxis, ...]
        data = data.astype(jnp.float32)

        a, b = data.min(), data.max()
        data = 2. * (data - a) / (b - a) - 1.

        def postprocess_fn(x): 
            return jnp.clip((1. + x) * 0.5 * (b - a) + a, min=0., max=1.)

        def target_fn(key, n): 
            y_range = jnp.arange(0, targets.max())
            return jr.choice(key, y_range, (n, 1)).astype(target_type)

    if dataset_name == "CIFAR10":
        data = jnp.asarray(dataset.data, dtype=jnp.uint8)
        targets = jnp.asarray(dataset.targets, dtype=target_type)
        targets = targets[:, jnp.newaxis]

        data = data / 255. 
        data = data.transpose(0, 3, 1, 2)
        data = data.astype(jnp.float32)

        a, b = data.min(), data.max()
        data = 2. * (data - a) / (b - a) - 1.

        def postprocess_fn(x): 
            return jnp.clip((1. + x) * 0.5 * (b - a) + a, min=0., max=1.)
        
        def target_fn(key, n): 
            y_range = jnp.arange(0, targets.max()) 
            return jr.choice(key, y_range, (n, 1)).astype(target_type)

    data = jax.image.resize(
        data, 
        shape=(data.shape[0], n_channels, img_size, img_size),
        method="bilinear"
    )

    print(
        "DATA:\n> {:.3E} {:.3E} {}\n> {} {}".format(
            data.min(), data.max(), data.dtype, 
            data.shape, targets.shape if targets is not None else None
        )
    )

    n_train = int(split * data.shape[0])
    x_train, x_valid = jnp.split(data, [n_train])

    if use_y:
        y_train, y_valid = jnp.split(targets, [n_train])
    else:
        y_train = y_valid = None

        # Null labels function if not using labels
        target_fn = lambda *args, **kwargs: None

    return (x_train, y_train), (x_valid, y_valid), target_fn, postprocess_fn


def add_noise(
    x: Float[Array, "... _ _ _"], 
    key: KeyType, 
    noise_type: NoiseType, 
    *, 
    eps_sigma: Optional[float]
) -> Float[Array, "... _ _ _"]:
    # Noise width is non-zero for uniform and Gaussian noise
    if eps_sigma is not None:
        if noise_type == "uniform":
            x_int = (x + 1.) * (255. / 2.)
            x = (x_int + jr.uniform(key, x_int.shape)) / 256.
            x = 2. * x - 1. 
        if noise_type == "gaussian":
            x = x + jr.normal(key, x.shape) * eps_sigma
    return x


@typecheck
def train(
    key: KeyType,
    # Model
    model: TransformerFlow,
    eps_sigma: Optional[float],
    noise_type: NoiseType,
    # Data
    dataset_name: DatasetName,
    dataset_path: Optional[str | Path],
    img_size: int,
    n_channels: int,
    use_y: bool = False,
    use_integer_labels: bool = False,
    train_split: float = 0.9,
    # Training
    batch_size: int = 256, 
    n_epochs: int = 100,
    lr: float = 2e-4,
    n_epochs_warmup: int = 1, # Cosine decay schedule 
    initial_lr: float = 1e-6, # Cosine decay schedule
    final_lr: float = 1e-6, # Cosine decay schedule
    max_grad_norm: Optional[float] = 1.0,
    use_ema: bool = False,
    ema_rate: Optional[float] = 0.9995,
    accumulate_gradients: bool = False,
    n_minibatches: Optional[int] = 4,
    policy: Optional[Policy] = None,
    # Sampling
    sample_every: int = 1000,
    n_sample: Optional[int] = 4,
    n_warps: Optional[int] = 1,
    denoise_samples: bool = False,
    get_state_fn: Callable[[None], eqx.nn.State] = None,
    cmap: Optional[str] = None,
    # Sharding: data and model
    sharding: Optional[NamedSharding] = None,
    replicated_sharding: Optional[PositionalSharding] = None
) -> TransformerFlow:

    imgs_dir = clear_and_get_results_dir(dataset_name)

    print("n_params={:.3E}".format(count_parameters(model)))

    key_data, valid_key, sample_key, *loader_keys = jr.split(key, 5)

    # Data
    (
        (x_train, y_train), 
        (x_valid, y_valid), 
        target_fn, 
        postprocess_fn
    ) = get_data(
        key_data,
        dataset_name, 
        img_size=img_size, 
        n_channels=n_channels, 
        split=train_split,
        use_y=use_y,
        use_integer_labels=use_integer_labels,
        dataset_path=dataset_path
    )

    # Optimiser & scheduler
    n_steps_per_epoch = int((x_train.shape[0] + x_valid.shape[0]) / batch_size) 
    n_steps = n_epochs * n_steps_per_epoch

    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=initial_lr, 
        peak_value=lr, 
        warmup_steps=n_epochs_warmup * n_steps_per_epoch,
        decay_steps=n_epochs * n_steps_per_epoch, 
        end_value=final_lr
    )

    opt = optax.adamw(
        learning_rate=scheduler, b1=0.9, b2=0.95, weight_decay=1e-4
    )
    if max_grad_norm is not None:
        opt = optax.chain(optax.clip_by_global_norm(max_grad_norm), opt)

    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array)) 

    # EMA model if required
    if use_ema:
        ema_model = deepcopy(model) 

    _batch_size = n_minibatches * batch_size if accumulate_gradients else batch_size

    losses, metrics = [], []
    with trange(n_steps) as bar: 
        for i, (x_t, y_t), (x_v, y_v) in zip(
            bar, 
            loader(x_train, y_train, _batch_size, key=loader_keys[0]), 
            loader(x_valid, y_valid, _batch_size, key=loader_keys[1])
        ):
            key_eps, key_step = jr.split(jr.fold_in(key, i))

            # Train 
            loss_t, metrics_t, model, opt_state = make_step(
                model, 
                add_noise(
                    x_t, key_eps, noise_type=noise_type, eps_sigma=eps_sigma
                ), 
                y_t, 
                key_step, 
                opt_state, 
                opt, 
                n_minibatches=n_minibatches,
                accumulate_gradients=accumulate_gradients,
                policy=policy,
                sharding=sharding,
                replicated_sharding=replicated_sharding
            )

            if use_ema:
                ema_model = apply_ema(ema_model, model, ema_rate, policy=policy)

            # Validate
            loss_v, metrics_v = evaluate(
                ema_model if use_ema else model, 
                valid_key, 
                add_noise(
                    x_v, key_eps, noise_type=noise_type, eps_sigma=eps_sigma
                ), 
                y_v, 
                policy=policy,
                sharding=sharding,
                replicated_sharding=replicated_sharding
            )

            # Record
            losses.append((loss_t, loss_v))
            metrics.append(
                (
                    (metrics_t["z"], metrics_v["z"]), 
                    (metrics_t["logdets"], metrics_v["logdets"])
                )
            )

            bar.set_postfix_str("Lt={} Lv={}".format(loss_t, loss_v))

            # Sample
            if (i % sample_every == 0) or (i in [10, 100, 500]):

                # Plot training data 
                if (i == 0) and (n_sample is not None):
                    x_fixed = x_t[:n_sample ** 2] # Fix first batch
                    y_fixed = y_t[:n_sample ** 2] if use_y else None

                    x_fixed_ = rearrange(
                        x_fixed, 
                        "(n1 n2) c h w -> (n1 h) (n2 w) c", 
                        n1=n_sample, 
                        n2=n_sample,
                        c=n_channels
                    )

                    plt.figure(dpi=200)
                    plt.imshow(add_spacing(postprocess_fn(x_fixed_), img_size), cmap=cmap) # NOTE: postprocessing (..., ...)
                    plt.colorbar() if cmap is not None else None
                    plt.axis("off")
                    plt.savefig(imgs_dir / "data.png", bbox_inches="tight")
                    plt.close()

                # Latents from model 
                if n_sample is not None:
                    latents_fixed, *_ = jax.vmap(model.forward)(x_fixed, y_fixed)
                    latents_fixed = jax.vmap(model.unpatchify)(latents_fixed)

                    latents_fixed = rearrange(
                        latents_fixed, 
                        "(n1 n2) c h w -> (n1 h) (n2 w) c", 
                        n1=n_sample,
                        n2=n_sample,
                        c=n_channels
                    )

                    plt.figure(dpi=200)
                    plt.imshow(add_spacing(latents_fixed, img_size), cmap=cmap)
                    plt.colorbar() if cmap is not None else None
                    plt.axis("off")
                    plt.savefig(imgs_dir / "latents/latents_{:05d}.png".format(i), bbox_inches="tight")
                    plt.close() 

                # Sample model 
                if n_sample is not None:
                    z = model.sample_prior(sample_key, n_sample ** 2) 
                    y = target_fn(sample_key, n_sample ** 2) 

                    samples = sample_model(
                        ema_model if use_ema else model, 
                        z, 
                        y, 
                        state=get_state_fn(),
                        denoise_samples=denoise_samples,
                        sharding=sharding,
                        replicated_sharding=replicated_sharding
                    )

                    samples = rearrange(
                        samples, 
                        "(n1 n2) c h w -> (n1 h) (n2 w) c", 
                        n1=n_sample,
                        n2=n_sample,
                        c=n_channels
                    )

                    plt.figure(dpi=200)
                    plt.imshow(add_spacing(postprocess_fn(samples), img_size), cmap=cmap)
                    plt.colorbar() if cmap is not None else None
                    plt.axis("off")
                    plt.savefig(imgs_dir / "samples/samples_{:05d}.png".format(i), bbox_inches="tight")
                    plt.close() 

                # Sample a warping from noise to data
                if n_warps is not None:
                    z = model.sample_prior(sample_key, n_warps)
                    y = target_fn(sample_key, n_warps) 

                    samples = sample_model(
                        ema_model if use_ema else model, 
                        z, 
                        y, 
                        state=get_state_fn(), 
                        return_sequence=True,
                        denoise_samples=denoise_samples,
                        sharding=sharding,
                        replicated_sharding=replicated_sharding
                    )

                    samples = rearrange(
                        samples, 
                        "(n1 n2) s c h w -> (n1 h) (s n2 w) c", 
                        n1=n_warps,
                        n2=1,
                        s=samples.shape[1], # Include initial noise (+ denoised if required)
                        c=n_channels
                    )

                    plt.figure(dpi=200)
                    plt.imshow(add_spacing(postprocess_fn(samples), img_size), cmap=cmap)
                    plt.axis("off")
                    plt.savefig(imgs_dir / "warps/warps_{:05d}.png".format(i), bbox_inches="tight")
                    plt.close()

                def filter_spikes(l: list, loss_max: float = 10.0) -> list[float]:
                    return [float(_l) for _l in l if _l < loss_max]

                # Losses and metrics
                fig, axs = plt.subplots(1, 3, figsize=(11., 3.))
                ax = axs[0]
                ax.plot(filter_spikes([l for l, _ in losses]), label="train") 
                ax.plot(filter_spikes([l for _, l in losses]), label="valid [ema]" if use_ema else "valid") 
                ax.set_title(r"$L$")
                ax.legend(frameon=False)
                ax = axs[1]
                ax.plot(filter_spikes([m[0][0] for m in metrics]))
                ax.plot(filter_spikes([m[0][1] for m in metrics]))
                ax.axhline(1., linestyle=":", color="k")
                ax.set_title(r"$z^2$")
                ax = axs[2]
                ax.plot(filter_spikes([m[1][0] for m in metrics]))
                ax.plot(filter_spikes([m[1][1] for m in metrics]))
                ax.set_title(r"$\sum_t^T\log|\mathbf{J}_t|$")
                for ax in axs:
                    ax.set_xscale("log")
                plt.savefig(imgs_dir / "losses.png", bbox_inches="tight")
                plt.close()

    return model


def get_config(dataset_name: str) -> ConfigDict:
    config = ConfigDict()

    config.seed                = 0

    # Data
    config.data = data = ConfigDict()
    data.dataset_name          = dataset_name
    data.dataset_path          = "/project/ls-gruen/" 
    data.n_channels            = {"CIFAR10" : 3, "MNIST" : 1}[dataset_name]
    data.img_size              = {"CIFAR10" : 32, "MNIST" : 28}[dataset_name]
    data.use_y                 = True
    data.use_integer_labels    = False # True

    # Model
    config.model = model = ConfigDict()
    model.img_size             = data.img_size
    model.in_channels          = data.n_channels
    model.patch_size           = 4 
    model.channels             = {"CIFAR10" : 512, "MNIST" : 128}[dataset_name]
    model.y_dim                = {"CIFAR10" : 1, "MNIST" : 1}[dataset_name] 
    model.n_classes            = {"CIFAR10" : 10, "MNIST" : 10}[dataset_name] 
    model.conditioning_type    = "layernorm" # "embed"
    model.n_blocks             = 3
    model.head_dim             = 64
    model.expansion            = 2
    model.layers_per_block     = 1

    if not data.use_y:
        model.y_dim = model.n_classes = None 
    else:
        if model.n_classes and ("embed" in model.conditioning_type):
            assert data.use_integer_labels, (
                "Can't use embedding with float labels!"
            )

    # Train
    config.train = train = ConfigDict()
    train.use_ema              = True
    train.ema_rate             = 0.9999 
    train.n_epochs             = 500 # Define epochs but use steps, same as paper
    train.n_epochs_warmup      = 1
    train.train_split          = 0.9
    train.batch_size           = 256
    train.initial_lr           = 1e-6
    train.lr                   = 2e-3
    train.final_lr             = 1e-6

    if not train.use_ema:
        train.ema_rate = None

    train.eps_sigma            = {"CIFAR10" : 0.05, "MNIST" : 0.1}[dataset_name]
    train.noise_type           = "gaussian"

    if train.noise_type == "uniform":
        train.eps_sigma = math.sqrt(1. / 3.) # Std of U[-1, 1]

    train.max_grad_norm        = 1.
    train.accumulate_gradients = False
    train.n_minibatches        = 4

    if not train.accumulate_gradients:
        train.n_minibatches = None

    train.sample_every         = 1000 # Steps
    train.n_sample             = jax.local_device_count() * 6
    train.n_warps              = jax.local_device_count() * 4
    train.denoise_samples      = True

    train.use_y                = data.use_y 
    train.use_integer_labels   = data.use_integer_labels
    train.dataset_name         = data.dataset_name
    train.dataset_path         = data.dataset_path
    train.img_size             = data.img_size
    train.n_channels           = data.n_channels
    train.cmap                 = {"CIFAR10" : None, "MNIST" : "gray_r"}[dataset_name]

    config.train.policy = policy = ConfigDict()
    train.use_policy           = True
    policy.param_dtype         = jnp.float32
    policy.compute_dtype       = jnp.bfloat16
    policy.output_dtype        = jnp.bfloat16

    return config


if __name__ == "__main__":

    dataset_name = "MNIST"

    config = get_config(dataset_name)

    key = jr.key(config.seed)
    key_model, key_train = jr.split(key)

    model, _ = eqx.nn.make_with_state(TransformerFlow)(**config.model, key=key_model)

    get_state_fn = partial(get_sample_state, config=config, key=key_model)

    sharding, replicated_sharding = get_shardings()

    if config.train.use_policy:
        policy = Policy(**config.train.policy)
    else:
        policy = None

    model = train(
        key_train, 
        # Model
        model, 
        eps_sigma=config.train.eps_sigma,
        noise_type=config.train.noise_type,
        # Data
        dataset_name=config.data.dataset_name,
        dataset_path=config.data.dataset_path,
        img_size=config.data.img_size,
        n_channels=config.data.n_channels,
        use_y=config.data.use_y,
        use_integer_labels=config.data.use_integer_labels,
        # Train
        train_split=config.train.train_split,
        batch_size=config.train.batch_size,
        n_epochs=config.train.n_epochs,
        lr=config.train.lr,
        n_epochs_warmup=config.train.n_epochs_warmup,
        initial_lr=config.train.initial_lr,
        final_lr=config.train.final_lr,
        max_grad_norm=config.train.max_grad_norm,
        use_ema=config.train.use_ema,
        ema_rate=config.train.ema_rate,
        accumulate_gradients=config.train.accumulate_gradients,
        n_minibatches=config.train.n_minibatches,
        # Sampling
        sample_every=config.train.sample_every,
        denoise_samples=config.train.denoise_samples,
        n_sample=config.train.n_sample,
        n_warps=config.train.n_warps,
        # Other
        cmap=config.train.cmap,
        policy=policy,
        get_state_fn=get_state_fn,
        sharding=sharding,
        replicated_sharding=replicated_sharding
    )