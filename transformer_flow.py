import os
import math
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
from jaxtyping import Array, Key, Float, Int, Bool, PyTree, jaxtyped
from beartype import beartype as typechecker

from einops import rearrange
from ml_collections import ConfigDict
import matplotlib.pyplot as plt
from tqdm.auto import trange
import torchvision as tv

from attention import MultiheadAttention, self_attention

KeyType = Key[jnp.ndarray, "..."]

MetricsDict = dict[str, Float[Array, ""]]

ScalarArray = Float[Array, ""]

Leaves = List[Array]

PARTITION_FILTER_TYPE = eqx.is_array 


def clear_and_get_results_dir(dataset_name: str) -> str:
    # Image save directories
    imgs_dir = "imgs/{}/".format(dataset_name.lower())
    rmtree(imgs_dir, ignore_errors=True) # Clear old ones
    if not os.path.exists(imgs_dir):
        for _dir in ["samples/", "warps/"]:
            os.makedirs(os.path.join(imgs_dir, _dir), exist_ok=True)
    return imgs_dir


def count_parameters(model: eqx.Module) -> int:
    n_parameters = sum(
        x.size 
        for x in 
        jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array))
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
    ema_rate: float
) -> eqx.Module:
    ema_fn = lambda p_ema, p: p_ema * ema_rate + p * (1. - ema_rate)
    m_, _m = eqx.partition(model, eqx.is_inexact_array) # Current model params
    e_, _e = eqx.partition(ema_model, eqx.is_inexact_array) # Old EMA params
    e_ = jax.tree.map(ema_fn, e_, m_) # New EMA params
    return eqx.combine(e_, _m)


def maybe_stop_grad(a: Array, stop: bool = True) -> Array:
    return jax.lax.stop_gradient(a) if stop else a


class Linear(eqx.Module):
    weight: Array
    bias: Array

    @jaxtyped(typechecker=typechecker)
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

    @jaxtyped(typechecker=typechecker)
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

    @jaxtyped(typechecker=typechecker)
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
        self.gamma_beta = Linear(y_dim, x_dim * 2, zero_init_weight=True, key=key) 

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, 
        x: Float[Array, "{self.x_dim}"], 
        y: Union[Float[Array, "{self.y_dim}"], Int[Array, "{self.y_dim}"]]
    ) -> Float[Array, "{self.x_dim}"]:
        params = self.gamma_beta(y.astype(jnp.float32))  
        gamma, beta = jnp.split(params, 2, axis=-1)  

        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_normalized = (x - mean) / jnp.sqrt(var + self.eps)
        
        out = jnp.exp(gamma) * x_normalized + beta
        return out


class Attention(eqx.Module):
    patch_size: int
    n_patches: int

    n_heads: int
    head_channels: int

    norm: eqx.nn.LayerNorm | AdaLayerNorm
    attention: MultiheadAttention

    y_dim: int
    conditioning_type: Optional[
        Literal["layernorm", "embed", "layernorm and embed"]
    ]

    @jaxtyped(typechecker=typechecker)
    def __init__(
        self, 
        in_channels: int, 
        head_channels: int, 
        patch_size: int,
        n_patches: int,
        y_dim: Optional[int],
        conditioning_type: Optional[
            Literal["layernorm", "embed", "layernorm and embed"]
        ] = None,
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
            if (y_dim is not None) and ("layernorm" in conditioning_type) else 
            eqx.nn.LayerNorm(in_channels)
        )

        self.attention = self_attention(
            self.n_heads,
            size=in_channels,
            state_length=n_patches,
            scale_factor=None, #head_channels ** 2., # NOTE: check this scale
            attn_weight_bias=True,
            key=keys[1]
        )

        self.y_dim = y_dim
        self.conditioning_type = conditioning_type

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, 
        x: Float[Array, "#s q"], # For autoregression
        y: Optional[
            Union[Float[Array, "{self.y_dim}"], Int[Array, "{self.y_dim}"]]
        ],
        mask: Optional[
            Union[
                Float[Array, "s s"],
                Int[Array, "s s"], 
                Bool[Array, "s s"], 
                Literal["causal"]
            ]
        ], 
        state: Optional[eqx.nn.State]
    ) -> Tuple[
        Float[Array, "#s q"], Optional[eqx.nn.State] # For autoregression
    ]:
        if (self.y_dim is not None) and ("layernorm" in self.conditioning_type):
            _norm = partial(self.norm, y=y)
        else:
            _norm = self.norm
        x = jax.vmap(_norm)(x) 

        if isinstance(mask, jax.Array):
            mask = mask.astype(jnp.bool)

        a = self.attention(x, x, x, mask=mask, state=state)

        # Return updated state if it was given
        if state is None:
            x = a
        else:
            x, state = a

        return x, state 


class MLP(eqx.Module):
    y_dim: int
    norm: eqx.nn.LayerNorm | AdaLayerNorm
    net: eqx.nn.Sequential

    conditioning_type: Literal["layernorm", "embed", "layernorm and embed"]

    @jaxtyped(typechecker=typechecker)
    def __init__(
        self, 
        channels: int, 
        expansion: int, 
        y_dim: Optional[int],
        conditioning_type: Optional[
            Literal["layernorm", "embed", "layernorm and embed"]
        ] = None,
        *, 
        key: KeyType
    ):
        keys = jr.split(key, 3)
        self.y_dim = y_dim
        self.norm = (
            AdaLayerNorm(channels, y_dim, key=keys[0])
            if (self.y_dim is not None) and ("layernorm" in self.conditioning_type) 
            else eqx.nn.LayerNorm(channels)
        )
        self.net = eqx.nn.Sequential(
            [
                Linear(channels, channels * expansion, key=keys[1]),
                eqx.nn.Lambda(jax.nn.gelu),
                Linear(channels * expansion, channels, key=keys[2]),
            ]
        )

        self.conditioning_type = conditioning_type

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, 
        x: Float[Array, "c"], 
        y: Optional[
            Union[Float[Array, "{self.y_dim}"], Int[Array, "{self.y_dim}"]]
        ]
    ) -> Float[Array, "c"]:
        return self.net(
            self.norm(x, y)
            if (self.y_dim is not None) and ("layernorm" in self.conditioning_type)
            else self.norm(x)
        )


class AttentionBlock(eqx.Module):
    attention: Attention
    mlp: MLP

    n_patches: int
    sequence_dim: int
    y_dim: int

    @jaxtyped(typechecker=typechecker)
    def __init__(
        self, 
        channels: int, 
        head_channels: int, 
        expansion: int, 
        patch_size: int,
        n_patches: int,
        y_dim: Optional[int] = None,
        conditioning_type: Optional[
            Literal["layernorm", "embed", "layernorm and embed"]
        ] = None,
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

    @jaxtyped(typechecker=typechecker)
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

    @jaxtyped(typechecker=typechecker)
    def __init__(
        self, 
        permute: Bool[Array, ""], 
        sequence_length: int
    ):
        self.permute = permute 
        self.sequence_length = sequence_length
    
    @jaxtyped(typechecker=typechecker)
    def forward(
        self, 
        x: Float[Array, "{self.sequence_length} q"], 
        axis: int = 0
    ) -> Float[Array, "{self.sequence_length} q"]:
        x = jax.lax.select(self.permute, x, jnp.flip(x, axis=axis))
        return x 
    
    @jaxtyped(typechecker=typechecker)
    def reverse(
        self, 
        x: Float[Array, "{self.sequence_length} q"], 
        axis: int = 0
    ) -> Float[Array, "{self.sequence_length} q"]:
        x = jax.lax.select(self.permute, x, jnp.flip(x, axis=axis))
        return x


class CausalTransformerBlock(eqx.Module):
    proj_in: Linear
    pos_embed: Array
    class_embed: Optional[Array]
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

    @jaxtyped(typechecker=typechecker)
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
        conditioning_type: Optional[
            Literal["layernorm", "embed", "layernorm and embed"]
        ] = None,
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

        self.channels = channels
        self.n_layers = n_layers
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.sequence_dim = in_channels 
        self.head_dim = head_dim
        self.y_dim = y_dim
    
        # Zero-init for identity mapping at first
        self.proj_out = Linear(
            channels, 
            in_channels * 2, 
            zero_init_weight=True, 
            key=keys[4]
        ) 

        self.permutation = permutation

    @jaxtyped(typechecker=typechecker)
    def forward(
        self, 
        x: Float[Array, "{self.n_patches} {self.sequence_dim}"], 
        y: Optional[
            Union[Float[Array, "{self.y_dim}"], Int[Array, "{self.y_dim}"]]
        ]
    ) -> Tuple[
        Float[Array, "{self.n_patches} {self.sequence_dim}"], ScalarArray
    ]: 
        all_params, struct = eqx.partition(self.attn_blocks, PARTITION_FILTER_TYPE)

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
            assert y.ndim == 1 and y.dtype == jnp.int_, (
                "Class embedding defined only for scalar classing."
                "y had shape {} and type {}".format(y.shape, y.dtype)
            )
            if y is not None:
                x = x + self.class_embed[y.squeeze()]
            else:
                x = x + self.class_embed.mean(dim=0)

        x, _ = jax.lax.scan(_block_step, x, all_params)

        # Project to input channels dimension, from hidden dimension
        x = jax.vmap(self.proj_out)(x)

        # Propagate no scaling to x_0
        x = jnp.concatenate([jnp.zeros_like(x[:1]), x[:-1]], axis=0) 

        # NVP scale and shift along token dimension 
        x_a, x_b = jnp.split(x, 2, axis=-1) 

        # Shift and scale all tokens in sequence; except first and last
        u = (x_in - x_b) * jnp.exp(-x_a) 

        u = self.permutation.reverse(u)

        return u, -x_a.mean() # Jacobian of transform on sequence

    @jaxtyped(typechecker=typechecker)
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
        # Autoregressive generation, start with i-th patch in sequence
        x_in = x[s].copy() 

        # Embed positional information to this patch
        x = (self.proj_in(x_in) + pos_embed[s])[jnp.newaxis, :] # Sequence dimension

        if self.class_embed is not None:
            assert y.ndim == 1 and y.dtype == jnp.int_, (
                "Class embedding defined only for scalar classing."
                "y had shape {} and type {}".format(y.shape, y.dtype)
            )
            if y is not None:
                x = x + self.class_embed[y.squeeze()]
            else:
                x = x + self.class_embed.mean(dim=0)

        all_params, struct = eqx.partition(self.attn_blocks, PARTITION_FILTER_TYPE)

        def _block_step(x, params__state):
            params, state = params__state

            block = eqx.combine(params, struct)

            x, state = block(x, y, attn_mask="causal", state=state)

            return x, None

        x, _ = jax.lax.scan(_block_step, x, (all_params, state)) 

        # Project to input channels dimension, from hidden dimension
        x = jax.vmap(self.proj_out)(x)

        x_a, x_b = jnp.split(x, 2, axis=-1) 

        # Shift and scale for i-th token, state with updated k/v
        return x_a, x_b, state 

    @jaxtyped(typechecker=typechecker)
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
        x = self.permutation.forward(x)
        pos_embed = self.permutation.forward(self.pos_embed) 

        def _autoregression_step(
            x_and_state: Tuple[
                Float[Array, "{self.n_patches} {self.sequence_dim}"], eqx.nn.State
            ], 
            s: Int[Array, ""]
        ) -> Tuple[
            Tuple[
                Float[Array, "{self.n_patches} {self.sequence_dim}"],
                eqx.nn.State
            ], 
            Int[Array, ""]
        ]:
            x, state = x_and_state

            z_a, z_b, state = self.reverse_step(
                x, y, pos_embed=pos_embed, s=s, state=state
            )

            x = x.at[s + 1].set(x[s + 1] * jnp.exp(z_a[0]) + z_b[0])

            return (x, state), s

        T = x.shape[0] 
        (x, state), _ = jax.lax.scan(
            _autoregression_step, 
            init=(x, state), 
            xs=jnp.arange(T - 1), 
            length=T - 1
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
    conditioning_type: Optional[Literal["layernorm", "embed", "layernorm and embed"]]

    eps_sigma: float

    @jaxtyped(typechecker=typechecker)
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
        eps_sigma: float = 0.05,
        y_dim: Optional[int] = None,
        n_classes: Optional[int] = None,
        conditioning_type: Optional[Literal["layernorm", "embed", "layernorm and embed"]] = None,
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
                    permute=permute, # Alternate permutations
                    sequence_length=self.n_patches
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
        self.blocks = eqx.filter_vmap(_make_block)(
            (jnp.arange(n_blocks) % 2).astype(jnp.bool), 
            block_keys
        )

        self.eps_sigma = eps_sigma

    @jaxtyped(typechecker=typechecker)
    def flatten(self, return_treedef: bool = False) -> Union[Tuple[Leaves, PyTree], Leaves]:
        leaves, treedef = jax.tree.flatten(self)
        return (leaves, treedef) if return_treedef else leaves

    @jaxtyped(typechecker=typechecker)
    def unflatten(self, leaves: Leaves) -> PyTree:
        treedef = self.flatten(return_treedef=True)[1]
        return jax.tree.unflatten(treedef, leaves)

    @jaxtyped(typechecker=typechecker)
    def sample_prior(
        self, 
        key: KeyType, 
        n_samples: int
    ) -> Float[Array, "#n {self.n_patches} {self.sequence_dim}"]:
        return jr.normal(key, (n_samples, self.n_patches, self.sequence_dim))

    @jaxtyped(typechecker=typechecker)
    def get_loss(
        self, 
        z: Float[Array, "{self.n_patches} {self.sequence_dim}"], 
        logdet: ScalarArray
    ) -> ScalarArray:
        return 0.5 * jnp.mean(jnp.square(z)) - logdet

    @jaxtyped(typechecker=typechecker)
    def log_prob(
        self, 
        x: Float[Array, "{self.n_channels} {self.img_size} {self.img_size}"], 
        y: Optional[Union[Float[Array, "..."], Int[Array, "..."]]] # Arbitrary shape conditioning is flattened
    ) -> ScalarArray:
        z, _, logdet = self.forward(x, y)
        log_prob = -self.get_loss(z, logdet)
        return log_prob
    
    @jaxtyped(typechecker=typechecker)
    def denoise(
        self, 
        x: Float[Array, "{self.n_channels} {self.img_size} {self.img_size}"], 
        y: Optional[Union[Float[Array, "..."], Int[Array, "..."]]] # Arbitrary shape conditioning is flattened
    ) -> Float[Array, "{self.n_channels} {self.img_size} {self.img_size}"]:
        score = jax.jacfwd(self.log_prob)(x, y)
        x = x + jnp.square(self.eps_sigma) * score(x)
        return x

    @jaxtyped(typechecker=typechecker)
    def sample_model(
        self, 
        key: KeyType,
        y: Optional[Union[Float[Array, "..."], Int[Array, "..."]]], # Arbitrary shape conditioning is flattened
        state: eqx.nn.State,
        *,
        return_sequence: bool = False,
    ) -> Union[
        Float[Array, "n _ _ _"], Float[Array, "n s _ _ _"]
    ]:
        z = self.sample_prior(key, n=1)[0] # Remove batch axis
        return sample_model(
            self, z, y, state=state, return_sequence=return_sequence
        )

    @jaxtyped(typechecker=typechecker)
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

    @jaxtyped(typechecker=typechecker)
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

    @jaxtyped(typechecker=typechecker)
    def forward(
        self, 
        x: Float[Array, "{self.n_channels} {self.img_size} {self.img_size}"], 
        y: Optional[Union[Float[Array, "..."], Int[Array, "..."]]] # Arbitrary shape conditioning is flattened
    ) -> Tuple[
        Float[Array, "{self.n_patches} {self.sequence_dim}"], 
        Float[Array, "{self.n_blocks} {self.n_patches} {self.sequence_dim}"],
        ScalarArray
    ]:
        if y is not None:
            y = y.flatten()

        all_params, struct = eqx.partition(self.blocks, PARTITION_FILTER_TYPE)

        def _block_step(x_logdet_s_sequence, params):
            x, logdet, s, sequence = x_logdet_s_sequence
            block = eqx.combine(params, struct)

            x, block_logdet = block.forward(x, y)
            logdet = logdet + block_logdet

            sequence = sequence.at[s].set(x)

            return (x, logdet, s + 1, sequence), None

        x = self.patchify(x)

        logdet = jnp.zeros(())
        sequence = jnp.zeros((self.n_blocks, self.n_patches, self.sequence_dim))

        (x, logdet, _, sequence), _ = jax.lax.scan(
            _block_step, (x, logdet, 0, sequence), all_params
        )

        return x, sequence, logdet

    @jaxtyped(typechecker=typechecker)
    def reverse(
        self, 
        z: Float[Array, "{self.n_patches} {self.sequence_dim}"], 
        y: Optional[Union[Float[Array, "..."], Int[Array, "..."]]], # Arbitrary shape conditioning is flattened
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

        all_params, struct = eqx.partition(self.blocks, PARTITION_FILTER_TYPE)

        def _block_step(z_s_sequence, params__state):
            z, s, sequence = z_s_sequence 
            params, state = params__state
            block = eqx.combine(params, struct)

            z, state = block.reverse(z, y, state=state)

            sequence = sequence.at[s].set(self.unpatchify(z))

            return (z, s + 1, sequence), None

        sequence = jnp.zeros((self.n_blocks + 1, self.n_channels, self.img_size, self.img_size))
        sequence = sequence.at[0].set(self.unpatchify(z))

        (z, _, sequence), _ = jax.lax.scan(
            _block_step, (z, 1, sequence), (all_params, state), reverse=True # NOTE: reverse over xs
        )

        x = self.unpatchify(z)

        return sequence if return_sequence else x, state


@jaxtyped(typechecker=typechecker)
def single_loss_fn(
    model: TransformerFlow, 
    key: KeyType, 
    x: Float[Array, "_ _ _"], 
    y: Optional[Union[Float[Array, "..."], Int[Array, "..."]]]
) -> Tuple[ScalarArray, MetricsDict]:
    z, _, logdets = model.forward(x, y)
    loss = model.get_loss(z, logdets)
    metrics = dict(z=jnp.mean(jnp.square(z)), logdets=logdets)
    return loss, metrics


@jaxtyped(typechecker=typechecker)
def batch_loss_fn(
    model: TransformerFlow, 
    key: KeyType, 
    X: Float[Array, "n _ _ _"], 
    Y: Optional[Union[Float[Array, "..."], Int[Array, "..."]]] = None
) -> Tuple[ScalarArray, MetricsDict]:
    keys = jr.split(key, X.shape[0])
    _fn = partial(single_loss_fn, model)
    loss, metrics = eqx.filter_vmap(_fn)(keys, X, Y)
    metrics = jax.tree.map(jnp.mean, metrics)
    return jnp.mean(loss), metrics


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit(donate="all-except-first")
def evaluate(
    model: TransformerFlow, 
    key: KeyType, 
    x: Float[Array, "n _ _ _"], 
    y: Optional[Union[Float[Array, "..."], Int[Array, "..."]]] = None,
    *,
    sharding: Optional[NamedSharding] = None,
    replicated_sharding: Optional[PositionalSharding] = None
) -> Tuple[ScalarArray, MetricsDict]:
    model = shard_model(model, sharding=replicated_sharding)
    x, y = shard_batch((x, y), sharding=sharding)
    loss, metrics = batch_loss_fn(model, key, x, y)
    return loss, metrics


@jaxtyped(typechecker=typechecker)
def accumulate_gradients_scan(
    model: eqx.Module,
    key: KeyType,
    x: Float[Array, "n _ _ _"], 
    y: Optional[Union[Float[Array, "..."], Int[Array, "..."]]],
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
    minibatch_size = batch_size // n_minibatches

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

    # Loop over minibatches to determine gradients and metrics.
    (grads, L, metrics) = _get_grads_loss_metrics_shapes()
    (grads, L, metrics), _ = jax.lax.scan(
        _scan_step, 
        init=(grads, L, metrics), 
        xs=jnp.arange(n_minibatches), 
        length=n_minibatches
    )

    # Average gradients/metrics over minibatches.
    grads = jax.tree.map(lambda g: g / n_minibatches, grads)
    metrics = jax.tree.map(lambda m: m / n_minibatches, metrics)

    return (L / n_minibatches, metrics), grads # Same signature as unaccumulated 


@jaxtyped(typechecker=typechecker)
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
    sharding: Optional[NamedSharding] = None,
    replicated_sharding: Optional[PositionalSharding] = None
) -> Tuple[
    ScalarArray, MetricsDict, TransformerFlow, optax.OptState
]:
    model, opt_state = shard_model(model, opt_state, replicated_sharding)
    x, y = shard_batch((x, y), sharding)

    grad_fn = eqx.filter_value_and_grad(batch_loss_fn, has_aux=True)

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

    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    return loss, metrics, model, opt_state


def get_sample_state(config: ConfigDict, key: Key) -> eqx.nn.State:
    return eqx.nn.make_with_state(TransformerFlow)(**config.model, key=key)[1]


@jaxtyped(typechecker=typechecker)
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
            samples = jnp.concatenate([samples, denoised[:, jnp.newaxis]], axis=1)
        else:
            samples = jax.vmap(model.denoise)(samples, y)

    return samples


@jaxtyped(typechecker=typechecker)
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
        if y is not None:
            batch = (x[perm], y[perm])
        else:
            batch = (x[perm], None)
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


@jaxtyped(typechecker=typechecker)
def get_data(
    dataset_name: Literal["MNIST", "CIFAR10", "FLOWERS"], 
    img_size: int, 
    n_channels: int,
    split: float = 0.9,
    use_y: bool = False,
    integer_labels: bool = True,
    *,
    dataset_path: Optional[str] 
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
    dataset_path = dataset_path if dataset_path is not None else "./" 
    dataset_path = os.path.join(dataset_path, "{}/".format(dataset_name.lower()))

    if dataset_name == "FLOWERS":
        _dataset_name = dataset_name.title() + "102"
    else:
        _dataset_name = dataset_name

    dataset = getattr(tv.datasets, _dataset_name)
    dataset = dataset(dataset_path, download=True)

    target_type = jnp.int32 if integer_labels else jnp.float32

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

        mu, std = data.mean(), data.std()
        data = (data - mu) / std

        def postprocess_fn(x): 
            return jnp.clip(mu + x * std, min=0., max=1.)
        
        def target_fn(key, n): 
            y_range = jnp.arange(0, targets.max()) 
            return jr.choice(key, y_range, (n, 1)).astype(target_type)

    if dataset_name == "FLOWERS":
        data = jnp.asarray(dataset.data, dtype=jnp.uint8)
        targets = jnp.asarray(dataset.targets, dtype=target_type)
        targets = targets[:, jnp.newaxis]

        data = data / 255.
        data = data.transpose(0, 3, 1, 2)
        data = data.astype(jnp.float32)

        mu, std = data.mean(), data.std()
        data = (data - mu) / std

        def postprocess_fn(x): 
            return jnp.clip(mu + x * std, min=0., max=1.)
        
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


@jaxtyped(typechecker=typechecker)
def train(
    key: KeyType,
    model: TransformerFlow,
    eps_sigma: float,
    # Data
    dataset_name: Literal["MNIST", "CIFAR10", "FLOWERS"],
    dataset_path: Optional[str],
    img_size: int,
    n_channels: int,
    use_y: bool = False,
    use_integer_labels: bool = False,
    # Training
    batch_size: int = 256, 
    n_epochs: int = 100,
    lr: float = 2e-4,
    n_epochs_warmup: int = 1, # Cosine decay schedule 
    initial_lr: float = 1e-6, # Cosine decay schedule
    final_lr: float = 1e-6, # Cosine decay schedule
    train_split: float = 0.9,
    max_grad_norm: Optional[float] = 1.0,
    use_ema: bool = False,
    ema_rate: float = 0.9995,
    accumulate_gradients: bool = False,
    n_minibatches: int = 4,
    # Sampling
    n_sample: Optional[int] = 4,
    n_warps: Optional[int] = 1,
    denoise_samples: bool = False,
    get_state_fn: Callable[[None], eqx.nn.State] = None,
    cmap: Optional[str] = "gray",
    # Sharding: data and model
    sharding: Optional[NamedSharding] = None,
    replicated_sharding: Optional[PositionalSharding] = None
) -> TransformerFlow:

    imgs_dir = clear_and_get_results_dir(dataset_name)

    # Data
    (
        (x_train, y_train), 
        (x_valid, y_valid), 
        target_fn, 
        postprocess_fn
    ) = get_data(
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

    valid_key, sample_key, *loader_keys = jr.split(key, 4)

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
            eps = eps_sigma * jr.normal(key_eps, x_t.shape) # Add Gaussian noise to inputs

            loss_t, metrics_t, model, opt_state = make_step(
                model, 
                x_t + eps, 
                y_t, 
                key_step, 
                opt_state, 
                opt, 
                n_minibatches=n_minibatches,
                accumulate_gradients=accumulate_gradients,
                sharding=sharding,
                replicated_sharding=replicated_sharding
            )

            if use_ema:
                ema_model = apply_ema(ema_model, model, ema_rate)

            # Validate
            eps = eps_sigma * jr.normal(valid_key, x_v.shape) # Add Gaussian noise to inputs

            loss_v, metrics_v = evaluate(
                ema_model if use_ema else model, 
                valid_key, 
                x_v + eps, 
                y_v, 
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
            bar.set_postfix_str("Lt={:.3E} Lv={:.3E}".format(loss_t, loss_v))

            # Sample
            if (i % 1000 == 0) or (i in [10, 100, 500]):

                # Plot training data 
                if (i == 0) and (n_sample is not None):
                    x_t = rearrange(
                        x_t[:n_sample ** 2], 
                        "(n1 n2) c h w -> (n1 h) (n2 w) c", 
                        n1=n_sample, 
                        n2=n_sample,
                        c=n_channels
                    )

                    plt.figure(dpi=200)
                    plt.imshow(add_spacing(postprocess_fn(x_t), img_size), cmap=cmap)
                    plt.colorbar() if cmap is not None else None
                    plt.axis("off")
                    plt.savefig(os.path.join(imgs_dir, "data.png"), bbox_inches="tight")
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
                    plt.savefig(os.path.join(imgs_dir, "samples/samples_{:05d}.png".format(i)), bbox_inches="tight")
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
                    plt.savefig(os.path.join(imgs_dir, "warps/warps_{:05d}.png".format(i)), bbox_inches="tight")
                    plt.close()

                # Losses and metrics
                fig, axs = plt.subplots(1, 3, figsize=(11., 3.))
                ax = axs[0]
                ax.plot([l for l, _ in losses if l < 10.0]) # Ignore spikes
                ax.plot([l for _, l in losses if l < 10.0]) 
                ax.set_title(r"$L$")
                ax = axs[1]
                ax.plot([m[0][0] for m in metrics])
                ax.plot([m[0][1] for m in metrics])
                ax.axhline(1., linestyle=":", color="k")
                ax.set_title(r"$z^2$")
                ax = axs[2]
                ax.plot([m[1][0] for m in metrics])
                ax.plot([m[1][1] for m in metrics])
                ax.set_title(r"$\sum_t^T\log|\mathbf{J}_t|$")
                plt.savefig(os.path.join(imgs_dir, "losses.png"), bbox_inches="tight")
                plt.close()

    return model


def get_config(dataset_name: str) -> ConfigDict:
    config = ConfigDict()

    # Data
    config.data = data = ConfigDict()
    data.dataset_name          = dataset_name
    data.dataset_path          = "/project/ls-gruen/" #users/jed.homer/
    data.n_channels            = {"CIFAR10" : 3, "MNIST" : 1, "FLOWERS" : 3}[dataset_name]
    data.img_size              = {"CIFAR10" : 32, "MNIST" : 28, "FLOWERS" : 64}[dataset_name]
    data.use_y                 = False 
    data.use_integer_labels    = True

    # Model
    config.model = model = ConfigDict()
    model.img_size             = data.img_size
    model.in_channels          = data.n_channels
    model.patch_size           = 4 
    model.channels             = {"CIFAR10" : 512, "MNIST" : 128, "FLOWERS" : 256}[dataset_name]
    model.y_dim                = {"CIFAR10" : 1, "MNIST" : 1, "FLOWERS" : 1}[dataset_name] 
    model.n_classes            = {"CIFAR10" : 10, "MNIST" : 10, "FLOWERS" : 102}[dataset_name] 
    model.conditioning_type    = "layernorm" 
    model.n_blocks             = 4
    model.head_dim             = 64
    model.expansion            = 4
    model.layers_per_block     = 4

    if not data.use_y:
        model.y_dim = model.n_classes = None 
    else:
        if model.n_classes:
            assert data.use_integer_labels, (
                "Can't use embedding with float labels!"
            )

    # Train
    config.train = train = ConfigDict()
    train.use_ema              = False 
    train.ema_rate             = 0.9995
    train.n_epochs             = 100 # Define epochs but use steps, same as paper
    train.batch_size           = 256
    train.lr                   = 2e-3

    train.eps_sigma            = {"CIFAR10" : 0.05, "MNIST" : 0.1, "FLOWERS" : 0.05}[dataset_name]

    train.max_grad_norm        = None 
    train.n_minibatches        = 0
    train.accumulate_gradients = False

    train.n_sample             = jax.local_device_count() * 4
    train.n_warps              = jax.local_device_count() * 4
    train.denoise_samples      = False

    train.use_y                = data.use_y 
    train.dataset_name         = data.dataset_name
    train.dataset_path         = data.dataset_path
    train.img_size             = data.img_size
    train.n_channels           = data.n_channels
    train.cmap                 = {"CIFAR10" : None, "MNIST" : "gray", "FLOWERS" : None}[dataset_name]

    return config


if __name__ == "__main__":
    key = jr.key(0)

    dataset_name = "MNIST"

    config = get_config(dataset_name)

    key_model, key_train = jr.split(key)

    model, _ = eqx.nn.make_with_state(TransformerFlow)(**config.model, key=key_model)

    get_state_fn = partial(get_sample_state, config=config, key=key_model)

    print("n_params={:.3E}".format(count_parameters(model)))

    sharding, replicated_sharding = get_shardings()

    model = train(
        key_train, 
        model, 
        **config.train, 
        get_state_fn=get_state_fn,
        sharding=sharding,
        replicated_sharding=replicated_sharding
    )