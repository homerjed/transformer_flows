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
from jaxtyping import Array, Key, Float, Int, Bool, PyTree, jaxtyped
from beartype import beartype as typechecker
import optax
from einops import rearrange
from ml_collections import ConfigDict
import matplotlib.pyplot as plt
from tqdm.auto import trange
import torchvision as tv

from attention import MultiheadAttention, self_attention

e = os.environ.get("DEBUG")
DEBUG = bool(int(e if e is not None else 0))


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
        jax.tree.leaves(eqx.filter(model, eqx.is_array))
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
    h, w, c = imgs.shape # Assuming channels last, square grid.
    idx = jnp.arange(img_size, h, img_size)
    if cols_only:
        imgs  = jnp.insert(imgs, idx, jnp.nan, axis=1)
    imgs  = jnp.insert(imgs, idx, jnp.nan, axis=0)
    return imgs


def get_shardings() -> Tuple[NamedSharding, PositionalSharding]:
    devices = jax.local_devices()
    n_devices = len(devices)

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


def apply_ema(
    ema_model: eqx.Module, 
    model: eqx.Module, 
    ema_rate: float
) -> eqx.Module:
    # Parameters of ema and model
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

    def __init__(
        self,
        in_size: int, 
        out_size: int, 
        *, 
        zero_init_weight: bool = False, 
        key: Key[jnp.ndarray, "..."]
    ):
        key_weight, key_bias = jr.split(key)
        if zero_init_weight:
            self.weight = jnp.zeros((out_size, in_size))
        else:
            self.weight = jr.uniform(
                key_weight, shape=(out_size, in_size), minval=-1., maxval=1.
            ) * math.sqrt(1. / in_size)
        self.bias = jr.uniform(
            key_bias, shape=(out_size,), minval=-1., maxval=1.
        ) * math.sqrt(1. / in_size)

    def __call__(
        self, 
        x: Float[Array, "i"], 
        key: Optional[Key[jnp.ndarray, "..."]] = None
    ) -> Float[Array, "o"]:
        return self.weight @ x + self.bias


class AdaLayerNorm(eqx.Module):
    x_dim: int
    y_dim: int
    gamma_beta: eqx.nn.Linear
    eps: float

    def __init__(
        self, x_dim: int, y_dim: int, *, key: Key[jnp.ndarray, "..."]
    ):
        self.x_dim = x_dim 
        self.y_dim = y_dim 
        self.gamma_beta = Linear(y_dim, x_dim * 2, zero_init_weight=True, key=key) 
        self.eps = 1e-5

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, 
        x: Float[Array, "{self.x_dim}"], 
        y: Float[Array, "{self.y_dim}"]
    ) -> Float[Array, "{self.x_dim}"]:
        # Zero-initialised gamma and beta parameters
        params = self.gamma_beta(y)  
        gamma, beta = jnp.split(params, 2, axis=-1)  

        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_normalized = (x - mean) / jnp.sqrt(var + self.eps)
        
        out = jnp.exp(gamma) * x_normalized + beta
        return out


class Permutation(eqx.Module):
    seq_length: int

    def __init__(self, seq_length: int):
        self.seq_length = seq_length

    def __call__(
        self, 
        x: Float[Array, "{self.seq_length} q"], 
        axis: int = 0, 
        inverse: bool = False
    ) -> Float[Array, "{self.seq_length} q"]:
        raise NotImplementedError('Overload me')


class PermutationIdentity(Permutation):
    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, 
        x: Float[Array, "{self.seq_length} q"], 
        axis: int = 0, 
        inverse: bool = False
    ) -> Float[Array, "{self.seq_length} q"]:
        return x


class PermutationFlip(Permutation):
    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, 
        x: Float[Array, "{self.seq_length} q"],
        axis: int = 0, 
        inverse: bool = False
    ) -> Float[Array, "{self.seq_length} q"]:
        return jnp.flip(x, axis=axis)


class Attention(eqx.Module):
    patch_size: int
    n_patches: int
    y_dim: int

    n_heads: int
    head_channels: int

    norm: eqx.nn.LayerNorm | AdaLayerNorm
    attention: MultiheadAttention

    @jaxtyped(typechecker=typechecker)
    def __init__(
        self, 
        in_channels: int, 
        head_channels: int, 
        patch_size: int,
        n_patches: int,
        y_dim: Optional[int],
        *, 
        key: Key[jnp.ndarray, "..."]
    ):
        assert in_channels % head_channels == 0

        keys = jr.split(key)

        self.patch_size = patch_size
        self.n_patches = n_patches
        self.y_dim = y_dim

        self.n_heads = int(in_channels / head_channels)
        self.head_channels = head_channels

        self.norm = (
            AdaLayerNorm(in_channels, y_dim=y_dim, key=keys[0])
            if y_dim is not None else
            eqx.nn.LayerNorm(in_channels)
        )

        self.attention = self_attention(
            self.n_heads,
            size=in_channels,
            state_length=n_patches,
            scale_factor=head_channels ** 2., # NOTE: check this scale
            attn_weight_bias=True,
            key=keys[1]
        )

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, 
        x: Float[Array, "#s q"], # For autoregression
        y: Optional[Float[Array, "{self.y_dim}"]], 
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

        _norm = partial(self.norm, y=y) if self.y_dim is not None else self.norm
        x = jax.vmap(_norm)(x) 

        if isinstance(mask, jax.Array):
            mask = mask.astype(jnp.bool)

        a = self.attention(x, x, x, mask=mask, state=state)

        # Return updated state if it was given
        if state is None:
            x = a
        else:
            x, state = a

        return x, state # Keys from cache of this attention mechanism are propagated by attn calculation above


class MLP(eqx.Module):
    y_dim: int
    norm: eqx.nn.LayerNorm | AdaLayerNorm
    net: eqx.nn.Sequential

    @jaxtyped(typechecker=typechecker)
    def __init__(
        self, 
        channels: int, 
        expansion: int, 
        y_dim: Optional[int],
        *, 
        key: Key[jnp.ndarray, "..."]
    ):
        keys = jr.split(key, 3)
        self.y_dim = y_dim
        self.norm = (
            AdaLayerNorm(channels, y_dim, key=keys[0])
            if y_dim is not None else
            eqx.nn.LayerNorm(channels)
        )
        self.net = eqx.nn.Sequential(
            [
                Linear(channels, channels * expansion, key=keys[1]),
                eqx.nn.Lambda(jax.nn.gelu),
                Linear(channels * expansion, channels, key=keys[2]),
            ]
        )

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, 
        x: Float[Array, "c"], 
        y: Optional[Float[Array, "{self.y_dim}"]]
    ) -> Float[Array, "c"]:
        return self.net(
            self.norm(x, y) if self.y_dim is not None else self.norm(x)
        )


class AttentionBlock(eqx.Module):
    attention: Attention
    mlp: MLP

    n_patches: int
    sequence_length: int
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
        *, 
        key: Key[jnp.ndarray, "..."]
    ):
        keys = jr.split(key)
        self.attention = Attention(
            channels, 
            head_channels, 
            patch_size=patch_size,
            n_patches=n_patches,
            y_dim=y_dim,
            key=keys[0]
        )
        self.mlp = MLP(channels, expansion, y_dim=y_dim, key=keys[1])

        self.n_patches = n_patches
        self.sequence_length = channels
        self.y_dim = y_dim

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, 
        x: Float[Array, "#{self.n_patches} {self.sequence_length}"], # 1 patch in autoregression step
        y: Optional[Float[Array, "{self.y_dim}"]] = None, 
        attn_mask: Optional[
            Union[
                Float[Array, "{self.n_patches} {self.n_patches}"],
                Int[Array, "{self.n_patches} {self.n_patches}"],
                Literal["causal"]
            ]
        ] = None, # No mask during sampling (key/value caching)
        state: Optional[eqx.nn.State] = None # No state during forward pass
    ) -> Union[
        Float[Array, "#{self.n_patches} {self.sequence_length}"],
        Tuple[
            Float[Array, "#{self.n_patches} {self.sequence_length}"], 
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


class CausalTransformerBlock(eqx.Module):
    proj_in: Linear
    pos_embed: Array
    attn_blocks: List[AttentionBlock]
    proj_out: Linear
    permutation: Permutation

    attn_mask: Array
    channels: int
    n_patches: int
    patch_size: int
    sequence_length: int
    head_dim: int
    y_dim: int

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
        *,
        key: Key[jnp.ndarray, "..."]
    ):
        key, *keys = jr.split(key, 4)

        self.proj_in = Linear(in_channels, channels, key=keys[0])

        self.pos_embed = jr.normal(keys[1], (n_patches, channels)) * 1e-2

        block_keys = jr.split(key, n_layers)
        self.attn_blocks = [
            AttentionBlock(
                channels, 
                head_dim, 
                expansion, 
                patch_size=patch_size,
                n_patches=n_patches,
                y_dim=y_dim,
                key=block_key
            ) 
            for block_key in block_keys
        ]

        self.channels = channels
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.sequence_length = in_channels 
        self.head_dim = head_dim
        self.y_dim = y_dim
    
        # Zero-init for identity mapping at first
        self.proj_out = Linear(
            channels, 
            in_channels * 2, 
            zero_init_weight=True, 
            key=keys[2]
        ) 

        self.permutation = permutation

        self.attn_mask = jnp.tril(jnp.ones((n_patches, n_patches)))

    @jaxtyped(typechecker=typechecker)
    def forward(
        self, 
        x: Float[Array, "{self.n_patches} {self.sequence_length}"], 
        y: Optional[Float[Array, "{self.y_dim}"]] = None
    ) -> Tuple[
        Float[Array, "{self.n_patches} {self.sequence_length}"], 
        Float[Array, ""]
    ]: 
        x_in = x.copy()

        # Permute position embedding and input together
        x = self.permutation(x)
        pos_embed = self.permutation(self.pos_embed) 

        # Encode each key and add positional information
        x = jax.vmap(self.proj_in)(x) + pos_embed 

        for block in self.attn_blocks:
            block: AttentionBlock
            x = block(x, y, attn_mask="causal")

        # Project to input channels dimension, from hidden dimension
        x = jax.vmap(self.proj_out)(x)

        # Splitting along sequence dimension. Always propagate first component of x_in without transforming it?
        x = jnp.concatenate([jnp.zeros_like(x[:1]), x[:-1]], axis=0) # Ensure first token the same, other tokens depend on ones before

        # NVP scale and shift along sequence dimension (not number of sequences e.g. patches in image?)
        x_a, x_b = jnp.split(x, 2, axis=-1) # NOTE: token dim; mu and alpha; mu,alpha up to T-1

        assert (
            x_a.shape == (self.n_patches, self.sequence_length)
            and
            x_b.shape == (self.n_patches, self.sequence_length)
        ), "xa: {}, xb: {}".format(x_a.shape, x_b.shape)

        # Shift and scale all tokens in sequence; except first and last
        u = (x_in - x_b) * jnp.exp(-x_a) # First token the same as input
        u = self.permutation(u, inverse=True)

        return u, -x_a.sum() # Jacobian of transform on sequence

    @jaxtyped(typechecker=typechecker)
    def reverse_step(
        self,
        x: Float[Array, "{self.n_patches} {self.sequence_length}"],
        y: Optional[Float[Array, "{self.y_dim}"]],
        pos_embed: Float[Array, "{self.n_patches} {self.channels}"],
        i: Int[Array, ""],
        state: eqx.nn.State
    ) -> Tuple[
        Float[Array, "1 {self.sequence_length}"], 
        Float[Array, "1 {self.sequence_length}"], # Autoregression
        eqx.nn.State
    ]:
        # Autoregressive generation, start with i-th patch in sequence
        x_in = x[i].copy() 

        # Embed positional information to this patch
        x = (self.proj_in(x_in) + pos_embed[i])[jnp.newaxis, :] # Sequence dimension

        for block in self.attn_blocks:
            block: AttentionBlock
            x, state = block(x, y, attn_mask="causal", state=state) # NOTE: no mask here, k/v caching

        # Project to input channels dimension, from hidden dimension
        x = jax.vmap(self.proj_out)(x)

        x_a, x_b = jnp.split(x, 2, axis=-1) # Split on 2 * token-length dim -> token-length

        # Shift and scale for i-th token, state with updated k/v
        return x_a, x_b, state 

    @jaxtyped(typechecker=typechecker)
    def reverse(
        self,
        x: Float[Array, "{self.n_patches} {self.sequence_length}"], 
        y: Optional[Float[Array, "{self.y_dim}"]],
        state: eqx.nn.State 
    ) -> Tuple[
        Float[Array, "{self.n_patches} {self.sequence_length}"], 
        eqx.nn.State
    ]:
        x = self.permutation(x)
        pos_embed = self.permutation(self.pos_embed)

        def _autoregression_step(
            x_and_state: Tuple[
                Float[Array, "{self.n_patches} {self.sequence_length}"], eqx.nn.State
            ], 
            i: Int[Array, ""]
        ) -> Tuple[
            Tuple[
                Float[Array, "{self.n_patches} {self.sequence_length}"],
                eqx.nn.State
            ], 
            Int[Array, ""]
        ]:
            x, state = x_and_state

            z_a, z_b, state = self.reverse_step(
                x, y, pos_embed=pos_embed, i=i, state=state
            )

            x = x.at[i + 1].set(x[i + 1] * jnp.exp(z_a[0]) + z_b[0])

            return (x, state), i

        T = x.shape[0] 
        (x, state), _ = jax.lax.scan(
            _autoregression_step, 
            init=(x, state), 
            xs=jnp.arange(T - 1), 
            length=T - 1
        )

        x = self.permutation(x, inverse=True)

        return x, state 


class TransformerFlow(eqx.Module):
    blocks: List[CausalTransformerBlock]

    img_size: int
    patch_size: int
    n_patches: int
    n_channels: int
    sequence_length: int
    y_dim: int

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
        *,
        key: Key[jnp.ndarray, "..."]
    ):
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = int(img_size / patch_size) ** 2
        self.n_channels = in_channels
        self.sequence_length = in_channels * patch_size ** 2
        self.y_dim = y_dim

        permutations = [
            PermutationIdentity(self.n_patches), 
            PermutationFlip(self.n_patches)
        ]

        blocks = []
        for i in range(n_blocks):
            key_block_i = jr.fold_in(key, i)
            blocks.append(
                CausalTransformerBlock(
                    self.sequence_length,
                    channels,
                    n_patches=self.n_patches,
                    permutation=permutations[i % 2], # Alternate permutations
                    n_layers=layers_per_block,
                    patch_size=patch_size,
                    head_dim=head_dim,
                    expansion=expansion,
                    y_dim=y_dim,
                    key=key_block_i
                )
            )
        self.blocks = blocks

        self.eps_sigma = eps_sigma

    @jaxtyped(typechecker=typechecker)
    def sample_prior(
        self, 
        key: Key[jnp.ndarray, "..."], 
        n_samples: int
    ) -> Float[Array, "#n {self.n_patches} {self.sequence_length}"]:
        return jr.normal(key, (n_samples, self.n_patches, self.sequence_length))

    @jaxtyped(typechecker=typechecker)
    def get_loss(
        self, 
        z: Float[Array, "{self.n_patches} {self.sequence_length}"], 
        logdet: Float[Array, ""]
    ) -> Float[Array, ""]:
        return 0.5 * jnp.sum(jnp.square(z)) - logdet

    @jaxtyped(typechecker=typechecker)
    def log_prob(
        self, 
        x: Float[Array, "{self.n_channels} {self.img_size} {self.img_size}"], 
        y: Optional[Float[Array, "..."]] # Arbitrary shape conditioning is flattened
    ) -> Float[Array, ""]:
        z, _, logdet = self.forward(x, y)
        log_prob = -self.get_loss(z, logdet)
        return log_prob
    
    @jaxtyped(typechecker=typechecker)
    def denoise(
        self, 
        x: Float[Array, "{self.n_channels} {self.img_size} {self.img_size}"], 
        y: Optional[Float[Array, "..."]] # Arbitrary shape conditioning is flattened
    ) -> Float[Array, "{self.n_channels} {self.img_size} {self.img_size}"]:
        score = jax.jacfwd(lambda x: self.log_prob(x, y))
        x = x + jnp.square(self.eps_sigma) * score(x)
        return x

    @jaxtyped(typechecker=typechecker)
    def sample_model(
        self, 
        key: Key[jnp.ndarray, "..."],
        y: Optional[Float[Array, "n ..."]], 
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
    ) -> Float[Array, "{self.n_patches} {self.sequence_length}"]:
        h = w = int(self.img_size / self.patch_size)
        ph = pw = self.patch_size
        u = rearrange(
            x, "c (h ph) (w pw) -> (h w) (c ph pw)", h=h, w=w, ph=ph, pw=pw
        )
        return u

    @jaxtyped(typechecker=typechecker)
    def unpatchify(
        self, 
        x: Float[Array, "{self.n_patches} {self.sequence_length}"]
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
        y: Optional[Float[Array, "..."]] # Arbitrary shape conditioning is flattened
    ) -> Tuple[
        Float[Array, "{self.n_patches} {self.sequence_length}"], 
        List[Float[Array, "{self.n_patches} {self.sequence_length}"]],
        Float[Array, ""]
    ]:
        x = self.patchify(x)
        if y is not None:
            y = y.flatten()

        sequence = []
        logdet = jnp.zeros(())
        for block in self.blocks:
            block: CausalTransformerBlock
            x, block_logdet = block.forward(x, y)
            logdet = logdet + block_logdet

            sequence.append(x)

        return x, sequence, logdet

    @jaxtyped(typechecker=typechecker)
    def reverse(
        self, 
        z: Float[Array, "{self.n_patches} {self.sequence_length}"], 
        y: Optional[Float[Array, "..."]], # Arbitrary shape conditioning is flattened
        state: eqx.nn.State,
        return_sequence: bool = False 
    ) -> Tuple[
        Union[
            Float[Array, "{self.n_channels} {self.img_size} {self.img_size}"],
            List[Float[Array, "{self.n_channels} {self.img_size} {self.img_size}"]] 
        ],
        eqx.nn.State # State used in sampling
    ]:
        sequence = [self.unpatchify(z)]
        if y is not None:
            y = y.flatten()

        for block in reversed(self.blocks):
            block: CausalTransformerBlock
            z, state = block.reverse(z, y, state=state)

            sequence.append(self.unpatchify(z))

        x = self.unpatchify(z)

        return sequence if return_sequence else x, state


@jaxtyped(typechecker=typechecker)
def single_loss_fn(
    model: TransformerFlow, 
    key: Key[jnp.ndarray, "..."], 
    x: Float[Array, "_ _ _"], 
    y: Optional[Float[Array, "..."]],
) -> Tuple[
    Float[Array, ""], 
    dict[str, Float[Array, ""]]
]:
    z, _, logdets = model.forward(x, y)
    loss = model.get_loss(z, logdets)
    metrics = dict(
        z=jnp.mean(jnp.square(z)), logdets=logdets
    )
    return loss, metrics


@jaxtyped(typechecker=typechecker)
def batch_loss_fn(
    model: TransformerFlow, 
    key: Key[jnp.ndarray, "..."], 
    X: Float[Array, "n _ _ _"], 
    Y: Optional[Float[Array, "n ..."]] = None
) -> Tuple[
    Float[Array, ""], 
    dict[str, Float[Array, ""]]
]:
    keys = jr.split(key, X.shape[0])
    _fn = partial(single_loss_fn, model)
    loss, metrics = eqx.filter_vmap(_fn)(keys, X, Y)
    metrics = jax.tree.map(jnp.mean, metrics)
    return jnp.mean(loss), metrics


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit(donate="all-except-first")
def evaluate(
    model: TransformerFlow, 
    key: Key[jnp.ndarray, "..."], 
    x: Float[Array, "n _ _ _"], 
    y: Optional[Float[Array, "n ..."]] = None,
    *,
    sharding: Optional[NamedSharding] = None,
    replicated_sharding: Optional[PositionalSharding] = None
) -> Tuple[
    Float[Array, ""], 
    dict[str, Float[Array, ""]]
]:
    if replicated_sharding is not None:
        model = eqx.filter_shard(model, replicated_sharding)
    if sharding is not None:
        x, y = shard_batch((x, y), sharding)

    loss, metrics = batch_loss_fn(model, key, x, y)
    return loss, metrics


@jaxtyped(typechecker=typechecker)
def accumulate_gradients_scan(
    model: eqx.Module,
    key: Key[jnp.ndarray, "..."],
    x: Float[Array, "n _ _ _"], 
    y: Optional[Float[Array, "n ..."]],
    n_minibatches: int,
    *,
    grad_fn: Callable[
        [
            eqx.Module, 
            Key[jnp.ndarray, "..."],
            Float[Array, "n _ _ _"],
            Optional[Float[Array, "n ..."]]
        ],
        Tuple[
            Float[Array, ""], 
            dict[str, Float[Array, ""]], 
        ]
    ]
) -> Tuple[
    Tuple[
        Float[Array, ""], 
        dict[str, Float[Array, ""]], 
    ], 
    PyTree
]:
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

    # Determine initial shapes for gradients and metrics.
    grads_shapes, L_shape, metrics_shape = jax.eval_shape(_minibatch_step, 0)
    grads = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), grads_shapes)
    L = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), L_shape)
    metrics = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), metrics_shape)

    # Loop over minibatches to determine gradients and metrics.
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
    y: Optional[Float[Array, "n ..."]], # Arbitrary conditioning shape is flattened
    key: Key[jnp.ndarray, "..."], 
    opt_state: optax.OptState, 
    opt: optax.GradientTransformation,
    *,
    max_grad_norm: Optional[float] = 1.,
    n_minibatches: Optional[int] = 4,
    accumulate_gradients: Optional[bool] = False,
    sharding: Optional[NamedSharding] = None,
    replicated_sharding: Optional[PositionalSharding] = None
) -> Tuple[
    Float[Array, ""], 
    dict[str, Float[Array, ""]], 
    TransformerFlow, 
    optax.OptState
]:
    if replicated_sharding is not None:
        model, opt_state = eqx.filter_shard(
            (model, opt_state), replicated_sharding
        )
    if sharding is not None:
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

    if (max_grad_norm is not None) and (max_grad_norm > 0.):
        grads = clip_grad_norm(grads, max_norm=max_grad_norm)

    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    return loss, metrics, model, opt_state


def check_caches(model: TransformerFlow, state: eqx.nn.State) -> None:
    for i in range(len(model.blocks)):
        for j in range(len(model.blocks[i].attn_blocks)):
            k_cache, v_cache, state_index = state.get(
                model
                .blocks[i]
                .attn_blocks[j]
                .attention
                .attention
                .autoregressive_index
            )
            jax.debug.print("{} {} {}", i, j, state_index)


def get_sample_state(config: ConfigDict, key: Key) -> eqx.nn.State:
    return eqx.nn.make_with_state(TransformerFlow)(**config.model, key=key)[1]


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def sample_model(
    model: TransformerFlow, 
    z: Float[Array, "n _ _"], 
    y: Optional[Float[Array, "n ..."]], 
    state: eqx.nn.State,
    *,
    return_sequence: bool = False,
    denoise: bool = False,
    sharding: Optional[NamedSharding] = None,
    replicated_sharding: Optional[PositionalSharding] = None
) -> Union[
    Float[Array, "n _ _ _"], Float[Array, "n s _ _ _"]
]:
    if replicated_sharding is not None:
        model = eqx.filter_shard(model, replicated_sharding)
    if sharding is not None:
        z, y = shard_batch((z, y), sharding)

    states = eqx.filter_vmap(lambda: state, axis_size=z.shape[0])() # Shard states?

    samples, state = eqx.filter_vmap(model.reverse)(z, y, states, return_sequence)

    # Denoising
    if denoise:
        if return_sequence:
            denoised = jax.vmap(model.denoise)(samples[-1], y)
            samples = samples + [denoised]
        else:
            samples = jax.vmap(model.denoise)(samples, y)

    # Stack sequence 
    if return_sequence:
        samples = jnp.stack(samples, axis=1)

    return samples


@jaxtyped(typechecker=typechecker)
def loader(
    data: Float[Array, "n _ _ _"], 
    targets: Optional[Float[Array, "n ..."]], 
    batch_size: int, 
    *, 
    key: Key[jnp.ndarray, "..."]
) -> Generator[
    Union[
        Tuple[
            Float[Array, "b _ _ _"], 
            Optional[Float[Array, "b ..."]]
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
    use_y: bool = False
) -> Tuple[
    Union[
        Tuple[
            Float[Array, "t _ _ _"], 
            Optional[Float[Array, "t ..."]]
        ], 
    ],
    Union[
        Tuple[
            Float[Array, "v _ _ _"], 
            Optional[Float[Array, "v ..."]]
        ], 
    ],
    Callable[[Key[jnp.ndarray, "..."], int], Float[Array, "_ ..."]], 
    Callable[[Float[Array, "_ _ _"]], Float[Array, "_ _ _"]]
]:
    dataset_path = "/project/ls-gruen/{}/".format(dataset_name.lower())

    if dataset_name == "FLOWERS":
        _dataset_name = dataset_name.title() + "102"
    else:
        _dataset_name = dataset_name

    dataset = getattr(tv.datasets, _dataset_name)
    dataset = dataset(dataset_path, download=True)

    if dataset_name == "MNIST":
        data = jnp.asarray(dataset.data.numpy(), dtype=jnp.uint8) 
        targets = jnp.asarray(dataset.targets.float().numpy(), dtype=jnp.float32)

        data = data / 255.
        data = data[:, jnp.newaxis, ...]
        data = data.astype(jnp.float32)

        # mu, std = data.mean(), data.std()
        # data = (data - mu) / std
        # postprocess_fn = lambda x: mu + x * std

        a, b = data.min(), data.max()
        data = 2. * (data - a) / (b - a) - 1.

        def postprocess_fn(x): 
            return jnp.clip((1. + x) * 0.5 * (b - a) + a, min=0., max=1.)

        def target_fn(key, n): 
            y_range = jnp.arange(0, targets.max())
            return jr.choice(key, y_range, (n, 1)).astype(jnp.float32)

    if dataset_name == "CIFAR10":
        data = jnp.asarray(dataset.data, dtype=jnp.uint8)
        targets = jnp.asarray(dataset.targets, dtype=jnp.float32)
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
            return jr.choice(key, y_range, (n, 1)).astype(jnp.float32)

    if dataset_name == "FLOWERS":
        data = jnp.asarray(dataset.data, dtype=jnp.uint8)
        targets = jnp.asarray(dataset.targets, dtype=jnp.float32)
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
            return jr.choice(key, y_range, (n, 1)).astype(jnp.float32)

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

        target_fn = lambda *args, **kwargs: None

    return (x_train, y_train), (x_valid, y_valid), target_fn, postprocess_fn


@jaxtyped(typechecker=typechecker)
def train(
    key: Key[jnp.ndarray, "..."],
    model: TransformerFlow,
    eps_sigma: float,
    # Data
    dataset_name: Literal["MNIST", "CIFAR10", "FLOWERS"],
    img_size: int,
    n_channels: int,
    # Training
    batch_size: int = 256, 
    n_steps: int = 500_000,
    n_sample: int = 4,
    n_warps: int = 1,
    lr: float = 2e-4,
    initial_lr: float = 1e-6,
    final_lr: float = 1e-6,
    train_split: float = 0.9,
    max_grad_norm: Optional[float] = 1.0,
    use_ema: bool = False,
    ema_rate: float = 0.9995,
    accumulate_gradients: bool = False,
    n_minibatches: int = 4,
    get_state_fn: Callable[[None], eqx.nn.State] = None,
    cmap: Optional[str] = "gray",
    use_y: bool = False,
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
        use_y=use_y
    )

    # Optimiser
    n_steps_per_epoch = int(x_train.shape[0] / batch_size) 
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=initial_lr, 
        peak_value=lr, 
        warmup_steps=1 * n_steps_per_epoch,
        decay_steps=100 * n_steps_per_epoch, # Same as paper
        end_value=final_lr
    )
    opt = optax.adamw(
        learning_rate=scheduler, b1=0.9, b2=0.95, weight_decay=1e-4
    )
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    # EMA model if required
    if use_ema:
        ema_model = deepcopy(model) 

    valid_key, *loader_keys = jr.split(key, 3)

    train_batch_size = n_minibatches * batch_size if accumulate_gradients else batch_size

    losses, metrics = [], []
    with trange(n_steps) as bar:
        for i, (x_t, y_t), (x_v, y_v) in zip(
            bar, 
            loader(x_train, y_train, train_batch_size, key=loader_keys[0]), 
            loader(x_valid, y_valid, batch_size, key=loader_keys[1])
        ):
            key_eps, key_step = jr.split(jr.fold_in(key, i))

            # Add Gaussian noise to inputs
            eps = eps_sigma * jr.normal(key_eps, x_t.shape)

            loss_t, metrics_t, model, opt_state = make_step(
                model, 
                x_t + eps, 
                y_t, 
                key_step, 
                opt_state, 
                opt, 
                max_grad_norm=max_grad_norm,
                n_minibatches=n_minibatches,
                accumulate_gradients=accumulate_gradients,
                sharding=sharding,
                replicated_sharding=replicated_sharding
            )

            if use_ema:
                ema_model = apply_ema(ema_model, model, ema_rate)

            # Add Gaussian noise to inputs
            eps = eps_sigma * jr.normal(valid_key, x_v.shape)

            loss_v, metrics_v = evaluate(
                ema_model if use_ema else model, 
                valid_key, 
                x_v + eps, 
                y_v, 
                sharding=sharding,
                replicated_sharding=replicated_sharding
            )

            losses.append((loss_t, loss_v))
            metrics.append(
                (
                    (metrics_t["z"], metrics_v["z"]), 
                    (metrics_t["logdets"], metrics_v["logdets"])
                )
            )
            bar.set_postfix_str("Lt={:.3E} Lv={:.3E}".format(loss_t, loss_v))

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
                    z = model.sample_prior(valid_key, n_sample ** 2) 
                    y = target_fn(valid_key, n_sample ** 2) 
                    samples = sample_model(
                        ema_model if use_ema else model, 
                        z, 
                        y, 
                        state=get_state_fn(),
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
                    plt.imshow(add_spacing(postprocess_fn(samples), img_size, cols_only=True), cmap=cmap)
                    plt.colorbar() if cmap is not None else None
                    plt.axis("off")
                    plt.savefig(os.path.join(imgs_dir, "samples/samples_{:05d}.png".format(i)), bbox_inches="tight")
                plt.close() 

                # Sample a warping from noise to data
                if n_warps is not None:
                    z = model.sample_prior(valid_key, n_warps)
                    y = target_fn(valid_key, n_warps) 
                    samples = sample_model(
                        ema_model if use_ema else model, 
                        z, 
                        y, 
                        state=get_state_fn(), 
                        return_sequence=True,
                        sharding=sharding,
                        replicated_sharding=replicated_sharding
                    )

                    samples = rearrange(
                        samples, 
                        "(n1 n2) s c h w -> (n1 h) (s n2 w) c", 
                        n1=n_warps,
                        n2=1,
                        s=len(model.blocks) + 1, # Include final + denoised 
                        c=n_channels
                    )

                    plt.figure(dpi=200)
                    plt.imshow(postprocess_fn(samples), cmap=cmap)
                    plt.axis("off")
                    plt.savefig(os.path.join(imgs_dir, "warps/warps_{:05d}.png".format(i)), bbox_inches="tight")
                    plt.close()

                # Losses and metrics
                fig, axs = plt.subplots(1, 3, figsize=(11., 3.))
                ax = axs[0]
                ax.plot([l for l in losses if l < 10.0]) # Ignore spikes
                ax.set_title(r"$L$")
                ax = axs[1]
                ax.plot([m[0][0] for m in metrics if m < 10.0])
                ax.plot([m[0][1] for m in metrics if m < 10.0])
                ax.axhline(1., linestyle=":", color="k")
                ax.set_title(r"$z^2$")
                ax = axs[2]
                ax.plot([m[1][0] for m in metrics if m < 10.0])
                ax.plot([m[1][1] for m in metrics if m < 10.0])
                ax.set_title(r"$\sum_t^T\log|\mathbf{J}_t|$")
                plt.savefig(os.path.join(imgs_dir, "losses.png"), bbox_inches="tight")
                plt.close()

    return model


def get_config(dataset_name: str) -> ConfigDict:
    config = ConfigDict()

    # Data
    config.data = data = ConfigDict()
    data.dataset_name          = dataset_name
    data.n_channels            = {"CIFAR10" : 3, "MNIST" : 1, "FLOWERS" : 3}[dataset_name]
    data.img_size              = {"CIFAR10" : 32, "MNIST" : 28, "FLOWERS" : 64}[dataset_name]
    data.use_y                 = True

    # Model
    config.model = model = ConfigDict()
    model.img_size             = data.img_size
    model.in_channels          = data.n_channels
    model.patch_size           = 4 
    model.channels             = {"CIFAR10" : 512, "MNIST" : 256, "FLOWERS" : 256}[dataset_name]
    model.y_dim                = {"CIFAR10" : 1, "MNIST" : 1, "FLOWERS" : 1}[dataset_name] if data.use_y else None
    model.n_blocks             = 4
    model.head_dim             = 64
    model.expansion            = 4
    model.layers_per_block     = 4

    # Train
    config.train = train = ConfigDict()
    train.use_ema              = False 
    train.ema_rate             = 0.9995
    train.batch_size           = 256
    train.lr                   = 1e-3
    train.eps_sigma            = {"CIFAR10" : 0.05, "MNIST" : 0.05, "FLOWERS" : 0.05}[dataset_name]
    train.max_grad_norm        = 1.
    train.n_minibatches        = 4
    train.accumulate_gradients = True
    train.n_sample             = jax.local_device_count() * 2
    train.n_warps              = jax.local_device_count() 
    train.use_y                = data.use_y 
    train.dataset_name         = data.dataset_name
    train.img_size             = data.img_size
    train.n_channels           = data.n_channels
    train.cmap                 = {"CIFAR10" : None, "MNIST" : "gray", "FLOWERS" : None}[dataset_name]

    return config


if __name__ == "__main__":
    key = jr.key(0)

    dataset_name = "CIFAR10"

    config = get_config(dataset_name)

    print(config.model)
    print(config.train)

    key_model, key_train = jr.split(key)

    model, _ = eqx.nn.make_with_state(TransformerFlow)(
        **config.model, key=key_model
    )

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