import os
from copy import deepcopy
from typing import Tuple, List, Optional, Sequence, Callable, Literal, Generator, Union
from functools import partial 
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.sharding import NamedSharding, PositionalSharding, Mesh, PartitionSpec
from jax.experimental import mesh_utils
import equinox as eqx
from jaxtyping import Array, Key, Float, Int, PyTree, jaxtyped
from beartype import beartype as typechecker
import optax
from einops import rearrange
import matplotlib.pyplot as plt
from tqdm.auto import trange
import torchvision as tv

from mha import MultiheadAttention

e = os.environ.get("DEBUG")
DEBUG = bool(int(e if e is not None else 0))

"""
    Fit a Transformer Flow to the MNIST dataset.
    - What is the NVP vars thing? Covariance of prior?

    torch: 
    - implementation in torch changes the order of the key / sequence
      because the torch linear layers act across first batch dimension 
      instead of vmapping. 
      => key / sequence axis in flip / identity permutations is wrong?

    flow:
    - acts on sequence dimension (not sequence length, which is iterated over
      in autoregressive sampling...)

    is warmup / decay too fast?

    aattention(x, x, x) works same as code..

    mask = None in sampling => MHA uses its own mask... masking out based on state index
    - mask is none should apply query to all keys and values (that are cached)

    how can z loss be around 1? E[x^2] = 1 if x ~ N[0, 1]

    eqx.filter for opt_state picking something up it shouldnt?

    sum/mean should be applied to MSE on a single z and logdets (-xa.sum())
    - the means are scaled by the same amount? N?
"""


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
        zero_init: bool = False, 
        key: Key
    ):
        if zero_init:
            self.weight = jnp.zeros((out_size, in_size))
        else:
            self.weight = jr.truncated_normal(
                key, shape=(out_size, in_size), lower=-2., upper=2.
            ) * jnp.sqrt(1. / in_size)
        self.bias = jnp.zeros((out_size,))

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
        self.gamma_beta = eqx.nn.Linear(y_dim, x_dim * 2, key=key) 
        self.eps = 1e-5

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, 
        x: Float[Array, "{self.x_dim}"], 
        y: Float[Array, "{self.y_dim}"]
    ) -> Float[Array, "{self.x_dim}"]:
        params = self.gamma_beta(y)  
        gamma, beta = jnp.split(params, 2, axis=-1)  

        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_normalized = (x - mean) / jnp.sqrt(var + self.eps)
        
        out = gamma * x_normalized + beta
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
    sqrt_scale: float
    head_channels: int

    norm: eqx.nn.LayerNorm | AdaLayerNorm
    attention: MultiheadAttention

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
        self.sqrt_scale = head_channels ** (-0.25)

        self.norm = (
            AdaLayerNorm(in_channels, y_dim=y_dim, key=keys[0])
            if y_dim is not None else
            eqx.nn.LayerNorm(in_channels)
        )

        self.attention = MultiheadAttention(
            num_heads=self.n_heads,
            query_size=in_channels,
            state_length=n_patches,
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
                Literal["causal"]
            ]
        ], 
        state: Optional[eqx.nn.State]
    ) -> Tuple[Float[Array, "s q"], Optional[eqx.nn.State]]:

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
                # NOTE: Forced coniditoning dim?
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

    def __init__(
        self, 
        channels: int, 
        head_channels: int, 
        expansion: int = 4, 
        patch_size: int = 2,
        n_patches: int = 256,
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
                Int[Array, "{self.n_patches} {self.n_patches}"]
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
        assert not (state and attn_mask), (
            "Mask during training and state during sampling, not both."
        )

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
    nvp: bool

    def __init__(
        self,
        in_channels: int,
        channels: int,
        n_patches: int,
        permutation: Permutation,
        n_layers: int = 1,
        patch_size: int = 2,
        head_dim: int = 64,
        expansion: int = 4,
        nvp: bool = True,
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

        self.nvp = nvp

        self.channels = channels
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.sequence_length = in_channels 
        self.head_dim = head_dim
        self.y_dim = y_dim
    
        # Zero-init for identity mapping at first
        self.proj_out = Linear(
            channels, 
            in_channels * 2 if nvp else in_channels, 
            zero_init=True, 
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
            x = block(x, y, attn_mask=maybe_stop_grad(self.attn_mask, stop=True))

        # Project to input channels dimension, from hidden dimension
        x = jax.vmap(self.proj_out)(x)

        # Splitting along sequence dimension. Always propagate first component of x_in without transforming it?
        x = jnp.concatenate([jnp.zeros_like(x[:1]), x[:-1]], axis=0) # Ensure first token the same, other tokens depend on ones before

        # NVP scale and shift along sequence dimension (not number of sequences e.g. patches in image?)
        if self.nvp:
            xa, xb = jnp.split(x, 2, axis=-1) # NOTE: token dim; mu and alpha; mu,alpha up to T-1
        else:
            xa, xb = jnp.zeros_like(x), x

        assert (
            xa.shape == (self.n_patches, self.sequence_length)
            and
            xb.shape == (self.n_patches, self.sequence_length)
        ), "xa: {}, xb: {}".format(xa.shape, xb.shape)

        # Shift and scale all tokens in sequence; except first and last
        scale = jnp.exp(-xa)
        u = (x_in - xb) * scale # First token the same as input

        u = self.permutation(u, inverse=True)

        return u, -xa.mean() # Jacobian of transform on sequence

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
        x_in = x[i].copy() # Get i-th patch but keep the sequence dimension

        # Embed positional information to this patch
        x = (self.proj_in(x_in) + pos_embed[i])[jnp.newaxis, :]

        for block in self.attn_blocks:
            block: AttentionBlock
            x, state = block(x, y, attn_mask=None, state=state) # NOTE: no mask here, k/v caching

        # Project to input channels dimension, from hidden dimension
        x = jax.vmap(self.proj_out)(x)

        if self.nvp:
            xa, xb = jnp.split(x, 2, axis=-1) # Split on sequence-length dim
        else:
            xa, xb = jnp.zeros_like(x), x

        # Shift and scale for i-th token, state with updated k/v
        return xa, xb, state 

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
        # Flip input => flip position embed also
        x = self.permutation(x)
        pos_embed = self.permutation(self.pos_embed)

        def _autoregression_step(x_and_state, i):
            x, state = x_and_state

            za, zb, state = self.reverse_step(
                x, y, pos_embed=pos_embed, i=i, state=state
            )

            scale = jnp.exp(za[0]) 
            x = x.at[i + 1].set(x[i + 1] * scale + zb[0])
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

    nvp: bool
    var: Float[Array, "..."]
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
        nvp: bool = True,
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
                    nvp=nvp,
                    y_dim=y_dim,
                    key=key_block_i
                )
            )
        self.blocks = blocks

        self.nvp = nvp

        self.eps_sigma = eps_sigma

        # Prior for nvp mode should be all ones, but needs to be learnd for the vp mode
        self.var = jnp.ones((self.n_patches, self.sequence_length)) # NOTE: stop_Grad usually?

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
        return 0.5 * jnp.mean(jnp.square(z)) - logdet

    @jaxtyped(typechecker=typechecker)
    def log_prob(
        self, 
        x: Float[Array, "{self.n_channels} {self.img_size} {self.img_size}"], 
        y: Optional[Float[Array, "..."]] # Arbitrary shape conditioning is flattened
    ) -> Float[Array, ""]:
        z, _, logdet = self.forward(x, y)
        log_prob = -self.get_loss(z, logdet)
        return log_prob
    
    def denoise(
        self, 
        x: Float[Array, "{self.n_channels} {self.img_size} {self.img_size}"], 
        y: Optional[Float[Array, "..."]] # Arbitrary shape conditioning is flattened
    ) -> Float[Array, "{self.n_channels} {self.img_size} {self.img_size}"]:
        score = jax.jacfwd(lambda x: self.log_prob(x, y))
        x = x + jnp.square(self.eps_sigma) * score(x)
        return x

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

        outputs = []
        logdets = jnp.zeros(())
        for block in self.blocks:
            block: CausalTransformerBlock
            x, logdet = block.forward(x, y)
            logdets = logdets + logdet
            outputs.append(x)

        return x, outputs, logdets

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

        z = z * jnp.sqrt(maybe_stop_grad(self.var, stop=self.nvp)) 

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
    dict[str, Float[Array, "..."]]
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
    dict[str, Float[Array, "..."]]
]:
    keys = jr.split(key, X.shape[0])
    _fn = partial(single_loss_fn, model)
    loss, metrics = jax.vmap(_fn)(keys, X, Y)
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
    dict[str, Float[Array, "..."]]
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
            dict[str, Float[Array, "..."]], 
        ]
    ]
) -> Tuple[
    Tuple[
        Float[Array, ""], 
        dict[str, Float[Array, "..."]], 
    ], 
    PyTree
]:
    batch_size = x.shape[0]
    minibatch_size = batch_size // n_minibatches
    keys = jr.split(key, n_minibatches)

    def _minibatch_step(minibatch_idx):
        """ Gradients and metrics for a single minibatch. """
        _x, _y = jax.tree.map(
            # Slicing with variable index (jax.Array).
            lambda x: jax.lax.dynamic_slice_in_dim(  
                x, 
                start_index=minibatch_idx * minibatch_size, 
                slice_size=minibatch_size, 
                axis=0
            ),
            (x, y)
        )

        (step_L, step_metrics), step_grads = grad_fn(
            model, keys[minibatch_idx], _x, _y
        )
        return step_grads, step_L, step_metrics

    def _scan_step(carry, minibatch_idx):
        """ Scan step function for looping over minibatches. """
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
    n_minibatches: int = 4,
    accumulate_gradients: bool = False,
    sharding: Optional[NamedSharding] = None,
    replicated_sharding: Optional[PositionalSharding] = None
) -> Tuple[
    Float[Array, ""], 
    dict[str, Float[Array, "..."]], 
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

    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return loss, metrics, model, opt_state


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def sample_model(
    model: TransformerFlow, 
    z: Float[Array, "n _ _"], 
    y: Optional[Float[Array, "n ..."]], 
    state: eqx.nn.State,
    *,
    return_sequence: bool = False,
) -> Union[
    Float[Array, "n _ _ _"], Float[Array, "n s _ _ _"]
]:
    # Check states are refreshed? Or used identically to if they are empty?

    # Caches - can be identically init'd since they're just empty holders
    # with starting index of zero for each sampling.
    states = eqx.filter_vmap(lambda: state, axis_size=z.shape[0])()

    sample_fn = lambda z, y, state: model.reverse(
        z, y, state=state, return_sequence=return_sequence
    ) 
    samples, _ = jax.vmap(sample_fn)(z, y, states)

    # Stack sequence + denoising
    if return_sequence:
        denoised = jax.vmap(model.denoise)(samples[-1], y)
        samples = jnp.concatenate(samples + [denoised], axis=1)
    else:
        samples = jax.vmap(model.denoise)(samples, y)

    return samples


def loader(
    data: Float[Array, "n _ _ _"], 
    targets: Optional[Float[Array, "n ..."]], 
    batch_size: int, 
    *, 
    key: Key[jnp.ndarray, "..."]
) -> Generator[
    Union[
        Tuple[Float[Array, "b _ _ _"], Float[Array, "b ..."]],
        Float[Array, "b _ _ _"]
    ], 
    None, 
    None
]:
    def _get_batch(perm, x, y):
        if y is not None:
            batch = (x[perm], y[perm])
        else:
            batch = x[perm]
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
    split: float = 0.9
) -> Tuple[
    Tuple[Float[Array, "t _ _ _"], Float[Array, "t ..."]], 
    Tuple[Float[Array, "v _ _ _"], Float[Array, "v ..."]], 
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
        data = jnp.asarray(dataset.data.float().numpy())
        targets = jnp.asarray(dataset.targets.float().numpy())

        data = data[:, jnp.newaxis, ...].astype(jnp.float32)
        data = data / data.max()
        mu, std = data.mean(), data.std()
        data = (data - mu) / std
        postprocess_fn = lambda x: mu + x * std

        # a, b = data.min(), data.max()
        # data = 2. * (data - a) / (b - a) - 1.
        # postprocess_fn = lambda x: x #(1. + x) * 0.5 * (b - a) + a

        target_fn = lambda key, n: jr.choice(key, jnp.arange(targets.max()), (n, 1)).astype(jnp.float32)

    if dataset_name == "CIFAR10":
        data = jnp.asarray(dataset.data)
        targets = jnp.asarray(dataset.targets).astype(jnp.float32)
        targets = targets[:, jnp.newaxis]

        data = data.transpose(0, 3, 1, 2).astype(jnp.float32)
        data = data / data.max()
        mu, std = data.mean(), data.std()
        data = (data - mu) / std
        postprocess_fn = lambda x: jnp.clip(mu + x * std, min=0., max=1.)
        
        target_fn = lambda key, n: jr.choice(key, jnp.arange(targets.max()), (n, 1)).astype(jnp.float32)

    if dataset_name == "FLOWERS":
        data = jnp.asarray(dataset.data)
        targets = jnp.asarray(dataset.targets).astype(jnp.float32)
        targets = targets[:, jnp.newaxis]

        data = data.transpose(0, 3, 1, 2).astype(jnp.float32)
        data = data / data.max()
        mu, std = data.mean(), data.std()
        data = (data - mu) / std
        postprocess_fn = lambda x: jnp.clip(mu + x * std, min=0., max=1.)
        
        target_fn = lambda key, n: jr.choice(key, jnp.arange(targets.max()), (n, 1)).astype(jnp.float32)

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
    y_train, y_valid = jnp.split(targets, [n_train])

    return (x_train, y_train), (x_valid, y_valid), target_fn, postprocess_fn


@jaxtyped(typechecker=typechecker)
def train(
    key: Key[jnp.ndarray, "..."],
    model: TransformerFlow,
    state: eqx.nn.State, # K/V cache, updates in place so no need to refresh it (?)
    eps_sigma: float = 0.05,
    # Data
    dataset_name: Literal["MNIST", "CIFAR10", "FLOWERS"] = "MNIST",
    img_size: int = 32,
    n_channels: int = 1,
    # Training
    batch_size: int = 256, 
    n_steps: int = 500_000,
    n_sample: int = 4,
    lr: float = 2e-4,
    initial_lr: float = 1e-6,
    final_lr: float = 1e-6,
    train_split: float = 0.9,
    use_ema: bool = False,
    ema_rate: float = 0.9995,
    accumulate_gradients: bool = False,
    n_minibatches: int = 4,
    sample_state_fn: Callable[[None], eqx.nn.State] = None
) -> TransformerFlow:

    # Image save directories
    imgs_dir = "imgs/{}/".format(dataset_name.lower())
    if not os.path.exists(imgs_dir):
        for _dir in ["samples/", "warps/"]:
            os.makedirs(os.path.join(imgs_dir, _dir), exist_ok=True)

    # Data
    (
        (x_train, y_train), 
        (x_valid, y_valid), 
        target_fn, 
        postprocess_fn
    ) = get_data(
        dataset_name, img_size, n_channels, train_split
    )

    # Optimiser
    n_steps_per_epoch = int(x_train.shape[0] / batch_size) # ! was data
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=initial_lr, 
        peak_value=lr, 
        warmup_steps=1 * n_steps_per_epoch,
        decay_steps=100 * n_steps_per_epoch, # Same as paper
        end_value=final_lr
    )
    opt = optax.adamw(
        # learning_rate=scheduler, b1=0.9, b2=0.95, weight_decay=1e-4
        learning_rate=scheduler, weight_decay=1e-4
    )
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    if use_ema:
        ema_model = deepcopy(model) 

    valid_key, *loader_keys = jr.split(key, 3)

    losses, metrics = [], []
    with trange(n_steps) as bar:
        for i, (x_t, y_t), (x_v, y_v) in zip(
            bar, 
            loader(x_train, y_train, batch_size, key=loader_keys[0]), 
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
                if i == 0:
                    x_t = rearrange(
                        x_t[:n_sample ** 2], 
                        "(n1 n2) c h w -> (n1 h) (n2 w) c", 
                        n1=n_sample, 
                        n2=n_sample,
                        c=n_channels
                    )

                    plt.figure(dpi=200)
                    plt.imshow(postprocess_fn(x_t), cmap="gray_r")
                    plt.axis("off")
                    plt.savefig("data_{}.png".format(dataset_name.lower()), bbox_inches="tight")
                    plt.close()

                # Sample model 
                z = model.sample_prior(valid_key, n_sample ** 2) 
                y = target_fn(key, n_sample ** 2) 
                samples = sample_model(
                    ema_model if use_ema else model, 
                    z, 
                    y, 
                    state=sample_state_fn() #state 
                )

                samples = rearrange(
                    samples, 
                    "(n1 n2) c h w -> (n1 h) (n2 w) c", 
                    n1=n_sample,
                    n2=n_sample,
                    c=n_channels
                )

                plt.figure(dpi=200)
                plt.imshow(postprocess_fn(samples), cmap="gray_r")
                plt.axis("off")
                plt.savefig(os.path.join(imgs_dir, "samples/samples_{:05d}.png".format(i)), bbox_inches="tight")
                plt.close() 

                # # Sample a warping from noise to data
                # z = model.sample_prior(valid_key, 1)
                # y = target_fn(key, 1) 
                # samples = sample_model(
                #     ema_model if use_ema else model, 
                #     z, 
                #     y, 
                #     state=state, 
                #     return_sequence=True
                # )

                # samples = rearrange(
                #     samples, 
                #     "(n1 n2) c h w -> (n1 h) (n2 w) c", 
                #     n1=1,
                #     n2=n_blocks + 1 + 1, # Include final + denoised 
                #     c=n_channels
                # )

                # plt.figure(dpi=200)
                # plt.imshow(postprocess_fn(samples), cmap="gray_r")
                # plt.axis("off")
                # plt.savefig("imgs/warps/samples_{:05d}.png".format(i), bbox_inches="tight")
                # plt.close()

                fig, axs = plt.subplots(1, 3, figsize=(11., 3.))
                ax = axs[0]
                ax.plot(losses)
                ax.set_title(r"$L$")
                ax = axs[1]
                ax.plot([m[0][0] for m in metrics])
                ax.plot([m[0][1] for m in metrics])
                ax.axhline(1., linestyle=":", color="k")
                ax.set_title(r"$z^2$")
                ax = axs[2]
                ax.plot([m[1][0] for m in metrics])
                ax.plot([m[1][1] for m in metrics])
                ax.set_title(r"$\sum\log|\mathbf{J}|$")
                plt.savefig("losses_{}.png".format(dataset_name.lower()), bbox_inches="tight")
                plt.close()

    return model


if __name__ == "__main__":
    key = jr.key(0)

    # Data
    dataset = "MNIST"
    n_channels = {"CIFAR10" : 3, "MNIST" : 1, "FLOWERS" : 3}[dataset]

    # Model
    nvp = True 
    img_size = {"CIFAR10" : 32, "MNIST" : 28, "FLOWERS" : 64}[dataset]
    patch_size = 4
    hidden_dim = {"CIFAR10" : 256, "MNIST" : 128, "FLOWERS" : 256}[dataset]
    y_dim = {"CIFAR10" : 1, "MNIST" : 1, "FLOWERS" : 1}[dataset]
    n_blocks = 4
    head_dim = 64
    expansion = 4
    layers_per_block = 4

    # Training
    use_ema = False 
    ema_rate = 0.9995
    batch_size = 256
    lr = 2e-3
    eps_sigma = {"CIFAR10" : 0.05, "MNIST" : 0.1, "FLOWERS" : 0.05}[dataset]
    n_minibatches = 0
    accumulate_gradients = True

    key_model, key_train = jr.split(key)

    model, sample_state = eqx.nn.make_with_state(TransformerFlow)(
        in_channels=n_channels,
        img_size=img_size,
        patch_size=patch_size,
        channels=hidden_dim,
        n_blocks=n_blocks,
        layers_per_block=layers_per_block,
        expansion=expansion,
        head_dim=head_dim,
        y_dim=y_dim,
        nvp=nvp,
        eps_sigma=eps_sigma,
        key=key_model
    )

    def get_sample_state():
        return eqx.nn.make_with_state(TransformerFlow)(
            in_channels=n_channels,
            img_size=img_size,
            patch_size=patch_size,
            channels=hidden_dim,
            n_blocks=n_blocks,
            layers_per_block=layers_per_block,
            expansion=expansion,
            head_dim=head_dim,
            y_dim=y_dim,
            nvp=nvp,
            eps_sigma=eps_sigma,
            key=key_model
        )[1]

    n_params = sum(
        x.size 
        for x in 
        jax.tree.leaves(eqx.filter(model, eqx.is_array))
    )
    print("n_params={:.3E}".format(n_params))

    sharding, replicated_sharding = get_shardings()

    model = train(
        key_train,
        model, 
        sample_state,
        dataset_name=dataset,
        eps_sigma=eps_sigma, 
        img_size=img_size, 
        n_channels=n_channels,
        use_ema=use_ema,
        ema_rate=ema_rate,
        accumulate_gradients=accumulate_gradients,
        n_minibatches=n_minibatches,
        sample_state_fn=get_sample_state
    )