
import math
from dataclasses import dataclass 
from pathlib import Path
import json
from shutil import rmtree
from copy import deepcopy
from typing import Tuple, List, Optional, Callable, Literal, Generator, Union
from functools import partial 

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.sharding import PartitionSpec, NamedSharding
import equinox as eqx
import optax
from jaxtyping import Array, PRNGKeyArray, Float, Int, Bool, Scalar, DTypeLike, PyTree, jaxtyped
from beartype import beartype as typechecker

from einops import rearrange
from ml_collections import ConfigDict
import matplotlib.pyplot as plt
from tqdm.auto import trange
from datasets import load_dataset

from transformer_flows import (
    TransformerFlow, Policy, Dataset,
    get_sample_state, get_shardings,
    exists, clear_and_get_results_dir,
    train, save, load
)

typecheck = jaxtyped(typechecker=typechecker) 

DatasetName = Literal["MNIST", "CIFAR10", "FLOWERS"]


@typecheck
def get_data(
    key: PRNGKeyArray,
    dataset_name: DatasetName,
    img_size: int, 
    n_channels: int,
    split: float = 0.9,
    use_y: bool = False,
    use_integer_labels: bool = True,
    *,
    dataset_path: Optional[str | Path] = None
) -> Dataset:

    target_type = jnp.int32 if use_integer_labels else jnp.float32

    if dataset_name == "MNIST":
        dataset = load_dataset("mnist").with_format("jax")

        data = jnp.concatenate([dataset["train"]["image"], dataset["test"]["image"]])
        data = data / 255.
        data = data[:, jnp.newaxis, ...]
        data = data.astype(jnp.float32)

        targets = jnp.concatenate([dataset["train"]["label"], dataset["test"]["label"]])
        targets = targets[:, jnp.newaxis]
        targets = targets.astype(target_type)

    if dataset_name == "CIFAR10":
        dataset = load_dataset("cifar10").with_format("jax")

        data = jnp.concatenate([dataset["train"]["img"], dataset["test"]["img"]])
        data = data / 255.
        data = data.transpose(0, 3, 1, 2)
        data = data.astype(jnp.float32)

        targets = jnp.concatenate([dataset["train"]["label"], dataset["test"]["label"]])
        targets = targets[:, jnp.newaxis]
        targets = targets.astype(target_type)

    if dataset_name == "FLOWERS":
        dataset = load_dataset("nelorth/oxford-flowers").with_format("jax")

        data_shape = (3, img_size, img_size)

        key_train, key_valid = jr.split(key)

        def random_crop(key, image, crop_size):
            img_shape = image.shape[-3:-1] # Extract (H, W)
            crop_h, crop_w = crop_size
            assert crop_h <= img_shape[0] and crop_w <= img_shape[1], (
                "Crop size must be <= image size"
            )
            key_top, key_left = jr.split(key)
            top = jr.randint(key_top, shape=(), minval=0, maxval=img_shape[0] - crop_h + 1)
            left = jr.randint(key_left, shape=(), minval=0, maxval=img_shape[1] - crop_w + 1)
            return image[..., top:top + crop_h, left:left + crop_w, :]

        imgs = []
        for i in trange(7169):
            key_i = jr.fold_in(key_train, i)
            img = jax.image.resize(
                # NOTE: index into an image dataset using the row index first and then the image column - dataset[0]["image"]
                random_crop(key_i, dataset["train"][i]["image"], crop_size=(500, 500)),
                reversed(data_shape),
                method="bilinear"
            )
            imgs.append(img)
        imgs = jnp.asarray(imgs)
        imgs_t = jnp.transpose(imgs, (0, 3, 1, 2))

        imgs = []
        for i in trange(1020):
            key_i = jr.fold_in(key_valid, i)
            img = jax.image.resize(
                random_crop(key_i, dataset["test"][i]["image"], crop_size=(500, 500)),
                reversed(data_shape),
                method="bilinear"
            )
            imgs.append(img)
        imgs = jnp.asarray(imgs)
        imgs_v = jnp.transpose(imgs, (0, 3, 1, 2))

        data = jnp.concatenate([imgs_t, imgs_v])
        data = data / 255.
        data = data.astype(jnp.float32)

        targets = jnp.concatenate([dataset["train"]["label"], dataset["test"]["label"]])
        targets = targets[:, jnp.newaxis]
        targets = targets.astype(target_type)

    data = jax.image.resize(
        data, 
        shape=(data.shape[0], n_channels, img_size, img_size),
        method="bilinear"
    )

    a, b = jnp.min(data), jnp.max(data)
    data = 2. * (data - a) / (b - a) - 1.

    def postprocess_fn(x: Array) -> Array: 
        return jnp.clip((1. + x) * 0.5 * (b - a) + a, min=0., max=1.)
        
    def target_fn(key: PRNGKeyArray, n: int) -> Array: 
        y_range = jnp.arange(0, targets.max()) 
        return jr.choice(key, y_range, (n, 1)).astype(target_type)

    print(
        "DATA:\n> {:.3E} {:.3E} {}\n> {} {}".format(
            data.min(), data.max(), data.dtype, 
            data.shape, targets.shape if exists(targets) else None
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

    return Dataset(
        dataset_name,
        x_train=x_train, 
        y_train=y_train, 
        x_valid=x_valid, 
        y_valid=y_valid, 
        target_fn=target_fn, 
        postprocess_fn=postprocess_fn
    )


def get_config(dataset_name: str) -> ConfigDict:

    config = ConfigDict()

    config.seed                = 0

    # Data
    config.data = data = ConfigDict()
    data.dataset_name          = dataset_name
    data.n_channels            = {"CIFAR10" : 3, "MNIST" : 1, "FLOWERS" : 3}[dataset_name]
    data.img_size              = {"CIFAR10" : 32, "MNIST" : 28, "FLOWERS" : 32}[dataset_name]
    data.use_y                 = False
    data.use_integer_labels    = True

    # Model
    config.model = model = ConfigDict()
    model.img_size             = data.img_size
    model.in_channels          = data.n_channels
    model.patch_size           = 4 
    model.channels             = {"CIFAR10" : 512, "MNIST" : 128, "FLOWERS" : 512}[dataset_name]
    model.y_dim                = {"CIFAR10" : 1, "MNIST" : 1, "FLOWERS" : 1}[dataset_name] 
    model.n_classes            = {"CIFAR10" : 10, "MNIST" : 10, "FLOWERS" : 101}[dataset_name] 
    model.conditioning_type    = "embed" # "layernorm" 
    model.n_blocks             = {"CIFAR10" : 8, "MNIST" : 3, "FLOWERS" : 8}[dataset_name]
    model.head_dim             = 64
    model.expansion            = 3
    model.layers_per_block     = {"CIFAR10" : 4, "MNIST" : 3, "FLOWERS" : 4}[dataset_name]

    if not data.use_y:
        model.n_classes = model.conditioning_type = None  
    else:
        if model.n_classes and ("embed" in model.conditioning_type):
            assert data.use_integer_labels, (
                "Can't use embedding with float labels!"
            )

    # Train
    config.train = train = ConfigDict()
    train.use_ema              = False
    train.ema_rate             = 0.9999 
    train.n_epochs             = {"CIFAR10" : 500, "MNIST" : 500, "FLOWERS" : 2000}[dataset_name] # Define epochs but use steps, same as paper
    train.n_epochs_warmup      = 1
    train.train_split          = 0.9
    train.batch_size           = {"CIFAR10" : 128, "MNIST" : 256, "FLOWERS" : 128}[dataset_name]
    train.initial_lr           = 1e-6
    train.lr                   = {"CIFAR10" : 5e-4, "MNIST" : 1e-3, "FLOWERS" : 5e-4}[dataset_name]
    train.final_lr             = 1e-6

    train.noise_type           = "gaussian"
    train.eps_sigma            = {"CIFAR10" : 0.05, "MNIST" : 0.1, "FLOWERS" : 0.05}[dataset_name]

    if train.noise_type == "uniform":
        train.eps_sigma = math.sqrt(1. / 3.) # Std of U[-1, 1]

    train.max_grad_norm        = 0.5
    train.accumulate_gradients = False
    train.n_minibatches        = 4

    n_sample = {"CIFAR10" : 1, "MNIST" : 5, "FLOWERS" : 1}[dataset_name]
    train.sample_every         = 1000 # Steps
    train.n_sample             = jax.local_device_count() * n_sample
    train.n_warps              = jax.local_device_count() * n_sample
    train.denoise_samples      = True

    train.use_y                = data.use_y 
    train.use_integer_labels   = data.use_integer_labels
    train.dataset_name         = data.dataset_name
    train.img_size             = data.img_size
    train.n_channels           = data.n_channels
    train.cmap                 = {"CIFAR10" : None, "MNIST" : "gray_r", "FLOWERS" : None}[dataset_name]

    config.train.policy = policy = ConfigDict()
    train.use_policy           = True
    policy.param_dtype         = jnp.float32
    policy.compute_dtype       = jnp.bfloat16 # Or bfloat16
    policy.output_dtype        = jnp.float32 

    return config


if __name__ == "__main__":

    dataset_name = "MNIST"

    reload_model = False

    config = get_config(dataset_name)

    imgs_dir = clear_and_get_results_dir(dataset_name)

    key = jr.key(config.seed)

    key_model, key_train, key_data = jr.split(key, 3)

    model, _ = eqx.nn.make_with_state(TransformerFlow)(**config.model, key=key_model)

    get_state_fn = partial(get_sample_state, config=config, key=key_model)

    sharding, replicated_sharding = get_shardings()

    if config.train.use_policy:
        policy = Policy(**config.train.policy)
    else:
        policy = None

    if reload_model:
        model = load(hyperparams=config.model)
    
    save_fn = partial(save, hyperparams=config.model)

    dataset = get_data(key_data, **config.data)

    model = train(
        key_train, 
        dataset,
        # Model
        model, 
        eps_sigma=config.train.eps_sigma,
        noise_type=config.train.noise_type,
        # Data
        dataset_name=config.data.dataset_name,
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
        replicated_sharding=replicated_sharding,
        imgs_dir=imgs_dir,
        save_fn=save_fn
    )