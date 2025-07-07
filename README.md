<h1 align='center'>Transformer flows</h1>

Implementation of Apple ML's Transformer Flow (or TARFlow) from [Normalising flows are capable generative models](https://arxiv.org/pdf/2412.06329) in `jax` and `equinox`.

Features:
- `jax.vmap` & `jax.lax.scan` construction & forward-pass, for layers respectively for fast compilation and execution,
- multi-device training, inference and sampling,
- score-based denoising step (see paper),
- conditioning via class embedding (for discrete class labels) or adaptive layer-normalisation (for continuous variables, like in DiT),
- array-typed to-the-teeth for dependable execution with `jaxtyping` and `beartype`.

To implement:
- [x] Guidance
- [x] Denoising
- [x] Mixed precision
- [x] EMA
- [x] AdaLayerNorm
- [x] Class embedding
- [x] Hyperparameter/model saving
- [x] Uniform and Gaussian noise for dequantisation

#### Usage 

```
pip install transformer-flows
```

```
uv run --all-extras python examples/main.py
```

```

#### Samples

I haven't optimised anything here (the authors mention varying the variance of noise used to dequantise the images), nor have I trained for very long. You can see slight artifacts due to the dequantisation noise.

<p align="center">
  <picture>
    <img src="assets/mnist_warp.gif" alt="Your image description">
  </picture>
</p>

<p align="center">
  <picture>
    <img src="assets/cifar10_warp.gif" alt="Your image description">
  </picture>
</p>

#### Citation 

```bibtex
@misc{zhai2024normalizingflowscapablegenerative,
      title={Normalizing Flows are Capable Generative Models}, 
      author={Shuangfei Zhai and Ruixiang Zhang and Preetum Nakkiran and David Berthelot and Jiatao Gu and Huangjie Zheng and Tianrong Chen and Miguel Angel Bautista and Navdeep Jaitly and Josh Susskind},
      year={2024},
      eprint={2412.06329},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.06329}, 
}
```