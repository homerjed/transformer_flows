<h1 align='center'>Transformer flow</h1>

Implementation of Apple ML's [Normalising flows are capable generative models](https://arxiv.org/pdf/2412.06329) in `jax` and `equinox`.

Features:
- `jax.vmap` & `jax.lax.scan` construction & forward-pass for layers respectively for fast compilation and execution,
- multi-device training, inference and sampling,
- score-based denoising step (see paper),
- conditioning via class embedding (for discrete class labels) or adaptive layer-normalisation (for continuous variables) for continuous quantities,
- array-typed to-the-teeth for dependable execution with `jaxtyping` and `beartype`.

To implement:
- [ ] Guidance
- [ ] Fix denoising
- [ ] Mixed precision
- [x] AdaLayerNorm
- [x] Class embedding
- [ ] Hyperparameter/model saving

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