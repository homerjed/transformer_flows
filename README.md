<h1 align='center'>Transformer flow</h1>

Implementation of Apple ML's Transformer Flow (or TARFlow) from [Normalising flows are capable generative models](https://arxiv.org/pdf/2412.06329) in `jax` and `equinox`.

Features:
- `jax.vmap` & `jax.lax.scan` construction & forward-pass, for layers respectively for fast compilation and execution,
- multi-device training, inference and sampling,
- score-based denoising step (see paper),
- conditioning via class embedding (for discrete class labels) or adaptive layer-normalisation (for continuous variables, like in DiT),
- array-typed to-the-teeth for dependable execution with `jaxtyping` and `beartype`.

To implement:
- [ ] Guidance
- [x] Denoising
- [x] Mixed precision
- [x] EMA
- [x] AdaLayerNorm
- [x] Class embedding
- [ ] Hyperparameter/model saving
- [x] Uniform noise for dequantisation

<!-- Notes:
- All-in-all, I think this paper implements a useful algorithm. However, it is not as easy as they imply to train. 
    - This could be due to the differences in attention implementations, but the model only really worked with EMA and gradient clipping.
    - The hyperparameters used in their code don't produce good results for me. 
- It's not clear which quantisation procedure you should use - it's allegedly a trade-off between sample quality and model log-likelihood.
- This model requires a lot of compute power. -->

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