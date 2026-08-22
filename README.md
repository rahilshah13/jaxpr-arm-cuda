#### generative audio transformer in JAX

- continuation of work from `https://github.com/rahilshah13/audio`

---

$$ \mathcal{L}_{\text{total}} = \min_{\theta} \mathbb{E}_{t, \mathbf{X}_0, \boldsymbol{\epsilon}, \mathbf{c}} \left[ \left\| \epsilon_\theta \left( \boldsymbol{\alpha}_t \odot \mathbf{X}_0 + \boldsymbol{\sigma}_t \odot \boldsymbol{\epsilon}, t, \mathbf{c} \right) - \boldsymbol{\epsilon} \right\|^2 + \lambda \left( \left\| \text{STFT}(\mathbf{X}_0) - \text{STFT}(\hat{\mathbf{X}}_0) \right\|_1 + \left\| \mathbf{X}_0 - \hat{\mathbf{X}}_0 \right\|_1 \right) \right] $$

* $\theta$: Learnable weights of the hierarchical transformer network (`gpt_forward`).
* $\mathbf{X}_0 \in \mathbb{R}^{B \times T \times L}$: Clean batch tensor of multi-channel audio frames.
* $\boldsymbol{\epsilon}$: Batch tensor of independent standard Gaussian noise samples.
* $t$: Batch diffusion time step indices.
* $\mathbf{c}$: Packed batch conditioning tuple containing scales, tempos, and stem identifiers.
* $\boldsymbol{\alpha}_t, \boldsymbol{\sigma}_t$: Broadcasted noise schedule signal and noise multiplier tensors.
* $\mathbf{X}_t$: Noisy batch latent tensor at diffusion step $t$.
* $\epsilon_\theta(\mathbf{X}_t, t, \mathbf{c})$: Predicted noise tensor.
* $\hat{\mathbf{X}}_0$: Reconstructed clean audio batch latent.
* $\lambda$: Weighting coefficient balancing reconstruction penalties.
* $\text{STFT}(\cdot)$: Short-Time Fourier Transform.

---

#### `model.py`

* `rms_norm`: Root Mean Square normalization across feature dimensions.
* `apply_rope`: Rotary Position Embeddings for queries and keys.
* `scaled_dot_product_attention`: Multi-head attention with optional causal masks.
* `clamp_frobenius_norm`: Projects tensor weights to a Frobenius norm limit of one.
* `combined_audio_loss`: Hybrid STFT, log-spectral, and L1 waveform objective.
* `compute_empirical_ntk`: Evaluates the empirical Neural Tangent Kernel via VJP Jacobians.
* `gpt_forward`: Hierarchical transformer diffusion block processing multi-channel stems and discrete metadata.
* `xavier_normal`: Xavier normal weight initialization.
* `init_params`: Constructs model weight parameters.
* `get_cached_metadata`: Reads JSON metadata vault records.
* `raw_memmap_loader`: Yields memory-mapped batch tensors and conditioning tuples.
* `load_checkpoint_safely`: Loads checkpoint bundles using file locks.
* `push_and_pull_gradients`: Coordinates distributed gradient accumulation.

---

#### `inference.py`

* `float_to_hex_halves`: Converts floats into 16-bit halves for assembly instructions.
* `compile_closed_jaxpr_to_arm64`: Translates JAX expressions into ARM64 assembly.
* `compile_closed_jaxpr_to_cuda`: Generates NVIDIA CUDA C execution kernels.
* `HeterogeneousRuntime`: Manages multithreaded CPU and GPU binary execution.

---

```bash
brew install deno;python3 venv .venv;source ./.venv/bin/activate;pip3 install jax jaxlib optax numpy demucs scipy
python3 processing.py
python3 model.py
python3 inference.py --compile --seconds 10
python3 inference.py --generate --seconds 10
