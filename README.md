
#### audio diffusion in JAX

---

##### 1. $\mathcal{O}(B, T, N, C)$ Scaling

$$\mathcal{C}(B, T, N, C) = \mathcal{O}\left(B \cdot T \cdot N \cdot C^2 + B \cdot T \cdot N^2 \cdot C\right)$$

##### 2. Variational Upper Bound

$$\mathcal{L}_{\text{infer}} \le \mathbb{E}_{t, x_0, \epsilon} \left[ \Vert{}\epsilon_\theta(x_t, t) - \epsilon\Vert{}^2 + \lambda \left( \Vert{}\text{STFT}(x_0) - \text{STFT}(\hat{x}_0)\Vert{}_1 + \Vert{}x_0 - \hat{x}_0\Vert{}_1 \right) \right]$$

---

#### Neural Network Kernel (`gpt_forward`)

$$\mathbf{h}_p = \text{RMSNorm}\left(\text{Attention}(\text{RoPE}(\mathbf{q}_p), \text{RoPE}(\mathbf{k}_p), \mathbf{v}_p) + \mathbf{p}_{\text{seq}}\right)$$

$$\mathbf{h}_t = \text{RMSNorm}\left(\text{Attention}(\text{RoPE}(\mathbf{q}_t), \text{RoPE}(\mathbf{k}_t), \mathbf{v}_t) + \mathbf{p}_{\text{time}}\right)$$

$$\mathbf{h}_{\text{out}} = \text{FFN}(\mathbf{h}_p + \mathbf{h}_t + \mathbf{c}_{\text{base}} + \sigma_t \mathbf{e}_{\sigma})$$

---

#### Diffusion Latent Trajectory

$$\hat{x}_0 = \frac{x_t - \sigma_t \epsilon_\theta(x_t, t)}{\alpha_t}$$

$$x_{t-1} = \alpha_{t-1} \hat{x}_0 + \sqrt{1 - \alpha_{t-1}^2 - \sigma_h^2} \, \epsilon_\theta(x_t, t) + \sigma_h z$$

* $x_t$: Noisy latent tensor at diffusion step $t$.
* $\epsilon_\theta(x_t, t)$: Predicted noise vector from the hierarchical transformer backbone.
* $\alpha_t, \sigma_t$: Noise schedule multipliers where $\alpha_t^2 + \sigma_t^2 = 1$.
* $\sigma_h$: Stochastic deviation coefficient determined by parameter $\eta$ (eta).
* $z \sim \mathcal{N}(0, I)$: Standard Gaussian noise.

---

```bash
brew install deno;python3 venv .venv;source ./.venv/bin/activate;pip3 install jax jaxlib optax numpy demucs scipy
python3 processing.py
python3 model.py
python3 inference.py --compile --seconds 10
python3 inference.py --generate --seconds 10
```

---

#### `model.py`

* `rms_norm`: Applies Root Mean Square normalization across the feature dimension with scaling and numerical stability epsilon.
* `apply_rope`: Applies Rotary Position Embeddings (RoPE) to query and key tensor representations.
* `scaled_dot_product_attention`: Computes scaled multi-head attention weights and applies optional causal masking.
* `clamp_frobenius_norm`: Iteratively projects tensor weights back to a Frobenius norm boundary limit of one.
* `combined_audio_loss`: Evaluates hybrid audio optimization objectives combining STFT spectral magnitude differences, log spectral convergence, and L1 waveform distance.
* `compute_empirical_ntk`: Evaluates the empirical Neural Tangent Kernel matrix and its condition number and trace via VJP Jacobians.
* `gpt_forward`: Implements hierarchical transformer diffusion blocks using RMSNorm, RoPE, multi-head attention, packed conditioning tuples, and weight norm clamping.
* `xavier_normal`: Initializes tensor weights using Xavier normal distribution scaling based on fan-in and fan-out dimensions.
* `init_params`: Constructs and validates the complete dictionary of model weight parameters initialized with float32 assertions.
* `get_cached_metadata`: Reads newline-delimited JSON metadata entries from audio vault records.
* `raw_memmap_loader`: Continuously yields batch tensors, condition metadata, and granular sample IDs sampled strictly within individual track duration bounds from memory-mapped shards.
* `load_checkpoint_safely`: Acquires an exclusive file lock to securely load the latest checkpoint bundle from disk.
* `push_and_pull_gradients`: Coordinates synchronized gradient accumulation across workers, advancing optimizer states and model versioning via file locks.

---

#### `inference.py`

* `float_to_hex_halves`: Converts a floating-point literal into lower and upper 16-bit integer halves for ARM immediate mov operations.
* `compile_closed_jaxpr_to_arm64`: Translates a closed JAX expression into optimized ARM64 assembly with vector register allocation and loop unrolling.
* `compile_closed_jaxpr_to_cuda`: Generates and wraps an NVIDIA CUDA C kernel from a closed JAX expression for parallel device execution.
* `HeterogeneousRuntime`: Manages compilation and concurrent multithreaded execution of compiled ARM64 and CUDA binary runtimes.
