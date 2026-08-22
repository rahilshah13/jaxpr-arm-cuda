#### audio diffusion in JAX


##### 1. $\mathcal{O}(B, T, N, C)$ Scaling

$$\mathcal{C}(B, T, N, C) = \mathcal{O}\left(B \cdot T \cdot N \cdot C^2 + B \cdot T \cdot N^2 \cdot C\right)$$

* Computational complexity scaling linearly with projections and quadratically with patch attention over batch $B$, time $T$, patches $N$, and channels $C$.

##### 2. Variational Upper Bound

$$\mathcal{L}_{\text{infer}} \le \mathbb{E}_{t, x_0, \epsilon} \left[ \Vert{}\epsilon_\theta(x_t, t) - \epsilon\Vert{}^2 + \lambda \left( \Vert{}\text{STFT}(x_0) - \text{STFT}(\hat{x}_0)\Vert{}_1 + \Vert{}x_0 - \hat{x}_0\Vert{}_1 \right) \right]$$

* Generative loss bounding noise prediction error alongside weighted spectral and temporal L1 reconstruction penalties.

---

#### Neural Network Kernel (`gpt_forward`)

$$\mathbf{h}_p = \text{RMSNorm}\left(\text{Attention}(\text{RoPE}(\mathbf{q}_p), \text{RoPE}(\mathbf{k}_p), \mathbf{v}_p) + \mathbf{p}_{\text{seq}}\right)$$

* Patch-level representation via normalized rotary attention and sequence positions.

$$\mathbf{h}_t = \text{RMSNorm}\left(\text{Attention}(\text{RoPE}(\mathbf{q}_t), \text{RoPE}(\mathbf{k}_t), \mathbf{v}_t) + \mathbf{p}_{\text{time}}\right)$$

* Temporal-level representation via normalized rotary attention and time positions.

$$\mathbf{h}_{\text{out}} = \text{FFN}(\mathbf{h}_p + \mathbf{h}_t + \mathbf{c}_{\text{base}} + \sigma_t \mathbf{e}_{\sigma})$$

* Output features combining patch, temporal, conditioning tuples, and noise embeddings through a feed-forward network.

---

#### Diffusion Latent Trajectory

$$\hat{x}_0 = \frac{x_t - \sigma_t \epsilon_\theta(x_t, t)}{\alpha_t}$$

* Estimated clean audio latent derived from noisy input and predicted noise.

$$x_{t-1} = \alpha_{t-1} \hat{x}_0 + \sqrt{1 - \alpha_{t-1}^2 - \sigma_h^2} \, \epsilon_\theta(x_t, t) + \sigma_h z$$

* Latent update step moving from diffusion step $t$ to $t-1$.
* $x_t$: Noisy multi-channel latent tensor at diffusion step $t$.
* $\epsilon_\theta(x_t, t)$: Predicted multi-channel noise vector.
* $\alpha_t, \sigma_t$: Noise schedule multipliers ($\alpha_t^2 + \sigma_t^2 = 1$).
* $\sigma_h$: Stochastic deviation coefficient.
* $z \sim \mathcal{N}(0, I)$: Standard Gaussian noise.

---

#### `model.py`

* `rms_norm`: Root Mean Square normalization across feature dimensions.
* `apply_rope`: Rotary Position Embeddings for queries and keys.
* `scaled_dot_product_attention`: Multi-head attention with optional causal masks.
* `clamp_frobenius_norm`: Projects tensor weights to a Frobenius norm limit of one.
* `combined_audio_loss`: Hybrid STFT, log-spectral, and L1 waveform objective.
* `compute_empirical_ntk`: Evaluates the empirical Neural Tangent Kernel via VJP Jacobians.
* `gpt_forward`: Hierarchical transformer diffusion block processing multi-channel stems and discrete metadata (scale, tempo/BPM, stem index).
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
```