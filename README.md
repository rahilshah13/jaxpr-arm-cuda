#### Audio Diffusion in JAX with an experimental Apple Arm64 Vectorized Backend

* **ARM64 NEON AOT Compiler (`AppleSiliconNeonRuntime`)**: Compiles JAX jaxprs into optimized ARM64 assembly via `clang` and executes through `ctypes`.
* **Diffusion Transformer Core (`gpt_forward`)**: Implements transformer blocks using RMSNorm, RoPE, attention, weight norm clamping, and STFT loss.
* **JAX-Native DDIM Sampler (`generate_audio_ddim`)**: Runs fast denoising loops via `jax.lax.scan` with classifier-free guidance (CFG).

---

#### Terminology

* **AOT (Ahead-Of-Time):** Compiles high-level jaxprs into native hardware assembly.
* **NEON:** ARM's SIMD (Single Instruction, Multiple Data) architecture extension.
* **DDIM (Denoising Diffusion Implicit Models):** A non-Markovian deterministic sampling formulation.
* **CFG (Classifier-Free Guidance):** Blends conditional and unconditional noise predictions.
* **RoPE (Rotary Position Embedding):** Encodes positions by rotating representations based on absolute token indices.
* **EMA (Exponential Moving Average):** Smooths weights across training updates to stabilize convergence.

---

The deterministic DDIM sampling update step implemented in `generate_audio_ddim` computes the estimated clean data point and steps backward to the previous latent state via:

$$\hat{x}_0 = \frac{x_t - \sigma_t \epsilon_\theta(x_t, t)}{\alpha_t}$$

$$x_{t-1} = \alpha_{t-1} \hat{x}_0 + \sqrt{1 - \alpha_{t-1}^2 - \sigma_h^2} \, \epsilon_\theta(x_t, t) + \sigma_h z$$

Where:

* $x_t$: Noisy latent tensor at diffusion step $t$.
* $\epsilon_\theta(x_t, t)$: Predicted noise vector from the transformer backbone.
* $\alpha_t, \sigma_t$: Noise schedule multipliers where $\alpha_t^2 + \sigma_t^2 = 1$.
* $\sigma_h$: Stochastic deviation coefficient determined by parameter $\eta$ (eta).
* $z \sim \mathcal{N}(0, I)$: Standard Gaussian noise.

---

### CLI Execution Modes

```bash
pip install jax jaxlib optax numpy

# native JAX Backend 
python3 model.py

# arm64 neon 
python3 model.py --aot

# audio generation
python3 model.py --sample

```
