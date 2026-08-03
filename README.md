#### Audio Diffusion in JAX + an experimental Arm64 Neon Backend

* **`AppleSiliconNeonRuntime`**: Compiles JAX jaxprs into ARM64 assembly via `clang` and executes through `ctypes`.
* **`gpt_forward`**: diffusion transformer blocks using RMSNorm, RoPE, attention, weight norm clamping, and STFT loss.
* **`generate_audio_ddim`**: Runs fast denoising loops via `jax.lax.scan` with classifier-free guidance (CFG).

---

The sampling update step `generate_audio_ddim` estimates the clean data point and steps backward to the previous latent state via:

$$\hat{x}_0 = \frac{x_t - \sigma_t \epsilon_\theta(x_t, t)}{\alpha_t}$$

$$x_{t-1} = \alpha_{t-1} \hat{x}_0 + \sqrt{1 - \alpha_{t-1}^2 - \sigma_h^2} \, \epsilon_\theta(x_t, t) + \sigma_h z$$


* $x_t$: Noisy latent tensor at diffusion step $t$.
* $\epsilon_\theta(x_t, t)$: Predicted noise vector from the transformer backbone.
* $\alpha_t, \sigma_t$: Noise schedule multipliers where $\alpha_t^2 + \sigma_t^2 = 1$.
* $\sigma_h$: Stochastic deviation coefficient determined by parameter $\eta$ (eta).
* $z \sim \mathcal{N}(0, I)$: Standard Gaussian noise.

---

```bash
pip install jax jaxlib optax numpy

# native JAX Backend 
python3 model.py

# arm64 neon 
python3 model.py --aot

# audio generation
python3 model.py --sample
```
