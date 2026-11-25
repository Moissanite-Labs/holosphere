# Kernel-Transformer Isomorphism in Project HoloSphere

## 1. Hardware-Geometry Identification

Traditional geometric transformers treat Lie groups ($SU(2)$, $SO(4)$, Hopf fibrations) as high-level tensor graphs that launch dozens of tiny elementwise PyTorch kernels per layer, paying `2.5×–3.9×` HBM bandwidth penalties (`S106 H6c`: `2.57×`; `S107 H7b`: `3.913×`).

**HoloSphere** eliminates the separation between kernel engineering and neural architecture through a single exact hardware-algebraic identification on NVIDIA Ampere (`sm_86`):

$$\text{1 unit quaternion } q = (w, x, y, z) \in SU(2) \cong S^3 \;\longleftrightarrow\; \text{1 aligned } \texttt{float4} \text{ quad } (\texttt{LDG.E.128}, \text{ 4 FP32 registers}).$$

Every head of dimension $D_h = 4 B$ decomposes into $B$ parallel quaternion lanes (`B = 8` for `d_head = 32`, `B = 10` for `d_head = 40`, `B = 16` for `d_head = 64`), so a single 32-thread warp holds `32` consecutive sequence positions $t \in [0, 31]$ for a quaternion lane entirely in registers!

---

## 2. Forward Associative Scan & Pure-Additive Backward Adjoint

### 2.1 Forward Pass (`5` Warp Shuffles, `0 B` Spills)
Given per-token $\mathfrak{su}(2)$ Lie-algebra generators $\omega_t \in \mathbb{R}^3$:
1. Compute bounded angle $\tilde{\omega}_t = \alpha \tanh(\omega_t)$ and smooth norm $\theta_t = \sqrt{\|\tilde{\omega}_t\|^2 + \epsilon^2}$.
2. Exponential map in registers: $\delta u_t = \left(\cos\theta_t, \frac{\sin\theta_t}{\theta_t}\tilde{\omega}_t\right) \in SU(2)$.
3. Associative Hillis–Steele right-accumulating prefix product across 32 warp lanes in **5** `__shfl_up_sync(0xFFFFFFFFu, ...)` rounds (plus hierarchical block-prefix combine for $T = 512..2048+$):
   $$\Pi_t = \delta u_0 \cdot \delta u_1 \cdots \delta u_t \in SU(2).$$
4. Because quaternion multiplication is strictly associative (mechanized in `lean/HoloSphereInvariants.lean`), parallel $O(\log T)$ training scan equals exact $O(1)$-per-token streaming inference $\Pi_t = \Pi_{t-1} \cdot \delta u_t$ bit-for-bit in exact arithmetic.

### 2.2 Reverse-Mode Adjoint Collapses to a Pure Additive Suffix Sum
Let $g_j = \frac{\partial \mathcal{L}}{\partial \Pi_j} \in \mathbb{R}^4$ be the upstream gradient at step $j$. The exact reverse recurrence for total adjoint $A_t$ satisfies:
$$A_t = g_t + A_{t+1} \cdot \overline{u_{t+1}} = \sum_{j=t}^{T-1} g_j \cdot \overline{u_{t+1} \cdots u_j}.$$
Because every $\Pi_k \in SU(2)$ is a unit quaternion ($\Pi_k^{-1} = \overline{\Pi_k}$), the interval conjugate factors as $\overline{u_{t+1} \cdots u_j} = \overline{\Pi_t^{-1}\Pi_j} = \overline{\Pi_j}\Pi_t$, yielding:
$$A_t = \underbrace{\left(\sum_{j=t}^{T-1} g_j \cdot \overline{\Pi_j}\right)}_{S_t} \cdot \Pi_t, \qquad \frac{\partial \mathcal{L}}{\partial u_t} = \overline{\Pi_{t-1}} \cdot S_t \cdot \Pi_t.$$

**Silicon Consequence:** The backward scan computes **ZERO quaternion multiplies inside the warp shuffle loop** — only one local `float4` product $g_j \cdot \overline{\Pi_j}$, one 5-step additive `__shfl_down_sync` FP32 suffix sum $S_t$, and two local `float4` products! This eliminates backward register pressure and guarantees `0 B` spills under `nvcc -O3 -arch=sm_86 --ptxas-options=-v`, verified to `< 1e-10` against PyTorch autograd and FP64 `torch.autograd.gradcheck` in `holosphere/kernels/torch_ops.py`.

---

## 3. Full Covariant $(Q, K, V)$ Parallel Transport (`H14c`)

In Track `S107 H7b`, only $Q$ and $K$ were transported by $\Pi_t$, giving exact interval holonomy in attention logits ($\langle L(\Pi_t) Q_t, L(\Pi_s) K_s \rangle = \langle Q_t, L(\Pi_t^{-1} \Pi_s) K_s \rangle$) while leaving $V_s$ untransported.

HoloSphere extends the *same* in-register $\Pi_t$ to transport $V_s$ and un-transport the attention output $O_t$ at **zero extra prefix-scan cost**:
$$\hat{Q}_t = L(\Pi_t) Q_t, \quad \hat{K}_s = L(\Pi_s) K_s, \quad \hat{V}_s = L(\Pi_s) V_s, \quad O_t^{\mathrm{cov}} = L(\Pi_t^{-1})\!\left(\sum_{s \le t} w_{ts} \hat{V}_s\right) = \sum_{s \le t} w_{ts} L(\Pi_t^{-1} \Pi_s) V_s.$$
Thus value vectors flowing from token $s$ to token $t$ are rotated by the exact non-Abelian path-ordered holonomy $\Pi_t^{-1} \Pi_s = (\delta u_{s+1} \cdots \delta u_t)^{-1} \in SU(2)$ without modifying FlashAttention / SDPA inner loops!

---

## 4. Non-Antipodal Acute-Cone & Rational Cayley Safe-Hopf Clocks (`W10-OBS-31` Cure)

Track `S107 H7c` uncovered **`W10-OBS-31`**: an unbounded coprime phase progression $\phi_r = 2\pi r / 11$ passes through $\phi_5 = 10\pi/11 \approx \pi$ at test-time loop depth $R_{\text{test}} = 6$, antipodally flipping the core update ($\cos\phi_5 \approx -0.96$) and crashing BPB from `3.12` to `8.07`.

HoloSphere provides two drop-in non-antipodal loop clocks guaranteeing strictly positive cosine and strictly increasing non-saturating phases across **all** $r \ge 0$:
1. **`acute_cone`:** $\phi_r = \phi_{\max} \frac{r}{r + \tau}$ with $0 < \phi_{\max} < \pi/2,\; \tau > 0 \implies \cos\phi_r > \cos\phi_{\max} > 0$.
2. **`cayley_acute` (SFU-free pure-rational form matching `Lean 4`):** $t_r = t_{\max} \frac{r}{r + \tau} \in [0, t_{\max}) \subset [0, 1)$,
   $$c_r = \frac{1 - t_r^2}{1 + t_r^2} > \frac{1 - t_{\max}^2}{1 + t_{\max}^2} > 0, \qquad s_r = \frac{2 t_r}{1 + t_r^2}, \qquad c_r^2 + s_r^2 = 1.$$
Because `cayley_acute` uses strictly rational arithmetic (zero `cosf`/`sinf`/`tanhf` SFU instructions, zero floating-point saturation at large $r$), it fuses directly into `sm_86` registers AND matches the exact rational inequality mechanized in `lean/HoloSphereInvariants.lean`.
