# Tutorial 2 — Point Clouds on $S^2$ and $S^3$

**MM845 — Tópicos de Geometria III: AI for Geometry**
Paired with **Lecture 2: Mathematical Foundations of Machine Learning**

---

## The idea

Lecture 2 defined learning as a variational problem over a function space, with the
measure known only through samples, and claimed that each of its three ingredients —
approximation theory, optimisation, statistics — carries geometric content. This
tutorial turns that claim into code on the simplest interesting manifolds available:
the round spheres $S^2 \subset \mathbb{R}^3$ and $S^3 \subset \mathbb{R}^4$.

Spheres are the right laboratory because **we know every answer in advance**. The
intrinsic dimension, the uniform measure, the geodesic distance, the eigenfunctions
of the Laplace–Beltrami operator, the isometry group — all in closed form. So when a
machine-learning method recovers one of them we can verify it, and when it fails we
can see precisely where. Real datasets never offer this, which is why the whole
course works with geometric data.

## Files

| File | What it is |
|---|---|
| [`point_clouds_on_spheres.ipynb`](point_clouds_on_spheres.ipynb) | The tutorial. 9 sections, 23 figures, 12 exercises. |
| `README.md` | This file. |

## Running it

You need the `aigeo` environment from [Tutorial 1](../tutorial_01/README.md).
Nothing beyond NumPy, SciPy and Matplotlib is used — no PyTorch, no scikit-learn.

```bash
$ conda activate aigeo
$ jupyter lab point_clouds_on_spheres.ipynb
```

Check the kernel indicator at the top right says `aigeo`. The whole notebook runs
top to bottom in well under a minute; **Kernel → Restart Kernel and Run All Cells**
should complete without error before you start editing.

For rotatable 3-D views instead of static images, `pip install ipympl` and uncomment
the `%matplotlib widget` line in the setup cell. Recommended — several figures are
much more convincing when you can turn them.

## Contents

| § | Topic | Lecture 2 connection |
|---|---|---|
| 1 | Sampling $S^1$, $S^2$: what "uniform" means, and the standard way to get it wrong | the measure $\mu$ |
| 2 | Fibonacci lattices, farthest-point sampling, convex-hull triangulation | representation choices |
| 3 | Chordal vs geodesic distance; the exact distribution of pairwise distances | metric geometry of data |
| 4 | $SO(3)$ acting on $S^2$: invariance, orbits, orbit-aware splits | slide 11 |
| 5 | $S^3$, unit quaternions, the Hopf fibration | visualising the undrawable |
| 6 | Concentration of measure and the curse of dimensionality | slide 9 |
| 7 | The manifold hypothesis: recovering $\dim = 2$ from $\mathbb{R}^{100}$ | slide 8 |
| 8 | Supervised learning on $S^2$ as energy minimisation; the U-curve | slides 4, 6, 7 |
| 9 | Summary and pointers | — |

## Results worth watching for

A few cells produce output that is more than an illustration, and it is worth
knowing in advance what to look at.

**§8.2 — the numerics rediscover spherical harmonics.** Monomials of degree $\le D$
in three variables number $\binom{D+3}{3}$, but their restrictions to $S^2$ span only
a $(D+1)^2$-dimensional space, because the restriction map kills the ideal generated
by $x^2+y^2+z^2-1$. The printed matrix ranks come out as $4, 9, 16, 25, \dots$
exactly — that is $\sum_{\ell \le D}(2\ell+1)$, the dimension of the span of the
Laplacian's eigenspaces. Nothing about the decomposition was put in by hand.

**§8.4 — the U-curve, then double descent.** With $N_{\text{train}} = 80$ noisy
samples, test error bottoms out at degree 3 (the true degree of the target), and
explodes by four orders of magnitude at degree 8 — which is where
$(D+1)^2 = 81 \approx N$, the interpolation threshold. Past it the error falls again
by an order of magnitude, because `np.linalg.lstsq` silently returns the
minimum-norm interpolant and that choice acts as an implicit regulariser. Lecture 2's
"modern twist" appears here in a model with no neural network in it.

**§7 — a local spectrum with three regimes.** Local PCA on the point cloud embedded
in $\mathbb{R}^{100}$ returns mean singular values $\approx (1,\ 0.86,\ 0.21,\ 0.06,
\ 0.045,\ 0.043)$. The first two are the tangent plane, the third is the second
fundamental form (it scales like $\kappa r^2$, so it shrinks quadratically as the
neighbourhood shrinks), and the rest are the ambient noise floor. Estimating
intrinsic dimension means choosing a radius inside the window
$\sigma_{\text{noise}} \ll r \ll 1/\kappa$; Exercise 8 asks you to find it.

**§4 — an identity that fails at `atol=1e-10`.** Rotations preserve inner products to
machine precision but geodesic distances only to about $10^{-8}$, because
$\arccos$ is ill-conditioned near $\pm 1$. Half the significant digits are lost for
nearly-coincident points. The lesson generalises: test the well-conditioned quantity.

## Exercises

Twelve, spread through the notebook. Solutions are discussed in the session; hints
are inline.

| # | § | Topic |
|---|---|---|
| 1 | 1 | Independent tests for uniformity; spherical-cap Monte Carlo |
| 2 | 1 | Why the naive scheme is correct on $S^1$ but not $S^2$; attempt Archimedes on $S^3$ |
| 3 | 2 | Euler characteristic; area convergence of a triangulation |
| 4 | 3 | The density $\rho_n(d) \propto \sin^{n-2} d$; $k$-NN graph distance vs true geodesic (Isomap) |
| **5** | 4 | **Orbit-aware train/test splitting — the leakage trap** |
| 6 | 5 | Quaternion group structure; disjointness and linking of Hopf fibres; $\mathbb{RP}^3$ |
| 7 | 6 | Sub-Gaussian concentration; covering numbers; the escape via intrinsic dimension |
| 8 | 7 | When tangent-plane recovery breaks: noise sweep, nonlinear embeddings, $S^3$ |
| 9 | 8 | Learning curves and the bias–variance regimes |
| 10 | 8 | Refitting in a symmetry-adapted (harmonic) basis — Lecture 10 in miniature |
| **11** | 8 | **Predicting Gaussian curvature from local point-cloud geometry — mini-project seed** |
| 12 | 8 | Double descent past the interpolation threshold |

Exercise 5 is the one to do if you do only one. Exercise 11 is a viable starting
point for the assessed mini-project.

## What to take away

- **Sampling is a modelling decision.** "Uniform" means invariant under the isometry
  group; the naive parametrisation is wrong and the error is invisible until you test
  a statistic whose exact distribution you know. Always have such a statistic.
- **Vectorise.** Every distance computation here is a Gram matrix. This is not only
  about speed — array-shaped thinking is how tensors, batches and differentiable
  programming work from Lecture 3 onwards.
- **Symmetry is data structure.** Ignoring it wastes capacity; ignoring it *when
  splitting the data* invalidates the experiment.
- **High dimension is not more of the same.** Distances concentrate, random vectors
  become orthogonal, grids become hopeless — and the same concentration is what makes
  empirical risk approximate true risk.
- **The manifold hypothesis is testable.** Global PCA sees the ambient span; local
  PCA sees the tangent space. Both give the right answer here, and you can watch them
  fail as noise grows.
- **The U-curve is real**, and it can be reproduced from first principles with
  nothing more exotic than least squares.

## Next

**Lecture 3** develops the convex, linear-model world where every claim above can be
proved. **Tutorial 3** applies it to geometric classification and regression on
random plane curves and triangulated surfaces.

## Further reading

- Vershynin, *High-Dimensional Probability* (CUP, 2018), ch. 3 & 5 — concentration, rigorously. Free online.
- Atkinson & Han, *Spherical Harmonics and Approximation on the Unit Sphere* (Springer, 2012).
- Lyons, "An elementary introduction to the Hopf fibration", *Math. Mag.* **76** (2003) 87–98 — the source of §5's pictures.
- Belkin, Hsu, Ma & Mandal, "Reconciling modern machine-learning practice and the classical bias–variance trade-off", *PNAS* **116** (2019) — double descent, for §8.4.
- Bronstein, Bruna, Cohen & Veličković, *Geometric Deep Learning* (2021), ch. 3 — symmetry as a design principle; where Exercise 10 leads.
