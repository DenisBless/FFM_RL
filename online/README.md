# FFM: online on-policy maximum-entropy RL

This folder contains the online RL experiments of
[Reinforcement Learning via Fixed-point Flow Matching](../README.md), where
flow policies are trained from scratch. The offline-to-online experiments
are in [`../offline2online`](../offline2online).

It implements **FFM** for online on-policy
reinforcement learning: a flow policy (a velocity field integrated as an
ODE) trained by a fixed-point flow-matching regression onto the critic's
action gradients, inside a massively parallel soft actor--critic loop
(TD(lambda) targets on soft rewards, HL-Gauss distributional critic,
divergence-form entropy via Hutchinson's estimator, velocity-space trust
region).

The trainer is a single file, `src/jaxrl/ffm_submission.py`; the default
configuration, `config/config.yaml`, is exactly the configuration used for
the published MuJoCo Playground experiments. All commands below are run
from this folder.

## Installation

We recommend [uv](https://docs.astral.sh/uv/getting-started/installation/)
for dependency management. With uv installed:

```bash
uv sync
```

installs the project and all dependencies into a local `.venv`. A GPU with
CUDA 12 compatible drivers is required. With other tools (e.g. conda),
create a `Python 3.12` environment and run `pip install -e .`.

## Running experiments

Configuration is handled with [hydra](https://hydra.cc/); any parameter can
be overridden on the command line.

MuJoCo Playground DMC tasks (the default configuration, discount 0.99):

```bash
uv run python src/jaxrl/ffm_submission.py env.name=CheetahRun seed=0
```

MJX humanoid joystick tasks (discount 0.97):

```bash
uv run python src/jaxrl/ffm_submission.py env=mjx_humanoid \
    env.name=G1JoystickFlatTerrain hyperparameters.gamma=0.97 seed=0
```

The available task names are listed in `config/env/mjx_dmc.yaml` and
`config/env/mjx_humanoid.yaml`. Logging goes to [wandb](https://wandb.ai)
(`wandb.mode=disabled` to turn it off). Evaluation runs the deterministic
central action (the ODE integrated from the prior mean); training rollouts
are stochastic.

## Flow-map policies: one- and two-step sampling

`src/jaxrl/flowmap_ffm.py` (config `config/flowmap_ffm.yaml`) trains the
policy as a flow map u(x, o, t, r), the average velocity of the flow from
time t to time r, instead of a velocity field. Actions are sampled with
`map_steps` Euler steps of the map: two network evaluations by default, or
one with `map_steps=1`. FFM integrates 16 ODE steps.

Critic, soft TD(lambda) backups, trust region, noise schedule and evaluation
protocol are FFM's. The changes are:

- **One network.** A single network represents the map, and at r = t the
  velocity.
- **Velocity loss.** Half of each actor minibatch carries FFM's fixed-point
  regression on the velocity.
- **Map loss.** The other half regresses the map on (t, r) pairs by
  Lagrangian self-distillation
  ([Boffi, Albergo & Vanden-Eijnden, 2025](https://arxiv.org/abs/2505.18825))
  from the network's own velocity: the endpoint
  X = x_t + (r - t) u(x_t, t, r) must satisfy dX/dr = u(X, r, r). The
  critic reaches the map only through the velocity.
- **Entropy.** The soft reward uses the exact entropy of the discrete
  sampler, computed from the log-determinants of the map's steps.
- **Temperature.** The temperature dual reads the divergence entropy of the
  velocity ODE, and has an anti-windup cap.

The docstring of `flowmap_ffm.py` gives the details. A single run:

```bash
uv run python src/jaxrl/flowmap_ffm.py env.name=CheetahRun seed=0              # two-step
uv run python src/jaxrl/flowmap_ffm.py env.name=CheetahRun seed=0 \
    hyperparameters.diffusion.flowmap.map_steps=1                              # one-step
uv run python src/jaxrl/flowmap_ffm.py env=mjx_humanoid \
    env.name=G1JoystickFlatTerrain hyperparameters.gamma=0.97 seed=0           # humanoid
```

The FFM baselines with few integration steps are the FFM trainer with
`hyperparameters.diffusion.diff_steps=K`, e.g.

```bash
uv run python src/jaxrl/ffm_submission.py env.name=CheetahRun seed=0 \
    hyperparameters.diffusion.diff_steps=2
```

### Reproducing the results

`scripts/run_flowmap.sh <map_steps> <task> <seed>` runs one experiment with
the settings of its task family. It writes
`outputs/flowmap_k<map_steps>/<task>__s<seed>/metrics.npz` and skips runs that
are already finished. `scripts/flowmap_jobs.sh` prints the full job list:
20 DMC and 4 humanoid tasks, seeds 0-9, so 240 runs per sampler. Run the
lists from this folder with the environment active
(`source .venv/bin/activate`):

```bash
bash scripts/flowmap_jobs.sh 2 > jobs_k2.txt        # two-step sampler
bash scripts/flowmap_jobs.sh 1 > jobs_k1.txt        # one-step sampler

bash jobs_k2.txt                                                     # one GPU, in turn
parallel -j 4 'CUDA_VISIBLE_DEVICES=$(({%} - 1)) {}' < jobs_k2.txt   # 4 GPUs, GNU parallel
mkdir -p logs && sbatch --array=1-240 scripts/slurm_array.sh jobs_k2.txt   # SLURM

python scripts/summarize_flowmap.py                 # per-task finals and IQM
```

A DMC run takes 15-35 minutes on a V100, and a humanoid run 1.5-3.5 hours,
so the humanoid lines checkpoint every two evaluation chunks
(`RESUME_EVERY=2`); relaunching an interrupted line resumes it. Every run
fits on a 16 GB V100.

The interquartile mean (IQM) of the final return over all runs, with 95%
stratified bootstrap intervals (10 seeds per task), is:

| | DMC (return / 1000) | Humanoids |
|---|---|---|
| flow map, two-step | 0.864 [0.838, 0.884] | 34.4 [34.0, 34.7] |
| flow map, one-step | 0.844 [0.831, 0.856] | 33.3 [32.9, 33.7] |
| FFM, 2 ODE steps | 0.876 [0.863, 0.888] | 20.2 [19.4, 20.8] |
| FFM, 1 ODE step | 0.830 [0.810, 0.847] | 6.3 [5.9, 6.6] |
| FFM, 16 ODE steps (default) | 0.869 [0.848, 0.888] | 35.4 [35.2, 35.6] |

With one or two network evaluations per action, the flow map comes close
to FFM with 16 integration steps, while FFM itself degrades with few steps
on the humanoids. The two-step sampler is the default; the second step costs
almost nothing in training.

<details>
<summary>Final return per task</summary>

The return of the last of the 21 evaluations, mean ± std over 10 seeds.

| task | flow map, two-step | flow map, one-step | FFM, 2 steps | FFM, 1 step | FFM, 16 steps |
|---|---|---|---|---|---|
| PendulumSwingup | 233 ± 292 | 227 ± 263 | 563 ± 337 | 191 ± 226 | 574 ± 336 |
| AcrobotSwingup | 328 ± 37 | 357 ± 28 | 412 ± 28 | 325 ± 35 | 398 ± 30 |
| AcrobotSwingupSparse | 0 ± 0 | 0 ± 0 | 15 ± 27 | 1 ± 2 | 8 ± 10 |
| BallInCup | 780 ± 390 | 877 ± 292 | 973 ± 6 | 974 ± 0 | 975 ± 0 |
| CartpoleBalance | 999 ± 1 | 989 ± 22 | 929 ± 93 | 926 ± 23 | 999 ± 1 |
| CartpoleBalanceSparse | 793 ± 358 | 984 ± 47 | 974 ± 77 | 914 ± 177 | 600 ± 420 |
| CartpoleSwingup | 863 ± 4 | 864 ± 3 | 840 ± 41 | 794 ± 29 | 865 ± 3 |
| CartpoleSwingupSparse | 795 ± 38 | 783 ± 45 | 772 ± 65 | 712 ± 248 | 691 ± 235 |
| CheetahRun | 912 ± 48 | 926 ± 5 | 928 ± 3 | 882 ± 68 | 928 ± 2 |
| FingerSpin | 729 ± 183 | 612 ± 83 | 880 ± 89 | 779 ± 50 | 839 ± 181 |
| FingerTurnEasy | 932 ± 14 | 927 ± 17 | 934 ± 17 | 917 ± 18 | 943 ± 13 |
| FingerTurnHard | 923 ± 17 | 912 ± 17 | 920 ± 14 | 914 ± 18 | 925 ± 13 |
| FishSwim | 644 ± 19 | 622 ± 16 | 634 ± 23 | 501 ± 33 | 609 ± 17 |
| HopperHop | 77 ± 88 | 105 ± 136 | 142 ± 176 | 126 ± 81 | 177 ± 196 |
| HopperStand | 553 ± 390 | 316 ± 295 | 266 ± 296 | 451 ± 367 | 276 ± 362 |
| ReacherEasy | 979 ± 2 | 979 ± 2 | 980 ± 2 | 977 ± 3 | 980 ± 1 |
| ReacherHard | 973 ± 2 | 975 ± 3 | 975 ± 2 | 976 ± 3 | 974 ± 2 |
| WalkerRun | 846 ± 5 | 839 ± 11 | 840 ± 11 | 750 ± 11 | 853 ± 3 |
| WalkerWalk | 976 ± 1 | 975 ± 1 | 976 ± 1 | 971 ± 6 | 977 ± 1 |
| WalkerStand | 985 ± 2 | 986 ± 2 | 982 ± 10 | 985 ± 1 | 987 ± 1 |
| G1JoystickFlatTerrain | 36.2 ± 0.2 | 34.8 ± 0.4 | 10.7 ± 4.3 | -1.2 ± 0.9 | 37.0 ± 0.4 |
| G1JoystickRoughTerrain | 31.9 ± 0.3 | 30.8 ± 0.3 | 17.9 ± 1.2 | 0.2 ± 0.8 | 31.7 ± 0.8 |
| T1JoystickFlatTerrain | 35.6 ± 1.7 | 34.8 ± 1.5 | 27.1 ± 1.3 | 18.7 ± 0.9 | 38.1 ± 0.6 |
| T1JoystickRoughTerrain | 33.6 ± 0.4 | 32.5 ± 0.7 | 21.8 ± 1.3 | 12.2 ± 1.0 | 33.8 ± 0.6 |

</details>

These runs used JAX 0.5.2 (jaxlib 0.5.1), Flax 0.10.6, Brax 0.13.0, MuJoCo
and MJX 3.3.7, and the playground fork at commit `1699a6d`. `uv sync`
installs JAX 0.6.0 and Flax 0.11 instead, which run the same code. Runs are
not bit-reproducible on GPU, not even twice in one environment, because
XLA's autotuning picks kernels by timing. Compare results as distributions
over seeds, not run by run.

## Acknowledgments

This implementation builds on the
[REPPO](https://arxiv.org/abs/2507.11019) codebase, which in turn builds on
[PureJaxRL](https://github.com/luchris429/purejaxrl) and
[FastTD3](https://github.com/younggyoseo/FastTD3). We thank the respective
authors for making their work available in open source.

## License

MIT -- see the top-level [LICENSE](../LICENSE). [LICENSE](LICENSE) in this
folder retains the notices of REPPO and FastTD3 (MIT) and PureJaxRL
(Apache 2.0).
