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
