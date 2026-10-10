# Reinforcement Learning via Fixed-point Flow Matching

Official code for **Fixed-point Flow Matching (FFM)**: a data-free procedure for training and
fine-tuning flow policies in RL. This repository reproduces the offline-to-online experiments of
the paper on the ten OGBench domains.

<p align="center">
  <img alt="The FFM policy-improvement step" src="assets/method.png" width="100%">
</p>

FFM trains a flow policy against the Boltzmann target π⁺ ∝ p_ref(a)·exp(τ_Q Q(s,a)) with a
flow-matching-style regression whose target is read off the critic instead of a dataset. One
policy-improvement step is: **(1)** generate clean actions with the current flow (forward only, no
gradients); **(2)** label them with the critic's action gradient ∇ₐQ; **(3)** noise them along the
interpolant X_t = t·X₁ + (1−t)·ε; **(4)** regress the velocity field onto the label. The optimal
policy is a fixed point of this loop. There is no backpropagation through the integrator, no
importance weighting, and no conversion to a diffusion model.

The entire method is `agents/ffm.py`. Everything else is environment, data and logging plumbing.

## Installation

```bash
conda create -n ffm python=3.11 -y
conda activate ffm

# JAX for your accelerator (the paper's runs used the CUDA 12 build):
pip install "jax[cuda12]"
pip install -r requirements.txt
```

OGBench pulls in MuJoCo. Rendering is headless:

```bash
export MUJOCO_GL=egl
```

## Datasets

```bash
python scripts/download_datasets.py                 # all ten domains
python scripts/download_datasets.py cube-double     # one domain
```

Datasets go to `~/.ogbench/data` (override with `OGBENCH_DATA`). The standard 1M datasets are
fetched through `ogbench`. Four domains use OGBench's larger releases — antmaze-giant,
cube-triple and puzzle-4x4 (10M) and cube-quadruple (100M) — which are stored as directories of
`.npz` shards that `main.py` cycles through during training (`--ogbench_dataset_dir`).

## Reproducing the paper

Each domain has five tasks. The protocol is: BC-pretrain a reference flow policy (300K steps,
once per domain), then for each task run FFM for 1M offline steps followed by 500K online
environment steps, evaluating every 50K steps over 50 episodes. The paper reports 8 seeds.

```bash
# 1. BC reference policy, once per (domain, seed). The BC policy is task-independent
#    within a domain, so it is trained on task 1 and reused by tasks 1-5.
scripts/run_bc.sh cube-double 10001

# 2. FFM offline -> online on a task
scripts/run_ffm.sh cube-double 3 10001
```

Results land in `exp/ffm/<run_group>/<env>/<run>/`: `eval.csv` holds the evaluation success rate
every 50K steps on a continuous 0–1.5M axis (offline evals at `step`, online evals at
`1,000,000 + step`), `offline_agent.csv` / `online_agent.csv` hold training diagnostics. Logging
also goes to Weights & Biases; set `WANDB_MODE=offline` to keep it local.

### The recipe

`scripts/domains.sh` encodes the per-domain settings used for every number in the paper. Two
regimes, chosen by domain family:

| | locomotion (Regime A) | manipulation (Regime B) |
|---|---|---|
| critic scale `tau_q` | 30 | 1 |
| Fisher normaliser in c(t) | fixed, `i_fisher=1` | adaptive (EMA) |
| KL trust region offline | on, ε_KL = 1 | off (`lam_scale=0`) |
| KL budget online | 8 | 8 |
| action chunking | no | yes, horizon 5 |

Domain-specific extras: humanoidmaze uses `discount=0.999, rho=0`; the four large-dataset
domains use 1024-wide networks with actor layer norm. All other hyperparameters are the
defaults in `agents/ffm.py:get_config`.

To run a single configuration by hand, the two scripts print the exact `main.py` command they
execute; e.g. cube-double task 3 is

```bash
python main.py --env_name cube-double-play-singletask-task3-v0 --agent agents/ffm.py \
  --agent.action_chunking=True --agent.tau_q=1 --agent.adaptive_fisher=True --agent.lam_scale=0 \
  --kl_budget_online=8.0 --offline_steps 1000000 --online_steps 500000 \
  --eval_interval 50000 --eval_episodes 50 --save_interval 250000 --save_last_checkpoint \
  --seed 10001 --run_group ffm --pretrained_actor_path exp/ffm/bc/<...>/params_300000.pkl
```

Interrupted runs resume from their last checkpoint when relaunched with the same arguments. To
start the online phase from a finished offline run, pass its `params_1000000.pkl` as
`--restore_agent_file` with `--offline_steps 0`.

## Code map

| file | what it is |
|---|---|
| `agents/ffm.py` | **the method**: critic loss, BC loss for the reference policy, the FFM regression, the path-space KL trust region with dual descent, flow sampling |
| `main.py` | the offline → online training loop, checkpointing, resume |
| `scripts/domains.sh` | the per-domain recipe |
| `scripts/run_bc.sh`, `scripts/run_ffm.sh` | BC pretraining and the FFM run |
| `scripts/download_datasets.py` | fetches the OGBench datasets |
| `evaluation.py` | rollout evaluation with action chunks |
| `utils/networks.py` | the velocity-field MLP and the critic ensemble |
| `utils/datasets.py`, `utils/flax_utils.py` | dataset / replay buffer and training-state helpers |
| `envs/` | OGBench environment and large-dataset loading |

## Acknowledgement

The training loop, data pipeline and critic are adapted from the
[TRQAM](https://arxiv.org/abs/2605.27079) codebase (MIT), itself built on the
[Q-chunking](https://github.com/ColinQiyangLi/qc) and [OGBench](https://github.com/seohongpark/ogbench)
releases. The MIT notice of the TRQAM authors is retained in `LICENSE`.

## Citation

The paper is under review; a citation will be added once it is public.

## License

MIT — see `LICENSE`.
