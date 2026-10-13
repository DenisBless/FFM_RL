"""Offline-to-online training of a flow policy with Fixed-point Flow Matching (FFM).

Protocol (as in the paper):
  1. BC pretraining of the reference flow policy on the dataset  (--bc_only=True)
  2. offline RL for --offline_steps on the dataset                 (--pretrained_actor_path)
  3. online fine-tuning for --online_steps in the environment

See scripts/run_bc.sh and scripts/run_ffm.sh for the exact per-domain commands.
"""
import glob
import json
import os
import pickle
import random
import time

import flax
import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_env_and_datasets
from envs.ogbench_utils import make_ogbench_env_and_datasets
from evaluation import evaluate
from log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb
from utils.datasets import Dataset, ReplayBuffer
from utils.flax_utils import restore_agent, restore_agent_with_file, save_agent

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group (second level of the save directory).')
flags.DEFINE_string('tags', 'Default', 'Comma-separated wandb tags; also prefixes the run name.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-double-play-singletask-task3-v0', 'OGBench environment name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline gradient steps.')
flags.DEFINE_integer('online_steps', 500000, 'Number of online environment steps.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 50000, 'Evaluation interval (0 disables intermediate evals).')
flags.DEFINE_integer('save_interval', 50000, 'Checkpoint interval (0 disables).')
flags.DEFINE_integer('start_training', 5000, 'Online steps collected before updates start.')
flags.DEFINE_integer('utd_ratio', 1, 'Online update-to-data ratio.')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of extra rendered episodes per evaluation.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

config_flags.DEFINE_config_file('agent', 'agents/ffm.py', lock_config=False)

flags.DEFINE_string('ogbench_dataset_dir', None,
                    'Directory of a large OGBench dataset split into several .npz files '
                    '(the 10M / 100M variants). The files are cycled during training.')
flags.DEFINE_integer('dataset_replace_interval', 1000,
                     'Steps between switching to the next dataset file when --ogbench_dataset_dir is set.')
flags.DEFINE_integer('horizon_length', 5, 'Action chunk length (used when agent.action_chunking=True).')

flags.DEFINE_bool('bc_only', False, 'Only train the reference flow policy with the BC flow-matching loss.')
flags.DEFINE_string('pretrained_actor_path', None,
                    'params_*.pkl of a BC run; its flow policy initialises both the reference and the '
                    'fine-tuned actor.')
flags.DEFINE_string('restore_agent_file', None,
                    'params_*.pkl to restore the FULL agent state (networks, optimiser, dual variables). '
                    'Use with --offline_steps 0 to start the online phase from a completed offline run.')
flags.DEFINE_float('kl_budget_online', None,
                   'KL budget for the online phase. If None, the offline agent.kl_budget is kept.')

flags.DEFINE_bool('save_last_checkpoint', False, 'Keep the last checkpoint at the end of the run.')
flags.DEFINE_bool('save_replay_buffer', False, 'Keep the replay buffer at the end of the run.')


def save_csv_loggers(csv_loggers, save_dir):
    for prefix, csv_logger in csv_loggers.items():
        csv_logger.save(os.path.join(save_dir, f"{prefix}_sv.csv"))


def restore_csv_loggers(csv_loggers, save_dir):
    for prefix, csv_logger in csv_loggers.items():
        if os.path.exists(os.path.join(save_dir, f"{prefix}_sv.csv")):
            csv_logger.restore(os.path.join(save_dir, f"{prefix}_sv.csv"))


def save_buffer_env_state(buffer, env, action_queue, save_dir):
    if hasattr(env, "get_state"):
        state = env.get_state()
    elif hasattr(env.unwrapped, "get_state"):
        state = env.unwrapped.get_state()
    elif hasattr(env.unwrapped, "data"):
        state = {"qpos": env.unwrapped.data.qpos, "qvel": env.unwrapped.data.qvel}
        if hasattr(env.unwrapped, "button_states"):
            state["button_states"] = env.unwrapped.button_states
    else:
        raise AttributeError(f"Cannot get state from env: {type(env)}")
    env_state = {"env_qpos": np.copy(state["qpos"]), "env_qvel": np.copy(state["qvel"])}
    if "button_states" in state:
        env_state["env_button_states"] = np.copy(state["button_states"])
    if action_queue:
        env_state["action_queue"] = np.stack(action_queue, axis=0)
    np.savez(os.path.join(save_dir, "buffer.npz"), **buffer, **env_state, pointer=buffer.pointer, size=buffer.size)


def restore_buffer_env_state(restore_path):
    buffer_dict = np.load(os.path.join(restore_path, "buffer.npz"))
    buffer_dict = {k: buffer_dict[k] for k in buffer_dict.files}
    pointer = buffer_dict.pop("pointer")
    size = buffer_dict.pop("size")
    state = {"qpos": buffer_dict.pop("env_qpos"), "qvel": buffer_dict.pop("env_qvel")}
    if "env_button_states" in buffer_dict:
        state["button_states"] = buffer_dict.pop("env_button_states")
    if "action_queue" in buffer_dict:
        state["action_queue"] = buffer_dict.pop("action_queue")
    return ReplayBuffer(buffer_dict, pointer=pointer, size=size), state


class LoggingHelper:
    def __init__(self, csv_loggers, wandb_logger):
        self.csv_loggers = csv_loggers
        self.wandb_logger = wandb_logger

    def log(self, data, prefix, step):
        assert prefix in self.csv_loggers, prefix
        self.csv_loggers[prefix].log(data, step=step)
        self.wandb_logger.log({f'{prefix}/{k}': v for k, v in data.items()}, step=step)


def load_pretrained_actor(agent, path):
    """Initialise actor_slow / actor_fast (and their targets) from a BC checkpoint."""
    print(f"Loading pretrained actor from {path}")
    with open(path, 'rb') as f:
        pretrained_params = pickle.load(f)['agent']['network']['params']
    if 'modules_actor_slow' not in pretrained_params:
        raise KeyError(f"No flow policy (modules_actor_slow) in {path}; keys: {list(pretrained_params)}")
    flow = pretrained_params['modules_actor_slow']
    new_params = dict(agent.network.params)
    for name in ('actor_slow', 'target_actor_slow', 'actor_fast', 'target_actor_fast'):
        new_params[f'modules_{name}'] = flow
    return agent.replace(network=agent.network.replace(params=new_params))


def main(_):
    exp_name = get_exp_name(FLAGS)
    env_tag = FLAGS.env_name.rsplit('-v', 1)[0]
    tags = FLAGS.tags.split(",") + [env_tag]
    run = setup_wandb(project='ffm', group=FLAGS.run_group, name=exp_name, tags=tags)
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, FLAGS.env_name, exp_name)

    # ------------------------------------------------------------------ data
    if FLAGS.ogbench_dataset_dir is not None:
        assert FLAGS.dataset_replace_interval != 0
        dataset_idx = 0
        dataset_paths = [
            f for f in sorted(glob.glob(f"{FLAGS.ogbench_dataset_dir}/*.npz")) if '-val.npz' not in f
        ]
        env, eval_env, train_dataset, val_dataset = make_ogbench_env_and_datasets(
            FLAGS.env_name, dataset_path=dataset_paths[dataset_idx], compact_dataset=False,
        )
    else:
        env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name)

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    online_rng, rng = jax.random.split(jax.random.PRNGKey(FLAGS.seed), 2)

    config = FLAGS.agent
    discount = FLAGS.agent.discount
    config["horizon_length"] = FLAGS.horizon_length
    config["bc_only"] = FLAGS.bc_only

    def process_train_dataset(ds):
        return Dataset.create(**ds)

    train_dataset = process_train_dataset(train_dataset)
    example_batch = train_dataset.sample(())

    # ----------------------------------------------------------------- agent
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(FLAGS.seed, example_batch['observations'], example_batch['actions'], config)

    if FLAGS.pretrained_actor_path is not None:
        agent = load_pretrained_actor(agent, FLAGS.pretrained_actor_path)

    if FLAGS.restore_agent_file is not None:
        agent = restore_agent_with_file(agent, FLAGS.restore_agent_file)

    params = {k: v for k, v in agent.network.params.items() if "target" not in k}
    print("param count:", sum(x.size for x in jax.tree_util.tree_leaves(params)))

    # --------------------------------------------------------------- logging
    prefixes = ["eval", "env"]
    if FLAGS.offline_steps > 0:
        prefixes.append("offline_agent")
    if FLAGS.online_steps > 0:
        prefixes.append("online_agent")
    csv_loggers = {prefix: CsvLogger(os.path.join(FLAGS.save_dir, f"{prefix}.csv")) for prefix in prefixes}

    # resume an interrupted run from its last checkpoint, if any
    last_save_path = None
    load_stage, load_step, replay_buffer, env_state = None, None, None, None
    success = False
    if os.path.isdir(FLAGS.save_dir):
        print("trying to load from", FLAGS.save_dir)
        if os.path.exists(os.path.join(FLAGS.save_dir, 'token.tk')):
            print("run already completed")
            return
        try:
            with open(os.path.join(FLAGS.save_dir, 'progress.tk'), 'r') as f:
                load_stage, load_step = f.read().split(",")
            load_step = int(load_step)
            agent = restore_agent(agent, restore_path=FLAGS.save_dir, restore_epoch=load_step)
            restore_csv_loggers(csv_loggers, FLAGS.save_dir)
            if load_stage == "online":
                replay_buffer, env_state = restore_buffer_env_state(restore_path=FLAGS.save_dir)
            success = True
        except Exception:
            load_stage, load_step, replay_buffer = None, None, None

    if not success:
        print("starting a fresh run")
        os.makedirs(FLAGS.save_dir, exist_ok=True)
        with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
            json.dump(get_flag_dict(), f)

    logger = LoggingHelper(csv_loggers=csv_loggers, wandb_logger=wandb)

    def run_eval(log_step, action_dim):
        eval_info, _, _ = evaluate(
            agent=agent, env=eval_env, action_dim=action_dim,
            num_eval_episodes=FLAGS.eval_episodes,
            num_video_episodes=FLAGS.video_episodes,
            video_frame_skip=FLAGS.video_frame_skip,
        )
        logger.log(eval_info, "eval", step=log_step)

    def next_dataset_file(cur_env):
        nonlocal dataset_idx, train_dataset, val_dataset
        dataset_idx = (dataset_idx + 1) % len(dataset_paths)
        print(f"Using new dataset: {dataset_paths[dataset_idx]}", flush=True)
        train_dataset, val_dataset = make_ogbench_env_and_datasets(
            FLAGS.env_name, dataset_path=dataset_paths[dataset_idx], compact_dataset=False,
            dataset_only=True, cur_env=cur_env,
        )
        train_dataset = process_train_dataset(train_dataset)

    action_dim = example_batch["actions"].shape[-1]

    # ------------------------------------------------------------ offline RL
    if load_stage == "online":
        print("skipping offline phase (resuming online)")
    else:
        start_step = load_step + 1 if load_stage == "offline" and load_step is not None else 1
        for i in tqdm.tqdm(range(start_step, FLAGS.offline_steps + 1)):
            log_step = i

            if FLAGS.ogbench_dataset_dir is not None and i % FLAGS.dataset_replace_interval == 0:
                next_dataset_file(env)

            batch = train_dataset.sample_sequence(
                config['batch_size'], sequence_length=FLAGS.horizon_length, discount=discount)
            agent, offline_info = agent.update(batch)

            if i % FLAGS.log_interval == 0:
                logger.log(offline_info, "offline_agent", step=log_step)

            if i == FLAGS.offline_steps or (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
                run_eval(log_step, action_dim)

            if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
                last_save_path = save_agent(agent, FLAGS.save_dir, log_step)
                save_csv_loggers(csv_loggers, FLAGS.save_dir)
                with open(os.path.join(FLAGS.save_dir, 'progress.tk'), 'w') as f:
                    f.write(f"offline,{i}")

    # ------------------------------------------------- offline -> online switch
    if FLAGS.kl_budget_online is not None:
        # config is a static field of the agent, so this triggers one recompilation
        config["kl_budget"] = FLAGS.kl_budget_online
        new_agent_config = dict(agent.config)
        new_agent_config["kl_budget"] = FLAGS.kl_budget_online
        object.__setattr__(agent, 'config', flax.core.FrozenDict(new_agent_config))
        print(f"Switching kl_budget to {FLAGS.kl_budget_online} for the online phase")

    if replay_buffer is None:
        replay_buffer = ReplayBuffer.create_from_initial_dataset(
            dict(train_dataset), size=train_dataset.size + FLAGS.online_steps
        )

    # ------------------------------------------------------------- online RL
    update_info = {}
    if load_stage == "online" and load_step is not None and env_state is not None:
        start_step = load_step + 1
        action_queue = list(np.reshape(env_state.pop("action_queue"), (-1, action_dim))) \
            if "action_queue" in env_state else []
        ob, _ = env.reset(options={"set_state": env_state})
        print(f"resuming from online step {start_step}")
    else:
        action_queue = []
        ob, _ = env.reset()
        start_step = 1

    for i in tqdm.tqdm(range(start_step, FLAGS.online_steps + 1)):
        log_step = FLAGS.offline_steps + i
        online_rng, key = jax.random.split(online_rng)

        if FLAGS.ogbench_dataset_dir is not None and i % FLAGS.dataset_replace_interval == 0:
            next_dataset_file(env)
            size = train_dataset.size
            for k in train_dataset:
                replay_buffer[k][:size] = train_dataset[k][:]

        # the action chunk is executed fully
        if len(action_queue) == 0:
            action = agent.sample_actions(observations=ob, rng=key)
            for a in np.array(action).reshape(-1, action_dim):
                action_queue.append(a)
        action = action_queue.pop(0)

        next_ob, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        env_info = {k: v for k, v in info.items() if k.startswith("distance")}
        logger.log(env_info, "env", step=log_step)

        replay_buffer.add_transition(dict(
            observations=ob,
            actions=action,
            rewards=reward,
            terminals=float(done),
            masks=1.0 - terminated,
            next_observations=next_ob,
        ))

        if done:
            ob, _ = env.reset()
            action_queue = []
        else:
            ob = next_ob

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample_sequence(
                config['batch_size'] * FLAGS.utd_ratio, sequence_length=FLAGS.horizon_length, discount=discount)
            batch = jax.tree.map(lambda x: x.reshape((FLAGS.utd_ratio, config["batch_size"]) + x.shape[1:]), batch)
            agent, update_info["online_agent"] = agent.batch_update(batch)

        if i % FLAGS.log_interval == 0:
            for key_, info_ in update_info.items():
                logger.log(info_, key_, step=log_step)
            update_info = {}

        if i == FLAGS.online_steps or (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
            run_eval(log_step, action_dim)

        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            last_save_path = save_agent(agent, FLAGS.save_dir, log_step)
            save_buffer_env_state(replay_buffer, env, action_queue, FLAGS.save_dir)
            save_csv_loggers(csv_loggers, FLAGS.save_dir)
            with open(os.path.join(FLAGS.save_dir, 'progress.tk'), 'w') as f:
                f.write(f"online,{i}")

    for csv_logger in logger.csv_loggers.values():
        csv_logger.close()

    with open(os.path.join(FLAGS.save_dir, 'token.tk'), 'w') as f:
        f.write(run.url or 'offline')

    # cleanup: drop intermediate checkpoints unless asked to keep them
    for name in os.listdir(FLAGS.save_dir):
        path = os.path.join(FLAGS.save_dir, name)
        if not os.path.isfile(path):
            continue
        if name.startswith("params") and (not FLAGS.save_last_checkpoint or path != last_save_path):
            os.remove(path)
        if name == "buffer.npz" and not FLAGS.save_replay_buffer:
            os.remove(path)

    wandb.finish()


if __name__ == '__main__':
    app.run(main)
