from functools import partial
from typing import Any, Optional, Tuple, Union

import chex
import gymnasium as gym
import gymnax
import jax
import jax.numpy as jnp
import mani_skill.envs
import numpy as np
import torch
from brax import envs
from brax.envs.wrappers.training import AutoResetWrapper, EpisodeWrapper
from flax import struct
from gymnasium.wrappers import TimeLimit
from gymnax.environments import environment, spaces
from gymnax.environments.environment import Environment
from gymnax.environments.spaces import Box
from jax.dlpack import from_dlpack
from jax.experimental import io_callback
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from ml_collections import ConfigDict
from mujoco_playground import MjxEnv, registry
from mujoco_playground._src.wrapper import Wrapper, wrap_for_brax_training
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch.utils.dlpack import to_dlpack

from src.env_utils.mixed_vec_env import MixedVecEnv


class MjxGymnaxWrapper(Environment):
    def __init__(
        self,
        env_or_name: str | MjxEnv,
        episode_length: int = 1000,
        action_repeat: int = 1,
        reward_scale: float = 1.0,
        push_distractions: bool = False,
        config: dict = None,
        asymmetric_observation: bool = False,
    ):
        if isinstance(env_or_name, str):
            if config is None:
                config = registry.get_default_config(env_or_name)
                is_humanoid_task = env_or_name in [
                    "G1JoystickRoughTerrain",
                    "G1JoystickFlatTerrain",
                    "T1JoystickRoughTerrain",
                    "T1JoystickFlatTerrain",
                ]
                if is_humanoid_task:
                    config.push_config.enable = push_distractions
            else:
                config = ConfigDict(config)
            env = registry.load(env_or_name, config=config)
            if episode_length is not None:
                env = wrap_for_brax_training(
                    env, episode_length=episode_length, action_repeat=action_repeat
                )
            self.env = env
        else:
            self.env = env_or_name
        self.reward_scale = reward_scale
        if isinstance(self.env.observation_size, int):
            self.dict_obs = False
        else:
            self.dict_obs = True
        if asymmetric_observation:
            self.dict_obs_key = "privileged_state"
        else:
            self.dict_obs_key = "state"
        print(self.dict_obs_key)
        super().__init__()

    def action_space(self, params):
        return gymnax.environments.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.env.action_size,),
        )

    def observation_space(self, params):
        if self.dict_obs:
            return Box(
                low=-float("inf"),
                high=float("inf"),
                shape=self.env.observation_size["state"],
            ), Box(
                low=-float("inf"),
                high=float("inf"),
                shape=self.env.observation_size[self.dict_obs_key],
            )
        else:
            return Box(
                low=-float("inf"),
                high=float("inf"),
                shape=(self.env.observation_size,),
            ), Box(
                low=-float("inf"),
                high=float("inf"),
                shape=(self.env.observation_size,),
            )

    @property
    def default_params(self) -> gymnax.EnvParams:
        return gymnax.EnvParams()

    def reset(self, key):
        state = self.env.reset(key)
        # state.info["truncation"] = 0.0
        obs = state.obs if not self.dict_obs else state.obs["state"]
        critic_obs = state.obs if not self.dict_obs else state.obs[self.dict_obs_key]
        return obs, critic_obs, state

    def step(self, key, state, action):
        # action = jnp.nan_to_num(action, 0.0)
        state = self.env.step(state, action)
        obs = state.obs if not self.dict_obs else state.obs["state"]
        critic_obs = state.obs if not self.dict_obs else state.obs[self.dict_obs_key]
        return (
            obs,
            critic_obs,
            state,
            state.reward * self.reward_scale,
            state.done > 0.5,
            {},
        )


@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: jnp.ndarray
    episode_lengths: jnp.ndarray
    episode_successes: jnp.ndarray
    returned_episode_returns: jnp.ndarray
    returned_episode_lengths: jnp.ndarray
    returned_episode_successes: jnp.ndarray
    timestep: jnp.ndarray
    truncated: jnp.ndarray
    info: Any = None

    def unwrapped(self):
        return self.env_state

    def set_env_state(self, env_state):
        if hasattr(self.env_state, "set_env_state"):
            return self.replace(env_state=self.env_state.set_env_state(env_state))
        return self.replace(env_state=env_state)


class LogWrapper(Wrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment, num_envs: int):
        super().__init__(env)
        self.num_envs = num_envs

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key) -> Tuple[chex.Array, environment.EnvState]:
        obs, critic_obs, env_state = self.env.reset(key)
        state = LogEnvState(
            env_state=env_state,
            episode_returns=jnp.zeros((self.num_envs,)),
            episode_lengths=jnp.zeros((self.num_envs,), dtype=jnp.int32),
            episode_successes=jnp.zeros((self.num_envs,), dtype=jnp.bool_),
            returned_episode_returns=jnp.zeros((self.num_envs,)),
            returned_episode_lengths=jnp.zeros((self.num_envs,), dtype=jnp.int32),
            returned_episode_successes=jnp.zeros((self.num_envs,), dtype=jnp.bool_),
            timestep=jnp.zeros((self.num_envs,), dtype=jnp.int32),
            truncated=jnp.ones((self.num_envs,), dtype=jnp.float32),
            info={
                "returned_episode": jnp.zeros((self.num_envs,), dtype=jnp.bool_),
                "returned_episode_returns": jnp.zeros((self.num_envs,)),
                "returned_episode_successes": jnp.zeros((self.num_envs,), dtype=jnp.bool_),
                "timestep": jnp.zeros((self.num_envs,), dtype=jnp.int32),
                "returned_episode_lengths": jnp.zeros(
                    (self.num_envs,), dtype=jnp.int32
                ),
            },
        )
        return obs, critic_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, critic_obs, env_state, reward, done, info = self.env.step(
            key, state.env_state, action
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        new_episode_success = state.episode_successes | env_state.info.get("success", jnp.zeros_like(state.episode_successes))
        info["returned_episode_returns"] = (
            state.returned_episode_returns * (1 - done) + new_episode_return * done
        )
        info["returned_episode_lengths"] = (
            state.returned_episode_lengths * (1 - done) + new_episode_length * done
        )
        info["returned_episode_successes"] = (
            state.returned_episode_successes & (~done) | new_episode_success & done
        )
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            episode_successes=new_episode_success & (~done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            returned_episode_successes=state.returned_episode_successes & (~done)
            | new_episode_success & done,
            timestep=state.timestep + 1,
            truncated=env_state.info["truncation"],
            info=info,
        )
        return obs, critic_obs, state, reward, done, info


class BraxGymnaxWrapper:
    def __init__(
        self,
        env_name,
        backend="generalized",
        episode_length=1000,
        reward_scaling=1.0,
        terminate=True,
    ):
        env = envs.get_environment(
            env_name=env_name, backend=backend, terminate_when_unhealthy=terminate
        )
        env = EpisodeWrapper(env, episode_length=episode_length, action_repeat=1)
        env = AutoResetWrapper(env)
        self.env = env
        self.action_size = self.env.action_size
        self.observation_size = (self.env.observation_size,)
        self.default_params = ()
        self.reward_scaling = reward_scaling

    def reset(self, key):
        state = self.env.reset(key)
        return state.obs, state

    def step(self, key, state, action):
        next_state = self.env.step(state, action)
        return (
            next_state.obs,
            next_state.obs,
            next_state,
            next_state.reward * self.reward_scaling,
            next_state.done > 0.5,
            {},
        )

    def observation_space(self):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self.env.observation_size,),
        ), spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self.env.observation_size,),
        )

    def action_space(self):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.env.action_size,),
        )


class ClipAction(Wrapper):
    def __init__(self, env, low=-0.999, high=0.999):
        super().__init__(env)
        self.low = low
        self.high = high

    def step(self, key, state, action):
        """TODO: In theory the below line should be the way to do this."""
        # action = jnp.clip(action, self.env.action_space.low, self.env.action_space.high)
        action = jnp.clip(action, self.low, self.high)
        return self.env.step(key, state, action)


@struct.dataclass
class NormalizeVecObsEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    critic_mean: jnp.ndarray
    critic_var: jnp.ndarray
    count: float
    env_state: environment.EnvState
    truncated: float
    info: Any = None

    def unwrapped(self):
        return self.env_state.unwrapped()

    def set_env_state(self, env_state):
        return self.replace(env_state=self.env_state.set_env_state(env_state))


class NormalizeVec(Wrapper):
    def __init__(self, env, enable=True):
        super().__init__(env)
        self.enable = enable

    def _init_state(self, key):
        obs, critic_obs, env_state = self.env.reset(key)
        return NormalizeVecObsEnvState(
            mean=jnp.mean(obs, axis=0),
            var=jnp.var(obs, axis=0),
            critic_mean=jnp.mean(critic_obs, axis=0),
            critic_var=jnp.var(critic_obs, axis=0),
            count=obs.shape[0],
            env_state=env_state,
        )

    def _compute_stats(self, mean, var, count, obs):
        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count

        return new_mean, new_var

    def reset(self, key, params=None):
        obs, critic_obs, env_state = self.env.reset(key)
        if params is not None:
            mean = params.mean
            var = params.var
            critic_mean = params.critic_mean
            critic_var = params.critic_var
            count = params.count
        else:
            mean = jnp.mean(obs, axis=0)
            var = jnp.var(obs, axis=0)
            critic_mean = jnp.mean(critic_obs, axis=0)
            critic_var = jnp.var(critic_obs, axis=0)
            count = obs.shape[0]
        state = NormalizeVecObsEnvState(
            mean=mean,
            var=var,
            critic_mean=critic_mean,
            critic_var=critic_var,
            count=count,
            env_state=env_state,
            truncated=env_state.truncated,
            info=env_state.info,
        )
        return (
            (
                (obs - state.mean) / jnp.sqrt(state.var + 1e-2),
                (critic_obs - state.critic_mean) / jnp.sqrt(state.critic_var + 1e-2),
                state,
            )
            if self.enable
            else (obs, critic_obs, state)
        )

    def step(self, key, state, action, update_stats=True):
        obs, critic_obs, env_state, reward, done, info = self.env.step(key, state.env_state, action)

        if update_stats:
            new_mean, new_var = self._compute_stats(state.mean, state.var, state.count, obs)
            new_critic_mean, new_critic_var = self._compute_stats(
                state.critic_mean, state.critic_var, state.count, critic_obs
            )

            new_count = state.count + obs.shape[0]
        else:
            new_mean, new_var = state.mean, state.var
            new_critic_mean, new_critic_var = state.critic_mean, state.critic_var
            new_count = state.count

        state = NormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            critic_mean=new_critic_mean,
            critic_var=new_critic_var,
            count=new_count,
            env_state=env_state,
            truncated=env_state.truncated,
            info=env_state.info,
        )
        return (
            (
                (obs - state.mean) / jnp.sqrt(state.var + 1e-2),
                (critic_obs - state.critic_mean) / jnp.sqrt(state.critic_var + 1e-2),
                state,
                reward,
                done,
                info,
            )
            if self.enable
            else (obs, critic_obs, state, reward, done, info)
        )


def to_jax(tensor):
    if isinstance(tensor, dict):
        return {k: to_jax(v) for k, v in tensor.items()}
    if not isinstance(tensor, torch.Tensor):
        return tensor
    # Ensure tensor is on the correct device and contiguous
    return from_dlpack(to_dlpack(tensor.contiguous()))


def to_torch(array):
    if isinstance(array, dict):
        return {k: to_torch(v) for k, v in array.items()}
    if not isinstance(array, (jax.Array, jnp.ndarray)):
        return array
    return torch.from_dlpack(jax.dlpack.to_dlpack(array))


@struct.dataclass
class ManiSkillEnvState:
    obs: Union[jnp.ndarray, dict]
    info: dict
    done: jnp.ndarray
    truncated: jnp.ndarray
    _wrapper: Any = struct.field(pytree_node=False)  # Reference to sync with engine

    def unwrapped(self):
        return self

    def set_env_state(self, new_state):
        """
        Sync manually set info['steps'] back to ManiSkill's internal PyTorch counters.
        """
        if "steps" in new_state.info:
            self._wrapper._sync_steps_to_torch(new_state.info["steps"])
        return new_state


class ManiSkillGymnaxWrapper:
    def __init__(
        self,
        env_name: str,
        num_envs: int,
        reconfiguration_freq: Optional[int] = None,
        partial_reset: bool = True,
        asymmetric_observation: bool = False,
        env_kwargs: dict = None,
    ):
        self.env_name = env_name
        self.num_envs = num_envs
        self.partial_reset = partial_reset
        self.asymmetric_observation = asymmetric_observation

        raw_env = gym.make(
            env_name,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **(env_kwargs or {}),
        )

        if isinstance(raw_env.action_space, gym.spaces.Dict):
            raw_env = FlattenActionSpaceWrapper(raw_env)

        self.env = ManiSkillVectorEnv(
            raw_env,
            num_envs,
            ignore_terminations=True,
            record_metrics=True,
        )

        self.max_episode_steps = gym_utils.find_max_episode_steps_value(self.env)
        self.action_size = self.env.action_space.shape[1]

        # Handle observation shapes (supporting Dict obs common in ManiSkill)
        if isinstance(self.env.observation_space, gym.spaces.Dict):
            self.obs_shape = self.env.observation_space["state"].shape[1:]
        else:
            self.obs_shape = self.env.observation_space.shape[1:]

    def action_space(self, params=None):
        return Box(low=-1.0, high=1.0, shape=(self.action_size,))

    def observation_space(self, params=None):
        return Box(low=-float("inf"), high=float("inf"), shape=self.obs_shape), Box(
            low=-float("inf"), high=float("inf"), shape=self.obs_shape
        )

    def _sync_steps_to_torch(self, jax_steps):
        """Helper to update the physical engine's clock from JAX."""

        def _do_sync(steps):
            torch_steps = to_torch(steps).to(self.env.device).to(torch.long)
            self.env.unwrapped.elapsed_steps[: self.num_envs] = torch_steps
            return steps  # io_callback needs a return

        io_callback(_do_sync, jax.ShapeDtypeStruct(jax_steps.shape, jax_steps.dtype), jax_steps)

    def _python_reset(self):
        """This function runs in pure Python, outside of JAX JIT."""
        # ManiSkill manages its own RNG via env_kwargs seed, but we can re-seed here if needed
        obs, info = self.env.reset()
        info = {"steps": info["elapsed_steps"].to(torch.float32), "success": info["success"].to(torch.bool)}
        return to_jax(obs), to_jax(info)

    def reset(self, key, params=None) -> Tuple[jnp.ndarray, jnp.ndarray, ManiSkillEnvState]:
        # TODO: asymmetric observation and dict obs
        # Define the expected return types/shapes for JAX
        obs_spec = jax.ShapeDtypeStruct((self.num_envs, *self.obs_shape), jnp.float32)
        info_spec = {"steps": jax.ShapeDtypeStruct((self.num_envs,), jnp.float32), "success": jax.ShapeDtypeStruct((self.num_envs,), jnp.bool_)}

        # Call the Python side
        obs, info = io_callback(self._python_reset, (obs_spec, info_spec))

        # Handle asymmetric observations if requested
        # In ManiSkill, 'state' is usually the proprioception,
        # 'privileged_state' contains the full simulator info.
        if isinstance(obs, dict):
            state_obs = obs["state"]
            critic_obs = obs.get("privileged_state", state_obs) if self.asymmetric_observation else state_obs
        else:
            state_obs = obs
            critic_obs = obs

        # info["steps"] = info["elapsed_steps"].astype(jnp.float32)
        info["truncation"] = jnp.zeros(self.num_envs)

        state = ManiSkillEnvState(
            obs=obs,
            info=info,
            done=jnp.zeros(self.num_envs, dtype=jnp.bool_),
            truncated=jnp.zeros(self.num_envs, dtype=jnp.float32),
            _wrapper=self,
        )
        return state_obs, critic_obs, state

    def _python_step(self, action_jax):
        """Step env via torch outside of JAX JIT."""
        action_torch = to_torch(action_jax)
        obs, reward, terminated, truncated, info = self.env.step(action_torch)
        # ManiSkill provides elapsed_steps in the info dict but we need it as "steps"
        info = {"steps": info["elapsed_steps"].to(torch.float32), "success": info["success"].to(torch.bool)}

        # Convert everything to JAX
        obs_j = to_jax(obs)
        reward_j = to_jax(reward).flatten()
        term_j = to_jax(terminated).flatten()
        trunc_j = to_jax(truncated).flatten()

        return obs_j, reward_j, term_j, trunc_j, to_jax(info)

    def step(self, key, state: ManiSkillEnvState, action: jnp.ndarray):
        obs_spec = jax.ShapeDtypeStruct((self.num_envs, *self.obs_shape), jnp.float32)
        rew_spec = jax.ShapeDtypeStruct((self.num_envs,), jnp.float32)
        done_spec = jax.ShapeDtypeStruct((self.num_envs,), jnp.bool_)
        trunc_spec = jax.ShapeDtypeStruct((self.num_envs,), jnp.bool_)
        info_spec = {"steps": jax.ShapeDtypeStruct((self.num_envs,), jnp.float32), "success": jax.ShapeDtypeStruct((self.num_envs,), jnp.bool_)}

        # Execute the step outside of JIT
        obs, reward, terminated, truncated, info = io_callback(
            self._python_step, (obs_spec, rew_spec, done_spec, trunc_spec, info_spec), action
        )

        if self.partial_reset:
            # maniskill continues bootstrap on terminated, which playground does on truncated.
            # This unifies the interfaces in a very hacky way
            # in other words: terminal is also just a truncation with bootstrap in maniskill
            actual_truncated = jnp.logical_or(terminated, truncated)
            done = actual_truncated
        else:
            done = jnp.logical_or(terminated, truncated)
            actual_truncated = jnp.zeros_like(done, dtype=jnp.bool_)

        if isinstance(obs, dict):
            state_obs = obs["state"]
            critic_obs = obs.get("privileged_state", state_obs) if self.asymmetric_observation else state_obs
        else:
            state_obs = obs
            critic_obs = obs

        info["truncation"] = actual_truncated.astype(jnp.float32)

        new_state = ManiSkillEnvState(
            obs=state_obs, info=info, done=done, truncated=actual_truncated.astype(jnp.float32), _wrapper=self
        )

        return state_obs, critic_obs, new_state, reward, done, {}


@struct.dataclass
class HumanoidBenchEnvState:
    obs: jnp.ndarray
    info: dict
    done: jnp.ndarray
    truncated: jnp.ndarray
    _wrapper: Any = struct.field(pytree_node=False)  # Reference to sync if needed

    def unwrapped(self):
        return self

    def set_env_state(self, new_state):
        return new_state


def _hb_make_env(env_name, rank, render_mode=None, seed=0):
    """
    Utility function for multiprocessed env.

    :param rank: (int) index of the subprocess
    :param seed: (int) the inital seed for RNG
    """

    if env_name in[
        "h1hand-push-v0",
        "h1-push-v0",
        "h1hand-cube-v0",
        "h1cube-v0",
        "h1hand-basketball-v0",
        "h1-basketball-v0",
        "h1hand-kitchen-v0",
        "h1-kitchen-v0",
    ]:
        max_episode_steps = 500
    else:
        max_episode_steps = 1000

    def _init():
        import humanoid_bench

        env = gym.make(env_name, render_mode=render_mode)
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env.unwrapped.seed(seed + rank)

        return env

    return _init


class HumanoidBenchGymnaxWrapper:
    """JAX wrapper for HumanoidBench following the Gymnax/MuJoCo Playground pattern."""

    def __init__(
        self,
        env_name: str,
        num_envs: int,
        render_mode: str = None,
        seed: int = 0,
        num_subproc: int = 1,
    ):
        self.env_name = env_name
        self.num_envs = num_envs
        self.asymmetric_observation = False
        self.seed_val = seed

        self.envs = MixedVecEnv(
            [_hb_make_env(env_name, i, render_mode=render_mode, seed=seed) for i in range(num_envs)],
            num_subproc=num_subproc,
        )

        if env_name in[
            "h1hand-push-v0",
            "h1-push-v0",
            "h1hand-cube-v0",
            "h1cube-v0",
            "h1hand-basketball-v0",
            "h1-basketball-v0",
            "h1hand-kitchen-v0",
            "h1-kitchen-v0",
        ]:
            self.max_episode_steps = 500
        else:
            self.max_episode_steps = 1000

        # Extract spaces directly from the SB3 VecEnv
        self.action_size = self.envs.action_space.shape[-1]
        self.obs_shape = self.envs.observation_space.shape

    def action_space(self, params=None):
        return Box(low=-1.0, high=1.0, shape=(self.action_size,))

    def observation_space(self, params=None):
        return Box(low=-float("inf"), high=float("inf"), shape=self.obs_shape), Box(
            low=-float("inf"), high=float("inf"), shape=self.obs_shape
        )

    def _python_reset(self):
        """This function runs in pure Python, outside of JAX JIT."""
        obs = self.envs.reset()
        info = {
            "steps": np.zeros(self.num_envs, dtype=np.float32),
            "real_next_obs": obs.copy()
        }
        return obs, info

    def reset(self, key, params=None) -> Tuple[jnp.ndarray, jnp.ndarray, HumanoidBenchEnvState]:
        # Define the expected return types/shapes for JAX
        obs_spec = jax.ShapeDtypeStruct((self.num_envs, *self.obs_shape), jnp.float32)
        info_spec = {
            "steps": jax.ShapeDtypeStruct((self.num_envs,), jnp.float32),
            "real_next_obs": jax.ShapeDtypeStruct((self.num_envs, *self.obs_shape), jnp.float32)
        }

        # Call the Python side safely across JIT boundaries via io_callback
        obs, info = io_callback(self._python_reset, (obs_spec, info_spec))

        info["truncation"] = jnp.zeros(self.num_envs, dtype=jnp.float32)

        state = HumanoidBenchEnvState(
            obs=obs,
            info=info,
            done=jnp.zeros(self.num_envs, dtype=jnp.bool_),
            truncated=jnp.zeros(self.num_envs, dtype=jnp.float32),
            _wrapper=self,
        )
        # Note: HumanoidBench does not natively differentiate state & privileged dict like MJX
        # so we pass standard `obs` for both `state_obs` and `critic_obs`.
        return obs, obs, state

    def _python_step(self, action_jax):
        """Step env via SB3 VecEnv pure Python process outside of JAX JIT."""
        action_np = np.asarray(action_jax)
        observations, rewards, dones, raw_infos = self.envs.step(action_np)

        truncateds = np.zeros(self.num_envs, dtype=bool)
        real_next_obs = observations.copy()

        for i in range(self.num_envs):
            # SB3 automatically resets completed envs and returns the first observation
            # of the new episode. We extract raw terminal observations to pass back correctly.
            if dones[i] and "terminal_observation" in raw_infos[i]:
                real_next_obs[i] = raw_infos[i]["terminal_observation"]

            if raw_infos[i].get("TimeLimit.truncated", False):
                truncateds[i] = True

        info_out = {
            "steps": np.zeros(self.num_envs, dtype=np.float32),
            "real_next_obs": real_next_obs.astype(np.float32)
        }

        return (
            observations.astype(np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=bool),
            truncateds,
            info_out
        )

    def step(self, key, state: HumanoidBenchEnvState, action: jnp.ndarray):
        obs_spec = jax.ShapeDtypeStruct((self.num_envs, *self.obs_shape), jnp.float32)
        rew_spec = jax.ShapeDtypeStruct((self.num_envs,), jnp.float32)
        done_spec = jax.ShapeDtypeStruct((self.num_envs,), jnp.bool_)
        trunc_spec = jax.ShapeDtypeStruct((self.num_envs,), jnp.bool_)
        info_spec = {
            "steps": jax.ShapeDtypeStruct((self.num_envs,), jnp.float32),
            "real_next_obs": jax.ShapeDtypeStruct((self.num_envs, *self.obs_shape), jnp.float32)
        }

        # Execute the step outside of JIT
        obs, reward, done, truncated, info = io_callback(
            self._python_step, (obs_spec, rew_spec, done_spec, trunc_spec, info_spec), action
        )

        info["truncation"] = truncated.astype(jnp.float32)

        new_state = HumanoidBenchEnvState(
            obs=obs,
            info=info,
            done=done,
            truncated=truncated.astype(jnp.float32),
            _wrapper=self
        )

        return obs, obs, new_state, reward, done, {}
