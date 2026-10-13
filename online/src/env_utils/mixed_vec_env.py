import itertools
import multiprocessing as mp
import warnings
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any, List, Optional, Type, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv


def _seq_worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
) -> None:
    """
    A sequential worker that runs a DummyVecEnv in a separated process.
    """
    parent_remote.close()
    envs = env_fn_wrapper.var()
    assert isinstance(envs, DummyVecEnv)
    
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observations, rewards, dones, infos = envs.step(data)
                # DummyVecEnv already performs auto-reset and updates `envs.reset_infos`
                remote.send((observations, rewards, dones, infos, envs.reset_infos))
            elif cmd == "reset":
                seeds, options = data
                envs._seeds = seeds
                envs._options = options
                observation = envs.reset()
                remote.send((observation, envs.reset_infos))
            elif cmd == "render":
                remote.send(envs.render(data))
            elif cmd == "get_images":
                remote.send(envs.get_images())
            elif cmd == "close":
                envs.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((envs.observation_space, envs.action_space))
            elif cmd == "env_method":
                method_name, method_args, indices, method_kwargs = data
                remote.send(envs.env_method(method_name, *method_args, indices=indices, **method_kwargs))
            elif cmd == "get_attr":
                attr_name, indices = data
                remote.send(envs.get_attr(attr_name, indices))
            elif cmd == "set_attr":
                attr_name, value, indices = data
                envs.set_attr(attr_name, value, indices)
                # Ensure we return a blocking signal matched dynamically to indices
                remote.send([None] * len(indices))
            elif cmd == "is_wrapped":
                wrapper_class, indices = data
                remote.send(envs.env_is_wrapped(wrapper_class, indices))
            elif cmd == "has_attr":
                attr_name = data
                remote.send(envs.has_attr(attr_name))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break
        except KeyboardInterrupt:
            break


def make_env_slice_dummy_vec_env(env_fns: Sequence[Callable[[], gym.Env]], start: int, end: int) -> Callable[[], DummyVecEnv]:
    """Make DummyVecEnv consisting of subset of environments based on slice indices."""
    def thunk() -> DummyVecEnv:
        return DummyVecEnv(list(env_fns[start:end]))
    return thunk


def _flatten_obs(obs_list: Sequence[VecEnvObs], space: spaces.Space) -> VecEnvObs:
    """Concatenate sequence of DummyVecEnv observations together into a single global batch."""
    if isinstance(space, spaces.Dict):
        assert isinstance(space.spaces, dict)
        return {key: np.concatenate([obs[key] for obs in obs_list], axis=0) for key in space.spaces.keys()}
    elif isinstance(space, spaces.Tuple):
        return tuple(np.concatenate([obs[i] for obs in obs_list], axis=0) for i in range(len(space.spaces)))
    else:
        return np.concatenate(obs_list, axis=0)


class MixedVecEnv(VecEnv):
    """
    Groups subsets of environments together sequentially via DummyVecEnv, then launches each
    subset concurrently into its own subprocess.

    This strikes a perfect balance between Multiprocessing capabilities and 
    reducing heavy inter-process-communication (IPC) bottlenecks for computationally demanding environments 
    such as humanoid-bench.

    :param env_fns: Environments to run in subprocesses
    :param num_subproc: Number of subprocesses to use (n_envs MUST be divisible by num_subproc)
    :param start_method: method used to start the subprocesses. Defaults to 'forkserver' or 'spawn'.
    """

    def __init__(
        self,
        env_fns: Sequence[Callable[[], gym.Env]],
        num_subproc: int,
        start_method: Optional[str] = None,
    ):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)
        
        assert n_envs % num_subproc == 0, f"Number of environments ({n_envs}) must be divisible by the number of subprocesses ({num_subproc})."
        
        self.num_subproc = num_subproc
        self.num_dummy_per_subproc = n_envs // num_subproc

        if start_method is None:
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        dummy_env_fns =[
            make_env_slice_dummy_vec_env(
                env_fns,
                self.num_dummy_per_subproc * i,
                self.num_dummy_per_subproc * (i + 1),
            )
            for i in range(num_subproc)
        ]

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(num_subproc)])
        self.processes =[]
        
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, dummy_env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_seq_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()

        super().__init__(n_envs, observation_space, action_space)

    def step_async(self, actions: Union[np.ndarray, List[Any]]) -> None:
        """Tell all the environments to start taking a step with the given actions."""
        # We manually chunk sequence arrays. This safely covers complex spaces (unlike naive reshaping)
        for i, remote in enumerate(self.remotes):
            start = i * self.num_dummy_per_subproc
            end = (i + 1) * self.num_dummy_per_subproc
            action_chunk = actions[start:end]
            remote.send(("step", action_chunk))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        """Wait for the step taken with step_async()."""
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos, reset_infos = zip(*results)
        
        self.reset_infos = list(itertools.chain.from_iterable(reset_infos))
        
        obs_stacked = _flatten_obs(obs, self.observation_space)
        rew_stacked = np.concatenate(rews, axis=0)
        dones_stacked = np.concatenate(dones, axis=0)
        infos_stacked = list(itertools.chain.from_iterable(infos))
        
        return obs_stacked, rew_stacked, dones_stacked, infos_stacked

    def reset(self) -> VecEnvObs:
        """Reset all environments and extract the array observations correctly mapped to Gym 0.29 standards."""
        for env_idx, remote in enumerate(self.remotes):
            start = env_idx * self.num_dummy_per_subproc
            end = (env_idx + 1) * self.num_dummy_per_subproc
            seeds = self._seeds[start:end]
            options = self._options[start:end]
            remote.send(("reset", (seeds, options)))
            
        results = [remote.recv() for remote in self.remotes]
        obs, reset_infos = zip(*results)
        
        self.reset_infos = list(itertools.chain.from_iterable(reset_infos))
        
        # SB3 specification mandates we discard seeds/options once used
        self._reset_seeds()
        self._reset_options()
        
        return _flatten_obs(obs, self.observation_space)

    def close(self) -> None:
        """Clean up the environment's resources."""
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        """Return RGB images from each environment when available."""
        if self.render_mode != "rgb_array":
            warnings.warn(
                f"The render mode is {self.render_mode}, but this method assumes it is `rgb_array` to obtain images."
            )
            return[None for _ in range(self.num_envs)]
        
        for pipe in self.remotes:
            pipe.send(("get_images", None))
        outputs = [pipe.recv() for pipe in self.remotes]
        return list(itertools.chain.from_iterable(outputs))

    def _do_batched_calls(
        self, command: str, args_fn: Callable[[List[int]], Any], indices: VecEnvIndices
    ) -> List[Any]:
        """
        Map global indices to the correct local environments grouped batched-processes, 
        and reconstruct the results asynchronously for performance.
        """
        indices_list = list(self._get_indices(indices))
        
        # Map out to relevant subprocesses
        indices_per_subproc = defaultdict(list)
        for i, idx in enumerate(indices_list):
            subproc_idx, dummy_idx = divmod(idx, self.num_dummy_per_subproc)
            indices_per_subproc[subproc_idx].append((i, dummy_idx))
            
        # Dispatch batched requests over IPC Pipes
        for subproc_idx, items in indices_per_subproc.items():
            dummy_indices =[dummy_idx for _, dummy_idx in items]
            remote = self.remotes[subproc_idx]
            remote.send((command, args_fn(dummy_indices)))
            
        # Wait back for them and reorder cleanly
        results = [None] * len(indices_list)
        for subproc_idx, items in indices_per_subproc.items():
            remote = self.remotes[subproc_idx]
            res = remote.recv()
            for (i, _), r in zip(items, res):
                results[i] = r
                
        return results

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        return self._do_batched_calls("get_attr", lambda d_idx: (attr_name, d_idx), indices)

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        self._do_batched_calls("set_attr", lambda d_idx: (attr_name, value, d_idx), indices)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        return self._do_batched_calls(
            "env_method", 
            lambda d_idx: (method_name, method_args, d_idx, method_kwargs), 
            indices
        )

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        return self._do_batched_calls("is_wrapped", lambda d_idx: (wrapper_class, d_idx), indices)

    def has_attr(self, attr_name: str) -> bool:
        for remote in self.remotes:
            remote.send(("has_attr", attr_name))
        return all(remote.recv() for remote in self.remotes)