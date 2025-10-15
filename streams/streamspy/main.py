#!/usr/bin/env python3
# Run STREAmS using the gymnasium environment wrapper

import json # collects values from input.json fields, converts them to a nested Python dictionary, which is then converted into a Config object (e.g. config.temporal.num_iter)
import h5py
import os
import runpy
from pathlib import Path
import numpy as np
from StreamsEnvironment import StreamsGymEnv
#from .StreamsEnvironment import StreamsGymEnv

env = StreamsGymEnv()

if env.config.jet.jet_method_name == "OpenLoop" or env.config.jet.jet_method_name == "None":

    rank = env.rank

    if rank == 0:
        if env.config.jet.jet_method_name == "OpenLoop":
            print(f'OPENLOOP')
        else:
            print(f'NO ACTUATION')
    # Instantiate environment and prep h5 files for standard datasets
    # env = StreamsGymEnv()
    save_dir = Path("/distribute_save")
    env.init_h5_io(save_dir)

    # Main solver loop
    obs = env.reset()
    for _ in range(env.max_episode_steps):
        # ``StreamsGymEnv`` internally handles any actuation logic depending on the
        # chosen jet type.  The action value is ignored for non-adaptive jets.
        obs, reward, done, info = env.step(np.array([0.0], dtype=np.float32))
        env.log_step_h5(info["jet_amplitude"])

    # Finalize solver and MPI
    env.close_h5_io()
    env.close()
    exit()
    
elif env.config.jet.jet_method_name == "Classical":
    # General imports
    import argparse # Used for attribute access, defining default values and data-type, and providing ready made help calls
    import json
    import signal
    from pathlib import Path
    from typing import Tuple
    import torch
    from mpi4py import rc
    rc.initialize = False
    rc.finalize = False
    from mpi4py import MPI
    from collections import deque

    # Script imports
    import importlib
    from base_Classical import BaseController
    import io_utils
    #from .base_Classical import BaseController
    #from . import io_utils

    STOP = False

    comm = env.comm
    rank = env.rank

    if rank == 0:
        print(f'CLASSICAL')

    save_dir = Path("/distribute_save")
    env.init_h5_io(save_dir)

    # Dynamically load the requested classical controller
    strategy = env.config.jet.jet_strategy_name
    module_path = f"Control.{strategy}"
    controller_module = importlib.import_module(module_path)
    controller_class = getattr(controller_module, "controller")

    if env.config.jet.jet_params.get("organized_motion") != "undefined":
        controller = controller_class(env)
        # controller.recompute_obs()

    obs = env.reset()
    if rank == 0 and hasattr(controller, "reset"):
        controller.reset()

    done = False
    step = 0
    while not done and step < env.max_episode_steps:
        if rank == 0:
            action = controller.compute_action(obs)
        else:
            action = None
        action = comm.bcast(action, root=0)
        obs, reward, done, info = env.step(np.array([action], dtype=np.float32))
        env.log_step_h5(info["jet_amplitude"])
        done = comm.bcast(done, root=0)
        step += 1
        
    # Finalize solver and MPI
    env.close_h5_io()
    env.close()
    exit()

elif env.config.jet.jet_method_name == "LearningBased":
    # General imports
    import argparse # Used for attribute access, defining default values and data-type, and providing ready made help calls
    import json
    import logging
    import shutil
    import signal
    from pathlib import Path
    from typing import Tuple
    import torch
    # from tqdm import trange (progress bar, could be a nice touch eventually)
    from mpi4py import rc
    rc.initialize = False
    rc.finalize = False
    from mpi4py import MPI
    from collections import deque

    # Script imports
    import importlib
    from base_LearningBased import BaseAgent
    import io_utils
    #from .base_LearningBased import BaseAgent
    #from . import io_utils

    STOP = False

    def train(env: StreamsGymEnv, agent) -> Path:
        """Train agent and return path to best checkpoint."""
        comm = MPI.COMM_WORLD
        rank = comm.rank
        if rank == 0:
            print(f'TRAINING')
        
        # print("[rl_control.py] Define reward objects")
        best_reward = -float("inf")
        episode_rewards = []
        
        best_path = Path(env.config.jet.jet_params["checkpoint_dir"]) / "best"
        
        # open output file for time, amp, reward, and obs in the training loop to be collected
        write_training = env.config.jet.jet_params["training_output"] is not None
        h5train = time_dset = amp_dset = reward_dset = obs_dset = None
        if write_training:
            if rank == 0:
                # Define path, create directory and h5, gather data to allocate shape
                training_output_path = Path(env.config.jet.jet_params["training_output"])
                training_output_path.parent.mkdir(parents=True, exist_ok=True)
                # h5train = io_utils.IoFile(str(training_output_path))
                h5train = io_utils.IoFile(str(training_output_path), comm=None)
                training_episodes = env.config.jet.jet_params["train_episodes"]
                training_steps = env.max_episode_steps
                observation_dim = env.observation_space.shape[0]
                # Create datasets within the h5 file
                time_dset = h5train.file.create_dataset("time", shape=(training_episodes, training_steps), dtype="f4")
                amp_dset = h5train.file.create_dataset("amplitude", shape=(training_episodes, training_steps), dtype="f4")
                reward_dset = h5train.file.create_dataset("reward", shape=(training_episodes, training_steps), dtype="f4")
                # If max simulation steps are greater than 100, chunk the data
                if env.config.temporal.num_iter > 100:
                    obs_dset = h5train.file.create_dataset("observation", shape=(training_episodes, training_steps, observation_dim), dtype="f4", chunks=(1, 100, observation_dim))
                else:
                    obs_dset = h5train.file.create_dataset("observation", shape=(training_episodes, training_steps, observation_dim), dtype="f4")
            
            else:
                training_output_path = None
            comm.Barrier()
                
        for ep in range(env.config.jet.jet_params["train_episodes"]): #, disable=rank != 0):
            if rank == 0:
                print(f'Beginning of training episode {ep + 1}')
            if STOP:
                break
            obs = env.reset()
            done = False
            ep_reward = 0.0
            step = 0
            sa_queue = deque()
            def _to_numpy_copy(array_like):
                if isinstance(array_like, torch.Tensor):
                    return array_like.detach().cpu().numpy().copy()
                return np.array(array_like, copy=True)
                
            while not done:
                if rank == 0:
                    obs_array = _to_numpy_copy(obs)
                    obs_tensor = torch.tensor(obs_array, dtype=torch.float32)
                    action_t = agent.choose_action(obs_tensor, step)
                    action_t, obs_t, obs_t_next, convection_complete = env.delay_action(action_t, obs_tensor) # Delay action if desired
                else:
                    action_t = None
                action_t = comm.bcast(action_t, root=0)
                next_obs, reward, done, info = env.step(action_t)
                done = comm.bcast(done, root=0)
                if rank == 0:
                    next_obs_array = _to_numpy_copy(next_obs)
                    if convection_complete:
                        sa_queue.append((_to_numpy_copy(obs_t), action_t, _to_numpy_copy(obs_t_next), info["time"]))
                    else:
                        sa_queue.append((obs_array, action_t, next_obs_array, info["time"]))
                if len(sa_queue) > env.lag_steps:
                    old_obs, old_action, old_next, old_time = sa_queue.popleft()
                    old_obs_np = _to_numpy_copy(old_obs)
                    old_next_np = _to_numpy_copy(old_next)
                    ep_reward += reward
                    agent.learn(old_obs_np, old_action, reward, old_next_np)
                    if write_training:
                        idx = step - env.lag_steps
                        time_dset[ep, idx] = old_time
                        amp_dset[ep, idx] = old_action
                        reward_dset[ep, idx] = reward
                        obs_dset[ep, idx, :] = old_obs_np
                step += 1
                obs = next_obs
            if rank == 0:
                episode_rewards.append(ep_reward)
                if ep_reward > best_reward: # 
                    best_reward = ep_reward
                    agent.save_checkpoint(Path(env.config.jet.jet_params["checkpoint_dir"]), "best") # Saves network parameters of the best performing episode
                if (ep + 1) % env.config.jet.jet_params["checkpoint_interval"] == 0:
                    agent.save_checkpoint(Path(env.config.jet.jet_params["checkpoint_dir"]), f"ep{ep + 1}") # Saves network parameters every "checkpoint_dir" number of episodes
        if rank == 0 and not STOP:
            agent.save_checkpoint(Path(env.config.jet.jet_params["checkpoint_dir"]), "final")
        if write_training:
            comm.Barrier()
            if rank == 0 and h5train is not None:
                h5train.close()
        
        return best_path

    def evaluate(env: StreamsGymEnv, agent, checkpoint: Path) -> None:
        """Run evaluation episodes using checkpoint."""
        comm = MPI.COMM_WORLD
        rank = comm.rank

        if rank == 0:
            print(f'EVALUATION')
            agent.load_checkpoint(checkpoint)

        # open output file for time, amp, reward, and obs in the evaluation loop to be collected
        write_eval = env.config.jet.jet_params["eval_output"] is not None
        h5eval = time_dset = amp_dset = reward_dset = obs_dset = None
        if write_eval:
            eval_output_path = Path(env.config.jet.jet_params["eval_output"])
            if rank == 0:
                eval_output_path.parent.mkdir(parents=True, exist_ok=True)
                h5eval = io_utils.IoFile(str(eval_output_path))
                eval_episodes = env.config.jet.jet_params["eval_episodes"]
                eval_steps = env.config.jet.jet_params["eval_max_steps"]
                observation_dim = env.observation_space.shape[0]

                time_dset = h5eval.file.create_dataset("time", shape=(eval_episodes, eval_steps), dtype="f4")
                amp_dset = h5eval.file.create_dataset("amplitude", shape=(eval_episodes, eval_steps), dtype="f4")
                reward_dset = h5eval.file.create_dataset("reward", shape=(eval_episodes, eval_steps), dtype="f4")
                if env.config.jet.jet_params["eval_max_steps"] > 100:
                    obs_dset = h5eval.file.create_dataset("observation", shape=(eval_episodes, eval_steps, observation_dim), dtype="f4", chunks=(1, 100, observation_dim))  
                else:
                    obs_dset = h5eval.file.create_dataset("observation", shape=(eval_episodes, eval_steps, observation_dim), dtype="f4")
                base_fields_dir = eval_output_path.parent.parent / "LB_EvalData"
                if base_fields_dir.exists():
                    shutil.rmtree(base_fields_dir)  # Remove the directory and all its contents
                base_fields_dir.mkdir(parents=True)
            else:
                base_fields_dir = None
            base_fields_dir = comm.bcast(base_fields_dir, root=0)
            comm.Barrier()              
                
        else:
            eval_output_path = None
            base_fields_dir = None
            
        comm.Barrier()
        
        for ep in range(env.config.jet.jet_params["eval_episodes"]): #, disable=rank != 0):
            if write_eval:
                if rank == 0:
                    print(f'Beginning of evaluation episode {ep + 1}')
                    ep_dir = base_fields_dir / f"ep_{ep:04d}"
                    ep_dir.mkdir(parents=True, exist_ok=True)
                    ep_dir_str = str(ep_dir)
                else:
                    ep_dir_str = None
                ep_dir_str = comm.bcast(ep_dir_str, root=0)
                env.init_h5_io(Path(ep_dir_str))
                
            obs = env.reset()
            done = False
            step = 0
            ep_reward = 0.0
            sa_queue = deque()
            def _to_numpy_copy(array_like):
                if isinstance(array_like, torch.Tensor):
                    return array_like.detach().cpu().numpy().copy()
                return np.array(array_like, copy=True)            
            
            while not done and step < env.config.jet.jet_params["eval_max_steps"]:
                if rank == 0:
                    obs_array = _to_numpy_copy(obs)
                    obs_tensor = torch.tensor(obs_array, dtype=torch.float32)
                    raw_action = agent.choose_action(obs_tensor, step)
                    actuation, obs_t, obs_t_next, convection_complete = env.delay_action(raw_action, obs_tensor)
                else:
                    actuation = None
                actuation = comm.bcast(actuation, root=0)
                obs, reward, done, info = env.step(actuation)
                done = comm.bcast(done, root=0)
                if write_eval:
                    env.log_step_h5(actuation)
                if rank == 0:
                    next_obs_array = _to_numpy_copy(obs)
                    if convection_complete:
                        delayed_obs = _to_numpy_copy(obs_t)
                        delayed_next_obs = _to_numpy_copy(obs_t_next)
                        sa_queue.append((delayed_obs, actuation, delayed_next_obs, info["time"]))
                    else:
                        sa_queue.append((obs_array, actuation, next_obs_array, info["time"]))
                    if len(sa_queue) > env.lag_steps:
                        old_obs, old_action, _, old_time = sa_queue.popleft()
                        old_obs_np = _to_numpy_copy(old_obs)
                        ep_reward += reward
                        idx = step - env.lag_steps
                        if write_eval:
                            time_dset[ep, idx] = old_time
                            amp_dset[ep, idx] = old_action
                            reward_dset[ep, idx] = reward
                            obs_dset[ep, idx, :] = old_obs_np
                step += 1
            if write_eval:
                comm.Barrier()
                env.close_h5_io()
                
        if write_eval:
            comm.Barrier()
            if rank == 0 and h5eval is not None:
                h5eval.close()
            env.close_h5_io()



    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--eval-only", action="store_true",
                            help="skip training and only run evaluation")
        parser.add_argument("--checkpoint", type=Path,
                            help="path to checkpoint to load for evaluation")
        args = parser.parse_args()

    # Generate random number via torch. Not used now but present for future restart use.
    torch.manual_seed(env.config.jet.jet_params["seed"])
    np.random.seed(env.config.jet.jet_params["seed"])

    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if env.rank == 0:
        print(f'state_dim: {env.observation_space.shape[0]}') # Collect the state dimension (tau x, equal to x grid dim)
        print(f'action_dim: {env.action_space.shape[0] }') # Collect the action dimension (integer valued jet amplitude)
        print(f'max_action: {float(env.action_space.high[0])}') # Specified in justfile
        print(f"Using device: {device} (CUDA available: {torch.cuda.is_available()})") # Display whether GPU was in fact available
    
    strategy = env.config.jet.jet_strategy_name
    module_path = f"Control.{strategy}"
    agent_module = importlib.import_module(module_path)
    #module_path = f".Control.{strategy}"
    #agent_module = importlib.import_module(module_path, package=__package__)
    agent_class = getattr(agent_module, "agent")
    
    # Clear the checkpoints directory before training runs, but not evaluation runs 
    checkpoint_dir = Path(env.config.jet.jet_params.get("checkpoint_dir"))
    if not args.eval_only:
        # only rank 0 should clear existing checkpoints to avoid race conditions
        if env.rank == 0 and checkpoint_dir.exists():
            for item in checkpoint_dir.iterdir():
                try:
                    if item.is_file() or item.is_symlink():
                        item.unlink()
                    else:
                        shutil.rmtree(item)
                except FileNotFoundError:
                    # another process may have removed the file
                    pass
        # make sure all ranks wait for the cleanup to finish
        env.comm.Barrier()

    comm = env.comm
    rank = env.rank

    if rank == 0:
        agent = agent_class(env)
    else:
        agent = None

    if args.eval_only:
        ckpt = args.checkpoint if args.checkpoint is not None else Path(env.config.jet.jet_params["checkpoint_dir"]) / "best"
        evaluate(env, agent, ckpt)
    else:
        best_ckpt = train(env, agent)
        evaluate(env, agent, best_ckpt)
    env.close()

else:
    if env.rank == 0:
        print(f'blowing_bc is set to {env.config.jet.jet_method_name}.')
        print("blowing_bc must be set to None, OpenLoop, Classical, or LearningBased")
