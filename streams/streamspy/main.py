#!/usr/bin/env python3
# Run STREAmS using the gymnasium environment wrapper

import json # collects values from input.json fields, converts them to a nested Python dictionary, which is then converted into a Config object (e.g. config.temporal.num_iter)
import h5py
import os
import runpy
from pathlib import Path
import numpy as np
from StreamsEnvironment import StreamsGymEnv

env = StreamsGymEnv()

if env.config.jet.jet_method_name == "OpenLoop":

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
    print("Classical control methods have yet to be implemented")
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
    from mpi4py import MPI

    # Script imports
    # from env.config.jet.parameters["strategy"] import agent
    import importlib
    # from DDPG import ddpg, ReplayBuffer
    # from config import Config, Jet
    from base_agent import BaseAgent
    import io_utils
    import inspect

    LOGGER = logging.getLogger(__name__)
    STOP = False


    def _signal_handler(signum, frame):
        global STOP
        STOP = True
        LOGGER.info("Received interrupt signal. Stopping after current episode...")

    signal.signal(signal.SIGINT, _signal_handler)

    def setup_logging() -> None: # Is this worth keeping now that we have H5 files generated for data... ?
        """Configure root logger to log to console and file."""
        LOGGER.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        log_path = Path("RL_metrics/rl_control.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        LOGGER.addHandler(fh)
        LOGGER.addHandler(ch)

    def train(env: StreamsGymEnv, agent) -> Path:
        """Train agent and return path to best checkpoint."""
        comm = MPI.COMM_WORLD
        rank = comm.rank
        if rank == 0:
            print(f'TRAINING')
        
        # print("[rl_control.py] Define objects for observation and action space")
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        # print("[rl_control.py] Define reward objects")
        best_reward = -float("inf")
        episode_rewards = []
        
        best_path = Path(env.config.jet.jet_params["checkpoint_dir"]) / "best"
        
        # open output file for training statistics on all ranks
        # metrics_path = Path(env.config.jet.jet_params["checkpoint_dir"]) / "training.h5"
        # metrics_path = Path(env.config.jet.jet_params["checkpoint_dir"])
        # metrics_path.parent.mkdir(parents=True, exist_ok=True)
        # h5 = io_utils.IoFile(str(metrics_path))
        # loss_writes = env.config.jet.jet_params["train_episodes"] * env.max_episode_steps
        # ep_dset = io_utils.Scalar0D(h5, [1], env.config.jet.jet_params["train_episodes"], "episode_reward", rank)
        
        # open output file for time, amp, reward, and obs in the training loop to be collected
        write_training = env.config.jet.jet_params["training_output"] is not None
        if write_training:
            # Define path, create directory and h5, gather data to allocate shape
            training_output_path = Path(env.config.jet.jet_params["training_output"])
            training_output_path.parent.mkdir(parents=True, exist_ok=True)
            h5train = io_utils.IoFile(str(training_output_path))
            training_episodes = env.config.jet.jet_params["train_episodes"]
            training_steps = env.max_episode_steps
            observation_dim = env.observation_space.shape[0]
            
            # Define frequency of collection (clunky... revisit)
            # if training_steps/10 >= 1:
            #     write_spacing = training_steps // 10
            # else:
            #     write_spacing = 1
            
            # Create datasets within the h5 file
            time_dset = h5train.file.create_dataset("time", shape = (training_episodes, training_steps), dtype = "f4")
            amp_dset = h5train.file.create_dataset("amplitude", shape = (training_episodes, training_steps), dtype = "f4")
            reward_dset = h5train.file.create_dataset("reward", shape = (training_episodes, training_steps), dtype = "f4")
            obs_dset = h5train.file.create_dataset("observation", shape = (training_episodes, training_steps, observation_dim), dtype = "f4")
                
        for ep in range(env.config.jet.jet_params["train_episodes"]): #, disable=rank != 0):
            if rank == 0:
                print(f'Beginning of training episode {ep + 1}')
            if STOP:
                break
            obs = env.reset()
            done = False
            ep_reward = 0.0
            step = 0
            while not done:
                if rank == 0:
                    obs_t = torch.tensor(obs, dtype=torch.float32)
                    action = agent.choose_action(obs_t, step)
                else:
                    action = None
                action = comm.bcast(action, root=0)
                next_obs, reward, done, info = env.step(action)
                done = comm.bcast(done, root=0)
                if rank == 0:
                    ep_reward += reward
                    agent.learn(obs, action, reward, next_obs)
                    # if write_training and step % write_spacing == 0:
                    time_dset[ep, step] = info["time"]
                    amp_dset[ep, step] = action
                    reward_dset[ep, step] = reward
                    obs_dset[ep, step, :] = obs
                step += 1
                obs = next_obs
            if rank == 0:
                episode_rewards.append(ep_reward)
                LOGGER.info("Training Episode %d reward %.6f", ep + 1, ep_reward)
                if ep_reward > best_reward: # 
                    best_reward = ep_reward
                    agent.save_checkpoint(Path(env.config.jet.jet_params["checkpoint_dir"]), "best") # Saves network parameters of the best performing episode
                if (ep + 1) % env.config.jet.jet_params["checkpoint_interval"] == 0:
                    agent.save_checkpoint(Path(env.config.jet.jet_params["checkpoint_dir"]), f"ep{ep + 1}") # Saves network parameters every "checkpoint_dir" number of episodes
        if rank == 0 and not STOP:
            agent.save_checkpoint(Path(env.config.jet.jet_params["checkpoint_dir"]), "final")
        if write_training:
            comm.Barrier()
            h5train.close()
        # ep_dset.close()
        # h5.close()
        
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
        h5eval = None
        if write_eval:
            # Define path, create directory and h5, gather data to allocate shape
            eval_output_path = Path(env.config.jet.jet_params["eval_output"])
            if rank == 0:
                eval_output_path.parent.mkdir(parents=True, exist_ok=True)
            comm.Barrier()
            
            h5eval = io_utils.IoFile(str(eval_output_path))
            eval_episodes = env.config.jet.jet_params["eval_episodes"]
            eval_steps = env.max_episode_steps
            observation_dim = env.observation_space.shape[0]
            
            # Define frequency of collection
            # if eval_steps/10 >= 1:
            #     write_spacing = eval_steps // 10
            # else:
            #     write_spacing = 1
            
            # Create datasets within the h5 file
            time_dset = h5eval.file.create_dataset("time", shape = (eval_episodes, eval_steps), dtype = "f4")
            amp_dset = h5eval.file.create_dataset("amplitude", shape = (eval_episodes, eval_steps), dtype = "f4")
            reward_dset = h5eval.file.create_dataset("reward", shape = (eval_episodes, eval_steps), dtype = "f4")
            obs_dset = h5eval.file.create_dataset("observation", shape = (eval_episodes, eval_steps, observation_dim), dtype = "f4")
                
            if rank == 0:
                base_fields_dir = eval_output_path.parent.parent / "LB_EvalData"
                if base_fields_dir.exists():
                    shutil.rmtree(base_fields_dir)  # Remove the directory and all its contents
                base_fields_dir.mkdir(parents=True)
                
            else:
                base_fields_dir = None
            base_fields_dir = comm.bcast(base_fields_dir, root=0)                
                
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
            
            while not done and step < env.config.jet.jet_params["eval_max_steps"]:
                if rank == 0:
                    action = agent.choose_action(torch.tensor(obs, dtype=torch.float32))
                else:
                    action = None
                action = comm.bcast(action, root=0)
                obs, reward, done, info = env.step(action)
                done = comm.bcast(done, root=0)
                if write_eval:
                    env.log_step_h5(action)
                    if rank == 0:
                        ep_reward += reward
                        # if write_eval and step % write_spacing == 0:
                        time_dset[ep, step] = info["time"]
                        amp_dset[ep, step] = action
                        reward_dset[ep, step] = reward
                        obs_dset[ep, step, :] = obs
                step += 1
            if write_eval:
                comm.Barrier()
                env.close_h5_io()
            if rank == 0:
                LOGGER.info("Eval Episode %d reward %.6f", ep + 1, ep_reward)
        if write_eval:
            comm.Barrier()
            if h5eval is not None:
                h5eval.close()
            env.close_h5_io()



    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--eval-only", action="store_true",
                            help="skip training and only run evaluation")
        parser.add_argument("--checkpoint", type=Path,
                            help="path to checkpoint to load for evaluation")
        args = parser.parse_args()
        
        # setup_logging()

    # Generate random number via torch. Not used now but present for future restart use.
    torch.manual_seed(env.config.jet.jet_params["seed"])
    np.random.seed(env.config.jet.jet_params["seed"])

    # Use GPU if available. Currently only used for open-loop control
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # env = StreamsGymEnv() # Import Streams Gym Environment
    state_dim = env.observation_space.shape[0] # Collect the state dimension (tau x, equal to x grid dim)
    action_dim = env.action_space.shape[0] # Collect the action dimension (integer valued jet amplitude)
    max_action = float(env.action_space.high[0]) # Specified in justfile

    if env.rank == 0:
        print(f'state_dim: {state_dim}')
        print(f'action_dim: {action_dim}')
        print(f'max_action: {max_action}')
    
    strategy = env.config.jet.jet_strategy_name
    module_path = f"Control.{strategy}"
    agent_module = importlib.import_module(module_path)
    agent_class = getattr(agent_module, "agent")

    # include all parameters required for the initialization of any LearningBased agent 
    # Note to self: Consider collapsing all learning-based structs into one single struct with default values and let Python
    # initialize an algorithm based on the match method below. 
    # Pros: Simpler Rust code and edit, less redundancy, and less frequent Rust edits. 
    # Cons: Giant Rust struct and json file when scaled up, and less informative Rust errors and help function. 
    
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
    
    agent_kwargs = {
        # "checkpoint_dir": env.config.jet.jet_params.get("checkpoint_dir"),
        "checkpoint_dir": str(checkpoint_dir),
        "hidden_width": env.config.jet.jet_params.get("hidden_width"),
        "buffer_size": env.config.jet.jet_params.get("buffer_size"),
        "batch_size": env.config.jet.jet_params.get("batch_size"),
        "lr": env.config.jet.jet_params.get("learning_rate"),
        "target_update" :env.config.jet.jet_params.get("target_update"),
        "GAMMA": env.config.jet.jet_params.get("gamma"),
        "TAU": env.config.jet.jet_params.get("tau"),
        "epsilon": env.config.jet.jet_params.get("epsilon"),
        "eps_clip": env.config.jet.jet_params.get("eps_clip"),
        "K_epochs": env.config.jet.jet_params.get("K_epochs"),
    } 
    
    # keeps only the parameters that the selected agent accepts
    sig = inspect.signature(agent_class.__init__)
    filtered = {k: v for k, v in agent_kwargs.items() if k in sig.parameters}

    agent = agent_class(state_dim, action_dim, max_action, **filtered)

    if args.eval_only:
        ckpt = args.checkpoint if args.checkpoint is not None else Path(env.config.jet.jet_params["checkpoint_dir"]) / "best"
        evaluate(env, agent, ckpt)
    else:
        best_ckpt = train(env, agent)
        evaluate(env, agent, best_ckpt)
    env.close()

else:
    print(f'blowing_bc is set to {env.config.jet.jet_method_name}.')
    print("blowing_bc must be set to None, OpenLoop, Classical, or LearningBased")
