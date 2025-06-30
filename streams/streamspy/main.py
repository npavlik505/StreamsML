#!/usr/bin/env python3
# Run STREAmS using the gymnasium environment wrapper

import json # collects values from input.json fields, converts them to a nested Python dictionary, which is then converted into a Config object (e.g. config.temporal.num_iter)
import os
import runpy
from pathlib import Path

import numpy as np
print("Made it back to main.py")
from StreamsEnvironment import StreamsGymEnv

env = StreamsGymEnv()

# If the configuration requests the adaptive jet, delegate to ``rl_control.py``
# print('we are here')
# with open("/input/input.json", "r") as f:
#    cfg_json = json.load(f)
#    cfg = Config.from_json(cfg_json)
#    print(cfg.jet.jet_method)
# if cfg.jet.jet_method != "LearningBased":
if env.config.jet.jet_method != "LearningBased":

    # Instantiate environment and prep h5 files for standard datasets
    # env = StreamsGymEnv()
    env.init_h5_io()

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

else:
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
    from DDPG import ddpg, ReplayBuffer
    from config import Config, JetMethod
    import io_utils

    LOGGER = logging.getLogger(__name__)
    STOP = False


    def _signal_handler(signum, frame):
        global STOP
        STOP = True
        LOGGER.info("Received interrupt signal. Stopping after current episode...")

    signal.signal(signal.SIGINT, _signal_handler)

    def setup_logging() -> None:
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

    def save_checkpoint(agent: ddpg, directory: Path, tag: str) -> None:
        """Save actor and critic weights."""
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(agent.actor.state_dict(), directory / f"actor_{tag}.pt")
        torch.save(agent.critic.state_dict(), directory / f"critic_{tag}.pt")


    def train(env: StreamsGymEnv, agent: ddpg, args: argparse.Namespace) -> Path:
        """Train agent and return path to best checkpoint."""
        comm = MPI.COMM_WORLD
        rank = comm.rank
        if rank == 0:
            print(f'[rl_control.py] TRAINING')
        
        # print("[rl_control.py] Define objects for observation and action space")
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        # print("[rl_control.py] Define object for replay buffer")
        buffer = ReplayBuffer(state_dim, action_dim, args.buffer_size)
        
        # print("[rl_control.py] Define reward objects")
        best_reward = -float("inf")
        best_path = Path(args.checkpoint_dir) / "best"
        episode_rewards = []
        
        # open output file for training statistics on all ranks
        metrics_path = Path(args.checkpoint_dir) / "training.h5"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        h5 = io_utils.IoFile(str(metrics_path))
        loss_writes = args.train_episodes * env.max_episode_steps
        actor_dset = io_utils.Scalar0D(h5, [1], loss_writes, "actor_loss", rank)
        critic_dset = io_utils.Scalar0D(h5, [1], loss_writes, "critic_loss", rank)
        ep_dset = io_utils.Scalar0D(h5, [1], args.train_episodes, "episode_reward", rank)
        
        # open output file for time, amp, reward, and obs in the training loop to be collected
        write_training = args.training_output is not None
        if write_training:
            # Define path, create directory and h5, gather data to allocate shape
            training_output_path = Path(args.training_output)
            training_output_path.parent.mkdir(parents=True, exist_ok=True)
            h5train = io_utils.IoFile(str(training_output_path))
            training_episodes = args.train_episodes
            training_steps = env.max_episode_steps
            observation_dim = env.observation_space.shape[0]
            
            # Define frequency of collection
            if training_steps/10 >= 1:
                write_spacing = training_steps // 10
            else:
                write_spacing = 1
            
            # Create datasets within the h5 file
            time_dset = h5train.file.create_dataset("time", shape = (training_episodes, training_steps), dtype = "f4")
            amp_dset = h5train.file.create_dataset("amplitude", shape = (training_episodes, training_steps), dtype = "f4")
            reward_dset = h5train.file.create_dataset("reward", shape = (training_episodes, training_steps), dtype = "f4")
            obs_dset = h5train.file.create_dataset("observation", shape = (training_episodes, training_steps, observation_dim), dtype = "f4")
                
        for ep in range(args.train_episodes): #, disable=rank != 0):
            if rank == 0:
                print(f'[rl_control.py] Beginning of training episode {ep}')
            if STOP:
                break
            obs = env.reset()
            done = False
            ep_reward = 0.0
            step = 0
            while not done:
                if rank == 0:
                    obs_t = torch.tensor(obs, dtype=torch.float32)
                    action = agent.choose_action(obs_t)
                else:
                    action = None
                action = comm.bcast(action, root=0)
                next_obs, reward, done, info = env.step(action)
                done = comm.bcast(done, root=0)
                if rank == 0:
                    ep_reward += reward
                    if write_training and step % write_spacing == 0:
                        time_dset[ep, step] = info["time"]
                        amp_dset[ep, step] = action
                        reward_dset[ep, step] = reward
                        obs_dset[ep, step, :] = obs

                    buffer.store(
                        torch.tensor(obs, dtype=torch.float32),
                        torch.tensor(action, dtype=torch.float32),
                        torch.tensor([reward], dtype=torch.float32),
                        torch.tensor(next_obs, dtype=torch.float32),
                    )
                    if buffer.size >= agent.batch_size:
                        actor_loss, critic_loss = ddpg_update(agent, buffer)
                        LOGGER.debug("actor_loss=%f critic_loss=%f", actor_loss, critic_loss)
                        if actor_dset is not None:
                            actor_dset.write_array(np.array([actor_loss], dtype=np.float32))
                            critic_dset.write_array(np.array([critic_loss], dtype=np.float32))
                step += 1
                obs = next_obs
            if rank == 0:
                episode_rewards.append(ep_reward)
                LOGGER.info("Training Episode %d reward %.6f", ep + 1, ep_reward)
                if ep_reward > best_reward:
                    best_reward = ep_reward
                    save_checkpoint(agent, Path(args.checkpoint_dir), "best")
                if (ep + 1) % args.checkpoint_interval == 0:
                    save_checkpoint(agent, Path(args.checkpoint_dir), f"ep{ep + 1}")
        if rank == 0 and not STOP:
            save_checkpoint(agent, Path(args.checkpoint_dir), "final")
        print("Just before h5train.close()")
        if write_training:
            comm.Barrier()
            h5train.close()
        actor_dset.close()
        critic_dset.close()
        ep_dset.close()
        h5.close()
        return best_path

    def evaluate(env: StreamsGymEnv, agent: ddpg, args: argparse.Namespace, checkpoint: Path) -> None:
        """Run evaluation episodes using checkpoint."""
        comm = MPI.COMM_WORLD
        rank = comm.rank
        if rank == 0:
            print(f'[rl_control.py] EVALUATION')
        agent.actor.load_state_dict(torch.load(checkpoint.with_name("actor_best.pt")))
        agent.critic.load_state_dict(torch.load(checkpoint.with_name("critic_best.pt")))

        # open output file for time, amp, reward, and obs in the evaluation loop to be collected
        write_eval = args.eval_output is not None
        if write_eval:
            # Define path, create directory and h5, gather data to allocate shape
            eval_output_path = Path(args.eval_output)
            eval_output_path.parent.mkdir(parents=True, exist_ok=True)
            h5eval = io_utils.IoFile(str(eval_output_path))
            eval_episodes = args.eval_episodes
            eval_steps = env.max_episode_steps
            observation_dim = env.observation_space.shape[0]
            
            # Define frequency of collection
            if eval_steps/10 >= 1:
                write_spacing = eval_steps // 10
            else:
                write_spacing = 1
            
            # Create datasets within the h5 file
            time_dset = h5eval.file.create_dataset("time", shape = (eval_episodes, eval_steps), dtype = "f4")
            amp_dset = h5eval.file.create_dataset("amplitude", shape = (eval_episodes, eval_steps), dtype = "f4")
            reward_dset = h5eval.file.create_dataset("reward", shape = (eval_episodes, eval_steps), dtype = "f4")
            obs_dset = h5eval.file.create_dataset("observation", shape = (eval_episodes, eval_steps, observation_dim), dtype = "f4")
        for ep in range(args.eval_episodes): #, disable=rank != 0):
            if rank == 0:
                print(f'[rl_control.py] Beginning of evaluation episode {ep}')
            obs = env.reset()
            done = False
            step = 0
            ep_reward = 0.0
            while not done and step < args.eval_max_steps:
                if rank == 0:
                    print(f'[rl_control.py] Evaluate step = {step}')
                    action = agent.choose_action(torch.tensor(obs, dtype=torch.float32))
                else:
                    action = None
                action = comm.bcast(action, root=0)
                obs, reward, done, info = env.step(action)
                done = comm.bcast(done, root=0)
                if rank == 0:
                    ep_reward += reward
                    if write_eval and step % write_spacing == 0:
                        time_dset[ep, step] = info["time"]
                        amp_dset[ep, step] = action
                        reward_dset[ep, step] = reward
                        obs_dset[ep, step, :] = obs
                step += 1
            if rank == 0:
                LOGGER.info("Eval Episode %d reward %.6f", ep + 1, ep_reward)
        if write_eval:
            comm.Barrier()
            h5eval.close()

    # Function providing all RL parameters with default values using argparse.
    def parse_args() -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description="DDPG control for STREAmS")
        parser.add_argument("--config", type=str, default="/output/input.json", help="Path to input.json")
        parser.add_argument("--train-episodes", type=int, default=10)
        parser.add_argument("--eval-episodes", type=int, default=5)
        parser.add_argument("--eval-max-steps", type=int, default=1000)
        parser.add_argument("--checkpoint-dir", type=str, default="./RL_metrics/checkpoint")
        parser.add_argument("--checkpoint-interval", type=int, default=5)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--learning-rate", type=float, default=3e-4)
        parser.add_argument("--gamma", type=float, default=0.99)
        parser.add_argument("--tau", type=float, default=0.005)
        parser.add_argument("--buffer-size", type=int, default=int(1e6))
        parser.add_argument("--training-output", type=str, default=None, help="Optional HDF5 file for training loop info")
        parser.add_argument("--eval-output", type=str, default=None, help="Optional HDF5 file for evaluation loop info")
        parser.add_argument("--verbose", type=bool, default=False, help = "Print network details") # action = "store_true", help = "Print network details")
        return parser.parse_args()

    if __name__ == "__main__":
        setup_logging()
        args = parse_args() # instance of parser assigned to object args 
        config = Config.from_json(args.config)
        extras = {**(config.jet.jet_gen_json or {}), **(config.jet.jet_alg_json or {}), } # assign the blowing-bc options from input.json to extras as a dictionary
        
        # Create an Python dictionary. While key and value are identical here, only the _overrides key must match the extras key collected from input.json entry
        _overrides = {
            "train_episodes":      "train_episodes",
            "eval_episodes":       "eval_episodes",
            "eval_max_steps":      "eval_max_steps",
            "checkpoint_dir":      "checkpoint_dir",
            "checkpoint_interval": "checkpoint_interval",
            "seed":                "seed",
            "learning_rate":       "learning_rate",
            "gamma":               "gamma",
            "tau":                 "tau",
            "buffer_size":         "buffer_size",
            "eval_output":         "eval_output",
            "training_output":     "training_output",
            "verbose":             "verbose",
    }

    # Loop through overrides and if _overrides and extras keys match, replace the _overrides value with extras value.
    for json_key, arg_name in _overrides.items():
        if json_key in extras:
            setattr(args, arg_name, extras[json_key])

    # Double check that the blowing-bc = adaptive, and exit false
    if config.jet.jet_method != LearningBased:
        LOGGER.error("JetMethod is not LearningBased. Exiting RL controller.")
        exit(1)

    # Generate random number via torch. Not used now but present for future restart use.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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

    agent = ddpg(state_dim, action_dim, max_action, verbose=args.verbose) # instantiate the ddpg algorithm
    agent.lr = args.learning_rate # RL parameters stored in args using argparse/json method above
    agent.GAMMA = args.gamma
    agent.TAU = args.tau
    agent.actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=agent.lr) #Initialize actor parameters
    agent.critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr=agent.lr) #Initialize critc parameters

    best_ckpt = train(env, agent, args) # Train the algorithm. Method above.

    evaluate(env, agent, args, best_ckpt) # Evaluate the algorithm. Method Above.

    env.close()

