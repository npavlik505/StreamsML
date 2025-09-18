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
            while not done:
                if rank == 0:
                    obs = torch.tensor(obs, dtype=torch.float32)
                    action_t = agent.choose_action(obs, step)
                    action_t, obs_t, obs_t_next, convection_complete = agent.delay_action(obs, action) # Delay action if desired
                else:
                    action_t = None
                action_t = comm.bcast(action, root=0)
                next_obs, reward, done, info = env.step(action_t)
                if convection_complete:
                    prev_obs_t = obs_t
                    next_obs_t = obs_t_next
                    done = comm.bcast(done, root=0)
                    if rank == 0:
                        sa_queue.append((prev_obs, action_t, next_obs_t, info["time"]))
                        if len(sa_queue) > env.lag_steps:
                            old_obs, old_action, old_next, old_time = sa_queue.popleft()
                            ep_reward += reward
                            agent.learn(old_obs, old_action, reward, old_next)
                            if write_training:
                                idx = step - env.lag_steps
                                time_dset[ep, idx] = old_time
                                amp_dset[ep, idx] = old_action
                                reward_dset[ep, idx] = reward
                                obs_dset[ep, idx, :] = old_obs
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
        # ep_dset.close()
        # h5.close()
        
        return best_path
