#!/usr/bin/env python3
"""
Usage:
    python train_dqn.py \
      --train-instances ta01 ta02 ta03 ta04 ta05 \
      --total-steps 500000 \
      --batch-size 64 \
      --buffer-capacity 100000 \
      --lr 1e-4 \
      --gamma 1.0 \
      --eps-start 1.0 \
      --eps-final 0.05 \
      --eps-decay-steps 200000 \
      --target-update-freq 1000 \
      --train-interval 4 \
      --device cpu \
      --log-dir runs/dqn
"""

import os
import argparse
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import JSSEnv  
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

torch.set_num_threads(os.cpu_count())

Transition = namedtuple(
    "Transition",
    ["real_obs", "action_mask", "action", "reward", "next_real_obs", "next_action_mask", "done"]
)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        # Store a transition
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        real_obs_batch = np.stack([t.real_obs for t in batch], axis=0)            
        action_mask_batch = np.stack([t.action_mask for t in batch], axis=0)      
        actions = np.array([t.action for t in batch], dtype=np.int64)             
        rewards = np.array([t.reward for t in batch], dtype=np.float32)           
        next_real_obs_batch = np.stack([t.next_real_obs for t in batch], axis=0)  
        next_action_mask_batch = np.stack([t.next_action_mask for t in batch], axis=0)  
        dones = np.array([t.done for t in batch], dtype=np.float32)               
        return (
            real_obs_batch, action_mask_batch, actions,
            rewards, next_real_obs_batch, next_action_mask_batch, dones
        )

    def __len__(self):
        return len(self.buffer)


# Q net
class QNetwork(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        jobs, feat = obs_shape
        input_size = jobs * feat

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, real_obs: torch.Tensor):
        B = real_obs.shape[0]
        x = real_obs.view(B, -1)  # flatten
        return self.net(x)


# DQN Trainer 
def train_dqn(
    train_instances,
    total_steps,
    batch_size,
    buffer_capacity,
    lr,
    gamma,
    eps_start,
    eps_final,
    eps_decay_steps,
    target_update_freq,
    train_interval,
    eval_interval,
    device,
    log_dir
):

    writer = SummaryWriter(log_dir) if log_dir is not None else None

    # Initialize buffer, networks, optimizer, and counters
    buffer = ReplayBuffer(buffer_capacity)


    first_path = os.path.join("JSSEnv", "envs", "instances", train_instances[0])
    env0 = gym.make("JSSEnv/JssEnv-v1", env_config={"instance_path": first_path})
    obs0, _ = env0.reset()
    jobs = obs0["real_obs"].shape[0]
    n_actions = obs0["action_mask"].shape[0]
    obs_shape = obs0["real_obs"].shape  
    env0.close()

    # Create networks
    q_net = QNetwork(obs_shape, n_actions).to(device)
    target_net = QNetwork(obs_shape, n_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    step_count = 0
    update_count = 0
    eps = eps_start

    episode_returns = []
    episode_lengths = []

    pbar = tqdm(total=total_steps, desc="DQN steps", unit="step")

    # Main training loop whihc rolls out episodes until total steps has been reached
    while step_count < total_steps:
        # At start of each episode picks a random instance
        instance_name = random.choice(train_instances)
        inst_path = os.path.join("JSSEnv", "envs", "instances", instance_name)
        env = gym.make("JSSEnv/JssEnv-v1", env_config={"instance_path": inst_path})
        obs_dict, _ = env.reset()
        real_obs = obs_dict["real_obs"].astype(np.float32)      
        action_mask = obs_dict["action_mask"].astype(np.bool_)  

        episode_return = 0.0
        episode_len = 0
        done = False
        truncated = False

        while not (done or truncated) and step_count < total_steps:
            #  Epsilon-greedy
            real_obs_tensor = torch.tensor(real_obs, dtype=torch.float32, device=device).unsqueeze(0)  
            mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=device).unsqueeze(0)    

            with torch.no_grad():
                q_values = q_net(real_obs_tensor)  
            q_values[~mask_tensor] = -1e8

            if random.random() < eps:
                valid_indices = np.where(action_mask)[0]
                action = random.choice(valid_indices)
            else:
                action = int(q_values.argmax(dim=1).item())

            next_obs_dict, reward, done, truncated, info = env.step(action)
            next_real_obs = next_obs_dict["real_obs"].astype(np.float32)
            next_action_mask = next_obs_dict["action_mask"].astype(np.bool_)

            episode_return += reward
            episode_len += 1
            step_count += 1
            pbar.update(1)

            buffer.push(
                real_obs, action_mask, action, float(reward),
                next_real_obs, next_action_mask, float(done or truncated)
            )

            real_obs = next_real_obs
            action_mask = next_action_mask

            # Linearly decay epsilon
            eps = max(eps_final, eps - (eps_start - eps_final) / eps_decay_steps)
            if len(buffer) >= batch_size and step_count % train_interval == 0:
                (
                    ro_batch, am_batch, a_batch,
                    r_batch, nro_batch, nam_batch, d_batch
                ) = buffer.sample(batch_size)

                ro_t = torch.tensor(ro_batch, dtype=torch.float32, device=device)      
                am_t = torch.tensor(am_batch, dtype=torch.bool, device=device)         
                a_t = torch.tensor(a_batch, dtype=torch.int64, device=device)          
                r_t = torch.tensor(r_batch, dtype=torch.float32, device=device)        
                nro_t = torch.tensor(nro_batch, dtype=torch.float32, device=device)    
                nam_t = torch.tensor(nam_batch, dtype=torch.bool, device=device)       
                d_t = torch.tensor(d_batch, dtype=torch.float32, device=device)        

                q_pred_all = q_net(ro_t)            
                q_pred = q_pred_all.gather(1, a_t.unsqueeze(1)).squeeze(1)  

                with torch.no_grad():
                    q_next_all = target_net(nro_t)      
                    q_next_all[~nam_t] = -1e8
                    q_next_max, _ = q_next_all.max(dim=1)  
                    q_target = r_t + gamma * q_next_max * (1.0 - d_t)

                loss = nn.MSELoss()(q_pred, q_target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                update_count += 1
                if update_count % target_update_freq == 0:
                    target_net.load_state_dict(q_net.state_dict())

                if writer:
                    writer.add_scalar("train/loss", loss.item(), update_count)
                    writer.add_scalar("train/epsilon", eps, update_count)

            if writer and eval_interval > 0 and step_count % eval_interval == 0:
                writer.add_scalar("train/episode_return", episode_return, step_count)
                writer.add_scalar("train/episode_length", episode_len, step_count)

        if writer:
            writer.add_scalar("episode/return", episode_return, step_count)
            writer.add_scalar("episode/length", episode_len, step_count)

        episode_returns.append(episode_return)
        episode_lengths.append(episode_len)
        env.close()

    pbar.close()

    # After total steps save Qnet and target network
    os.makedirs("models", exist_ok=True)
    torch.save(q_net.state_dict(), "models/dqn_model.pth")
    torch.save(target_net.state_dict(), "models/dqn_target_model.pth")
    if writer:
        writer.close()

    print("Training finished.")
    print(f"Average return (last 10 eps): {np.mean(episode_returns[-10:]):.2f}")
    print(f"Average length (last 10 eps): {np.mean(episode_lengths[-10:]):.2f}")
    print("Saved models to models/dqn_model.pth and models/dqn_target_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DQN on JSSenv")
    parser.add_argument(
        "--train-instances", nargs="+",
        default=["ta01", "ta02", "ta03", "ta04", "ta05"],
        help="List of ta instances for training"
    )
    parser.add_argument(
        "--total-steps", type=int, default=300000,
        help="Total env steps to train "
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for DQN updates"
    )
    parser.add_argument(
        "--buffer-capacity", type=int, default=100000,
        help="Replay buffer capacity"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate for Adam"
    )
    parser.add_argument(
        "--gamma", type=float, default=1.0,
        help="Discount factor "
    )
    parser.add_argument(
        "--eps-start", type=float, default=1.0,
        help="Initial epsilon for exploration"
    )
    parser.add_argument(
        "--eps-final", type=float, default=0.05,
        help="Final epsilon after decay"
    )
    parser.add_argument(
        "--eps-decay-steps", type=int, default=150000,
        help="Number of steps over which epsilon decays"
    )
    parser.add_argument(
        "--target-update-freq", type=int, default=1000,
        help="Number of DQN updates between copying to target network"
    )
    parser.add_argument(
        "--train-interval", type=int, default=4,
        help="Perform a DQN update every train_interval environment steps"
    )
    parser.add_argument(
        "--eval-interval", type=int, default=0,
        help="If >0, log training metrics every `eval_interval` steps"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"],
        help="Device to train on"
    )
    parser.add_argument(
        "--log-dir", type=str, default="runs/dqn",
        help="TensorBoard log directory (set to '' to disable)"
    )

    args = parser.parse_args()

    train_dqn(
        train_instances=args.train_instances,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        lr=args.lr,
        gamma=args.gamma,
        eps_start=args.eps_start,
        eps_final=args.eps_final,
        eps_decay_steps=args.eps_decay_steps,
        target_update_freq=args.target_update_freq,
        train_interval=args.train_interval,
        eval_interval=args.eval_interval,
        device=args.device,
        log_dir=(args.log_dir if args.log_dir else None)
    )
