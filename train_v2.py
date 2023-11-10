import argparse
import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from typing import Callable
from agents import ModelAgent
from custom_cnn import CustomCNN
from environment import QuantumTiqTaqToe
from sb3_contrib.common.wrappers import ActionMasker
import torch
from unitary.examples.tictactoe.enums import TicTacSquare
from utils import enemy_map, valid_fn

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def train_agents(
          N,
          folder,
          policy,
          timestemps,
          verbose,
          logs,
          
          lr,
          linear_lr,
          n_steps

):
    env_X = QuantumTiqTaqToe(player=TicTacSquare.X)
    env_X = ActionMasker(env_X, valid_fn)
    env_O = QuantumTiqTaqToe(player=TicTacSquare.O)
    env_O = ActionMasker(env_O, valid_fn)
    if(policy == "CnnPolicy"):
        policy_kwargs = dict(
                features_extractor_class=CustomCNN,
                features_extractor_kwargs=dict(features_dim=128),
                activation_fn=torch.nn.ReLU,
                net_arch=dict(pi=[128, 128], vf=[128, 128])
            )
    elif(policy == "MlpPolicyV2"):
        policy = "MlpPolicy"    
        policy_kwargs = dict(
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[256, 128, 128], vf=[256, 128, 128])
        )
    else:
        policy_kwargs = dict()

    if(linear_lr):
        lr = linear_schedule(lr)
    
    model_X = MaskablePPO(policy, env_X, verbose = verbose, n_steps=n_steps, policy_kwargs=policy_kwargs,learning_rate=lr,tensorboard_log=logs)
    model_O = MaskablePPO(policy, env_O, verbose = verbose, n_steps=n_steps, policy_kwargs=policy_kwargs,learning_rate=lr,tensorboard_log=logs)

    model_path = folder + "/model"
    for i in range(N):
        if(verbose > 0):
             print(f"Train iteration:{i}, player: X")
        model_X.learn(total_timesteps = timestemps, tb_log_name=f"player_X",reset_num_timesteps=False )

        if(verbose > 0):
             print(f"Train iteration:{i}, player: O")
        model_O.learn(total_timesteps = timestemps, tb_log_name=f"player_O",reset_num_timesteps=False )
        
        model_X_path = f"{model_path}_X_{str(i+1)}"
        model_O_path = f"{model_path}_O_{str(i+1)}"
        model_X.save(model_X_path)
        model_O.save(model_O_path)

        
        enemy_agent_X = ModelAgent(model_X_path)
        enemy_agent_O = ModelAgent(model_O_path)
        
        env_X.set_enemy_agent(enemy_agent_O)
        env_O.set_enemy_agent(enemy_agent_X)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quantum TIQ-TAQ-TOE self-play PPO')
    argParser = argparse.ArgumentParser()
    
    argParser.add_argument("-N","--N", type=int, default=10, help="number of enemy change")
    argParser.add_argument("-f","--folder", type=str, default="models", help="folder to save models")
    argParser.add_argument("-p","--policy", type=str, default="MlpPolicy", choices=['MlpPolicy','MlpPolicyV2', 'CnnPolicy'], help="what type of policy to use")
    argParser.add_argument("-t","--timestemps", type=int, default=1000, help="timestemps to train the model during a single run")
    argParser.add_argument("-v","--verbose", type=int, default=1, help="verbose")
    argParser.add_argument("-d","--device", type=str, default='cpu', choices=['cpu', 'cuda'], help="device")
    argParser.add_argument("-l","--logs", type=str, default='logs', help="logs file")
    argParser.add_argument("-lr","--lr", type=float, default=0.0003, help="learning rate")
    argParser.add_argument("--linear", type=bool, default=False, help="linear lr")
    argParser.add_argument("-s","--no_steps", type=int, default=2048, help="no_steps")
    args = argParser.parse_args()
    torch.set_default_device(args.device)
    train_agents(args.N, args.folder, args.policy, args.timestemps, args.verbose, args.logs, args.lr, args.linear, args.no_steps)
        