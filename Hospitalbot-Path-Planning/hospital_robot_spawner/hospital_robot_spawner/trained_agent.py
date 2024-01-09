import rclpy
from rclpy.node import Node
from gymnasium.envs.registration import register
from hospital_robot_spawner.hospitalbot_env import HospitalBotEnv
from hospital_robot_spawner.hospitalbot_simplified_env import HospitalBotSimpleEnv
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os
import optuna
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

class TrainingNode(Node):

    def __init__(self):
        super().__init__("hospitalbot_training", allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self._training_mode = "training"


def main(args=None):
    rclpy.init()
    node = TrainingNode()
    node.get_logger().info("Training node has been created")

    home_dir = os.path.expanduser('~')
    pkg_dir = 'ros2_ws/src/Hospitalbot-Path-Planning/hospital_robot_spawner'
    trained_models_dir = os.path.join(home_dir, pkg_dir, 'rl_models')
    log_dir = os.path.join(home_dir, pkg_dir, 'logs_latest_A2C1')
    
    if not os.path.exists(trained_models_dir):
        os.makedirs(trained_models_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    register(
        id="HospitalBotEnv-v0",
        entry_point="hospital_robot_spawner.hospitalbot_env:HospitalBotEnv",
        max_episode_steps=300,
    )

    node.get_logger().info("The environment has been registered")

    env = gym.make('HospitalBotEnv-v0')
    env = Monitor(env)

    check_env(env)
    node.get_logger().info("Environment check finished")

    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=900, verbose=1)
    eval_callback = EvalCallback(env, callback_on_new_best=stop_callback, eval_freq=100000, best_model_save_path=trained_models_dir, n_eval_episodes=40)
    
    if node._training_mode == "random_agent":
        episodes = 10
        node.get_logger().info("Starting the RANDOM AGENT now")
        for ep in range(episodes):
            obs = env.reset()
            done = False
            while not done:
                obs, reward, done, truncated, info = env.step(env.action_space.sample())
                node.get_logger().info("Agent state: [" + str(info["distance"]) + ", " + str(info["angle"]) + "]")
                node.get_logger().info("Reward at step " + ": " + str(reward))
    
    elif node._training_mode == "training":
        model = A2C("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir, n_steps=20480, gamma=0.9880614935504514, gae_lambda=0.9435887928788405, ent_coef=0.00009689939917928778, vf_coef=0.6330533453055319, learning_rate=0.00001177011863371444)
        try:
            model.learn(total_timesteps=int(40000000), reset_num_timesteps=False, callback=eval_callback, tb_log_name="A2C_m2_r7")
        except KeyboardInterrupt:
            model.save(f"{trained_models_dir}/A2C_MP21")
        model.save(f"{trained_models_dir}/A2C_MP21")
    
    elif node._training_mode == "retraining":
        node.get_logger().info("Retraining an existent model")
        trained_model_path = os.path.join(home_dir, pkg_dir, 'rl_models', 'PPO_risk_seeker.zip')
        custom_obj = {'action_space': env.action_space, 'observation_space': env.observation_space}
        model = A2C.load(trained_model_path, env=env, custom_objects=custom_obj)
        
        try:
            model.learn(total_timesteps=int(20000000), reset_num_timesteps=False, callback=eval_callback, tb_log_name="A2C_risk_seeker")
        except KeyboardInterrupt:
            model.save(f"{trained_models_dir}/A2C_risk_seeker_21")
        model.save(f"{trained_models_dir}/A2C_risk_seeker_21")

    elif node._training_mode == "hyperparam_tuning":
        env.close()
        del env
        study = optuna.create_study(direction='maximize')
        study.optimize(optimize_agent, n_trials=10, n_jobs=1)
        node.get_logger().info("Best Hyperparameters: " + str(study.best_params))

    node.get_logger().info("The training is finished, now the node is destroyed")
    node.destroy_node()
    rclpy.shutdown()

def optimize_ppo(trial):
    return {
        'n_steps': trial.suggest_int('n_steps', 2048, 8192),
        'gamma': trial.suggest_loguniform('gamma', 0.8, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-6, 1e-3),
        'clip_range': trial.suggest_uniform('clip_range', 0.1, 0.4),
        'gae_lambda': trial.suggest_uniform('gae_lambda', 0.8, 0.99),
        'ent_coef': trial.suggest_loguniform('ent_coef', 0.00000001, 0.1),
        'vf_coef': trial.suggest_uniform('vf_coef', 0, 1),
    }

def optimize_ppo_refinement(trial):
    return {
        'n_steps': trial.suggest_int('n_steps', 2048, 14336),
        'gamma': trial.suggest_loguniform('gamma', 0.96, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 9e-4),
        'clip_range': trial.suggest_uniform('clip_range', 0.15, 0.37),
        'gae_lambda': trial.suggest_uniform('gae_lambda', 0.94, 0.99),
        'ent_coef': trial.suggest_loguniform('ent_coef', 0.00000001, 0.00001),
        'vf_coef': trial.suggest_uniform('vf_coef', 0.55, 0.65),
    }

def optimize_agent(trial):
    try:
        env_opt = gym.make('HospitalBotEnv-v0')
        HOME_DIR = os.path.expanduser('~')
        PKG_DIR = 'ros2_ws/src/Hospitalbot-Path-Planning/hospital_robot_spawner'
        LOG_DIR = os.path.join(HOME_DIR, PKG_DIR, 'logs')
        SAVE_PATH = os.path.join(HOME_DIR, PKG_DIR, 'tuning', 'trial_{}'.format(trial.number))
        model_params = optimize_ppo_refinement(trial)
        model = A2C("MultiInputPolicy", env_opt, tensorboard_log=LOG_DIR, verbose=0, **model_params)
        model.learn(total_timesteps=150000)
        mean_reward, _ = evaluate_policy(model, env_opt, n_eval_episodes=20)
        env_opt.close()
        del env_opt
        model.save(SAVE_PATH)
        return mean_reward

    except Exception as e:
        return -10000

if __name__ == "__main__":
    main()








# #!usr/bin/env python3

# import rclpy
# from rclpy.node import Node
# from gymnasium.envs.registration import register
# from hospital_robot_spawner.hospitalbot_env import HospitalBotEnv
# from hospital_robot_spawner.hospitalbot_simplified_env import HospitalBotSimpleEnv
# import gymnasium as gym
# from stable_baselines3 import PPO
# from stable_baselines3 import A2C
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.env_checker import check_env
# import os
# import numpy as np

# class TrainedAgent(Node):

#     def __init__(self):
#         super().__init__("trained_hospitalbot", allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

# def main(args=None):
#     rclpy.init()
#     node = TrainedAgent()
#     node.get_logger().info("Trained agent node has been created")

#     # We get the dir where the models are saved
#     home_dir = os.path.expanduser('~')
#     pkg_dir = 'ros2_ws/src/Hospitalbot-Path-Planning/hospital_robot_spawner'
#     trained_model_path = os.path.join(home_dir, pkg_dir, 'rl_models', 'PPO_risk_seeker.zip')

#     # Register the gym environment
#     register(
#         id="HospitalBotEnv-v0",
#         entry_point="hospital_robot_spawner.hospitalbot_env:HospitalBotEnv",
#         #entry_point="hospital_robot_spawner.hospitalbot_simplified_env:HospitalBotSimpleEnv",
#         max_episode_steps=3000,
#     )

#     env = gym.make('HospitalBotEnv-v0')
#     env = Monitor(env)

#     check_env(env)

#     episodes = 10

#     # This is done to bypass the problem between using two different distros of ROS (humble and foxy)
#     # They use different python versions, for this reason the action and observation space cannot be deserialized from the trained model
#     # The solution is passing them as custom_objects, so that they won't be loaded from the model
#     custom_obj = {'action_space': env.action_space, 'observation_space': env.observation_space}

#     # Here we load the rained model
#     model = PPO.load(trained_model_path, env=env, custom_objects=custom_obj)

#     # Evaluating the trained agent
#     Mean_ep_rew, Num_steps = evaluate_policy(model, env=env, n_eval_episodes=100, return_episode_rewards=True, deterministic=True)

#     # Print harvested data
#     node.get_logger().info("Mean Reward: " + str(np.mean(Mean_ep_rew)) + " - Std Reward: " + str(np.std(Mean_ep_rew)))
#     node.get_logger().info("Max Reward: " + str(np.max(Mean_ep_rew)) + " - Min Reward: " + str(np.min(Mean_ep_rew)))
#     node.get_logger().info("Mean episode length: " + str(np.mean(Num_steps)))

#     # Close env to print harvested info and destroy the hospitalbot node
#     env.close()

#     node.get_logger().info("The script is completed, now the node is destroyed")
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == "__main__":
#     main()