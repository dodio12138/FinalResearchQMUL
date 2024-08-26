import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env_name = "LunarLander-v2"
# env_name = "Acrobot-v1"
env = gym.make(env_name)
env = DummyVecEnv([lambda : env])
model = DQN(
    "MlpPolicy", 
    env=env, 
    learning_rate = 6.3e-4,
    batch_size = 128,
    buffer_size = 50000,
    learning_starts = 0,
    gamma = 0.99,
    target_update_interval = 250,
    train_freq=4,
    gradient_steps= -1,
    exploration_fraction= 0.12,
    exploration_final_eps= 0.1,
    policy_kwargs={"net_arch" : [256, 256]},
    verbose=1,
    tensorboard_log="./learning/tensorboard/LunarLander-v2/"
)

model.learn(total_timesteps=1e5)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
env.close()
print(mean_reward, std_reward)

model.save("./learning/model/LunarLander3.pkl")