import gymnasium as gym
from stable_baselines3 import DDPG

# env_name = "LunarLander-v2"
env_name = "BipedalWalker-v3"
env = gym.make(env_name,render_mode="human")
model = DDPG.load("./learning/model/BipedalWalker-v3.pkl")

state, _ = env.reset()
done = False 
score = 0
while not done:
    action, _ = model.predict(observation=state)
    state, reward, done, truncated, info  = env.step(action=action)
    score += reward
    env.render()
env.close()