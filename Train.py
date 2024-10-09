## Importing Libraries 
import numpy as np
import subprocess
from stable_baselines3 import SAC, PPO, TD3, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv
from datetime import datetime
from GymEnv_noRobot import GymEnv
#from CustomCallback import StateActionRewardHistory



# Decorating function to count how many times the function is called
def count_calls(func):
    def wrapper(*args, **kwargs):
        wrapper.call_count += 1
        port = 23001 + wrapper.call_count
        #print(f"Function '{func.__name__}' has been called {wrapper.call_count} times.")
        return func(port, *args, **kwargs)
    wrapper.call_count = 0
    return wrapper

# Retrieving date & time
date = datetime.now().strftime("%Y%m%d-%I:%M%p")

numberEnv = 4
eval_freq = max( 100 // numberEnv, 1)
subprocess.run(["/home/major/Desktop/Simulation/Main/./CsRun.sh",f"{numberEnv}"])
# callable function to create the environment
@count_calls
def make_my_env(port):
    #subprocess.run(["bash", "CsRun.sh", str(port)])
    env = GymEnv(port, imageObservation = False)
    return env

# Create & Wrap env
env = GymEnv(port = 23001,imageObservation = False)
distanceThreshold = env.distanceThreshold
multienv = make_vec_env(lambda:make_my_env(), n_envs=numberEnv)


# Noise 
n_actions = env.action_space.shape[-1]
normal_action_noise = NormalActionNoise(mean = np.zeros(n_actions), sigma = 0.1 * np.ones(n_actions))
Orn_action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma = 0.1 * np.ones(n_actions),theta=0.15, dt=0.01)

# Callbacks
evalenv = Monitor(env)
Name = f"{date}_TD3_{distanceThreshold}"
stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=50, verbose=1)
evalcallback = EvalCallback(evalenv, best_model_save_path = f"./models/{Name}", log_path = f"./logs/{Name}",
                          eval_freq = 1000, deterministic = True, render = False,callback_after_eval=stop_train_callback)
#callback = CallbackList([evalcallback,stop_train_callback])

# customCallback = StateActionRewardHistory(env=env)

# # Model
model = TD3("MlpPolicy",multienv , verbose= 1, batch_size = 512,
            tensorboard_log=f"./tensorboard/{Name}",train_freq = numberEnv,gradient_steps=2,action_noise= normal_action_noise)#, action_noise= normal_action_noise)

# model = SAC("MlpPolicy",env , verbose= True, batch_size = 512,
#             tensorboard_log=f"./tensorboard/{Name}", action_noise= normal_action_noise)

# model = SAC("MlpPolicy", env, verbose= True, action_noise = action_noise)
# model = SAC("MlpPolicy", env)

# Learning
model.learn(total_timesteps = 10e6,log_interval = 10,callback=evalcallback)
# Saving
model.save(f"./tensorboard/{Name}/model")

# Closing the env
env.close()

# ## Saving State Action Reward
# customCallback.save(date)