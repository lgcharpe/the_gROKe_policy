from matplotlib import pyplot as plt
from pymgrid import Microgrid
from pymgrid.envs import DiscreteMicrogridEnv
from pymgrid.algos import RuleBasedControl
import gymnasium as gym
import ray
from ray.rllib.algorithms import ppo
from ray.tune.registry import register_env
from pymgrid.microgrid.utils.step import MicrogridStep
from pymgrid.modules.grid_module import GridModule
import numpy as np
import yaml
from pymgrid.microgrid import DEFAULT_HORIZON
from pymgrid.modules.base import BaseTimeSeriesMicrogridModule
from utills import RenewableModuleCustom, GridModuleCustom, GensetModuleDiscrete, MicrogridStepCustom, Microgrid2
from pymgrid.modules import (
    BatteryModule,
    LoadModule,
    RenewableModule,
    GridModule,
    GensetModule)

import pandas as pd
from ray.rllib.algorithms.dqn import DQNConfig
from ray import tune, train
from ray.tune.logger import pretty_print
from tqdm import tqdm
import logging
import argparse

NUM_BUCKETS = 10

def parse_args() -> argparse.Namespace:
    """
    Parse the command line arguments and return the parsed arguments.

    Returns:
        args (Namespace): The parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--random", action='store_true', help='Evaluate a random policy')
    parser.add_argument("-c", "--new_cost_function", action='store_true', help='Whether or not to use the redefined cost function from Q4')
    parser.add_argument("-n", "--number_of_houses", type=int, required=True, help='Number of houses to load consumer load from')
    args = parser.parse_args()
    return args

def cost_function(reward, info, cost_info):
    # Grid
    if "grid" in info:
        if "provided_energy" in info["grid"][0]:
            E_u = 0.25 * info["grid"][0]["provided_energy"]**2 * cost_info["grid"][0]["production_marginal_cost"] / 1000 +  0.5 * info["grid"][0]["provided_energy"] * cost_info["grid"][0]["production_marginal_cost"]
        else: 
            E_u = 0
        if "absorbed_energy" in info["grid"][0]:
            S_u = info["grid"][0]["absorbed_energy"] * cost_info["grid"][0]["absorption_marginal_cost"]
        else:
            S_u = 0
    else:
        E_u = 0
        S_u = 0

    # Solar
    if "solar_pv" in info:
        E_s = info["solar_pv"][0]["provided_energy"]
        C_s = cost_info["solar_pv"][0]["production_marginal_cost"]
    else:
        E_s = 0.0
        C_s = 0.0
        

    # Wind
    if "wind_turbine" in info:
        E_w = info["wind_turbine"][0]["provided_energy"]
        C_w = cost_info["wind_turbine"][0]["production_marginal_cost"]
    else:
        E_w = 0.0
        C_w = 0.0

    # Generator
    if "genset" in info:
        E_g = info["genset"][0]["provided_energy"]
        C_g = cost_info["genset"][0]["production_marginal_cost"]
    else:
        E_g = 0.0
        C_g = 0.0

    # Battery
    if "battery" in info:
        E_b = info["battery"][0]["absorbed_energy"] if "absorbed_energy" in info["battery"][0] else info["battery"][0]["provided_energy"]
        C_b = cost_info["battery"][0]["production_marginal_cost"]
    else:
        E_b = 0.0
        C_b = 0.0
    
    # Microgrid
    O_m = E_s * C_s + E_w * C_w + E_g * C_g + E_b * C_b

    reward = -(E_u + O_m - S_u) 

    return reward

def env_creator(env_context):
    return MicrogridEnv(microgrid, action_list)  # bad code using global variables but no idea how to do it differently for now....

class MicrogridEnv(gym.Env):
    def __init__(self, microgrid, action_list):
        self.action_space = gym.spaces.Discrete(len(action_list))
        self.observation_space = gym.spaces.Dict({
            module.name[0]: gym.spaces.Box(
                low=module.observation_space["normalized"].low,
                high=module.observation_space["normalized"].high,
                shape=module.observation_space["normalized"].shape,
                dtype=module.observation_space["normalized"].dtype
            )
        for module in microgrid.module_list})

        self._action_to_direction = {
            i: action for i, action in enumerate(action_list)
        }

        self.microgrid = microgrid
        
    def reset(self, seed=None, options=None):
        obs = self.microgrid.reset()
        #obs.pop("balance")
        #obs.pop("other")
        return obs, {}
        
    def step(self, action):
        action = self._action_to_direction[action]
        obs, reward, done, info = self.microgrid.run(action)
        return obs, reward, done, False, info
    

if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    data = pd.read_csv("./data/EnergyGenerationRenewable_round.csv")
    time_solar = data["Solar Generation"].values
    time_wind = data["Wind Generation"].values
    
    battery = BatteryModule(
    min_capacity=15,
    max_capacity=285,
    max_charge=2.5,
    max_discharge=2.5,
    efficiency=0.99,
    battery_cost_cycle=0.95,
    init_soc=0.1
    )
    gas_turbine_generator = GensetModuleDiscrete(
        running_min_production=0,
        running_max_production=600,
        genset_cost=0.55,
        num_buckets=NUM_BUCKETS
    )
    solar_pv = RenewableModuleCustom(
        time_series=time_solar,
        operating_cost=0.15
    )
    wind_turbine = RenewableModuleCustom(
        time_series=time_wind,
        operating_cost=0.085
    )

    buy_price = pd.read_csv("./data/rate_consumption_charge.csv")["Grid Elecricity Price（$/kWh）"].values
    sell_price = np.ones(len(buy_price)) * .2
    co2 = np.zeros(len(buy_price))

    time_grid = np.concatenate([buy_price[:, None], sell_price[:, None], co2[:, None]], axis=1)

    grid = GridModuleCustom(
        time_series=time_grid,
        max_export=10000,
        max_import=10000
    )


    time_load = pd.read_csv(f"./data/Load{args.number_of_houses}Households.csv")["load"].values[:8640]

    load = LoadModule(
        time_series=time_load
    )

    modules = [
        gas_turbine_generator,
        ("solar_pv", solar_pv),
        ("wind_turbine", wind_turbine),
        grid,
        load
    ]

    # Define the possible actions. In this case, solar + wind + generator
    renewable1 = np.tile(np.repeat([0, 0, 1, 1], 8), NUM_BUCKETS+1)
    renewable2 = np.tile(np.repeat([0, 1, 0, 1], 8), NUM_BUCKETS+1)
    genset = np.repeat(np.arange(NUM_BUCKETS+1), 32)
    g = [0, 1, 0, 1, 0, 1, 0, 1] * 4 * (NUM_BUCKETS+1)
    s = [0, 1, 0, 1, 1, 0, 1, 0] * 4 * (NUM_BUCKETS+1)
    w = [0, 1, 1, 0, 0, 1, 1, 0] * 4 * (NUM_BUCKETS+1)
    action_list = [{"genset": i, "solar_pv": j, "wind_turbine": k, "status": {"genset": l, "solar_pv": m, "wind_turbine": n}} for i, j, k, l, m, n in zip(genset, renewable1, renewable2, g, s, w)]

    if args.random:
        logging.info('Executing a random policy on the microgrid with load profiles from {args.number_of_houses} houses.')
        evaluation_grid_random = Microgrid2(modules=modules, add_unbalanced_module=False, reward_shaping_func=cost_function)
        actions = list()
        for _ in range(8640): #One year
            action = np.random.choice(action_list)
            actions.append(action)
            evaluation_grid_random.run(action)
        ax = evaluation_grid_random.log.loc[:, pd.IndexSlice[:, :, "reward"]].cumsum().plot()
        evaluation_grid_random.log.loc[:, pd.IndexSlice["balance", :, "shaped_reward"]].cumsum().plot(ax=ax)
        #plt.savefig("../figures/last_task_random.pdf", format="pdf", bbox_inches="tight")
        plt.show()
    else:
        logging.info('Executing a DQN policy on the microgrid with load profiles from {args.number_of_houses} houses.')
        ray.init()
        global microgrid
        if args.new_cost_function:
            microgrid = MicrogridEnv(Microgrid2(modules=modules, add_unbalanced_module=False), action_list)
        else:
            microgrid = MicrogridEnv(Microgrid2(modules=modules, add_unbalanced_module=False, reward_shaping_func=cost_function), action_list)

        register_env("my_env", env_creator)
        exploration_config = {'type': 'EpsilonGreedy',
                            'initial_epsilon': 1.0,
                            'final_epsilon': 0.0,
                            'epsilon_timesteps': 30000}
        logging.info(f'Hyperparameters: {exploration_config}, lr= {1e-4}, gamma={0.9}, epochs={5}')
        config = DQNConfig().training(gamma=0.9, lr=1e-4).exploration(explore=False, exploration_config=exploration_config).environment("my_env")
        algo = config.build()
        epochs = 5
        logging.info('Training...')
        for i in tqdm(range((8640//1000)*epochs)):
            result = algo.train()

        env = MicrogridEnv(microgrid, action_list)
        
        episode_reward = 0
        obs = microgrid.reset()
        terminated = False
        truncated = False
        actions = []
        while not terminated and not truncated:
            action = algo.compute_single_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            actions.append(action)
        
        evaluation_grid_dqn = Microgrid2(modules=modules, add_unbalanced_module=False, reward_shaping_func=cost_function)
        for ac in actions:
            evaluation_grid_dqn.run(action_list[ac])
        
        ax = evaluation_grid_dqn.log.loc[:, pd.IndexSlice[:, :, "reward"]].cumsum().plot()
        evaluation_grid_dqn.log.loc[:, pd.IndexSlice["balance", :, "shaped_reward"]].cumsum().plot(ax=ax)
        #plt.savefig("../figures/shaped_reward.pdf", format="pdf", bbox_inches="tight")
        plt.show()
                        


        
