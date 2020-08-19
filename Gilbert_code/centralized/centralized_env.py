"""Centralized single and multi-agent environment for networks with traffic lights.

These environments are used to train traffic lights to regulate traffic flow
through an n x m traffic light grid.

Note: Centralized Environment:
                            concatenates all observations of all agents into one array (see get_state method)
                            sums  all reward values of all agents(see compute_reward method)
                            the observation space is (observation space for single agent * number of agents)
                            Action space is defined for all agents
"""

import numpy as np
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from flow.core.traffic_light_utils import log_rewards, log_travel_times, get_training_iter, execute_action
from flow.envs.traffic_light_grid import TrafficLightGridPOEnv
import pandas as pd
import os
from flow.envs import thesis
from flow.envs import presslight
from flow.core import benchmark_params
import itertools

# index to split traffic lights string eg. "center0".split("center")[ID_IDX] = 0
ID_IDX = 1

modules = [presslight, thesis]

# These sample parameters can be imported in the simulation config files as
# from flow.envs.centralized_env import PRESSURE_SAMPLE_PARAMS, THESIS_SAMPLE_PARAMS

PRESSURE_SAMPLE_PARAMS = {
    # minimum switch time for each traffic light (in seconds)
    "switch_time": 2.0,
    # whether the traffic lights should be actuated by sumo or RL
    # options are "controlled" and "actuated"
    "tl_type": "actuated",
    # determines whether the action space is meant to be discrete or continuous
    "discrete": True,
    "target_velocity": 11,
    "yellow_phase_duration": 4,
    "num_observed": 2,
    "num_local_edges": 4,
    "num_local_lights": 4,
    "benchmark": "PressureLightGridEnv",  # This should be the string name of the benchmark class
    "benchmark_params": "BenchmarkParams"
}

THESIS_SAMPLE_PARAMS = {
    # minimum switch time for each traffic light (in seconds)
    "switch_time": 2.0,
    # whether the traffic lights should be actuated by sumo or RL
    # options are "controlled" and "actuated"
    "tl_type": "actuated",
    # determines whether the action space is meant to be discrete or continuous
    "discrete": True,
    "target_velocity": 11,
    "yellow_phase_duration": 4,
    "num_observed": 2,
    "num_local_edges": 4,
    "num_local_lights": 4,
    "benchmark": "ThesisLightGridEnv",  # This should be the string name of the benchmark class
    "benchmark_params": "BenchmarkParams"
}


class CentralizedGridEnv(TrafficLightGridPOEnv):

    """ Inherited for Baseline Implementation

    Centralized model version of TrafficLightGridPOEnv.

    Required from env_params: See parent class

    States
        See parent class

    Actions
        See parent class

    Rewards
        See parent class

    Termination
        See parent class
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        """Initialized centralized environment with benchmark string parameters"""

        # obtain the class objects (parameters and experiments) from the defined benchmark string names
        self.benchmark_params = getattr(benchmark_params, env_params.additional_params["benchmark_params"])()
        for mod in modules:
            try:
                self.benchmark = getattr(mod, env_params.additional_params["benchmark"])(self.benchmark_params)
                break
            except AttributeError:
                continue

        self.action_dict = dict()
        self.rew_list = []
        self.yellow_phase_duration = 4 # env_params.additional_params["yellow_phase_duration"]

    def compute_reward(self, rl_actions, **kwargs):

        """Reward function for the RL agent(s).
          Calls benchmark object to compute reward for each traffic light

        Parameters
        ----------
        rl_actions : array_like
            actions performed by rl agents
        kwargs : dict
            other parameters of interest. Contains a "fail" element, which
            is True if a vehicle crashed, and False otherwise

        Returns
        -------
        reward : float
        """
        rl_ids = self.k.traffic_light.get_ids()
        reward = {}

        # if no rl_action, SUMO default (either actuated or fixed) is executed. reward is computed.
        if rl_actions is None:
            for rl_id in rl_ids:
                reward[rl_id] = self.benchmark.compute_reward(self.step_counter, rl_id)

            # add all rewards for each traffic light
            final_reward = sum(list(reward.values()))
            return final_reward

        # collect required iterable for rl_actions and rl_ids
        if not self.action_dict:
            rl_id_action_dict = rl_actions.items()

        else:
            actions = self.action_dict[rl_actions]
            rl_id_action_dict = zip(rl_ids, actions)

        # compute rewards for each rl_id
        for rl_id, rl_action in rl_id_action_dict:
            reward[rl_id] = self.benchmark.compute_reward(self.step_counter, rl_id)

        # add all rewards for each traffic light
        final_reward = sum(list(reward.values()))
        return final_reward

    def get_state(self):
        """Return the state of the simulation as perceived by the RL agent.
           Calls benchmark object to compute state for each traffic light

        Returns
        -------
        final_obs : array_like
            information on the state of the traffic lights, which is provided to the
            agent
        """

        observations = {}

        # collect states/observations for for each rl_id
        for rl_id in self.k.traffic_light.get_ids():
            observations[rl_id] = self.benchmark.get_state(kernel=self.k,
                                                           network=self.network,
                                                           _get_relative_node=self._get_relative_node,
                                                           direction=self.direction,
                                                           step_counter=self.step_counter,
                                                           rl_id=rl_id)

        # concatenate all states for each traffic light
        final_obs = np.concatenate(list((observations.values())))
        return final_obs

    @property
    def observation_space(self):
        """Identify the dimensions and bounds of the observation space.

        Returns:
        ---------
        gym.spaces.box object
            contains shape and bounds of observation space characterized
            by the benchmark and number of traffic lights
        """

        tl_box = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.benchmark.obs_shape_func() * self.num_traffic_lights,),
            dtype=np.float32)
        return tl_box

    @property
    def action_space(self):
        """Identify the dimensions and bounds of the action space.
          Actions characterized using a dict that map agent choices to traffic light actions:
            ie. for single light: 0, 1
                                 to not switch or to switch traffic light respectively
                                 {0: (0), 1:(1)}
                for multi light: 0, 1, 2, 3, 4 ...
                                 agents values corresponding to action for each traffic light
                                 example for 3 lights: {1: (1,0,0), 2:(0,1,0) ..

        Returns
        -------
        gym.spaces.Discrete object
            contains shape and bounds of action space characterized

        """

        # get all combinations of actions [(1,0,0), (0,1,0)...
        lst = list(itertools.product([0, 1], repeat=self.num_traffic_lights))

        # create dict mapping agents actions to list {1: (1,0,0), 2:(0,1,0) ..
        for i in np.arange(len(lst)):
            self.action_dict.update({i: lst[i]})

        return Discrete(len(lst))

    def step(self, rl_actions):
        """Advance the environment by one step.

        See parent class

        Parameters
        ----------
        rl_actions : array_like
            action provided by the rl algorithm
            Note: self.action_dict =  {rl_actions: (action_for each_traffic_light)}

        Returns
        -------
        observation : array_like
            agent's observation of the current environment
        reward : float
            amount of reward associated with the previous state/action pair
        done : bool
            indicates whether the episode has ended
        info : dict
            contains other diagnostic information from the previous action
        """

        # advance simulation
        next_observation, reward, done, infos = super().step(rl_actions)

        # log average reward and average travel times if simulation is over
        self.rew_list.append(reward)

        # log reward during simulation
        if self.benchmark_params.log_rewards_during_iteration:
            log_rewards(self.rew_list, rl_actions, self.benchmark_params, 0, self.step_counter, during_simulation=True)

        if done:
            # current training iteration
            iter_ = get_training_iter(self.benchmark_params.full_path)
            # log average travel time
            log_travel_times(rl_actions, iter_, self.benchmark_params, self.network, self.sim_params, self.step_counter)
            # log average reward
            log_rewards(self.rew_list, rl_actions, self.benchmark_params, iter_, self.step_counter)

        return next_observation, reward, done, infos

    def _apply_rl_actions(self, rl_actions):
        """Specify the actions to be performed by the rl agent(s).

        If no actions are provided at any given step, the rl agents default to
        performing actions specified by SUMO.

        Parameters
        ----------
        rl_actions : int
            actions provided by the RL algorithm
            Note: self.action_dict =  {rl_actions: (action_for each_traffic_light)}
            ie. for single light: 0, 1
                                 switch or not to switch traffic light respectively
                                 {0: (0), 1:(1)}
                for multi light: 0, 1, 2, 3, 4 ...
                                 agents values corresponding to action for each traffic light
                                 example for 3 lights: {1: (1,0,0), 2:(0,1,0) ..

        """
        # flag to activate sumo actuate baselines or not to
        if self.benchmark_params.sumo_actuated_baseline:

            # check file to track number of simulations completed during training.
            if not os.path.isfile(self.benchmark_params.full_path):
                return
            else:
                # read csv file if it exists
                df = pd.read_csv(self.benchmark_params.full_path, index_col=False)
                n_iter = df.training_iteration.iat[-1]

            if n_iter < self.benchmark_params.sumo_actuated_simulations:
                return

        rl_ids = np.arange(self.num_traffic_lights)
        actions = self.action_dict[rl_actions]

        for i, rl_action in zip(rl_ids, actions):
            execute_action(self, i, rl_action)
