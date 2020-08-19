"""Decentralized single and multi-agent environment for networks with traffic lights.

These environments are used to train traffic lights to regulate traffic flow
through an n x m traffic light grid.

Note: Decentralized Environment:
                            returns dict of each observation of each agents (see get_state method)
                            returns dict of each reward value of each agents(see compute_reward method)
                            the observation space is (observation space for single agent) defined locally for each
                            Action space is locally for each agents
"""

from flow.envs.multiagent.traffic_light_grid import MultiTrafficLightGridPOEnv
from flow.envs.centralized_env import CentralizedGridEnv
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from flow.core.traffic_light_utils import log_rewards, log_travel_times, get_training_iter, execute_action
import pandas as pd
import os
import numpy as np

# index to split traffic lights string eg. "center0".split("center")[ID_IDX] = 0
ID_IDX = 1


class DeCentralizedGridEnv(CentralizedGridEnv, MultiTrafficLightGridPOEnv):

    """ Inherited for Baseline Implementation

    CentralizedGridEnv is the immediate class inherited from and handles primarily the __init__ method

    MultiTrafficLightGridPOEnv contains special methods initializing decentralized properties
        (ie. dict for state, reards, actions etc)

    Decentralized Centralized model version of TrafficLightGridPOEnv.

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

    def step(self, rl_actions):

        """Advance the environment by one step.

        See parent class

        Parameters
        ----------
        rl_actions : dict
            an dict of rl_ids as keys and actions as values provided by the rl algorithm

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
        next_observation, reward, done, infos = MultiTrafficLightGridPOEnv.step(self, rl_actions)

        # log average reward and average travel times if simulation is over
        self.rew_list.append(sum(reward.values()))

        # log reward during simulation
        if self.benchmark_params.log_rewards_during_iteration:
            log_rewards(self.rew_list, rl_actions, self.benchmark_params, 0, self.step_counter, during_simulation=True)

        if done["__all__"]:
            # current training iteration
            iter_ = get_training_iter(self.benchmark_params.full_path)
            # log average travel time
            log_travel_times(rl_actions, iter_, self.benchmark_params, self.network, self.sim_params, self.step_counter)
            # log average reward
            log_rewards(self.rew_list, rl_actions, self.benchmark_params, iter_, self.step_counter)

        return next_observation, reward, done, infos

    def get_state(self):
        """Return the state of the simulation as perceived by the RL agent.
           Calls benchmark object to compute state for each traffic light

        Returns
        -------
        observation : dict
            information on the state of each traffic light, which is provided to the
            agent
        """
        observation = {}

        # collect states/observations for for each rl_id
        for rl_id in self.k.traffic_light.get_ids():
            observation[rl_id] = self.benchmark.get_state(kernel=self.k,
                                                          network=self.network,
                                                          _get_relative_node=self._get_relative_node,
                                                          direction=self.direction,
                                                          step_counter=self.step_counter,
                                                          rl_id=rl_id)

        return observation

    def reset(self, new_inflow_rate=None):
        """ Resets the experiment

            see parent class: MultiTrafficLightGridPOEnv
        """

        return MultiTrafficLightGridPOEnv.reset(self, new_inflow_rate)

    def clip_actions(self, rl_actions=None):
        """ Clip the actions passed from the RL agent.

            see parent class: MultiTrafficLightGridPOEnv
         """

        return MultiTrafficLightGridPOEnv.clip_actions(self, rl_actions)

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
           reward : dict
                reward values for each traffic light
           """

        # if no rl_action, SUMO default (either actuated or fixed) is executed,
        # non reward required.
        if rl_actions is None:
            return {}
        rews = {}

        # collect required iterable for rl_actions and rl_ids
        rl_ids = self.k.traffic_light.get_ids()
        if not self.action_dict:
            rl_id_action_dict = rl_actions.items()

        else:
            actions = self.action_dict[rl_actions]
            rl_id_action_dict = zip(rl_ids, actions)

        # compute rewards for each rl_id
        for rl_id, rl_action in rl_id_action_dict:
            rews[rl_id] = self.benchmark.compute_reward(self.step_counter, rl_id)

        return rews

    def _apply_rl_actions(self, rl_actions):
        """Specify the actions to be performed by the rl agent(s).

        If no actions are provided at any given step, the rl agents default to
        performing actions specified by SUMO.

        Parameters
        ----------
        rl_actions : dict
            dictionary in which the keys correspond to rl_ids/traffic lights
            and the values correspond to either 0  or 1 (to not switch or switch respectively
        """

        # flag to activate sumo actuate baselines or not to
        if self.benchmark_params.sumo_actuated_baseline:
            if not os.path.isfile(self.benchmark_params.full_path):
                return
            else:
                # read csv
                df = pd.read_csv(self.benchmark_params.full_path, index_col=False)
                n_iter = df.training_iteration.iat[-1]

            if n_iter < self.benchmark_params.sumo_actuated_simulations:
                return

        for rl_id, rl_action in rl_actions.items():
            i = int(rl_id.split("center")[ID_IDX])
            execute_action(self, i, rl_action)


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
            shape=(self.benchmark.obs_shape_func(),),
            dtype=np.float32)
        return tl_box

    @property
    def action_space(self):
        """Identify the dimensions and bounds of the action space.
          For each single light: 0, 1
                              to not switch or to switch traffic light respectively

        Returns
        -------
        gym.spaces.Discrete object
            contains shape and bounds of action space characterized
        """
        return Discrete(2)
