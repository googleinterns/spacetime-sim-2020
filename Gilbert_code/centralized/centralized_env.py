"""Multi-agent environments for networks with traffic lights.

These environments are used to train traffic lights to regulate traffic flow
through an n x m traffic light grid.
"""

import numpy as np
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from flow.core.traffic_light_utils import log_rewards, log_travel_times, get_training_iter
from flow.envs.traffic_light_grid import TrafficLightGridPOEnv
import pandas as pd
import os

ADDITIONAL_ENV_PARAMS = {
    # minimum switch time for each traffic light (in seconds)
    "switch_time": 2.0,
    # whether the traffic lights should be actuated by sumo or RL
    # options are "controlled" and "actuated"
    "tl_type": "controlled",
    # determines whether the action space is meant to be discrete or continuous
    "discrete": False,
}

ADDITIONAL_PO_ENV_PARAMS = {
    # num of vehicles the agent can observe on each incoming edge
    "num_observed": 2,
    # velocity to use in reward functions
    "target_velocity": 30,
}

# Index for retrieving ID when splitting node name, e.g. ":center#"
ID_IDX = 1


class MultiTrafficLightGridPOEnvTH(TrafficLightGridPOEnv):

    """ Inherited for PressLight baseline Implementation

    Multiagent shared model version of TrafficLightGridPOEnv.

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

        self.rew_list = []
        self.benchmark_params = env_params.additional_params["benchmark_params"]()
        self.benchmark = env_params.additional_params["benchmark"](self.benchmark_params)

    def compute_reward(self, rl_actions, **kwargs):

        return self.benchmark.compute_reward(rl_actions, self.step_counter, **kwargs)

    def get_state(self):

        return self.benchmark.get_state(kernel=self.k,
                                        network=self.network,
                                        _get_relative_node=self._get_relative_node,
                                        direction=self.direction,
                                        currently_yellow=self.currently_yellow,
                                        step_counter=self.step_counter)

    @property
    def observation_space(self):
        """State space that is partially observed.

        Velocities, distance to intersections, edge number (for nearby
        vehicles) from each direction, local edge information, and traffic
        light state.
        """
        tl_box = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.benchmark.obs_shape(),),
            dtype=np.float32)
        return tl_box

    @property
    def action_space(self):
        """See class definition."""
        return Discrete(2 * self.num_traffic_lights)

    def step(self, rl_actions):
        """Advance the environment by one step.

        See parent class

        Parameters
        ----------
        rl_actions : array_like
            an list of actions provided by the rl algorithm

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
        self.rew_list += [reward]
        if done:
            # current training iteration
            iter_ = get_training_iter(self.benchmark_params.full_path)
            # log average travel time
            log_travel_times(rl_actions, iter_, self.benchmark_params, self.network, self.sim_params, self.step_counter)
            # log average reward
            log_rewards(self.rew_list, rl_actions, self.benchmark_params, iter_, self.step_counter)

        return next_observation, reward, done, infos

    def _apply_rl_actions(self, rl_actions):
        """
        TODO: Test for multi-centralized vs single centralized
        See parent class.

        Issues action for each traffic light agent.
        """
        if self.benchmark_params.sumo_actuated_baseline:
            # return
            if not os.path.isfile(self.benchmark_params.full_path):
                return
            else:
                # read csv
                df = pd.read_csv(self.benchmark_params.full_path, index_col=False)
                n_iter = df.training_iteration.iat[-1]

            if n_iter < 6:
                return
        i = 0
        for rl_action in rl_actions:

            if self.discrete:
                action = rl_action
            else:
                # convert values less than 0.0 to zero and above to 1. 0's
                # indicate that we should not switch the direction
                action = rl_action > 0.0

            if self.currently_yellow[i] == 1:  # currently yellow
                self.last_change[i] += self.sim_step
                # Check if our timer has exceeded the yellow phase, meaning it
                # should switch to red
                if self.last_change[i] >= self.min_switch_time:
                    if self.direction[i] == 0:
                        self.k.traffic_light.set_state(
                            node_id='center{}'.format(i), state="GrGr")
                    else:
                        self.k.traffic_light.set_state(
                            node_id='center{}'.format(i), state='rGrG')
                    self.currently_yellow[i] = 0
            else:
                if action:
                    if self.direction[i] == 0:
                        self.k.traffic_light.set_state(
                            node_id='center{}'.format(i), state='yryr')
                    else:
                        self.k.traffic_light.set_state(
                            node_id='center{}'.format(i), state='ryry')
                    self.last_change[i] = 0.0
                    self.direction[i] = not self.direction[i]
                    self.currently_yellow[i] = 1
            i += 1

