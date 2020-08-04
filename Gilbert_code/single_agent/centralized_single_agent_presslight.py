"""Environments for networks with traffic lights.

These environments are used to train traffic lights to regulate traffic flow
through an n x m traffic light grid.
"""

from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from flow.envs.traffic_light_grid import TrafficLightGridPOEnv
from tensorboardX import SummaryWriter
import numpy as np
import os
from datetime import datetime
from flow.core.traffic_light_utils import trip_info_emission_to_csv
import pandas as pd
import time


#######################################################
# ######### Gilbert code below for PressLight ####### #
#######################################################
now = datetime.now()
current_time = now.strftime("%Y-%H-%M-%S")
home_dir = os.path.expanduser('~')
if not os.path.exists(home_dir + '/ray_results/real_time_metrics'):
    os.makedirs(home_dir + '/ray_results/real_time_metrics')

summaries_dir = home_dir + '/ray_results/real_time_metrics'
log_rewards_during_iteration = False

# choose look-ahead distance
# look_ahead = 80
# look_ahead = 160
# look_ahead = 240
look_ahead = 43

# choose demand pattern
# demand = "L_analysis32x32x32"
demand = "H_learning_rate_0.01_exp_0.5_16x16x16_"

# choose exp running
exp = "rl"
# exp = "non_rl"

# log title for tensorboard
log_title = '/simulation_1x1_{}_{}_{}'.format(exp, look_ahead, demand)
filename = "/iterations_{}_{}.csv".format(look_ahead, demand)
root_dir = os.path.expanduser('~')
file_location = root_dir + '/ray_results/grid1x3_learning_rate_0.01'
full_path = file_location + filename

RED = (255, 0, 0)
BLUE = (0, 0, 255)
CYAN = (0, 255, 255)

#######################################################
# ######### Gilbert code above for PressLight ####### #
#######################################################

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


# class PressLightSingleEnv(TrafficLightGridPOEnv):
class MyGridEnv(TrafficLightGridPOEnv):
    """See parent class.
`   Inherited for PressLight baseline Implementation
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        # check whether the action space is meant to be discrete or continuous
        # self.discrete = env_params.additional_params.get("discrete", False)
        self.discrete = True

        # set up Tensorboard logging info
        self.exp_name = log_title
        self.writer = SummaryWriter(summaries_dir + self.exp_name)

        # set look ahead distance
        self.look_ahead = look_ahead

        # initialize edge pressure and reward list
        self.edge_pressure = None
        self.rew_list = []

        self.obs_shape = 14 * self.num_traffic_lights

    def get_state(self):
        """See parent class.

        Returns edge pressures of an intersection, edge_numbers,
         traffic light state. This is partially observed
        """

        self.edge_pressure = []
        edge_number = []
        observed_ids_ahead = []
        observed_ids_behind = []

        # collect edge pressures for single intersection
        for rl_id, edges in self.network.node_mapping:
            incoming_edges = edges
            for edge_ in incoming_edges:
                # for each incoming edge, log the incoming and outgoing vehicle ids
                if self.k.network.rts[edge_]:
                    index_ = self.k.network.rts[edge_][0][0].index(edge_)
                    observed_ids_ahead = \
                        self.get_id_within_dist(edge_, direction="ahead")
                    outgoing_lane = self.k.network.rts[edge_][0][0][index_+1]
                    observed_ids_behind = self.get_id_within_dist(outgoing_lane, direction="behind")

                # color vehicles
                self.color_vehicles(observed_ids_ahead, CYAN)
                self.color_vehicles(observed_ids_behind, RED)
                # compute pressure
                self.edge_pressure += [len(observed_ids_ahead) - len(observed_ids_behind)]
                # print(self.edge_pressure )

            # assigning unique number id for each incoming edge
            for edge in edges:
                edge_number += \
                    [self._convert_edge(edge) /
                     (self.k.network.network.num_edges - 1)
                     ]

        light_states = self.k.traffic_light.get_state(rl_id)
        if light_states == "GrGr":
            # green state
            light_states_ = [1]
        elif light_states == ["yryr"]:
            # yellow state
            light_states_ = [0.6]
        else:
            # all other states are red
            light_states_ = [0.2]

        observation = np.array(np.concatenate(
            [self.edge_pressure,
             edge_number,
             self.direction.flatten().tolist(),
             light_states_
             ]))

        return observation

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""

        rew = -sum(self.edge_pressure)
        if log_rewards_during_iteration:
            # write current reward to tensorboard (during simulation)
            self.log_rewards(rew, rl_actions, during_simulation=True)
        return rew

    @property
    def observation_space(self):
        """See class definition."""

        tl_box = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_shape,),
            dtype=np.float32)
        return tl_box

    @property
    def action_space(self):
        """See class definition."""
        # Discrete space for DQN
        return Discrete(2 ** self.num_traffic_lights)

    def get_id_within_dist(self, edge, direction):
        """Collect vehicle ids within looking ahead or behind distance

        Parameters
        ----------
        edge: string
            the name of the edge to observe

        direction: string
            the direction of the edge relative to the traffic lights.
            Can be either "ahead" or "behind

        Returns
        ----------
        list
            list of observed string ids of vehicles either
            ahead or behind traffic light

        """
        if direction == "ahead":
            ids_in_scope = filter(self.is_within_look_ahead, self.k.vehicle.get_ids_by_edge(edge))
            return list(ids_in_scope)

        if direction == "behind":
            ids_in_scope = filter(self.is_within_look_behind, self.k.vehicle.get_ids_by_edge(edge))
            return list(ids_in_scope)

    def is_within_look_ahead(self, veh_id):
        """Check if vehicle is within the looking distance

        Parameters
        ----------
        veh_id: string
            string id of vehicle in pre-defined lane

        Returns
        ----------
        bool
            True or False
        """

        if self.get_distance_to_intersection(veh_id) <= self.look_ahead:
            return True
        else:
            return False

    def is_within_look_behind(self, veh_id):
        """Check if vehicle is within the looking distance

        Parameters
        ----------
        veh_id: string
            string id of vehicle in pre-defined lane

        Returns
        ----------
        bool
            True or False
        """

        if self.k.vehicle.get_position(veh_id) <= self.look_ahead:
            return True
        else:
            return False

    def color_vehicles(self, ids, color):
        """Color observed vehicles to visualize during simulation

        Parameters
        ----------
        ids: list
            list of string ids of vehicles to color
        color: tuple
            tuple of RGB color pattern to color vehicles

        """
        for veh_id in ids:
            self.k.vehicle.set_color(veh_id=veh_id, color=color)

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
            iter_ = self.get_training_iter()
            # log average travel time
            self.log_travel_times(rl_actions, iter_)
            # log average reward
            self.log_rewards(self.rew_list, rl_actions, during_simulation=False, n_iter=iter_)

        return next_observation, reward, done, infos

    def log_travel_times(self, rl_actions, iter_):
        """log average travel time to tensorboard

        Parameters
        ----------
         rl_actions : array_like
            a list of actions provided by the rl algorithm
         iter_: int
            value of training iteration currently being simulated

        """

        # wait a short period of time to ensure the xml file is readable
        time.sleep(0.1)

        # collect the location of the emission file
        dir_path = self.sim_params.emission_path
        emission_filename = \
            "{0}-emission.xml".format(self.network.name)
        emission_path = os.path.join(dir_path, emission_filename)

        # convert the emission file into a csv adn return trip info in dict
        trip_info = trip_info_emission_to_csv(emission_path)

        # log travel times to tensorbord
        info = pd.DataFrame(trip_info)

        # Delete the .xml version of the emission file.
        # os.remove(emission_path)

        if rl_actions is None:
            n_iter = self.step_counter
            string = "untrained"
        else:
            n_iter = iter_
            string = "trained"

        # get average of full trip durations
        avg = info.travel_times.mean()
        print("avg_travel_time = " + str(avg))
        self.writer.add_scalar(self.exp_name + '/travel_times ' + string, avg, n_iter)

    def log_rewards(self, rew, action, during_simulation=False, n_iter=None):
        """log current reward during simulation or average reward after simulation to tensorboard

        Parameters
        ----------
         rew : array_like or int
            single value of current time-step's reward if int
            array or rewards for each time-step for entire simulation
         action : array_like
            a list of actions provided by the rl algorithm
         during_simulation : bool
            an list of actions provided by the rl algorithm
         n_iter: int
            value of training iteration currently being simulated

        """

        if action is None:
            string = "untrained"
        else:
            string = "trained"

        if during_simulation:
            self.writer.add_scalar(
                self.exp_name + '/reward_per_simulation_step ' + string,
                rew,
                self.step_counter
            )
        else:
            avg = np.mean(np.array(rew))
            print("avg_reward = " + str(avg))
            self.writer.add_scalar(
                self.exp_name + '/average_reward ' + string,
                avg,
                n_iter
            )

    def get_training_iter(self):
        """Create csv file to track train iterations
        iteration steps and update the values

        Returns
        ----------
        n_iter: int
            value of training iteration currently being simulated

        """
        # filename = "/iterations_{}_{}.csv".format(self.look_ahead, demand)
        # root_dir = os.path.expanduser('~')
        # file_location = root_dir + '/ray_results/grid-trail-analysis'
        # full_path = file_location+filename

        # check if file exists in directory
        if not os.path.isfile(full_path):
            # create dataframe with training_iteration = 0
            data = {"training_iteration": 0}
            file_to_convert = pd.DataFrame([data])

            # convert to csv
            file_to_convert.to_csv(full_path, index=False)
            return 0

        else:
            # read csv
            df = pd.read_csv(full_path, index_col=False)
            n_iter = df.training_iteration.iat[-1]

            # increase iteration by 1
            data = {"training_iteration": n_iter+1}
            file_to_convert = df.append(data, ignore_index=True)

            # convert to csv
            file_to_convert.to_csv(full_path, index=False)

            return n_iter+1
