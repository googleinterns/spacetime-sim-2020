"""Environments for networks with traffic lights.

These environments are used to train traffic lights to regulate traffic flow
through an n x m traffic light grid.
"""

import numpy as np
import re

from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.spaces import Tuple

from flow.core import rewards
from flow.envs.base import Env
from tensorboardX import SummaryWriter
import numpy as np
import os
from datetime import datetime
from flow.core.util import trip_info_emission_to_csv
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
# demand = "L_testing"
demand = "H_testing"

# choose exp running
exp = "rl"
# exp = "non_rl"

# log title for tensorboard
log_title = '/simulation_test_1x1_{}_{}_{}'.format(exp, look_ahead, demand)

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


class TrafficLightGridEnv(Env):
    """Environment used to train traffic lights.

    Required from env_params:

    * switch_time: minimum time a light must be constant before
      it switches (in seconds).
      Earlier RL commands are ignored.
    * tl_type: whether the traffic lights should be actuated by sumo or RL,
      options are respectively "actuated" and "controlled"
    * discrete: determines whether the action space is meant to be discrete or
      continuous

    States
        An observation is the distance of each vehicle to its intersection, a
        number uniquely identifying which edge the vehicle is on, and the speed
        of the vehicle.

    Actions
        The action space consist of a list of float variables ranging from 0-1
        specifying whether a traffic light is supposed to switch or not. The
        actions are sent to the traffic light in the grid from left to right
        and then top to bottom.

    Rewards
        The reward is the negative per vehicle delay minus a penalty for
        switching traffic lights

    Termination
        A rollout is terminated once the time horizon is reached.

    Additional
        Vehicles are rerouted to the start of their original routes once they
        reach the end of the network in order to ensure a constant number of
        vehicles.

    Attributes
    ----------
    grid_array : dict
        Array containing information on the traffic light grid, such as the
        length of roads, row_num, col_num, number of initial cars
    rows : int
        Number of rows in this traffic light grid network
    cols : int
        Number of columns in this traffic light grid network
    num_traffic_lights : int
        Number of intersection in this traffic light grid network
    tl_type : str
        Type of traffic lights, either 'actuated' or 'static'
    steps : int
        Horizon of this experiment, see EnvParams.horion
    obs_var_labels : dict
        Referenced in the visualizer. Tells the visualizer which
        metrics to track
    node_mapping : dict
        Dictionary mapping intersections / nodes (nomenclature is used
        interchangeably here) to the edges that are leading to said
        intersection / node
    last_change : np array [num_traffic_lights]x1 np array
        Multi-dimensional array keeping track, in timesteps, of how much time
        has passed since the last change to yellow for each traffic light
    direction : np array [num_traffic_lights]x1 np array
        Multi-dimensional array keeping track of which direction in traffic
        light is flowing. 0 indicates flow from top to bottom, and
        1 indicates flow from left to right
    currently_yellow : np array [num_traffic_lights]x1 np array
        Multi-dimensional array keeping track of whether or not each traffic
        light is currently yellow. 1 if yellow, 0 if not
    min_switch_time : np array [num_traffic_lights]x1 np array
        The minimum time in timesteps that a light can be yellow. Serves
        as a lower bound
    discrete : bool
        Indicates whether or not the action space is discrete. See below for
        more information:
        https://github.com/openai/gym/blob/master/gym/spaces/discrete.py
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):

        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        self.grid_array = network.net_params.additional_params["grid_array"]
        self.rows = self.grid_array["row_num"]
        self.cols = self.grid_array["col_num"]
        # self.num_observed = self.grid_array.get("num_observed", 3)
        self.num_traffic_lights = self.rows * self.cols
        self.tl_type = env_params.additional_params.get('tl_type')

        super().__init__(env_params, sim_params, network, simulator)

        # Saving env variables for plotting
        self.steps = env_params.horizon
        self.obs_var_labels = {
            'edges': np.zeros((self.steps, self.k.vehicle.num_vehicles)),
            'velocities': np.zeros((self.steps, self.k.vehicle.num_vehicles)),
            'positions': np.zeros((self.steps, self.k.vehicle.num_vehicles))
        }

        # Keeps track of the last time the traffic lights in an intersection
        # were allowed to change (the last time the lights were allowed to
        # change from a red-green state to a red-yellow state.)
        self.last_change = np.zeros((self.rows * self.cols, 1))
        # Keeps track of the direction of the intersection (the direction that
        # is currently being allowed to flow. 0 indicates flow from top to
        # bottom, and 1 indicates flow from left to right.)
        self.direction = np.zeros((self.rows * self.cols, 1))
        # Value of 1 indicates that the intersection is in a red-yellow state.
        # value 0 indicates that the intersection is in a red-green state.
        self.currently_yellow = np.zeros((self.rows * self.cols, 1))

        # when this hits min_switch_time we change from yellow to red
        # the second column indicates the direction that is currently being
        # allowed to flow. 0 is flowing top to bottom, 1 is left to right
        # For third column, 0 signifies yellow and 1 green or red
        self.min_switch_time = env_params.additional_params["switch_time"]

        if self.tl_type != "actuated":
            for i in range(self.rows * self.cols):
                self.k.traffic_light.set_state(
                    node_id='center' + str(i), state="GrGr")
                self.currently_yellow[i] = 0

        # # Additional Information for Plotting
        # self.edge_mapping = {"top": [], "bot": [], "right": [], "left": []}
        # for i, veh_id in enumerate(self.k.vehicle.get_ids()):
        #     edge = self.k.vehicle.get_edge(veh_id)
        #     for key in self.edge_mapping:
        #         if key in edge:
        #             self.edge_mapping[key].append(i)
        #             break

        # check whether the action space is meant to be discrete or continuous
        self.discrete = env_params.additional_params.get("discrete", False)

    @property
    def action_space(self):
        """See class definition."""
        if self.discrete:
            return Discrete(2 ** self.num_traffic_lights)
        else:
            return Box(
                low=-1,
                high=1,
                shape=(self.num_traffic_lights,),
                dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        speed = Box(
            low=0,
            high=1,
            shape=(self.initial_vehicles.num_vehicles,),
            dtype=np.float32)
        dist_to_intersec = Box(
            low=0.,
            high=np.inf,
            shape=(self.initial_vehicles.num_vehicles,),
            dtype=np.float32)
        edge_num = Box(
            low=0.,
            high=1,
            shape=(self.initial_vehicles.num_vehicles,),
            dtype=np.float32)
        traffic_lights = Box(
            low=0.,
            high=1,
            shape=(3 * self.rows * self.cols,),
            dtype=np.float32)
        return Tuple((speed, dist_to_intersec, edge_num, traffic_lights))

    def get_state(self):
        """See class definition."""
        # compute the normalizers
        grid_array = self.net_params.additional_params["grid_array"]
        max_dist = max(grid_array["short_length"],
                       grid_array["long_length"],
                       grid_array["inner_length"])

        # get the state arrays
        speeds = [
            self.k.vehicle.get_speed(veh_id) / self.k.network.max_speed()
            for veh_id in self.k.vehicle.get_ids()
        ]
        dist_to_intersec = [
            self.get_distance_to_intersection(veh_id) / max_dist
            for veh_id in self.k.vehicle.get_ids()
        ]
        edges = [
            self._convert_edge(self.k.vehicle.get_edge(veh_id)) /
            (self.k.network.network.num_edges - 1)
            for veh_id in self.k.vehicle.get_ids()
        ]

        state = [
            speeds, dist_to_intersec, edges,
            self.last_change.flatten().tolist(),
            self.direction.flatten().tolist(),
            self.currently_yellow.flatten().tolist()
        ]
        return np.array(state)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        # check if the action space is discrete
        if self.discrete:
            # convert single value to list of 0's and 1's
            rl_mask = [int(x) for x in list('{0:0b}'.format(rl_actions))]
            rl_mask = [0] * (self.num_traffic_lights - len(rl_mask)) + rl_mask
        else:
            # convert values less than 0 to zero and above 0 to 1. 0 indicates
            # that should not switch the direction, and 1 indicates that switch
            # should happen
            rl_mask = rl_actions > 0.0

        for i, action in enumerate(rl_mask):
            if self.currently_yellow[i] == 1:  # currently yellow
                self.last_change[i] += self.sim_step
                # Check if our timer has exceeded the yellow phase, meaning it
                # should switch to red
                if self.last_change[i] >= self.min_switch_time:
                    if self.direction[i] == 0:
                        self.k.traffic_light.set_state(
                            node_id='center{}'.format(i),
                            state="GrGr")
                    else:
                        self.k.traffic_light.set_state(
                            node_id='center{}'.format(i),
                            state='rGrG')
                    self.currently_yellow[i] = 0
            else:
                if action:
                    if self.direction[i] == 0:
                        self.k.traffic_light.set_state(
                            node_id='center{}'.format(i),
                            state='yryr')
                    else:
                        self.k.traffic_light.set_state(
                            node_id='center{}'.format(i),
                            state='ryry')
                    self.last_change[i] = 0.0
                    self.direction[i] = not self.direction[i]
                    self.currently_yellow[i] = 1

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        return - rewards.min_delay_unscaled(self) \
            - rewards.boolean_action_penalty(rl_actions >= 0.5, gain=1.0)

    # ===============================
    # ============ UTILS ============
    # ===============================

    def get_distance_to_intersection(self, veh_ids):
        """Determine the distance from a vehicle to its next intersection.

        Parameters
        ----------
        veh_ids : str or str list
            vehicle(s) identifier(s)

        Returns
        -------
        float (or float list)
            distance to closest intersection
        """
        if isinstance(veh_ids, list):
            return [self.get_distance_to_intersection(veh_id)
                    for veh_id in veh_ids]
        return self.find_intersection_dist(veh_ids)

    def find_intersection_dist(self, veh_id):
        """Return distance from intersection.

        Return the distance from the vehicle's current position to the position
        of the node it is heading toward.
        """
        edge_id = self.k.vehicle.get_edge(veh_id)
        # FIXME this might not be the best way of handling this
        if edge_id == "":
            return -10
        if 'center' in edge_id:
            return 0
        edge_len = self.k.network.edge_length(edge_id)
        relative_pos = self.k.vehicle.get_position(veh_id)
        dist = edge_len - relative_pos
        return dist

    def _convert_edge(self, edges):
        """Convert the string edge to a number.

        Start at the bottom left vertical edge and going right and then up, so
        the bottom left vertical edge is zero, the right edge beside it  is 1.

        The numbers are assigned along the lowest column, then the lowest row,
        then the second lowest column, etc. Left goes before right, top goes
        before bottom.

        The values are zero indexed.

        Parameters
        ----------
        edges : list of str or str
            name of the edge(s)

        Returns
        -------
        list of int or int
            a number uniquely identifying each edge
        """
        if isinstance(edges, list):
            return [self._split_edge(edge) for edge in edges]
        else:
            return self._split_edge(edges)

    def _split_edge(self, edge):
        """Act as utility function for convert_edge."""
        if edge:
            if edge[0] == ":":  # center
                center_index = int(edge.split("center")[1][0])
                base = ((self.cols + 1) * self.rows * 2) \
                    + ((self.rows + 1) * self.cols * 2)
                return base + center_index + 1
            else:
                pattern = re.compile(r"[a-zA-Z]+")
                edge_type = pattern.match(edge).group()
                edge = edge.split(edge_type)[1].split('_')
                row_index, col_index = [int(x) for x in edge]
                if edge_type in ['bot', 'top']:
                    rows_below = 2 * (self.cols + 1) * row_index
                    cols_below = 2 * (self.cols * (row_index + 1))
                    edge_num = rows_below + cols_below + 2 * col_index + 1
                    return edge_num if edge_type == 'bot' else edge_num + 1
                if edge_type in ['left', 'right']:
                    rows_below = 2 * (self.cols + 1) * row_index
                    cols_below = 2 * (self.cols * row_index)
                    edge_num = rows_below + cols_below + 2 * col_index + 1
                    return edge_num if edge_type == 'left' else edge_num + 1
        else:
            return 0

    def _get_relative_node(self, agent_id, direction):
        """Yield node number of traffic light agent in a given direction.

        For example, the nodes in a traffic light grid with 2 rows and 3
        columns are indexed as follows:

            |     |     |
        --- 3 --- 4 --- 5 ---
            |     |     |
        --- 0 --- 1 --- 2 ---
            |     |     |

        See flow.networks.traffic_light_grid for more information.

        Example of function usage:
        - Seeking the "top" direction to ":center0" would return 3.
        - Seeking the "bottom" direction to ":center0" would return -1.

        Parameters
        ----------
        agent_id : str
            agent id of the form ":center#"
        direction : str
            top, bottom, left, right

        Returns
        -------
        int
            node number
        """
        ID_IDX = 1
        agent_id_num = int(agent_id.split("center")[ID_IDX])
        if direction == "top":
            node = agent_id_num + self.cols
            if node >= self.cols * self.rows:
                node = -1
        elif direction == "bottom":
            node = agent_id_num - self.cols
            if node < 0:
                node = -1
        elif direction == "left":
            if agent_id_num % self.cols == 0:
                node = -1
            else:
                node = agent_id_num - 1
        elif direction == "right":
            if agent_id_num % self.cols == self.cols - 1:
                node = -1
            else:
                node = agent_id_num + 1
        else:
            raise NotImplementedError

        return node

    def additional_command(self):
        """See parent class.

        Used to insert vehicles that are on the exit edge and place them
        back on their entrance edge.
        """
        for veh_id in self.k.vehicle.get_ids():
            self._reroute_if_final_edge(veh_id)

    def _reroute_if_final_edge(self, veh_id):
        """Reroute vehicle associated with veh_id.

        Checks if an edge is the final edge. If it is return the route it
        should start off at.
        """
        edge = self.k.vehicle.get_edge(veh_id)
        if edge == "":
            return
        if edge[0] == ":":  # center edge
            return
        pattern = re.compile(r"[a-zA-Z]+")
        edge_type = pattern.match(edge).group()
        edge = edge.split(edge_type)[1].split('_')
        row_index, col_index = [int(x) for x in edge]

        # find the route that we're going to place the vehicle on if we are
        # going to remove it
        route_id = None
        if edge_type == 'bot' and col_index == self.cols:
            route_id = "bot{}_0".format(row_index)
        elif edge_type == 'top' and col_index == 0:
            route_id = "top{}_{}".format(row_index, self.cols)
        elif edge_type == 'left' and row_index == 0:
            route_id = "left{}_{}".format(self.rows, col_index)
        elif edge_type == 'right' and row_index == self.rows:
            route_id = "right0_{}".format(col_index)

        if route_id is not None:
            type_id = self.k.vehicle.get_type(veh_id)
            lane_index = self.k.vehicle.get_lane(veh_id)
            # remove the vehicle
            self.k.vehicle.remove(veh_id)
            # reintroduce it at the start of the network
            self.k.vehicle.add(
                veh_id=veh_id,
                edge=route_id,
                type_id=str(type_id),
                lane=str(lane_index),
                pos="0",
                speed="max")

    def get_closest_to_intersection(self, edges, num_closest, padding=False):
        """Return the IDs of the vehicles that are closest to an intersection.

        For each edge in edges, return the IDs (veh_id) of the num_closest
        vehicles in edge that are closest to an intersection (the intersection
        they are heading towards).

        This function performs no check on whether or not edges are going
        towards an intersection or not, it just gets the vehicles that are
        closest to the end of their edges.

        If there are less than num_closest vehicles on an edge, the function
        performs padding by adding empty strings "" instead of vehicle ids if
        the padding parameter is set to True.

        Parameters
        ----------
        edges : str | str list
            ID of an edge or list of edge IDs.
        num_closest : int (> 0)
            Number of vehicles to consider on each edge.
        padding : bool (default False)
            If there are less than num_closest vehicles on an edge, perform
            padding by adding empty strings "" instead of vehicle ids if the
            padding parameter is set to True (note: leaving padding to False
            while passing a list of several edges as parameter can lead to
            information loss since you will not know which edge, if any,
            contains less than num_closest vehicles).

        Usage
        -----
        For example, consider the following network, composed of 4 edges
        whose ids are "edge0", "edge1", "edge2" and "edge3", the numbers
        being vehicles all headed towards intersection x. The ID of the vehicle
        with number n is "veh{n}" (edge "veh0", "veh1"...).

                            edge1
                            |   |
                            | 7 |
                            | 8 |
               -------------|   |-------------
        edge0    1 2 3 4 5 6  x                 edge2
               -------------|   |-------------
                            | 9 |
                            | 10|
                            | 11|
                            edge3

        And consider the following example calls on the previous network:

        >>> get_closest_to_intersection("edge0", 4)
        ["veh6", "veh5", "veh4", "veh3"]

        >>> get_closest_to_intersection("edge0", 8)
        ["veh6", "veh5", "veh4", "veh3", "veh2", "veh1"]

        >>> get_closest_to_intersection("edge0", 8, padding=True)
        ["veh6", "veh5", "veh4", "veh3", "veh2", "veh1", "", ""]

        >>> get_closest_to_intersection(["edge0", "edge1", "edge2", "edge3"],
                                         3, padding=True)
        ["veh6", "veh5", "veh4", "veh8", "veh7", "", "", "", "", "veh9",
         "veh10", "veh11"]

        Returns
        -------
        str list
            If n is the number of edges given as parameters, then the returned
            list contains n * num_closest vehicle IDs.

        Raises
        ------
        ValueError
            if num_closest <= 0
        """
        if num_closest <= 0:
            raise ValueError("Function get_closest_to_intersection called with"
                             "parameter num_closest={}, but num_closest should"
                             "be positive".format(num_closest))

        if isinstance(edges, list):
            ids = [self.get_closest_to_intersection(edge, num_closest)
                   for edge in edges]
            # flatten the list and return it
            return [veh_id for sublist in ids for veh_id in sublist]

        # get the ids of all the vehicles on the edge 'edges' ordered by
        # increasing distance to end of edge (intersection)
        veh_ids_ordered = sorted(self.k.vehicle.get_ids_by_edge(edges),
                                 key=self.get_distance_to_intersection)

        # return the ids of the num_closest vehicles closest to the
        # intersection, potentially with ""-padding.
        pad_lst = [""] * (num_closest - len(veh_ids_ordered))
        return veh_ids_ordered[:num_closest] + (pad_lst if padding else [])


class TrafficLightGridPOEnv(TrafficLightGridEnv):
    """Environment used to train traffic lights.

    Required from env_params:

    * switch_time: minimum switch time for each traffic light (in seconds).
      Earlier RL commands are ignored.
    * num_observed: number of vehicles nearest each intersection that is
      observed in the state space; defaults to 2

    States
        An observation is the number of observed vehicles in each intersection
        closest to the traffic lights, a number uniquely identifying which
        edge the vehicle is on, and the speed of the vehicle.

    Actions
        The action space consist of a list of float variables ranging from 0-1
        specifying whether a traffic light is supposed to switch or not. The
        actions are sent to the traffic light in the grid from left to right
        and then top to bottom.

    Rewards
        The reward is the delay of each vehicle minus a penalty for switching
        traffic lights

    Termination
        A rollout is terminated once the time horizon is reached.

    Additional
        Vehicles are rerouted to the start of their original routes once they
        reach the end of the network in order to ensure a constant number of
        vehicles.

    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        for p in ADDITIONAL_PO_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        # number of vehicles nearest each intersection that is observed in the
        # state space; defaults to 2
        self.num_observed = env_params.additional_params.get("num_observed", 2)

        # used during visualization
        self.observed_ids = []
        self.path = summaries_dir + "/"

    @property
    def observation_space(self):
        """State space that is partially observed.

        Velocities, distance to intersections, edge number (for nearby
        vehicles) from each direction, edge information, and traffic light
        state.
        """
        tl_box = Box(
            low=0.,
            high=3,
            shape=(3 * 4 * self.num_observed * self.num_traffic_lights +
                   2 * len(self.k.network.get_edge_list()) +
                   3 * self.num_traffic_lights,),
            dtype=np.float32)
        return tl_box

    def get_state(self):
        """See parent class.

        Returns self.num_observed number of vehicles closest to each traffic
        light and for each vehicle its velocity, distance to intersection,
        edge_number traffic light state. This is partially observed
        """
        speeds = []
        dist_to_intersec = []
        edge_number = []
        max_speed = max(
            self.k.network.speed_limit(edge)
            for edge in self.k.network.get_edge_list())
        grid_array = self.net_params.additional_params["grid_array"]
        max_dist = max(grid_array["short_length"], grid_array["long_length"],
                       grid_array["inner_length"])
        all_observed_ids = []

        for _, edges in self.network.node_mapping:
            for edge in edges:
                observed_ids = \
                    self.get_closest_to_intersection(edge, self.num_observed)
                all_observed_ids += observed_ids

                # check which edges we have so we can always pad in the right
                # positions
                speeds += [
                    self.k.vehicle.get_speed(veh_id) / max_speed
                    for veh_id in observed_ids
                ]
                dist_to_intersec += [
                    (self.k.network.edge_length(
                        self.k.vehicle.get_edge(veh_id)) -
                        self.k.vehicle.get_position(veh_id)) / max_dist
                    for veh_id in observed_ids
                ]
                edge_number += \
                    [self._convert_edge(self.k.vehicle.get_edge(veh_id)) /
                     (self.k.network.network.num_edges - 1)
                     for veh_id in observed_ids]

                if len(observed_ids) < self.num_observed:
                    diff = self.num_observed - len(observed_ids)
                    speeds += [0] * diff
                    dist_to_intersec += [0] * diff
                    edge_number += [0] * diff

        # now add in the density and average velocity on the edges
        density = []
        velocity_avg = []
        for edge in self.k.network.get_edge_list():
            ids = self.k.vehicle.get_ids_by_edge(edge)
            if len(ids) > 0:
                vehicle_length = 5
                density += [vehicle_length * len(ids) /
                            self.k.network.edge_length(edge)]
                velocity_avg += [np.mean(
                    [self.k.vehicle.get_speed(veh_id) for veh_id in
                     ids]) / max_speed]
            else:
                density += [0]
                velocity_avg += [0]
        self.observed_ids = all_observed_ids
        return np.array(
            np.concatenate([
                speeds, dist_to_intersec, edge_number, density, velocity_avg,
                self.last_change.flatten().tolist(),
                self.direction.flatten().tolist(),
                self.currently_yellow.flatten().tolist()
            ]))

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        if self.env_params.evaluate:
            return - rewards.min_delay_unscaled(self)
        else:
            return (- rewards.min_delay_unscaled(self) +
                    rewards.penalize_standstill(self, gain=0.2))

    def additional_command(self):
        """See class definition."""
        # specify observed vehicles
        [self.k.vehicle.set_observed(veh_id) for veh_id in self.observed_ids]


class TrafficLightGridBenchmarkEnv(TrafficLightGridPOEnv):
    """Class used for the benchmarks in `Benchmarks for reinforcement learning inmixed-autonomy traffic`."""

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        if self.env_params.evaluate:
            return - rewards.min_delay_unscaled(self)
        else:
            return rewards.desired_velocity(self)


class TrafficLightGridTestEnv(TrafficLightGridEnv):
    """
    Class for use in testing.

    This class overrides RL methods of traffic light grid so we can test
    construction without needing to specify RL methods
    """

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        pass

    def compute_reward(self, rl_actions, **kwargs):
        """No return, for testing purposes."""
        return 0

    #######################################################
    # ######### Gilbert code below for PressLight ####### #
    #######################################################


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
        for _, edges in self.network.node_mapping:
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

            # assigning unique number id for each incoming edge
            for edge in edges:
                edge_number += \
                    [self._convert_edge(edge) /
                     (self.k.network.network.num_edges - 1)
                     ]

        return np.array(np.concatenate([
            self.edge_pressure, edge_number,
            self.last_change.flatten().tolist(),
            self.direction.flatten().tolist(),
            self.currently_yellow.flatten().tolist()
            ]))

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
            # hardcoded in for single traffic light
            shape=(3 * self.rows * self.cols + 4 + 4,),
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
        os.remove(emission_path)

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
        filename = "/iterations_{}_{}.csv".format(self.look_ahead, demand)
        root_dir = os.path.expanduser('~')
        file_location = root_dir + '/ray_results/grid-trail'
        full_path = file_location+filename

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

            return n_iter
