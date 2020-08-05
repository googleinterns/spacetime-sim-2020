"""Multi-agent environments for networks with traffic lights.

These environments are used to train traffic lights to regulate traffic flow
through an n x m traffic light grid.
"""

import numpy as np
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from flow.envs.multiagent.traffic_light_grid import MultiTrafficLightGridPOEnv
from flow.core.traffic_light_utils import trip_info_emission_to_csv
from tensorboardX import SummaryWriter
import pandas as pd
import time
import os
from datetime import datetime
import matplotlib.pyplot as plt

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
sumo_actuated_baseline = True
save_plots = False
exp_being_run = "masters demand sumo"

# choose look-ahead distance
# look_ahead = 80
# look_ahead = 160
# look_ahead = 240
look_ahead = 43

# choose demand pattern
# demand = "L_32x32x32"
demand = "H_grid1x3_masters_monday_v2_BS"

# choose exp running
exp = "rl"
# exp = "non_rl"

# log title for tensorboard
log_title = '/simulation_1x3_analysis_{}_{}_{}'.format(exp, look_ahead, demand)
filename = "/iterations_{}_{}.csv".format(look_ahead, demand)
root_dir = os.path.expanduser('~')
file_location = root_dir + '/ray_results/grid1x3_learning_rate_0.01'
full_path = file_location + filename
ADDITIONAL_ENV_PARAMS = {
    # num of nearby lights the agent can observe {0, ..., num_traffic_lights-1}
    "num_local_lights": 4,  # FIXME: not implemented yet
    # num of nearby edges the agent can observe {0, ..., num_edges}
    "num_local_edges": 4,  # FIXME: not implemented yet
}

# Index for retrieving ID when splitting node name, e.g. ":center#"
ID_IDX = 1

RED = (255, 0, 0)
BLUE = (0, 0, 255)
CYAN = (0, 255, 255)


#######################################################
# ######### Gilbert code above for PressLight ####### #
#######################################################


class MultiTrafficLightGridPOEnvPL(MultiTrafficLightGridPOEnv):

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

        # set up tensorboard logging info
        self.exp_name = log_title
        self.writer = SummaryWriter(summaries_dir + self.exp_name)
        self.rew_list = []
        # set look ahead distance
        self.look_ahead = look_ahead
        self.edge_pressure_dict = None
        self.waiting_times = None
        self.num_of_emergency_stops= dict()
        self.delays = dict()
        self.num_of_switch_actions = dict()
        self.current_state = dict()
        self.prev_state = dict()

        if look_ahead == 43:
            self.cars_in_scope = 24
            self.obs_shape = 24 * 3 + 10

        # elif look_ahead == 80:
        #     cars_in_scope = """TODO"""
        #     obs_shape = 124 * 3 + 10
        #
        # elif look_ahead == 160:
        #     cars_in_scope = """TODO"""
        #     obs_shape = """TODO""" * 3 + 10

        elif look_ahead == 240:
            self.cars_in_scope = 124
            self.obs_shape = 124 * 3 + 10

    def get_state(self):
        """Observations for each traffic light agent.

        :return: dictionary which contains agent-wise observations as follows:
        - For the self.num_observed number of vehicles closest and incoming
        towards traffic light agent, gives the vehicle velocity, distance to
        intersection, edge number.
        - For edges in the network, gives the density and average velocity.
        - For the self.num_local_lights number of nearest lights (itself
        included), gives the traffic light information, including the last
        change time, light direction (i.e. phase), and a currently_yellow flag.
        """

        # Traffic light information
        direction = self.direction.flatten()
        currently_yellow = self.currently_yellow.flatten()
        # This is a catch-all for when the relative_node method returns a -1
        # (when there is no node in the direction sought). We add a last
        # item to the lists here, which will serve as a default value.
        # TODO(cathywu) are these values reasonable?
        direction = np.append(direction, [0])
        currently_yellow = np.append(currently_yellow, [1])

        obs = {}
        # TODO(cathywu) allow differentiation between rl and non-rl lights

        # collect list of names of inner edges
        internal_edges = []
        self.edge_pressure_dict = dict()
        self.waiting_times = dict()
        for i in self.k.network.rts:
            if self.k.network.rts[i]:
                if self.k.network.rts[i][0][0][1:-1]:
                    internal_edges += [self.k.network.rts[i][0][0][1:]]

        node_to_edges = self.network.node_mapping

        all_ids_incoming = dict()
        for rl_id in self.k.traffic_light.get_ids():
            all_ids_incoming[rl_id] = []

            # collect observations for each traffic light
            rl_id_num = int(rl_id.split("center")[ID_IDX])
            local_edges = node_to_edges[rl_id_num][1]
            local_edge_numbers = [self.k.network.get_edge_list().index(e)
                                  for e in local_edges]
            local_id_nums = [rl_id_num, self._get_relative_node(rl_id, "top"),
                             self._get_relative_node(rl_id, "bottom"),
                             self._get_relative_node(rl_id, "left"),
                             self._get_relative_node(rl_id, "right")]

            # define incoming edges
            incoming = local_edges
            outgoing_edges = []
            edge_pressure = []
            for edge_ in incoming:
                if self.k.network.rts[edge_]:
                    # if edge is an outer(global) incoming edge,
                    # outgoing edge is the next edge in the route
                    index_ = self.k.network.rts[edge_][0][0].index(edge_)
                    outgoing_edges = self.k.network.rts[edge_][0][0][index_ + 1]
                else:
                    for lst in internal_edges:
                        # if edge is an inner edges, outgoing is the next edge in the list
                        if len(lst) > 1 and edge_ in lst:
                            index_ = lst.index(edge_)
                            outgoing_edges = lst[index_ + 1]

                # get vehicle ids in incoming edge
                observed_ids = \
                    self.get_id_within_dist(edge_, direction="ahead")
                all_ids_incoming[rl_id] += observed_ids

                # get ids in outgoing edge
                observed_ids_behind = \
                    self.get_id_within_dist(outgoing_edges, direction="behind")

                # get edge pressures
                edge_pressure += [len(observed_ids) - len(observed_ids_behind)]

                # color incoming and outgoing vehicles
                self.color_vehicles(observed_ids, CYAN)
                self.color_vehicles(observed_ids_behind, RED)

            # for each incoming edge, store the pressure terms to be used in compute reward
            self.edge_pressure_dict[rl_id] = edge_pressure

            # initialize obs arrays
            veh_positions = np.zeros(self.cars_in_scope)
            relative_speeds = np.zeros(self.cars_in_scope)
            accelerations = np.zeros(self.cars_in_scope)

            # initailize more local obs at intersections
            num_of_emergency_stops = 0
            local_waiting_time = 0
            delays = 0

            i = 0
            for all_veh_ids in all_ids_incoming[rl_id]:

                local_waiting_time += self.k.kernel_api.vehicle.getWaitingTime(all_veh_ids)
                veh_positions[i] = self.k.vehicle.get_position(all_veh_ids)
                relative_speeds[i] = self.k.kernel_api.vehicle.getSpeed(all_veh_ids) /\
                                     self.k.kernel_api.vehicle.getMaxSpeed(all_veh_ids)
                num_of_emergency_stops += self.k.kernel_api.vehicle.getAcceleration(all_veh_ids) < -4.5
                accelerations[i] = self.k.kernel_api.vehicle.getAcceleration(all_veh_ids)
                delays += self.k.kernel_api.vehicle.getAllowedSpeed(all_veh_ids) - self.k.kernel_api.vehicle.getSpeed(all_veh_ids)
                i += 1

            light_states = self.k.traffic_light.get_state(rl_id)

            if self.step_counter == 1:
                self.current_state[rl_id] = light_states
            if self.step_counter > 1:
                self.prev_state[rl_id] = self.current_state[rl_id]
                self.current_state[rl_id] = light_states

            if light_states == "GrGr":
                light_states_ = [1]
            elif light_states == ["yryr"]:
                light_states_ = [0.6]
            else:
                light_states_ = [0.2]

            self.waiting_times[rl_id] = local_waiting_time
            self.num_of_emergency_stops[rl_id] = num_of_emergency_stops
            self.delays[rl_id] = delays

            observation = np.array(np.concatenate(
                [veh_positions,
                 relative_speeds,
                 accelerations,
                 local_edge_numbers,
                 direction[local_id_nums],
                 light_states_,
                 ]))

            obs.update({rl_id: observation})

        return obs

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        if rl_actions is None:
            return {}
        rews = {}
        for rl_id, rl_action in rl_actions.items():

            if self.step_counter == 1:
                changed = 0
            elif self.step_counter > 1:
                if self.prev_state[rl_id] == self.current_state[rl_id]:
                    changed = 0
                else:
                    changed = 1

            if self.step_counter == 1:
                self.num_of_switch_actions[rl_id] = [changed]
            elif self.step_counter < 121:
                self.num_of_switch_actions[rl_id] += [changed]
            else:
                self.num_of_switch_actions[rl_id] = self.num_of_switch_actions[rl_id][1:] + [changed]

            rews[rl_id] = - (0.1 * sum(self.num_of_switch_actions[rl_id]) +
                                 0.2 * self.num_of_emergency_stops[rl_id] +
                                 0.3 * self.delays[rl_id] +
                                 0.3 * self.waiting_times[rl_id]/60
                                 )

        return rews

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
            # hardcoded in for 1 x 3 traffic light
            shape=(self.obs_shape,),
            dtype=np.float32)
        return tl_box

    @property
    def action_space(self):
        """See class definition."""
        if self.discrete:
            return Discrete(2)
        else:
            return Box(
                low=-1,
                high=1,
                shape=(1,),
                dtype=np.float32)

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
        self.rew_list += list(reward.values())
        if done["__all__"]:
            # current training iteration
            iter_ = self.get_training_iter()
            # log average travel time
            self.log_travel_times(rl_actions, iter_)
            # log average reward
            self.log_rewards(self.rew_list, rl_actions, during_simulation=False, n_iter=iter_)

        return next_observation, reward, done, infos

    def _apply_rl_actions(self, rl_actions):
        """
        See parent class.

        Issues action for each traffic light agent.
        """
        if sumo_actuated_baseline:
            # return
            if not os.path.isfile(full_path):
                return
            else:
                # read csv
                df = pd.read_csv(full_path, index_col=False)
                n_iter = df.training_iteration.iat[-1]

            if n_iter < 6:
                return

        for rl_id, rl_action in rl_actions.items():
            i = int(rl_id.split("center")[ID_IDX])
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
        print("arrived cars = {}".format(len(info.travel_times)))
        print("last car at = {}".format(max(info.arrival)))
        if save_plots:
            plt.hist(info.travel_times, bins=150)
            plt.xlabel("Travel Times (sec)")
            plt.ylabel("Number of vehicles/frequency")
            plt.title("1x3 {} Travel Time Distribution\n "
                      "{} Avg Travel Time \n"
                      " {} arrived cars,  last car at {}".format(exp_being_run,
                                                                 int(avg),
                                                                 len(info.travel_times),
                                                                 max(info.arrival)))
            plt.savefig("{}.png".format(exp_being_run))
            # plt.show()
        self.writer.add_scalar(self.exp_name + '/travel_times ' + string, avg, n_iter)
        self.writer.add_scalar(self.exp_name + '/arrived cars ' + string, len(info.travel_times), n_iter)

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
            data = {"training_iteration": n_iter + 1}
            file_to_convert = df.append(data, ignore_index=True)

            # convert to csv
            file_to_convert.to_csv(full_path, index=False)

            return n_iter+1
