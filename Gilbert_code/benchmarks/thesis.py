import numpy as np
from flow.core.traffic_light_utils import color_vehicles, get_id_within_dist

ID_IDX = 1


class ThesisDecentLightGridEnv:

    def __init__(self, params_obj):

        self.benchmark_params = params_obj
        self.look_ahead = self.benchmark_params.look_ahead
        self.num_of_emergency_stops = dict()
        self.delays = dict()
        self.num_of_switch_actions = dict()
        self.current_state = dict()
        self.prev_state = dict()
        self.edge_pressure_dict = dict()
        self.waiting_times = dict()
    
    def obs_shape_func(self):
        """Define the shape of the observation space for the Masters Thesis benchmark"""

        cars_in_scope = self.get_cars_inscope()
        obs_shape = (cars_in_scope * 3 + 10)

        return obs_shape

    def get_state(self,
                  kernel,
                  network,
                  _get_relative_node,
                  direction,
                  currently_yellow,
                  step_counter):
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
        direction = direction.flatten()
        currently_yellow = currently_yellow.flatten()
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
        for i in kernel.network.rts:
            if kernel.network.rts[i]:
                if kernel.network.rts[i][0][0][1:-1]:
                    internal_edges += [kernel.network.rts[i][0][0][1:]]

        node_to_edges = network.node_mapping

        all_ids_incoming = dict()
        for rl_id in kernel.traffic_light.get_ids():
            all_ids_incoming[rl_id] = []

            # collect observations for each traffic light
            rl_id_num = int(rl_id.split("center")[ID_IDX])
            local_edges = node_to_edges[rl_id_num][1]
            local_edge_numbers = [kernel.network.get_edge_list().index(e)
                                  for e in local_edges]
            local_id_nums = [rl_id_num, _get_relative_node(rl_id, "top"),
                             _get_relative_node(rl_id, "bottom"),
                             _get_relative_node(rl_id, "left"),
                             _get_relative_node(rl_id, "right")]

            # define incoming edges
            incoming = local_edges
            outgoing_edges = []
            edge_pressure = []
            for edge_ in incoming:
                if kernel.network.rts[edge_]:
                    # if edge is an outer(global) incoming edge,
                    # outgoing edge is the next edge in the route
                    index_ = kernel.network.rts[edge_][0][0].index(edge_)
                    outgoing_edges = kernel.network.rts[edge_][0][0][index_ + 1]
                else:
                    for lst in internal_edges:
                        # if edge is an inner edges, outgoing is the next edge in the list
                        if len(lst) > 1 and edge_ in lst:
                            index_ = lst.index(edge_)
                            outgoing_edges = lst[index_ + 1]

                # get vehicle ids in incoming edge
                observed_ids = \
                    get_id_within_dist(edge_, "ahead", kernel, self.benchmark_params)
                all_ids_incoming[rl_id] += observed_ids

                # get ids in outgoing edge
                observed_ids_behind = \
                    get_id_within_dist(outgoing_edges, "behind", kernel, self.benchmark_params)

                # get edge pressures
                edge_pressure += [len(observed_ids) - len(observed_ids_behind)]

                # color incoming and outgoing vehicles
                color_vehicles(observed_ids, self.benchmark_params.CYAN, kernel)
                color_vehicles(observed_ids_behind, self.benchmark_params.RED, kernel)

            # for each incoming edge, store the pressure terms to be used in compute reward
            self.edge_pressure_dict[rl_id] = edge_pressure

            # initialize obs arrays
            veh_positions = np.zeros(self.get_cars_inscope())
            relative_speeds = np.zeros(self.get_cars_inscope())
            accelerations = np.zeros(self.get_cars_inscope())

            # initailize more local obs at intersections
            num_of_emergency_stops = 0
            local_waiting_time = 0
            delays = 0

            i = 0
            for all_veh_ids in all_ids_incoming[rl_id]:

                local_waiting_time += kernel.kernel_api.vehicle.getWaitingTime(all_veh_ids)
                veh_positions[i] = kernel.vehicle.get_position(all_veh_ids)
                relative_speeds[i] = kernel.kernel_api.vehicle.getSpeed(all_veh_ids) /\
                                     kernel.kernel_api.vehicle.getMaxSpeed(all_veh_ids)
                num_of_emergency_stops += kernel.kernel_api.vehicle.getAcceleration(all_veh_ids) < -4.5
                accelerations[i] = kernel.kernel_api.vehicle.getAcceleration(all_veh_ids)
                delays += kernel.kernel_api.vehicle.getAllowedSpeed(all_veh_ids) - kernel.kernel_api.vehicle.getSpeed(all_veh_ids)
                i += 1

            light_states = kernel.traffic_light.get_state(rl_id)

            if step_counter == 1:
                self.current_state[rl_id] = light_states
            if step_counter > 1:
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

    def compute_reward(self, rl_actions, step_counter, **kwargs):
        """See class definition."""
        if rl_actions is None:
            return {}
        rews = {}
        for rl_id, rl_action in rl_actions.items():

            if step_counter == 1:
                changed = 0
            elif step_counter > 1:
                if self.prev_state[rl_id] == self.current_state[rl_id]:
                    changed = 0
                else:
                    changed = 1

            if step_counter == 1:
                self.num_of_switch_actions[rl_id] = [changed]
            elif step_counter < 121:
                self.num_of_switch_actions[rl_id] += [changed]
            else:
                self.num_of_switch_actions[rl_id] = self.num_of_switch_actions[rl_id][1:] + [changed]

            rews[rl_id] = - (0.1 * sum(self.num_of_switch_actions[rl_id]) +
                                 0.2 * self.num_of_emergency_stops[rl_id] +
                                 0.3 * self.delays[rl_id] +
                                 0.3 * self.waiting_times[rl_id]/60
                                 )

        return rews

    def get_cars_inscope(self):
        """"compute the maximum number of vehicles that can be observed given the look ahead distance
        Returns: int
            number of vehicles that can be observed"""

        if self.benchmark_params.look_ahead == 43:
            cars_in_scope = 24

            # elif look_ahead == 80:
            #     cars_in_scope = """TODO"""
            #     obs_shape = (124 * 3 + 10) * self.num_traffic_lights
            #
            # elif look_ahead == 160:
            #     cars_in_scope = """TODO"""
            #     obs_shape = ("""TODO""" * 3 + 10) * self.num_traffic_lights

        elif self.benchmark_params.look_ahead == 240:
            cars_in_scope = 124

        return cars_in_scope
    

class ThesisCentLightGridEnv(ThesisDecentLightGridEnv):
    """TODO: cite paper"""

    def obs_shape_func(self):
        """Define the shape of the observation space for the Masters Thesis benchmark"""

        obs_shape = super().obs_shape_func()
        return obs_shape * self.num_traffic_lights

    def get_state(self,
                  kernel,
                  network,
                  _get_relative_node,
                  direction,
                  currently_yellow,
                  step_counter):
        """ see parent class"""

        obs = super().get_state(kernel,
                                network,
                                _get_relative_node,
                                direction,
                                currently_yellow,
                                step_counter)

        final_obs = np.concatenate(list((obs.values())))
        return final_obs

    def compute_reward(self, rl_actions, step_counter, **kwargs):
        """Compute the pressure reward for this time step
        Args:
            TODO
        Returns:
            TODO
        """
        rews = super().compute_reward(rl_actions, step_counter)
        final_rews = sum(list((rews.values())))
        return final_rews

