import numpy as np
from flow.core.traffic_light_utils import get_light_states, \
    get_observed_ids, get_edge_params, get_internal_edges, get_outgoing_edge

ID_IDX = 1


class ThesisLightGridEnv:

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
        self.exp_type = None
    
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
                  step_counter,
                  rl_id):
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
        # This is a catch-all for when the relative_node method returns a -1
        # (when there is no node in the direction sought). We add a last
        # item to the lists here, which will serve as a default value.
        # TODO(cathywu) are these values reasonable?
        direction = np.append(direction, [0])

        # collect list of names of inner edges
        # collect list of names of inner edges
        internal_edges = get_internal_edges(kernel)
        all_ids_incoming = dict()

        # collect and set edge names and ids
        incoming_edges, local_edge_numbers, local_id_nums = get_edge_params(rl_id,
                                                                            network,
                                                                            kernel,
                                                                            _get_relative_node)

        # define incoming edges
        edge_pressure_state = []
        for edge_ in incoming_edges:
            # get outgoing edge
            outgoing_edge = get_outgoing_edge(kernel, edge_, internal_edges)

            # collect observed ids within distances and color them
            observed_ids_ahead, observed_ids_behind = get_observed_ids(kernel,
                                                                       edge_,
                                                                       outgoing_edge,
                                                                       self.benchmark_params)

            all_ids_incoming[rl_id] += observed_ids_ahead

            # get edge pressures of those edges
            edge_pressure_state += [len(observed_ids_ahead) - len(observed_ids_behind)]

        # for each incoming edge, store the pressure terms to be used in compute reward
        self.edge_pressure_dict[rl_id] = edge_pressure_state
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

        light_states_id = get_light_states(kernel, rl_id)

        self.waiting_times[rl_id] = local_waiting_time
        self.num_of_emergency_stops[rl_id] = num_of_emergency_stops
        self.delays[rl_id] = delays

        observation = np.array(np.concatenate(
            [veh_positions,
             relative_speeds,
             accelerations,
             local_edge_numbers,
             direction[local_id_nums],
             light_states_id,
             ]))

        return observation

    def compute_reward(self,
                       rl_actions,
                       step_counter,
                       action_dict=None,
                       rl_id=None, **kwargs):
        """See class definition."""

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

        rews = - (0.1 * sum(self.num_of_switch_actions[rl_id]) +
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