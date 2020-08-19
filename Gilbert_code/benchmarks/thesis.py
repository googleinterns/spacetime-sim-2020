""" Deep Reinforcement Learning for Coordination in Traffic Light Control benchmark
    implementation from thesis paper:
    https://esc.fnwi.uva.nl/thesis/centraal/files/f632158773.pdf

    The methods in this file define the rewards functions, observations,
    and observation space of the defined benchmark
"""
import numpy as np
import math
from flow.core.traffic_light_utils import get_light_states, \
    get_observed_ids, get_edge_params, get_internal_edges, get_outgoing_edge


class ThesisLightGridEnv:
    """ van der Pol, Elise. (2016).
        Deep Reinforcement Learning for Coordination in Traffic Light Control (MSc thesis).

       An implementation of the thesis benchmark.
       This benchmark utilizes vehicle properties (such as positions, speeds, accelerations)
        and traffic light states as observation, and simulation features such as waiting times, switch frequency etc.
        as the reward function:
    """

    def __init__(self, params_obj):
        """Initialize the class with given BenchmarkParams object"""

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
        """Define the shape of the observation space for the Masters Thesis benchmark

        Returns:
        ------
        obs_shape: int
            Value of shape the observation space. Each value in the observation
            space is described in the get_state method
        """

        cars_in_scope = self.get_cars_inscope()
        obs_shape = (cars_in_scope * 3 + 10)

        return obs_shape

    def get_state(self,
                  kernel,
                  network,
                  _get_relative_node,
                  direction,
                  step_counter,
                  rl_id):
        """ Collect the observations for a single traffic light.

        Parameters:
        ---------
        kernel: obj
            Traci API obj to collect current simulation information
            (from flow.kernel in parent class)
                example: kernel.vehicle.get_accel(veh_id)
                        returns the acceleration of the vehicle id

        network: obj
            (from flow.network)
            object to collect network information
            example: kernel.network.rts
                    returns a dict containing list of strings or
                    all the route edge names

        direction : np array [num_traffic_lights]x1 np array
            Multi-dimensional array keeping track of which direction in traffic
            light is flowing. 0 indicates flow from top to bottom, and
            1 indicates flow from left to right

        step_counter: int
            current simulation time step

        rl_id: string
            Name of current traffic light node/intersection being observed

        Returns:
        ---------
        observation: np array
             observations of current traffic light described by the rl_id:
             ie.  np.concatenate([
                                  vehicle positions of each observed vehicle,
                                  relative speeds of each observed vehicle compared to max allowable speed,
                                  accelerations of each observed vehicle,
                                  local edge ids at the intersection,
                                  direction that is flowing for specific rl_id,
                                  traffic light states values
                                ])

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
        all_ids_incoming[rl_id] = []

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
            edge_pressure_state.append(len(observed_ids_ahead) - len(observed_ids_behind))

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
            relative_speeds[i] = kernel.kernel_api.vehicle.getSpeed(all_veh_ids) / \
                                 kernel.kernel_api.vehicle.getMaxSpeed(all_veh_ids)
            num_of_emergency_stops += kernel.kernel_api.vehicle.getAcceleration(all_veh_ids) < -4.5
            accelerations[i] = kernel.kernel_api.vehicle.getAcceleration(all_veh_ids)
            delays += kernel.kernel_api.vehicle.getAllowedSpeed(all_veh_ids) - kernel.kernel_api.vehicle.getSpeed(
                all_veh_ids)
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

    def compute_reward(self, step_counter, rl_id):
        """Compute the reward for single traffic light as described in class definition.

           Note: Because we are minimizing the collected metrics described below,
           we take the negate it in order to maximize reward

        Parameters:
        -----------
        step_counter: int
            current simulation time step

        rl_id: string
            Name of current traffic light node/intersection being observed

        Returns:
        ---------
        rew : int
            negative of the following weighted metrics:
                [number of switch action in the past 120 seconds,
                number of emergency stops on that intersection for the observed vehicles,
                sum of delays for each observed vehicle,
                sum of waiting times for each observed vehicle]

            """

        # track the time steps in order to know when 120 seconds
        # has been reached in order to stop updating memory pool for switching actions

        if step_counter == 1:
            changed = 0
        elif step_counter > 1:
            if self.prev_state[rl_id] == self.current_state[rl_id]:
                changed = 0
            else:
                changed = 1

        # update the list of action switches
        if step_counter == 1:
            self.num_of_switch_actions[rl_id] = [changed]
        elif step_counter < 121:
            self.num_of_switch_actions[rl_id] += [changed]
        else:
            self.num_of_switch_actions[rl_id] = self.num_of_switch_actions[rl_id][1:] + [changed]

        reward = - (0.1 * sum(self.num_of_switch_actions[rl_id]) +
                  0.2 * self.num_of_emergency_stops[rl_id] +
                  0.3 * self.delays[rl_id] +
                  0.3 * self.waiting_times[rl_id] / 60
                  )

        return reward

    def get_cars_inscope(self):
        """"compute the maximum number of vehicles that can be observed given the look ahead distance
        Returns: int
            number of vehicles that can be observed"""

        vehicle_length = 5
        cars_in_scope_single_lane = math.floor(self.benchmark_params.look_ahead / vehicle_length)

        # multiply by number of lanes at local intersection
        cars_in_scope = cars_in_scope_single_lane * 4

        return cars_in_scope
