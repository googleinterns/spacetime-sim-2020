""" PressLight benchmark implementation from paper:
    https://dl.acm.org/doi/pdf/10.1145/3292500.3330949

    The methods in this file define the rewards functions, observations,
    and observation space of the defined benchmark
"""

import numpy as np
from flow.core.traffic_light_utils import get_light_states, \
    get_observed_ids, get_edge_params, get_internal_edges, get_outgoing_edge


class PressureLightGridEnv:
    """ Wei, Hua & Chen, Chacha & Zheng, Guanjie & Wu, Kan & Gayah, Vikash & Xu, Kai & Li, Zhenhui. (2019).
        PressLight: Learning Max Pressure Control to Coordinate Traffic Signals in Arterial Network.
        1290-1298. 10.1145/3292500.3330949.

       An implementation of the PressLight benchmark.
       PressLight utilizes number of vehicles and traffic light states as observation,
       and "pressure" as the reward function:
        ie. single edge pressure is described as the difference between incoming vehicles
            and outgoing vehicles on an edge.
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

    def obs_shape_func(self):
        """Define the shape of the observation space for the PressLight benchmark

        Returns:
        ------
        obs_shape: int
            Value of shape the observation space. Each value in the observation
            space is described in the get_state method
        """

        obs_shape = 14
        return obs_shape

    def get_state(self,
                  kernel,
                  network,
                  _get_relative_node,
                  direction,
                  currently_yellow,
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
        network: obj TODO: Remove
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
            name of current traffic light node/intersection being observed

        Returns:
        ---------
        observation: np array
             observations of current traffic light described by the rl_id:
             ie.  np.concatenate([
                                  edge pressures for each lane pair,
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
        direction = np.append(direction, [0])

        # collect list of names of inner edges
        internal_edges = get_internal_edges(kernel)

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

            # get edge pressures of those edges
            edge_pressure_state += [len(observed_ids_ahead) - len(observed_ids_behind)]

        # for each incoming edge, store the pressure terms to be used in compute reward
        self.edge_pressure_dict[rl_id] = edge_pressure_state

        # for each  intersection, collect traffic light state
        light_states_ = get_light_states(kernel, rl_id)

        observation = np.array(np.concatenate(
            [self.edge_pressure_dict[rl_id],
             local_edge_numbers,
             direction[local_id_nums],
             light_states_
             ]))

        return observation

    def compute_reward(self, rl_actions, step_counter, action_dict=None,
                       rl_id=None, **kwargs):
        """Compute the reward for single traffic light as described in class definition.
           ie. compute the total sum for all pressure values for an edge pair

           Note: Because we are minimizing pressure, we take the negate it in order to maximize reward
                max(rew) = max(-pressure) = min (pressure)

        Parameters:
        -----------
        step_counter: int
            current simulation time step

        rl_id: string
            Name of current traffic light node/intersection being observed

        Returns:
        ---------
        rew : int
            negative of the pressure values
        """

        rew = -sum(self.edge_pressure_dict[rl_id])
        return rew

