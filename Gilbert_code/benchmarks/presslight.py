import numpy as np
from flow.core.traffic_light_utils import color_vehicles, get_id_within_dist

ID_IDX = 1


class PressureDecentLightGridEnv:

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
        """Define the shape of the observation space for the PressLight benchmark"""

        obs_shape = 14
        return obs_shape

    def get_state(self,
                  kernel,
                  network,
                  _get_relative_node,
                  direction,
                  currently_yellow,
                  step_counter):

        """See parent class.

        Returns edge pressures of an intersection, edge_numbers,
         traffic light state. This is partially observed
         TODO: Explian in detail
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

            light_states = kernel.traffic_light.get_state(rl_id)

            if light_states == "GrGr":
                light_states_ = [1]
            elif light_states == ["yryr"]:
                light_states_ = [0.6]
            else:
                light_states_ = [0.2]

            observation = np.array(np.concatenate(
                [self.edge_pressure_dict[rl_id],
                 local_edge_numbers,
                 direction[local_id_nums],
                 light_states_
                 ]))
            obs.update({rl_id: observation})

        return obs

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        if rl_actions is None:
            return {}
        rews = {}
        for rl_id in rl_actions.keys():

            # get edge pressures
            rews[rl_id] = -sum(self.edge_pressure_dict[rl_id])

        return rews


class PressureCentLightGridEnv(PressureDecentLightGridEnv):
    """TODO: cite paper"""

    def obs_shape_func(self):
        """Define the shape of the observation space for the PressLight benchmark"""

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

    def compute_reward(self, rl_actions, **kwargs):
        """Compute the pressure reward for this time step
        Args:
            TODO
        Returns:
            TODO
        """
        rew = -np.sum(list(self.edge_pressure_dict.values()))

        # if self.benchmark_params.log_rewards_during_iteration:
        #     # write current reward to tensorboard (during simulation)
        #     self.log_rewards(rew, rl_actions, during_simulation=True)

        return rew
