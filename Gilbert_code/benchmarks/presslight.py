import numpy as np
# from flow.envs.traffic_light_grid import TrafficLightGridPOEnv as base

ID_IDX = 1

RED = (255, 0, 0)
BLUE = (0, 0, 255)
CYAN = (0, 255, 255)
log_rewards_during_iteration = False


class PressureCentLightGridEnv:
    """TODO: cite paper"""

    def obs_shape_func(self):
        """Define the shape of the observation space for the PressLight benchmark"""

        obs_shape = 14 * self.num_traffic_lights
        return obs_shape

    def get_state(self):

        """See parent class.

        Returns edge pressures of an intersection, edge_numbers,
         traffic light state. This is partially observed
         TODO: Explian in detail
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

            observation = np.array(np.concatenate(
                [self.edge_pressure_dict[rl_id],
                 local_edge_numbers,
                 direction[local_id_nums],
                 light_states_
                 ]))
            obs.update({rl_id: observation})

        # merge all observations into one list
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

        if log_rewards_during_iteration:
            # write current reward to tensorboard (during simulation)
            self.log_rewards(rew, rl_actions, during_simulation=True)

        return rew

# class PressureDecentLightGridEnv(base, PressLight):
#  def reward(....):
#    return pressure_for_one_light(...)