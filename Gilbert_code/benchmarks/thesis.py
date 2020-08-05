import numpy as np

ID_IDX = 1

RED = (255, 0, 0)
BLUE = (0, 0, 255)
CYAN = (0, 255, 255)
log_rewards_during_iteration = False
look_ahead = 43


class ThesisCentLightGridEnv:
    """TODO: cite paper"""

    def obs_shape_func(self):
        """Define the shape of the observation space for the Masters Thesis benchmark"""

        cars_in_scope = self.get_cars_inscope()
        obs_shape = (cars_in_scope * 3 + 10) * self.num_traffic_lights

        return obs_shape

    def get_state(self):

        """Observations for each traffic light agent.
            TODO: FIX This signature
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
            cars_in_scope = self.get_cars_inscope()
            veh_positions = np.zeros(cars_in_scope)
            relative_speeds = np.zeros(cars_in_scope)
            accelerations = np.zeros(cars_in_scope)

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
        final_rews = sum(list((rews.values())))
        return final_rews

    def get_cars_inscope(self):
        """"compute the maximum number of vehicles that can be observed given the look ahead distance
        Returns: int
            number of vehicles that can be observed"""

        if look_ahead == 43:
            cars_in_scope = 24

            # elif look_ahead == 80:
            #     cars_in_scope = """TODO"""
            #     obs_shape = (124 * 3 + 10) * self.num_traffic_lights
            #
            # elif look_ahead == 160:
            #     cars_in_scope = """TODO"""
            #     obs_shape = ("""TODO""" * 3 + 10) * self.num_traffic_lights

        elif look_ahead == 240:
            cars_in_scope = 124

        return cars_in_scope

# class ThesisDecentLightGridEnv(base, PressLight):
#  def reward(....):
#    return pressure_for_one_light(...)