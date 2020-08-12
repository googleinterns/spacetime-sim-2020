import numpy as np
from flow.core.traffic_light_utils import color_vehicles, get_id_within_dist

ID_IDX = 1


class PressureDecentLightGridEnv:

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

    def obs_shape_func(self, num_traffic_lights):
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
        # This is a catch-all for when the relative_node method returns a -1
        # (when there is no node in the direction sought). We add a last
        # item to the lists here, which will serve as a default value.
        # TODO(cathywu) are these values reasonable?
        direction = np.append(direction, [0])

        obs = {}

        # collect list of names of inner edges
        internal_edges = get_internal_edges(kernel)

        for rl_id in kernel.traffic_light.get_ids():
            # collect observations for each traffic light

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
            obs.update({rl_id: observation})

        return obs

    def compute_reward(self, rl_actions, step_counter, action_dict=None,
                       rl_id=None, **kwargs):
        """See class definition."""

        return -sum(self.edge_pressure_dict[rl_id])


class PressureCentLightGridEnv(PressureDecentLightGridEnv):
    """TODO: cite paper"""

    def obs_shape_func(self, num_traffic_lights):
        """Define the shape of the observation space for the PressLight benchmark"""

        obs_shape = super().obs_shape_func(num_traffic_lights)
        return obs_shape * num_traffic_lights

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


def get_internal_edges(kernel):
    """Collect all internal edges in network including last outgoing edge of specified route
    ie. Inner edges are Internal edges
                  (outer)         (outer)       (outer)
                     |              |              |
        (outer) x----|----Inner-----|----Inner-----|----x (outer)
                   Inner          Inner          Inner
        (outer) x----|----Inner-----|----Inner-----|----x (outer)
                     |              |              |
                 (outer)         (outer)       (outer)


    """
    internal_edges = []
    for i in kernel.network.rts:
        if kernel.network.rts[i]:
            if kernel.network.rts[i][0][0][1:-1]:
                internal_edges += [kernel.network.rts[i][0][0][1:]]
    return internal_edges


def get_outgoing_edge(kernel, edge_, internal_edges):
    """Collect the next edge for vehicles given the incoming edge id""
    ie.
                    incoming
                        |
        -> --incoming---|---outgoing--->
                        |
                     outgoing
    """
    if kernel.network.rts[edge_]:
        # if edge is an outer(global) incoming edge,
        # outgoing edge is the next edge in the route
        index_ = kernel.network.rts[edge_][0][0].index(edge_)
        outgoing_edge = kernel.network.rts[edge_][0][0][index_ + 1]
    else:
        for lst in internal_edges:
            # if edge is an inner edges, outgoing is the next edge in the list
            if len(lst) > 1 and edge_ in lst:
                index_ = lst.index(edge_)
                outgoing_edge = lst[index_ + 1]

    return outgoing_edge


def get_light_states(kernel, rl_id):
    """Return current traffic light stats as number:
    ie either GrGrGr, yryryr, ryryry, rGrGrG"""

    light_states = kernel.traffic_light.get_state(rl_id)

    if light_states == "GrGr":
        light_states__ = [1]
    elif light_states == "yryr":
        light_states__ = [0.6]
    else:
        light_states__ = [0.2]

    return light_states__


def get_observed_ids(kernel, edge_, outgoing_edge, benchmark_params):
    """Return lists of observed vehicles within look ahead and behind distances"""

    # get vehicle ids in incoming edge
    observed_ids_ahead = \
        get_id_within_dist(edge_, "ahead", kernel, benchmark_params)

    # get ids in outgoing edge
    observed_ids_behind = \
        get_id_within_dist(outgoing_edge, "behind", kernel, benchmark_params)

    # color incoming and outgoing vehicles
    color_vehicles(observed_ids_ahead, benchmark_params.CYAN, kernel)
    color_vehicles(observed_ids_behind, benchmark_params.RED, kernel)

    return observed_ids_ahead, observed_ids_behind


def get_edge_params(rl_id, network, kernel, _get_relative_node):

    """Collect ids and names of edges"""
    node_to_edges = network.node_mapping
    rl_id_num = int(rl_id.split("center")[ID_IDX])
    local_edges = node_to_edges[rl_id_num][1]
    local_edge_numbers = [kernel.network.get_edge_list().index(e)
                          for e in local_edges]
    local_id_nums = [rl_id_num, _get_relative_node(rl_id, "top"),
                     _get_relative_node(rl_id, "bottom"),
                     _get_relative_node(rl_id, "left"),
                     _get_relative_node(rl_id, "right")]

    return local_edges, local_edge_numbers, local_id_nums