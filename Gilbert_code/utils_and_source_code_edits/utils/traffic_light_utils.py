# TODO: Describe briefly what is in the functions do
from lxml import etree
from xml.etree import ElementTree
import random
from flow.core.params import InitialConfig, NetParams
from flow.core.params import InFlows
import numpy as np
import time
import os
import pandas as pd
import matplotlib.pyplot as plt

# index to split traffic lights string eg. "center0".split("center")[ID_IDX] = 0
ID_IDX = 1


def trip_info_emission_to_csv(emission_path, output_path=None):
    """Convert an trip_info file generated by sumo into a csv file.

    Note that the trip_info file contains information generated by sumo, not
    flow.

    Parameters
    ----------
    emission_path : str
        path to the trip_info file that should be converted
    output_path : str
        path to the csv file that will be generated, default is the same
        directory as the trip_info file, with the same name
    """
    parser = etree.XMLParser(recover=True)
    tree = ElementTree.parse(emission_path, parser=parser)
    root = tree.getroot()

    # parse the xml data into a dict
    out_data = []
    for car in root.findall("tripinfo"):
        out_data.append(dict())
        try:
            out_data[-1]['travel_times'] = float(car.attrib['duration'])
            out_data[-1]['arrival'] = float(car.attrib['arrival'])
            out_data[-1]['id'] = car.attrib['id']
        except KeyError:
            del out_data[-1]

    # sort the elements of the dictionary by the vehicle id
    out_data = sorted(out_data, key=lambda k: k['id'])

    # default output path
    if output_path is None:
        output_path = emission_path[:-3] + 'csv'

    # output the dict data into a csv file
    # keys = out_data[0].keys()
    # with open(output_path, 'w') as output_file:
    #     dict_writer = csv.DictWriter(output_file, keys)
    #     dict_writer.writeheader()
    #     dict_writer.writerows(out_data)

    return out_data


def gen_edges(col_num, row_num):
    """Generate the names of the outer edges in the grid network.

    Parameters
    ----------
    col_num : int
        number of columns in the grid
    row_num : int
        number of rows in the grid

    Returns
    -------
    list of str
        names of all the outer edges
    """
    edges_col = []
    edges_row = []

    # build the left and then the right edges
    for i in range(col_num):
        edges_col += ['left' + str(row_num) + '_' + str(i)]
        edges_col += ['right' + '0' + '_' + str(i)]

    # build the bottom and then top edges
    for i in range(row_num):
        edges_row += ['bot' + str(i) + '_' + '0']
        edges_row += ['top' + str(i) + '_' + str(col_num)]

    return edges_col, edges_row


def get_flow_params(col_num, row_num, horizon, num_veh_per_row, num_veh_per_column, additional_net_params):
    """Define the network and initial params in the presence of inflows.

    Parameters
    ----------
    col_num : int
        number of columns in the grid
    row_num : int
        number of rows in the grid
    horizon : int
        time period is seconds over which to generate inflows
    num_veh_per_row : int
        total vehicles to be inserted via each row in the given time horizon.
    num_veh_per_column : int
        total vehicles to be inserted via each column in the given time horizon.
    additional_net_params : dict
        network-specific parameters that are unique to the grid

    Returns
    -------
    flow.core.params.InitialConfig
        parameters specifying the initial configuration of vehicles in the
        network
    flow.core.params.NetParams
        network-specific parameters used to generate the network
    """
    initial = InitialConfig(
        spacing='custom', lanes_distribution=float('inf'), shuffle=True)

    col_edges, row_edges = gen_edges(col_num, row_num)

    inflow = gen_demand(horizon, num_veh_per_row, num_veh_per_column, col_edges, row_edges)

    net = NetParams(
        inflows=inflow,
        additional_params=additional_net_params)

    return initial, net


def gen_demand(horizon,
               num_veh_per_row,
               num_veh_per_column,
               col_edges,
               row_edges,
               is_uniform=True):
    """Generate an inflow object of demands.
        format: object (see imported class: flow.core.params.InFlows)
                veh_type='human',
                edge="?",
                probability=1,
                depart_lane='free',
                depart_speed=5,
                begin="?",
                number=1)

    Parameters:
    ----------
      horizon: float
        time period is seconds over which to generate inflows

      num_veh_per_row: int
        total vehicles to be inserted via each row in the given time horizon.

      num_veh_per_column: int
        total vehicles to be inserted via each column in the given time horizon.

      col_edges: list
        a list of strings [top_segment,bottom_segment..] of all incoming column segments
        where each segment is the column segment id in sumo.

      row_edges: list
        a list of strings [right_segment, left_segment..] of all incoming row segments
        where each segment is the row segment id in sumo.

      is_uniform: bool,
        if false then generate a normal distribution

    Returns:
    ---------
    inflow: object
      `an inflow object containing predefined demands for FLOW.
    """

    inflow = InFlows()
    rows = row_edges
    col = col_edges
    mean = horizon / 2
    std = 10
    row_time = []
    row_edges = []
    col_time = []
    col_edges = []

    # for each row
    for _ in np.arange(num_veh_per_row):
        # pick time
        if is_uniform:
            row_time += [random.choice(range(1, horizon))]
        else:
            # we center demand around horizon/2
            row_time += [get_truncated_normal(mean, std, 1, horizon)]

        # pick edge randomly
        row_edges += [random.choice(rows)]

        # for each column
    for _ in np.arange(num_veh_per_column):
        # pick time
        if is_uniform:
            col_time += [random.choice(range(1, horizon))]
        else:
            # we center demand around horizon/2
            col_time += [get_truncated_normal(mean, std, 1, horizon)]

        # pick edge randomly
        col_edges += [random.choice(col)]

    # merge lists
    merged_times = row_time + col_time
    merged_edges = row_edges + col_edges

    # sort by depart time (SUMO requires them to be in order of time)
    sorted_times_and_edges = sorted(zip(merged_times, merged_edges), key=lambda x: x[0])

    # add inflow
    for time_, edge in sorted_times_and_edges:
        inflow.add(
            veh_type='human',
            edge=edge,
            probability=1,
            depart_lane='free',
            depart_speed=5,
            begin=time_,
            number=1)

        # store histogram of demand
    # if save_hist:
    #     home_dir = os.path.expanduser('~')
    #     ensure_dir('%s' % home_dir + '/ray_results/real_time_metrics/hist')
    #     hist_path = home_dir + '/ray_results/real_time_metrics/hist/'
    #
    #     if is_uniform:
    #         title_flag = "Random Distribution"
    #     else:
    #         title_flag = "Peak Distribution: Mean = {} secs, Standard Dev ={} secs,".format(mean, std)
    #
    #     plt.hist(vehicle_str.keys(), edgecolor='white')
    #     plt.ylabel("Frequency")
    #     plt.xlabel("Depart time INTO the Network (secs)")
    #     plt.title("Demand Data \n {} vehicles \n".format(num_of_vehicles) + title_flag)
    #     plt.savefig(hist_path + '%s.png' % network_name)
    #     plt.close()

    return inflow


def get_truncated_normal(mean=0, sd=1800, low=0, upp=10):
    """Generate a peak distribution of values centred at the mean
    Args:
        mean: int value to center distribution on
        sd: int value of standard deviation of interest
        low: int value of lowest value to consider as lower bound when sampling
        upp: int value of highest value to consider as higher bound when sampling

    Returns:
        int: randomly selected value given bounds and parameters above
        """

    while True:
        rd = random.normalvariate(mean, sd)
        if low <= rd <= upp:
            return int(rd)


def get_non_flow_params(enter_speed, add_net_params):
    """Define the network and initial params in the absence of inflows.

    Note that when a vehicle leaves a network in this case, it is immediately
    returns to the start of the row/column it was traversing, and in the same
    direction as it was before.

    Parameters
    ----------
    enter_speed : float
        initial speed of vehicles as they enter the network.
    add_net_params: dict
        additional network-specific parameters (unique to the grid)

    Returns
    -------
    flow.core.params.InitialConfig
        parameters specifying the initial configuration of vehicles in the
        network
    flow.core.params.NetParams
        network-specific parameters used to generate the network
    """
    additional_init_params = {'enter_speed': enter_speed}
    initial = InitialConfig(
        spacing='custom', additional_params=additional_init_params)
    net = NetParams(additional_params=add_net_params)

    return initial, net


def log_travel_times(rl_actions,
                     iter_,
                     obj,
                     network,
                     sim_params,
                     step_counter
                     ):
    """log average travel time to tensorboard

    Parameters
    ----------
     rl_actions : dict or int
        actions provided by the rl algorithm

     iter_: int
        value of simulation step currently being trained/run
        Note: if the training is at simulation 50, iter_ = 50

     obj: object
        BenchmarkParams object containing benchmark parameters

     network: obj
        (from flow.network)
        object to collect network information
        example: kernel.network.rts
                returns a dict containing list of strings or
                all the route edge names

    sim_params : flow.core.params.SimParams
        simulation-specific parameters

    step_counter: int
        current simulation time step

    """

    save_plots = obj.save_plots
    exp_being_run = obj.grid_demand
    writer = obj.writer
    exp_name = obj.exp_name

    # wait a short period of time to ensure the xml file is readable
    time.sleep(0.1)

    # collect the location of the emission file
    dir_path = sim_params.emission_path
    emission_filename = \
        "{0}-emission.xml".format(network.name)
    emission_path = os.path.join(dir_path, emission_filename)

    # convert the emission file into a csv adn return trip info in dict
    trip_info = trip_info_emission_to_csv(emission_path)

    # log travel times to tensorbord
    info = pd.DataFrame(trip_info)

    # Delete the .xml version of the emission file.
    os.remove(emission_path)

    if rl_actions is None:
        n_iter = step_counter
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
        plt.title("{} Travel Time Distribution\n "
                  "{} Avg Travel Time \n"
                  " {} arrived cars,  last car at {}".format(exp_being_run,
                                                             int(avg),
                                                             len(info.travel_times),
                                                             max(info.arrival)))

        plt.savefig("{}.png".format(obj.full_histogram_path))
        # plt.show()
    writer.add_scalar(exp_name + '/travel_times ' + string, avg, n_iter)
    writer.add_scalar(exp_name + '/arrived cars ' + string, len(info.travel_times), n_iter)


def log_rewards(rew,
                action,
                obj,
                n_iter,
                step_counter,
                during_simulation=False):
    """log current reward during simulation or average reward after simulation to tensorboard

    Parameters
    ----------
     rew : array_like or int
        single value of current time-step's reward if int
        array or rewards for each time-step for entire simulation

     action : array_like
        a list of actions provided by the rl algorithm

     obj : object
        object containing tensorboard parameters

     n_iter: int
        value of simulation step currently being trained/run
        Note: if the training is at simulation 50, iter_ = 50

    step_counter: int
        current simulation time step

    during_simulation: bool
        flag for whether to plot reward during iteration

    """
    writer = obj.writer
    exp_name = obj.exp_name
    if action is None:
        string = "untrained"
    else:
        string = "trained"

    if during_simulation:
        if len(rew) == 1:
            plt.plot([step_counter - 1, step_counter], [0, rew[0]], color="r")
        else:
            plt.plot([step_counter-1, step_counter], rew[-2:], color="r")
        plt.title("Reward Evolution during Simulation")
        plt.xlabel("iteration")
        plt.ylabel("reward")
        plt.pause(0.0001)

        writer.add_scalar(
            exp_name + '/reward_per_simulation_step ' + string,
            rew[-1],
            step_counter
        )
    else:
        avg = np.mean(np.array(rew))
        print("avg_reward = " + str(avg))
        writer.add_scalar(
            exp_name + '/average_reward ' + string,
            avg,
            n_iter
        )


def get_training_iter(full_path):
    """Create csv file to track train iterations
    iteration steps and update the values

    Parameters:
    -----------
    full_path: string
        file location containing current simulation step

    Returns
    ----------
    n_iter: int
        n_iter: int
        value of simulation step currently being trained/run
        Note: if the training is at simulation 50, iter_ = 50

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

        return n_iter + 1


def color_vehicles(ids, color, kernel):
    """Color observed vehicles to visualize during simulation

    Parameters
    ----------
    ids: list
        list of string ids of vehicles to color

    color: tuple
        tuple of RGB color pattern to color vehicles

    kernel: obj
    Traci API obj to collect current simulation information
    (from flow.kernel in parent class)
        example- kernel.vehicle.get_accel(veh_id)
                returns the acceleration of the vehicle id

    """
    for veh_id in ids:
        kernel.vehicle.set_color(veh_id=veh_id, color=color)


def get_id_within_dist(edge, direction, kernel, obj):
    """Collect vehicle ids within looking ahead or behind distance

    Parameters
    ----------
    edge: string
        the name of the edge to observe

    direction: string
        the direction of the edge relative to the traffic lights.
        Can be either "ahead" or "behind

    kernel: obj
    Traci API obj to collect current simulation information
    (from flow.kernel in parent class)
        example- kernel.vehicle.get_accel(veh_id)
                returns the acceleration of the vehicle id

    obj : object
        object containing tensorboard parameters

    Returns
    ----------
    list
        list of observed string ids of vehicles either
        ahead or behind traffic light

    """
    if direction == "ahead":
        filter_func = is_within_look_ahead(kernel, obj.look_ahead)
        ids_in_scope = filter(filter_func, kernel.vehicle.get_ids_by_edge(edge))
        return list(ids_in_scope)

    if direction == "behind":
        filter_func = is_within_look_behind(kernel, obj.look_ahead)
        ids_in_scope = filter(filter_func, kernel.vehicle.get_ids_by_edge(edge))
        return list(ids_in_scope)


def is_within_look_ahead(kernel, look_ahead):
    """Check if vehicle is within the looking distance

    Parameters
    ----------
    kernel: obj
    Traci API obj to collect current simulation information
    (from flow.kernel in parent class)
        example- kernel.vehicle.get_accel(veh_id)
                returns the acceleration of the vehicle id

    look_ahead: int
        distance for traffic light to look ahead and observe vehicles

    Returns
    ----------
    deep_filter: function
        parameterized function to check whether vehicles are within looking distance
    """

    def deep_filter(veh_id):
        """Return bool whether vehicle is in look_ahead distance
        Parameters:
        ----------
        veh_id: string
            string id of vehicle

        Returns:
        --------
        bool: True or False
            whether vehicle is within look ahead distance

        """
        if find_intersection_dist(veh_id, kernel) <= look_ahead:
            return True
        else:
            return False

    return deep_filter


def is_within_look_behind(kernel, look_ahead):
    """Check if vehicle is within the looking distance

    Parameters
    ----------
    kernel: obj
    Traci API obj to collect current simulation information
    (from flow.kernel in parent class)
        example- kernel.vehicle.get_accel(veh_id)
                returns the acceleration of the vehicle id

    look_ahead: int
        distance for traffic light to look behind and observe vehicles

    Returns
    ----------
    deep_filter: function
        parameterized function to check whether vehicles are within looking distance
    """

    def deep_filter(veh_id):
        """Return bool whether vehicle is in look_ahead distance
        Parameters:
        ----------
        veh_id: string
            string id of vehicle

        Returns:
        --------
        bool: True or False
            whether vehicle is within look behinddistance

        """

        if kernel.vehicle.get_position(veh_id) <= look_ahead:
            return True
        else:
            return False

    return deep_filter


def find_intersection_dist(veh_id, kernel):
    """Return distance from intersection.

    Return the distance from the vehicle's current position to the position
    of the node it is heading toward.
    """
    edge_id = kernel.vehicle.get_edge(veh_id)
    # FIXME this might not be the best way of handling this
    if edge_id == "":
        return -10
    if 'center' in edge_id:
        return 0
    edge_len = kernel.network.edge_length(edge_id)
    relative_pos = kernel.vehicle.get_position(veh_id)
    dist = edge_len - relative_pos
    return dist


def get_light_states(kernel, rl_id):
    """Map the traffic light state into an unique float number.

    Parameters:
    ----------
    kernel: obj
    Traci API obj to collect current simulation information
    (from flow.kernel in parent class)
        example- kernel.vehicle.get_accel(veh_id)
                returns the acceleration of the vehicle id


     rl_id: string
        name id of current traffic light node/intersection

    Returns:
    ---------
    light_states__ : list
         list that contains only one float number, which uniquely represents the traffic light state.
            ie. GrGrGr = [1]"""

    light_states = kernel.traffic_light.get_state(rl_id)

    if light_states == "GrGr":
        encoded_light_state = [1]
    elif light_states == "yryr":
        encoded_light_state = [0.6]
    else:
        # ryry or rGrG
        encoded_light_state = [0.2]

    return encoded_light_state


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


    Parameters:
    -----------
    kernel: obj
    Traci API obj to collect current simulation information
    (from flow.kernel in parent class)
        example- kernel.vehicle.get_accel(veh_id)
                returns the acceleration of the vehicle id

    Returns:
    ---------
    internal_edges: list
        list of all internal edges for each route (including last outgoing edge)
     """

    internal_edges = []
    for i in kernel.network.rts:
        if kernel.network.rts[i]:
            if kernel.network.rts[i][0][0][1:-1]:
                internal_edges += [kernel.network.rts[i][0][0][1:]]
    return internal_edges


def get_outgoing_edge(kernel, edge_, internal_edges):
    """Collect the next (outgoing) edge for vehicles given the incoming edge id""
    ie.
                    incoming
                        |
        -> --incoming---|---outgoing--->
                        |
                     outgoing
    Parameters:
    ----------
    kernel: obj
    Traci API obj to collect current simulation information
    (from flow.kernel in parent class)
        example- kernel.vehicle.get_accel(veh_id)
                returns the acceleration of the vehicle id

    edge_: string
        string name of incoming edge

    Returns:
    ---------
    outgoing_edge: string
        outgoing edge (next edge given the incoming edge)
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


def execute_action(self, i, rl_action):
    """Execute action on traffic light

    Parameters:
    ---------
    self: object
        environment object to collect variables, simulation properties and set actions in simulation

    i: string
        string name of traffic light node/intersection being acted on

    rl_action: int
        agent action value to be executed (see action space for detailed explanation)

    """
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
        if self.last_change[i] >= self.yellow_phase_duration:
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
