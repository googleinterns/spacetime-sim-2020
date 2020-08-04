"""Grid example."""
from flow.controllers import GridRouter
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import TrafficLightParams
from flow.core.params import InFlows, SumoCarFollowingParams, VehicleParams
from flow.envs.ring.accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.envs.centralized_single_agent_presslight import TrafficLightGridPOEnv, MyGridEnv
from flow.envs.multiagent.decentralized_multi_light_thesis import MultiTrafficLightGridPOEnvPL
# from flow.envs.centralized_multi_agent_thesis import TrafficLightSingleMultiEnv
from flow.networks import TrafficLightGridNetwork
from flow.controllers import SimCarFollowingController, GridRouter
import random
import numpy as np

# # # exp 1
arterial = 600
side_street = 180

# # exp 2
arterial = 1400
side_street = 420


WHITE = (255, 255, 255)
CYAN = (0, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

USE_INFLOWS = True


v_enter = 5
inner_length = 240
long_length = 240
short_length = 240
n_rows = 1
n_columns = 3
num_cars_left = 0 #up
num_cars_right = 0 #bottom
num_cars_top = 0 #right
num_cars_bot = 0#left
tot_cars = (num_cars_left + num_cars_right) * n_columns \
           + (num_cars_top + num_cars_bot) * n_rows

grid_array = {
    "short_length": short_length,
    "inner_length": inner_length,
    "long_length": long_length,
    "row_num": n_rows,
    "col_num": n_columns,
    "cars_left": num_cars_left,
    "cars_right": num_cars_right,
    "cars_top": num_cars_top,
    "cars_bot": num_cars_bot
}


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
        edges_col  += ['left' + str(row_num) + '_' + str(i)]
        edges_col  += ['right' + '0' + '_' + str(i)]

    # build the bottom and then top edges
    for i in range(row_num):
        edges_row += ['bot' + str(i) + '_' + '0']
        edges_row += ['top' + str(i) + '_' + str(col_num)]

    return edges_col, edges_row


def get_flow_params(col_num, row_num, additional_net_params):
    """Define the network and initial params in the presence of inflows.

    Parameters
    ----------
    col_num : int
        number of columns in the grid
    row_num : int
        number of rows in the grid
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

    inflow = gen_demand(3600, arterial, side_street, col_edges, row_edges)

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

    # time = 50
    inflow = InFlows()
    rows = row_edges
    col = col_edges
    mean = horizon / 2
    std = 10
    # vehicle_str = dict()
    row_time = []
    row_edges = []
    col_time = []
    col_edges = []

    # for each row
    for i in np.arange(num_veh_per_row):
        # pick time
        if is_uniform:
            row_time += [random.choice(range(1, horizon))]
        else:
            # we center demand around horizon/2
            row_time += [get_truncated_normal(mean, std, 1, horizon)]

        # pick edge randomly
        row_edges += [random.choice(rows)]

        # for each column
    for i in np.arange(num_veh_per_column):
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

    sorted_times_and_edges = sorted(zip(merged_times, merged_edges), key=lambda x: x[0])

    # add inflow
    for time, edge in sorted_times_and_edges:
        inflow.add(
            veh_type='human',
            edge=edge,
            probability=1,
            depart_lane='free',
            depart_speed=5,
            begin=time,
            number=1)


    return inflow


def get_truncated_normal(mean=0, sd=1800, low=0, upp=10):
    while True:
        rd = random.normalvariate(mean, sd)
        if rd >= low and rd <= upp:
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


vehicles = VehicleParams()
# vehicles.add(
#     veh_id="human",
#     routing_controller=(GridRouter, {}),
#     car_following_params=SumoCarFollowingParams(
#         min_gap=2.5,
#         decel=7.5,  # avoid collisions at emergency stops
#     ),
#     num_vehicles=tot_cars)
vehicles.add(
    veh_id="human",
    acceleration_controller=(SimCarFollowingController, {}),
    car_following_params=SumoCarFollowingParams(
        min_gap=2.5,
        decel=7.5,  # avoid collisions at emergency stops
        speed_mode="right_of_way",
    ),
    routing_controller=(GridRouter, {}),
    num_vehicles=0)

# env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

tl_logic = TrafficLightParams(baseline=False)
phases = [{
    "duration": "31",
    "minDur": "8",
    "maxDur": "45",
    "state": "GrGr"
}, {
    "duration": "4",
    "minDur": "3",
    "maxDur": "6",
    "state": "yryr"
}, {
    "duration": "31",
    "minDur": "8",
    "maxDur": "45",
    "state": "rGrG"
}, {
    "duration": "4",
    "minDur": "3",
    "maxDur": "6",
    "state": "ryry"
}]
tl_logic.add("center0", phases=phases, programID=1, tls_type="actuated")
tl_logic.add("center1", phases=phases, programID=1, tls_type="actuated")
tl_logic.add("center2", phases=phases, programID=1, tls_type="actuated")

additional_net_params = {
    "grid_array": grid_array,
    "speed_limit": 11,
    "horizontal_lanes": 1,
    "vertical_lanes": 1
}

if USE_INFLOWS:
    initial_config, net_params = get_flow_params(
        col_num=n_columns,
        row_num=n_rows,
        additional_net_params=additional_net_params)
else:
    initial_config, net_params = get_non_flow_params(
        enter_speed=v_enter,
        add_net_params=additional_net_params)


flow_params = dict(
    # name of the experiment
    exp_tag='grid-trail',

    # name of the flow environment the experiment is running on
    env_name=MyGridEnv,

    # name of the network class the experiment is running on
    network=TrafficLightGridNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(restart_instance=True,
        sim_step=1,
        render=False,
        emission_path='~/flow/data',
    ),

    # environment related parameters (see flow.core.params.EnvParams)

    env=EnvParams(
        horizon=5400,
        additional_params={
            "target_velocity": 11,
            "switch_time": 4,
            "num_observed": 2,
            "discrete": True,
            "tl_type": "actuated",
            "num_local_edges": 4,
            "num_local_lights": 4,
        }
        # additional_params=ADDITIONAL_ENV_PARAMS,
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=net_params,

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=initial_config,

    # traffic lights to be introduced to specific nodes (see
    # flow.core.params.TrafficLightParams)
    tls=tl_logic,
)
