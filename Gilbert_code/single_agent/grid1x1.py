"""Grid example."""
from flow.controllers import GridRouter
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams
from flow.core.params import TrafficLightParams
from flow.core.params import SumoCarFollowingParams
from flow.core.params import InFlows
from flow.envs.ring.accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.envs import TrafficLightGridPOEnv, MyGridEnv
from flow.networks import TrafficLightGridNetwork
import numpy as np
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.spaces import Tuple

# # # exp 1
arterial = 600
side_street = 180

# # exp 2
# arterial = 1400
# side_street = 420


WHITE = (255, 255, 255)
CYAN = (0, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

USE_INFLOWS = True
ADDITIONAL_ENV_PARAMS = {
        'target_velocity': 11,
        'switch_time': 3.0, # min switch time
        'num_observed': 1, # num of cars we can observe
        'discrete': False,
        'tl_type': 'controlled' # actuated by SUMO
    }
v_enter = 11
inner_length = 240
long_length = 240
short_length = 240
n_rows = 1
n_columns = 1
num_cars_left = 1 #up
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
    edges = []

    # build the left and then the right edges
    for i in range(col_num):
        edges += ['left' + str(row_num) + '_' + str(i)]
        edges += ['right' + '0' + '_' + str(i)]

    # build the bottom and then top edges
    for i in range(row_num):
        edges += ['bot' + str(i) + '_' + '0']
        edges += ['top' + str(i) + '_' + str(col_num)]

    return edges


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

    inflow = InFlows()
    outer_edges = gen_edges(col_num, row_num)
    for i in range(len(outer_edges)):
        if outer_edges[i].__contains__("top") or outer_edges[i].__contains__("bot"):
            vph = arterial
        else:
            vph = side_street
        inflow.add(
            veh_type='human',
            edge=outer_edges[i],
            vehs_per_hour=vph,
            depart_lane='free',
            depart_speed=5)  # was 20

    net = NetParams(
        inflows=inflow,
        additional_params=additional_net_params)

    return initial, net


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
vehicles.add(
    veh_id="human",
    routing_controller=(GridRouter, {}),
    car_following_params=SumoCarFollowingParams(
        min_gap=2.5,
        decel=7.5,  # avoid collisions at emergency stops
    ),
    num_vehicles=tot_cars)

env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

tl_logic = TrafficLightParams(baseline=False)
phases = [{
    "duration": "31",
    "minDur": "8",
    "maxDur": "45",
    "state": "GrGr"
}, {
    "duration": "6",
    "minDur": "3",
    "maxDur": "6",
    "state": "yryr"
}, {
    "duration": "31",
    "minDur": "8",
    "maxDur": "45",
    "state": "rGrG"
}, {
    "duration": "6",
    "minDur": "3",
    "maxDur": "6",
    "state": "ryry"
}]
tl_logic.add("center0", phases=phases, programID=1, tls_type="actuated")
# tl_logic.add("center1", phases=phases, programID=1)
# tl_logic.add("center2", phases=phases, programID=1, tls_type="actuated")

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
    sim=SumoParams(
        sim_step=1,
        render=False,
        emission_path='~/flow/data',
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=1000,
        additional_params=ADDITIONAL_ENV_PARAMS.copy(),
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
