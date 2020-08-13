"""Multi-agent traffic light example (single shared policy)."""
from ray.rllib.agents.dqn.dqn_policy import DQNTFPolicy
from ray.tune.registry import register_env
from flow.utils.registry import make_create_env
from flow.core.params import SumoParams, EnvParams
from flow.core.params import TrafficLightParams
from flow.core.params import SumoCarFollowingParams, VehicleParams
from flow.envs.centralized_env import MultiTrafficLightGridPOEnvTH
from flow.envs.multiagent.decentralized_env import MultiTrafficLightGridPOEnvPL
from flow.networks import TrafficLightGridNetwork
from flow.controllers import SimCarFollowingController, GridRouter
from flow.core.traffic_light_utils import get_non_flow_params, get_flow_params

N_ROLLOUTS = 1  # number of rollouts per training iteration
N_CPUS = 1  # number of parallel workers
"""Grid example."""

# # # exp 1
# arterial = 600
# side_street = 180

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

vehicles = VehicleParams()

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
        horizon=3600,
        num_veh_per_row=arterial,
        num_veh_per_column=side_street,
        additional_net_params=additional_net_params)
else:
    initial_config, net_params = get_non_flow_params(
        enter_speed=v_enter,
        add_net_params=additional_net_params)

env_name_ = MultiTrafficLightGridPOEnvTH, MultiTrafficLightGridPOEnvPL

flow_params = dict(
    # name of the experiment
    exp_tag='THESIS_test',

    # name of the flow environment the experiment is running on
    env_name=MultiTrafficLightGridPOEnvPL,

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
            "benchmark": "PressureLightGridEnv",  #explain why this should be a string
            "benchmark_params": "BenchmarkParams"
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

create_env, env_name = make_create_env(params=flow_params, version=0)

# Register as rllib env
register_env(env_name, create_env)

test_env = create_env()
obs_space = test_env.observation_space
act_space = test_env.action_space


def gen_policy():
    """Generate a policy in RLlib."""
    return DQNTFPolicy, obs_space, act_space, {}


# Setup PG with a single policy graph for all agents
POLICY_GRAPHS = {'av': gen_policy()}


def policy_mapping_fn(_):
    """Map a policy in RLlib."""
    return 'av'


POLICIES_TO_TRAIN = ['av']
