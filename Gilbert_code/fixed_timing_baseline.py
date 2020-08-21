from flow.core.experiment import Experiment
import numpy as np
import os
import pandas as pd
import time
import matplotlib.pyplot as plt
from flow.core.params import SumoParams, EnvParams
from flow.core.params import TrafficLightParams
from flow.core.params import SumoCarFollowingParams, VehicleParams
from flow.envs.centralized_env import CentralizedGridEnv
from flow.networks import TrafficLightGridNetwork
from flow.controllers import SimCarFollowingController, GridRouter
from flow.core.traffic_light_utils import get_non_flow_params, get_flow_params, trip_info_emission_to_csv


def fix_timing(total_cycle_time=None, e_w_green_time=None):
    """ Calculate the average travel time for a fixed timing plan 
    Args:
        total_cycle_time:
        e_w_green_time: 
    Returns:
        Average trip duration
    """
    # create flow parameters
    flow_params = create_flow_params(total_cycle_time, e_w_green_time)

    # create experiment object
    exp = Experiment(flow_params)

    # run the simulation
    # run the sumo simulation
    _ = exp.run(1, convert_to_csv=False)

    travel_time_ = get_travel_times(exp)

    return travel_time_


def create_flow_params(total_cycle_time, e_w_green_time):
    """ TODO: document args and rturns

    Parameters
    ----------
    total_cycle_time:
    e_w_green_time:

    Returns
    ---------
    flow_params:

    """

    # Set up the number of vehicles to be inserted in the NS and EW directions
    arterial = 1400
    side_street = 420

    # use inflows specified above
    USE_INFLOWS = True

    # set up road network parameters
    v_enter = 5
    inner_length = 240
    long_length = 240
    short_length = 240
    n_rows = 1
    n_columns = 1

    # number of vehicles inflow (these inflows are used if USE_INFLOWS = False)
    num_cars_left = 0  # up
    num_cars_right = 0  # bottom
    num_cars_top = 0  # right
    num_cars_bot = 0  # left

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

    # specify vehicle parameters to be added
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

    # Set up traffic light parameters
    tl_logic = TrafficLightParams(baseline=False)
    phases = [{
        "duration": str(e_w_green_time),
        "state": "GrGr"
    }, {
        "duration": "4",
        "state": "yryr"
    }, {
        "duration": str(total_cycle_time - 8 - e_w_green_time),
        "state": "rGrG"
    }, {
        "duration": "4",
        "state": "ryry"
    }]

    # add the specified phases and traffic lights: These should match the num_rows + num_col
    # NOTE: Iif tls_type="actuated", SUMO activates the actuated phases timing plan
    tl_logic.add("center0", phases=phases, programID=1)
    # tl_logic.add("center1", phases=phases, programID=1)
    # tl_logic.add("center2", phases=phases, programID=1)
    # tl_logic.add("center3", phases=phases, programID=1)

    # specify network paramters
    additional_net_params = {
        "grid_array": grid_array,
        "speed_limit": 11,
        "horizontal_lanes": 1,
        "vertical_lanes": 1
    }

    # add inflows specified above
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

    # set up flow_params
    flow_params = dict(
        # name of the experiment
        exp_tag='test',

        # name of the flow environment the experiment is running on
        env_name=CentralizedGridEnv,

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
                "yellow_phase_duration": 4,
                "num_observed": 2,
                "discrete": True,
                "tl_type": "actuated",
                "num_local_edges": 4,
                "num_local_lights": 4,
                "benchmark": "PressureLightGridEnv",  # This should be the string name of the benchmark class
                "benchmark_params": "BenchmarkParams"
            }
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

    return flow_params


def get_travel_times(exp):
    """ Get simulation travel time
    Parameters
    ----------
     exp : object
        python object of experiment to extract path info
     avg: int
        average total travel time of experiment

    """

    # wait a short period of time to ensure the xml file is readable
    time.sleep(0.1)

    # collect the location of the emission file
    dir_path = exp.env.sim_params.emission_path
    emission_filename = \
        "{0}-emission.xml".format(exp.env.network.name)
    emission_path = os.path.join(dir_path, emission_filename)

    # convert the emission file into a csv adn return trip info in dict
    trip_info = trip_info_emission_to_csv(emission_path)

    # log travel times to tensorbord
    info = pd.DataFrame(trip_info)

    # Delete the .xml version of the emission file.
    os.remove(emission_path)

    # get average of full trip durations
    avg = info.travel_times.mean()

    return avg


if __name__ == "__main__":
    # If the model is still worse than this fix-timing baseline,
    # visualize the simulation to see what is wrong with the model.

    # create dict, add sumo and store here
    for cycle_length in [120]:
        for demand in ["H"]:
            # between 4 and 116 secs

            # cycle length
            # cycle_length = 120
            # cycle_length = 108

            # set green time
            green_times = np.arange(4, cycle_length-12)
            # green_times = np.arange(34, 35)
            # demand = "H"
            # set sumo
            # if demand == "L":
                # sumo_travel_time = 56.6
                # sumo_pressure = -10.2
                # # #  exp 1
                # arterial = 600
                # side_street = 180

            if demand == "H":
                sumo_travel_time = 97.74
                sumo_pressure = -2.9

                # # exp 2
                arterial = 1400
                side_street = 420

            travel_info = {"sumo\nactuated": sumo_travel_time}
            for time_ in green_times:
                print("testing green_time = {} secs now".format(time_))
                travel_time = fix_timing(total_cycle_time=cycle_length, e_w_green_time=time_)

                # store dict of "green timing and "travel_time"
                travel_info.update({str(time_): travel_time})
            # plot a barchart
            plt.bar(list(travel_info.keys())[0:1], list(travel_info.values())[0:1], color="red")
            plt.bar(list(travel_info.keys())[1:], list(travel_info.values())[1:])

            # display hist and save hist
            plt.ylabel("Average travel time (sec)")
            plt.title("Fixed vs Actuated Travel times: SUMO \n "
                      "1x1 Grid {} Cycle length".format(cycle_length))
            plt.xticks(list(travel_info.keys())[::10])
            plt.savefig("baselines_{}_{}1x1.png".format(cycle_length, demand))
            # plt.show()
            plt.clf()

            # save to csv
            info = pd.DataFrame(travel_info, index=[0])
            info.to_csv("baselines_{}_{}1x1.csv".format(cycle_length, demand), index=False)
