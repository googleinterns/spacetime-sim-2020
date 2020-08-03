from flow.controllers import GridRouter
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams
from flow.core.params import TrafficLightParams
from flow.core.params import SumoCarFollowingParams
from flow.core.params import InFlows
from flow.envs import TrafficLightGridPOEnv, MyGridEnv
from flow.networks import TrafficLightGridNetwork
from flow.core.experiment import Experiment
import numpy as np
import os
from datetime import datetime
from flow.core.util import trip_info_emission_to_csv
import pandas as pd
import time
import matplotlib.pyplot as plt


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
    """Grid example."""

    USE_INFLOWS = True
    ADDITIONAL_ENV_PARAMS = {
        'target_velocity': 11,
        'switch_time': 3.0,  # min switch time
        'num_observed': 1,  # num of cars we can observe
        'discrete': False,
        'tl_type': 'controlled'  # actuated by SUMO
    }
    v_enter = 11
    inner_length = 240
    long_length = 240
    short_length = 240
    n_rows = 2
    n_columns = 2
    num_cars_left = 1  # up
    num_cars_right = 0  # bottom
    num_cars_top = 0  # right
    num_cars_bot = 0  # left
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
    tl_logic.add("center0", phases=phases, programID=1)
    tl_logic.add("center1", phases=phases, programID=1)
    tl_logic.add("center2", phases=phases, programID=1)
    tl_logic.add("center3", phases=phases, programID=1)

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
            restart_instance=True,
        ),

        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            horizon=3600,
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
        for demand in ["L", "H"]:
            # between 4 and 116 secs

            # cycle length
            # cycle_length = 120
            # cycle_length = 108

            # set green time
            green_times = np.arange(4, cycle_length-12)
            # green_times = np.arange(34, 35)
            # demand = "H"
            # set sumo
            if demand == "L":
                sumo_travel_time = 56.6
                sumo_pressure = -10.2
                # #  exp 1
                arterial = 600
                side_street = 180

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
                      "2x2 Grid {} Cycle length\n Demand = {}".format(cycle_length, demand))
            plt.xticks(list(travel_info.keys())[::10])
            plt.savefig("baselines_{}_{}2x2.png".format(cycle_length, demand))
            # plt.show()
            plt.clf()

            # save to csv
            info = pd.DataFrame(travel_info, index=[0])
            info.to_csv("baselines_{}_{}2x2.csv".format(cycle_length, demand), index=False)
