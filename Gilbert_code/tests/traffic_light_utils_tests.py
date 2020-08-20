"""
Tests for Presslight and Thesis implementations on single traffic light 1x1 grid light network.

Consider a network with the following configuration with the following edges:

    ie.
                            |
                          route2
                            |
                            |
                            |
                            |
        -> --route1---------|----------route1outgoing--->
                            |
                            |
                            |   
                    route2_outgoing
                            |
                            
                            
 Distances/Measurements from the center of a node in the xy defined as follows                       
                            0
                            |
                            |
                            |
                            |
                            |
                           240
                            __
       >0 ---------240 |  center | 0 -------------240>
                            __
                            0
                            |
                            |
                            |
                            |
                            |
                           240


Vehicles on the network are placed in the following manner:

                            |
                            |
                          veh_4
                            |
                            |
        -> ---------veh_3---|---veh_1----veh_2---->
                            |
                            |
                          veh_5
                            |

The positions of specified vehicle in meters:

                            |
                            |
                           200
                            |
                            |
        -> ---------200-----|---50----99---->     
                            |
                            |
                           200  
                            |

The speeds, accelerations, waiting time, of specified vehicles:

        Waiting times (sec):
            "veh_1" = 1
            veh_2" =  2
            "veh_3" = 3
            "veh_4" = 4
            "veh_5" = 5
        
        Speeds (m/s):
            "veh_1" = 5
            veh_2" = 10
            "veh_3" = 15
            "veh_4" = 20
            "veh_5" = 25
        
        
        Accelerations (m/s^2):
            "veh_1" = 0
            veh_2" =  1
            "veh_3" = 2
            "veh_4" = 3
            "veh_5" = 4
        
        Allowable speed = 30m/s
        MaxSpeed = 26m/s

"""

import unittest
from flow.core.traffic_light_utils import get_light_states, \
    get_observed_ids, get_edge_params, get_internal_edges, get_outgoing_edge
from flow.envs.presslight import PressureLightGridEnv
from flow.envs.thesis import ThesisLightGridEnv
from collections import defaultdict
import numpy as np


class Network:
    """Create network class to have network configuration"""

    rts = defaultdict(list)
    rts["route1"] = [(["route1", "route1_outgoing"], 1)]
    rts["route2"] = [(["route2", "route2_outgoing"], 1)]
    node_mapping = [('center0', ["route1", "route2"])]

    def edge_length(self, edge):
        """Return length of specified edge"""
        return 240

    def get_edge_list(self):
        """Return edge edge name in list"""
        return ["route1", "route1_outgoing", "route2", "route2_outgoing"]


class Trafficlight:
    """Create Trafficlight class to have light states"""

    states = {"center0": "GrGr"}

    def get_state(self, rl_id):
        """"Return Traffic lights State"""
        return self.states[rl_id]


class BenchmarkParams:
    """Create benchmark parameter class"""
    look_ahead = 100
    CYAN = None
    RED = None


class Vehicle:
    """Create vehicle class to have properties of alln vehicles"""

    def getWaitingTime(self, id):
        """Return waiting times for specified vehicle"""

        if id == "veh_1":
            return 1
        elif id == "veh_2":
            return 2
        elif id == "veh_3":
            return 3
        elif id == "veh_4":
            return 4
        elif id == "veh_5":
            return 5

    def getSpeed(self, id):
        """Return Speeds for specified vehicle"""

        if id == "veh_1":
            return 5
        elif id == "veh_2":
            return 10
        elif id == "veh_3":
            return 15
        elif id == "veh_4":
            return 20
        elif id == "veh_5":
            return 25

    def getMaxSpeed(self, id):
        """Return maximum speeds for specified vehicle"""
        return 26

    def getAcceleration(self, id):
        """Return accelerations for specified vehicle"""

        if id == "veh_1":
            return 0
        elif id == "veh_2":
            return 1
        elif id == "veh_3":
            return 2
        elif id == "veh_4":
            return 3
        elif id == "veh_5":
            return 4

    def getAllowedSpeed(self, id):
        """Return allowable speed for specified vehicle"""
        return 30

    def get_ids_by_edge(self, edge):
        """Return vehciles located on specified"""

        if edge == "route1_outgoing":
            return ["veh_1", "veh_2"]

        elif edge == "route1":
            return ["veh_3"]

        elif edge == "route2_outgoing":
            return ['veh_5']

        elif edge == "route2":
            return ["veh_4"]

    def get_position(self, id):
        """Return position on edge for specified vehicle"""

        if id == "veh_1":
            return 50
        elif id == "veh_2":
            return 99
        elif id == "veh_3":
            return 200
        elif id == "veh_4":
            return 200
        elif id == "veh_5":
            return 200

    def get_edge(self, id):
        """Return edge where specified vehicle is located"""

        if id == "veh_1":
            return "route1_outgoing"
        elif id == "veh_2":
            return "route1_outgoing"
        elif id == "veh_3":
            return "route1"
        elif id == "veh_4":
            return "route2"
        elif id == "veh_5":
            return "route2_outgoing"

    def set_color(self, veh_id, color):
        pass


class kernel_api:
    """Create kernel api class for vehicle value retrieval"""
    vehicle = Vehicle()


class Kernel:
    """Create kernel api class for vehicle, network and traffic light value retrieval"""
    network = Network()
    traffic_light = Trafficlight()
    vehicle = Vehicle()
    kernel_api = kernel_api()

    def get_relative_node(self, rl_id, str):
        """Return value of node:
        ie. "center0" = 0
            "center1" = 1
        """

        return 0


class TestEnv(unittest.TestCase):
    """Tests for presslight and thesis methods given configured network"""

    # initialize kernel class
    kernel_ = Kernel()

    def test_get_internal_edges(self):
        """Tests get_internal_edges method"""
        expected_value = []
        internal_edges = get_internal_edges(self.kernel_)
        self.assertEquals(internal_edges, expected_value)

    def test_get_outgoing_edge(self):
        """Tests get_outgoing_edge method"""
        expected_value1 = "route1_outgoing"
        expected_value2 = "route2_outgoing"

        internal_edges = []
        result_1 = get_outgoing_edge(self.kernel_, "route1", internal_edges)
        result_2 = get_outgoing_edge(self.kernel_, "route2", internal_edges)
        self.assertEquals(result_1, expected_value1)
        self.assertEquals(result_2, expected_value2)

    def test_get_light_states(self):
        """Tests get_light_states method"""
        expected_value = [1]

        tl_state_1 = get_light_states(self.kernel_, "center0")
        self.assertEquals(tl_state_1, expected_value)

    def test_get_observed_ids(self):
        """Tests get_observed_ids method"""
        expected_value1 = (['veh_3'], ['veh_1', "veh_2"])
        expected_value2 = (['veh_4'], [])

        id_1 = get_observed_ids(self.kernel_, "route1", "route1_outgoing", BenchmarkParams())
        id_2 = get_observed_ids(self.kernel_, "route2", "route2_outgoing", BenchmarkParams())
        self.assertEquals(id_1, expected_value1)
        self.assertEquals(id_2, expected_value2)

    def test_get_edge_params(self):
        """Tests get_edge_params method"""
        expected_value1 = ["route1", "route2"]
        expected_value2 = [0, 2]
        expected_value3 = [0, 0, 0, 0, 0]

        incoming_edges, local_edge_numbers, local_id_nums = \
            get_edge_params(rl_id="center0",
                            network=self.kernel_.network,
                            kernel=self.kernel_,
                            _get_relative_node=self.kernel_.get_relative_node)
        self.assertEquals(incoming_edges, expected_value1)
        self.assertEquals(local_edge_numbers, expected_value2)
        self.assertEquals(local_id_nums, expected_value3)

    def test_get_state_pressure(self):
        """Tests get_state method for pressure benchmark

        [self.edge_pressure_dict[rl_id], = [2-1 ,1]
        local_edge_numbers, = [0,2]
        direction[local_id_nums], [ 0,0,0,0,0]
        light_states_= [1]
        ]))
        """
        expected_value = np.array([-1, 1, 0, 2, 0, 0, 0, 0, 0, 1])

        experiment = PressureLightGridEnv(BenchmarkParams())
        observation = experiment.get_state(kernel=self.kernel_,
                                           network=self.kernel_.network,
                                           _get_relative_node=self.kernel_.get_relative_node,
                                           direction=np.array([0]),
                                           step_counter=1,
                                           rl_id="center0")

        np.testing.assert_array_equal(observation, expected_value)

    def test_compute_reward_pressure(self):
        """Tests compute_reward method for pressure benchmark"""
        expected_value = 0

        experiment = PressureLightGridEnv(BenchmarkParams())
        experiment.get_state(kernel=self.kernel_,
                             network=self.kernel_.network,
                             _get_relative_node=self.kernel_.get_relative_node,
                             direction=np.array([0]),
                             step_counter=1,
                             rl_id="center0")
        reward = experiment.compute_reward(step_counter=0, rl_id="center0")
        self.assertEquals(reward, expected_value)

    def test_get_state_thesis(self):
        """Tests get_state method for thesis benchmark"""

        veh_positions = np.zeros(80)
        relative_speeds = np.zeros(80)
        accelerations = np.zeros(80)
        veh_positions[0:2] = 200
        relative_speeds[0:2] = [15 / 26, 20 / 26]
        accelerations[0:2] = [2, 3]
        expected_value = np.array(np.concatenate(
            [veh_positions,
             relative_speeds,
             accelerations,
             [0, 2],
             [0, 0, 0, 0, 0],
             [1],
             ]))

        experiment = ThesisLightGridEnv(BenchmarkParams())
        observation = experiment.get_state(kernel=self.kernel_,
                                           network=self.kernel_.network,
                                           _get_relative_node=self.kernel_.get_relative_node,
                                           direction=np.array([0]),
                                           step_counter=1,
                                           rl_id="center0")
        np.testing.assert_array_equal(observation, expected_value)

    def test_compute_reward_thesis(self):
        """Tests compute_reward method for thesis benchmark"""

        waiting_times = 7
        num_of_emergency_stops = 0
        delays = 30 - 15 + 30 - 20
        expected_reward = - (0.1 * 0 +
                             0.2 * num_of_emergency_stops +
                             0.3 * delays +
                             0.3 * waiting_times / 60)

        experiment = ThesisLightGridEnv(BenchmarkParams())
        experiment.get_state(kernel=self.kernel_,
                             network=self.kernel_.network,
                             _get_relative_node=self.kernel_.get_relative_node,
                             direction=np.array([0]),
                             step_counter=1,
                             rl_id="center0")

        reward = experiment.compute_reward(step_counter=1, rl_id="center0")
        self.assertEquals(reward, expected_reward)


if __name__ == "__main__":
    unittest.main()
