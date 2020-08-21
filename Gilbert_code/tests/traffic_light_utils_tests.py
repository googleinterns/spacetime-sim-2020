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
                            
                            
 Distances/Measurements from the center of a node in the x y direction in meters defined as follows                       
                           240
                            |
                            |
                            |
                            |
                            |
                            0
                            __
       >240 ---------0 |  center | 0 -------------240>
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
                           40
                            |
                            |
        -> ----------40-----|---50----99---->
                            |
                            |
                           40
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
        """Return position on edge for specified vehicle


            Note: SUMO's geometry is structured in the manner below;

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

        Therefore, to get the actual position of the vehicles, we need to subtract the relative distance
        for some vehicles to the node from the total distance of the edge:
        ie: Nothboud and Westbound

        Therefore;          |
                            |
                           40
                            |
                            |
        -> ----------40-----|---50----99---->
                            |
                            |
                           40
                            |

        ends up being;
                            |
                            |
                        240 - 40 = 200
                            |
                            |
        -> --240 - 40 = 200-----|---50----99---->
                            |
                            |
                        240 - 40 = 200
                            |

          and thus, finally;
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


        """

        if id == "veh_1":
            return 50
        elif id == "veh_2":
            return 99
        elif id == "veh_3":
            return 240-40
        elif id == "veh_4":
            return 240-40
        elif id == "veh_5":
            return 240-40

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
    rows = 1
    cols = 1

    def get_relative_node(self, agent_id, direction):

        """code implementation is found in
            flow/flow/envs/traffic_light_grid.py

        Yield node number of traffic light agent in a given direction.

        For example, the nodes in a traffic light grid with 2 rows and 3
        columns are indexed as follows:

            |     |     |
        --- 3 --- 4 --- 5 ---
            |     |     |
        --- 0 --- 1 --- 2 ---
            |     |     |

        See flow.networks.traffic_light_grid for more information.

        Example of function usage:
        - Seeking the "top" direction to ":center0" would return 3.
        - Seeking the "bottom" direction to ":center0" would return -1.

        Parameters
        ----------
        agent_id : str
            agent id of the form ":center#"
        direction : str
            top, bottom, left, right

        Returns
        -------
        int
            node number
        """
        ID_IDX = 1
        agent_id_num = int(agent_id.split("center")[ID_IDX])
        if direction == "top":
            node = agent_id_num + self.cols
            if node >= self.cols * self.rows:
                node = -1
        elif direction == "bottom":
            node = agent_id_num - self.cols
            if node < 0:
                node = -1
        elif direction == "left":
            if agent_id_num % self.cols == 0:
                node = -1
            else:
                node = agent_id_num - 1
        elif direction == "right":
            if agent_id_num % self.cols == self.cols - 1:
                node = -1
            else:
                node = agent_id_num + 1
        else:
            raise NotImplementedError

        return node


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

        # edges observed vehicles
        expected_value1 = ["route1", "route2"]

        # index of nodes in node list (Network.get_edge_list)
        expected_value2 = [0, 2]

        # node_ids: center0 = 0, center2 = 2
        # [node_id, adjacent nodes ..]
        # [center0, top node, bottom node, left node, right node]
        # [0, -1, -1, -1, -1] = if there's no adjacent node, return -1
        expected_value3 = [0, -1, -1, -1, -1]

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
        direction = array => [current node,   "center0"
                            top adjacent node,
                            bottom adjacent node,
                            left adjacent node,
                            right adjacent node]
        Note: each value 0 if cars are flowing in the NS direction and 0 if flowing in the EW direction
                current node is GrGr therefore = 0

             Note that we have no adjacent nodes so all adjacent nodes default to 0:
                top and bottom node direction = 0
                left and right node direction = 0



        see: flow/envs/traffic_light_grid.py


        There are two pressures for each of the two incoming segments.
        The first pressure is on E-W direction. There is 1 incoming and 2 outgoing vehicles
        within the look-ahead distance, so the pressure is 1 - 2 = -1

       The second pressure is N-W direction. There is 1 incoming and 0 vehicles within the look-ahead distance,
       thus the total pressure is 1-0=1

       Thus, the reward = -sum(pressure) =  -(-1+1) = 0

        [self.edge_pressure_dict[rl_id], = [1-2 ,1]
        local_edge_numbers, = [0,2]
        direction[rl_id], [0]
        light_states_= [1]
        ]))
        """
        expected_value = np.array([-1, 1,  # pressure for observed vehicles on edges
                                   0, 2,  # index of nodes with vehicles in node list (Network.get_edge_list)
                                   0,  # current node direction
                                   1])  # traffic light states GrGr

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

        # total pressure (-1+1) = 0
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
        """Tests get_state method for thesis benchmark

        Considering vehicle length = 5m, look-ahead = 100m
        cars in scope = math.floor(100 / 5) x 2 lanes = 40

        40 is the number of vehicles that can be observed given the look ahead distance of 100.
        To keep the observation space static, we set unobserved vehicle values as zeros (placeholders)

        In this example, only incoming two vehicles are observed in the NS and EW directions
        ie. veh_3 and veh_4

        """

        # placeholders
        veh_positions = np.zeros(40)
        relative_speeds = np.zeros(40)
        accelerations = np.zeros(40)

        # [200, 200, 0, 0....] position for observed first two observed vehicles,
        # 0 (placeholders) for all non observed vehicles
        veh_positions[0:2] = 200

        # observed relative speeds for each observed vehicle (speed/max_speed)
        relative_speeds[0:2] = [15 / 26, 20 / 26]

        # observed accelarations for each observed vehicle
        accelerations[0:2] = [2, 3]
        expected_value = np.array(np.concatenate(
            [veh_positions,
             relative_speeds,
             accelerations,
             [0, 2],  # index of nodes with vehicles in node list (Network.get_edge_list)
             [0],  # current node direction
             [1]]))  # traffic light states GrGr

        experiment = ThesisLightGridEnv(BenchmarkParams())
        experiment.num_local_lanes = 2  # local lanes
        observation = experiment.get_state(kernel=self.kernel_,
                                           network=self.kernel_.network,
                                           _get_relative_node=self.kernel_.get_relative_node,
                                           direction=np.array([0]),
                                           step_counter=1,
                                           rl_id="center0")
        np.testing.assert_array_equal(observation, expected_value)

    def test_compute_reward_thesis(self):
        """Tests compute_reward method for thesis benchmark"""

        # 4 secs + 3 secs for observed incoming vehicles
        waiting_times = 7
        num_of_emergency_stops = 0

        # sum (maximum allowable speed - current speed) for all observed vehicles
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
