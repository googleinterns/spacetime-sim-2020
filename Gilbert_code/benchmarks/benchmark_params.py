from tensorboardX import SummaryWriter
import os


class BenchmarkParams:
    """
    Parameters required by environments:
       mostly contains tensorboard setup and naming parameters,
       flags for when to log info on tensorboard,
       saving simulation results,
       training flags such as actuated baselines

    """

    def __init__(self):

        # specify path for tensorboard logging avg travel times, avg reward, no. of vehicles in network
        home_dir = os.path.expanduser('~')
        if not os.path.exists(home_dir + '/ray_results/real_time_metrics'):
            os.makedirs(home_dir + '/ray_results/real_time_metrics')
        self.summaries_dir = home_dir + '/ray_results/real_time_metrics'

        # logging evolution of reward during a simulation
        self.log_rewards_during_iteration = False

        # activate sumo actuated baseline during training for first 6 iterations
        self.sumo_actuated_baseline = False
        self.sumo_actuated_simulations = 6

        # plot and save histogram of travel time distribution at end of simulation
        self.save_plots = True
        # name and title of histogram (from above) to be saved
        self.exp_being_run = "PRESSURE_CENTRALIZED"

        # choose look-ahead distance can be either 43, 80, 160 or 240
        self.look_ahead = 43

        # choose demand pattern for naming the tensorboard directory logging
        # self.demand = "L"
        # self.demand = "2x2DECENTRALIZED_Pressure_B"
        self.demand = "TEST"

        # choose exp running, can either be "rl" or "non_rl"
        self.exp = "rl"

        # log title for tensorboard
        self.log_title = '/simulation_1x3_analysis_{}_{}_{}'.format(self.exp, self.look_ahead, self.demand)
        self.filename = "/iterations_{}_{}.csv".format(self.look_ahead, self.demand)
        root_dir = os.path.expanduser('~')

        # file location to store simulation numbers csv while training
        # ie. if the training is at simulation 50, the last value add the created csv file, will be 50
        # this helps us keep track of the average travel times, rewards and other values at the end
        # of each simulation
        file_location = root_dir + '/ray_results/grid-trail'
        self.full_path = file_location + self.filename

        # RGB colors for vehicles
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.CYAN = (0, 255, 255)

        # set up more tensorboard logging info
        self.exp_name = self.log_title
        self.writer = SummaryWriter(self.summaries_dir + self.exp_name)
        self.rew_list = []
