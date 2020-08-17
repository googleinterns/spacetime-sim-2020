from tensorboardX import SummaryWriter
import os
from flow.core.util import ensure_dir


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
        # create it if it directory doesn't exist
        root_dir = os.path.expanduser('~')
        tensorboard_log_dir = root_dir + '/ray_results/real_time_metrics'
        self.summaries_dir = ensure_dir(tensorboard_log_dir)

        # file location to store simulation numbers csv while training
        # ie. if the training is at simulation 50, the last value add the created csv file, will be 50
        # this helps us keep track of the average travel times, rewards and other values at the end
        # of each simulation
        simulation_log_location = root_dir + '/ray_results/num_of_simulations'
        ensure_dir(simulation_log_location)

        # saving flag, file location and name of histogram of results
        # plot and save histogram of travel time distribution at end of simulation
        self.save_plots = False
        self.experiment_title = "1x1_DECENTRALIZED_Thesis"
        hist_dir = root_dir + '/ray_results/histograms'
        self.full_histogram_path = ensure_dir(hist_dir) + "/" + self.experiment_title

        # logging evolution of reward during a simulation
        self.log_rewards_during_iteration = False

        # activate sumo actuated baseline during training for first 6 iterations
        self.sumo_actuated_baseline = True
        self.sumo_actuated_simulations = 6

        # choose grid type for naming the tensorboard directory logging
        # will be the name and title of histogram (from save_plots above) to be saved
        self.grid_demand = self.experiment_title

        # choose exp running, can either be "rl" or "non_rl"
        self.exp = "rl"
        # choose look-ahead distance can be either 43, 80, 160 or 240
        self.look_ahead = 43

        # log title for tensorboard and name + file path for simulation log
        self.log_title = '/simulation_analysis_{}_{}_{}'.format(self.exp, self.look_ahead, self.grid_demand)
        self.filename = "/iterations_{}_{}.csv".format(self.look_ahead, self.grid_demand)
        self.full_path = simulation_log_location + self.filename

        # RGB colors for vehicles if coloring a vehicle. We color the observed incoming
        # vehicles BLUE and the observed outgoing vehicles as RED at each intersection
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.CYAN = (0, 255, 255)

        # set up more tensorboard logging info
        self.exp_name = self.log_title
        self.writer = SummaryWriter(self.summaries_dir + self.exp_name)
        self.rew_list = []
