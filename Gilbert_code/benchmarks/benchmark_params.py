from tensorboardX import SummaryWriter
import os
from datetime import datetime


class BenchmarkParams:

    def __init__(self):
        now = datetime.now()
        current_time = now.strftime("%Y-%H-%M-%S")
        home_dir = os.path.expanduser('~')
        if not os.path.exists(home_dir + '/ray_results/real_time_metrics'):
            os.makedirs(home_dir + '/ray_results/real_time_metrics')

        self.summaries_dir = home_dir + '/ray_results/real_time_metrics'
        self.log_rewards_during_iteration = False
        self.sumo_actuated_baseline = False
        self.save_plots = False
        self. exp_being_run = "masters demand sumo"
        self.experiment = "Presslight"

        # choose look-ahead distance
        # self.look_ahead = 80
        # self.look_ahead = 160
        # self.look_ahead = 240
        self.look_ahead = 43

        # choose demand pattern
        # self.demand = "L"
        self.demand = "H"

        # choose exp running
        self.exp = "rl"
        # exp = "non_rl"

        # log title for tensorboard
        self.log_title = '/simulation_1x3_analysis_{}_{}_{}'.format(self.exp, self.look_ahead, self.demand)
        self.filename = "/iterations_{}_{}.csv".format(self.look_ahead, self.demand)
        root_dir = os.path.expanduser('~')
        file_location = root_dir + '/ray_results/grid1x3_learning_rate_0.01'
        self.full_path = file_location + self.filename

        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.CYAN = (0, 255, 255)

        # set up tensorboard logging info
        self.exp_name = self.log_title
        self.writer = SummaryWriter(self.summaries_dir + self.exp_name)
        self.rew_list = []