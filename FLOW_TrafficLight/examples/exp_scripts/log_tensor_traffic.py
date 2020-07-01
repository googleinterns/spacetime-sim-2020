# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
from tensorboardX import SummaryWriter
import pandas as pd
import numpy as np
import os
from datetime import datetime
#
# now = datetime.now()
# current_time = now.strftime("%Y-%H-%M-%S")
home_dir = os.path.expanduser('~')
# if not os.path.exists(home_dir + '/ray_results/tens_test'):
#     os.makedirs(home_dir + '/ray_results/tens_test')
#


# get current directory
# home_dir = os.path.expanduser('~')
directory = home_dir + '/ray_results/grid-trail'
# directory = '~/ray_results/'
folder = max([os.path.join(directory,d) for d in os.listdir(directory)], key=os.path.getmtime)
print(folder)
# dir = '~/ray_results/grid-trail/DQN_MyGridEnv-v0_0_2020-06-16_18-49-41r6lemi3x/progress.csv'
# df = pd.read_csv(dir)
# iter = df.training_iteration.iat[-1]