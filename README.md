**This is not an officially supported Google product.** 
Traffic light optimization using sumo.

# Traffic Light Optimization using Deep-RL
In this project, we study and implement traffic light optimization techniques using a Deep-Reinforcement Learning Benchmarking tool, FLOW and a traffic simulator, SUMO.\
FLOW (https://flow-project.github.io/)  was developed by the Mobile Sensing Laboratory at the University of California, Berkeley with the goal of providing structured and scalable RL optimization tools in traffic management for autonomous vehicles and traffic lights. \
SUMO (https://sumo.dlr.de/docs/index.html) is an open source, highly portable, microscopic and continuous traffic simulation package designed to handle large networks. 

# Benchmarks Implemented
In the project, we are implement two Traffic Light Optimization Baenchmarks:
- PressLight: Learning Max Pressure Control to Coordinate Traffic Signals in Arterial Network
    - https://dl.acm.org/doi/10.1145/3292500.3330949
    - https://github.com/wingsweihua/presslight

- Coordinated Deep Reinforcement Learners for Traffic Light Control 
    - https://esc.fnwi.uva.nl/thesis/centraal/files/f632158773.pdf

# Installation
In order to begin implementing the code, the user needs to install FLOW in their home directory (user root directory). Please follow the instructions provided in (https://github.com/flow-project/flow) to install and begin using FLOW.
The instructions will guide the user to install SUMO and rllib which are important pieces of the project too.\
Ensure the "spacetime-sim-2020" repo is cloned in the home directory (user root directory) too.

# Files in this Repo
The files located in the directory Gilbert_code correspond to the edited files from the FLOW source code needed to implement our traffic light optimization methods and experiment. These files are located in different places within the FLOW source code.
#  Description of Files and Functionality + How to Run
## In the root directory:
- ####  copy_to_dir.sh
   This is scripts copies all the necessary files/code from FLOW source code into Gilbert_code in order to keep track of the changes made.
- #### copy_from_dir.sh:
    This is scripts copies all the necessary files/code from Gilbert_code into into FLOW source code in order to run.

## In Gilbert_code directory:
### benchmarks directory:
- #### benchmark_params.py
    Source Location: edited from ~/flow/flow/core\
    Contains parameters mainly logging and naming parameters for experiment. The class containined in this file is initialized in the the init statement in centrelized_env.py described below. \ Note: self.look_ahead and self.sumo_actuated_baseline are the only paramters that affect training.

- #### grid_simulation_non_rl.py
    Source Location: edited from ~//flow/examples/exp_configs/non_rl\
    Sets simulation parameters for a non-rl experiment. This environment spawns and renders a SUMO simulation. Traffic light control can either be SUMO inbuilt policies or pre-assigned phases timing plans. To run this file, in the ~/flow/examples directory, run:
    ###### $ python simulate.py --exp_config grid_simulation_non_rl

- #### presslight.py
    Source Location: edited from ~/flow/flow/envs\
    Contains observations and reward functions implementations of Presslight benchmark. If used, the class containined in this file is initialized in the the init statement in centrelized_env.py 

- #### thesis.py
    Source Location: edited from ~/flow/flow/envs\
    Contains observations and reward functions implementations of thesis benchmark. If used, the class containined in this file is initialized in the the init statement in centrelized_env.py 
    
### centralized directory:
- ####  centralized_env.py 
    Source Location: edited from ~/flow/flow/envs\
    Contains gym compatible environment class and methods for centralized experiments. Centralized experiments concatenate all obersevations into a single array and trained that way. The class called has all the implemented methods that called the benchmark classes (eg in presslight.py, thesis.py) to set the observation and action spaces, collect states, compute rewards, and step functions.
- ####  grid_rl_centralized.py 
    Source Location: edited from ~/flow/examples/exp_configs/rl/multiagent\
        Sets simulation parameters for a rl experiment. To train this file, in the ~/flow/examples directory, run:
    ###### $ python train.py --exp_config  grid_rl_centralized
   
### decentralized directory:
- ####  decentralized_env.py 
    Source Location: edited from ~/flow/flow/envs\
    Contains gym compatible environment class and methods for decentralized experiments. Decentralized experiments return obersevations, actions and rewards as dictionaries with agent ids as keys and trained that way. The class called has all the implemented methods that called the benchmark classes (eg in presslight.py, thesis.py) to set the observation and action spaces, collect states, compute rewards, and step functions.
- ####  grid_rl_decentralized.py 
    Source Location: edited from ~/flow/examples/exp_configs/rl/multiagent\
        Sets simulation parameters for a rl experiment. To train this file, in the ~/flow/examples directory, run:
    ###### $ python train.py --exp_config  grid_rl_decentralized

### single_agent directory:    
- #### __init__.py 
    Source Location: edited from ~/flow/flow/envs\
    Registers the creates environments for FLOW to use. Contains Centralized environment.
    
### multi_agent directory:
- ####  __init__.py Source Location: 
    Source Location: edited from ~/flow/flow/envs/multiagent\
    Registers the creates environments for FLOW to use. Contains Decentralized environment.
    
### utils_and_source_code_edits directory:

#### /simulation:
- #### traci.py 
    Source Location: edited from ~/flow/flow/core/kernel/simulation\
    This file contains code/methods that are shared amongst environments in FLOW source code. The lines of interest are 119 to 122. These lines enable SUMO to output an xml containing trip infos that we collect travel times from.
    
    ##### Note (TODO) : this file file edition overwrites emission output path with trip-info output. (line ~121). Adding a seperate trip-info output path to flow would be difficult (without changed flow's source code) at this time.

#### /training:
- #### train.py 
    Source Location: edited from ~/flow/examples\
    This file contains training hyperparameters (such as Neural Network configurations, learning rate, etc)  and spawns a training session for an experiment.

#### /utils:
- ####  traffic_light_utils.py 
    Source Location: edited from ~/flow/flow/core\
    This file contains helper functions that are imported and used in the benchmark classes and environment classes.


## Visualizing trained RL policies
In order to visualize the policies, from the ~/flow/flow/visualize directory, run:
###### $ python visualizer_rllib.py --result_dir "result_dir here" --checkpoint_num "checkpoint_num here"
where "checkpoint_num here" and "result_dir here" correspond to the checkpoint number we are trying to visualize and the directory containing the trained policy respectively. 
