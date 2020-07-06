**This is not an officially supported Google product.** 
Traffic light optimization using sumo.

# Traffic Light Optimization using Deep-RL
In this project, we study and implemetent traffic light optimaization techniques using a Deep-Reinforcment Learning Benchmarking tool, FLOW and a traffic simulator, SUMO.
FLOW (https://flow-project.github.io/)  was developed by the Mobile Sensing Laboratory at the University of California, Berkeley with the goal of providing structured and scalable RL optimization tools in traffic management for autonmous vehicles and traffic lights. 
SUMO (https://sumo.dlr.de/docs/index.html) is an open source, highly portable, microscopic and continuous traffic simulation package designed to handle large networks. 

# Baselines Implemented
In the Project, we are implement 2 Traffic Light Optimization Baselines:
1.PressLight: Learning Max Pressure Control to Coordinate Traffic Signals in Arterial Network
https://dl.acm.org/doi/10.1145/3292500.3330949
https://github.com/wingsweihua/presslight

2.Coordinated Deep Reinforcement Learners for Traffic Light Control 
https://esc.fnwi.uva.nl/thesis/centraal/files/f632158773.pdf

# Installation
In order to begin implementing the code, the user needs to install FLOW in their home directory (user root directory). Please follow the instructions provided in https://github.com/flow-project/flow to install and begin using FLOW.
The intructions will guide the user to install SUMO and rllib which are important pieces of the project too.

# Files in this Repo
The files located in the directory Gilbert_code correspond to the edited files from the FLOW source code needed to implement our traffic light optimization methods and experiment. These files are located in different places within the FLOW source code.
##  Description of Files and Functionality + How to Run
### In the root directory:
#### copy_to_dir.sh
This is scripts copies all the necessary files/code from FLOW source code into Gilbert_code in order to keep track of the changes made.
#### copy_from_dir.sh: [NOT implemented yet]
This is scripts copies all the necessary files/code from Gilbert_code into into FLOW source code in order to run.

### In Gilbert_code directory:
Single_agent directory:
#### traffic_light_grid.py
From where: edited from ~/flow/flow/envs
Contains gym environment for a 1x1 single intersection experiment. The class called MyGridEnv has all the implemented methods that set the obersavation and action spaces, collects states, computes rewards, and logs rewards and travel time on tensoboard.
#### __init__.py 
From where: edited from ~/flow/flow/envs
Registers the created environments for FLOW to use
#### grid1x1.py -
From where: edited from ~/flow/examples/exp_configs/non_rl
Sets parameters for single intersection non-rl experiment. Traffic light control can either be SUMO inbuilt policies or pre-assigned phases timing plans. To run this file, in the flow/examples directory, run:
$ python simulatate.py grid1x1

#### grid1x1_rl.py 
From where: edited from ~/flow/examples/exp_configs/rl
Sets parameters for single intersection rl experiment. Traffic light control can either be SUMO inbuilt policies or pre-assigned phases timing plans. To run this file, in the ~/flow/examples directory, run:
$ python train.py grid1x1_rl

Multi_agent directory:
#### traffic_light_grid.py 
From where: edited from ~/flow/flow/envs/multiagent
Contains gym environment for a 1x3 and 2x2 intersection experiments. The class called MultiTrafficLightGridPOEnvPL has all the implemented methods that set the obersavation and action spaces, collects states, computes rewards, and logs rewards and travel time on tensoboard.
#### __init__.py From where: 
From where: edited from ~/flow/flow/envs/multiagent
Registers the created environments for FLOW to use
#### grid1x3.py 
From where: edited from ~/flow/examples/exp_configs/non_rl
similar purpose to grid1x1 above but for 1x3 multi-agent scenario, to run this file, in the flow/examples directory, run:
$ python simulatate.py grid1x3
#### grid2x2.py 
From where: edited from ~/flow/examples/exp_configs/non_rl
similar purpose to grid1x1 above but for 2x2 multi-agent scenario, to run this file, in the flow/examples directory, run:
$ python simulatate.py grid2x2
#### grid1x3_rl.py 
From where:  edited from ~/flow/examples/exp_configs/rl
similar purpose to grid1x1_rl above but for 1x3 multi-agent scenario, to run this file, in the flow/examples directory, run:
$ python simulatate.py grid1x3_rl
#### grid2x2_rl.py 
From where: edited from ~/flow/examples/exp_configs/rl
similar purpose to grid1x1_rl above but for 2x2 multi-agent scenario, to run this file, in the flow/examples directory, run:
$ python simulatate.py grid2x2_rl

utils_and_source_code_edits directory:
#### util.py 
From where: edited from ~/flow/flow/core
This file contains code/methods that are shared amongst environemnts such as logging ot travel times.
### traci.py 
From where: edited from ~/flow/flow/core/kernel/simulation
This file contains code/methods that are shared amongst environemnts but are deeper into FLOW source code. The lines of interest are 119 to 122. These lines enable SUMO to output an xml containing trip_infos that we collect travel times from.

## Visualizing trained RL policies
In order to viualize the policies, from in the ~/flow/flow/visualize run:
$ python visualizer_rllib.py --result_dir "result_dir here" --checkpoint_num "checkpoint_num here"
where "checkpoint_num here" and "result_dir here" correspond to the checkpoint number we are trying to visualize and the directory containing the trained policy respectively.


