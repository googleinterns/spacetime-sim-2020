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

After installing FLOW, ensure the ["spacetime-sim-2020"](https://github.com/googleinterns/spacetime-sim-2020.git) repo is cloned in the home directory (user root directory) too by running:
   ```shell
        git clone https://github.com/googleinterns/spacetime-sim-2020.git
   ```

# Files in this Repo
The files located in the directory Gilbert_code correspond to the files that interact with the FLOW's source code in order implement our traffic light optimization methods and experiment. These files are copied to different directories within the FLOW codebase in order to run smoothly.

### Note: The FLOW's source code files that were explicitly edited are included in this repo and are; 
 - \_\_init\_\_.py located in [single_agent](https://github.com/googleinterns/spacetime-sim-2020/tree/master/Gilbert_code/single_agent)
 - \_\_init\_\_.py located in [multi_agent](https://github.com/googleinterns/spacetime-sim-2020/tree/master/Gilbert_code/multi_agent)
 - traci.py located in [simulation](https://github.com/googleinterns/spacetime-sim-2020/tree/master/Gilbert_code/utils_and_source_code_edits/simulation)
 - train.py located in [training](https://github.com/googleinterns/spacetime-sim-2020/tree/master/Gilbert_code/utils_and_source_code_edits/training)

The above files overwrite FLOW's source code in order to ensure our experiments are able to run.

# How to use example:
This is a summarized guide on on to use this repo and run it's files (ensure your flow conda env is acticvated by running:  ```conda activate flow```).

In your root directory, run:
   ```shell
        cd spacetime-sim-2020
        sh spacetime-sim-2020/copy_from_dir.sh
        cd ../flow
        python examples/train.py --exp_config grid_rl_centralized
   ```

To visualize tensorboard logging while training, run:
     
```shell
    tensorboard --logdir=~/ray_results/
```

When training is finished, to visualize policy, run:
    
```shell
    python flow/visualize/visualizer_rllib.py --result_dir "result_dir here" --checkpoint_num "checkpoint_num here"
```
where ```checkpoint_num here``` and ```result_dir here ``` correspond to the checkpoint number we are trying to visualize and the directory containing the trained policy respectively(found in ```~/ray_results ```). 

To run a non-rl simulation (no training), run:
```shell
    python examples/simulate.py --exp_config grid_simulation_non_rl
```

#  Detailed Description of Files and Functionality
## In the root directory:
- ####  copy_to_dir.sh
   This is scripts copies all the necessary files/code from FLOW source code into Gilbert_code in order to keep track of the changes made.
- #### copy_from_dir.sh:
    This is scripts copies all the necessary files/code from Gilbert_code into into FLOW source code in order to run.

## In Gilbert_code directory:
### 1. [benchmarks](https://github.com/googleinterns/spacetime-sim-2020/tree/master/Gilbert_code/benchmarks)
### 2. [centralized](https://github.com/googleinterns/spacetime-sim-2020/tree/master/Gilbert_code/centralized)
### 3. [decentralized](https://github.com/googleinterns/spacetime-sim-2020/tree/master/Gilbert_code/decentralized)
### 4. [single_agent](https://github.com/googleinterns/spacetime-sim-2020/tree/master/Gilbert_code/single_agent)    
### 5. [multi_agent](https://github.com/googleinterns/spacetime-sim-2020/tree/master/Gilbert_code/multi_agent)
### 6. [utils_and_source_code_edits](https://github.com/googleinterns/spacetime-sim-2020/tree/master/Gilbert_code/utils_and_source_code_edits)
   - #### [simulation](https://github.com/googleinterns/spacetime-sim-2020/tree/master/Gilbert_code/utils_and_source_code_edits/simulation)
   - #### [training](https://github.com/googleinterns/spacetime-sim-2020/tree/master/Gilbert_code/utils_and_source_code_edits/training)
   - #### [utils](https://github.com/googleinterns/spacetime-sim-2020/tree/master/Gilbert_code/utils_and_source_code_edits/utils)

## Important Info about visualizing trained RL policies:
After training, if ``` "tls=tl_logic" ``` was passed into into ``` flow_params ``` in the configuration files at teh start of the training session, ensure that this command is renamed or removed in ``` ~/ray_results/../../params.json``` of the trained policy. \
   - This will ensure any SUMO default actions are NOT performed. (ie. all actions being visualized are purely from the trained agent).

