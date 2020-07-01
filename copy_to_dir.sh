#!/bin/bash
#copy changed files in flow/ to Gilbert_code/
#envs
cp ../flow/envs/traffic_light_grid.py Gilbert_code/single_agent
cp ../flow/envs/__init__.py Gilbert_code/single_agent
cp ../flow/envs/multiagent/traffic_light_grid.py Gilbert_code/multi_agent
cp ../flow/envs/multiagent/__init__.py Gilbert_code/multi_agent

#experiments
#non-rl
cp ../examples/exp_configs/non_rl/grid1x1.py Gilbert_code/single_agent
cp ../examples/exp_configs/non_rl/grid1x3.py Gilbert_code/multi_agent
cp ../examples/exp_configs/non_rl/grid2x2.py Gilbert_code/multi_agent

#rl
cp ../examples/exp_configs/rl/singleagent/grid1x1_rl.py Gilbert_code/single_agent
cp ../examples/exp_configs/rl/multiagent/grid1x3_rl.py Gilbert_code/multi_agent
#cp ../examples/exp_configs/rl/multiagent/grid2x2_rl.py Gilbert_code/multi_agent

#utils and source_code_edits
cp ../flow/core/util.py Gilbert_code/utils_and_source_code_edits
cp ../flow/core/kernel/simulation/traci.py Gilbert_code/utils_and_source_code_edits
