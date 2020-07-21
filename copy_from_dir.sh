#!/bin/bash
#copy changed files in Gilbert_code/ to ~/flow/...
#envs
cp Gilbert_code/single_agent/traffic_light_grid.py ~/flow/flow/envs
cp Gilbert_code/single_agent/__init__.py ~/flow/flow/envs
cp Gilbert_code/multi_agent/traffic_light_grid.py ~/flow/flow/envs/multiagent
cp Gilbert_code/multi_agent/__init__.py ~/flow/flow/envs/multiagent

#changes added
cp Gilbert_code/single_agent/presslight_single_agent.py ~/flow/flow/envs
cp Gilbert_code/multi_agent/presslight_multi_agent.py ~/flow/flow/envs/multiagent

#experiments
#non-rl
cp Gilbert_code/single_agent/grid1x1.py ~/flow/examples/exp_configs/non_rl
cp Gilbert_code/multi_agent/grid1x3.py ~/flow/examples/exp_configs/non_rl
cp Gilbert_code/multi_agent/grid2x2.py ~/flow/examples/exp_configs/non_rl

#rl
cp Gilbert_code/single_agent/grid1x1_rl.py ~/flow/examples/exp_configs/rl/singleagent
cp Gilbert_code/multi_agent/grid1x3_rl.py ~/flow/examples/exp_configs/rl/multiagent
cp Gilbert_code/multi_agent/grid2x2_rl.py ~/flow/examples/exp_configs/rl/multiagent

#utils and source_code_edits
cp Gilbert_code/utils_and_source_code_edits/util.py ~/flow/flow/core
cp Gilbert_code/utils_and_source_code_edits/traci.py ~/flow/flow/core/kernel/simulation
cp Gilbert_code/utils_and_source_code_edits/train.py ~/flow/examples

#changes added
cp Gilbert_code/utils_and_source_code_edits/presslight_utils.py ~/flow/flow/core