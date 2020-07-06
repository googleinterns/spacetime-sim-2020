#!/bin/bash
#copy changed files in flow/ to Gilbert_code/
#envs
cp Gilbert_code/single_agent ~/flow/flow/envs/traffic_light_grid.py
cp Gilbert_code/single_agent ~/flow/flow/envs/__init__.py
cp Gilbert_code/multi_agent ~/flow/flow/envs/multiagent/traffic_light_grid.py
cp Gilbert_code/multi_agent ~/flow/flow/envs/multiagent/__init__.py

#experiments
#non-rl
cp Gilbert_code/single_agent ~/flow/examples/exp_configs/non_rl/grid1x1.py
cp Gilbert_code/multi_agent ~/flow/examples/exp_configs/non_rl/grid1x3.py
cp Gilbert_code/multi_agent ~/flow/examples/exp_configs/non_rl/grid2x2.py

#rl
cp Gilbert_code/single_agent ~/flow/examples/exp_configs/rl/singleagent/grid1x1_rl.py
cp Gilbert_code/multi_agent ~/flow/examples/exp_configs/rl/multiagent/grid1x3_rl.p
#cp Gilbert_code/multi_agent ~/flow/examples/exp_configs/rl/multiagent/grid2x2_rl.py

#utils and source_code_edits
cp Gilbert_code/utils_and_source_code_edits ~/flow/flow/core/util.py
cp Gilbert_code/utils_and_source_code_edits ~/flow/flow/core/kernel/simulation/traci.py
