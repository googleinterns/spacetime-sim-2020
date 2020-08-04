#!/bin/bash
#copy changed files in flow/ to Gilbert_code/
#envs
#cp ~/flow/flow/envs/traffic_light_grid.py Gilbert_code/single_agent
cp ~/flow//flow/envs/__init__.py Gilbert_code/single_agent
#cp ~/flow/flow/envs/multiagent/traffic_light_grid.py Gilbert_code/multi_agent
cp ~/flow/flow/envs/multiagent/__init__.py Gilbert_code/multi_agent

#changes added
#cp ~/flow/flow/envs/presslight_single_agent.py Gilbert_code/single_agent
#cp ~/flow/flow/envs/multiagent/presslight_multi_agent.py Gilbert_code/multi_agent
cp ~/flow/flow/envs/centralized_multi_agent_thesis.py Gilbert_code/single_agent
cp ~/flow/flow/envs/centralized_single_agent_presslight.py Gilbert_code/single_agent
cp ~/flow/flow/envs/multiagent/decentralized_multi_light_presslight.py Gilbert_code/multi_agent
cp ~/flow/flow/envs/multiagent/decentralized_multi_light_thesis.py Gilbert_code/multi_agent

#experiments
#non-rl
cp ~/flow/examples/exp_configs/non_rl/grid1x1.py Gilbert_code/single_agent
cp ~/flow/examples/exp_configs/non_rl/grid1x3.py Gilbert_code/multi_agent
cp ~/flow/examples/exp_configs/non_rl/grid2x2.py Gilbert_code/multi_agent

#rl
cp ~/flow/examples/exp_configs/rl/singleagent/grid1x1_rl.py Gilbert_code/single_agent
cp ~/flow/examples/exp_configs/rl/multiagent/grid1x3_rl.py Gilbert_code/multi_agent
cp ~/flow/examples/exp_configs/rl/multiagent/grid2x2_rl.py Gilbert_code/multi_agent

#utils and source_code_edits
cp ~/flow/flow/core/util.py Gilbert_code/utils_and_source_code_edits/utils/
cp ~/flow/flow/core/kernel/simulation/traci.py Gilbert_code/utils_and_source_code_edits/simulation/
cp ~/flow/flow/core/kernel/network/traci.py Gilbert_code/utils_and_source_code_edits/network/
cp ~/flow/examples/train.py Gilbert_code/utils_and_source_code_edits/training/

#changes added
cp ~/flow/flow/core/traffic_light_utils.py Gilbert_code/utils_and_source_code_edits/utils