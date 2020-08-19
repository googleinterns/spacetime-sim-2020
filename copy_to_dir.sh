#!/bin/bash
#copy changed files in flow/ to Gilbert_code/
#envs
cp ~/flow/flow/envs/__init__.py Gilbert_code/single_agent
cp ~/flow/flow/envs/multiagent/__init__.py Gilbert_code/multi_agent

#environments
cp ~/flow/flow/envs/centralized_env.py Gilbert_code/centralized
cp ~/flow/flow/envs/multiagent/decentralized_env.py Gilbert_code/decentralized

#benchmarks and params
cp ~/flow/flow/core/benchmark_params.py Gilbert_code/benchmarks
cp ~/flow/flow/envs/thesis.py Gilbert_code/benchmarks
cp ~/flow/flow/envs/presslight.py Gilbert_code/benchmarks

#experiments
#non-rl
cp ~/flow/examples/exp_configs/non_rl/grid_simulation_non_rl.py Gilbert_code/benchmarks
#rl
cp ~/flow/examples/exp_configs/rl/multiagent/grid_rl_centralized.py Gilbert_code/centralized
cp ~/flow/examples/exp_configs/rl/multiagent/grid_rl_decentralized.py Gilbert_code/decentralized

#utils and source_code_edits
cp ~/flow/flow/core/kernel/simulation/traci.py Gilbert_code/utils_and_source_code_edits/simulation/
cp ~/flow/examples/train.py Gilbert_code/utils_and_source_code_edits/training/

#changes added
cp ~/flow/flow/core/traffic_light_utils.py Gilbert_code/utils_and_source_code_edits/utils