#!/bin/bash
#copy changed files in Gilbert_code/ to ~/flow/...

#envs
cp Gilbert_code/single_agent/__init__.py ~/flow/flow/envs
cp Gilbert_code/multi_agent/__init__.py ~/flow/flow/envs/multiagent

#environments
cp Gilbert_code/centralized/centralized_env.py ~/flow/flow/envs
cp Gilbert_code/decentralized/decentralized_env.py ~/flow/flow/envs/multiagent

#benchmarks and params
cp Gilbert_code/benchmarks/benchmark_params.py ~/flow/flow/core
cp Gilbert_code/benchmarks/thesis.py ~/flow/flow/envs
cp Gilbert_code/benchmarks/presslight.py ~/flow/flow/envs

#experiments
#non-rl
cp Gilbert_code/benchmarks/grid_simulation_non_rl.py ~/flow/examples/exp_configs/non_rl
#rl
cp Gilbert_code/centralized/grid_rl_centralized.py ~/flow/examples/exp_configs/rl/multiagent
cp Gilbert_code/decentralized/grid_rl_decentralized.py ~/flow/examples/exp_configs/rl/multiagent

#utils and source_code_edits
cp Gilbert_code/utils_and_source_code_edits/simulation/traci.py ~/flow/flow/core/kernel/simulation/
cp Gilbert_code/utils_and_source_code_edits/training/train.py ~/flow/examples/

#changes added
cp Gilbert_code/utils_and_source_code_edits/utils/traffic_light_utils.py  ~/flow/flow/core


#tests
cp Gilbert_code/tests/test_traffic_light_grid.py  ~/flow/tests/fast_tests/