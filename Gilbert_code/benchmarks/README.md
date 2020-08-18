## File Descriptions:
- #### benchmark_params.py
    Source Location: edited from ~/flow/flow/core\
    Contains parameters mainly logging and naming parameters for experiment. The class containined in this file is initialized in the the init statement in centrelized_env.py described below. \ Note: self.look_ahead and self.sumo_actuated_baseline are the only paramters that affect training.

- #### grid_simulation_non_rl.py
    Source Location: edited from ~//flow/examples/exp_configs/non_rl\
    Sets simulation parameters for a non-rl experiment. This environment spawns and renders a SUMO simulation. Traffic light control can either be SUMO inbuilt policies or pre-assigned phases timing plans. To run this file, in the ~/flow directory, run:
    ```shell
        python examples/simulate.py --exp_config grid_simulation_non_rl
    ```

- #### presslight.py
    Source Location: edited from ~/flow/flow/envs\
    Contains observations and reward functions implementations of Presslight benchmark. If used, the class containined in this file is initialized in the the init statement in centrelized_env.py 

- #### thesis.py
    Source Location: edited from ~/flow/flow/envs\
    Contains observations and reward functions implementations of thesis benchmark. If used, the class containined in this file is initialized in the the init statement in centrelized_env.py 
