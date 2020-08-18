## File Descriptions:
- ####  centralized_env.py 
    Source Location: edited from ~/flow/flow/envs\
    Contains gym compatible environment class and methods for centralized experiments. Centralized experiments concatenate all obersevations into a single array and trained that way. The class called has all the implemented methods that called the benchmark classes (eg in presslight.py, thesis.py) to set the observation and action spaces, collect states, compute rewards, and step functions.
- ####  grid_rl_centralized.py 
    Source Location: edited from ~/flow/examples/exp_configs/rl/multiagent\
        Sets simulation parameters for a rl experiment. To train this file, in the ~/flow directory, run:
    
    ```shell
        python examples/train.py --exp_config  grid_rl_centralized
    ```
