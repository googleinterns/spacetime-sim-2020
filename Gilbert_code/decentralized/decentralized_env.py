"""Multi-agent environments for networks with traffic lights.

These environments are used to train traffic lights to regulate traffic flow
through an n x m traffic light grid.
"""
from flow.envs.multiagent.traffic_light_grid import MultiTrafficLightGridPOEnv
from flow.envs.centralized_env import MultiTrafficLightGridPOEnvTH
from gym.spaces.discrete import Discrete
from flow.core.traffic_light_utils import log_rewards, log_travel_times, get_training_iter

ID_IDX = 1


class MultiTrafficLightGridPOEnvPL(MultiTrafficLightGridPOEnvTH, MultiTrafficLightGridPOEnv):

    """ Inherited for PressLight baseline Implementation

    Multiagent shared model version of TrafficLightGridPOEnv.

    Required from env_params: See parent class

    States
        See parent class

    Actions
        See parent class

    Rewards
        See parent class

    Termination
        See parent class
    """

    def step(self, rl_actions):

        """Advance the environment by one step.

        See parent class

        Parameters
        ----------
        rl_actions : array_like
            an list of actions provided by the rl algorithm

        Returns
        -------
        observation : array_like
            agent's observation of the current environment
        reward : float
            amount of reward associated with the previous state/action pair
        done : bool
            indicates whether the episode has ended
        info : dict
            contains other diagnostic information from the previous action
        """

        # advance simulation
        next_observation, reward, done, infos = MultiTrafficLightGridPOEnv.step(self, rl_actions)

        # log average reward and average travel times if simulation is over
        self.rew_list += list(reward.values())
        if done["__all__"]:
            # current training iteration
            iter_ = get_training_iter(self.benchmark_params.full_path)
            # log average travel time
            log_travel_times(rl_actions, iter_, self.benchmark_params, self.network, self.sim_params, self.step_counter)
            # log average reward
            log_rewards(self.rew_list, rl_actions, self.benchmark_params, iter_, self.step_counter)

        return next_observation, reward, done, infos

    def reset(self, new_inflow_rate=None):

        return MultiTrafficLightGridPOEnv.reset(self, new_inflow_rate)

    def clip_actions(self, rl_actions=None):

        return MultiTrafficLightGridPOEnv.clip_actions(self, rl_actions)

    def compute_reward(self, rl_actions, **kwargs):

        """TODO add for loop here"""
        if rl_actions is None:
            return {}
        rews = {}

        rl_ids = self.k.traffic_light.get_ids()
        if not self.action_dict:
            rl_id_action_dict = rl_actions.items()

        else:
            actions = self.action_dict[rl_actions]
            rl_id_action_dict = zip(rl_ids, actions)

        for rl_id, rl_action in rl_id_action_dict:
            rews[rl_id] = self.benchmark.compute_reward(rl_action,
                                          self.step_counter,
                                          action_dict=self.action_dict,
                                          rl_id=rl_id,
                                          **kwargs)
        return rews

    def _apply_rl_actions(self, rl_actions):
        """
        See parent class.

        Issues action for each traffic light agent.
        """
        for rl_id, rl_action in rl_actions.items():
            i = int(rl_id.split("center")[ID_IDX])
            if self.discrete:
                action = rl_action
            else:
                # convert values less than 0.0 to zero and above to 1. 0's
                # indicate that we should not switch the direction
                action = rl_action > 0.0

            if self.currently_yellow[i] == 1:  # currently yellow
                self.last_change[i] += self.sim_step
                # Check if our timer has exceeded the yellow phase, meaning it
                # should switch to red
                if self.last_change[i] >= self.min_switch_time:
                    if self.direction[i] == 0:
                        self.k.traffic_light.set_state(
                            node_id='center{}'.format(i), state="GrGr")
                    else:
                        self.k.traffic_light.set_state(
                            node_id='center{}'.format(i), state='rGrG')
                    self.currently_yellow[i] = 0
            else:
                if action:
                    if self.direction[i] == 0:
                        self.k.traffic_light.set_state(
                            node_id='center{}'.format(i), state='yryr')
                    else:
                        self.k.traffic_light.set_state(
                            node_id='center{}'.format(i), state='ryry')
                    self.last_change[i] = 0.0
                    self.direction[i] = not self.direction[i]
                    self.currently_yellow[i] = 1

    @property
    def action_space(self):
        """See class definition."""
        return Discrete(2)

