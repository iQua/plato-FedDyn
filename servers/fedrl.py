"""
A federated server with a reinforcement learning agent.
This federated server uses reinforcement learning
to tune the number of local aggregations on edge servers.
"""

import logging
import asyncio
import sys
import time
import os

from config import Config
from servers import FedAvgCrossSiloServer
import servers
from utils import csv_processor

FLServer = FedAvgCrossSiloServer
if Config().rl:
    from stable_baselines3.common.env_checker import check_env
    from rl_envs import FLEnv

    # The central server of FL
    FLServer = {
        "fedavg_cross_silo": servers.fedavg_cs.FedAvgCrossSiloServer
    }[Config().rl.fl_server]


class FedRLServer(FLServer):
    """Federated server using RL."""
    def __init__(self):
        super().__init__()

        self.rl_env = FLEnv(self)
        self.rl_episode = 0
        self.rl_tuned_para_value = None
        self.rl_state = None
        self.is_rl_tuned_para_got = False
        self.is_rl_episode_done = False
        self.rl_episode_start_time = None
        self.cumulative_reward = 0

        # An RL agent waits for the event that the tuned parameter
        # is passed from RL environment
        self.rl_tuned_para_got = asyncio.Event()

        # Indicate if server response (contains RL tuned para that will be sent to edge servers)
        # is generated in this RL time step, as customize_server_response() is called everytime
        # when the central server sends response to one edge server,
        # but the response only need to be generated once in an RL time step (an FL round)
        self.generated_server_response = False

        # An RL agent waits for the event that RL environment is reset to aviod
        # directly starting a new time step after the previous episode ends
        self.new_episode_begin = asyncio.Event()

        # Since RL training (function start_rl()) runs as a coroutine
        # It needs to wait until wrap_up_an_episode() is complete to start a new RL episode
        self.wrapped_previous_episode = asyncio.Event()
        self.wrapped_previous_episode.set()

        if Config().results:
            self.rl_recorded_items = [
                'episode', 'cumulative_reward', 'rl_training_time'
            ]
            # Directory of results (figures etc.)
            result_dir = f'./results/{Config().trainer.dataset}/{Config().trainer.model}/{Config().server.type}/'
            result_csv_file = result_dir + 'result_rl.csv'
            csv_processor.initialize_csv(result_csv_file,
                                         self.rl_recorded_items, result_dir)

    def configure(self):
        """
        Booting the RL agent and the FL server
        """
        logging.info('Configuring a RL agent and a %s server...',
                     Config().rl.fl_server)
        logging.info(
            "This RL agent will tune the number of aggregations on edge servers."
        )

        total_episodes = Config().rl.episodes
        target_reward = Config().rl.target_reward

        if target_reward:
            logging.info('RL Training: %s episodes or %s%% reward\n',
                         total_episodes, 100 * target_reward)
        else:
            logging.info('RL Training: %s episodes\n', total_episodes)

    def start_clients(self, as_server=False):
        """Start all clients and RL training."""
        super().start_clients(as_server)

        # The starting point of RL training
        # Run RL training as a coroutine
        if not as_server:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(asyncio.gather(self.start_rl()))

    def start_rl(self):
        """The starting point of RL training."""
        # Test the environment of reinforcement learning.
        #self.check_with_sb3_env_checker(FedRLServer.rl_env)
        FedRLServer.try_a_random_agent(self.rl_env)

    def reset_rl_env(self):
        """Reset the RL environment at the beginning of each episode."""
        current_loop = asyncio.get_event_loop()
        task = current_loop.create_task(self.wrapped_previous_episode.wait())
        current_loop.run_until_complete(task)
        self.wrapped_previous_episode.clear()

        # The number of finished FL training round
        self.current_round = 0

        self.is_rl_episode_done = False
        self.cumulative_reward = 0
        self.rl_episode_start_time = time.time()

        self.rl_episode += 1
        logging.info('\nRL Agent: Starting episode %s...', self.rl_episode)

        # Configure the FL central server
        super().configure()

        # starting time of a gloabl training round
        self.round_start_time = 0

    async def wrap_up(self):
        """Wrapping up when one RL time step (one FL round) is done."""
        self.generated_server_response = False
        # Get the RL state
        # Use accuracy as state for now
        self.rl_state = self.accuracy
        target_accuracy = Config().trainer.target_accuracy

        if target_accuracy and self.accuracy >= target_accuracy:
            logging.info('Target accuracy of FL reached.')
            self.is_rl_episode_done = True

        if self.current_round >= Config().trainer.rounds:
            logging.info('Target number of FL training rounds reached.')
            self.is_rl_episode_done = True

        # Pass the RL state to the RL env
        self.rl_env.get_state(self.rl_state, self.is_rl_episode_done)

        # Give RL env some time to finish step() before FL select clients to start next round
        await self.rl_env.step_done.wait()
        self.rl_env.step_done.clear()

        if self.is_rl_episode_done:
            await self.wrap_up_an_episode()

    async def customize_server_response(self, server_response):
        """Wrap up generating the server response with any additional information."""
        if not self.generated_server_response:
            await self.update_rl_tuned_parameter()
            self.generated_server_response = True
        server_response['fedrl'] = Config().cross_silo.rounds
        server_response['current_global_round'] = self.current_round
        print("CURRENT GLOBAL ROUND", self.current_round)
        return server_response

    async def update_rl_tuned_parameter(self):
        """
        Wait for getting RL tuned parameter from env,
        and update this parameter in Config().
        """
        await self.rl_tuned_para_got.wait()
        self.rl_tuned_para_got.clear()

        Config().cross_silo = Config().cross_silo._replace(
            rounds=self.rl_tuned_para_value)

    def get_tuned_para(self, rl_tuned_para_value, time_step):
        """
        Get tuned parameter from RL env.
        This function is called by RL env.
        """
        assert time_step == self.current_round + 1
        self.rl_tuned_para_value = rl_tuned_para_value
        # Signal the RL agent that it gets the tuned parameter
        self.rl_tuned_para_got.set()
        print("RL Agent: Get tuned para of time step", time_step)

    async def wrap_up_an_episode(self):
        """Wrapping up when one RL episode (the FL training) is done."""
        if Config().results:
            new_row = []
            for item in self.rl_recorded_items:
                item_value = {
                    'episode': self.rl_episode,
                    'cumulative_reward': self.cumulative_reward,
                    'rl_training_time':
                    time.time() - self.rl_episode_start_time
                }[item]
                new_row.append(item_value)

            result_dir = f'./results/{Config().trainer.dataset}/{Config().trainer.model}/{Config().server.type}/'
            result_csv_file = result_dir + 'result_rl.csv'
            csv_processor.write_csv(result_csv_file, new_row)
        self.wrapped_previous_episode.set()

        if self.rl_episode >= Config().rl.episodes:
            if Config().results:
                # Delete the csv file created when edge servers called super().__init__() as it is useless
                os.remove(
                    f'./results/{Config().trainer.dataset}/{Config().trainer.model}/{Config().rl.fl_server}/result.csv'
                )

            logging.info(
                'RL Agent: Target number of training episodes reached.')
            await self.close_connections()
            sys.exit()
        else:
            # Wait until RL env resets and starts a new RL episode
            self.new_episode_begin.clear()
            await self.new_episode_begin.wait()
            self.new_episode_begin.clear()

    @staticmethod
    def check_with_sb3_env_checker(env):
        """
        Use helper provided by stable_baselines3
        to check that the environment runs without error.
        """
        # It will check the environment and output additional warnings if needed
        check_env(env)

    @staticmethod
    def try_a_random_agent(env):
        """Quickly try a random agent on the environment."""
        # pylint: disable=unused-variable
        obs = env.reset()
        episodes = Config().rl.episodes
        n_steps = Config().trainer.rounds

        for i in range(episodes):
            for _ in range(n_steps):
                # Random action
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                if done:
                    if i < episodes:
                        obs = env.reset()
                    break