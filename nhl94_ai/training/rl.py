#!/usr/bin/env python3
"""
Train a Model on NHL 94
"""
from nhl94_ai.agents.registry import CONTROLLERS

from nhl94_ai.artifacts import save_checkpoint

from nhl94_ai.config import default_config_path
import os
import sys
import time
import argparse
from nhl94_ai.training.logging import get_model_file_name, com_print, init_logger, create_output_dir
from nhl94_ai.models.factory import init_model
from nhl94_ai.env.factory import init_env
from nhl94_ai.config import load_hyperparams, resolve_hyperparams_for_model


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--alg', type=str, default='ppo2')
    parser.add_argument('--nn', type=str, default='CnnPolicy')
    parser.add_argument('--nnsize', type=int, default='256')
    parser.add_argument('--env', type=str, default='NHL941on1-Genesis-v0')
    parser.add_argument('--state', type=str, default=None)
    parser.add_argument('--num_players', type=int, default='1')
    parser.add_argument('--num_env', type=int, default=24)
    parser.add_argument('--num_timesteps', type=int, default=6000000)
    parser.add_argument('--output_basedir', type=str, default='~/OUTPUT')
    parser.add_argument('--load_p1_model', type=str, default='')
    parser.add_argument('--load_opponent_model', type=str, default='')
    parser.add_argument('--display_width', type=int, default='1440')
    parser.add_argument('--display_height', type=int, default='810')
    parser.add_argument('--alg_verbose', default=True, action='store_true')
    parser.add_argument('--info_verbose', default=True, action='store_true')
    parser.add_argument('--play', default=False, action='store_true')
    parser.add_argument('--rf', type=str, default='')
    parser.add_argument('--deterministic', default=True, action='store_true')
    parser.add_argument('--hyperparams', type=str, default=default_config_path("default"))
    parser.add_argument('--selfplay', default=False, action='store_true')
    parser.add_argument('--selfplay_role', type=str, default='offense', choices=['offense', 'defense'])
    parser.add_argument('--seq_len', type=int, default=16,
                       help='Frame history length for temporal policies such as HybridMambaPolicy or GRUMlpPolicy')

    parser.add_argument('--action_type', type=str, default='FILTERED',
                       choices=['FILTERED', 'DISCRETE', 'MULTI_DISCRETE', 'HOCKEY_INTENT_DPAD', 'TARGET_POSITION'],
                       help='Buttons, hockey intents, or target-only positioning')

    parser.add_argument("--seed", type=int, default=0, help="Environment random seed")
    parser.add_argument("--policy_seed", type=int, default=None, help="Optional PPO initialization seed")
    return parser


def parse_cmdline(argv):
    args = build_parser().parse_args(argv)

    #if args.info_verbose is False:
    #    logger.set_level(logger.DISABLED)

    return args

class ModelTrainer:
    def __init__(self, args, logger):
        self.args = args

        #if self.args.alg_verbose:
        #    logger.log('========= Init =============')

        self.output_fullpath = create_output_dir(args)
        model_savefile_name = get_model_file_name(args)
        self.model_savepath = os.path.join(self.output_fullpath, model_savefile_name)

        self.env = init_env(self.output_fullpath, args.num_env, args.state, args.num_players, args, args.hyperparams_dict)

        try:
            self.p1_model = init_model(self.output_fullpath, args.load_p1_model, args.alg, args, self.env, logger, args.hyperparams_dict)
        except BaseException:
            self.env.close()
            raise

        #if self.args.alg_verbose:
        com_print('OUTPUT PATH:   %s' % self.output_fullpath)
        com_print('ENV:           %s' % args.env)
        com_print('STATE:         %s' % args.state)
        com_print('NN:            %s' % args.nn)
        com_print('ALGO:          %s' % args.alg)
        com_print('NUM TIMESTEPS: %s' % args.num_timesteps)
        com_print('NUM ENV:       %s' % args.num_env)
        com_print('NUM PLAYERS:   %s' % args.num_players)

        print(self.env.observation_space)

    def train(self, callback=None):
        com_print('========= Start Training ==========')

        if self.args.nn in CONTROLLERS:
            raise NotImplementedError(f'{self.args.nn} is inference-only. Use nhl94 play to run it.')

        if self.args.alg == 'es':
            self.p1_model.train(num_generations=500)
        else:
            self.p1_model.learn(total_timesteps=self.args.num_timesteps, callback=callback)

        com_print('========= End Training ==========')

        save_checkpoint(self.p1_model, self.model_savepath)
        com_print('Model saved to:%s' % self.model_savepath)

        return self.model_savepath

    def play(self, args, continuous=True):
        #if self.args.alg_verbose:
        com_print('========= Start Play Loop ==========')

        # Special case of ES
        if self.args.alg == 'es':
            final_reward = self.p1_model.evaluate(render=True, num_episodes=5)
            print(f"Final evaluation reward: {final_reward:.2f}")
            return

        state = self.env.reset()
        while True:
            self.env.render(mode='human')

            p1_actions = self.p1_model.predict(state, deterministic=args.deterministic)

            state, reward, done, info = self.env.step(p1_actions[0])
            time.sleep(0.05)
            #print(reward)

            if done[0]:
                state = self.env.reset()

            if not continuous and done is True:
                return info


def main(argv):

    args = parse_cmdline(argv[1:])

    logger = init_logger(args)
    com_print("=========== Params ===========")
    com_print(args)

    args.hyperparams_dict = resolve_hyperparams_for_model(load_hyperparams(
        args.hyperparams,
        required=True,
        base_dir=os.path.dirname(__file__),
    ), args.nn)

    if args.selfplay:
        if not args.rf:
            args.rf = 'SelfPlayOffenseFinetune' if args.selfplay_role == 'offense' else 'SelfPlayDefenseFinetune'
        if not args.load_opponent_model:
            args.load_opponent_model = args.load_p1_model

    trainer = ModelTrainer(args, logger)

    try:
        trainer.train()
        if args.play:
            trainer.play(args)
    finally:
        trainer.env.close()


if __name__ == '__main__':
    main(sys.argv)
