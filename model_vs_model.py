"""
Pit two models together on NHL 94
"""

import retro
import sys
import argparse
import logging
import numpy as np
import pygame
from stable_baselines import logger

from common import init_env, init_model, init_play_env, get_num_parameters

def parse_cmdline(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--p1_alg', type=str, default='ppo2')
    parser.add_argument('--p2_alg', type=str, default='ppo2')
    #parser.add_argument('--nn', type=str, default='CnnPolicy')
    parser.add_argument('--model1_desc', type=str, default='CNN')
    parser.add_argument('--model2_desc', type=str, default='MLP')
    parser.add_argument('--env', type=str, default='NHL941on1-Genesis')
    parser.add_argument('--state', type=str, default=None)
    parser.add_argument('--num_players', type=int, default='2')
    parser.add_argument('--num_env', type=int, default=1)
    parser.add_argument('--num_timesteps', type=int, default=0)
    parser.add_argument('--output_basedir', type=str, default='~/OUTPUT')
    parser.add_argument('--load_p1_model', type=str, default='')
    parser.add_argument('--load_p2_model', type=str, default='')
    parser.add_argument('--display_width', type=int, default='1440')
    parser.add_argument('--display_height', type=int, default='810')
    parser.add_argument('--deterministic', default=True, action='store_true')

    args = parser.parse_args(argv)

    logger.log("=========== Params ===========")
    logger.log(argv[1:])

    return args


def main(argv):

    args = parse_cmdline(argv[1:])
    
    logger.log('========= Init =============')
    play_env, uw_display_env = init_play_env(args)
    p1_env = init_env(None, 1, None, 1, args)
    p2_env = init_env(None, 1, None, 1, args)
    
    p1_model = init_model(None, args.load_p1_model, args.p1_alg, args, p1_env)
    p2_model = init_model(None, args.load_p2_model, args.p2_alg, args, p2_env)

    uw_display_env.model1_params = get_num_parameters(p1_model)
    uw_display_env.model2_params = get_num_parameters(p2_model)

    logger.log('========= Start Play Loop ==========')

    state = play_env.reset()

    p1_actions = []
    p2_actions = []

    while True:
        p1_actions = p1_model.predict(state)
        p2_actions = p2_model.predict(state)

        uw_display_env.p1_action_probabilities = p1_model.action_probability(state)
        uw_display_env.p2_action_probabilities = p2_model.action_probability(state)

            
        actions2 = np.append(p1_actions[0], p2_actions[0])

        state, reward, done, info = play_env.step(actions2)

        if done:
            state = play_env.reset()
    

if __name__ == '__main__':
    main(sys.argv)