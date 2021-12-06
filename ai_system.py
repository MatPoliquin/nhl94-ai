"""
AI System
"""

import math
from common import init_env, init_model, init_play_env, get_model_file_name, print_model_info, get_num_parameters
from nhl94_const import GameConsts

AI_STATE_IDLE = 0
AI_STATE_ISHOOTING = 1


class AISystem():
    def __init__(self, args, env):
        self.test = 0
        self.args = args
        self.use_model = True
        self.p1_model = None
        self.state = AI_STATE_IDLE
        if args.load_p1_model is '':
            self.use_model = False
        else:
            self.p1_model = init_model(None, args.load_p1_model, args.alg, args, env)


    def GotoTarget(self, p1_actions, target_vec):
        if target_vec[0] > 0:
            p1_actions[GameConsts.INPUT_LEFT] = 1
        else:
            p1_actions[GameConsts.INPUT_RIGHT] = 1

        if target_vec[1] > 0:
            p1_actions[GameConsts.INPUT_DOWN] = 1
        else:
            p1_actions[GameConsts.INPUT_UP] = 1


    def DistToPos(self, vec1, vec2):
        tmp = (vec1[0] - vec2[0])**2 + (vec1[1] - vec2[1])**2
    
        return math.sqrt(tmp)



    def Think(self, info):
        p1_actions = [0] * GameConsts.INPUT_MAX

        p1_x = info.get('p1_x')
        p1_y = info.get('p1_y')
        g1_x = info.get('g1_x')
        g1_y = info.get('g1_y')
        puck_x = info.get('puck_x')
        puck_y = info.get('puck_y')


        pp_vec = [p1_x - puck_x, p1_y - puck_y]
        tmp = (p1_x - puck_x)**2 + (p1_y - puck_y)**2
        pp_dist = math.sqrt(tmp)

        has_puck = True
        if pp_dist > GameConsts.MAX_PLAYER_PUCK_DIST:
            has_puck = False
    
        
        goalie_has_puck = True
        dist = self.DistToPos([g1_x, g1_y], [puck_x, puck_y])
        if dist > GameConsts.MAX_PLAYER_PUCK_DIST:            
            goalie_has_puck = False


        print(goalie_has_puck)
        #print(has_puck)    

        if has_puck:
            dist = self.DistToPos([p1_x, p1_y], [GameConsts.SHOOT_POS_X, GameConsts.SHOOT_POS_Y])

            if dist < 60:
                #self.state = AI_STATE_ISHOOTING
                p1_actions[GameConsts.INPUT_C] = 1
            else:
                self.GotoTarget(p1_actions, [p1_x - GameConsts.SHOOT_POS_X, p1_y - GameConsts.SHOOT_POS_Y])
                print('GOTO SHOOT POSITION')
        elif goalie_has_puck:
            p1_actions[GameConsts.INPUT_B] = 1
            print('GOALIE PASS')
        else:
            self.GotoTarget(p1_actions, pp_vec)
            print('FIND PUCK')

        return p1_actions




    def predict(self, state, info, deterministic):
        if self.use_model:
            p1_actions = self.p1_model.predict(state, deterministic=deterministic)[0]
        else:
            p1_actions = [0] * GameConsts.INPUT_MAX

            if info is not None:

                return self.Think(info)

        

        return p1_actions

    
    
    #def predict