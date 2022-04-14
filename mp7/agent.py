import numpy as np
import utils

class Agent:    
    def __init__(self, actions, Ne=40, C=40, gamma=0.7):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path,self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None
    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''
        #first generate the state
        s_prime = self.generate_state(environment)

        # TODO: write your function here
#update the n table 
#then update the Q-table
#when t=0 both s and a will be none no update
        action =self.a
        if dead:
            reward= -1
        elif self.points != points:
            self.points = points
            reward=1
        else:
            reward=-0.1

        if self._train is True:
            max_Q=-100
            #select an action
            max_act=-10000
            choice_act=0
            for act in reversed(range(4)):
                #if best action is less than NE we can choose it
                if self.N[s_prime[0], s_prime[1], s_prime[2], s_prime[3], s_prime[4], s_prime[5], s_prime[6],s_prime[7], act] < self.Ne:
                    #choose the action
                    F=1
                else:
                    F=self.Q[s_prime[0], s_prime[1], s_prime[2], s_prime[3], s_prime[4], s_prime[5], s_prime[6], s_prime[7], act]

                #choose best action that is less explored
                if F>max_act:
                    max_act=F
                    choice_act=act
            if self.a != None and self.s != None:
                self.N[self.s][self.a] += 1
                #find maxa′Q(s′,a′)
                for act in reversed(range(4)):
                    if self.Q[s_prime[0], s_prime[1], s_prime[2], s_prime[3], s_prime[4], s_prime[5], s_prime[6], s_prime[7], act] > max_Q:
                        max_Q=self.Q[s_prime[0], s_prime[1], s_prime[2], s_prime[3], s_prime[4], s_prime[5], s_prime[6], s_prime[7], act]
                
                Curr_Q=self.Q[self.s[0], self.s[1], self.s[2], self.s[3], self.s[4], self.s[5], self.s[6], self.s[7],self.a]
                #get alpha
                alpha = self.C/(self.C + self.N[self.s[0],self.s[1],self.s[2],self.s[3],self.s[4],self.s[5],self.s[6],self.s[7],action])
                #update Q
                self.Q[self.s[0], self.s[1], self.s[2], self.s[3], self.s[4], self.s[5], self.s[6], self.s[7],action]+= alpha*(reward-Curr_Q+self.gamma*max_Q)
                
         
            if dead:
                self.reset()
            else:
                self.s = s_prime
                self.points = points
#                 for act in reversed(range(4)):
#                     #if best action is less than NE we can choose it
#                     if self.N[s_prime[0], s_prime[1], s_prime[2], s_prime[3], s_prime[4], s_prime[5], s_prime[6],s_prime[7], act] < self.Ne:
#                         #choose the action
#                         F=1
#                     else:
#                         F=self.Q[s_prime[0], s_prime[1], s_prime[2], s_prime[3], s_prime[4], s_prime[5], s_prime[6], s_prime[7], act]

#                     #choose best action that is less explored
#                     if F>max_act:
#                         max_act=F
#                         choice_act=act
#             if not dead:
#                 self.N[s_prime[0], s_prime[1], s_prime[2], s_prime[3], s_prime[4], s_prime[5], s_prime[6],s_prime[7],choice_act] += 1
        #testing phase
        else:
            
            max_Q=-100
            for act in reversed(range(4)):
                if self.Q[s_prime[0], s_prime[1], s_prime[2], s_prime[3], s_prime[4], s_prime[5], s_prime[6], s_prime[7], act] > max_Q:
                    max_Q=self.Q[s_prime[0], s_prime[1], s_prime[2], s_prime[3], s_prime[4], s_prime[5], s_prime[6], s_prime[7], act]
                    choice_act=act
        if dead:
            self.reset()
        self.a=choice_act
        
            
    #snake dies if:
    #head touches wall
    #it touches own body
    #after taking DISPLAY_SIZE / GRID_SIZE  steps without food
    #testing choose action without exploration function
        return choice_act

    def generate_state(self, environment):
        # TODO: Implement this helper function that generates a state given an environment 
        #if on same axis value 0, otherwise left or right
        if environment[0] >environment[3]:
            food_dir_x=1
        elif environment[0]<environment[3]:
            food_dir_x=2
        else:
            food_dir_x=0
        if environment[1] >environment[4]:
            food_dir_y=1
        elif environment[1]<environment[4]:
            food_dir_y=2
        else:
            food_dir_y=0
            
        #need to factor in if snake runs out of board boundaries its 0
        if environment[0]<utils.WALL_SIZE*2 and environment[0]>0 :
            adjoining_wall_x=1
        elif environment[0]>utils.DISPLAY_SIZE-utils.WALL_SIZE*3 and environment[0]<utils.DISPLAY_SIZE-utils.WALL_SIZE:
            adjoining_wall_x=2
        else:
            adjoining_wall_x=0
        if environment[1]<utils.WALL_SIZE*2 and environment[1]>utils.WALL_SIZE:
            adjoining_wall_y=1
        elif environment[1]>utils.DISPLAY_SIZE-utils.WALL_SIZE*3 and environment[1]<utils.DISPLAY_SIZE-utils.WALL_SIZE:
            adjoining_wall_y=2
        else:
            adjoining_wall_y=0
        adjoining_body_bottom=0
        adjoining_body_left=0
        adjoining_body_right=0
        adjoining_body_top=0
        if (environment[0] - utils.WALL_SIZE, environment[1]) in environment[2]:
            adjoining_body_left = 1
        if (environment[0] + utils.WALL_SIZE, environment[1]) in environment[2]:
            adjoining_body_right = 1
        if (environment[0], environment[1] - utils.WALL_SIZE) in environment[2]:
            adjoining_body_top = 1
        if (environment[0], environment[1] + utils.WALL_SIZE) in environment[2]:
            adjoining_body_bottom = 1
        
        #8*((utils.DISPLAY_SIZE/utils.GRID_SIZE)-1)**2
        final_state =(food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)
        return final_state
