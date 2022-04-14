# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by James Gao (jamesjg2@illinois.edu) on 9/03/2021
# Inspired by work done by Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP2
"""

import math
import numpy as np
from alien import Alien
from numpy.linalg import norm
def dot(v,w):
    x,y = v
    X,Y = w
    return x*X + y*Y
def typed(a,b):
    x,y = a
    X,Y = b
    return (X-x, Y-y)
def point_to_line(start,end,pnt):
    seg = typed(start, end)
    vpnt = typed(start, pnt)
    x,y = seg
    if (x*x + y*y)!=0:
        mag = math.sqrt(x*x + y*y)
        seg_unit = (x/mag,y/mag)
        i,j = vpnt
        pntscaled = (i*1.0/mag,j*1.0/mag)
        t = dot(seg_unit, pntscaled)    
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        closest = (x*t,y*t)
        final = typed(closest, vpnt)
        x,y=final
        dist=math.sqrt(x*x + y*y)
        return (dist)
    if x*x + y*y ==0:
        return 1000
def does_alien_touch_wall(alien, walls,granularity):
    loc = []
    for wall in walls:
        loc = alien.get_centroid()
        endpt=alien.get_head_and_tail()
        head=endpt[0]
        tail=endpt[1]
        B=[wall[2],wall[3]] #wall endpoint
        A=[wall[0], wall[1]] #wall start point
        C=[loc[0],loc[1]] #center of alien
        AB = [None, None]
        AC = [None, None]
        BC = [None, None]
        AB[0] = B[0] - A[0]
        AB[1] = B[1] - A[1]
       
        BC[0] = C[0] - B[0]
        BC[1] = C[1] - B[1]
       
        AC[0] = C[0] - A[0]
        AC[1] = C[1] - A[1]
        ABBC = AB[0] * BC[0] + AB[1] * BC[1]
        ABAC = AB[0] * AC[0] + AB[1] * AC[1]

        if alien.is_circle():
            ABBC = AB[0] * BC[0] + AB[1] * BC[1]
            ABAC = AB[0] * AC[0] + AB[1] * AC[1]
            #is point is not in region to the goal and is closer to 
#             if (ABBC == 0) or (ABAC == 0):
#                 return False
            if (ABBC >= 0) :
                if math.sqrt(BC[0]**2+BC[1]**2)<(alien.get_width()++granularity/math.sqrt(2)) or np.isclose(math.sqrt(BC[0]**2+BC[1]**2),(granularity/math.sqrt(2)+alien.get_width())):
                    return True
            elif (ABAC <= 0):
                if math.sqrt(AC[0]**2+AC[1]**2)<(alien.get_width()++granularity/math.sqrt(2)) or np.isclose(math.sqrt(AC[0]**2+AC[1]**2),(+granularity/math.sqrt(2)+alien.get_width())):
                    return True
            else:
                x1 = AB[0]
                y1 = AB[1]
                x2 = AC[0]
                y2 = AC[1]
                d = abs(x1 * y2 - y1 * x2) / math.sqrt(x1 * x1 + y1 * y1)
                if float(d)<(alien.get_width()+granularity/math.sqrt(2)) or np.isclose(float(d),(alien.get_width()+granularity/math.sqrt(2))):
                     return True
                continue
            #find the closest endpoint to the wall
            d_str=math.sqrt(((wall[0]-loc[0])**2)+(wall[1]-loc[1])**2)
            d_end=math.sqrt(((wall[2]-loc[0])**2)+(wall[3]-loc[1])**2)
            closest_endpt=min([d_str,d_end])
            if closest_endpt<(alien.get_width()+granularity/math.sqrt(2)) or np.isclose(closest_endpt,(alien.get_width()+granularity/math.sqrt(2))):
                    return True
            continue 
        #alien is shape horizontal or vertical
        else:
            start= [wall[0],wall[1]]
            end= [wall[2],wall[3]]
            startx= float(start[0])
            starty= float(start[1])
            endx= float(end[0])
            endy= float(end[1])
            headx= float(head[0])
            heady= float(head[1])
            tailx= float(tail[0])
            taily= float(tail[1])
            
            #check if alien intersects
            xwall =  float(endx - startx)
            ywall =  float(endy - starty)
            xalien =  float(tailx - headx )    
            yalien =  float(taily - heady)
            determinate= (-1*xalien*ywall)+(xwall*yalien)
            if determinate != 0:
                v = float((-ywall*(startx-headx)+xwall*(starty-heady))/ (-xalien*ywall+xwall*yalien))
                w = float((-yalien*(startx-headx)+(xalien*(starty-heady)))/ (-xalien*ywall+xwall*yalien))

                if v >= 0 and v <= 1 and w >= 0 and w <= 1:
                    return True
            #check if the lines dont intersect that they dont touch the walls
            start= [wall[0],wall[1]]
            end= [wall[2],wall[3]]
            # distance from tail to wallseg 
            d_tail=point_to_line(start,end,tail)
            # distance from head to wallseg 
            d_head=point_to_line(start,end,head)
            #dis from wall start to alien line seg
            d_start=point_to_line(head,tail,start)
            #dis from wall end to alien line seg
            d_end=point_to_line(head,tail,end)
            min_dist= min([d_tail,d_head,d_start,d_end])
            if float(min_dist)<=(alien.get_width()+(granularity/math.sqrt(2))) or np.isclose(float(min_dist),(alien.get_width()+granularity/math.sqrt(2)) ):
                return True
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format [(startx, starty, endx, endx), ...]
            granularity (int): The granularity of the map

        Return:
            True if touched, False if not
    """
    return False

def does_alien_touch_goal(alien, goals):
    loc = []
    pt_dist=[]
    for goal in goals:
        loc = alien.get_centroid()
        if alien.is_circle():
            distance= math.sqrt(((loc[0]-goal[0])**2)+(loc[1]- goal[1])**2)
            if distance< (goal[2]+alien.get_width()) or np.isclose(distance,(goal[2]+alien.get_width()) ):
                return True
            else:
                continue
        else:
            endpt=alien.get_head_and_tail()
            B=list(endpt[1])
            A=list(endpt[0])
            C=[float(goal[0]),float(goal[1])]
            AB = [None, None]
            BC = [None, None]
            AC = [None, None]
            AB[0] = B[0] - A[0]
            AB[1] = B[1] - A[1]
            BC[0] = C[0] - B[0]
            BC[1] = C[1] - B[1]
            AC[0] = C[0] - A[0]
            AC[1] = C[1] - A[1]
            # Calculating the dot product
            ABBC = AB[0] * BC[0] + AB[1] * BC[1]
            ABAC = AB[0] * AC[0] + AB[1] * AC[1]
            #is point is not in region to the goal and is closer to 
            if (ABBC > 0) :
                if math.sqrt(BC[0]**2+BC[1]**2)<(alien.get_width()+goal[2]) or np.isclose(math.sqrt(BC[0]**2+BC[1]**2),(goal[2]+alien.get_width())):
                    return True
            elif (ABAC < 0):
                if math.sqrt(AC[0]**2+AC[1]**2)<(alien.get_width()+goal[2]) or np.isclose(math.sqrt(AC[0]**2+AC[1]**2),(goal[2]+alien.get_width())):
                    return True
            else:
                x1 = AB[0];
                y1 = AB[1];
                x2 = AC[0];
                y2 = AC[1];
                if x1 * x1 + y1 * y1 !=0:
                    d = abs(x1 * y2 - y1 * x2) / math.sqrt(x1 * x1 + y1 * y1)
                else:
                    d=0
                if float(d)<(alien.get_width()+goal[2]) or np.isclose(float(d),(goal[2]+alien.get_width())):
                        return True
               
    """Determine whether the alien touches a goal
        
        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            goals (list): x, y coordinate and radius of goals in the format [(x, y, r), ...]. There can be multiple goals
        
        Return:
            True if a goal is touched, False if not.
    """
    return False

def is_alien_within_window(alien, window,granularity):

    loc=alien.get_centroid()

    if alien.is_circle():
        if (loc[0] <= granularity/math.sqrt(2)+alien.get_width()) or (loc[0] >= window[0] -granularity/math.sqrt(2)-alien.get_width()):
            return False
        if (loc[1] <= granularity/math.sqrt(2)+alien.get_width()) or (loc[1] >= window[1] -granularity/math.sqrt(2)-alien.get_width()):
            return False
    else:
        endpt=alien.get_head_and_tail()
        head=endpt[0]
        tail=endpt[1]
        
        if (tail[0] <= granularity/math.sqrt(2)+alien.get_width()) or (head[0] <= granularity/math.sqrt(2)+alien.get_width()) or (tail[0] >= window[0] -granularity/math.sqrt(2)-alien.get_width()) or (head[0] >= window[0] -granularity/math.sqrt(2)-alien.get_width()) or np.isclose(head[0],window[0] -granularity/math.sqrt(2)-alien.get_width()) or np.isclose(tail[0],window[0]-granularity/math.sqrt(2)-alien.get_width()) or np.isclose(head[0],granularity/math.sqrt(2)+alien.get_width()) or np.isclose(tail[0],granularity/math.sqrt(2)+alien.get_width()):
            return False
        if (head[1] <= granularity/math.sqrt(2)+alien.get_width()) or (tail[1] <= granularity/math.sqrt(2)+alien.get_width()) or (tail[1] >= window[1] -granularity/math.sqrt(2)-alien.get_width()) or (head[1] >= window[1] -granularity/math.sqrt(2)-alien.get_width()) or np.isclose(head[1], window[1] -granularity/math.sqrt(2)-alien.get_width()) or np.isclose(tail[1], window[1] -granularity/math.sqrt(2)-alien.get_width()) or np.isclose(head[1],granularity/math.sqrt(2)+alien.get_width()) or np.isclose(tail[1],granularity/math.sqrt(2)+alien.get_width()):
            return False
        
        
      
    """Determine whether the alien stays within the window
        
        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
            granularity (int): The granularity of the map
    """
    return True


if __name__ == '__main__':
    #Walls, goals, and aliens taken from Test1 map
    walls =   [(0,100,100,100),  
                (0,140,100,140),
                (100,100,140,110),
                (100,140,140,130),
                (140,110,175,70),
                (140,130,200,130),
                (200,130,200,10),
                (200,10,140,10),
                (175,70,140,70),
                (140,70,130,55),
                (140,10,130,25),
                (130,55,90,55),
                (130,25,90,25),
                (90,55,90,25)]
    goals = [(110, 40, 10)]
    window = (220, 200)

    def test_helper(alien : Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()

        touch_wall_result = does_alien_touch_wall(alien, walls, 0) 
        touch_goal_result = does_alien_touch_goal(alien, goals)
        in_window_result = is_alien_within_window(alien, window, 0)

        assert touch_wall_result == truths[0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {touch_wall_result}, expected: {truths[0]}'
        assert touch_goal_result == truths[1], f'does_alien_touch_goal(alien, goals) with alien config {config} returns {touch_goal_result}, expected: {truths[1]}'
        assert in_window_result == truths[2], f'is_alien_within_window(alien, window) with alien config {config} returns {in_window_result}, expected: {truths[2]}'

    #Initialize Aliens and perform simple sanity check. 
    alien_ball = Alien((30,120), [40, 0, 40], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Ball', window)
    test_helper(alien_ball, alien_ball.get_centroid(), (False, False, True))

    alien_horz = Alien((30,120), [40, 0, 40], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Horizontal', window)	
    test_helper(alien_horz, alien_horz.get_centroid(), (False, False, True))

    alien_vert = Alien((30,120), [40, 0, 40], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Vertical', window)	
    test_helper(alien_vert, alien_vert.get_centroid(), (True, False, True))

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Horizontal', window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Vertical', window)

    alien_positions = [
                        #Sanity Check
                        (0, 100),

                        #Testing window boundary checks
                        (25.6, 25.6),
                        (25.5, 25.5),
                        (194.4, 174.4),
                        (194.5, 174.5),

                        #Testing wall collisions
                        (30, 112),
                        (30, 113),
                        (30, 105.5),
                        (30, 105.6), # Very close edge case
                        (30, 135),
                        (140, 120),
                        (187.5, 70), # Another very close corner case, right on corner
                        
                        #Testing goal collisions
                        (110, 40),
                        (145.5, 40), # Horizontal tangent to goal
                        (110, 62.5), # ball tangent to goal
                        
                        #Test parallel line oblong line segment and wall
                        (50, 100),
                        (200, 100),
                        (205.5, 100) #Out of bounds
                    ]

    #Truths are a list of tuples that we will compare to function calls in the form (does_alien_touch_wall, does_alien_touch_goal, is_alien_within_window)
    alien_ball_truths = [
                            (True, False, False),
                            (False, False, True),
                            (False, False, True),
                            (False, False, True),
                            (False, False, True),
                            (True, False, True),
                            (False, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (False, True, True),
                            (False, False, True),
                            (True, True, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True)
                        ]
    alien_horz_truths = [
                            (True, False, False),
                            (False, False, True),
                            (False, False, False),
                            (False, False, True),
                            (False, False, False),
                            (False, False, True),
                            (False, False, True),
                            (True, False, True),
                            (False, False, True),
                            (True, False, True),
                            (False, False, True),
                            (True, False, True),
                            (True, True, True),
                            (False, True, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, False),
                            (True, False, False)
                        ]
    alien_vert_truths = [
                            (True, False, False),
                            (False, False, True),
                            (False, False, False),
                            (False, False, True),
                            (False, False, False),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (False, False, True),
                            (True, True, True),
                            (False, False, True),
                            (True, True, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True)
                        ]

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    #Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110,55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))


    print("Geometry tests passed\n")