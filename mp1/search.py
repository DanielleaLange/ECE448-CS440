# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)


# Feel free to use the code below as you wish
# Initialize it with a list/tuple of objectives
# Call compute_mst_weight to get the weight of the MST with those objectives
# TODO: hint, you probably want to cache the MST value for sets of objectives you've already computed...
import queue
import heapq
import copy
from collections import deque
class MST:
    def __init__(self, objectives):
        self.elements = {key: None for key in objectives}

        # TODO: implement some distance between two objectives 
        # ... either compute the shortest path between them, or just use the manhattan distance between the objectives
        self.distances   = {
                (i, j): abs(i[0]-j[0]) + abs(i[0]-j[0])
                for i, j in self.cross(objectives)
            }
        
    # Prim's algorithm adds edges to the MST in sorted order as long as they don't create a cycle
    def compute_mst_weight(self):
        weight      = 0
        for distance, i, j in sorted((self.distances[(i, j)], i, j) for (i, j) in self.distances):
            if self.unify(i, j):
                weight += distance
        return weight

    # helper checks the root of a node, in the process flatten the path to the root
    def resolve(self, key):
        path = []
        root = key 
        while self.elements[root] is not None:
            path.append(root)
            root = self.elements[root]
        for key in path:
            self.elements[key] = root
        return root
    
    # helper checks if the two elements have the same root they are part of the same tree
    # otherwise set the root of one to the other, connecting the trees
    def unify(self, a, b):
        ra = self.resolve(a) 
        rb = self.resolve(b)
        if ra == rb:
            return False 
        else:
            self.elements[rb] = ra
            return True

    # helper that gets all pairs i,j for a list of keys
    def cross(self, keys):
        return (x for y in (((i, j) for j in keys if i < j) for i in keys) for x in y)
def bfs(maze):
    found = set()
    parent = {}
    path = []
    #print(1)
    queue= deque([maze.start])
    found.add(maze.start)
    while len(queue):
        temp = queue.pop()
        #print(temp)
        for e in maze.neighbors(temp[0], temp[1]):
            if e not in found:
                found.add(e)
                parent[e]=temp
                queue.appendleft(e)
            if e == maze.waypoints[0]:
                queue.clear()
                break             
    d = maze.waypoints
    d = d[0]
    while True:
        path.append(d)
        d = parent[d]
        if d == maze.start:
            path.append(maze.start)
            break
    #print(path)
    path = path[::-1]

    return path
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    
# def getPath(maze,parent,d,start):
#     path = []
#     while True:
#         path.append(d)
#         d = parent[d]
        
#         if d == start:
#             path.append(start)
#             break
#     #print(path)
#     path = path[::-1]
#     return path
def astar_single(maze):
    found = set()
    parent = {}
    g= {}
    h= {}
    path = []
    cost={}
    heap= []
    
    #convert type of starting and ending point for calculations
    start= [maze.start[0], maze.start[1]]
    waypointslist= maze.waypoints[0]
    goal= [waypointslist[0], waypointslist[1]]
    
  
    #calculate initial compenents of cost 
    g[tuple(start)] =0
    h[tuple(start)]= abs(goal[0]-start[0])+abs(goal[1]-start[1])
    print(goal)
    # mark start as found
    found.add(maze.start) 
     
    #calc init est cost and push to heap     
    heapq.heappush(heap, (g[start[0],start[1]]+h[start[0],start[1]], tuple(start)))
    #print(heap)

    while waypointslist not in found: #temp != waypointslist:
        temp = heapq.heappop(heap) # temp has current node
        curr= tuple(temp[1])
        cost[curr]= g[curr] + abs(temp[1][0]-goal[0])+ abs(temp[1][1]-goal[1]) #estimated total cost
        h[curr]= abs(curr[0]-goal[0])+ abs(curr[1]-goal[1])
        if curr== waypointslist:
            #getPath(maze,parent,tuple(goal),start)
            break
            
        for e in maze.neighbors(temp[1][0], temp[1][1]):
            g[e]=g[curr]+1
            h[e]= abs(e[0]-goal[0])+ abs(e[1]-goal[1])
            if e not in found:
                #update cost since its a new state
                cost[e]=h[e]+g[e] #can g go down 
                f=cost[e]
                heapq.heappush(heap, (f, tuple(e)))
                parent[e] = temp[1]#parents holds pointer to previous node at current node 
                if e not in found:
                    found.add(e)
            elif e in found:
                #only update cost if its lower than the previous
                if (h[e]+g[e])<cost[e]:# or e not in found:
                    cost[e]=h[e]+g[e] #can g go down 
                    f=cost[e]
                    heapq.heappush(heap, (f, tuple(e)))
                    parent[e] = temp[1]#parents holds pointer to previous node at current node 
           
    d=tuple(goal)
  
    print(parent)
    while True:
        path.append(d)
        #print(path)
        d = parent[d]
        #print(d)
        #print(tuple(start))
        if d == tuple(start):
            path.append(tuple(start))
            break
    path = path[::-1]
    return path
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

def astar_multiple(maze): 
    found=[]
    parent = {}
    g= {}
    h= {}
    path = []
    cost={}
    heap= []
    mst_goals= {}
    mini={}
    goalorder=[]
    main={}
    temp_last=[]
    
    #convert type of starting and ending point for calculations
    start= [maze.start[0], maze.start[1]]
    
    waypointslist= maze.waypoints #tuple
    goal= list(maze.waypoints) #list
    print(type(goal))
    future_mst=goal
    for i in range(len(goal)):
        future_mst= (goal[:i]+goal[i+1:])
        my_mst=MST(future_mst)
        mst_weight=my_mst.compute_mst_weight()
        mst_goals[tuple(future_mst)] = mst_weight
    print(mst_goals)
        
   # print(mst_goals)
    #find first goal
    for e in waypointslist:
        mini[e]= abs(start[0]-e[0]) +abs(start[1]-e[1]) 
        
    first_goal= min(mini, key=mini.get)
   #order waypoints so that first is in first position
    my_mst=MST(maze.waypoints)
    mst_weight=my_mst.compute_mst_weight()
    
    #mst_val= goal.remove(first_goal)
    #store mst value once computed
    mst_goals[maze.waypoints] = mst_weight
 
     #set up state
    state =( maze.start, waypointslist)
    
    #calculate initial compenents of cost 
    g[state] =0
    h[state]=  mst_weight + abs(start[0]-first_goal[0]) +abs(start[1]-first_goal[1])
    cost[state]=mst_weight + abs(start[0]-first_goal[0]) +abs(start[1]-first_goal[1])
   
   
    #calc init est cost and push to heap     
    heapq.heappush(heap, (mst_weight + abs(start[0]-first_goal[0]) +abs(start[1]-first_goal[1]), state))
    main[state] = mst_weight + abs(start[0]-first_goal[0]) +abs(start[1]-first_goal[1])
    temp=heap[0]
    curr= tuple(temp[1])
 
    future_mst=list(curr[1])
    while len(curr[1]) != 0: #fix this?????
        #print(heap)
        temp_last.append(goal)
        if heap== []:
            break
        temp = heapq.heappop(heap) # temp has current node
        stored_curr=curr
        curr= tuple(temp[1])
        state=(curr[0],curr[1])
            #calculate cost and h of parent node
       
        h[curr]= abs(curr[0][0]-first_goal[0])+ abs(curr[0][1]-first_goal[1])+ mst_goals[state[1]]
        cost[curr]= g[curr] + h[curr]
        
        #every time we reach a goal state
        if curr[0] in goal:
            
            #while we are not at the last goal state but at a goal state
            if len(curr[1]) != 0:
                #print(goal)
                my_mst=MST(goal)
                mst_weight=my_mst.compute_mst_weight()
                mst_goals[tuple(goal)] = mst_weight
                #print(mst_goals)
                future_mst=list(curr[1])
                mini.clear()
                for i in range(len(goal)):
                    future_mst= (goal[:i]+goal[i+1:])
                    my_mst=MST(future_mst)
                    mst_weight=my_mst.compute_mst_weight()
                    mst_goals[tuple(future_mst)] = mst_weight
#                     next_goal=tuple(goal[i])
#                     p=next_goal[0]
#                     q=next_goal[1]
#                     mini[e]= abs(start[0]-next_goal[0]) +abs(start[1]-next_goal[1]) + mst_goals[tuple(future_mst)]
                    print(mst_goals)
                #find next closest goal 
                temp_list=goal
                
                   
                #first_goal= min(mini, key=mini.get)
                #print(goal)
                goal.remove(curr[0])
                
            state=(curr[0],goal)
           # curr=(curr[0],goal)
        #if we have removed all waypoints break to get final path
        if len(curr[1]) == 0:
            state=(curr[0])
            break     
        
        #for all neighbors
        #print(mst_goals)
        g[curr]=g[stored_curr]+1
        
        
        for e in maze.neighbors(curr[0][0], curr[0][1]):
            state=(e, curr[1])
            g[state]=g[curr]+1
            h[state]= abs(state[0][0]-first_goal[0])+ abs(state[0][1]-first_goal[1]) + mst_goals[state[1]]
            #when its a state we havent seen update cost and he
            if state not in main:
            
                cost[state]=h[state]+g[state] #can g go down 
                f=cost[state]
                heapq.heappush(heap, (f, state))
                parent[e] = curr[0]#parents holds pointer to previous node at current node 
                if state not in main:
                    main[state]=cost[state]
            #when its a state seen see if previous cost is worse than new cost at state
            elif state in main:
                if (h[state]+g[state])<cost[state]:# or e not in found:
                    cost[state]=h[state]+g[state] #can g go down 
                    f=cost[state]
                    heapq.heappush(heap, (f, state))
                    #poniter[]
                    parent[e] = curr[0]#parents holds pointer to previous node at current node 
   
  
    #print(main)
    #print(state)
    #print(temp_last)
   
    d=curr[0]
    goal= list(maze.waypoints)
   
    print(parent)
    print(goal)
    while True:
        print(d)
        if d in goal:
            goal.remove(d)
            print(goal)
        path.append(d)
        #print(parent)
        d = parent[d]
        print(len(goal))
        if d ==(maze.start) and len(goal)==0: #start has to equal last starting point waypoint?
#                 break
#     #print(path)
#     path.append(tuple(maze.start))
#     path = path[::-1]
    return path

    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

def fast(maze):
#     found = set()
#     foundall=set()
#     parent = {}
#     g= {}
#     h= {}
#     path = []
#     cost={}
#     heap= []
#     mst_goals= {}
#     mini={}
#     goalorder=[]
#     main={}
    
#     #convert type of starting and ending point for calculations
#     start= [maze.start[0], maze.start[1]]
    
#     waypointslist= maze.waypoints #tuple
#     goal= list(maze.waypoints) #list
#     print(waypointslist)
#     #find first goal
#     for e in waypointslist:
#         mini[e]= abs(start[0]-e[0]) +abs(start[1]-e[1])
        
#     first_goal= min(mini, key=mini.get)
#    #order waypoints so that first is in first position
#     my_mst=MST(maze.waypoints)
#     mst_weight=my_mst.compute_mst_weight()
    
#     #store mst value once computed
#     mst_goals[first_goal] = mst_weight
    
#     #calculate initial compenents of cost 
#     g[tuple(start)] =0
#     h[tuple(start)]=  mst_weight + abs(start[0]-first_goal[0]) +abs(start[1]-first_goal[1])
#     # mark start as found
#     found.add(maze.start) 
#     foundall.add(maze.start) 
    
    
#     #set up state
#     state =( maze.start, waypointslist)
#     #calc init est cost and push to heap     
#     heapq.heappush(heap, (g[start[0],start[1]]+h[start[0],start[1]], state))
#     main[state] = g[start[0],start[1]]+h[start[0],start[1]]
#     print(main)
#     while len(waypointslist) != 0: #fix this?????
#         temp = heapq.heappop(heap) # temp has current node
#         curr= tuple(temp[1])
#         print("current cell is: ", curr)
#         cost[curr]= g[curr] + abs(temp[1][0]-first_goal[0])+ abs(temp[1][1]-first_goal[1]) #estimated total cost
#         h[curr]= abs(curr[0]-first_goal[0])+ abs(curr[1]-first_goal[1])+ mst_weight
#         #every time we reach a goal state
#         if curr== first_goal:
#             #update start and save order of goals
#             start= list(first_goal)
#             goalorder.append(first_goal)
            
#             #remove previous goal
#             goal.remove(first_goal)
#             waypointslist = tuple(goal)
            
#             #while we are not at the last goal state
#             if len(goal) != 0:
#                 #find next closest goal
#                 mini.clear()
#                 for e in tuple(goal): #change this maybe?
#                     mini[e]= abs(start[0]-e[0]) +abs(start[1]-e[1])
#                 first_goal= min(mini, key=mini.get)
                
#                 #update MST
#                 my_mst=MST(waypointslist)
#                 mst_weight=my_mst.compute_mst_weight()
    
#                 #store mst value once computed
#                 mst_goals[first_goal] = mst_weight

#             cost.clear()#will old cost values mess with path between different goals????
#             found.clear()
            
#             found.add(tuple(start)) 

#         #if we have removed all waypoints break to get final path
#         if len(waypointslist) == 0:
#                 break     
        
#         #calculate cost and h of parent node
#         cost[curr]= g[curr] + abs(temp[1][0]-first_goal[0])+ abs(temp[1][1]-first_goal[1]) 
#         h[curr]= abs(curr[0]-first_goal[0])+ abs(curr[1]-first_goal[1])+ mst_weight
        
#         #for all neighbors
#         for e in maze.neighbors(temp[1][0], temp[1][1]):
#             g[e]=g[curr]+1
#             h[e]= abs(e[0]-first_goal[0])+ abs(e[1]-first_goal[1]) + mst_weight
#             #when its a state we havent seen update cost and heap
#             if e not in found:
#                 cost[e]=h[e]+g[e] #can g go down 
#                 f=cost[e]
#                 heapq.heappush(heap, (f, tuple(e)))
#                 parent[e] = temp[1]#parents holds pointer to previous node at current node 
#                 if e not in found:
#                     found.add(e)
#             #when its a state seen see if previous cost is worse than new cost at state
#             elif e in found:
#                 if (h[e]+g[e])<cost[e]:# or e not in found:
#                     cost[e]=h[e]+g[e] #can g go down 
#                     f=cost[e]
#                     heapq.heappush(heap, (f, tuple(e)))
#                     parent[e] = temp[1]#parents holds pointer to previous node at current node 
           
           
#     print(maze.start)
#     print(parent)
#     for e in goalorder:
#         print(goalorder)
#         d=e
#         while True:
#             path.append(d)
#             d = parent[d]
#             if d in goalorder or d ==(maze.start): #start has to equal last starting point waypoint?
#                 break
#     print(path)
#     path.append(tuple(maze.start))
#     path = path[::-1]
#     return path

    found = set()
    parent = {}
    g= {}
    h= {}
    path = []
    cost={}
    heap= []
    mst_goals= {}
    mini={}
    
    #convert type of starting and ending point for calculations
    start= [maze.start[0], maze.start[1]]
    waypointslist= maze.waypoints #tuple
    goal= list(maze.waypoints) #list
    #find first goal
    for e in waypointslist:
        mini[e]= abs(start[0]-e[0]) +abs(start[1]-e[1])
        
    first_goal= min(mini, key=mini.get)
   #order waypoints so that first is in first position
    my_mst=MST(maze.waypoints)
    mst_weight=my_mst.compute_mst_weight()
    
    #store mst value once computed
    mst_goals[first_goal] = mst_weight
    
    #calculate initial compenents of cost 
    g[tuple(start)] =0
    h[tuple(start)]=  mst_weight + abs(start[0]-first_goal[0]) +abs(start[1]-first_goal[1])
    # mark start as found
    found.add(maze.start) 
     
    #calc init est cost and push to heap     
    heapq.heappush(heap, (g[start[0],start[1]]+h[start[0],start[1]], tuple(start)))

    while True: #fix this?????
        temp = heapq.heappop(heap) # temp has current node
        curr= tuple(temp[1])
        cost[curr]= g[curr] + abs(temp[1][0]-first_goal[0])+ abs(temp[1][1]-first_goal[1]) #estimated total cost
        h[curr]= abs(curr[0]-first_goal[0])+ abs(curr[1]-first_goal[1])+ mst_weight
        #every time we reach a goal state
        if curr== first_goal:
            print(1)
            #store path 
            d=first_goal
            print(parent)
            while True:
                path.append(d)
                #print(path)
                d = parent[d]
                if d == tuple(start): #start has to equal last starting point waypoint?
                    path.append(tuple(start))
                    break
                    
            #update start
            start= list(first_goal)
            #remove waypoint from remaining goals
            goal.remove(first_goal)
            print(goal)
            waypointslist = tuple(goal)
            if len(goal) != 0:
                #find next closest goal
                mini.clear()
                for e in tuple(goal):
                    mini[e]= abs(start[0]-e[0]) +abs(start[1]-e[1])
  
                first_goal= min(mini, key=mini.get)
                
                #update MST
                my_mst=MST(waypointslist)
                mst_weight=my_mst.compute_mst_weight()
    
                #store mst value once computed
                mst_goals[first_goal] = mst_weight
            h.clear()
            g.clear()
            cost.clear()
            parent.clear()
            found.clear()
            found.add(tuple(start)) 
            h[tuple(start)]= abs(curr[0]-first_goal[0])+ abs(curr[1]-first_goal[1])+ mst_weight
            g[tuple(start)]=0
            cost[tuple(start)]=h[tuple(start)]
            heap=[]
            
            heapq.heappush(heap, (g[start[0],start[1]]+h[start[0],start[1]], tuple(start)))
            
            if len(waypointslist) == 0:
                #add if statement for if its the last waypoint
                path = path[::-1]
                return path
        cost[curr]= g[curr] + abs(temp[1][0]-first_goal[0])+ abs(temp[1][1]-first_goal[1]) #estimated total cost
        h[curr]= abs(curr[0]-first_goal[0])+ abs(curr[1]-first_goal[1])+ mst_weight
        for e in maze.neighbors(temp[1][0], temp[1][1]):
            
        
            g[e]=g[curr]+1
            h[e]= abs(e[0]-first_goal[0])+ abs(e[1]-first_goal[1]) + mst_weight
            if e not in found:
                #update cost since its a new state
                cost[e]=h[e]+g[e] #can g go down 
                f=cost[e]
                heapq.heappush(heap, (f, tuple(e)))
                parent[e] = temp[1]#parents holds pointer to previous node at current node 
                if e not in found:
                    found.add(e)
            elif e in found:
                #only update cost if its lower than the previous
                if (h[e]+g[e])<cost[e]:# or e not in found:
                    
                    cost[e]=h[e]+g[e] #can g go down 
                    f=cost[e]
                    heapq.heappush(heap, (f, tuple(e)))
                    parent[e] = temp[1]#parents holds pointer to previous node at current node 
#                 elif cost[curr]>cost[e]
#                     del parent[temp[1]]
                    
           
   
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return path
    
            
