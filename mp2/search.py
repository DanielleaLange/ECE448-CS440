# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

from collections import deque
from heapq import heappop, heappush

def search(maze, searchMethod):
    return {
        "bfs": bfs,
    }.get(searchMethod, [])(maze)

def bfs(maze, ispart1=False):
    # Write your code here
    found = set()
    parent = {}
    path = []
    print(maze.getStart())
    queue= deque([maze.getStart()])
    #print(maze.getStart())
    found.add(maze.getStart())
    print(maze.getObjectives())
    while len(queue):
        temp = queue.pop()
        #print(temp)    
        for e in maze.getNeighbors(temp[0], temp[1], temp[2], ispart1):
            if e not in found:
                found.add(e)
                parent[e]=temp
                queue.appendleft(e)
            if maze.isObjective(e[0],e[1], e[2], ispart1 ):
                queue.clear()
                break             
#     d = maze.waypoints
#     d = d[0]
    if maze.isObjective(e[0],e[1], e[2], ispart1):
        print(parent)
        print(e)
        while True:
            path.append(e)
            e = parent[e]
            if e == maze.getStart():
                path.append(maze.getStart())
                break
        print(path)
        path = path[::-1]

        return path
    return None

    """
    This function returns optimal path in a list, which contains start and objective.
    If no path found, return None. 

    Args:
        maze: Maze instance from maze.py
        ispart1: pass this variable when you use functions such as getNeighbors and isObjective. DO NOT MODIFY THIS
    """
    []