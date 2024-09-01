### Task 1.1  - State Representation

### Task 1.2  - Initial & Goal States

### Task 1.3  - Representation Invariant

### Task 1.4  - Which Search Algorithm Should We Pick?

### Task 1.5  - Completeness and Optimality

### Task 1.6  - Implement Tree Search

def mnc_tree_search(m, c):  
    '''
    Solution should be the action taken from the root node (initial state) to 
    the leaf node (goal state) in the search tree.

    Parameters
    ----------    
    m: no. of missionaries
    c: no. of cannibals
    
    Returns
    ----------    
    Returns the solution to the problem as a tuple of steps. Each step is a tuple of two numbers x and y, indicating the number of missionaries and cannibals on the boat respectively as the boat moves from one side of the river to another. If there is no solution, return False.
    '''
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_1_6():
    # NOTE: There may be other optimal solutions.
    
    print(mnc_tree_search(2,1)) # possible solution ((2, 0), (1, 0), (1, 1))
    print(mnc_tree_search(2,2)) # possible solution ((1, 1), (1, 0), (2, 0), (1, 0), (1, 1))
    print(mnc_tree_search(3,3)) # possible solution ((1, 1), (1, 0), (0, 2), (0, 1), (2, 0), (1, 1), (2, 0), (0, 1), (0, 2), (1, 0), (1, 1))

### Task 1.7 - Implement Graph Search

def mnc_graph_search(m,c):
    '''
    Graph search requires to deal with the redundant path: cycle or loopy path.
    Modify the above implemented tree search algorithm to accelerate your AI.

    Parameters
    ----------    
    m: no. of missionaries
    c: no. of cannibals
    
    Returns
    ----------    
    Returns the solution to the problem as a tuple of steps. Each step is a tuple of two numbers x and y, indicating the number of missionaries and cannibals on the boat respectively as the boat moves from one side of the river to another. If there is no solution, return False.
    '''
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_1_7():
    # NOTE: There may be other optimal solutions.
    
    print(mnc_graph_search(2,1)) # possible solution ((2, 0), (1, 0), (1, 1))
    print(mnc_graph_search(2,2)) # possible solution ((1, 1), (1, 0), (2, 0), (1, 0), (1, 1))
    print(mnc_graph_search(3,3)) # possible solution ((1, 1), (1, 0), (0, 2), (0, 1), (2, 0), (1, 1), (2, 0), (0, 1), (0, 2), (1, 0), (1, 1))

### Task 1.8 - Tree vs Graph Search

import copy
import heapq
import math
import os
import random
import sys
import time

import utils
import cube

from typing import List, Tuple, Callable
from functools import partial

"""
We provide implementations for the Node and PriorityQueue classes in utils.py, but you can implement your own if you wish
"""
from utils import Node
from utils import PriorityQueue

### Task 2.1: Design a heuristic for A* Search

def heuristic_func(problem: cube.Cube, state: cube.State) -> float:
    r"""
    Computes the heuristic value of a state
    
    Args:
        problem (cube.Cube): the problem to compute
        state (cube.State): the state to be evaluated
        
    Returns:
        h_n (float): the heuristic value 
    """
    h_n = 0.0
    goals = problem.goal

    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

    return h_n

# goal state
cube_goal = {
    'initial': [['N', 'U', 'S'],
                ['N', 'U', 'S'],
                ['N', 'U', 'S']],
    'goal': [['N', 'U', 'S'],
             ['N', 'U', 'S'],
             ['N', 'U', 'S']],
    'solution': [],
}

# one step away from goal state
cube_one_step = {
    'initial': [['S', 'N', 'U'],
                ['N', 'U', 'S'],
                ['N', 'U', 'S']],
    'goal': [['N', 'U', 'S'],
             ['N', 'U', 'S'],
             ['N', 'U', 'S']],
    'solution': [[0, 'left']],
}

# transposes the cube
cube_transpose = {
    'initial': [['S', 'O', 'C'],
                ['S', 'O', 'C'],
                ['S', 'O', 'C']],
    'goal': [['S', 'S', 'S'],
             ['O', 'O', 'O'],
             ['C', 'C', 'C']],
    'solution': [[2, 'right'], [1, 'left'], [1, 'down'], [2, 'up']],
}

# flips the cube
cube_flip = {
    'initial': [['N', 'U', 'S'],
                ['N', 'U', 'S'],
                ['N', 'U', 'S']],
    'goal': [['S', 'U', 'N'],
             ['N', 'S', 'U'],
             ['U', 'N', 'S']],
    'solution': [[0, 'left'], [1, 'right'], [0, 'up'], [1, 'down']],
}

# intermediate state for cube_flip
cube_flip_intermediate = {
    'initial': [['U', 'S', 'N'],
                ['N', 'U', 'S'],
                ['N', 'U', 'S']],
    'goal': [['S', 'U', 'N'],
             ['N', 'S', 'U'],
             ['U', 'N', 'S']],
    'solution': [[1, 'right'], [0, 'up'], [1, 'down']],
}


# 3x4 cube
cube_3x4 = {
    'initial': [[1, 1, 9, 0],
                [2, 2, 0, 2],
                [9, 0, 1, 9]],
    'goal': [[1, 0, 9, 2],
             [2, 1, 0, 9],
             [2, 1, 0, 9]],
    'solution': [[1, 'down'], [3, 'up'], [2, 'left']],
}

def test_task_2_1():
    def test_heuristic(heuristic_func, case):
        problem = cube.Cube(cube.State(case['initial']), cube.State(case['goal']))
        assert heuristic_func(problem, problem.goal) == 0, "Heuristic is not 0 at the goal state"
        assert heuristic_func(problem, problem.initial) <= len(case['solution']), "Heuristic is not admissible"
    
    test_heuristic(heuristic_func, cube_goal)
    test_heuristic(heuristic_func, cube_one_step)
    test_heuristic(heuristic_func, cube_transpose)
    test_heuristic(heuristic_func, cube_flip)
    test_heuristic(heuristic_func, cube_flip_intermediate)
    test_heuristic(heuristic_func, cube_3x4)

### Task 2.2: Implement A* search 

def astar_search(problem: cube.Cube, heuristic_func: Callable):
    r"""
    A* Search finds the solution to reach the goal from the initial.
    If no solution is found, return False.
    
    Args:
        problem (cube.Cube): Cube instance
        heuristic_func (Callable): heuristic function for the A* search

    Returns:
        solution (List[Action]): the action sequence
    """
    fail = True
    solution = []

    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """
    
    if fail:
        return False
    return solution

def test_search(algorithm, case):
    problem = cube.Cube(cube.State(case['initial']), cube.State(case['goal']))
    start_time = time.perf_counter()
    solution = algorithm(problem)
    print(f"{algorithm.__name__}(goal={case['goal']}) took {time.perf_counter() - start_time:.4f} seconds")
    if solution is False:
        assert case['solution'] is False
        return
    verify_output = problem.verify_solution(solution, _print=False)
    assert verify_output['valid'], f"Fail to reach goal state with solution {solution}"
    assert verify_output['cost'] <= len(case['solution']), f"Cost is not optimal."

def test_task_2_2():
    def astar_heuristic_search(problem): 
        return astar_search(problem, heuristic_func=heuristic_func)
        
    test_search(astar_heuristic_search, cube_goal)
    test_search(astar_heuristic_search, cube_one_step)
    test_search(astar_heuristic_search, cube_transpose)
    test_search(astar_heuristic_search, cube_flip)
    test_search(astar_heuristic_search, cube_flip_intermediate)
    test_search(astar_heuristic_search, cube_3x4)

### Task 2.3: Consistency & Admissibility

### Task 2.4: Implement Uninformed Search

def uninformed_search(problem: cube.Cube):
    r"""
    Uninformed Search finds the solution to reach the goal from the initial.
    If no solution is found, return False.
    
    Args:
        problem (cube.Cube): Cube instance

    Returns:
        solution (List[Action]): the action sequence
    """
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_2_4():
    test_search(uninformed_search, cube_goal)
    test_search(uninformed_search, cube_one_step)
    test_search(uninformed_search, cube_transpose)
    test_search(uninformed_search, cube_flip)
    test_search(uninformed_search, cube_flip_intermediate)
    test_search(uninformed_search, cube_3x4)

### Task 2.5: Uninformed vs Informed Search


if __name__ == '__main__':
    test_task_1_6()
    test_task_1_7()
    test_task_2_1()
    test_task_2_2()
    test_task_2_4()