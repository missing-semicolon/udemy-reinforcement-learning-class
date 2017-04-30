import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

SMALL_ENOUGH = 10e-4
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

# this is deterministic
# all p(s',r|s,a) = 1 or 0

if __name__ == '__main__':
    # this grid gives you a reward of -0.1 for every non-terminal state
    # we want to see if this will encourage finding a shorter path to the goal
    grid = negative_grid()

    # print rewards
    print("rewards:")
    print_values(grid.rewards, grid)

    # state -> action
    # we'll randomly choose an action and update as we learn
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

    # initial policy
    print("initial policy:")
    print_policy(policy, grid)

    # initialize V(s)
    V = {}
    states = grid.all_states()
    for s in states:
        # V[s] = 0
        if s in grid.actions:
            V[s] = np.random.random()
        else:
            # terminal state
            V[s] = 0

    # This is the Value Iteration Step!!!
    # repeat until convergence - will break out when policy does not change
    # V[s] = max[a]{ sum[s',r] { p(s',r|s,a)[r + GAMMA*V[s']] } }
    # Note that in the code below p(s',r|s,a) = 1 -- Non-stochastic rewards or 
    # states conditional upon the current state and policy-driven action.
    while True:
        biggest_change = 0
        for s in states:
            old_v = V[s]

            # V(s) only has value if it's not a terminal state
            if s in policy:
                new_v = float('-inf')
                for a in ALL_POSSIBLE_ACTIONS:
                    grid.set_state(s)
                    r = grid.move(a)
                    v = r + GAMMA * V[grid.current_state()]
                    if v > new_v:
                        new_v = v
                V[s] = new_v
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))
        if biggest_change < SMALL_ENOUGH:
            break

    # find a policy that leads to the optimal value function
    for s in policy.keys():
        best_a = None
        best_value = float('-inf')
        # loop through all possible action to find the best current action
        for a in ALL_POSSIBLE_ACTIONS:
            grid.set_state(s)
            r = grid.move(a)
            v = r + GAMMA * V[grid.current_state()]
            if v > best_value:
                best_value = v
                best_a = a
        policy[s] = best_a

    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)