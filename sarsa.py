import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy
from monte_carlo_es import max_dict
from td0_prediction import random_action

GAMMA = 0.9
ALPHA = 0.1
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')


# NOTE: Determine policy with sarsa


def play_game(grid, policy):
    # returns a list of states and corresponding rewards (not returns!)

    # Start at the designated start state

    s = (2, 0)
    grid.set_state(s)
    states_and_rewards = [(s, 0)]
    while not grid.game_over():
        a = policy[s]
        a = random_action(a)
        r = grid.move(a)
        s = grid.current_state()
        states_and_rewards.append((s, r))

    return states_and_rewards


if __name__ == '__main__':

    grid = negative_grid(step_cost=-0.1)

    # print rewards
    print("rewards:")
    print_values(grid.rewards, grid)

    # No policy initialization. We will derive our policy from the most recent Q

    # Initialize Q(s, a)

    Q = {}
    states = grid.all_states()
    for s in states:
        Q[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            Q[s][a] = 0

    # let's keep track of how many times Q[s] has been updated
    update_counts = {}
    update_counts_sa = {}
    for s in states:
        update_counts_sa[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            update_counts_sa[s][a] = 1.0


    # repeat until convergence
    t = 1.0
    deltas = []
    for it in range(10000):
        if it % 100 == 0:
            t += 10e-3
        if it % 2000 == 0:
            print("it: {}".format(it))

        # instead of generating an episode, we will play an episode withint this loop
        s = (2, 0)
        grid.set_state(s)

        # the first (s, r) tuple is the state we start in w/ reward = 0
        a = max_dict(Q[s])[0]
        a = random_action(a, eps=0.5/t)
        biggest_change = 0
        while not grid.game_over():
            r = grid.move(a)
            s2 = grid.current_state()

            # we will need the next action as well since Q(s, a) depends on Q(s', a')
            # if s2 not in policy then it's a terminal state, all Q are 0
            a2 = max_dict(Q[s2])[0]
            a2 = random_action(a2, eps=0.5/t)  # epsilon greedy

            # we will update Q(s, a) as we experience the episode 
            alpha = ALPHA / update_counts_sa[s][a]
            update_counts_sa[s][a] += .005
            old_qsa = Q[s][a]
            Q[s][a] = Q[s][a] + alpha * (r + GAMMA * Q[s2][a2] - Q[s][a])
            biggest_change = max(biggest_change, np.abs(old_qsa - Q[s][a]))

            # we would like to know how often Q(s) has been updated too
            update_counts[s]  = update_counts.get(s, 0) + 1

            # next state becomes current state
            s = s2
            a = a2

        deltas.append(biggest_change)

    plt.plot(deltas)
    plt.show()

    # Determine policy from Q*
    # Find V* from Q*
    policy = {}
    V = {}
    for s in list(grid.actions):
        a, max_q = max_dict(Q[s])
        policy[s] = a
        V[s] = max_q

    # what's the proportion of time we spend updating each part of Q?
    print("Update counts")
    total = np.sum(list(update_counts.values()))
    for k, v in update_counts.items():
        update_counts[k] = float(v) / total
    print_values(update_counts, grid)

    print("final policy:")
    print_policy(policy, grid)

    print("values:")
    print_values(V, grid)
