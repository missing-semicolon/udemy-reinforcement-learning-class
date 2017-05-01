import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy
from monte_carlo_es import max_dict

SMALL_ENOUGH = 10e-4
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')


# find optimal policy and value function using on-policy first-visit MC

def random_action(a, eps=0.1):
    p = np.random.random()
    if p < (1 - eps):
        return a
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)


def play_game(grid, policy):
    # returns a list of states and corresponding returns

    # In this version we will NOT be using exploring starts method
    # instead  we will explore using an epsilon-soft policy

    s = (2, 0)
    grid.set_state(s)
    a = random_action(policy[s])

    # be aware of the timing
    # each triple is s(t), a(t), r(t)
    # bur r(t) results from taking actions a(t-1) from s(t-1) and landing in s(t)

    states_actions_rewards = [(s, a, 0)]
    while True:
        old_s = grid.current_state()
        r = grid.move(a)
        s = grid.current_state()
        if s == old_s:
            # hack so that we don't end up in infinitely long episodes
            # print("I'm stuck!")
            states_actions_rewards.append((s, None, -100))
            break
        elif grid.game_over():
            # print("game over!")
            states_actions_rewards.append((s, None, r))
            break
        else:
            a = random_action(policy[s])
            states_actions_rewards.append((s, a, r))

    G = 0
    states_actions_returns = []
    first = True
    for s, a, r in reversed(states_actions_rewards):
        # the value of the terminal state is 0 by definition
        # we should ignore the first state we encounter
        if first:
            first = False
        else:
            states_actions_returns.append((s, a, G))
        G = r + GAMMA * G
    states_actions_returns.reverse()
    return states_actions_returns


if __name__ == '__main__':

    grid = negative_grid(step_cost=-0.1)

    # print rewards
    print("rewards:")
    print_values(grid.rewards, grid)

    # create random policy
    policy = {}
    for s in list(grid.actions):
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)
    print_policy(policy, grid)

    # initialize Q(s,a) and returns
    Q = {}
    returns = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            Q[s] = {}
            for a in ALL_POSSIBLE_ACTIONS:
                Q[s][a] = 0  # needs to be initialized into something we can argmax
                returns[(s, a)] = []
        else:
            # terminal state
            pass


    # repeat until convergence
    deltas = []
    for t in range(20000):
        if t % 1000 == 0:
            print(t)

        # generate an episode using policy
        biggest_change = 0
        states_actions_returns = play_game(grid, policy)
        seen_state_action_pairs = set()
        for s, a, G in states_actions_returns:
            # check if we have already seen s
            # called first-visit MC policy evaluation
            sa = (s, a)
            if sa not in seen_state_action_pairs:
                old_Q = Q[s][a]
                returns[sa].append(G)
                Q[s][a] = np.mean(returns[sa])
                biggest_change = max(biggest_change, np.abs(old_Q - Q[s][a]))
                seen_state_action_pairs.add(sa)
        deltas.append(biggest_change)

        # update policy
        for s in list(policy):
            policy[s] = max_dict(Q[s])[0]

    plt.plot(deltas)
    plt.show()

    print("final policy:")
    print_policy(policy, grid)

    # Find V
    V = {}
    for s, Qs in Q.items():
        V[s] = max_dict(Qs)[1]

    print("values:")
    print_values(V, grid)
