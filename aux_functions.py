import numpy as np
import math as m


def find_possible_Pi(T):
    '''
    Find the possible actions for each state

    Parameters
    ----------
    T : list
      T[s][a][s'] is equal to the transition probability T(s,a,s') to go from s to s' with action a

    Returns
    -------
    possible_Pi : list
      possible_Pi[s] is a list of possible actions from state s
    '''
    # init
    n_state = len(T)
    possible_Pi = [[]]*n_state
    # find passiple actions a from states s
    for s in range(n_state):
        for a in range(len(T[s])):
            # an action a is not available from state s if and only if all the probability transitions to states s' are equal to 0. otherwise it is equal to 1
            is_action = sum(T[s][a]) > 0
            if is_action:
                possible_Pi[s].append(a)
    return possible_Pi


def update_V(cur_V, gamma, T, R, possible_Pi):
    '''
    compute the new value for each state

    Parameters
    ----------
    cur_V : list
        list of the current values for each state
    gamma : float
        value of gamma parameter
    T : list
        transition matrix
    R : list
        reward matrix
    possible_Pi : list
        possible actions for each state

    Returns
    -------
    new_V : list
        list of the new values for each state
    '''
    n = len(cur_V)
    new_V=np.zeros((n))
    for s in range(n):
        new_V[s]=R[s]
        sum=np.zeros(len(possible_Pi[s]))
        for a in possible_Pi[s]:
            for s2 in range(n):
                sum[a] += gamma*T[s][a][s2]*cur_V[s2]
        new_V[s] += max(sum)
    return new_V


def RMS_error(cur_V, new_V):
    '''
    compute the root mean square error between the current and the new set of values for each state

    Parameters
    ----------
    cur_V : list
        list of the current values for each state
    new_V : list
        list of the new values for each state

    Returns
    -------
    rms : float
        root mean square error
    '''

    n = len(cur_V)
    if len(new_V) != n:
        print('Error : new V and current V doesn t have the same size.')
        return 0
    rms = 0
    for s in range(n):
        rms += (cur_V[s] - new_V[s])**2
    rms = m.sqrt(rms/n)
    return rms


def optimal_policy(cur_V, T, possible_pi):
    '''
    compute the optimal policy 

    Parameters
    ----------
    cur_V : list
        list of the current values for each state
    T : list
        transition matrix
    possible_pi : list
        list of possible actions for each state

    Returns
    -------
    pi : list
        list of the action for each state for the optimal policy
    '''

    n=len(cur_V)
    pi=np.zeros(n)
    for s in range(n):
        if len(possible_pi[s]) == 1:
            pi[s] = possible_pi[s]
        else:
            sum = np.zeros(len(possible_pi[s]))
            for a in possible_pi[s]:
                for s2 in range(n):
                    sum[a] += T[s][a][s2]*cur_V[s2]
            pi[s] = np.argmax(sum)
    return pi
