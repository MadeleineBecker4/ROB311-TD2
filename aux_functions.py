import numpy as np
import math as m


def update_V(cur_V,gamma,T,R,possible_Pi):
    n = len(cur_V)
    new_V=np.zeros((1,n))
    for s in range(1,4):
        new_V[s]=R[s]
        sum=np.zeros(len(possible_Pi[s]))
        for a in possible_Pi[s]:
            for s2 in range(n):
                sum[a] += gamma*T[s][a][s2]*cur_V[s2]
        new_V[s]+=max(sum)
    return new_V


def RMS_error(cur_V, new_V):
    n = len(cur_V)
    if len(new_V) != n:
        print('Error : new V and current V doesn t have the same size.')
        return 0
    rms=0
    for s in range(n):
        rms += (cur_V - new_V)**2
    rms = m.sqrt(rms)/n
    return rms

def optimal_policy(cur_V, T,possible_pi):
    n=len(cur_V)
    pi=np.zeros((1,n))
    for s in range(n):
        if len(possible_pi[s])==1:
            pi[s]=possible_pi[s]
        else:
            sum = np.zeros((1,len(possible_pi[s])))
            for a in possible_pi[s]:
                for s2 in range(n):
                    sum[a] += T[s][a][s2]*cur_V[s2]
            pi[s] = np.argmax(sum)
    return pi