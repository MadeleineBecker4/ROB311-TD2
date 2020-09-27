import aux_functions as af
import numpy as np
import matplotlib.pyplot as plt

# MAIN

# environnement parameters
R = [0, 0, 1, 10]
x = 0.25
y = 0.25

T = np.zeros((4, 3, 4))
T[0][0] = [0, 0, 0, 0]
T[0][1] = [0, 1, 0, 0]
T[0][2] = [0, 0, 1, 0]

T[1][0] = [0, 1-x, 0, x]
T[1][1] = [0, 0, 0, 0]
T[1][2] = [0, 0, 0, 0]

T[2][0] = [1-y, 0, 0, y]
T[2][1] = [0, 0, 0, 0]
T[2][2] = [0, 0, 0, 0]

T[3][0] = [1, 0, 0, 0]
T[3][1] = [0, 0, 0, 0]
T[3][2] = [0, 0, 0, 0]

# iteration value algorithm parameters
gamma = 0.9
eps = 0.01
# instrumentation parameters
isInstrumentionON = True  # if True, save variable values during the iteration
# of the algorithm and plot them

# compute the possible action for each state
possible_Pi = af.find_possible_Pi(T)

# iteration value algorithm to find the optimal policy
# init
n_state = len(T)
curr_V = [0]*n_state
next_V = [0]*n_state
i = 0
diff = eps + 1
# init instruentation variables
if isInstrumentionON:
    instrum_diff = []
    instrum_pi = [[] for k in range(n_state)]

# compute V*
while diff >= eps:
    next_V = af.update_V(curr_V, gamma, T, R, possible_Pi)
    diff = af.RMS_error(curr_V, next_V)
    curr_V = next_V.copy()
    i = i+1
    # intrumentation
    if isInstrumentionON:
        optimal_Pi = af.optimal_policy(curr_V, T, possible_Pi)
        for s in range(n_state):
            instrum_pi[s].append(optimal_Pi[s])
        instrum_diff.append(diff)

# find the optimal policy
optimal_Pi = af.optimal_policy(curr_V, T, possible_Pi)

# present results
print("the estimated optimal policy is :", optimal_Pi)
print("number of iteration :", i)
print("final V =", curr_V)
print("final RMS :", diff)
print("parameters : gamma =", gamma, ", eps =", eps)


if isInstrumentionON:
    if i == 0:
        print("error : 0 iteration in the iteration value algorithm")
    else:
        index = np.arange(1, i+1)
        plt.plot(index, instrum_diff, "r", label="RMS")
        plt.xlabel('number of iteration')
        plt.ylabel("RMS between consecutive vector V*")
        plt.plot(index, eps*np.ones(i), "b", label="eps")
        plt.legend()
        plt.show()
        for s in range(n_state):
            plt.plot(index, instrum_pi[s], "+")
            plt.xlabel('number of iteration')
            plt.ylabel("Optimal pi(" + str(s) + ")")
            plt.show()
