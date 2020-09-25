#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 09:28:33 2020

@author: iad
"""





#entree
R
T
gamma
eps
#var intermediaire
possible_Pi = find_possible_Pi (T)
#var init
n_state = len(T)
curr_V = [0]*n_state
next_V = [0]*n_state
i = 0
diff = eps +1

while diff >= eps:
  next_V = update_V(cur_V,gamma,T,R,possible_Pi)
  diff = RMS_error(curr_V,next_V)
  curr_V = next_V.copy()
  i = i+1
  
optimal_Pi = optimal_policy (curr_V,T,possible_Pi)


def find_possible_Pi (T):
  '''
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
  for s in range (n_state):
    for a in range (len(T[s])):
      # an action a is not available from state s if and only if all the probability transitions to states s' are equal to 0. otherwise it is equal to 1
      is_action = sum(T[s][a]) > 0
      if is_action:
        possible_Pi[s].append(a)
  return possible_Pi


def test (a,b):
  return a+b