#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 09:28:33 2020

@author: iad
"""

import aux_functions as af

# MAIN

#entree
R = [0,0,3,10]
T = []
gamma = 0.9
eps = 0.0001


#var intermediaire
possible_Pi = af.find_possible_Pi (T)
#var init
n_state = len(T)
curr_V = [0]*n_state
next_V = [0]*n_state
i = 0
diff = eps +1

while diff >= eps:
  next_V = af.update_V(curr_V,gamma,T,R,possible_Pi)
  diff = af.RMS_error(curr_V,next_V)
  curr_V = next_V.copy()
  i = i+1
  
optimal_Pi = af.optimal_policy (curr_V,T,possible_Pi)

