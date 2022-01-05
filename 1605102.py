import pandas as pd
import numpy as np
from scipy.linalg import eig

# obs = ("normal", "cold", "dizzy")
obs = list()
with open('data.txt') as f:
    for value in f:
        obs.append(float(value))
        # print(value)


states = ("el_nino", "la_nina")

# trans_p = {
#     "Healthy": {"Healthy": 0.7, "Fever": 0.3},
#     "Fever": {"Healthy": 0.4, "Fever": 0.6},
# }
lines = []
with open('parameters.txt') as f:
    lines = f.readlines()
count = int(lines[0])
trans_array = np.empty((count, count), float)


for i in range(count):
    j = 0
    for value in lines[i+1].split():
        trans_array[i][j] = float(value)
        j += 1
# print(trans_array)


# start_p = {"Healthy": 0.6, "Fever": 0.4}
transition_mat = np.matrix(trans_array)

S, U = eig(transition_mat.T)
stationary = np.array(U[:, np.where(np.abs(S - 1.) < 1e-8)[0][0]].flat)
start_p = stationary / np.sum(stationary)
print(start_p)


# emit_p = {
#     "Healthy": {"normal": 0.5, "cold": 0.4, "dizzy": 0.1},
#     "Fever": {"normal": 0.1, "cold": 0.3, "dizzy": 0.6},
# }


# def viterbi(obs, states, start_p, trans_p, emit_p):
#     V = [{}]
#     for st in states:
#         V[0][st] = {"prob": start_p[st] * emit_p[st][obs[0]], "prev": None}
#     # Run Viterbi when t > 0
#     for t in range(1, len(obs)):
#         V.append({})
#         for st in states:
#             max_tr_prob = V[t - 1][states[0]]["prob"] * trans_p[states[0]][st]
#             prev_st_selected = states[0]
#             for prev_st in states[1:]:
#                 tr_prob = V[t - 1][prev_st]["prob"] * trans_p[prev_st][st]
#                 if tr_prob > max_tr_prob:
#                     max_tr_prob = tr_prob
#                     prev_st_selected = prev_st

#             max_prob = max_tr_prob * emit_p[st][obs[t]]
#             V[t][st] = {"prob": max_prob, "prev": prev_st_selected}

#     for line in dptable(V):
#         print(line)

#     opt = []
#     max_prob = 0.0
#     best_st = None
#     # Get most probable state and its backtrack
#     for st, data in V[-1].items():
#         if data["prob"] > max_prob:
#             max_prob = data["prob"]
#             best_st = st
#     opt.append(best_st)
#     previous = best_st

#     # Follow the backtrack till the first observation
#     for t in range(len(V) - 2, -1, -1):
#         opt.insert(0, V[t + 1][previous]["prev"])
#         previous = V[t + 1][previous]["prev"]

#     print("The steps of states are " + " ".join(opt) +
#           " with highest probability of %s" % max_prob)


# def dptable(V):
#     # Print a table of steps from dictionary
#     yield " " * 5 + "     ".join(("%3d" % i) for i in range(len(V)))
#     for state in V[0]:
#         yield "%.7s: " % state + " ".join("%.7s" % ("%lf" % v[state]["prob"]) for v in V)


# viterbi(obs,
#         states,
#         start_p,
#         trans_p,
#         emit_p)
