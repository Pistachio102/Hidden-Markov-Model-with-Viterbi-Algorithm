import pandas as pd
import numpy as np
from scipy.linalg import eig
from scipy.stats import norm


obs = list()
with open('data.txt') as f:
    for value in f:
        obs.append(float(value))
        # print(value)


states = ("el_nino", "la_nina")


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

mean_array = []
std_array = []

for value in lines[-2].split():
    i = 0
    mean_array.append(float(value))
    i += 1

for value in lines[-1].split():
    i = 0
    std_array.append(float(value))
    i += 1


# print(mean_array)
# print(std_array)

transition_mat = np.matrix(trans_array)

S, U = eig(transition_mat.T)
stationary = np.array(U[:, np.where(np.abs(S - 1.) < 1e-8)[0][0]].flat)
start_p = stationary / np.sum(stationary)
# print(start_p)


# norm.pdf(emission_val,  mean_of_distribution, standard_deviation_of_distribution)


def viterbi(obs, states, start_p, trans_p):
    V = [{}]
    for st in states:
        i = 0
        V[0][st] = {"prob": start_p[i] *
                    norm.pdf(obs[0],  mean_array[i], std_array[i]), "prev": None}
        # print(mean_array[i])
        i += 1

# Run Viterbi when t > 0
    for t in range(1, len(obs)):
        V.append({})
        for st in states:
            i = 0
            max_tr_prob = V[t - 1][states[0]]["prob"] * trans_p[0][i]
            prev_st_selected = 0
            for prev_st in states[1:]:
                j = 1
                tr_prob = V[t - 1][prev_st]["prob"] * trans_p[j][i]
                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_st_selected = prev_st
                j += 1
            max_prob = max_tr_prob * \
                norm.pdf(obs[t],  mean_array[i], std_array[i])
            # max_prob = max_tr_prob * emit_p[st][obs[t]]
            V[t][st] = {"prob": max_prob, "prev": prev_st_selected}
            i += 1

    for line in dptable(V):
        print(line)

    opt = []
    max_prob = 0.0
    best_st = None
    # Get most probable state and its backtrack
    for st, data in V[-1].items():
        if data["prob"] > max_prob:
            max_prob = data["prob"]
            best_st = st
    opt.append(best_st)
    previous = best_st

    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]

    print("The steps of states are " + " ".join(opt) +
          " with highest probability of %s" % max_prob)


def dptable(V):
    # Print a table of steps from dictionary
    yield " " * 5 + "     ".join(("%3d" % i) for i in range(len(V)))
    for state in V[0]:
        yield "%.7s: " % state + " ".join("%.7s" % ("%lf" % v[state]["prob"]) for v in V)


viterbi(obs,
        states,
        start_p,
        trans_array)
