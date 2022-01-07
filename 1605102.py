import pandas as pd
import numpy as np
from scipy.linalg import eig
from scipy.stats import norm
import math


observation = list()
with open('data.txt') as f:
    for value in f:
        observation.append(float(value))
        # print(value)
    f.close()


states = ["el_nino", "la_nina"]
# print(type(states))

lines = []
with open('parameters.txt') as f:
    lines = f.readlines()
    f.close()
count = int(lines[0])
transition_probability = np.empty((count, count), float)

for i in range(count):
    j = 0
    for value in lines[i+1].split():
        transition_probability[i][j] = float(value)
        j += 1
# print(transition_probability)

mean_array = []
variance_array = []

for value in lines[-2].split():
    i = 0
    mean_array.append(float(value))
    i += 1

for value in lines[-1].split():
    i = 0
    variance_array.append(math.sqrt(float(value)))
    i += 1


# print(mean_array)
# print(variance_array)

transition_matrix = np.matrix(transition_probability)

S, U = eig(transition_matrix.T)
stationary = np.array(U[:, np.where(np.abs(S - 1.) < 1e-8)[0][0]].flat)
start_p = stationary / np.sum(stationary)
# print(start_p)


def viterbi(observation, states, start_p, transition_probability):
    Viterbi = [{}]
    i = 0
    for state in states:
        emission_probability = math.log(
            norm.pdf(observation[0],  mean_array[i], variance_array[i]))
        Viterbi[0][state] = {"prob": math.log(start_p[i]) +
                             emission_probability, "prev": None}

        i += 1

    for t in range(1, len(observation)):
        Viterbi.append({})
        i = 0
        for state in states:

            max_tr_prob = Viterbi[t - 1][states[0]]["prob"] + \
                math.log(transition_probability[0][i])
            prev_st_selected = states[0]
            j = 1
            for prev_st in states[1:]:
                tr_prob = Viterbi[t - 1][prev_st]["prob"] + \
                    math.log(transition_probability[j][i])
                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_st_selected = prev_st
                j += 1
            emission_probability = math.log(
                norm.pdf(observation[t], mean_array[i], variance_array[i]))
            maximum_probability = max_tr_prob + emission_probability

            Viterbi[t][state] = {"prob": maximum_probability,
                                 "prev": prev_st_selected}
            i += 1

    # print(Viterbi)
    opt = []
    maximum_probability = 0.0
    best_state = None

    for state, data in Viterbi[-1].items():
        if -1 * data["prob"] > maximum_probability:
            maximum_probability = data["prob"]
            best_state = state

    opt.append(best_state)
    previous = best_state

    for t in range(len(Viterbi) - 2, -1, -1):
        opt.insert(0, Viterbi[t + 1][previous]["prev"])
        previous = Viterbi[t + 1][previous]["prev"]

    textfile = open("viterbi_only.txt", "w")
    for i in range(len(opt)):
        textfile.write("\"" + opt[i]+"\"" + "\n")
    textfile.close()


viterbi(observation,
        states,
        start_p,
        transition_probability)
