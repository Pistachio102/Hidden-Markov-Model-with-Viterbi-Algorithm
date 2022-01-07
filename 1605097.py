import pandas as pd
import numpy as np
from scipy.linalg import eig
from scipy.stats import norm
import math


def viterbi(states, observation, transition, mean_sd, stationary):
    V = [{}]
    state_len = len(states)

    for st in range(state_len):
        if st == 0:
            st_v = "El Nino"
        elif st == 1:
            st_v = "La Nina"
        emission = norm.pdf(
            float(observation[0]), mean_sd[0][st], mean_sd[1][st])
        V[0][st_v] = {"probability": math.log(
            stationary[st])+math.log(emission), "prev_state": None}

    for t in range(1, len(observation)):
        V.append({})
        for st in range(state_len):
            if st == 0:
                st_v = "El Nino"
            elif st == 1:
                st_v = "La Nina"
            max_transmission = V[t - 1][states[0]
                                        ]["probability"] + math.log(transition[0][st])
            print("max tr...", max_transmission)
            prev_state_selected = states[0]
            for prev_st in range(1, state_len):
                if(prev_st == 1):
                    prev_st_v = "La Nina"
                if prev_st == 0:
                    prev_st_v = "El Nino"
                # print("Checking...",V[t-1][prev_st_v]["probability"])
                #print("Checking trans....",transition[prev_st][st])
                tr_prob = V[t - 1][prev_st_v]["probability"] + \
                    math.log(transition[prev_st][st])
                print("tr prob...", tr_prob)
                if tr_prob > max_transmission:
                    max_transmission = tr_prob
                    prev_state_selected = prev_st_v
            emission = norm.pdf(
                float(observation[t]), mean_sd[0][st], mean_sd[1][st])
            max_prob = (max_transmission + math.log(emission))
            V[t][st_v] = {"probability": max_prob,
                          "prev_state": prev_state_selected}
    opt = []
    max_prob = 0.0
    best_st = None
    # for i in range(len(V)):
    #      print("len.......hala.",V[i].items())
    for st, data in V[-1].items():
        print("data prob...", data["probability"])
        if (data["probability"]*-1) > max_prob:
            print("dhukse....")
            max_prob = data["probability"]
            best_st = st
    opt.append(best_st)
    previous = best_st
    for t in range(len(V) - 2, -1, -1):
        # print("Final....",previous)
        opt.insert(0, V[t + 1][previous]["prev_state"])
        previous = V[t + 1][previous]["prev_state"]
    textfile = open("output_without_baum.txt", "w")
    for i in range(len(opt)):
        textfile.write("\"" + opt[i]+"\"" + "\n")
    textfile.close()
