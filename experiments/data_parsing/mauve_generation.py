import matplotlib.pyplot as plt
from parse_data import get_data, get_modified_values, get_binary_values, make_data_scalar
from datetime import datetime
import torch
import pickle
import numpy as np
import json
import mauve 
import numpy as np
import pandas as pd
from os import listdir
import math
import scipy.stats as st
from data_gen import Datagen

gen = Datagen(None)
_, true_data, _ = gen.get_generated_data(1, 0, 0)
true_data = true_data.cpu().numpy()
    
print(true_data)


def get_file(filename=""):
    with open(filename, "r") as f:
        return json.loads(f.read())

def get_all_files(folder=""):
    files = listdir(folder)
    return [i for i in files if "json" in i ]


get_mauve_cache= {}
def get_mauve(model:str):
    if model in get_mauve_cache:
        return get_mauve_cache[model]

    options = [model + "/" + i for i in get_all_files(model)]

    best_score = 0
    best = None
    for o in options:
        f = get_file(o)
        for r in f:
            res = _get_mauve(true_data,r["y_hat"][0])
            if res > best_score:
                print("New_best ", res, " was ", best_score)
                best_score = res
                best = o
            else:
                print("Worse ", res, " best still stands ", best_score)
    get_mauve_cache[model] =      {"name": best, "value": best_score}   
    return get_mauve_cache[model]

    

def _get_mauve(true, generated):

    res = {}
    a = (np.round(np.array(true) - 0.0001,decimals=4))

    a_str = ""
    for i in a:
        a_str += str(i[0]) + ","
    b = np.round(np.array(generated), decimals=4)
    b_str = ""
    for i in b:
        b_str += str(i) + ","
    print("generated: ", a_str[:1000])
    print("true data: ", b_str[:1000])
    n = 20
    p = [b_str[i:i+n] for i in range(0, len(b_str), n)]
    q = [a_str[i:i+n] for i in range(0, len(a_str), n)]
    
    out = mauve.compute_mauve(p_text=p, q_text=q, device_id=0, 
                                max_text_length=256, verbose=False)
    
    
    return out.mauve
    