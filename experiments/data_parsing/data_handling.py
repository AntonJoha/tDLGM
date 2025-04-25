
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os import listdir
import math
import scipy.stats as st
from data_gen import Datagen

gen = Datagen(None)

_, y_true, _ = gen.get_true_data(1)
_, y_test, _ = gen.get_test_data(1)

best_future_eval_sum_cache = {}
best_reconstruction_cache = {}

def get_file(filename=""):
    with open(filename, "r") as f:
        return json.loads(f.read())

def get_all_files(folder=""):
    files = listdir(folder)
    return [i for i in files if "json" in i ]

def bin_plot(y, low, high, step_size, extra_str=""):
    count = get_bin(y, low, high, step_size)
    names = list(count.keys())
    values = list(count.values())
    fig, ax = plt.subplots(figsize=(12,12))
    ax.bar(range(len(count) -1), values[:-1], tick_label=names[:-1])

def _bin_diff(q_bar, p_bar):
    q_val = np.array([q_bar[k] for k in q_bar])
    p_val = np.array([p_bar[k] for k in p_bar])
    q = q_val/np.sum(q_val)
    p = p_val/np.sum(p_val)
    score = 0
    p_len = 0
    for i in p:
        if i != 0:
            p_len += 1
    for i in range(len(q)):
        if p[i] == 0:
            continue
        score += min(q[i]/p[i],1)/p_len
    return score

bin_diff_cache = {}
def bin_diff(model, low = 0, high = 1, stepsize = 0.05):
    cache = model + str(low) + str(high) + str(stepsize)
    if cache in bin_diff_cache:
        return bin_diff_cache[cache]

    options = [model + "/" + i for i in get_all_files(model)]

    best_score = 0
    best = None

    p = get_bin(y_true, low, high, stepsize)
    for o in options:
        f = get_file(o)[0]
        arr = np.concatenate([i for i in f["y_hat"]])
        q = get_bin(arr, low, high, stepsize)
        res = _bin_diff(q,p)
        if res > best_score:
            best_score = res
            best = o
    bin_diff_cache[cache] = {"name": best, "value": best_score}
    return bin_diff_cache[cache]
        
def get_bin(y, low, high, step_size):
    count = {}
    values = np.around(np.arange(low, high + step_size*2, step_size), decimals=2)
    for i in values:
        count[i] = 0
    for i in y:
        prev = values[0]
        for j in values:
            if i < j:
                count[prev] += 1
                break
            prev = j
            
    
    return count

def get_bin_order(y, low, high, step_size):
    to_return = []
    values = np.around(np.arange(low, high + step_size*2, step_size), decimals=2)
    
    for i in y:
        prev = values[0]
        for j in values:
            if i < j:
                to_return.append(prev)
                break
            prev = j
            
    
    return to_return

def get_variance_configs():

    return [{"variance": 0.001, "probability": 0.2}, 
            {"variance": 0.01, "probability": 0.4},
            {"variance": 0.005, "probability": 0.6},
            {"variance": 0.05, "probability": 0.8},
            {"variance": 0, "probability": 1}]

def plot_all_future(model: str,path):

    options = [model + "/" + i for i in get_all_files(model)]
    count = 0
    for o in options:
        plot_future(o,title=str(count),name=path + o.split("/")[-1][:-4] + ".png")
        count+=1
        

def plot_future(path, name=None, title=None):
    fig, ax = plt.subplots(4, layout="constrained")

    f = get_file(path)[0]
    for i in range(4):
        ev = f["future"][-1]["result"][i]
        ax[i].plot(ev["generated"])
        ax[i].plot(ev["true"])
    
    for a in ax:
        a.set_xlabel("Timestep")
        a.set_ylabel("Fraction of GPU")

    if title is not None:
        fig.suptitle(title)
    if name is not None:
        fig.savefig(name)
        plt.close("all")
    else:
        plt.show()
        plt.close()



def plot_file(path, name=None, title=None):
    fig, ax = plt.subplots(2, layout="constrained")

    f = get_file(path)[0]
    y_hat = np.array(f["y_hat"][0])
    y_hat_test = np.array(f["test_run"][0]["results"][0])
    
    
    ax[0].plot(range(1,201) , y_true[:200])
    ax[0].plot(range(1,201) , y_hat[:200])
    ax[0].set_title("True generation")
    
    ax[1].plot(range(1,201) , y_test[:200])
    ax[1].plot(range(1,201), y_hat_test[:200])
    ax[1].set_title("Test dataset recognition")
    
    for a in ax:
        a.set_xlabel("Timestep")
        a.set_ylabel("Fraction of GPU")

    if title is not None:
        fig.suptitle(title)
    if name is not None:
        fig.savefig(name)
        plt.close("all")
    else:
        plt.show()
        plt.close()


def _lowest_mean_squared_individual_reconstruction_error(f):


    score = 100
    arr = np.array([])
    
    for i in f:
        res = np.mean(np.array([[abs(k) for k in i] for j in i]))
        if res < score:
            score = res
        
    return res


def filter_rnn(model):
    
    f = get_file("output_rnn_prob.json")
    to_return = []
    for p in f:
        if p["p_score"] < 0.9:
            to_return.append(model + "/"  + p["name"])
    return to_return

lowest_mean_squared_individual_reconstruction_error_cache = {}

def lowest_mean_squared_individual_reconstruction(model:str, variance:float = 0.001, probability:float = 0.2):

    index = model + str(variance) + str(probability)
    if index in lowest_mean_squared_individual_reconstruction_error_cache:
        return lowest_mean_squared_individual_reconstruction_error_cache[index]

    options = []
    if model.find("rnn") != -1:
        options = filter_rnn(model)
    else:
        options = [model + "/" + i for i in get_all_files(model)]

        
    
    best_score = 100000000
    best = None
    for o in options:
        f = get_file(o)
        for r in f:
            for i in r["test_run"]:
                if variance == i["variance"] and probability == i["probability"]:
                    res = _lowest_mean_squared_individual_reconstruction_error(i["diff"])
                    if res < best_score:
                        best_score = res
                        best = o
    lowest_mean_squared_individual_reconstruction_error_cache[index] = {"name": best, "variance": variance, "probability": probability, "value":best_score}
    return lowest_mean_squared_individual_reconstruction_error_cache[index]
    

def _lowest_mean_squared_reconstruction(f):


    score = 0
    arr = np.array([])

    for i in f:
        a = np.array(i)
        arr = np.concatenate((arr,a))
    res = np.mean(np.array([abs(i) for i in arr]))
    return res


lowest_mean_squared_reconstruction_error_cache = {}

def lowest_mean_squared_reconstruction(model:str, variance:float = 0.001, probability:float = 0.2):

    index = model + str(variance) + str(probability)
    if index in lowest_mean_squared_reconstruction_error_cache:
        return lowest_mean_squared_reconstruction_error_cache[index]

    options = []
    if "rnn" in model:
        options = filter_rnn(model)
    else:
        options = [model + "/" + i for i in get_all_files(model)]


    best_score = 100000000
    best = None
    for o in options:
        f = get_file(o)
        for r in f:
            for i in r["test_run"]:
                if variance == i["variance"] and probability == i["probability"]:
                    res = _lowest_mean_squared_reconstruction(i["diff"])
                    if res < best_score:
                        best_score = res
                        best = o
    lowest_mean_squared_reconstruction_error_cache[index] = {"name": best, "variance": variance, "probability": probability, "value":best_score}
    return lowest_mean_squared_reconstruction_error_cache[index]
    

    

def _distribution_width(f, lower, upper, variance, prob):

    low = -10000000000
    high = 1000000000
    best_mean = 0
    best_var = 0
    for i in f:
        if i["variance"] != 0:
            continue
        count = 0
        mean = 0
        for d in i["diff"]:
            mean += sum([j**2 for j in d])
            count += len(d)
        mean /=count
        var = 0
        for d in i["diff"]:
            var += sum([(j-mean)**2 for j in d])
        var /= (count - 1)
        std = math.sqrt(var)

        curr_low = lower*std + mean
        curr_high = upper*std + mean

        if curr_low > low and curr_high < high and curr_high - curr_low < high - low:
            low = curr_low
            high = curr_high
            best_mean = mean
            best_var = var
            
    return low, high, best_mean, best_var
        
    
    

# Calculate the ranges inside which % of all values lie (according to the normal dist)
def distribution_width(model:str, perc:float = 0.5, variance=0, prob=1):
    if perc < 0 or perc > 1:
        return None

    upper = (1-perc)/2 + perc # Two tailed
    lower = 1-upper
    upper = st.norm.ppf(upper)
    lower = st.norm.ppf(lower)

    options = [model + "/" + i for i in get_all_files(model)]

    best = 1000000
    best_low = -10000000000
    best_high = 1000000000
    best_mean = 0
    best_var = 0
    best = None
    for o in options:
        f = get_file(o)
        for r in f:
            low, high, mean, var = _distribution_width(r["test_run"], lower,upper, variance, prob)
            if low > best_low and high < best_high and high - low < best_high - best_low:
                best_low = low
                best_high = high
                best_mean = mean
                best_var = var
                best = o

    return {"name": best, "low": best_low, "high": best_high, "mean": best_mean, "var": best_var}
    




def _best_future_eval_sum(f):
    total = 0
    for i in f:
        total += float(i["diff_squared_mean"])
    return total
        
def _best_reconstruction(f):
    scores = []
    for i in f:
        for j in i["diff"]:
            scores.append(sum([abs(v) for v in j])/len(j))
    return sum(scores)/len(scores)

    
def best_reconstruction(model:str):

    if model in best_reconstruction_cache:
        return best_reconstruction_cache[model]

    
    options = [model + "/" + i for i in get_all_files(model)]

    best_score = 100000000
    best = None
    for o in options:
        f = get_file(o)
        for r in f:
            res = _best_reconstruction(r["test_run"])
            if res < best_score:
                best_score = res
                best = o
    best_reconstruction_cache[model] = {"name": best, "value":best_score}
    return best_reconstruction_cache[model]


def _future_eval_allowed(f):
    return 0 #f[0]["diff_squared_mean"]/f[-1]["diff_squared_mean"]

lowest_future_eval_cache = {}
def lowest_future_eval(model:str):

    if model in lowest_future_eval_cache:
        return lowest_future_eval_cache[model]

    options = [model + "/" + i for i in get_all_files(model)]

    best_score = 10000000
    best = None

    for o in options:
        f = get_file(o)
        for r in f:
            if _future_eval_allowed(r["future_eval"]) > 0.9:
                continue
            if best_score > r["future_eval"][0]["diff_squared_mean"]:
                best_score = r["future_eval"][0]["diff_squared_mean"]
                best = o
    lowest_future_eval_cache[model] = {"name": best, "value": best_score}
    return lowest_future_eval_cache[model]


def best_future_eval_sum(model:str):


    if model in best_future_eval_sum_cache:
        return best_future_eval_sum_cache[model]
        
    options = [model + "/" + i for i in get_all_files(model)]

    best_score = 1000000000000000000
    best = None
    for o in options:
        f = get_file(o)
        for r in f:
            res = _best_future_eval_sum(r["future_eval"])
            if res < best_score:
                best_score = res
                best = o
    best_future_eval_sum_cache[model] = {"name": best, "value": best_score}
    return best_future_eval_sum_cache[model]
        
        