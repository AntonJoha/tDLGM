import matplotlib.pyplot as plt
from parse_data import get_data, get_modified_values, get_binary_values, make_data_scalar
from datetime import datetime
import torch
import pickle
import numpy as np
import json
import random
from data_gen import Datagen


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gen = Datagen(device)


def get_lots_yhat(m,x):
    res = []
    m.eval()
    m.make_xi()
    prev = x[0]
    for i in range(100000):
        un = prev.unsqueeze(0)
        val, _ = m(un)
        val = val.detach()
        prev = torch.cat([prev[1:], val[0]], dim=0)
        res.append(val.detach().cpu()[0])
        
    return res

def get_yhat(m,m_r, m_t, x, x_1):
    res = []
    m.eval()

    for i, i_1 in zip(x,x_1):
        _,_, state = m_t(i.unsqueeze(0))
        rec = m_r(i_1.unsqueeze(0)[:,-1,:])
        m.set_xi(rec[-1])
        val = m(state)
        res.append(val.detach().cpu()[0][-1])
    return torch.tensor(res)

def get_bin(y, low, high, conf, step_size):
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

def bin_plot(y, low, high, conf, step_size, extra_str=""):
    count = get_bin(y, low, high, conf, step_size)
    names = list(count.keys())
    values = list(count.values())
    fig, ax = plt.subplots(figsize=(12,12))
    ax.bar(range(len(count) -1), values[:-1], tick_label=names[:-1])
    fig.savefig("images/%s%s.png" % (str(conf), "barfig_" + extra_str ))

   
def add_run(s,to_add, filename="entries.json"):
    
    entries = None
    filename = "results/%s%s" % (s, filename)
    try:
        with open(filename, "r") as f:
            entries = json.loads(f.read())
    except:
        # If there is no file
        entries = []
    entries.append(to_add)
    
    with open(filename, "w") as f:
        f.write(json.dumps(entries, indent=4))
    

def _generate_test(m, m_r, m_t,variance=0, prob=0,times=10, seq_len=10):

    x_test, y_test, x_test_1 = gen.get_test_data(seq_len, variance, prob)

    
    results = []
    results_diff = []
    y_true = y_test.squeeze()

    for i in range(times):
        res = get_yhat(m,m_r, m_t, x_test, x_test_1).to(device)
        results.append(res)
        diff = torch.sub(res, y_true)
        results_diff.append(diff)
        
    return results, results_diff

def generate_test(m, m_r, m_t, conf, times=10):

    to_return = []

    var = [0.001, 0.01, 0.005, 0.05, 0]
    data_prob = [i/5 for i in range(1,6)]
    for v, p in zip(var, data_prob):
        results, results_diff = _generate_test(m, m_r, m_t, conf["variance"], conf["data_prob"], times, conf["seq_len"])
        to_return.append({ "variance": v, 
                          "probability": p,
                          "results": [r.detach().cpu().tolist() for r in results],
                          "diff": [r.detach().cpu().tolist() for r in results_diff]})
    
    t = results_diff[0]
    for i in range(1,len(results_diff)):
        t = torch.cat((t, results_diff[i]))

   
    return to_return
    

def _future_steps(m,m_r, m_t, x, x_1,y, steps=1):
    res = []
    m.eval()

    for i in range(len(x)):
        

        x_u = x[i].unsqueeze(0)
        x_1_u = x_1[i].unsqueeze(0)
        _,_,t = m_t(x_u)
        rec = m_r(x_1_u[:,-1,:])
        m.set_xi(rec[-1])
        r= m(t)
        x_u = torch.cat((x_u[:,1:,:], r.unsqueeze(0)), dim=1)
        # Cheap way of protecting against out of index errors.
        y_hat = []
        y_true = []
        try:
            for j in range(steps):
                m.make_xi()
                _,_,t = m_t(x_u)
                val = m(t)
                y_hat.append(val.detach().cpu()[0][-1])

                x_u = torch.cat((x_u[:,1:,:], val.unsqueeze(0)), dim=1)

                y_true.append(y[j+1+i].detach().cpu())
            res.append({"generated": [r.detach().cpu().tolist() for r in y_hat], "true": [r.detach().cpu().tolist()[0] for r in y_true]})
        except:
            pass

    return res
        
            
def future_steps(m,m_r, m_t, x, x_1,y ):
    to_return = []

    steps = [1,2,5,8,10,15,20,25,30]

    for s in steps:
        to_return.append({"steps": s, "result": _future_steps(m,m_r, m_t, x, x_1,y, s)})

    return to_return
   
def evaluate_future_steps(f):
    to_return = []

    for i in f:
        count = 0
        squared_sum = 0
        for v in i["result"]:
            for y_hat, y in zip(v["generated"], v["true"]):
                squared_sum += (y_hat - y)**2
                count += 1
        to_return.append({"steps": i["steps"], "diff_squared_mean": squared_sum/count})
    return to_return
    

def evaluate_model(m ,m_r, m_t,x,y, x_1,x_test,y_test, x_test_1,conf, draw_images=True):
    to_add = conf
    conf_str = str(conf).replace(" ", "").replace("/","").replace(":","").replace("'", "").replace("{","").replace("}","")
    model_name = conf_str


    to_add["test_run"] = generate_test(m, m_r, m_t, conf)

    to_add["future"] = future_steps(m, m_r, m_t, x, x_1, y)
    to_add["future_eval"] = evaluate_future_steps(to_add["future"])



    to_add["y_hat_forced"] = [get_yhat(m, m_r, m_t,x, x_1).cpu().numpy().tolist() for _ in range(2)]

    
    add_run(conf_str, to_add)
    
