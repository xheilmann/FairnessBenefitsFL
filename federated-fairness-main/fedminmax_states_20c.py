"""
-------------------------------------------------------------------------------------------------------------

fedminmax_cifar_iid_5c_v1.py
by Oscar, April 2024

-------------------------------------------------------------------------------------------------------------

Simulating using Flower:
  The FedMinMax strategy
  On the CIFAR-10 dataset with the iid partition.
  For 5 clients
Data is saved to JSON.

-------------------------------------------------------------------------------------------------------------

Declarations, functions and classes:
  client_fn - creates Flower clients when required to avoid depleting RAM
  fit_config - defines the config parameters pass to the client during fit/training
  evaluate - the central evaluation function
  fit_callback - the callback following a complete round of client fit, this is used for calculating and
    aggregating the fairness metrics

-------------------------------------------------------------------------------------------------------------

Usage:
Ensure the necessary packages are installed using:
  $ pip install -r requirements.txt
Alter the config parameters as required.
Run the script.
View results in ./Results/{path_extension}.json

-------------------------------------------------------------------------------------------------------------

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

-------------------------------------------------------------------------------------------------------------
"""
import argparse
import ast
# Library imports
# from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# from torchvision.datasets import EMNIST
# from torch.utils.data import DataLoader, random_split
# import random
# from matplotlib import pyplot as plt
# from math import comb
# from itertools import combinations
import json
from datetime import timedelta
import time

from torch.utils.hipify.hipify_python import compute_stats

import wandb

start = time.perf_counter()
import flwr as fl
from flwr.common import Metrics
# User defined module imports:
from source.shapley import Shapley
from source.states_net import Net, train, test
from source.client import FlowerClient, DEVICE, get_parameters, set_parameters
from source.load_states import load_iid
from source.fedminmax import FedMinMax, data_preprocess


print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)

parser = argparse.ArgumentParser()
parser.add_argument('--num-clients', default = 20, help='number of FL clients', type= int)
parser.add_argument('--epochs', help = "epochs for  local training", default = 10, type=int)
parser.add_argument('--rounds', default = 10, type= int)
parser.add_argument('--selection-rate', type = float, default= 1.0,  help='proportion of clients selected in each round of training')
parser.add_argument('--sensitive-attribute', type= str, default="MAR", help='name of the sensitive attributes')
parser.add_argument('--comp-attribute',  type= str, default="SEX",  help='name of the comparision attributes')
parser.add_argument("--fedminmax-lr", default = 0.02, help="value of overall lr", type = float)
parser.add_argument("--fedminmax-adverse-lr", default = 0.001, help="value of local lr", type= float)
parser.add_argument("--dataset-name", type=str, default="income", help="name of the dataset")
parser.add_argument("--batch-size", default=528, type=int, help="batch size for training")
parser.add_argument("--cluster", default=0, type=int, help="cluster = 0 all clients, cluster = 1 clients that are unfair towards SEX, cluster=2 clients that are unfair towards MAR")

opt = parser.parse_args()



 
# Key parameter and data storage variables:
NUM_CLIENTS = opt.num_clients
LOCAL_EPOCHS = opt.epochs
NUM_ROUNDS = opt.rounds
BATCH_SIZE = opt.batch_size
SELECTION_RATE = opt.selection_rate # what proportion of clients are selected per round
SENSITIVE_ATTRIBUTES = [(0,0), (0,1), (1,0), (1,1)]
SENS_ATT = opt.sensitive_attribute
COMP_ATT= opt.comp_attribute
FEDMINMAX_LR = opt.fedminmax_lr
FEDMINMAX_ADVERSE_LR = opt.fedminmax_adverse_lr
cluster = opt.cluster
path_extension = f'FedMinMax_states_iid_{NUM_CLIENTS}C_{int(SELECTION_RATE * 20)}PC_{LOCAL_EPOCHS}E_{NUM_ROUNDS}R'
dataset_name =opt.dataset_name

if dataset_name == "income":
    in_feat= 40
else:
    in_feat = 79
data = {
    "rounds": [],
    "general_fairness": {
          "f_j": [],
          "f_g": [],
          "f_r": [],
          "f_o": [],
          "f_comp_g":[]},
      "config": {
          "num_clients": NUM_CLIENTS,
          "local_epochs": LOCAL_EPOCHS,
          "num_rounds": NUM_ROUNDS,
          "batch_size": BATCH_SIZE,
          "selection_rate": SELECTION_RATE,
          "sensitive_attributes": SENSITIVE_ATTRIBUTES,
          "sens_att":SENS_ATT,
          "comp_att":COMP_ATT,
          "cluster":cluster,
          "datasetname":dataset_name,
          "fedminmax_lr": FEDMINMAX_LR,
          "fedminmax_adverse_lr": FEDMINMAX_ADVERSE_LR
          },
      "per_client_data": {
            "shap": [],
            "accuracy": [],
            "avg_eop": [],
          "comp_eop":[],
            "gains": [],
      "stats":[],
      "stats_comp":[]}}

wandb.init(project="multiobj-FL-wo-race", config=data["config"])


# Key experiment specific functions:
def client_fn(cid) -> FlowerClient:
    """
    Instances of clients are only created when required to avoid depleting RAM.
    """
    net = Net(in_feat).to(DEVICE)
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    return FlowerClient(cid, net, trainloader, valloader, test, train)

def fit_config(server_round: int):
    """
    Return training configuration dict for each round.
    """
    config = {
        "server_round": server_round, # The current round of federated learning
        "local_epochs": LOCAL_EPOCHS,
        "sensitive_attributes": SENSITIVE_ATTRIBUTES,
        "sens_att":SENS_ATT,
        "comp_att":COMP_ATT
    }
    return config

def evaluate(server_round: int,
             parameters: fl.common.NDArrays,
             config: Dict[str, fl.common.Scalar]) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    """
    Used for centralised evaluation. This is enacted by flower before the Federated Evaluation.
    Runs initially before FL begins as well.
    """
    net = Net(in_feat).to(DEVICE)
    shap.aggregatedRoundParams = parameters
    set_parameters(net, parameters)
    loss, accuracy, _,_,_,_ = test(net, testloader, sensitive_labels= SENSITIVE_ATTRIBUTES, sens_att=SENS_ATT, comp_att=COMP_ATT)
    shap.f_o = accuracy # stored incase the user wants to define orchestrator fairness by the central eval performance, usused by default
    shap.centralLoss = loss
    shap.round = server_round
    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
    return loss, {"accuracy": accuracy}

def fit_callback(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Called at the end of the clients fit method
    Used to call the Shapley calculation as it is before weight aggregation
    """
    clients = set()
    client_list = []
    parameters = [None for client in range(NUM_CLIENTS)]
    strategy.risks = [metric["risks"] for _,metric in metrics] # updating the risk parameters
    # Why are the parameters we get here the post aggregation ones...?
    for client in metrics:

      cid = client[1]["cid"]
      clients.add(cid)
      parameters[cid] = client[1]["parameters"]
      client_list.append(cid)
      print(cid)
    #if True:
    #  shap.fedSV(clients, parameters)
    # Jain's fairness index (JFI) is used to evaluate uniformity for the fairness metrics
    JFI = lambda x: ((np.sum(x)**2) / (len(x) * np.sum(x**2)))
    # We determine individual fairness using the FedEval accuracy and JFI
    accuracies = np.array([metric["accuracy"] for _,metric in metrics])
    print(accuracies)
    rewards = np.array([metric["reward"] for _,metric in metrics])
    contributions = shap.resultsFedSV
    gains = np.array([accuracies[i] / contributions[metrics[i][1]["cid"]] for i in range(len(metrics))])
    f_j = JFI(gains)
    print(f"Individual fairness, f_j = {f_j}")
    # As we have passed the sensitive labels into fedEval, we calculate f_g
    group_fairness = np.array([metric["group_fairness"] for _,metric in metrics])
    print(group_fairness)
    group_comp_fairness = np.array([metric["group_comp_fairness"] for _, metric in metrics])
    print(group_comp_fairness)
    f_g = [[0 for j in range(len(SENSITIVE_ATTRIBUTES))] for i in range(len(SENSITIVE_ATTRIBUTES))]
    for _, metric in metrics:
        for i in range(len(metric["group_stats"])):
            for j in range(len(metric["group_stats"][i])):
                f_g[i][j] += metric["group_stats"][i][j]
    f_g = [float((f_g[index][0] / f_g[index][2] - f_g[index][1] / f_g[index][3])) for index in range(len(f_g))]
    # Linear mapping the average EOD which is between [-1,1] to [0,1] by taking mod (this is okay as either are unfair just represents difference in false )
    #f_g = 1 - np.mean(np.median(np.absolute(np.array([np.mean(np.array(list(group_dict.values()))) for group_dict in group_fairness]))))
    print(f" {SENS_ATT} Group fairness, f_g = {f_g}")

    f_comp_g = [[0 for j in range(len(SENSITIVE_ATTRIBUTES))] for i in range(len(SENSITIVE_ATTRIBUTES))]
    for _, metric in metrics:
        for i in range(len(metric["group_comp_stats"])):
            for j in range(len(metric["group_comp_stats"][i])):
                f_comp_g[i][j] += metric["group_comp_stats"][i][j]
    f_comp_g = [float((f_comp_g[index][0] / f_comp_g[index][2] - f_comp_g[index][1] / f_comp_g[index][3])) for index in range(len(f_comp_g))]
    # Linear mapping the average EOD which is between [-1,1] to [0,1] by taking mod (this is okay as either are unfair just represents difference in false )
    # f_g = 1 - np.mean(np.median(np.absolute(np.array([np.mean(np.array(list(group_dict.values()))) for group_dict in group_fairness]))))
    print(f" {COMP_ATT} Group fairness, f_g = {f_comp_g}")
    # We calculate incentive fairness using the reward (accuracy) and contributions
    # We only get accuracies of the evaluated client set, we match the contribution by cid
    reward_gains = np.array([rewards[i] / contributions[metrics[i][1]["cid"]] for i in range(len(metrics))])
    if np.sum(reward_gains) == 0: # Used to catch the case where the accuracy is zero for all clients which would break JFI
      f_r = 1 # if all are zero, it is technically uniform
    else:
      f_r = JFI(reward_gains)
    print(f"Incentive fairness, f_r = {f_r}")
    # Obtain the orchestrator fairness back from the Shapley class
    # f_o = shap.f_o # centralised evaluation option
    sum_column0 = np.sum(accuracies[:, 0])
    sum_column1 = np.sum(accuracies[:, 1])
    f_o = sum_column0/sum_column1

    print(f"Global Accuracy = {f_o}")
    shap_v=[t[1] for t in sorted(zip(client_list, contributions))]
    accuracy=[t[1] for t in sorted(zip(client_list, accuracies))]
    avg_eop =[t[1] for t in sorted(zip(client_list, group_fairness))]
    comp_eop=[t[1] for t in sorted(zip(client_list, group_comp_fairness))]
    gains=[t[1] for t in sorted(zip(client_list, gains))]
    stats=[t[1] for t in sorted(zip(client_list, np.array([metric["group_stats"] for _, metric in metrics])))]
    stats_comp=[t[1] for t in sorted(zip(client_list, np.array([metric["group_comp_stats"] for _, metric in metrics])))]

    # Saving metrics to dictionary for JSON storage:
    data["rounds"].append(shap.round)
    data["general_fairness"]["f_j"].append(f_j)
    data["general_fairness"]["f_g"].append(f_g)
    data["general_fairness"]["f_comp_g"].append(f_comp_g)
    data["general_fairness"]["f_r"].append(f_r)
    data["general_fairness"]["f_o"].append(f_o)
    data["per_client_data"]["shap"].append(shap_v)
    data["per_client_data"]["accuracy"].append(accuracy)
    data["per_client_data"]["avg_eop"].append(avg_eop)
    data["per_client_data"]["comp_eop"].append(comp_eop)
    data["per_client_data"]["gains"].append(gains)
    data["per_client_data"]["stats"].append(stats)
    data["per_client_data"]["stats_comp"].append(stats_comp)
    wandb.log({"global_acc":f_o,"disparity":np.max(f_g), "client_acc":accuracy, "global_sens_fairness":f_g ,"global_comp_fairness":f_comp_g,
                "client_comp_fairness": comp_eop, "client_stats":stats, "client_comp_stats":stats_comp}, step=shap.round)
    return {"f_j": f_j, "f_g": f_g, "f_r": f_r, "f_o": f_o}


# Gathering the data:
trainloaders, valloaders, testloader, _ = load_iid(NUM_CLIENTS, BATCH_SIZE, SENS_ATT, COMP_ATT, dataset_name, cluster)
# Creating Shapley instance:
shap = Shapley(testloader, test, set_parameters, NUM_CLIENTS, Net(in_feat).to(DEVICE))
# Create FedAvg strategy:
strategy = FedMinMax(
    lr = FEDMINMAX_LR,
    adverse_lr = FEDMINMAX_ADVERSE_LR,
    sensitive_attributes = SENSITIVE_ATTRIBUTES,
    sens_att=SENS_ATT,
    comp_att=COMP_ATT,
    dataset_information = data_preprocess(NUM_CLIENTS, trainloaders, valloaders, SENSITIVE_ATTRIBUTES, SENS_ATT),
    fraction_fit=SELECTION_RATE, # sample all clients for training
    fraction_evaluate=0.0, # Disabling federated evaluation
    min_fit_clients=int(NUM_CLIENTS*SELECTION_RATE), # never sample less that this for training
    min_evaluate_clients=int(NUM_CLIENTS*SELECTION_RATE), # never sample less than this for evaluation
    min_available_clients=NUM_CLIENTS, # has to wait until all clients are available
    # Passing initial_parameters prevents flower asking a client:
    initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(Net(in_feat))),
    # Called whenever fit or evaluate metrics are received from clients:
    fit_metrics_aggregation_fn = fit_callback,
    # Evaluate function is called by flower every round for central evaluation:
    evaluate_fn=evaluate,
    # Altering client behaviour with the config dictionary:
    on_fit_config_fn=fit_config,
    accept_failures = False
)
# Specifying client resources:
client_resources =  {"num_cpus": 3, "num_gpus": 0.0}
if DEVICE.type == "cuda":
    # here we are asigning an entire GPU for each client.
    client_resources = {"num_cpus": 3, "num_gpus": 0.5}
# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),  # We can configure the number of rounds - we want this to have a threshold at which it cuts off
    strategy=strategy,
    client_resources=client_resources,
)
# Saving data:
with open('./Results/'+ path_extension + '.json', "w") as outfile:
    data =repr(data)
    json.dump(data, outfile)
#with open('./Results/'+ path_extension + '.json') as out:
#    d=json.load(out)
#    print(d)
print(f"Elapsed time = {timedelta(seconds=time.perf_counter()-start)}")
