"""
-------------------------------------------------------------------------------------------------------------

states_net.py, , v1.0
by Xenia, Nov 2024

-------------------------------------------------------------------------------------------------------------

A binary classification model for the 20 states .

Note: the aim of this project is not model optimisation, well respected baseline models have been selected
such that the development time can be spend on fairness analytics. 

-------------------------------------------------------------------------------------------------------------

Declarations, functions and classes:
- Net - a nn.Module derived class defining the architecture of the neural net/model.
- train - a training function using BinaryCrossEntropyLoss and the Adam optimiser.
- test - testing and evaluating on a separate testset and gathering collecting data on the protected group
    performance.

-------------------------------------------------------------------------------------------------------------

Usage:
Import from root directory using:
    >>> from source.nslkdd_net import Net, train, test
Instantiate:
    >>> net = Net().to(DEVICE)
Gather initial parameters if required:
    >>> get_parameters(Net())

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
# from collections import OrderedDict
# from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# from torchvision.datasets import EMNIST
# from torch.utils.data import DataLoader, random_split
# import random
# from matplotlib import pyplot as plt
# from math import comb
# from itertools import combinations
# import flwr as fl
# from flwr.common import Metrics
# Local import
from .client import DEVICE

class Net(nn.Module):
    """
    A linear model consisting of four fully connected layers:
    Design influenced by: https://machinelearningmastery.com/building-a-binary-classification-model-in-pytorch/ 
    """
    def __init__(self) -> None:
        super(Net, self).__init__()
        # Needs to start with input space as wide as preprocessed inputs, 123 wide including the class label
        self.layer1 = nn.Linear(16, 100, dtype=torch.float64)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(100, 100, dtype=torch.float64)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(100, 50, dtype=torch.float64)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(50, 1, dtype=torch.float64) # ends with single output binary classifier
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        """ A forward pass through the network. """
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x

def train(net, trainloader, epochs: int, option = None, sens_attr="SEX"):
    """
    Train the network on the training set.
    
    Inputs:
        net - the instance of the model
        trainloader - a pytorch DataLoader object.
        epochs - the number of local epochs to train over
        option - a flag to enable alternative training regimes such as ditto
    """
    sum_risks = None # Placeholder for fedminmax return
    def ditto_manual_update(lr, lam, glob):
        """ Manual parameter updates for ditto """
        with torch.no_grad():
            counter = 0
            q = [torch.from_numpy(g).to(DEVICE) for g in glob]
            for p in net.parameters():
                new_p = p - lr*(p.grad + (lam * (p - q[counter])))
                p.copy_(new_p)
                counter += 1
            return

    def fedminmax_manual_update(lr, risk):
        """ Manual parameter updates for FedMinMax strategy"""
        with torch.no_grad():
            for p in net.parameters():
                new_p = p - (lr*(p.grad)*risk)
                p.copy_(new_p)
            return

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0

        for data in trainloader:
            if "ESR" in data:
                label_name = "ESR"
            else:
                label_name = ">50K"  # accounts for NSL-KDD
            labels = (torch.Tensor([[x] for x in data[label_name]]).double()).to(DEVICE)
            data.pop(label_name)
            inputs = torch.from_numpy(np.array([values.numpy() for key,values in data.items()], dtype=float)).to(DEVICE)
            inputs = inputs.mT # transpose required
            batch_size = len(labels)
            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs.squeeze(1), labels.squeeze(1))

            loss.backward()
            if option is not None:

                if option["opt"] == "ditto":
                    ditto_manual_update(option["eta"], option["lambda"], option["global_params"])
                if option["opt"] == "fedminmax":
                    sensitive_attributes = option["attributes"]
                    if sum_risks is None:
                        sum_risks = np.array([0.0 for a in sensitive_attributes])
                    subset_losses = [0 for a in sensitive_attributes]
                    subsets = [[] for a in sensitive_attributes]
                    number_sensitive = [0 for group in sensitive_attributes]
                    # Building mini datasets for each sensitive attribute:
                    sens_labels=[0,1]
                    sens_sex=[0,1]
                    for l in range(batch_size):
                        try:
                            i = sens_labels.index(int(labels[l]))
                            j= sens_sex.index(data[sens_attr][l])
                        except:
                            continue
                        if (i+j)==0 or (i+j)==2:
                            subsets[i*3].append(inputs[l])
                            number_sensitive[i*3] += 1
                        elif i==0:
                            subsets[1].append(inputs[l])
                            number_sensitive[1] += 1
                        else:
                            subsets[2].append(inputs[l])
                            number_sensitive[2] += 1

                    # Testing the mini datasets to determine the risks:

                    for s in range(len(sensitive_attributes)):
                        #print(subsets[s])
                        if subsets[s] == []:
                            continue
                        lbls = torch.Tensor([sensitive_attributes[s][0] for img in range(len(subsets[s]))]).double().to(DEVICE)
                        inpt = (torch.stack(subsets[s])).to(DEVICE)
                        subset_loss = criterion(net(inpt).squeeze(1), lbls)
                        subset_losses[s] += float(subset_loss)
                    # Calculating the key risk parameters
                    #print(lbls, inpt, subset_loss, subset_losses)
                    risks = np.nan_to_num(np.array(subset_losses))
                    print(f"risks:{risks}")
                    sum_risks += risks
                    print(f"sumrisks:{sum_risks}")
                    risk = np.sum((np.array(option["w"]) * risks) / batch_size)
                    print(f"risk:{risk}")
                    fedminmax_manual_update(option["lr"], risk)
            else:
                optimizer.step()
            # Train metrics:
            epoch_loss += loss
            total += labels.size(0)
            correct += (outputs.round() == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")
    return {"fedminmax_risks": sum_risks}


def test(net, testloader, sensitive_labels=[], sens_att = "SEX", comp_att = "MAR"):
    """
    Evaluate the network on the inputted test set and determine the equalised odds for each protected group.
    
    Inputs:
        net - the instance of the model
        testloader - a pytorch DataLoader object.
        sensitive_labels - a list of the class indexes associated with the protected groups in question.

    Outputs:
        loss - average loss 
        accuracy - accuracy calculated as the number of correct classificatins out of the total
        group_performance - a list of equalised odds measurers for each protected group given.
    """

    criterion = torch.nn.BCELoss()
    correct, total, loss = 0, 0, 0.0
    group_performance = [[0,0,0,0] for label in range(len(sensitive_labels))] # preset for EOP calc, will store the performance
    group_fairness = [0 for label in range(len(sensitive_labels))] # preset for EOP calc, will store the performance
    group_comp_performance = [[0, 0, 0, 0] for label in
                         range(len(sensitive_labels))]  # preset for EOP calc, will store the performance
    group_comp_fairness = [0 for label in range(len(sensitive_labels))]  # preset for EOP calc, will store the performance

    net.eval()
    with torch.no_grad():
        for data in testloader:
            if "ESR" in data:
                label_name = "ESR"
            else:
                label_name= ">50K" # accounts for NSL-KDD
            labels = (torch.Tensor([[x] for x in data[label_name]]).double()).to(DEVICE)
            data.pop(label_name)
            inputs = torch.from_numpy(np.array([values.numpy() for key,values in data.items()], dtype=float)).to(DEVICE)
            inputs = inputs.mT # transpose required
            outputs = net(inputs)
            loss += criterion(outputs.squeeze(1), labels.squeeze(1)).item()
            predicted = outputs.round()
            # Comparing the predicted to the inputs in order to determine EOD
            matched = (predicted == labels)
            matched=torch.flatten(matched)
            sex =data[sens_att].to(DEVICE)
            l=0
            for label in sensitive_labels:

                lab = (labels == label[0])
                lab=torch.flatten(lab)
                lab1= (sex == label[1])
                labelled = (lab == lab1) * (lab1 == True)
                nlab = (labels == label[0])
                nlab=torch.flatten(nlab)
                nlab1 = (sex != label[1])
                not_labelled = (nlab == nlab1) * (nlab1 == True)
                group_performance[l][0] += int(torch.sum((matched == labelled) * (labelled ==True)))
                group_performance[l][1] += int(torch.sum((matched == not_labelled) * (not_labelled ==True)))
                group_performance[l][2] += int(lab1.sum())
                group_performance[l][3] += int(nlab1.sum())
                l = l + 1
            total += labels.size(0)
            correct += matched.sum().item()
            comp = data[comp_att].to(DEVICE)
            l = 0
            for label in sensitive_labels:
                lab = (labels == label[0])
                lab = torch.flatten(lab)
                lab1 = (comp == label[1])
                labelled = (lab == lab1) * (lab1 == True)
                nlab = (labels == label[0])
                nlab = torch.flatten(nlab)
                nlab1 = (comp != label[1])
                not_labelled = (nlab == nlab1) * (nlab1 == True)
                group_comp_performance[l][0] += int(torch.sum((matched == labelled) * (labelled == True)))
                group_comp_performance[l][1] += int(torch.sum((matched == not_labelled) * (not_labelled == True)))
                group_comp_performance[l][2] += int(lab1.sum())
                group_comp_performance[l][3] += int(nlab1.sum())
                l = l + 1
    for index in range(len(group_performance)):
        # Calculating EOD: P(Y.=1|A=1,Y=y) - P(Y.=1|A=0,Y=y) for each:
        group_fairness[index] = float((group_performance[index][0]/group_performance[index][2] - group_performance[index][1] / group_performance[index][3]))
    for index in range(len(group_comp_performance)):
        # Calculating EOD: P(Y.=1|A=1,Y=y) - P(Y.=1|A=0,Y=y) for each:
        group_comp_fairness[index] = float((group_comp_performance[index][0]/group_comp_performance[index][2] - group_comp_performance[index][1] / group_comp_performance[index][3]))
    loss /= len(testloader.dataset)
    accuracy = [correct, total]
    #print(group_performance)
    return loss, accuracy, group_fairness,group_performance, group_comp_fairness, group_comp_performance

