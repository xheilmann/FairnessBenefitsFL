"""
-------------------------------------------------------------------------------------------------------------

load_cifar.py, v1.0
by Oscar, March 2024

-------------------------------------------------------------------------------------------------------------

load data from 20 states preprocessed

-------------------------------------------------------------------------------------------------------------

Declarations, functions and classes:
- apply_transforms - manipulates the data as required to be compatible with pytorch nn model.
- load_iid - iid loading of client trainloaders (90% split) and validation loaders (10%) as well as central testset.


-------------------------------------------------------------------------------------------------------------

Usage:
Import from root directory using:
    >>> from source.load_cifar import load_niid, load_iid
Use either function to automatically generated loaders depending on number of clients and batch size:
    >>> trainloaders, valloaders, testloader, features = load_iid(NUM_CLIENTS, BATCH_SIZE)

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
from sndhdr import tests

import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import wandb
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader  # , random_split
import wandb
import os
from datasets import Dataset


def compute_disparity(
    sensitive_attributes, targets, possible_sensitive_attributes, possible_targets
):
    disparities = []
    combinations = []
    # Compute the disparity
    for target in possible_targets:
        for sensitive_attribute in possible_sensitive_attributes:
            Z_equal_z = len(
                sensitive_attributes[sensitive_attributes == sensitive_attribute]
            )
            Z_not_equal_z = len(sensitive_attributes) - Z_equal_z

            Z_equal_z_and_Y_equal_target = len(
                sensitive_attributes[
                    (sensitive_attributes == sensitive_attribute) & (targets == target)
                ]
            )
            Z_not_equal_z_and_Y_equal_target = len(
                sensitive_attributes[
                    (sensitive_attributes != sensitive_attribute) & (targets == target)
                ]
            )

            disparities.append(
                abs(
                    Z_equal_z_and_Y_equal_target / Z_equal_z
                    - Z_not_equal_z_and_Y_equal_target / Z_not_equal_z
                )
            )
            combinations.append((int(sensitive_attribute), int(target)))
    return disparities, combinations


path = "/media/heilmann/MultiObjFairFL/Improving-Fairness-via-Federated-Learning-main/FedFB/sampled_data"
#dirs = ['UT.csv', 'WI.csv', 'NH.csv', 'IN.csv', 'SD.csv', 'LA.csv','WV.csv', 'ND.csv', 'WY.csv', 'KS.csv']
dirs=['TX.csv', 'FL.csv', 'CA.csv', 'IL.csv', 'PA.csv', 'VT.csv', 'RI.csv', 'CT.csv', 'NM.csv', 'CO.csv']
print(dirs)





def load_iid(num_clients, b_size, sens_attr, comp_attr):
    """
    Load iid split

    Inputs:
        num_clients - the number of clients that require datasets
        b_size - the batch size used

    Outputs:
        trainloaders - a list of pytorch DataLoader for 90% train sets indexed by client.
        valloaders - a list of pytorch DataLoader for 10% test sets indexed by client.
        testloader - a single DataLoader for the centralised testset
        features - dataset information for displaying information.

    """
    # Download and transform CIFAR-10 (train and test)
    # Loading the central testset:
    testset =pd.read_csv("/media/heilmann/MultiObjFairFL/Improving-Fairness-via-Federated-Learning-main/FedFB/sampled_data/WY.csv")
    normalized_test= (testset - testset.min()) / (testset.max() - testset.min())
    normalized_test[">50K"]=testset[">50K"]
    # Divide data on each node: 90% train, 10% validation
    test = Dataset.from_pandas(normalized_test.loc[1:500], preserve_index=False)
    testloader= DataLoader(test, batch_size=b_size, shuffle=True)
    features = test.features

    trainloaders = []
    valloaders = []
    # Looping through each client, splitting train into train and val and turning into Pytorch DataLoader
    for file in dirs:
        print(file)
        df = pd.read_csv(path+"/"+file)
        sensitive_attributes_list = np.array(list(df[sens_attr]))
        comp_attributes_list = np.array(list(df[comp_attr]))
        targets_list = np.array(list(df[">50K"]))

        possible_sensitive_attributes = list(set(sensitive_attributes_list))
        possible_comp_attributes = list (set(comp_attributes_list))
        possible_targets = list(set(targets_list))

        disparities, combinations = compute_disparity(
            sensitive_attributes=sensitive_attributes_list,
            targets=targets_list,
            possible_sensitive_attributes=possible_sensitive_attributes,
            possible_targets=possible_targets
        )
        print(sens_attr, disparities, combinations)
        #wandb.log({f"{file}_initial_disparity_{sens_attr}": disparities})
        disparities, combinations = compute_disparity(
            sensitive_attributes=comp_attributes_list,
            targets=targets_list,
            possible_sensitive_attributes=possible_comp_attributes,
            possible_targets=possible_targets
        )
        print(comp_attr, disparities, combinations)
        #wandb.log({f"{file}_initial_disparity_{comp_attr}":disparities})

        normalized_df = (df - df.min()) / (df.max() - df.min())
        normalized_df[">50K"] = df[">50K"]
        # Divide data on each node: 90% train, 10% validation
        train, test = train_test_split(normalized_df, test_size=0.1)
        partition_train = Dataset.from_pandas(train, preserve_index=False)
        partition_test = Dataset.from_pandas(test, preserve_index=False)
        trainloaders.append(DataLoader(partition_train, batch_size=b_size, shuffle=True))
        valloaders.append(DataLoader(partition_test, batch_size=b_size))
    return trainloaders, valloaders, testloader, features


