import numpy as np
import pandas as pd
from ray import state

from utils import *
import torch

def read_data(dataset):
    if dataset == "states_income_sex":
        Z = 20
        # states income sex
        sensitive_attributes = ['SEX']
        categorical_attributes = []
        continuous_attributes = ["AGEP", "COW", "SCHL", "OCCP", "POBP", "RELP", "RAC1P", "WKHP"]
        features_to_keep = ["MAR", "SEX", '>50K', "AGEP", "COW", "SCHL", "OCCP", "POBP", "RELP", "RAC1P", "WKHP"]
        label_name = '>50K'
        #path = "/home/heilmann/Dokumente/Multiple Objective FL/Improving-Fairness-via-Federated-Learning-main/FedFB/sampled_data"
        dirs = [ 'ND.csv', 'WY.csv', 'KS.csv',
                'CT.csv', 'NM.csv', 'CO.csv']
        client_idx = []
        train = []
        test= []
        i = 0
        for file in dirs:

            states_income_sex = process_csv('sampled_data', file, label_name, 1, sensitive_attributes, None,
                                            categorical_attributes, continuous_attributes, features_to_keep,
                                            na_values=[], header=0, columns=features_to_keep)
            train.append(states_income_sex.iloc[:int(len(states_income_sex) * .7)])
            test.append(states_income_sex.iloc[int(len(states_income_sex) * .7):])
            number_points = len(states_income_sex.iloc[:int(len(states_income_sex) * .7)])
            client = pd.Index([j for j in range(i, i+number_points)])
            i += number_points
            client_idx.append(client)



        train = pd.concat(train)
        test = pd.concat(test)
        num_features = 10
        states_test = LoadData(test, '>50K', 'z')
        states_train = LoadData(train, '>50K', 'z')
        torch.manual_seed(0)
        info = [states_train, states_test, client_idx]
    elif dataset == "states_income_mar":
        Z = 20
        # states income mar
        sensitive_attributes = ['MAR']
        categorical_attributes = ["SEX"]
        continuous_attributes = ["AGEP", "COW", "SCHL", "OCCP", "POBP", "RELP", "RAC1P", "WKHP"]
        features_to_keep = ["MAR", "SEX", '>50K', "AGEP", "COW", "SCHL", "OCCP", "POBP", "RELP", "RAC1P", "WKHP"]
        label_name = '>50K'
        path = "/home/heilmann/Dokumente/Multiple Objective FL/Improving-Fairness-via-Federated-Learning-main/FedFB/sampled_data"
        dirs = ['UT.csv', 'WI.csv', 'NH.csv', 'IN.csv', 'SD.csv', 'LA.csv', 'WV.csv', 'ND.csv', 'WY.csv', 'KS.csv',
                'TX.csv', 'FL.csv', 'CA.csv', 'IL.csv', 'PA.csv', 'VT.csv', 'RI.csv', 'CT.csv', 'NM.csv', 'CO.csv']
        client_idx = []
        train = []
        i = 0
        for file in dirs:
            states_income_sex = process_csv('sampled_data', file, label_name, "1", sensitive_attributes, None,
                                            categorical_attributes, continuous_attributes, features_to_keep,
                                            na_values=[], header=None, columns=features_to_keep)
            train.append(states_income_sex)
            number_points = len(states_income_sex.index)
            client = np.array([i for i in range(i, number_points)])
            i += len(states_income_sex.index)
            client_idx.append(client)

        states_train = pd.concat(train)
        num_features = len(states_train.columns)
        train = states_train.iloc[:int(len(states_train) * .7)]
        test = states_train.iloc[int(len(states_train) * .7):]
        states_test = LoadData(test, '>50K', 'z')
        states_train = LoadData(train, '>50K', 'z')
        torch.manual_seed(0)
        info = [states_train, states_test, client_idx]
    elif dataset == "states_income_sex_mar":
        Z = 20
        # states sex and mar
        sensitive_attributes = ['SEX', 'MAR']
        categorical_attributes = []
        continuous_attributes = ["AGEP", "COW", "SCHL", "OCCP", "POBP", "RELP", "RAC1P", "WKHP"]
        features_to_keep = ["MAR", "SEX", '>50K', "AGEP", "COW", "SCHL", "OCCP", "POBP", "RELP", "RAC1P", "WKHP"]
        label_name = '>50K'
        path = "/home/heilmann/Dokumente/Multiple Objective FL/Improving-Fairness-via-Federated-Learning-main/FedFB/sampled_data"
        dirs = ['UT.csv', 'WI.csv', 'NH.csv', 'IN.csv', 'SD.csv', 'LA.csv', 'WV.csv', 'ND.csv', 'WY.csv', 'KS.csv',
                'TX.csv', 'FL.csv', 'CA.csv', 'IL.csv', 'PA.csv', 'VT.csv', 'RI.csv', 'CT.csv', 'NM.csv', 'CO.csv']
        client_idx = []
        train = []
        i = 0
        for file in dirs:
            states_income_sex = process_csv('sampled_data', file, label_name, "1", sensitive_attributes, None,
                                            categorical_attributes, continuous_attributes, features_to_keep,
                                            na_values=[], header=None, columns=features_to_keep)
            train.append(states_income_sex)
            number_points = len(states_income_sex.index)
            client = np.array([j for j in range(i, i+number_points)])
            i += len(states_income_sex.index)
            client_idx.append(client)

        states_train = pd.concat(train)
        num_features = len(states_train.columns)
        train = states_train.iloc[:int(len(states_train) * .7)]
        test = states_train.iloc[int(len(states_train) * .7):]
        states_test = LoadData(test, '>50K', 'z')
        states_train = LoadData(train, '>50K', 'z')
        torch.manual_seed(0)
        info = [states_train, states_test, client_idx]
    return Z, num_features, info





