from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from argparse import ArgumentParser
from functools import partial
import pandas as pd
import os
import numpy as np
import json



def load_state_data(folder_path):
    all_data = pd.DataFrame()
    data_dict = {}
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            state_name = filename.split('.')[0]  # Assumes the file name is StateAbbreviation.csv
            file_path = os.path.join(folder_path, filename)
            state_data = pd.read_csv(file_path)
            
            # Add a 'State' column to each state's DataFrame
            state_data['State'] = state_name
            
            data_dict[state_name] = state_data
            all_data = pd.concat([all_data, state_data], ignore_index=True)
    
    return all_data, data_dict


def run_model_special(data, n_splits=5, fair=False):
    data['SEX_MAR'] = data['SEX'].astype(str) + "_" + data['MAR'].astype(str)
    
    # Prepare data partitions
    features = data.drop(columns=['State', 'SEX', 'MAR', 'RAC1P', 'ESR', 'SEX_MAR'])
    target = data['ESR']
    
    # Initialize cross-validation and classifier
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    clf = RandomForestClassifier(n_estimators=2, random_state=42)
    
    # Dictionary to collect metrics by state
    state_metrics = {state: [] for state in data['State'].unique()}
    
    for train_index, test_index in skf.split(features, target):
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]
        groups_train, groups_test = data.iloc[train_index], data.iloc[test_index]
        
        if fair:
            constraint = DemographicParity(difference_bound=0.05)
            mitigator = ExponentiatedGradient(clf, constraint)
            mitigator.fit(X_train, y_train, sensitive_features=groups_train['SEX_MAR'])
            y_pred = mitigator.predict(X_test)
        else:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

        for state in state_metrics:
            state_mask = groups_test['State'] == state
            if state_mask.any():
                # State-specific sensitive features
                sensitive_features_mar = groups_test[state_mask]['MAR']
                sensitive_features_sex = groups_test[state_mask]['SEX']

                # Calculate DPD within the state
                dpd_mar = demographic_parity_difference(y_test[state_mask], y_pred[state_mask], sensitive_features=sensitive_features_mar)
                dpd_sex = demographic_parity_difference(y_test[state_mask], y_pred[state_mask], sensitive_features=sensitive_features_sex)

                # General metrics
                mf = MetricFrame(
                    metrics={'selection_rate': selection_rate},
                    y_true=y_test[state_mask],
                    y_pred=y_pred[state_mask],
                    sensitive_features=groups_test[state_mask]['SEX_MAR']
                )

                metric_result = {
                    'Accuracy': accuracy_score(y_test[state_mask], y_pred[state_mask]),
                    'Selection Rate': mf.overall['selection_rate'],
                    'Worst Difference in Selection Rate': mf.by_group['selection_rate'].max() - mf.by_group['selection_rate'].min(),
                    'Demographic Parity MAR': dpd_mar,
                    'Demographic Parity SEX': dpd_sex
                }
                state_metrics[state].append(metric_result)

    # Print and return aggregated results
    for state, results in state_metrics.items():
        print(f"Metrics for {state}:")
        for metric in results:
            print(metric)

    return state_metrics

def save_results_to_json(state_metrics, filename='metrics_results.json'):
    # Dictionary to hold the final results including the average
    results_with_average = {}
    
    # Initialize sums to compute averages later
    sum_accuracy = 0
    sum_selection_rate = 0
    sum_worst_diff = 0
    sum_dpd_mar = 0
    sum_dpd_sex = 0
    count = 0
    
    # Iterate through each state to populate the results and calculate sums
    for state, metrics in state_metrics.items():
        state_avg = {
            'Accuracy': np.mean([m['Accuracy'] for m in metrics]),
            'Selection Rate': np.mean([m['Selection Rate'] for m in metrics]),
            'Worst Difference in Selection Rate': np.mean([m['Worst Difference in Selection Rate'] for m in metrics]),
            'Demographic Parity MAR': np.mean([m['Demographic Parity MAR'] for m in metrics]),
            'Demographic Parity SEX': np.mean([m['Demographic Parity SEX'] for m in metrics])
        }
        results_with_average[state] = state_avg
        
        # Accumulate sums
        sum_accuracy += state_avg['Accuracy']
        sum_selection_rate += state_avg['Selection Rate']
        sum_worst_diff += state_avg['Worst Difference in Selection Rate']
        sum_dpd_mar += state_avg['Demographic Parity MAR']
        sum_dpd_sex += state_avg['Demographic Parity SEX']
        count += 1
    
    # Calculate averages
    if count > 0:
        average_metrics = {
            'Average Accuracy': sum_accuracy / count,
            'Average Selection Rate': sum_selection_rate / count,
            'Average Worst Difference in Selection Rate': sum_worst_diff / count,
            'Average Demographic Parity MAR': sum_dpd_mar / count,
            'Average Demographic Parity SEX': sum_dpd_sex / count
        }
        results_with_average['average'] = average_metrics
    
    # Write results to a JSON file
    with open(filename, 'w') as f:
        json.dump(results_with_average, f, indent=4)
    
    print(f"Results saved to {filename}")

if __name__ == '__main__':
    parser = ArgumentParser(description='Run global model optimization')
    parser.add_argument('--dataset', choices=['employment', 'other'], default='employment')
    args = parser.parse_args()
    
    if args.dataset == 'employment':
        path = 'employment_data_unfair'
    else:
        path = 'sampled_csv'
    all_data, data_dict = load_state_data(path)
    mf_unfair = run_model_special(all_data, n_splits=2)
    save_results_to_json(mf_unfair, filename='unfair_metrics.json')
    mf_fair = run_model_special(all_data, n_splits=2, fair=True)
    save_results_to_json(mf_fair, filename='fair_metrics.json')


