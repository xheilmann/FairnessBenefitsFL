from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from argparse import ArgumentParser
import numpy as np
import os
import json

def load_state_data(base_path):
    data_dict = {}

    # Iterate through each subfolder in the base path
    for folder_name in sorted(os.listdir(base_path), key=int):
        folder_path = os.path.join(base_path, folder_name)

        # Construct file paths for training and test data
        train_features_file = os.path.join(folder_path, f'income_dataframes_{folder_name}_train.npy')
        train_sex_file = os.path.join(folder_path, f'income_groups_{folder_name}_train.npy')
        train_mar_file = os.path.join(folder_path, f'income_second_groups_{folder_name}_train.npy')
        train_labels_file = os.path.join(folder_path, f'income_labels_{folder_name}_train.npy')
        test_features_file = os.path.join(folder_path, f'income_dataframes_{folder_name}_test.npy')
        test_sex_file = os.path.join(folder_path, f'income_groups_{folder_name}_test.npy')
        test_mar_file = os.path.join(folder_path, f'income_second_groups_{folder_name}_test.npy')
        test_labels_file = os.path.join(folder_path, f'income_labels_{folder_name}_test.npy')

        # Load training and test data from files
        train_X = np.load(train_features_file)
        train_sex = np.load(train_sex_file)
        train_mar = np.load(train_mar_file)
        train_y = np.load(train_labels_file)
        test_X = np.load(test_features_file)
        test_sex = np.load(test_sex_file)
        test_mar = np.load(test_mar_file)
        test_y = np.load(test_labels_file)

        # Store data in dictionaries
        data_dict[folder_name] = {
            'train': {
                'features': train_X,
                'sex': train_sex,
                'mar': train_mar,
                'labels': train_y
            },
            'test': {
                'features': test_X,
                'sex': test_sex,
                'mar': test_mar,
                'labels': test_y
            }
        }

    return data_dict


def run_model_global(data_dict, fair=True):
    global_metrics = {}
    state_metrics = {}

    # Concatenate data from each state
    global_features = []
    global_labels = []
    global_sex_mar = []
    global_sex = []
    global_mar = []
    
    # Additional lists to keep track of state labels for masking
    global_state_labels = []

    for folder_name, datasets in data_dict.items():
        train_data = datasets['train']
        test_data = datasets['test']

        # Concatenate training and testing data
        global_features.extend(np.vstack((train_data['features'], test_data['features'])))
        global_labels.extend(np.concatenate((train_data['labels'], test_data['labels'])))
        global_sex.extend(np.concatenate((train_data['sex'], test_data['sex'])))
        global_mar.extend(np.concatenate((train_data['mar'], test_data['mar'])))
        global_sex_mar.extend([f"{sex}_{mar}" for sex, mar in zip(np.concatenate((train_data['sex'], test_data['sex'])).astype(str), np.concatenate((train_data['mar'], test_data['mar'])).astype(str))])

        # Append state identifier to track which data belongs to which state
        global_state_labels.extend([folder_name] * (len(train_data['features']) + len(test_data['features'])))

    # Convert lists to numpy arrays
    global_features = np.array(global_features)
    global_labels = np.array(global_labels)
    global_sex_mar = np.array(global_sex_mar)
    global_state_labels = np.array(global_state_labels)

    # Initialize classifier
    clf = LogisticRegression(max_iter=1000)
    if fair:
        constraint = DemographicParity()
        mitigator = ExponentiatedGradient(clf, constraint)
        mitigator.fit(global_features, global_labels, sensitive_features=global_sex_mar)
        y_pred = mitigator.predict(global_features)
    else:
        clf.fit(global_features, global_labels)
        y_pred = clf.predict(global_features)

    # Evaluate metrics globally
    global_mf = MetricFrame(
        metrics={'selection_rate': selection_rate, 'accuracy': accuracy_score},
        y_true=global_labels,
        y_pred=y_pred,
        sensitive_features=global_sex_mar
    )

    global_metrics['Average'] = {
        'Accuracy': global_mf.overall['accuracy'],
        'Selection Rate': global_mf.overall['selection_rate'],
        'Worst Difference in Selection Rate': global_mf.by_group['selection_rate'].max() - global_mf.by_group['selection_rate'].min(),
        'Demographic Parity': global_mf.difference(method='between_groups')['selection_rate']
    }

    # Evaluate metrics per state for reporting
    for folder_name in data_dict:
        state_mask = global_state_labels == folder_name
        state_labels = global_labels[state_mask]
        state_pred = y_pred[state_mask]
        state_sex_mar = global_sex_mar[state_mask]

        state_mf = MetricFrame(
            metrics={'selection_rate': selection_rate, 'accuracy': accuracy_score},
            y_true=state_labels,
            y_pred=state_pred,
            sensitive_features=state_sex_mar
        )

        state_metrics[folder_name] = {
            'Accuracy': state_mf.overall['accuracy'],
            'Selection Rate': state_mf.overall['selection_rate'],
            'Worst Difference in Selection Rate': state_mf.by_group['selection_rate'].max() - state_mf.by_group['selection_rate'].min(),
            'Demographic Parity': state_mf.difference(method='between_groups')['selection_rate']
        }

    return state_metrics


def run_model_global_fix(data_dict, fair=True):
    global_metrics = {}
    state_metrics = {}

    # Concatenate data from each state
    global_features = []
    global_labels = []
    global_sex_mar = []
    global_sex = []
    global_mar = []
    global_state_labels = []

    for folder_name, datasets in data_dict.items():
        train_data = datasets['train']
        test_data = datasets['test']

        # Concatenate training and testing data
        global_features.extend(np.vstack((train_data['features'], test_data['features'])))
        global_labels.extend(np.concatenate((train_data['labels'], test_data['labels'])))
        global_sex.extend(np.concatenate((train_data['sex'], test_data['sex'])))
        global_mar.extend(np.concatenate((train_data['mar'], test_data['mar'])))
        global_sex_mar.extend([f"{sex}_{mar}" for sex, mar in zip(np.concatenate((train_data['sex'], test_data['sex'])).astype(str), np.concatenate((train_data['mar'], test_data['mar'])).astype(str))])
        global_state_labels.extend([folder_name] * (len(train_data['features']) + len(test_data['features'])))

    # Convert lists to numpy arrays for efficiency in indexing and manipulation
    global_features = np.array(global_features)
    global_labels = np.array(global_labels)
    global_sex_mar = np.array(global_sex_mar)
    global_sex = np.array(global_sex)
    global_mar = np.array(global_mar)
    global_state_labels = np.array(global_state_labels)

    # Initialize classifier
    clf = LogisticRegression(max_iter=1000)
    if fair:
        constraint = DemographicParity()
        mitigator = ExponentiatedGradient(clf, constraint)
        mitigator.fit(global_features, global_labels, sensitive_features=global_sex_mar)
        y_pred = mitigator.predict(global_features)
    else:
        clf.fit(global_features, global_labels)
        y_pred = clf.predict(global_features)

    # Global metrics evaluation
    global_mf = MetricFrame(
        metrics={'selection_rate': selection_rate, 'accuracy': accuracy_score},
        y_true=global_labels,
        y_pred=y_pred,
        sensitive_features=global_sex_mar
    )

    dpd_sex = demographic_parity_difference(global_labels, y_pred, sensitive_features=global_sex)
    dpd_mar = demographic_parity_difference(global_labels, y_pred, sensitive_features=global_mar)

    state_metrics['average'] = {
        'Accuracy': global_mf.overall['accuracy'],
        'Selection Rate': global_mf.overall['selection_rate'],
        'Worst Difference in Selection Rate': global_mf.by_group['selection_rate'].max() - global_mf.by_group['selection_rate'].min(),
        'Demographic Parity SEX': dpd_sex,
        'Demographic Parity MAR': dpd_mar
    }

    # Evaluate metrics per state
    for folder_name in data_dict:
        state_mask = global_state_labels == folder_name
        state_labels = global_labels[state_mask]
        state_pred = y_pred[state_mask]
        state_sex = global_sex[state_mask]
        state_mar = global_mar[state_mask]
        state_sex_mar = global_sex_mar[state_mask]

        state_mf = MetricFrame(
            metrics={'selection_rate': selection_rate, 'accuracy': accuracy_score},
            y_true=state_labels,
            y_pred=state_pred,
            sensitive_features=state_sex_mar
        )

        # Calculate demographic parity differences for SEX and MAR separately
        dpd_sex = demographic_parity_difference(state_labels, state_pred, sensitive_features=state_sex)
        dpd_mar = demographic_parity_difference(state_labels, state_pred, sensitive_features=state_mar)

        state_metrics[folder_name] = {
            'Accuracy': state_mf.overall['accuracy'],
            'Selection Rate': state_mf.overall['selection_rate'],
            'Worst Difference in Selection Rate': state_mf.by_group['selection_rate'].max() - state_mf.by_group['selection_rate'].min(),
            'Demographic Parity SEX': dpd_sex,
            'Demographic Parity MAR': dpd_mar
        }

    return state_metrics



def run_model_local_intersectional(data_dict, fair=False):
    state_metrics = {}

    for folder_name, datasets in data_dict.items():
        train_data = datasets['train']
        test_data = datasets['test']

        # Create intersectional attribute for training and testing
        SEX_MAR_train = np.array([f"{sex}_{mar}" for sex, mar in zip(train_data['sex'].astype(str), train_data['mar'].astype(str))])
        SEX_MAR_test = np.array([f"{sex}_{mar}" for sex, mar in zip(test_data['sex'].astype(str), test_data['mar'].astype(str))])

        # Initialize classifier
        clf = LogisticRegression(max_iter=1000)
        if fair:
            constraint = DemographicParity(difference_bound=0.05)
            mitigator = ExponentiatedGradient(clf, constraint)
            mitigator.fit(train_data['features'], train_data['labels'], sensitive_features=SEX_MAR_train)
            y_pred = mitigator.predict(test_data['features'])
        else:
            clf.fit(train_data['features'], train_data['labels'])
            y_pred = clf.predict(test_data['features'])

        # Evaluate metrics on the test set using MetricFrame as per your specific setup
        mf = MetricFrame(
            metrics={'selection_rate': selection_rate},
            y_true=test_data['labels'],
            y_pred=y_pred,
            sensitive_features=SEX_MAR_test
        )

        # Calculate demographic parity differences
        dpd_mar = demographic_parity_difference(test_data['labels'], y_pred, sensitive_features=test_data['mar'])
        dpd_sex = demographic_parity_difference(test_data['labels'], y_pred, sensitive_features=test_data['sex'])

        metric_result = {
            'Accuracy': accuracy_score(test_data['labels'], y_pred),
            'Selection Rate': mf.overall['selection_rate'],
            'Worst Difference in Selection Rate': mf.by_group['selection_rate'].max() - mf.by_group['selection_rate'].min(),
            'Demographic Parity MAR': dpd_mar,
            'Demographic Parity SEX': dpd_sex
        }
        # Append the result to state-specific metrics
        state_metrics[folder_name] = metric_result

    return state_metrics

def run_model_local(data_dict, unfair_sex, unfair_mar, fair=False):
    state_metrics = {}

    for folder_name, datasets in data_dict.items():
        train_data = datasets['train']
        test_data = datasets['test']

        # Create intersectional attribute for training and testing
        SEX_MAR_train = np.array([f"{sex}_{mar}" for sex, mar in zip(train_data['sex'].astype(str), train_data['mar'].astype(str))])
        SEX_MAR_test = np.array([f"{sex}_{mar}" for sex, mar in zip(test_data['sex'].astype(str), test_data['mar'].astype(str))])

        # Initialize classifier
        clf = LogisticRegression(max_iter=1000)
        
        if int(folder_name) in unfair_sex:
            sensitive_features_train = train_data['sex']
            sensitive_features_test = test_data['sex']
        elif int(folder_name) in unfair_mar:
            sensitive_features_train = train_data['mar']
            sensitive_features_test = test_data['mar']
        else:
            # If the folder is not specified as unfair wrt any attribute, proceed without fairness intervention
            sensitive_features_train = None
            sensitive_features_test = None

        if sensitive_features_train is not None and fair:
            # Apply ExponentiatedGradient with DemographicParity for specified sensitive feature
            constraint = DemographicParity()
            mitigator = ExponentiatedGradient(clf, constraint)
            mitigator.fit(train_data['features'], train_data['labels'], sensitive_features=sensitive_features_train)
            y_pred = mitigator.predict(test_data['features'])
        else:
            clf.fit(train_data['features'], train_data['labels'])
            y_pred = clf.predict(test_data['features'])

        # Evaluate metrics on the test set
        mf = MetricFrame(
            metrics={'selection_rate': selection_rate},
            y_true=test_data['labels'],
            y_pred=y_pred,
            sensitive_features=SEX_MAR_test
        )

        # Calculate demographic parity differences
        dpd_mar = demographic_parity_difference(test_data['labels'], y_pred, sensitive_features=test_data['mar'])
        dpd_sex = demographic_parity_difference(test_data['labels'], y_pred, sensitive_features=test_data['sex'])

        metric_result = {
            'Accuracy': accuracy_score(test_data['labels'], y_pred),
            'Selection Rate': mf.overall['selection_rate'],
            'Worst Difference in Selection Rate': mf.by_group['selection_rate'].max() - mf.by_group['selection_rate'].min(),
            'Demographic Parity MAR': dpd_mar,
            'Demographic Parity SEX': dpd_sex
        }
        state_metrics[folder_name] = metric_result

    return state_metrics


def save_results_to_json(state_metrics, filename='metrics_results.json', do_avg=True):
    # Initialize sums to compute averages later
    sum_accuracy = 0
    sum_selection_rate = 0
    sum_worst_diff = 0
    sum_dpd_mar = 0
    sum_dpd_sex = 0
    count = 0

    if do_avg:
        # Iterate through each state to populate the results and calculate sums
        for state, metrics in state_metrics.items():
            # Accumulate sums
            sum_accuracy += metrics['Accuracy']
            sum_selection_rate += metrics['Selection Rate']
            sum_worst_diff += metrics['Worst Difference in Selection Rate']
            sum_dpd_mar += metrics['Demographic Parity MAR']
            sum_dpd_sex += metrics['Demographic Parity SEX']
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
            state_metrics['average'] = average_metrics
    
    # Write results to a JSON file
    with open(filename, 'w') as f:
        json.dump(state_metrics, f, indent=4)
    
    print(f"Results saved to {filename}")

if __name__ == '__main__':
    parser = ArgumentParser(description='Run global model optimization')
    parser.add_argument('--dataset', choices=['employment', 'income'], default='employment')
    parser.add_argument('--which', choices=['global', 'local', 'local_intersectional'], default='local_intersectional', help='Run the local model evaluation rather than global')
    args = parser.parse_args()
    
    if args.dataset == 'employment':
        path = 'FL_employment_data_train_test_NO_RACE/federated/'
    else:
        path = 'FL_income_data_train_test_NO_RACE/federated/'
    
    data_dict = load_state_data(path)
    if args.which == 'local_intersectional':
        state_metrics = run_model_local_intersectional(data_dict, fair=False)
        save_results_to_json(state_metrics, filename='unfair_metrics_{}_{}.json'.format(args.dataset, args.which))
        state_metrics = run_model_local_intersectional(data_dict, fair=True)
        save_results_to_json(state_metrics, filename='fair_metrics_{}_{}.json'.format(args.dataset, args.which))
    elif args.which == 'global':
        state_metrics = run_model_global_fix(data_dict, fair=False)
        save_results_to_json(state_metrics, filename='unfair_metrics_{}_{}.json'.format(args.dataset, args.which), do_avg=False)
        state_metrics = run_model_global_fix(data_dict, fair=True)
        save_results_to_json(state_metrics, filename='fair_metrics_{}_{}.json'.format(args.dataset, args.which), do_avg=False)
    elif args.which == 'local':
        unfair_sex = [15, 17, 9, 5, 13, 7, 18, 8, 19, 6]
        unfair_mar = [14, 3, 0, 4, 11, 16, 12, 2, 10, 1]
        state_metrics = run_model_local(data_dict, unfair_sex, unfair_mar, fair=False)
        save_results_to_json(state_metrics, filename='unfair_metrics_{}_{}.json'.format(args.dataset, args.which))
        state_metrics = run_model_local(data_dict, unfair_sex, unfair_mar, fair=True)
        save_results_to_json(state_metrics, filename='fair_metrics_{}_{}.json'.format(args.dataset, args.which))
    else:
        print('comic sans') 
