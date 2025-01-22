
# Income 

# Baseline
for i in $(seq 0 4);
do
    poetry run python /home/XXXXXX/multi_fairness/PUFFLE/experiments/income_reduced_NO_RACE/../../puffle/main.py --batch_size=604 --run_name inc_Baseline --project_name Multi_Results_paper_new --node_shuffle_seed $i --epochs=5 --lr=0.09006815489620071 --optimizer=adam --dataset income_NO_RACE --num_rounds 10 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 20 --sampled_clients 1 --sampled_clients_test 1 --sampled_clients_validation 0 --debug False --base_path ../../../FL_income_data_train_test_NO_RACE/ --dataset_path ../../../FL_income_data_train_test_NO_RACE/ --seed 41 --wandb True  --training_nodes 1 --validation_nodes 0 --test_nodes 1 --tabular_data True --update_lambda False --metric disparity --splitted_data_dir federated --ratio_unfair_nodes 0.5 --cross_silo True
done 

# 005
for i in $(seq 0 4);
do
    poetry run python /home/XXXXXX/multi_fairness/PUFFLE/experiments/income_reduced_NO_RACE/../../puffle/main.py --run_name inc_005 --project_name Multi_Results_paper_new --node_shuffle_seed $i --batch_size=1115 --epochs=2 --lr=0.04080554931586709 --optimizer=adam --regularization_lambda=0.6142028086670507 --dataset income_NO_RACE --num_rounds 10 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 20 --sampled_clients 1 --sampled_clients_test 1 --sampled_clients_validation 0 --debug False --base_path ../../../FL_income_data_train_test_NO_RACE/ --dataset_path ../../../FL_income_data_train_test_NO_RACE/ --seed 41 --wandb True  --training_nodes 1 --validation_nodes 0 --test_nodes 1 --tabular_data True --update_lambda False --metric disparity --splitted_data_dir federated --ratio_unfair_nodes 0.5 --update_lambda False --regularization_mode fixed --regularization True --target 0.05 --cross_silo True
done 

# 010
for i in $(seq 0 4);
do
    poetry run python /home/XXXXXX/multi_fairness/PUFFLE/experiments/income_reduced_NO_RACE/../../puffle/main.py --run_name inc_010 --project_name Multi_Results_paper_new --node_shuffle_seed $i --batch_size=1286 --epochs=3 --lr=0.08020724393222783 --optimizer=adam --regularization_lambda=0.20085987258247073 --dataset income_NO_RACE --num_rounds 10 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 20 --sampled_clients 1 --sampled_clients_test 1 --sampled_clients_validation 0 --debug False --base_path ../../../FL_income_data_train_test_NO_RACE/ --dataset_path ../../../FL_income_data_train_test_NO_RACE/ --seed 41 --wandb True  --training_nodes 1 --validation_nodes 0 --test_nodes 1 --tabular_data True --update_lambda False --metric disparity --splitted_data_dir federated --ratio_unfair_nodes 0.5 --update_lambda False --regularization_mode fixed --regularization True --target 0.1 --cross_silo True
done 

# # baseline_last
# for i in $(seq 0 4);
# do
#     poetry run python /home/XXXXXX/multi_fairness/PUFFLE/experiments/income_reduced_last_10_NO_RACE/../../puffle/main.py --run_name inc_Baseline_last --project_name Multi_Results_paper --node_shuffle_seed $i  --batch_size=1194 --epochs=3 --lr=0.08349094976667246 --optimizer=adam --dataset income_NO_RACE --num_rounds 10 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 10 --sampled_clients 1 --sampled_clients_test 1 --sampled_clients_validation 0 --debug False --base_path ../../../FL_income_data_train_test_last_10_NO_RACE/ --dataset_path ../../../FL_income_data_train_test_last_10_NO_RACE/ --seed 41 --wandb True  --training_nodes 1 --validation_nodes 0 --test_nodes 1 --tabular_data True --update_lambda False --metric disparity --splitted_data_dir federated --ratio_unfair_nodes 0.5 --cross_silo True --sensitive_attribute MAR
# done 

# # 005_last
# for i in $(seq 0 4);
# do
#     poetry run python /home/XXXXXX/multi_fairness/PUFFLE/experiments/income_reduced_last_10_NO_RACE/../../puffle/main.py --run_name inc_005_last --project_name Multi_Results_paper --node_shuffle_seed $i --batch_size=698 --epochs=4 --lr=0.058909714502664835 --optimizer=adam --regularization_lambda=0.39469550843185625 --dataset income_NO_RACE --num_rounds 10 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 10 --sampled_clients 1 --sampled_clients_test 1 --sampled_clients_validation 0 --debug False --base_path ../../../FL_income_data_train_test_last_10_NO_RACE/ --dataset_path ../../../FL_income_data_train_test_last_10_NO_RACE/ --seed 41 --wandb True  --training_nodes 1 --validation_nodes 0 --test_nodes 1 --tabular_data True --update_lambda False --metric disparity --splitted_data_dir federated --ratio_unfair_nodes 0.5 --update_lambda False --regularization_mode fixed --regularization True --target 0.05 --cross_silo True --sensitive_attribute MAR
# done 

# # 010_last
# for i in $(seq 0 4);
# do
#     poetry run python /home/XXXXXX/multi_fairness/PUFFLE/experiments/income_reduced_last_10_NO_RACE/../../puffle/main.py --run_name inc_010_last --project_name Multi_Results_paper --node_shuffle_seed $i --batch_size=1982 --epochs=3 --lr=0.04908832373730579 --optimizer=adam --regularization_lambda=0.1892716184914648 --dataset income_NO_RACE --num_rounds 10 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 10 --sampled_clients 1 --sampled_clients_test 1 --sampled_clients_validation 0 --debug False --base_path ../../../FL_income_data_train_test_last_10_NO_RACE/ --dataset_path ../../../FL_income_data_train_test_last_10_NO_RACE/ --seed 41 --wandb True  --training_nodes 1 --validation_nodes 0 --test_nodes 1 --tabular_data True --update_lambda False --metric disparity --splitted_data_dir federated --ratio_unfair_nodes 0.5 --update_lambda False --regularization_mode fixed --regularization True --target 0.1 --cross_silo True --sensitive_attribute MAR
# done 

# # baseline_inverse
# for i in $(seq 0 4);
# do
#     poetry run python /home/XXXXXX/multi_fairness/PUFFLE/experiments/income_reduced_inverse_NO_RACE/../../puffle/main.py --run_name inc_Baseline_inverse --project_name Multi_Results_paper --node_shuffle_seed $i --batch_size=1307 --epochs=3 --lr=0.08431260665383802 --optimizer=adam --dataset income_NO_RACE --num_rounds 10 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 20 --sampled_clients 1 --sampled_clients_test 1 --sampled_clients_validation 0 --debug False --base_path ../../../FL_income_data_train_test_NO_RACE/ --dataset_path ../../../FL_income_data_train_test_NO_RACE/ --seed 41 --wandb True  --training_nodes 1 --validation_nodes 0 --test_nodes 1 --tabular_data True --update_lambda False --metric disparity --splitted_data_dir federated --ratio_unfair_nodes 0.5 --cross_silo True --sensitive_attribute MAR
# done 

# # 005_inverse
# for i in $(seq 0 4);
# do
#     poetry run python /home/XXXXXX/multi_fairness/PUFFLE/experiments/income_reduced_inverse_NO_RACE/../../puffle/main.py --run_name inc_005_inverse --project_name Multi_Results_paper --node_shuffle_seed $i --batch_size=1535 --epochs=2 --lr=0.09176155927568644 --optimizer=sgd --regularization_lambda=0.7507232591303207 --dataset income_NO_RACE --num_rounds 10 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 20 --sampled_clients 1 --sampled_clients_test 1 --sampled_clients_validation 0 --debug False --base_path ../../../FL_income_data_train_test_NO_RACE/ --dataset_path ../../../FL_income_data_train_test_NO_RACE/ --seed 41 --wandb True  --training_nodes 1 --validation_nodes 0 --test_nodes 1 --tabular_data True --update_lambda False --metric disparity --splitted_data_dir federated --ratio_unfair_nodes 0.5 --update_lambda False --regularization_mode fixed --regularization True --target 0.05 --cross_silo True --sensitive_attribute MAR
# done 

# # 010_inverse
# for i in $(seq 0 4);
# do
#     poetry run python /home/XXXXXX/multi_fairness/PUFFLE/experiments/income_reduced_inverse_NO_RACE/../../puffle/main.py --run_name inc_010_inverse --project_name Multi_Results_paper --node_shuffle_seed $i  --batch_size=1940 --epochs=3 --lr=0.0733069188663143 --optimizer=sgd --regularization_lambda=0.3441656052294894 --dataset income_NO_RACE --num_rounds 10 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 20 --sampled_clients 1 --sampled_clients_test 1 --sampled_clients_validation 0 --debug False --base_path ../../../FL_income_data_train_test_NO_RACE/ --dataset_path ../../../FL_income_data_train_test_NO_RACE/ --seed 41 --wandb True  --training_nodes 1 --validation_nodes 0 --test_nodes 1 --tabular_data True --update_lambda False --metric disparity --splitted_data_dir federated --ratio_unfair_nodes 0.5 --update_lambda False --regularization_mode fixed --regularization True --target 0.1 --cross_silo True --sensitive_attribute MAR
# done 

# # baseline_first
# for i in $(seq 0 4);
# do
#     poetry run python /home/XXXXXX/multi_fairness/PUFFLE/experiments/income_reduced_first_10_NO_RACE/../../puffle/main.py --run_name inc_baseline_first --project_name Multi_Results_paper --node_shuffle_seed $i  --batch_size=1186 --epochs=5 --lr=0.049699642989754794 --optimizer=adam --dataset income_NO_RACE --num_rounds 10 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 10 --sampled_clients 1 --sampled_clients_test 1 --sampled_clients_validation 0 --debug False --base_path ../../../FL_income_data_train_test_first_10_NO_RACE/ --dataset_path ../../../FL_income_data_train_test_first_10_NO_RACE/ --seed 41 --wandb True  --training_nodes 1 --validation_nodes 0 --test_nodes 1 --tabular_data True --update_lambda False --metric disparity --splitted_data_dir federated --ratio_unfair_nodes 0.5 --cross_silo True --sensitive_attribute SEX
# done 

# # 005_first
# for i in $(seq 0 4);
# do
#     poetry run python /home/XXXXXX/multi_fairness/PUFFLE/experiments/income_reduced_first_10_NO_RACE/../../puffle/main.py --run_name inc_005_first --project_name Multi_Results_paper --node_shuffle_seed $i  --batch_size=531 --epochs=5 --lr=0.07361801513898486 --optimizer=adam --regularization_lambda=0.31137432481787214 --dataset income_NO_RACE --num_rounds 10 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 10 --sampled_clients 1 --sampled_clients_test 1 --sampled_clients_validation 0 --debug False --base_path ../../../FL_income_data_train_test_first_10_NO_RACE/ --dataset_path ../../../FL_income_data_train_test_first_10_NO_RACE/ --seed 41 --wandb True  --training_nodes 1 --validation_nodes 0 --test_nodes 1 --tabular_data True --update_lambda False --metric disparity --splitted_data_dir federated --ratio_unfair_nodes 0.5 --update_lambda False --regularization_mode fixed --regularization True --target 0.05 --cross_silo True --sensitive_attribute SEX
# done 

# # 010_first
# for i in $(seq 0 4);
# do
#     poetry run python /home/XXXXXX/multi_fairness/PUFFLE/experiments/income_reduced_first_10_NO_RACE/../../puffle/main.py --run_name inc_010_first --project_name Multi_Results_paper --node_shuffle_seed $i --batch_size=1988 --epochs=5 --lr=0.09199149349857597 --optimizer=adam --regularization_lambda=0.22101836913474815 --dataset income_NO_RACE --num_rounds 10 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 10 --sampled_clients 1 --sampled_clients_test 1 --sampled_clients_validation 0 --debug False --base_path ../../../FL_income_data_train_test_first_10_NO_RACE/ --dataset_path ../../../FL_income_data_train_test_first_10_NO_RACE/ --seed 41 --wandb True  --training_nodes 1 --validation_nodes 0 --test_nodes 1 --tabular_data True --update_lambda False --metric disparity --splitted_data_dir federated --ratio_unfair_nodes 0.5 --update_lambda False --regularization_mode fixed --regularization True --target 0.1 --cross_silo True --sensitive_attribute SEX
# done 