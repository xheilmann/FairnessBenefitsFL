cd ./employment_reduced_reweight_first_10
CUDA_VISIBLE_DEVICES=4 sh run_sweep_1.sh
cd ..


cd ./employment_reduced_reweight_last_10
CUDA_VISIBLE_DEVICES=4 sh run_sweep_1.sh
cd ..

