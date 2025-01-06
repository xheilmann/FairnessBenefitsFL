import subprocess

for i in range(5):
    cmd = ("python fedminmax_states_20c.py" )
    subprocess.run(cmd, shell=True)
for i in range(5):
    cmd = ("python fedminmax_states_20c.py --cluster 1 --num-clients 10 --sensitive-attribute SEX --comp-attribute MAR" )
    subprocess.run(cmd, shell=True)
for i in range(5):
    cmd = ("python fedminmax_states_20c.py --sensitive-attribute SEX --comp-attribute MAR")
    subprocess.run(cmd, shell=True)
for i in range(5):
    cmd = ("python fedminmax_states_20c.py --cluster 2 --num-clients 10 ")
    subprocess.run(cmd, shell=True)



for i in range(5):
    cmd = (f"python fedminmax_states_20c.py --dataset-name employment --fedminmax-lr 0.02 --fedminmax-adverse-lr 0.1" )
    subprocess.run(cmd, shell=True)
for i in range(5):
    cmd = ("python fedminmax_states_20c.py --cluster 1 --num-clients 10 --dataset-name employment --sensitive-attribute SEX --comp-attribute MAR --fedminmax-lr 0.02 --fedminmax-adverse-lr 0.1" )
    subprocess.run(cmd, shell=True)
for i in range(5):
    cmd = ("python fedminmax_states_20c.py --sensitive-attribute SEX --comp-attribute MAR --dataset-name employment --fedminmax-lr 0.02 --fedminmax-adverse-lr 0.1")
    subprocess.run(cmd, shell=True)
for i in range(5):
    cmd = ("python fedminmax_states_20c.py --cluster 2 --num-clients 10  --dataset-name employment --fedminmax-lr 0.02 --fedminmax-adverse-lr 0.1")
    subprocess.run(cmd, shell=True)


