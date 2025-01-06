import subprocess

for i in [0.02, 0.002, 0.008]:
    for j in [0.1, 0.01, 0.001]:
        cmd = (f"python fedminmax_states_20c.py --dataset-name employment --fedminmax-lr {i} --fedminmax-adverse-lr {j}" )
        subprocess.run(cmd, shell=True)
for i in range(1):
    cmd = ("python fedminmax_states_20c.py --cluster 1 --num-clients 10" )
    subprocess.run(cmd, shell=True)
for i in range(1):
    cmd = ("python fedminmax_states_20c.py --sensitive-attribute SEX --comp-attribute MAR")
    subprocess.run(cmd, shell=True)
for i in range(1):
    cmd = ("python fedminmax_states_20c.py --cluster 2 --num-clients 10 --sensitive-attribute SEX --comp-attribute MAR")
    subprocess.run(cmd, shell=True)



for i in range(1):
    cmd = ("python fedminmax_states_20c.py" )
    subprocess.run(cmd, shell=True)
for i in range(1):
    cmd = ("python fedminmax_states_20c.py --cluster 1 --num-clients 10 --dataset-name employment" )
    subprocess.run(cmd, shell=True)
for i in range(1):
    cmd = ("python fedminmax_states_20c.py --sensitive-attribute SEX --comp-attribute MAR --dataset-name employment")
    subprocess.run(cmd, shell=True)
for i in range(1):
    cmd = ("python fedminmax_states_20c.py --cluster 2 --num-clients 10 --sensitive-attribute SEX --comp-attribute MAR --dataset-name employment")
    subprocess.run(cmd, shell=True)


