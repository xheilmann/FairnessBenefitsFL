import subprocess

for i in range(5):
    cmd = ("python fedminmax_states_20c.py" )
    subprocess.run(cmd, shell=True)
for i in range(5):
    cmd = ("python fedminmax_states_20c.py --cluster=2 --clients=10" )
    subprocess.run(cmd, shell=True)
for i in range(5):
    cmd = ("python fedminmax_states_20c.py --sensitive-attribute=SEX --comp-attribute=MAR")
    subprocess.run(cmd, shell=True)
for i in range(5):
    cmd = ("python fedminmax_states_20c.py --cluster=1 --clients=10--sensitive-attribute=SEX --comp-attribute=MAR")
    subprocess.run(cmd, shell=True)



for i in range(5):
    cmd = ("python fedminmax_states_20c.py --dataset-name=employment" )
    subprocess.run(cmd, shell=True)
for i in range(5):
    cmd = ("python fedminmax_states_20c.py --cluster=2 --clients=10 --dataset-name=employment" )
    subprocess.run(cmd, shell=True)
for i in range(5):
    cmd = ("python fedminmax_states_20c.py --sensitive-attribute=SEX --comp-attribute=MAR --dataset-name=employment")
    subprocess.run(cmd, shell=True)
for i in range(5):
    cmd = ("python fedminmax_states_20c.py --cluster=1 --clients=10--sensitive-attribute=SEX --comp-attribute=MAR --dataset-name=employment")
    subprocess.run(cmd, shell=True)


