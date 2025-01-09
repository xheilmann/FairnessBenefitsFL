import subprocess

#for i in [0.15, 0.05, 0.2]:
##    for j in [0.05, 0.02, 0.2]:

#income all sens MAR
for i in range(1):
    cmd = ("python fedminmax_states_20c.py" )
    subprocess.run(cmd, shell=True)
#income group sens MAR
for i in range(1):
    cmd = ("python fedminmax_states_20c.py --cluster 2 --num-clients 10 --fedminmax-lr 0.02 --fedminmax-adverse-lr 0.01")
    subprocess.run(cmd, shell=True)
#income group sens SEX
for i in range(1):
    cmd = ("python fedminmax_states_20c.py --cluster 1 --num-clients 10 --sensitive-attribute SEX --comp-attribute MAR --fedminmax-lr 0.02 --fedminmax-adverse-lr 0.1")
    subprocess.run(cmd, shell=True)
#income all sens SEX
for i in range(1):
    cmd = ("python fedminmax_states_20c.py --sensitive-attribute SEX --comp-attribute MAR")
    subprocess.run(cmd, shell=True)

#employment group sens MAR
for i in range(1):
        cmd = (f"python fedminmax_states_20c.py --dataset-name employment --cluster 1 --num-clients 10 --fedminmax-lr 0.02 --fedminmax-adverse-lr 0.2 ")
        subprocess.run(cmd, shell=True)
#employment all sens MAR
for i in range(1):
    cmd = (f"python fedminmax_states_20c.py --dataset-name employment --fedminmax-lr 0.02 --fedminmax-adverse-lr 0.1" )
    subprocess.run(cmd, shell=True)
#employment group sens SEX
for i in range(1):
    cmd = ("python fedminmax_states_20c.py --cluster 2 --num-clients 10  --sensitive-attribute SEX --comp-attribute MAR --dataset-name employment --fedminmax-lr 0.002 --fedminmax-adverse-lr 0.001")
    subprocess.run(cmd, shell=True)
#employment all sens SEX
for i in range(1):
    cmd = (f"python fedminmax_states_20c.py --dataset-name employment --sensitive-attribute SEX --comp-attribute MAR --fedminmax-lr 0.02 --fedminmax-adverse-lr 0.2" )
    subprocess.run(cmd, shell=True)




