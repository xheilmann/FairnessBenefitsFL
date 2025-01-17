import subprocess

for i in [0.1, 0.05, 0.2]:
    for j in [0.02, 0.08, 0.1, 0.005, 0.01]:

#income all sens MAR
#for i in range(1):
#        cmd = (f"python fedminmax_states_20c.py --fedminmax-lr 0.02 --fedminmax-adverse-lr 0 --epochs 3 --rounds 50 --batch-size 256" )
#        subprocess.run(cmd, shell=True)
#income group sens MAR
#for i in range(1):
#        cmd = (f"python fedminmax_states_20c.py --cluster 2 --num-clients 10 --fedminmax-lr 0.02 --fedminmax-adverse-lr 0 --epochs 10 --rounds 20")
#        subprocess.run(cmd, shell=True)
#income group sens SEX
#for i in range(1):
#        cmd = (f"python fedminmax_states_20c.py --cluster 1 --num-clients 10 --sensitive-attribute SEX --comp-attribute MAR --fedminmax-lr 0.002 --fedminmax-adverse-lr 0 --epochs 3 --rounds 20")
#        subprocess.run(cmd, shell=True)
#income all sens SEX
#for i in range(1):
#        cmd = (f"python fedminmax_states_20c.py --sensitive-attribute SEX --comp-attribute MAR --fedminmax-lr 0.01 --fedminmax-adverse-lr 0 --epochs 3 --rounds 20")
#        subprocess.run(cmd, shell=True)

#employment group sens MAR
#for i in range(1):
        cmd = (f"python fedminmax_states_20c.py --dataset-name employment --cluster 1 --num-clients 10 --fedminmax-lr {j} --fedminmax-adverse-lr {i} --epochs 5 --rounds 50 ")
        subprocess.run(cmd, shell=True)
#employment all sens MAR
#for i in range(1):
        cmd = (f"python fedminmax_states_20c.py --dataset-name employment --fedminmax-lr {j} --fedminmax-adverse-lr {i} --epochs 5 --rounds 50" )
        subprocess.run(cmd, shell=True)
#employment group sens SEX
#for i in range(5):
#        cmd = (f"python fedminmax_states_20c.py --cluster 2 --num-clients 10  --sensitive-attribute SEX --comp-attribute MAR --dataset-name employment --fedminmax-lr 0.2 --fedminmax-adverse-lr 0.02 --epochs 3 --rounds 30")
#        subprocess.run(cmd, shell=True)
#employment all sens SEX
#for i in range(5):
#        cmd = (f"python fedminmax_states_20c.py --dataset-name employment --sensitive-attribute SEX --comp-attribute MAR --fedminmax-lr 0.2 --fedminmax-adverse-lr 0.002 --epochs 5 --rounds 50" )
#        subprocess.run(cmd, shell=True)
  



