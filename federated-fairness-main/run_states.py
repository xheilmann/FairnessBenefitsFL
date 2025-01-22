import subprocess

#for i in [0.1]:
#    for j in [ 0.5]:

#income all sens MAR
#for i in range(5):
#        cmd = (f"python fedminmax_states_20c.py --fedminmax-lr 0.85  --fedminmax-adverse-lr 0.5 --epochs 3 --rounds 250 " )
#        subprocess.run(cmd, shell=True)
#income group sens MAR
#for i in range(5):
#        cmd = (f"python fedminmax_states_20c.py --cluster 2 --num-clients 10 --fedminmax-lr 1.4 --fedminmax-adverse-lr 0.5 --epochs 3 --rounds 200")
#        subprocess.run(cmd, shell=True)
#income group sens SEX
for i in range(1):
        cmd = (f"python fedminmax_states_20c.py --cluster 1 --num-clients 10 --sensitive-attribute SEX --comp-attribute MAR --fedminmax-lr 0.85 --fedminmax-adverse-lr 0.5 --epochs 3 --rounds 150")
        subprocess.run(cmd, shell=True)
#income all sens SEX
#for i in range(2):
#        cmd = (f"python fedminmax_states_20c.py --sensitive-attribute SEX --comp-attribute MAR --fedminmax-lr 0.85  --fedminmax-adverse-lr 0.5 --epochs 3 --rounds 350")
#        subprocess.run(cmd, shell=True)

#employment group sens MAR
#for i in range(4):
#        cmd = (f"python fedminmax_states_20c.py --dataset-name employment --cluster 1 --num-clients 10 --fedminmax-lr 0.5 --fedminmax-adverse-lr 0.1 --epochs 5 --rounds 30 ")
#        subprocess.run(cmd, shell=True)
#employment all sens MAR
#for i in range(3):
#        cmd = (f"python fedminmax_states_20c.py --dataset-name employment --fedminmax-lr 0.5 --fedminmax-adverse-lr 0.1 --epochs 5 --rounds 100" )
#        subprocess.run(cmd, shell=True)
#employment group sens SEX
#for i in range(5):
#        cmd = (f"python fedminmax_states_20c.py --cluster 2 --num-clients 10  --sensitive-attribute SEX --comp-attribute MAR --dataset-name employment --fedminmax-lr 0.5 --fedminmax-adverse-lr 0.1 --epochs 3 --rounds 30")
#        subprocess.run(cmd, shell=True)
#employment all sens SEX
#for i in range(5):
#        cmd = (f"python fedminmax_states_20c.py --dataset-name employment --sensitive-attribute SEX --comp-attribute MAR --fedminmax-lr 0.5 --fedminmax-adverse-lr 0.1 --epochs 5 --rounds 50" )
#        subprocess.run(cmd, shell=True)
  



