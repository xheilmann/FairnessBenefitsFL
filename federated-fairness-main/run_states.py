import subprocess

for i in range(5):
    cmd = ("python fedminmax_states_20c.py" )
    subprocess.run(cmd, shell=True)