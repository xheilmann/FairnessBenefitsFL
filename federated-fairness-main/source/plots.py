import wandb
api = wandb.Api()
run = api.run("/xehe/multiobj-FL/runs/tvpaq9p8")



print(run.history())