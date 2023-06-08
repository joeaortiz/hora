# import os
# import argparse
from datetime import datetime
import submitit


exp_settings = {
    "model": {
        "frac_time_perception": [1.0],
        # "refine_poses": [0, 1],
        # "optim_embedding": [0, 1],
        # "do_active": [0, 1],
        # "n_embed_funcs": [11, 18],
        # "gauss_embed": [0],
        # "scale_input": [0.06, 0.08, 0.1],
        # "hidden_layers_block": [2],
        # "hidden_feature_size": [256, 400],
    },
}

log_folder = "slurm_out/%j"
executor = submitit.AutoExecutor(folder=log_folder)

executor.update_parameters(
    # slurm_partition="local",
    nodes=1,
    gpus_per_node=8,
    timeout_min=120,
    name="hora_ablation",
)

priv_infos = {
    "task.env.privInfo.enableObjPos": True,
    "task.env.privInfo.enableObjScale": True,
    "task.env.privInfo.enableObjMass": True,
    "task.env.privInfo.enableObjCOM": True,
    "task.env.privInfo.enableObjFriction": True
}

now = datetime.now()
time_str = now.strftime("%m-%d-%y_%H-%M-%S")

jobs = []
with executor.batch():
    for i in range(len(priv_infos.keys())):
        info = priv_infos.copy()
        info[list(info.keys())[i]] = False

        fn_string = ["scripts/train_s1.sh", f"{i}", "0", f"{time_str}/{i:02d}"]
        
        for k, v in info.items():
            fn_string.append(f"{k}={str(v)}")

        function = submitit.helpers.CommandFunction(fn_string)
        job = executor.submit(function)
        jobs.append(job)


print([job.job_id for job in jobs])  # ID of your job

outputs = [job.result()[0][-1] for job in jobs]

print(outputs)
