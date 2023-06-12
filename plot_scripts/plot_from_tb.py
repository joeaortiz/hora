import matplotlib.pyplot as plt
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Path to your TensorBoard TB file
exp_root = "outputs/AllegroHandHora/06-08-23_07-22-54/"
exps = os.listdir(exp_root)
exps.sort()

events = {}
labels = [
    "No ObjPos", "No ObjScale", "No ObjMass", "No ObjCOM", "No ObjFriction"
]

for i, exp in enumerate(exps):
    tb_file_dir = f"{exp_root}/{exp}/stage1_tb/"
    fname = os.listdir(tb_file_dir)
    tb_file_path = tb_file_dir + fname[0]

    # Create an EventAccumulator to load the TB file
    event_acc = EventAccumulator(tb_file_path)
    event_acc.Reload()
    events[labels[i]] = event_acc

baseline_file = "outputs/AllegroHandHora/hora/stage1_tb/events.out.tfevents"
event_acc = EventAccumulator(tb_file_path)
event_acc.Reload()
events["All"] = event_acc

# Get all the tags/scalars available in the TB file
# tags = event_acc.Tags()["scalars"]

plot_tags = ['episode_rewards/step', 'episode_lengths/step', 'rotation_reward']
save_files = ['ep_reward.png', 'ep_len.png', 'rot_reward.png']


# Plot the scalar events for each tag
for tag, file in zip(plot_tags, save_files):

    fig = plt.figure()

    for label, event_acc in events.items():
        # Get the scalar events for the 'reward' tag
        scalar_events = event_acc.Scalars(tag)

        # Extract the steps and values from the scalar events
        steps = []
        values = []
        for scalar_event in scalar_events:
            steps.append(scalar_event.step)
            values.append(scalar_event.value)

        smoothing_parameter = 0.95
        smoothed_values = [values[0]]
        for i in range(1, len(values)):
            smoothed_value = smoothing_parameter * smoothed_values[-1] + (1 - smoothing_parameter) * values[i]
            smoothed_values.append(smoothed_value)

        # Plot the reward values
        plt.plot(steps, smoothed_values, label=label)

    plt.legend()
    plt.xlabel('Step')
    plt.ylabel(tag)
    plt.savefig(f'outputs/plots/{file}')
