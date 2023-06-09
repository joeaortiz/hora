import os
import csv
import matplotlib.pyplot as plt

# Function to load CSV file and return a list of dictionaries
def load_csv(file_path):
    data = []
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data

# Function to plot the data from a list of dictionaries
def plot_data(data):
    x = []
    y = []
    for row in data:
        x.append(float(row['Step']))
        y.append(float(row['Value']))
    plt.plot(x, y)

# Directory containing the CSV files
directory = 'outputs/AllegroHandHora/plots/priv_abl/'

# Iterate over CSV files in the directory
files = os.listdir(directory)
files.sort()
print(files)
for filename in files:
    if filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)
        data = load_csv(file_path)
        plot_data(data)

labels = [
    "No ObjPos", "No ObjScale", "No ObjMass",
    "No ObjCOM", "No ObjFriction", "All"
]

# Customize the plot
plt.xlabel('Env steps')
plt.ylabel('Rotation reward')
plt.title('Privileged info ablation')
plt.legend(labels)  # Add a legend based on file names
plt.show()
