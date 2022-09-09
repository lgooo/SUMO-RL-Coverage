import argparse
import os
import pandas as pd
import json
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='offline dataset generator')
parser.add_argument(
    '--size',
    type=int,
    default=300000,
    help='size of the dataset')
parser.add_argument(
    '--beta',
    type=float,
    default=0.8,
    help='proportion of safe samples',
)
parser.add_argument(
    '--root',
    default='./data/',
    help='root directory of data files'
)
args = parser.parse_args()

root = args.root
beta = args.beta
dataset_size = args.size

experiments = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]

# load every .txt file under {root}/{experiment_name}/data/
df_list = []
for experiment in experiments:
    data_root = os.path.join(root, experiment, 'data')
    if os.path.exists(data_root):
        for f in os.listdir(data_root):
            if f.split('.')[1] == 'txt':
                df_list.append(pd.read_csv(os.path.join(data_root, f).replace('\\', '/'), sep='\t'))

offline_dataset = pd.concat(df_list)
offline_dataset['obs'] = offline_dataset['obs'].apply(lambda x: json.loads(x))
offline_dataset['next_obs'] = offline_dataset['next_obs'].apply(lambda x: json.loads(x))
offline_dataset['safety'] = offline_dataset['safety'].apply(lambda x: json.loads(x))
offline_dataset['info'] = offline_dataset['info'].apply(lambda x: json.loads(x))


def check_safe(safety):
    for key in safety:
        if safety[key] != 0:
            return False
    return True


def check_unsafe(safety):
    return not check_safe(safety)


dangerous_set = offline_dataset.loc[offline_dataset['safety'].apply(check_unsafe)]
safe_set = offline_dataset.loc[offline_dataset['safety'].apply(check_safe)]

safe_set_len = len(safe_set.index)
dangerous_set_len = len(dangerous_set.index)
safe_rows, dangerous_rows = 0, 0

# Ensure that safe tuples occupy beta% of the dataset.
if safe_set_len >= int(dataset_size * beta) and dangerous_set_len >= int(dataset_size * (1 - beta)):
    safe_rows, dangerous_rows = int(dataset_size * beta), int(dataset_size * (1 - beta))
else:
    if (safe_set_len / dangerous_set_len) > (beta / (1 - beta)):
        safe_rows, dangerous_rows = int(beta / (1 - beta) * dangerous_set_len), dangerous_set_len
    else:
        safe_rows, dangerous_rows = safe_set_len, int(safe_set_len * (1 - beta) / beta)

print("Number of safe tuples: ", safe_rows)
print("Number of dangerous tuples: ", dangerous_rows)

final_dataset = pd.concat([dangerous_set.head(dangerous_rows), safe_set.head(safe_rows)])

if not os.path.exists('./data/dataset/'):
    os.mkdir('./data/dataset/')

with open('./data/dataset/beta{}.txt'.format(int(beta * 100)), 'w') as f:
    print('\t'.join(['ID', 'timestep', 'initial_state', 'obs', 'next_obs', 'action', 'reward', 'safety', 'terminate',
                     'done', 'info']), file=f)
    for index, row in final_dataset.iterrows():
        print('\t'.join([json.dumps(row['ID']), json.dumps(row['timestep']),
                         json.dumps(row['initial_state']),
                         json.dumps(row['obs']), json.dumps(row['next_obs']),
                         json.dumps(row['action']),
                         json.dumps(row['reward']), json.dumps(row['safety']),
                         json.dumps(row['terminate']), json.dumps(row['done']),
                         json.dumps(row['info'])]), file=f)
f.close()
