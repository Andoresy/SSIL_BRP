import torch
import os
import re
import numpy as np
from tqdm import tqdm
import scipy.stats as stats
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
"""	GENERATE DATASET FOR TEST (CVS Dataset)
"""
def transform_format(instance_file):
    with open(instance_file, 'r') as file:
        lines = file.readlines()
    # Extracting number_of_stacks and number_of_blocks
    num_stacks, num_blocks = map(int, lines[0].split())
    # Initializing the result list
    result = []
    # Loop through each row and transform the data
    for i in range(1, num_stacks + 1):
        # 0~1 unifrom distribution
        block_values = list(map(lambda x: ((num_blocks+1) - int(x))/(num_blocks+1), lines[i].split()[1:]))
        row = block_values + [0] * 2
        result.append(row)
    return torch.tensor(result)


def process_files_with_regex(directory_path, file_regex):
    # Use re to find files matching the specified regex pattern
    files = [file for file in os.listdir(directory_path) if re.search(file_regex, file)]
    transform_datas = []
    for file_name in files:
        #print(file_name)
        file_path = os.path.join(directory_path, file_name)
        transformed_data = transform_format(file_path)
        transform_datas.append(transformed_data.unsqueeze(0))
    return torch.cat(transform_datas)

def data_from_caserta(H=3, W=3):
    file_regex= f"data{H}-{W}-.*"
    directory_path  = 'brp-instances-caserta-etal-2012\\CRPTestcases_Caserta' # This path could be changed in different environment
    transform_datas = process_files_with_regex(directory_path, file_regex)
    return transform_datas


""" GENERATE Random Data
    Copied and slightly modified
    https://github.com/binarycopycode/CRP_DAM
"""
def generate_data(device,n_samples=10,n_containers = 8,max_stacks=4,max_tiers=4, seed = None, plus_tiers = 2, plus_stacks = 0):

	if seed is not None:
		torch.manual_seed(seed)
		np.random.seed(seed)
	#
	dataset = torch.zeros((n_samples, max_stacks+plus_stacks, max_tiers + plus_tiers - 2), dtype=float).to(device)
	if max_stacks * max_tiers < n_containers:  # 放不下就寄
		print("max_stacks*max_tiers<n_containers")
		assert max_stacks * max_tiers >= n_containers

	for i in range(n_samples):
		per = np.arange(0, n_containers, 1)
		np.random.shuffle(per)
		per=torch.FloatTensor((per+1)/(n_containers+1.0)) #Uniform(0,1)
		#per =torch.FloatTensor(1/(per+1)) #1/N
		data=torch.reshape(per,(max_stacks,max_tiers-2)).to(device)
		data = torch.cat([torch.zeros(plus_stacks, max_tiers-2).to(device),data], dim=0)
		data = data[torch.randperm(data.size()[0])]
		add_empty= torch.zeros((max_stacks+plus_stacks,plus_tiers),dtype=float).to(device)
		#add_empty=-1 * torch.ones((max_stacks+plus_stacks,plus_tiers),dtype=float).to(device)
		dataset[i]=torch.cat( (data,add_empty) ,dim=1).to(device)

	dataset=dataset.to(torch.float32)
	return dataset

"""	GENERATE DATASET FOR TRAIN/VAL
"""
def transform_format_BC(instance_file):
    # Read the instance file
    with open(instance_file, 'r') as file:
        lines = file.readlines()

    # Extracting number_of_stacks and number_of_blocks
    num_stacks = 5
    # Initializing the result list and number_of_blocks
    result = []
    num_blocks = 0

    #Count number of blocks in Given Data
    for i in range(0, num_stacks):
        num_blocks += sum(list(map(lambda x: 1 if int(x) else 0, lines[i].split()))) # 1 if non-empty (i.e. not 0)
    
    #Convert to Uniform(0,1)
    for i in range(0, num_stacks):
        stack = list(map(lambda x: 0 if int(x)==0 else ((num_blocks+1) - int(x))/(num_blocks+1), lines[i].split()))
        result.append(stack)
    
    #Create label (optimal action)
    label = list(map(lambda x: int(x), lines[num_stacks].split()))

    return torch.tensor(result), label[0]*num_stacks + label[1]
def generate_train():
    directory_path  = 'uBRP_Imitation/uBRP_Exact_Solutions/5x5_solution'
    files = [file for file in os.listdir(directory_path) ]
    transform_datas = []
    labels = []
    for file_name in tqdm(files, desc='Creating Train file'):
        file_path = os.path.join(directory_path, file_name)
        transformed_data, label = transform_format_BC(file_path)
        transform_datas.append(transformed_data)
        labels.append(label)
    return transform_datas, labels
class Generator(Dataset):
	""" https://github.com/utkuozbulak/pytorch-custom-dataset-examples
	"""
	def __init__(self, data=None):
		self.data_pos, self.labels = data if data!=None else generate_train()
		self.n_samples=len(self.data_pos)

	def __getitem__(self, idx):
		return self.data_pos[idx], self.labels[idx]

	def __len__(self):
		return self.n_samples


if __name__ == '__main__':
    print(data_from_caserta())
    datas, labels = generate_train()