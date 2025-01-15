import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

samplesize = 5000
batchsize = 4
base_layer = 16
patiencee= 50
factore = 0.75
num_epochs = 100000
alpha = 100
ILR=0.001
file_dataset='mathematica/filteredDatasly1713.csv'
attributes='00_00_00_005_005_005_data1713_FULL_drop_penalty_squared_10g3_N_xxx'

###### load before directory change
def load_dataset(filename):
    df = pd.read_csv(filename)
    return df

loaded_dataset = load_dataset(file_dataset)
###### END load before directory change

directory_name = f"output_sz{samplesize}_b{batchsize}_bl{base_layer}_p{patiencee}_a{alpha}_{attributes}"
current_directory = Path.cwd()
output_directory = current_directory / directory_name
output_directory.mkdir(parents=True, exist_ok=True)
os.chdir(output_directory)

def save_model_with_logging(model, dir_name, baselayer, samplesize, batchsize, patiencee, alpha, log_filename='output.txt'):
    # Construct the base filename using the provided parameters
    base_filename = f'_ISITTHEMODEL_bl{baselayer}_{samplesize}_{batchsize}_p{patiencee}_a{alpha}_{attributes}'
    PATH = os.path.join(dir_name, base_filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g., 20240818_143210
    new_filename = f"{base_filename}_{timestamp}"
    PATH = os.path.join(dir_name, new_filename)  # Update the path with the new filename
    torch.save(model, PATH)

    # Log that the model was saved successfully
    with open(log_filename, 'a') as f:
        print(f"Model saved successfully at: {PATH}\n###################\n\n", file=f)

with open('output.txt', 'w') as f:
    print("Using device:", device, file=f)
with open('output.txt', 'a') as f:
    print(f"Dataset loaded from {file_dataset}", file=f) 

def log_gpu_info(filename='output.txt'):
    if torch.cuda.is_available():
        gpu_id = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_id)
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024 ** 3)  # Convert bytes to GB
        with open(filename, 'a') as f:
            print(f"GPU ID: {gpu_id}", file=f)
            print(f"GPU Name: {gpu_name}", file=f)
            print(f"Total Memory: {total_memory:.2f} GB", file=f)
    else:
        with open(filename, 'a') as f:
            print("No GPU available.", file=f)

log_gpu_info()

datasetDF=pd.DataFrame(loaded_dataset)
#datasetDF = datasetDF[(datasetDF["raggio"] < 20)]

def df_column_switch(df, column1, column2):
    temp = df[column1].copy()
    df[column1] = df[column2]
    df[column2] = temp
    i = list(df.columns)
    a, b = i.index(column1), i.index(column2)
    i[b], i[a] = i[a], i[b]
    df.columns = i
    
    return df

def log(x):
    return np.log(x)

def cgspressure(x):
    c = 2.9979e10
    G = 6.67408e-8
    Msun = 1.989e33
    return x*c**8/(G**3*Msun*Msun)

datasetDF['Λns'] = pd.to_numeric(datasetDF['Λns'], errors='coerce')
datasetDF['Λns'] = datasetDF['Λns'].apply(log)
datasetDF['ρ0'] = datasetDF['ρ0'].apply(log)
# datasetDF['p1'] = datasetDF['p1'].apply(cgspressure)
datasetDF['p1'] = datasetDF['p1'].apply(log)

# swapping to columns. I prefer it this way
datasetDF = datasetDF.dropna(subset=['Λns'])
datasetDF = df_column_switch(datasetDF, "ρ0","raggio") 
datasetDF = datasetDF[datasetDF["raggio"] < 17]
datasetDF = datasetDF[datasetDF["massa"] > 0.2]


class Encoder(nn.Module):
    def __init__(self,n,input_size):
        super(Encoder, self).__init__()
        
        self.inp = nn.Linear(input_size, 5*n)
        self.fc1 = nn.Linear(5*n, 4*n)
        self.fc2 = nn.Linear(4*n, 3*n)
        self.fc3 = nn.Linear (3*n, 2*n)
        self.fc4 = nn.Linear (2*n, 1*n)
        self.fc_latent = nn.Linear(1*n, 4)  # Latent space with dimensions for [Mass, Radius, Tidal, Compactness]
        self.relu = nn.ReLU()
        self.silu = nn.SiLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        dv1 = 0.
        dv2 = 0.
        dv3 = 0.
        x = self.inp(x)
        x = self.silu(x)
        x = self.fc1(x)
        x = self.silu(x)
        x = nn.functional.dropout(x, dv1, training=self.training)
        x = self.fc2(x)
        x = self.silu(x)
        x = nn.functional.dropout(x, dv2, training=self.training)
        x = self.fc3(x)
        x = self.silu(x)
        x = nn.functional.dropout(x, dv3, training=self.training)
        x = self.fc4(x)
        x = self.silu(x)
        latent = self.fc_latent(x)
        return latent

class Decoder(nn.Module):
    def __init__(self,n,input_size):
        super(Decoder, self).__init__()
        self.fc_latent = nn.Linear(4, 1*n)
        self.fc1 = nn.Linear(1*n, 2*n)
        self.fc2 = nn.Linear(2*n, 3*n)
        self.fc3 = nn.Linear(3*n, 4*n)
        self.fc4 = nn.Linear(4*n, 5*n)
        self.out = nn.Linear(5*n, input_size)
        self.silu = nn.SiLU()
        self.tanh = nn.Tanh()


    def forward(self, z):
        dv1 = 0.05
        dv2 = 0.05
        dv3 = 0.05
        z = self.silu(z)
        z = self.fc_latent(z)
        z = self.silu(z)
        z = self.fc1(z)
        z = self.silu(z)
        z = nn.functional.dropout(z, dv3, training=self.training)
        z = self.fc2(z)
        z = self.silu(z)
        z = nn.functional.dropout(z, dv2, training=self.training)
        z = self.fc3(z)
        z = self.silu(z)
        z = nn.functional.dropout(z, dv1, training=self.training)
        z = self.fc4(z)
        z = self.silu(z)
        reconstructed = self.out(z)
        return reconstructed

class Autoencoder(nn.Module):
    def __init__(self,n,input_size):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(n,input_size)
        self.decoder = Decoder(n,input_size)

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed
    
    
from torch.utils.data import random_split

# Define a safe division by std, replacing zero std with 1 to avoid division by zero
def safe_normalize(data, mean, std):
    std = torch.where(std <1e-8, torch.ones_like(std), std)  # Replace zeros in std with ones
    return (data - mean) / std

def safe_normalize(data, mean, std):
    std_zero_mask = std < 1e-8
    # Replace zeros in std with ones (to avoid division by zero)
    std_safe = torch.where(std_zero_mask, torch.ones_like(std), std)
    # Normalize the data
    normalized_data = (data - mean) / std_safe
    # Set normalized data to 1 where std was effectively zero
    normalized_data = torch.where(std_zero_mask, torch.ones_like(data), normalized_data)
    return normalized_data
def denormalize(tensor, mean, std): 
    return tensor * std + mean




# [p1, gamma1, gamma2, gamma3, ρ0, massa, raggio, Λns]
sampledf=datasetDF.sample(n=samplesize)		#, random_state=1)
dataset = torch.tensor(sampledf.values).double()  
fdata = torch.tensor(datasetDF.values).double() 

input_full = fdata[:, :5]
target_full = fdata[:, 5:]

input_features = dataset[:, :5]
target_features = dataset[:, 5:]



# Normalizzazione degli input
input_mean = input_full.mean(0, keepdim=True)
input_std = input_full.std(0, keepdim=True)
normalized_inputs = safe_normalize(input_features, input_mean, input_std).to(device)

# Normalizzazione degli output latenti
target_mean = target_full.mean(0, keepdim=True)
target_std = target_full.std(0, keepdim=True)
normalized_targets = safe_normalize(target_features, target_mean, target_std).to(device)

# Riconfigura il DataLoader con dati normalizzati
train_dataset = TensorDataset(normalized_inputs, normalized_targets)


valid_size = int(0.2 * len(train_dataset))  # 20% del dataset per la validazione
train_size = len(train_dataset) - valid_size

train_subset, valid_subset = random_split(train_dataset, [train_size, valid_size])

# Crea DataLoader per i subset di training e validazione
train_loader = DataLoader(train_subset, batch_size=batchsize, shuffle=True)
test_loader = DataLoader(valid_subset, batch_size=batchsize, shuffle=True)


############################### Penalty

feature_mins = normalized_inputs.min(dim=0)[0]  # Minimum for each feature
feature_maxs = normalized_inputs.max(dim=0)[0] # Maximum for each feature

def boundary_loss(output, feature_mins, feature_maxs, penalty_weight=1.0):
    # Penalize values outside the feature-wise bounds
    lower_penalty = torch.relu(feature_mins - output)*torch.relu(feature_mins - output)  # Penalty for going below min
    upper_penalty = torch.relu(output - feature_maxs)*torch.relu(output - feature_maxs)  # Penalty for exceeding max
    
    # Sum the penalties for all samples and features
    penalty = penalty_weight * (lower_penalty + upper_penalty).sum()
    
    return penalty
############################### Penalty END


class PhysicsInformedLoss(nn.Module):
    def __init__(self, alpha):
        super(PhysicsInformedLoss, self).__init__()
        self.alpha = alpha

    def forward(self, output, data, latent, target_latent,feature_mins, feature_maxs):
        # Basic MSE Loss for reconstruction
        # mse_loss = F.mse_loss(output, data)
        loss_feature_p1 = F.mse_loss(output[:, 0], data[:, 0])
        loss_feature_1 = F.mse_loss(output[:, 1], data[:, 1])
        loss_feature_2 = F.mse_loss(output[:, 2], data[:, 2])
        loss_feature_3 = F.mse_loss(output[:, 3], data[:, 3])
        loss_feature_rho = F.mse_loss(output[:, 4], data[:, 4])

        # Combine the individual losses into a single value if needed
        mse_loss = (loss_feature_p1 + loss_feature_1 + loss_feature_2 + 10*loss_feature_3 + loss_feature_rho) / 14


        boundary_penalty = boundary_loss(output, feature_mins, feature_maxs)

        internal_loss = F.mse_loss(denormalize(latent[:,0],target_mean[0][0],target_std[0][0])*1.47/denormalize(latent[:,1],target_mean[0][1],target_std[0][1]), 
                                  denormalize(target_latent[:,3],target_mean[0][3],target_std[0][3]))
        
        x = denormalize(latent[:,3],target_mean[0][3],target_std[0][3])
        internal_loss2a = F.mse_loss((3.388812515157282 - 29.777405993083043*(x - 0.25) + 30.337557708081047*(x - 0.25) ** 2 - 61.81649320998202*(x - 0.25) ** 3 + 157.44835450951092*(x - 0.25) ** 4 - 449.14845563901287*(x - 0.25) ** 5 + 1372.7925658409486*(x - 0.25) ** 6 - 4395.652103938753*(x - 0.25) ** 7 + 14554.605110543052*(x - 0.25) ** 8 - 49427.98717501882*(x - 0.25) ** 9 + 171216.49234524113*(x - 0.25) ** 10),
                                     denormalize(latent[:,2],target_mean[0][2],target_std[0][2]))
        
        physical_loss = F.mse_loss(latent, target_latent)
        # Combine the losses
        total_loss = mse_loss + alpha*physical_loss  + internal_loss + boundary_penalty + 0.1*internal_loss2a
        return total_loss


input_size = len(input_features[0])
model = Autoencoder(base_layer,input_size).double().to(device)

optimizer = optim.Adam(model.parameters(), lr=ILR)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factore, patience=patiencee)


with open('output.txt', 'a') as f:
    print(f'TOTAL EPOCHS: {num_epochs}\nDATA SIZE = {samplesize}    BATCH SIZE = {batchsize}\nSCHEDULER -- PATIENCE: {patiencee}    FACTOR: {factore}    INITIAL LR = {ILR}\nNETWORK -- BASE LAYER = {base_layer}\nLOSS FUNCTION -- ALPHA = {alpha}', file=f)


# Initialize weights
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)
            
            
# Assume 'model' is defined elsewhere
model.apply(weights_init)

# Initialize the custom loss function with a physics regularization factor
criterion = PhysicsInformedLoss(alpha=alpha)

def train(epoch, model, train_loader, optimizer, criterion, device, feature_mins, feature_maxs):
    model.train()
    train_loss = 0
    for batch_idx, (data, target_latent) in enumerate(train_loader):
        data, target_latent = data.to(device), target_latent.to(device)
        optimizer.zero_grad()

        output = model(data)
        latent = model.encoder(data)  # Extract latent vector to apply physical loss
        # Now pass all necessary components to the criterion
        loss = criterion(output, data, latent, target_latent, feature_mins, feature_maxs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        train_loss += loss.item()

    average_train_loss = train_loss / len(train_loader)
    #print(f'Train Epoch: {epoch}\tAverage Loss: {average_train_loss:.6f}')
    return average_train_loss

def test(model, test_loader, criterion, device, feature_mins, feature_maxs):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target_latent in test_loader:
            data, target_latent = data.to(device), target_latent.to(device)
            output = model(data)
            latent = model.encoder(data)
            loss = criterion(output, data, latent, target_latent, feature_mins, feature_maxs)
            test_loss += loss.item()

    average_test_loss = test_loss / len(test_loader)
    return average_test_loss

# Prepare the plot
fig, ax = plt.subplots()
train_losses, test_losses = [], []
train_line, = ax.plot(train_losses, label='Training Loss')
test_line, = ax.plot(test_losses, label='Test Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Initial Learning Rate: {:.5f}'.format(optimizer.param_groups[0]['lr']))
ax.legend()
plt.yscale("log")
plt.ion()

# Function to update plot for each epoch
def update_plot(epoch, avg_train_loss, avg_test_loss, train_losses, test_losses, current_lr):
    train_line.set_data(np.arange(len(train_losses)), train_losses)
    test_line.set_data(np.arange(len(test_losses)), test_losses)
    ax.relim()
    ax.autoscale_view()
    ax.set_title(f'Training and Test Loss (LR: {current_lr:.1e})')
    ax.legend([f'Training Loss: {avg_train_loss:.1e}', f'Test Loss: {avg_test_loss:.1e}'])
    plt.savefig('training_plot.png')
    plt.draw()

# training starts HERE

tot_time=0
for epoch in range(1, num_epochs + 1):
    start_time = time.time()
    avg_train_loss = train(epoch, model, train_loader, optimizer, criterion, device, feature_mins, feature_maxs)
    avg_test_loss = test(model, test_loader, criterion, device, feature_mins, feature_maxs)
    
    train_losses.append(avg_train_loss)
    test_losses.append(avg_test_loss)
    
    
    scheduler.step(avg_test_loss)  # Update learning rate based on test loss
    current_lr = scheduler.get_last_lr()[0]

    end_time = time.time()
    epoch_time = end_time-start_time
    tot_time +=epoch_time

    if (epoch) % 10 == 0:
            ts = tot_time/10
            with open('output.txt', 'a') as f:
                print(f'average epoch time: {ts}       train Loss: {avg_train_loss:.6f}, test Loss: {avg_test_loss:.6f}, epoch: {epoch}, LR = {current_lr:.6f}', file=f)
            tot_time=0

    if current_lr <= 1e-6:
        with open('output.txt', 'a') as f:
                print(f'\n\n#######\nsaving model EndOfDataset', file=f)
        save_model_with_logging(model, output_directory, base_layer, samplesize=samplesize, batchsize=batchsize, patiencee=patiencee, alpha=alpha)
        break
    if (epoch + 1) % 50 == 0:
        update_plot(epoch, avg_train_loss, avg_test_loss, train_losses, test_losses, current_lr)
    if (epoch + 1) % 1500 == 0:
        save_model_with_logging(model, output_directory, baselayer=base_layer, samplesize=samplesize, batchsize=batchsize, patiencee=patiencee, alpha=alpha)
   

with open('output.txt', 'a') as f:
            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!\nsaving the last model - EPOCHS LIMIT REACHED', file=f)
save_model_with_logging(model, output_directory, baselayer=base_layer, samplesize=samplesize, batchsize=batchsize, patiencee=patiencee, alpha=alpha)



