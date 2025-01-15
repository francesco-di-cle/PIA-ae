import numba
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d, UnivariateSpline, InterpolatedUnivariateSpline, splder, splrep
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



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

def safe_normalize(data, mean, std):
    std_zero_mask = std < 1e-8
    std_safe = torch.where(std_zero_mask, torch.ones_like(std), std)
    normalized_data = (data - mean) / std_safe
    normalized_data = torch.where(std_zero_mask, torch.ones_like(data), normalized_data)
    return normalized_data
def denormalize(tensor, mean, std):
    return tensor * std + mean

def load_dataset(filename):
    df = pd.read_csv(filename)
    print(f"Dataset loaded from {filename}")
    return df

loaded_dataset = load_dataset('datasets/filteredDatasly1713.csv')
datasetDF=pd.DataFrame(loaded_dataset)

datasetDF['Λns'] = pd.to_numeric(datasetDF['Λns'], errors='coerce') 
datasetDF['Λns'] = datasetDF['Λns'].apply(log)
datasetDF['ρ0'] = datasetDF['ρ0'].apply(log)
datasetDF['p1'] = datasetDF['p1'].apply(log)

datasetDF = datasetDF.dropna(subset=['Λns']) 
datasetDF=df_column_switch(datasetDF, "ρ0","raggio")

datasetDF = datasetDF[datasetDF["raggio"] < 17]
datasetDF = datasetDF[datasetDF["massa"] > 0.2]


dataset = torch.tensor(datasetDF.values).double()

input_features = dataset[:, :5]
target_features = dataset[:, 5:]

input_mean = input_features.mean(0, keepdim=True)
input_std = input_features.std(0, keepdim=True)

normalized_inputs = safe_normalize(input_features, input_mean, input_std).to(device)

target_mean = target_features.mean(0, keepdim=True)
target_std = target_features.std(0, keepdim=True)

normalized_targets = safe_normalize(target_features, target_mean, target_std).to(device)

# input_mean = torch.tensor([[79.3268,  2.6713,  2.5508,  2.6069, 34.2310]], dtype=torch.float64)
# input_std = torch.tensor([[0.5162, 0.6283, 0.7768, 0.7528, 0.6330]], dtype=torch.float64)
# target_mean = torch.tensor([[ 1.1441, 12.8125,  7.8177,  0.1378]], dtype=torch.float64)
# target_std = torch.tensor([[0.7290, 1.9179, 4.2745, 0.0911]], dtype=torch.float64)