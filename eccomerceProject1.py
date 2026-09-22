import pandas as pd
import numpy as np
import torch
import torch.nn
from sklearn.linear_model import LinearRegression

df = pd.load_csv("data.csv")

X = df["product_weight"]
y = df["product_category"]

categories = {
  "":  0,
  "": 1,
  "": 2,
  "": 3,
  "": 4,
  "": 5,
  "": 6,
}

def productClean(x):
  return categories[x]
  
X = X.apply(productClean)  

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, 32)
        self.out = nn.Linear(32, 6)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.out(x)
        return x
