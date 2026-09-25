import pandas as pd
import numpy as np
import torch
import torch.nn
from sklearn.linear_model import LinearRegression

df = pd.load_csv("data.csv")

X = df["product_weight"]
y = df["product_category"]

categories = {
  "Home & Kitchen":  0,
  "Beauty": 1,
  "Toys": 2,
  "Grocery": 3,
  "Electronics": 4,
  "Health & Personal Care": 5,
  "Automotive": 6,
  "Office Supplies": 7,
  "Books":8,
  "Sports & Fitness": 9,
  "Fashion": 10,
  "Pet Supplies": 11,
    
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
        x = self.out(x)
        return x
