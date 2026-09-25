import pandas as pd
import numpy as np
import torch
import torch.nn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("/kaggle/input/datasets/datascikhan/e-commerce-delivery-and-shipping-data-2026/E-commerce_Delivery_Shipping_Data_2026.csv.csv")

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
  return float(categories[x])
  
y = y.apply(productClean)  

x2 = X
y2 = y

X = torch.FloatTensor(X)
y = torch.LongTensor(y)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, 32)
        self.out = nn.Linear(32, 11)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.out(x)
        return x


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
torch.manual_seed(41)
model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 1500

for i in range(epochs):
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        predictions = torch.argmax(y_pred, dim=1)
        accuracy = (predictions == y_train).float().mean()
        print(accuracy)
        
with torch.no_grad():
    model.eval()
    test_outputs = model(X_test)
    predictions = torch.argmax(test_outputs, dim=1)
    accuracy = (predictions == y_test).float().mean()
    print(f"Test Accuracy: {accuracy.item():.4f}")

with torch.no_grad():
    model.eval()

