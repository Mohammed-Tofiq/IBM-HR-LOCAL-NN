import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os

# 1. Device Setup (Mac M4)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training on: {device}")

# 2. Load Dataset & Select Top 7 Features
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

# Only keep the 7 features we want, plus the target (Attrition)
selected_columns = ['Age', 'MonthlyIncome', 'OverTime', 'JobRole', 
                    'JobSatisfaction', 'EnvironmentSatisfaction', 'YearsAtCompany', 'Attrition']
df = df[selected_columns]

# 3. Data Preprocessing
label_encoders = {}
# Encode Categorical features
for col in ['OverTime', 'JobRole', 'Attrition']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop('Attrition', axis=1).values
y = df['Attrition'].values

# Scale Numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save preprocessors for the Streamlit UI
with open('hr_preprocessors.pkl', 'wb') as f:
    pickle.dump({'scaler': scaler, 'encoders': label_encoders}, f)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to PyTorch Tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

# 4. Neural Network Architecture
class AttritionNet(nn.Module):
    def __init__(self, input_shape):
        super(AttritionNet, self).__init__()
        self.fc1 = nn.Linear(input_shape, 32)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.relu2(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

input_shape = X_train.shape[1] # Will be 7
model = AttritionNet(input_shape).to(device)

# 5. Loss & Optimizer
# Added weights to handle imbalanced HR data (fewer people leave than stay)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# 6. Training Loop
print("Training Custom AI Model...")
epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 7. Save the Model
torch.save({'input_shape': input_shape, 'model_state_dict': model.state_dict()}, 'attrition_model.pth')
print("✅ Training Complete! Model and Preprocessors Saved locally.")