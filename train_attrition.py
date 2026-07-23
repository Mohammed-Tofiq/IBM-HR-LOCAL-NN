import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from sklearn.inspection import permutation_importance
import pickle
import copy

# 1. Device Setup (Mac M4)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training on: {device}")

# 2. Load Dataset & Select Features
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

# predictors of attrition in this dataset that were previously dropped.
NUMERIC_FEATURES = [
    'Age', 'MonthlyIncome', 'JobSatisfaction', 'EnvironmentSatisfaction',
    'YearsAtCompany', 'TotalWorkingYears', 'StockOptionLevel',
    'WorkLifeBalance', 'YearsInCurrentRole', 'DistanceFromHome',
    'NumCompaniesWorked'
]
CATEGORICAL_FEATURES = ['OverTime', 'JobRole', 'MaritalStatus']
TARGET = 'Attrition'

selected_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
df = df[selected_columns].copy()

# 3. Data Preprocessing
label_encoders = {}
for col in CATEGORICAL_FEATURES + [TARGET]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

FEATURE_ORDER = NUMERIC_FEATURES + CATEGORICAL_FEATURES  
X = df[FEATURE_ORDER].values
y = df[TARGET].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train / Validation / Test Split (stratified, 70/15/15)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
print(f"Attrition rate -> train: {y_train.mean():.3f}, val: {y_val.mean():.3f}, test: {y_test.mean():.3f}")

X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

# 5. Neural Network Architecture
class AttritionNet(nn.Module):
    def __init__(self, input_shape):
        super(AttritionNet, self).__init__()
        self.fc1 = nn.Linear(input_shape, 32)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 1)  

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

input_shape = X_train.shape[1]
model = AttritionNet(input_shape).to(device)
# 6. Loss & Optimizer (class-weighted for imbalance)
n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
print(f"pos_weight for minority class (Attrition=Yes): {pos_weight.item():.2f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.005)

# 7. Training Loop with Validation Tracking + Early Stopping
print("\nTraining Custom AI Model...")
epochs = 300
patience = 25
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_state = None

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_t)
        val_loss = criterion(val_outputs, y_val_t).item()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no val improvement for {patience} epochs).")
            break

# Restore best-performing weights on validation set
model.load_state_dict(best_model_state)

# 8. Evaluation on Held-Out Test Set
model.eval()
with torch.no_grad():
    test_logits = model(X_test_t)
    test_probs = torch.sigmoid(test_logits).cpu().numpy().flatten()
    test_preds = (test_probs >= 0.5).astype(int)

acc = accuracy_score(y_test, test_preds)
prec = precision_score(y_test, test_preds)
rec = recall_score(y_test, test_preds)
f1 = f1_score(y_test, test_preds)
auc = roc_auc_score(y_test, test_probs)
cm = confusion_matrix(y_test, test_preds)

print("\n" + "=" * 50)
print("TEST SET EVALUATION")
print("=" * 50)
print(f"Accuracy:  {acc:.3f}")
print(f"Precision: {prec:.3f}  (of predicted leavers, how many actually left)")
print(f"Recall:    {rec:.3f}  (of actual leavers, how many we caught)")
print(f"F1 Score:  {f1:.3f}")
print(f"ROC-AUC:   {auc:.3f}")
print("\nConfusion Matrix (rows=actual, cols=predicted) [No, Yes]:")
print(cm)
print("\nFull classification report:")
print(classification_report(y_test, test_preds, target_names=['No', 'Yes']))
print("=" * 50)

# 9. Feature Importance (permutation importance on test set)
class TorchWrapper:
    """Wraps the torch model so sklearn's permutation_importance can call it."""
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def fit(self, X, y):
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32).to(self.device)
            probs = torch.sigmoid(self.model(t)).cpu().numpy().flatten()
        return (probs >= 0.5).astype(int)

    def score(self, X, y):
        preds = self.predict(X)
        return f1_score(y, preds)

wrapper = TorchWrapper(model, device)
perm_result = permutation_importance(
    wrapper, X_test, y_test, scoring=None, n_repeats=15, random_state=42
)

importance_df = pd.DataFrame({
    'feature': FEATURE_ORDER,
    'importance_mean': perm_result.importances_mean,
    'importance_std': perm_result.importances_std
}).sort_values('importance_mean', ascending=False)

print("\nFeature importance (drop in F1 score when feature is shuffled):")
print(importance_df.to_string(index=False))

# 10. Save Preprocessors and Model
with open('hr_preprocessors.pkl', 'wb') as f:
    pickle.dump({
        'scaler': scaler,
        'encoders': label_encoders,
        'feature_order': FEATURE_ORDER
    }, f)

torch.save({
    'input_shape': input_shape,
    'feature_order': FEATURE_ORDER,
    'model_state_dict': model.state_dict(),
    'test_metrics': {
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1': f1, 'roc_auc': auc
    }
}, 'attrition_model.pth')

print("\nTraining Complete! Model and Preprocessors Saved locally.")
print("NOTE: the model now outputs raw logits (no Sigmoid layer) and uses")
print("an expanded feature set (14 features instead of 7). terminal_app.py")
print("needs matching updates before it will work with this checkpoint.")