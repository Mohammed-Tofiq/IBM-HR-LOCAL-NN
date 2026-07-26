import torch
import torch.nn as nn
import numpy as np
import pickle

# 1. Neural Network Architecture (must exactly match train_attrition.py)
class AttritionNet(nn.Module):
    def __init__(self, input_shape):
        super(AttritionNet, self).__init__()
        self.fc1 = nn.Linear(input_shape, 32)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 1)  # raw logits, no sigmoid here

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

def ask_float(prompt, low=None, high=None):
    while True:
        try:
            val = float(input(prompt))
            if low is not None and val < low:
                print(f"   Please enter a value >= {low}.")
                continue
            if high is not None and val > high:
                print(f"   Please enter a value <= {high}.")
                continue
            return val
        except ValueError:
            print("   That's not a valid number, try again.")

def ask_choice(prompt, choices):
    choice_str = ', '.join(choices)
    while True:
        val = input(f"{prompt} [{choice_str}]: ").strip()
        if val in choices:
            return val
        print(f"   Please type it exactly as shown: {choice_str}")

def main():
    print("\n" + "="*50)
    print("CORPORATE AI HR ANALYTICS (TERMINAL MODE)")
    print("Built From Scratch | Zero Cloud Dependency")
    print("="*50 + "\n")

    # 2. Load Preprocessors and Model
    try:
        with open('hr_preprocessors.pkl', 'rb') as f:
            preprocessors = pickle.load(f)

        scaler = preprocessors['scaler']
        encoders = preprocessors['encoders']
        feature_order = preprocessors['feature_order']

        checkpoint = torch.load('attrition_model.pth', map_location=torch.device('cpu'), weights_only=True)
        model = AttritionNet(checkpoint['input_shape'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    except FileNotFoundError:
        print("Error: Model files not found. Please run 'train_attrition.py' first.")
        return

    print("Local AI Model loaded successfully!")
    if 'test_metrics' in checkpoint:
        m = checkpoint['test_metrics']
        print(f"(Model test performance -> Recall: {m['recall']:.2f}, "
              f"Precision: {m['precision']:.2f}, ROC-AUC: {m['roc_auc']:.2f})\n")

    print("Enter Employee Details:")
    print("-" * 25)

    try:
        # 3. Collect inputs for all 14 features, in the exact order the model expects
        values = {}
        values['Age'] = ask_float("Age (e.g., 30): ", low=18, high=75)
        values['MonthlyIncome'] = ask_float("Monthly Income in USD (e.g., 5000): ", low=0)
        values['JobSatisfaction'] = ask_float("Job Satisfaction (1-4): ", low=1, high=4)
        values['EnvironmentSatisfaction'] = ask_float("Environment Satisfaction (1-4): ", low=1, high=4)
        values['YearsAtCompany'] = ask_float("Years at Company: ", low=0)
        values['TotalWorkingYears'] = ask_float("Total Working Years (career total): ", low=0)
        values['StockOptionLevel'] = ask_float("Stock Option Level (0-3): ", low=0, high=3)
        values['WorkLifeBalance'] = ask_float("Work Life Balance (1-4): ", low=1, high=4)
        values['YearsInCurrentRole'] = ask_float("Years in Current Role: ", low=0)
        values['DistanceFromHome'] = ask_float("Distance From Home (miles/km): ", low=0)
        values['NumCompaniesWorked'] = ask_float("Number of Companies Worked At: ", low=0)

        overtime = ask_choice("\nWorks OverTime?", list(encoders['OverTime'].classes_))
        job_role = ask_choice("\nJob Role", list(encoders['JobRole'].classes_))
        marital_status = ask_choice("\nMarital Status", list(encoders['MaritalStatus'].classes_))

        values['OverTime'] = encoders['OverTime'].transform([overtime])[0]
        values['JobRole'] = encoders['JobRole'].transform([job_role])[0]
        values['MaritalStatus'] = encoders['MaritalStatus'].transform([marital_status])[0]

        # 4. Build input vector in the exact feature order used during training
        input_data = np.array([[values[feat] for feat in feature_order]])

        scaled_data = scaler.transform(input_data)
        tensor_data = torch.tensor(scaled_data, dtype=torch.float32)

        # 5. AI Inference (model outputs a logit -> apply sigmoid for probability)
        with torch.no_grad():
            logit = model(tensor_data)
            risk_probability = torch.sigmoid(logit).item()

        # 6. Print Results
        print("\n" + "="*40)
        print("AI PREDICTION RESULT")
        print("="*40)

        if risk_probability >= 0.5:
            print("HIGH RISK OF ATTRITION!")
            print(f"Resignation Probability: {risk_probability*100:.1f}%")
            print("Recommendation: Schedule a 1-on-1 meeting to discuss concerns.")
        else:
            print("SAFE / RETAINED")
            print(f"Resignation Probability: {risk_probability*100:.1f}%")
            print("Employee is likely to stay.")
        print("="*40 + "\n")

    except Exception as e:
        print(f"\nInput Error: {e}")
        print("Make sure you type numbers correctly and spell Job Role/OverTime/Marital Status exactly as shown.")

if __name__ == "__main__":
    main()