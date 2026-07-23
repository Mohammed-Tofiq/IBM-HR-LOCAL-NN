import torch
import torch.nn as nn
import numpy as np
import pickle

# 1. Neural Network Architecture (Training wali exact same class)
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

def main():
    print("\n" + "="*50)
    print("🏢 CORPORATE AI HR ANALYTICS (TERMINAL MODE)")
    print("🔒 Built From Scratch | Zero Cloud Dependency")
    print("="*50 + "\n")
    
    # 2. Load Preprocessors and Model
    try:
        with open('hr_preprocessors.pkl', 'rb') as f:
            preprocessors = pickle.load(f)
        
        scaler = preprocessors['scaler']
        encoders = preprocessors['encoders']
        
        checkpoint = torch.load('attrition_model.pth', map_location=torch.device('cpu'), weights_only=True)
        model = AttritionNet(checkpoint['input_shape'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    except FileNotFoundError:
        print("❌ Error: Model files not found. Please run 'train_attrition.py' first.")
        return

    print("✅ Local AI Model loaded successfully!\n")
    print("Enter Employee Details:")
    print("-" * 25)
    
    try:
        # 3. Take inputs directly from terminal
        age = float(input("👉 Age (e.g., 30): "))
        income = float(input("👉 Monthly Income in USD (e.g., 5000): "))
        
        print(f"\nAvailable Job Roles: {', '.join(encoders['JobRole'].classes_)}")
        job_role = input("👉 Job Role (Type EXACTLY as shown above): ")
        
        print(f"\nAvailable OverTime options: {', '.join(encoders['OverTime'].classes_)}")
        overtime = input("👉 Works OverTime? (Type EXACTLY as shown above): ")
        
        job_satisfaction = float(input("\n👉 Job Satisfaction (1 to 4): "))
        env_satisfaction = float(input("👉 Environment Satisfaction (1 to 4): "))
        years_at_company = float(input("👉 Years at Company: "))

        # 4. Process Inputs
        encoded_role = encoders['JobRole'].transform([job_role])[0]
        encoded_overtime = encoders['OverTime'].transform([overtime])[0]

        input_data = np.array([[age, income, encoded_overtime, encoded_role, 
                                job_satisfaction, env_satisfaction, years_at_company]])
        
        scaled_data = scaler.transform(input_data)
        tensor_data = torch.tensor(scaled_data, dtype=torch.float32)
        
        # 5. AI Inference
        with torch.no_grad():
            risk_probability = model(tensor_data).item()
        
        # 6. Print Results
        print("\n" + "="*40)
        print("📊 AI PREDICTION RESULT")
        print("="*40)
        
        if risk_probability >= 0.5:
            print("⚠️  HIGH RISK OF ATTRITION!")
            print(f"📉 Resignation Probability: {risk_probability*100:.1f}%")
            print("💡 Recommendation: Schedule a 1-on-1 meeting to discuss concerns.")
        else:
            print("✅ SAFE / RETAINED")
            print(f"📈 Resignation Probability: {risk_probability*100:.1f}%")
            print("💡 Employee is satisfied and likely to stay.")
        print("="*40 + "\n")

    except Exception as e:
        print(f"\n❌ Input Error: {e}")
        print("Make sure you type numbers correctly and spell Job Roles/OverTime exactly as shown.")

if __name__ == "__main__":
    main()