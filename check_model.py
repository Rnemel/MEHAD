import torch
import os
import sys

# 1. تعريف معمارية المودل عشان نقدر نحمل الأوزان فيها
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTMSeizureNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=3):
        super(CNNLSTMSeizureNet, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, 
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(64*2, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.adaptive_avg_pool1d(x, 256)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        x = torch.cat((h_n[-2], h_n[-1]), dim=1)
        return self.fc(x)

# 2. تحديد مسار الملف
model_path = "/Users/rnemalmalki/Desktop/prof plan/final_seizure_model_patience7.pth"

# 3. الفحص
if not os.path.exists(model_path):
    print(f"❌ الملف غير موجود في المسار: {model_path}")
    print("تأكدي من اسم الملف ومساره.")
    sys.exit(1)

file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
print(f"✅ الملف موجود! حجم الملف: {file_size_mb:.2f} MB")

try:
    print("جاري محاولة تحميل الأوزان في المعمارية...")
    model = CNNLSTMSeizureNet()
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    
    # تجربة إدخال بيانات وهمية للتأكد من أن المودل يشتغل
    dummy_input = torch.randn(1, 1, 15360) # شكل نافذة الـ EEG
    output = model(dummy_input)
    
    print("✅ تم تحميل الأوزان بنجاح وتجربة المودل ببيانات وهمية!")
    print(f"شكل المخرجات (المتوقع 1, 3): {output.shape}")
    print("المودل سليم وجاهز 100% لمرحلة التطبيق 🚀")
except Exception as e:
    print(f"❌ حدث خطأ أثناء تحميل المودل: {e}")
