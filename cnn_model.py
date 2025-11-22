# models.py
# File chứa 5 kiến trúc model CNN cho bài toán QuickDraw 28x28, 25 classes.
# Lựa chọn 5: MiniMobileNet (thay thế cho EfficientNet)

import torch
import torch.nn as nn
from torch.nn import functional as F
from math import pow
# (torchvision không cần thiết nữa nếu không dùng model pre-trained)

# ===============================================================
# 1. MODEL TỪ FILE `model.py` (Custom V1)
# ===============================================================
class QuickDrawV1(nn.Module):
    """
    Model gốc từ file 'model.py' của bạn.
    Conv1(28x28) -> Pool(12x12) -> Conv2(8x8) -> Pool(4x4)
    Flattened size: 64 * 4 * 4 = 1024
    """
    def __init__(self, input_size=28, num_classes=25):
        super(QuickDrawV1, self).__init__()
        self.num_classes = num_classes
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, bias=False),  # 28x28 -> 24x24
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2, 2)  # 24x24 -> 12x12
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, bias=False),  # 12x12 -> 8x8
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2, 2)  # 8x8 -> 4x4
        )
        
        dimension = int(64 * pow(input_size/4 - 3, 2)) # 1024
        
        self.fc1 = nn.Sequential(nn.Linear(dimension, 512), nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(512, 128), nn.Dropout(0.5))
        self.fc3 = nn.Sequential(nn.Linear(128, num_classes))

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = output.view(output.size(0), -1) # Flatten
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        return output

# ===============================================================
# 2. MODEL TỪ FILE `QuickDraw_Classifier_Class_v2_pytorch.py` (Custom V2)
# ===============================================================
class QuickDrawV2(nn.Module):
    """
    Model từ file 'QuickDraw_Classifier_Class_v2_pytorch.py'.
    Đã điều chỉnh cho 28x28 (output 256*1*1).
    Trace: 28x28 -> Pool(14x14) -> Pool(7x7) -> Pool(3x3) -> Pool(1x1)
    """
    def __init__(self, num_classes=25):
        super(QuickDrawV2, self).__init__()
        self.features = nn.Sequential(
            # --- Block 1 (Input: 28x28) ---
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 28 -> 14

            # --- Block 2 (Input: 14x14) ---
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 14 -> 7

            # --- Block 3 (Input: 7x7) ---
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),  # 7 -> 3

            # --- Block 4 (Input: 3x3) ---
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),  # 3 -> 1
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 1 * 1, 512), # Kích thước flatten đã điều chỉnh
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ===============================================================
# 3. MODEL LeNet-5
# ===============================================================
class LeNet5(nn.Module):
    """
    Kiến trúc LeNet-5 cổ điển, điều chỉnh cho 25 classes.
    Input: 28x28x1 -> C1(24x24x6) -> S2(12x12x6) -> C3(8x8x16) -> S4(4x4x16)
    Flatten: 4*4*16 = 256
    """
    def __init__(self, num_classes=25):
        super(LeNet5, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16 * 4 * 4, out_features=120), nn.ReLU(),
            nn.Linear(in_features=120, out_features=84), nn.ReLU(),
            nn.Linear(in_features=84, out_features=num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

# ===============================================================
# 4. MODEL CUSTOM SỬ DỤNG RESIDUAL BLOCK
# ===============================================================

class BasicBlock(nn.Module):
    """Helper class cho CustomResNet: một khối residual cơ bản."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity # Phép cộng residual
        out = F.relu(out)
        return out

class CustomResNet(nn.Module):
    """
    Một model nhỏ giống ResNet sử dụng BasicBlock.
    Trace: 28x28 -> (Conv 16) -> Block1(32,s=2) -> 14x14
           -> Block2(64,s=2) -> 7x7 -> Block3(128,s=2) -> 4x4
           -> AdaptiveAvgPool -> 1x1x128
    """
    def __init__(self, num_classes=25):
        super(CustomResNet, self).__init__()
        self.in_channels = 16
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.layer1 = self._make_layer(16, 2, stride=1) # 28x28
        self.layer2 = self._make_layer(32, 2, stride=2) # 14x14
        self.layer3 = self._make_layer(64, 2, stride=2) # 7x7
        self.layer4 = self._make_layer(128, 2, stride=2) # 4x4
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# ===============================================================
# 5. MODEL Mini-MobileNet (Tự code lại)
# ===============================================================

class DepthwiseSeparableConv(nn.Module):
    """
    Khối cơ bản của MobileNetV1: Depthwise + Pointwise Convolution.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Sequential(
            # Conv 3x3 depthwise, 'groups=in_channels' là mấu chốt
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.pointwise = nn.Sequential(
            # Conv 1x1 pointwise
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class MiniMobileNet(nn.Module):
    """
    Một kiến trúc giống MobileNetV1 thu nhỏ cho 28x28.
    Trace: 28x28 -> (Conv 32)
           -> DS-Conv(64, s=2)  -> 14x14
           -> DS-Conv(128, s=1) -> 14x14
           -> DS-Conv(128, s=2) -> 7x7
           -> DS-Conv(256, s=1) -> 7x7
           -> DS-Conv(256, s=2) -> 4x4
           -> AdaptiveAvgPool   -> 1x1x256
    """
    def __init__(self, num_classes=25):
        super(MiniMobileNet, self).__init__()
        
        # Lớp Conv đầu tiên (chuẩn, không phải depthwise)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1, bias=False), # 28x28
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Các khối Depthwise Separable
        self.features = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=2),   # -> 14x14
            DepthwiseSeparableConv(64, 128, stride=1),  # -> 14x14
            DepthwiseSeparableConv(128, 128, stride=2), # -> 7x7
            DepthwiseSeparableConv(128, 256, stride=1), # -> 7x7
            DepthwiseSeparableConv(256, 256, stride=2), # -> 4x4
        )
        
        # Lớp classifier
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ===============================================================
# KIỂM TRA NHANH (Bỏ comment để chạy test)
# ===============================================================
if __name__ == "__main__":
    # Tạo một tensor input giả lập (Batch=2, Channel=1, H=28, W=28)
    dummy_input = torch.randn(2, 1, 28, 28)
    
    models_list = {
        "QuickDrawV1": QuickDrawV1(num_classes=25),
        "QuickDrawV2": QuickDrawV2(num_classes=25),
        "LeNet5": LeNet5(num_classes=25),
        "CustomResNet": CustomResNet(num_classes=25),
        "MiniMobileNet": MiniMobileNet(num_classes=25) # Đã thay thế
    }
    
    print("Kiểm tra kích thước output của 5 model với input 2x1x28x28:\n")
    
    for name, model in models_list.items():
        try:
            output = model(dummy_input)
            print(f"✅ {name}: OK! Output shape: {output.shape}")
        except Exception as e:
            print(f"❌ {name}: FAILED! Error: {e}")
            
    # Output mong đợi cho tất cả: torch.Size([2, 25])