import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os

# 设置设备
device = torch.device("cuda")

# 使用预训练的 ResNet18 模型，并进行微调
class BicycleClassifier(nn.Module):
    def __init__(self):
        super(BicycleClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.model(x)

# 加载训练好的模型
model = BicycleClassifier().to(device)
model.load_state_dict(torch.load('test.pth', map_location=device))
model.eval()  # 将模型设置为评估模式

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 使用预训练模型的标准化参数
])

# 预测函数
def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # 添加批次维度
    with torch.no_grad():
        output = model(image)
        probability = torch.sigmoid(output).item()
    return probability

# 示例：遍历 predict 文件夹中的所有 PNG 图片并进行预测
if __name__ == '__main__':
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    predict_dir = os.path.join(script_dir, 'predict')

    # 遍历 predict 文件夹中的所有图像文件
    for img_file in os.listdir(predict_dir):
        if img_file.endswith(('.png', '.jpg')):
            img_path = os.path.join(predict_dir, img_file)
            result = predict(img_path)
            if result > 0.5:
                print(f'图像 {img_file} 中包含xx，置信度为：{result:.4f}')
            else:
                print(f'图像 {img_file} 中不包含xx，置信度为：{result:.4f}')
