import torch
import torch.nn as nn

class SimpleOCRModel(nn.Module):
    def __init__(self):
        super(SimpleOCRModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  
        self.fc2 = nn.Linear(128, 10) 

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 64 * 8 * 8)  
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = SimpleOCRModel()
torch.save(model.state_dict(), 'model_gpu.pth')
print("model_gpu.pth file has been created successfully.")
