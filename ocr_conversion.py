import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


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
        return self.fc2(x)

def check_file_exists(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

def load_model(model_path, device='cpu'):
    check_file_exists(model_path)
    model = SimpleOCRModel()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()
    return model

def perform_ocr(model, device, video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    results, processing_times = [], []

    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        input_frame = transform(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            results.append(model(input_frame))
        processing_times.append(time.time() - start_time)

    cap.release()
    avg_fps = len(processing_times) / sum(processing_times) if processing_times else 0
    return results, avg_fps

def evaluate_results(gpu_results, cpu_results):
    accuracy = np.mean([
        torch.equal(gpu, cpu) for gpu, cpu in zip(gpu_results, cpu_results)
    ])
    return accuracy * 100

def main():
    # Paths
    gpu_model_path = "model_gpu.pth"
    cpu_model_path = "models/model_cpu.pth"
    video_path = "videos/input_video.mp4"

    # Check files
    check_file_exists(video_path)

    # Load Models
    device_gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_model = load_model(gpu_model_path, device_gpu)

    device_cpu = torch.device('cpu')
    cpu_model = load_model(gpu_model_path, device_cpu)
    torch.save(cpu_model.state_dict(), cpu_model_path)

    # Perform OCR
    gpu_results, gpu_fps = perform_ocr(gpu_model, device_gpu, video_path)
    print(f"GPU Model FPS: {gpu_fps}")

    cpu_results, cpu_fps = perform_ocr(cpu_model, device_cpu, video_path)
    print(f"CPU Model FPS: {cpu_fps}")

    # Evaluate Accuracy
    accuracy = evaluate_results(gpu_results, cpu_results)
    print(f"Accuracy: {accuracy:.2f}%")

    # Plot FPS Comparison
    plt.figure(figsize=(10, 5))
    plt.bar(['GPU', 'CPU'], [gpu_fps, cpu_fps], color=['blue', 'green'])
    plt.ylabel('Frames Per Second (FPS)')
    plt.title('FPS Comparison: GPU vs CPU')
    plt.show()
if __name__ == "__main__":
    main()
