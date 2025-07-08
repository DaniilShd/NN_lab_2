import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from utilits import save


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution - базовая единица MobileNet"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 1. Depthwise convolution (фильтрация по каналам)
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,  # ключевой параметр для depthwise
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)

        # 2. Pointwise convolution (объединение каналов)
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.depthwise(x)))
        x = F.relu(self.bn2(self.pointwise(x)))
        return x


class SimpleMobileNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Первый обычный сверточный слой
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # Последовательность Depthwise Separable блоков
        self.conv2 = DepthwiseSeparableConv(32, 64, stride=1)
        self.conv3 = DepthwiseSeparableConv(64, 128, stride=2)
        self.conv4 = DepthwiseSeparableConv(128, 128, stride=1)
        self.conv5 = DepthwiseSeparableConv(128, 256, stride=2)
        self.conv6 = DepthwiseSeparableConv(256, 256, stride=1)
        self.conv7 = DepthwiseSeparableConv(256, 512, stride=2)

        # Несколько одинаковых блоков
        self.conv8 = DepthwiseSeparableConv(512, 512, stride=1)
        self.conv9 = DepthwiseSeparableConv(512, 512, stride=1)
        self.conv10 = DepthwiseSeparableConv(512, 512, stride=1)
        self.conv11 = DepthwiseSeparableConv(512, 512, stride=1)
        self.conv12 = DepthwiseSeparableConv(512, 512, stride=1)

        # Завершающие слои
        self.conv13 = DepthwiseSeparableConv(512, 1024, stride=2)
        self.conv14 = DepthwiseSeparableConv(1024, 1024, stride=1)

        # Глобальный пулинг и классификатор
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def get_dataset(path):

    if not os.path.exists(f'{path}data'):
        download = True
    else:
        download = False

    return get_MNIST(path, download)

def get_MNIST(base_path, download):
    # Преобразования для нормализации данных
    transform = transforms.Compose([
        transforms.ToTensor(),  # Преобразует изображение в тензор [1, 28, 28]
        transforms.Normalize((0.1307,), (0.3081,))  # Нормализация (mean, std) MNIST
    ])

    # Загрузка тренировочного и тестового наборов
    train = datasets.MNIST(
        root=f'{base_path}data',
        train=True,
        download=download,
        transform=transform
    )
    test= datasets.MNIST(
        root=f'{base_path}data',
        train=False,
        download=download,
        transform=transform
    )

    return train, test


def get_dataloader(train_dataset, test_dataset, batch_size, shuffle):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, test_loader


def train_model(model, criterion, device, optimiser, num_epochs, train_loader, test_loader):

    train_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0

        for batch_id, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)

            optimiser.zero_grad()

            output = model(data)

            loss = criterion(output, labels)

            running_loss += loss.item()

            loss.backward()
            optimiser.step()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        model.eval()
        total = 0
        correct = 0

        with torch.no_grad():
            for _, (data, labels) in enumerate(test_loader):
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Test Accuracy: {accuracy:.2f}%')

    return train_losses, test_accuracies

def export_model(model, dummy_input):
    torch.onnx.export(
        model,  # модель
        dummy_input,  # пример входа
        "onnx/mobilenet_cnn.onnx",  # имя выходного файла
        input_names=['input'],  # имя входа
        output_names=['output'],  # имя выхода
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},  # поддержка разных batch size
        opset_version=11  # версия ONNX (11 — совместимая)
    )
    print("Модель успешно экспортирована в ONNX.")



if __name__ == "__main__":
    base_path = './'

    model = SimpleMobileNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)  # Перенос модели на устройство

    train_dataset, test_dataset = get_dataset(base_path)

    train_loader, test_loader = get_dataloader(train_dataset,
                                               test_dataset,
                                               32,
                                               True)

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses, test_accuracies = train_model(model,
                                                criterion,
                                                device,
                                                optimiser,
                                                5,
                                                train_loader,
                                                test_loader)

    save.to_json('metrics/mobilenet_nn', "train_losses", train_losses)
    save.to_json('metrics/mobilenet_nn', "test_accuracies", test_accuracies)

    model.eval()  # Переводим модель в режим оценки

    tensor_image, label = train_dataset[0]

    export_model(model, tensor_image.unsqueeze(0).to(device))