import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from utilits import save


# Простейшая кастомная нейросеть для классификации изображений (например, 1x28x28)
class BasicBlock(nn.Module):
    """Базовый блок с двумя свёртками и skip-connection."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip-connection: если размерность меняется (stride > 1 или каналов разное количество)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)  # Важно: skip-connection!
        out = F.relu(out)
        return out

class TinyResNet(nn.Module):
    """Упрощённый ResNet"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 16

        # Первый слой
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Блоки ResNet
        self.layer1 = self._make_layer(16, 2, stride=1)   # 64x28x28 -> 64x28x28
        self.layer2 = self._make_layer(128, 1, stride=2)  # 64x28x28 -> 128x14x14
        self.layer3 = self._make_layer(256, 1, stride=2)  # 128x14x14 -> 256x7x7
        # self.layer4 = self._make_layer(512, 2, stride=2)  # 256x8x8 -> 512x4x4

        # Финальный полносвязный слой
        self.linear = nn.Linear(256, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        """Создаёт слой из нескольких BasicBlock."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 1x28x28 -> 16x28x28
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)
        out = F.avg_pool2d(out, 7)  # Global Average Pooling: 256x7x7 -> 256x1x1
        out = out.view(out.size(0), -1)  # Flatten
        out = self.linear(out)
        return out


def get_dataset(path):

    if not os.path.exists(f'{path}data'):
        download = True
    else:
        download = False

    return get_MNIST(path, download)



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
        "onnx/resnet_cnn.onnx",  # имя выходного файла
        input_names=['input'],  # имя входа
        output_names=['output'],  # имя выхода
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},  # поддержка разных batch size
        opset_version=11  # версия ONNX (11 — совместимая)
    )
    print("Модель успешно экспортирована в ONNX.")

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
    test = datasets.MNIST(
        root=f'{base_path}data',
        train=False,
        download=download,
        transform=transform
    )

    return train, test


if __name__ == "__main__":
    base_path = './'

    model = TinyResNet(num_classes=10)
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

    save.to_json('metrics/resnet_nn', "train_losses", train_losses)
    save.to_json('metrics/resnet_nn', "test_accuracies", test_accuracies)

    model.eval()  # Переводим модель в режим оценки

    tensor_image, label = train_dataset[0]

    print(tensor_image.size())

    export_model(model, tensor_image.unsqueeze(0).to(device))
