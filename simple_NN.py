import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from utilits import save


# Простейшая кастомная нейросеть для классификации изображений (например, 1x28x28)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Первый сверточный слой: вход 1 канал, выход 8 каналов, ядро 3x3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Второй сверточный слой: вход 8 каналов, выход 16 каналов, ядро 3x3
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 3 * 3, 256)  # предполагается input size 28x28
        self.fc2 = nn.Linear(256, 10)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Первый сверточный слой + ReLU + MaxPool
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.max_pool2d(x, 2)  # уменьшаем размер в 2 раза
        x = self.dropout(x)

        # Второй сверточный слой + ReLU + MaxPool
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.max_pool2d(x, 2)  # снова уменьшаем
        x = self.dropout(x)

        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = F.max_pool2d(x, 2)  # снова уменьшаем
        x = self.dropout(x)

        # Преобразуем тензор в вектор перед полносвязным слоем

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # Полносвязный слой
        x = F.relu(self.fc2(x))
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
        "onnx/simple_cnn.onnx",  # имя выходного файла
        input_names=['input'],  # имя входа
        output_names=['output'],  # имя выхода
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},  # поддержка разных batch size
        opset_version=11  # версия ONNX (11 — совместимая)
    )
    print("Модель успешно экспортирована в ONNX.")



if __name__ == "__main__":
    base_path = './'

    model = SimpleCNN()
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
                                                10,
                                                train_loader,
                                                test_loader)

    save.to_json('metrics/simple_nn', "train_losses", train_losses)
    save.to_json('metrics/simple_nn', "test_accuracies", test_accuracies)

    model.eval()  # Переводим модель в режим оценки

    tensor_image, label = train_dataset[0]

    export_model(model, tensor_image.unsqueeze(0).to(device))






