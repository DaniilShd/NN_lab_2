import matplotlib.pyplot as plt
import json

def read_json_to_list(file_path):
    try:
        with open(f"{file_path}", 'r', encoding='utf-8') as file:
            data = json.load(file)

            # Если данные не являются списком, помещаем их в список
            if not isinstance(data, list):
                data = [data]
        return data
    except FileNotFoundError:
        print(f"Ошибка: Файл {file_path} не найден.")
        return []
    except json.JSONDecodeError:
        print(f"Ошибка: Файл {file_path} содержит некорректный JSON.")
        return []


base_path = "../metrics/resnet_nn"

dt1 = read_json_to_list(f"{base_path}/test_accuracies.json")
dt2 = read_json_to_list(f"{base_path}/train_losses.json")


plt.figure(figsize=(12, 7))
plt.subplot(1, 2, 1)
plt.plot(dt1, 'o-r', alpha=0.7, label="test_accuracies", lw=5, mec='b', mew=2, ms=10)
# Подпись точек
# for i, y in enumerate(dt1):
#     plt.text(i, y, f'{y}',
#              ha='center', va='bottom')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(dt2, 'o-r', alpha=0.7, label="train_losses", lw=5, mec='b', mew=2, ms=10)
plt.legend()
# # Подпись точек
# for i, y in enumerate(dt2):
#     plt.text(i, y, f'{y}',
#              ha='center', va='bottom')
plt.grid(True)
plt.show()