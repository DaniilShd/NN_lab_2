import json
import os

def to_json(path, name, my_list):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except Exception as e:
        print(f"Ошибка при создании папки: {e}")
    # Save list to file
    with open(f'{path}/{name}.json', 'w') as f:
        json.dump(my_list, f)