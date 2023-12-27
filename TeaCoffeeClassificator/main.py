import numpy as np
import random


# Генерация случайных данных
def generate_data(num_samples):
    data = []
    for _ in range(num_samples):
        age = random.randint(1, 100)  # Возраст
        gender = random.choice([0, 1])  # 0 - М, 1 - Ж
        weight = random.randint(50, 90)  # Вес
        commute_time = random.randint(10, 300)  # время пути на работу в минутах
        sleep_time = random.randint(0, 24)  # время сна в часах
        work_duration = random.randint(0, 24)  # продолжительность рабочего дня в часах
        beverage = random.choice(['чай', 'кофе'])  # метка

        data.append([age, gender, weight, commute_time, sleep_time, work_duration, beverage])

    return data


# Функция для вычисления евклидова расстояния между двумя точками
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


# Функция для поиска k ближайших соседей
def k_nearest_neighbors(train_data, test_instance, k):
    distances = []
    for i in range(len(train_data)):
        # Вычисляем расстояние между тестовым экземпляром и каждым обучающим экземпляром
        dist = euclidean_distance(np.array(train_data[i][:6]), np.array(test_instance))
        distances.append((train_data[i], dist))

    # Сортируем по расстоянию
    distances.sort(key=lambda x: x[1])

    # Получаем k ближайших соседей
    neighbors = [item[0] for item in distances[:k]]

    return neighbors


# Функция для определения класса на основе большинства среди соседей
def predict_class(neighbors):
    class_votes = {}
    for neighbor in neighbors:
        label = neighbor[-1]
        if label in class_votes:
            class_votes[label] += 1
        else:
            class_votes[label] = 1

    # Возвращаем класс, имеющий наибольшее количество голосов
    return max(class_votes, key=class_votes.get)


# Пример данных
data = generate_data(1000)
# Новый тестовый экземпляр
new_instance = [27, 0, 65, 25, 7, 8]

# Задаем количество соседей
k_neighbors = 1

# Находим ближайших соседей
neighbors = k_nearest_neighbors(data, new_instance, k_neighbors)

# Предсказываем класс на основе большинства среди соседей
prediction = predict_class(neighbors)

print(f'Прогноз напитка для новых данных: {prediction}')
