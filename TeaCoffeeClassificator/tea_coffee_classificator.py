import pickle

import numpy as np
import numpy.typing as npt


from TeaCoffeeClassificator.settings import WEIGHTS_PATH


class TeaCoffeeClassificator:
    def __init__(self) -> None:
        self.weights: tuple[npt.NDArray[npt.NDArray[np.int64]]] = self.load()

    def predict(self, k: int, data: list[int]) -> str:
        """
        Получить предсказание от модели.
        :param

        k: количество соседей
        data: данные для предсказания

        :return: наиболее релевантное предложение.
        """
        distances = []
        for i in range(self.weights[0].shape[0]):
            distance = np.sqrt(np.sum((self.weights[0][i] - data) ** 2))
            distances.append((distance, self.weights[1][i]))
        distances.sort(key=lambda x: x[0])
        k_nearest_neighbors = distances[:k]
        class_counts = {0: 0, 1: 0}
        for _, label in k_nearest_neighbors:
            class_counts[label] += 1
        predict_class: int = max(class_counts, key=class_counts.get)
        return "кофе" if predict_class else "чай"

    @staticmethod
    def train():
        # Признаки (возраст, пол, вес, время пути на работу, время сна, длительность рабочего дня, )
        x_train = np.array([[25, 1, 70],
                            [30, 0, 65],
                            [40, 1, 80],
                            [35, 0, 75],
                            [28, 1, 68]])

        # Классы (чай - 0, кофе - 1)
        y_train = np.array([0, 1, 0, 1, 0])
        # Сохранение весов
        weights = (x_train, y_train)

        # Загрузка весов модели
        with open(WEIGHTS_PATH / 'model_weights.pkl', 'wb') as file:
            pickle.dump(weights, file)

    @staticmethod
    def load() -> tuple[npt.NDArray[npt.NDArray[np.int64]]]:
        """
        Загрузить из pickle модель.
        :param:
        :return: объект модели в памяти.
        """
        with open(WEIGHTS_PATH / 'model_weights.pkl', 'rb') as file:
            return pickle.load(file)
