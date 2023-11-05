import mnist  # Potrzebujesz zainstalowanej biblioteki mnist
from network_chat import Network
import numpy as np


# Załaduj dane treningowe i testowe
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Przygotuj dane do treningu
training_data = [(np.reshape(x, (784, 1)) / 255, np.array([[int(y == i)] for i in range(10)])) for x, y in zip(train_images, train_labels)]

# Utwórz instancję sieci neuronowej
sizes = [784, 30, 10]  # Liczby neuronów w warstwach
network = Network(sizes)

# Trenuj sieć neuronową
epochs = 30
mini_batch_size = 10
learning_rate = 3.0
network.train(training_data, epochs, mini_batch_size, learning_rate)

# Ocena skuteczności na zbiorze testowym
test_data = [(np.reshape(x, (784, 1)) / 255, y) for x, y in zip(test_images, test_labels)]
accuracy = network.evaluate(test_data)

print(f"Skuteczność sieci neuronowej: {accuracy * 100}%")
