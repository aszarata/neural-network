import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import sys
sys.path.append('network') 
from network import Network



image = np.zeros((28, 28), dtype=np.uint8) 

drawing = False
accepted = False

model = Network()

model = model.load('models/mnist1model.pickle')

def draw_on_image(x, y):
    image[max(0, y-1):min(28, y+1), max(0, x-1):min(28, x+1)] = 255

def onmotion(event):
    global drawing
    if drawing:
        x = int(event.xdata)
        y = int(event.ydata)
        draw_on_image(x, y)
        update_plot()

def onpress(event):
    global drawing
    drawing = True

def onrelease(event):
    global drawing
    drawing = False

def accept_drawing(event):

    read_input(image)

    # Clear the image
    plt.close()

def update_plot():
    plt.imshow(image, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
    plt.draw()


def read_input(image):
    pixel_values = image.flatten()
    X = np.array(pixel_values)
    y = model.predict(X)
    output = np.unravel_index(np.argmax(y), y.shape)
    print("prediction: ", output)

while True:
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray', interpolation='nearest', vmin=0, vmax=255)

    fig.canvas.mpl_connect('motion_notify_event', onmotion)
    fig.canvas.mpl_connect('button_press_event', onpress)
    fig.canvas.mpl_connect('button_release_event', onrelease)
    fig.canvas.mpl_connect('key_press_event', accept_drawing)

    cursor = plt.Circle((0, 0), 1, color='gray', fill=False, linewidth=2)
    ax.add_patch(cursor)


    plt.show()
    image = np.zeros((28, 28), dtype=np.uint8) 