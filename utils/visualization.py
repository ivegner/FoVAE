import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow_unnorm(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

