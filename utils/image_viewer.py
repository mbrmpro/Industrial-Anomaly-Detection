from PIL import Image


def load_images(image_paths):
    """
    Load multiple images.
    """

    images = []

    for path in image_paths:

        img = Image.open(path)

        images.append(img)

    return images

import random


def sample_images(image_paths, n=6):
    """
    Randomly sample images.
    """

    if len(image_paths) <= n:
        return image_paths

    return random.sample(image_paths, n)



def first_images(image_paths, n=6):
    """
    Return first n images.
    """

    return image_paths[:n]


def image_size(image):

    return image.size