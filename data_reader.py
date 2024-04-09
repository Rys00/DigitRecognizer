import numpy as np
from PIL import Image
from typing import Literal


def show_img(buf, row: int, col: int):
    buf = buf.reshape(col, row)
    img = Image.fromarray(buf)
    img.show()


def save_img(buf, row: int, col: int, filename: str):
    buf = buf.reshape(col, row)
    img = Image.fromarray(buf)
    img.save(f"{filename}.jpg")


def get_info(filename: str):
    with open(filename, "rb") as f:
        f.read(4)  # magic number
        count = int.from_bytes(f.read(4))
        row = int.from_bytes(f.read(4))
        col = int.from_bytes(f.read(4))
        f.close()
        return count, row, col


def get_images(filename: str, mode: Literal["0-255", "0-1"] = "0-1"):
    with open(filename, "rb") as f:
        f.read(4)  # magic number
        count = int.from_bytes(f.read(4))
        row = int.from_bytes(f.read(4))
        col = int.from_bytes(f.read(4))
        pixels = row * col
        images = np.frombuffer(f.read(pixels * count), dtype=np.uint8)
        images = images.reshape(count, pixels)
        if mode == "0-1":
            images = images.astype(np.float32)
            images = images / 255
        f.close()
        return images


def get_labels(filename: str):
    with open(filename, "rb") as f:
        f.read(4)  # magic number
        count = int.from_bytes(f.read(4))
        labels = np.frombuffer(f.read(count), dtype=np.uint8)
        f.close()
        return labels


def get_train_data():
    count, row, col = get_info("train-images.idx3-ubyte")
    return (
        count,
        row,
        col,
        get_images("train-images.idx3-ubyte", mode="0-1"),
        get_labels("train-labels.idx1-ubyte"),
    )


def get_test_data():
    count, row, col = get_info("t10k-images.idx3-ubyte")
    return (
        count,
        row,
        col,
        get_images("t10k-images.idx3-ubyte", mode="0-1"),
        get_labels("t10k-labels.idx1-ubyte"),
    )


if __name__ == "__main__":
    count, row, col = get_info("train-images.idx3-ubyte")
    images = get_images("train-images.idx3-ubyte", mode="0-255")
    labels = get_labels("train-labels.idx1-ubyte")
    for i in range(10):
        save_img(images[i], row, col, f"images/{labels[i]}_imgnr{i}")
