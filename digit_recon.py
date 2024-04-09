import data_reader as data
import numpy as np


class DigitRecognizer:

    def __init__(self, size) -> None:
        np.random.seed(1)

        self.i_size = size
        self.h1_size = 16
        self.h2_size = 16
        self.o_size = 10

        self.i_state = np.zeros((self.i_size), dtype=np.float32)
        self.h1_state = np.zeros((self.h1_size), dtype=np.float32)
        self.h2_state = np.zeros((self.h1_size), dtype=np.float32)
        self.o_state = np.zeros((self.o_size), dtype=np.float32)
        self.h1_state_raw = np.zeros((self.h1_size), dtype=np.float32)
        self.h2_state_raw = np.zeros((self.h1_size), dtype=np.float32)
        self.o_state_raw = np.zeros((self.o_size), dtype=np.float32)

        self.weights_i_h1 = 2 * np.random.random((self.i_size, self.h1_size)) - 1
        self.weights_h1_h2 = 2 * np.random.random((self.h1_size, self.h2_size)) - 1
        self.weights_h2_o = 2 * np.random.random((self.h2_size, self.o_size)) - 1

        self.bias_radius = 10
        m = 2 * self.bias_radius
        b = self.bias_radius
        self.biases_i_h1 = m * np.random.random((self.h1_size)) - b
        self.biases_h1_h2 = m * np.random.random((self.h2_size)) - b
        self.biases_h2_o = m * np.random.random((self.o_size)) - b

        # cdo stands for cost derivative over ...
        # "s" stands for state, "w" for weight and "b" for bias
        self.cdo_cache_ih1_w = np.zeros((self.i_size, self.h1_size), dtype=np.float32)
        self.cdo_cache_ih1_b = np.zeros((self.h1_size), dtype=np.float32)
        self.cdo_cache_h1h2_w = np.zeros((self.h1_size, self.h2_size), dtype=np.float32)
        self.cdo_cache_h1h2_b = np.zeros((self.h2_size), dtype=np.float32)
        self.cdo_cache_h2o_w = np.zeros((self.h2_size, self.o_size), dtype=np.float32)
        self.cdo_cache_h2o_b = np.zeros((self.o_size), dtype=np.float32)

    def __clear_cache(self):
        self.cdo_cache_ih1_w = np.zeros((self.i_size, self.h1_size), dtype=np.float32)
        self.cdo_cache_ih1_b = np.zeros((self.h1_size), dtype=np.float32)
        self.cdo_cache_h1h2_w = np.zeros((self.h1_size, self.h2_size), dtype=np.float32)
        self.cdo_cache_h1h2_b = np.zeros((self.h2_size), dtype=np.float32)
        self.cdo_cache_h2o_w = np.zeros((self.h2_size, self.o_size), dtype=np.float32)
        self.cdo_cache_h2o_b = np.zeros((self.o_size), dtype=np.float32)

    def __divide_cache(self, divisor: int):
        self.cdo_cache_ih1_w /= divisor
        self.cdo_cache_ih1_b /= divisor
        self.cdo_cache_h1h2_w /= divisor
        self.cdo_cache_h1h2_b /= divisor
        self.cdo_cache_h2o_w /= divisor
        self.cdo_cache_h2o_b /= divisor

    def __apply_cache(self):
        self.weights_i_h1 -= self.cdo_cache_ih1_w
        self.biases_i_h1 -= self.cdo_cache_ih1_b
        self.weights_h1_h2 -= self.cdo_cache_h1h2_w
        self.biases_h1_h2 -= self.cdo_cache_h1h2_b
        self.weights_h2_o -= self.cdo_cache_h2o_w
        self.biases_h2_o -= self.cdo_cache_h2o_b

    def save(self):
        np.save("wih1", self.weights_i_h1)
        np.save("bih1", self.biases_i_h1)
        np.save("wh1h2", self.weights_h1_h2)
        np.save("bh1h2", self.biases_h1_h2)
        np.save("wh2o", self.weights_h2_o)
        np.save("bh2o", self.biases_h2_o)

    def load(self):
        self.weights_i_h1 = np.load("wih1.npy")
        self.biases_i_h1 = np.save("bih1.npy")
        self.weights_h1_h2 = np.save("wh1h2.npy")
        self.biases_h1_h2 = np.save("bh1h2.npy")
        self.weights_h2_o = np.save("wh2o.npy")
        self.biases_h2_o = np.save("bh2o.npy")

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, inputs, answers, iterations):

        count = inputs.shape[0]
        expected = np.zeros((count, 10), dtype=np.float32)
        for i in range(count):
            expected[i][answers[i]] = 1.0

        for iteration in range(iterations):
            deb = f"Running training session... (id: {iteration+1}/{iterations}) progress: "
            print(f"{deb}0%", end="\r")
            batch = 1000

            avg_cost = 0
            for i in range(0, count, batch):
                avg_diff = np.zeros(expected[i].shape)
                for j in range(batch):
                    idx = i + j
                    self.i_state = inputs[idx]
                    self.__think()
                    self.__back_propagation(expected[idx], batch)
                    avg_diff += self.o_state - expected[idx]
                avg_cost += np.sum(np.power(avg_diff / batch, 2))
                # self.__divide_cache(batch)
                self.__apply_cache()
                self.__clear_cache()
                print(f"{deb}{int(i/count*100*100)/100}%", end="\r")

            avg_cost /= count / batch
            print(f"{deb}100% done!")
            print(f"Cost: {avg_cost}")

    def __back_propagation(self, expected, divisor: int):
        # "cdo" stands for cost derivative over ...
        # "s" stands for state, "w" for weight, "b" for bias and "r" for raw
        # also cdo_?_b = cdo_?_r so it will be skipped
        cdo_o_s = 2 * (self.o_state - expected)
        # because cost = (self.o_state - expected)^2
        # and now derivatives for h2:
        cdo_o_r = cdo_o_s * self.__sigmoid_derivative(self.o_state_raw)
        # because o_state = sigmoid(o_raw)
        cdo_h2o_w = np.array([cdo_o_r * self.h2_state[i] for i in range(self.h2_size)])
        cdo_h2_s = np.matmul(self.weights_h2_o, cdo_o_r)
        # because o_raw = h2o_weight * h2_state + h2_bias

        # using the same principal:
        # for h1:
        cdo_h2_r = cdo_h2_s * self.__sigmoid_derivative(self.h2_state_raw)
        cdo_h1h2_w = np.array(
            [cdo_h2_r * self.h1_state[i] for i in range(self.h1_size)]
        )
        cdo_h1_s = np.matmul(self.weights_h1_h2, cdo_h2_r)

        # and for i:
        cdo_h1_r = cdo_h1_s * self.__sigmoid_derivative(self.h1_state_raw)
        cdo_ih1_w = np.array([cdo_h1_r * self.i_state[i] for i in range(self.i_size)])

        # updating cache
        self.cdo_cache_ih1_w += cdo_ih1_w / divisor
        self.cdo_cache_ih1_b += cdo_h1_r / divisor
        self.cdo_cache_h1h2_w += cdo_h1h2_w / divisor
        self.cdo_cache_h1h2_b += cdo_h2_r / divisor
        self.cdo_cache_h2o_w += cdo_h2o_w / divisor
        self.cdo_cache_h2o_b += cdo_o_r / divisor

    def __think(self):
        self.h1_state_raw = (
            np.matmul(self.i_state, self.weights_i_h1) + self.biases_i_h1
        )
        self.h1_state = self.__sigmoid(self.h1_state_raw)
        self.h2_state_raw = (
            np.matmul(self.h1_state, self.weights_h1_h2) + self.biases_h1_h2
        )
        self.h2_state = self.__sigmoid(self.h2_state_raw)
        self.o_state_raw = (
            np.matmul(self.h2_state, self.weights_h2_o) + self.biases_h2_o
        )
        self.o_state = self.__sigmoid(self.o_state_raw)

    def answer(self, input):
        self.i_state = input
        self.__think()
        return np.argmax(self.o_state)


if __name__ == "__main__":
    count, row, col, images, labels = data.get_train_data()
    recognizer = DigitRecognizer(row * col)
    print(recognizer.answer(images[0]))
    recognizer.train(images, labels, 1)
    recognizer.save()
    for i in range(10):
        print(recognizer.answer(images[i]), labels[i])
