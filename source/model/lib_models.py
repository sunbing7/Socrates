import autograd.numpy as np
import torch

class Model:
    def __init__(self, shape, lower, upper, layers, path):
        self.shape = shape
        self.lower = lower
        self.upper = upper
        self.layers = layers

        if layers == None and path != None:
            self.ptmodel = torch.load(path)

    def __apply_ptmodel(self, x):
        x = torch.from_numpy(x).view(self.shape.tolist())

        with torch.no_grad():
            output = self.ptmodel(x)

        output = output.numpy()

        return output

    def apply(self, x):
        if self.layers == None:
            return self.__apply_ptmodel(x)

        shape_i = [1, *self.shape[1:]]
        size_i = np.prod(shape_i)

        len = int(x.size / size_i)

        for i in range(len):
            x_i = x[size_i * i : size_i * (i + 1)].reshape(shape_i)
            output = x_i
            for layer in self.layers:
                output = layer.apply(output)

        for layer in self.layers:
            layer.reset()

        return output

    def apply_intermediate(self, x):
        if self.layers == None:
            return self.__apply_ptmodel(x)

        shape_i = [1, *self.shape[1:]]
        size_i = np.prod(shape_i)

        len = int(x.size / size_i)

        # only handle single input
        if len != 1:
            return None, None

        layer_output = []

        for i in range(len):
            x_i = x[size_i * i : size_i * (i + 1)].reshape(shape_i)
            output = x_i
            for layer in self.layers:
                output = layer.apply(output)
                layer_output.append(output)

        for layer in self.layers:
            layer.reset()

        return output, layer_output