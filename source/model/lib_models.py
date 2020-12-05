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
            j = 0
            for layer in self.layers:
                output = layer.apply(output)
                #sunbing
                #if j == 2:
                #    output[0][20] = 0.0*output[0][20]
                #if j == 0:
                #    output[0][15] =  0.0*output[0][15]
                    #output[0][13] =  0.03*output[0][13]
                #layer_output.append(output)
                j = j + 1

        for layer in self.layers:
            layer.reset()

        return output, layer_output


    def apply_lstm_inter(self, x):
        if self.layers == None:
            return self.__apply_ptmodel(x)

        shape_i = [1, *self.shape[1:]]
        size_i = np.prod(shape_i)

        len = int(x.size / size_i)

        hidden_state = [] # should contain xlen elements

        #lstm_cell_mean = 0

        for i in range(len):
            x_i = x[size_i * i : size_i * (i + 1)].reshape(shape_i)
            output = x_i

            layer_index = 0
            for layer in self.layers:
                output = layer.apply(output)
                if layer_index == 0:
                    # lstm layer
                    # test first cell: output[0][0]
                    #lstm_cell_mean = lstm_cell_mean + output[0][0]
                    #output[0][0] = 1.397 * output[0][0];
                    hidden_state.append(output[0][0])

                layer_index = layer_index + 1

        #lstm_cell_mean = lstm_cell_mean / len

        for layer in self.layers:
            layer.reset()

        return output, hidden_state


    def apply_lstm_repair(self, x, layer_r=None, weight=None):
        if self.layers == None:
            return self.__apply_ptmodel(x)

        shape_i = [1, *self.shape[1:]]
        size_i = np.prod(shape_i)

        len = int(x.size / size_i)

        hidden_state = [] # should contain xlen elements

        #lstm_cell_mean = 0

        for i in range(len):
            x_i = x[size_i * i : size_i * (i + 1)].reshape(shape_i)
            output = x_i

            layer_index = 0
            for layer in self.layers:
                output = layer.apply(output)
                if layer_index == layer_r:
                    # lstm layer
                    # test first cell: output[0][0]
                    #lstm_cell_mean = lstm_cell_mean + output[0][0]
                    output[0][layer_r] = (weight + 1) * output[0][layer_r];
                    hidden_state.append(output[0][layer_r])

                layer_index = layer_index + 1

        #lstm_cell_mean = lstm_cell_mean / len

        for layer in self.layers:
            layer.reset()

        return output, hidden_state

    # mean of all timestep of one lstm cell
    '''
    def apply_lstm_inter(self, x):
        if self.layers == None:
            return self.__apply_ptmodel(x)

        shape_i = [1, *self.shape[1:]]
        size_i = np.prod(shape_i)

        len = int(x.size / size_i)

        lstm_cell_mean = 0

        for i in range(len):
            x_i = x[size_i * i : size_i * (i + 1)].reshape(shape_i)
            output = x_i

            layer_index = 0
            for layer in self.layers:
                output = layer.apply(output)
                if layer_index == 0:
                    # lstm layer
                    # test first cell
                    lstm_cell_mean = lstm_cell_mean + output[0][0]
                    output[0][0] = 0.2710
                layer_index = layer_index + 1

        lstm_cell_mean = lstm_cell_mean / len

        for layer in self.layers:
            layer.reset()

        return output, lstm_cell_mean
    '''