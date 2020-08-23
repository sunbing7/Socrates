import numpy as np
import ast
import math

from autograd import grad
from utils import *
import time


class DTMCImpl():
    def __init__(self):
        self.model = None
        self.assertion = None
        self.display = None
        self.s = [] #states
        self.s_idx = [] #index of states
        self.s0 = 0
        self.delta = 0.001
        self.error = 0.01
        self.n_ij = []
        self.n_i = []
        self.A = []
        self.m = 0
        self.offset = []
        self.bitshift = []
        self.num_of_path = 0
        self.gen_path = '../debug/gen_path.txt'   #debug purpose
        self.step = 50  #get_new_sample step
        self.sensitive = []
        self.feature = [] # additional feature to analyze
        self.intermediate_layer = []    # intermediate layer to analyze
        self.neurons = []    # neuron index at intermediate layer to analyze
        self.neuron = []  # current neuron index at intermediate layer analyzing
        self.under_analyze = []
        self.final = [] #final state
        self.timeout = 20   #timeout value set to 20min
        self.starttime = time.time()    #start time
        self.label_diff = 0

    def __generate_x(self, shape, lower, upper):
        size = np.prod(shape)
        x = np.random.rand(size)

        x = (upper - lower) * x + lower
        out = np.array([0.0] * x.size)
        i = 0
        for x_i in x:
            x_i = round(x_i)
            out[i] = x_i
            i = i + 1
        return out

    def solve(self, model, assertion, display=None, delta=0.0001, error=0.001, step=50):
        self.model = model
        self.assertion = assertion
        self.display = display
        self.delta = delta
        self.error = error
        self.step = step

        spec = assertion

        if 'fairness' in spec:
            self.sensitive = np.array(ast.literal_eval(read(spec['fairness'])))

        if 'feature' in spec:
            self.feature = np.array(ast.literal_eval(read(spec['feature'])))

        if 'intermediate' in spec:
            self.intermediate_layer = np.array(ast.literal_eval(read(spec['intermediate'])))

        if 'neurons' in spec:
            self.neurons = np.array(ast.literal_eval(read(spec['neurons'])))

        if 'timeout' in spec:
            self.timeout = read(spec['timeout']) * 60

        # analyze sensitive feature only
        self.under_analyze = np.concatenate((self.feature, self.sensitive), 0)
        print('Sensitive features: {}'. format(self.sensitive))
        print('Other features: {}'.format(self.feature))
        print('Intermediate layers: {}'.format(self.intermediate_layer))
        print('Intermediate neuron index: {}'.format(self.neurons))
        print('Timeout: {}'.format(self.timeout))

        #array = self.aequitas_test()

        if len(self.neurons) != 0:

            for i in range (0, len(self.neurons)):
                self.neuron = self.neurons[i]
                print('\nNeuron: {}'.format(self.neuron))
                # calculate offset to keep each state identical
                self.calc_offset()

                # calculate bitshift for state coding
                self.calc_bitshift()

                #print('Learning DTMC model...')
                self.learn_dtmc_model()

                # analyze fairness
                #print('Analyzing fairness...')
                self.analyze_fairness()

                self.starttime = time.time()
        else:
            # calculate offset to keep each state identical
            self.calc_offset()

            # calculate bitshift for state coding
            self.calc_bitshift()

            # print('Learning DTMC model...')
            self.learn_dtmc_model()

            # analyze fairness
            # print('Analyzing fairness...')
            self.analyze_fairness()



    def calc_offset(self):
        lower = self.model.lower
        upper = self.model.upper

        size = upper.size
        self.offset = []
        self.offset.append(1)
        for i in range (1, size):
            self.offset.append(self.offset[i - 1] + upper[i - 1] - lower[i - 1] + 1)

    def calc_bitshift(self):
        self.bitshift = []
        total_analyze = len(self.intermediate_layer) + 4 #start + input layer (sensitive + other feature) + output layer

        for i in range (0, total_analyze):
            if i == 0:  #start
                self.bitshift.append(0)
            elif i == 1: #input sensitive
                self.bitshift.append(4)
            elif i == 2: #input other feature
                # sensitive feature range
                cal_range = self.model.upper[self.sensitive[0]] - self.model.lower[self.sensitive[0]]
                sensitive_range = int(cal_range) + 1

                # how many bits needed? how many nibbles
                nibbles = int(sensitive_range / 16 + 1) # + 1 to handle floating point result

                self.bitshift.append(nibbles * 4 + self.bitshift[i - 1])
            elif i!= total_analyze - 1:
                if len(self.feature) == 0:
                    self.bitshift.append(self.bitshift[i - 1])
                    continue

                # other feature range
                cal_range = self.model.upper[self.feature[0]] - self.model.lower[self.feature[0]]
                feature_range = int(cal_range) + 1

                # how many bits needed? how many nibbles
                nibbles = int(feature_range / 16 + 1) # + 1 to handle floating point result

                self.bitshift.append(nibbles * 4 + self.bitshift[i - 1])
            elif i == total_analyze - 1:
                self.bitshift.append(1)


    '''
    Update model when add a new trace path
    '''
    def update_model(self, path):
        if path[-1] not in self.final:
            self.final.append(path[-1])
        for i in range (0, len(path)):
            if path[i] not in self.s:
                self.s.append(path[i])
                self.s_idx.append(i)
                self.m = self.m + 1
                # add corresponding A fields
                if len(self.A) == 0:
                    self.A.append([0.0])
                else:
                    for row in self.A:
                        row.append(0.0)
                    self.A.append([0.0] * len(self.A[0]))

                # add corresponding n_ij fields
                if len(self.n_ij) == 0:
                    self.n_ij.append([0.0])
                else:
                    for row in self.n_ij:
                        row.append(0.0)
                    self.n_ij.append([0.0] * len(self.n_ij[0]))

                # add corresponding n_i fields
                self.n_i.append(0.0)

            # link previous state to n_ij and n_i
            if i == 0:
                continue
            idx_start = self.s.index(path[i - 1])
            idx_end = self.s.index(path[i])
            self.n_ij[idx_start][idx_end] = self.n_ij[idx_start][idx_end] + 1
            self.n_i[idx_start] = self.n_i[idx_start] + 1
            self.A[idx_start][idx_end] = self.n_ij[idx_start][idx_end] / self.n_i[idx_start]

        return

    def add_state(self, state, sequence):


        if state not in self.s:
            self.s.append(state)
            self.s_idx.append(sequence)
            self.m = self.m + 1
            # add corresponding A fields
            if len(self.A) == 0:
                self.A.append([0.0])
            else:
                for row in self.A:
                    row.append(0.0)
                self.A.append([0.0] * len(self.A[0]))

            # add corresponding n_ij fields
            if len(self.n_ij) == 0:
                self.n_ij.append([0.0])
            else:
                for row in self.n_ij:
                    row.append(0.0)
                self.n_ij.append([0.0] * len(self.n_ij[0]))

            # add corresponding n_i fields
            self.n_i.append(0.0)

        return

    '''
    init model
    '''
    def init_model(self):
        self.s = []
        self.s_idx = []
        self.m = 0
        self.n_ij = []
        self.n_i = []
        self.A = []
        self.num_of_path = 0
        return

    '''
    check if more samples needed
    '''
    def is_more_sample_needed(self):
        if (self.m == 0) or (self.m == 1):
            return True

        for i in range (0, self.m):
            if self.s[i] in self.final:
                continue
            max_diff = 0.0
            for j in range (0, self.m):
                if (self.n_i[i] == 0.0):
                    continue
                diff = abs(0.5 - self.n_ij[i][j] / self.n_i[i])
                if diff > max_diff:
                    max_diff = diff
            H = 2.0 / (self.error * self.error) * math.log(2.0 / (self.delta / self.m)) * (0.25 - (max_diff - 2.0 * self.error / 3.0) * (max_diff - 2.0 * self.error / 3.0))
            if self.n_i[i] < H:
                # timeout?
                if ((time.time() - self.starttime) > self.timeout):
                    print('\nTimeout! States that need more sample:')
                    needs_more = []
                    for k in range(i, self.m):
                        if self.s[k] in self.final:
                            continue
                        max_diff = 0.0
                        for p in range(0, self.m):
                            if self.n_i[k] == 0.0:
                                continue
                            diff = abs(0.5 - self.n_ij[k][p] / self.n_i[k])
                            if diff > max_diff:
                                max_diff = diff
                        H = 2.0 / (self.error * self.error) * math.log(2.0 / (self.delta / self.m)) * (
                                0.25 - (max_diff - 2.0 * self.error / 3.0) * (max_diff - 2.0 * self.error / 3.0))
                        if self.n_i[k] < H:
                            needs_more.append(self.s[k])
                            print("0x%016X: %d" % (int(self.s[k]), int(self.n_i[k])))
                    print("\n")

                    return False

                return True

        return False

    '''
    
    '''
    def get_new_sample(self):
        lower = self.model.lower
        upper = self.model.upper

        generated = self.step
        out = []

        while generated:
            x = self.__generate_x(self.model.shape, lower, upper)
            y, layer_op = self.model.apply_intermediate(x)
            y = np.argmax(y, axis=1)[0]
            #y = np.argmax(self.model.apply(x), axis=1)[0]

            #calculate label difference
            y_ori = np.argmax(self.model.apply(x), axis=1)[0]

            if y != y_ori:
                self.label_diff = self.label_diff + 1

            intermediate_result = []
            for i in range (0, len(layer_op)):
                if i in self.intermediate_layer:
                    layer_sign = np.sign(layer_op[i])

                    # code into one state: each neuron represendted by 2 bits
                    # TODO: only support positive and non-positive now
                    '''
                    #encode all neurtons in this layer
                    layer_state = 0
                    for j in range (0, len(layer_sign[0])):
                        layer_state = layer_state | (int((layer_sign[0][j] + 1)) << (2 * j))
                    '''
                    '''
                    #count number of activated neuron
                    ayer_activated = np.count_nonzero(layer_sign)
                    layer_state = layer_activated
                    '''

                    #neuron by neuron
                    layer_state = int(layer_sign[0][self.neuron] + 1)
                    self.add_state((1 << self.bitshift[3]), 2)
                    self.add_state((2 << self.bitshift[3]), 2)

                    intermediate_result.append((layer_state << self.bitshift[3]))

            path = [self.s0]

            # input feature under analysis
            # TODO: support only one feature and one sensitive
            to_add = 0
            for i in range (0, len(x)):
                if i in self.feature:
                    new = (int(x[i]) + 1) << self.bitshift[2]
                    to_add = to_add | new
                if i in self.sensitive:
                    new = (int(x[i]) + 1) << self.bitshift[1]
                    to_add = to_add | new

            path.append(to_add)

            # intermediate layer result under analysis
            # TODO: only support positive and non-positive now
            for i in range (0, len(intermediate_result)):
                path.append(intermediate_result[i])

            path.append(int(y + self.offset[-1]))
            self.num_of_path = self.num_of_path + 1
            out.append(path)
            generated = generated - 1

        return out


    '''
    learn dtmc model based on given network
    '''
    def learn_dtmc_model(self):
        file = open(self.gen_path, 'w+')
        self.init_model()
        while (self.is_more_sample_needed() == True):
            path = self.get_new_sample()
            for i in range (0, self.step):
                self.update_model(path[i])
                for item in path[i]:
                    file.write("%f\t" % item)
                file.write("\n")

        self.finalize_model()

        print('Error tolerance: {}'.format(self.error))
        print('Accuracy: {}'.format(self.delta))
        print('Number of traces generated: {} \n'.format(self.num_of_path))
        print('Number of states: {}'.format(self.m))
        file.close()
        return

    def finalize_model(self):
        for i in range(0, self.m):
            if self.n_i[i] == 0.0:
                for j in range (0, self.m):
                    self.n_ij[i][j] = 1 / self.m
        return

    def analyze_fairness(self):
        # generate weight matrix
        weight = []
        from_symbol = []
        to_symbol = []
        for i in range (1, (len(self.sensitive)  + len(self.intermediate_layer)) + 2):
            w = []
            _from_symbol = []
            _to_symbol = []
            for idx_row in range (0, self.m):
                if self.s_idx[idx_row] == i - 1:
                    _from_symbol.append(idx_row)
                    w_row = []
                    for j in range (0, self.m):
                        if self.s_idx[j] == i:
                            w_row.append(self.A[idx_row][j])
                            if j not in _to_symbol:
                                _to_symbol.append(j)
                    w.append(w_row)

            weight.append(w)
            from_symbol.append(_from_symbol)
            to_symbol.append(_to_symbol)

        # analyze independence fairness
        res = []
        res.append(weight[(len(self.sensitive) + len(self.intermediate_layer))])
        print("Probabilities: \n")
        #print('Sensitive feature {}:'.format(self.sensitive[-1]))

        # print index
        print("transition from:")
        for item in from_symbol[(len(self.sensitive) + len(self.intermediate_layer))]:
            print("0x%016X" % int(self.s[item]))
        print("transition to:")
        for item in to_symbol[(len(self.sensitive) + len(self.intermediate_layer))]:
            print("0x%016X" % int(self.s[item]))

        #print transformation matrix
        print("\n")
        for item in weight[(len(self.sensitive) + len(self.intermediate_layer))]:
            print(item)
        #print(np.matrix(weight[len(self.sensitive)]))
        print("\n")
        for i in range (0, (len(self.sensitive) + len(self.intermediate_layer))):
            result = np.matmul(weight[(len(self.sensitive) + len(self.intermediate_layer)) - i - 1], res[i])
            res.append(result)
            '''
            if i != (len(self.sensitive) + len(self.intermediate_layer)) - 1:
                print('Sensitive feature {}:'.format(self.sensitive[(len(self.sensitive) + len(self.intermediate_layer)) - i - 2]))
            else:
                print("Overal probabilities:")
            '''
            # print index
            print("transition from:")
            for item in from_symbol[(len(self.sensitive) + len(self.intermediate_layer)) - i - 1]:
                print("0x%016X" % int(self.s[item]))
            print("transition to:")
            for item in to_symbol[(len(self.sensitive) + len(self.intermediate_layer))]:
                print("0x%016X" % int(self.s[item]))

            print("\n")
            for item in np.matrix(result):
                print(item)

            #print(np.matrix(result))
            print("\n")

        #print bitshift
        print("State coding bitshift (0: start, 1: sensitive input feature, 2: other input feature, 3+: intermediate state, last: output):")
        for item in self.bitshift:
            print(item)


        print("Total execution time: %fs\n" % (time.time() - self.starttime))

        accuracy_diff = self.label_diff / self.num_of_path
        print("Total label difference: %f\n" % (accuracy_diff))

        print("debug message:")

        for i in range (0, len(weight)):
            print("weight: %d" % i)
            for item in weight[i]:
                print(item)

        return res

    def aequitas_test(self):
        num_trials = 400
        samples = 1000

        estimate_array = []
        rolling_average = 0.0

        for i in range(num_trials):
            disc_count = 0
            total_count = 0
            for j in range(samples):
                total_count = total_count + 1


                if (self.aeq_test_new_sample()):
                    disc_count = disc_count + 1

            estimate = float(disc_count) / total_count
            rolling_average = ((rolling_average * i) + estimate) / (i + 1)
            estimate_array.append(estimate)

            print(estimate, rolling_average)

        print("Total execution time: %fs\n" % (time.time() - self.starttime))
        return estimate_array


    def aeq_test_new_sample(self):
        lower = self.model.lower
        upper = self.model.upper


        x = self.__generate_x(self.model.shape, lower, upper)
        y = np.argmax(self.model.apply(x), axis=1)[0]
        x_g = x

        sensitive_feature = self.sensitive[0]
        sens_range = upper[sensitive_feature] - lower[sensitive_feature] + 1

        for val in range (int(lower[sensitive_feature]), int(upper[sensitive_feature]) + 1):
            if val != x[sensitive_feature]:
                x_g[sensitive_feature] = float(val)
                y_g = np.argmax(self.model.apply(x_g), axis=1)[0]

                if y != y_g:
                    return 1
        return 0



