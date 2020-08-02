import numpy as np
import ast
import math

from autograd import grad
from utils import *


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
        self.num_of_path = 0
        self.gen_path = '../debug/gen_path.txt'   #debug purpose
        self.step = 50  #get_new_sample step
        self.sensitive = []
        self.under_analyze = []
        self.final = [] #final state


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

        # analyze sensitive feature only
        self.under_analyze = self.sensitive

        print('Sensitive features: {}'. format(self.sensitive))
        # calculate offset to keep each state identical
        self.calc_offset()

        #print('Learning DTMC model...')
        self.learn_dtmc_model()

        # analyze fairness
        #print('Analyzing fairness...')
        self.analyze_fairness()

        '''
        spec = assertion
        lower = model.lower
        upper = model.upper
        x0 = np.array(ast.literal_eval(read(spec['x0'])))
        y0 = np.argmax(model.apply(x0), axis=1)[0]
        if 'fairness' in spec:
            sensitive = np.array(ast.literal_eval(read(spec['fairness'])))
            for index in range(x0.size):
                if not (index in sensitive):
                    lower[index] = x0[index]
                    upper[index] = x0[index]

        x = self.__generate_x(model.shape, lower, upper)
        y = np.argmax(model.apply(x), axis=1)[0]
        '''

    def calc_offset(self):
        lower = self.model.lower
        upper = self.model.upper

        size = upper.size

        self.offset.append(256)
        for i in range (1, size):
            self.offset.append(self.offset[i - 1] + upper[i - 1] - lower[i - 1] + 1)

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

    '''
    init model
    '''
    def init_model(self):
        self.s = []
        self.m = 0
        self.n_ij = []
        self.n_i = []
        self.A = []
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
                diff = abs(0.5 - self.n_ij[i][j] / self.n_i[i])
                if diff > max_diff:
                    max_diff = diff
            H = 2.0 / (self.error * self.error) * math.log(2.0 / (self.delta / self.m)) * (0.25 - (max_diff - 2.0 * self.error / 3.0) * (max_diff - 2.0 * self.error / 3.0))
            if self.n_i[i] < H:
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
            y = np.argmax(self.model.apply(x), axis=1)[0]
            path = [self.s0]
            for i in range (0, len(x)):
                if i in self.under_analyze:
                    x[i] = x[i] + self.offset[i]
                    path.append(x[i])

            path.append((y + self.offset[-1]))
            self.num_of_path = self.num_of_path + 1
            out.append(path)
            generated = generated - 1
            '''
            x = self.__generate_x(model.shape, lower, upper)
            y = np.argmax(model.apply(x), axis=1)[0]
            for i in range (0, len(x)):
                x[i] = x[i] + self.offset[i]
            path = x.tolist()
            path.append(y)
            self.num_of_path = self.num_of_path + 1
            out.append(path)
            generated = generated - 1
            '''
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

        print('Error tolerance: {}'.format(self.error))
        print('Accuracy: {}'.format(self.delta))
        print('Number of traces generated: {} \n'.format(self.num_of_path))
        file.close()
        return



    def analyze_fairness(self):
        # generate weight matrix
        weight = []
        from_symbol = []
        to_symbol = []
        for i in range (1, len(self.under_analyze) + 2):
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
        res.append(weight[len(self.under_analyze)])
        print("Probabilities: \n")
        print('Sensitive feature {}:'.format(self.under_analyze[-1]))

        # print index
        print("transition from:")
        for item in from_symbol[len(self.under_analyze)]:
            print("%d" % self.s[item])
        print("transition to:")
        for item in to_symbol[len(self.under_analyze)]:
            print("%d" % self.s[item])

        print(np.matrix(weight[len(self.under_analyze)]))
        print("\n")
        for i in range (0, len(self.under_analyze)):
            result = np.matmul(weight[len(self.under_analyze) - i - 1], res[i])
            res.append(result)
            if i != len(self.under_analyze) - 1:
                print('Sensitive feature {}:'.format(self.under_analyze[len(self.under_analyze) - i - 2]))
            else:
                print("Overal probabilities:")

            # print index
            print("transition from:")
            for item in from_symbol[len(self.under_analyze) - i - 1]:
                print("%d" % self.s[item])
            print("transition to:")
            for item in to_symbol[len(self.under_analyze)]:
                print("%d" % self.s[item])

            print(np.matrix(result))
            print("\n")



        return res


