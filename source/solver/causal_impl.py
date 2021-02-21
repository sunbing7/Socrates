import numpy as np
import ast
import math
from sklearn.cluster import MiniBatchKMeans

from autograd import grad
import os.path
from os import path
from utils import *
import time
import pyswarms as ps
import matplotlib.pyplot as plt

class CausalImpl():
    def __init__(self):
        self.model = None
        self.assertion = None
        self.display = None
        self.timeout = 5
        self.datapath = None    # path to accuracy test data
        self.resultpath = None  # path to output result folder
        self.datalen = 0        # len of existing data
        self.stepsize = 16      # step size for intervension
        self.do_layer = []
        self.do_neuron = []

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

    def solve(self, model, assertion, display=None):
        overall_starttime = time.time()
        self.model = model
        self.assertion = assertion
        self.display = display

        spec = assertion

        if 'timeout' in spec:
            self.timeout = read(spec['timeout']) * 60

        if 'stepsize' in spec:
            self.stepsize = spec['stepsize']

        if 'datalen' in spec:
            self.datalen = spec['datalen']

        if 'datapath' in spec:
            self.datapath = spec['datapath']

        if 'resultpath' in spec:
            self.resultpath = spec['resultpath']

        if 'do_layer' in spec:
            self.do_layer = np.array(ast.literal_eval(read(spec['do_layer'])))

        if 'do_neuron' in spec:
            self.do_neuron = np.array(ast.literal_eval(read(spec['do_neuron'])))

        print('Timeout: {}s'.format(self.timeout))

        # get ace for each hidden neuron

        if len(self.do_layer) == 0 or len(self.do_neuron) == 0:
            print('Nothing to analyze!')
            return

        plt_row = 0
        for i in range (0, len(self.do_layer)):
            if len(self.do_neuron[i]) > plt_row:
                plt_row = len(self.do_neuron[i])
        plt_col = len(self.do_layer)
        fig, ax = plt.subplots(plt_row, plt_col, figsize=(3.5*plt_col, 2.5*plt_row), sharex=False, sharey=False)
        fig.tight_layout()

        row = 0
        col = 0
        for do_layer in self.do_layer:
            row = 0
            print('Analyzing layer {}'.format(col))
            for do_neuron in self.do_neuron[col]:
                ie, min, max = self.get_ie_do_h(do_layer, do_neuron, self.stepsize, 0)

                # plot ACE
                #ax[row, col].set_title('N_' + str(do_layer) + '_' + str(do_neuron))
                ax[row, col].set_xlabel('Intervention Value(alpha)')
                ax[row, col].set_ylabel('Causal Attributions(ACE)')

                # Baseline is np.mean(expectation_do_x)
                ax[row, col].plot(np.linspace(min, max, self.stepsize), np.array(ie) - np.mean(np.array(ie)), label = str(do_layer) + '_' + str(do_neuron), color='b')
                ax[row, col].legend()

                row = row + 1
            if row == len(self.do_neuron[col]):
                for off in range(row, plt_row):
                    ax[off, col].set_axis_off()
            col = col + 1

        plt.savefig(self.resultpath + '/' + 'all' + ".png")
        plt.show()

        # timing measurement
        print('Total execution time(s): {}'.format(time.time() - overall_starttime))

    #
    # get expected value of y with hidden neuron intervention
    #
    def get_y_do_h(self, do_layer, do_neuron, do_value):
        pathX = self.datapath + '/'
        pathY = self.datapath + '/labels.txt'

        y0s = np.array(ast.literal_eval(read(pathY)))

        l_pass = 0
        l_fail = 0

        y_sum = 0.0

        for i in range(self.datalen):
            x0_file = pathX + 'data' + str(i) + '.txt'
            x0 = np.array(ast.literal_eval(read(x0_file)))

            y = self.model.apply_intervention(x0, do_layer, do_neuron, do_value)

            lbl_x0 = np.argmax(y, axis=1)[0]

            y_sum = y_sum + y

            # accuracy test
            if lbl_x0 == y0s[i]:
                l_pass = l_pass + 1
            else:
                l_fail = l_fail + 1
        acc = l_pass / (l_pass + l_fail)

        avg = y_sum / self.datalen

        #self.debug_print("Accuracy of network: %f.\n" % (acc))

        return avg, acc

    #
    # given number of steps, get expected ys for each step
    #
    def get_ie_do_h(self, do_layer, do_neuron, num_step=16, class_n=0):
        # get value range of given hidden neuron
        pathX = self.datapath + '/'
        pathY = self.datapath + '/labels.txt'

        y0s = np.array(ast.literal_eval(read(pathY)))

        hidden_max = 0.0
        hidden_min = 0.0

        for i in range(self.datalen):
            x0_file = pathX + 'data' + str(i) + '.txt'
            x0 = np.array(ast.literal_eval(read(x0_file)))

            y, hidden = self.model.apply_get_h(x0, do_layer, do_neuron)

            if i == 0:
                hidden_max = hidden
                hidden_min = hidden
            else:
                if hidden > hidden_max:
                    hidden_max = hidden
                if hidden < hidden_min:
                    hidden_min = hidden

        # now we have hidden_min and hidden_max

        # compute interventional expectation for each step
        ie = []
        if hidden_max == hidden_min:
            ie = [hidden_min] * num_step
        else:
            for h_val in np.linspace(hidden_min, hidden_max, num_step):
                y,_ = self.get_y_do_h(do_layer, do_neuron, h_val)
                ie.append(y[0][class_n])

        return ie, hidden_min, hidden_max

    def net_accuracy_test(self, r_neuron=0, r_weight=0, r_layer=0):
        pathX = self.datapath + '/'
        pathY = self.datapath + '/labels.txt'

        y0s = np.array(ast.literal_eval(read(pathY)))

        l_pass = 0
        l_fail = 0

        for i in range(self.datalen):
            x0_file = pathX + 'data' + str(i) + '.txt'
            x0 = np.array(ast.literal_eval(read(x0_file)))
            if len(r_neuron) != 0:
                y = self.model.apply_repair(x0, r_neuron, r_weight, r_layer)
            else:
                y = self.model.apply(x0)

            lbl_x0 = np.argmax(y, axis=1)[0]

            # accuracy test
            if lbl_x0 == y0s[i]:
                l_pass = l_pass + 1
            else:
                l_fail = l_fail + 1
        acc = l_pass / (l_pass + l_fail)

        #self.debug_print("Accuracy of network: %f.\n" % (acc))

        return acc

    def test_repaired_net(self, weight):

        self.repair_w = weight

        self.repair = True
        accuracy = self.net_accuracy_test(self.repair_neuron, weight, self.repair_layer)

        self.starttime = time.time()

        self.learn_dtmc_model()

        _, _, prob_diff, _ = self.analyze_fairness()
        self.repair = False
        return prob_diff, accuracy

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

        for val in range(int(lower[sensitive_feature]), int(upper[sensitive_feature]) + 1):
            if val != x[sensitive_feature]:
                x_g[sensitive_feature] = float(val)
                y_g = np.argmax(self.model.apply(x_g), axis=1)[0]

                if y != y_g:
                    return 1
        return 0

    def detail_print(self, x):
        if self.sens_analysis and self.dbgmsg:
            print(x)

    def debug_print(self, x):
        if self.dbgmsg:
            print(x)
        pass

    def d_detail_print(self, x):
        if self.dbgmsg:
            print(x)

