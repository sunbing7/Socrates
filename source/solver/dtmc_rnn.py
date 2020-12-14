import numpy as np
import ast
import math

from autograd import grad
from utils import *
import time
import ast
from scipy import spatial
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import pyswarms as ps

import gensim

class DTMCImpl_rnn():
    def __init__(self):
        self.model = None
        self.assertion = None
        self.display = None
        self.s = []  # states
        self.s_idx = []  # index of states
        self.s0 = 0
        self.delta = 0.05
        self.error = 0.005
        self.n_ij = []
        self.n_i = []
        self.A = []
        self.m = 0
        self.offset = []
        self.bitshift = []
        self.num_of_path = 0
        #self.gen_path = '../debug/gen_path.txt'  # debug purpose
        self.step = 50  # get_new_sample step
        self.sensitive = []   #sesitive vectors
        self.sens_group = 1 # grouping of senstivie vectors
        self.feature = []  # additional feature to analyze
        self.intermediate_layer = []  # intermediate layer to analyze
        self.neurons = []  # neuron index at intermediate layer to analyze
        self.neuron = None  # current neuron index at intermediate layer analyzing
        self.under_analyze = []
        self.final = []  # final state
        self.timeout = 20  # timeout value set to 20min
        self.starttime = time.time()  # start time
        self.label_diff = 0
        self.criteria = 0.1
        self.sens_analysis = False  # sensitivity analysis
        self.dbgmsg = True
        self.type = 'nlp'
        self.data_path = None   #path to training data
        self.data_len = 0   #len of training data
        self.embedding_index = None #embedding index
        self.nlp_model = None   #embedding model
        self.data_used = 0
        self.prism_model_path = '../benchmark/rnn_fairness/nnet/jigsaw_lstm_6/prism_model/model.pm'
        #self.reuse_sample = False
        #self.gen_f = None   # file handler of gen.txt
        #self.gen_f_w = None  # file handler of gen.txt
        #self.generated_samples = 0  #number of samples generated and stored in gen.txt file
        #self.gen_offset = 0 # pos of gen.txt being read
        self.repair = False
        self.repair_neuron = None
        self.repair_layer = None
        self.repair_w = None
        self.most_sens_cell = None

    def __generate_x(self):

        x, x_w = self.__get_x_random()

        # select feature randomly from x0
        word = ' '
        vec = np.zeros(50)
        while word == ' ' or np.all((vec == 0)):
            index = np.random.randint(100)
            word = x_w[index]
            vec = x[50 * index : 50 * (index + 1)]

        synonym = self.nlp_model.most_similar(positive=[word], topn=5)

        index2 = np.random.randint(5)
        s_vec = self.nlp_model.wv[synonym[index2][0]]
        x[50 * index : 50 * (index + 1)] = s_vec

        x_w[index] = synonym[index2][0]

        '''
        if self.reuse_sample == False:
            self.gen_f.write(str(x.tolist()) + '\n')
            self.gen_f_w.write(str(x_w.tolist()) + '\n')
            self.generated_samples = self.generated_samples + 1
        '''
        return x, x_w

    def __get_x(self):
        # select x0 randomly from training data
        index = self.data_used
        data = self.data_path + '/' + str(index) + '.txt'
        x0 = np.array(ast.literal_eval(read(data)))

        data = self.data_path + '_w/' + str(index) + '.txt'
        x0_w = np.array(ast.literal_eval(read(data)))

        return x0, x0_w

    def __get_x_random(self):
        # select x0 randomly from training data
        index = np.random.randint(self.data_len)
        data = self.data_path + '/' + str(index) + '.txt'
        x0 = np.array(ast.literal_eval(read(data)))

        data = self.data_path + '_w/' + str(index) + '.txt'
        x0_w = np.array(ast.literal_eval(read(data)))

        return x0, x0_w

    '''
    def __get_x_reuse(self):
        # select x0 randomly from training data
        index = self.data_used

        if self.gen_offset >= self.generated_samples:
            #print('not enough generated samples')
            return self.__generate_x()

        try:
            rd_line = self.gen_f.readline()
            rd_w_line = self.gen_f_w.readline()
            self.gen_offset = self.gen_offset + 1
        except:
            return self.__generate_x()

        x0 = np.array(ast.literal_eval(rd_line))
        x0_w = np.array(ast.literal_eval(rd_w_line))

        return x0, x0_w
    '''

    def solve(self, model, assertion, display=None):
        overall_starttime = time.time()
        self.model = model
        self.assertion = assertion
        self.display = display

        spec = assertion

        if 'type' in spec:
            self.type = read(spec['type'])

        # sensitive feature vector table
        if 'sensitive_path' in spec:
            sens_path = (spec['sensitive_path'])

            input = open(sens_path, 'r')
            self.sensitive = np.array(ast.literal_eval(input.readline()))
            input.close()

        if 'data_path' in spec:
            self.data_path = spec['data_path']

        if 'data_len' in spec:
            self.data_len = spec['data_len']

        # sensitive feature vector table
        if 'embedding_index' in spec:
            mbed_path = (spec['embedding_index'])

            self.nlp_model = gensim.models.KeyedVectors.load_word2vec_format(mbed_path, binary=False, limit=500000)
            #synonym = self.nlp_model.most_similar(positive=['asian'], topn=10)
            #print(synonym)

        if 'sensitive_group' in spec:
            self.sens_group = read(spec['sensitive_group'])

        if 'timeout' in spec:
            self.timeout = read(spec['timeout']) * 60

        if 'criteria' in spec:
            self.criteria = spec['criteria']

        if 'error' in spec:
            self.error = (spec['error']) / 2

        if 'confidence' in spec:
            self.delta = 1 - math.sqrt(1 - (spec['confidence']))

        preparation_time = time.time() - overall_starttime
        '''
        if self.reuse_sample == False:
            self.gen_f = open(self.data_path + '/' + 'gen.txt', 'w+')
            self.gen_f_w = open(self.data_path + '_w/' + 'gen.txt', 'w+')
        else:
            self.gen_f = open(self.data_path + '/' + 'gen.txt', 'r')
            self.gen_f_w = open(self.data_path + '_w/' + 'gen.txt', 'r')
        '''

        print('Error tolerance: {:.5f}'.format(self.error))
        print('Confidence: {:.5f}'.format(self.delta))
        print('Fairness Criteria: {}'.format(self.criteria))
        print('Timeout: {}s'.format(self.timeout))
        
        # print('Learning DTMC model...')
        self.learn_dtmc_model()
        '''
        self.gen_f.close()
        self.gen_f_w.close()
        '''
        # analyze fairness
        # print('Analyzing fairness...')
        res, is_fair, prob_diff, weight_matrix = self.analyze_fairness()
        print('\nprob diff: {}\n'.format(prob_diff))

        self.export_prism_model()

        analyze_time = time.time() - preparation_time - overall_starttime

        if is_fair:
            return

        self.sens_analysis = True

        # adjust accuracy for sensitivity analysis
        ori_delta = self.delta
        ori_error = self.error
        ori_timeout = self.timeout

        self.delta = 0.15
        self.error = 0.025
        self.timeout = 300

        # perform sensitivity analysis
        print("Perform sensitivity analysis:")

        # other features
        '''
        print('Other Feature Analysis:')
        if 'feature' in spec:
            self.feature = np.array(ast.literal_eval(read(spec['feature'])))

            print('Other features: {}'.format(self.feature))

            # learn model with other features
            # print matrix

            # calculate offset to keep each state identical
            self.calc_offset()

            # calculate bitshift for state coding
            self.calc_bitshift()

            # print('Learning DTMC model...')
            self.learn_dtmc_model()

            # analyze fairness
            self.analyze_fairness()

            self.feature = []
        '''
        # other neurons
        if 'intermediate' in spec:
            self.intermediate_layer = np.array(ast.literal_eval(read(spec['intermediate'])))

        if 'neurons' in spec:
            self.neurons = np.array(ast.literal_eval(read(spec['neurons'])))

        print('Hidden Neuron Analysis:')
        print('Intermediate layers: {}'.format(self.intermediate_layer))
        print('Intermediate neuron index: {}'.format(self.neurons))

        if len(self.neurons) != 0:
            weight_matrix = []
            for i in range(0, len(self.neurons)):
                self.starttime = time.time()
                self.neuron = self.neurons[i]
                print('\nNeuron: {}'.format(self.neuron))
                '''
                if self.reuse_sample == False:
                    self.gen_f = open(self.data_path + '/' + 'gen.txt', 'w+')
                    self.gen_f_w = open(self.data_path + '_w/' + 'gen.txt', 'w+')
                else:
                    self.gen_f = open(self.data_path + '/' + 'gen.txt', 'r')
                    self.gen_f_w = open(self.data_path + '_w/' + 'gen.txt', 'r')
                '''

                # print('Learning DTMC model...')
                self.learn_dtmc_model()
                '''
                self.gen_f.close()
                self.gen_f_w.close()
                '''
                # analyze fairness
                # print('Analyzing fairness...')
                _, _, _, weight = self.analyze_fairness()
                weight_matrix.append(weight)
            # compare sensitivity
            sens_cell = []
            sens_rank = []
            idx = 0
            for m_cell in weight_matrix:
                # now m_cell contain the transition weight matrix for current cell
                # m_cell[1] contain probability to each sensitive group
                # m_cell[2] contain probability to output
                m_ph = m_cell[1]
                m_ph.sort()

                max_ph = m_ph[len(m_ph) - 1][0] - m_ph[0][0]

                s_ho = []
                s_ho.append(m_cell[2][0][0] * max_ph)
                s_ho.append(m_cell[2][1][0] * max_ph)
                s_r = []
                sens_cell.append(s_ho)
                s_r.append(max(s_ho[0], s_ho[1]))
                s_r.append(idx)
                sens_rank.append(s_r)
                idx = idx + 1

            sens_rank.sort()

            self.debug_print('Sensitivity ranking:')
            for item in sens_rank:
                self.debug_print(item)

            self.debug_print('Sensitivity details:')
            for item in sens_cell:
                self.debug_print(item)

            self.most_sens_cell = sens_rank[-1][1]


        self.sens_analysis = False

        sensitivity_time = time.time() - analyze_time - preparation_time - overall_starttime

        # repair
        self.repair = True
        if self.repair == True:
            # repair
            print('Start reparing...')
            options = {'c1': 0.41, 'c2': 0.41, 'w': 0.8}

            optimizer = ps.single.GlobalBestPSO(n_particles=5, dimensions=1, options=options,
                                                bounds=([-1.0], [1.0]),
                                                init_pos=np.array(
                                                    [[0.0],[0.0],[0.0],[0.0],[0.0]]))

            # Perform optimization
            best_cost, best_pos = optimizer.optimize(self.pso_fitness_func, iters=20)

            # Obtain the cost history
            print(optimizer.cost_history)
            # Obtain the position history
            print(optimizer.pos_history)
            # Obtain the velocity history
            #print(optimizer.velocity_history)
            print('neuron to repair: {} at layter: {}'.format(self.repair_neuron, self.repair_layer))
            #print('best cost: {}'.format(best_cost))
            #print('best pos: {}'.format(best_pos))
        self.repair = False

        repair_time = time.time() - sensitivity_time - analyze_time - preparation_time - overall_starttime

        # change back acc req for testing
        self.delta = ori_delta
        self.error = ori_error
        self.timeout = ori_timeout

        # verify prob diff and model accuracy after repair
        r_prob_diff, r_acc = self.test_repaired_net(best_pos[0])
        print('Probability difference after repair: {}'.format(r_prob_diff))
        print('Network Accuracy after repair: {}'.format(r_acc))

        valid_time = time.time() - repair_time - sensitivity_time - analyze_time - preparation_time - overall_starttime

        # timing measurement
        print('Total execution time(s): {}'.format(time.time() - overall_starttime))
        print('Model verification time (s): {}'.format(analyze_time))
        print('Sensitivity analysis time (s): {}'.format(sensitivity_time))
        print('Repair time (s): {}'.format(repair_time))
        print('Validation time (s): {}'.format(valid_time))


    def pso_fitness_func(self, weight):

        #self.reuse_sample = True

        result = []
        for i in range (0, len(weight)):
            self.repair_neuron = self.most_sens_cell
            self.repair_layer = self.intermediate_layer[0]
            self.repair_w = weight[i][0]

            accuracy = self.net_accuracy_test(self.repair_neuron, weight[i][0], self.intermediate_layer[0])

            self.starttime = time.time()
            '''
            self.gen_f = open(self.data_path + '/' + 'gen.txt', 'r')
            self.gen_f_w = open(self.data_path + '_w/' + 'gen.txt', 'r')
            '''

            self.learn_dtmc_model()

            '''
            self.gen_f.close()
            self.gen_f_w.close()
            '''

            _, _, prob_diff, _ = self.analyze_fairness()

            _result = prob_diff + 0.3 * (1 - accuracy)

            self.debug_print('Repaired prob_diff: {}, accuracy: {}'.format(prob_diff, accuracy))

            result.append(_result)
        print(result)

        return result

    def test_repaired_net(self, weight):
        #self.reuse_sample = True
        self.repair_neuron = self.most_sens_cell
        self.repair_layer = self.intermediate_layer[0]
        self.repair_w = weight

        self.repair = True
        accuracy = self.net_accuracy_test(self.repair_neuron, weight, self.intermediate_layer[0])

        self.starttime = time.time()
        '''
        self.gen_f = open(self.data_path + '/' + 'gen.txt', 'r')
        self.gen_f_w = open(self.data_path + '_w/' + 'gen.txt', 'r')
        '''
        self.learn_dtmc_model()
        '''
        self.gen_f.close()
        self.gen_f_w.close()
        '''
        _, _, prob_diff, _ = self.analyze_fairness()
        self.repair = False
        return prob_diff, accuracy

    def net_accuracy_test(self, r_neuron=0, r_weight=0, r_layer=0):
        pathX = '../benchmark/rnn_fairness/data/jigsaw/sensitive/'
        pathY = '../benchmark/rnn_fairness/data/jigsaw/sensitive/labels.txt'

        y0s = np.array(ast.literal_eval(read(pathY)))

        l_pass = 0
        l_fail = 0

        for i in range(300):
            x0_file = pathX + 'data' + str(i) + '.txt'
            x0 = np.array(ast.literal_eval(read(x0_file)))

            #print('Data {}'.format(i))

            y, _ = self.model.apply_lstm_repair(x0, r_neuron, r_weight, r_layer)
            #y = self.model.apply(x0)
            lbl_x0 = np.argmax(y, axis=1)[0]

            # accuracy test
            if lbl_x0 == y0s[i]:
                l_pass = l_pass + 1
            else:
                l_fail = l_fail + 1
        acc = l_pass / (l_pass + l_fail)

        #self.debug_print("Accuracy of ori network: %f.\n" % (acc))

        return acc

    def pso_fitness_func_test(self, weight):
        prob_diff = 0.0
        result = []
        for i in range(0, len(weight)):
            result.append(weight[i][0])
        #print('\n {}'.format(prob_diff))

        return result


    '''
    Update model when add a new trace path
    '''

    def update_model(self, path):

        for i in range(0, len(path)):
            # link previous state to n_ij and n_i
            if i == 0:
                continue
            idx_start = self.s.index(path[i - 1])
            idx_end = self.s.index(path[i])
            self.n_ij[idx_start][idx_end] = self.n_ij[idx_start][idx_end] + 1
            self.n_i[idx_start] = self.n_i[idx_start] + 1
            # self.A[idx_start][idx_end] = self.n_ij[idx_start][idx_end] / self.n_i[idx_start]

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
        self.data_used = 0
        self.num_of_path = 0
        #self.gen_offset = 0

        # initialize states of inputs:

        # start node
        self.add_state(0, 0)

        # intermediat layer
        # states: 0: start
        # 1: output 1
        # 2: output 2
        # 3 ... self.sens_group + 3: sens groups
        # self.sens_group + 3 + 1: hidden low
        # self.sens_group + 3 + 1 + 1: hidden high

        if self.sens_analysis == True and self.neuron != None:
            self.add_state(self.sens_group + 3 + 1, 2)
            self.add_state(self.sens_group + 3 + 2, 2)

            self.add_state(1, 3)
            self.add_state(2, 3)
        else:
            self.add_state(1, 2)
            self.add_state(2, 2)

        self.final.append(1)
        self.final.append(2)

        # input layer
        for i in range(0, self.sens_group + 1):
            self.add_state(i + 3, 1)

        return

    '''
    check if more samples needed
    '''

    def is_more_sample_needed(self):
        if (self.m == 0) or (self.m == 1):
            return True

        for i in range(0, self.m):
            if self.s[i] in self.final:
                continue
            max_diff = 0.0
            for j in range(0, self.m):
                if (self.n_i[i] == 0.0):
                    continue
                diff = abs(0.5 - self.n_ij[i][j] / self.n_i[i])
                if diff > max_diff:
                    max_diff = diff
            H = 2.0 / (self.error * self.error) * math.log(2.0 / (self.delta / self.m)) * (
                        0.25 - (max_diff - 2.0 * self.error / 3.0) * (max_diff - 2.0 * self.error / 3.0))
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

        generated = self.step
        out = []

        while generated:
            if self.data_used < self.data_len:
                x, x_w = self.__get_x() # x_w: word vector
                self.data_used = self.data_used + 1
            else:
                x, x_w = self.__generate_x()
            '''
            elif self.reuse_sample == False:
                x, x_w = self.__generate_x()
            elif self.reuse_sample == True:
                x, x_w = self.__get_x_reuse()
            '''

            #y = self.model.apply(x)
            #y = np.argmax(y, axis=1)[0]

            # intermediat layer

            if self.repair == False:
                y, cell = self.model.apply_lstm_inter(x, self.neuron)
            else:
                y, cell = self.model.apply_lstm_repair(x, self.repair_neuron, self.repair_w, self.repair_layer)
            # now cell contain a sequence of hidden cell values of length equal to number of timesteps
            y = np.argmax(y, axis=1)[0]

            path = [self.s0]

            # input feature under analysis
            # TODO: support only one feature and one sensitive
            to_add = 3
            for i in range (0, len(self.sensitive)):
                if (self.sensitive[i] in x_w):
                    to_add = i + 4

            path.append(to_add)

            # add intermediate
            if self.sens_analysis == True and (self.neuron) != None:
                to_add = cell
                #path.append(to_add)
                path = path + to_add

            path.append(y + 1)

            self.num_of_path = self.num_of_path + 1
            out.append(path)
            generated = generated - 1

        return out

    '''
    learn dtmc model based on given network
    '''
    '''
    def learn_dtmc_model(self):
        #file = open(self.gen_path, 'w+')
        self.init_model()
        while (self.is_more_sample_needed() == True):
            path = self.get_new_sample()
            for i in range(0, self.step):
                self.update_model(path[i])
                #for item in path[i]:
                #    file.write("%f\t" % item)
                #file.write("\n")

        self.finalize_model()

        print('Number of traces generated: {}'.format(self.num_of_path))
        print('Number of states: {}'.format(self.m))
        #file.close()
        return
    '''

    '''
    learn dtmc model based on given network; with k measn clustering on hidden neuronß
    '''

    def learn_dtmc_model(self):
        #file = open(self.gen_path, 'w+')
        self.init_model()
        path_gen = []

        cluster_label = np.array([])

        kmeans = MiniBatchKMeans(n_clusters=2, init='k-means++', batch_size=self.step)
        #kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)
        if self.sens_analysis == True and (self.neuron) != None:
            self.debug_print('Debug: Cluster centers at iterations:')

        while (self.is_more_sample_needed() == True):
            new_path = self.get_new_sample()
            # new_path: s0, input, hidden0, hidden1 ... hiddent, output
            # number of hidden states = len(new_path) - 3
            num_hidden_state = len(new_path[0]) - 3
            cluster_label = []
            if self.sens_analysis == True and (self.neuron) != None:
                path_gen = path_gen + new_path
                new_path_array = np.array(new_path)

                for h in range (0, num_hidden_state):
                    hidden_state = (new_path_array[:, h + 2]).reshape(-1, 1)
                    kmeans = kmeans.partial_fit(hidden_state)
                    cluster_label.append(kmeans.labels_)

                #hidden_state = (new_path_array[:,2]).reshape(-1, 1)
                #kmeans = kmeans.partial_fit(hidden_state)
                #cluster_label = kmeans.labels_
            #if (self.neuron) != None:
            #    self.debug_print('{}: {}, {}'.format(self.num_of_path, kmeans.cluster_centers_[0][0], kmeans.cluster_centers_[1][0]))

            for i in range(0, self.step):
                path = new_path[i]
                if self.sens_analysis == True and (self.neuron) != None:
                    for h in range(0, num_hidden_state):
                        path[2 + h] = cluster_label[h][i] + self.sens_group + 4
                self.update_model(path)
                # for item in path[i]:
                #    file.write("%f\t" % item)
                # file.write("\n")

        #apply kmeans
        #hidden_states = np.array(path_gen)[:, 2]
        #kmeans = kmeans.fit(hidden_states.reshape(-1, 1))

        # now we have clusters
        #cluster_label = kmeans.labels_

        # print cluster center
        if self.sens_analysis == True and (self.neuron) != None:
            print('Hidden cell cluster centers: \n{}, {}'.format(kmeans.cluster_centers_[0][0], kmeans.cluster_centers_[1][0]))

        self.finalize_model()

        print('Number of traces generated: {}'.format(self.num_of_path))
        print('Number of states: {}'.format(self.m))
        #file.close()
        return


    def finalize_model(self):
        for i in range(0, self.m):
            for j in range (0, self.m):
                if self.n_i[i] == 0.0:
                    self.A[i][j] = 1.0 / self.m
                else:
                    self.A[i][j] = self.n_ij[i][j] / self.n_i[i]

        return

    def analyze_fairness(self):
        hidden_num = 0
        if self.sens_analysis == True and self.neuron != None:
            hidden_num = 1
        # generate weight matrix
        weight = []
        from_symbol = []
        to_symbol = []
        for i in range(1, (1 + hidden_num) + 2):
            w = []
            _from_symbol = []
            _to_symbol = []
            for idx_row in range(0, self.m):
                if self.s_idx[idx_row] == i - 1:
                    _from_symbol.append(idx_row)
                    w_row = []
                    for j in range(0, self.m):
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
        res.append(weight[(1 + hidden_num)])
        self.detail_print("\nProbabilities:")
        # print('Sensitive feature {}:'.format(self.sensitive[-1]))

        # print index
        self.detail_print("transition from:")
        for item in from_symbol[(1 + hidden_num)]:
            self.detail_print("0x%016X" % int(self.s[item]))
        self.detail_print("transition to:")
        for item in to_symbol[(1 + hidden_num)]:
            self.detail_print("0x%016X" % int(self.s[item]))

        # print transformation matrix
        self.detail_print("\n")
        for item in weight[(1 + hidden_num)]:
            self.detail_print(item)
        # print(np.matrix(weight[len(self.sensitive)]))
        self.detail_print("\n")
        for i in range(0, (1 + hidden_num)):
            result = np.matmul(weight[(1 + hidden_num) - i - 1], res[i])
            res.append(result)
            '''
            if i != (len(self.sensitive) + hidden_num) - 1:
                print('Sensitive feature {}:'.format(self.sensitive[(len(self.sensitive) + hidden_num) - i - 2]))
            else:
                print("Overal probabilities:")
            '''
            # print index
            self.detail_print("transition from:")
            for item in from_symbol[(1+ hidden_num) - i - 1]:
                self.detail_print("0x%016X" % int(self.s[item]))
            self.detail_print("transition to:")
            for item in to_symbol[(1 + hidden_num)]:
                self.detail_print("0x%016X" % int(self.s[item]))

            self.detail_print("\n")
            for item in np.matrix(result):
                self.detail_print(item)

            # print(np.matrix(result))
            self.detail_print("\n")

        # print bitshift
        #self.detail_print(
        #    "State coding bitshift (0: start, 1: sensitive input feature, 2: other input feature, 3+: intermediate state, last: output):")
        #for item in self.bitshift:
        #    self.detail_print(item)

        # check against criteria
        weight_to_check = weight[(1 + hidden_num)]
        # TODO: to handle more than 2 labels

        weight_to_check.sort()

        for non_zero_i in range (0, len(weight_to_check)):
            if weight_to_check[non_zero_i][0] == 0.0:
                continue
            else:
                break

        prob_diff = weight_to_check[len(weight_to_check) - 1][0] - weight_to_check[non_zero_i][0]

        fairness_result = 1
        if self.sens_analysis == False:
            if prob_diff > self.criteria:
                fairness_result = 0
                print("Failed accurcay criteria!")
            else:
                fairness_result = 1
                print("Passed accurcay criteria!")

            print('Probability difference: {:.4f}\n'.format(prob_diff))

            print("Total execution time: %fs\n" % (time.time() - self.starttime))

        # accuracy_diff = self.label_diff / self.num_of_path
        # self.debug_print("Total label difference: %f\n" % (accuracy_diff))

        self.debug_print("Debug message:")

        for i in range(0, len(weight)):
            self.debug_print("weight: %d" % i)
            for item in weight[i]:
                self.debug_print(item)

        self.debug_print("\n")

        for i in range (0, len(self.n_i)):
            self.debug_print('n_{}: {}'.format(i, self.n_i[i]))

        return res, fairness_result, prob_diff, weight

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
        if self.sens_analysis or self.dbgmsg:
            print(x)

    def debug_print(self, x):
        if self.dbgmsg:
            print(x)

    def find_closest_embeddings(word_vec, embeddings_index):
        return sorted(embeddings_index.keys(),
                      key=lambda word: spatial.distance.euclidean(embeddings_index[word], word_vec))


    def export_prism_model(self):
        fout = open(self.prism_model_path, 'w')

        fout.write('dtmc\n\n')

        # start module
        fout.write('module model_learned\n')

        # states
        to_write = 's:[' + str(min(self.s)) + '..' + str(max(self.s)) + '] init ' + str(min(self.s)) + ';\n'
        fout.write(to_write)

        # state transitions
        for i in range (0, len(self.s)):
            to_write = '[]s=' + str(self.s[i]) + ' -> '
            first = True
            is_empty = True
            for j in range (0, len(self.A[i])):
                if self.A[i][j] == 0:
                    continue
                is_empty = False
                if first == True:
                    to_write = to_write + str(self.A[i][j]) + ':(s\'=' + str(self.s[j]) + ')'
                    first = False
                else:
                    to_write = to_write + ' + ' + str(self.A[i][j]) + ':(s\'=' + str(self.s[j]) + ')'
            to_write = to_write + ';\n'
            if is_empty == False:
                fout.write(to_write)

        # end module
        fout.write('\nendmodule')
        fout.flush()


        fout.close()
        return