import numpy as np
import time as time
import ConfigParser
from src.mp_gbrbm import *
from src.mp_config import *

class mp_analyse:

    def __init__(self, config_file = ''):
        self.config_file = config_file

        cfg = ConfigParser.ConfigParser()
        cfg.read(config_file)

        try:
            self.machine = cfg.get('analyse', 'machine').strip().lower()
            self.machine_name = {
                'gbrbm': 'Gauss-Bernoulli RBM',
                'rbm': 'Binary RBM'
            }[self.machine]
        except:
            self.machine = 'gbrbm'
            self.machine_name = 'Gauss-Bernoulli RBM'

        try:
            self.machine_init = \
                _strtodict(cfg.get('analyse', 'init').replace('\n', ''))
        except:
            self.machine_init = {}

        try:
            self.train_steps = cfg.getint('analyse', 'training_steps')
        except:
            self.train_steps = 0

        # create list which contains all training params
        self.train_params = []
        if self.train_steps == 0:
            try:
                param = cfg.get('analyse', 'training').replace('\n', '')
            except:
                param = ''

            self.train_params.append(self._strtodict(param))
        else:
            for step in range(1, self.train_steps + 1):
                try:
                    param = cfg.get('analyse', 'step_%i' % (step)).replace('\n', '')
                except:
                    param = ''

                self.train_params.append(self._strtodict(param))

        # get plot configuration
        try:
            self.create_plots = cfg.get('plot', 'create_plots').lower() in ['true', '1', 'yes']
        except:
            self.create_plots = 1

        try:
            self.plot_points = cfg.get('plot', 'plot_points')
        except:
            self.plot_points = 150

    def train(self, obj_network = None, obj_dataset = None):
        print "running analyse: '%s'" % (self.config_file)
        num_visible = obj_network.adjacency.shape[0]
        num_hidden = obj_network.adjacency.shape[1]
        print '%s (%s, %s)' % (self.machine_name, num_visible, num_hidden)

        machine = {
            'gbrbm': lambda: gbrbm(
                adjacency_matrix = obj_network.adjacency,
                debug = self.create_plots,
                **self.machine_init),
            'rbm': lambda: rbm(
                adjacency_matrix = obj_network.adjacency,
                debug = self.create_plots,
                **self.machine_init)
        }[self.machine]()

        start = time.clock()
        for params in self.train_params:
            machine.train(data = obj_dataset.data, **params)
        stop = time.clock()
        print "total time: %.2fs" % (stop - start)

        self.results = machine.results
        self.plot = machine.plot

    def saveresults(data, path):
        return 1

#        i = 1
#        while os.path.exists('%s/report_%s/' % (path_reports, i)):
#            i += 1
#
#        path_report = '%s/report_%s/' % (path_reports, i)
#        os.makedirs(path_report)
#
#        # save calculated data
#        print "saving results in '%s'" % (path_report)
#        np.savetxt('%s/weights.csv' % (path_report), machine.W, delimiter = ';')
#        np.savetxt('%s/vbias.csv' % (path_report), machine.v_bias, delimiter = ';')
#        np.savetxt('%s/hbias.csv' % (path_report), machine.h_bias, delimiter = ';')
#        np.savetxt('%s/sdev.csv' % (path_report), np.sqrt(np.exp(machine.v_lvar)), delimiter = ';')


    def _strtodict(self, str=''):
        dict = {}
        for item in str.split(','):
            key, val = item.split('=')
            dict[key.strip().lower()] = eval(val)
        return dict

#        try:
#            file_adjacency = cfg.get('network', 'adjacency_matrix')
#            self.adjacency = np.loadtxt(file_adjacency, delimiter = ',', dtype = bool)
#        except:
#            gene_S = cfg.get('network', 's').replace(' ', '').split(',')
#            gene_TF = cfg.get('network', 'tf').replace(' ', '').split(',')
#            gene_E = cfg.get('network', 'e').replace(' ', '').split(',')
#
#            # create list with bindings
#            all_bindings = []
#            for s in gene_S:
#                try:
#                    s_tf_bindings = cfg.get('s-tf binding', s).replace(' ', '').split(',')
#                    for tf in s_tf_bindings:
#                        all_bindings.append('%s,%s' % (s, tf))
#                except:
#                    pass
#
#            for e in gene_E:
#                try:
#                    e_tf_bindings = cfg.get('e-tf binding', e).replace(' ', '').split(',')
#                    for tf in e_tf_bindings:
#                        all_bindings.append('%s,%s' % (e, tf))
#                except:
#                    pass
#
#            # create adjacency matrix
#            list_visible = gene_S + gene_E
#            list_hidden = gene_TF
#            adjacency = np.empty([len(list_visible), len(list_hidden)], dtype=bool)
#            for i, v in enumerate(list_visible):
#                for j, h in enumerate(list_hidden):
#                    adjacency[i, j] = all_bindings.count('%s,%s' % (v, h)) > 0
#
#            self.adjacency = adjacency
