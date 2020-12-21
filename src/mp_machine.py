class mp_machine:

    def __init__(self, config_file = None):
        if not config_file == None:
            self.load_config_file(config_file)

    def load_config_file(self, config_file = None):
        if config_file == None:
            print "mp_machine.load_config_file: no configuration file given!"
            quit()

        # check if python module 'os' is available
        try:
            import os
        except:
            print "mp_machine.load_config_file: could not import python module 'os'"
            quit()

        # check if configuration file exists
        if not os.path.isfile(config_file):
            print "mp_machine.load_config_file: configuration file '%s' does not exist!" % (config_file)
            quit()

        # check if python module 'ConfigParser' is available
        try:
            import ConfigParser
        except:
            print "mp_machine.load_config_file: could not import python module 'ConfigParser'"
            quit()

        # read configuration file
        cfg = ConfigParser.ConfigParser()
        cfg.optionxform = str
        cfg.read(config_file)

        # check if configuration file contains section 'machine'
        cfg_sections = cfg.sections()
        if not 'machine' in cfg_sections:
            print "mp_machine.load_config_file: configuration file '%s' does not contain section 'machine'!" % (config_file)
            quit()

        # check if configuration file contains option 'machine' in section 'machine'
        try:
            self.machine = cfg.get('machine', 'machine').strip().lower()
            self.machine_name = {
                'gbrbm': 'Gauss-Bernoulli RBM',
                'rbm': 'Binary RBM'
            }[self.machine]
        except:
            self.machine = 'gbrbm'
            self.machine_name = 'Gauss-Bernoulli RBM'

        try:
            self.machine_init = \
                _str_to_dict(cfg.get('machine', 'init').replace('\n', ''))
        except:
            self.machine_init = {}

        try:
            self.train_steps = cfg.getint('machine', 'training_steps')
        except:
            self.train_steps = 0

        # create list which contains all training params
        self.train_params = []
        if self.train_steps == 0:
            try:
                param = cfg.get('machine', 'training').replace('\n', '')
            except:
                param = ''

            self.train_params.append(self._str_to_dict(param))
        else:
            for step in range(1, self.train_steps + 1):
                try:
                    param = cfg.get('machine', 'step_%i' % (step)).replace('\n', '')
                    self.train_params.append(self._str_to_dict(param))
                except:
                    pass

        # get plot configuration
        try:
            self.create_plots = \
                cfg.get('machine', 'create_plots').lower().strip() in ['true', '1', 'yes']
        except:
            self.create_plots = 1

        try:
            self.plot_points = cfg.getint('machine', 'plot_points')
        except:
            self.plot_points = 150

        # calculate total number of epochs
        # and update number of plot_points per step
        if self.create_plots:
            epochs = 0
            for params in self.train_params:
                if 'epochs' in params:
                    epochs += params['epochs']

            for params in self.train_params:
                if 'epochs' in params:
                    plot_points = self.plot_points * params['epochs'] / epochs
                    params['plot_points'] = plot_points

    def train(self, obj_network = None, obj_dataset = None):

        # check if python module 'time' is available
        try:
            import time
        except:
            print "mp_machine.train: could not import python module 'time'"
            quit()

        # check if python module 'numpy' is available
        try:
            import numpy as np
        except:
            print "mp_machine.train: could not import python module 'numpy'"
            quit()

        # check if python module 'src.gbrbm' is available
        try:
            from src.mp_gbrbm import gbrbm
        except:
            print "mp_machine.train: could not import python module 'src.gbrgm'"
            quit()

        # define lambda function for machine
        machine = {
            'gbrbm': lambda: gbrbm(
                adjacency_matrix = obj_network.A,
                create_plots = self.create_plots,
                **self.machine_init),
            'rbm': lambda: rbm(
                adjacency_matrix = obj_network.A,
                create_plots = self.create_plots,
                **self.machine_init)
        }[self.machine]()

        print 'starting training on %s' % \
          (time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))

        start = time.clock()
        for params in self.train_params:
            machine.train(data = obj_dataset.data, **params)
        stop = time.clock()

        print "total time: %.2fs" % (stop - start)

        # update data
        self.results = machine.results
        self.plot = machine.plot

        # update network
        if np.max(np.abs(machine.W)) > 0:
            obj_network.W = \
                np.abs(machine.W) / np.max(np.abs(machine.W))
        else:
            obj_network.W = machine.W

        obj_network.update_graph()

    def save(self, path = ''):
        if path == '':
            print "mp_machine.save: no save path was given!"
            quit()

        print "saving parameters to %s" % (path)

        # check if python module 'numpy' is available
        try:
            import numpy as np
        except:
            print "mp_machine.save: could not import python module 'numpy'"
            quit()

        for key, val in self.results.iteritems():
            np.savetxt('%s/%s.csv' % (path, key), val, delimiter = ';')

    # TODO: load

    def save_plots(self, path = ''):
        if path == '':
            print "mp_machine.save_plots: no save path was given"
            quit()

        print "saving plots to %s" % (path)
        try:
            import matplotlib.pyplot as plt
        except:
            print "mp_machine.save_plots: could not import python module 'pyplot'"
            quit()

        for key, val in self.plot.items():
            if key == 'x' or key[0:1] == '_':
                continue

            file_plot = '%s/%s.pdf' % (path, key.lower())

            # get labels
            if '_x' in self.plot:
                xlabel = self.plot['_x']
            else:
                xlabel = 'Epochs'

            ylabel = key

            if '_' + key in self.plot:
                description = self.plot['_' + key] + '\n\n'

            plt.figure()
            plt.plot(self.plot['x'], val, 'b,')
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.savefig(file_plot)

    #
    # elementar functions 
    #
    
    def _str_to_dict(self, str = '', delim = ','):
        dict = {}
        for item in str.split(delim):
            key, val = item.split('=')
            key = key.strip().lower()
            val = val.strip()
            try:
                dict[key] = eval(val)
            except:
                pass
        return dict
    
    def _str_to_list(self, str = '', delim = ','):
        list = []
        for item in str.split(delim):
            list.append(item.strip())
        return list