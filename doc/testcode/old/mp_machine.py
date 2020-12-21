class mp_machine:

    def __init__(self, config_file = None):
        # initialize logger
        import logging
        self.logger = logging.getLogger('metapath')
        
        # load config
        if not config_file == None:
            self.load_config_file(config_file)

    def load_config_file(self, config_file = None):
        if config_file == None:
            self.logger.error("no configuration file given!")
            quit()

        # check if configuration file exists
        import os
        if not os.path.isfile(config_file):
            self.logger.error("configuration file '%s' does not exist!" % (config_file))
            quit()

        # check if python module 'ConfigParser' is available
        try:
            import ConfigParser
        except:
            self.logger.error("could not import python module 'ConfigParser'")
            quit()

        # read configuration file
        cfg = ConfigParser.ConfigParser()
        cfg.optionxform = str
        cfg.read(config_file)

        # check if configuration file contains section 'machine'
        if not 'machine' in cfg.sections():
            self.logger.error("configuration file '%s' does not contain section 'machine'!" % (config_file))
            quit()

        # check if configuration file contains option 'machine' in section 'machine'
        if 'machine' in cfg.options('machine'):
            self.machine = cfg.get('machine', 'machine').strip().lower()
            self.machine_name = {
                'gbrbm': 'Gauss-Bernoulli RBM',
                'rbm': 'Binary RBM'
            }[self.machine]
        else:
            self.machine = 'gbrbm'
            self.machine_name = 'Gauss-Bernoulli RBM'

        if 'init' in cfg.options('machine'):
            self.machine_init = \
                self._str_to_dict(cfg.get('machine', 'init').replace('\n', ''))
        else:
            self.machine_init = {}

        if 'training_stages' in cfg.options('machine'):
            self.training_stages = cfg.getint('machine', 'training_stages')
        else:
            self.training_stages = 0

        if 'training_iterations' in cfg.options('machine'):
            self.training_iterations = cfg.getint('machine', 'training_iterations')
        else:
            self.training_iterations = 1

        # create list with training parameters
        self.training_parameters = []
        if self.training_stages == 0:
            try:
                param = cfg.get('machine', 'training').replace('\n', '')
            except:
                param = ''

            self.training_parameters.append(self._str_to_dict(param))
        else:
            for stage in range(1, self.training_stages + 1):
                try:
                    param = cfg.get('machine', 'training_stage_%i' % (stage)).replace('\n', '')
                    self.training_parameters.append(self._str_to_dict(param))
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

        # calculate total number of updates
        # and update number of plot_points per stage
        if self.create_plots:
            updates = 0
            for params in self.training_parameters:
                if 'updates' in params:
                    updates += params['updates']

            for params in self.training_parameters:
                if 'updates' in params:
                    plot_points = self.plot_points * params['updates'] / updates
                    params['plot_points'] = plot_points

    def init_model(self, obj_network = None, obj_dataset = None):
        # initialize model
        if self.machine == 'gbrbm':
            try:
                from src.mp_gbrbm import gbrbm
            except:
                self.logger.critical("could not import python module 'src.gbrbm'")
                quit()
                
            self.model = gbrbm(
                adjacency_matrix = obj_network.A,
                create_plots = self.create_plots,
                data = obj_dataset.data,
                **self.machine_init)
        elif self.machine == 'rbm':
            try:
                from src.mp_rbm import rbm
            except:
                self.logger.critical("could not import python module 'src.rbm'")
                quit()

            self.model = rbm(
                adjacency_matrix = obj_network.A,
                create_plots = self.create_plots,
                data = obj_dataset.data,
                **self.machine_init)

    def train(self, obj_network = None, obj_dataset = None):
        # check if python module 'numpy' is available
        try:
            import numpy as np
        except:
            self.logger.error("could not import python module 'numpy'")
            quit()

        self.init_model(obj_network, obj_dataset)

        import time
        self.logger.info("starting time: %s" % \
            (time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())))
        start = time.clock()

        # iterate trainings and save parameters to list
        self.training = []
        for training in range(0, self.training_iterations):
            if training == 0:
                training_start = time.clock()
            else:
                self.model.reset()

            if len(self.training_parameters) > 1:
                for i, params in enumerate(self.training_parameters):
                    if 'name' in params:
                        self.logger.info("training stage %i: '%s'" % (i + 1, params['name']))
                    else:
                        self.logger.info("trining stage %i")
                    self.model.train(data = obj_dataset.data, **params)
            else:
                params = self.training_parameters[0]
                self.model.train(data = obj_dataset.data, **params)
            
            # append params of current traing to list
            self.training.extend([self.model.params.get()])
                
            if training == 0:
                training_stop = time.clock()
                self.logger.info("estimated time: %.2fs" % \
                    ((training_stop - training_start) * self.training_iterations))
            
        stop = time.clock()
        self.logger.info("total time: %.2fs" % (stop - start))

##        quit()
##
##        np.savetxt('v_lvar',
##            X =self.proc['v_lvar'],
##            fmt = '%2.8f', delimiter='\t', newline='\n')
##        np.savetxt('v_expect.txt',
##            X = self.proc['v_expect'],
##            fmt = '%2.8f', delimiter='\t', newline='\n')
##        np.savetxt('h_expect.txt',
##            X =self.proc['h_expect'],
##            fmt = '%2.8f', delimiter='\t', newline='\n')
##        np.savetxt('W',
##            X =self.proc['W'],
##            fmt = '%2.8f', delimiter='\t', newline='\n')

        self.plot = machine.plot

        # update network weight matrix
        if np.max(np.abs(self.model.params.W)) > 0:
            obj_network.W = \
                np.abs(self.model.params.W) / np.max(np.abs(self.model.params.W))
        else:
            obj_network.W = self.model.params.W

        # update network graph
        obj_network.update_graph()

    #
    # save all parameters
    #

    def save(self, folder = None, filename = None, fileformat = 'npz'):
        if folder == None:
            self.logger.error("no folder was given!")
            quit()
        
        # create path if not available
        import os
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # save all params
        import numpy as np
        for i, params in enumerate(self.training):
            if fileformat == 'npz':
                np.savez("%s_%i.npz" % (folder + filename, i), **params)

    def save_h_mean_model(self, file = None, fileformat = 'csv'):
        if file == None:
            self.logger.error("no file was given!")
            quit()
        
        # create path if not available
        import os
        if not os.path.exists(os.path.dirname(file)):
            os.makedirs(os.path.dirname(file))
        
        # save all params
        import numpy as np
        
        for i, params in enumerate(self.training):
            if i == 0:
                h_mean_model = params['h_mean_model']
            else:
                h_mean_model = np.append(h_mean_model, params['h_mean_model'], 0)
        
        np.savetxt(file,
            X = h_mean_model,
            fmt = '%2.8f', delimiter='\t', newline='\n')

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
    
if __name__ == '__main__':
    machine = mp_machine('')