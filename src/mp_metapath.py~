import copy
import time
import pickle
import os
from src.mp_common import *

#
# metapath main class
#

class metapath:
    
    version = '0.3.9-20120712'
    logger  = None

    def __init__(self, workspace = None, default = False):
        
        # init metapath base configuration
        from src.mp_config import mp_config
        self.cfg = mp_config()
        
        # init logger (console) and print welcome message
        import logging
        self.logger = logging.getLogger('metapath')
        self.logger.info('MetaPath %s' % (self.version))
        
        # load configuration
        if default or workspace:
            self.cfg.load(workspace)
    
    #
    # CREATE NETWORK INSTANCE
    #
    
    def network(self, name = None, quiet = False):
        
        # create empty network if non name is given
        if name == None:
            from src.mp_network import mp_network
            return mp_network()
        
        # get dataset configuration
        cfg_network = self.cfg.get('network', name = name)
        if not cfg_network and not quit:
            self.logger.error('unkown network "' + name + '"')
            return None
        
        # create console message
        if not quit:
            self.logger.info("create network instance from '" + name + "'")
        
        # create dataset instance
        from src.mp_network import mp_network
        return mp_network(cfg_network)
    
    #
    # CREATE DATASET INSTANCE
    #
    
    def dataset(self, name = None, quiet = False):
        
        # create empty dataset if no name is given
        if name == None:
            from src.mp_dataset import mp_dataset
            return mp_dataset()
        
        # get dataset configuration
        cfg_dataset = self.cfg.get('dataset', name = name)
        if not cfg_dataset and not quit:
            self.logger.error('unkown dataset "' + name + '"')
            return None
        
        # create console message
        if not quit:
            self.logger.info("create dataset instance of '" + name + "'")
        
        # create dataset instance
        from src.mp_dataset import mp_dataset
        return mp_dataset(cfg_dataset)
    
    #
    # CREATE MODEL INSTANCE
    #

    def model(self, name = None, network = None, dataset = None, quiet = False):
        
        # create empty model instance if no name is given
        if name == None:
            from src.mp_model import mp_model
            return mp_model(
                network = self.network(),
                dataset = self.dataset() )
                
        # get model configuration
        cfg_model = self.cfg.get('model', name = name)
        if not cfg_model:
            self.logger.error('unkown model type "' + name + '"')
            return None
        else:
            name = cfg_model['name']
        
        # create console message
        if not quiet:
            self.logger.info("create new model")
            self.logger.info(" * model configuration")
            self.logger.info("   "
                "class: '" + name + "', "
                "network: '" + network + "', "
                "dataset: '" + dataset + "'")
        
        # create model instance
        from src.mp_model import mp_model
        return mp_model(
            config  = cfg_model,
            network = self.network(network, quiet = True),
            dataset = self.dataset(dataset, quiet = True) )

    #
    # SAVE MODEL TO FILE
    #
        
    def save(self, model, file = None, quiet = False):
        
        # check object class
        if not model.__class__.__name__ == 'mp_model':
            self.logger.error("could not save model: 'model' has to be mp_model instance!")
            return False
        
        # get filename
        if file == None:
            file = self.cfg.path['reports'] + 'model-%s/model-%s.mp' % (model.id, model.type)
            
        file = get_empty_file(file)
        
        # dump model parameters to file
        file_handler = open(file, 'wb')
        pickle.dump(model.get(), file_handler)
        file_handler.close()
        
        # create console message
        if not quiet:
            self.logger.info("save model to file: '" + file + "'")
        
        return file
        
    #
    # LOAD MODEL FROM FILE
    #
        
    def load(self, file, quiet = False):
        
        # check file
        if not os.path.exists(file):
            self.logger.error("could not open file: '%s'!" % file)
            return None
        
        # get model parameters from file
        file_handler = open(file, 'rb')
        model_dict = pickle.load(file_handler)
        file_handler.close()
        
        # create empty model instance
        model = self.model()
        
        # set model parameters
        model.set(model_dict)
        
        # create console message
        if not quiet:
            self.logger.info("load model from file: '" + file + "'")
        
        return model

    #
    # OPTIMIZE MODEL INSTANCE
    #

    def optimize(self, obj_model, schedule = None):
        
        # get training schedule configuration
        cfg_schedule = self.cfg.get('schedule', name = schedule)
        if not cfg_schedule:
            self.logger.error('unkown training schedule "' + schedule + '"')
            return None
        
        # make copy of model
        model = copy.copy(obj_model)
        
        # start training schedule
        self.logger.info("optimize model:")
        self.logger.info(" * starting time: %s" % \
            (time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())))

        for stage, params in enumerate(cfg_schedule['stage']):
            model.optimize(**params)
    
        return model
    
    #
    # ANALYSE MODEL INSTANCE
    #
    
    # analysis in batch mode
    def analyse(self, obj_model, name = None):
        
        # get analyse configuration
        cfg_analyse = self.cfg.get('analyse', name = name)
        if not cfg_analyse:
            self.logger.error('unkown analyser configuration "' + name + '"')
            return False
        
        # make copy of model
        model = copy.copy(obj_model)

        # create analyse instance
        from src.mp_analyse import mp_analyse
        analyser = mp_analyse(config = cfg_analyse)
        
        # start analyse
        self.logger.info("analyse model:")
        return analyser.run(model)

    # wrapper to simply plot any object type
    def plot(self, obj_generic, config = None, type = ''):
    
        # get analyse configuration
        cfg_analyse = self.cfg.get('analyse', name = config)
        if not cfg_analyse:
            self.logger.error('unkown analyser configuration "' + config + '"')
            return None
        
        # create analyse instance
        from src.mp_analyse import mp_analyse
        analyser = mp_analyse(config = cfg_analyse)
        
        return analyser.plot(obj_generic, type = type)
        
##    def analyse_plot_knockout(self, model):
##        
##        
##        #
##        # create table
##        #
##        
##        x = np.array([(1.0, 2), (3.0, 4)], dtype=[('x', float), ('y', int)])
##        
##        size = self.model.params.v_size
##        table_records = size ** 2
##        table_layout = {
##            'names': [
##                'gene1_name',
##                'gene1_type',
##                'gene2_name',
##                'gene2_type',
##                'correlation',
##                'effect' ],
##            'formats': [
##                16 * [np.str_] +
##                2 * [np.str_] +
##                16 * [np.str_] +
##                2 * [np.str_] +
##                [np.float] +
##                [np.float]
##            ]
##        }
##        
##        tbl = np.array(
##            shape = (6, table_records),
##            dtype = table_layout)
##        
##        # calcualte entries
##        corr = np.corrcoef(self.dataset.data.T)
##        
##        for v1 in range(size):
##            for v2 in range(size):
##                index = v1 * size + v2
##                analyse_array[0, index] = self.dataset.label[v1]
##                analyse_array[1, index] = '?'
##                analyse_array[2, index] = self.dataset.label[v2]
##                analyse_array[3, index] = '?'
##                analyse_array[4, index] = corr[v1, v2]
##                analyse_array[5, index] = v_impact_on_v[v1, v2]
##        
##        analyse_table = {
##            'gene1_name': [],
##            'gene1_type': [],
##            'gene2_name': [],
##            'gene2_type': [],
##            'corr': [],
##            'impact': []
##        }
##        
##        for v1 in range(v_impact_on_v.shape[0]):
##            for v2 in range(v_impact_on_v.shape[1]):
##                analyse_table['gene1_name'].extend([self.dataset.label[v1]])
##                analyse_table['gene2_name'].extend([self.dataset.label[v2]])
##                analyse_table['corr'].extend([corr[v1, v2]])
##                analyse_table['impact'].extend([v_impact_on_v[v1, v2]])
##                
##        print(analyse_table)


##    
##        quit()
##        
##        if len(self.schedule_parameters) > 1:
##            for i, params in enumerate(self.schedule_parameters):
##                if 'name' in params:
##                    self.logger.info("schedule stage %i: '%s'" % (i + 1, params['name']))
##                else:
##                    self.logger.info("schedule stage %i")
##                self.model.optimize(data = self.dataset.data, **params)
##        
##        
##        
##        
##        
##        
##        self.schedule = []
##        for i in range(1, cfg_schedule['iterations'] + 1):
####            # TODO: measure time within optimization
####            if schedule == 1:
####                schedule_start = time.time()
##                            
##            if len(self.schedule_parameters) > 1:
##                for i, params in enumerate(self.schedule_parameters):
##                    if 'name' in params:
##                        self.logger.info("schedule stage %i: '%s'" % (i + 1, params['name']))
##                    else:
##                        self.logger.info("schedule stage %i")
##                    self.model.optimize(data = self.dataset.data, **params)
##        
##        quit()
##        
##        

##        
##        
##        
##        self.schedule = []
##        for schedule in range(1, self.schedule_iterations + 1)
##    
##        #if not scenario == None:
##        #    self.load_scenario(scenario)
##        #if not output == None:
##        #    self.output = output
##        
##        #
##        # iterative schedule
##        #
##        
##        # plot graph
##        if 'network:plot:graph' in self.output:
##            self.plot_network(self.save_path + "network.png")
##                
##        # plot correlation
##        if 'dataset:plot:correlation' in self.output:
##            self.dataset.plot(
##                file = self.save_path + "correlation.png",
##                relationship = 'pearson')
##                
####        # calculate total number of updates
####        # and update number of plot_points per schedule stage
####        if self.create_plots:
####            updates = 0
####            for params in self.schedule_parameters:
####                if 'updates' in params:
####                    updates += params['updates']
####
####            for params in self.schedule_parameters:
####                if 'updates' in params:
####                    plot_points = self.plot_points * params['updates'] / updates
####                    params['plot_points'] = plot_points                
##                
##        self.schedule = []
##        for schedule in range(1, self.schedule_iterations + 1):
##            
##            if schedule == 1:
##                schedule_start = time.time()
##                
##                # plot model evolution
##                if 'model:evolution:plot' in self.output:
##                    self.plot_model(file = self.save_path + "model 0.png")
##                    
##            elif self.schedule_mode in ['scan', 'breadth']:
##                
##                # reset model before each optimization
##                self.model.reset()
##
##            if len(self.schedule_parameters) > 1:
##                for i, params in enumerate(self.schedule_parameters):
##                    if 'name' in params:
##                        self.logger.info("schedule stage %i: '%s'" % (i + 1, params['name']))
##                    else:
##                        self.logger.info("schedule stage %i")
##                    self.model.optimize(data = self.dataset.data, **params)
##            else:
##                params = self.schedule_parameters[0]
##                self.model.optimize(data = self.dataset.data, **params)
##            
##            # append model params to list
##            self.schedule.extend([self.model.params.get()])
##            
##            # save model to file
##            if 'model:evolution:file' in self.output:
##                self.model.params.save(self.save_path + "model %i" % (schedule))
##            # plot model evolution
##            if 'model:evolution:plot' in self.output:
##                self.plot_model(file = self.save_path + 'model %i.png' % (schedule))
##            # plot model evolution video frame
##            if 'model:evolution:video' in self.output:
##                self.plot_frame(file = self.cache + '/frame %i.png' % (schedule))
##                
##            if schedule == 1:
##                schedule_stop = time.time()
##                delta = ((schedule_stop - schedule_start) * self.schedule_iterations)
##                est = time.localtime(schedule_stop + delta)
##                self.logger.info("estimated finishing time: %s (%.2fs)" % (
##                    time.strftime("%a, %d %b %Y %H:%M:%S", est),
##                    delta))
##            
##        stop = time.time()
##        self.logger.info("total time: %.2fs" % (stop - start))
##        
##        # plot model
##        if 'model:plot' in self.output:
##            self.plot_model(file = self.save_path + 'model.png')
##        # create model evolution video
##        if 'model:evolution:video' in self.output:
##            self.create_video(
##                frames = self.cache + '/frame %d.png',
##                file = self.save_path + 'model_evolution.mp4'
##            )
##            
##            
##            
##            
##            
##            
##            
##            
##            
##            
##            
##            
##            
##            
##            
##            
##            
##            
##            
##            
##            
##            
##            
##
##
##
##
##        # create dataset instance
##        cfg_dataset = self.cfg.get('dataset', name = dataset)
##        from src.mp_dataset import mp_dataset
##        dataset = mp_dataset(cfg_dataset)
##        
##        # create network instance
##        cfg_network = self.cfg.get('network', name = network)
##        from src.mp_network import mp_network
##        network = mp_network(cfg_network)
##        
##        # search for cache file
##        cache_file = self.cfg.path['cache'] + 'data_%s_%s.npz' \
##            % (cfg_dataset['id'], cfg_network['id'])
##            
##        import os
##        if not os.path.isfile(cache_file):
##            self.create_schedule_data(cache_file)
##            
##        # load cache file
##        #dataset.load_data(cache_file)
##
##        # update network configuration
##        network.update(s = dataset.gene['s'], e = dataset.gene['e'])
##
##        # create network instance


##        cfg_network = self.cfg.get('network', name = network)
##        cfg_model = self.cfg.get('model', name = type)
##                
##        if schedule:
##            cfg_schedule = self.cfg.get('schedule', name = schedule)
##        else:
##            cfg_schedule = None
##            
##        if analyse:
##            cfg_analyse = self.cfg.get('analyse', name = analyse)
##        else:
##            cfg_analyse = None
            
            
##        # Annotation
        
        
        ## create dataset instance

        
##        
##        

##        
##        # create dataset instance
##            
##        # prepare data
##        cache_file = self.cfg.path['cache'] + 'data_%s_%s.npz' \
##            % (cfg_dataset['id'], cfg_network['id'])
##        

            
        
##        # init network object
##        from src.mp_network import mp_network
##        self.network = mp_network(self.scenario)
##    
##        #
##        # section: dataset
##        #
##    
##        # init dataset object
##        from src.mp_dataset import mp_dataset
##        self.dataset = mp_dataset(self.scenario)
            
##        # initialize data
##        #import os
##
##        
##        #dataset_id = self.cfg.dataset[dataset]
        
##        dataset_cache_file = self.cache + 'data.npz'
##        
##        # check if dataset file exists
##        if os.path.isfile(dataset_file):
##            self.dataset.load_data(dataset_file)
##        else:
##            self.create_schedule_data(dataset_file)
##
##        # reduce nodes in network to available data
##        self.network.update(
##            e = self.dataset.gene['e'],
##            s = self.dataset.gene['s'])


##        # import model classes
##        import src.mp_model as mp_model
##
##        if 'type' in cfg.options('model'):
##            model = cfg.get('model', 'type').strip().lower()
##        else:
##            model = 'grbm'
##
##        if 'init' in cfg.options('model'):
##            model_init = \
##                self._str_to_dict(cfg.get('model', 'init').replace('\n', ''))
##        else:
##            model_init = {}
##        
##        # create instance of model
##        self.model = mp_model.init(
##            type = model,
##            adjacency = self.network.A, data = self.dataset.data,
##            kwargs = model_init
##        )
##
##
##
##    def load_scenario(self, file = None):
##
##        # check file and set scenario
##        import os
##        if file == None:
##            self.load_scenario(self.default_scenario)
##            return
##        elif os.path.isfile(file):
##            self.scenario = file
##        elif os.path.isfile(self.path_scenarios + '/' + file):
##            self.scenario = self.path_scenarios + '/' + file
##        else:
##            self.logger.error("scenario file '%s' does not exist!" % (file))
##            return
##
##        # init save_path
##        self.save_path = self.init_save_path()
##
##        # create cache
##        self.cache = self.init_cache_path(self.path_cache)
##
##        # init logger
##        self.logger =  self.init_logger(self.save_path + '/metapath.log')
##        self.logger.info('loading scenario file: ' + self.scenario)
##
##        # init parser for scenario file
##        import ConfigParser
##        cfg = ConfigParser.ConfigParser()
##        cfg.optionxform = str
##        cfg.read(self.scenario)
##        
##        # check if configuration file contains network section
##        if not 'network' in cfg.sections():
##            self.logger.critical("configuration file '%s' does not contain section 'network'" % (config_file))
##            quit()
##            
##        if 'file' in cfg.options('network'):
##            network_file = cfg.get('network', 'file').strip()
##            
##            if 'file_format' in cfg.options('network'):
##                file_format = cfg.get('network', 'file_format').strip().lower()
##            else:
##                file_format = 'ini'
##            
##            self.load(file = network_file)
##        else:
##            self.load(file = config_file, format = 'ini')
##        
##
##        #
##        # section: network
##        #
##
##        # init network object
##        from src.mp_network import mp_network
##        self.network = mp_network(self.scenario)
##    
##        #
##        # section: dataset
##        #
##    
##        # init dataset object
##        from src.mp_dataset import mp_dataset
##        self.dataset = mp_dataset(self.scenario)
##            
##        # initialize data
##        self.prepare_data()
##
##        #
##        # SECTION: model
##        #
##
##        # import model classes
##        import src.mp_model as mp_model
##
##        if 'type' in cfg.options('model'):
##            model = cfg.get('model', 'type').strip().lower()
##        else:
##            model = 'grbm'
##
##        if 'init' in cfg.options('model'):
##            model_init = \
##                self._str_to_dict(cfg.get('model', 'init').replace('\n', ''))
##        else:
##            model_init = {}
##        
##        # create instance of model
##        self.model = mp_model.init(
##            type = model,
##            adjacency = self.network.A, data = self.dataset.data,
##            kwargs = model_init
##        )
##        
##        #
##        # SECTION: output
##        #
##
##        if 'plot_points' in cfg.options('output'):
##            self.plot_points = cfg.getint('output', 'plot_points')
##        else:
##            self.plot_points = 150
##            
##        self.output = []
##        if ('network_file' in cfg.options('output')
##            and cfg.getboolean('output', 'network_file')):
##            self.output.append('network:file')
##        if ('network_plot' in cfg.options('output')
##            and cfg.getboolean('output', 'network_plot')):
##            self.output.append('network:plot:graph')
##        if ('model_file' in cfg.options('output')):
##            s = cfg.get('output', 'model_file').strip().lower()
##            if s in ['single']:
##                self.output.append('model:file')
##            elif s == 'all':
##                self.output.append('model:evolution:file')
##        if ('model_plot' in cfg.options('output')):
##            s = cfg.get('output', 'model_plot').strip().lower()
##            if s in ['single']:
##                self.output.append('model:plot')
##            elif s == 'all':
##                self.output.append('model:evolution:plot')
##        if ('model_evolution_video' in cfg.options('output')
##            and cfg.getboolean('output', 'model_evolution_video')):
##            self.output.append('model:evolution:video')
##        if ('model_evolution_energy' in cfg.options('output')
##            and cfg.getboolean('output', 'model_evolution_energy')):
##            self.output.append('model:evolution:energy')
##        if ('model_evolution_error' in cfg.options('output')
##            and cfg.getboolean('output', 'model_evolution_error')):
##            self.output.append('model:evolution:error')
##        if 'model_weight_scale' in cfg.options('output'):
##            self.network.W_scale = cfg.getfloat('output', 'model_weight_scale')
##        if ('dataset_correlation_plot' in cfg.options('output')
##            and cfg.getboolean('output', 'dataset_correlation_plot')):
##            self.output.append('dataset:plot:correlation')
##        
##        #
##        # SECTION: schedule
##        #
##        
##        if 'iterations' in cfg.options('schedule'):
##            self.schedule_iterations = cfg.getint('schedule', 'iterations')
##        else:
##            self.schedule_iterations = 1
##        
##        if 'stages' in cfg.options('schedule'):
##            schedule_stages = cfg.getint('schedule', 'stages')
##        else:
##            schedule_stages = 0
##
##        if 'mode' in cfg.options('schedule'):
##            self.schedule_mode = cfg.get('schedule', 'mode')
##        else:
##            self.schedule_mode = 'scan'
##
##        self.schedule_parameters = []
##        if schedule_stages == 0:
##            if 'schedule' in cfg.options('schedule'):
##                param = cfg.get('schedule', 'schedule').replace('\n', '')
##            else:
##                param = ''
##
##            self.schedule_parameters.append(self._str_to_dict(param))
##        else:
##            for stage in range(1, schedule_stages + 1):
##                if 'stage_%i' % (stage) in cfg.options('schedule'):
##                    param = cfg.get('schedule', 'stage_%i' % (stage)).replace('\n', '')
##                    self.schedule_parameters.append(self._str_to_dict(param))

##    def load_model(self, file):
##        self.logger.info('loading model: ' + file)
##        self.model.params.load(file)
##        self.network.W = self.model.params.W
##        self.network.update_graph()

##    def save(self, savelist = []):
##        if 'allparams' in savelist:
##            self.model.save(self.save_path + 'params/', 'params', 'npz')
##        if 'mean_model' in savelist:
##            self.model.save_h_mean_model(self.save_path + 'params/h_mean.txt')
##        if 'params_txt' in savelist:
##            self.model.params.save(self.save_path + 'params/', format = 'txt')
##        if 'schedule_plots' in savelist:
##            self.model.plot.save(self.save_path + 'plots/')
##        if 'graph' in savelist:
##            self.network.save_graph(self.save_path + 'network/graph.gml')
##        if 'graph_plot' in savelist:
##            self.network.plot(self.save_path + 'network/graph.png')
##        if 'graph_plot_adjacency' in savelist:
##            self.network.plot(self.save_path + 'network/graph_adjacency.png', edges = 'adjacency')
            
##    def plot(self, filename = None):
##        self.network.plot(self.save_path + 'plots/' + filename)

##    def init_logger(self, logfile = None):
##        # initialize logger
##        import logging
##        
##        logger = logging.getLogger('metapath')
##        logger.setLevel(logging.INFO)
##        
##        # remove all previous handlers
##        for h in logger.handlers:
##            logger.removeHandler(h)
##        
##        # set up file handler for logfile
##        if not logfile == None:
##            file_formatter = logging.Formatter(
##                fmt = '%(asctime)s %(levelname)s %(module)s -> %(funcName)s: %(message)s',
##                datefmt = '%m/%d/%Y %H:%M:%S')
##            file_handler = logging.FileHandler(logfile)
##            file_handler.setFormatter(file_formatter)
##            logger.addHandler(file_handler)
##        
##        # set up console handler
##        console_formatter = logging.Formatter(
##            fmt = '%(message)s')
##        console_handler = logging.StreamHandler()
##        console_handler.setFormatter(console_formatter)
##        logger.addHandler(console_handler)
##        
##        return logger

##    def init_save_path(self, scenario = None):
##        import os
##        
##        # use configuration name as save_path
##        if scenario == None:
##            configuration = os.path.basename(self.scenario)
##            scenario = os.path.splitext(configuration)[0]
##        
##        # find iterative subdirectory
##        i = 1
##        while os.path.exists('%s/%s/report %s/' % \
##            (self.path_reports, scenario, i)):
##            i += 1
##
##        path_report = '%s/%s/report %s/' % \
##            (self.path_reports, scenario, i)
##        os.makedirs(path_report)
##
##        return path_report
##
##    def init_cache_path(self, path = None, scenario = None):
##        # set scenario
##        if scenario == None:
##            scenario = self.scenario
##        
##        # get cache path
##        import binascii
##        import os
##        cache_path = '%s/%s/' % (
##            path, abs(binascii.crc32(os.path.abspath(self.scenario))))
##        
##        # create cache path if necessary
##        if not os.path.exists(cache_path):
##            os.makedirs(cache_path)
##        
##        return cache_path

##    def prepare_data(self):
##        
##        import os
##        dataset_file = self.cache + 'data.npz'
##        
##        # check if dataset file exists
##        if os.path.isfile(dataset_file):
##            self.dataset.load_data(dataset_file)
##        else:
##            self.create_schedule_data(dataset_file)
##
##        # reduce nodes in network to available data
##        self.network.update(
##            e = self.dataset.gene['e'],
##            s = self.dataset.gene['s'])
##
##    def create_schedule_data(self, dataset_file):
##        
##        #
##        # ANNOTATION
##        #
##        
##        # convert network and dataset geneIDs to EntrezIDs
##        # using R/bioconductor
##
##        from src.mp_R_wrapper import bioconductor
##        bioc = bioconductor()
##
##        # convert network geneIDs to EntrezIDs
##        self.logger.info("mapping network genes to EntrezID's")
##        
##        network_e_geneIDs, network_e_not_found = bioc.convert_geneids(
##            self.network.gene['e'],
##            self.network.label_format, 'entrezid')
##        network_s_geneIDs, network_s_not_found = bioc.convert_geneids(
##            self.network.gene['s'],
##            self.network.label_format, 'entrezid')
##        network_geneIDs = \
##            network_e_geneIDs + network_s_geneIDs
##        network_genes_not_found = \
##            network_e_not_found + network_s_not_found
##        
##        if network_genes_not_found:
##            self.logger.warning("%i of %i network geneIDs could not be assigned to EntrezIDs: %s" % (
##                len(network_genes_not_found),
##                len(network_geneIDs),
##                ",".join(network_genes_not_found)
##            ))
##
##        # convert dataset geneIDs to EntrezIDs
##        self.logger.info("mapping dataset genes to EntrezID's")
##
##        dataset_geneIDs, dataset_genes_not_found  = bioc.convert_geneids(
##            self.dataset.label,
##            self.dataset.label_format, 'entrezid')
##        
##        if dataset_genes_not_found:
##            self.logger.warning("%i of %i dataset geneIDs could not be assigned to EntrezIDs: %s" % (
##                len(dataset_genes_not_found),
##                len(dataset_geneIDs),
##                ','.join(dataset_genes_not_found)
##            ))
##
##        # search network genes in dataset via EntrezIDs
##        self.logger.info("searching network genes in dataset via assigned EntrezIDs")
##
##        columns = []
##        dataset_label = []
##        columns_not_found = []
##
##        # search for signaling genes in dataset (type s)
##        dataset_label_s = []
##        for index, geneid in enumerate(network_s_geneIDs):
##            label = self.network.gene['s'][index]
##            if geneid in dataset_geneIDs:
##                columns.append(dataset_geneIDs.index(geneid))
##                dataset_label_s.append(label)
##            else:
##                columns_not_found.append(label)
##
##        # search for metabolism genes in dataset (type e)
##        dataset_label_e = []
##        for index, geneid in enumerate(network_e_geneIDs):
##            label = self.network.gene['e'][index]
##            if geneid in dataset_geneIDs:
##                columns.append(dataset_geneIDs.index(geneid))
##                dataset_label_e.append(label)
##            else:
##                columns_not_found.append(label)
##
##        dataset_label = dataset_label_s + dataset_label_e
##
##        if columns_not_found:
##            self.logger.info("%i of %i genes could not be found in dataset" % \
##                (len(columns_not_found), len(network_geneIDs)))
##
##        # set the list of columns in dataset according to the search results
##        self.dataset.set_source_file_columns(columns, format = "index")
##        
##        # set label and label format
##        self.dataset.label = dataset_label
##        self.dataset.label_format = self.network.label_format
##        self.dataset.gene = {'e': dataset_label_e, 's': dataset_label_s}
##
##        #
##        # NORMALIZATION
##        #
##
##        # load source dataset file
##        self.dataset.import_csv()
##        
##        # preprocessing data (normalization etc)
##        self.dataset.run_preprocessing()
##        
##        # save compressed and filtered dataset file
##        self.dataset.save(dataset_file)
##
##    #
##    # CREATE MODEL
##    #
##
##    def create_model_old(self, scenario = None, output = None):
##        if not scenario == None:
##            self.load_scenario(scenario)
##        if not output == None:
##            self.output = output
##        
##        import numpy as np
##        import time
##        self.logger.info(bcolors.HEADER + "Create Model" + bcolors.ENDC)
##        self.logger.info("starting time: %s" % \
##            (time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())))
##
##        start = time.time()
##
##        #
##        # iterative schedule
##        #
##        
##        # plot graph
##        if 'network:plot:graph' in self.output:
##            self.plot_network(self.save_path + "network.png")
##                
##        # plot correlation
##        if 'dataset:plot:correlation' in self.output:
##            self.dataset.plot(
##                file = self.save_path + "correlation.png",
##                relationship = 'pearson')
##                
####        # calculate total number of updates
####        # and update number of plot_points per schedule stage
####        if self.create_plots:
####            updates = 0
####            for params in self.schedule_parameters:
####                if 'updates' in params:
####                    updates += params['updates']
####
####            for params in self.schedule_parameters:
####                if 'updates' in params:
####                    plot_points = self.plot_points * params['updates'] / updates
####                    params['plot_points'] = plot_points                
##                
##        self.schedule = []
##        for schedule in range(1, self.schedule_iterations + 1):
##            
##            if schedule == 1:
##                schedule_start = time.time()
##                
##                # plot model evolution
##                if 'model:evolution:plot' in self.output:
##                    self.plot_model(file = self.save_path + "model 0.png")
##                    
##            elif self.schedule_mode in ['scan', 'breadth']:
##                
##                # reset model before each optimization
##                self.model.reset()
##
##            if len(self.schedule_parameters) > 1:
##                for i, params in enumerate(self.schedule_parameters):
##                    if 'name' in params:
##                        self.logger.info("schedule stage %i: '%s'" % (i + 1, params['name']))
##                    else:
##                        self.logger.info("schedule stage %i")
##                    self.model.optimize(data = self.dataset.data, **params)
##            else:
##                params = self.schedule_parameters[0]
##                self.model.optimize(data = self.dataset.data, **params)
##            
##            # append model params to list
##            self.schedule.extend([self.model.params.get()])
##            
##            # save model to file
##            if 'model:evolution:file' in self.output:
##                self.model.params.save(self.save_path + "model %i" % (schedule))
##            # plot model evolution
##            if 'model:evolution:plot' in self.output:
##                self.plot_model(file = self.save_path + 'model %i.png' % (schedule))
##            # plot model evolution video frame
##            if 'model:evolution:video' in self.output:
##                self.plot_frame(file = self.cache + '/frame %i.png' % (schedule))
##                
##            if schedule == 1:
##                schedule_stop = time.time()
##                delta = ((schedule_stop - schedule_start) * self.schedule_iterations)
##                est = time.localtime(schedule_stop + delta)
##                self.logger.info("estimated finishing time: %s (%.2fs)" % (
##                    time.strftime("%a, %d %b %Y %H:%M:%S", est),
##                    delta))
##            
##        stop = time.time()
##        self.logger.info("total time: %.2fs" % (stop - start))
##        
##        # plot model
##        if 'model:plot' in self.output:
##            self.plot_model(file = self.save_path + 'model.png')
##        # create model evolution video
##        if 'model:evolution:video' in self.output:
##            self.create_video(
##                frames = self.cache + '/frame %d.png',
##                file = self.save_path + 'model_evolution.mp4'
##            )
##        # save model params to txt
##        #self.model.params.save(self.save_path + 'txt/', format = 'txt')
        
    #
    # ANALYSE MODEL
    #

    def analyse_model_old(self):
        
        self.logger.info(bcolors.HEADER + "Evaluate Model" + bcolors.ENDC)
        
        import numpy as np
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        
        #
        # calculate impact
        #
        
        # simulate impact of visible units
        v_impact_on_v, v_impact_on_h = self.model.params.v_impact()

        # simulate impact of hidden units
        h_impact_on_v, h_impact_on_h = self.model.params.h_impact(self.dataset.data)
        
        #
        # plot impact
        #
        
        file = self.save_path + "impact.png"        
        data = v_impact_on_v / np.max(np.abs(v_impact_on_v))

        title = 'Impact of visible Units'
        
        # create figure object
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.grid(True)
        num = len(self.dataset.label)
        cax = ax.imshow(data,
            cmap = matplotlib.cm.hot_r,
            interpolation = 'nearest',
            #interpolation = 'kaiser',
            extent = (0, num, 0, num))
        
        # set ticks and labels
        plt.xticks(
            np.arange(num) + 0.5,
            tuple(self.dataset.label),
            fontsize = 9,
            rotation = 70)
        plt.yticks(
            num - np.arange(num) - 0.5,
            tuple(self.dataset.label),
            fontsize = 9)
        
        # add colorbar
        cbar = fig.colorbar(cax)
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontsize(9)
        
        # add title
        plt.title(title, fontsize = 11)
        
        if file == None:
            plt.show()
        else:
            ## self.logger.info("saving dataset correlation plot to %s" % (file))
            fig.savefig(file, dpi = 300)
        
        # clear current figure object and release memory
        plt.clf()
        plt.close(fig)
        
        #
        # create table
        #
        
        x = np.array([(1.0, 2), (3.0, 4)], dtype=[('x', float), ('y', int)])
        
        size = self.model.params.v_size
        table_records = size ** 2
        table_layout = {
            'names': [
                'gene1_name',
                'gene1_type',
                'gene2_name',
                'gene2_type',
                'correlation',
                'effect' ],
            'formats': [
                16 * [np.str_] +
                2 * [np.str_] +
                16 * [np.str_] +
                2 * [np.str_] +
                [np.float] +
                [np.float]
            ]
        }
        
        tbl = np.array(
            shape = (6, table_records),
            dtype = table_layout)
        
        # calcualte entries
        corr = np.corrcoef(self.dataset.data.T)
        
        for v1 in range(size):
            for v2 in range(size):
                index = v1 * size + v2
                analyse_array[0, index] = self.dataset.label[v1]
                analyse_array[1, index] = '?'
                analyse_array[2, index] = self.dataset.label[v2]
                analyse_array[3, index] = '?'
                analyse_array[4, index] = corr[v1, v2]
                analyse_array[5, index] = v_impact_on_v[v1, v2]
        
        analyse_table = {
            'gene1_name': [],
            'gene1_type': [],
            'gene2_name': [],
            'gene2_type': [],
            'corr': [],
            'impact': []
        }
        
        for v1 in range(v_impact_on_v.shape[0]):
            for v2 in range(v_impact_on_v.shape[1]):
                analyse_table['gene1_name'].extend([self.dataset.label[v1]])
                analyse_table['gene2_name'].extend([self.dataset.label[v2]])
                analyse_table['corr'].extend([corr[v1, v2]])
                analyse_table['impact'].extend([v_impact_on_v[v1, v2]])
                
        print(analyse_table)

    def plot_network(self, file = None):
        plot_caption = r'$\mathbf{Network:}\,\mathrm{%s}$' % (self.network.name)
        self.network.plot(
            file = file,
            edges = 'adjacency',
            draw_edge_labels = False,
            draw_node_captions = False,
            title = '',
            caption = plot_caption)

    def plot_model(self, file = None):
        # update model graph
        #self.network.W = self.model.params.W
        self.network.W = self.model.params.single_link_knockout(self.dataset.data)
        self.network.v_approx = self.model.params.v_approx(self.dataset.data)
        self.network.update_graph()

        # plot model graph
        approx, std = self.model.params.approx(self.dataset.data)
        plot_title = \
            r'$\mathbf{Network:}\,\mathrm{%s}' % (self.network.name) \
            + r',\,\mathbf{Dataset:}\,\mathrm{%s}$' % (self.dataset.name)
        plot_caption = \
            r'$\mathbf{Approximation:}\,\mathrm{%.1f' % (100 * approx) + '\%}$'
            
        self.network.plot(
            file = file,
            edges = 'weights',
            draw_edge_labels = True,
            draw_node_captions = True,
            edge_threshold = 0.1,
            title = plot_title,
            caption = plot_caption,
            colors = 'colors')
            
    def plot_frame(self, file = None):
        # update model graph
        self.network.W = self.model.params.W
        self.network.update_graph()
        # plot model graph
        plot_caption = \
            r'$\mathbf{Network:}\,\mathrm{%s}' % (self.network.name) \
            + r',\,\mathbf{Dataset:}\,\mathrm{%s}$' % (self.dataset.name)
        
        self.network.plot(
            file = file,
            edges = 'weights',
            draw_edge_labels = False,
            edge_threshold = 0.0,
            title = '',
            caption = plot_caption,
            colors = 'colors',
            dpi = 200)

    #
    # create mp4-video using ffmpeg and delete single frames
    #

    def create_video(self, file = None, frames = None):
        
        # create video
        cmd = 'ffmpeg -qscale 1 -r 10 -b 9600 -i "' + frames + '" "' + file + '"'
        import os
        os.system(cmd + ' >/dev/null 2>&1')
        
        # delete frames        
        for i in range(1, self.schedule_iterations + 1):
            frame = frames % (i)
            if os.path.isfile(frame):
                os.remove(frame)

    def init_model(self, type = None, kwargs = {}):
        
        # initialize model
        if type == 'grbm':
            from src.mp_model import GRBM
            return GRBM(
                adjacency_matrix = self.network.A,
                data = self.dataset.data,
                **kwargs)
        else:
            self.logger.error("Model '" + type + '" is unknown!')

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

#
# whatever
#

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    HEADER = OKGREEN

    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.ENDC = ''
