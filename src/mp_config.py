import logging
import os
import ConfigParser
import re
import inspect
import binascii
import time

#
# metapath configuration class
#

class mp_config:
    
    # configuration is stored in dictionaries
    model    = {}
    network  = {}
    dataset  = {}
    schedule = {}
    analyse  = {}
    path     = {}
    basepath = {}
    
    # default basepaths
    basepath['workspace'] = './workspace/'
    basepath['datasets']  = './data/'
    basepath['reports']   = './reports/'
    basepath['networks']  = './networks/'
    basepath['cache']     = './cache/'
    
    # default paths
    path['config']  = 'metapath.ini'
    path['logfile'] = 'metapath.log'
    path['cache']   = basepath['cache']
    path['reports'] = basepath['reports']
    
    path['default_workspace'] = basepath['workspace'] + 'default.ini'
    
    def __init__(self, workspace = None):
        
        # init basic logger (console)
        self.logger = self.init_logger()
        
        # get configuration
        if os.path.exists(self.path['config']):   
            
            cfg = ConfigParser.ConfigParser()
            cfg.optionxform = str
            cfg.read(self.path['config'])
            
            # section: folders
            if 'workspace' in cfg.options('folders'):
                path = cfg.get('folders', 'workspace').strip()
                if os.path.exists(path):
                    self.basepath['workspace'] = path
                else:
                    self.logger.error("workspace folder '%s' does not exist" % (path))
                    
            if 'datasets' in cfg.options('folders'):
                path = cfg.get('folders', 'datasets').strip()
                if os.path.exists(path):
                    self.basepath['datasets'] = path
                else:
                    self.logger.error("dataset folder '%s' does not exist" % (path))
                    
            if 'networks' in cfg.options('folders'):
                path = cfg.get('folders', 'networks').strip()
                if os.path.exists(path):
                    self.basepath['networks'] = path
                else:
                    self.logger.error("network folder '%s' does not exist" % (path))
                    
            if 'reports' in cfg.options('folders'):
                path = cfg.get('folders', 'reports').strip()
                if os.path.exists(path):
                    self.basepath['reports'] = path
                else:
                    self.logger.error("report folder '%s' does not exist" % (path))
                    
            if 'cache' in cfg.options('folders'):
                path = cfg.get('folders', 'cache').strip()
                if os.path.exists(path):
                    self.basepath['cache'] = path
                else:
                    self.logger.error("cache folder '%s' does not exist" % (path))
                    
            # section: files
            if 'default_workspace' in cfg.options('files'):
                path = cfg.get('files', 'default_workspace').strip()
                if os.path.exists(self.basepath['workspace'] + path):
                    self.path['default_workspace'] = self.basepath['workspace'] + path
                elif os.path.exists(path):
                    self.path['default_workspace'] = path
                else:
                    self.logger.error("default workspace file '%s' does not exist" % (path))
                    
            if 'default_log' in cfg.options('files'):
                path = cfg.get('files', 'default_log').strip()
        
        # load workspace
        if workspace:
            self.load(workspace)

    def load(self, file = None):

        # search workspace file
        if file == None:
            file = self.path['default_workspace']
        elif os.path.isfile(self.basepath['workspace'] + file):
            config_file = self.basepath['workspace'] + file
        elif os.path.isfile(file):
            config_file = file
        else:
            self.logger.error("workspace file '%s' does not exist!" % (file))
            return False
        
        # update current reports directory
        file_name = os.path.basename(file)
        file_basename = os.path.splitext(file_name)[0]
        basepath = self.basepath['reports'] + file_basename + '/'
        self.path['reports'] = basepath # self.get_iterative_subfolder(basepath)
        
        # update current cache directory
        string = os.path.abspath(file)
        self.path['cache'] = self.get_unique_subfolder(self.basepath['cache'], string)

        # update logger to creaty logfile in reports directory
        self.path['logfile'] = self.path['reports'] + 'metapath.log'
        self.logger = self.init_logger(self.path['logfile'])

        # init parser
        cfg = ConfigParser.ConfigParser()
        cfg.optionxform = str
        cfg.read(file)
        
        # use regular expressions to match sections
        re_network = re.compile('network[0-9]*')
        re_dataset = re.compile('dataset[0-9]*')
        re_model = re.compile('model[0-9]*')
        re_schedule = re.compile('schedule[0-9]*')
        re_analyse = re.compile('analyse[0-9]*')
        
        for section in cfg.sections():
            if re_network.match(section):
                network_cfg = self.get_network_cfg(cfg, section)
                if network_cfg:
                    self.network[network_cfg['name']] = network_cfg
            elif re_dataset.match(section):
                dataset_cfg = self.get_dataset_cfg(cfg, section)
                if dataset_cfg:
                    self.dataset[dataset_cfg['name']] = dataset_cfg
            elif re_model.match(section):
                model_cfg = self.get_model_cfg(cfg, section)
                if model_cfg:
                    self.model[model_cfg['name']] = model_cfg
            elif re_schedule.match(section):
                schedule_cfg = self.get_schedule_cfg(cfg, section)
                if schedule_cfg:
                    self.schedule[schedule_cfg['name']] = schedule_cfg
            elif re_analyse.match(section):
                analyse_cfg = self.get_analyse_cfg(cfg, section)
                if analyse_cfg:
                    self.analyse[analyse_cfg['name']] = analyse_cfg
    
    #
    # parse network* section
    #
    
    def get_network_cfg(self, cfg, section):
        
        # create new network config dict
        network = {}
        network['id'] = int(section.lstrip('network'))
        
        # get network file
        if 'file' in cfg.options(section):
            file = cfg.get(section, 'file').strip()
            
            # check network file
            if os.path.isfile(self.basepath['networks'] + file):
                file = os.path.abspath(self.basepath['networks'] + file)
            elif os.path.isfile(file):
                file = os.path.abspath(file)
            else:
                self.logger.warning("skipping section '" + 
                    section + "': network file '" + file + "' does not exist!")
                return False
            
            network['file'] = file
        else:
            self.logger.warning("skipping section '" +
                section + "': missing parameter 'file'")
            return None
        
        # get network fileformat, default: get from file extension
        if 'file_format' in cfg.options(section):
            network['file_format'] = cfg.get(section, 'file_format').strip().lower()
        else:
            file_name = os.path.basename(network['file'])
            file_ext = os.path.splitext(file_name)[1]
            network['file_format'] = file_ext.lstrip('.')
        
        # read network file
        if network['file_format'] == 'ini':

            netcfg = ConfigParser.ConfigParser()
            netcfg.optionxform = str
            netcfg.read(network['file'])

            # validate network file
            if not 'network' in netcfg.sections():
                self.logger.warning("skipping section '" + section
                    + "': network file '" + network['file']
                    + "' does not contain section 'network'")
                return None
            if not 'e-tf binding' in netcfg.sections():
                self.logger.warning("skipping section '" + section
                    + "': network file '" + network['file']
                    + "' does not contain section 'e-tf binding'")
                return None
            if not 'tf-s binding' in netcfg.sections():
                self.logger.warning("skipping section '" + section
                    + "': network file '" + network['file']
                    + "' does not contain section 'tf-s binding'")
                return None
                
            # parse network file
            
            # section 'network'
            
            # get network name
            if 'name' in netcfg.options('network'):
                network['name'] = netcfg.get('network', 'name').strip()
            else:
                network['name'] = network['file']
                
            # get gene label format
            if 'gene_format' in netcfg.options('network'):
                network['label_format'] = netcfg.get('network', 'gene_format').strip()
            else:
                network['label_format'] = 'alias'

            # get nodes and interactions as lists
            list_e = []
            list_tf = []
            list_e_tf = []
            list_s = []
            list_s_tf = []
            
            # section 'e-tf binding'
            for e in netcfg.options('e-tf binding'):
                e = e.strip()
                if e == '':
                    continue
                list_e.append(e)
                e_tf = netcfg.get('e-tf binding', e).split(',')
                for tf in e_tf:
                    tf = tf.strip()
                    if tf == '':
                        continue
                    list_tf.append(tf)
                    list_e_tf.append((e, tf))

            # section 'tf-s binding'
            for tf in netcfg.options('tf-s binding'):
                tf = tf.strip()
                if tf == '':
                    continue
                list_tf.append(tf)
                tf_s = netcfg.get('tf-s binding', tf).split(',')
                for s in tf_s:
                    s = s.strip()
                    if s == '':
                        continue
                    list_s.append(s)
                    list_s_tf.append((s, tf))
                
        # save lists
        network['nodes'] = {
            'e': list(set(list_e)),
            'tf': list(set(list_tf)),
            's': list(set(list_s))}
        network['edges'] = {
            'e-tf': list(set(list_e_tf)),
            's-tf': list(set(list_s_tf))}
        
        return network

    #
    # parse dataset* section
    #

    def get_dataset_cfg(self, cfg, section):
        
        # create new dataset config dict
        dataset = {
            'id': int(section.lstrip('dataset')),
            'cache_path': self.path['cache']}
        
        # get dataset file
        if 'file' in cfg.options(section):
            file = cfg.get(section, 'file').strip()
            
            # check dataset file
            if os.path.isfile(self.basepath['datasets'] + file):
                file = os.path.abspath(self.basepath['datasets'] + file)
            elif os.path.isfile(file):
                file = os.path.abspath(file)
            else:
                self.logger.warning("skipping section '" + section
                    + "': file '" + file + "' does not exist!")
                return None
            
            dataset['file'] = file
        else:
            self.logger.warning("skipping section '" + section
                + "': missing parameter 'file'")
            return None

        # get dataset fileformat, default: get from file extension
        if 'file_format' in cfg.options(section):
            dataset['file_format'] = cfg.get(section, 'file_format').strip().lower()
        else:
            file_name = os.path.basename(dataset['file'])
            file_ext = os.path.splitext(file_name)[1]
            dataset['file_format'] = file_ext.lstrip('.')
            
        # validate dataset file format
        if not dataset['file_format'] in ['csv', 'txt', 'tsv']:
            self.logger.warning("skipping section '" + section
                + "': unknown dataset file format'" + dataset['file_format'] + "'!")
            return None
        else:
            dataset['file_format'] = 'csv'
        
        # get dataset name
        if 'name' in cfg.options(section):
            dataset['name'] = cfg.get(section, 'name').strip()
        else:
            file_name = os.path.basename(dataset['file'])
            file_basename = os.path.splitext(file_name)[0]
            dataset['name'] = file_basename
            
        # get dataset file format specific options as dict, default {}
        dataset['file_options'] = {}
        if 'file_options' in cfg.options(section):
            file_options = \
                self._str_to_dict(cfg.get(section, 'file_options').replace('\n', ''))
        else:
            file_options = {}

        # csv: 'comma separated values' file format
        if dataset['file_format'] == 'csv':

            # delimiter, default: comma
            if 'delimiter_ascii' in file_options:
                dataset['file_options']['delimiter'] = \
                    chr(file_options['delimiter_ascii'])
            else:
                dataset['file_options']['delimiter'] = ','

            # dtype, default: float
            if 'data_type' in file_options \
                and file_options['data_type'].lower() in [
                    'bool', 'int', 'int8', 'int16', 'int32', 'int64',
                    'float', 'float16', 'float32', 'float64']:
                dataset['file_options']['dtype'] = \
                    file_options['data_type']
            else:
                dataset['file_options']['dtype'] = 'float'

            # skiprows, default: 0
            if 'include_labels' in file_options \
                and file_options['include_labels'].lower() in ['true', '1', 'yes']:
                dataset['file_options']['skiprows'] = 1
                dataset['file_include_labels'] = True
            else:
                dataset['file_options']['skiprows'] = 0
                dataset['file_include_labels'] = False

            # label_format, default: 'entrezid'
            if 'label_format' in file_options:
                dataset['label_format'] = file_options['label_format'].lower()
            else:
                dataset['label_format'] = 'entrezid'

        # get labels
        if 'label' in cfg.options(section):
            labels = cfg.get(section, 'label').split(',')
            dataset['label'] = []
            for label in labels:
                dataset['label'].extend(label.strip())
        else:
            if not dataset['file_include_labels']:
                self.logger.warning("skipping section '" + section
                    + "': missing label information!")
                return None

            try:
                f = open(dataset['file'], 'r')
                labels = f.readline().replace('"', '')
                f.close()
                
                dataset['label'] = self._str_to_list(labels, dataset['file_options']['delimiter'])
            except:
                self.logger.warning("skipping section '" + section
                    + "': could not read file!")
                return None

        # get pre-processing options
        if 'preprocessing' in cfg.options(section):
            dataset['preprocessing'] = \
                self._str_to_dict(cfg.get(section, 'preprocessing').replace('\n', ''))
        else:
            dataset['preprocessing'] = {}

        return dataset

    #
    # parse model* section
    #

    def get_model_cfg(self, cfg, section):
        
        # create new model config dict
        model = {}
        model['id'] = int(section.lstrip('model'))
        
        # get model class
        if 'class' in cfg.options(section):
            model['class'] = cfg.get(section, 'class').strip()
        else:
            self.logger.warning("skipping section '" + section
                + "': missing parameter 'class'!")
            return None
        
        # verify model class
        found = False
        try:
            package_name = 'src.mp_model_' + model['class'].lower()
            exec "import " + package_name + " as package"
            
            for name, obj in inspect.getmembers(package):
                if name == model['class']:
                    found = True
                    break
        except:
            pass
        
        if not found:
            self.logger.warning("skipping section '" + section
                + "': unkown model class '" + model['class'] + "'!")
            return None
        
        # get model name
        if 'name' in cfg.options(section):
            model['name'] = cfg.get(section, 'name').strip()
        else:
            model['name'] = section
            
        # get model description
        if 'description' in cfg.options(section):
            model['description'] = \
                cfg.get(section, 'description').replace('\n', '')
        else:
            model['description'] = model['name']
            
        # get model init parameters
        if 'init' in cfg.options(section):
            model['init'] = self._str_to_dict(cfg.get(section, 'init'))
        else:
            model['init'] = {}
        
        return model

    #
    # parse schedule* section
    #
    
    def get_schedule_cfg(self, cfg, section):
        
        # create new schedule config dict
        schedule = {}
        schedule['id'] = int(section.lstrip('schedule'))
        
        if 'iterations' in cfg.options(section):
            schedule['iterations'] = cfg.getint(section, 'iterations')
        else:
            schedule['iterations'] = 1

        if 'mode' in cfg.options(section):
            schedule['mode'] = cfg.get(section, 'mode')
        else:
            schedule['mode'] = 'scan'

        if 'stages' in cfg.options(section):
            schedule['stages'] = cfg.getint(section, 'stages')
        else:
            schedule['stages'] = 0

        # use regular expressions to match stages
        re_stage = re.compile('stage_[0-9]*')
        
        schedule['stage'] = []
        for key in cfg.options(section):
            if not re_stage.match(key):
                continue
            
            if schedule['stages'] > 0 and \
                len(schedule['stage']) >= schedule['stages']:
                break

            value = cfg.get(section, key).replace('\n', '')
            schedule['stage'].append(self._str_to_dict(value))
            
        schedule['stages'] = len(schedule['stage'])
            
        if schedule['stages'] == 0:
            self.logger.warning("skipping section '" + section
                + "': missing parameter 'stage_*'!")
            return None
        
        # get schedule name
        if 'name' in cfg.options(section):
            schedule['name'] = cfg.get(section, 'name').strip()
        else:
            schedule['name'] = section

        return schedule

    #
    # parse analyse* section
    #
    
    def get_analyse_cfg(self, cfg, section):
        
        # create new analyse config dict
        analyse = {}
        analyse['plot'] = {
            'dataset': {
                'correlation': {} },
            'network': {
                'graph': {} },
            'model': {
                'graph': {},
                'knockout': {} } }

        analyse['id'] = int(section.lstrip('analyse'))
        analyse['output'] = []

        if 'name' in cfg.options(section):
            analyse['name'] = cfg.get(section, 'name').strip()
        else:
            analyse['name'] = section

        if 'plot_points' in cfg.options(section):
            analyse['plot_points'] = cfg.getint(section, 'plot_points')
        else:
            analyse['plot_points'] = 150

        if 'model_weight_scale' in cfg.options(section):
            analyse['plot_weight_scale'] = cfg.getfloat(section, 'model_weight_scale')
        else:
            analyse['plot_weight_scale'] = 1.0

        if 'network_file' in cfg.options(section) \
            and cfg.getboolean(section, 'network_file'):
            analyse['output'].append('network:file')
            
        if 'model_file' in cfg.options(section):
            s = cfg.get(section, 'model_file').strip().lower()
            if s in ['single']:
                analyse['output'].append('model:file')
            elif s == 'all':
                analyse['output'].append('model:evolution:file')
                
        if 'model_plot' in cfg.options(section):
            s = cfg.get(section, 'model_plot').strip().lower()
            if s in ['single']:
                analyse['output'].append('model:plot')
            elif s == 'all':
                analyse['output'].append('model:evolution:plot')
                
        if 'model_evolution_video' in cfg.options(section) \
            and cfg.getboolean(section, 'model_evolution_video'):
            analyse['output'].append('model:evolution:video')
            
        if 'model_evolution_energy' in cfg.options(section) \
            and cfg.getboolean(section, 'model_evolution_energy'):
            analyse['output'].append('model:evolution:energy')
            
        if 'model_evolution_error' in cfg.options(section) \
            and cfg.getboolean(section, 'model_evolution_error'):
            analyse['output'].append('model:evolution:error')
            
        if 'path' in cfg.options(section):
            analyse['path'] = path
        else:            
            analyse['path'] = self.path['reports']

        if 'report' in cfg.options(section):
            dict = self._str_to_dict(cfg.get(section, 'report'))
            analyse['report'] = dict
            if 'include' in dict and dict['include'] in ['yes']:
                analyse['output'].append('report')

        if 'model_graph_plot' in cfg.options(section):
            dict = self._str_to_dict(cfg.get(section, 'model_graph_plot'))
            analyse['plot']['model']['graph'] = dict
            if 'include' in dict and dict['include'] in ['yes']:
                analyse['output'].append('plot:graph')
                
        if 'model_knockout_plot' in cfg.options(section):
            dict = self._str_to_dict(cfg.get(section, 'model_knockout_plot'))
            analyse['plot']['model']['knockout'] = dict
            if 'include' in dict and dict['include'] in ['yes']:
                analyse['output'].append('plot:knockout')

        if 'network_graph_plot' in cfg.options(section):
            dict = self._str_to_dict(cfg.get(section, 'network_graph_plot'))
            analyse['plot']['network']['graph'] = dict
            if 'include' in dict and dict['include'] in ['yes']:
                analyse['output'].append('plot:network')
            
        if 'dataset_correlation_plot' in cfg.options(section):
            dict = self._str_to_dict(cfg.get(section, 'dataset_correlation_plot'))
            analyse['plot']['dataset']['correlation'] = dict
            if 'include' in dict and dict['include'] in ['yes']:
                analyse['output'].append('plot:correlation')

        return analyse

    #
    # get configuration
    #

    def get(self, section, id = None, name = None):
        if section == 'network':
            if id:
                for network in self.network:
                    if not network['id'] == id:
                        continue
                    return network
                return None
            if name:
                if name in self.network:
                    return self.network[name]
                return None
            return self.network[self.network.keys()[0]]
        
        if section == 'dataset':
            if id:
                for dataset in self.dataset:
                    if not dataset['id'] == id:
                        continue
                    return dataset
                return None
            if name:
                if name in self.dataset:
                    return self.dataset[name]
                return None
            return self.dataset[self.dataset.keys()[0]]
        
        if section == 'model':
            if id:
                for model in self.model:
                    if not model['id'] == id:
                        continue
                    return model
                return None
            if name:
                if name in self.model:
                    return self.model[name]
                return None
            return self.model[self.model.keys()[0]]
        
        if section == 'schedule':
            if id:
                for schedule in self.schedule:
                    if not schedule['id'] == id:
                        continue
                    return schedule
                return None
            if name:
                if name in self.schedule:
                    return self.schedule[name]
                return None
            return self.schedule[self.schedule.keys()[0]]
        
        if section == 'analyse':
            if id:
                for analyse in self.analyse:
                    if not analyse['id'] == id:
                        continue
                    return analyse
                return None
            if name:
                if name in self.analyse:
                    return self.analyse[name]
                return None
            return self.analyse[self.analyse.keys()[0]]

    #
    # logging
    #

    def init_logger(self, logfile = None):
        
        # initialize logger
        
        logger = logging.getLogger('metapath')
        logger.setLevel(logging.INFO)
        
        # remove all previous handlers
        for h in logger.handlers:
            logger.removeHandler(h)
        
        # set up file handler for logfile
        if not logfile == None:
            file_formatter = logging.Formatter(
                fmt = '%(asctime)s %(levelname)s %(module)s -> %(funcName)s: %(message)s',
                datefmt = '%m/%d/%Y %H:%M:%S')
            file_handler = logging.FileHandler(logfile)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        # set up console handler
        console_formatter = logging.Formatter(
            fmt = '%(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        return logger

    #
    # create subfolders
    #

    def get_iterative_subfolder(self, basepath, folder = None):
        
        # check basepath
        try:
            if not os.path.exists(basepath):
                os.makedirs(basepath)
        except:
            self.logger.error("could not create directory '" + basepath + "'")
            return None
        
        #
        if folder:
            pre = basepath + folder + '_'
        else:
            pre = basepath

        # search for unused subfolder, starting with 1
        i = 1
        while os.path.exists('%s%s/' % (pre, i)):
            i += 1
        
        # create new subfolder
        new = '%s%s/' % (pre, i)
        try:
            os.makedirs(new)
        except:
            self.logger.error("could not create directory '" + new + "'")
            return None
        
        return os.path.abspath(new) + '/'

    def get_unique_subfolder(self, basepath, string):
        
        # check basepath
        try:
            if not os.path.exists(basepath):
                os.makedirs(basepath)
        except:
            self.logger.error("could not create directory '" + basepath + "'")
            return None
        
        # create unique foldername using string -> crc32(string)
        i = abs(binascii.crc32(string))
        
        # create new subfolder
        new = '%s%s/' % (basepath, i)
        try:
            if not os.path.exists(new):
                os.makedirs(new)
        except:
            self.logger.error("could not create directory '" + new + "'")
            return None
        
        return os.path.abspath(new) + '/'
    
    #
    # string converter
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