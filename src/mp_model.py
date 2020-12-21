import numpy as np
import logging
import os
import time

class mp_model:
    
    id      = None
    logger  = None
    type    = 'empty'
    history = []
    cfg     = {}
    
    # class instances
    network = None
    dataset = None
    machine = None
    
    def __init__(self, config = None, network = None, dataset = None):
        
        # initialize logger
        self.logger = logging.getLogger('metapath')
        
        # initialize id
        self.id = int(time.time() * 100)
        
        # reference network and dataset
        self.network = network
        self.dataset = dataset
        
        # create empty model instance if no configuration is given
        if not config:
            return

        # configure machine
        self.configure(config = config, network = network, dataset = dataset)
        
        # initialize machine
        self.initialize()
    
    # configure model
    def configure(self, config = None, network = None, dataset = None):
        
        if not config == None:
            self.cfg = config
        elif self.cfg == None:
            self.logger.warning('could not configure model: no configuration was given!')
            return False
        
        # reference network instance
        if not network == None:
            self.network = network
        elif self.network == None:
            self.logger.warning('could not configure model: no network was given!')
            return False

        # reference dataset instance
        if not dataset == None:
            self.dataset = dataset
        elif self.dataset == None:
            self.logger.warning('could not configure model: no network was given!')
            return False

        # create machine instance
        self.configure_machine()


    # configure machine
    def configure_machine(self):
        
        params = {
            'visible': self.network.nodes(type = 'e') + self.network.nodes(type = 's'),
            'hidden':  self.network.nodes(type = 'tf'),
            'edges':   self.network.edges(),
            'init':    self.cfg['init'],
            'data':    self.dataset.data}
        
        class_name = self.cfg['class'].upper()
        package_name = 'src.mp_model_' + class_name.lower()
        
        try:
            exec "from %s import %s" % (package_name, class_name)
            exec "self.machine = %s(**params)" % (class_name)
            self.type = class_name
        except:
            self.machine = None
            self.logger.warning("model class '" + class_name + "' is not supported!")
            return

    # initialize model parameters to dataset        
    def initialize(self, dataset = None):
        
        if not dataset == None:
            self.dataset = dataset
        elif self.dataset == None:
            self.logger.warning('could not initialize model: no dataset was given!')
            return False
        
        if self.machine == None:
            self.logger.warning('could not initialize model: model has not been configured!')
            return False
        
        # search network in dataset and exclude missing nodes
        cache = self.dataset.cfg['cache_path'] + \
            'data-network%s-dataset%s.npz' % \
            (self.network.cfg['id'], self.dataset.cfg['id']) 
        
        if os.path.isfile(cache):
            self.logger.info(' * found cachefile: "' + cache + '"')
            self.dataset.load(cache)
        else:
            self.logger.info(' * search network nodes in dataset and exclude missing')
            label_format = self.network.cfg['label_format']
            label_lists = {
                's': self.network.node_labels(type = 's'),
                'e': self.network.node_labels(type = 'e')}
            self.dataset = self.dataset.subset(label_format, label_lists)
            self.dataset.update_from_source()
            self.logger.info(' * create cachefile: "' + cache + '"')
            self.dataset.save(cache)
            
        # update network to available data
        for type in self.dataset.sub:
            self.network.update(nodelist = {
                'type': type,
                'list': self.dataset.sub[type]})

        # update model to network
        self.configure()
        
        # normalize data
        self.dataset.normalize()
        
        # initialize model parameters with data
        self.machine.initialize(self.dataset.data)

    # optimize model
    def optimize(self, **params):
        
        # append current model parameters to model history
        self.history.append({self.id: self.machine.get()})
        
        # optimize model
        self.machine.run(self.dataset.data, **params)
        
        # update model id
        self.id = int(time.time() * 100)
        
    # get all model parameters as dictionary
    def get(self):
        
        dict = {
            'id': self.id,
            'type': self.type,
            'history': self.history,
            'cfg': self.cfg,
            'network': self.network.get(),
            'dataset': self.dataset.get(),
            'machine': self.machine.get()
        }
    
        return dict
    
    def set(self, dict):
        
        self.id = dict['id']
        self.type = dict['type']
        self.history = dict['history']
        self.cfg = dict['cfg']
        self.network.set(**dict['network'])
        self.dataset.set(**dict['dataset'])
        self.configure_machine()
        self.machine.set(**dict['machine'])
    
    #
    # getter methods for model simulations
    #
    
    def get_approx(self, type = 'rel_approx'):
        if type == 'rel_approx':
            v_approx = self.machine.v_rel_approx(self.dataset.data)
        elif type == 'abs_approx':
            v_approx = self.machine.v_abs_approx(self.dataset.data)
        
        
        # calc mean
        mean_approx = np.mean(v_approx)
        
        # create node dictionary
        v_label = self.machine.v['label']
        approx_dict = {}
        for i, node in enumerate(v_label):
            approx_dict[node] = v_approx[i]
        
        return mean_approx, approx_dict

    def get_knockout_approx(self):
        h_knockout_approx = self.machine.h_knockout_approx(self.dataset.data)
        v_knockout_approx = self.machine.v_knockout_approx(self.dataset.data)
        
        # create node dictionary
        approx_dict = {}
        v_label = self.machine.v['label']
        for i, node in enumerate(v_label):
            approx_dict[node] = v_knockout_approx[i]
        h_label = self.machine.h['label']
        for i, node in enumerate(h_label):
            approx_dict[node] = h_knockout_approx[i]
            
        return approx_dict

    def get_knockout_matrix(self):
        v_impact_on_v, v_impact_on_h = self.machine.v_knockout(self.dataset.data)
        
        return v_impact_on_v

    # get weights
    def get_weights(self, type = 'weights'):
        
        weights = {}
        
        if type == 'weights':
            W = self.machine.W
            A = self.machine.A
            
            for v, v_label in enumerate(self.machine.v['label']):
                for h, h_label in enumerate(self.machine.h['label']):
                    if not A[v, h]:
                        continue
                    
                    weights[(h_label, v_label)] = W[v, h]
                    weights[(v_label, h_label)] = W[v, h]
        elif type == 'link_energy':
            weights_directed = self.machine.link_energy(self.dataset.data)
            
            # make weights symmetric
            for (n1, n2) in weights_directed:
                if (n1, n2) in weights:
                    continue
                
                weights[(n1, n2)] = weights_directed[(n1, n2)]
                weights[(n2, n1)] = weights_directed[(n1, n2)]
        else:
            return None
        
        return weights

##
##class GRBM_plot:
##
##    def __init__(self):
##        
##        # initialize logger
##        self.logger = logging.getLogger('metapath')
##        
##        self.label = {
##            'energy': 'Energy = $- \sum \\frac{1}{2 \sigma_i^2}(v_i - b_i)^2 ' +\
##                '- \sum \\frac{1}{\sigma_i^2} w_{ij} v_i h_j ' +\
##                '- \sum c_j h_j$',
##            'error': 'Error = $\sum (data - p[v = data|\Theta])^2$'
##        }
##
##        self.density = 1
##        self.reset()
##
##        
##    def reset(self):
##        self.data = {
##            'epoch': np.empty(1),
##            'energy': np.empty(1),
##            'error': np.empty(1)
##        }
##
##        self.buffer = {
##            'energy': 0,
##            'error': 0
##        }
##
##        self.last_epoch = 0
##    
##    def set_density(self, updates, points):
##        self.density = max(int(updates / points), 1)
##    
##    def add(self, epoch, v_data, h_data, v_model, h_model, params):
##        
##        # calculate energy, error etc.
##        self.buffer['error'] += params.error(v_data)
##        self.buffer['energy'] += params.energy(v_data)
##        
##        if (epoch - self.last_epoch) % self.density == 0:
##            self.data['epoch'] = \
##                np.append(self.data['epoch'], epoch)
##            self.data['error'] = \
##                np.append(self.data['error'], self.buffer['error'] / self.density)
##            self.data['energy'] = \
##                np.append(self.data['energy'], self.buffer['energy'] / self.density)
##
##            # reset energy and error
##            self.buffer['error'] = 0
##            self.buffer['energy'] = 0
##
##    def save(self, path = None):
##        if path == None:
##            self.logger.error("no save path was given")
##            quit()
##
##        # create path if not available
##        if not os.path.exists(path):
##            os.makedirs(path)
##
##        for key, val in self.data.items():
##            if key == 'epoch':
##                continue
##
##            file_plot = '%s/%s.pdf' % (path, key.lower())
##
##            # get labels
##            xlabel = 'updates'
##            ylabel = key
##
##            plt.figure()
##            plt.plot(self.data['epoch'], val, 'b,')
##            plt.xlabel(xlabel)
##            plt.ylabel(ylabel)
##            plt.savefig(file_plot)

