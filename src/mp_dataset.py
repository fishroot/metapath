import logging
import numpy as np
import copy
import time

class mp_dataset:
    
    id     = None
    data   = None
    sub    = {}
    logger = None
    cfg    = {}

    def __init__(self, config = None):
        
        # initialize id
        self.id = int(time.time() * 100)
        
        # initialize logger
        self.logger = logging.getLogger('metapath')
        
        # create empty dataset instance if no configuration is passed
        if config == None:
            return
        
        # reference configuration
        self.cfg  = config

    def normalize(self, type = None):
        
        # get type of normalization
        if type == None:
            if 'preprocessing' in self.cfg \
                and 'normalize' in self.cfg['preprocessing']:
                type = self.cfg['preprocessing']['normalize']
        
        # create console message
        if (type == None or type.lower() == 'none'):
            self.logger.info(" * using plain dataset (no normalization)")
        else:
            self.logger.info(" * normalizing dataset: '" + type + "'")
        
        # normalize data
        if type == 'mean':
            self.data = self.data - np.mean(self.data, axis = 0)
            return True
        if type == 'scale':
            self.data = self.data / np.mean(self.data, axis = 0) - 1
        if type == 'var':
            self.data = self.data / np.var(self.data, axis = 0)
            return True
        if type == 'z-trans':
            self.data = \
                (self.data - np.mean(self.data, axis = 0)) \
                / np.var(self.data, axis = 0)
            return True

        return False

    def update_from_source(self):
        file = self.cfg['file']
        format = self.cfg['file_format']
        options = self.cfg['file_options']

        self.data = np.loadtxt(file, **options)

    def save(self, file = ''):
        np.savez(file, cfg = self.cfg, data = self.data, sub = self.sub)

    def load(self, file = ''):
        
        npzfile = np.load(file)
        self.cfg = npzfile['cfg'].item()
        self.data = npzfile['data']
        self.sub = npzfile['sub'].item()

    def subset(self, label_format = 'alias', labels = {}):
        
        # make copy of self
        dataset = copy.copy(self)
        
        #
        # ANNOTATION
        #
        
        # convert network and dataset geneIDs to EntrezIDs
        # using R/bioconductor

        from src.mp_R_wrapper import bioconductor
        bioc = bioconductor()

        # mapping network labels to EntrezIDs
        self.logger.info("   (1) mapping network node labels to EntrezIDs")
        
        network_labels = labels
        network_format = label_format
        network_list_IDs = {}
        network_list_lost = {}
        network_IDs = []
        network_lost = []
        
        for l in labels:
            network_list_IDs[l], network_list_lost[l] = bioc.convert_geneids(
                network_labels[l], network_format, 'entrezid')
            network_IDs += network_list_IDs[l]
            network_lost += network_list_lost[l]
        
        #if network_lost:
        #    self.logger.warning(" * %i of %i network labels could not be mapped to EntrezIDs: %s" % (
        #        len(network_lost), len(network_IDs), ",".join(network_lost)))

        # mapping dataset labels to EntrezIDs
        self.logger.info("   (2) mapping dataset column labels to EntrezIDs")
        
        dataset_IDs, dataset_lost = bioc.convert_geneids(
            dataset.cfg['label'], dataset.cfg['label_format'], 'entrezid')
        
        #if dataset_lost:
        #    self.logger.warning(" * %i of %i dataset labels could not be assigned to EntrezIDs: %s" % (
        #        len(dataset_lost), len(dataset_IDs), ','.join(dataset_lost)))

        # intersect EntrezIDs
        self.logger.info("   (3) intersecting network labels with dataset labels via assigned EntrezIDs")

        columns = []
        columns_not_found = []
        dataset_label = []
        subset = {}
        
        for l in labels:
            subset[l] = []
            for index, geneid in enumerate(network_list_IDs[l]):
                label = network_labels[l][index]
                
                if geneid in dataset_IDs:
                    source_culumn_id = dataset_IDs.index(geneid)
                    columns.append(source_culumn_id)
                    dataset_label.append(label)
                    subset[l].append(label)
                else:
                    columns_not_found.append(label)

        if columns_not_found:
            self.logger.warning(" * %i of %i genes could not be found in dataset: %s" % \
                (len(columns_not_found), len(network_IDs), ','.join(columns_not_found)))

        # set the list of columns in dataset according to the search results
        self.cfg['file_options']['usecols'] = tuple(columns)
        
        # set label and label format
        dataset.cfg['label'] = dataset_label
        dataset.cfg['label_format'] = network_format
        dataset.sub = subset
        
        return dataset

    def get(self):

        return {
            'id': self.id,
            'data': self.data,
            'sub': self.sub,
            'cfg': self.cfg
        }
    
    def set(self, **params):
        
        if 'id' in params:
            self.id = params['id']
        if 'data' in params:
            self.data = params['data']
        if 'sub' in params:
            self.sub = params['sub']
        if 'cfg' in params:
            self.cfg = params['cfg']
        
        return True