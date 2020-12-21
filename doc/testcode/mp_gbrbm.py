import numpy as np

class machine:
    
    def __init__(self,
        visible = 0, hidden = 0, adjacency_matrix = None,
        std_dev = 0.25, weight_distribution = 0.3, data = None,
        create_plots = False, **kwargs):
        
        # initialize logger
        import logging
        self.logger = logging.getLogger('metapath')
        
        # check adjacency matrix
        if not hasattr(adjacency_matrix, 'shape'):
            if visible > 0 and hidden > 0:
                adjacency_matrix = np.ones((visible, hidden))
            else:
                self.logger.critical("you have to specify an adjacency matrix")
                quit()
        
        # initialize params
        self.params = gbrbm_params(data, adjacency_matrix, std_dev, weight_distribution)
        self.epoch = 0

        # initialize list for minima
        self.minima = []
        self.lowest_energy = 0

        # initialize arrays for plots
        self.create_plots = create_plots
        self.plot = gbrbm_plot()

    def reset(self):
        self.epoch = 0
        self.params.reset()
        self.plot.reset()

    #
    # MODEL OPTIMIZATION
    #

    def optimize(self, data, updates = 10000,
        sampling_algorithm = 'cd-k', sampling_steps = 3, sampling_iterations = 1,
        learning_rate = 0.0001, plot_points = 200,
        selection = 'last', **kwargs):

        # initialize sampling function         
        self.sample = {
            'cd': lambda data:
                self.sample_cd1(data),
            'cd-k': lambda data:
                self.sample_cdk(data, k = sampling_steps, m = sampling_iterations),
            'ml': lambda data:
                self.sample_ml(data, k = sampling_steps, m = sampling_iterations)
        }[sampling_algorithm.lower()]

        # initialize update rates
        self.params.set_update_rates(learning_rate)

        # initialize plot density
        if self.create_plots:
            self.plot.set_density(updates, plot_points)

##        sampling_algorithm_str = {
##            'cd': 'CD',
##            'cd-k': 'CD-k (%d, %d)' % (sampling_steps, sampling_iterations),
##            'ml': 'ML'
##        }[sampling_algorithm.lower()]

        # iterative optimization
        for epoch in xrange(1, updates + 1):
            # sample from data and update params
            v_data, h_data, v_model, h_model = self.sample(data)
            self.params.update(v_data, h_data, v_model, h_model)
            self.epoch += 1
            
##            # detect optima
##            if not selection == None:
##                energy = self.params.energy(v_data)
##                if energy < self.lowest_energy:
##                    self.lowest_energy = energy

##            # plots
##            if self.create_plots:
##                self.plot.add(self.epoch, v_data, h_data, v_model, h_model, self.params)
    
    #
    # GENERAL SAMPLING ALGORITHMS
    #

    #
    # (1-step) contrastive divergency sampling (CD)
    #

    def sample_cd1(self, v_data):
        h_data = self.params.h_expect(v_data)
        v_model = self.params.v_expect(self.params.h_sample(h_data))
        h_model = self.params.h_expect(v_model)

        return v_data, h_data, v_model, h_model

    #
    # k-step contrastive divergency sampling (CD-k)
    #

    def sample_cdk(self, v_data, k = 1, m = 1):
        h_data = self.params.h_expect(v_data)
        
        # calculate v_model and h_model
        v_model = np.zeros(shape = v_data.shape)
        h_model = np.zeros(shape = h_data.shape)
        for i in range(m):
            for j in range(k):
                
                # calculate h_sample from h_expect
                # in first sampling step init h_sample with h_data
                if j == 0:
                    h_sample = self.params.h_sample(h_data)
                else:
                    h_sample = self.params.h_sample(h_expect)

                # calculate v_expect from h_sample
                v_expect = self.params.v_expect(h_sample)

                # calculate h_expect from v_sample
                # in last sampling step use v_expect
                # instead of v_sample to reduce noise
                if j + 1 == k:
                    h_expect = self.params.h_expect(v_expect)
                else:
                    v_sample = self.params.v_sample(v_expect)
                    h_expect = self.params.h_expect(v_sample)

            v_model += v_expect / m
            h_model += h_expect / m

        return v_data, h_data, v_model, h_model

    #
    # maximum likelihood sampling (ML)
    # 

    def sample_ml(self, v_data, k = 3, m = 100):
        h_data = self.params.h_expect(v_data)
        
        # calculate v_model and h_model
        v_model = np.zeros(shape = v_data.shape)
        h_model = np.zeros(shape = h_data.shape)
        for i in range(m):
            for step in range(1, k + 1):
                
                # calculate h_sample from h_expect
                # in first sampling step init h_sample with random values
                if step == 1:
                    h_sample = self.params.h_random()
                else:
                    h_sample = self.params.h_sample(h_expect)

                # calculate v_expect from h_sample
                v_expect = self.params.v_expect(h_sample)

                # calculate h_expect from v_sample
                # in last sampling step use v_expect
                # instead of v_sample to reduce noise
                if step == k:
                    h_expect = self.params.h_expect(v_expect)
                else:
                    v_sample = self.params.v_sample(v_expect)
                    h_expect = self.params.h_expect(v_sample)

            v_model += v_expect / m
            h_model += h_expect / m

        return v_data, h_data, v_model, h_model

class gbrbm_params:
    
    def __init__(self, data = None, adjacency_matrix = None, std_dev = 0.25, range = 0.3):
        
        # initialize logger
        import logging
        self.logger = logging.getLogger('metapath')
        
        # check adjacency matrix
        if adjacency_matrix == None or not hasattr(adjacency_matrix, 'shape'):
            self.logger.error("adjacency matrix is not a numpy array!")
            quit()
        
        # check data matrix
        if data == None or not hasattr(data, 'shape'):
            self.logger.error("data matrix is not a numpy array!")
            quit()
        
        # initialize adjacency matrix and number of visible and hidden units
        self.A = adjacency_matrix
        self.v_size = self.A.shape[0]
        self.h_size = self.A.shape[1]
        
        # initializa data properties
        self.data_size = data.shape[0]
        self.v_mean_data = np.mean(data, axis = 0).reshape(1, self.A.shape[0])
        self.data_sdev = np.std(data, axis = 0).reshape(1, self.A.shape[0])
        
        # initialize reinit parameters
        self.init_W_range = range
        self.init_sdev = std_dev

        # initialize default update rates
        self.set_update_rates()
        
        # initialize params
        self.reset()
        
    def set_update_rates(self, learning_rate = 0.0001,
        learning_factor_weights = 1.0, learning_factor_vbias = 0.1,
        learning_factor_hbias = 0.1, learning_factor_vlvar = 0.01):
        
        self.W_rate = learning_rate * learning_factor_weights
        self.v_bias_rate = learning_rate * learning_factor_vbias
        self.v_lvar_rate = learning_rate * learning_factor_vlvar
        self.h_bias_rate = learning_rate * learning_factor_hbias
    
    def reset(self):
        # (re)initialize params
        self.W = self.init_W()
        self.v_bias = self.init_v_bias()
        self.v_lvar = self.init_v_lvar()
        self.h_bias = self.init_h_bias()
    
        # (re)initialize model evaluation quantities
        self.v_mean_model = self.init_v_mean_model()
        self.h_mean_model = self.init_h_mean_model()
    
    #
    # MODEL INITIALIZATION
    #
        
    # input: none
    # output: weight matrix with normal distributed random values
    # dimensions: (visible, hidden)
    def init_W(self):
        return self.A * np.random.normal(np.zeros(shape = self.A.shape),
            self.init_W_range * self.data_sdev.T)

    # input: none
    # output: mean values of data
    # dimensions: (1, visible)
    def init_v_bias(self):
        return self.v_mean_data

    # init visible units logarithmic variances with given standard deviation
    # dimensions: (1, visible)
    def init_v_lvar(self):
        return np.log((self.init_sdev * np.ones((1, self.A.shape[0]))) ** 2)

    # init hidden units biases with zeros
    # dimensions: (1, hidden)
    def init_h_bias(self):
        return np.zeros((1, self.A.shape[1]))

    #
    # UNIT RECONSTRUCTION
    #

    # input: samples of hidden units
    # output: expected values for visible units
    def v_expect(self, h_sample):
        return self.v_bias + np.dot(h_sample, self.W.T)
    
    # input: expected values for visible units
    # output: samples of visible units (gauss distribution)
    def v_sample(self, v_expect):
        return np.random.normal(v_expect, np.exp(self.v_lvar))
    
    # input: none
    # output: random samples of visible units (gauss distribution)
    def v_random(self):
        return np.random.normal(self.v_bias + \
            np.dot(self.h_random, self.W.T), np.exp(self.v_lvar))

    # input: samples of visible units
    # output: expected values of hidden units
    def h_expect(self, v_sample):
        return 1.0 / (1 + np.exp(-(self.h_bias +
            np.dot(v_sample / np.exp(self.v_lvar), self.W))))

    # input: expected values of hidden units
    # output: samples of hidden units (bernoulli distribution)
    def h_sample(self, h_expect):
        return h_expect > np.random.rand(h_expect.shape[0], h_expect.shape[1])

    # input: none
    # output: random samples of hidden units (bernoulli distribution)
    def h_random(self):
        return 1.0 / (1 + np.exp(-(self.h_bias))) > \
            np.random.rand(self.data_size, self.h_size)

    #
    # PARAMETER UPDATE FUNCTIONS
    #

    # update params
    def update(self, v_data, h_data, v_model, h_model):
        # calculate update for each param
        delta_W = self.delta_W(v_data, h_data, v_model, h_model)
        delta_v_bias = self.delta_v_bias(v_data, v_model)
        delta_h_bias = self.delta_h_bias(h_data, h_model)
        delta_v_lvar = self.delta_v_lvar(v_data, h_data, v_model, h_model)
        
        # update params
        self.W += self.W_rate * delta_W
        self.v_bias += self.v_bias_rate * delta_v_bias
        self.h_bias += self.h_bias_rate * delta_h_bias
        self.v_lvar += self.v_lvar_rate * delta_v_lvar

    # update rule for weight matrix
    def delta_W(self, v_data, h_data, v_model, h_model):
        data = np.dot(v_data.T, h_data) / v_data.shape[0]
        model = np.dot(v_model.T, h_model) / v_model.shape[0]
        return (data - model) * self.A / np.exp(self.v_lvar).T

    # update rule for visible units biases
    def delta_v_bias(self, v_data, v_model):
        data = np.mean(v_data, axis = 0).reshape(self.v_bias.shape)
        model = np.mean(v_model, axis = 0).reshape(self.v_bias.shape)
        return (data - model) / np.exp(self.v_lvar)

    # update rule for hidden units biases
    def delta_h_bias(self, h_data, h_model):
        data = np.mean(h_data, axis = 0).reshape(self.h_bias.shape)
        model = np.mean(h_model, axis = 0).reshape(self.h_bias.shape)
        return data - model

    # update rule for visible units logarithmic variance
    def delta_v_lvar(self, v_data, h_data, v_model, h_model):
        data = np.mean(0.5 * (v_data - self.v_bias) ** 2 - v_data *
            np.dot(h_data, self.W.T), axis = 0).reshape(self.v_lvar.shape)
        model = np.mean(0.5 * (v_model - self.v_bias) ** 2 - v_model *
            np.dot(h_model, self.W.T), axis = 0).reshape(self.v_lvar.shape)
        return (data - model) / np.exp(self.v_lvar)
   
    #
    # MODEL VALIDATION
    #
    
    # 
    # unit reconstruction error using cd-k (3, 100)
    #
    
    def v_error(self, v_data, k = 3, m = 100):
        h_data = self.h_expect(v_data)
        
        # allocate reconstruction array
        v_recon = np.ndarray(shape = (m, v_data.shape[0], v_data.shape[1]))

        # sample from data
        for i in range(m):
            h_sample = self.h_sample(h_data)
            v_expect = self.v_expect(h_sample)
            
            for j in range(k):
                
                # calculate h_sample from h_expect
                if j > 0:
                    h_sample = self.h_sample(h_expect)
                    v_expect = self.v_expect(h_sample)

                # calculate h_expect from v_sample
                if j + 1 < k:
                    v_sample = self.v_sample(v_expect)
                    h_expect = self.h_expect(v_sample)
            
            v_recon[i] = v_expect
        
        # calculate mean error and mean variance per sample and unit
        v_err_sample = np.abs(v_data - np.mean(v_recon, axis = 0))
        v_std_sample = np.std(v_recon, axis = 0)
        
        # calculate mean error and mean variance per unit
        v_err = np.mean(v_err_sample, axis = 0)
        v_std = np.sqrt(np.sum(v_std_sample ** 2, axis = 0))
        
        return v_err, v_std

    # 
    # model reconstruction error using cd-k (3, 100)
    #

    def error(self, v_data):
        # get mean error and mean variance per unit
        v_err, v_std = self.v_error(v_data)
        
        # calculate mean error and mean variance
        err = np.mean(v_err)
        std = np.sqrt(np.sum(v_std ** 2))
        
        return err, std
    
    #
    # model energy
    #

    def energy(self, v_data):
        h_data = self.h_expect(v_data)
        v_term = np.sum((v_data - self.v_bias) ** 2 / np.exp(self.v_lvar)) / 2
        h_term = np.sum(h_data * self.h_bias)
        W_term = np.sum(v_data * np.dot(h_data, self.W.T) / np.exp(self.v_lvar))
        return - (v_term + h_term + W_term)

    # input: none
    # output:
    # dimensions: (1, visible)
    def init_v_mean_model(self):
        return self.v_mean_data

    # input: none
    # output:
    # dimensions: (1, hidden)
    def init_h_mean_model(self):
        return np.zeros((1, self.A.shape[1]))

    # input: none
    # output: maximum likelihood expected values
    def update_expect_ml(self, k = 4, m = 1000):
        
        # calculate v_model and h_model
        v_model = np.zeros(shape = (self.data_size, self.v_size))
        h_model = np.zeros(shape = (self.data_size, self.h_size))
        
        for i in range(m):
            for step in range(1, k + 1):
                
                # calculate h_sample from h_expect
                if step == 1:
                    h_sample = self.h_random()
                else:
                    h_sample = self.h_sample(h_expect)
                    
                # calculate v_expect from h_sample
                v_expect = self.v_expect(h_sample)
                
                # calculate h_expect from v_sample
                # in last sampling step use v_expect
                # instead of v_sample to reduce noise
                if step == k:
                    h_expect = self.h_expect(v_expect)
                else:
                    v_sample = self.v_sample(v_expect)
                    h_expect = self.h_expect(v_sample)
                    
            v_model += v_expect / m
            h_model += h_expect / m
            
        v_mean = np.mean(v_model, axis = 0).reshape(1, v_model.shape[1])
        h_mean = np.mean(h_model, axis = 0).reshape(1, h_model.shape[1])
        
        self.v_mean_model = v_mean
        self.h_mean_model = h_mean

    #
    # model validation functions
    #



    # calculate error
    #def error(self, v_data, v_model):
    #    return np.sum((v_data - v_model) ** 2)
    
    # calculate maximum likelihood error
    def ml_error(self):
        self.update_expect_ml()
        return np.sum((self.v_mean_data - self.v_mean_model) ** 2)

    #
    # MODEL PARAMETER INTERFACES
    #
    
    # get all params as dict
    def get(self):
        self.update_expect_ml()
        
        params = {
            'A': self.A,
            'W': self.W,
            'v_bias': self.v_bias,
            'h_bias': self.h_bias,
            'v_lvar': self.v_lvar,
            'v_mean_model': self.v_mean_model,
            'h_mean_model': self.h_mean_model,
            'v_mean_data': self.v_mean_data
        }
        
        return params

    # set all params use a dict
    def set(self, params):
        self.A = params['A']
        self.W = params['W']
        self.v_bias = params['v_bias']
        self.v_lvar = params['v_lvar']
        self.h_bias = params['h_bias']
        self.v_mean_model = params['v_mean_model']
        self.h_mean_model = params['h_mean_model']
        self.v_mean_data = params['v_mean_data']

    # save params to file (npz, txt)
    def save(self, file = None, format = 'npz'):
        if file == None:
            self.logger.error("no file was given!")
            quit()
            
        # create path if not available
        import os
        if not os.path.exists(os.path.dirname(file)):
            os.makedirs(os.path.dirname(file))
        
        params = self.get()
        
        self.logger.info("saving model parameters to " + file)
        if format == 'npz':
            np.savez(file, **params)
        elif format == 'txt':
            folder = os.path.dirname(file)
            for param in params:
                np.savetxt(folder + '/' + param + '.txt',
                    params[param], delimiter = '\t', fmt='%.5f')
        
    # load params from file (npz)
    def load(self, file = None):
        if file == None:
            self.logger.error("no file was given!")
            quit()
            
        params = np.load(file)
        self.set(params) 

class gbrbm_plot:

    def __init__(self):
        
        # initialize logger
        import logging
        self.logger = logging.getLogger('metapath')
        
        self.label = {
            'energy': 'Energy = $- \sum \\frac{1}{2 \sigma_i^2}(v_i - b_i)^2 ' +\
                '- \sum \\frac{1}{\sigma_i^2} w_{ij} v_i h_j ' +\
                '- \sum c_j h_j$',
            'error': 'Error = $\sum (data - p[v = data|\Theta])^2$'
        }

        self.density = 1
        self.reset()

        
    def reset(self):
        self.data = {
            'epoch': np.empty(1),
            'energy': np.empty(1),
            'error': np.empty(1)
        }

        self.buffer = {
            'energy': 0,
            'error': 0
        }

        self.last_epoch = 0
    
    def set_density(self, updates, points):
        self.density = max(int(updates / points), 1)
    
    def add(self, epoch, v_data, h_data, v_model, h_model, params):
        
        # calculate energy, error etc.
        self.buffer['error'] += params.error(v_data)
        self.buffer['energy'] += params.energy(v_data)
        
        if (epoch - self.last_epoch) % self.density == 0:
            self.data['epoch'] = \
                np.append(self.data['epoch'], epoch)
            self.data['error'] = \
                np.append(self.data['error'], self.buffer['error'] / self.density)
            self.data['energy'] = \
                np.append(self.data['energy'], self.buffer['energy'] / self.density)

            # reset energy and error
            self.buffer['error'] = 0
            self.buffer['energy'] = 0

    def save(self, path = None):
        if path == None:
            self.logger.error("no save path was given")
            quit()

        # create path if not available
        import os
        if not os.path.exists(path):
            os.makedirs(path)

        # check if python module 'pyplot' is available
        try:
            import matplotlib.pyplot as plt
        except:
            self.logger.error("could not import python module 'pyplot'")
            quit()

        # everything seems to be fine
        self.logger.info("saving training plots: %s" % (path))

        for key, val in self.data.items():
            if key == 'epoch':
                continue

            file_plot = '%s/%s.pdf' % (path, key.lower())

            # get labels
            xlabel = 'updates'
            ylabel = key

            plt.figure()
            plt.plot(self.data['epoch'], val, 'b,')
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.savefig(file_plot)

if __name__ == '__main__':
    gbrbm = gbrbm()

