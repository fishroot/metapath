import numpy as np


class gbrbm:

    def __init__(self, visible = 0, hidden = 0,
        weights = 0.1, vbias = 0, stddev = 0.25, hbias = 0,
        adjacency_matrix = None, create_plots = False):

        # get dimensions (number of visible and hidden units)
        if hasattr(adjacency_matrix, 'shape'):
            visible = adjacency_matrix.shape[0]
            hidden = adjacency_matrix.shape[1]
        
        if not (visible > 0 and hidden > 0):
            print "gbrbm.__init__: number of visible and hidden units have to be > 0"
            quit()

        # initialize adjacency matrix
        # dimensions (visible, hidden)
        if hasattr(adjacency_matrix, 'shape') \
            and adjacency_matrix.shape == (visible, hidden):
            self.A = adjacency_matrix
        else:
            self.A = np.ones((visible, hidden))

        # initialize weight matrix
        # dimensions (visible, hidden)
        if hasattr(weights, 'shape') \
            and weights.shape == (visible, hidden):
            self.W = weights
        else:
            self.W = weights * np.random.randn(visible, hidden)

        # initialize bias of visible units
        # dimensions (1, visible)
        try:
            vbias *= np.ones((1, visible))
            self.v_bias = vbias
        except:
            print "gbrbm.__init__: param 'vbias' has no valid dimension (1 x visible)"
            quit()
        
        # initialize bias of hidden units
        # dimensions (1, hidden)
        try:
            hbias *= np.ones((1, hidden))
            self.h_bias = hbias
        except:
            print "gbrbm.__init__: param 'hbias' has no valid dimension (1 x hidden)"
            quit()

        # initialize logarithmic variance of visible units
        # dimensions (1, visible)
        try:
            stddev *= np.ones((1, visible))
            self.v_lvar = np.log(stddev ** 2)
        except:
            print "gbrbm.__init__: param 'stddev' has no valid dimension (1 x visible)"
            quit()

        # initialize arrays for plots
        self.create_plots = create_plots
        self.plot = {}
        if create_plots:
            self.plot['x'] = np.empty(1)
            self.plot['Energy'] = np.empty(1)
            self.plot['_Energy'] = \
                "Energy = $- \sum \\frac{1}{2 \sigma_i^2}(v_i - b_i)^2 " +\
                "- \sum \\frac{1}{\sigma_i^2} w_{ij} v_i h_j " +\
                "- \sum c_j h_j$"
            self.plot['Error'] = np.empty(1)
            self.plot['_Error'] = \
                "Error = $\sum (data - p[v = data|\Theta])^2$"
            self.plot['Update'] = np.empty(1)
            self.plot['_Update'] = \
                "Update = $\\lambda \\cdot \\left|\\left|\\Delta W\\right|\\right|$"
        
        self.results = {}


    #
    # iterative training
    #

    def train(self, data, epochs = 10000,
        method = 'cdn', sampling_steps = 3, sampling_stat = 1,
        learning_rate = 0.025,
        learning_factor_weights = 1.0, learning_factor_vbias = 0.1,
        learning_factor_hbias = 0.1, learning_factor_vlvar = 0.01,
        plot_points = 200):
        
        # check if python module 'time' is available
        try:
            import time
        except:
            print "mp_gbrbm.train: could not import python module 'time'"
            quit()

        # initialize learning rates for weights, biases and logvar
        # using relative factors
        self.W_rate = learning_rate * learning_factor_weights
        self.v_bias_rate = learning_rate * learning_factor_vbias
        self.v_lvar_rate = learning_rate * learning_factor_vlvar
        self.h_bias_rate = learning_rate * learning_factor_hbias

        # initialize weights with respect to adjacence
        self.W *= self.A

        # initialize start time
        self.time_start = time.clock()

        if self.create_plots:
            # initialize epoch offset
            if self.plot['x'].shape[0] == 1:
                epoch_offset = 0
            else:
                epoch_offset = \
                    self.plot['x'][self.plot['x'].shape[0] - 1]

            # initialize values for plots
            count = max(int(epochs / plot_points), 1)
            error = 0
            energy = 0
            update = 0

        estim_epoch = min(200, epochs)

        # define sampling function
        sample = {
            'cd': lambda data:
                self.sample_cd(data),
            'cdn': lambda data:
                self.sample_cdn(data, sampling_steps, sampling_stat),
            'ml': lambda data:
                self.sample_ml(data)
        }[method]

        method_str = {
            'cd': 'CD',
            'cdn': 'CD-k (%d, %d)' % (sampling_steps, sampling_stat),
            'ml': 'ML'
        }[method]

        # main loop
        for epoch in xrange(1, epochs + 1):
            # use prefered sampling method
            v_data, h_data, v_model, h_model = sample(data)

            # estimate time
            if epoch == estim_epoch:
                delta = time.clock() - self.time_start
                estim_time = delta * epochs / float(epoch)
                print "...training %s epochs with %s, est. time: %.2fs" % \
                    (epochs, method_str, estim_time)

            # arrays for plots
            if self.create_plots:
                # calculate energy, error etc.
                error += self.error(v_data, v_model)
                energy += self.energy(v_model, h_model)
                update += self.norm_delta_W(v_data, h_data, v_model, h_model)

                # insert data for plots
                if epoch % count == 0:
                    self.plot['x'] = \
                        np.append(self.plot['x'], epoch + epoch_offset)
                    self.plot['Error'] = \
                        np.append(self.plot['Error'], error / count)
                    self.plot['Energy'] = \
                        np.append(self.plot['Energy'], energy / count)
                    self.plot['Update'] = \
                        np.append(self.plot['Update'], update / count)

                    # reset energy and error
                    error = 0
                    energy = 0
                    update = 0

            # update params
            self.update_params(v_data, h_data, v_model, h_model)

        #copy results
        self.results['weights'] = self.W
        self.results['vbias'] = self.v_bias
        self.results['hbias'] = self.h_bias
        self.results['vsdev'] = np.sqrt(np.exp(self.v_lvar))

    #
    # sampling
    #

    #
    # contrastive divergency sampling (CD)
    #

    def sample_cd(self, data):
        v_data = data
        h_data = self.h_expect(v_data)
        v_model = self.v_expect(self.h_values(h_data))
        h_model = self.h_expect(v_model)

        return v_data, h_data, v_model, h_model

    #
    # k-step contrastive divergency sampling (CD-k)
    #

    def sample_cdn(self, data, n = 1, m = 1):
        v_data = data
        h_data = self.h_expect(data)
        v_model = np.zeros(shape = v_data.shape)
        h_model = np.zeros(shape = h_data.shape)

        for i in range(m):
            for step in range(1, n + 1):
                if step == 1:
                    h_values = self.h_values(h_data)
                else:
                    h_values = self.h_values(h_expect)

                v_expect = self.v_expect(h_values)

                if step < n:
                    v_values = self.v_values(v_expect)
                    h_expect = self.h_expect(v_values)
                else:
                    h_expect = self.h_expect(v_expect)

            v_model += v_expect
            h_model += h_expect

        v_model /= m
        h_model /= m

        return v_data, h_data, v_model, h_model

    #
    # persistent contrastive divergency sampling (persistentCD)
    # TODO: implement

    def sample_persistentCD(self, data, n = 1, m = 1):
        return

    #
    # maximum likelihood sampling (ML)
    # TODO: don't sample from visible units

    def sample_ml(self, data, n = 5, m = 10):
        v_data = data
        h_data = self.h_expect(data)
        v_model = np.zeros(shape = v_data.shape)
        h_model = np.zeros(shape = h_data.shape)

        for i in range(m):
            for step in range(1, n + 1):
                if step == 1:
                    h_values = self.h_values(h_data)
                else:
                    h_values = self.h_values(h_expect)

                v_expect = self.v_expect(h_values)

                if step < n:
                    v_values = self.v_values(v_expect)
                    h_expect = self.h_expect(v_values)
                else:
                    h_expect = self.h_expect(v_expect)

            v_model += v_expect
            h_model += h_expect

        v_model /= m
        h_model /= m

        return v_data, h_data, v_model, h_model

    #
    # unit reconstruction
    #

    # expected values of visible units
    def v_expect(self, h_values):
        return self.v_bias + np.dot(h_values, self.W.T)

    # gauss distributed random values of visible units
    def v_values(self, expect):
        return np.random.normal(expect, np.exp(self.v_lvar))

    # expected values of hidden units
    def h_expect(self, v_values):
        return 1.0 / (1 + np.exp(-(self.h_bias +
            np.dot(v_values / np.exp(self.v_lvar), self.W))))

    # bernoulli distributed random values of hidden units
    def h_values(self, expect):
        return expect > np.random.rand(expect.shape[0], expect.shape[1])

    #
    # update params
    #

    # calculate all deltas using same params and update
    def update_params(self, v_data, h_data, v_model, h_model):
        delta_W = self.delta_W(v_data, h_data, v_model, h_model)
        delta_v_bias = self.delta_v_bias(v_data, v_model)
        delta_h_bias = self.delta_h_bias(h_data, h_model)
        delta_v_lvar = self.delta_v_lvar(v_data, h_data, v_model, h_model)
        self.W += self.W_rate * delta_W
        self.v_bias += self.v_bias_rate * delta_v_bias
        self.h_bias += self.h_bias_rate * delta_h_bias
        self.v_lvar += self.v_lvar_rate * delta_v_lvar

    # update rule for weight matrix
    def delta_W(self, v_data, h_data, v_model, h_model):
        data = np.dot(v_data.T, h_data) / v_data.shape[0]
        model = np.dot(v_model.T, h_model) / v_model.shape[0]
        delta_W = (data - model) * self.A / np.exp(self.v_lvar).T
        return delta_W

    # update rule for visible units biases
    def delta_v_bias(self, v_data, v_model):
        data = np.mean(v_data, axis = 0).reshape(self.v_bias.shape)
        model = np.mean(v_model, axis = 0).reshape(self.v_bias.shape)
        delta_v_bias = (data - model) / np.exp(self.v_lvar)
        return delta_v_bias

    # update rule for hidden units biases
    def delta_h_bias(self, h_data, h_model):
        data = np.mean(h_data, axis = 0).reshape(self.h_bias.shape)
        model = np.mean(h_model, axis = 0).reshape(self.h_bias.shape)
        delta_h_bias = data - model
        return delta_h_bias

    # update rule for visible units logarithmic variance
    def delta_v_lvar(self, v_data, h_data, v_model, h_model):
        data = np.mean(0.5 * (v_data - self.v_bias) ** 2 - v_data *
            np.dot(h_data, self.W.T), axis = 0).reshape(self.v_lvar.shape)
        model = np.mean(0.5 * (v_model - self.v_bias) ** 2 - v_model *
            np.dot(h_model, self.W.T), axis = 0).reshape(self.v_lvar.shape)
        delta_v_lvar = (data - model) / np.exp(self.v_lvar)
        return delta_v_lvar

    #
    # energy, error etc.
    #

    # calculate energy
    def energy(self, v_model, h_model):
        v_term = np.sum((v_model - self.v_bias) ** 2 / np.exp(self.v_lvar)) / 2
        h_term = np.sum(h_model * self.h_bias)
        W_term = np.sum(v_model * np.dot(h_model, self.W.T) / np.exp(self.v_lvar))
        energy = - (v_term + h_term + W_term)
        return energy

    # calculate error
    def error(self, v_data, v_model):
        error = np.sum((v_data - v_model) ** 2)
        return error

    # calculate update
    def norm_delta_W(self, v_data, h_data, v_model, h_model):
        delta_W = self.delta_W(v_data, h_data, v_model, h_model)
        norm = np.linalg.norm(delta_W)
        return self.W_rate * norm


##  def run_visible(self, data):
##    """
##    Assuming the RBM has been trained (so that weights for the network have been learned),
##    run the network on a set of visible units, to get a sample of the hidden units.
##
##    Parameters
##    ----------
##    data: A matrix where each row consists of the states of the visible units.
##
##    Returns
##    -------
##    hidden_states: A matrix where each row consists of the hidden units activated from the visible
##    units in the data matrix passed in.
##    """
##
##    # get Number of training samples
##    int_samples = data.shape[0]
##
##    # Create a matrix, where each row is to be the hidden units (plus a bias unit)
##    # sampled from a training example.
##    hidden_states = np.ones((int_samples, self.hidden + 1))
##
##    # Insert bias units of 1 into the first column of data.
##    data = np.insert(data, 0, 1, axis = 1)
##
##    # Calculate the activations of the hidden units.
##    hidden_activations = np.dot(data, self.weights)
##    # Calculate the probabilities of turning the hidden units on.
##    hidden_probs = self._logistic(hidden_activations)
##    # Turn the hidden units on with their specified probabilities.
##    hidden_states[:,:] = hidden_probs > np.random.rand(int_samples, self.hidden + 1)
##    # Always fix the bias unit to 1.
##    # hidden_states[:,0] = 1
##
##    # Ignore the bias units.
##    hidden_states = hidden_states[:,1:]
##    return hidden_states
##
##  # TODO: Remove the code duplication between this method and `run_visible`?
##  def run_hidden(self, data):
##    """
##    Assuming the RBM has been trained (so that weights for the network have been learned),
##    run the network on a set of hidden units, to get a sample of the visible units.
##
##    Parameters
##    ----------
##    data: A matrix where each row consists of the states of the hidden units.
##
##    Returns
##    -------
##    visible_states: A matrix where each row consists of the visible units activated from the hidden
##    units in the data matrix passed in.
##    """
##
##    # get Number of training samples
##    int_samples = data.shape[0]
##
##    # Create a matrix, where each row is to be the visible units (plus a bias unit)
##    # sampled from a training example.
##    visible_states = np.ones((int_samples, self.visible + 1))
##
##    # Insert bias units of 1 into the first column of data.
##    data = np.insert(data, 0, 1, axis = 1)
##
##    # Calculate the activations of the visible units.
##    visible_activations = np.dot(data, self.weights.T)
##    # Calculate the probabilities of turning the visible units on.
##    visible_probs = self._logistic(visible_activations)
##    # Turn the visible units on with their specified probabilities.
##    visible_states[:,:] = visible_probs > np.random.rand(int_samples, self.visible + 1)
##    # Always fix the bias unit to 1.
##    # visible_states[:,0] = 1
##
##    # Ignore the bias units.
##    visible_states = visible_states[:,1:]
##    return visible_states
##
##  def daydream(self, num_samples):
##    """
##    Randomly initialize the visible units once, and start running alternating Gibbs sampling steps
##    (where each step consists of updating all the hidden units, and then updating all of the visible units),
##    taking a sample of the visible units at each step.
##    Note that we only initialize the network *once*, so these samples are correlated.
##
##    Returns
##    -------
##    samples: A matrix, where each row is a sample of the visible units produced while the network was
##    daydreaming.
##    """
##
##    # Create a matrix, where each row is to be a sample of of the visible units
##    # (with an extra bias unit), initialized to all ones.
##    samples = np.ones((num_samples, self.visible + 1))
##
##    # Take the first sample from a uniform distribution.
##    samples[0,1:] = np.random.rand(self.visible)
##
##    # Start the alternating Gibbs sampling.
##    # Note that we keep the hidden units binary states, but leave the
##    # visible units as real probabilities. See section 3 of Hinton's
##    # "A Practical Guide to Training Restricted Boltzmann Machines"
##    # for more on why.
##    for i in range(1, num_samples):
##      visible = samples[i-1,:]
##
##      # Calculate the activations of the hidden units.
##      hidden_activations = np.dot(visible, self.weights)
##      # Calculate the probabilities of turning the hidden units on.
##      hidden_probs = self._logistic(hidden_activations)
##      # Turn the hidden units on with their specified probabilities.
##      hidden_states = hidden_probs > np.random.rand(self.hidden + 1)
##      # Always fix the bias unit to 1.
##      hidden_states[0] = 1
##
##      # Recalculate the probabilities that the visible units are on.
##      visible_activations = np.dot(hidden_states, self.weights.T)
##      visible_probs = self._logistic(visible_activations)
##      visible_states = visible_probs > np.random.rand(self.visible + 1)
##      samples[i,:] = visible_states
##
##    # Ignore the bias units (the first column), since they're always set to 1.
##    return samples[:,1:]

#if __name__ == '__main__':
#    sdev = 0.025 * np.ones((1, 4))
#    gbrbm = gbrbm(1, 3, stddev = 0.5)

