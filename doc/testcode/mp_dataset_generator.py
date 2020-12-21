import numpy as np
import ConfigParser
import os

class mp_dataset_generator:

    def __init__(self, count = 0, num_e = 0, num_s = 0, dataset_file = ''):

        if (count == 0 or num_e == 0 or num_s == 0):
            print 'error'
            quit()

        num_vis = num_e + num_s

        data = np.random.randn(count, num_vis)
        for i in range(data.shape[0]):
            data[i,11] = data[i,5]
            data[i,6] = data[i,0]
        np.savetxt(dataset_file, data, delimiter = ';')
        
if __name__ == '__main__':
    create = mp_dataset_generator('test.csv')