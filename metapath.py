#!/usr/bin/env python

import argparse
from src.mp_metapath import metapath

def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', '-w', dest = 'workspace',
        help = 'ini file, containing metapath workspace (default: default.ini)')
    parser.add_argument('--model', '-m', dest = 'model',
        help = 'name of model ("GRBM")')
    parser.add_argument('--network', '-n', dest = 'network',
        help = 'name of network')
    parser.add_argument('--dataset', '-d', dest = 'dataset',
        help = 'name of dataset')
    parser.add_argument('--schedule', '-s', dest = 'schedule',
        help = 'name of optimization schedule')
    parser.add_argument('--analyse', '-a', dest = 'analyse',
        help = 'name of collection of analysis')
    args = parser.parse_args()
    
    # create metapath instance
    mp = metapath(args.workspace, default = True)
    
    # load model
    model = mp.load('./reports/default/model-134210344929/model-GRBM.mp')
    
    # analyse model
    #mp.analyse(
    #    model,
    #    name = args.analyse)
    
    #quit()
    
##    # create new model
##    model = mp.model(
##        name    = 'GRBM',
##        network = 'Smallest',
##        dataset = 'TCGA-normal')
##    
##    # optimize model
##    model = mp.optimize(
##        model,
##        schedule = args.schedule)
##        
##    mp.save(model)
        
    mp.analyse(model, 'model_weights')
    mp.analyse(model, 'model_link_energy')
    mp.analyse(model, 'model_weights_abs_approx')
    mp.analyse(model, 'model_link_energy_abs_approx')
    #
    mp.analyse(model, 'model_report')
    mp.analyse(model, 'knockout')
    mp.analyse(model, 'dataset')
    mp.analyse(model, 'network')
    
    
    quit()
        
    

    
    # save optimized model
    file = mp.save(model)
        
    # analyse model
    mp.analyse(
        model,
        name = args.analyse)
        
    quit()
    
    #
    # do all
    #
    
    for net in mp.cfg.network.keys():
        for data in mp.cfg.dataset.keys():
    
            # create new model
            model = mp.model(
                name = args.model,
                network = net,
                dataset = data)
            
            # opimize model
            model = mp.optimize(
                model,
                schedule = args.schedule)
            
            # analyse model
            mp.analyse(
                model,
                name = args.analyse)

if __name__ == "__main__":
    main()