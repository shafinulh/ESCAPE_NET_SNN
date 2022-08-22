import os 
import torch 

path  = 'C:/Users/shafi/OneDrive - University of Toronto/ISML_22/snn_conversion_2/trained_models/snn'
list_of_files = {}
for (dirpath, dirnames, filenames) in os.walk(path):
    for filename in filenames:
        if filename.endswith('.pth'):
            trained_snn = os.sep.join([dirpath, filename])
            # print(trained_snn)
            state = torch.load(trained_snn, map_location='cpu')
            print('data for model: ', filename)
            for key in state.keys():
              if key != 'thresholds' and key != 'state_dict' and key != 'optimizer':
                print(key, ': ', state[key])