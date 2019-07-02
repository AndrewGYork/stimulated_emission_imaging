import os
import numpy as np
import np_tif

# each raw data stack is a red/green delay scan (3 different delays,
# middle one 0 delay)

angles = [
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    '10',
    '11',
    '12',
    '13',
    '14',
    ]

green_powers = [
    '_0mW',
    '_1010mW',
    ]

red_powers = [
    '_240mW',
    ]

for gr_pow in green_powers:
    data_list = []
    for ang in angles:
        for rd_pow in red_powers:
            filename = (
                'STE_' +
                'phase_angle_' + ang +
                '_green' + gr_pow +
                '_red' + rd_pow +
                '.tif')
            assert os.path.exists(filename)
            print("Loading", filename)
            data = np_tif.tif_to_array(filename).astype(np.float64)
            assert data.shape == (3, 128, 380)
            data = data.reshape(1, data.shape[0], data.shape[1], data.shape[2])
            data_list.append(data)
    print("Done loading.")
    data = np.concatenate(data_list)
    print('data shape is',data.shape)
    tif_shape = (data.shape[0] * data.shape[1], data.shape[2], data.shape[3])
    
    


    print("Saving...")
    np_tif.array_to_tif(data.reshape(tif_shape),
                        ('dataset_green'+gr_pow+'_single_shot.tif'))
    print("Done saving.")
