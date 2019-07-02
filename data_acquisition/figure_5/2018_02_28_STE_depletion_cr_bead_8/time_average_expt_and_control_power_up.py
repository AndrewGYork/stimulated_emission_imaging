import os
import numpy as np
import np_tif

# each raw data stack is a red/green delay scan (five different delays,
# middle one 0 delay) over many repetitions

angles = [
    ''
##    '_116_0',
##    '_116_5',
##    '_117_0',
##    '_117_5',
##    '_118_0',
##    '_118_5',
##    '_119_0',
##    '_119_5',
##    '_120_0',
##    '_120_5',
##    '_121_0',
##    '_121_5',
##    '0',
##    '1',
##    '2',
##    '3',
##    '4',
##    '5',
##    '6',
##    '7',
##    '8',
##    '9',
##    '10',
##    '11',
##    '12',
##    '13',
##    '14',
##    '15',
##    '16',
##    '17',
##    '18',
##    '19',
##    '20',
##    '21',
##    '22',
##    '23',
##    '24',
##    '25',
##    '26',
##    '27',
##    '28',
##    '29',
##    '30',
##    '31',
    ]

green_powers = [
    '_0mW',
##    '_25mW',
    '_31mW',
##    '_100mW',
    '_95mW',
##    '_225mW',
    '_190mW',
##    '_400mW',
##    '_500mW',
    '_380mW',
##    '_700mW',
##    '_800mW',
##    '_900mW',
##    '_1000mW',
    '_696mW',
##    '_1200mW',
    '_950mW',
    ]

red_powers = [
##    '_blocked',
##    '_0mW',
##    '_75mW',
##    '_150mW',
##    '_225mW',
    '_240mW',
    ]

z_voltages = [
    '',
##    '_22000mV',
##    '_23000mV',
##    '_24000mV',
##    '_25000mV',
##    '_26000mV',
##    '_27000mV',
##    '_28000mV',
##    '_29000mV',
##    '_30000mV',
##    '_31000mV',
##    '_32000mV',
##    '_33000mV',
##    '_34000mV',
##    '_35000mV',
##    '_36000mV',
##    '_37000mV',
##    '_38000mV',
##    '_39000mV',
##    '_40000mV',
##    '_51000mV',
##    '_52000mV',
##    '_53000mV',
##    '_54000mV',
##    '_58000mV',
##    '_60000mV',
##    '_62000mV',
    ]



data_list = []
for gr_pow in green_powers:
    for ang in angles:
        for rd_pow in red_powers:
            for z_v in z_voltages:
                filename = (
                    'fluorescence' +
##                    'phase_angle_' + ang +
                    '_green' + gr_pow +
                    '_red' + rd_pow +
                    z_v +
                    '_up.tif')
                assert os.path.exists(filename)
                print("Loading", filename)
                data = np_tif.tif_to_array(filename).astype(np.float64)
                assert data.shape == (3*1, 128, 380)
                data = data.reshape(1, 3, 128, 380) # Stack to hyperstack
                data = data.mean(axis=0, keepdims=True) # Sum over reps
                data_list.append(data)
print("Done loading.")
data = np.concatenate(data_list)
##    print('data shape is',data.shape)
##    mean_subtracted = data - data.mean(axis=-3, keepdims=True)
tif_shape = (data.shape[0] * data.shape[1], data.shape[2], data.shape[3])
    
    


print("Saving...")
np_tif.array_to_tif(data.reshape(tif_shape),('dataset_green_all_powers_up.tif'))
##    np_tif.array_to_tif(mean_subtracted.reshape(tif_shape),
##                        'dataset_green'+gr_pow+'_mean_subtracted.tif')
##    np_tif.array_to_tif(data_controlled,('data_controlled_green'+gr_pow+'.tif'))
print("Done saving.")
