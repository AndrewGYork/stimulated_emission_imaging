import numpy as np

import matplotlib.pyplot as plt
import pickle

def main():

    filename = 'voltages_green_600mW_red_0mW_fluorescence.pickle'
    
    with open(filename, 'rb') as f:
        order1_voltages = pickle.load(f)

    print('green power =', np.sum(order1_voltages[:,1]))
    print('red power =', np.sum(order1_voltages[:,2]))
    print('trigger power =', np.sum(order1_voltages[:,0]))
    print('max green =', np.max(order1_voltages[:,1]))
    print('min green =', np.min(order1_voltages[:,1]))
    print('max red =', np.max(order1_voltages[:,2]))
    print('min red =', np.min(order1_voltages[:,2]))

    plt.figure()
    plt.plot(order1_voltages[:, 1], '.-', label='Green AOM', color='green')
    plt.plot(order1_voltages[:, 2], '.-', label='Red AOM', color='red')
    plt.plot(order1_voltages[:, 0]/15, '.-', label='Camera trig', color='blue')
    plt.legend()
    plt.grid()
    plt.show()

    return None

if __name__ == '__main__':
    main()
