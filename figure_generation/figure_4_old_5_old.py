import os
import numpy as np
from stack_registration import bucket
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as patches
import np_tif

def main():

    assert os.path.isdir('./../images')
    if not os.path.isdir('./../images/figure_4_old'):
        os.mkdir('./../images/figure_4_old')
    if not os.path.isdir('./../images/figure_5_old'):
        os.mkdir('./../images/figure_5_old')

    data_path = ('./../../stimulated_emission_imaging-data' +
                 '/2016_11_18_modulated_imaging_darkfield_nanodiamond_7' +
                 '_extra_green_filter/')

    num_reps = 200 # number of times a power/delay stack was taken
    num_delays = 5

    # power calibration
    # red max power is 300 mW
    # green max power is 1450 mW
    # green powers calibrated using camera
    green_max_mW = 1450
    green_powers = np.array(
        (113.9,119.6,124.5,135,145.5,159.5,175.3,193.1,234.5,272.2,334.1,385.7,446.1))
    # 0th power is AOM with 0 volts
    green_powers = green_powers - min(green_powers)
    green_powers = green_powers * green_max_mW / max(green_powers)
    green_powers = np.around(green_powers).astype(int)
    sparse_green_nums = [0,7,10,12]
    green_powers_sparse = green_powers[sparse_green_nums]

    # red powers calibrated using camera
    red_max_mW = 300
    red_powers = np.array(
        (26.6, 113, 198, 276, 353, 438, 537))
    # 0th power is AOM with 0 volts
    red_powers = red_powers - min(red_powers)
    red_powers = red_powers * red_max_mW / max(red_powers)
    red_powers = np.around(red_powers).astype(int)
    sparse_red_nums = [0,3,6]
    red_powers_sparse = red_powers[sparse_red_nums]

    # load representative images
    STE_image_filename = (data_path + 'STE_image_avg.tif')
    STE_image = np_tif.tif_to_array(STE_image_filename).astype(np.float64)
    STE_image_bg_filename = (data_path + 'STE_image_bg_avg.tif')
    STE_image_bg = np_tif.tif_to_array(STE_image_bg_filename).astype(np.float64)
    darkfield_image_filename = (data_path + 'darkfield_image_avg.tif')
    darkfield_image = np_tif.tif_to_array(
        darkfield_image_filename).astype(np.float64)
    darkfield_image_bg_filename = (data_path + 'darkfield_image_bg_avg.tif')
    darkfield_image_bg = np_tif.tif_to_array(
        darkfield_image_bg_filename).astype(np.float64)
    STE_delta_image = STE_image - darkfield_image
    STE_delta_image_bg = STE_image_bg - darkfield_image_bg
    # crosstalk correction
    STE_delta_image_corrected = STE_delta_image - STE_delta_image_bg

    # load space-averaged signal data
    filename = (data_path + 'signal_all_scaled.tif')
    data = np_tif.tif_to_array(filename).astype(np.float64)
    bg_filename = (data_path + 'signal_green_blocked_all_scaled.tif')
    bg = np_tif.tif_to_array(bg_filename).astype(np.float64)
    
    # reshape to hyperstack
    data = data.reshape((
        num_reps,
        len(red_powers),
        len(green_powers),
        num_delays,
        ))
    bg = bg.reshape((
        num_reps,
        len(red_powers),
        len(green_powers),
        num_delays,
        ))

    # compute STE signal
    # (delay #2 -> 0 us delay, delay #0 and #4 -> +/- 2.5 us delay)
    STE_signal = data[:,:,:,2] - 0.5*(data[:,:,:,0] + data[:,:,:,4])
    STE_signal_bg = bg[:,:,:,2] - 0.5*(bg[:,:,:,0] + bg[:,:,:,4])

    # crosstalk correction
    STE_signal_corrected = STE_signal - STE_signal_bg

    # remove bad repetition(s)
    STE_signal_corrected = np.delete(STE_signal_corrected, 188, 0)
    STE_signal_corrected = np.delete(STE_signal_corrected, 185, 0)
    STE_signal_corrected = np.delete(STE_signal_corrected, 148, 0)
    STE_signal_corrected = np.delete(STE_signal_corrected, 63, 0)
    STE_signal_corrected = np.delete(STE_signal_corrected, 5, 0)
    # also for regular signal in order to compute correct standard
    # deviation without outliers
    STE_signal = np.delete(STE_signal, 188, 0)
    STE_signal = np.delete(STE_signal, 185, 0)
    STE_signal = np.delete(STE_signal, 148, 0)
    STE_signal = np.delete(STE_signal, 63, 0)
    STE_signal = np.delete(STE_signal, 5, 0)

    # standard deviation
    STE_signal_std = np.std(STE_signal,axis=0)

    # repetition average
    STE_signal_corrected = STE_signal_corrected.mean(axis=0)

    plt.figure()
    for (pow_num,rd_pow) in enumerate(red_powers):
        plt.errorbar(
            green_powers,
            STE_signal_corrected[pow_num,:],
            yerr=STE_signal_std[pow_num,:]/(200**0.5),
            label=('Stimulation power = '+str(rd_pow)+' mW'))
    plt.xlabel('Excitation power (mW)')
    plt.ylabel('Change in scattered light signal (CMOS pixel counts)')
    plt.legend(loc='lower left')
    plt.ylim(-52, 6)
    plt.xlim(-15, 1465)
    plt.grid()
    plt.savefig('./../images/figure_5_old/STE_v_green_power.svg')
    plt.show()

    # plot sparse power scan data
    # all green powers, sparse red powers
    plt.figure()
    for (pow_num,rd_pow) in enumerate(red_powers):
        if pow_num in sparse_red_nums:
            plt.errorbar(
                green_powers,
                STE_signal_corrected[pow_num,:],
                yerr=STE_signal_std[pow_num,:]/(200**0.5),
                label=('Stimulation power = '+str(rd_pow)+' mW'))
    plt.xlabel('Excitation power (mW)')
    plt.ylabel('Change in scattered light signal (CMOS pixel counts)')
    plt.legend(loc='lower left')
    plt.ylim(-40, 6)
    plt.xlim(-15, 1465)
    plt.grid()
    plt.savefig('./../images/figure_5_old/STE_v_green_power_sparse.svg')
    plt.show()

    #all red powers, sparse green powers
    plt.figure()
    for (pow_num,gr_pow) in enumerate(green_powers):
        if pow_num in sparse_green_nums:
            plt.errorbar(
                red_powers,
                STE_signal_corrected[:, pow_num],
                yerr=STE_signal_std[:, pow_num] / (200 ** 0.5),
                label=('Excitation power = ' + str(gr_pow) + ' mW'))
    plt.xlabel('Stimulation power (mW)')
    plt.ylabel('Change in scattered light signal (CMOS pixel counts)')
    plt.legend(loc='lower left')
    plt.xlim(-3, 303)
    plt.grid()
    plt.savefig('./../images/figure_5_old/STE_v_red_power_sparse.svg')
    plt.show()


    # plot darkfield and stim emission images

    darkfield_image = darkfield_image[0,0:-1,:]
    STE_delta_image = STE_delta_image[0,0:-1,:]
    STE_delta_image_corrected = STE_delta_image_corrected[0,0:-1,:]

    # downsample darkfield and STE delta images
    bucket_width = 8
    bucket_shape = (bucket_width, bucket_width)
    darkfield_image = bucket(
        darkfield_image, bucket_shape) / bucket_width**2
    STE_delta_image = bucket(
        STE_delta_image, bucket_shape) / bucket_width**2
    STE_delta_image_corrected = bucket(
        STE_delta_image_corrected, bucket_shape) / bucket_width**2

    # scale bar
    darkfield_image[-2:-1, 1:6] = np.max(darkfield_image)
    STE_delta_image[-2:-1, 1:6] = np.min(STE_delta_image)
    STE_delta_image_corrected[-2:-1, 1:6] = np.min(STE_delta_image_corrected)

    fig, (ax0, ax1) = plt.subplots(nrows=1,ncols=2,figsize=(19,5))

    cax0 = ax0.imshow(darkfield_image, cmap=plt.cm.gray,
                      interpolation='nearest')
    ax0.axis('off')
    cbar0 = fig.colorbar(cax0, ax = ax0)

    cax1 = ax1.imshow(STE_delta_image_corrected, cmap=plt.cm.gray,
                      interpolation='nearest')
    cbar1 = fig.colorbar(cax1, ax = ax1)
    ax1.axis('off')

    ax0.text(0.4,2.2,'A',fontsize=72,color='white',fontweight='bold')
    ax1.text(0.4,2.2,'B',fontsize=72,color='black',fontweight='bold')
    plt.savefig('./../images/figure_4_old/darkfield_STE_image.svg')
    plt.show()

    return None


if __name__ == '__main__':
    main()
