import os
import numpy as np
import np_tif
from stack_registration import stack_registration
import matplotlib.pyplot as plt

def main():

    # the data to be plotted by this program is generated from raw tifs
    # and repetition_average_expt_and_control.py

    assert os.path.isdir('./../images')
    if not os.path.isdir('./../images/figure_6'):
        os.mkdir('./../images/figure_6')
    if not os.path.isdir('./../images/figure_7'):
        os.mkdir('./../images/figure_7')

    num_reps = 10 # number of times a power/delay stack was taken
    num_delays = 5
    image_h = 128
    image_w = 380

    # power calibration
    # red max power is 300 mW
    # green max power is 1450 mW
    # green powers calibrated using camera
    green_bg = 113.9
    green_max_mW = 1450
    green_powers = np.array(
        (113.9,119.6,124.5,135,145.5,159.5,175.3,193.1,234.5,272.2,334.1,385.7,446.1))
    green_powers -= green_bg
    green_powers = green_powers * green_max_mW / max(green_powers)
    green_powers = np.around(green_powers).astype(int)

    # red powers calibrated using camera
    red_bg = 26.6
    red_max_mW = 300
    red_powers = np.array(
        (26.6, 113, 198, 276, 353, 438, 537))
    red_powers -= red_bg
    red_powers = red_powers * red_max_mW / max(red_powers)
    red_powers = np.around(red_powers).astype(int)

    filename = './../../stimulated_emission_data/figure_6_7/dataset.tif'
    data = np_tif.tif_to_array(filename).astype(np.float64)

    # get rid of overexposed rows at top and bottom of images
    less_rows = 3
    data = data[:,0+less_rows:data.shape[1]-less_rows,:]

    # reshape to hyperstack
    data = data.reshape((
        len(red_powers),
        len(green_powers),
        num_delays,
        data.shape[1],
        data.shape[2],
        ))

    # from the image where red/green are simultaneous, subtract the
    # average of images taken when the delay magnitude is greatest
    depletion_stack = (
        data[:,:,2,:,:] - # zero red/green delay
        0.5 * (data[:,:,0,:,:] + data[:,:,4,:,:]) # max red/green delay
        )

    # fluorescence image (no STE) stack
    fluorescence_stack = 0.5 * (data[:,:,0,:,:] + data[:,:,4,:,:])
    depleted_stack = data[:,:,2,:,:] # zero red/green delay

    # save processed stacks
##    tif_shape = (
##        len(red_powers)*len(green_powers),
##        depletion_stack.shape[2],
##        depletion_stack.shape[3],
##        )
##    np_tif.array_to_tif(
##        depletion_stack.reshape(tif_shape),'depletion_stack.tif')
##    np_tif.array_to_tif(
##        fluorescence_stack.reshape(tif_shape),'fluorescence_stack.tif')
##    np_tif.array_to_tif(
##        depleted_stack.reshape(tif_shape),'depleted_stack_ctrl.tif')

    # plot darkfield and stim emission signal
    # first crop and spatially filter images
    top = 0
    bot = 112
    left = 122
    right = 249
    fluorescence_cropped = fluorescence_stack[-1,-1,top:bot,left:right]-102
    fluorescence_cropped = fluorescence_cropped.reshape(
        1,fluorescence_cropped.shape[0],fluorescence_cropped.shape[1])
    fluorescence_cropped = annular_filter(fluorescence_cropped,r1=0,r2=0.03)
    fluorescence_cropped = fluorescence_cropped[0,:,:]
    depletion_cropped = depletion_stack[-1,-1,top:bot,left:right]
    depletion_cropped = depletion_cropped.reshape(
        1,depletion_cropped.shape[0],depletion_cropped.shape[1])
    depletion_cropped = annular_filter(depletion_cropped,r1=0,r2=0.03)
    depletion_cropped = depletion_cropped[0,:,:]

    fig, (ax0, ax1) = plt.subplots(nrows=1,ncols=2,figsize=(16,5))

    cax0 = ax0.imshow(fluorescence_cropped, cmap=plt.cm.gray)
    ax0.axis('off')
    cbar0 = fig.colorbar(cax0, ax = ax0)
    ax0.set_title('Fluorescence image of nanodiamond')

    cax1 = ax1.imshow(depletion_cropped, cmap=plt.cm.gray)
    cbar1 = fig.colorbar(cax1, ax = ax1)
    ax1.set_title('Fluorescence intensity decreased due to stim. emission')
    ax1.axis('off')
    plt.savefig('./../images/figure_6/fluorescence_depletion_image.svg')
    
    # average points around center lobe of the nanodiamond image to get
    # "average signal level" for darkfield and STE images
    top = 31
    bot = 84
    left = 160
    right = 215
    depletion_signal = (
        depletion_stack[:,:,top:bot,left:right].mean(axis=3).mean(axis=2))
    depleted_signal = (
        depleted_stack[:,:,top:bot,left:right].mean(axis=3).mean(axis=2))
    fluorescence_signal = (
        fluorescence_stack[:,:,top:bot,left:right].mean(axis=3).mean(axis=2))
##    np_tif.array_to_tif(STE_signal,'STE_signal_array.tif')
##    np_tif.array_to_tif(crosstalk_signal,'crosstalk_signal_array.tif')
##    np_tif.array_to_tif(darkfield_signal,'darkfield_signal_array.tif')
    
    # plot signal

    plt.figure()
    for (pow_num,rd_pow) in enumerate(red_powers):
        plt.plot(
            green_powers,depleted_signal[pow_num,:],
            '.-',label=('Red power ='+str(rd_pow)))
    plt.title('Depleted signal in main fluorescence lobe')
    plt.xlabel('Green power (mW)')
    plt.ylabel('Fluorescence light signal (CMOS pixel counts)')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig('./../images/figure_7/fluorescence_v_green_power.svg')    
    
    plt.figure()
    for (pow_num,gr_pow) in enumerate(green_powers):
        plt.plot(
            red_powers,depletion_signal[:,pow_num],
            '.-',label=('Green power ='+str(gr_pow)))
    plt.title('Depletion signal in main fluorescent lobe')
    plt.xlabel('Red power (mW)')
    plt.ylabel('Change in fluorescent light signal (CMOS pixel counts)')
    plt.legend(loc='lower left')
    plt.grid()
    plt.figure()
    for (pow_num,rd_pow) in enumerate(red_powers):
        plt.plot(
            green_powers,depletion_signal[pow_num,:],
            '.-',label=('Red power ='+str(rd_pow)))
    plt.title('Depletion signal in main fluorescence lobe')
    plt.xlabel('Green power (mW)')
    plt.ylabel('Change in fluorescent light signal (CMOS pixel counts)')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
    

    return None




def annular_filter(x, r1, r2):
    assert r2 > r1 >= 0

    x_ft = np.fft.fftn(x)
    n_y, n_x = x.shape[-2:]
    kx = np.fft.fftfreq(n_x).reshape(1, 1, n_x)
    ky = np.fft.fftfreq(n_y).reshape(1, n_y, 1)

    x_ft[kx**2 + ky**2 > r2**2] = 0
    x_ft[kx**2 + ky**2 < r1**2] = 0

    x_filtered = np.fft.ifftn(x_ft).real

    return x_filtered

if __name__ == '__main__':
    main()
