import os
import numpy as np
import np_tif
from stack_registration import stack_registration
import matplotlib.pyplot as plt

def main():

    # the data to be plotted by this program is generated from raw tifs
    # and repetition_average_expt_and_control.py

    assert os.path.isdir('./../images')
    if not os.path.isdir('./../images/figure_7'):
        os.mkdir('./../images/figure_7')

    filename = (
        './../../stimulated_emission_data/figure_7/dataset_green_1500mW.tif')
    filename_ctrl = (
        './../../stimulated_emission_data/figure_7/dataset_green_0mW.tif')
    data = np_tif.tif_to_array(filename).astype(np.float64)
    data_ctrl = np_tif.tif_to_array(filename_ctrl).astype(np.float64)

    # get rid of overexposed rows at top and bottom of images
    less_rows = 3
    data = data[:,0+less_rows:data.shape[1]-less_rows,:]
    data_ctrl = data_ctrl[:,0+less_rows:data_ctrl.shape[1]-less_rows,:]

    # combine experiment and control images
    data_combined = np.zeros((2,data.shape[0],data.shape[1],data.shape[2]))
    data_combined[0] = data
    data_combined[1] = data_ctrl

##    # register each control slice with the corresponding experimental slice
##    fmm = 0.02 #fourier mask magnitude is a carefully tuned parameter
##    for which_slice in range(data.shape[0]):
##        stack_registration(
##            data_combined[:,which_slice,:,:],
##            fourier_mask_magnitude = fmm,
##            )

    # reshape to hyperstack
    data = data_combined[0]
    data_ctrl = data_combined[1]
    num_delays = 5
    data = data.reshape((
        data.shape[0]/num_delays,
        num_delays,
        data.shape[1],
        data.shape[2],
        ))
    data_ctrl = data_ctrl.reshape((
        data_ctrl.shape[0]/num_delays,
        num_delays,
        data_ctrl.shape[1],
        data_ctrl.shape[2],
        ))
    # from the image where red/green are simultaneous, subtract the
    # average of images taken when the delay magnitude is greatest
    STE_stack = (
        data[:,2,:,:] - # zero red/green delay
        0.5 * (data[:,0,:,:] + data[:,4,:,:]) # max red/green delay
        )
    crosstalk_stack = (
        data_ctrl[:,2,:,:] - # zero red/green delay
        0.5 * (data_ctrl[:,0,:,:] + data_ctrl[:,4,:,:]) # max red/green delay
        )
    # phase contrast image (no STE) stack
    phase_stack = 0.5 * (data[:,0,:,:] + data[:,4,:,:])
    phase_stack_ctrl = 0.5 * (data_ctrl[:,0,:,:] + data_ctrl[:,4,:,:])

    # save processed stacks
##    np_tif.array_to_tif(STE_stack,'STE_stack.tif')
##    np_tif.array_to_tif(crosstalk_stack,'crosstalk_stack.tif')
##    np_tif.array_to_tif(phase_stack,'phase_stack.tif')
##    np_tif.array_to_tif(phase_stack_ctrl,'phase_ctrl_stack.tif')

    # plot phase contrast image and stim emission signal
    top = 0
    bot = 122
    left = 59
    right = 311
    phase_cropped = phase_stack[:,top:bot,left:right]
    STE_cropped = (STE_stack[:,top:bot,left:right] -
                   crosstalk_stack[:,top:bot,left:right])

    for angle_num in range(STE_cropped.shape[0]):
        # filter darkfield and STE images
        STE_image = STE_cropped[angle_num,:,:]
        STE_image = STE_image.reshape(
            1,STE_image.shape[0],STE_image.shape[1])
        STE_image = annular_filter(STE_image,r1=0,r2=0.03)
        STE_image = STE_image[0,:,:]
        phase_image = phase_cropped[angle_num,:,:]
        phase_image = phase_image.reshape(
            1,phase_image.shape[0],phase_image.shape[1])
        phase_image = annular_filter(phase_image,r1=0,r2=0.03)
        phase_image = phase_image[0,:,:]

        # generate and save plot
        STE_image[0,0] = 12.8 # cheap way to conserve colorbar
        STE_image[1,0] = -36.5 # cheap way to conserve colorbar
        phase_image[0,0] = 35500 #cheap way to conserve colorbar
        phase_image[1,0] = 4400 #cheap way to conserve colorbar
        fig, (ax0, ax1) = plt.subplots(nrows=2,ncols=1,figsize=(9,8))

        cax0 = ax0.imshow(phase_image, cmap=plt.cm.gray)
        ax0.axis('off')
        cbar0 = fig.colorbar(cax0,ax=ax0)
        ax0.set_title('Phase contrast image of nanodiamond')

        cax1 = ax1.imshow(STE_image, cmap=plt.cm.gray)
        cbar1 = fig.colorbar(cax1, ax = ax1)
        ax1.set_title('Change in phase contrast image due to stim. emission')
        ax1.axis('off')
        plt.savefig('./../images/figure_7/phase_STE_image_' +
                    str(angle_num)+'.svg')
    
    
    # average points around center lobe of the nanodiamond image to get
    # "average signal level" for darkfield and STE images
    top = 9
    bot = 84
    left = 153
    right = 232
    STE_signal = (
        STE_stack[:,top:bot,left:right].mean(axis=2).mean(axis=1))
    crosstalk_signal = (
        crosstalk_stack[:,top:bot,left:right].mean(axis=2).mean(axis=1))
    phase_signal = (
        phase_stack[:,top:bot,left:right].mean(axis=2).mean(axis=1))
    
    # plot signal v phase
    angles = range(32)
    true_signal = STE_signal - crosstalk_signal
    plt.figure()
    plt.plot(angles,phase_signal,'.-',color='black')
    plt.title('Phase contrast image main lobe brightness')
    plt.xlabel('Relative phase (arb. units)')
    plt.ylabel('Average intensity (CMOS pixel counts)')
    plt.grid()
##    plt.savefig('darkfield_v_z.svg')
    plt.figure()
    plt.plot(angles,STE_signal,'.-',label='STE signal',color='blue')
    plt.plot(angles,crosstalk_signal,'.-',label='AOM crosstalk',color='green')
    plt.title('Stimulated emission signal main lobe intensity')
    plt.xlabel('Relative phase (arb. units)')
    plt.ylabel('Change in phase contrast signal (CMOS pixel counts)')
    plt.legend(loc='lower right')
    plt.grid()
##    plt.savefig('darkfield_STE_v_z.svg')
    plt.figure()
    plt.plot(angles,true_signal,'.-',label='STE signal',color='red')
    plt.title('Corrected stimulated emission signal main lobe intensity')
    plt.xlabel('Relative phase (arb. units)')
    plt.ylabel('Change in phase contrast signal (CMOS pixel counts)')
    plt.legend(loc='lower right')
    plt.grid()
##    plt.savefig('darkfield_STE_v_z.svg')
##    plt.show()
    

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

main()
