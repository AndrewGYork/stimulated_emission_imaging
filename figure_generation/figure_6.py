import os
import numpy as np
import np_tif
from stack_registration import stack_registration
import matplotlib.pyplot as plt

def main():

    # the data to be plotted by this program is generated from raw tifs
    # and repetition_average_expt_and_control.py

##    filename = ('dataset_green_1500mW.tif')
##    filename_ctrl = ('dataset_green_0mW.tif')
    assert os.path.isdir('./../images')
    if not os.path.isdir('./../images/figure_6'):
        os.mkdir('./../images/figure_6')

    filename = (
        './../../stimulated_emission_data/figure_6/dataset_green_1500mW.tif')
    filename_ctrl = (
        './../../stimulated_emission_data/figure_6/dataset_green_0mW.tif')

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

    # register each control slice with the corresponding experimental slice
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
    # darkfield image (no STE) stack
    darkfield_stack = 0.5 * (data[:,0,:,:] + data[:,4,:,:])

    # save processed stacks
##    np_tif.array_to_tif(STE_stack,'STE_stack.tif')
##    np_tif.array_to_tif(crosstalk_stack,'crosstalk_stack.tif')
##    np_tif.array_to_tif(darkfield_stack,'darkfield_stack.tif')

    # plot darkfield and stim emission signal
    top = 1
    bot = 116
    left = 74+30
    right = 249-30
    darkfield_cropped = darkfield_stack[:,top:bot,left:right]
    STE_cropped = (STE_stack[:,top:bot,left:right] -
                   crosstalk_stack[:,top:bot,left:right])

    STE_y_z = np.zeros((STE_cropped.shape[0]*23,STE_cropped.shape[1]))
    darkfield_y_z = np.zeros((STE_cropped.shape[0]*23,STE_cropped.shape[1]))
    STE_x_z = np.zeros((STE_cropped.shape[0]*23,STE_cropped.shape[2]))
    darkfield_x_z = np.zeros((STE_cropped.shape[0]*23,STE_cropped.shape[2]))

    for z_num in range(STE_cropped.shape[0]):
        # filter darkfield and STE images
        STE_image = STE_cropped[z_num,:,:]
        STE_image = STE_image.reshape(
            1,STE_image.shape[0],STE_image.shape[1])
        STE_image = annular_filter(STE_image,r1=0,r2=0.03)
        STE_image = STE_image[0,:,:]
        STE_y_z[z_num*23:(z_num+1)*23,:] = STE_image.mean(axis=1)
        STE_x_z[z_num*23:(z_num+1)*23,:] = STE_image.mean(axis=0)
        
        darkfield_image = darkfield_cropped[z_num,:,:]
        darkfield_image = darkfield_image.reshape(
            1,darkfield_image.shape[0],darkfield_image.shape[1])
        darkfield_image = annular_filter(darkfield_image,r1=0,r2=0.03)
        darkfield_image = darkfield_image[0,:,:]
        darkfield_y_z[z_num*23:(z_num+1)*23,:] = darkfield_image.mean(axis=1)
        darkfield_x_z[z_num*23:(z_num+1)*23,:] = darkfield_image.mean(axis=0)

        # generate and save plot
        STE_image[0,1] = 10 # cheap way to conserve colorbar
        STE_image[0,0] = -149 # cheap way to conserve colorbar
        darkfield_image[0,1] = -200 #cheap way to conserve colorbar
        darkfield_image[0,0] = 64060 #cheap way to conserve colorbar
        darkfield_image[103:109,5:34] =  np.max(darkfield_image,(0,1))# scale bar
        STE_image[103:109,5:34] =  np.min(STE_image,(0,1))# scale bar
        
        fig, (ax0, ax1) = plt.subplots(nrows=1,ncols=2,figsize=(14,5))

        cax0 = ax0.imshow(darkfield_image, cmap=plt.cm.gray)
        ax0.axis('off')
        cbar0 = fig.colorbar(cax0,ax=ax0)
        ax0.set_title('A', fontsize=30)

        cax1 = ax1.imshow(STE_image, cmap=plt.cm.gray)
        cbar1 = fig.colorbar(cax1, ax = ax1)
        ax1.set_title('B', fontsize=30)
        ax1.axis('off')
        plt.savefig('./../images/figure_6/darkfield_STE_image_' +
                    str(z_num)+'.svg')
        plt.close()

    # generate and save x projection of stack
    darkfield_y_z[328:334,5:34] =  np.max(darkfield_y_z,(0,1))# scale bar
    STE_y_z[328:334,5:34] =  np.min(STE_y_z,(0,1))# scale bar
##    np_tif.array_to_tif(darkfield_y_z,'darkfield_y_z.tif')
##    np_tif.array_to_tif(STE_y_z,'STE_y_z.tif')
    fig, (ax0, ax1) = plt.subplots(nrows=1,ncols=2,figsize=(14,14))
    cax0 = ax0.imshow(darkfield_y_z, cmap=plt.cm.gray)
    ax0.axis('off')
    cbar0 = fig.colorbar(cax0,ax=ax0)
    ax0.set_title('C', fontsize=30)

    cax1 = ax1.imshow(STE_y_z, cmap=plt.cm.gray)
    cbar1 = fig.colorbar(cax1, ax = ax1)
    ax1.set_title('D', fontsize=30)
    ax1.axis('off')
    plt.savefig('./../images/figure_6/darkfield_STE_image_yz.svg')
    plt.close()

    # generate and save y projection of stack
##    darkfield_x_z = darkfield_x_z[:,30:175-30]
##    STE_x_z = STE_x_z[:,30:175-30]
    darkfield_x_z[328:334,5:34] =  np.max(darkfield_x_z,(0,1))# scale bar
    STE_x_z[328:334,5:34] =  np.min(STE_x_z,(0,1))# scale bar
##    np_tif.array_to_tif(darkfield_x_z,'darkfield_x_z.tif')
##    np_tif.array_to_tif(STE_x_z,'STE_x_z.tif')
    fig, (ax0, ax1) = plt.subplots(nrows=1,ncols=2,figsize=(14,14))
    cax0 = ax0.imshow(darkfield_x_z, cmap=plt.cm.gray)
    ax0.axis('off')
    cbar0 = fig.colorbar(cax0,ax=ax0)
    ax0.set_title('C', fontsize=30)

    cax1 = ax1.imshow(STE_x_z, cmap=plt.cm.gray)
    cbar1 = fig.colorbar(cax1, ax = ax1)
    ax1.set_title('D', fontsize=30)
    ax1.axis('off')
    plt.savefig('./../images/figure_6/darkfield_STE_image_xz.svg')
    plt.close()
    
    
    # average points around center lobe of the nanodiamond image to get
    # "average signal level" for darkfield and STE images
    top = 38
    bot = 74
    left = 144
    right = 177
    STE_signal = (
        STE_stack[:,top:bot,left:right].mean(axis=2).mean(axis=1))
    crosstalk_signal = (
        crosstalk_stack[:,top:bot,left:right].mean(axis=2).mean(axis=1))
    darkfield_signal = (
        darkfield_stack[:,top:bot,left:right].mean(axis=2).mean(axis=1))
    
    # plot signal v z
    z_list = a = np.arange(-1400,1401,200)
    plt.figure()
    plt.plot(z_list,darkfield_signal,'.-',color='black')
    plt.title('Darkfield image main lobe brightness')
    plt.xlabel('Z (nm)')
    plt.ylabel('Average intensity (CMOS pixel counts)')
    plt.grid()
##    plt.savefig('darkfield_v_z.svg')
    plt.figure()
    plt.plot(z_list,STE_signal,'.-',label='STE signal',color='blue')
    plt.plot(z_list,crosstalk_signal,'.-',label='AOM crosstalk',color='green')
    plt.title('Stimulated emission signal main lobe intensity')
    plt.xlabel('Z (nm)')
    plt.ylabel('Change in scattered light signal (CMOS pixel counts)')
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
