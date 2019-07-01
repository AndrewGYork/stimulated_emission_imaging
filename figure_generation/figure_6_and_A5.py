import os
import numpy as np
import np_tif
from stack_registration import stack_registration
from stack_registration import bucket
import matplotlib.pyplot as plt

def main():


    assert os.path.isdir('./../images')
    if not os.path.isdir('./../images/figure_6'):
        os.mkdir('./../images/figure_6')
    assert os.path.isdir('./../images')
    if not os.path.isdir('./../images/figure_A5'):
        os.mkdir('./../images/figure_A5')

    z_voltages = [
        '_34000mV',
        '_36000mV',
        '_38000mV',
        '_40000mV',
        '_42000mV',
        '_44000mV',
        '_46000mV',
        '_48000mV',
        '_50000mV',
        '_52000mV',
        '_54000mV',
        '_56000mV',
        '_58000mV',
        '_60000mV',
        '_62000mV',
        ]

    

    # for each piezo voltage (axial location) we: 
    # 1. brightness-correct each delay scan by using the entire image
    # brightness at max/min delay as an estimate of the red laser
    # brightness over that particular delay scan
    # 2. repetition average the images for each delay
    # 3. register the repetition averaged "green off" image to the
    # corresponding "green on" image

    num_delays = 5
    num_reps = 200
    num_z = len(z_voltages)
    width = 380
    height = 128
    less_rows = 3
    data = np.zeros((num_z, num_delays, height - less_rows * 2, width))
    data_ctrl = np.zeros((num_z, num_delays, height - less_rows * 2, width))
    
    for z_index, z_v in enumerate(z_voltages):
        print('Piezo voltage', z_v)
        filename = (
            './../../stimulated_emission_imaging-data' +
            '/2016_11_02_STE_z_stack_darkfield' +
            '/STE_darkfield_113_green_1500mW_red_300mW' +
            z_v + '_many_delays.tif')
        filename_ctrl = (
            './../../stimulated_emission_imaging-data' +
            '/2016_11_02_STE_z_stack_darkfield' +
            '/STE_darkfield_113_green_0mW_red_300mW' +
            z_v + '_many_delays.tif')
        data_z = np_tif.tif_to_array(filename).astype(np.float64)
        data_z_ctrl = np_tif.tif_to_array(filename_ctrl).astype(np.float64)
        # reshape data arrays: 0th axis is rep number, 1st is delay number
        data_z = data_z.reshape(num_reps, num_delays, height, width)
        data_z_ctrl = data_z_ctrl.reshape(num_reps, num_delays, height, width)
        # crop to remove overexposed rows
        data_z = data_z[:, :, less_rows:height - less_rows, :]
        data_z_ctrl = data_z_ctrl[:, :, less_rows:height - less_rows, :]
        # get slice for reference brightness
        ref_slice_1 = data_z[:, 0, :, :].mean(axis=0)
        ref_slice_2 = data_z[:, 4, :, :].mean(axis=0)
        ref_slice = (ref_slice_1 + ref_slice_2) / 2
        # red beam brightness correction
        red_avg_brightness = ref_slice.mean(axis=1).mean(axis=0)
        # for green on data
        local_laser_brightness = ((
            data_z[:, 0, :, :] + data_z[:, 4, :, :])/2
                                  ).mean(axis=2).mean(axis=1)
        local_calibration_factor = red_avg_brightness / local_laser_brightness
        local_calibration_factor = local_calibration_factor.reshape(
            num_reps, 1, 1, 1)
        data_z = data_z * local_calibration_factor
        # for green off data
        local_laser_brightness = ((
            data_z_ctrl[:, 0, :, :] + data_z_ctrl[:, 4, :, :])/2
                                  ).mean(axis=2).mean(axis=1)
        local_calibration_factor = red_avg_brightness / local_laser_brightness
        local_calibration_factor = local_calibration_factor.reshape(
            num_reps, 1, 1, 1)
        data_z_ctrl = data_z_ctrl * local_calibration_factor

        # repetition average both image sets
        data_z_rep_avg = data_z.mean(axis=0)
        data_z_ctrl_rep_avg = data_z_ctrl.mean(axis=0)

        # registration shift control data to match "green on" data
        align_to_this_slice = data_z_rep_avg[0, :, :]
        print("Computing registration shifts (no green)...")
        shifts = stack_registration(
            data_z_ctrl_rep_avg,
            align_to_this_slice=align_to_this_slice,
            refinement='integer',
            register_in_place=True,
            background_subtraction='edge_mean')
        print("... done computing shifts.")

        # build arrays with repetition averaged images
        data[z_index, ...] = data_z_rep_avg
        data_ctrl[z_index, ...] = data_z_ctrl_rep_avg

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

    # plot darkfield and stim emission signal
    top = 1
    bot = 116
    left = 104
    right = 219
    darkfield_cropped = darkfield_stack[:,top:bot,left:right]
    STE_cropped = (STE_stack[:,top:bot,left:right] -
                   crosstalk_stack[:,top:bot,left:right])

    # Our pixels are tiny (8.7 nm/pixel) to give large dynamic range.
    # This is not great for viewing, because fluctuations can swamp the
    # signal. This step bins the pixels into a more typical size.
    bucket_width = 8 # bucket width in pixels
    darkfield_cropped = bucket(
        darkfield_cropped, (1, bucket_width, bucket_width)) / bucket_width ** 2
    STE_cropped = bucket(
        STE_cropped, (1, bucket_width, bucket_width)) / bucket_width ** 2

    STE_y_z = np.zeros((STE_cropped.shape[0]*3,STE_cropped.shape[1]))
    darkfield_y_z = np.zeros((STE_cropped.shape[0]*3,STE_cropped.shape[1]))
    STE_x_z = np.zeros((STE_cropped.shape[0]*3,STE_cropped.shape[2]))
    darkfield_x_z = np.zeros((STE_cropped.shape[0]*3,STE_cropped.shape[2]))
    

    for z_num in range(STE_cropped.shape[0]):
        # get darkfield and STE images and create yz and xz views
        STE_image = STE_cropped[z_num,:,:]
        STE_y_z[z_num*3:(z_num+1)*3,:] = STE_image.mean(axis=1)
        STE_x_z[z_num*3:(z_num+1)*3,:] = STE_image.mean(axis=0)
        
        darkfield_image = darkfield_cropped[z_num,:,:]
        darkfield_y_z[z_num*3:(z_num+1)*3,:] = darkfield_image.mean(axis=1)
        darkfield_x_z[z_num*3:(z_num+1)*3,:] = darkfield_image.mean(axis=0)

        # generate and save plot

        # scale bar
        darkfield_image[-2:-1, 1:6] = 60105
        STE_image[-2:-1, 1:6] = -138
        
        fig, (ax0, ax1) = plt.subplots(nrows=1,ncols=2,figsize=(14,5))

        cax0 = ax0.imshow(darkfield_image, cmap=plt.cm.gray,
                          interpolation='nearest', vmax=60106, vmin=282)
        ax0.axis('off')
        cbar0 = fig.colorbar(cax0,ax=ax0)
        ax0.set_title('A', fontsize=30)

        cax1 = ax1.imshow(STE_image, cmap=plt.cm.gray,
                          interpolation='nearest', vmax=5.5, vmin=-138)
        cbar1 = fig.colorbar(cax1, ax = ax1)
        ax1.set_title('B', fontsize=30)
        ax1.axis('off')
        plt.savefig('./../images/figure_6/darkfield_STE_image_' +
                    str(z_num)+'.svg')
        plt.close()

    # save x projection of stack
    darkfield_y_z[-2:-1, 1:6] =  np.max(darkfield_y_z,(0,1))# scale bar
    STE_y_z[-2:-1, 1:6] =  np.min(STE_y_z,(0,1))# scale bar
    fig, (ax0, ax1) = plt.subplots(nrows=1,ncols=2, figsize=(14,14))
    cax0 = ax0.imshow(darkfield_y_z, cmap=plt.cm.gray,
                      interpolation='nearest')
    ax0.axis('off')
    cbar0 = fig.colorbar(cax0,ax=ax0)
    ax0.set_title('A', fontsize=30)

    cax1 = ax1.imshow(STE_y_z, cmap=plt.cm.gray,
                      interpolation='nearest')
    cbar1 = fig.colorbar(cax1, ax = ax1)
    ax1.set_title('B', fontsize=30)
    ax1.axis('off')
    plt.savefig('./../images/figure_A5/darkfield_STE_image_yz.svg')
    plt.show()

    # save y projection of stack
    darkfield_x_z[-2:-1, 1:6] =  np.max(darkfield_x_z,(0,1))# scale bar
    STE_x_z[-2:-1, 1:6] =  np.min(STE_x_z,(0,1))# scale bar

    fig, (ax0, ax1) = plt.subplots(nrows=1,ncols=2,figsize=(14,14))
    cax0 = ax0.imshow(darkfield_x_z, cmap=plt.cm.gray,
                      interpolation='nearest')
    ax0.axis('off')
    cbar0 = fig.colorbar(cax0,ax=ax0)
    ax0.set_title('A', fontsize=30)

    cax1 = ax1.imshow(STE_x_z, cmap=plt.cm.gray,
                      interpolation='nearest')
    cbar1 = fig.colorbar(cax1, ax = ax1)
    ax1.set_title('B', fontsize=30)
    ax1.axis('off')
    plt.savefig('./../images/figure_A5/darkfield_STE_image_xz.svg')
    plt.show()
    
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
    plt.show()
    plt.figure()
    plt.plot(z_list,STE_signal,'.-',label='STE signal',color='blue')
    plt.plot(z_list,crosstalk_signal,'.-',label='AOM crosstalk',color='green')
    plt.title('Stimulated emission signal main lobe intensity')
    plt.xlabel('Z (nm)')
    plt.ylabel('Change in scattered light signal (CMOS pixel counts)')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()
    

    return None



main()
