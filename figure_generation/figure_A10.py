import os
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from stack_registration import bucket
import np_tif

def main():

    # the data to be plotted by this program is generated from raw tifs
    # and repetition_average_expt_and_control.py

    assert os.path.isdir('./../images')
    if not os.path.isdir('./../images/figure_A10'):
        os.mkdir('./../images/figure_A10')

    filename = (
        './../../stimulated_emission_imaging-data' +
        '/2016_10_31_STE_phase_scan_re-center_phase_plate'
        '/dataset_green_1500mW.tif')
    filename_ctrl = (
        './../../stimulated_emission_imaging-data' +
        '/2016_10_31_STE_phase_scan_re-center_phase_plate'
        '/dataset_green_0mW.tif')
    data = np_tif.tif_to_array(filename).astype(np.float64)
    data_ctrl = np_tif.tif_to_array(filename_ctrl).astype(np.float64)

    # reshape to hyperstack
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

    # Get the average pixel brightness in the background region of the
    # phase contrast images. We'll use it to account for laser intensity
    # fluctuations. This was already done for all reps at a particular
    # angle, but now we should do it across all angles
    avg_laser_brightness = get_bg_level(data.mean(axis=(0, 1)))

    # scale all images to have the same background brightness. This
    # amounts to a correction of roughly 1% or less
    # ... for "green on" data
    local_laser_brightness = get_bg_level(data)
    data = data * (avg_laser_brightness / local_laser_brightness).reshape(
        data.shape[0], data.shape[1], 1, 1)
    # ... for "green off" data
    # scale all images to have the same background brightness. This
    # amounts to a correction of roughly 1% or less
    local_laser_brightness = get_bg_level(data_ctrl)
    data_ctrl = data_ctrl * (
        avg_laser_brightness / local_laser_brightness).reshape(
            data_ctrl.shape[0], data_ctrl.shape[1], 1, 1)

    # get zero delay images, max delay images and phase contrast images
    zero_delay_images = data[:, 2, :, :] # zero red/green delay
    max_delay_images = data[:, 0, :, :] # red before green (min delay)
    phase_stack = max_delay_images
    max_delay_images = phase_stack # just use red before green
    # also for green off data
    zero_delay_images_ctrl = data_ctrl[:, 2, :, :] # zero red/green delay
    max_delay_images_ctrl = data_ctrl[:, 0, :, :] # red before green (min delay)

    # from the image where red/green are simultaneous, subtract the
    # average of the max and min delay images
    STE_stack = zero_delay_images - max_delay_images
    STE_stack_ctrl = zero_delay_images_ctrl - max_delay_images_ctrl
    # subtract crosstalk signal
    STE_stack -= STE_stack_ctrl

    # The background phase contrast image noise is comprised of stripes
    # that are completely outside of the microscope's spatial pass-band.
    # The smoothing step below strongly attenuates this striping
    # artifact with almost no effect on spatial frequencies due to the
    # sample.
    sigma = 9 # tune this parameter to reject high spatial frequencies
    STE_stack = gaussian_filter(STE_stack, sigma=(0, sigma, sigma))
    STE_stack_ctrl = gaussian_filter(STE_stack_ctrl, sigma=(0, sigma, sigma))
    phase_stack = gaussian_filter(phase_stack, sigma=(0, sigma, sigma))

    # crop images to center bead
    top = 0
    bot = 122
    left = 59
    right = 311
    phase_cropped = phase_stack[:,top:bot,left:right]
    # also subtract crosstalk
    STE_cropped = (STE_stack[:,top:bot,left:right] -
                   STE_stack_ctrl[:,top:bot,left:right])

    # Our pixels are tiny (8.7 nm/pixel) to give large dynamic range.
    # This is not great for viewing, because fluctuations can swamp the
    # signal. This step bins the pixels into a more typical size.
    bucket_width = 8 # bucket width in pixels
    phase_cropped = bucket(
        phase_cropped, (1, bucket_width, bucket_width)) / bucket_width**2
    STE_cropped = bucket(
        STE_cropped, (1, bucket_width, bucket_width)) / bucket_width**2

    # find min and max values of 
    phase_max = np.amax(phase_cropped)
    phase_min = np.amin(phase_cropped)
    STE_max = np.amax(STE_cropped)
    STE_min = np.amin(STE_cropped)

    for angle_num in range(STE_cropped.shape[0]):
        STE_image = STE_cropped[angle_num,:,:]
        phase_image = phase_cropped[angle_num,:,:]

        # generate and save plot
        phase_image[-2:-1, 1:6] = phase_max# scale bar
        STE_image[-2:-1, 1:6] = STE_min# scale bar
        
        fig, (ax0, ax1) = plt.subplots(nrows=2,ncols=1,figsize=(9,8))

        cax0 = ax0.imshow(phase_image, cmap=plt.cm.gray,
                          interpolation='nearest',
                          vmax=phase_max, vmin=phase_min)
        ax0.axis('off')
        cbar0 = fig.colorbar(cax0,ax=ax0)
        ax0.set_title('Phase contrast image of nanodiamond')

        cax1 = ax1.imshow(STE_image, cmap=plt.cm.gray,
                          interpolation='nearest',
                          vmax=STE_max, vmin=STE_min)
        cbar1 = fig.colorbar(cax1, ax = ax1)
        ax1.set_title('Change in phase contrast image due to N-v excitation')
        ax1.axis('off')
        plt.savefig('./../images/figure_A10/phase_STE_image_' +
                    str(angle_num)+'.svg')
        plt.close()
    
    # average points around center lobe of the nanodiamond image to get
    # "average signal level" for darkfield and STE images
    top = 9
    bot = 84
    left = 153
    right = 232
    STE_signal = (
        STE_stack[:,top:bot,left:right].mean(axis=2).mean(axis=1))
    STE_signal_ctrl = (
        STE_stack_ctrl[:,top:bot,left:right].mean(axis=2).mean(axis=1))
    phase_signal = (
        phase_stack[:,top:bot,left:right].mean(axis=2).mean(axis=1))
    
    # plot signal v phase
    angles = range(32)
    true_signal = STE_signal - STE_signal_ctrl
    plt.figure()
    plt.plot(angles,phase_signal,'.-',color='black')
    plt.title('Phase contrast image main lobe brightness')
    plt.xlabel('Relative phase (arb. units)')
    plt.ylabel('Average intensity (CMOS pixel counts)')
    plt.grid()
##    plt.savefig('darkfield_v_z.svg')
    plt.figure()
    plt.plot(angles,STE_signal,'.-',label='STE signal',color='blue')
    plt.plot(angles,STE_signal_ctrl,'.-',label='AOM crosstalk',color='green')
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
    plt.show()
    

    return None

def get_bg_level(data):
    num_regions = 2
    
    # region 1
    bg_up = 2
    bg_down = 120
    bg_left = 285
    bg_right = 379
    bg_level = data[..., bg_up:bg_down, bg_left:bg_right].mean(axis=(-2, -1))

    # region 2
    bg_up = 2
    bg_down = 120
    bg_left = 1
    bg_right = 81
    bg_level += data[..., bg_up:bg_down, bg_left:bg_right].mean(axis=(-2, -1))

    return(bg_level / num_regions)

main()
