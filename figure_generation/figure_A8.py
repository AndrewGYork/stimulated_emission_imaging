import os
import numpy as np
import matplotlib.pyplot as plt
import np_tif
from stack_registration import bucket


def main():

    assert os.path.isdir('./../images')
    if not os.path.isdir('./../images/figure_A8'):
        os.mkdir('./../images/figure_A8')
    folder_string = (
        './../../stimulated_emission_imaging-data' +
        '/2017_05_09_pulse_length_scan_')

    pulsewidths = np.array([8,4,2,1])
    
    # location of center lobe
    bright_spot_x = np.array([219, 235, 204, 207])-1
    bright_spot_y = np.array([47, 71, 37, 66])-1

    # cropping area defined
    top_left_x = [133, 146, 114, 118]
    top_left_y = [0, 4, 0, 4]#[0, 7, 0, 5]
    crop_width = 175
    crop_height = 118

    # where on the plot should the cropped images be
    plot_pos_y = [0.12, 0.26, 0.4, 0.62]
    plot_pos_x = [0.27, 0.42, 0.58, 0.77]
    
    STE_signal = np.zeros(4)
    STE_signal_relative = np.zeros(4)
    STE_image_cropped = np.zeros((4,14,21))
    qtr_width = 12
    num_reps = 50
    num_delays = 3
    for i in range(4):
        pulsewidth = pulsewidths[i]
        extra_text = ''
        if pulsewidth == 4: extra_text = '_again_good'
        rest_of_folder_string = (str(pulsewidth) + 'us' + extra_text)
        filename = (folder_string + rest_of_folder_string +
                    '/STE_phase_angle_2_green_1395mW_red_300mW.tif')
        assert os.path.exists(filename)
        data = np_tif.tif_to_array(filename).astype(np.float64)
        filename = (folder_string + rest_of_folder_string +
                    '/STE_phase_angle_2_green_0mW_red_300mW.tif')
        assert os.path.exists(filename)
        data_ctrl = np_tif.tif_to_array(filename).astype(np.float64)
        # get rid of overexposed rows at top and bottom of images
        less_rows = 3
        data = data[:, less_rows:data.shape[1] - less_rows, :]
        data_ctrl = data_ctrl[:,less_rows:data_ctrl.shape[1] - less_rows, :]
        # Get the average pixel brightness in the background region of the
        # phase contrast image. We'll use it to account for laser intensity
        # fluctuations
        avg_laser_brightness = get_bg_level(data.mean(axis=0))
        # scale all images to have the same background brightness. This
        # amounts to a correction of roughly 1% or less
        local_laser_brightness = get_bg_level(data)
        data = data * (
            avg_laser_brightness / local_laser_brightness).reshape(
                data.shape[0], 1, 1)
        # do the same for control (green off) data
        local_laser_brightness_ctrl = get_bg_level(data_ctrl)
        data_ctrl = data_ctrl * (
            avg_laser_brightness / local_laser_brightness_ctrl).reshape(
                data_ctrl.shape[0], 1, 1)
        # reshape to hyperstack
        data = data.reshape(
            num_reps, num_delays, data.shape[1], data.shape[2])
        data_ctrl = data_ctrl.reshape(
            num_reps, num_delays, data_ctrl.shape[1], data_ctrl.shape[2])
        # now that brightness is corrected, repetition average the data
        data = data.mean(axis=0)
        data_ctrl = data_ctrl.mean(axis=0)
        # subtract avg of green off images from green on images
        data_simult = data[1, :, :]
        data_non_simult = 0.5 * (data[0, :, :] + data[2, :, :])
        STE_image = data_simult - data_non_simult
        data_simult_ctrl = data_ctrl[1, :, :]
        data_non_simult_ctrl = 0.5 * (data_ctrl[0, :, :] + data_ctrl[2, :, :])
        STE_image_ctrl = data_simult_ctrl - data_non_simult_ctrl
        # subtract AOM effects (even though it doesn't seem like there are any)
        STE_image = STE_image# - STE_image_ctrl
        # capture stim emission signal
        my_col = bright_spot_x[i]
        my_row = bright_spot_y[i]
        main_lobe = STE_image[
            my_row-qtr_width:my_row+qtr_width,
            my_col-qtr_width:my_col+qtr_width]
        left_edge = STE_image[:,qtr_width*2]
        STE_signal[i] = np.mean(main_lobe)
        STE_signal_relative[i] = STE_signal[i] - np.mean(left_edge)
        # crop stim emission image
        STE_image_cropped_single = STE_image[
            top_left_y[i]:top_left_y[i] + crop_height,
            top_left_x[i]:top_left_x[i] + crop_width,
            ]
        # Our pixels are tiny (8.7 nm/pixel) to give large dynamic range.
        # This is not great for viewing, because fluctuations can swamp the
        # signal. This step bins the pixels into a more typical size.
        bucket_width = 8 # bucket width in pixels
        STE_image_cropped_single = bucket(
            STE_image_cropped_single, (bucket_width, bucket_width)
            ) / bucket_width**2
        STE_image_cropped[i, :, :] = STE_image_cropped_single

    # get max/min values for plot color scaling
    STE_max = np.amax(STE_image_cropped)
    STE_min = np.amin(STE_image_cropped)
    STE_image_cropped[:, -2:-1, 1:6] = STE_max # scale bar

##    my_zero = np.zeros(1)
##    STE_signal_relative = np.concatenate((my_zero,STE_signal_relative))
##    my_intensity = np.concatenate((my_zero,1/pulsewidths))
    my_intensity = 1/pulsewidths

    fig, ax1 = plt.subplots()
    lines = ax1.plot(my_intensity,STE_signal_relative,'o--',color='blue')
    plt.setp(lines, linewidth=3, color='b')
    ax1.plot(my_intensity,STE_signal_relative,'o',color='blue', markersize=10)
    ax1.set_xlim(xmin=-0.05,xmax=1.05)
    plt.ylim(ymin=-5,ymax=196)
    ax1.set_ylabel('Average signal brightness (pixel counts)', color='blue')
    ax1.tick_params('y', colors='b')
    plt.xlabel('Normalized laser intensity (constant energy)')
    plt.grid()
    for i in range(4):
        a = plt.axes([plot_pos_x[i], plot_pos_y[i], .12, .12])
        plt.imshow(STE_image_cropped[i,:,:], cmap=plt.cm.gray,
                   interpolation='nearest', vmax=STE_max, vmin=STE_min)
        plt.xticks([])
        plt.yticks([])
    # plot energy per exposure
    green_uJ = np.array([10, 10, 10, 10])
    red_uJ = np.array([2, 2, 2, 2])
    ax2 = ax1.twinx()
    ax2.plot(my_intensity, green_uJ, '--og', linewidth=2)
    ax2.plot(my_intensity, red_uJ, '--or', linewidth=2)
    ax2.set_ylabel('Energy deposited per exposure (µJ)')
    ax2.set_ylim(ymin=-0.3, ymax=11.4)
    ax1.set_xlim(xmin=-0.02,xmax=1.05)

    # annotate with red/green pulses
    im = plt.imread('green_shortpulse.png')
    a = plt.axes([0.82, 0.813, .08, .08], frameon=False)
    plt.imshow(im)
    plt.xticks([])
    plt.yticks([])
    im = plt.imread('green_longpulse.png')
    a = plt.axes([0.175, 0.77, .1, .1], frameon=False)
    plt.imshow(im)
    plt.xticks([])
    plt.yticks([])
    im = plt.imread('red_shortpulse.png')
    a = plt.axes([0.82, 0.265, .08, .08], frameon=False)
    plt.imshow(im)
    plt.xticks([])
    plt.yticks([])
    im = plt.imread('red_longpulse.png')
    a = plt.axes([0.175, 0.225, .1, .1], frameon=False)
    plt.imshow(im)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('./../images/figure_A8/phase_contrast_dye_pulse_length_scan.svg')
    plt.show()
##    plt.close()

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
