import os
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import np_tif
from stack_registration import bucket

def main():

    assert os.path.isdir('./../images')
    if not os.path.isdir('./../images/figure_4'):
        os.mkdir('./../images/figure_4')

    less_rows = 3 # top and bottom image rows tend to saturate
    num_reps = 3000 #original number of reps
    reps_avgd = 1
    reps_per_set = int(num_reps/reps_avgd)
    num_delays = 3
    dark_counts = 100
    sets = ['a', 'b', 'c']
    image_center = np.array([
        [52, 192], [52, 192], [52, 192]])
    # change image center coordinate height due to cropping
    image_center = image_center - np.array([less_rows, 0])
    assert len(sets) == len(image_center)
    height = 128
    width = 380
    lbhw = 28 # half width of box around main image lobe
    # get bg level for brightness calibration across all data sets
    ref_data_set = 'a'
    ref_filename = (
        './../../stimulated_emission_imaging-data' +
        '/2018_03_05_STE_phase_bleaching_orange_bead_14_' + ref_data_set +
        '/STE_phase_angle_1_green_130mW_red_230mW.tif')
    set_data = np_tif.tif_to_array(
        ref_filename).astype(np.float64) - dark_counts
    set_data = set_data.reshape(reps_per_set,num_delays,height,width)
    # get rid of overexposed rows at top and bottom of images
    set_data = set_data[:, :, 0+less_rows:height-less_rows, :]
    
    avg_laser_brightness = get_bg_level(set_data.mean(axis=(0, 1)))
    

    all_STE_images = np.zeros((reps_per_set*len(sets),height-less_rows*2,width))
    STE_signal = np.zeros((reps_per_set*len(sets)))
    bg_signal = np.zeros((reps_per_set*len(sets)))
    
    for my_index, my_set in enumerate(sets):
        filename = (
            './../../stimulated_emission_imaging-data' +
            '/2018_03_05_STE_phase_bleaching_orange_bead_14_' + my_set +
            '/STE_phase_angle_1_green_130mW_red_230mW.tif')
        set_data = np_tif.tif_to_array(
            filename).astype(np.float64) - dark_counts
        assert set_data.shape == (reps_per_set*num_delays,height,width)
        set_data = set_data.reshape(reps_per_set,num_delays,height,width)
        set_data = set_data[:, :, 0+less_rows:height-less_rows, :]

        # scale all images to have the same background brightness. This
        # amounts to a correction of roughly 1% or less
        local_laser_brightness = get_bg_level(set_data)
        set_data = set_data * (
            avg_laser_brightness / local_laser_brightness).reshape(
                set_data.shape[0], set_data.shape[1], 1, 1)

        # get zero delay and max delay images
        zero_delay_images = set_data[:, 1, :, :]
        max_delay_images = set_data[:, 0:3:2, :, :].mean(axis=1)

        # local range in global set
        begin = my_index * reps_per_set
        end = begin + reps_per_set

        # stim emission image is the image with green/red simultaneous minus
        # image with/red green not simultaneous
        STE_image_set = zero_delay_images - max_delay_images
        all_STE_images[begin:end] = STE_image_set

        # average points around main STE image lobe and add to STE_signal list
        ste_y, ste_x = image_center[my_index]
        STE_signal[begin:end] = STE_image_set[
            :,
            ste_y - lbhw:ste_y + lbhw,
            ste_x - lbhw:ste_x + lbhw
            ].mean(axis=2).mean(axis=1)

    
    # average consecutive STE images for better SNR (mandatory)
    bucket_width = 50
    all_STE_images = bucket(
        all_STE_images, (bucket_width, 1, 1)) / bucket_width

    # average consecutive STE signal levels for better SNR (optional)
    signal_bucket_width = 50
    orig_STE_signal = STE_signal
    STE_signal = np.array([STE_signal])
    STE_signal = bucket(
        STE_signal, (1, signal_bucket_width)) / signal_bucket_width
    STE_signal = STE_signal[0, :]

    # choose images to display and laterally smooth them
    sigma = 9 # tune this parameter to reject high spatial frequencies
    STE_display_imgs = np.array([
        all_STE_images[0, :, :],
        all_STE_images[int(all_STE_images.shape[0] / 2), :, :],
        all_STE_images[-1, :, :]])    
    STE_display_imgs = gaussian_filter(
        STE_display_imgs, sigma=(0, sigma, sigma))

    # Our pixels are tiny (8.7 nm/pixel) to give large dynamic range.
    # This is not great for viewing, because fluctuations can swamp the
    # signal. This step bins the pixels into a more typical size.
    bucket_width = 8 # bucket width in pixels
    STE_display_imgs = bucket(
        STE_display_imgs, (1, bucket_width, bucket_width)) / bucket_width**2

    # crop images left and right sides
    less_cols = 5
    STE_display_imgs = STE_display_imgs[:, :, less_cols:-less_cols]

    # get max and minimum values to display images with unified color scale
    max_pixel_value = np.max(STE_display_imgs)
    min_pixel_value = np.min(STE_display_imgs)
    print(max_pixel_value, min_pixel_value)
    max_pixel_value = 102.6 # from figure_8_crimson.py
    min_pixel_value = -49 # from figure_8_crimson.py

    STE_display_imgs[:, -2, 1:6] = max_pixel_value # scale bar

    # number of accumulated excitation pulses for x axis of bleaching plot
    pulses_per_exposure = 8
    exposures_per_delay_scan = 3
    num_delay_scans = len(sets) * num_reps
    orig_pulses_axis = (np.arange(num_delay_scans) *
                        pulses_per_exposure *
                        exposures_per_delay_scan)
    pulses_axis = (
        (np.arange(num_delay_scans / signal_bucket_width) + 0.5) *
        pulses_per_exposure *
        exposures_per_delay_scan *
        signal_bucket_width)

    plt.figure(figsize=(13,5))
    plt.plot(
        orig_pulses_axis, orig_STE_signal,
        'o', markersize = 2.5,
        markerfacecolor='none', markeredgecolor='blue')
    plt.plot(pulses_axis, STE_signal, color='red')

    # lines from images to data points
##    plt.plot(
##        [pulses_axis[0], 50189],
##        [STE_signal[0], 76],
##        'k--', lw=2
##        )
##    plt.plot(
##        [pulses_axis[0], 162113],
##        [STE_signal[0], 76],
##        'k--', lw=2
##        )
    for x in np.arange(50189, 162114, 1000):
        plt.plot(
            [pulses_axis[0], x],
            [STE_signal[0], 76],
            'k', lw=0.1,
            )
##    plt.plot(
##        [pulses_axis[int(pulses_axis.shape[0] / 2)], 213673],
##        [STE_signal[int(STE_signal.shape[0] / 2)], 76],
##        'k--', lw=2
##        )
##    plt.plot(
##        [pulses_axis[int(pulses_axis.shape[0] / 2)], 325597],
##        [STE_signal[int(STE_signal.shape[0] / 2)], 76],
##        'k--', lw=2
##        )
    for x in np.arange(213673, 325598, 1000):
        plt.plot(
            [pulses_axis[int(pulses_axis.shape[0] / 2)], x],
            [STE_signal[int(STE_signal.shape[0] / 2)], 76],
            'k', lw=0.1,
            )
##    plt.plot(
##        [pulses_axis[-1], 377157],
##        [STE_signal[-1], 76],
##        'k--', lw=2
##        )
##    plt.plot(
##        [pulses_axis[-1], 489080],
##        [STE_signal[-1], 76],
##        'k--', lw=2
##        )
    for x in np.arange(377157, 489081, 1000):
        plt.plot(
            [pulses_axis[-1], x],
            [STE_signal[-1], 76],
            'k', lw=0.1,
            )

    
##    plt.axis([0-2000, 74000, -25, 110])
    plt.axis([-2000, 502800 + 2000, -25, 110])
##    plt.axis([0, np.max(pulses_axis), -25, 50])
    plt.grid()
    plt.ylabel('Average pixel brightness (sCMOS counts)', fontsize=14)
    plt.xlabel('Number of excitation pulses delivered to sample', fontsize=18)
    a = plt.axes([.2, .7, .18, .18])
    plt.imshow(
        STE_display_imgs[0, :, :],
        cmap=plt.cm.gray,
        interpolation='nearest',
        vmax = max_pixel_value,
        vmin = min_pixel_value)
    plt.xticks([])
    plt.yticks([])
    a = plt.axes([.45, .7, .18, .18])
    plt.imshow(
        STE_display_imgs[1, :, :],
        cmap=plt.cm.gray,
        interpolation='nearest',
        vmax = max_pixel_value,
        vmin = min_pixel_value)
    plt.xticks([])
    plt.yticks([])
    a = plt.axes([.7, .7, .18, .18])
    plt.imshow(
        STE_display_imgs[2, :, :],
        cmap=plt.cm.gray,
        interpolation='nearest',
        vmax = max_pixel_value,
        vmin = min_pixel_value)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('./../images/figure_4/STE_v_fluence_orange.svg')
    plt.savefig('./../images/figure_4/STE_v_fluence_orange.png', dpi=300)
    plt.show()


    return None

def get_bg_level(data):
    num_regions = 2
    
    # region 1
    bg_up = 5#9
    bg_down = 123#115
    bg_left = 285#335
    bg_right = 379#373
    bg_level = data[..., bg_up:bg_down, bg_left:bg_right].mean(axis=(-2, -1))

    # region 2
    bg_up = 5
    bg_down = 123
    bg_left = 1
    bg_right = 81
    bg_level += data[..., bg_up:bg_down, bg_left:bg_right].mean(axis=(-2, -1))

    return(bg_level / num_regions)

main()
