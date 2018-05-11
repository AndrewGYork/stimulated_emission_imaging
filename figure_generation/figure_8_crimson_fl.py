import os
import numpy as np
import np_tif
import matplotlib.pyplot as plt
from stack_registration import bucket

def main():

    assert os.path.isdir('./../images')
    if not os.path.isdir('./../images/figure_8'):
        os.mkdir('./../images/figure_8')

    crop_rows = 3 # these image rows tend to saturate
    num_reps = 3000 #original number of reps
    reps_avgd = 1
    reps_per_set = int(num_reps/reps_avgd)
    num_delays = 3
    sets = ['a', 'b', 'c', 'd', 'e']
    image_center = [
        [75, 179],
        [77, 180],
        [79, 180],
        [82, 180],
        [84, 177]]
    assert len(sets) == len(image_center)
    height = 128
    width = 380
    lbhw = 28 # half width of box around main image lobe
    crop_half_height = 43
    crop_half_width = 80
    bg_up = 9
    bg_down = 115
    bg_left = 335
    bg_right = 373
    bg_level = 101.7

    all_fl_images = np.zeros((
        reps_per_set*len(sets),
        crop_half_height * 2,
        crop_half_width * 2))
    fl_signal = np.zeros((reps_per_set*len(sets)))
    bg_signal = np.zeros((reps_per_set*len(sets)))
    
    for my_index, my_set in enumerate(sets):
        filename = (
            './../../stimulated_emission_imaging-data' +
            '/2018_05_08_crimson_fluorescence_depletion_bleaching_bead_8' +
            '/STE_depletion_green_1010mW_red_230mW_' + my_set + '.tif')
        set_data = np_tif.tif_to_array(filename).astype(np.float64)
        assert set_data.shape == ((reps_per_set+1)*num_delays,height,width)
        set_data = set_data.reshape(reps_per_set+1,num_delays,height,width)
        set_data = set_data[1::, ...]
        begin = my_index * reps_per_set
        end = begin + reps_per_set

        fl_y, fl_x = image_center[my_index]

        # for fluorescence images just average all three images in delay scan
        fl_image_set = set_data.mean(axis=1)
        all_fl_images[begin:end, :, :] = fl_image_set[
            :,
            fl_y - crop_half_height:fl_y + crop_half_height,
            fl_x - crop_half_width:fl_x + crop_half_width]

        # average points around main STE image lobe and add to STE_signal list
        fl_signal[begin:end] = fl_image_set[
            :,
            fl_y - lbhw:fl_y + lbhw,
            fl_x - lbhw:fl_x + lbhw
            ].mean(axis=2).mean(axis=1)

        # average lots of points away from STE image
        bg_signal[begin:end] = fl_image_set[
            :, bg_up:bg_down, bg_left:bg_right].mean(axis=2).mean(axis=1)

    # subtract background signal
    fl_signal -= bg_level
    all_fl_images -= bg_level

    # average consecutive fl images for better SNR of weak fluorescence (optional)
    bucket_width = 1
    all_fl_images = bucket(
        all_fl_images, (bucket_width, 1, 1)) / bucket_width

    # average consecutive STE signal levels for better SNR (optional)
    signal_bucket_width = 50
    orig_fl_signal = fl_signal
    fl_signal = np.array([fl_signal])
    fl_signal = bucket(
        fl_signal, (1, signal_bucket_width)) / signal_bucket_width
    fl_signal = fl_signal[0, :]

    # choose images to display
    fl_display_imgs = np.array([
        all_fl_images[0, :, :],
        all_fl_images[int(all_fl_images.shape[0] / 2), :, :],
        all_fl_images[-1, :, :]])

    # Our pixels are tiny (8.7 nm/pixel) to give large dynamic range.
    # This is not great for viewing, because fluctuations can swamp the
    # signal. This step bins the pixels into a more typical size.
    bucket_width = 8 # bucket width in pixels
    fl_display_imgs = bucket(
        fl_display_imgs, (1, bucket_width, bucket_width)) / bucket_width**2

    # get max and minimum values to display images with unified color scale
    max_pixel_value = np.max(fl_display_imgs)
    min_pixel_value = np.min(fl_display_imgs)
    print(max_pixel_value, min_pixel_value)

    fl_display_imgs[:, -2, 1:6] = max_pixel_value # scale bar

    orig_fl_signal_norm = orig_fl_signal/max(orig_fl_signal) # normalize

    half_level = np.argmin(np.absolute(orig_fl_signal_norm - 0.5))
    quarter_level = np.argmin(np.absolute(orig_fl_signal_norm - 0.25))
    eighth_level = np.argmin(np.absolute(orig_fl_signal_norm - 0.125))
    pulses_100h = 3 * 8 * half_level
    pulses_hq = 3 * 8 * (quarter_level - half_level)
    pulses_qe = 3 * 8 * (eighth_level - quarter_level)

    print('100% to half level in', pulses_100h, 'pulses')
    print('Half to quarter level in', pulses_hq, 'pulses')
    print('Quarter to eighth level in', pulses_qe, 'pulses')
    
    # number of accumulated excitation pulses for x axis of bleaching plot
    pulses_per_exposure = 8
    exposures_per_delay_scan = 3
    num_delay_scans = len(sets) * num_reps
    orig_pulses_axis = (
        np.arange(num_delay_scans) *
        pulses_per_exposure *
        exposures_per_delay_scan)
    pulses_axis = (
        np.arange(num_delay_scans / signal_bucket_width) *
        pulses_per_exposure *
        exposures_per_delay_scan *
        signal_bucket_width)

    # finally plot
    plt.figure(figsize=(13,5))
    plt.plot(
        orig_pulses_axis, orig_fl_signal_norm,
        'o', markersize = 2.5,
        markerfacecolor='none', markeredgecolor='blue')
    plt.plot(pulses_axis, fl_signal, color='red')
##    plt.axis([0-2000, 74000, -25, 110])
    plt.axis([0-2000, 502800 + 2000, -.25, 1.10])
##    plt.axis([0-2000, np.max(pulses_axis)+2000, -5, 110])
    plt.grid()
    plt.ylabel('Fluorescence brightness (arb. units)', fontsize=14)
    plt.xlabel('Number of excitation pulses delivered to sample', fontsize=18)
    a = plt.axes([.2, .7, .18, .18])
    plt.imshow(
        fl_display_imgs[0, :, :],
        cmap=plt.cm.gray,
        interpolation='nearest',
        vmax = max_pixel_value,
        vmin = min_pixel_value)
    plt.xticks([])
    plt.yticks([])
    a = plt.axes([.45, .7, .18, .18])
    plt.imshow(
        fl_display_imgs[1, :, :],
        cmap=plt.cm.gray,
        interpolation='nearest',
        vmax = max_pixel_value,
        vmin = min_pixel_value)
    plt.xticks([])
    plt.yticks([])
    a = plt.axes([.7, .7, .18, .18])
    plt.imshow(
        fl_display_imgs[2, :, :],
        cmap=plt.cm.gray,
        interpolation='nearest',
        vmax = max_pixel_value,
        vmin = min_pixel_value)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('./../images/figure_8/fl_v_fluence_crimson.svg')
    plt.show()

##    STE_stack = np_tif.tif_to_array(filename)
##
##    # crop images
##    STE_cropped = STE_stack[:,:,80:STE_stack.shape[2]]
##
##    # get STE images ready to plot
##    for image_num in range(STE_cropped.shape[0]):
##        # filter darkfield and STE images
##        STE_image = STE_cropped[image_num,:,:]
##        STE_image = STE_image.reshape(
##            1,STE_image.shape[0],STE_image.shape[1])
##        STE_image = annular_filter(STE_image,r1=0,r2=0.025)
##        STE_image = STE_image[0,:,:]
####        phase_image = phase_cropped[image_num,:,:]
####        phase_image = phase_image.reshape(
####            1,phase_image.shape[0],phase_image.shape[1])
####        phase_image = annular_filter(phase_image,r1=0,r2=0.02)
####        phase_image = phase_image[0,:,:]
##
##        # generate and save plot
####        print(np.max(STE_image,(0,1)))
####        print(np.min(STE_image,(0,1)))
##        STE_image[0,0] =  167 # cheap way to conserve colorbar
##        STE_image[1,0] = -88 # cheap way to conserve colorbar
####        phase_image[0,0] = 18500 #cheap way to conserve colorbar
####        phase_image[1,0] = -13000 #cheap way to conserve colorbar
##        STE_image[108:114,5:34] = 167 # scale bar
####        phase_image[108:114,5:34] = 1500 # scale bar
##        STE_cropped[image_num,:,:] = STE_image
##
##
##    # choose three representative STE images
##    STE_first = STE_cropped[0,:,:]
##    STE_middle = STE_cropped[59,:,:]
##    STE_last = STE_cropped[118,:,:]
##
##    #plot and save image
####    fig, (ax0) = plt.subplots(nrows=1,ncols=1)
####
####    cax0 = ax0.imshow(phase_image, cmap=plt.cm.gray)
####    ax0.axis('off')
####    cbar0 = fig.colorbar(cax0,ax=ax0)
####    ax0.set_title('Phase contrast image of crimson bead')
####
####    cax0 = ax0.imshow(STE_image, cmap=plt.cm.gray)
####    cbar0 = fig.colorbar(cax0, ax = ax0)
####    ax0.set_title('Change in phase contrast image due to stim. emission')
####    ax0.axis('off')
####    plt.savefig(
####        './../images/figure_8/STE_imag_delay_' + str(image_num)+'.svg')
####    plt.close()
##
##    # compute and plot signal v accumulated fluence
##    
##    STE_signal = np.mean(STE_stack[:,25:86,187:252],(1,2))
####    STE_signal = STE_signal - np.min(STE_signal)
##
##    intensity = 4e6 #W/cm^2
##    bead_radius_nm = 100
##    bead_radius_cm = bead_radius_nm * 1e-7
##    bead_area = 3.14* bead_radius_cm**2
##    pulse_duration_s = 1e-6
##    fluence_per_pulse_Joules = intensity*bead_area*pulse_duration_s
##    pulses_per_frame = 10
##    pulses_per_delay_scan = pulses_per_frame * 3
##    delay_scans_per_data_point = 100
##    fluence_per_time_unit = (fluence_per_pulse_Joules *
##                             pulses_per_delay_scan *
##                             delay_scans_per_data_point
##                             )
####    time_units_elapsed = np.array((1,
####                                   2,
####                                   3,
####                                   4,
####                                   5,
####                                   6,
####                                   3*14,
####                                   ))
##    time_units_elapsed = np.arange(1,121)
##    accumulated_fluence_uJ = fluence_per_time_unit * time_units_elapsed * 1e6
##
##    plt.figure(figsize=(13,5))
##    plt.plot(accumulated_fluence_uJ,STE_signal,'o',color='red')
####    plt.title('Phase contrast stimulated emission peak signal')
##    plt.ylabel('Average pixel count',fontsize=18)
##    plt.xlabel('Accumulated excitation fluence (microJoules)',fontsize=18)
##    plt.grid()
##    a = plt.axes([.2, .7, .18, .18])
##    plt.imshow(STE_first, cmap=plt.cm.gray)
##    plt.xticks([])
##    plt.yticks([])
##    a = plt.axes([.42, .43, .18, .18])
##    plt.imshow(STE_middle, cmap=plt.cm.gray)
##    plt.xticks([])
##    plt.yticks([])
##    a = plt.axes([.7, .26, .18, .18])
##    plt.imshow(STE_last, cmap=plt.cm.gray)
##    plt.xticks([])
##    plt.yticks([])
####    plt.savefig('./../images/figure_8/STE_v_fluence.svg')
##    plt.show()
####    plt.close()
    
    


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
