import os
import numpy as np
import matplotlib.pyplot as plt
import np_tif
from stack_registration import bucket

def main():

    assert os.path.isdir('./../images')
    if not os.path.isdir('./../images/figure_A7'):
        os.mkdir('./../images/figure_A7')
    root_string = (
        './../../stimulated_emission_imaging-data' +
        '/2016_10_24_pulse_length')
    pulse_text = [
        '_4x_long_pulse/',
        '_3x_long_pulse/',
        '_2x_long_pulse/',
        '/',
        ]

    pulsewidths = np.array([4,3,2,1])
    
    # location of center lobe
    bright_spot_x = np.array([160, 160, 162, 182])
    bright_spot_y = np.array([86, 86, 83, 78])-3
    # half of roughtly half width of center lobe
    qtr_width = 10
    
    # cropping area defined
    top_left_x = np.array([110, 110, 111, 130])
    top_left_y = np.array([39, 39, 37, 33]) - 3
    crop_width = 98
    crop_height = 85

    # where on the plot should the cropped images be
    plot_pos_y = [0.655, 0.515, 0.39, 0.11]
    plot_pos_x = [0.16, 0.22, 0.34, 0.67]
    
    
    STE_signal = np.zeros(4)
    STE_signal_relative = np.zeros(4)
    STE_image_cropped = np.zeros((4,10,12))
    num_reps = 200
    num_delays = 5

    for i in range(4):
        pulsewidth = pulsewidths[i]
        folder_string = root_string + '/nanodiamond_7' + pulse_text[i]
        filename = (folder_string +
                    'STE_darkfield_117_5_green_1500mW_red_300mW_many_delays.tif')
        assert os.path.exists(filename)
        data = np_tif.tif_to_array(filename).astype(np.float64)
        filename = (folder_string +
                    'STE_darkfield_117_5_green_0mW_red_300mW' +
                    '_many_delays.tif')
        assert os.path.exists(filename)
        data_bg = np_tif.tif_to_array(filename).astype(np.float64)
        # reshape to hyperstack
        data = data.reshape(
            num_reps, num_delays, data.shape[1], data.shape[2])
        data_bg = data_bg.reshape(
            num_reps, num_delays, data_bg.shape[1], data_bg.shape[2])
        # get rid of overexposed rows at top and bottom of images
        less_rows = 3
        data = data[:, :, less_rows:data.shape[2]-less_rows, :]
        data_bg = data_bg[:, :, less_rows:data_bg.shape[2]-less_rows, :]
        # repetition average
        data = data.mean(axis=0)
        data_bg = data_bg.mean(axis=0)
        # subtract crosstalk
        data = data - data_bg
        # subtract avg of green off images from green on images
        data_simult = data[2,:,:]
        data_non_simult = 0.5 * (data[0,:,:] + data[4,:,:])
        STE_image = data_simult - data_non_simult
        # capture stim emission signal
        my_col = bright_spot_x[i]
        my_row = bright_spot_y[i]
        main_lobe = STE_image[
            my_row-qtr_width:my_row+qtr_width,
            my_col-qtr_width:my_col+qtr_width]
        STE_signal[i] = np.mean(main_lobe)
        STE_signal_relative[i] = STE_signal[i]
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
        # load into final image array
        STE_image_cropped[i, :, :] = STE_image_cropped_single

    # get max and min vals
    STE_max = np.amax(STE_image_cropped)
    STE_min = np.amin(STE_image_cropped)
    STE_image_cropped[:, -2:-1, 1:5] = STE_min # scale bar

    my_intensity = 1/pulsewidths

    fig, ax1 = plt.subplots()
    ax1.plot(my_intensity,STE_signal_relative,'o',color='black',markersize=10)
    plt.ylim(ymin=-140,ymax=0)
    ax1.set_ylabel('Average signal brightness (pixel counts)')
    ax1.tick_params('y', colors='k')
    plt.xlabel('Normalized laser intensity (constant energy)')
    plt.grid()
    for i in range(4):
        a = plt.axes([plot_pos_x[i], plot_pos_y[i], .12, .12])
        plt.imshow(STE_image_cropped[i,:,:], cmap=plt.cm.gray,
                   interpolation='nearest', vmax=STE_max, vmin=STE_min)
        plt.xticks([])
        plt.yticks([])

    # plot energy per exposure
    green_uJ = np.array([54, 54, 54, 54])
    red_uJ = np.array([10, 10, 10, 10])
    ax2 = ax1.twinx()
    ax2.plot(my_intensity, green_uJ, '--b', linewidth=2)
    ax2.plot(my_intensity, red_uJ, '--b', linewidth=2)
    ax2.set_ylabel('Fluence per exposure (µJ)',color='blue')
    ax2.tick_params('y', colors='b')
    ax2.set_ylim(ymin=-1, ymax=61.5)
    ax1.set_xlim(xmin=0,xmax=1.125)

    # annotate with red/green pulses
    im = plt.imread('green_shortpulse.png')
    a = plt.axes([0.773, 0.812, .08, .08], frameon=False)
    plt.imshow(im)
    plt.xticks([])
    plt.yticks([])
    im = plt.imread('green_longpulse.png')
    a = plt.axes([0.24, 0.772, .1, .1], frameon=False)
    plt.imshow(im)
    plt.xticks([])
    plt.yticks([])
    im = plt.imread('red_shortpulse.png')
    a = plt.axes([0.773, 0.25, .08, .08], frameon=False)
    plt.imshow(im)
    plt.xticks([])
    plt.yticks([])
    im = plt.imread('red_longpulse.png')
    a = plt.axes([0.24, 0.21, .1, .1], frameon=False)
    plt.imshow(im)
    plt.xticks([])
    plt.yticks([])

    plt.savefig('./../images/figure_A7/darkfield_nd_pulse_length_scan.svg')
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

main()
