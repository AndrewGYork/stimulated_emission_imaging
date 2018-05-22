import os
import numpy as np
import matplotlib.pyplot as plt
import np_tif
from stack_registration import bucket

def main():

    # the data to be plotted by this program is generated from raw tifs
    # and repetition_average_expt_and_control.py

    assert os.path.isdir('./../images')
    if not os.path.isdir('./../images/figure_A6'):
        os.mkdir('./../images/figure_A6')

    filename = (
        './../../stimulated_emission_imaging-data' +
        '/2017_08_25_STE_phase_thermal_nanodiamond_4_5x_longest_delays' +
        '/STE_phase_angle_10_green_1350mW_red_300mW.tif')
    filename_ctrl = (
        './../../stimulated_emission_imaging-data' +
        '/2017_08_25_STE_phase_thermal_nanodiamond_4_5x_longest_delays' +
        '/STE_phase_angle_10_green_0mW_red_300mW.tif')
    data = np_tif.tif_to_array(filename).astype(np.float64)
    data_ctrl = np_tif.tif_to_array(filename_ctrl).astype(np.float64)

    # get rid of overexposed rows at top and bottom of images
    less_rows = 3
    data = data[:,0+less_rows:data.shape[1]-less_rows,:]
    data_ctrl = data_ctrl[:,0+less_rows:data_ctrl.shape[1]-less_rows,:]

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

    # get delays from code copied from
    # modulated_imaging_one_z_STE_phase_thermal.py
    num_reps = 50
    daq_rate = 8e5
    red_pulse_duration_pixels = 4
    green_pulse_duration_pixels = 20
    red_beginning_pixels = np.array([
        -10,
        0,
        30,
        50,
        100,
        150,
        ])
    red_step_pixels = 200
    red_last_pixel = 7200
    red_end_pixels = np.arange(
        red_step_pixels, red_last_pixel, red_step_pixels)
    red_start_pixel_array = np.append(red_beginning_pixels, red_end_pixels)
    red_delays = red_start_pixel_array/daq_rate
    red_pulse_duration = red_pulse_duration_pixels/daq_rate
    green_pulse_duration = green_pulse_duration_pixels/daq_rate
    
    num_delays = red_delays.shape[0]

    # reshape data by reps
    data = data.reshape(
        num_reps, num_delays, data.shape[1], data.shape[2])
    data_ctrl = data_ctrl.reshape(
        num_reps, num_delays, data_ctrl.shape[1], data_ctrl.shape[2])

    # now that brightness is corrected, repetition average the data
    data = data.mean(axis=0)
    data_ctrl = data_ctrl.mean(axis=0)

    # The first delay is negative. Subtract this from the rest of the
    # data in order to find the time delayed change in the phase
    # contrast image due to the excitation pulse.
    early_data = np.array([data[0,:,:]]*num_delays) # red precedes green
    thermal_stack = data - early_data
    early_data_ctrl = np.array([data_ctrl[0,:,:]]*num_delays) # red precedes green
    crosstalk_stack = data_ctrl - early_data_ctrl
    # subtract AOM effects (even though it doesn't seem like there are any)
    thermal_stack = thermal_stack# - crosstalk_stack

    # plot phase contrast image and thermal signal
    top = 0
    bot = 98
    left = 94
    right = 252
    thermal_cropped = thermal_stack[:,top:bot,left:right]

    # Our pixels are tiny (8.7 nm/pixel) to give large dynamic range.
    # This is not great for viewing, because fluctuations can swamp the
    # signal. This step bins the pixels into a more typical size.
    bucket_width = 8 # bucket width in pixels
    thermal_cropped = bucket(
        thermal_cropped, (1, bucket_width, bucket_width)) / bucket_width**2

    # choose three representative thermal images
    thermal_first = thermal_cropped[6,:,:]
    thermal_middle = thermal_cropped[10,:,:]
    thermal_last = thermal_cropped[28,:,:]

    # combine into single array
    three_images = np.array([thermal_first, thermal_middle, thermal_last])
    # get max/min values
    thermal_max = np.amax(three_images)
    thermal_min = np.amin(three_images)
    # scale bar
    three_images[:, -2:-1, 1:6] = thermal_min
    
    
    # average points around center lobe of the nanodiamond image to get
    # "average signal level" for darkfield and STE images
    top = 13
    bot = top + 48
    left = 148
    right = left + 48
    thermal_signal = (
        thermal_stack[:,top:bot,left:right].mean(axis=2).mean(axis=1))
    green_pulse = np.array([
        0,
        0,
        1,
        1,
        0,
        0,
        ])
##    green_pulse = (1 - green_pulse) * np.amin(thermal_signal)
    dt = 1e-9
    green_pulse_tail = 5e-5
    green_pulse_time_axis = np.array([
        -green_pulse_tail,
        0-dt,
        0,
        green_pulse_duration,
        green_pulse_duration + dt,
        green_pulse_duration + green_pulse_tail,
        ])
        

    
    # plot signal v phase
    max_delay_num = 28
    fig = plt.figure(figsize = (12.5,4.5))
    ax = fig.add_subplot(111)
    lns1 = ax.plot(
        red_delays[0:max_delay_num]*1e3,
        thermal_signal[0:max_delay_num],
        'o-',color='red',
        label='Delayed phase contrast signal',
        )
    ax2 = ax.twinx()
    lns2 = ax2.plot(
        green_pulse_time_axis*1e3,
        green_pulse,
        '-',color='green',
        label='Excitation laser duration',
        )

    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='lower right')

    ax.grid()
    ax.set_xlabel('Time delay between red and green pulses (ms)')
    ax.set_ylabel('Average pixel count', color='red')
    ax.tick_params('y', colors='red')
    ax2.set_ylabel('Laser power (arb. units)', color='green')
    ax2.tick_params('y', colors='green')
    ax.set_xlim([-0.25,5.75])
    ax.set_ylim(-550,25)
    ax2.set_ylim([-0.05,1.1])

    a = plt.axes([.21, .13, .18, .18])
    plt.imshow(three_images[0], cmap=plt.cm.gray,
               interpolation='nearest', vmax=thermal_max, vmin=thermal_min)
    plt.xticks([])
    plt.yticks([])
    a = plt.axes([.33, .36, .18, .18])
    plt.imshow(three_images[1], cmap=plt.cm.gray,
               interpolation='nearest', vmax=thermal_max, vmin=thermal_min)
    plt.xticks([])
    plt.yticks([])
    a = plt.axes([.7, .53, .18, .18])
    plt.imshow(three_images[2], cmap=plt.cm.gray,
               interpolation='nearest', vmax=thermal_max, vmin=thermal_min)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('./../images/figure_A6/phase_contrast_nd_delayed_signal.svg')
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
