import os
import numpy as np
import np_tif
from stack_registration import stack_registration
import matplotlib.pyplot as plt

def main():

    # the data to be plotted by this program is generated from raw tifs
    # and repetition_average_expt_and_control.py

    assert os.path.isdir('./../images')
    if not os.path.isdir('./../images/figure_A6'):
        os.mkdir('./../images/figure_A6')

    filename = (
        './../../stimulated_emission_data/figure_A6/dataset_green_1350mW.tif')
    filename_ctrl = (
        './../../stimulated_emission_data/figure_A6/dataset_green_0mW.tif')
    data = np_tif.tif_to_array(filename).astype(np.float64)
    data_ctrl = np_tif.tif_to_array(filename_ctrl).astype(np.float64)

    # get rid of overexposed rows at top and bottom of images
    less_rows = 3
    data = data[:,0+less_rows:data.shape[1]-less_rows,:]
    data_ctrl = data_ctrl[:,0+less_rows:data_ctrl.shape[1]-less_rows,:]

##    # combine experiment and control images
##    data_combined = np.zeros((2,data.shape[0],data.shape[1],data.shape[2]))
##    data_combined[0] = data
##    data_combined[1] = data_ctrl

##    # register each control slice with the corresponding experimental slice
##    fmm = 0.02 #fourier mask magnitude is a carefully tuned parameter
##    for which_slice in range(data.shape[0]):
##        stack_registration(
##            data_combined[:,which_slice,:,:],
##            fourier_mask_magnitude = fmm,
##            )

##    # reshape to hyperstack
##    data = data_combined[0]
##    data_ctrl = data_combined[1]
##    num_delays = 41
##    print(data.shape)
##    data = data.reshape((
##        data.shape[0]/num_delays,
##        num_delays,
##        data.shape[1],
##        data.shape[2],
##        ))
##    print (data.shape)
##    print(data.shape[4444444444])
##    data_ctrl = data_ctrl.reshape((
##        data_ctrl.shape[0]/num_delays,
##        num_delays,
##        data_ctrl.shape[1],
##        data_ctrl.shape[2],
##        ))

    # get delays from code copied from
    # modulated_imaging_one_z_STE_phase_thermal.py
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

    # The first delay is negative. Subtract this from the rest of the
    # data in order to find the time delayed change in the phase
    # contrast image due to the excitation pulse.
    early_data = np.array([data[0,:,:]]*num_delays) # red precedes green
    thermal_stack = data - early_data
    early_data_ctrl = np.array([data_ctrl[0,:,:]]*num_delays) # red precedes green
    crosstalk_stack = data_ctrl - early_data_ctrl
    thermal_stack = thermal_stack - crosstalk_stack

    # plot phase contrast image and thermal signal
    top = 0
    bot = 98
    left = 94
    right = 252
    thermal_cropped = thermal_stack[:,top:bot,left:right]

    for delay_num in range(thermal_cropped.shape[0]):
        # filter thermal images
        thermal_image = thermal_cropped[delay_num,:,:]
        thermal_image = thermal_image.reshape(
            1,thermal_image.shape[0],thermal_image.shape[1])
        thermal_image = annular_filter(thermal_image,r1=0,r2=0.03)
        thermal_image = thermal_image[0,:,:]
        

        # generate and save plot
##        print(np.amax(thermal_image))
##        print(np.amin(thermal_image))
        thermal_image[0,0] = 416 # cheap way to conserve colorbar
##        thermal_image[1,0] = -1091 # cheap way to conserve colorbar
        thermal_image[88:94,5:34] = -1091 # scale bar
        thermal_cropped[delay_num,:,:] = thermal_image

        
        fig, ax0 = plt.subplots(nrows=1,ncols=1)

        cax0 = ax0.imshow(thermal_image, cmap=plt.cm.gray)
        ax0.axis('off')

##        plt.savefig('./../images/figure_A6/phase_STE_image_' +
##                    str(angle_num)+'.svg')
        plt.close()

    # choose three representative thermal images
    thermal_first = thermal_cropped[6,:,:]
    thermal_middle = thermal_cropped[10,:,:]
    thermal_last = thermal_cropped[28,:,:]
    
    
    # average points around center lobe of the nanodiamond image to get
    # "average signal level" for darkfield and STE images
    top = 26
    bot = 55
    left = 156
    right = 286
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
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Average pixel count', color='red')
    ax.tick_params('y', colors='red')
    ax2.set_ylabel('Laser power (arb. units)', color='green')
    ax2.tick_params('y', colors='green')
    ax.set_xlim([-0.25,5.75])
    ax.set_ylim(-400,25)
    ax2.set_ylim([-0.05,1.1])

    a = plt.axes([.18, .13, .18, .18])
    plt.imshow(thermal_first, cmap=plt.cm.gray)
    plt.xticks([])
    plt.yticks([])
    a = plt.axes([.26, .4, .18, .18])
    plt.imshow(thermal_middle, cmap=plt.cm.gray)
    plt.xticks([])
    plt.yticks([])
    a = plt.axes([.7, .6, .18, .18])
    plt.imshow(thermal_last, cmap=plt.cm.gray)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('./../images/figure_A6/phase_contrast_nd_delayed_signal.svg')
##    plt.show()
    plt.close()
    

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
