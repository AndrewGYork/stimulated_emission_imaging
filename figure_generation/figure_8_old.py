import os
import numpy as np
import np_tif
import matplotlib.pyplot as plt

def main():

    


##    filename = 'STE_phase_angle_1_green_1400mW_red_300mW.tif'
##    assert os.path.exists(filename)
##    data1 = np_tif.tif_to_array(filename).astype(np.float64)
##    num_reps = 4000
##    assert data1.shape == (3*num_reps, 128, 380)
##
##    filename = 'STE_phase_angle_2_green_1400mW_red_300mW.tif'
##    assert os.path.exists(filename)
##    data2 = np_tif.tif_to_array(filename).astype(np.float64)
##    num_reps = 4000
##    assert data2.shape == (3*num_reps, 128, 380)
##
##    filename = 'STE_phase_angle_3_green_1400mW_red_300mW.tif'
##    assert os.path.exists(filename)
##    data3 = np_tif.tif_to_array(filename).astype(np.float64)
##    num_reps = 4000
##    assert data3.shape == (3*num_reps, 128, 380)
##
##    num_reps = 12000
##    data = np. zeros((num_reps*3,128,380))
##    data[0:4000*3,:,:] = data1
##    data[4000*3:8000*3,:,:] = data2
##    data[8000*3:12000*3,:,:] = data3
##
##
##    data = data.reshape(num_reps, 3, 128, 380) # Stack to hyperstack
##    frames_per_interval = 100
##    num_points = int(num_reps/frames_per_interval)
##    image_stack = np.zeros((num_points,3,128,380))
##    for point in range(num_points):
##        image_stack[point,:,:,:] = data[
##            point*frames_per_interval:(point+1)*frames_per_interval,:,:,:,
##            ].mean(axis=0,keepdims=True)
####    image_stack[0,:,:,:] = data[0:99,:,:,:].mean(axis=0,keepdims=True)
####    image_stack[1,:,:,:] = data[100:199,:,:,:].mean(axis=0,keepdims=True)
####    image_stack[2,:,:,:] = data[200:299,:,:,:].mean(axis=0,keepdims=True)
####    image_stack[3,:,:,:] = data[300:399,:,:,:].mean(axis=0,keepdims=True)
####    image_stack[4,:,:,:] = data[400:499,:,:,:].mean(axis=0,keepdims=True)
##
##    # get rid of overexposed rows at top and bottom of images
##    less_rows = 3
##    image_stack = image_stack[:,:,0+less_rows:image_stack.shape[2]-less_rows,:]
##
####    # crop images
####    less_cols = 80
####    image_stack = image_stack[:,:,:,0+less_cols:image_stack.shape[3]]
##    
##    # compute control and stim emission images
##    phase_stack = 0.5 * (image_stack[:,0,:,:]+image_stack[:,2,:,:])
##    STE_stack = image_stack[:,1,:,:] - phase_stack
##    control_level = np.mean(STE_stack,(1,2),keepdims=True)
####    STE_stack = STE_stack-control_level
##
##    np_tif.array_to_tif(STE_stack,'ste_stack.tif')

    assert os.path.isdir('./../images')
    if not os.path.isdir('./../images/figure_8'):
        os.mkdir('./../images/figure_8')

    filename = (
        './../../stimulated_emission_data/figure_8/ste_stack.tif')

    STE_stack = np_tif.tif_to_array(filename)

    # crop images
    STE_cropped = STE_stack[:,:,80:STE_stack.shape[2]]

    # get STE images ready to plot
    for image_num in range(STE_cropped.shape[0]):
        # filter darkfield and STE images
        STE_image = STE_cropped[image_num,:,:]
        STE_image = STE_image.reshape(
            1,STE_image.shape[0],STE_image.shape[1])
        STE_image = annular_filter(STE_image,r1=0,r2=0.025)
        STE_image = STE_image[0,:,:]
##        phase_image = phase_cropped[image_num,:,:]
##        phase_image = phase_image.reshape(
##            1,phase_image.shape[0],phase_image.shape[1])
##        phase_image = annular_filter(phase_image,r1=0,r2=0.02)
##        phase_image = phase_image[0,:,:]

        # generate and save plot
##        print(np.max(STE_image,(0,1)))
##        print(np.min(STE_image,(0,1)))
        STE_image[0,0] =  167 # cheap way to conserve colorbar
        STE_image[1,0] = -88 # cheap way to conserve colorbar
##        phase_image[0,0] = 18500 #cheap way to conserve colorbar
##        phase_image[1,0] = -13000 #cheap way to conserve colorbar
        STE_image[108:114,5:34] = 167 # scale bar
##        phase_image[108:114,5:34] = 1500 # scale bar
        STE_cropped[image_num,:,:] = STE_image


    # choose three representative STE images
    STE_first = STE_cropped[0,:,:]
    STE_middle = STE_cropped[59,:,:]
    STE_last = STE_cropped[118,:,:]

    #plot and save image
##    fig, (ax0) = plt.subplots(nrows=1,ncols=1)
##
##    cax0 = ax0.imshow(phase_image, cmap=plt.cm.gray)
##    ax0.axis('off')
##    cbar0 = fig.colorbar(cax0,ax=ax0)
##    ax0.set_title('Phase contrast image of crimson bead')
##
##    cax0 = ax0.imshow(STE_image, cmap=plt.cm.gray)
##    cbar0 = fig.colorbar(cax0, ax = ax0)
##    ax0.set_title('Change in phase contrast image due to stim. emission')
##    ax0.axis('off')
##    plt.savefig(
##        './../images/figure_8/STE_imag_delay_' + str(image_num)+'.svg')
##    plt.close()

    # compute and plot signal v accumulated fluence
    
    STE_signal = np.mean(STE_stack[:,25:86,187:252],(1,2))
##    STE_signal = STE_signal - np.min(STE_signal)

    intensity = 4e6 #W/cm^2
    bead_radius_nm = 100
    bead_radius_cm = bead_radius_nm * 1e-7
    bead_area = 3.14* bead_radius_cm**2
    pulse_duration_s = 1e-6
    fluence_per_pulse_Joules = intensity*bead_area*pulse_duration_s
    pulses_per_frame = 10
    pulses_per_delay_scan = pulses_per_frame * 3
    delay_scans_per_data_point = 100
    fluence_per_time_unit = (fluence_per_pulse_Joules *
                             pulses_per_delay_scan *
                             delay_scans_per_data_point
                             )
##    time_units_elapsed = np.array((1,
##                                   2,
##                                   3,
##                                   4,
##                                   5,
##                                   6,
##                                   3*14,
##                                   ))
    time_units_elapsed = np.arange(1,121)
    accumulated_fluence_uJ = fluence_per_time_unit * time_units_elapsed * 1e6

    plt.figure(figsize=(13,5))
    plt.plot(accumulated_fluence_uJ,STE_signal,'o',color='red')
##    plt.title('Phase contrast stimulated emission peak signal')
    plt.ylabel('Average pixel count',fontsize=18)
    plt.xlabel('Accumulated excitation fluence (microJoules)',fontsize=18)
    plt.grid()
    a = plt.axes([.2, .7, .18, .18])
    plt.imshow(STE_first, cmap=plt.cm.gray)
    plt.xticks([])
    plt.yticks([])
    a = plt.axes([.42, .43, .18, .18])
    plt.imshow(STE_middle, cmap=plt.cm.gray)
    plt.xticks([])
    plt.yticks([])
    a = plt.axes([.7, .26, .18, .18])
    plt.imshow(STE_last, cmap=plt.cm.gray)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('./../images/figure_8/STE_v_fluence.svg')
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
