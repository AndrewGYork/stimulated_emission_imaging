import os
import numpy as np
import np_tif
import matplotlib.pyplot as plt

def main():

##    assert os.path.isdir('./../images')
##    if not os.path.isdir('./../images/figure_8'):
##        os.mkdir('./../images/figure_8')

    crop_rows = 3 # these image rows tend to saturate
    num_reps = 3000 #original number of reps
    reps_avgd = 1
    reps_per_set = int(num_reps/reps_avgd)
    num_delays = 3
    sets = ['a', 'b', 'c', 'd', 'e']
    image_center = [
        [86,196],
        [91,193],
        [99,188],
        [103,186],
        [108,181]]
    assert len(sets) == len(image_center)
    height = 128
    width = 380
    lbhw = 13 # half width of box around main image lobe
    bg_up = 9
    bg_down = 115
    bg_left = 335
    bg_right = 373

    all_STE_images = np.zeros((reps_per_set*len(sets),height,width))
    STE_signal = np.zeros((reps_per_set*len(sets)))
    bg_signal = np.zeros((reps_per_set*len(sets)))
    
    for my_index, my_set in enumerate(sets):
        filename = (
            './../../stimulated_emission_data/figure_8/STE_depletion/' +
            'dataset_' + my_set + '_' + str(reps_avgd) + '_rep_avg.tif')
        set_data = np_tif.tif_to_array(filename).astype(np.float64)
        assert set_data.shape == (reps_per_set*num_delays,height,width)
        set_data = set_data.reshape(reps_per_set,num_delays,height,width)
        begin = my_index * reps_per_set
        end = begin + reps_per_set

        # for fluorescence images just average all three images in delay scan
        STE_image_set = set_data.mean(axis=1)
        all_STE_images[begin:end] = STE_image_set

        # average points around main STE image lobe and add to STE_signal list
        ste_y, ste_x = image_center[my_index]
        STE_signal[begin:end] = STE_image_set[
            :,
            ste_y - lbhw:ste_y + lbhw,
            ste_x - lbhw:ste_x + lbhw
            ].mean(axis=2).mean(axis=1)

        # average lots of points away from STE image
        bg_signal[begin:end] = STE_image_set[
            :, bg_up:bg_down, bg_left:bg_right].mean(axis=2).mean(axis=1)

    # correct STE signal for stimulating beam power fluctuations
    STE_signal -= bg_signal

    # dust particle in 

    # crop images
    all_STE_images = all_STE_images[:, crop_rows:height - crop_rows, :]

    plt.figure()
    plt.plot(STE_signal)
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
