import os
import numpy as np
import np_tif
import matplotlib.pyplot as plt

def main():

    assert os.path.isdir('./../images')
    if not os.path.isdir('./../images/figure_A8'):
        os.mkdir('./../images/figure_A8')
    folder_string = './../../stimulated_emission_data/figure_A8/'

    pulsewidths = np.array([8,4,2,1])
    
    # location of center lobe
    bright_spot_x = np.array([219, 235, 204, 207])-1
    bright_spot_y = np.array([47, 71, 37, 66])-1

    # cropping area defined
    top_left_x = [133, 146, 114, 118]
    top_left_y = [0, 7, 0, 5]
    crop_width = 175
    crop_height = 118

    # where on the plot should the cropped images be
    plot_pos_y = [0.12, 0.26, 0.43, 0.63]
    plot_pos_x = [0.26, 0.4, 0.58, 0.77]

    
    
    STE_signal = np.zeros(4)
    STE_signal_relative = np.zeros(4)
    STE_image_cropped = np.zeros((4,crop_height,crop_width))
    qtr_width = 12 
    for i in range(4):
        pulsewidth = pulsewidths[i]
        filename = (folder_string +
                    'dataset_green_1395mW_' + str(pulsewidth) + 'us.tif')
        assert os.path.exists(filename)
        data = np_tif.tif_to_array(filename).astype(np.float64)
        filename = (folder_string +
                    'dataset_green_0mW_' + str(pulsewidth) + 'us.tif')
        assert os.path.exists(filename)
        data_bg = np_tif.tif_to_array(filename).astype(np.float64)
        data = data - data_bg
        # get rid of overexposed rows at top and bottom of images
        less_rows = 3
        data = data[:,0+less_rows:data.shape[2]-less_rows,:]
        # subtract avg of green off images from green on images
        data_simult = data[1,:,:]
        data_non_simult = 0.5 * (data[0,:,:] + data[2,:,:])
        STE_image = data_simult - data_non_simult
        # filter high frequency noise
        STE_image = STE_image.reshape(
            1,STE_image.shape[0],STE_image.shape[1])
        STE_image = annular_filter(STE_image,r1=0,r2=0.02)
        STE_image = STE_image[0,:,:]
        # crop stim emission image
        STE_image_cropped[i,:,:] = STE_image[
            top_left_y[i]:top_left_y[i] + crop_height,
            top_left_x[i]:top_left_x[i] + crop_width,
            ]
##        print(np.amax(STE_image_cropped[i,:,:]))
##        print(np.amin(STE_image_cropped[i,:,:]))
        STE_image_cropped[i,0,0] = -63
        STE_image_cropped[i,0,-1] = 178
        STE_image_cropped[i,108:114,5:34] = 178
        # capture stim emission signal
        my_col = bright_spot_x[i]
        my_row = bright_spot_y[i]
        main_lobe = STE_image[
            my_row-qtr_width:my_row+qtr_width,
            my_col-qtr_width:my_col+qtr_width]
        left_edge = STE_image[:,qtr_width*2]
        STE_signal[i] = np.mean(main_lobe)
        STE_signal_relative[i] = STE_signal[i] - np.mean(left_edge)

    my_zero = np.zeros(1)
    STE_signal_relative = np.concatenate((my_zero,STE_signal_relative))
    my_intensity = np.concatenate((my_zero,1/pulsewidths))

    fig = plt.figure()
    lines = plt.plot(my_intensity,STE_signal_relative,'o--',color='red')
    plt.setp(lines, linewidth=2, color='r')
    plt.plot(my_intensity,STE_signal_relative,'o',color='black')
    plt.xlim(xmin=-0.04,xmax=1.04)
    plt.ylim(ymin=-5,ymax=176)
    plt.ylabel('Average signal brightness (pixel counts)')
    plt.xlabel('Normalized laser intensity (constant energy)')
    plt.grid()
    for i in range(4):
        a = plt.axes([plot_pos_x[i], plot_pos_y[i], .12, .12])
        plt.imshow(STE_image_cropped[i,:,:], cmap=plt.cm.gray)
        plt.xticks([])
        plt.yticks([])
##    plt.show()
    plt.savefig('./../images/figure_A8/phase_contrast_dye_pulse_length_scan.svg')
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
