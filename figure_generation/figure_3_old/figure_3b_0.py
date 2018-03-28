import os
import numpy as np
import np_tif
from stack_registration import stack_registration
import matplotlib.pyplot as plt

def main():

    # the data to be plotted by this program is generated from raw tifs
    # and repetition_average_expt_and_control.py

    assert os.path.isdir('./../images')
    if not os.path.isdir('./../images/figure_3'):
        os.mkdir('./../images/figure_3')

    num_reps = 10 # number of times a power/delay stack was taken
    num_delays = 3
    image_h = 128
    image_w = 380

    # power calibration
    # red max power is 300 mW
    # green max power is 1450 mW
    # green powers calibrated using camera
    green_max_mW = 1330
    green_powers = np.array(
        (113.9,124.5,145.5,175.3,234.5,334.1,446.1))
    green_powers = green_powers - min(green_powers)
    green_powers = green_powers * green_max_mW / max(green_powers)
    green_powers = np.around(green_powers).astype(int)

    # red powers calibrated using camera
    red_bg = 26.6
    red_max_mW = 300
    red_powers = np.array(
        (26.6, 537))
    red_powers -= red_bg
    red_powers = red_powers * red_max_mW / max(red_powers)
    red_powers = np.around(red_powers).astype(int)

    filename = (
        './../../stimulated_emission_data/figure_3b/dataset_green_all_powers_up_0.tif')
    data = np_tif.tif_to_array(filename).astype(np.float64)

    # get rid of overexposed rows at top and bottom of images
    less_rows = 3
    data = data[:,0+less_rows:data.shape[1]-less_rows,:]

    # reshape to hyperstack
    data = data.reshape((
        len(green_powers),
        num_delays,
        data.shape[1],
        data.shape[2],
        ))

    # from the image where red/green are simultaneous, subtract the
    # average of images taken when the delay magnitude is greatest
    depletion_stack = (
        data[:,1,:,:] - # zero red/green delay
        0.5 * (data[:,0,:,:] + data[:,2,:,:]) # max red/green delay
        )

    # fluorescence image (no STE depletion) stack
    fluorescence_stack = 0.5 * (data[:,0,:,:] + data[:,2,:,:])
    # fluorescence image (with STE depletion) stack
    depleted_stack = data[:,1,:,:] # zero red/green delay

    # save processed stacks
##    tif_shape = (
##        len(red_powers)*len(green_powers),
##        depletion_stack.shape[2],
##        depletion_stack.shape[3],
##        )
##    np_tif.array_to_tif(
##        depletion_stack.reshape(tif_shape),'depletion_stack.tif')
##    np_tif.array_to_tif(
##        fluorescence_stack.reshape(tif_shape),'fluorescence_stack.tif')
##    np_tif.array_to_tif(
##        depleted_stack.reshape(tif_shape),'depleted_stack_ctrl.tif')

    # plot darkfield and stim emission signal
    # get background signal level for brightest image
    top_bg = 10
    bot_bg = 20
    left_bg = 10
    right_bg = 20
    fluorescence_signal_bg = (
        fluorescence_stack[:,top_bg:bot_bg,left_bg:right_bg
                           ].mean(axis=2).mean(axis=1)
        )
    depleted_signal_bg = (
        depleted_stack[:,top_bg:bot_bg,left_bg:right_bg
                       ].mean(axis=2).mean(axis=1)
        )
    # crop, bg subtract, and spatially filter image
    top = 0
    bot = 112
    left = 95 + 15 + 25
    right = 255 - 15 + 25
    fluorescence_cropped = (fluorescence_stack[-1,top:bot,left:right] -
                            fluorescence_signal_bg[-1])
    fluorescence_cropped = fluorescence_cropped.reshape(
        1,fluorescence_cropped.shape[0],fluorescence_cropped.shape[1])
    fluorescence_cropped = annular_filter(fluorescence_cropped,r1=0,r2=0.03)
    fluorescence_cropped = fluorescence_cropped[0,:,:]
    depletion_cropped = depletion_stack[-1,top:bot,left:right]
    depletion_cropped = depletion_cropped.reshape(
        1,depletion_cropped.shape[0],depletion_cropped.shape[1])
    depletion_cropped = annular_filter(depletion_cropped,r1=0,r2=0.03)
    depletion_cropped = depletion_cropped[0,:,:]
    fluorescence_cropped[101:107,5:34] = np.max(fluorescence_cropped) # scale bar
    depletion_cropped[101:107,5:34] = np.min(depletion_cropped) # scale bar

##    fig, (ax0, ax1) = plt.subplots(nrows=1,ncols=2,figsize=(16,5))
##
##    cax0 = ax0.imshow(fluorescence_cropped, cmap=plt.cm.gray)
##    ax0.axis('off')
##    cbar0 = fig.colorbar(cax0, ax = ax0)
##    ax0.set_title('Fluorescence image of nanodiamond')
##
##    cax1 = ax1.imshow(depletion_cropped, cmap=plt.cm.gray)
##    cbar1 = fig.colorbar(cax1, ax = ax1)
##    ax1.set_title('Fluorescence intensity decreased due to stim. emission')
##    ax1.axis('off')
##    plt.show()
##    plt.savefig('./../images/figure_3b/fluorescence_depletion_image.svg')
    
    # average points around center lobe of the nanodiamond image to get
    # "average signal level" for darkfield and STE images
    top = 34
    bot = 80
    left = 152 + 25
    right = 200 + 25
    depletion_signal = (
        depletion_stack[:,top:bot,left:right].mean(axis=2).mean(axis=1))
    depleted_signal = (
        depleted_stack[:,top:bot,left:right].mean(axis=2).mean(axis=1))
    depleted_signal = depleted_signal - depleted_signal_bg
    fluorescence_signal = (
        fluorescence_stack[:,top:bot,left:right].mean(axis=2).mean(axis=1))
    fluorescence_signal = fluorescence_signal - fluorescence_signal_bg
##    np_tif.array_to_tif(STE_signal,'STE_signal_array.tif')
##    np_tif.array_to_tif(crosstalk_signal,'crosstalk_signal_array.tif')
##    np_tif.array_to_tif(darkfield_signal,'darkfield_signal_array.tif')
    
    # plot signal
    mW_per_kex = 1100
    kex = green_powers / mW_per_kex

    mW_per_kdep = 1400
    kdep = red_powers[-1] / mW_per_kdep

    brightness = 65
    model_fl = kex / (1 + kex)
    model_fl_dep = kex / (1 + kex + kdep)

##    nd_brightness = depleted_signal[0,:]
    nd_brightness = fluorescence_signal
##    nd_brightness_depleted = depleted_signal[-1,:]
    nd_brightness_depleted = depleted_signal
    
    plt.figure()
    plt.plot(kex,nd_brightness/brightness,'o',
             label='Crimson bead fluorescence', color='green')
    plt.plot(kex,nd_brightness_depleted/brightness,'o',
             label='Depleted fluorescence', color='red')
    plt.plot(kex, model_fl, '-',
             label='Crimson bead fluorescence (model)', color='green')
    plt.plot(kex, model_fl_dep, '-',
             label='Depleted fluorescence (model)', color='red')
    plt.xlabel('k_ex = excitation power (mW) / %i\n'%(mW_per_kex))
    plt.ylabel('Excitation fraction')
    plt.axis([-0.01,1.4,-0.01,0.65])
    plt.legend(loc='lower right')
##    plt.title("k_ex  = mW/%i\n"%(mW_per_kex) + "k_dep = mW/%i"%(mW_per_kdep))
    plt.grid()
    a = plt.axes([0.17, 0.6, .25, .25])
    plt.imshow(fluorescence_cropped, cmap=plt.cm.gray)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('./../images/figure_3/fluorescence_depletion_dye.svg')
    plt.show()
    

##    plt.figure()
##    for (pow_num,rd_pow) in enumerate(red_powers):
##        plt.plot(
##            green_powers,depleted_signal[pow_num,:],
##            '.-',label=('Red power = '+str(rd_pow)+' mW'))
##    plt.title('Average fluorescence signal in main lobe')
##    plt.xlabel('Green power (mW)')
##    plt.ylabel('Fluorescence light signal (CMOS pixel counts)')
##    plt.legend(loc='lower right')
##    plt.ylim(0,80)
##    plt.grid()
####    plt.savefig('./../images/figure_4/fluorescence_v_green_power.svg')    
##    
##    plt.figure()
##    for (pow_num,gr_pow) in enumerate(green_powers):
##        plt.plot(
##            red_powers,depletion_signal[:,pow_num],
##            '.-',label=('Green power = '+str(gr_pow)+' mW'))
##    plt.title('Average fluorescence signal in main lobe')
##    plt.xlabel('Red power (mW)')
##    plt.ylabel('Change in fluorescent light signal (CMOS pixel counts)')
##    plt.legend(loc='lower left')
##    plt.grid()
##    plt.figure()
##    for (pow_num,rd_pow) in enumerate(red_powers):
##        plt.plot(
##            green_powers,depletion_signal[pow_num,:],
##            '.-',label=('Red power = '+str(rd_pow)+' mW'))
##    plt.title('Average fluorescence signal in main lobe')
##    plt.xlabel('Green power (mW)')
##    plt.ylabel('Change in fluorescent light signal (CMOS pixel counts)')
##    plt.legend(loc='upper right')
##    plt.grid()
##    plt.show()
    

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

if __name__ == '__main__':
    main()
