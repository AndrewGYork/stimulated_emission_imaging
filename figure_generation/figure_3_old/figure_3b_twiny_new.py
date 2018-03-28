import os
import numpy as np
import np_tif
from stack_registration import stack_registration
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as patches

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
    green_max_mW = 950
    green_powers = np.array(
        (169,1949,4795,8406,15215,25940,37337))
    green_powers = green_powers - min(green_powers)
    green_powers = green_powers * green_max_mW / max(green_powers)
    green_powers = np.around(green_powers).astype(int)

    # red powers calibrated using camera
    red_bg = 26.6
    red_max_mW = 240
    red_powers = np.array(
        (26.6, 537))
    red_powers -= red_bg
    red_powers = red_powers * red_max_mW / max(red_powers)
    red_powers = np.around(red_powers).astype(int)

    filename = (
        './../../stimulated_emission_data/figure_3b/dataset_green_all_powers_up.tif')
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
    top = 2
    bot = top + 112
    left = 141
    right = left + 130
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

    # IMPORTANT PARAMETER FOR FIT BELOW /1.2 - 73, reg 81 *1.2 - 89
    reg_brightness = 97
    reg_mW_per_kex = 960
    reg_mW_per_kdep = 1480
    brightness = reg_brightness
    rate_const_mod = 1.5
    brightness_mod_hi = 1.30
    brightness_mod_lo = 1.27
    dep_mult = "{:.2f}".format(red_powers[-1] / reg_mW_per_kdep)
    dep_mult_hi = "{:.2f}".format(red_powers[-1] / reg_mW_per_kdep / rate_const_mod)
    dep_mult_lo = "{:.2f}".format(red_powers[-1] / reg_mW_per_kdep * rate_const_mod)
    
    # average points around center lobe of the nanodiamond image to get
    # "average signal level" for darkfield and STE images
    top = 37
    bot = top + 46
    left = 184
    right = left + 46
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
    mW_per_kex = reg_mW_per_kex
    kex = green_powers / mW_per_kex
    kex_min = kex * 1.1
    kex_max = kex / 1.1

    mW_per_kdep = reg_mW_per_kdep
    kdep = red_powers[-1] / mW_per_kdep
    kdep_min = kdep / 1.4
    kdep_max = kdep * 1.4


    model_fl = kex / (1 + kex)
    model_fl_max = kex_max / (1 + kex_max)
    model_fl_min = kex_min / (1 + kex_min)
    model_fl_dep = kex / (1 + kex + kdep)
    model_fl_dep_max = kex / (1 + kex + kdep_max)
    model_fl_dep_min = kex / (1 + kex + kdep_min)
##    model_fl_dep_max = kex_max / (1 + kex_max + kdep)
##    model_fl_dep_min = kex_min / (1 + kex_min + kdep)

##    nd_brightness = depleted_signal[0,:]
    nd_brightness = fluorescence_signal
##    nd_brightness_depleted = depleted_signal[-1,:]
    nd_brightness_depleted = depleted_signal
    
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(green_powers, model_fl, '-',
             label=r'Model, $h_{stim}\sigma_{23}=0$', color='green')
    ax1.fill_between(green_powers, model_fl_max, model_fl_min,color="#C0FFC0")
    ax1.plot(green_powers, model_fl_dep, '-',
             label=r'Model, $h_{stim}\sigma_{23}=' + dep_mult + r'(1/\tau_{fluor})$', color='red')
    ax1.fill_between(green_powers, model_fl_dep_max, model_fl_dep_min,color='#FFD0D0')
    plt.ylabel(r'Excitation fraction $n_2$',fontsize=17)
    ax1.plot(green_powers,fluorescence_signal/brightness,'o',
             label='Measured, 0 mW stimulation', color='green')
    ax1.plot(green_powers,depleted_signal/brightness,'o',
             label='Measured, 240 mW stimulation', color='red')
    ax1.set_xlabel('Excitation power (mW)',fontsize=16)
    plt.axis([0,1100,0,0.65])
    leg = plt.legend(loc='lower right',title='Fluorescence',fontsize=14)
    plt.setp(leg.get_title(),fontsize=15)
    plt.grid()
    ax2 = ax1.twiny()
    formatter = FuncFormatter(
        lambda green_powers, pos: '{:0.2f}'.format(green_powers/mW_per_kex))
    ax2.xaxis.set_major_formatter(formatter)
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xlabel(r'$h_{exc}\sigma_{01}/(1/\tau_{fluor})$',fontsize=17)
    ax2 = ax1.twinx()
    ax2.set_yticks(np.round(np.linspace(0,0.65,num=8)*brightness))
    ax2.set_ylabel('Average pixel brightness (sCMOS counts)',fontsize=15)
    
    a = plt.axes([0.17, 0.6, .25, .25])
    plt.imshow(fluorescence_cropped, cmap=plt.cm.gray)
    plt.xticks([])
    plt.yticks([])
    a.text(8,15,'Crimson bead',fontsize=14,color='white',fontweight='bold')
    rect = patches.Rectangle(
        (43,33),48,46,linewidth=1,linestyle='dashed',edgecolor='y',facecolor='none')
    a.add_patch(rect)
    plt.savefig('./../images/figure_3/fluorescence_depletion_crimson_dye_rate_optimal.svg')
    plt.show()

    # with excitation and rate constants 20% higher and scaled to fit
    # IMPORTANT PARAMETER FOR FIT BELOW /1.2 - 73, reg 81 *1.2 - 92
    brightness = reg_brightness * brightness_mod_hi
    
    # average points around center lobe of the nanodiamond image to get
    # "average signal level" for darkfield and STE images
    top = 37
    bot = top + 46
    left = 184
    right = left + 46
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
    mW_per_kex = reg_mW_per_kex * rate_const_mod
    kex = green_powers / mW_per_kex
    kex_min = kex * 1.1
    kex_max = kex / 1.1

    mW_per_kdep = reg_mW_per_kdep * rate_const_mod
    kdep = red_powers[-1] / mW_per_kdep
    kdep_min = kdep / 1.4
    kdep_max = kdep * 1.4


    model_fl = kex / (1 + kex)
    model_fl_max = kex_max / (1 + kex_max)
    model_fl_min = kex_min / (1 + kex_min)
    model_fl_dep = kex / (1 + kex + kdep)
    model_fl_dep_max = kex / (1 + kex + kdep_max)
    model_fl_dep_min = kex / (1 + kex + kdep_min)
##    model_fl_dep_max = kex_max / (1 + kex_max + kdep)
##    model_fl_dep_min = kex_min / (1 + kex_min + kdep)

##    nd_brightness = depleted_signal[0,:]
    nd_brightness = fluorescence_signal
##    nd_brightness_depleted = depleted_signal[-1,:]
    nd_brightness_depleted = depleted_signal
    
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(green_powers, model_fl, '-',
             label=r'Model, $h_{stim}\sigma_{23}=0$', color='green')
    ax1.fill_between(green_powers, model_fl_max, model_fl_min,color="#C0FFC0")
    ax1.plot(green_powers, model_fl_dep, '-',
             label=r'Model, $h_{stim}\sigma_{23}=' + dep_mult_hi + r'(1/\tau_{fluor})$', color='red')
    ax1.fill_between(green_powers, model_fl_dep_max, model_fl_dep_min,color='#FFD0D0')
    plt.ylabel(r'Excitation fraction $n_2$',fontsize=17)
    ax1.plot(green_powers,fluorescence_signal/brightness,'o',
             label='Measured, 0 mW stimulation', color='green')
    ax1.plot(green_powers,depleted_signal/brightness,'o',
             label='Measured, 240 mW stimulation', color='red')
    ax1.set_xlabel('Excitation power (mW)',fontsize=16)
    plt.axis([0,1100,0,0.65])
    leg = plt.legend(loc='lower right',title='Fluorescence',fontsize=14)
    plt.setp(leg.get_title(),fontsize=15)
    plt.grid()
    ax2 = ax1.twiny()
    formatter = FuncFormatter(
        lambda green_powers, pos: '{:0.2f}'.format(green_powers/mW_per_kex))
    ax2.xaxis.set_major_formatter(formatter)
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xlabel(r'$h_{exc}\sigma_{01}/(1/\tau_{fluor})$',fontsize=17)
    ax2 = ax1.twinx()
    ax2.set_yticks(np.round(np.linspace(0,0.65,num=8)*brightness))
    ax2.set_ylabel('Average pixel brightness (sCMOS counts)',fontsize=15)
    
    a = plt.axes([0.17, 0.6, .25, .25])
    plt.imshow(fluorescence_cropped, cmap=plt.cm.gray)
    plt.xticks([])
    plt.yticks([])
    a.text(8,15,'Crimson bead',fontsize=14,color='white',fontweight='bold')
    rect = patches.Rectangle(
        (43,33),48,46,linewidth=1,linestyle='dashed',edgecolor='y',facecolor='none')
    a.add_patch(rect)
    plt.savefig('./../images/figure_3/fluorescence_depletion_crimson_dye_rate_lo.svg')
    plt.show()

    # with excitation and rate constants 20% higher and scaled to fit
    # IMPORTANT PARAMETER FOR FIT BELOW /1.3 - 69, reg 81 *1.3 - 92
    brightness = reg_brightness / brightness_mod_lo
    
    # average points around center lobe of the nanodiamond image to get
    # "average signal level" for darkfield and STE images
    top = 37
    bot = top + 46
    left = 184
    right = left + 46
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
    mW_per_kex = reg_mW_per_kex / rate_const_mod
    kex = green_powers / mW_per_kex
    kex_min = kex * 1.1
    kex_max = kex / 1.1

    mW_per_kdep = reg_mW_per_kdep / rate_const_mod
    kdep = red_powers[-1] / mW_per_kdep
    kdep_min = kdep / 1.4
    kdep_max = kdep * 1.4


    model_fl = kex / (1 + kex)
    model_fl_max = kex_max / (1 + kex_max)
    model_fl_min = kex_min / (1 + kex_min)
    model_fl_dep = kex / (1 + kex + kdep)
    model_fl_dep_max = kex / (1 + kex + kdep_max)
    model_fl_dep_min = kex / (1 + kex + kdep_min)
##    model_fl_dep_max = kex_max / (1 + kex_max + kdep)
##    model_fl_dep_min = kex_min / (1 + kex_min + kdep)

##    nd_brightness = depleted_signal[0,:]
    nd_brightness = fluorescence_signal
##    nd_brightness_depleted = depleted_signal[-1,:]
    nd_brightness_depleted = depleted_signal
    
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(green_powers, model_fl, '-',
             label=r'Model, $h_{stim}\sigma_{23}=0$', color='green')
    ax1.fill_between(green_powers, model_fl_max, model_fl_min,color="#C0FFC0")
    ax1.plot(green_powers, model_fl_dep, '-',
             label=r'Model, $h_{stim}\sigma_{23}=' + dep_mult_lo + r'(1/\tau_{fluor})$', color='red')
    ax1.fill_between(green_powers, model_fl_dep_max, model_fl_dep_min,color='#FFD0D0')
    plt.ylabel(r'Excitation fraction $n_2$',fontsize=17)
    ax1.plot(green_powers,fluorescence_signal/brightness,'o',
             label='Measured, 0 mW stimulation', color='green')
    ax1.plot(green_powers,depleted_signal/brightness,'o',
             label='Measured, 240 mW stimulation', color='red')
    ax1.set_xlabel('Excitation power (mW)',fontsize=16)
    plt.axis([0,1100,0,0.65])
    leg = plt.legend(loc='lower right',title='Fluorescence',fontsize=14)
    plt.setp(leg.get_title(),fontsize=15)
    plt.grid()
    ax2 = ax1.twiny()
    formatter = FuncFormatter(
        lambda green_powers, pos: '{:0.2f}'.format(green_powers/mW_per_kex))
    ax2.xaxis.set_major_formatter(formatter)
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xlabel(r'$h_{exc}\sigma_{01}/(1/\tau_{fluor})$',fontsize=17)
    ax2 = ax1.twinx()
    ax2.set_yticks(np.round(np.linspace(0,0.65,num=8)*brightness))
    ax2.set_ylabel('Average pixel brightness (sCMOS counts)',fontsize=15)
    
    a = plt.axes([0.17, 0.6, .25, .25])
    plt.imshow(fluorescence_cropped, cmap=plt.cm.gray)
    plt.xticks([])
    plt.yticks([])
    a.text(8,15,'Crimson bead',fontsize=14,color='white',fontweight='bold')
    rect = patches.Rectangle(
        (43,33),48,46,linewidth=1,linestyle='dashed',edgecolor='y',facecolor='none')
    a.add_patch(rect)
    plt.savefig('./../images/figure_3/fluorescence_depletion_crimson_dye_rate_hi.svg')
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

if __name__ == '__main__':
    main()
