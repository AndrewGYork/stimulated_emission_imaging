import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as patches
import np_tif
from stack_registration import bucket

def main():
    assert os.path.isdir('./../images')
    if not os.path.isdir('./../images/figure_3'):
        os.mkdir('./../images/figure_3')

    num_reps = 10 # number of times a power/delay stack was taken
    num_delays = 5

    # power calibration:
    # for a given voltage input to the AOM, how many milliwatts do we
    # expect the AOM to output
    green_max_mW = 1450 # measured with power meter before sample
    # When the experiment is run, the acquisition code sends voltages to
    # the AOM via the analog out card. The maximum voltage is the same
    # as was used to deliver the max power to the power meter (see
    # previous line). We replaced the green filter with neutral density
    # filters and measured green power on the camera while running the
    # same acquisition code used to take the fluorescence data.
    # calibration units: camera counts
    green_powers = np.array(
        (113.9, 119.6, 124.5, 135, 145.5, 159.5, 175.3, 193.1, 234.5,
         272.2, 334.1, 385.7, 446.1))
    # 100-102 is roughly the baseline level, but ~0.4 s exposure caused
    # ~13 pixel counts of extra light, so we subtract 113.9
    green_powers = green_powers - min(green_powers)
    green_powers = green_powers * green_max_mW / max(green_powers) # units: mW

    # red powers calibrated using power meter and AOM transmitting 1.25us pulses
    red_max_mW = 300 # measured with power meter before sample
    red_bg = 26.6
    red_powers = np.array(
        (26.6, 113, 198, 276, 353, 438, 537))
    red_powers -= red_bg
    red_powers = red_powers * red_max_mW / max(red_powers)

    # load fluorescence signal data (no spatial resolution)
    data = np_tif.tif_to_array(
        './../../stimulated_emission_imaging-data' +
        '/2016_11_14_modulated_imaging_depletion_nanodiamond_7' +
        '/data_point_signal.tif').astype(np.float64)
    bg = np_tif.tif_to_array(
        './../../stimulated_emission_imaging-data' +
        '/2016_11_14_modulated_imaging_depletion_nanodiamond_7' +
        '/data_point_bg.tif').astype(np.float64)

    # subtract signal from corner of image where there is no fluorophore
    data = data - bg
    
    # reshape to hyperstack
    data = data.reshape((
        num_reps,
        len(red_powers),
        len(green_powers),
        num_delays,
        ))

    # zero red/green delay is 3rd out of 5 delays (hence index 2)
    depleted_stack = data[:, :, :, 2]
    # regular fluorescence is with zero power in the depletion beam,
    # (hence index 0)
    fluorescence_signal_mean = depleted_stack[:,0,:].mean(axis=0)
    fluorescence_signal_max = depleted_stack[:,0,:].max(axis=0)
    fluorescence_signal_min = depleted_stack[:,0,:].min(axis=0)
    fluorescence_signal_std = depleted_stack[:,0,:].std(axis=0)
    # maximally depleted fluorescence is with max power in the depletion
    # beam (hence index -1)
    depleted_signal_mean = depleted_stack[:,-1,:].mean(axis=0)
    depleted_signal_max = depleted_stack[:,-1,:].max(axis=0)
    depleted_signal_min = depleted_stack[:,-1,:].min(axis=0)
    depleted_signal_std = depleted_stack[:,-1,:].std(axis=0)

    # make inset image of representative data
    # image is already cropped, averaged (10 reps) and bg subtracted
    inset_image = np_tif.tif_to_array(
        './../../stimulated_emission_imaging-data' +
        '/2016_11_14_modulated_imaging_depletion_nanodiamond_7' +
        '/representative_image_avg.tif').astype(np.float64)
    inset_image = inset_image[0,:,:] # there's only one image
    inset_image = bucket(inset_image, bucket_size=(8, 8))
    inset_image[-2:-1, 1:6] = np.max(inset_image) # scale bar
    

    # IMPORTANT PARAMETERS FOR FIT
    brightness = 160
    mW_per_kex = 1150
    mW_per_kdep = 1150
    
    # compute rate constants
    kex = green_powers / mW_per_kex
    kex_min = kex / 1.1
    kex_max = kex * 1.1

    kdep = red_powers[-1] / mW_per_kdep
    kdep_min = kdep / 1.4
    kdep_max = kdep * 1.4

    model_fl = kex / (1 + kex)
    model_fl_max = kex_max / (1 + kex_max)
    model_fl_min = kex_min / (1 + kex_min)
    model_fl_dep = kex / (1 + kex + kdep)
    model_fl_dep_max = kex / (1 + kex + kdep_max)
    model_fl_dep_min = kex / (1 + kex + kdep_min)


    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    # plot measured data mean and std
    ax1.errorbar(green_powers,
                 fluorescence_signal_mean,
                 yerr=fluorescence_signal_std,
                 fmt='o', linewidth=3, capthick=2,
                 label='Measured, 0 mW stimulation',
                 color='green')
    ax1.errorbar(green_powers,
                 depleted_signal_mean,
                 yerr=depleted_signal_std,
                 fmt='o', linewidth=3, capthick=2,
                 label='Measured, 300 mW stimulation',
                 color='red')
    ax1.set_xlabel('Excitation power (mW)',fontsize=16)
    # plot model fit
    ax1.plot(green_powers,
             model_fl * brightness,
             '-',
             label=r'Model, $h_{stim}\sigma_{23}=0$',
             color='green')
    ax1.fill_between(green_powers,
                     model_fl_max * brightness,
                     model_fl_min * brightness,
                     color="#C0FFC0")
    dep_mult = ("{:.2f}".format(kdep))
    ax1.plot(green_powers, model_fl_dep * brightness, '-',
             label=(
                 r'Model, $h_{stim}\sigma_{23}=' +
                 dep_mult + r'(1/\tau_{fluor})$'),
             color='red')
    ax1.fill_between(green_powers,
                     model_fl_dep_max * brightness,
                     model_fl_dep_min * brightness,
                     color='#FFD0D0')
    plt.ylabel('Average pixel brightness (sCMOS counts)', fontsize=15)
    plt.axis([0, 1600, 0, 110])
    leg = plt.legend(loc='lower right', title='Fluorescence', fontsize=14)
    plt.setp(leg.get_title(), fontsize=15)
    plt.grid()
    # plot other axes
    ax2 = ax1.twiny()
    formatter = FuncFormatter(
        lambda green_powers, pos: '{:0.2f}'.format(green_powers/mW_per_kex))
    ax2.xaxis.set_major_formatter(formatter)
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xlabel(r'$h_{exc}\sigma_{01}/(1/\tau_{fluor})$', fontsize=17)
    ax2 = ax1.twinx()
    formatter = FuncFormatter(
        lambda model_fl, pos: '{:0.2f}'.format(model_fl/brightness))
    ax2.yaxis.set_major_formatter(formatter)
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_ylabel(r'Excitation fraction $n_2$', fontsize=17)
    # inset image
    a = plt.axes([0.17, 0.6, .25, .25])
    plt.imshow(inset_image, cmap=plt.cm.gray, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    a.text(
        0.4, 1.5, 'Nanodiamond', fontsize=14, color='white', fontweight='bold')
    rect = patches.Rectangle(
        (4.6, 3.9), 6, 6,
        linewidth=1, linestyle='dashed', edgecolor='y', facecolor='none')
    a.add_patch(rect)
    plt.savefig(
        './../images/figure_3/fluorescence_depletion_nd_brightness_optimal.svg')
    plt.show()

    # change brightness * 1.5 and scale rate constant to fit
    brightness_hi = brightness * 1.5
    rate_const_mod = 1.83
    extra_kdep_mod = 0.6
    
    # compute rate constants
    mW_per_kex_hi = mW_per_kex * rate_const_mod
    kex = green_powers / mW_per_kex_hi
    kex_min = kex * 1.1
    kex_max = kex / 1.1

    mW_per_kdep_hi = mW_per_kdep * rate_const_mod * extra_kdep_mod
    kdep = red_powers[-1] / mW_per_kdep_hi
    kdep_min = kdep / 1.4
    kdep_max = kdep * 1.4


    model_fl = kex / (1 + kex)
    model_fl_max = kex_max / (1 + kex_max)
    model_fl_min = kex_min / (1 + kex_min)
    model_fl_dep = kex / (1 + kex + kdep)
    model_fl_dep_max = kex / (1 + kex + kdep_max)
    model_fl_dep_min = kex / (1 + kex + kdep_min)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    # plot measured data mean and std
    ax1.errorbar(green_powers,
                 fluorescence_signal_mean,
                 yerr=fluorescence_signal_std,
                 fmt='o', linewidth=3, capthick=2,
                 label='Measured, 0 mW stimulation',
                 color='green')
    ax1.errorbar(green_powers,
                 depleted_signal_mean,
                 yerr=depleted_signal_std,
                 fmt='o', linewidth=3, capthick=2,
                 label='Measured, 300 mW stimulation',
                 color='red')
    ax1.set_xlabel('Excitation power (mW)', fontsize=16)
    # plot model fit
    ax1.plot(green_powers,
             model_fl * brightness_hi,
             '-',
             label=r'Model, $h_{stim}\sigma_{23}=0$',
             color='green')
    ax1.fill_between(green_powers,
                     model_fl_max * brightness_hi,
                     model_fl_min * brightness_hi,
                     color="#C0FFC0")
    dep_mult = ("{:.2f}".format(kdep))
    ax1.plot(green_powers, model_fl_dep * brightness_hi, '-',
             label=(r'Model, $h_{stim}\sigma_{23}=' +
                    dep_mult + r'(1/\tau_{fluor})$'),
             color='red')
    ax1.fill_between(green_powers,
                     model_fl_dep_max * brightness_hi,
                     model_fl_dep_min * brightness_hi,
                     color='#FFD0D0')
    plt.ylabel('Average pixel brightness (sCMOS counts)', fontsize=15)
    plt.axis([0, 1600, 0, 110])
    leg = plt.legend(loc='lower right', title='Fluorescence', fontsize=14)
    plt.setp(leg.get_title(), fontsize=15)
    plt.grid()
    # plot other axes
    ax2 = ax1.twiny()
    formatter = FuncFormatter(
        lambda green_powers, pos: '{:0.2f}'.format(green_powers/mW_per_kex))
    ax2.xaxis.set_major_formatter(formatter)
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xlabel(r'$h_{exc}\sigma_{01}/(1/\tau_{fluor})$', fontsize=17)
    ax2 = ax1.twinx()
    formatter = FuncFormatter(
        lambda model_fl, pos: '{:0.2f}'.format(model_fl/brightness_hi))
    ax2.yaxis.set_major_formatter(formatter)
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_ylabel(r'Excitation fraction $n_2$', fontsize=17)
    # inset image
    a = plt.axes([0.17, 0.6, .25, .25])
    plt.imshow(inset_image, cmap=plt.cm.gray, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    a.text(
        0.4, 1.5, 'Nanodiamond', fontsize=14, color='white', fontweight='bold')
    rect = patches.Rectangle(
        (4.6, 3.9), 6, 6,
        linewidth=1, linestyle='dashed', edgecolor='y', facecolor='none')
    a.add_patch(rect)
    plt.savefig(
        './../images/figure_3/fluorescence_depletion_nd_brightness_hi.svg')
    plt.show()
    

    # change brightness / 1.5 and scale rate constant to fit
    brightness_lo = brightness / 1.5
    rate_const_mod = 2.2
    extra_kdep_mod = 0.8
    
    # compute rate constants
    mW_per_kex_lo = mW_per_kex / rate_const_mod
    kex = green_powers / mW_per_kex_lo
    kex_min = kex * 1.1
    kex_max = kex / 1.1

    mW_per_kdep_lo = mW_per_kdep / rate_const_mod / extra_kdep_mod
    kdep = red_powers[-1] / mW_per_kdep_lo
    kdep_min = kdep / 1.4
    kdep_max = kdep * 1.4

    model_fl = kex / (1 + kex)
    model_fl_max = kex_max / (1 + kex_max)
    model_fl_min = kex_min / (1 + kex_min)
    model_fl_dep = kex / (1 + kex + kdep)
    model_fl_dep_max = kex / (1 + kex + kdep_max)
    model_fl_dep_min = kex / (1 + kex + kdep_min)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    # plot measured data mean and std
    ax1.errorbar(green_powers,
                 fluorescence_signal_mean,
                 yerr=fluorescence_signal_std,
                 fmt='o', linewidth=3, capthick=2,
                 label='Measured, 0 mW stimulation',
                 color='green')
    ax1.errorbar(green_powers,
                 depleted_signal_mean,
                 yerr=depleted_signal_std,
                 fmt='o', linewidth=3, capthick=2,
                 label='Measured, 300 mW stimulation',
                 color='red')
    ax1.set_xlabel('Excitation power (mW)', fontsize=16)
    # plot model fit
    ax1.plot(green_powers,
             model_fl * brightness_lo,
             '-',
             label=r'Model, $h_{stim}\sigma_{23}=0$',
             color='green')
    ax1.fill_between(green_powers,
                     model_fl_max * brightness_lo,
                     model_fl_min * brightness_lo,
                     color="#C0FFC0")
    dep_mult = ("{:.2f}".format(kdep))
    ax1.plot(green_powers, model_fl_dep * brightness_lo, '-',
             label=(r'Model, $h_{stim}\sigma_{23}=' +
                    dep_mult + r'(1/\tau_{fluor})$'),
             color='red')
    ax1.fill_between(green_powers,
                     model_fl_dep_max * brightness_lo,
                     model_fl_dep_min * brightness_lo,
                     color='#FFD0D0')
    plt.ylabel('Average pixel brightness (sCMOS counts)', fontsize=15)
    plt.axis([0, 1600, 0, 110])
    leg = plt.legend(loc='lower right', title='Fluorescence', fontsize=14)
    plt.setp(leg.get_title(), fontsize=15)
    plt.grid()
    # plot other axes
    ax2 = ax1.twiny()
    formatter = FuncFormatter(
        lambda green_powers, pos: '{:0.2f}'.format(green_powers/mW_per_kex))
    ax2.xaxis.set_major_formatter(formatter)
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xlabel(r'$h_{exc}\sigma_{01}/(1/\tau_{fluor})$', fontsize=17)
    ax2 = ax1.twinx()
    formatter = FuncFormatter(
        lambda model_fl, pos: '{:0.2f}'.format(model_fl/brightness_lo))
    ax2.yaxis.set_major_formatter(formatter)
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_ylabel(r'Excitation fraction $n_2$', fontsize=17)
    # inset image
    a = plt.axes([0.17, 0.6, .25, .25])
    plt.imshow(inset_image, cmap=plt.cm.gray, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    a.text(
        0.4, 1.5, 'Nanodiamond', fontsize=14, color='white', fontweight='bold')
    rect = patches.Rectangle(
        (4.6, 3.9), 6, 6,
        linewidth=1, linestyle='dashed', edgecolor='y', facecolor='none')
    a.add_patch(rect)
    plt.savefig(
        './../images/figure_3/fluorescence_depletion_nd_brightness_lo.svg')
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
