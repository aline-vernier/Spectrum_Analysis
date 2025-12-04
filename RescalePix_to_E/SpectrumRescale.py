import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
#from pathlib import Path
import time as t
import pyqtgraph as pg
from scipy.interpolate import interp1d
from pyqtgraph import QtCore


class SpectrumRescale:

    def __init__(self,
                 imPath='./calib/'
                        'SET6/SET6/magnet0.4T_Soectrum_isat4.9cm_26bar_gdd25850_HeAr_0002.TIFF',
                 calPath='./calib/dsdE_Small_LHC.txt',
                 pixel_per_mm=1,
                 plotCal=False,
                 deflection='L',
                 **kwds):

        self.imPath = imPath
        self.calPath = calPath
        self.pixel_per_mm = pixel_per_mm

        self.E_file = None
        self.E = None
        self.dsdE_file = None
        self.dsdE = None
        self.s_file = None
        self.s = None
        self.im = None
        self.zero_mm = 0
        self.zero_px = 0
        self.s_ref = None
        self.deflection = deflection

        self.y_scale = None

        self.load_calib()
        self.load_image()

        if plotCal:
            plt.plot(self.s, self.E)
            plt.show()
    '''Load calibration file. Currently the format is 
    cal[0]:Energy in Mev
    cal[1]:ds/dE in mm/MeV, currently unused
    cal[2]:s in mm'''
    def load_calib(self, calPath=None):
        if calPath is not None:
            self.calPath = calPath
        cal = np.loadtxt(self.calPath).T
        '''Data is flipped to allow usage of numpy interpolation'''
        self.E_file = np.flip(cal[0])
        self.dsdE_file = np.flip(cal[1])
        self.s_file = np.flip(cal[2])

    def load_image(self):
        self.im = np.array(Image.open(self.imPath))
        mean_im = np.average(self.im)
        self.im[self.im<mean_im]=0
        

    '''Placeholder for filtering out meaningless data'''
    def remove_meaningless_points(self, x_array, y_array):
        pass

    '''TWO DIFFERENT SPECTROMETER GEOMETRIES :
        - With zero (ref without magnet deflection)
        - Without zero (ref only from geometry)'''

    '''Rescaled1D_zero works with a zero in pixels or in mm.
    Low energies are on the LEFT, flipping options to be added? 
    Zero definition from cursors or automated to be added to interface.'''
    def rescale1D_zero(self, zero_px=None, zero_mm=None):
        if zero_px is not None:
            self.zero_px = zero_px
            self.zero_mm = zero_px/self.pixel_per_mm
        if zero_mm is not None:
            self.zero_mm = zero_mm
            self.zero_px = zero_mm*self.pixel_per_mm

        try:
            offset = self.im.shape[1]-self.zero_px
            self.s = (np.array(range(0, self.im.shape[1]))-offset)/self.pixel_per_mm
            self.E = np.interp(self.s, self.s_file, self.E_file, right=float('nan'), left=float('nan'))
            self.E = np.flip(self.E)

            self.dsdE = np.interp(self.s, self.s_file, self.dsdE_file, right=float('nan'), left=float('nan'))

        except Exception as e:
            print(f'Exception: {e}')

    def rescale2D_zero(self, zero_px=None, zero_mm=None):
        self.rescale1D_zero(zero_px, zero_mm)
        self.y_scale = np.array(range(0, self.im.shape[0]))/self.pixel_per_mm

    '''Rescaled1D_ref works without a zero but with a reference point 
    tuple (position in mm, energy in MeV).
    
        - refPoint defined from real lanex coordinate and energy E_ref
        - s_ref calculated from interpolation of E_ref in calibration file
        - Zero assumed on the right of the image (right edge of the lanex).
        - Low energies are on the LEFT, flipping options to be added soon. 
        
    Reference point definition to be added to interface.'''

    def rescale1D_ref(self, refPoint=(40,10)):# Default value, 40mm <=> 10MeV

        self.s_ref = np.interp(refPoint[1], np.flip(self.E_file), np.flip(self.s_file)
                               , right=float('nan'), left=float('nan')) # Flip not too clean, to be sanitized ;)
        self.s = np.array(range(0, self.im.shape[1]))/self.pixel_per_mm-(self.s_ref - refPoint[0])
        self.E = np.interp(self.s, self.s_file, self.E_file, right=float('nan'), left=float('nan'))

    def rescale2D_ref(self, refPoint=(40,10)):
        self.rescale1D_ref(refPoint)
        self.y_scale = np.array(range(0, self.im.shape[0])) / self.pixel_per_mm




if __name__ == "__main__":

    '''Experimental data: spatial calibration 49µm/mm, zero position {x=1953px, y=635px}'''
    r = SpectrumRescale(plotCal=False, pixel_per_mm=21.28)
    '''Both functions work, to be thoroughly tested ;)'''
    r.rescale2D_zero(zero_px=1953)
    #r.rescale2D_ref((40,10))

    '''Analysis'''
    data = r.im # shape: (n_y, n_x)
    energy = r.E
    angle = r.y_scale
    dsdE = r.dsdE

    '''To be integrated to class'''
    # Find valid x indices (non-NaN)
    valid_mask = ~np.isnan(energy)
    energy_valid = energy[valid_mask]
    dsdE_valid = dsdE[valid_mask]
    data_valid = data[:, valid_mask]  # Keep all rows, filter columns
    
    energy_regular = np.linspace(energy_valid.min(), energy_valid.max(), len(energy_valid))
    interp_func = interp1d(energy_valid, data_valid, axis=1, kind='linear')
    image_resampled = interp_func(energy_regular)


    app = pg.mkQApp()
    win = pg.GraphicsLayoutWidget()
    win.show()

    plot = win.addPlot()
    img = pg.ImageItem(image_resampled.T)  # transpose so axes align correctly
    plot.addItem(img)

    # map the image coordinates to physical coordinates
    img.setRect(
        energy_regular[0],           # x origin
        angle[0],                    # y origin
        energy_regular[-1] - energy_regular[0],  # width
        angle[-1] - angle[0]         # height
    )

    # add axis labels
    plot.setLabel('bottom', 'Energy', units='MeV')  # adjust units
    plot.setLabel('left', 'Angle', units='mrad')

    # optional: add a color bar
    from pyqtgraph import ColorBarItem
    bar = ColorBarItem(values=(image_resampled.min(), image_resampled.max()))
    img.setColorMap(pg.colormap.get('viridis'))
    bar.setImageItem(img)
    win.addItem(bar)

    # start the event loop if running standalone
    pg.exec()
