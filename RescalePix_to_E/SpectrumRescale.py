import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
#from pathlib import Path
import time
import pyqtgraph as pg
from scipy.interpolate import interp1d
from pyqtgraph import QtCore
from pyqtgraph import ColorBarItem

VIRIDIS = pg.colormap.get('viridis')

class SpectrumRescale:

    def __init__(self,
                 imPath='./calib/SET6/SET6/'
                        'magnet0.4T_Soectrum_isat4.9cm_26bar_gdd25850_HeAr_0002.TIFF',
                 calPath='./calib/dsdE_Small_LHC.txt',
                 pixel_per_mm=1.,
                 plotCal=False,
                 deflection='L',
                 **kwds):

        self.imPath = imPath
        self.calPath = calPath
        self.pixel_per_mm = pixel_per_mm

        # self.E_file = None
        # self.E = None
        # self.dsdE_file = None
        # self.dsdE = None
        # self.s_file = None
        # self.s = None
        # self.im = None
        self.zero_mm = 0
        self.zero_px = 0
        self.s_ref = None
        self.deflection = deflection

        self.y_scale = None

        self.load_calib()
        self.load_image()

        # if plotCal:
        #     plt.plot(self.s, self.E)
        #     plt.show()
            # Load calibration file. Currently the format is 
            # cal[0]:Energy in Mev
            # cal[1]:ds/dE in mm/MeV, currently unused
            # cal[2]:s in mm
    
    def load_calib(self, calPath=None):
        if calPath is not None:
            self.calPath = calPath
        cal = np.loadtxt(self.calPath).T
        # Data is flipped to allow usage of numpy interpolation
        self.E_file = np.flip(cal[0])
        self.dsdE_file = np.flip(cal[1])
        self.s_file = np.flip(cal[2])

    def load_image(self):
        self.im = np.array(Image.open(self.imPath))
        mean_im = np.average(self.im)
        self.im[self.im<mean_im]=0
        

        # TWO DIFFERENT SPECTROMETER GEOMETRIES :
        #    - With zero (ref without magnet deflection)
        #    - Without zero (ref only from geometry)


    def rescale1D_zero(self, zero_px=None, zero_mm=None):
        '''
        Rescaled1D_zero works with a zero in pixels or in mm.
        Low energies are on the LEFT, flipping options to be added? 
        Zero definition from cursors or automated to be added to interface.
        '''
        if zero_px is not None:
            self.zero_px = zero_px
            self.zero_mm = zero_px / self.pixel_per_mm
        if zero_mm is not None:
            self.zero_mm = zero_mm
            self.zero_px = zero_mm * self.pixel_per_mm

        try:
            offset = self.im.shape[1] - self.zero_px
            self.s = (np.array(range(0, self.im.shape[1])) - offset) / self.pixel_per_mm # x-axis in space unit
            print(f'self.s min = {min(self.s)}; self.s max = {max(self.s)}')
            self.E = np.interp(self.s, self.s_file, self.E_file, right=np.nan, left=np.nan)
            self.E = np.flip(self.E)

            self.dsdE = np.interp(self.s, self.s_file, self.dsdE_file, right=np.nan, left=np.nan)

        except Exception as e:
            print(f'Exception: {e}')

    def rescale2D_zero(self, zero_px=None, zero_mm=None):
        self.rescale1D_zero(zero_px, zero_mm)
        self.y_scale = np.array(range(0, self.im.shape[0])) / self.pixel_per_mm


    def rescale1D_ref(self, refPoint=(40,10)):
        '''
        Rescaled1D_ref works without a zero but with a reference point 
        tuple (position in mm, energy in MeV).
        
             - refPoint defined from real lanex coordinate and energy E_ref
             - s_ref calculated from interpolation of E_ref in calibration file
             - Lanex zero coordinate assumed on the right of the image (right edge of the lanex).
             - Low energies are on the LEFT, flipping options to be added soon. 
            
        Reference point definition to be added to interface.

        Default value:
                refPoint: tuple
                    (40, 10) corresponds to 40mm and 10MeV.
        '''

        self.s_ref = np.interp(refPoint[1], np.flip(self.E_file), np.flip(self.s_file),
                               right=np.nan, left=np.nan) # Flip not too clean, to be sanitized ;)
        print(f'self.s_ref = {self.s_ref}')
        self.s = np.array(range(0, self.im.shape[1])) / self.pixel_per_mm - (self.s_ref - refPoint[0])
        print(f'self.s min = {min(self.s)}; self.s max = {max(self.s)}')
        self.E = np.interp(self.s, self.s_file, self.E_file, right=np.nan, left=np.nan)
        self.E = np.flip(self.E)

        self.dsdE = np.interp(self.s, self.s_file, self.dsdE_file, right=np.nan, left=np.nan)

    def rescale2D_ref(self, refPoint=(40,10)):
        self.rescale1D_ref(refPoint)
        self.y_scale = np.array(range(0, self.im.shape[0])) / self.pixel_per_mm


class SpectrumImage:
    def __init__(self, spectrum: SpectrumRescale):
        data = spectrum.im # shape: (n_y, n_x)
        energy = spectrum.E
        self.angle = spectrum.y_scale

        valid_mask = ~np.isnan(energy) # take only not nan elements
        energy_valid = energy[valid_mask]
        data_valid = data[:, valid_mask]  # keep all rows, filter columns

        self.energy_regular = np.linspace(energy_valid.min(), energy_valid.max(), len(energy_valid))
        interp_func = interp1d(energy_valid, data_valid, axis=1, kind='linear')
        self.image_resampled = interp_func(self.energy_regular)

class SpectrumGraph:
    def __init__(self, _spectrum_image: SpectrumImage):
        t0=time.time()
        self.app = pg.mkQApp()
        self.win = pg.GraphicsLayoutWidget()
        self.win.show()

        self.plot = self.win.addPlot()
        print(f'Time elapsed for window creation: {time.time()-t0}s')
    
        #self.img = pg.ImageItem(_spectrum_image.image_resampled.T)  # transpose so axes align correctly
        #self.plot.addItem(self.img)

        # add axis labels
        t0=time.time()
        self.set_labels()
        print(f'Time elapsed for setting labels {time.time()-t0}s')
        t0=time.time()
        self.set_image(_spectrum_image)
        print(f'Time elapsed for setting image {time.time()-t0}s')
        t0=time.time()
        self.set_colorbar(_spectrum_image)
        print(f'Time elapsed for setting colorbar {time.time()-t0}s')

        

    def set_labels(self):
        self.plot.setLabel('bottom', 'Energy', units='MeV')  # adjust units
        self.plot.setLabel('left', 'Angle', units='mrad')
    

    def set_image(self, _spectrum_image):
        self.img = pg.ImageItem(_spectrum_image.image_resampled.T)  # transpose so axes align correctly
        self.plot.addItem(self.img)
        self.img.setRect(
        _spectrum_image.energy_regular[0],           # x origin
        _spectrum_image.angle[0],                    # y origin
        _spectrum_image.energy_regular[-1] - _spectrum_image.energy_regular[0],  # width
        _spectrum_image.angle[-1] - _spectrum_image.angle[0]         # height
        )
    def set_colorbar(self, _spectrum_image):
        self.bar = ColorBarItem(values=(_spectrum_image.image_resampled.min(), _spectrum_image.image_resampled.max()))
        self.img.setColorMap(VIRIDIS)
        self.bar.setImageItem(self.img)
        self.win.addItem(self.bar)



if __name__ == "__main__":
    
    '''Experimental data: spatial calibration 49µm/mm, zero position {x=1953px, y=635px}'''
    t0=time.time()
    spectrum = SpectrumRescale(plotCal=False, pixel_per_mm=21.28)
    #spectrum.rescale2D_zero(zero_px=1953)
    spectrum.rescale2D_ref(refPoint=(38.78, 10))
    print(f'{time.time()-t0}s to rescale spectrum')

    t0=time.time()
    spectrum_image = SpectrumImage(spectrum)
    print(f'{time.time()-t0}s to process data for plotting')

    t0=time.time()
    spectrum_graph = SpectrumGraph(spectrum_image)
    print(f'{time.time()-t0}s to create graphical objects')

    # start the event loop if running standalone
    pg.exec()
