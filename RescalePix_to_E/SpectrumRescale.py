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
        self.im[self.im<mean_im*2]=0
        

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
            print(f'offset: {offset/self.pixel_per_mm}')
            self.s = (np.array(range(0, self.im.shape[1])) - offset) / self.pixel_per_mm # x-axis in space unit
            print(f'self.s = {self.s}')
            self.E = np.interp(self.s, self.s_file, self.E_file, right=np.nan, left=np.nan)
            self.E = np.flip(self.E)

            self.dsdE = np.interp(self.s, self.s_file, self.dsdE_file, right=np.nan, left=np.nan)
            self.dsdE = np.flip(self.dsdE)
            plt.plot(self.E, self.dsdE)
            plt.show()

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
        self.s = np.array(range(0, self.im.shape[1])) / self.pixel_per_mm - (self.s_ref - refPoint[0])
        self.E = np.interp(self.s, self.s_file, self.E_file, right=np.nan, left=np.nan)
        self.E = np.flip(self.E)

        self.dsdE = np.interp(self.s, self.s_file, self.dsdE_file, right=np.nan, left=np.nan)
        self.dsdE = np.flip(self.dsdE)


    def rescale2D_ref(self, refPoint=(40,10)):
        self.rescale1D_ref(refPoint)
        self.y_scale = np.array(range(0, self.im.shape[0])) / self.pixel_per_mm


    def dNdpx_to_dNdE(self, dndpix_array, bkg_array, dsdE_array):
        dNdE = lambda dndpix, bkg, dsdE: (dndpix-bkg) * abs(dsdE) * self.pixel_per_mm
        dNdE_vect = np.vectorize(dNdE)
        return dNdE_vect(dndpix_array, bkg_array, dsdE_array)
    
    def bin_1MeV(self, dn_dE):
    
        # Define bin edges at integer MeV values
        E_min = np.floor(min(self.E))
        E_max = np.ceil(max(self.E))
   
        bin_edges = np.arange(E_min, E_max + 1, 1.0)  # 1 MeV bins
        
        # Initialize arrays for binned data
        n_bins = len(bin_edges) - 1
        binned_counts = np.zeros(n_bins)
        bin_centers = bin_edges[:-1] + 0.5
        
        # For each original data point, add its contribution to the appropriate bin
        for i in range(len(self.E)):
            # Find which bin this energy falls into
            bin_idx = np.searchsorted(bin_edges[:-1], self.E[i], side='right') - 1
            
            if 0 <= bin_idx < n_bins:
                # Estimate the energy width this point represents
                if i == 0:
                    dE = self.E[1] - self.E[0]
                elif i == len(self.E) - 1:
                    dE = self.E[-1] - self.E[-2]
                else:
                    dE = (self.E[i+1] - self.E[i-1]) / 2
                
                # Add counts from this point (counts/MeV * MeV = counts)
                binned_counts[bin_idx] += dn_dE[i] * dE
        
        # Average counts per MeV in each bin
        binned_counts_per_MeV = binned_counts / 1.0  # divided by bin width (1 MeV)
        return bin_centers, binned_counts, binned_counts_per_MeV

    def sum_dNdE_cursors(self, cur_Em: int, cur_Ep: int, cur_Bm: int, cur_Bp: int):
        '''
        :param cur_Ep: high electron signal cursor position, in px
        :param cur_Em: low electron cursor position, in px
        :param cur_Bp: high background signal cursor position, in px
        :param cur_Bm: low background cursor position, in px
        :return:
        '''
        if cur_Em or cur_Ep is None:
            self.cur_Ep = int(self.im.shape[0]/2+100)
            self.cur_Em = int(self.im.shape[0]/2-100)
        if cur_Bm or cur_Bp is None:
            self.cur_Bm = 0
            self.cur_Bp = 100

        sub_im = self.im[cur_Em:cur_Ep, :]
        sub_im_bkg = self.im[cur_Bm:cur_Bp, :]
        dn_dpix = np.sum(sub_im, axis=0)
        bkg = np.sum(sub_im_bkg, axis=0)*(cur_Ep-cur_Em)/(cur_Bp-cur_Bm)
        dn_dE = self.dNdpx_to_dNdE(dn_dpix, bkg, self.dsdE)
        bin_centers, binned_counts, binned_counts_per_MeV = self.bin_1MeV(dn_dE)
        plt.plot(bin_centers, binned_counts_per_MeV)
        #plt.xlim(5, 80)
        plt.show()
    
    

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
    spectrum = SpectrumRescale(plotCal=False, pixel_per_mm=20.408)
    spectrum.rescale2D_zero(zero_px=1953)
    #spectrum.rescale2D_ref(refPoint=(38.78, 10))
    spectrum.sum_dNdE_cursors(515, 755, 300, 400)
    print(f'{time.time()-t0}s to rescale spectrum')

    t0=time.time()
    spectrum_image = SpectrumImage(spectrum)
    print(f'{time.time()-t0}s to process data for plotting')

    t0=time.time()
    spectrum_graph = SpectrumGraph(spectrum_image)
    print(f'{time.time()-t0}s to create graphical objects')


    # start the event loop if running standalone
    pg.exec()
