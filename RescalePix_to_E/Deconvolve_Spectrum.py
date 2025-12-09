import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import time
import pyqtgraph as pg
from pyqtgraph import ColorBarItem
from scipy.interpolate import interp1d

VIRIDIS = pg.colormap.get('viridis')


def spectrum_image(im_path: str, revert: bool):
    """Opens image from file, returns 2D numpy array"""
    if revert:
        return np.flip(np.array(Image.open(im_path)), axis=1)
    else:
        return np.array(Image.open(im_path))


class CalibrationData:
    """
    :param cal_path: path to calibration file (absolute)
    :param spacing: spacing between energy points, in MeV
    Calibration Data takes a calibration file formatted as:
        1st column: energy in MeV
        2nd column: ds/dE in mm/MeV
        3rd column: s in mm (longitudinal coordinate along the lanex
                    with respect to beam position without magnet)
    Attributes:
        energy: array with equal spacing in energy
        dsde: ds/dE interpolated for each energy value
        s: s interpolated for each energy value
    """
    def __init__(self, cal_path: str, spacing: float):
        cal = np.loadtxt(cal_path).T
        e_file = cal[0]
        dsde_file = cal[1]
        s_file = cal[2]

        self.spacing = spacing

        self.energy_equal_spacing(e_file, self.spacing)
        self.interpolated_ds_de(e_file, dsde_file)
        self.interpolated_s(e_file, s_file)


    def energy_equal_spacing(self, _e_file, _spacing):
        emin = _e_file.min()
        emax = _e_file.max()

        self.energy = np.linspace(emin, emax, int((emax - emin)/_spacing))
        self.dsde = None
        self.s = None

    def interpolated_ds_de(self, _e_file, _dsde_file):
        self.dsde = np.interp(self.energy, _e_file, _dsde_file, right=np.nan, left=np.nan)

    def interpolated_s(self, _e_file, _s_file):
        self.s = np.interp(self.energy, _e_file, _s_file, right=np.nan, left=np.nan)


class DeconvolvedSpectrum:
    """
    :param image: image is a 2D numpy array, not an image :)
    :param pixel_per_mm: calibration in pixel per mm
    :param mrad_per_pix: calibration mrad per pixel
    :param ref_mode: string, should be "zero" or "refpoint"
    :param ref_point: tuple, either (x, y) coordinates of zero in pixels if chosen method is "zero"
                             or (x, E) x-coordinate (mm) of a given energy (MeV) is chosen method is "refpoint"

    Public attributes:

    """
    def __init__(self, image: np.ndarray, calibration: CalibrationData,
                 pixel_per_mm: float, mrad_per_pix: float,
                 ref_mode: str, ref_point: tuple):
        self.pixel_per_mm = pixel_per_mm
        self.mrad_per_pix = mrad_per_pix
        self.ref_mode = ref_mode
        self.ref_point = ref_point
        self.calibration = calibration
        self._image_dimensions = image.shape

        self.s_offset_zero(image)
        self.set_angle()
    def s_offset_zero(self, image):
        x_min = self.ref_point[0]-self._image_dimensions[1]
        x_max = self.ref_point[0]
        x_lanex = np.linspace(x_min, x_max, self._image_dimensions[1]) / self.pixel_per_mm

        # Filter data with Yamask
        x_lanex[x_lanex < min(self.calibration.s)] = np.nan
        x_lanex[x_lanex > max(self.calibration.s)] = np.nan
        self._energy_uneven = np.interp(x_lanex, np.flip(self.calibration.s), np.flip(self.calibration.energy),
                                        right=np.nan, left=np.nan)

        valid_yamask = ~np.isnan(x_lanex)  # take only not nan elements
        self._energy_uneven = self._energy_uneven[valid_yamask]
        self._filtered_image = image[:, valid_yamask]  # keep all rows, filter columns
        interp_func = interp1d(self._energy_uneven, self._filtered_image, axis=1, kind='linear')

        emin = min(self._energy_uneven)
        emax = max(self._energy_uneven)

        self.energy = np.linspace(emin, emax, int((emax-emin)/self.calibration.spacing))
        self.image = interp_func(self.energy)


    def set_angle(self):
        self.angle = np.linspace(-self._image_dimensions[0] / 2,
                                 self._image_dimensions[0] / 2, self._image_dimensions[0]) * self.mrad_per_pix


class SpectrumGraph:
    def __init__(self, _spectrum_image: DeconvolvedSpectrum):
        self.app = pg.mkQApp()
        self.win = pg.GraphicsLayoutWidget()
        self.win.show()

        self.plot = self.win.addPlot()
        self.set_labels()
        self.set_image(_spectrum_image)
        self.set_colorbar(_spectrum_image)

    def set_labels(self):
        self.plot.setLabel('bottom', 'Energy', units='MeV')  # adjust units
        self.plot.setLabel('left', 'Angle', units='mrad')

    def set_image(self, _spectrum_image):
        self.img = pg.ImageItem(_spectrum_image.image.T)  # transpose so axes align correctly
        self.plot.addItem(self.img)

        self.img.setRect(
            _spectrum_image.energy[0],  # x origin
            _spectrum_image.angle[0],  # y origin
            _spectrum_image.energy[-1] - _spectrum_image.energy[0],  # width
            _spectrum_image.angle[-1] - _spectrum_image.angle[0]  # height
        )

    def set_colorbar(self, _spectrum_image):
        self.bar = ColorBarItem(values=(_spectrum_image.image.min(), _spectrum_image.image.max()))
        self.img.setColorMap(VIRIDIS)
        self.bar.setImageItem(self.img)
        self.win.addItem(self.bar)


if __name__ == "__main__":
    """
    Experimental data: 
        spatial calibration 49µm/px, 
        zero position {x=1953px, y=635px}, 
        signal calibration 4.33e-6pC/count
    """
    # Load image and calibration
    spImage = spectrum_image(im_path=".\\calib\\magnet0.4T_Soectrum_isat4.9cm_26bar_gdd25850_HeAr_0002.TIFF",
                           revert=True)
    calibration_data = CalibrationData(cal_path=".\\calib\\dsdE_Small_LHC.txt", spacing=0.1)

    # Deconvolve data
    deconvolved_spectrum = DeconvolvedSpectrum(spImage, calibration_data,
                                               20.408, 0.1,
                                               "zero", (1953, 635))
    # Show 2D plot
    graph = SpectrumGraph(deconvolved_spectrum)
    pg.exec()
