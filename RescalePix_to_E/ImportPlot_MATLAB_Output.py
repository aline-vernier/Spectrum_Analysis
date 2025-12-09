import numpy as np
from matplotlib import pyplot as plt


def load_file(spectrum_path: str):
    spectrum = np.loadtxt(spectrum_path).T
    # Data is flipped to allow usage of numpy interpolation
    return spectrum

if __name__ == "__main__":
    spectrum = load_file("MATLAB_Output/MATLAB_Output_2.txt")
    plt.plot(spectrum[0], spectrum[1])
    plt.show()