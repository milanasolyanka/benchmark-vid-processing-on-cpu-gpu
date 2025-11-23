import numpy as np

def build_lut(n):
    # Каждому значению 0..255 находим квант и центр кванта
    bins = np.linspace(0, 256, n+1)
    centers = ((bins[:-1] + bins[1:]) / 2).astype(np.uint8)

    lut = np.zeros(256, dtype=np.uint8)

    for value in range(256):
        bin_index = np.searchsorted(bins, value, side='right') - 1
        bin_index = min(max(bin_index, 0), n-1)
        lut[value] = centers[bin_index]

    return lut
