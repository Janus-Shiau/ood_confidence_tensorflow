import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter


def draw_confidence_histogram(correct, wrong, n_bins=20):
    _, ax = plt.subplots()

    # We can set the number of bins with the `bins` kwarg
    ax.hist(correct, bins=n_bins, density=True, alpha = 0.5, label='correct prediction')
    ax.hist(wrong, bins=n_bins, density=True, alpha = 0.5, label='wrong prediction')

    ax.set_xlabel('Confidence')
    ax.set_ylabel('Normalized number of sample')
    ax.set_title('Histogram of Confidence Estimation')

    plt.legend(loc='upper right')
    plt.show()
