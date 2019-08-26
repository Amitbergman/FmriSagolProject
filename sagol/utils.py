import sys

import logbook
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from sagol import config


def setup_log_handlers():
    logbook.StreamHandler(sys.stdout, level=config.LOGGING_LEVEL, bubble=True).push_application()


def show_matplotlib_pre_loaded_figure_in_jupyter(fig: Figure):
    # Suppresses: `Matplotlib is currently using module://ipykernel.pylab.backend_inline, which is a non-GUI backend, so cannot show the figure.`
    # https://stackoverflow.com/questions/49503869/attributeerror-while-trying-to-load-the-pickled-matplotlib-figure?noredirect=1&lq=1
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)
