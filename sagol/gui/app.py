from sagol.utils import setup_log_handlers
import matplotlib

from sagol.gui.initial_window_new import InitialWindowNew

setup_log_handlers()
matplotlib.rcParams.update({'font.size': 6})

InitialWindowNew().open()
