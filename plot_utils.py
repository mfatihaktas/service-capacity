import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('Agg')
import matplotlib.pyplot as plot
import itertools

from log_utils import *

NICE_BLUE = '#66b3ff'
NICE_RED = '#ff9999'
NICE_GREEN = '#99ff99'
NICE_ORANGE = '#ffcc99'

nice_color = itertools.cycle((NICE_BLUE, NICE_RED, NICE_GREEN, NICE_ORANGE))
dark_color = itertools.cycle(('green', 'purple', 'blue', 'magenta', 'purple', 'gray', 'brown', 'turquoise', 'gold', 'olive', 'silver', 'rosybrown', 'plum', 'goldenrod', 'lightsteelblue', 'lightpink', 'orange', 'darkgray', 'orangered'))
light_color = itertools.cycle(('silver', 'rosybrown', 'plum', 'lightsteelblue', 'lightpink', 'orange', 'turquoise'))
linestyle = itertools.cycle(('-', '--', '-.', ':') )
marker_cycle = itertools.cycle(('x', '+', 'v', '^', 'p', 'd', '<', '>', '1' , '2', '3', '4') )
skinny_marker_l = ['x', '+', '1', '2', '3', '4']

mew, ms = 1, 2 # 3, 5

def prettify(ax):
  plot.tick_params(top='off', right='off', which='both')
  ax.patch.set_alpha(0.2)
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)

def plot_points(x_y_l, fname):
  x_l, y_l = [], []
  for x_y in x_y_l:
    x_l.append(x_y[0] )
    y_l.append(x_y[1] )
  
  plot.plot(x_l, y_l, color=NICE_BLUE, marker='o', ls='None')
  plot.savefig('{}.png'.format(fname), bbox_inches='tight')
  plot.gcf().clear()
  log(INFO, "done.")
