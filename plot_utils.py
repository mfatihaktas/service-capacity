import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('Agg')
import matplotlib.pyplot as plot
import itertools

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
  ax.patch.set_alpha(0.2)
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)

def plot_hull(x_y_l, fname, title):
  # ps = np.empty((len(x_y_l), 2))
  # for i, x_y in enumerate(x_y_l):
  #   # ps[i, :] = [[p[0], p[1]]]
  #   ps[i, 0] = x_y[0]
  #   ps[i, 1] = x_y[1]
  
  # print("ps= {}".format(ps) )
  # plot.plot(ps[:,0], ps[:,1], 'o')
  
  plot.plot(ps[:,0], ps[:,1], 'o')
  
  """
  hull = ConvexHull(ps)
  for simplex in hull.simplices:
    plot.plot(ps[simplex, 0], ps[simplex, 1], 'k-')
  plot.plot(ps[hull.vertices,0], ps[hull.vertices,1], 'r--', lw=2)
  plot.plot(ps[hull.vertices[0],0], ps[hull.vertices[0],1], 'ro')
  """