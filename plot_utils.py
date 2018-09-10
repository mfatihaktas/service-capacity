


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