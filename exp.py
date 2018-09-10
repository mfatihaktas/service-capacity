import numpy as np

def plot_rand_2dconvexhull():
  ps = np.random.rand(30, 2)
  hull = ConvexHull(ps)
  
  print("ps= {}".format(ps) )
  
  plot.plot(ps[:,0], ps[:,1], 'o')
  for simplex in hull.simplices:
    plot.plot(ps[simplex, 0], ps[simplex, 1], 'k-')
  plot.plot(ps[hull.vertices,0], ps[hull.vertices,1], 'r--', lw=2)
  plot.plot(ps[hull.vertices[0],0], ps[hull.vertices[0],1], 'ro')
  # plot.show()
  plot.savefig('plot_rand_2dconvexhull.png')
  plot.gcf().clear()

def opt():
  # Problem data.
  m = 30
  n = 20
  np.random.seed(1)
  A = np.random.randn(m, n)
  b = np.random.randn(m)
  
  # Construct the problem.
  x = Variable(n)
  obj = Minimize(sum_squares(A*x - b))
  constraints = [0 <= x, x <= 1]
  prob = Problem(obj, constraints)
  
  # The optimal objective value is returned by prob.solve()
  result = prob.solve()
  # The optimal value for x is stored in x.value
  print("x= {}".format(x.value) )
  # The optimal Lagrange multiplier for a constraint is stored in constraint.dual_value
  print("constraints[0].dual_value= {}".format(constraints[0].dual_value) )
