
def stability_region_mds_4_2():
  ps = np.array([ [0, 0],
    [2.5, 0], [0, 2.5], [2, 0], [0, 2], [2, 1], [1, 2] ] )
  hull = ConvexHull(ps)
  
  for simplex in hull.simplices:
    plot.plot(ps[simplex, 0], ps[simplex, 1], 'k-')
  plot.plot(ps[hull.vertices,0], ps[hull.vertices,1], 'r--', lw=2)
  plot.plot(ps[hull.vertices[0],0], ps[hull.vertices[0],1], 'ro')
  # plot.show()
  plot.savefig("stability_region_mds_4_2.png")
  plot.gcf().clear()

def conf_matrix_to_M_C(G):
  n = G.shape[1]
  sys_repgroup_l = [[], []]
  
  for s in range(0, 2):
    for c in range(n):
      m = np.column_stack((G[:, s], G[:, c] ) )
      if np.linalg.det(m) == 0: # c is a systematic node for s
        sys_repgroup_l[s].append((c,))
    
    for subset in itertools.combinations(range(n), 2):
      if s in subset:
        continue
      m = np.column_stack((G[:,subset[0]], G[:,subset[1]]))
      # print("m= {}".format(m) )
      if np.linalg.det(m):
        # print("columns {}, {} are LI".format(os, c) )
        sys_repgroup_l[s].append((subset[0], subset[1]))
  print("sys_repgroup_l= {}".format(pprint.pformat(sys_repgroup_l) ) )
  
  r_0 = len(sys_repgroup_l[0] )
  r_1 = len(sys_repgroup_l[1] )
  r = r_0 + r_1
  # if r != len(sys_repgroup_l[1] ):
  #   log(ERROR, "Code was supposed to be symmetric, but it is not.")
  #   return 1
  x = sys_repgroup_l[0] + sys_repgroup_l[1]
  M = np.zeros((n, r))
  for i in range(n):
    for j in range(r):
      if i in x[j]:
        M[i, j] = 1
  print("M= {}".format(M) )
  
  C = np.zeros((2, r))
  C[0, 0:r_0] = 1
  C[1, r_0:r] = 1
  print("C= {}".format(C) )
  
  return r, M, C

def plot_xy_stability_region(n):
  # Rep(2)
  # G = np.matrix([[1,0], [0,1]])
  # G = np.matrix([[1,0], [0,1], [1,0], [0,1] ])
  # MDS(3,2)
  # G = np.matrix([[1,0], [0,1], [1,1]])
  # MDS(4,2)
  # G = np.matrix([[1,0], [0,1], [1,1], [2,1]])
  # MDS(5,2)
  # G = np.matrix([[1,0], [0,1], [1,1], [2,1], [3,1]])
  # MDS(6,2)
  # G = np.matrix([[1,0], [0,1], [1,1], [2,1], [3,1], [4,1]])
  # MDS(7,2)
  # G = np.matrix([[1,0], [0,1], [1,1], [2,1], [3,1], [4,1], [5,1]])
  # Mixed
  # G = np.matrix([[1,0], [0,1], [1,1], [2,1], [3,1], [1,0] ])
  # G = np.matrix([[1,0], [0,1], [1,1], [2,1], [3,1], [1,1] ])
  # G = np.matrix([[1,0], [0,1], [1,1], [1,0], [0,1], [1,1] ])
  # G = G.transpose()
  
  code = 'MDS' # 'Rep' # 'MDS' # 'Mixed'
  G = conf_matrix(code, n)
  
  print("G= {}".format(G) )
  n = G.shape[1]
  r, M, C = conf_matrix_to_M_C(G)
  p_l = []
  # 
  x = cvxpy.Variable(r, 1, name='x')
  for b in np.linspace(0, 1, 20):
    # print("b= {}".format(b) )
    
    length = math.sqrt((1-b)**2 + b**2)
    w = np.matrix([[(1-b)/length, b/length]] )
    # print("w.shape= {}, w= {}".format(w.shape, w) )
    w_ = w*C
    # print("w_= {}".format(w_) )
    # obj = cvxpy.Maximize(w*(C*x) )
    obj = cvxpy.Maximize(w_*x)
    # print("obj= {}".format(obj) )
    constraints = [M*x == 1, x >= 0] # [M*x <= 1, x >= 0]
    prob = cvxpy.Problem(obj, constraints)
    # print("prob= {}".format(prob) )
    prob.solve()
    print("status= {}".format(prob.status) )
    # print("optimal value= {}".format(prob.value) )
    y = C*(x.value)
    # print("optimal y= {}".format(y) )
    p_l.append((y[0], y[1]) )
  plot_hull(p_l, "plot_xy_stability_region_{}_n_{}.png".format(code, n) )
  plot.title('{}, n= {}, k= 2'.format(code, n) )
  plot.savefig(fname)
  plot.gcf().clear()
  log(WARNING, "done, code= {}, n= {}".format(code, n) )

if __name__ == "__main__":
  # random_2d_convex_hull()
  # stability_region_mds_4_2()
  # opt()
  
  plot_xy_stability_region(n=4)
  # for n in range(3, 10):
  #   plot_xy_stability_region(n)
  # plot_xy_stability_region(n=100)