import math, scipy, cvxpy, pprint, itertools
import numpy as np
from scipy.spatial import ConvexHull

from plot_utils import *
from conf_gen import *
from log_utils import *

# ########################################  CapFinder  ########################################### #
'''
Storage for k symbols is configured by matrix G.
Left multiplying a row vector of symbols with G gives the content stored across the ordered nodes.

Each of the k symbols can be recovered from a repair group (single node in case of systematic repair).
Let ai be the rate of `a` supplied by its ith repair group.
x = [a0, a1, ..., b0, b1, ...]
Total rate supplied for each symbol can then be easily computed as Cx where
C = [[1, 1, ..., 0, 0, ...], [0, 0, ..., 1, 1, ...]]

Each xi is supplied by contribution of a set of nodes. This connection is made by the matrix M such that
M[j, i] = 1 if node-j contributes supplying xi.
Naturally, Mx <= 1 where the capacity of each node is 1.
'''
class CapFinder(object):
  def __init__(self, G):
    self.k = G.shape[0]
    self.n = G.shape[1]
    
    print("G= {}".format(G) )
  
  def __repr__(self):
    return 'CapFinder[k= {}, n= {}]'.format(self.k, self.n)
  
  def r_M_C(self):
    def does_right_contain_left(t1, t2):
      for e in t1:
        if e not in t2:
          return False
      return True
    
    sys__repgroup_l_l = [[] for s in range(self.k) ]
    for s in range(self.k):
      y = G[:, s].reshape((self.k, 1))
      repgroup_l = []
      for repair_size in range(1, self.k+1):
        for subset in itertools.combinations(range(self.n), repair_size):
          # Check if subset contains any previously registered smaller repair group
          skip = False
          for rg in repgroup_l:
            if does_right_contain_left(rg, subset):
              skip = True
              break
          if skip:
            continue
          l = [G[:, i] for i in subset]
          A = np.array(l).reshape((self.k, len(l) ))
          # print("\n")
          # blog(A=A, y=y)
          x, residuals, _, _ = np.linalg.lstsq(A, y)
          # blog(x=x, residuals=residuals)
          if np.sum(residuals) < 0.0001:
            repgroup_l.append(subset)
      sys__repgroup_l_l[s] = repgroup_l
    blog(sys__repgroup_l_l=sys__repgroup_l_l)
    # C
    x = []
    for rg_l in sys__repgroup_l_l:
      x.extend(rg_l)
    
    r = len(x)
    C = np.zeros((self.k, r))
    c = 0
    for s, rg_l in enumerate(sys__repgroup_l_l):
      c_ = c + len(rg_l)
      C[s, c:c_] = 1
      c = c_
    # M
    M = np.zeros((self.n, r))
    for i in range(self.n):
      for j in range(r):
        if i in x[j]:
          M[i, j] = 1
    
    blog(r=r, C=C, M=M)
    return r, M, C
  
  def cap_point_l(self):
    r, M, C = cf.r_M_C()
    p_l = []
    
    x = cvxpy.Variable(shape=(r, 1), name='x')
    counter = 1
    while counter < 10**self.k:
      counter_str = str(counter).zfill(self.k)
      l = [int(char) for char in counter_str]
      length = math.sqrt(sum([e**2 for e in l] ) )
      w = np.array(l).reshape((1, self.k))/length
      
      # blog(w=w)
      w_ = np.dot(w, C) # w*C
      # blog(w_=w_)
      obj = cvxpy.Maximize(w_*x)
      
      constraints = [M*x == 1, x >= 0] # [M*x <= 1, x >= 0]
      prob = cvxpy.Problem(obj, constraints)
      prob.solve()
      y = np.dot(C, x.value)
      # blog(prob=prob, status=prob.status, opt_val=prob.value, y=y)
      # p_l.append(tuple(map(tuple, y) ) )
      # y = (e[0] for e in y)
      p_l.append([e[0] for e in tuple(y) ] )
      
      counter += 1
    return p_l
  
  def plot_2d_servcap(self):
    point_l = self.cap_point_l()
    point_l.append((0, 0))
    # print("point_l= {}".format(point_l) )
    
    # x_l, y_l = [], []
    # for x_y in point_l:
    #   x_l.append(x_y[0] )
    #   y_l.append(x_y[1] )
    # plot.plot(x_l, y_l, c=NICE_BLUE, marker='o', ls=':')
    
    points_inrows = np.array(point_l).reshape((len(point_l), self.k))
    # print("points_inrows= \n{}".format(points_inrows) )
    hull = ConvexHull(points_inrows)
    for simplex in hull.simplices:
      plot.plot(points_inrows[simplex, 0], points_inrows[simplex, 1], 'k-')
    plot.plot(points_inrows[hull.vertices, 0], points_inrows[hull.vertices, 1], 'r--', lw=2)
    plot.plot(points_inrows[hull.vertices[0], 0], points_inrows[hull.vertices[0], 1], 'ro')
    
    prettify(plot.gca() )
    plot.title('n= {}, k= {}'.format(self.n, self.k) )
    plot.xlabel('a', fontsize=14)
    plot.ylabel('b', fontsize=14)
    fig = plot.gcf()
    fig.set_size_inches(5, 5)
    plot.savefig('plot_2d_servcap_n{}_k{}.png'.format(self.n, self.k), bbox_inches='tight')
    fig.clear()
    log(INFO, "done.")
  
if __name__ == "__main__":
  # G = conf_mds_matrix(3, k=2)
  G = conf_mds_matrix(5, k=2)
  cf = CapFinder(G)
  # r, M, C = cf.r_M_C()
  # cf.cap_point_l()
  cf.plot_2d_servcap()
