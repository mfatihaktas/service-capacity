import scipy, cvxpy, pprint, itertools
import numpy as np
from scipy.spatial import ConvexHull

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
  
  # def cap_x_y_l
  
if __name__ == "__main__":
  # G = conf_mds_matrix(3, k=2)
  G = conf_mds_matrix(4, k=2)
  cf = CapFinder(G)
  r, M, C = cf.r_M_C()
