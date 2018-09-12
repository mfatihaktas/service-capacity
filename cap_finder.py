import math, scipy, cvxpy, pprint, itertools
import numpy as np
from scipy import spatial
from fillplots import plot_regions

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
  
  def cap_boundary_point_l(self):
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
  
  def cap_hyperplane(self):
    point_l = self.cap_boundary_point_l()
    point_l.append((0, 0))
    points_inrows = np.array(point_l).reshape((len(point_l), self.k))
    hull = scipy.spatial.ConvexHull(points_inrows)
    
    # `hull` := {x| Ax <= b}
    m, n = hull.equations.shape
    A = np.mat(hull.equations[:, 0:n-1] )
    b = np.mat(-hull.equations[:, n-1] ).T
    # A = A[A > 0.001]
    # b = b[b > 0.001]
    A[A < 0.001] = 0
    rowi_tokeep_l = []
    for i in range(A.shape[0] ):
      if A[i, :].sum() > 0.001:
        rowi_tokeep_l.append(i)
    A = A[rowi_tokeep_l, :]
    b = b[rowi_tokeep_l, :]
    
    # blog(A=A, b=b)
    return A, b
  
  def integrate_jointpdf_overcaphyperlane(self, jointpdf):
    A, b = self.cap_hyperplane()
    blog(A=A, b=b)
    
    ranges = []
    for i in range(self.k):
      if i == 0:
        ranges.append((0, self.n) )
      else:
        def u(*args, i=i):
          # print("u:: i= {}".format(i) )
          A_ = A[:, [j for j in range(self.k) if j != i] ]
          x_ = np.array(args).reshape((self.k-1, 1))
          # blog(A_=A_, x_=x_)
          
          b_ = b - A_*x_
          a = A[:, i]
          # blog(b_=b_, a=a)
          for j in range(a.shape[0] ):
            b_[j, 0] /= a[j, 0]
          # print("final b_= {}".format(b_) )
          max_val = b_.min() # b_.max()
          return max(max_val, 0)
        ranges.append(u)
        # ranges.append((0, self.n) )
    # Plot to check if u() works
    x_l = np.linspace(*ranges[0], 50)
    y_l = [ranges[1](x) for x in x_l]
    plot.plot(x_l, y_l, c=NICE_BLUE, marker='o', ls=':')
    plot.savefig('plot_deneme.png', bbox_inches='tight')
    
    # result, abserr = scipy.integrate.nquad(jointpdf, ranges)
    # blog(result=result, abserr=abserr)
  
  def plot_2d_servcap(self):
    '''
    point_l = self.cap_boundary_point_l()
    point_l.append((0, 0))
    
    points_inrows = np.array(point_l).reshape((len(point_l), self.k))
    hull = scipy.spatial.ConvexHull(points_inrows)
    for simplex in hull.simplices:
      # print("x_l= {}, y_l= {}".format(points_inrows[simplex, 0], points_inrows[simplex, 1] ) )
      plot.plot(points_inrows[simplex, 0], points_inrows[simplex, 1], c=NICE_BLUE, marker='o', ls='-', lw=3)
    # plot.plot(points_inrows[hull.vertices, 0], points_inrows[hull.vertices, 1], 'r--', lw=2)
    # plot.plot(points_inrows[hull.vertices[0], 0], points_inrows[hull.vertices[0], 1], 'ro')
    
    plot.fill(points_inrows[simplex, 0], points_inrows[simplex, 1], c=NICE_BLUE, alpha=0.5)
    '''
    
    # plot_regions([[(lambda x: (1.0 - x ** 2) ** 0.5, True), (lambda x: x,) ] ], xlim=(0, 1), ylim=(0, 1) )
    A, b = self.cap_hyperplane()
    blog(A=A, b=b)
    # for i in range(A.shape[0] ):
    #   print("A[i, 0]= {}, A[i, 1]= {}".format(A[i, 0], A[i, 1] ) )
    
    eq_l = [lambda x, i=i: b[i, 0]/A[i, 1] - A[i, 0]/A[i, 1]*x for i in range(A.shape[0] ) if A[i, 0] > 0.001]
    x_l = [x for x in np.linspace(0, 5, 20) ]
    # for i, eq in enumerate(eq_l):
    #   plot.plot(x_l, [eq(x) for x in x_l], label='i= {}'.format(i), c=next(dark_color), marker='.', ls=':')
    # plot.xlim((0, 10))
    # plot.ylim((0, 10))
    
    plot.legend()
    plot_regions(
      [[(eq, True) for eq in eq_l] ],
      xlim=(0, 10), ylim=(0, 10) )
    
    # plot_regions(
    #   [[(lambda x: 1 - x, True), (lambda x: 2 - 4*x, True) ] ],
    #   xlim=(0, 1), ylim=(0, 1) )
    
    # prettify(plot.gca() )
    plot.title('n= {}, k= {}'.format(self.n, self.k) )
    plot.xlabel('a', fontsize=14)
    plot.ylabel('b', fontsize=14)
    fig = plot.gcf()
    fig.set_size_inches(5, 5)
    plot.savefig('plot_2d_servcap_n{}_k{}.png'.format(self.n, self.k), bbox_inches='tight')
    fig.clear()
    log(INFO, "done.")

if __name__ == "__main__":
  G = conf_mds_matrix(10, k=2)
  cf = CapFinder(G)
  # r, M, C = cf.r_M_C()
  # cf.cap_boundary_point_l()
  cf.plot_2d_servcap()
  # cf.cap_hyperplane()
  cf.integrate_jointpdf_overcaphyperlane(lambda x, y: 1)
  
  # M = np.array(list(range(16) ) ).reshape((4, 4))
  # M_ = M[:, [1, 2]] # M[[1, 2],]
  # blog(M=M, M_=M_)
  
