import math, scipy, cvxpy, pprint, itertools, string
import numpy as np
from scipy import spatial
from fillplots import plot_regions

from mpl_toolkits import mplot3d

from plot_utils import *
from conf_gen import *
from log_utils import *

# ########################################  ConfInspector  ########################################### #
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
class ConfInspector(object):
  def __init__(self, G):
    self.G = G
    
    self.k = G.shape[0]
    self.n = G.shape[1]
    
    self.sys__repgroup_l_l = self.get_sys__repgroup_l_l()
    blog(ConfInspector=self, sys__repgroup_l_l=self.sys__repgroup_l_l)
    self.A, self.b = self.cap_hyperplane()
  
  def __repr__(self):
    return 'ConfInspector[k= {}, n= {}, G=\n{}]'.format(self.k, self.n, self.G)
  
  def to_sysrepr(self):
    sym_l = string.ascii_lowercase[:self.k]
    node_l = []
    for c in range(self.n):
      l = []
      for r in range(self.k):
        if G[r, c] != 0:
          num = int(G[r, c])
          l.append('{}{}'.format(num, sym_l[r] ) if num != 1 else '{}'.format(sym_l[r] ) )
      node_l.append('+'.join(l) )
    return str(node_l)
  
  def get_sys__repgroup_l_l(self):
    def does_right_contain_left(t1, t2):
      for e in t1:
        if e not in t2:
          return False
      return True
    
    sys__repgroup_l_l = [[] for s in range(self.k) ]
    for s in range(self.k):
      y = self.G[:, s].reshape((self.k, 1))
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
          l = [self.G[:, i] for i in subset]
          A = np.array(l).reshape((self.k, len(l) ))
          # print("\n")
          x, residuals, _, _ = np.linalg.lstsq(A, y)
          # log(INFO, "", A=A, y=y, x=x, residuals=residuals)
          if np.sum(residuals) < 0.0001:
            repgroup_l.append(subset)
      sys__repgroup_l_l[s] = repgroup_l
    # blog(sys__repgroup_l_l=sys__repgroup_l_l)
    return sys__repgroup_l_l
  
  def r_M_C(self):
    # C
    x = []
    for rg_l in self.sys__repgroup_l_l:
      x.extend(rg_l)
    
    r = len(x)
    C = np.zeros((self.k, r))
    c = 0
    for s, rg_l in enumerate(self.sys__repgroup_l_l):
      c_ = c + len(rg_l)
      C[s, c:c_] = 1
      c = c_
    # M
    M = np.zeros((self.n, r))
    for i in range(self.n):
      for j in range(r):
        if i in x[j]:
          M[i, j] = 1
    
    # log(INFO, "", r=r, C=C, M=M)
    return r, M, C
  
  def cap_boundary_point_l(self):
    r, M, C = self.r_M_C()
    p_l = []
    
    x = cvxpy.Variable(shape=(r, 1), name='x')
    counter = 1
    while counter < 10**self.k:
      counter_str = str(counter).zfill(self.k)
      l = [int(char) for char in counter_str]
      length = math.sqrt(sum([e**2 for e in l] ) )
      w = np.array(l).reshape((1, self.k))/length
      
      # log(INFO, "", w=w)
      w_ = np.dot(w, C) # w*C
      # blog(w_=w_)
      obj = cvxpy.Maximize(w_*x)
      
      constraints = [M*x == 1, x >= 0] # [M*x <= 1, x >= 0]
      prob = cvxpy.Problem(obj, constraints)
      prob.solve()
      # log(INFO, "optimization problem;", prob=prob, status=prob.status, opt_val=prob.value)
      y = np.dot(C, x.value)
      # p_l.append(tuple(map(tuple, y) ) )
      # y = (e[0] for e in y)
      p = [e[0] for e in tuple(y) ]
      p_l.append(p)
      ## Add projections on the axes as well to make sure point_l form a convex hull
      for i in range(self.k):
        p_ = list(p)
        p_[i] = 0
        p_l.append(p_)
      
      counter += 1
    return p_l
  
  def cap_hyperplane(self):
    point_l = self.cap_boundary_point_l()
    point_l.append((0, 0))
    # log(INFO, "", point_l=point_l)
    # plot_points(point_l, fname='points')
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
  
  def integrate_overcaphyperlane(self, func):
    ranges = []
    for i in range(self.k):
      '''
      nquad feeds arguments to u by reducing one from the head at a time.
      e.g., for func(x0, x1, x2, x3)
      ranges shall be 
      [lim0(x1, x2, x3)
       lim1(x2, x3)
       lim2(x3)
       lim3() ]
      '''
      def u(*args, i=i):
        # print("i= {}, args= {}".format(i, args) )
        if len(args) == 0:
          return 0, self.n
        
        # print("u:: i= {}".format(i) )
        A_ = self.A[:, [j for j in range(self.k) if j != i] ]
        x_ = np.array(args).reshape((self.k-1, 1))
        # blog(A_=A_, x_=x_)
        
        b_ = self.b - A_*x_
        a = self.A[:, i]
        # blog(b_=b_, a=a)
        for j in range(a.shape[0] ):
          b_[j, 0] /= a[j, 0]
        # print("final b_= {}".format(b_) )
        max_val = b_.min() # b_.max()
        return 0, max(max_val, 0)
      ranges.append(u)
      # ranges.append((0, self.n) )
    # Plot to check if u() works
    # x_l = np.linspace(*ranges[0], 50)
    # y_l = [ranges[1](x) for x in x_l]
    # plot.plot(x_l, y_l, c=NICE_BLUE, marker='o', ls=':')
    # plot.savefig('plot_deneme.png', bbox_inches='tight')
    
    result, abserr = scipy.integrate.nquad(func, ranges)
    # blog(result=result, abserr=abserr)
    return round(result, 2)
  
  def plot_servcap_2d(self):
    # '''
    point_l = self.cap_boundary_point_l()
    point_l.append((0, 0))
    points_inrows = np.array(point_l).reshape((len(point_l), self.k))
    hull = scipy.spatial.ConvexHull(points_inrows)
    for simplex in hull.simplices:
      # print("x_l= {}, y_l= {}".format(points_inrows[simplex, 0], points_inrows[simplex, 1] ) )
      plot.plot(points_inrows[simplex, 0], points_inrows[simplex, 1], c=NICE_ORANGE, marker='o', ls='-', lw=3)
    # plot.plot(points_inrows[hull.vertices, 0], points_inrows[hull.vertices, 1], 'r--', lw=2)
    # plot.plot(points_inrows[hull.vertices[0], 0], points_inrows[hull.vertices[0], 1], 'ro')
    # plot.fill(points_inrows[simplex, 0], points_inrows[simplex, 1], c=NICE_ORANGE, alpha=0.5)
    plot.fill(points_inrows[hull.vertices, 0], points_inrows[hull.vertices, 1], c=NICE_ORANGE, alpha=0.3)
    # '''
    
    '''
    A, b = self.A, self.b
    # blog(A=A, b=b)
    # for i in range(A.shape[0] ):
    #   print("A[i, 0]= {}, A[i, 1]= {}".format(A[i, 0], A[i, 1] ) )
    
    eq_l = [lambda x, i=i: b[i, 0]/A[i, 1] - A[i, 0]/A[i, 1]*x for i in range(A.shape[0] ) if A[i, 0] > 0.001]
    x_l = [x for x in np.linspace(0, 5, 20) ]
    # for i, eq in enumerate(eq_l):
    #   plot.plot(x_l, [eq(x) for x in x_l], label='i= {}'.format(i), c=next(dark_color), marker='.', ls=':')
    # plot.xlim((0, 10))
    # plot.ylim((0, 10))
    
    plot.legend()
    # plot_regions(
    #   [[(lambda x: 1 - x, True), (lambda x: 2 - 4*x, True) ] ],
    #   xlim=(0, 1), ylim=(0, 1) )
    plot_regions(
      [[(eq, True) for eq in eq_l] ],
      xlim=(0, 10), ylim=(0, 10) )
    '''
    
    # prettify(plot.gca() )
    # plot.title('n= {}, k= {}'.format(self.n, self.k) )
    plot.xlabel(r'$\lambda_a$', fontsize=16)
    plot.ylabel(r'$\lambda_b$', fontsize=16)
    fig = plot.gcf()
    fig.set_size_inches(5, 5)
    plot.savefig('plot_servcap_2d_n{}_k{}.png'.format(self.n, self.k), bbox_inches='tight')
    fig.clear()
    log(INFO, "done.")
  
  def cost(self, *args):
    x = np.array(args).reshape((self.k, 1))
    if np.any(np.greater(np.dot(self.A, x), self.b) ):
      # log(ERROR, "outside of capacity region;", x=x)
      return 0
    
    total_cost = 0
    for sys, rg_l in enumerate(self.sys__repgroup_l_l):
      cap_demand = x[sys, 0]
      i, cost = 0, 0
      while cap_demand - i > 0.001 and i < len(rg_l):
        cost += min(cap_demand - i, 1)*len(rg_l[i] )
        i += 1
      
      if cap_demand - i > 0.001:
        log(ERROR, "should have been supplied!", x=x)
        return
      total_cost += cost
    return total_cost
  
  def moment_cost(self, pop_jointpdf, i):
    func = lambda *args: self.cost(*args)**i * pop_jointpdf(*args)
    return self.integrate_overcaphyperlane(func)
  
  def plot_cost_2d(self):
    X, Y = np.mgrid[0:self.n:100j, 0:self.n:100j]
    # blog(X_shape=X.shape, Y_shape=Y.shape)
    
    x_l, y_l = X.ravel(), Y.ravel()
    cost_l = []
    for i, x in enumerate(x_l):
      cost_l.append(self.cost(x, y_l[i] ) )
    cost_m = np.array(cost_l).reshape(X.shape)
    
    fig = plot.gcf()
    # ax = plot.axes(projection='3d')
    ax = plot.gca()
    # ax.plot_surface(x_l, y_l, cost_l, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    # ax.scatter(x_l, y_l, cost_l, c=cost_l, cmap='viridis', lw=0.5)
    c = ax.pcolormesh(X, Y, cost_m, cmap='Reds')
    fig.colorbar(c, ax=ax)
    
    ax.set_title('n= {}, k= {}'.format(self.n, self.k) )
    ax.set_xlabel(r'$\lambda_a$', fontsize=14)
    ax.set_ylabel(r'$\lambda_b$', fontsize=14)
    
    fig = plot.gcf()
    fig.set_size_inches(5, 5)
    plot.savefig('plot_cost_2d_n{}_k{}.png'.format(self.n, self.k) ) # , bbox_extra_artists=[ax], bbox_inches='tight'
    fig.clear()
    log(INFO, "done.")
  
  def plot_all_2d(self, popmodel):
    fig, axs = plot.subplots(1, 2, sharey=True)
    fontsize = 18
    
    ### Service cap and Popularity heatmap
    ax = axs[0]
    plot.sca(ax)
    # Service cap
    point_l = self.cap_boundary_point_l()
    point_l.append((0, 0))
    points_inrows = np.array(point_l).reshape((len(point_l), self.k))
    hull = scipy.spatial.ConvexHull(points_inrows)
    # label='Capacity boundary'
    for simplex in hull.simplices:
      plot.plot(points_inrows[simplex, 0], points_inrows[simplex, 1], c=NICE_ORANGE, marker='o', ls='-', lw=3)
    plot.fill(points_inrows[simplex, 0], points_inrows[simplex, 1], c=NICE_ORANGE, alpha=0.5)
    # Popularity heatmap
    [xmax, ymax] = popmodel.max_l
    X, Y = np.mgrid[0:xmax:100j, 0:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel() ] )
    Z = np.reshape(popmodel.kernel(positions).T, X.shape)
    # label='Popularity heatmap', 
    plot.imshow(np.rot90(Z), cmap=plot.cm.gist_earth_r, extent=[0, xmax, 0, ymax] )
    # plot.plot(values[0, :], values[1, :], 'k.', markersize=2)
    # plot.legend()
    plot.xlim([0, xmax] )
    plot.ylim([0, ymax] )
    plot.xlabel('a', fontsize=fontsize)
    plot.ylabel('b', fontsize=fontsize)
    
    covered_mass = self.integrate_overcaphyperlane(popmodel.joint_pdf)
    plot.text(0.7, 0.85, 'Covered mass= {}'.format(covered_mass),
      horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
    
    ### Cost map
    ax = axs[1]
    plot.sca(ax)
    
    x_l, y_l = X.ravel(), Y.ravel()
    cost_l = []
    for i, x in enumerate(x_l):
      cost_l.append(self.cost(x, y_l[i] ) )
    cost_m = np.array(cost_l).reshape(X.shape)
    c = ax.pcolormesh(X, Y, cost_m, cmap='Reds')
    plot.gcf().colorbar(c, ax=ax)
    plot.xlim([0, xmax] )
    plot.ylim([0, ymax] )
    plot.xlabel('a', fontsize=fontsize)
    plot.ylabel('b', fontsize=fontsize)
    
    EC = self.moment_cost(popmodel.joint_pdf, i=1)
    plot.text(0.7, 0.85, 'E[Cost]= {}'.format(EC),
      horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
    
    plot.suptitle('n= {}, k= {}'.format(self.n, self.k), fontsize=fontsize)
    fig.set_size_inches(2*5, 5)
    plot.savefig('plot_all_2d_n{}_k{}.png'.format(self.n, self.k), bbox_inches='tight')
    fig.clear()
    log(INFO, "done;", n=self.n, k=self.k)

if __name__ == "__main__":
  # G = mds_conf_matrix(5, k=2)
  G = custom_conf_matrix(4, k=2)
  cf = ConfInspector(G)
  print("cf.to_sysrepr= {}".format(cf.to_sysrepr() ) )
  
  # r, M, C = cf.r_M_C()
  # cf.cap_boundary_point_l()
  cf.plot_servcap_2d()
  # cf.cap_hyperplane()
  # cf.integrate_overcaphyperlane(lambda x, y: 1)
  
  # cf.plot_cost_2d()
