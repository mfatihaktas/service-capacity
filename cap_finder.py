import math, scipy, cvxpy, pprint, itertools, string
import numpy as np
from scipy import spatial
# from fillplots import plot_regions

from mpl_toolkits import mplot3d

from plot_utils import *
from conf_gen import *
from log_utils import *
from popularity import *

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
    self.cost_ofgoingtocloud = 5
    
    self.k = G.shape[0]
    self.n = G.shape[1]
    
    self.sys__repgroup_l_l = self.get_sys__repgroup_l_l()
    blog(ConfInspector=self, sys__repgroup_l_l=self.sys__repgroup_l_l)
    
    self.point_l = self.cap_boundary_point_l()
    self.points_inrows = np.array(self.point_l).reshape((len(self.point_l), self.k))
    self.hull = scipy.spatial.ConvexHull(self.points_inrows)
    
    self.A, self.b = self.cap_hyperplane()
    
  def __repr__(self):
    return 'ConfInspector[k= {}, n= {}, G=\n{}]'.format(self.k, self.n, self.G)
  
  def to_sysrepr(self):
    sym_l = string.ascii_lowercase[:self.k]
    node_l = []
    for c in range(self.n):
      l = []
      for r in range(self.k):
        if self.G[r, c] != 0:
          num = int(self.G[r, c] )
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
          # A = np.array(l).reshape((self.k, len(l) ))
          A = np.column_stack(l)
          # print("\n")
          x, residuals, _, _ = np.linalg.lstsq(A, y)
          residuals = y - np.dot(A, x)
          # log(INFO, "", A=A, y=y, x=x, residuals=residuals)
          if np.sum(np.absolute(residuals) ) < 0.0001: # residuals.size > 0 and 
            repgroup_l.append(subset)
      sys__repgroup_l_l[s] = repgroup_l
    # blog(sys__repgroup_l_l=sys__repgroup_l_l)
    return sys__repgroup_l_l
  
  def r_M_C(self):
    ## C
    x = []
    for rg_l in self.sys__repgroup_l_l:
      x.extend(rg_l)
    
    r = len(x)
    C = np.zeros((self.k, r))
    i = 0
    for s, rg_l in enumerate(self.sys__repgroup_l_l):
      j = i + len(rg_l)
      C[s, i:j] = 1
      i = j
    ## M
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
      # print("w= {}, p= {}".format(w, p) )
      p_l.append(p)
      ## Add projections on the axes as well to make sure point_l form a convex hull
      for i in range(self.k):
        p_ = list(p)
        p_[i] = 0
        p_l.append(p_)
      
      counter += 1
    p_l.append((0,)*self.k)
    return p_l
  
  def cap_hyperplane(self):
    # plot_points(self.point_l, fname='points')
    
    # `hull` := {x| Ax <= b}
    m, n = self.hull.equations.shape
    A = np.mat(self.hull.equations[:, 0:n-1] )
    b = np.mat(-self.hull.equations[:, n-1] ).T
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
  
  def plot_servcap_3d(self):
    fig = plot.figure()
    ax = plot.axes(projection='3d')
    
    for simplex in self.hull.simplices:
      ax.scatter3D(self.points_inrows[simplex, 0], self.points_inrows[simplex, 1], self.points_inrows[simplex, 2], c=NICE_BLUE, marker='o')
      # print("extreme vertices=\n{}".format(self.points_inrows[simplex, :] ) )
    # ax.fill_between(self.points_inrows[self.hull.vertices, 0], self.points_inrows[self.hull.vertices, 1], self.points_inrows[self.hull.vertices, 2], c=NICE_BLUE, alpha=0.3)
    
    ax.set_xlabel(r'$\lambda_a$', fontsize=20)
    ax.set_ylabel(r'$\lambda_b$', fontsize=20)
    ax.set_zlabel(r'$\lambda_c$', fontsize=20)
    plot.title('{}'.format(self.to_sysrepr() ) )
    ax.view_init(30, -105)
    plot.savefig('plot_servcap_3d_n{}_k{}.png'.format(self.n, self.k), bbox_inches='tight')
    fig.clear()
  
  def plot_servcap_2d(self, popmodel=None):
    ## Service cap
    # '''
    for simplex in self.hull.simplices:
      # print("x_l= {}, y_l= {}".format(self.points_inrows[simplex, 0], self.points_inrows[simplex, 1] ) )
      plot.plot(self.points_inrows[simplex, 0], self.points_inrows[simplex, 1], c=NICE_BLUE, marker='o', ls='-', lw=3)
    # plot.plot(self.points_inrows[self.hull.vertices, 0], self.points_inrows[self.hull.vertices, 1], 'r--', lw=2)
    # plot.plot(self.points_inrows[self.hull.vertices[0], 0], self.points_inrows[self.hull.vertices[0], 1], 'ro')
    # plot.fill(self.points_inrows[simplex, 0], self.points_inrows[simplex, 1], c=NICE_BLUE, alpha=0.5)
    plot.fill(self.points_inrows[self.hull.vertices, 0], self.points_inrows[self.hull.vertices, 1], c=NICE_BLUE, alpha=0.3)
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
    ## Popularity heatmap
    if popmodel is not None:
      [xmax, ymax] = popmodel.max_l
      X, Y = np.mgrid[0:xmax:200j, 0:ymax:200j]
      positions = np.vstack([X.ravel(), Y.ravel() ] )
      Z = np.reshape(popmodel.kernel(positions).T, X.shape)
      plot.imshow(np.rot90(Z), cmap=plot.cm.gist_earth_r, extent=[0, xmax, 0, ymax] )
      covered_mass = self.integrate_overcaphyperlane(popmodel.joint_pdf)
      plot.text(0.6, 0.85, 'Covered mass= {}'.format(covered_mass),
        horizontalalignment='center', verticalalignment='center', transform = plot.gca().transAxes)
    
    # plot.xlim((0, 2.1))
    # plot.ylim((0, 2.1))
    prettify(plot.gca() )
    # plot.title('n= {}, k= {}'.format(self.n, self.k) )
    plot.title('{}'.format(self.to_sysrepr() ) )
    plot.xlabel(r'$\lambda_a$', fontsize=20)
    plot.ylabel(r'$\lambda_b$', fontsize=20)
    fig = plot.gcf()
    fig.set_size_inches(3, 3)
    plot.savefig('plot_servcap_2d_n{}_k{}.png'.format(self.n, self.k), bbox_inches='tight')
    fig.clear()
    log(INFO, "done.")
  
  def util(self, *args):
    x = np.array(args).reshape((self.k, 1))
    if np.any(np.greater(np.dot(self.A, x), self.b) ):
      return 1
    else:
      return np.sum(x)/self.n
    
  def cost(self, *args):
    def cost_insidecapregion(x):
      total_cost = 0
      for sys, rg_l in enumerate(self.sys__repgroup_l_l):
        cap_demand = x[sys]
        i, cost = 0, 0
        while cap_demand - i > 0.001 and i < len(rg_l):
          cost += min(cap_demand - i, 1)*len(rg_l[i] )
          i += 1
        
        if cap_demand - i > 0.001:
          log(ERROR, "should have been supplied!", x=x)
          return
        total_cost += cost # - cap_demand
      return total_cost
    
    x = np.array(args).reshape((self.k, 1))
    if np.any(np.greater(np.dot(self.A, x), self.b) ):
      # log(ERROR, "outside of capacity region;", x=x)
      
      # https://stackoverflow.com/questions/42248202/find-the-projection-of-a-point-on-the-convex-hull-with-scipy
      def proj_mindistance(pt1, pt2, p):
        # blog(x=x, pt1=pt1, pt2=pt2)
        """ returns the projection of point p (and the distance) on the closest edge formed by the two points pt1 and pt2"""
        l = np.sum((pt2-pt1)**2) # compute the squared distance between the 2 vertices
        # blog(l=l, dot=np.dot(p-pt1, pt2-pt1)[0] )
        t = np.max([0., np.min([1., np.dot(p-pt1, pt2-pt1)[0]/l] ) ] )
        proj = pt1 + t*(pt2-pt1)
        return proj, np.sum((proj-p)**2) # project the point
      
      proj, mindistance = None, float('Inf')
      for i in range(len(self.hull.vertices)):
        p, m = proj_mindistance(self.points_inrows[self.hull.vertices[i] ], self.points_inrows[self.hull.vertices[(i+1) % len(self.hull.vertices) ] ], x)
        if m < mindistance:
          mindistance = m
          proj = p
      if proj is None:
        log(ERROR, "proj= {}".format(proj) )
        return 0
      # blog(proj=proj)
      return cost_insidecapregion(proj) + self.cost_ofgoingtocloud*np.sum(x.T[0] - proj)
    else:
      return cost_insidecapregion(x.T[0] )
  
  def moment_cost(self, i, popmodel):
    func = lambda *args: self.cost(*args)**i * popmodel.joint_pdf(*args)
    return popmodel.integrate_overpopmodel(func)
  
  def plot_cost_2d(self, popmodel=None):
    # point_l = self.cap_boundary_point_l()
    point_l = popmodel.cap_l_
    x_max = max([p[0] for p in point_l] )
    y_max = max([p[1] for p in point_l] )
    X, Y = np.mgrid[0:x_max:100j, 0:y_max:100j]
    # blog(X_shape=X.shape, Y_shape=Y.shape)
    
    x_l, y_l = X.ravel(), Y.ravel()
    cost_l, util_l = [], []
    for i, x in enumerate(x_l):
      cost = self.cost(x, y_l[i] )
      if popmodel is not None:
        cost *= popmodel.joint_pdf(x, y_l[i] )
      cost_l.append(cost)
    cost_m = np.array(cost_l).reshape(X.shape)
    
    fig = plot.gcf()
    # ax = plot.axes(projection='3d')
    ax = plot.gca()
    # ax.plot_surface(x_l, y_l, cost_l, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    # ax.scatter(x_l, y_l, cost_l, c=cost_l, cmap='viridis', lw=0.5)
    c = ax.pcolormesh(X, Y, cost_m, cmap='Reds')
    fig.colorbar(c, ax=ax)
    
    if popmodel is not None:
      covered_mass = self.integrate_overcaphyperlane(popmodel.joint_pdf)
      plot.text(0.7, 0.9, 'Covered mass= {}'.format(covered_mass),
        horizontalalignment='center', verticalalignment='center', transform = plot.gca().transAxes)
      
      ## Service cap
      for simplex in self.hull.simplices:
        plot.plot(self.points_inrows[simplex, 0], self.points_inrows[simplex, 1], c=NICE_BLUE, marker='o', ls='-', lw=3)
      
      EC = self.moment_cost(1, popmodel)
      blog(EC=EC)
      plot.text(0.7, 0.8, 'E[Cost]= {}'.format(EC),
        horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
      
      Eutil = popmodel.integrate_overpopmodel(lambda x, y: self.util(x, y) * popmodel.joint_pdf(x, y) )
      blog(Eutil=Eutil)
      plot.text(0.7, 0.7, 'E[Utilization]= {}'.format(round(Eutil, 2) ),
        horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    
    prettify(plot.gca() )
    ax.set_title('{}'.format(self.to_sysrepr() ) )
    ax.set_xlabel(r'$\lambda_a$', fontsize=20)
    ax.set_ylabel(r'$\lambda_b$', fontsize=20)
    
    fig = plot.gcf()
    fig.set_size_inches(5, 4)
    plot.savefig('plot_cost_2d_n{}_k{}.png'.format(self.n, self.k), bbox_inches='tight') # , bbox_extra_artists=[ax], 
    fig.clear()
    log(INFO, "done.")
  
  def plot_all_2d(self, popmodel):
    fig, axs = plot.subplots(1, 2, sharey=True)
    fontsize = 20
    
    ### Service cap and Popularity heatmap
    ax = axs[0]
    plot.sca(ax)
    # Service cap
    # label='Capacity boundary'
    for simplex in self.hull.simplices:
      plot.plot(self.points_inrows[simplex, 0], self.points_inrows[simplex, 1], c=NICE_BLUE, marker='o', ls='-', lw=3)
    plot.fill(self.points_inrows[simplex, 0], self.points_inrows[simplex, 1], c=NICE_BLUE, alpha=0.5)
    # Popularity heatmap
    [xmax, ymax] = popmodel.max_l
    X, Y = np.mgrid[0:xmax:200j, 0:ymax:200j]
    positions = np.vstack([X.ravel(), Y.ravel() ] )
    Z = np.reshape(popmodel.kernel(positions).T, X.shape)
    # label='Popularity heatmap', 
    plot.imshow(np.rot90(Z), cmap=plot.cm.gist_earth_r, extent=[0, xmax, 0, ymax] )
    # plot.plot(values[0, :], values[1, :], 'k.', markersize=2)
    # plot.legend()
    prettify(ax)
    plot.xlim([0, xmax] )
    plot.ylim([0, ymax] )
    plot.xlabel(r'$\lambda_a$', fontsize=fontsize)
    plot.ylabel(r'$\lambda_b$', fontsize=fontsize)
    
    covered_mass = self.integrate_overcaphyperlane(popmodel.joint_pdf)
    plot.text(0.7, 0.85, 'Covered mass= {}'.format(covered_mass),
      horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    
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
    prettify(ax)
    plot.xlim([0, xmax] )
    plot.ylim([0, ymax] )
    plot.xlabel(r'$\lambda_a$', fontsize=fontsize)
    plot.ylabel(r'$\lambda_b$', fontsize=fontsize)
    
    EC = self.moment_cost(1, popmodel)
    plot.text(0.7, 0.85, 'E[Cost]= {}'.format(EC),
      horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    
    plot.suptitle('{}'.format(self.to_sysrepr() ) )
    fig.set_size_inches(2*3.5, 3.5)
    plot.savefig('plot_all_2d_n{}_k{}.png'.format(self.n, self.k), bbox_inches='tight')
    fig.clear()
    log(INFO, "done;", n=self.n, k=self.k)

if __name__ == "__main__":
  # G = mds_conf_matrix(4, k=2)
  G = custom_conf_matrix(4, k=2)
  # G = mds_conf_matrix(4, k=3)
  cf = ConfInspector(G)
  print("cf.to_sysrepr= {}".format(cf.to_sysrepr() ) )
  
  # r, M, C = cf.r_M_C()
  # cf.cap_boundary_point_l()
  
  # cf.plot_servcap_2d()
  pm = PopModel_wZipf(k=2, zipf_tailindex_rv=TNormal(1, 2), arrate_rv=TNormal(1.5, 0.4) )
  # cf.plot_servcap_2d(pm)
  cf.plot_cost_2d(pm)
  
  # cf.cap_hyperplane()
  # cf.integrate_overcaphyperlane(lambda x, y: 1)
  
  # cf.plot_servcap_3d()
  