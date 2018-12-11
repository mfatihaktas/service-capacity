# import sys
# sys.path.append('/home/mfa51/.local/lib/python3.5/site-packages/')
# sys.path = ['/home/mfa51/.local/lib/python3.5/site-packages/']
# sys.path.remove('/opt/sw/packages/gcc-4.8/python/3.5.2/lib/python3.5/site-packages')

import cvxpy
import numpy as np
from scipy import spatial
from scipy.spatial import HalfspaceIntersection

from bucket_model import *

class BucketConfInspector(object):
  def __init__(self, m, C, obj__bucket_l_m):
    self.m = m # number of buckets
    self.C = C # capacity of each bucket
    self.obj__bucket_l_m = obj__bucket_l_m
    
    self.k = len(obj__bucket_l_m)
    self.l = sum([len(obj__bucket_l_m[obj] ) for obj in range(self.k) ] )
    ## T
    self.T = np.zeros((self.k, self.l))
    i = 0
    for obj, bucket_l in obj__bucket_l_m.items():
      j = i + len(bucket_l)
      self.T[obj, i:j] = 1
      i = j
    
    ## M
    self.M = np.zeros((m, self.l))
    obj = 0
    i = 0
    while obj < self.k:
      for bucket in obj__bucket_l_m[obj]:
        self.M[bucket, i] = 1
        i += 1
      obj += 1
  
  def __repr__(self):
    return 'BucketConfInspector[m= {}, C= {}, \nobj__bucket_l_m= {}, \nM=\n{}, \nT=\n{}]'.format(self.m, self.C, self.obj__bucket_l_m, self.M, self.T)
  
  def is_stable(self, ar_l):
    x = cvxpy.Variable(shape=(self.l, 1), name='x')
    
    # obj = cvxpy.Maximize(np.ones((1, self.l))*x)
    obj = cvxpy.Maximize(0)
    constraints = [self.M*x <= self.C, x >= 0, self.T*x == np.array(ar_l).reshape((self.k, 1)) ]
    prob = cvxpy.Problem(obj, constraints)
    try:
      prob.solve()
    except cvxpy.SolverError:
      prob.solve(solver='SCS')
    
    # blog(x_val=x.value)
    return prob.status == 'optimal'
  
  def is_stable_w_naivesplit(self, ar_l):
    '''
     Rather than searching through all distribution vector x's, we restrict ourselves here on
     the naive splitting of demand for object-i uniformly across all its buckets.
    Result: Naive split cannot capture the benefit of multiple choice;
     Pr{robustness} with naive split is much lower than actual.
    '''
    x = np.zeros((self.l, 1) ) # self.l*[0]
    i = 0
    for obj, bucket_l in self.obj__bucket_l_m.items():
      j = i + len(bucket_l)
      x[i:j, 0] = ar_l[obj]/len(bucket_l)
      i = j
    # y = np.matmul(self.M, x)
    # z = np.less_equal(y, np.array(self.m*[self.C] ).reshape((self.m, 1)) )
    # log(INFO, "", y=y, z=z)
    return np.all(np.less_equal(np.matmul(self.M, x), np.array(self.m*[self.C] ).reshape((self.m, 1)) ) )
  
  def sim_frac_stable(self, cum_demand, nsamples=10**3, w_naivesplit=False):
    nstable = 0
    for i in range(nsamples):
      rand_l = sorted(np.random.uniform(size=(self.k-1, 1) ) )
      ar_l = np.array([rand_l[0]] + \
        [rand_l[i+1] - rand_l[i] for i in range(len(rand_l)-1) ] + \
        [1 - rand_l[-1]] ) * cum_demand
      stable = self.is_stable(ar_l) # if not w_naivesplit else self.is_stable_w_naivesplit(ar_l)
      # blog(ar_l=ar_l, stable=stable)
      
      nstable += int(stable)
    return nstable/nsamples
  
  def frac_stable(self, E):
    # np.set_printoptions(threshold=np.nan)
    # M = np.zeros((self.m + 1, self.m))
    # for i in range(self.m):
    #   M[i, i] = E
    # log(INFO, "M=\n{}".format(M) )
    # hull = scipy.spatial.ConvexHull(M)
    # simplex_area = hull.area
    # # log(INFO, "simplex_area= {}".format(simplex_area) )
    
    ar_u = 0
    for ar in np.arange(E/self.m, E+0.01, 0.01):
      # ar_l = [ar]
      # ar_l.extend((self.m-1)*[(E - ar)/(self.m-1) ] )
      ar_l = [ar, 0]
      ar_l.extend((self.m-2)*[(E - ar)/(self.m-2) ] )
      if not self.is_stable(ar_l):
        break
      ar_u = ar
    log(INFO, "ar_u= {}".format(ar_u) )
    
    # E_ = ar_u if ar_u < E/2 else E
    # M_ = np.zeros((self.m*(self.m-1), self.m))
    # r = 0
    # for i in range(self.m):
    #   for j in range(self.m):
    #     if j == i:
    #       continue
    #     # c = i*(self.m-1) + j
    #     M_[r, i] = ar_u
    #     M_[r, j] = E_ - ar_u
    #     r += 1
    # log(INFO, "M_=\n{}".format(M_) )
    # hull_ = scipy.spatial.ConvexHull(M_)
    # robust_area = hull_.area
    # # log(INFO, "robust_area= {}".format(robust_area) )
    # return robust_area/simplex_area
    
    return Pr_max_uniform_spacing_leq_x(self.m, ar_u/E)
  
  def plot_cap(self, d):
    if self.k != 3:
      log(ERROR, "implemented only for m= 3;", m=self.m)
      return
    np.set_printoptions(threshold=np.nan)
    
    halfspaces = np.zeros((self.m+self.l, self.l+1))
    for r in range(self.m):
      halfspaces[r, -1] = -self.C
    halfspaces[:self.m, :-1] = self.M
    for r in range(self.m, self.m+self.l):
      halfspaces[r, r-self.m] = -1
    log(INFO, "halfspaces= \n{}".format(halfspaces) )
    
    feasible_point = np.array([self.C/self.l]*self.l)
    # feasible_point = np.array([0.01]*self.k)
    blog(feasible_point=feasible_point)
    hs = HalfspaceIntersection(halfspaces, feasible_point)
    
    fig = plot.figure()
    ax = plot.axes(projection='3d')
    
    # print("hs.intersections= \n{}".format(hs.intersections) )
    x_l, y_l, z_l = [], [], []
    for x in hs.intersections:
      # print("x= {}".format(x) )
      y = np.matmul(self.T, x)
      x_l.append(y[0] )
      y_l.append(y[1] )
      z_l.append(y[2] )
    # '''
    # x_l, y_l, z_l = zip(*hs.intersections)
    ax.scatter3D(x_l, y_l, z_l, label='Vertices', c='red')
    
    intersections = hs.intersections
    hull = scipy.spatial.ConvexHull(intersections)
    # for simplex in hull.simplices:
    #   ax.plot3D(intersections[simplex, 0], intersections[simplex, 1], intersections[simplex, 2], c=NICE_BLUE, marker='o')
    
    cent = np.mean(intersections, 0)
    pts = []
    for pt in points[hull.simplices]:
        pts.append(pt[0].tolist())
        pts.append(pt[1].tolist())
    
    pts.sort(key=lambda p: np.arctan2(p[1] - cent[1],
                                    p[0] - cent[0]))
    pts = pts[0::2]  # Deleting duplicates
    pts.insert(len(pts), pts[0])
    k = 1.1
    color = 'green'
    poly = Polygon(k*(np.array(pts)- cent) + cent,
                   facecolor=color, alpha=0.2)
    poly.set_capstyle('round')
    plt.gca().add_patch(poly)
    
    plot.legend()
    fontsize = 18
    ax.set_xlabel(r'$\lambda_1$', fontsize=fontsize)
    ax.set_xlim(xmin=0)
    ax.set_ylabel(r'$\lambda_2$', fontsize=fontsize)
    ax.set_ylim(ymin=0)
    ax.set_zlabel(r'$\lambda_3$', fontsize=fontsize)
    ax.set_zlim(zmin=0)
    plot.title(r'$k= {}$, $m= {}$, $C= {}$, $d= {}$'.format(self.k, self.m, self.C, d), fontsize=fontsize)
    ax.view_init(20, 30)
    plot.savefig('plot_cap_C{}_d{}.png'.format(self.C, d), bbox_inches='tight')
    fig.clear()
    # '''
    log(INFO, "done.")

def BucketConfInspector_regularbalanced(k, m, C, choice):
  obj__bucket_l_m = {obj: [] for obj in range(k) }
  for obj, bucket_l in obj__bucket_l_m.items():
    bucket = obj % m
    bucket_l.extend([(bucket+i) % m for i in range(choice) ] )
  return BucketConfInspector(m, C, obj__bucket_l_m)

if __name__ == "__main__":
  # blog(np_version=np.__version__)
  
  k, m, C = 3, 3, 5
  d = 2
  bci = BucketConfInspector_regularbalanced(k, m, C, d)
  bci.plot_cap(d)
  
