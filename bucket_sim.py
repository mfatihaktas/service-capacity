import cvxpy
import numpy as np

from log_utils import *
from bucket_model import *

class BucketConfInspector(object):
  def __init__(self, m, C, obj__bucket_l_m):
    self.m = m # number of buckets
    self.C = C # capacity of each bucket
    # self.S = S # total demand
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
    M = np.zeros((self.k, self.k))
    for i in range(self.k):
      M[i, i] = E
    hull = scipy.spatial.ConvexHull(M)
    simplex_area = hull.area
    log(INFO, "simplex_area= {}".format(simplex_area) )
    
    for ar1 in np.arange(0, E, 0.01):
      ar_l = [ar1, E - ar1]
      ar_l.append((self.k-2)*[0] )
      if self.is_stable(ar_l):
        break
    
    M_ = np.zeros((self.k, self.k*(self.k-1) ))
    for i in range(self.k):
      for j in range(self.k):
        if j == i:
          continue
        c = i*(self.k-1)
        M_[i, c] = ar1
        M_[j, c] = E - ar1
    hull_ = scipy.spatial.ConvexHull(M_)
    robust_area = hull_.area
    log(INFO, "robust_area= {}".format(robust_area) )

def plot_Pr_robust_sim_vs_model():
  m, C = 2, 10
  # obj__bucket_l_m = {0: [0], 1: [0], 2: [1], 3: [1] }
  obj__bucket_l_m = {0: [0], 1: [1] }
  bci = BucketConfInspector(m, C, obj__bucket_l_m)
  print("bci= {}".format(bci) )
  
  E_l = []
  Pr_robust_sim_l, Pr_robust_model_l, Pr_robust_approx_l = [], [], []
  for E in np.linspace(C, 2*C, 10):
    E_l.append(E)
    sim = bci.sim_frac_stable(cum_demand=E)
    model = Pr_no_overflow_cont(E, m, C)
    approx = Pr_no_overflow_cont_approx(E, m, C)
    print("E= {}, sim= {}, model= {}, approx= {}".format(E, sim, model, approx) )
    
    Pr_robust_sim_l.append(sim)
    Pr_robust_model_l.append(model)
    Pr_robust_approx_l.append(approx)
  
  plot.plot(E_l, Pr_robust_sim_l, label='sim', c=NICE_RED, marker='o', ls=':', lw=2)
  plot.plot(E_l, Pr_robust_model_l, label='model', c=NICE_BLUE, marker='o', ls=':', lw=2)
  plot.plot(E_l, Pr_robust_approx_l, label='approx', c=NICE_ORANGE, marker='o', ls=':', lw=2)
  plot.legend()
  plot.ylim([0, 1] )
  plot.title('m= {}, C= {}'.format(m, C) )
  plot.xlabel('E', fontsize=14)
  plot.ylabel('Pr{robust}', fontsize=14)
  plot.savefig('plot_Pr_robust_sim_vs_model.png', bbox_inches='tight')
  log(INFO, "done; m= {}, C= {}".format(m, C) )

def BucketConfInspector_regularbalanced(k, m, C, choice):
  obj__bucket_l_m = {obj: [] for obj in range(k) }
  for obj, bucket_l in obj__bucket_l_m.items():
    bucket = obj % m
    bucket_l.extend([(bucket+i) % m for i in range(choice) ] )
  return BucketConfInspector(m, C, obj__bucket_l_m)

def plot_Pr_robust_wchoice():
  # k, m, C = 4, 4, 5
  # k, m, C = 10, 10, 5
  k, m, C = 50, 50, 5
  
  def plot_(choice):
    log(INFO, "choice= {}".format(choice) )
    bci = BucketConfInspector_regularbalanced(k, m, C, choice)
    E_l, sim_Pr_robust_l = [], []
    # Pr_robust_l, Pr_robust_w_naivesplit_l = [], []
    for E in np.linspace(C, (m+1)*C, 10):
      print(">> E= {}".format(E) )
      E_l.append(E)
      sim_Pr_robust = bci.sim_frac_stable(cum_demand=E)
      blog(sim_Pr_robust=sim_Pr_robust)
      sim_Pr_robust_l.append(sim_Pr_robust)
      # Pr_robust_w_naivesplit = bci.sim_frac_stable(cum_demand=E, w_naivesplit=True)
      # blog(Pr_robust_w_naivesplit=Pr_robust_w_naivesplit)
      
      Pr_robust = bci.frac_stable(E)
      blog(Pr_robust=Pr_robust)
      
      # Pr_robust = Pr_no_overflow_wchoice_cont(E, m, C, choice)
      # blog(Pr_robust=Pr_robust)
      # Pr_robust_l.append(Pr_robust)
      # Pr_robust_w_naivesplit_l.append(Pr_robust_w_naivesplit)
      
      if sim_Pr_robust < 0.1:
        break
    plot.plot(E_l, sim_Pr_robust_l, label='Sim, d={}'.format(choice), c=next(dark_color_c), marker=next(marker_c), ls=':', lw=3, mew=0.5, ms=7)
    # plot.plot(E_l, Pr_robust_l, label='d={}'.format(choice), c=next(dark_color_c), marker=next(marker_c), ls=':', lw=3, mew=0.5, ms=7)
    # plot.plot(E_l, Pr_robust_w_naivesplit_l, label='d={}, w/ naive split'.format(choice), c=next(dark_color_c), marker=next(marker_c), ls=':', lw=3, mew=0, ms=7)
  
  plot_(choice=1)
  # plot_(choice=2)
  # plot_(choice=3)
  # plot_(choice=4)
  # plot_(choice=8)
  
  # for c in range(1, m+1):
  # plot_(choice=1)
  # plot_(choice=2)
  
  # plot_(choice=1)
  # for c in range(1, 8):
  #   plot_(choice=c)
  # plot_(choice=10)
  # plot_(choice=m)
  
  fontsize = 18
  plot.legend(loc='lower left', framealpha=0.5, fontsize=fontsize)
  plot.ylim([0, 1] )
  plot.title('k= {}, m= {}, C= {}'.format(k, m, C), fontsize=fontsize)
  plot.xlabel(r'$\Sigma$', fontsize=fontsize)
  plot.ylabel('Pr{$\Sigma$-robustness}', fontsize=fontsize)
  # plot.gcf().set_size_inches(8, 7)
  plot.savefig('plot_Pr_robust_wchoice_k{}_m{}_C{}.png'.format(k, m, C), bbox_inches='tight')
  # plot.gcf().clear()
  log(INFO, "done; m= {}, C= {}".format(m, C) )

def checking_Conjecture_following_Godfrey_claim():
  # Claim: as soon as the number of choice is \Omega(log(m)) in a regular balanced setting,
  # throwing m balls into m urns will result in O(1) maximum load w.h.p.
  # [Godfrey, Balls and Bins with Structure: Balanced Allocations on Hypergraphs]
  # (Our) Conjecture: Pouring E amount of water over m buckets set up with a regular balanced setting 
  # with log(m) choice, maximum load will be E/m + O(1) w.h.p.
  k = 400
  log(INFO, "k= {}".format(k) )
  
  def test_conjecture(m, E):
    choice = math.ceil(math.log(m))
    C = E/m*math.log(2*math.log(math.log(m)) + math.log(m) ) # E/m*math.sqrt(2*math.log(math.log(m)) + math.log(m) ) # + 1 # E/m + math.sqrt(E/m) # 2*E/m
    bci = BucketConfInspector_regularbalanced(k, m, C, choice)
    Pr_robust = bci.sim_frac_stable(E)
    print("m= {}, E= {}, C= {}, choice= {}, Pr_robust= {}".format(m, E, C, choice, Pr_robust) )
  
  def test(m):
    # for i in range(8):
    # for i in range(7, 15):
    # for i in range(11, 15):
    for i in range(12):
      test_conjecture(m, E=2**i*100)
  
  # test(m=10)
  test(m=50)
  # test(m=100)
  
  log(INFO, "done; k= {}".format(k) )

if __name__ == "__main__":
  # m, C = 2, 10
  # obj__bucket_l_m = {0: [0], 1: [0], 2: [1], 3: [1] }
  # bci = BucketConfInspector(m, C, obj__bucket_l_m)
  # bci.sim_frac_stable(cum_demand=15)
  # print("bci= {}".format(bci) )
  
  # plot_Pr_robust_sim_vs_model()
  plot_Pr_robust_wchoice()
  # checking_Conjecture_following_Godfrey_claim()
