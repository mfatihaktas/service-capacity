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
  
  def is_stable(self, d_l):
    x = cvxpy.Variable(shape=(self.l, 1), name='x')
    
    # obj = cvxpy.Maximize(np.ones((1, self.l))*x)
    obj = cvxpy.Maximize(0)
    constraints = [self.M*x <= self.C, x >= 0, self.T*x == np.array(d_l).reshape((self.k, 1)) ]
    prob = cvxpy.Problem(obj, constraints)
    try:
      prob.solve()
    except cvxpy.SolverError:
      prob.solve(solver='SCS')
    
    # blog(x_val=x.value)
    return prob.status == 'optimal'
  
  def sim_frac_stable(self, cum_demand, nsamples=400):
    nstable = 0
    for i in range(nsamples):
      rand_l = sorted(np.random.uniform(size=(self.k-1, 1) ) )
      d_l = np.array([rand_l[0]] + \
        [rand_l[i+1] - rand_l[i] for i in range(len(rand_l)-1) ] + \
        [1 - rand_l[-1]] ) * cum_demand
      stable = self.is_stable(d_l)
      # blog(d_l=d_l, stable=stable)
      
      nstable += int(stable)
    return nstable/nsamples

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

def BucketConfInspector_regularbalanced(choice, k, m, C):
  obj__bucket_l_m = {obj: [] for obj in range(k) }
  for obj, bucket_l in obj__bucket_l_m.items():
    bucket = obj % m
    bucket_l.extend([(bucket+i) % m for i in range(choice) ] )
  return BucketConfInspector(m, C, obj__bucket_l_m)

def plot_Pr_robust_wchoice():
  # k, m, C = 4, 4, 5
  # k, m, C = 10, 10, 5
  k, m, C = 100, 100, 5
  
  def plot_(choice):
    print("choice= {}".format(choice) )
    bci = BucketConfInspector_regularbalanced(choice, k, m, C)
    E_l, Pr_robust_l = [], []
    for E in np.linspace(C, (m+1)*C, 20):
      E_l.append(E)
      Pr_robust = bci.sim_frac_stable(cum_demand=E)
      print("E= {}, Pr_robust= {}".format(E, Pr_robust) )
      Pr_robust_l.append(Pr_robust)
      if Pr_robust < 0.01:
        break
    plot.plot(E_l, Pr_robust_l, label='{}-choice'.format(choice), c=next(dark_color), marker='o', ls=':', lw=2)
  
  # plot_(choice=1)
  # plot_(choice=2)
  # plot_(choice=3)
  
  # for c in range(1, m+1):
  for c in range(1, 5):
    plot_(choice=c)
  
  plot_(choice=10)
  plot_(choice=25)
  plot_(choice=50)
  plot_(choice=100)
  
  plot.legend(loc='lower left')
  plot.ylim([0, 1] )
  plot.title('k= {}, m= {}, C= {}'.format(k, m, C) )
  plot.xlabel('E', fontsize=14)
  plot.ylabel('Pr{robust}', fontsize=14)
  plot.savefig('plot_Pr_robust_wchoice_k{}_m{}_C{}.png'.format(k, m, C), bbox_inches='tight')
  log(INFO, "done; m= {}, C= {}".format(m, C) )

def checking_Conjecture_following_Godfrey_claim():
  # Claim: as soon as the number of choice is \Omega(log(m)) in a regular balanced setting,
  # throwing m balls into m urns will result in O(1) maximum load w.h.p.
  # [Godfrey, Balls and Bins with Structure: Balanced Allocations on Hypergraphs]
  # (Our) Conjecture: Pouring E amount of water over m buckets set up with a regular balanced setting 
  # with log(m) choice, maximum load will be E/m + O(1) w.h.p.
  k = 100
  
  def test_conjecture(m, E):
    choice = math.ceil(math.log2(m))
    C = 2*E/m # + math.sqrt(E*math.log2(m)/m) + 2
    bci = BucketConfInspector_regularbalanced(choice, k, m, C)
    Pr_robust = bci.sim_frac_stable(E)
    print("m= {}, E= {}, C= {}, choice= {}, Pr_robust= {}".format(m, E, C, choice, Pr_robust) )
  
  def test(m):
    # for i in range(8):
    for i in range(7, 15):
      test_conjecture(m, E=2**i*100)
  
  # test(m=10)
  # test(m=100)
  test(m=1000)
  
  log(INFO, "done; k= {}".format(k) )

if __name__ == "__main__":
  # m, C = 2, 10
  # obj__bucket_l_m = {0: [0], 1: [0], 2: [1], 3: [1] }
  # bci = BucketConfInspector(m, C, obj__bucket_l_m)
  # bci.sim_frac_stable(cum_demand=15)
  # print("bci= {}".format(bci) )
  
  # plot_Pr_robust_sim_vs_model()
  # plot_Pr_robust_wchoice()
  checking_Conjecture_following_Godfrey_claim()
