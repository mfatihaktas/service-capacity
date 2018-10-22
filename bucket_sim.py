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
    for obj in range(self.k):
      for bucket in obj__bucket_l_m[obj]:
        self.M[bucket, obj] = 1
  
  def __repr__(self):
    return 'BucketConfInspector[m= {}, C= {}, \n\tobj__bucket_l_m= {}, \n\tM= {}, \n\tT= {}]'.format(self.m, self.C, self.obj__bucket_l_m, self.M, self.T)
  
  def is_stable(self, d_l):
    x = cvxpy.Variable(shape=(self.l, 1), name='x')
    
    # obj = cvxpy.Maximize(np.ones((1, self.l))*x)
    obj = cvxpy.Maximize(0)
    constraints = [self.M*x <= self.C, x >= 0, self.T*x == np.array(d_l).reshape((self.k, 1)) ]
    prob = cvxpy.Problem(obj, constraints)
    prob.solve()
    # blog(x_val=x.value)
    return prob.status == 'optimal'
  
  def sim_frac_stable(self, cum_demand, nsamples=400):
    nstable = 0
    for i in range(nsamples):
      rand_l = np.random.uniform(size=(self.k-1, 1) )
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

if __name__ == "__main__":
  # m, C = 2, 10
  # obj__bucket_l_m = {0: [0], 1: [0], 2: [1], 3: [1] }
  # bci = BucketConfInspector(m, C, obj__bucket_l_m)
  # bci.sim_frac_stable(cum_demand=15)
  # print("bci= {}".format(bci) )
  
  plot_Pr_robust_sim_vs_model()
