import math, random, numpy, scipy
from scipy.stats import *

class RV(): # Random Variable
  def __init__(self, l_l, u_l):
    self.l_l = l_l
    self.u_l = u_l

class Normal(RV):
  def __init__(self, mu, sigma):
    super().__init__(l_l=-np.inf, u_l=np.inf)
    self.mu = mu
    self.sigma = sigma
    
    self.dist = scipy.stats.norm(mu, sigma)
  
  def __repr__(self):
    return 'Normal[mu= {}, sigma= {}]'.format(self.mu, self.sigma)
  
  def cdf(self, x):
    return self.dist.cdf(x)
  
  def tail(self, x):
    return 1 - self.cdf(x)
  
  def mean(self):
    return self.mu
  
  def sample(self):
    return self.dist.rvs(size=1)[0]

class TNormal(RV):
  def __init__(self, mu, sigma):
    super().__init__(l_l=0, u_l=float('Inf') )
    self.mu = mu
    self.sigma = sigma
    
    lower, upper = 0, mu + 10*sigma
    self.u_l = upper
    self.dist = scipy.stats.truncnorm(
      a=(lower - mu)/sigma, b=(upper - mu)/sigma, loc=mu, scale=sigma)
  
  def __repr__(self):
    # return 'TNormal[mu= {}, sigma= {}]'.format(self.mu, self.sigma)
    # return r'N^+(\mu= {}, \sigma= {})'.format(self.mu, self.sigma)
    return r'N^+({}, {})'.format(self.mu, self.sigma)
  
  def cdf(self, x):
    return self.dist.cdf(x)
  
  def tail(self, x):
    return 1 - self.cdf(x)
  
  def mean(self):
    return self.dist.mean()
  
  def std(self):
    return self.dist.std()
  
  def sample(self):
    return self.dist.rvs(size=1)[0]

class Exp(RV):
  def __init__(self, mu, D=0):
    super().__init__(l_l=D, u_l=float("inf") )
    self.D = D
    self.mu = mu
  
  def __repr__(self):
    if self.D == 0:
      return r'Exp(\mu={})'.format(self.mu)
    return r'{} + Exp(\mu={})'.format(self.D, self.mu)
  
  def tail(self, x):
    if x <= self.l_l:
      return 1
    return math.exp(-self.mu*(x - self.D) )
  
  def cdf(self, x):
    if x <= self.l_l:
      return 0
    return 1 - math.exp(-self.mu*(x - self.D) )
  
  def pdf(self, x):
    if x <= self.l_l:
      return 0
    return self.mu*math.exp(-self.mu*(x - self.D) )
  
  def mean(self):
    return self.D + 1/self.mu
  
  def var(self):
    return 1/self.mu**2
  
  def moment(self, i):
    return moment_ith(i, self)
  
  def laplace(self, s):
    if self.D > 0:
      log(ERROR, "D= {} != 0".format(D) )
    return self.mu/(s + self.mu)
  
  def sample(self):
    return self.D + random.expovariate(self.mu)

class Uniform(RV):
  def __init__(self, lb, ub):
    super().__init__(l_l=lb, u_l=ub)
    
    self.dist = scipy.stats.uniform(loc=lb, scale=ub-lb)
  
  def __repr__(self):
    return 'Uniform[{}, {}]'.format(self.l_l, self.u_l)
  
  def sample(self):
    return self.dist.rvs()

class DUniform(RV):
  def __init__(self, lb, ub):
    super().__init__(l_l=lb, u_l=ub)
    
    self.v = numpy.arange(self.l_l, self.u_l+1)
    w_l = [1 for v in self.v]
    self.p = [w/sum(w_l) for w in w_l]
    self.dist = scipy.stats.rv_discrete(name='duniform', values=(self.v, self.p) )
  
  def __repr__(self):
    return 'DUniform[{}, {}]'.format(self.l_l, self.u_l)
  
  def mean(self):
    return (self.u_l + self.l_l)/2
  
  def pdf(self, x):
    return self.dist.pmf(x)
  
  def cdf(self, x):
    if x < self.l_l:
      return 0
    elif x > self.u_l:
      return 1
    return self.dist.cdf(math.floor(x) )
  
  def tail(self, x):
    return 1 - self.cdf(x)
  
  def moment(self, i):
    return self.dist.moment(i)
  
  def sample(self):
    return self.dist.rvs() # [0]

class BZipf(RV):
  def __init__(self, lb, ub, a=1):
    super().__init__(l_l=lb, u_l=ub)
    self.a = a
    
    self.v = numpy.arange(self.l_l, self.u_l+1) # values
    w_l = [float(v)**(-a) for v in self.v] # self.v**(-a) # weights
    self.p = [w/sum(w_l) for w in w_l]
    self.dist = scipy.stats.rv_discrete(name='bounded_zipf', values=(self.v, self.p) )
  
  def __repr__(self):
    return "BZipf([{}, {}], a= {})".format(self.l_l, self.u_l, self.a)
  
  def pdf(self, x):
    return self.dist.pmf(x)
  
  def cdf(self, x):
    # if x < self.l_l: return 0
    # elif x >= self.u_l: return 1
    # else:
    #   return sum(self.p[:(x-self.l_l+1) ] )
    return self.dist.cdf(x)
  
  def inv_cdf(self, p):
    return self.dist.ppf(p)
  
  def tail(self, x):
    return 1 - self.cfd(x)
  
  def mean(self):
    # return sum([v*self.p(i) for i,v in enumerate(self.v) ] )
    return self.dist.mean()
  
  def sample(self):
    return self.dist.rvs(size=1)[0]
