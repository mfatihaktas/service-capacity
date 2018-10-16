import math, scipy
from scipy import special
import numpy as np

from mpl_toolkits import mplot3d
import matplotlib.tri as mtri

from functools import reduce
from math import gcd

from plot_utils import *
from log_utils import *

def binom(n, k):
  return scipy.special.binom(n, k)

'''
M: Maximal uniform spacing within the order statistics of m-1 iid Uniform[0, 1] samples.
:= max{X(1), X(2)-X(1), ..., X(m-1)-X(m-2), 1-X(m-1) }

Lemma 2.4: [https://projecteuclid.org/download/pdf_1/euclid.aop/1176994313]
  Pr{M < y} --> exp(-exp(-(my - log(m) ) ) ) as m --> infty.
'''
def Pr_no_overflow_cont_approx(E, m, C): # = Pr{M < C/E}
  return math.exp(-math.exp(-(m*C/E - math.log(m) ) ) )

def Pr_no_overflow_cont(E, m, C): # = Pr{M < C/E}
  if C == 0:
    return E == 0
  s = 0
  x = C/E
  i = 1
  while True:
    if i*x >= 1:
      break
    s += (-1)**(i+1)*(1 - i*x)**(m-1)*binom(m, i)
    i += 1
  # return min(1, max(1 - s, 0) )
  p = 1 - s
  if p < 0 or p >= 1.001:
    p = 0
  return p

def min_max_required_C(E, m):
  def maximal_uniform_spacing_ub(m):
    return (2*math.log2(m) + math.log(m) )/m
  def maximal_uniform_spacing_lb(m):
    return (math.log(m) - math.log(m, 3) - math.log(2) )/m
  return E*maximal_uniform_spacing_lb(m), E*maximal_uniform_spacing_ub(m)

def min_required_g(k, E):
  for g in range(1, k):
    m = int(k/g)
    if min_max_required_C(E, m)[1] <= 2**(g-1):
      return g
  return k

'''
n balls are thrown uniformly at random into m urns of capacity b.
What is the probability that no urns overflow?

Recurrence relation is found in (4) of M.V. Ramakrishna, Computing the probability of hash table/urn overflow.
'''
def Pr_no_overflow(n, m, b):
  map_ = {}
  def P(n, m, b):
    if m == 1:
      # log(ERROR, "m= {}!".format(m) )
      return int(n <= b)
    if n <= b:
      return 1
    n_ = n-1
    
    t1 = (n_, m, b)
    if t1 in map_:
      p1 = map_[t1]
    else:
      p1 = P(*t1)
      map_[t1] = p1
    
    t2 = (n_-b, m-1, b)
    if t2 in map_:
      p2 = map_[t2]
    else:
      p2 = P(*t2)
      map_[t2] = p2
    
    # return p1 - binom(n_, b)*p2*(m-1)**(n_-b)/m**n_
    ## To avoid numeric overflow while taking factorial of large integers
    log_fact = lambda n: sum(math.log(i) for i in range(1, n+1) )
    if p1 <= 0:
      p1 = 0.000000001
    if p2 <= 0:
      p2 = 0.000000001
    # print("p2= {}, m-1= {}".format(p2, m-1) )
    r = log_fact(n_) - log_fact(n_-b) - log_fact(b) \
        + math.log(p2) + (n_-b)*math.log(m-1) - n_*math.log(m)
    return p1 - math.exp(r)
  p = P(n, m, b)
  if np.isnan(p):
    p = 0
  return p

def plot_Pr_no_overflow():
  def plot_w_varying_b(n, m):
    print("Min, Max b= {}".format(min_max_required_C(n, m) ) )
    
    b_l, p_l, p_cont_l, p_cont_approx_l = [], [], [], []
    lb = max(int(n/m)-5, 0)
    for b in range(lb, lb+20):
      b_l.append(b)
      p_l.append(Pr_no_overflow(n, m, b) )
      p_cont_l.append(Pr_no_overflow_cont(n, m, b) )
      p_cont_approx_l.append(Pr_no_overflow_cont_approx(n, m, b) )
    
    plot.plot(b_l, p_l, label='Discrete', c=NICE_RED, marker='.', ls='-', lw=2)
    plot.plot(b_l, p_cont_l, label='Cont', c=NICE_BLUE, marker='.', ls='-', lw=2)
    plot.plot(b_l, p_cont_approx_l, label='Cont approx', c=NICE_ORANGE, marker='.', ls='-', lw=2)
    plot.legend()
    plot.ylim([0, 1] )
    plot.title("n= {}, m= {}".format(n, m) )
    plot.xlabel('b', fontsize=18)
    plot.ylabel('Pr{no overflow}', fontsize=18)
    log(INFO, "done; n= {}, m= {}".format(n, m) )
  
  def plot_w_varying_m(n, b):
    m_l, p_l, p_cont_l, p_cont_approx_l = [], [], [], []
    for m in range(1, int(n/b)*10):
      m_l.append(m)
      p_l.append(Pr_no_overflow(n, m, b) )
      p_cont_l.append(Pr_no_overflow_cont(n, m, b) )
      p_cont_approx_l.append(Pr_no_overflow_cont_approx(n, m, b) )
    
    plot.plot(m_l, p_l, label='Discrete', c=NICE_RED, marker='.', ls='-', lw=2)
    plot.plot(m_l, p_cont_l, label='Cont', c=NICE_BLUE, marker='.', ls='-', lw=2)
    plot.plot(m_l, p_cont_approx_l, label='Cont approx', c=NICE_ORANGE, marker='.', ls='-', lw=2)
    plot.legend()
    plot.ylim([0, 1] )
    plot.title("n= {}, b= {}".format(n, b) )
    plot.xlabel('m', fontsize=18)
    plot.ylabel('Pr{no overflow}', fontsize=18)
    log(INFO, "done; n= {}, b= {}".format(n, b) )
  
  def plot_w_varying_n(m, b):
    n_l, p_l = [], []
    for n in range(1, b*m+1):
      n_l.append(n)
      p_l.append(Pr_no_overflow(n, m, b) )
    
    plot.plot(n_l, p_l, c=NICE_BLUE, marker='.', ls='-', lw=2)
    plot.ylim([0, 1] )
    plot.title("m= {}, b= {}".format(m, b) )
    plot.xlabel('n', fontsize=18)
    plot.ylabel('Pr{no overflow}', fontsize=18)
    log(INFO, "done; m= {}, b= {}".format(m, b) )
  
  '''
  fig, axs = plot.subplots(1, 2, sharey=True)
  plot.sca(axs[0] )
  plot_w_varying_b(n=100, m=20)
  plot.sca(axs[1] )
  plot_w_varying_m(n=100, b=10)
  # plot.sca(axs[2] )
  # plot_w_varying_n(m=10, b=10)
  
  fig.set_size_inches(2*3.5, 3.5)
  plot.savefig('plot_Pr_no_overflow.png', bbox_inches='tight')
  fig.clear()
  '''
  
  def plot_w_varying_C_m(E):
    C_l, m_l, p_l = [], [], []
    p_critical = 0.99
    C_critical_l, m_critical_l = [], []
    C_approxcritical_l, m_approxcritical_l = [], []
    # for b in range(1, int(E/2) ):
    #   for m in range(1, int(E/2) ):
    #     C_l.append(b)
    #     m_l.append(m)
    #     p = Pr_no_overflow(E, m, b)
    for C in np.linspace(0, E, 100):
      for m in range(1, int(E/2) ):
        C_l.append(C)
        m_l.append(m)
        p = Pr_no_overflow_cont(E, m, C)
        p_l.append(p)
        
        if m == 2:
          print("m= {}, C= {}, p= {}".format(m, C, p) )
        
        if p >= p_critical:
          C_critical_l.append(C)
          m_critical_l.append(m)
        
        if Pr_no_overflow_cont_approx(E, m, C) >= p_critical:
          C_approxcritical_l.append(C)
          m_approxcritical_l.append(m)
    
    fig = plot.figure()
    ax = plot.axes(projection='3d')
    # ax.plot3D(C_l, m_l, p_l, c=NICE_RED) # cmap=plot.cm.coolwarm
    # ax.contour3D(C_l, m_l, p_l, cmap=)
    
    # ax.scatter3D(C_l, m_l, p_l, c=np.abs(p_l), cmap=plot.cm.coolwarm)
    triang = mtri.Triangulation(C_l, m_l)
    ax.plot_trisurf(triang, p_l, cmap=plot.cm.Spectral, edgecolor='none')
    
    ax.set_xlabel('C', fontsize=18)
    ax.set_xlim(xmin=0)
    ax.set_ylabel('m', fontsize=18)
    ax.set_ylim(ymin=0)
    ax.set_zlabel('Pr{E-robust}', fontsize=18)
    ax.set_zlim(zmin=0, zmax=1)
    plot.title('E= {}'.format(E), fontsize=18)
    ax.view_init(30, -105)
    plot.savefig('plot_Pr_no_overflow_w_varying_C_m.png', bbox_inches='tight')
    fig.clear()
    
    plot.plot(C_critical_l, m_critical_l, label='{} >= {}'.format('Pr{E-robust}', p_critical), c=NICE_BLUE, marker='.', ls='none')
    plot.plot(C_approxcritical_l, m_approxcritical_l, label='Approx {} >= {}'.format('Pr{E-robust}', p_critical), c=NICE_RED, marker='.', ls='none')
    # x_l, y_l = [], []
    # constant = 150
    # for x in np.linspace(0, max(C_critical_l), 1000):
    #   x_l.append(x)
    #   y_l.append(constant/x)
    # plot.plot(x_l, y_l, label='x*y={}'.format(constant), c=NICE_RED, marker='.', ls='-')
    
    plot.legend()
    plot.xlabel('C', fontsize=18)
    plot.ylabel('m', fontsize=18)
    plot.ylim([0, max(m_critical_l) ] )
    plot.title('E= {}'.format(E), fontsize=18)
    plot.savefig('plot_Pr_no_overflow_critical_boundary.png', bbox_inches='tight')
    fig.clear()
    # for angle in range(0, 360):
    #   ax.view_init(30, angle)
    #   plot.draw()
    #   plot.pause(.001)
  plot_w_varying_C_m(E=100)
  log(INFO, "done.")

def compare_varying_groupsize(max_gs):
  E = 420 # 200 # 100
  k = reduce(lambda x,y: x*y//gcd(x, y), range(1, max_gs+1) ) # least common multiple
  print("max_gs= {}, k= {}".format(max_gs, k) )
  
  min_g = min_required_g(k, E)
  print("min_g= {}".format(min_g) )
  
  g_l, p_l, nservers_l = [], [], []
  for g in range(1, max_gs+3):
    g_l.append(g)
    m = int(k/g)
    
    # Encode with simplex
    num_servers = m*(2**g - 1)
    b = 2**(g-1)
    
    # p = Pr_no_overflow(E, m, b)
    p = Pr_no_overflow_cont(E, m, b)
    print('g= {}, p= {}, num_servers= {}'.format(g, p, num_servers) )
    p_l.append(p)
    
    nservers_l.append(num_servers)
  
  fig, axs = plot.subplots(1, 2)
  fontsize = 14
  ## Pr{robust}
  ax = axs[0]
  plot.sca(ax)
  plot.plot(g_l, p_l, c=NICE_BLUE, marker='o', ls='-')
  plot.xlabel('g', fontsize=fontsize)
  plot.ylabel('Pr{E-robust}', fontsize=fontsize)
  plot.ylim([0, 1.1] )
  prettify(ax)
  
  ## Number of servers
  ax = axs[1]
  plot.sca(ax)
  plot.plot(g_l, nservers_l, c=NICE_RED, marker='o', ls='-')
  plot.xlabel('g', fontsize=fontsize)
  plot.ylabel('Total number of servers', fontsize=fontsize)
  prettify(ax)
  
  st = plot.suptitle('k= {}, E= {}, asymptotic min g= {}'.format(k, E, min_g), fontsize=fontsize)
  plot.subplots_adjust(wspace=0.3)
  fig.set_size_inches(2*4.5, 3.5)
  plot.savefig('plot_Pr_robustness_w_varying_g_E{}.png'.format(E), bbox_extra_artists=[st], bbox_inches='tight')
  plot.gcf().clear()
  log(INFO, "done; E= {}, max_gs= {}".format(E, max_gs) )

if __name__ == "__main__":
  '''
  def exp(n, m, b):
    p = Pr_no_overflow(n, m, b)
    log(INFO, "n= {}, m= {}, b= {}, Pr_no_overflow= {}".format(n, m, b, p) )
  
  m, b = 3, 3
  for n in range(1, m*b+1):
    exp(n, m, b)
  '''
  
  # plot_Pr_no_overflow()
  compare_varying_groupsize(max_gs=7)
  