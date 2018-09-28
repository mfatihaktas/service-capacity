import math, scipy
from scipy import special
import numpy as np

from mpl_toolkits import mplot3d
import matplotlib.tri as mtri

from plot_utils import *
from log_utils import *

def binom(n, k):
  return scipy.special.binom(n, k)

'''
n balls are thrown uniformly at random into m urns of capacity b.
What is the probability that no urns overflow?

Recurrence relation is found in (4) of M.V. Ramakrishna, Computing the probability of hash table/urn overflow.
'''
def Pr_no_overflow(n, m, b):
  map_ = {}
  def P(n, m, b):
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
    
    return p1 - binom(n_, b)*p2*(m-1)**(n_-b)/m**n_
  return P(n, m, b)

def plot_Pr_no_overflow():
  
  def plot_w_varying_b(n, m):
    b_l, p_l = [], []
    for b in range(1, int(n/2) ):
      b_l.append(b)
      p_l.append(Pr_no_overflow(n, m, b) )
    
    plot.plot(b_l, p_l, c=NICE_RED, marker='.', ls='-', lw=3)
    plot.ylim([0, 1] )
    plot.title("n= {}, m= {}".format(n, m) )
    plot.xlabel('b', fontsize=14)
    plot.ylabel('Pr{no overflow}', fontsize=14)
    log(INFO, "done; n= {}, m= {}".format(n, m) )
  
  def plot_w_varying_m(n, b):
    m_l, p_l = [], []
    for m in range(1, int(n/b)*10):
      m_l.append(m)
      p_l.append(Pr_no_overflow(n, m, b) )
    
    plot.plot(m_l, p_l, c=NICE_ORANGE, marker='.', ls='-', lw=3)
    plot.ylim([0, 1] )
    plot.title("n= {}, b= {}".format(n, b) )
    plot.xlabel('m', fontsize=14)
    plot.ylabel('Pr{no overflow}', fontsize=14)
    log(INFO, "done; n= {}, b= {}".format(n, b) )
  
  def plot_w_varying_n(m, b):
    n_l, p_l = [], []
    for n in range(1, b*m+1):
      n_l.append(n)
      p_l.append(Pr_no_overflow(n, m, b) )
    
    plot.plot(n_l, p_l, c=NICE_BLUE, marker='.', ls='-', lw=3)
    plot.ylim([0, 1] )
    plot.title("m= {}, b= {}".format(m, b) )
    plot.xlabel('n', fontsize=14)
    plot.ylabel('Pr{no overflow}', fontsize=14)
    log(INFO, "done; m= {}, b= {}".format(m, b) )
  
  '''
  fig, axs = plot.subplots(1, 2, sharey=True)
  plot.sca(axs[0] )
  plot_w_varying_b(n=100, m=10)
  plot.sca(axs[1] )
  plot_w_varying_m(n=100, b=4)
  # plot.sca(axs[2] )
  # plot_w_varying_n(m=10, b=10)
  fig.set_size_inches(2*3.5, 3.5)
  '''
  
  def plot_w_varying_b_m(n):
    b_l, m_l, p_l = [], [], []
    for b in range(1, int(n/2) ):
      for m in range(1, int(n/2) ):
        b_l.append(b)
        m_l.append(m)
        p = Pr_no_overflow(n, m, b)
        p = 0 if np.isnan(p) else p
        p_l.append(p)
    
    fig = plot.figure()
    ax = plot.axes(projection='3d')
    # ax.plot3D(b_l, m_l, p_l, c=NICE_RED) # cmap=plot.cm.coolwarm
    # ax.contour3D(b_l, m_l, p_l, cmap=)
    
    # ax.scatter3D(b_l, m_l, p_l, c=np.abs(p_l), cmap=plot.cm.coolwarm)
    triang = mtri.Triangulation(b_l, m_l)
    ax.plot_trisurf(triang, p_l, cmap=plot.cm.Spectral, edgecolor='none')
    
    ax.set_xlabel('b')
    ax.set_xlim(xmin=0)
    ax.set_ylabel('m')
    ax.set_ylim(ymin=0)
    ax.set_zlabel('Pr{no overflow}')
    ax.set_zlim(zmin=0, zmax=1)
    plot.title('n= {}'.format(n) )
    
    ax.view_init(30, -120)
    
    # for angle in range(0, 360):
    #   ax.view_init(30, angle)
    #   plot.draw()
    #   plot.pause(.001)
  plot_w_varying_b_m(n=100)
  
  plot.savefig('plot_Pr_no_overflow.png', bbox_inches='tight')
  # fig.clear()
  log(INFO, "done.")

if __name__ == "__main__":
  '''
  def exp(n, m, b):
    p = Pr_no_overflow(n, m, b)
    log(INFO, "n= {}, m= {}, b= {}, Pr_no_overflow= {}".format(n, m, b, p) )
  
  m, b = 3, 3
  for n in range(1, m*b+1):
    exp(n, m, b)
  '''
  
  plot_Pr_no_overflow()
  