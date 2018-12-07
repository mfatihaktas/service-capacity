import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('Agg')
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import scipy, seaborn
seaborn.set(style='white', color_codes=True)

from plot_utils import *
from log_utils import *
from rvs import *

class PopModel_wZipf(object):
  def __init__(self, k, zipf_tailindex_rv, arrate_rv):
    self.k = k
    self.zipf_tailindex_rv = zipf_tailindex_rv
    self.arrate_rv = arrate_rv
    
    self.cap_l_ = self.cap_l(5000)
    # self.values = np.array(self.cap_l_).reshape((self.k, len(self.cap_l_) )).T
    self.values = np.column_stack(tuple(self.cap_l_) )
    blog(values=self.values)
    
    self.kernel, self.max_l = self.gaussian_kde()
  
  def __repr__(self):
    return 'PopModel_wZipf[k= {}, zipf_tailindex_rv= {}, arrate_rv= {}]'.format(self.k, self.zipf_tailindex_rv, self.arrate_rv)
  
  def p_l(self, a):
    self.v_l = np.arange(1, self.k+1)
    w_l = [float(v)**(-a) for v in self.v_l]
    return [w/sum(w_l) for w in w_l]
  
  def cap_l(self, npoints):
    l = []
    a_l = [self.zipf_tailindex_rv.sample() for _ in range(npoints) ]
    for a in a_l:
      p_l = self.p_l(a)
      if random.uniform(0, 1) < 0.5: # each symbol is equally likely to be more popular
        p_l.reverse()
      ar = self.arrate_rv.sample()
      # if ar > 2:
      #   print("ar= {} > 2!".format(ar) )
      l.append(np.array(p_l)*ar)
    return l
  
  def integrate_overpopmodel(self, func):
    ranges = []
    for m in self.max_l:
      ranges.append((0, m) )
    log(INFO, "", ranges=ranges)
    result, abserr = scipy.integrate.nquad(func, ranges, opts={'limit': 200, 'epsabs': 1.49e-04})
    return round(result, 2)
  
  def gaussian_kde(self, npoints=10000):
    max_l = np.amax(self.values, axis=1).tolist()
    kernel = scipy.stats.gaussian_kde(self.values) # bw_method='silverman'
    return kernel, max_l
  
  def joint_pdf(self, *args):
    return self.kernel(np.array(args).reshape((self.k, 1)) )[0]
  
  def plot_heatmap_2d(self):
    plot.plot(self.values[0, :], self.values[1, :], 'k.', markersize=2)
    fig = plot.gcf()
    fig.set_size_inches(5, 5)
    plot.savefig('plot_scatter_2d.png', bbox_inches='tight')
    fig.clear()
    
    data = pd.DataFrame(self.cap_l_, columns=['a', 'b'] )
    # print('data= {}'.format(data) )
    # seaborn.jointplot(x='a', y='b', data=data)
    seaborn.jointplot(x='a', y='b', data=data, kind='kde', space=0) # color='red'
    
    plot.xlim(xmin=0)
    plot.ylim(ymin=0)
    st = plot.suptitle(r'$k= {}$, $a \sim {}$, $\lambda \sim {}$'.format(self.k, self.zipf_tailindex_rv, self.arrate_rv) )
    fig = plot.gcf()
    fig.set_size_inches(5, 5)
    plot.savefig('plot_heatmap_2d.png', bbox_extra_artists=[st], bbox_inches='tight')
    fig.clear()
    log(INFO, "done.")
  
  def plot_kde_2d(self):
    xmin, xmax = min(self.values[0, :]), max(self.values[0, :])
    ymin, ymax = min(self.values[1, :]), max(self.values[1, :])
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel() ] )
    # blog(positions=positions)
    Z = np.reshape(self.kernel(positions).T, X.shape)
    # blog(Z=Z)
    
    plot.imshow(np.rot90(Z), cmap=plot.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax] )
    plot.plot(self.values[0, :], self.values[1, :], 'k.', markersize=2)
    plot.xlim([xmin, xmax] )
    plot.ylim([ymin, ymax] )
    plot.xlabel(r'$\lambda_a$', fontsize=20)
    plot.ylabel(r'$\lambda_b$', fontsize=20)
    # plot.title(r'$\lambda \sim {}$, $\alpha \sim {}$'.format(self.arrate_rv, self.zipf_tailindex_rv), fontsize=18)
    prettify(plot.gca() )
    fig = plot.gcf()
    fig.set_size_inches(4, 4)
    plot.savefig('plot_kde_2d.png', bbox_inches='tight')
    fig.clear()
    log(INFO, "done.")
  
if __name__ == "__main__":
  pm = PopModel_wZipf(k=2, zipf_tailindex_rv=TNormal(1, 2), arrate_rv=TNormal(1.5, 0.4) )
  # # pm.plot_heatmap_2d()
  pm.plot_kde_2d()
  
  # rv = TNormal(2, 0.01)
  # print("samples= \n{}".format([rv.sample() for _ in range(1000) ] ) )
