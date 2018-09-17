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
      l.append(np.array(p_l)*self.arrate_rv.sample() )
    return l
  
  def gaussian_kde(self, npoints=10000):
    cap_l = self.cap_l(npoints)
    values = np.array(cap_l).reshape((self.k, len(cap_l) ))
    # blog(values=values)
    
    max_l = np.amax(values, axis=1).tolist()
    kernel = scipy.stats.gaussian_kde(values) # bw_method='silverman'
    return kernel, max_l
  
  def joint_pdf(self, *args):
    return self.kernel(np.array(args).reshape((self.k, 1)) )[0]
  
  def plot_heatmap_2d(self):
    cap_l = self.cap_l(npoints=1000)
    
    values = np.array(cap_l).reshape((self.k, len(cap_l) ))
    plot.plot(values[0, :], values[1, :], 'k.', markersize=2)
    fig = plot.gcf()
    fig.set_size_inches(5, 5)
    plot.savefig('plot_scatter_2d.png', bbox_inches='tight')
    fig.clear()
    
    data = pd.DataFrame(cap_l, columns=['a', 'b'] )
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
    cap_l = self.cap_l(npoints=10000)
    
    values = np.array(cap_l).reshape((self.k, len(cap_l) ))
    xmin, xmax = min(values[0, :]), max(values[0, :])
    ymin, ymax = min(values[1, :]), max(values[1, :])
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel() ] )
    # blog(positions=positions)
    Z = np.reshape(self.kernel(positions).T, X.shape)
    # blog(Z=Z)
    
    plot.imshow(np.rot90(Z), cmap=plot.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax] )
    plot.plot(values[0, :], values[1, :], 'k.', markersize=2)
    plot.xlim([xmin, xmax] )
    plot.ylim([ymin, ymax] )
    plot.xlabel('a', fontsize=14)
    plot.ylabel('b', fontsize=14)
    plot.title(r'$\lambda \sim {}$, $\alpha \sim {}$'.format(self.arrate_rv, self.zipf_tailindex_rv), fontsize=14)
    prettify(plot.gca() )
    fig = plot.gcf()
    fig.set_size_inches(5, 5)
    plot.savefig('plot_kde_2d.png', bbox_inches='tight')
    fig.clear()
    log(INFO, "done.")
  
if __name__ == "__main__":
  pm = PopModel_wZipf(k=2, zipf_tailindex_rv=TNormal(1.2, 1), arrate_rv=TNormal(3, 1) )
  # pm.plot_heatmap_2d()
  pm.plot_kde_2d()
