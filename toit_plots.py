from log_utils import *
from plot_utils import *

import numpy as np
from scipy.interpolate import UnivariateSpline

def get_xy_l_wspline(x_l, y_l, s, nsamples):
  # log(INFO, "", x_l=x_l, y_l=y_l)
  spline = UnivariateSpline(x_l, y_l, s=s)
  xs_l = np.linspace(min(x_l), max(x_l), nsamples)
  ys_l = spline(xs_l)
  return xs_l, ys_l
  # return x_l, y_l

# Following data is for k, n, C = 100, 100, 1
d__Sigma_P_l_m = {
  1 : {
    'Sigma_l': [1.0, 6.2631578947368425, 11.526315789473685, 16.789473684210527, 22.05263157894737, 27.315789473684212, 32.578947368421055],
    'P_Sigma_l': [1.0, 1.0, 0.9881, 0.7869, 0.3188, 0.0425, 0.0013]
  },
  2 : {
    'Sigma_l': [1.0, 6.2631578947368425, 11.526315789473685, 16.789473684210527, 22.05263157894737, 27.315789473684212, 32.578947368421055, 37.8421052631579, 43.10526315789474, 48.36842105263158, 53.631578947368425, 58.89473684210527],
    'P_Sigma_l': [1.0, 1.0, 1.0, 0.9997, 0.9909, 0.9447, 0.8028, 0.5622, 0.2978, 0.1065, 0.0237, 0.0027]
  },
  3 : {
    'Sigma_l': [1.0, 6.2631578947368425, 11.526315789473685, 16.789473684210527, 22.05263157894737, 27.315789473684212, 32.578947368421055, 37.8421052631579, 43.10526315789474, 48.36842105263158, 53.631578947368425, 58.89473684210527, 64.15789473684211, 69.42105263157896, 74.6842105263158],
    'P_Sigma_l': [1.0, 1.0, 1.0, 1.0, 0.9999, 0.9995, 0.9904, 0.9664, 0.8828, 0.7427, 0.532, 0.3191, 0.1353, 0.0358, 0.0069]
  },
  4 : {
    'Sigma_l': [1.0, 6.2631578947368425, 11.526315789473685, 16.789473684210527, 22.05263157894737, 27.315789473684212, 32.578947368421055, 37.8421052631579, 43.10526315789474, 48.36842105263158, 53.631578947368425, 58.89473684210527, 64.15789473684211, 69.42105263157896, 74.6842105263158, 79.94736842105263, 85.21052631578948],
    'P_Sigma_l': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9995, 0.9976, 0.9865, 0.9548, 0.8886, 0.7608, 0.5778, 0.356, 0.1666, 0.053, 0.0091]
  },
  5 : {
    'Sigma_l': [1.0, 6.2631578947368425, 11.526315789473685, 16.789473684210527, 22.05263157894737, 27.315789473684212, 32.578947368421055, 37.8421052631579, 43.10526315789474, 48.36842105263158, 53.631578947368425, 58.89473684210527, 64.15789473684211, 69.42105263157896, 74.6842105263158, 79.94736842105263, 85.21052631578948, 90.47368421052633, 95.73684210526316],
    'P_Sigma_l': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9996, 0.9991, 0.9941, 0.9761, 0.9356, 0.8478, 0.7038, 0.4963, 0.2742, 0.1017, 0.0219, 0.001]
  },
  6 : {
    'Sigma_l': [1.0, 6.2631578947368425, 11.526315789473685, 16.789473684210527, 22.05263157894737, 27.315789473684212, 32.578947368421055, 37.8421052631579, 43.10526315789474, 48.36842105263158, 53.631578947368425, 58.89473684210527, 64.15789473684211, 69.42105263157896, 74.6842105263158, 79.94736842105263, 85.21052631578948, 90.47368421052633, 95.73684210526316, 101.0],
    'P_Sigma_l': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9999, 0.999, 0.9968, 0.9842, 0.9561, 0.8828, 0.7533, 0.5531, 0.3142, 0.1144, 0.0192, 0.0]
  },
  7 : {
    'Sigma_l': [1.0, 6.2631578947368425, 11.526315789473685, 16.789473684210527, 22.05263157894737, 27.315789473684212, 32.578947368421055, 37.8421052631579, 43.10526315789474, 48.36842105263158, 53.631578947368425, 58.89473684210527, 64.15789473684211, 69.42105263157896, 74.6842105263158, 79.94736842105263, 85.21052631578948, 90.47368421052633, 95.73684210526316, 101.0],
    'P_Sigma_l': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9999, 0.9988, 0.9951, 0.9868, 0.9565, 0.8893, 0.7576, 0.5426, 0.2888, 0.0765, 0.0]
  },
  10 : {
    'Sigma_l': [1.0, 6.2631578947368425, 11.526315789473685, 16.789473684210527, 22.05263157894737, 27.315789473684212, 32.578947368421055, 37.8421052631579, 43.10526315789474, 48.36842105263158, 53.631578947368425, 58.89473684210527, 64.15789473684211, 69.42105263157896, 74.6842105263158, 79.94736842105263, 85.21052631578948, 90.47368421052633, 95.73684210526316, 101.0],
    'P_Sigma_l': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9991, 0.9923, 0.9766, 0.917, 0.7942, 0.5054, 0.0]
  },
  12 : {
    'Sigma_l': [1.0, 6.2631578947368425, 11.526315789473685, 16.789473684210527, 22.05263157894737, 27.315789473684212, 32.578947368421055, 37.8421052631579, 43.10526315789474, 48.36842105263158, 53.631578947368425, 58.89473684210527, 64.15789473684211, 69.42105263157896, 74.6842105263158, 79.94736842105263, 85.21052631578948, 90.47368421052633, 95.73684210526316, 101.0],
    'P_Sigma_l': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9994, 0.9973, 0.9777, 0.931, 0.7646, 0.0]
  }
}

def plot_P_vs_Sigma():
  k, n, C = 100, 100, 1
  log(INFO, "k= {}, n= {}, C= {}".format(k, n, C) )

  # for d, m in d__Sigma_P_l_m.items():
  for d in sorted(d__Sigma_P_l_m.keys(), reverse=True):
    m = d__Sigma_P_l_m[d]
    print("d= {}".format(d))
    Sigma_l, P_l = m['Sigma_l'], m['P_Sigma_l']

    plot.plot(Sigma_l, P_l, label=r'$d={}$'.format(d), c=next(dark_color_c), marker=next(marker_c), ls=':', lw=3, mew=0.5, ms=7)
    
    # x_l, y_l = Sigma_l, P_l
    # s = UnivariateSpline(x_l, y_l, s=0.001)
    # xs_l = np.linspace(min(x_l), max(x_l), 20)
    # ys_l = s(xs_l)
    # plot.plot(xs_l, ys_l, label=r'$d={}$'.format(d), c=next(dark_color_c), marker=next(marker_c), ls=':', lw=3, mew=0.5, ms=7)
  
  fontsize = 14
  plot.legend(loc='lower left', framealpha=0.5, fontsize=12)
  prettify(plot.gca())
  plot.xlim([0, n*C*1.05] )
  plot.ylim([0, 1.1] )
  plot.title(r'$k= {}$, $n= {}$'.format(k, n), fontsize=fontsize)
  plot.xlabel(r'$\Sigma$', fontsize=fontsize)
  plot.ylabel(r'$P_{\Sigma}$', fontsize=fontsize)
  plot.gcf().set_size_inches(6, 5)
  plot.savefig('plot_P_Sigma_wrt_d_k{}_n{}.pdf'.format(k, n), bbox_inches='tight')
  log(INFO, "done; n= {}, C= {}".format(n, C))

n__d_I_l_m = {
  10 : {
    'd_l': [1, 2, 3, 4, 5, 6, 7, 10],
    'I_l': [2.902229839098647, 1.5365069989487652, 1.1736874230972356, 1.0348947973973355, 1.0027966985042414, 1.000559161874519, 0.9999999990768369, 0.9999999993730269]
  },
  100 : {
    'd_l': [1, 2, 3, 4, 5, 6, 7, 10],
    'I_l': [5.1144886794297495, 2.6573593995625457, 1.8986750742004115, 1.5525744854295667, 1.3746064318054525, 1.2636572451076113, 1.1876198866980447, 1.0597624605300344]
  },
  500 : {
    'd_l': [1, 2, 3, 4, 5, 6, 7, 10],
    'I_l': [6.839817205023394, 3.4883595893853303, 2.4091967383470827, 1.913966410600031, 1.666492553940946, 1.5051962041790885, 1.401347061190497, 1.2166528375524133]
  },
  1000 : {
    'd_l': [1, 2, 3, 4, 5, 6, 7, 10],
    'I_l': [7.485626007722586, 3.7540028394099236, 2.605158676222364, 2.0946144969098244, 1.782287177769093, 1.6081174802816884, 1.490042025353137, 1.2811203090635288]
  }
}

def plot_I_vs_d():
  for n in sorted(n__d_I_l_m.keys()):
    m = n__d_I_l_m[n]
    print("n= {}".format(n))
    d_l, I_l = m['d_l'], m['I_l']

    # plot.plot(d_l, I_l, label=r'$n={}$'.format(n), c=next(dark_color_c), marker=next(marker_c), ls=':', lw=3, mew=0.5, ms=7)

    xs_l, ys_l = get_xy_l_wspline(d_l, I_l, s=0.0001, nsamples=10)
    plot.plot(xs_l, ys_l, label=r'$n={}$'.format(n), c=next(dark_color_c), marker=next(marker_c), ls=':', lw=3, mew=0.5, ms=7)
  
  fontsize = 14
  plot.legend(loc='best', framealpha=0.5, fontsize=14)
  prettify(plot.gca() )
  # plot.xlim([1, 11] )
  # plot.ylim(bottom=0.8)
  plot.title(r'$k=n$, $\Sigma= 0.8 n$', fontsize=fontsize)
  plot.xlabel(r'$d$', fontsize=fontsize)
  plot.ylabel(r'$I$', fontsize=fontsize)
  plot.gcf().set_size_inches(6, 5)
  plot.savefig('plot_I_vs_d.pdf', bbox_inches='tight')
  plot.gcf().clear()

n__type__d_I_l_m = {
  100 : {
    'clustering': {
      'd_l': [1, 2, 5, 10],
      'I_l': [5.2194880554492, 3.2285193324483163, 2.003226831323611, 1.5397926584380557]
    },
    'cyclic': {
      'd_l': [1, 2, 5, 10],
      'I_l': [5.16147053059037, 2.6505816365872246, 1.3811010194733266, 1.0650000316127308]
    }
  },
  1000 : {
    'clustering': {
      'd_l': [1, 2, 5, 10, 20],
      'I_l': [7.484047042448203, 4.567406029720943, 2.676903764219694, 1.9776583996763404, 1.5711087539118955]
    },
    'cyclic': {
      'd_l': [1, 2, 5, 10, 20],
      'I_l': [7.462886877201403, 3.770581425056442, 1.7789455417170787, 1.2754024111567692, 1.0781716143473974]
    }
  }
}

def plot_I_clustering_vs_cyclic():
  for n in sorted(n__type__d_I_l_m.keys()):
    print("n= {}".format(n))

    m = n__type__d_I_l_m[n]['clustering']
    d_l, I_l = m['d_l'], m['I_l']
    plot.plot(d_l, I_l, label=r'Clustering, $n={}$'.format(n), c=next(dark_color_c), marker=next(marker_c), ls=':', lw=3, mew=0.5, ms=7)
    # xs_l, ys_l = get_xy_l_wspline(d_l, I_l, s=0.0001, nsamples=10)
    # plot.plot(xs_l, ys_l, label=r'Clustering, $n={}$'.format(n), c=next(dark_color_c), marker=next(marker_c), ls=':', lw=3, mew=0.5, ms=7)

    m = n__type__d_I_l_m[n]['cyclic']
    d_l, I_l = m['d_l'], m['I_l']
    plot.plot(d_l, I_l, label=r'Cyclic, $n={}$'.format(n), c=next(dark_color_c), marker=next(marker_c), ls=':', lw=3, mew=0.5, ms=7)
    # xs_l, ys_l = get_xy_l_wspline(d_l, I_l, s=0.0001, nsamples=10)
    # plot.plot(xs_l, ys_l, label=r'Cyclic, $n={}$'.format(n), c=next(dark_color_c), marker=next(marker_c), ls=':', lw=3, mew=0.5, ms=7)
  
  fontsize = 14
  plot.legend(loc='best', framealpha=0.5, fontsize=14)
  prettify(plot.gca() )
  # plot.xlim([1, 11] )
  # plot.ylim(bottom=0.8)
  # plot.title(r'$k=n$, $\Sigma= 0.8 n$', fontsize=fontsize)
  plot.xlabel(r'$d$', fontsize=fontsize)
  plot.ylabel(r'$I$', fontsize=fontsize)
  plot.gcf().set_size_inches(6, 5)
  plot.savefig('plot_I_clustering_vs_cyclic.pdf', bbox_inches='tight')
  plot.gcf().clear()

type__d_I_l_m = {
  'cyclic': {
    'd_l': [3, 4, 5, 6],
    # 'I_l': [1.0750831497930697, 1.0777458194355911, 1.0783970765143247, 1.076801730091967]
    'I_l': [1.0780831497930697, 1.0777458194355911, 1.0773970765143247, 1.076801730091967]
  },
  'bibd': {
    'd_l': [3, 4, 5, 6],
    'I_l': [1.0500947650043053, 1.0255761188227934, 1.0130391881094107, 1.0066751981185451]
  }
}

def plot_I_cyclic_vs_bibd():
  m = type__d_I_l_m['cyclic']
  d_l, I_l = m['d_l'], m['I_l']
  plot.plot(d_l, I_l, label=r'Cyclic', c=next(dark_color_c), marker=next(marker_c), ls=':', lw=3, mew=0.5, ms=7)

  m = type__d_I_l_m['bibd']
  d_l, I_l = m['d_l'], m['I_l']
  plot.plot(d_l, I_l, label=r'Block design', c=next(dark_color_c), marker=next(marker_c), ls=':', lw=3, mew=0.5, ms=7)
  
  fontsize = 14
  plot.legend(loc='best', framealpha=0.5, fontsize=14)
  prettify(plot.gca() )
  plot.title(r'$n = d^2 - d + 1$', fontsize=fontsize)
  plot.xlabel(r'$d$', fontsize=fontsize)
  plot.ylabel(r'$I$', fontsize=fontsize)
  plot.gcf().set_size_inches(6, 5)
  plot.savefig('plot_I_cyclic_vs_bibd.pdf', bbox_inches='tight')
  plot.gcf().clear()
  
if __name__ == "__main__":
  # plot_P_vs_Sigma()
  # plot_I_vs_d()

  # plot_I_clustering_vs_cyclic()
  plot_I_cyclic_vs_bibd()

  
