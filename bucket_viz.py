from bucket_sim import *

def plot_robustornot():
  k, m, C = 3, 3, 4
  d = 1 # 2 # 1
  bci = BucketConfInspector_regularbalanced(k, m, C, d)
  E = 10
  
  ar1_robust_l, ar2_robust_l, ar3_robust_l = [], [], []
  ar1_overflow_l, ar2_overflow_l, ar3_overflow_l = [], [], []
  for ar1 in np.linspace(0, E, 50):
    for ar2 in np.linspace(0, E-ar1, 50):
      ar3 = E - ar1 - ar2
      if bci.is_stable([ar1, ar2, ar3] ):
        ar1_robust_l.append(ar1)
        ar2_robust_l.append(ar2)
        ar3_robust_l.append(ar3)
      else:
        ar1_overflow_l.append(ar1)
        ar2_overflow_l.append(ar2)
        ar3_overflow_l.append(ar3)
  fig = plot.figure()
  ax = plot.axes(projection='3d')
  ax.scatter3D(ar1_robust_l, ar2_robust_l, ar3_robust_l, label='Robust', c='blue')
  ax.scatter3D(ar1_overflow_l, ar2_overflow_l, ar3_overflow_l, c='red')
  
  fontsize = 18
  ax.set_xlabel(r'$\lambda_1$', fontsize=fontsize)
  ax.set_xlim(xmin=0)
  ax.set_ylabel(r'$\lambda_2$', fontsize=fontsize)
  ax.set_ylim(ymin=0)
  ax.set_zlabel(r'$\lambda_3$', fontsize=fontsize)
  ax.set_zlim(zmin=0)
  plot.title(r'$k= {}$, $m= {}$, $C= {}$, $\Sigma= {}$, $d= {}$'.format(k, m, C, E, d), fontsize=fontsize)
  # ax.view_init(30, -105)
  ax.view_init(20, 30)
  plot.savefig('plot_robustornot_C{}_d{}.png'.format(C, d), bbox_inches='tight')
  fig.clear()
  log(INFO, "done.")

if __name__ == "__main__":
  plot_robustornot()
