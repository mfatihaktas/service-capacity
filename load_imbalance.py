from bucket_wchoice import *
from bucket_wcode import *
from math_utils import *

def I_loadimbalance(k, n, d):
  if d == 1:
    r = k // n
    a = 1 - (r-1)*mlog(mlog(n))/mlog(n) + mlog(math.factorial(r-1))/mlog(n)
    return mlog(n)/r/a
  else:
    log(WARNING, "not implemented yet.")
    return None

def I_wm():
  d = 1
  C = 1
  n = 100 # 20
  
  def w_m(m):
    log(INFO, "m= {}".format(m) )
    k = m*n
    bci = BucketConfInspector_roundrobin(k, n, C, d)
    
    E_l, I_l, sim_I_l = [], [], []
    # for E in np.linspace(C, n+1, 10):
    for E in np.linspace(C, n, 1):
      print(">> E= {}".format(E) )
      E_l.append(E)
      
      I = I_loadimbalance(m*n, n, d)
      sim_mincap = bci.sim_min_bucketcap_forstability(E, nsamples=10**4)
      sim_I = sim_mincap/(E/n)
      print("I= {}, sim_I= {}".format(I, sim_I) )
      I_l.append(I)
      sim_I_l.append(sim_I)
    c = next(dark_color_c)
    plot.plot(E_l, I_l, label='model, m= {}'.format(m), c=c, marker=next(marker_c), ls=':', lw=3, mew=0.5, ms=7)
    plot.plot(E_l, sim_I_l, label='sim, m= {}'.format(m), c=c, marker=next(marker_c), ls=':', lw=3, mew=0.5, ms=7)
  
  for m in [3, 4]: # [1, 2, 3]:
    w_m(m)
  
  fontsize = 20
  plot.legend(loc='best', framealpha=0.5, fontsize=14)
  # plot.ylim([0, 1] )
  prettify(plot.gca() )
  plot.title('n= {}'.format(n), fontsize=fontsize)
  plot.xlabel(r'$\Sigma$', fontsize=fontsize)
  plot.ylabel(r'$I$', fontsize=fontsize)
  plot.gcf().set_size_inches(7, 5)
  plot.savefig('I_wm.png', bbox_inches='tight')
  plot.gcf().clear()
  log(INFO, "done; n= {}".format(n) )

def I_wd():
  C = 1
  
  def plot_(n, ro):
    print("\n>> n= {}, ro= {}".format(n, ro) )
    E = ro*n*C
    d_l, sim_I_l = [], []
    for d in [*range(1, 8), 10]:
      d_l.append(d)
      
      bci = BucketConfInspector_roundrobin(n, n, C, d)
      sim_mincap = bci.sim_min_bucketcap_forstability(E, nsamples=10**2) # 10**4
      sim_I = sim_mincap/(E/n)
      print("d= {}, sim_I= {}".format(d, sim_I) )
      sim_I_l.append(sim_I)
    plot.plot(d_l, sim_I_l, label=r'$n= {}$'.format(n), c=next(dark_color_c), marker=next(marker_c), ls=':', lw=3, mew=0.5, ms=7)
  
  plot_(n=10, ro=0.8)
  plot_(n=100, ro=0.8)
  plot_(n=500, ro=0.8)
  plot_(n=1000, ro=0.8)
  
  fontsize = 20
  plot.legend(loc='best', framealpha=0.5, fontsize=14)
  prettify(plot.gca() )
  plot.xlim([1, 10] )
  plot.ylim(bottom=1)
  plot.title(r'$k=n$, $\Sigma= 0.8 n$', fontsize=fontsize)
  plot.xlabel(r'$d$', fontsize=fontsize)
  plot.ylabel(r'$I$', fontsize=fontsize)
  plot.gcf().set_size_inches(7, 5)
  plot.savefig('I_wd.pdf', bbox_inches='tight')
  plot.gcf().clear()
  log(INFO, "done.")

def compare_I_clustering_vs_rr():
  C = 1
  
  def plot_(n, d_l, t):
    log(INFO, ">> n= {}, d_l= {}, t= {}".format(n, d_l, t) )
    k = n
    E = n*C*0.8
    
    nsamples = 10**2 # 10**5
    I_l = []
    for d in d_l:
      if t == 'cl':
        bci = BucketConfInspector_clustering(k, n, C, d)
        name = "Clustering"
      elif t == 'rr':
        bci = BucketConfInspector_roundrobin(k, n, C, d)
        name = "Cyclic"
      else:
        log(ERROR, "Unknown t= {}".format(t) )
        return
      
      I = bci.sim_min_bucketcap_forstability(E, nsamples)/(E/n)
      blog(I=I)
      I_l.append(I)
    plot.plot(d_l, I_l, label='{}, $n= {}$'.format(name, n), c=next(dark_color_c), marker=next(marker_c), ls=':', lw=3, mew=0.5, ms=7)
  
  # plot_(n=9, d_l=[1, 3], t='cl')
  # plot_(n=9, d_l=[1, 2, 3, 4], t='rr')
  
  # plot_(n=20, d_l=[1, 2, 4, 5], t='cl')
  # plot_(n=20, d_l=[1, 2, 3, 4, 5], t='rr')
  
  plot_(n=100, d_l=[1, 2, 5, 10], t='cl')
  plot_(n=100, d_l=[1, 2, 5, 10], t='rr')
  
  plot_(n=1000, d_l=[1, 2, 5, 10, 20], t='cl')
  plot_(n=1000, d_l=[1, 2, 5, 10, 20], t='rr')
  
  fontsize = 20
  plot.legend(loc='best', framealpha=0.5, fontsize=12)
  prettify(plot.gca() )
  plot.xlabel(r'$d$', fontsize=fontsize)
  plot.ylabel(r'$I$', fontsize=fontsize)
  plot.gcf().set_size_inches(5, 5)
  plot.savefig('compare_I_clustering_vs_rr.pdf', bbox_inches='tight')
  plot.gcf().clear()
  log(INFO, "done.")

def compare_I_rr_vs_bibd():
  C = 1
  def plot_(d):
    log(INFO, "started; d= {}".format(d) )
    n = k = d**2 - d + 1
    bci_rr = BucketConfInspector_roundrobin(k, n, C, d)
    if d == 3:
      obj__bucket_l_m = {
        0: [0, 1, 2],
        1: [0, 3, 4],
        2: [0, 5, 6],
        3: [2, 3, 5],
        4: [2, 4, 6],
        5: [1, 3, 6],
        6: [1, 4, 5] }
    elif d == 4:
      
    elif d == 5:
      # https://rdrr.io/cran/ibd/man/bibd.html
      # bibd(v=21, b=21, r=5, k=5, lambda=1)
      #           [,1] [,2] [,3] [,4] [,5]
      # Block-1     6    7   15   18   19
      # Block-2     4    5    7   10   12
      # Block-3     4    6   11   16   21
      # Block-4     8   10   11   17   19
      # Block-5     6   12   14   17   20
      # Block-6     3   11   12   13   15
      # Block-7     3   10   14   16   18
      # Block-8     1   10   15   20   21
      # Block-9     8    9   12   18   21
      # Block-10    1    7    9   11   14
      # Block-11    5   13   14   19   21
      # Block-12    7    8   13   16   20
      # Block-13    1    3    5    6    8
      # Block-14    3    4    9   19   20
      # Block-15    5    9   15   16   17
      # Block-16    2    3    7   17   21
      # Block-17    1    4   13   17   18
      # Block-18    2    6    9   10   13
      # Block-19    2    5   11   18   20
      # Block-20    2    4    8   14   15
      # Block-21    1    2   12   16   19
      obj__bucket_l_m = {
        0: [5, 6, 14, 17, 18],
        1: [3, 4, 6, 9, 11],
        2: [3, 5, 10, 15, 20],
        3: [7, 9, 10, 16, 18],
        4: [5, 11, 13, 16, 19],
        5: [2, 10, 11, 12, 14],
        6: [2, 9, 13, 15, 17],
        7: [0, 9, 14, 19, 20],
        8: [7, 8, 11, 17, 20],
        9: [0, 6, 8, 10, 13],
        10: [4, 12, 13, 18, 20],
        11: [6, 7, 12, 15, 19],
        12: [0, 2, 4, 5, 7],
        13: [2, 3, 8, 18, 19],
        14: [4, 8, 14, 15, 16],
        15: [1, 2, 6, 16, 20],
        16: [0, 3, 12, 16, 17],
        17: [1, 5, 8, 9, 12],
        18: [1, 4, 10, 17, 19],
        19: [1, 3, 7, 13, 14],
        20: [0, 1, 11, 15, 18] }
    else:
      log(ERROR, "not implemented for d= {}".format(d) )
      return
    bci_bibd = BucketConfInspector(n, C, obj__bucket_l_m)
    
    nsamples = 10**5 # 10**6
    E_l, I_rr_l, I_bibd_l = [], [], []
    for E in np.linspace(C, n*C, 7):
      print(">> E= {}".format(E) )
      E_l.append(E)
      
      I_rr = bci_rr.sim_min_bucketcap_forstability(E, nsamples)/(E/n)
      blog(I_rr=I_rr)
      I_rr_l.append(I_rr)
      
      
      I_bibd = bci_bibd.sim_min_bucketcap_forstability(E, nsamples)/(E/n)
      blog(I_bibd=I_bibd)
      I_bibd_l.append(I_bibd)
      
      # if I_rr < 0.01 or I_bibd < 0.01:
      #   break
    plot.plot(E_l, I_rr_l, label='Cyclic, $n= {}$, $d= {}$'.format(n, d), c=next(dark_color_c), marker=next(marker_c), ls=':', lw=3, mew=0.5, ms=7)
    plot.plot(E_l, I_bibd_l, label='Block design, $n= {}$, $d= {}$'.format(n, d), c=next(dark_color_c), marker=next(marker_c), ls=':', lw=3, mew=0.5, ms=7)
  
  plot_(d=3)
  plot_(d=5)
  
  
  def plot_(n, d_l, t):
    log(INFO, ">> n= {}, d_l= {}, t= {}".format(n, d_l, t) )
    k = n
    E = n*C*0.8
    
    nsamples = 10**4 # 10**5
    I_l = []
    for d in d_l:
      if t == 'cl':
        bci = BucketConfInspector_clustering(k, n, C, d)
        name = "Clustering"
      elif t == 'rr':
        bci = BucketConfInspector_roundrobin(k, n, C, d)
        name = "Cyclic"
      else:
        log(ERROR, "Unknown t= {}".format(t) )
        return
      
      I = bci.sim_min_bucketcap_forstability(E, nsamples)/(E/n)
      blog(I=I)
      I_l.append(I)
    plot.plot(d_l, I_l, label='{}, $n= {}$'.format(name, n), c=next(dark_color_c), marker=next(marker_c), ls=':', lw=3, mew=0.5, ms=7)
  
  plot_(n=9, d_l=[1, 3], t='cl')
  plot_(n=9, d_l=[1, 2, 3, 4], t='rr')
  
  plot_(n=20, d_l=[1, 2, 4, 5], t='cl')
  plot_(n=20, d_l=[1, 2, 3, 4, 5], t='rr')
  
  fontsize = 20
  plot.legend(loc='best', framealpha=0.5, fontsize=12)
  prettify(plot.gca() )
  plot.xlabel(r'$d$', fontsize=fontsize)
  plot.ylabel(r'$I$', fontsize=fontsize)
  plot.gcf().set_size_inches(5, 5)
  plot.savefig('compare_I_rr_vs_bibd.pdf', bbox_inches='tight')
  plot.gcf().clear()
  log(INFO, "done.")

def compare_I_rr_vs_rXORs():
  C = 1
  
  def plot_(n, d, r):
    log(INFO, "started; n= {}, d= {}".format(n, d) )
    k = n
    bci_rr = BucketConfInspector_roundrobin(k, n, C, d)
    
    if n == 7 and d == 3 and r == 2:
      bucket__objdesc_l_l = \
        [[((0, 1),), ((2, 1), (3, 1)) ],
         [((1, 1),), ((5, 1), (6, 1)) ],
         [((2, 1),), ((0, 1), (1, 1)) ],
         [((3, 1),), ((1, 1), (4, 1)) ],
         [((4, 1),), ((0, 1), (3, 1)) ],
         [((5, 1),), ((2, 1), (6, 1)) ],
         [((6, 1),), ((4, 1), (5, 1)) ] ]
    elif n == 8 and d == 2 and r == 2:
      bucket__objdesc_l_l = \
        [[((0, 1),), ((6, 1), (7, 1)) ],
         [((1, 1),) ],
         [((2, 1),), ((0, 1), (1, 1)) ],
         [((3, 1),) ],
         [((4, 1),), ((2, 1), (3, 1)) ],
         [((5, 1),) ],
         [((6, 1),), ((4, 1), (5, 1)) ],
         [((7, 1),) ] ]
    else:
      log(ERROR, "don't have bucket__objdesc_l_l for", n=n, d=d)
      return
    
    m, G, obj_bucket_m = get_m_G__obj_bucket_m(k, bucket__objdesc_l_l)
    log(INFO, "G=\n{}".format(pprint.pformat(list(G) ) ), m=m, obj_bucket_m=obj_bucket_m)
    bci_xor = BucketConfInspector_wCode(m, C, G, obj_bucket_m)
    
    nsamples = 10**3 # 10**5
    E_l, I_rr_l, I_xor_l = [], [], []
    for E in np.linspace(C, n*C, 7):
      print(">> E= {}".format(E) )
      E_l.append(E)
      
      I_rr = bci_rr.sim_min_bucketcap_forstability(E, nsamples)/(E/n)
      blog(I_rr=I_rr)
      I_rr_l.append(I_rr)
      
      I_xor = bci_xor.sim_min_bucketcap_forstability(E, nsamples)/(E/n)
      blog(I_xor=I_xor)
      I_xor_l.append(I_xor)
    
    plot.plot(E_l, I_rr_l, label='Cyclic, $n= {}$, $d= {}$'.format(n, d), c=next(dark_color_c), marker=next(marker_c), ls=':', lw=3, mew=0.5, ms=7)
    plot.plot(E_l, I_xor_l, label='{}-XOR, $n= {}$, $d= {}$'.format(r, n, d), c=next(dark_color_c), marker=next(marker_c), ls=':', lw=3, mew=0.5, ms=7)
  
  # plot_(n=7, d=3, r=2)
  plot_(n=8, d=2, r=2)
  
  fontsize = 20
  plot.legend(loc='best', framealpha=0.5, fontsize=12)
  prettify(plot.gca() )
  plot.xlabel(r'$\Sigma$', fontsize=fontsize)
  plot.ylabel(r'$I$', fontsize=fontsize)
  plot.gcf().set_size_inches(5, 5)
  plot.savefig('compare_I_rr_vs_rXORs.png', bbox_inches='tight')
  plot.gcf().clear()
  log(INFO, "done.")

if __name__ == "__main__":
  # I_wm()
  I_wd()
  # compare_I_clustering_vs_rr()
  # compare_I_rr_vs_bibd()
  # compare_I_rr_vs_rXORs()
