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
  plot.xlim(right=21)
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
  
  def bd_bci(d):
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
      # https://rdrr.io/cran/ibd/man/bibd.html
      # bibd(v=13, b=13, r=4, k=4, lambda=1)
      #           [,1] [,2] [,3] [,4]
      # Block-1     1    2    5    7
      # Block-2     3    4    6    7
      # Block-3     3    5   10   11
      # Block-4     7    9   11   12
      # Block-5     7    8   10   13
      # Block-6     1    6    9   10
      # Block-7     4    5    8    9
      # Block-8     5    6   12   13
      # Block-9     2    4   10   12
      # Block-10    1    4   11   13
      # Block-11    2    3    9   13
      # Block-12    1    3    8   12
      # Block-13    2    6    8   11
      obj__bucket_l_m = {
        0: [0, 1, 4, 6],
        1: [2, 3, 5, 6],
        2: [2, 4, 9, 10],
        3: [6, 8, 10, 11],
        4: [6, 7, 9, 12],
        5: [0, 5, 8, 9],
        6: [3, 4, 7, 8],
        7: [4, 5, 11, 12],
        8: [1, 3, 9, 11],
        9: [0, 3, 10, 12],
        10: [1, 2, 8, 12],
        11: [0, 2, 7, 11],
        12: [1, 5, 7, 10] }
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
    elif d == 6:
      # bibd(v=31, b=31, r=6, k=6, lambda=1)
      #           [,1] [,2] [,3] [,4] [,5] [,6]
      # Block-1     2    3    5    6    9   17
      # Block-2     3    7   10   16   22   25
      # Block-3     3    8   11   21   30   31
      # Block-4     4    6   11   25   27   29
      # Block-5     4   14   17   20   21   22
      # Block-6     1   10   15   17   29   31
      # Block-7     6    8   10   20   23   24
      # Block-8     4    5    7    8   15   19
      # Block-9     5   10   11   12   14   18
      # Block-10    5   16   21   23   26   29
      # Block-11   17   18   19   23   25   30
      # Block-12    1    5   22   24   27   30
      # Block-13    5   13   20   25   28   31
      # Block-14    2   12   15   21   24   25
      # Block-15    2    7   14   23   27   31
      # Block-16    9   11   13   15   22   23
      # Block-17    9   10   19   21   27   28
      # Block-18    1    8    9   14   25   26
      # Block-19    8   12   13   16   17   27
      # Block-20    1    6    7   13   18   21
      # Block-21    7   11   17   24   26   28
      # Block-22    6   12   19   22   26   31
      # Block-23    4    9   16   18   24   31
      # Block-24    3   13   14   19   24   29
      # Block-25    1    3    4   12   23   28
      # Block-26    7    9   12   20   29   30
      # Block-27    6   14   15   16   28   30
      # Block-28    2    8   18   22   28   29
      # Block-29    2    4   10   13   26   30
      # Block-30    1    2   11   16   19   20
      # Block-31    3   15   18   20   26   27
      obj__bucket_l_m = {
        0: [1, 2, 4, 5, 8, 16],
        1: [2, 6, 9, 15, 21, 24],
        2: [2, 7, 10, 20, 29, 30],
        3: [3, 5, 10, 24, 26, 28],
        4: [3, 13, 16, 19, 20, 21],
        5: [0, 9, 14, 16, 28, 30],
        6: [5, 7, 9, 19, 22, 23],
        7: [3, 4, 6, 7, 14, 18],
        8: [4, 9, 10, 11, 13, 17],
        9: [4, 15, 20, 22, 25, 28],
        10: [16, 17, 18, 22, 24, 29],
        11: [0, 4, 21, 23, 26, 29],
        12: [4, 12, 19, 24, 27, 30],
        13: [1, 11, 14, 20, 23, 24],
        14: [1, 6, 13, 22, 26, 30],
        15: [8, 10, 12, 14, 21, 22],
        16: [8, 9, 18, 20, 26, 27],
        17: [0, 7, 8, 13, 24, 25],
        18: [7, 11, 12, 15, 16, 26],
        19: [0, 5, 6, 12, 17, 20],
        20: [6, 10, 16, 23, 25, 27],
        21: [5, 11, 18, 21, 25, 30],
        22: [3, 8, 15, 17, 23, 30],
        23: [2, 12, 13, 18, 23, 28],
        24: [0, 2, 3, 11, 22, 27],
        25: [6, 8, 11, 19, 28, 29],
        26: [5, 13, 14, 15, 27, 29],
        27: [1, 7, 17, 21, 27, 28],
        28: [1, 3, 9, 12, 25, 29],
        29: [0, 1, 10, 15, 18, 19],
        30: [2, 14, 17, 19, 25, 26] }
    else:
      log(ERROR, "not implemented for d= {}".format(d) )
      return None
    n = d**2 - d + 1
    return BucketConfInspector(n, C, obj__bucket_l_m)
  
  def plot_(d_l, t):
    log(INFO, ">> d_l= {}, t= {}".format(d_l, t) )
    
    nsamples = 10**4
    I_l = []
    for d in d_l:
      n = d**2 - d + 1
      k = n
      E = n*C*0.8
      if t == 'rr':
        bci = BucketConfInspector_roundrobin(k, n, C, d)
        name = "Cyclic"
      elif t == 'bd':
        bci = bd_bci(d)
        name = "Block design"
      else:
        log(ERROR, "Unknown t= {}".format(t) )
        return
      # blog(bci=bci)
      I = bci.sim_min_bucketcap_forstability(E, nsamples)/(E/n)
      # log(INFO, "E/n= {}".format(E/n), I=I)
      I_l.append(I)
    plot.plot(d_l, I_l, label=name, c=next(dark_color_c), marker=next(marker_c), ls=':', lw=3, mew=0.5, ms=7)
  
  # plot_(d_l=[3], t='rr')
  # plot_(d_l=[3], t='bd')
  plot_(d_l=[3, 4, 5, 6], t='rr')
  plot_(d_l=[3, 4, 5, 6], t='bd')
  
  fontsize = 20
  plot.legend(loc='best', framealpha=0.5, fontsize=12)
  prettify(plot.gca() )
  plot.xlabel(r'$d$', fontsize=fontsize)
  plot.ylabel(r'$I$', fontsize=fontsize)
  plot.title(r'$n = d^2 - d + 1$', fontsize=fontsize)
  plot.xlim(right=7)
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
  # I_wd()
  # compare_I_clustering_vs_rr()
  compare_I_rr_vs_bibd()
  # compare_I_rr_vs_rXORs()
