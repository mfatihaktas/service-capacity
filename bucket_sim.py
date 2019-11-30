from log_utils import *
from bucket_wchoice import *

def plot_P_Sigma_sim_vs_model():
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
  plot.savefig('plot_P_Sigma_sim_vs_model.png', bbox_inches='tight')
  log(INFO, "done; m= {}, C= {}".format(m, C) )

def plot_P_Sigma_wchoice():
  k, n, C = 100, 100, 1
  log(INFO, "k= {}, n= {}, C= {}".format(k, n, C) )
  
  def plot_(d):
    log(INFO, "d= {}".format(d) )
    bci = BucketConfInspector_roundrobin(k, n, C, d)
    E_l, sim_P_Sigma_l = [], []
    Pr_robust_l, Pr_robust_w_naivesplit_l = [], []
    for E in np.linspace(C, (n+1)*C, 20):
    # for E in np.linspace((d-1)*C, d*C, 10):
    # C_min = C*n/(2*math.log(math.log(n)) + math.log(n) )
    # for E in np.linspace(d*C_min, (d-1)*C_min, 10):
      print(">> E= {}".format(E) )
      E_l.append(E)
      sim_Pr_robust = bci.sim_frac_stable(cum_demand=E, nsamples=10**4) # 10**4
      blog(sim_Pr_robust=sim_Pr_robust)
      sim_P_Sigma_l.append(sim_Pr_robust)
      # Pr_robust_w_naivesplit = bci.sim_frac_stable(cum_demand=E, w_naivesplit=True)
      # blog(Pr_robust_w_naivesplit=Pr_robust_w_naivesplit)
      
      # frac_stable = bci.frac_stable(E)
      # blog(frac_stable=frac_stable)
      
      # Pr_robust = Pr_no_overflow_wchoice_cont(E, n, C, d)
      # blog(Pr_robust=Pr_robust)
      # Pr_robust_l.append(Pr_robust)
      # # Pr_robust_w_naivesplit_l.append(Pr_robust_w_naivesplit)
      
      if sim_Pr_robust < 0.01:
      # if sim_Pr_robust >= 0.99:
      #   print("*** sim_Pr_robust >= 0.99 when E/C_min <= {}".format(E/C_min) )
        break
    plot.plot(E_l, sim_P_Sigma_l, label='d={}'.format(d), c=next(dark_color_c), marker=next(marker_c), ls=':', lw=3, mew=0.5, ms=7)
    # plot.plot(E_l, sim_P_Sigma_l, label='Sim, d={}'.format(d), c=next(dark_color_c), marker=next(marker_c), ls=':', lw=3, mew=2, ms=7)
    # plot.plot(E_l, Pr_robust_l, label='d={}'.format(d), c=next(dark_color_c), marker=next(marker_c), ls=':', lw=3, mew=2, ms=7)
    # plot.plot(E_l, Pr_robust_w_naivesplit_l, label='d={}, w/ naive split'.format(d), c=next(dark_color_c), marker=next(marker_c), ls=':', lw=3, mew=0, ms=7)
  
  # plot_(d=1)
  # plot_(d=2)
  # plot_(d=3)
  # plot_(d=4)
  # plot_(d=8)
  
  # for c in range(2, math.ceil(math.log(n)) + 2):
  #   plot_(d=c)
  
  # for c in range(1, n+1):
  # plot_(d=1)
  # plot_(d=2)
  
  # plot_(d=1)
  for c in range(1, 8):
    plot_(d=c)
  plot_(d=10)
  plot_(d=12)
  
  fontsize = 20
  plot.legend(loc='lower left', framealpha=0.5, fontsize=14)
  prettify(plot.gca() )
  plot.xlim([0, n*C*1.05] )
  plot.ylim([0, 1.1] )
  # plot.title('k= {}, n= {}, C= {}'.format(k, n, C), fontsize=fontsize)
  plot.title(r'$k= {}$, $n= {}$'.format(k, n), fontsize=fontsize)
  plot.xlabel(r'$\Sigma$', fontsize=fontsize)
  # plot.ylabel('Pr{$\Sigma$-robustness}', fontsize=fontsize)
  plot.ylabel(r'$P_{\Sigma}$', fontsize=fontsize)
  plot.gcf().set_size_inches(7, 5)
  plot.savefig('plot_P_Sigma_wrt_d_k{}_n{}.pdf'.format(k, n), bbox_inches='tight')
  # plot.gcf().clear()
  log(INFO, "done; n= {}, C= {}".format(n, C) )

def compare_P_Sigma_wrt_overlaps():
  choice = 3
  C = 1
  m = k = 9
  bci_roundrobin = BucketConfInspector_roundrobin(k, m, C, choice)
  
  obj__bucket_l_m = {
    0: [0, 1, 2],
    1: [0, 1, 2],
    2: [0, 1, 2],
    3: [3, 4, 5],
    4: [3, 4, 5],
    5: [3, 4, 5],
    6: [6, 7, 8],
    7: [6, 7, 8],
    8: [6, 7, 8] }
  bci_clustering = BucketConfInspector(m, C, obj__bucket_l_m)
  
  # obj__bucket_l_m = {
  #   0: [0, 8, 3],
  #   1: [1, 0, 4],
  #   2: [2, 1, 5],
  #   3: [3, 4, 6],
  #   4: [4, 3, 7],
  #   5: [5, 7, 8],
  #   6: [6, 5, 1],
  #   7: [7, 6, 2],
  #   8: [8, 2, 0] }
  obj__bucket_l_m = {
    0: [0, 1, 2],
    1: [0, 1, 2],
    2: [3, 1, 2],
    3: [3, 4, 5],
    4: [3, 4, 5],
    5: [0, 4, 5],
    6: [6, 7, 8],
    7: [6, 7, 8],
    8: [6, 7, 8] }
  bci_roundrobin_wless_overlap = BucketConfInspector(m, C, obj__bucket_l_m)
  
  E_l, Pr_robust_rr_l, Pr_robust_cl_l, Pr_robust_rr_wlessoverlap_l = [], [], [], []
  for E in np.linspace(C, (m+1)*C, 15): # 20
    print(">> E= {}".format(E) )
    E_l.append(E)
    
    Pr_robust_rr = bci_roundrobin.sim_frac_stable(cum_demand=E, nsamples=2*10**3)
    blog(Pr_robust_rr=Pr_robust_rr)
    Pr_robust_rr_l.append(Pr_robust_rr)
    
    Pr_robust_clustering = bci_clustering.sim_frac_stable(cum_demand=E, nsamples=2*10**3)
    blog(Pr_robust_clustering=Pr_robust_clustering)
    Pr_robust_cl_l.append(Pr_robust_clustering)
    
    Pr_robust_rr_wlessoverlap = bci_roundrobin_wless_overlap.sim_frac_stable(cum_demand=E, nsamples=2*10**3)
    blog(Pr_robust_rr_wlessoverlap=Pr_robust_rr_wlessoverlap)
    Pr_robust_rr_wlessoverlap_l.append(Pr_robust_rr_wlessoverlap)
    
    if Pr_robust_rr < 0.01:
      break
  plot.plot(E_l, Pr_robust_rr_l, label='Round-robin', c=next(dark_color_c), marker=next(marker_c), ls=':', lw=3, mew=0.5, ms=7)
  plot.plot(E_l, Pr_robust_cl_l, label='Clustering', c=next(dark_color_c), marker=next(marker_c), ls=':', lw=3, mew=0.5, ms=7)
  plot.plot(E_l, Pr_robust_rr_wlessoverlap_l, label='Round-robin w/ less overlap', c=next(dark_color_c), marker=next(marker_c), ls=':', lw=3, mew=0.5, ms=7)
  
  fontsize = 20
  plot.legend(loc='lower left', framealpha=0.5, fontsize=14)
  plot.ylim([0, 1] )
  prettify(plot.gca() )
  plot.title('k= {}, m= {}, C= {}'.format(k, m, C), fontsize=fontsize)
  plot.xlabel(r'$\Sigma$', fontsize=fontsize)
  plot.ylabel('Pr{$\Sigma$-robustness}', fontsize=fontsize)
  plot.gcf().set_size_inches(7, 5)
  plot.savefig('compare_P_Sigma_wrt_overlaps_k{}_m{}_C{}.png'.format(k, m, C), bbox_inches='tight')
  # plot.gcf().clear()
  log(INFO, "done; m= {}, C= {}".format(m, C) )

def compare_P_Sigma_fano_vs_roundrobin():
  choice = 3
  C = 1
  m = k = 7
  bci_roundrobin = BucketConfInspector_roundrobin(k, m, C, choice)
  
  obj__bucket_l_m = {
    0: [0, 1, 2],
    1: [0, 3, 4],
    2: [0, 5, 6],
    3: [2, 3, 5],
    4: [2, 4, 6],
    5: [1, 3, 6],
    6: [1, 4, 5] }
  bci_fano = BucketConfInspector(m, C, obj__bucket_l_m)
  
  E_l, Pr_robust_rr_l, Pr_robust_fano_l = [], [], []
  for E in np.linspace(C, (m+1)*C, 15): # 20
    print(">> E= {}".format(E) )
    E_l.append(E)
    
    Pr_robust_rr = bci_roundrobin.sim_frac_stable(cum_demand=E, nsamples=2*10**3)
    blog(Pr_robust_rr=Pr_robust_rr)
    Pr_robust_rr_l.append(Pr_robust_rr)
    
    Pr_robust_fano = bci_fano.sim_frac_stable(cum_demand=E, nsamples=2*10**3)
    blog(Pr_robust_fano=Pr_robust_fano)
    Pr_robust_fano_l.append(Pr_robust_fano)
    
    if Pr_robust_rr < 0.01 or Pr_robust_fano < 0.01:
      break
  plot.plot(E_l, Pr_robust_rr_l, label='Round-robin', c=next(dark_color_c), marker=next(marker_c), ls=':', lw=3, mew=0.5, ms=7)
  plot.plot(E_l, Pr_robust_fano_l, label='Fano', c=next(dark_color_c), marker=next(marker_c), ls=':', lw=3, mew=0.5, ms=7)
  
  fontsize = 20
  plot.legend(loc='lower left', framealpha=0.5, fontsize=14)
  plot.ylim([0, 1] )
  prettify(plot.gca() )
  plot.title('k= {}, m= {}, C= {}'.format(k, m, C), fontsize=fontsize)
  plot.xlabel(r'$\Sigma$', fontsize=fontsize)
  plot.ylabel('Pr{$\Sigma$-robustness}', fontsize=fontsize)
  plot.gcf().set_size_inches(7, 5)
  plot.savefig('compare_P_Sigma_fano_vs_roundrobin_k{}_m{}_C{}.png'.format(k, m, C), bbox_inches='tight')
  # plot.gcf().clear()
  log(INFO, "done; m= {}, C= {}".format(m, C) )

def compare_P_Sigma_singlechoice_many_vs_few_objs():
  choice = 1
  C = 1
  k = m = 7
  bci_1obj_pernode = BucketConfInspector_roundrobin(k, m, C, choice)
  
  k = 3*m
  bci_3obj_pernode = BucketConfInspector_roundrobin(k, m, C, choice)
  
  E_l, Pr_robust_1obj_l, Pr_robust_3obj_l = [], [], []
  for E in np.linspace(C, (m+1)*C, 15): # 20
    print(">> E= {}".format(E) )
    E_l.append(E)
    
    Pr_robust_1obj = bci_1obj_pernode.sim_frac_stable(cum_demand=E, nsamples=10**3)
    blog(Pr_robust_1obj=Pr_robust_1obj)
    Pr_robust_1obj_l.append(Pr_robust_1obj)
    
    Pr_robust_3obj = bci_3obj_pernode.sim_frac_stable(cum_demand=E, nsamples=10**3)
    blog(Pr_robust_3obj=Pr_robust_3obj)
    Pr_robust_3obj_l.append(Pr_robust_3obj)
    
    if Pr_robust_1obj < 0.01 or Pr_robust_3obj < 0.01:
      break
  plot.plot(E_l, Pr_robust_1obj_l, label='1 obj/node', c=next(dark_color_c), marker=next(marker_c), ls=':', lw=3, mew=0.5, ms=7)
  plot.plot(E_l, Pr_robust_3obj_l, label='3 obj/node', c=next(dark_color_c), marker=next(marker_c), ls=':', lw=3, mew=0.5, ms=7)
  
  fontsize = 20
  plot.legend(loc='lower left', framealpha=0.5, fontsize=14)
  plot.ylim([0, 1] )
  prettify(plot.gca() )
  plot.title('m= {}, C= {}'.format(m, C), fontsize=fontsize)
  plot.xlabel(r'$\Sigma$', fontsize=fontsize)
  plot.ylabel('Pr{$\Sigma$-robustness}', fontsize=fontsize)
  plot.gcf().set_size_inches(7, 5)
  plot.savefig('compare_P_Sigma_singlechoice_many_vs_few_objs_m{}_C{}.png'.format(m, C), bbox_inches='tight')
  # plot.gcf().clear()
  log(INFO, "done; m= {}, C= {}".format(m, C) )

def checking_Conjecture_following_Godfrey_claim():
  # Claim: as soon as the number of choice is \Omega(log(m)) in a regular balanced setting,
  # throwing m balls into m urns will result in O(1) maximum load w.h.p.
  # [Godfrey, Balls and Bins with Structure: Balanced Allocations on Hypergraphs]
  # (Our) Conjecture: Pouring E amount of water over m buckets set up with a regular balanced setting 
  # with log(m) choice, maximum load will be E/m + O(1) w.h.p.
  k = 400
  log(INFO, "k= {}".format(k) )
  
  def test_conjecture(m, E):
    choice = math.ceil(math.log(m))
    C = E/m*math.log(2*math.log(math.log(m)) + math.log(m) ) # E/m*math.sqrt(2*math.log(math.log(m)) + math.log(m) ) # + 1 # E/m + math.sqrt(E/m) # 2*E/m
    bci = BucketConfInspector_roundrobin(k, m, C, choice)
    Pr_robust = bci.sim_frac_stable(E)
    print("m= {}, E= {}, C= {}, choice= {}, Pr_robust= {}".format(m, E, C, choice, Pr_robust) )
  
  def test(m):
    # for i in range(8):
    # for i in range(7, 15):
    # for i in range(11, 15):
    for i in range(12):
      test_conjecture(m, E=2**i*100)
  
  # test(m=10)
  test(m=50)
  # test(m=100)
  
  log(INFO, "done; k= {}".format(k) )

if __name__ == "__main__":
  # m, C = 2, 10
  # obj__bucket_l_m = {0: [0], 1: [0], 2: [1], 3: [1] }
  # bci = BucketConfInspector(m, C, obj__bucket_l_m)
  # bci.sim_frac_stable(cum_demand=15)
  # print("bci= {}".format(bci) )
  
  # plot_P_Sigma_sim_vs_model()
  plot_P_Sigma_wchoice()
  # compare_P_Sigma_fano_vs_roundrobin()
  # compare_P_Sigma_wrt_overlaps()
  # compare_P_Sigma_singlechoice_many_vs_few_objs()
  # checking_Conjecture_following_Godfrey_claim()
