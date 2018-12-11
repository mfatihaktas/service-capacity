from log_utils import *
from bucket_wchoice import *

def plot_Pr_robust_sim_vs_model():
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
  plot.savefig('plot_Pr_robust_sim_vs_model.png', bbox_inches='tight')
  log(INFO, "done; m= {}, C= {}".format(m, C) )

def plot_Pr_robust_wchoice():
  # k, m, C = 3, 3, 5
  # k, m, C = 10, 10, 5
  k, m, C = 50, 50, 5
  log(INFO, "k= {}, m= {}, C= {}".format(k, m, C) )
  
  def plot_(choice):
    log(INFO, "choice= {}".format(choice) )
    bci = BucketConfInspector_regularbalanced(k, m, C, choice)
    E_l, sim_Pr_robust_l = [], []
    Pr_robust_l, Pr_robust_w_naivesplit_l = [], []
    # for E in np.linspace(C, (m+1)*C, 10):
    # for E in np.linspace((choice-1)*C, choice*C, 10):
    C_min = C*m/(2*math.log(math.log(m)) + math.log(m) )
    for E in np.linspace(choice*C_min, (choice-1)*C_min, 10):
      print(">> E= {}".format(E) )
      E_l.append(E)
      sim_Pr_robust = bci.sim_frac_stable(cum_demand=E, nsamples=10**3)
      blog(sim_Pr_robust=sim_Pr_robust)
      sim_Pr_robust_l.append(sim_Pr_robust)
      # Pr_robust_w_naivesplit = bci.sim_frac_stable(cum_demand=E, w_naivesplit=True)
      # blog(Pr_robust_w_naivesplit=Pr_robust_w_naivesplit)
      
      # frac_stable = bci.frac_stable(E)
      # blog(frac_stable=frac_stable)
      
      Pr_robust = Pr_no_overflow_wchoice_cont(E, m, C, choice)
      blog(Pr_robust=Pr_robust)
      Pr_robust_l.append(Pr_robust)
      # Pr_robust_w_naivesplit_l.append(Pr_robust_w_naivesplit)
      
      # if sim_Pr_robust < 0.1:
      if sim_Pr_robust >= 0.99:
        print("*** sim_Pr_robust >= 0.99 when E/C_min <= {}".format(E/C_min) )
        break
    plot.plot(E_l, sim_Pr_robust_l, label='Sim, d={}'.format(choice), c=next(dark_color_c), marker=next(marker_c), ls=':', lw=3, mew=0.5, ms=7)
    plot.plot(E_l, Pr_robust_l, label='d={}'.format(choice), c=next(dark_color_c), marker=next(marker_c), ls=':', lw=3, mew=0.5, ms=7)
    # plot.plot(E_l, Pr_robust_w_naivesplit_l, label='d={}, w/ naive split'.format(choice), c=next(dark_color_c), marker=next(marker_c), ls=':', lw=3, mew=0, ms=7)
  
  # plot_(choice=1)
  # plot_(choice=2)
  # plot_(choice=3)
  # plot_(choice=4)
  # plot_(choice=8)
  
  for c in range(2, math.ceil(math.log(m)) + 2):
    plot_(choice=c)
  
  # for c in range(1, m+1):
  # plot_(choice=1)
  # plot_(choice=2)
  
  # plot_(choice=1)
  # for c in range(1, 8):
  #   plot_(choice=c)
  # plot_(choice=10)
  # plot_(choice=m)
  
  fontsize = 18
  plot.legend(loc='lower left', framealpha=0.5, fontsize=fontsize)
  plot.ylim([0, 1] )
  plot.title('k= {}, m= {}, C= {}'.format(k, m, C), fontsize=fontsize)
  plot.xlabel(r'$\Sigma$', fontsize=fontsize)
  plot.ylabel('Pr{$\Sigma$-robustness}', fontsize=fontsize)
  # plot.gcf().set_size_inches(8, 7)
  plot.savefig('plot_Pr_robust_wchoice_k{}_m{}_C{}.png'.format(k, m, C), bbox_inches='tight')
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
    bci = BucketConfInspector_regularbalanced(k, m, C, choice)
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
  
  # plot_Pr_robust_sim_vs_model()
  plot_Pr_robust_wchoice()
  # checking_Conjecture_following_Godfrey_claim()
