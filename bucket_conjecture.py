from bucket_sim import *

def Pr_robust_lb(E, m, C, d):
  k = d
  x = m*d*C/E - math.log(m) - (k-1)*math.log(math.log(m)) + math.log(math.factorial(k-1))
  # x = m*C/E - math.log(m) - (k-1)*math.log(math.log(m)) + math.log(math.factorial(k-1))
  # log(INFO, "x= {}".format(x) )
  return math.exp(-math.exp(-x) )

def Pr_robust_ub(E, m, C, d):
  k = d
  x = m*(2*d-1)*C/E - math.log(m) - (k-1)*math.log(math.log(m)) + math.log(math.factorial(k-1))
  # log(INFO, "x= {}".format(x) )
  return math.exp(-math.exp(-x) )

def Pr_robust_conj(E, m, C, d):
  '''
  def get_x(i, k):
    return m*i*C/E - math.log(m) - (k-1)*math.log(math.log(m)) + math.log(math.factorial(k-1))
  
  # if d*C <= E <= (d+1)*C:
  #   return Pr_no_overflow_cont(E, m, d*C)
  # elif (d+1)*C < E <= (d+2)*C:
  # #   """
  # #   f0, f1 = d*C, (d-1)*C
  # #   E0, E1 = (d+1)*C, (d+2)*C
  # #   a = (f0-f1)/(E0-E1)
  # #   b = (f1*E0 - f0*E1)/(E0-E1)
  # #   return Pr_no_overflow_cont(E, m, a*E+b)
  # #   """
  #   d, k = d, 2
  
  # k = min(E//C - d, d)
  k = E//C - d
  i = d # max(d, k)
  log(INFO, "k= {}, i= {}".format(k, i) )
  x = get_x(i, k)
  log(INFO, "x= {}".format(x) )
  return math.exp(-math.exp(-x) )
  '''
  k = d
  x = m*(d+1)*C/E - math.log(m) - (k-1)*math.log(math.log(m)) + math.log(math.factorial(k-1))
  return math.exp(-math.exp(-x) )

def Pr_robust_lb2(E, m, C, d):
  G = min(d*C, E)
  if E - (m-1)*G <= 0:
    return max(0, 1 - m*(1 - G/E)**(m-1) )
  else:
    log(WARNING, "G > E/(m-1)!")
    return (1 - m*G/E)**(m-1)

def Pr_robust_ub2(E, m, C, d):
  G = (2*d-1)*C
  if E - (m-1)*G <= 0:
    return max(1 - m*(1 - G/E)**(m-1), 0)
  else:
    log(WARNING, "G > E/(m-1)!")
    return (1 - m*G/E)**(m-1)

def test_conjecture():
  # k, m, C = 5, 5, 1 # 50, 50, 5
  # d = 2
  k, m, C = 100, 100, 1
  # k, m, C = 10, 10, 1
  d = 4 # 2 # 10
  log(INFO, "k= {}, m= {}, C= {}, d= {}".format(k, m, C, d) )
  bci = BucketConfInspector_regularbalanced(k, m, C, d)
  E_l, sim_Pr_robust_l = [], []
  Pr_robust_ub_l, Pr_robust_lb_l, Pr_robust_conj_l = [], [], []
  Pr_robust_lb2_l, Pr_robust_ub2_l = [], []
  
  # for E in [*np.linspace(d*C, (d+1)*C, 2, endpoint=False), \
  #           *np.linspace((d+1)*C, (d+2)*C, 5) ]:
  # for E in np.linspace((d+1)*C, (d+2)*C, 5):
  # for E in [(d+2)*C]:
  # for E in [(d+1)*C, (d+2)*C, (d+3)*C, (d+4)*C]:
  # for E in [(d+i)*C for i in range(1, 10) ]:
  # for E in [(d+i)*C for i in range(30, 60) ]:
  # for E in np.logspace(math.log(C), math.log(m*C), 11):
  # for E in np.linspace(C, m*C, 11):
  for E in np.linspace(C, m*C, 11):
    print(">> E= {}".format(E) )
    E_l.append(E)
    sim_Pr_robust = 0 # bci.sim_frac_stable(cum_demand=E, nsamples=10**3)
    blog(sim_Pr_robust=sim_Pr_robust)
    sim_Pr_robust_l.append(sim_Pr_robust)
    
    _Pr_robust_lb = Pr_robust_lb(E, m, C, d)
    _Pr_robust_ub = Pr_robust_ub(E, m, C, d)
    _Pr_robust_conj = Pr_robust_conj(E, m, C, d)
    _Pr_robust_lb2 = Pr_robust_lb2(E, m, C, d)
    _Pr_robust_ub2 = Pr_robust_ub2(E, m, C, d)
    blog(_Pr_robust_lb=_Pr_robust_lb, _Pr_robust_ub=_Pr_robust_ub, _Pr_robust_conj=_Pr_robust_conj)
    blog(_Pr_robust_lb2=_Pr_robust_lb2, _Pr_robust_ub2=_Pr_robust_ub2)
    Pr_robust_lb_l.append(_Pr_robust_lb)
    Pr_robust_ub_l.append(_Pr_robust_ub)
    Pr_robust_conj_l.append(_Pr_robust_conj)
    Pr_robust_lb2_l.append(_Pr_robust_lb2)
    Pr_robust_ub2_l.append(_Pr_robust_ub2)
  plot.plot(E_l, sim_Pr_robust_l, label='Simulation', c='black', marker='o', ls=':', lw=2)
  plot.plot(E_l, Pr_robust_lb_l, label='Lower-bound', c=NICE_BLUE, marker='o', ls=':', lw=2)
  plot.plot(E_l, Pr_robust_ub_l, label='Upper-bound', c=NICE_RED, marker='o', ls=':', lw=2)
  plot.plot(E_l, Pr_robust_conj_l, label='Conjecture', c=NICE_ORANGE, marker='o', ls=':', lw=2)
  plot.plot(E_l, Pr_robust_lb2_l, label='Lower-bound-2', c=next(dark_color_c), marker='o', ls=':', lw=2)
  plot.plot(E_l, Pr_robust_ub2_l, label='Upper-bound-2', c=next(dark_color_c), marker='o', ls=':', lw=2)
  
  fontsize = 18
  plot.legend(loc='lower left', framealpha=0.5)
  plot.ylim([0, 1] )
  plot.title('$k= {}$, $m= {}$, $C= {}$, $d= {}$'.format(k, m, C, d), fontsize=fontsize)
  plot.xlabel(r'$\Sigma$', fontsize=fontsize)
  plot.ylabel(r'Pr{$\Sigma$-robustness}', fontsize=fontsize)
  plot.gcf().set_size_inches(5, 4)
  plot.savefig('plot_P_sim_lb_ub_conj.png', bbox_inches='tight')
  log(INFO, "done.")

if __name__ == "__main__":
  test_conjecture()
