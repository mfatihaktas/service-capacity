import numpy as np

def get_uspacings_l(k, E=1, nsamples=10**3):
  uspacings_l = []
  
  for _ in range(nsamples):
    U_ordered_l = sorted(np.random.uniform(size=(k-1, 1) ) )
    uspacings_l.append(
      np.array(
        [U_ordered_l[0]] + \
        [U_ordered_l[i+1] - U_ordered_l[i] for i in range(len(U_ordered_l)-1) ] + \
        [1 - U_ordered_l[-1]] ) * E)
  return uspacings_l
