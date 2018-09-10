import numpy as np

def conf_mds_matrix(n, k=2):
  if k != 2:
    log(ERROR, "Unexpected k;", k=k)
    return None
  
  G = np.zeros((2, n))
  for j in range(n):
    if j == 0:
      G[0, j] = 1
      G[1, j] = 0
    elif j == 1:
      G[0, j] = 0
      G[1, j] = 1
    else:
      G[0, j] = j-1
      G[1, j] = 1
  return G