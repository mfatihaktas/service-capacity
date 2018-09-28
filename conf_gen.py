import numpy as np

def mds_conf_matrix(n, k=2):
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
      G[0, j] = 1
      G[1, j] = j-1
  return G

def custom_conf_matrix(n, k=2):
  if n == 4:
    # G = np.array([
    #   [1, 0, 1, 0],
    #   [0, 1, 0, 1] ])
    G = np.array([
      [1, 0, 1, 1],
      [0, 1, 1, 0] ])
  elif n == 3:
    G = np.array([
      [1, 0, 1],
      [0, 1, 1] ])
  elif n == 2:
    G = np.array([
      [1, 0],
      [0, 1] ])
  return G
