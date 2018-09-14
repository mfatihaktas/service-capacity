
from log_utils import *
from cap_finder import *
from popularity import *

if __name__ == "__main__":
  G = conf_mds_matrix(4, k=2)
  cf = CapFinder(G)
  
  pm = PopModel_wZipf(k=2, zipf_tailindex_rv=TNormal(1.2, 1), arrate_rv=TNormal(3, 1) )
  
  # for x in np.linspace(0, 2, 10):
  #   for y in np.linspace(0, 2, 10):
  #     p = pm.joint_pdf([x, y] )
  #     print("at [{}, {}], pdf= {}".format(x, y, p) )
  