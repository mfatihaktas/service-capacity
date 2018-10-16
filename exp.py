from log_utils import *
from cap_finder import *
from popularity import *

if __name__ == "__main__":
  pm = PopModel_wZipf(k=2, zipf_tailindex_rv=TNormal(1.2, 1), arrate_rv=TNormal(1.5, 1) )
  
  # for x in np.linspace(0, 2, 10):
  #   for y in np.linspace(0, 2, 10):
  #     p = pm.joint_pdf(x, y)
  #     print("at [{}, {}], pdf= {}".format(x, y, p) )
  
  # for n in range(3, 10):
  #   G = mds_conf_matrix(n, k=2)
  #   ci = ConfInspector(G)
  #   covered_pdf_mass = ci.integrate_jointpdf_overcaphyperlane(pm.joint_pdf)
  #   print("n= {}, covered_pdf_mass= {}".format(n, covered_pdf_mass) )
  
  G = mds_conf_matrix(4, k=2)
  cf = ConfInspector(G)
  cf.plot_all_2d(pm)
  # EC = cf.moment_cost(pm.joint_pdf, i=1)
  # blog(EC=EC)
