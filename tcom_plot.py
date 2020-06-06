from cap_finder import *
# from popularity import *

if __name__ == "__main__":
  popmodel = PopModel_wZipf(k=2, zipf_tailindex_rv=TNormal(1, 2), arrate_rv=TNormal(2, 0.4) )
  heatmap_grid_xmax_ymax = gen_pop_heatmap_grid_xmax_ymax(popmodel)
  
  def plot_(G, savename):
    cf = ConfInspector(G)
    print("cf.to_sysrepr= {}".format(cf.to_sysrepr() ) )
    cf.plot_servcap_2d(popmodel, heatmap_grid_xmax_ymax, savename)
  
  plot_(G=custom_conf_matrix(4, k=2), savename='plot_servcap_rep42.png')
  plot_(G=mds_conf_matrix(4, k=2), savename='plot_servcap_mds42.png')
