from cap_finder import *

def plot_capregion_blending_rep_coding():
  def plot_capregion(G, c, ls):
    cf = ConfInspector(G)
    for simplex in cf.hull.simplices:
      plot.plot(cf.points_inrows[simplex, 0], cf.points_inrows[simplex, 1], c=c, marker='.', ls=ls, lw=4)
  
  # a, a, a, a, b, b, b, b
  G = np.array([[1, 0], [0, 1], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1] ] ).T
  plot_capregion(G, NICE_BLUE, '-')
  # a, a, a, b, b, b, a+b, a+2b
  G = np.array([[1, 0], [0, 1], [1, 0], [1, 0], [0, 1], [0, 1], [1, 1], [1, 2] ] ).T
  plot_capregion(G, NICE_RED, '--')
  # a, b, a+b, a+2b, a+3b, a+4b, a+5b, a+6b
  G = np.array([[1, 0], [0, 1], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6] ] ).T
  plot_capregion(G, NICE_ORANGE, ':')
  # a+b, a+2b, a+3b, a+4b, a+5b, a+6b, a+7b, a+8b
  G = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8] ] ).T
  plot_capregion(G, 'purple', '-.')
  
  prettify(plot.gca() )
  plot.xlabel(r'$\lambda_a$', fontsize=20)
  plot.ylabel(r'$\lambda_b$', fontsize=20)
  fig = plot.gcf()
  fig.set_size_inches(3, 3)
  plot.savefig('plot_capregion_blending_rep_coding.png', bbox_inches='tight')
  fig.clear()
  log(INFO, "done.")
  
def plot_bsf_proposal():
  '''
  G_savename_l = [
    (np.array([
      [1, 0, 1, 0],
      [0, 1, 0, 1] ]), 'plot_cost_a-a-b-b'),
    (np.array([
      [1, 0, 1],
      [0, 1, 1] ]), 'plot_cost_a-b-a+b'),
    (np.array([
      [1, 0, 1, 1],
      [0, 1, 1, 2] ]), 'plot_cost_a-b-a+b-a+2b') ]
  for G, savename in G_savename_l:
    cf = ConfInspector(G)
    print("cf.to_sysrepr= {}".format(cf.to_sysrepr() ) )
    cf.plot_cost_2d(savename=savename)
  
  pm = PopModel_wZipf(k=2, zipf_tailindex_rv=TNormal(1, 2), arrate_rv=TNormal(1.5, 0.4) )
  G_savename_l = [
    (np.array([
      [1, 0, 1, 0],
      [0, 1, 0, 1] ]), 'plot_cost_wpop_a-a-b-b'),
    (np.array([
      [1, 0, 1],
      [0, 1, 1] ]), 'plot_cost_wpop_a-b-a+b'),
    (np.array([
      [1, 0, 1, 1],
      [0, 1, 1, 2] ]), 'plot_cost_wpop_a-b-a+b-a+2b') ]
  for G, savename in G_savename_l:
    cf = ConfInspector(G)
    print("cf.to_sysrepr= {}".format(cf.to_sysrepr() ) )
    cf.plot_cost_2d(pm, savename=savename)
  '''
  pm = PopModel_wZipf(k=2, zipf_tailindex_rv=TNormal(1, 2), arrate_rv=TNormal(1.5, 0.4) )
  G_savename_l = [
    (np.array([
      [1, 0, 1],
      [0, 1, 0] ]), 'plot_capregion_a-a-b_wheatmap'),
    (np.array([
      [1, 0, 1],
      [0, 1, 1] ]), 'plot_capregion_a-b-a+b_wheatmap'),
    (np.array([
      [1, 0, 1, 0],
      [0, 1, 0, 1] ]), 'plot_capregion_a-a-b-b_wheatmap') ]
  for G, savename in G_savename_l:
    cf = ConfInspector(G)
    print("cf.to_sysrepr= {}".format(cf.to_sysrepr() ) )
    cf.plot_servcap_2d(pm, savename=savename)

def plot_rep_vs_coding():
  def plot_(G, c, ls):
    cf = ConfInspector(G)
    print("cf.to_sysrepr= {}".format(cf.to_sysrepr() ) )
    cf.plot_servcap_2d(color=c, ls=ls)
  
  color_c = itertools.cycle((NICE_BLUE, 'black', NICE_RED))
  ls_c = itertools.cycle(('-', ':', '--'))
  
  G = mds_conf_matrix(2, k=2)
  plot_(G, next(color_c), next(ls_c) )
  G = custom_conf_matrix(4, k=2)
  plot_(G, next(color_c), next(ls_c) )
  G = mds_conf_matrix(4, k=2)
  plot_(G, next(color_c), next(ls_c) )
  
  plot.gca().legend(
    [matplotlib.lines.Line2D([0], [0], color=next(color_c), ls=next(ls_c), lw=3),
     matplotlib.lines.Line2D([0], [0], color=next(color_c), ls=next(ls_c), lw=3),
     matplotlib.lines.Line2D([0], [0], color=next(color_c), ls=next(ls_c), lw=3) ],
    ['[a, b]', '[a, b, a, b]', '[a, b, a+b, a+2b]'],
    loc='upper center', framealpha=0.5,
    bbox_to_anchor=(0.5, 1.35), ncol=2)
  plot.gcf().set_size_inches(3, 3)
  plot.savefig('plot_rep_vs_coding.png', bbox_inches='tight')
  plot.gcf().clear()
  log(INFO, "done.")

if __name__ == "__main__":
  # G = mds_conf_matrix(4, k=2)
  # G = custom_conf_matrix(4, k=2)
  # G = mds_conf_matrix(4, k=3)
  # cf = ConfInspector(G)
  # print("cf.to_sysrepr= {}".format(cf.to_sysrepr() ) )
  
  # r, M, C = cf.r_M_C()
  # cf.cap_boundary_point_l()
  
  # cf.plot_servcap_2d()
  # cf.plot_cost_2d()
  # pm = PopModel_wZipf(k=2, zipf_tailindex_rv=TNormal(1, 2), arrate_rv=TNormal(1.5, 0.4) )
  # cf.plot_servcap_2d(pm)
  # cf.plot_cost_2d(pm)
  
  plot_rep_vs_coding()