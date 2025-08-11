
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl 
from matplotlib.patches import Rectangle

TEXTWIDTH_FULL = 452
TEXTWIDTH = 221

tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 10,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
}

rcparams = {
    "text.usetex": True,
    # 'mathtext.default': 'regular',
    'text.latex.preamble': r'\usepackage{bm}\usepackage{amsmath}\usepackage{siunitx}'
            }    

def reset_params():
    plt.rcdefaults() 

def set_params(override_params=dict()):
    plt.rcParams.update(rcparams)  
    plt.rcParams.update(override_params)  

def metallic_mean(n=1):
    return (n+np.sqrt(n**2+4))/2

def set_size(width, height = None, fraction=1, n=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    if height is None:
        fig_height_in = fig_width_in /metallic_mean(n)
    else:
        fig_height_pt = height * fraction
        fig_height_in = fig_height_pt * inches_per_pt
    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim       


def set_astect_equal_to(ax2,ax1):
    asp = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
    asp /= np.abs(np.diff(ax1.get_xlim())[0] / np.diff(ax1.get_ylim())[0])
    ax2.set_aspect(asp)    

def remove_axis(axis, despine = False):
    axis.set_xticks([])
    axis.set_yticks([])
    if despine:
        for spine in axis.spines.values():
            # pass
            spine.set_visible(False)

def subplots(ncols=1,nrows=1,mode = 'column', aspect=1):
    textwidth = TEXTWIDTH_FULL if mode=='full' else TEXTWIDTH 
    print('size',set_size(textwidth,n=aspect))
    return plt.subplots(ncols=ncols, nrows = nrows, figsize=set_size(textwidth,n=aspect)) 

def add_border_to_axis(ax, pad = 0, **kwargs):
    # Border around axis

    autoAxis = ax.axis()
    p0 = (autoAxis[0]-pad,autoAxis[2]+pad)
    width = (autoAxis[1]-autoAxis[0])+2*pad
    height = (autoAxis[3]-autoAxis[2])-2*pad
    rec = Rectangle(p0,width, height,fill=False,**kwargs)
    rec = ax.add_patch(rec)
    rec.set_clip_on(False)