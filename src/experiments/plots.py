"""Figures F1--F3 for the selected solver settings."""
import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "t16-matplotlib"))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def make_plots(problems, selected, output, verified=None):
    methods=['GD','GD-BT','Newton pure','Newton damped','Momentum','Adam']
    colors = dict(zip(methods, [plt.get_cmap('tab10')(i) for i in range(len(methods))]))
    fig,ax=plt.subplots(figsize=(9,6))
    paths=[selected[('R1',m)][1][1] for m in methods if selected[('R1',m)][1] is not None]
    points=np.concatenate(paths)
    lo=points.min(axis=0)-.25;hi=points.max(axis=0)+.25
    xx,yy=np.meshgrid(np.linspace(lo[0],hi[0],500),np.linspace(lo[1],hi[1],500))
    zz=(1-xx)**2+100*(yy-xx**2)**2
    ax.contour(xx,yy,zz,levels=np.logspace(-1,3.5,20),colors='lightgray',linewidths=.6)
    for m in methods:
        out=selected[('R1',m)][1]
        if out is not None:ax.plot(out[1][:,0],out[1][:,1],label=m,color=colors[m],linewidth=1.2)
    ax.plot(1,1,'k*',markersize=10);ax.set(xlabel='$x_1$',ylabel='$x_2$',title='F1 Rosenbrock R1 trajectories')
    ax.legend();fig.tight_layout();fig.savefig(output/'F1.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
    for ax,p in zip(axes,problems[2:4]):
        curves(ax,p,selected,methods,colors)
    fig.tight_layout();fig.savefig(output/'F2.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5));curves(ax,problems[4],selected,methods,colors)
    fig.tight_layout();fig.savefig(output/'F3.png',dpi=180);plt.close(fig)
    if verified:
        fig,ax=plt.subplots(figsize=(9,5));curves(ax,problems[4],verified,methods,colors)
        ax.set_title('Project: supplementary verified-optimum comparison')
        fig.tight_layout();fig.savefig(output/'F3_verified_optimum.png',dpi=180);plt.close(fig)


def curves(ax,p,selected,methods,colors):
    for m in methods:
        row,out=selected[(p.name,m)]
        if out is None:continue
        norms=np.array([np.linalg.norm(p.grad(x)) for x in out[1]])
        label=m+(' [stationary, nonoptimal]' if row['optimum_verified'] is False else '')
        ax.semilogy(np.arange(len(norms)),np.maximum(norms,1e-18),label=label,color=colors[m],
                    linestyle='--' if m=='Newton pure' else '-',marker='o' if m.startswith('Newton') else None,markersize=4,
                    markerfacecolor='none' if m=='Newton pure' else colors[m],zorder=3 if m=='Newton pure' else 2)
    threshold=1e-6*np.linalg.norm(p.grad(p.x0)) if p.relative else 1e-6
    ax.axhline(threshold,color='black',linestyle=':',linewidth=.8,label='stopping threshold')
    ax.set(xlabel='Updates k',ylabel='Gradient norm',title=p.name)
    ax.grid(True,which='both',alpha=.2);ax.legend(fontsize=8)
