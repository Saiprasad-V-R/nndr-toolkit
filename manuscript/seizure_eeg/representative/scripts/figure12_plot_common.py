from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
DATA=PACKAGE/'data'; FIGURES=PACKAGE/'figures'
sys.path.insert(0,str(HERE))
from figure_style import configure_style, format_axis, panel_label, METHOD_STYLE
TICK_FONT_SIZE=7.0
AXIS_LABEL_SIZE=9.5
PANEL_LABEL_SIZE=10.5
LEGEND_FONT_SIZE=8.0
RAW_LINE_WIDTH=.58
SMOOTH_LINE_WIDTH=1.5
FIGURE_SIZE=(7.2,6.8)
COLORS={'M2':METHOD_STYLE['M2_OPT']['color'],'M3':METHOD_STYLE['M3_COMM']['color']}

def render(include_sigma=False, sigma_only=False, recording="chb01_03", focus=False):
    configure_style(font_size=9.5,axes_labelsize=AXIS_LABEL_SIZE,xtick_labelsize=TICK_FONT_SIZE,ytick_labelsize=TICK_FONT_SIZE,legend_fontsize=LEGEND_FONT_SIZE,savefig_dpi=550)
    w=pd.read_csv(DATA/f'{recording}_window_diagnostics.csv')
    eeg=pd.read_csv(DATA/f'{recording}_eeg_mean_field.csv')
    event=pd.read_csv(DATA/f'{recording}_event_metadata.csv').iloc[0]
    t=w.window_end_relative_s.to_numpy();te=eeg.time_relative_to_onset_s.to_numpy()
    fig,axs=plt.subplots(4,1,figsize=FIGURE_SIZE,sharex=True,gridspec_kw={'height_ratios':[1.05,1.08,1,1],'hspace':0.16})
    for ax,label in zip(axs,'abcd'):
        format_axis(ax,nbx=8,nby=5)
        ax.axvspan(0,event.seizure_duration_s,color='0.87',alpha=.7,zorder=0)
        for x in [0,event.seizure_duration_s]:ax.axvline(x,color='0.43',lw=.8,ls='--',zorder=1)
        panel_label(ax,label,x=-.08,y=1.01,fontsize=PANEL_LABEL_SIZE)
        ax.set_xlim((-120,65) if focus else ((-180,240) if recording=='chb01_04' else (-160,250)))
        ax.set_yscale('linear')
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5,steps=[1,2,2.5,5,10]))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    axs[0].plot(te,eeg.eeg_mean_field,color='0.12',lw=.62,rasterized=False)
    axs[0].set_ylabel(r'Mean EEG ($\mu$V)')
    axs[0].set_ylim(-250,250)
    axs[0].annotate('Onset',xy=(0,.82),xycoords=('data','axes fraction'),xytext=(-46,.82),textcoords=('data','axes fraction'),arrowprops={'arrowstyle':'->','lw':.85},fontsize=8,ha='center',va='center')
    def trace(ax,metric,method,ls='-'):
        col=COLORS[method]
        ax.plot(t,w[f'{metric}_{method}_raw'],color=col,lw=RAW_LINE_WIDTH,alpha=.38,ls=ls,zorder=2)
        ax.plot(t,w[f'{metric}_{method}_smoothed'],color=col,lw=SMOOTH_LINE_WIDTH,alpha=1,ls=ls,zorder=3)
    for method in ['M2','M3']:
        trace(axs[1],'Rsigma' if sigma_only else 'RA',method)
        if include_sigma and not sigma_only:trace(axs[1],'Rsigma',method,'--')
        if focus:
            trace(axs[2],'Rsigma',method)
        else:
            trace(axs[2],'support',method)
            trace(axs[3],'delta',method)
    axs[1].axhline(1,color='0.45',lw=.75,ls='--',zorder=1)
    axs[1].set_ylim(0,9 if sigma_only else 4)
    axs[1].set_ylabel(r'$R_\sigma$' if sigma_only else (r'$R_A,\ R_\sigma$' if include_sigma else r'$R_A$'))
    if focus:
        axs[2].axhline(1,color='0.45',lw=.75,ls='--',zorder=1)
        axs[2].set_ylabel(r'$R_\sigma$')
        axs[2].set_ylim(0,9)
        axs[3].plot(t,w['kappa_eig_full_raw'],color='0.5',lw=.58,alpha=.5)
        axs[3].plot(t,w['kappa_eig_full_smoothed'],color='0.12',lw=1.5)
        axs[3].set_ylabel(r'$\kappa_{\mathrm{eig}}(F)$')
        axs[3].set_ylim(0,450)
    else:
        axs[2].set_ylabel('Reaction support')
        axs[2].set_ylim(0,.8)
        axs[3].set_ylabel(r'$\Delta$')
        axs[3].set_ylim(0,1)
    axs[3].set_xlabel('Window end time from seizure onset (s)')
    method_handles=[Line2D([0],[0],color=COLORS[m],lw=1.5,label=f'{m} '+('optimization' if m=='M2' else 'commutator')) for m in ['M2','M3']]
    if include_sigma and not focus:
        method_handles += [Line2D([0],[0],color='0.25',lw=1.5,ls='-',label=r'$R_A$'),Line2D([0],[0],color='0.25',lw=1.5,ls='--',label=r'$R_\sigma$')]
    fig.legend(handles=method_handles,loc='lower center',bbox_to_anchor=(.5,.015),ncol=4 if include_sigma and not focus else 2,frameon=False,handlelength=2.7,columnspacing=1.6)
    fig.subplots_adjust(left=.12,right=.985,top=.975,bottom=.12)
    stem=(f'figure_{recording}_RA_Rsigma_kappa_focus' if focus else (f'figure12_{recording}_Rsigma_supp' if sigma_only else (f'figure12_{recording}_RA_Rsigma_supp' if include_sigma else f'figure12_{recording}_RA_main')))
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES/f'{stem}.pdf')
    fig.savefig(FIGURES/f'{stem}.png',dpi=550)
    plt.close(fig)
if __name__=='__main__':render()
