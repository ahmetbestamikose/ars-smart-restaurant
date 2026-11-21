"""Confusion-matrix figures of the article (Figures 4 and 5).

Usage:  python evaluation/plot_confusion.py --results evaluation/results --out docs/figures
"""
import argparse, os
import json
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif'],'font.size':11})
nice=['Anger','Contempt','Disgust','Fear','Happiness','Sadness','Surprise']
def plot(cm,out):
    cm=np.array(cm); cmn=cm/cm.sum(1,keepdims=True)
    fig,ax=plt.subplots(figsize=(5.6,4.7))
    im=ax.imshow(cmn,cmap='Greys',vmin=0,vmax=1)
    for i in range(7):
        for j in range(7):
            if cm[i,j]: ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=11,color='white' if cmn[i,j]>0.55 else 'black')
    ax.set_xticks(range(7)); ax.set_yticks(range(7))
    ax.set_xticklabels(nice,rotation=40,ha='right'); ax.set_yticklabels(nice)
    ax.set_xlabel('Predicted class'); ax.set_ylabel('True class')
    cb=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04); cb.set_label('Row-normalised share')
    fig.tight_layout(); fig.savefig(out+'.pdf'); fig.savefig(out+'.png',dpi=300)
# thesis run (read from the notebook's confusion-matrix figure, 825 validation images)
A=[[118,0,0,0,0,4,0],[0,112,0,0,1,0,0],[1,0,112,0,1,0,1],[0,0,0,105,0,0,0],[0,0,0,0,133,0,0],[0,0,0,0,0,122,0],[0,0,0,1,0,0,114]]
assert np.trace(A)==816 and np.sum(A)==825
ap = argparse.ArgumentParser(); ap.add_argument('--results', default='evaluation/results'); ap.add_argument('--out', default='docs/figures')
args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
plot(A, os.path.join(args.out, 'fig4_protocolA_cm'))
plot(json.load(open(os.path.join(args.results, 'B_summary.json')))['pooled_cm'], os.path.join(args.out, 'fig5_protocolB_cm'))
