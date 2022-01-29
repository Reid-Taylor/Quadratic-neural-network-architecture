import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

qEpochs = pd.read_csv('quadEpochs.csv')
epochs = pd.read_csv('epochs.csv')
bestQEpochs = pd.DataFrame()
bestEpochs = pd.DataFrame()
time = []
qTime = []

qEpochs['training_error'] = 1-qEpochs['training_error']
epochs['training_error'] = 1-epochs['training_error']
attempt =[]
epoch=[]
training_error=[]
nonzero_params=[]
delta_time=[]

for i in range(51):
    index = qEpochs.loc[qEpochs['training_error'] == qEpochs[qEpochs['attempt'] == i]['training_error'].max()].values
    try: 
        diction = {'attempt': index[0][0], 'epoch':index[0][1], 'training_error':index[0][2], 'nonzero_params':index[0][3], 'delta_time':index[0][4]}
        bestQEpochs= bestQEpochs.append(diction, ignore_index=True)
    except:
        pass
    index = epochs.loc[epochs['training_error'] == epochs[epochs['attempt'] == i]['training_error'].max()].values
    try: 
        diction = {'attempt': index[0][0], 'epoch':index[0][1], 'training_error':index[0][2], 'nonzero_params':index[0][3], 'delta_time':index[0][4]}
        bestEpochs= bestEpochs.append(diction, ignore_index=True)
    except:
        pass
    # bestEpochs.append(epochs.loc[epochs['training_error'] == epochs[epochs['attempt'] == i]['training_error'].max()])
    time.append(epochs[epochs['attempt']==i]['delta_time'].sum())
    qTime.append(qEpochs[qEpochs['attempt']==i]['delta_time'].sum())
# bestQEpochs['attempt']=attempt
bestQEpochs['total_time'] = qTime
bestEpochs['total_time'] = time

gain = []
qGain = []
for x in range(len(epochs['training_error'])):
    if epochs.iloc[x,1] == 0:
        dog = 0.0
    else:
        dog = epochs.iloc[x,2] - epochs.iloc[x-1,2]
    gain.append(dog)
    if qEpochs.iloc[x,1] == 0:
        dog = 0.0
    else:
        dog = qEpochs.iloc[x,2] - qEpochs.iloc[x-1,2]
    qGain.append(dog)
epochs['gain'] = gain
qEpochs['gain'] = qGain

bestQEpochs.reset_index()
bestEpochs.reset_index()

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))

#make color green for quad and red for conventional for all plots
axes[0,0].plot(epochs['epoch'].sort_values(), epochs['training_error'].sort_values(), color='r', label='Conventional')
axes[0,0].plot(qEpochs['epoch'].sort_values(), qEpochs['training_error'].sort_values(), color='g', label='Quadratic')
axes[0,0].set_title('Accuracy over Epochs')
axes[0,0].set_ylabel('Accuracy (%)')
axes[0,0].set_xlabel('Epoch')
# axes[1,0].set_xlim([0,50])
# axes[1,0].set_xticks(np.arange(0,51, step=5))

#make color green for quad and red for conventional for all plots
axes[0,1].hist(bestEpochs['training_error'], alpha=0.5, color='r', label='Conventional')
axes[0,1].hist(bestQEpochs['training_error'], alpha=0.5, color='g', label='Quadratic')
axes[0,1].set_title('Distribution of Optimal Accuracy\nby Class (Histogram)')
axes[0,1].set_ylabel('Count')
axes[0,1].set_xlabel('Accuracy (%)')
# axes[2,0].set_xlim([0,50])
# axes[2,0].set_xticks(np.arange(0,51, step=5))

#make color green for quad and red for conventional for all plots
axes[1,0].scatter(bestQEpochs['total_time']/60, bestQEpochs['training_error'], color='g',label='Quadratic')
axes[1,0].scatter(bestEpochs['total_time']/60, bestEpochs['training_error'], color='r', label='Conventional')
axes[1,0].set_title('Optimal Performance and\nTotal Elapsed Time')
axes[1,0].set_ylabel('Optimal Accuracy\nDuring Training')
axes[1,0].set_xlabel('Total Training Time (Minutes)')
# axes[3,0].set_xlim([0,60])
# axes[3,0].set_xticks(np.arange(0,61, step=5))

#make color green for quad and red for conventional for all plots
axes[1,1].scatter(epochs['delta_time'], epochs['gain'], color='r', label='Conventional')
axes[1,1].scatter(qEpochs['delta_time'], qEpochs['gain'], color='g',label='Quadratic')
axes[1,1].set_title('Changes to Accuracy per Epoch')
axes[1,1].set_ylabel('Gain in Accuracy')
axes[1,1].set_xlabel('Epoch Training Duration (Seconds)')
# axes[3,1].set_xlim([0,5])
# axes[3,1].set_xticks(np.arange(0,6, step=.5))
plt.tight_layout()
plt.savefig('MassGraphs.png', dpi=600)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10,10))

#make color green for quad and red for conventional for all plots
axes.boxplot(x=[bestEpochs['training_error'],bestQEpochs['training_error']], patch_artist=True, notch=True, labels=['Conventional', 'Quadratic'])
axes.set_title('Distribution of Accuracy by Class\n(BoxPlot)')
axes.set_ylabel('Accuracy (%)')
# axes.set_xlabel('Epoch')
axes.set_ylim([0.92,1])
# axes[0,1].set_xticks(np.arange(0,51, step=5))
plt.savefig('Boxplot.png', dpi=600)

"""
axes[0,0].plot(epochs['epoch'], epochs['training_error'], color='r', label='Conventional')
axes[0,0].set_title('Accuracy over Epochs')
axes[0,0].set_ylabel('Loss')
axes[0,0].set_xlabel('Epoch')
# axes[0,0].set_xlim([0,50])
# axes[0,0].set_xticks(np.arange(0,51, step=5))"""

"""
#make color green for quad and red for conventional for all plots
axes[1,1].plot(epochs['epoch'], epochs['nonzero_params'], color='r', label='Conventional')
axes[1,1].plot(qEpochs['epoch'], qEpochs['nonzero_params'], color='g', label='Quadratic')
axes[1,1].set_title('Parameter Count by Epochs')
axes[1,1].set_ylabel('Count of Non-Zero\nParameters')
axes[1,1].set_xlabel('Epoch')
# axes[1,1].set_xlim([0,50])
# axes[1,1].set_xticks(np.arange(0,51, step=5))"""

"""#make color green for quad and red for conventional for all plots
axes[2,1].bar(bestEpochs['nonzero_params'], height=bestEpochs['training_error'], color='r', label='Conventional')
axes[2,1].bar(bestQEpochs['nonzero_params'], height=bestQEpochs['training_error'], color='g',label='Quadratic')
axes[2,1].set_title('Accuracy over Epochs')
axes[2,1].set_ylabel('Loss')
axes[2,1].set_xlabel('Epoch')
# axes[2,1].set_xlim([0,50])
# axes[2,1].set_xticks(np.arange(0,51, step=5))"""

'''
plt.figtext(0.5, 0.15, f"Final Error: {round(list(errorArray.values())[-1],6)*100}%", ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
# plt.savefig('NN-constant.png', dpi=400)
plt.show()
'''