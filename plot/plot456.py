# %%
# load test data 
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import numpy as np
sns.set()
sns.set_context("paper") # notebook, poster,talk, paper
sns.set_style('whitegrid')
sns.palplot(sns.color_palette('hls', 8))
dd = []
with open("eval456.pickle_data", 'rb') as f:
    dd = pickle.load(f)
print(dd)
# %%
# Figure 4
fig, axes = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)
x = np.arange(1, 29)
x_ticks = np.arange(1, 29)
axes.tick_params(axis='x', labelsize=12,colors='b')
axes.tick_params(axis='y', labelsize=12, colors='b')
axes.bar(x, dd['slew_time'])
axes.set_xlabel('Targets Index',fontsize=14)
axes.set_ylabel('Slew Time (seconds)',fontsize=14)
axes.set_title('')
axes.set_xticks(x_ticks)
axes.grid(True)
plt.savefig("4.pdf", dpi="figure",format="pdf")

# %%
# Figure 5
fig, axes = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)
x = np.arange(1, 29)
x_ticks = np.arange(1, 29)
axes.tick_params(axis='x', labelsize=12,colors='b') 
axes.tick_params(axis='y', labelsize=12, colors='b') 
axes.plot(x, dd['start_el'], 's', label="Beginning")
axes.plot(x, dd['end_el'], 'o', label="Ending")
axes.set_xlabel('Targets Index',fontsize=14)
axes.set_ylabel('Elevation Angle(${\degree}$)',fontsize=14)
axes.set_title('')
axes.set_xticks(x_ticks)

axes.legend(loc='upper right')
axes.grid(axis='y')
plt.savefig("5.pdf", dpi="figure",format="pdf")

# %%
# Figure 6
from astropy.time import Time
import astropy.units as u
time_range = Time(['2021-05-21 00:00:00', "2021-05-22 00:00:00"])
time_ut = time_range[0] + np.linspace(0, 24, 24*60)*u.hour

fig, ax = plt.subplots(1, 1, figsize=(12, 4), tight_layout=True)

ax.set_yticks(range(len(dd['target'])))
ax.set_yticklabels(dd['target'])
ax.set_ylim([0, 8])

y=3
text_delt_y = [1, 2, 3, 4, -2 , -3 ,-4]

for i in range(len(dd['target'])-7):
    text_delt_y.append(text_delt_y[len(text_delt_y)-7])

text_delt_x = 0
i = 0
for idx, k in enumerate(dd['target']):
    text_delt_x = 0
    s,e = dd['start_time'][idx],dd['end_time'][idx]
    line1, = ax.plot(range(int(s), int(e)), [y for i in range(int(s), int(e))], '--', linewidth=20,
                    label=k)
    delta_y = text_delt_y[i]
    if k=='0742-2822':
        delta_y = -3
    if delta_y == -4:
        delta_y = 1   
        text_delt_x = -100   
    ax.annotate(s=k, rotation=18, fontsize=10,xycoords='data',textcoords='data',xy=(int(s)+10, y), 
    xytext=(int(s)+text_delt_x, y+delta_y),
    arrowprops=dict(arrowstyle="->",connectionstyle="arc",color="black"),bbox=dict(boxstyle="round,pad=0.1",alpha=0.1,facecolor='green'))
    i = i + 1

ax.set_xlim([0, 1440])
ax.set_yticklabels([])
ax.axes.yaxis.set_visible(False) 
xticks = list(range(0, 1400, 100))
#最后一个观测完的时间last_time = 1384
last_time = 1384
xticks.append(last_time)
xticks.append(1440)
ax.set_xticks(xticks)

ax.vlines(last_time, 0, 8, 'gray','--')
plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

d = min(time_ut).datetime
ax.set_xlabel("Minutes offset since {0} {1}(UTC)".format(d.date(), d.time()),fontsize=14)
ax.grid(False)

ax.bar(last_time + (1440-last_time)/2, 8, width=1440-last_time, hatch='//', color='white', edgecolor='gray')

plt.savefig("6.pdf", dpi="figure",format="pdf")
# %%

