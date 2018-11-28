import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set(color_codes=True)
sns.set_style('ticks')
lbd = 1e-2
k = 50

with open('output/exp3_jlist.pkl', 'rb') as f:
    jList = pkl.load(f)
with open('output/exp3_rmselist.pkl', 'rb') as f:
    rmseList = pkl.load(f)
with open('output/exp3_gauss_jlist.pkl', 'rb') as f:
    jList_gauss = pkl.load(f)
with open('output/exp3_gauss_rmselist.pkl', 'rb') as f:
    rmseList_gauss = pkl.load(f)

fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(rmseList, 'r-')
line1_gauss, = ax1.plot(rmseList_gauss, 'r-.')
plt.ylabel("RMSE")
ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
line2, = ax2.plot(jList, 'b-')
line2_gauss, = ax2.plot(jList_gauss, 'b-.')
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
plt.ylabel("J")
plt.legend([line1, line2, line1_gauss, line2_gauss], ["Jacobi RMSE", "Jacobi J", "Gauss RMSE", "Gauss J"])


fig.savefig('output/exp3_combined_k%dl%d.png' % (k, -np.log10(lbd)), dpi=300)
