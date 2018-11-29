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
line1_gauss, = ax1.plot(rmseList_gauss, 'b-')
plt.ylabel("RMSE")
ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
line2, = ax2.plot(jList, 'r-.')
line2_gauss, = ax2.plot(jList_gauss, 'b-.')
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
plt.ylabel("J")
plt.legend([line1, line2, line1_gauss, line2_gauss], ["Jacobi RMSE", "Jacobi J", "Gauss RMSE", "Gauss J"])

fig.savefig('output/exp3_combined_k%dl%d.png' % (k, -np.log10(lbd)), dpi=300)

plt.rc('text', usetex=True)

def handleFile(substr=''):
    with open('output/paramRes%sList.pkl' % substr, 'rb') as f:
        resList = pkl.load(f)
    rmse1, J1 = resList[(0.01,50)]
    rmse2, J2 = resList[(10,70)]
    rmse1= rmse1[1:]
    rmse2= rmse2[1:]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    line1, = ax1.plot(rmse1, 'r-')
    line2, = ax1.plot(rmse2, 'b-')
    plt.ylabel("RMSE")
    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
    line3, = ax2.plot(J1, 'r-.')
    line4, = ax2.plot(J2, 'b-.')
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    plt.ylabel("J")
    plt.legend([line1, line3, line2, line4], ["RMSE $k=50,\lambda=0.01$", "J $k=50,\lambda=0.01$", "RMSE $k=70,\lambda=10$", "J $k=70,\lambda=10$"])

    fig.savefig('output/exp3_%s.png' % substr, dpi=300)


handleFile('L3')
handleFile('LM')
