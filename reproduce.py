import matplotlib.pyplot as plt
import numpy as np

R = 20

linewidth = 2
plt.grid(True)
x_ticks = np.arange(0, 20, 2)
x_ticks_val = x_ticks

y_ticks = [1, 2, 3, 4, 5, 6, 7]
#N_array = [2, 5, 10]
N_array = [2]
Rate = []

colors = ['BLUE', 'RED', 'YELLOW']

cnt = 0
legend = []
reward = []
for n in N_array:
    SNR_range = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
    reward = np.load(f"Learning Curves/{n}_{R}.npy", allow_pickle=True).squeeze()
    plt.plot(SNR_range, reward, marker='*', linewidth=linewidth, color=colors[cnt])
    legend.append(f"IRS-assisted, R = 20, N = {n}")
    x_label = "SNR"
    y_label = "rate"
    # print("r_average:", r_average)
    cnt = cnt + 1
#
# t = 0
# for n in N_array:
#     SNR_range = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
#     reward2 = np.load(f"{n}_No_RIS.npy", allow_pickle=True).squeeze()
#     plt.plot(SNR_range, reward2, marker='_', linestyle='dashed', linewidth=linewidth, color=colors[t])
#     legend.append(f"No RIS, N = {n}")
#     # print("r_average:", r_average)
#     t = t + 1

y_ticks_val = y_ticks
legend_loc = 'upper left'
plt.xlabel('SNR (dB)')
plt.ylabel('Rate')
plt.legend(legend, loc=legend_loc, fontsize=10, ncol=1)
ax = plt.gca()
ax.set_xlim([0, 30])
ax.set_ylim([0, 22])
leg = ax.get_legend()
leg.legendHandles[0].set_color('BLUE')
# leg.legendHandles[1].set_color('RED')
# leg.legendHandles[2].set_color('YELLOW')

plt.show()
