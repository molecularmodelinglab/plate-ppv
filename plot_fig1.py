import numpy as np
import matplotlib.pyplot as plt
import pickle


data = pickle.load(open("./results_xgboost_v2_bedroc.pkl", "rb"))


pos_bal = {}
pos_unbal = {}
for df_name in data.keys():
    pos_bal[df_name] = {"mean": [], "std": []}
    pos_unbal[df_name] = {"mean": [], "std": []}
    for j in range(1, 1000):
        cb = []
        cu = []
        for i in range(10):
            real_top_bal = data[df_name][i]["y_test"][
                np.argpartition(data[df_name][i]["III_prob"],
                                len(data[df_name][i]["y_test"]) - j)[-j:]]
            real_top_unbal = data[df_name][i]["y_test"][
                np.argpartition(data[df_name][i]["IV_prob"],
                                len(data[df_name][i]["y_test"]) - j)[-j:]]
            pred_top_bal = data[df_name][i]["III"][
                np.argpartition(data[df_name][i]["III_prob"],
                                len(data[df_name][i]["y_test"]) - j)[-j:]]
            pred_top_unbal = data[df_name][i]["IV"][
                np.argpartition(data[df_name][i]["IV_prob"],
                                len(data[df_name][i]["y_test"]) - j)[-j:]]
            cb.append(real_top_bal.sum()/j)
            cu.append(real_top_unbal.sum()/j)
        pos_bal[df_name]["mean"].append(np.mean(cb))
        pos_unbal[df_name]["mean"].append(np.mean(cu))
        pos_bal[df_name]["std"].append(np.std(cb))
        pos_unbal[df_name]["std"].append(np.std(cu))

aids = [
    "AID_485314",
    "AID_485341",
    "AID_504466",
    "AID_624202",
    "AID_651820"
]

plt.figure(figsize=(12, 8))
fig = plt.gcf()
gs = fig.add_gridspec(2, 12)
ax1 = fig.add_subplot(gs[0:1, 0:4])
ax2 = fig.add_subplot(gs[0:1, 4:8])
ax3 = fig.add_subplot(gs[0:1, 8:12])
ax4 = fig.add_subplot(gs[1:2, 2:6])
ax5 = fig.add_subplot(gs[1:2, 6:10])
axes = [ax1, ax2, ax3, ax4, ax5]
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for ax, df_name, title, col in zip(axes, data.keys(), aids, colors[:5]):
    ax.plot(np.arange(16, 1000), pos_bal[df_name]["mean"][15:], label="Balanced", color=col, alpha=0.4)
    ax.fill_between(np.arange(16, 1000), np.array(pos_bal[df_name]["mean"][15:])+np.array(pos_bal[df_name]["std"][15:]), np.array(pos_bal[df_name]["mean"][15:])-np.array(pos_bal[df_name]["std"][15:]), color=col, alpha=0.4)
    ax.plot(np.arange(16, 1000), pos_unbal[df_name]["mean"][15:], label="Unbalanced", color=col)
    ax.fill_between(np.arange(16, 1000), np.array(pos_unbal[df_name]["mean"][15:])+np.array(pos_unbal[df_name]["std"][15:]), np.array(pos_unbal[df_name]["mean"][15:])-np.array(pos_unbal[df_name]["std"][15:]), color=col, alpha=0.8)
    ax.vlines(128, 0, 1, linestyles="dashed", alpha=0.8, colors="black")
    ax.vlines(256, 0, 1, linestyles="dashed", alpha=0.8, colors="black")
    ax.set_xlim(16, 1000)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_xticks(list(range(128, 1000, 128)))
    ax.legend()
fig.suptitle("PPV in top N Nominations")
fig.supxlabel("Number of Nominated Compounds")
fig.supylabel("PPV")
plt.tight_layout()
plt.savefig("fig1.svg")
plt.savefig("fig1.png")
plt.show()
