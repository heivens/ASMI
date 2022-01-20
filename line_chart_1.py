import matplotlib.pyplot as plt
import numpy as np

x_list = [50, 60, 70, 80, 90]
# 3分类的acc
A_net_acc_ave_3 = [0.815, 0.821, 0.845, 0.871, 0.893]
A_net_acc_std_3 = [0.053, 0.034, 0.014, 0.021, 0.023]
Loss_A_acc_ave_3 = [0.896, 0.927, 0.941, 0.949, 0.964]
Loss_A_acc_std_3 = [0.036, 0.021, 0.014, 0.043, 0.023]
ASMI_acc_ave_3 = [0.911, 0.944, 0.984, 0.989, 0.991]
ASMI_acc_std_3 = [0.018, 0.012, 0.016, 0.012, 0.010]
# 5分类的acc
A_net_acc_ave_5 = [0.595, 0.642, 0.677, 0.698, 0.756]
A_net_acc_std_5 = [0.073, 0.052, 0.044, 0.030, 0.048]
Loss_A_acc_ave_5 = [0.662, 0.751, 0.777, 0.808, 0.828]
Loss_A_acc_std_5 = [0.053, 0.049, 0.044, 0.041, 0.048]
ASMI_acc_ave_5 = [0.886, 0.911, 0.922, 0.931, 0.941]
ASMI_acc_std_5 = [0.029, 0.021, 0.022, 0.016, 0.016]
# 10分类的acc
A_net_acc_ave_10 = [0.325, 0.356, 0.374, 0.412, 0.448]
A_net_acc_std_10 = [0.078, 0.030, 0.028, 0.021, 0.043]
Loss_A_acc_ave_10 = [0.389, 0.421, 0.451, 0.459, 0.461]
Loss_A_acc_std_10 = [0.035, 0.028, 0.028, 0.013, 0.011]
ASMI_acc_ave_10 = [0.627, 0.652, 0.673, 0.683, 0.698]
ASMI_acc_std_10 = [0.051, 0.031, 0.011, 0.016, 0.011]

# 3分类f1
A_net_f1_ave_3 = [0.823, 0.829, 0.856, 0.881, 0.889]
A_net_f1_std_3 = [0.034, 0.032, 0.011, 0.022, 0.023]
Loss_A_f1_ave_3 = [0.900, 0.920, 0.943, 0.951, 0.967]
Loss_A_f1_std_3 = [0.029, 0.023, 0.034, 0.023, 0.017]
ASMI_f1_ave_3 = [0.914, 0.941, 0.980, 0.987, 0.990]
ASMI_f1_std_3 = [0.010, 0.015, 0.012, 0.012, 0.021]
# 5分类f1
A_net_f1_ave_5 = [0.603, 0.637, 0.671, 0.691, 0.751]
A_net_f1_std_5 = [0.062, 0.031, 0.044, 0.037, 0.036]
Loss_A_f1_ave_5 = [0.651, 0.743, 0.769, 0.802, 0.831]
Loss_A_f1_std_5 = [0.021, 0.035, 0.024, 0.019, 0.023]
ASMI_f1_ave_5 = [0.885, 0.901, 0.919, 0.927, 0.941]
ASMI_f1_std_5 = [0.022, 0.019, 0.017, 0.036, 0.020]
# 10分类f1
A_net_f1_ave_10 = [0.344, 0.366, 0.382, 0.422, 0.451]
A_net_f1_std_10 = [0.048, 0.073, 0.048, 0.021, 0.032]
Loss_A_f1_ave_10 = [0.394, 0.431, 0.462, 0.471, 0.480]
Loss_A_f1_std_10 = [0.029, 0.031, 0.024, 0.013, 0.011]
ASMI_f1_ave_10 = [0.623, 0.654, 0.669, 0.689, 0.691]
ASMI_f1_std_10 = [0.031, 0.021, 0.039, 0.036, 0.021]

# 3分类auc
A_net_auc_ave_3 = [0.911, 0.923, 0.930, 0.957, 0.973]
A_net_auc_std_3 = [0.023, 0.014, 0.004, 0.041, 0.013]
Loss_A_auc_ave_3 = [0.913, 0.930, 0.937, 0.952, 0.970]
Loss_A_auc_std_3 = [0.031, 0.019, 0.022, 0.041, 0.029]
ASMI_auc_ave_3 = [0.914, 0.952, 0.979, 0.981, 0.992]
ASMI_auc_std_3 = [0.005, 0.012, 0.012, 0.032, 0.017]

# 5分类auc
A_net_auc_ave_5 = [0.896, 0.914, 0.930, 0.942, 0.956]
A_net_auc_std_5 = [0.023, 0.014, 0.004, 0.041, 0.013]
Loss_A_auc_ave_5 = [0.902, 0.922, 0.929, 0.947, 0.966]
Loss_A_auc_std_5 = [0.031, 0.019, 0.022, 0.041, 0.029]
ASMI_auc_ave_5 = [0.961, 0.969, 0.972, 0.974, 0.988]
ASMI_auc_std_5 = [0.015, 0.002, 0.021, 0.024, 0.019]

# 10分类auc
A_net_auc_ave_10 = [0.612, 0.651, 0.671, 0.682, 0.690]
A_net_auc_std_10 = [0.055, 0.079, 0.044, 0.041, 0.023]
Loss_A_auc_ave_10 = [0.723, 0.746, 0.792, 0.843, 0.861]
Loss_A_auc_std_10 = [0.062, 0.031, 0.029, 0.032, 0.011]
ASMI_auc_ave_10 = [0.922, 0.929, 0.936, 0.944, 0.947]
ASMI_auc_std_10 = [0.037, 0.012, 0.014, 0.034, 0.009]

fig = plt.figure(figsize=(8, 8))
sub_fig_3 = fig.add_subplot(3, 1, 1)
sub_fig_5 = fig.add_subplot(3, 1, 2)
sub_fig_10 = fig.add_subplot(3, 1, 3)

# 3分类
sub_fig_3.errorbar(x_list, ASMI_auc_ave_3, yerr=ASMI_auc_std_3, fmt='d--', lw=2, markersize=8, capsize=4, label='ASMI')
sub_fig_3.errorbar(x_list, Loss_A_auc_ave_3, yerr=Loss_A_auc_std_3, fmt='s--', lw=2, markersize=8, capsize=4, label='Loss-A.')
sub_fig_3.errorbar(x_list, A_net_auc_ave_3, yerr=A_net_auc_std_3, fmt='*--', lw=2, markersize=8, capsize=4, label='A.-net')
sub_fig_3.set_title('Corel-3', fontsize=16)
sub_fig_3.set_xticks(x_list)
sub_fig_3.set_ylabel('AUC', fontsize=16)
sub_fig_3.legend(loc='lower right')
sub_fig_3.grid(alpha=1)
# 5分类
sub_fig_5.errorbar(x_list, ASMI_auc_ave_5, yerr=ASMI_auc_std_5, fmt='d--', lw=2, markersize=8, capsize=4, label='ASMI')
sub_fig_5.errorbar(x_list, Loss_A_auc_ave_5, yerr=Loss_A_auc_std_5, fmt='s--', lw=2, markersize=8, capsize=4, label='Loss-A.')
sub_fig_5.errorbar(x_list, A_net_auc_ave_5, yerr=A_net_auc_std_5, fmt='*--', lw=2, markersize=8, capsize=4, label='A.-net')
sub_fig_5.set_title('Corel-5', fontsize=16)
sub_fig_5.set_xticks(x_list)
sub_fig_5.set_ylabel('AUC', fontsize=16)
sub_fig_5.legend(loc='lower right')
sub_fig_5.grid(alpha=1)
# 10分类
sub_fig_10.errorbar(x_list, ASMI_auc_ave_10, yerr=ASMI_auc_std_10, fmt='d--', lw=2, markersize=8, capsize=4, label='ASMI')
sub_fig_10.errorbar(x_list, Loss_A_auc_ave_10, yerr=Loss_A_auc_std_10, fmt='s--', lw=2, markersize=8, capsize=4, label='Loss-A.')
sub_fig_10.errorbar(x_list, A_net_auc_ave_10, yerr=A_net_auc_std_10, fmt='*--', lw=2, markersize=8, capsize=4, label='A.-net')
sub_fig_10.set_title('Corel-10', fontsize=16)
sub_fig_10.set_xticks(x_list)
sub_fig_10.set_ylabel('AUC', fontsize=16)
sub_fig_10.set_xlabel('Number of training bags per category', fontsize=16)
sub_fig_10.legend(loc='lower right')
sub_fig_10.grid(alpha=1)

plt.tight_layout()
plt.savefig('main/figures/chart_corel_auc.pdf', format='pdf')
plt.show()