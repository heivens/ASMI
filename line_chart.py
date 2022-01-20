import matplotlib.pyplot as plt

x_list = ['$2^6$', '$2^7$', '$2^8$', '$2^9$', '$2^{10}$', '$2^{11}$']
y_10 = [94.7, 97.2, 98.3, 99.1, 99.0, 99.1]
y_50 = [91.4, 94.2, 95.6, 97.4, 97.3, 97.4]
y_100 = [86.7, 92.1, 93.3, 95.1, 95.6, 95.4]

# plt.style.use('ggplot')
# mean bag siz = 10
plt.figure(figsize=(6, 5), dpi=300)
plt.plot(x_list, y_10, label="10", linestyle="-.", marker='.')
for a, b in zip(x_list, y_10):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

# mean bag siz = 50的折线图分布
plt.plot(x_list, y_50, label="50", linestyle="-.", marker='o')
for a, b in zip(x_list, y_50):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
# mean bag siz = 100的折线图分布
plt.plot(x_list, y_100, label="100", linestyle="-.", marker='*')
for a, b in zip(x_list, y_100):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
plt.grid(alpha=1)
plt.legend(loc='lower right')
plt.xlabel(r"Value of $\mathtt{l}$", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.tight_layout()
plt.savefig('main/figures/line_mnist.pdf', format='pdf')
plt.show()
