import matplotlib.pyplot as plt

def plot_curves(x, x_label, *argv):
    n = len(argv)
    if n == 6:
        plt.errorbar(x, argv[0], yerr = argv[1], marker='o', label='DT')
        plt.errorbar(x, argv[2], yerr = argv[3], marker='o', label='BT')
        plt.errorbar(x, argv[4], yerr = argv[5], marker='o', label='RF')
        plt.title('10-Fold Cross Validation: DT, BT and RF')
    elif n == 4:
        plt.errorbar(x, argv[0], yerr = argv[1], marker='o', label='BT')
        plt.errorbar(x, argv[2], yerr = argv[3], marker='o', label='RF')
        plt.title('10-Fold Cross Validation: BT and RF')
    plt.legend(loc='upper left')
    plt.xlabel(x_label)
    plt.ylabel('Model Accuracy and Standard Error')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.show()
