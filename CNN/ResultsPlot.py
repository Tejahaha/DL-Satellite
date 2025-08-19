import matplotlib.pyplot as plt

def plot_optim_results(res):
    optims = list(res.keys())
    acc = [res[opt]["acc"] for opt in optims]
    loss = [res[opt]["loss"] for opt in optims]

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.bar(optims, acc , color='skyblue')
    plt.title('Optimisers VS Accuracy ')
    plt.ylabel("Accuracy (%)")
    plt.ylim(0,100)
    plt.grid(axis='y' , linestyle='--' , alpha=0.7)

    plt.subplot(1,2,2)
    plt.bar(optims, loss , color='red')
    plt.title('Optimisers VS Loss ')
    plt.ylabel("Loss (%)")
    plt.ylim(0,100)
    plt.grid(axis='y' , linestyle='--' , alpha=0.7)

    plt.tight_layout()
    plt.show()
