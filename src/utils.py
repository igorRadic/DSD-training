import matplotlib.pyplot as plt
import numpy as np
import random
import torch

plt.style.use("ggplot")


def plot_wb(model, fig_path, ranges=None, only_fc=False):
    tmp = list(model.named_parameters())
    layers = []
    for i in range(0, len(tmp), 2):
        w, b = tmp[i], tmp[i + 1]
        if only_fc:
            if "fc" in w[0] or "fc" in b[0]:
                layers.append((w, b))
        else:
            if ("conv" in w[0] or "conv" in b[0]) or ("fc" in w[0] or "fc" in b[0]):
                layers.append((w, b))

    num_rows = len(layers)

    fig = plt.figure(figsize=(20, 40))

    i = 1
    for w, b in layers:
        w_flatten = w[1].flatten().detach().cpu().numpy()
        b_flatten = b[1].flatten().detach().cpu().numpy()

        fig.add_subplot(num_rows, 2, i)
        plt.title(w[0])
        plt.hist(w_flatten, bins=100, range=ranges)

        fig.add_subplot(num_rows, 2, i + 1)
        plt.title(b[0])
        plt.hist(b_flatten, bins=100, range=ranges)

        i += 2

    fig.tight_layout()
    plt.savefig(fig_path)
    plt.close()


def set_all_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(self, path: str, best_valid_loss=float("inf")):
        self.best_valid_loss = best_valid_loss
        self.path = path

    def __call__(self, current_valid_loss, epoch, model, optimizer, criterion):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": criterion,
                },
                self.path,
            )
