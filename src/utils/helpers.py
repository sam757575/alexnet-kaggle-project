import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def augment_data(image):
    # Function to perform data augmentation on the input image
    pass


def save_model(model, filepath):
    # Function to save the trained model to the specified filepath
    pass


def load_model(filepath):
    # Function to load a model from the specified filepath
    pass


def plot_training_history(history, save_path='training_history.png', show=False):
    """Plot every train/validation KPI found in a Keras history object.

    Produces one subplot per tracked metric (loss, accuracy, and any other
    metric passed to `model.compile(metrics=[...])`), each showing the
    training curve alongside its validation counterpart when available.
    """
    if hasattr(history, 'history'):
        history = history.history

    if not isinstance(history, dict):
        raise TypeError('history must be a Keras History object or a dict containing training metrics')

    metric_names = [key for key in history if not key.startswith('val_') and history[key]]
    if not metric_names:
        raise ValueError('history does not contain any metrics to plot')

    fig, axes = plt.subplots(len(metric_names), 1, figsize=(8, 4 * len(metric_names)), dpi=150, squeeze=False)
    fig.patch.set_facecolor('white')

    for ax, metric in zip(axes[:, 0], metric_names):
        train_values = history[metric]
        epochs = range(1, len(train_values) + 1)
        label = metric.replace('_', ' ').title()

        ax.set_facecolor('white')
        ax.plot(epochs, train_values, color='#1f77b4', linewidth=2, marker='o', label=f'Train {label}')

        val_key = f'val_{metric}'
        if val_key in history:
            ax.plot(epochs, history[val_key], color='#ff7f0e', linewidth=2, marker='s', label=f'Validation {label}')

        ax.set_title(f'{label} per Epoch')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.2)
        ax.legend()

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path
