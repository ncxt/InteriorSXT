import matplotlib.pyplot as plt


def plot_imgs(height=3, cmap="gray", clim=(None, None), colorbar=True, **kwargs):
    fig, axes = plt.subplots(
        nrows=1, ncols=len(kwargs), figsize=(height * len(kwargs), height)
    )
    if len(kwargs) == 1:
        axes = [axes]
    for ax, (k, v) in zip(axes, kwargs.items()):
        pcm = ax.imshow(v, cmap=cmap, clim=clim)
        if colorbar:
            fig.colorbar(pcm, ax=ax)
        ax.set_title(k)
    fig.tight_layout()
