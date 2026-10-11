from matplotlib.figure import Figure


def plot_series(
    x, y, output, smoothed=None, fit=None, outliers=None, title=None, xlabel="x", ylabel="y"
):
    fig = Figure(figsize=(8, 4.5))
    ax = fig.add_subplot()
    ax.plot(x, y, "o", color="0.6", markersize=3, label="raw data")
    if smoothed is not None:
        ax.plot(x, smoothed, "-", color="black", linewidth=1.5, label="moving average")
    if fit is not None:
        ax.plot(x, fit.predict(x), "--", color="black", linewidth=1, label="linear fit")
    if outliers is not None and outliers.any():
        ax.plot(x[outliers], y[outliers], "x", color="black", markersize=8, label="outliers")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle=":", color="0.8")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    return output
