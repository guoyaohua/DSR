import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# sphinx_gallery_thumbnail_number = 2
plt.rcParams['font.sans-serif']=['SIMHEI'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """
    
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

label = ["加速", "碰撞", "匀速", "左转", "右转"]

# 1.DA-FA-MVCNN
harvest = np.array([[11, 0, 4, 0, 0],
                    [1, 22, 0, 0,  0],
                    [0,  0, 27,  0,  0],
                    [0,  0, 0, 20,  0],
                    [0,  0, 0,  0, 20]])

plt.figure(figsize=(3,3),frameon = True)
fig, ax = plt.subplots()

im, cbar = heatmap(harvest, label, label, ax=ax,
                   cmap="YlGn", cbarlabel="DA-FA-MVCNN")
texts = annotate_heatmap(im, valfmt="{x:1d}")

fig.tight_layout()
plt.gcf().savefig("./ExperimentData/DA_FA_MVCNN.png",dpi = 500,bbox_inches='tight')

# 2.DA-FA-CNN
harvest = np.array([[11, 0, 4, 0, 0],
                    [0, 22, 1, 0,  0],
                    [1,  0, 26,  0,  0],
                    [0,  0, 0, 20,  0],
                    [0,  0, 0,  0, 20]])

plt.figure(figsize=(3,3),frameon = True)
fig, ax = plt.subplots()

im, cbar = heatmap(harvest, label, label, ax=ax,
                   cmap="YlGn", cbarlabel="DA-FA-CNN")
texts = annotate_heatmap(im, valfmt="{x:1d}")

fig.tight_layout()
plt.gcf().savefig("./ExperimentData/DA_FA_CNN.png",dpi = 500,bbox_inches='tight')


# 3.DA-FA-LSTM
harvest = np.array([[6, 0, 9, 0, 0],
                    [0, 22, 1, 0,  0],
                    [2,  0, 25,  0,  0],
                    [0,  0, 0, 20,  0],
                    [0,  0, 0,  0, 20]])

plt.figure(figsize=(3,3),frameon = True)
fig, ax = plt.subplots()

im, cbar = heatmap(harvest, label, label, ax=ax,
                   cmap="YlGn", cbarlabel="DA-FA-LSTM")
texts = annotate_heatmap(im, valfmt="{x:1d}")

fig.tight_layout()
plt.gcf().savefig("./ExperimentData/DA_FA_LSTM.png",dpi = 500,bbox_inches='tight')

# 4.DA-FA-RNN
harvest = np.array([[4, 0, 11, 0, 0],
                    [1, 20, 1, 0,  1],
                    [0,  0, 27,  0,  0],
                    [1,  0, 0, 19,  0],
                    [0,  0, 0,  0, 20]])

plt.figure(figsize=(3,3),frameon = True)
fig, ax = plt.subplots()

im, cbar = heatmap(harvest, label, label, ax=ax,
                   cmap="YlGn", cbarlabel="DA-FA-RNN")
texts = annotate_heatmap(im, valfmt="{x:1d}")

fig.tight_layout()
plt.gcf().savefig("./ExperimentData/DA_FA_RNN.png",dpi = 500,bbox_inches='tight')


# 5.DA-MVCNN
harvest = np.array([[12, 0, 3, 0, 0],
                    [1, 20, 2, 0,  0],
                    [0,  0, 27,  0,  0],
                    [0,  0, 0, 20,  0],
                    [0,  0, 0,  0, 20]])

plt.figure(figsize=(3,3),frameon = True)
fig, ax = plt.subplots()

im, cbar = heatmap(harvest, label, label, ax=ax,
                   cmap="YlGn", cbarlabel="DA-MVCNN")
texts = annotate_heatmap(im, valfmt="{x:1d}")

fig.tight_layout()
plt.gcf().savefig("./ExperimentData/DA_MVCNN.png",dpi = 500,bbox_inches='tight')

# 6.DA-CNN
harvest = np.array([[9, 0, 6, 0, 0],
                    [0, 21, 2, 0,  0],
                    [1,  0, 26,  0,  0],
                    [0,  0, 0, 20,  0],
                    [0,  0, 0,  0, 20]])

plt.figure(figsize=(3,3),frameon = True)
fig, ax = plt.subplots()

im, cbar = heatmap(harvest, label, label, ax=ax,
                   cmap="YlGn", cbarlabel="DA-CNN")
texts = annotate_heatmap(im, valfmt="{x:1d}")

fig.tight_layout()
plt.gcf().savefig("./ExperimentData/DA_CNN.png",dpi = 500,bbox_inches='tight')


# 7.DA-LSTM
harvest = np.array([[6, 0, 9, 0, 0],
                    [5, 6, 12, 0,  0],
                    [6,  0, 21,  0,  0],
                    [0,  0, 0, 20,  0],
                    [0,  0, 0,  0, 20]])

plt.figure(figsize=(3,3),frameon = True)
fig, ax = plt.subplots()

im, cbar = heatmap(harvest, label, label, ax=ax,
                   cmap="YlGn", cbarlabel="DA-LSTM")
texts = annotate_heatmap(im, valfmt="{x:1d}")

fig.tight_layout()
plt.gcf().savefig("./ExperimentData/DA_LSTM.png",dpi = 500,bbox_inches='tight')

# 8.DA-RNN
harvest = np.array([[6, 0, 9, 0, 0],
                    [2, 19, 1, 0,  1],
                    [1,  0, 26,  0,  0],
                    [0,  0, 0, 20,  0],
                    [0,  0, 0,  0, 20]])

plt.figure(figsize=(3,3),frameon = True)
fig, ax = plt.subplots()

im, cbar = heatmap(harvest, label, label, ax=ax,
                   cmap="YlGn", cbarlabel="DA-RNN")
texts = annotate_heatmap(im, valfmt="{x:1d}")

fig.tight_layout()
plt.gcf().savefig("./ExperimentData/DA_RNN.png",dpi = 500,bbox_inches='tight')

# 9.FA-MVCNN
harvest = np.array([[15, 0, 0, 0, 0],
                    [1, 21, 1, 0,  0],
                    [8,  0, 19,  0,  0],
                    [0,  0, 0, 20,  0],
                    [0,  0, 0,  0, 20]])

plt.figure(figsize=(3,3),frameon = True)
fig, ax = plt.subplots()

im, cbar = heatmap(harvest, label, label, ax=ax,
                   cmap="YlGn", cbarlabel="FA-MVCNN")
texts = annotate_heatmap(im, valfmt="{x:1d}")

fig.tight_layout()
plt.gcf().savefig("./ExperimentData/FA_MVCNN.png",dpi = 500,bbox_inches='tight')

# 10.FA-CNN
harvest = np.array([[0, 0, 15, 0, 0],
                    [0, 21, 1, 0,  1],
                    [0,  0, 27,  0,  0],
                    [0,  0, 1, 19,  0],
                    [0,  0, 0,  0, 20]])

plt.figure(figsize=(3,3),frameon = True)
fig, ax = plt.subplots()

im, cbar = heatmap(harvest, label, label, ax=ax,
                   cmap="YlGn", cbarlabel="FA-CNN")
texts = annotate_heatmap(im, valfmt="{x:1d}")

fig.tight_layout()
plt.gcf().savefig("./ExperimentData/FA_CNN.png",dpi = 500,bbox_inches='tight')


# 11.FA-LSTM
harvest = np.array([[0, 0, 15, 0, 0],
                    [0, 10, 6, 2,  5],
                    [0,  0, 27,  0,  0],
                    [0,  0, 11, 0,  9],
                    [0,  0, 0,  0, 20]])

plt.figure(figsize=(3,3),frameon = True)
fig, ax = plt.subplots()

im, cbar = heatmap(harvest, label, label, ax=ax,
                   cmap="YlGn", cbarlabel="FA-LSTM")
texts = annotate_heatmap(im, valfmt="{x:1d}")

fig.tight_layout()
plt.gcf().savefig("./ExperimentData/FA_LSTM.png",dpi = 500,bbox_inches='tight')

# 12.FA-RNN
harvest = np.array([[0, 0, 15, 0, 0],
                    [0, 8, 6, 6,  3],
                    [0,  0, 27,  0,  0],
                    [0,  0, 1, 19,  0],
                    [0,  0, 0,  0, 20]])

plt.figure(figsize=(3,3),frameon = True)
fig, ax = plt.subplots()

im, cbar = heatmap(harvest, label, label, ax=ax,
                   cmap="YlGn", cbarlabel="FA-RNN")
texts = annotate_heatmap(im, valfmt="{x:1d}")

fig.tight_layout()
plt.gcf().savefig("./ExperimentData/FA_RNN.png",dpi = 500,bbox_inches='tight')



# 13.MVCNN
harvest = np.array([[5, 0, 10, 0, 0],
                    [0, 20, 3, 0,  0],
                    [1,  0, 26,  0,  0],
                    [0,  0, 0, 20,  0],
                    [0,  0, 0,  0, 20]])

plt.figure(figsize=(3,3),frameon = True)
fig, ax = plt.subplots()

im, cbar = heatmap(harvest, label, label, ax=ax,
                   cmap="YlGn", cbarlabel="MVCNN")
texts = annotate_heatmap(im, valfmt="{x:1d}")

fig.tight_layout()
plt.gcf().savefig("./ExperimentData/MVCNN.png",dpi = 500,bbox_inches='tight')

# 14.CNN
harvest = np.array([[0, 0, 15, 0, 0],
                    [0, 3, 14, 0,  6],
                    [0,  0, 27,  0,  0],
                    [0,  0, 0, 20,  0],
                    [0,  0, 0,  0, 20]])

plt.figure(figsize=(3,3),frameon = True)
fig, ax = plt.subplots()

im, cbar = heatmap(harvest, label, label, ax=ax,
                   cmap="YlGn", cbarlabel="CNN")
texts = annotate_heatmap(im, valfmt="{x:1d}")

fig.tight_layout()
plt.gcf().savefig("./ExperimentData/CNN.png",dpi = 500,bbox_inches='tight')


# 15.LSTM
harvest = np.array([[0, 1, 13, 1, 0],
                    [0, 4, 12, 4,  3],
                    [0,  0, 24,  3,  0],
                    [0,  0, 0, 20,  0],
                    [0,  0, 0,  0, 20]])

plt.figure(figsize=(3,3),frameon = True)
fig, ax = plt.subplots()

im, cbar = heatmap(harvest, label, label, ax=ax,
                   cmap="YlGn", cbarlabel="LSTM")
texts = annotate_heatmap(im, valfmt="{x:1d}")

fig.tight_layout()
plt.gcf().savefig("./ExperimentData/LSTM.png",dpi = 500,bbox_inches='tight')

# 16.FA-RNN
harvest = np.array([[0, 1, 14, 0, 0],
                    [2, 6, 14, 0,  1],
                    [0,  0, 27,  0,  0],
                    [0,  0, 0, 20,  0],
                    [0,  0, 0,  0, 20]])

plt.figure(figsize=(3,3),frameon = True)
fig, ax = plt.subplots()

im, cbar = heatmap(harvest, label, label, ax=ax,
                   cmap="YlGn", cbarlabel="RNN")
texts = annotate_heatmap(im, valfmt="{x:1d}")

fig.tight_layout()
plt.gcf().savefig("./ExperimentData/RNN.png",dpi = 500,bbox_inches='tight')