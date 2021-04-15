import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches


def invert(mat):
    ones = mat == 1
    zeros = mat == 0
    mat[ones] = 0
    mat[zeros] = 1
    return mat

simData = np.load("simulated-data-10000.npz")
# variantData = invert(simData["variantMatrixArray"][0][:200, ].transpose())
variantData = invert(simData["variantMatrixArray"][2][:200, :40].transpose())
invariantData = invert(simData["variantMatrixArray"][2][:200, 40:].transpose())


# plt.imshow(variantData, interpolation="nearest", cmap=cm.Greys_r)
# plt.xlabel("Site")
# plt.ylabel("Sample")
# plt.savefig("variant-alignment.png", bbox_inches="tight", pad_inches=0.1)


# # Variant Site Alignment
# fig, ax = plt.subplots()
# ax.imshow(variantData, interpolation="nearest", cmap=cm.Greys_r)
# fig.savefig("variant-alignment.png", bbox_inches="tight", pad_inches=0.1)

# # plt.clf()
# # plt.imshow(invariantData, interpolation="nearest", cmap=cm.Greys_r)
# # plt.xlabel("Site")
# # plt.ylabel("Sample")
# # plt.savefig("invariant-alignment.png", bbox_inches="tight", pad_inches=0.1)


# # Invariant Site Alignment
# fig, ax = plt.subplots()
# ax.imshow(invariantData, interpolation="nearest", cmap=cm.Greys_r)
# fig.savefig("invariant-alignment.png", bbox_inches="tight", pad_inches=0.1)


# # Show 1D convolution
# fig, ax = plt.subplots()
# ax.imshow(invariantData, interpolation="nearest", cmap=cm.Greys_r)
# # rect = patches.Rectangle((5, 5), 5, 5, linewidth=1, edgecolor='r', facecolor='none')
# # ax.add_patch(rect)
# fig.savefig("1d-convolution-alignment.png", bbox_inches="tight", pad_inches=0.1)



# fig, ax = plt.subplots()
# ax.imshow(invariantData, interpolation="nearest", cmap=cm.Greys_r)
# # rect = patches.Rectangle((0, 0), 200, 5, linewidth=1, edgecolor='r', facecolor='none')
# # ax.add_patch(rect)
# # ax.xlabel("Site")
# # ax.ylabel("Sample")
# fig.savefig("1d-convolution-alignment.png", bbox_inches="tight", pad_inches=0.1)



import plotly.express as px

fig = px.imshow(variantData)


fig.write_image("1d-convolution-alignment.png")