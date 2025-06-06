import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# 1) Load the GDV‐per‐layer data:
file_path = 'results/gpt2_gdv/gpt2_bank_gdv.pkl'
with open(file_path, 'rb') as f:
    data = pickle.load(f)

layers        = data['sorted_layers']
layer_data    = data['layer_data']
gdv_per_layer = data['gdv_per_layer']
meta          = data['meta']

# 2) Extract PCA coordinates, group labels, sample sentences, and GDV values:
x_all    = [layer_data[layer]['x']        for layer in layers]
y_all    = [layer_data[layer]['y']        for layer in layers]
group    = layer_data[layers[0]]['group']   # same “group” vector for every layer
sentences= layer_data[layers[0]]['sentence']# same “sentence” list for every layer
gdv_all  = [gdv_per_layer[layer]           for layer in layers]

# 3) Compute global axis limits (with a 10% margin):
all_x = np.concatenate(x_all)
all_y = np.concatenate(y_all)
margin_x = (all_x.max() - all_x.min()) * 0.10
margin_y = (all_y.max() - all_y.min()) * 0.10

x_min, x_max = all_x.min() - margin_x, all_x.max() + margin_x
y_min, y_max = all_y.min() - margin_y, all_y.max() + margin_y

# 4) Pick two indices to display full sample text (one from each group):
idx0 = 0 
idx1 = np.where(np.array(group) == 1)[0][0]

# 5) Animation parameters:
n_layers     = len(layers)
n_interp     = 5                     # number of “in‐between” steps
total_frames = (n_layers - 1) * n_interp + 1

# 6) Switch to a dark‐background style:
plt.style.use('dark_background')

# 7) Set up a wide (landscape) figure and push axes to the right (so we can write text on the left):
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('black')
ax.set_facecolor('#222222')
fig.subplots_adjust(left=0.30, right=0.95, top=0.90, bottom=0.10)

# 8) Initialize an empty scatter (with circular dots, white edge):
scatter = ax.scatter([], [], s=100, marker='o', edgecolors='white', linewidth=0.5)

# 9) Two text‐annotations for sample sentences:
dx = (x_max - x_min) * 0.02
dy = (y_max - y_min) * 0.02
text0 = ax.text(0, 0, sentences[idx0], color='white', fontsize=9,
                backgroundcolor='#000000', alpha=0.6)
text1 = ax.text(0, 0, sentences[idx1], color='white', fontsize=9,
                backgroundcolor='#000000', alpha=0.6)

# 10) Fix the PC‐axes limits and labels:
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel("PC 1", fontsize=14, color='white')
ax.set_ylabel("PC 2", fontsize=14, color='white')
ax.tick_params(colors='white')

# 11) Static text on the left (titles/meta):
fig.text(0.05, 0.85, "GPT2", fontsize=24, fontweight='bold', color='white')
fig.text(
    0.048, 0.7,
    f"Word='bank'  \nLayers=1–{n_layers} \nBest GDV Layer=13",
    fontsize=16, color='white'
)
dynamic_text = fig.text(0.05, 0.70, "", fontsize=18, color='white')  # will update each frame

# 12) Legend for Group 0 / Group 1 (pink vs. green):
pink  = '#ff69b4'
green = '#00ff00'
from matplotlib.lines import Line2D
handles = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor=pink,  markersize=10, linestyle=''),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=green, markersize=10, linestyle='')
]
legend = ax.legend(handles, ['Group 0','Group 1'], loc='upper right',
                   facecolor='#333333', edgecolor='white')
for text in legend.get_texts():
    text.set_color('white')

colors = np.array([pink if g == 0 else green for g in group])

# 13) The frame‐update function:
def update(frame):
    # Which “base” layer index we’re between:
    if frame == total_frames - 1:
        i_layer = n_layers - 1
        t = 0.0
    else:
        i_layer = frame // n_interp
        t = (frame % n_interp) / n_interp

    # Linear interpolation of the PCA points and GDV:
    if i_layer < n_layers - 1:
        x_interp  = (1 - t)*x_all[i_layer] + t*x_all[i_layer+1]
        y_interp  = (1 - t)*y_all[i_layer] + t*y_all[i_layer+1]
        gdv_interp = (1 - t)*gdv_all[i_layer] + t*gdv_all[i_layer+1]
    else:
        x_interp  = x_all[i_layer]
        y_interp  = y_all[i_layer]
        gdv_interp = gdv_all[i_layer]

    coords = np.vstack((x_interp, y_interp)).T
    scatter.set_offsets(coords)
    scatter.set_facecolors(colors)

    # “Display” the nearest integer layer in the left text:
    if t < 0.5:
        display_idx = i_layer
    else:
        display_idx = i_layer + 1 if (i_layer + 1) < n_layers else i_layer
    display_gdv = gdv_all[display_idx]
    dynamic_text.set_text(f"Layer {display_idx+1}\nGDV = {display_gdv:.4f}")

    # Reposition the two sample annotations:
    text0.set_position((x_interp[idx0] + dx, y_interp[idx0] + dy))
    text1.set_position((x_interp[idx1] + dx, y_interp[idx1] - dy))

    return scatter, dynamic_text, text0, text1

# 14) Build & save the animation:
ani = FuncAnimation(
    fig, update,
    frames=total_frames,
    blit=True,
    interval=100,   # 100 ms between frames → 10 fps
    repeat=True
)

output_path = 'gpt2_bank_animation_dark_lefttext.gif'
writer = PillowWriter(fps=10)
ani.save(output_path, writer=writer)

plt.close(fig)
print("Saved GIF to:", output_path)
