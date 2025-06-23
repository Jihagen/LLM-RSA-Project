import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

# 1) Load the GDV‐per‐layer data:
file_path = 'results/gpt2_gdv/gpt2_bank_gdv.pkl'
with open(file_path, 'rb') as f:
    data = pickle.load(f)

layers        = data['sorted_layers']
layer_data    = data['layer_data']
gdv_per_layer = data['gdv_per_layer']

# 2) Extract PCA coords, group labels, sentences, and GDV values:
x_all     = [layer_data[layer]['x']        for layer in layers]
y_all     = [layer_data[layer]['y']        for layer in layers]
group     = layer_data[layers[0]]['group']
sentences = layer_data[layers[0]]['sentence']
gdv_all   = [gdv_per_layer[layer]           for layer in layers]

# 3) Compute global axis limits with 10% margin:
all_x     = np.concatenate(x_all)
all_y     = np.concatenate(y_all)
mx, Mx    = all_x.min(), all_x.max()
my, My    = all_y.min(), all_y.max()
dx_margin = (Mx - mx) * 0.10
dy_margin = (My - my) * 0.10
x_min, x_max = mx - dx_margin, Mx + dx_margin
y_min, y_max = my - dy_margin, My + dy_margin

# 4) Pick two example indices (one per group):
idx0 = 0
idx1 = np.where(np.array(group) == 1)[0][0]

# 5) Animation parameters:
n_layers     = len(layers)
n_interp     = 5
total_frames = (n_layers - 1) * n_interp + 1

# 6) Switch to light‐background style:
plt.style.use('default')

# 7) Figure & axes (landscape), leaving room on left for text:
fig, ax = plt.subplots(figsize=(10, 6))
fig.subplots_adjust(left=0.30, right=0.95, top=0.90, bottom=0.10)

# 8) Initialize empty scatter:
scatter = ax.scatter([], [], s=100, marker='o',
                     edgecolors='black', linewidth=0.5)

# 9) Two text‐annotations with white semi‐transparent boxes:
dx = (x_max - x_min) * 0.02
dy = (y_max - y_min) * 0.02
text0 = ax.text(0, 0, sentences[idx0], color='black', fontsize=9,
                backgroundcolor='white', alpha=0.6)
text1 = ax.text(0, 0, sentences[idx1], color='black', fontsize=9,
                backgroundcolor='white', alpha=0.6)

# 10) Fix axes limits & labels in dark text:
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel("PC 1", fontsize=14, color='black')
ax.set_ylabel("PC 2", fontsize=14, color='black')
ax.tick_params(colors='black')

# 11) Static meta text on left:
fig.text(0.05, 0.85, "GPT2", fontsize=24, fontweight='bold', color='black')
fig.text(
    0.05, 0.70,
    f"Word = 'bank'\nLayers = 1–{n_layers}\nBest GDV Layer = 13",
    fontsize=16, color='black'
)
dynamic_text = fig.text(0.05, 0.60, "", fontsize=18, color='black')

# 12) Legend for Group 0 / Group 1 (pink vs. green):
pink  = '#ff69b4'
green = '#00ff00'
handles = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor=pink,  markersize=10, linestyle=''),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=green, markersize=10, linestyle='')
]
legend = ax.legend(handles, ['Group 0','Group 1'], loc='upper right',
                   facecolor='white', edgecolor='black')
for text in legend.get_texts():
    text.set_color('black')

colors = np.array([pink if g == 0 else green for g in group])

# 13) Frame‐update function:
def update(frame):
    # Determine base layer & interpolation factor
    if frame == total_frames - 1:
        i_layer, t = n_layers - 1, 0.0
    else:
        i_layer = frame // n_interp
        t       = (frame % n_interp) / n_interp

    # Interpolate coords & GDV
    if i_layer < n_layers - 1:
        x_interp   = (1 - t)*x_all[i_layer]   + t*x_all[i_layer+1]
        y_interp   = (1 - t)*y_all[i_layer]   + t*y_all[i_layer+1]
        gdv_interp = (1 - t)*gdv_all[i_layer] + t*gdv_all[i_layer+1]
    else:
        x_interp, y_interp, gdv_interp = x_all[i_layer], y_all[i_layer], gdv_all[i_layer]

    scatter.set_offsets(np.vstack((x_interp, y_interp)).T)
    scatter.set_facecolors(colors)

    # Update dynamic text:
    display_idx = i_layer + (t >= 0.5 and i_layer < n_layers - 1)
    display_gdv = gdv_all[int(display_idx)]
    dynamic_text.set_text(f"Layer {int(display_idx)+1}\nGDV = {display_gdv:.4f}")

    # Reposition sample annotations:
    text0.set_position((x_interp[idx0] + dx, y_interp[idx0] + dy))
    text1.set_position((x_interp[idx1] + dx, y_interp[idx1] - dy))

    return scatter, dynamic_text, text0, text1

# 14) Build & save the animation:
ani = FuncAnimation(
    fig, update,
    frames=total_frames,
    blit=True,
    interval=100,
    repeat=True
)

output_path = 'gpt2_bank_animation_light.gif'
writer = PillowWriter(fps=10)
ani.save(output_path, writer=writer)

plt.close(fig)
print("Saved GIF to:", output_path)
