import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Define the notes including low and high octaves
notes = ['C₀', 'C#₀', 'D₀', 'D#₀', 'E₀', 'F₀',
         'F#₀', 'G₀', 'G#₀', 'A₀', 'A#₀', 'B₀', 'C₁']

# Define E(n) values corresponding to each interval
# Note: Diminished Fifth has been removed
interval_e_n = {
    1: 11,   # Minor Second
    2: 8,    # Major Second
    3: 8,    # Minor Third
    4: 7,    # Major Third
    5: 5,    # Perfect Fourth
    6: 14,   # Tritone
    7: 4,    # Perfect Fifth (corrected)
    8: 8,    # Minor Sixth
    9: 7,    # Major Sixth
    10: 9,   # Minor Seventh
    11: 10,  # Major Seventh
    12: 2    # Octave
}

# Create a 13x13 matrix to store E(n) values
matrix = np.zeros((13, 13))

for i in range(13):
    for j in range(13):
        diff = j - i
        if diff == 0:
            e_value = 1    # Unison
        elif diff == 12:
            e_value = 2    # Octave
        elif 0 < diff < 12:
            e_value = interval_e_n.get(diff, np.nan)
        elif diff == -12:
            e_value = 2    # Octave (reverse)
        elif diff < 0:
            # Handle cases lower than the current octave, e.g., C₀ and B₀
            # Use positive difference to get corresponding E(n) value
            e_value = interval_e_n.get((-diff) % 12, np.nan)
        else:
            e_value = np.nan  # Undefined in other cases
        matrix[i, j] = e_value

# Set color mapping: lower E(n) values are more harmonious, darker colors
cmap = plt.cm.viridis_r  # Reverse viridis colormap, making lower E(n) darker

# Create custom normalization for clearer color mapping
norm = colors.Normalize(vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))

# Increase figure size to accommodate all labels
fig, ax = plt.subplots(figsize=(14, 14))

# Plot pseudocolor map
cax = ax.imshow(matrix, cmap=cmap, norm=norm, origin='lower')

# Add color bar, adjusting its width and position
cbar = fig.colorbar(cax, ax=ax, shrink=0.85, aspect=40)
cbar.set_label('Harmony E(n)', fontsize=14)

# Set axis ticks and labels
ax.set_xticks(np.arange(13))
ax.set_yticks(np.arange(13))
ax.set_xticklabels(notes, fontsize=12)
ax.set_yticklabels(notes, fontsize=12)

# Set axis label positions and rotation angles
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Add finer grid lines with a softer color
ax.set_xticks(np.arange(-0.5, 13, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 13, 1), minor=True)
ax.grid(which='minor', color='lightgrey', linestyle='-', linewidth=0.5)

# Remove major grid lines
ax.tick_params(which='minor', bottom=False, left=False)

# Set title
ax.set_title('Pitch Harmony Heatmap', fontsize=20, pad=20)

# Add E(n) values to each grid
for i in range(13):
    for j in range(13):
        e_value = matrix[i, j]
        if not np.isnan(e_value):
            # Adjust text color based on brightness
            color = cmap(norm(e_value))
            # Calculate brightness (YIQ model)
            brightness = (0.299*color[0] + 0.587*color[1] + 0.114*color[2])
            text_color = 'black' if brightness > 0.5 else 'white'
            ax.text(j, i, int(e_value), ha='center', va='center',
                    color=text_color, fontsize=8)

# Adjust layout to ensure labels are not cut off
plt.subplots_adjust(bottom=0.10, left=0.10, right=0.9, top=0.9)

plt.show()