import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

modes_data = [
    {"Mode": "(0, 0)", "Ratios": [1, 2]},  # I
    {"Mode": "(1, 0)", "Ratios": [2, 3, 4]},  # II
    {"Mode": "(0, 1)", "Ratios": [4, 5, 8]},  # III
    {"Mode": "(2, 0)", "Ratios": [8, 9, 12, 16]},  # IV
    {"Mode": "(1, 1)", "Ratios": [8, 10, 12, 15, 16]},  # V
    {"Mode": "(0, 2)", "Ratios": [16, 20, 25, 32]},  # VI
    {"Mode": "(3, 0)", "Ratios": [16, 18, 24, 27, 32]},  # VII
    {"Mode": "(2, 1)", "Ratios": [32, 36, 40, 45, 48, 60, 64]},  # VIII
    {"Mode": "(1, 2)", "Ratios": [54, 75, 80, 96, 100, 120, 128]},  # IX
    {"Mode": "(0, 3)", "Ratios": [64, 80, 100, 125, 128]},  # X
    {"Mode": "(4, 0)", "Ratios": [64, 72, 81, 96, 108, 128]},  # XI
    {"Mode": "(3, 1)", "Ratios": [128, 135, 144, 160, 180, 192, 216, 240, 256]},  # XII
    {"Mode": "(2, 2)", "Ratios": [128, 144, 150, 160, 180, 192, 200, 225, 240, 256]},  # XIII
    {"Mode": "(1, 3)", "Ratios": [256, 300, 320, 375, 384, 400, 480, 500, 512]},  # XIV
    {"Mode": "(0, 4)", "Ratios": [512, 625, 640, 800, 1000, 1024]},  # XV
    {"Mode": "(5, 0)", "Ratios": [128, 144, 162, 1922, 216, 243, 256]},  # XVI
    {"Mode": "(4, 1)", "Ratios": [256, 270, 288, 320, 324, 360, 384, 405, 432, 480, 512]},   # XVII
    {"Mode": "(3, 2)", "Ratios" : [512, 540, 576, 600, 640, 675, 720, 768, 800, 864, 900, 960, 1024]},
    {"Mode": "(3, 3)", "Ratios" : [2048, 2160, 2250, 2304, 2400, 2560, 2700, 2880, 3000, 3072, 3200, 3375, 3456, 3600, 3840, 4000, 4096]}
]

twelve_tet_ratios = [2**(n/12) for n in range(13)]
twelve_tet_notes = ['F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B', 'C', 'C♯', 'D', 'D♯', 'E', 'F']

# 创建一个映射函数，将频率比例转为对数值
def to_log_scale(ratio):
    return np.log2(ratio)

# 创建 DataFrame
mode_entries = []

for mode in modes_data:
    ratios = mode["Ratios"]
    min_ratio = min(ratios)
    normalized_ratios = [r / min_ratio for r in ratios]
    for ratio in normalized_ratios:
        mode_entries.append({"Mode": mode["Mode"], "Frequency": to_log_scale(ratio), "Note": None})

for ratio, note in zip(twelve_tet_ratios, twelve_tet_notes):
    mode_entries.append({"Mode": "12-TET", "Frequency": to_log_scale(ratio), "Note": note})

df = pd.DataFrame(mode_entries)
modes_sorted = ["12-TET"] + [m["Mode"] for m in modes_data]
plt.figure(figsize=(12, 6))
color_map = {
    'F': '#1f77b4', 'F♯': '#ff7f0e', 'G': '#2ca02c', 'G♯': '#d62728',
    'A': '#9467bd', 'A♯': '#8c564b', 'B': '#e377c2', 'C': '#7f7f7f',
    'C♯': '#bcbd22', 'D': '#17becf', 'D♯': '#aec7e8', 'E': '#ffbb78'
}

for mode in modes_sorted:
    mode_df = df[df['Mode'] == mode]
    y = modes_sorted.index(mode)
    if mode == "12-TET":
        for ratio, note in zip(twelve_tet_ratios, twelve_tet_notes):
            plt.scatter(to_log_scale(ratio), y, color=color_map.get(note, '#000000'), s=100)
    else:
        plt.scatter(mode_df['Frequency'], [y]*len(mode_df), color='gray', s=50, alpha=0.7)

plt.yticks(range(len(modes_sorted)), modes_sorted, fontsize=12)
plt.ylim(-1, len(modes_sorted))
log_tet_ratios = [to_log_scale(r) for r in twelve_tet_ratios]
plt.xticks(log_tet_ratios, labels=twelve_tet_notes, fontsize=10)
plt.xlim(min(log_tet_ratios) - 0.1, max(log_tet_ratios) + 0.1)
plt.title('Comparison of Scales', fontsize=14)

# legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=note,
#                               markerfacecolor=color, markersize=10) 
#                    for note, color in color_map.items()]
# plt.legend(handles=legend_elements, title='Notes', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.grid(True, which='both', axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()