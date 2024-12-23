import numpy as np
import pandas as pd

def generate_equal_temperament_a440(base_frequency=440, num_notes=12):
    """生成十二平均律音阶频率"""
    return [base_frequency * (2 ** (n / num_notes)) for n in range(num_notes + 1)]

def scale_three_to_a440(base_frequency=440):
    """将音阶三的比例调整到以 A = 440 Hz 为基准"""
    scale_three_ratios = [512, 540, 576, 600, 640, 675, 720, 768, 800, 864, 900, 960, 1024]
    return [base_frequency * (r / 512) for r in scale_three_ratios]

def compare_scales_with_pandas(scale1, scale2):
    data = []
    for i in range(min(len(scale1), len(scale2))):
        diff_hz = scale2[i] - scale1[i]  # 差异（Hz）
        cents = 1200 * np.log2(scale2[i] / scale1[i])  # 差异（音分）
        data.append({
            "音阶三频率 (Hz)": scale1[i],
            "十二平均律频率 (Hz)": scale2[i],
            "差异（Hz）": diff_hz,
            "差异（音分）": cents
        })
    return pd.DataFrame(data)

scale_et_a440 = generate_equal_temperament_a440(base_frequency=440)
scale_three_a440 = scale_three_to_a440(base_frequency=440)
df_comparison = compare_scales_with_pandas(scale_three_a440, scale_et_a440)
df_comparison = df_comparison.round(4)
print(df_comparison)

df_comparison.to_csv("comparison_full_scale_a440.csv", index=False, float_format="%.4f")