from fractions import Fraction
from collections import defaultdict
import math
from functools import reduce

# 定义半音差到频率比率的映射
SEMITONE_TO_RATIO = {
    0: (1, 1),      # Unison
    1: (16, 15),    # Minor Second
    2: (9, 8),      # Major Second
    3: (6, 5),      # Minor Third
    4: (5, 4),      # Major Third
    5: (4, 3),      # Perfect Fourth
    6: (45, 32),    # Tritone
    7: (3, 2),      # Perfect Fifth
    8: (8, 5),      # Minor Sixth
    9: (5, 3),      # Major Sixth
    10: (9, 5),     # Minor Seventh
    11: (15, 8),    # Major Seventh
    12: (2, 1),     # Octave
}

def prime_factors(n):
    """
    将一个数分解为质因数，并返回一个字典，
    其中键是质因数，值是对应的指数
    """
    factors = defaultdict(int)
    original_n = n  # 保存原始的n用于打印
    # 处理2
    while n % 2 == 0:
        factors[2] += 1
        n //= 2
    # 处理奇数
    for i in range(3, int(math.isqrt(n)) + 1, 2):
        while n % i == 0:
            factors[i] += 1
            n //= i
    # 如果n本身是质数
    if n > 1:
        factors[n] += 1
    # 打印质因数分解结果
    factors_str = ' * '.join([f'{p}^{exp}' if exp >1 else f'{p}' for p, exp in factors.items()])
    print(f"质因数分解结果：{original_n} = {factors_str}")
    return factors

def calculate_E(n):
    """
    根据欧拉的公式计算次数 E(n)
    """
    if n == 1:
        print("n=1，E(n)=1")
        return 1
    factors = prime_factors(n)
    E = 1
    terms = []
    for p, a in factors.items():
        term = f"{a}*({p}-1)"
        E += a * (p - 1)
        terms.append(term)
    terms_str = " + ".join(terms)
    print(f"计算不和谐度 E(n): E(n) = 1 + {terms_str} = {E}")
    return E

def gcd(a, b):
    """计算两个数的最大公约数"""
    return math.gcd(a, b)

def lcm(a, b):
    """计算两个数的最小公倍数"""
    return a * b // gcd(a, b)

def lcm_list(lst):
    """计算一个列表中所有数的最小公倍数"""
    return reduce(lcm, lst, 1)

def compute_cumulative_frequencies(positions, semitone_to_ratio):
    """
    根据半音位置列表计算每个音相对于前一个音的频率比率
    并返回累积频率比率列表
    """
    if not positions:
        return []
    frequencies = [Fraction(1, 1)]  # 基础音
    print("\n计算累积频率比率（基于连续半音差）:")
    for i in range(1, len(positions)):
        delta = positions[i] - positions[i - 1]
        if delta not in semitone_to_ratio:
            raise ValueError(f"半音差 {delta} 不在映射表中。")
        ratio_num, ratio_den = semitone_to_ratio[delta]
        ratio = Fraction(ratio_num, ratio_den)
        current_freq = frequencies[-1] * ratio
        frequencies.append(current_freq)
        print(f"位置 {positions[i -1]} -> {positions[i]}: 半音差 {delta}，比率 {ratio_num}/{ratio_den}，累积频率 {current_freq}")
    print(f"累积频率比率列表: {[f'{f.numerator}/{f.denominator}' for f in frequencies]}")
    return frequencies

def compute_root_relative_frequencies(positions, semitone_to_ratio):
    """
    计算每个音相对于根音的频率比率，并返回根音相对频率比率列表
    """
    if not positions:
        return []
    root = positions[0]
    frequencies = [Fraction(1, 1)]  # 根音
    print("\n计算根音相对频率比率:")
    for i in range(1, len(positions)):
        delta = positions[i] - root
        if delta not in semitone_to_ratio:
            raise ValueError(f"半音差 {delta} 不在映射表中。")
        ratio_num, ratio_den = semitone_to_ratio[delta]
        ratio = Fraction(ratio_num, ratio_den)
        frequencies.append(ratio)
        print(f"位置 {root} -> {positions[i]}: 半音差 {delta}，比率 {ratio_num}/{ratio_den}")
    print(f"根音相对频率比率列表: {[f'{f.numerator}/{f.denominator}' for f in frequencies]}")
    return frequencies

def scale_frequencies(frequencies):
    """
    将频率比率转换为互质整数，并计算最小公倍数 n
    """
    denominators = [f.denominator for f in frequencies]
    lcm_den = lcm_list(denominators)
    print(f"\n所有分母的最小公倍数 (LCM) 为: {lcm_den}")
    scaled_frequencies = [f * lcm_den for f in frequencies]
    # 转换为整数
    scaled_frequencies_int = [f.numerator for f in scaled_frequencies]
    print(f"缩放后的频率（整数）: {scaled_frequencies_int}")
    # 计算 n，即所有频率的最小公倍数
    n = lcm_list(scaled_frequencies_int)
    print(f"所有缩放后频率的最小公倍数 (n) 为: {n}")
    return n, scaled_frequencies_int

def compute_E_of_chord(positions):
    """
    给定半音位置列表，计算和弦的不和谐度 E(n)
    并输出中间步骤
    """
    print("\n=== 计算和弦的不和谐度 ===")
    print(f"输入的半音位置列表: {positions}")
    
    # 计算累积频率比率
    root_relative_frequencies = compute_cumulative_frequencies(positions, SEMITONE_TO_RATIO)
    
    # 计算根音相对频率比率
    # root_relative_frequencies = compute_root_relative_frequencies(positions, SEMITONE_TO_RATIO)
    
    # 缩放根音相对频率比率
    n, scaled_frequencies_int = scale_frequencies(root_relative_frequencies)
    
    # 打印缩放后的频率比率为冒号分隔
    scaled_frequencies_str = ":".join(map(str, scaled_frequencies_int))
    print(f"缩放后频率比率（冒号分隔）: {scaled_frequencies_str}")
    
    # 计算不和谐度 E(n)
    E = calculate_E(n)
    print(f"\n最终和弦的不和谐度 E(n) 为: {E}\n")
    return E

# 示例用法
if __name__ == "__main__":
    # 示例 1：大七和弦 [0, 4, 7, 11]
    semitone_positions = [0, 4, 7]
    E_value_1 = compute_E_of_chord(semitone_positions)

    semitone_positions = [0, 3, 7]
    E_value_1 = compute_E_of_chord(semitone_positions)

    semitone_positions = [-1, 2, 5]
    E_value_1 = compute_E_of_chord(semitone_positions)