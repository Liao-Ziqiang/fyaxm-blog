import math
from functools import reduce
from math import gcd

def get_divisors(n):
    """返回 n 的所有正因数（从小到大排序）"""
    divisors = set()
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divisors.add(i)
            divisors.add(n // i)
    return sorted(divisors)

def generate_scale(a, b, c = 0, d = 0):
    """
    根据公式 2^m * 3^a * 5^b 生成频率
    """
    A = (3 ** a) * (5 ** b) * (7 ** c) * (11 ** d)
    print('===========================================')
    print(f"计算 A = 3^{a} * 5^{b} * 7^{c} * 11^{d} = {A}")
    
    # 找到 A 的所有正因数
    divisors = get_divisors(A)
    print(f"A 的因数: {divisors}")
    
    # 尝试不同的 m，保证频率落在八度内
    m = 0
    while True:
        E = 2 ** m
        frequencies = []
        success = True
        
        for d in divisors:
            if d == 0:
                success = False
                break
            ratio = E / d
            if ratio <= 1:
                m_i = 0
            else:
                m_i = math.ceil(math.log2(ratio))
            
            freq = (2 ** m_i) * d
            
            if E <= freq < 2 * E:
                frequencies.append(freq)
            else:
                success = False
                break
        
        if success:
            frequencies.append(2 * E)
            frequencies = sorted(frequencies)
            
            # 缩放，保证比例互质
            scale_gcd = reduce(gcd, frequencies)
            scaled_frequencies = [freq // scale_gcd for freq in frequencies]
            
            scale_ratio = ":".join(map(str, scaled_frequencies))
            print(f"生成音阶: {scale_ratio}")
            print(f"音阶数（不含八度音）: {len(scaled_frequencies) - 1}")
            return scale_ratio
        
        m += 1
        
        if m > 20:
            print("无法生成音阶。")
            return ""


if __name__ == '__main__':
    generate_scale(1, 1)
    generate_scale(4, 1)
    generate_scale(2, 2, 1)