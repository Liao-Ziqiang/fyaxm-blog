import math
from collections import defaultdict

def gcd(a, b):
    """计算两个数的最大公约数"""
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    """计算两个数的最小公倍数"""
    return abs(a * b) // gcd(a, b)

def prime_factors(n):
    """
    将一个数分解为质因数，并返回一个字典，
    其中键是质因数，值是对应的指数
    """
    factors = defaultdict(int)
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
    return factors

def calculate_E(n):
    """
    根据欧拉的公式计算次数 E(n)
    """
    if n == 1:
        return 1
    factors = prime_factors(n)
    E = 1
    for p, a in factors.items():
        E += a * (p - 1)
    return E

# 定义音程列表
intervals = [
    {"name": "纯一度（Unison）", "ratio": "1:1"},
    {"name": "小二度（Minor Second）", "ratio": "16:15"},
    {"name": "大二度（Major Second）", "ratio": "9:8"},
    {"name": "小三度（Minor Third）", "ratio": "6:5"},
    {"name": "大三度（Major Third）", "ratio": "5:4"},
    {"name": "纯四度（Perfect Fourth）", "ratio": "4:3"},
    {"name": "增四度（Tritone）", "ratio": "45:32"},
    {"name": "减五度（Diminished Fifth）", "ratio": "64:45"},
    {"name": "纯五度（Perfect Fifth）", "ratio": "3:2"},
    {"name": "小六度（Minor Sixth）", "ratio": "8:5"},
    {"name": "大六度（Major Sixth）", "ratio": "5:3"},
    {"name": "小七度（Minor Seventh）", "ratio": "9:5"},
    {"name": "大七度（Major Seventh）", "ratio": "15:8"},
    {"name": "纯八度（Octave）", "ratio": "2:1"},
]

# 计算并打印结果
print(f"{'序号':<5}{'名称':<25}{'比率':<10}{'最小公倍数 (n)':<20}{'质因数分解':<30}{'次数 E(n)':<10}")
print("-" * 80)

for idx, interval in enumerate(intervals, start=1):
    ratio = interval["ratio"]
    a, b = map(int, ratio.split(":"))
    n = lcm(a, b)
    E_n = calculate_E(n)
    # 获取质因数分解用于显示
    if n == 1:
        factors_display = "-"
    else:
        factors = prime_factors(n)
        factors_display = " * ".join([f"{p}^{exp}" if exp >1 else f"{p}" for p, exp in sorted(factors.items())])
    print(f"{idx:<5}{interval['name']:<25}{ratio:<10}{n:<20}{factors_display:<30}{E_n:<10}")