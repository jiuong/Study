# 从3进制转10进制
def from_base3_digits(digits: list[int]) -> int:
    n = 0
    for i, d in enumerate(digits):
        n += d * (3 ** i)
    return n

# 获取3进制位数
def get_base3_digits(n: int, length: int) -> list[int]:
    digits = []
    for _ in range(length):
        # 取当前最低位
        digits.append(n % 3)
        # 右移一位（除以3）
        n = n // 3
    # 逆序：digits[0] = 最低位，也可以反过来
    return digits