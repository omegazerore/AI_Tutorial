"""
Solving the quadratic equation: a*x^2+b*x+c=0
"""

a, b, c = map(int, input().split(" "))

sqrt_ = b**2 - 4 * a * c
if sqrt_ < 0:
    print("No real root")
elif sqrt_ == 0:
    x = int(-b/(2 * a))
    print(f"Two same roots x={x}")
else:
    sqrt_root = sqrt_**(0.5)
    x1 = (-b + sqrt_root)/(2 * a)
    x2 = (-b - sqrt_root)/(2 * a)
    if x1 > x2:
        x_large = x1
        x_small = x2
    else:
        x_large = x2
        x_small = x1
    print(f"Two different roots x1={int(x_large)} , x2={int(x_small)}")