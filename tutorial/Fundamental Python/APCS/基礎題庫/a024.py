"""
https://en.wikipedia.org/wiki/Euclidean_algorithm

給定兩個整數，請求出它們的最大公因數

Given two integers, find their greatest common divisor (GCD).
"""

a, b = map(int, input("Give the numbers: ").split())

if a < b:
    r1 = b
    r2 = a
else:
    r2 = b
    r1 = a

print(f"r1: {r1}; r2: {r2}")

residue = r1 % r2
print(f"residue: {residue}")

while residue != 0:
    print("****************************")
    r1, r2 = r2, residue
    print(f"r1: {r1}; r2: {r2}")
    residue = r1 % r2
    print(f"residue: {residue}")
    if residue == 0:
        print("Get GCD")
    print("****************************")
print(r2)