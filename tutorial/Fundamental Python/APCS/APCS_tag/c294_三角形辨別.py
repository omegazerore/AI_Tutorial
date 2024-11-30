"""
Ac 34ms 3.3MB
"""

segments = list(map(int, input().split()))

segments.sort()

a, b, c = segments

print(*segments)

if a + b <= c:
    print("No")
elif (a*a + b*b) < c*c:
    print("Obtuse")
elif (a*a + b*b) == c*c:
    print("Right")
elif (a*a + b*b) > c*c:
    print("Acute")