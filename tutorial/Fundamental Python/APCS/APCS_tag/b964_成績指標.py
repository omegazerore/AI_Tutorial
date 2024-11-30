"""
AC
18ms
"""


N = int(input())
scores = list(map(int, input().split()))

scores.sort()

print(*scores)

highest_fail = -100

for ix, score in enumerate(scores):
    if score < 60:
        if ix == N-1:
            print(score)
            print("worst case")
    else:
        if ix == 0:
            print("best case")
            print(score)
        else:
            if scores[ix-1] < 60:
                print(scores[ix-1])
                print(score)
