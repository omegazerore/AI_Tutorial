'''
AC 18ms 3.3MB
'''

n = 0

# game 1:

for i in range(2):
    if i == 0:
        home_scores = list(map(int, input().split()))
    else:
        away_scores = list(map(int, input().split()))

print(f"{sum(home_scores)}:{sum(away_scores)}")

n += int(sum(home_scores) > sum(away_scores))

# game 2:

for i in range(2):
    if i == 0:
        home_scores = list(map(int, input().split()))
    else:
        away_scores = list(map(int, input().split()))

print(f"{sum(home_scores)}:{sum(away_scores)}")

n += int(sum(home_scores) > sum(away_scores))

if n == 2:
    print("WIN")
elif n == 1:
    print("Tie")
else:
    print("Lose")