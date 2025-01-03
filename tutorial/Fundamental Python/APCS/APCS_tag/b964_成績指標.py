"""
一次考試中，於所有及格學生中獲取最低分數者最為幸運，反之，於所有不及格同學中，獲取最高分數者，可以說是最為不幸，而此二種分數，可以視為成績指標。

請你設計一支程式，讀入全班成績(人數不固定)，請對所有分數進行排序，並分別找出不及格中最高分數，以及及格中最低分數。

當找不到最低及格分數，表示對於本次考試而言，這是一個不幸之班級，此時請你印出「worst case」；反之，當找不到最高不及格分數時，請你印出「best case」。

( 註：假設及格分數為 60 )。


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
