"""
In a magical land, there is a type of rabbit that only eats carrots, and it eats only one carrot per day.

There are four types of carrots, categorized by their colors: red carrot, white carrot, yellow carrot, and moldy carrot (black).

After eating a carrot, the rabbit's weight will change as follows:

Eating a red carrot will increase the rabbit's weight by x grams.

Eating a white carrot will increase the rabbit's weight by y grams.

Eating a yellow carrot will decrease the rabbit's weight by z grams (pickled yellow carrots taste terrible…).

Eating a moldy carrot will decrease the rabbit's weight by w grams and will also inflict a poisoned status.

When poisoned, the rabbit loses n grams per day (the day it gets poisoned does not count).

Moreover, the poisoned status is cumulative. The initial weight of the rabbit is m. Each morning,
the rabbit first suffers the effects of poisoning (if any), and in the evening, it eats a carrot.

Now, given x,y,z,w,n, and m, determine the rabbit's weight after a certain number of days!

P.S. The details above are very important, so read them carefully!

"""


N = int(input())

for _ in range(N):
    poison = 0
    x, y, z, w, n, m = list(map(int, input().split()))

    map_ = {'0': 0,
            '1': x,
            '2': y,
            '3': -z,
            '4': -w}

    for idx, carrot in enumerate(list(input().split())):
        m -= poison * n
        if m < 0:
            print("bye~Rabbit")
            break
        m += map_[carrot]
        print(f"Day {idx+1}: weight = {m}g")
        if m < 0:
            print("bye~Rabbit")
            break
        if carrot == '4':
            poison += 1
    if m >= 0:
        print(f"{m}g")


