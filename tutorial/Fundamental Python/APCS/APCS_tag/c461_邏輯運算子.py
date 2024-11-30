'''

AC 19ms 3.3MB
'''

input_ = list(map(int, input().split()))

printed = False

if (bool(input_[0]) and (bool(input_[1]))) == bool(input_[2]):
    print('AND')
    printed = True

if (bool(input_[0]) or (bool(input_[1]))) == bool(input_[2]):
    print('OR')
    printed = True

if (bool(input_[0]) != (bool(input_[1]))) == bool(input_[2]):
    print('XOR')
    printed = True

if not printed:
    print("IMPOSSIBLE")

