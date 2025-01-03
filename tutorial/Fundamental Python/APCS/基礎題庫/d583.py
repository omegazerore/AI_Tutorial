while True:
    try:
        _ = input()

        my_list = list(map(int, input().split()))

        my_list.sort()

        print(" ".join([str(a) for a in my_list]))
    except EOFError:
        break