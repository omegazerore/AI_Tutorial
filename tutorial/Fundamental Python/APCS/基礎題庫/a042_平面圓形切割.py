"""
對任意正整數n，平面上的n 個圓最多可將平面切成幾個區域？
"""


while True:

    try:
        n = int(input())
        print(n ** 2 - n + 2)
    except EOFError:
        break

