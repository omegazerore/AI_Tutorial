'''
actions: 0: 左, 1: 上, 2: 右, 3: 下

update apex iteratively

if touch apex(N//2 + i, N//2 - i): 往上
if touch apex(N//2 - i, N//2 - i): 往右
if touch apex(N//2 - i, N//2 + i): 往下
if touch apex(N//2 + i, N//2 + i): 往左

AC 27ms 3.6MB
'''


def update_location(x, y, action):
    if action == 0:
        return x-1, y
    if action == 1:
        return x, y-1
    if action == 2:
        return x+1, y
    if action == 3:
        return x, y+1

N = int(input())
initial_action = int(input())

data = []

while True:
    try:
        data.append(list(map(int, input().split())))
    except EOFError:
        break

# N = 5
# initial_action = 0
# data = [[3, 4, 2, 1, 4],
#         [4, 2, 3, 8, 9],
#         [2, 1, 9, 5, 6],
#         [4, 2, 3, 7, 8],
#         [1, 2, 6, 4, 3]]

center = N//2

x = center
y = center

output = [data[y][x]]

x_start, y_start = update_location(x, y, initial_action)

action = (initial_action + 1) % 4

count = 0

for i in range(1, N//2 + 1):

    for count in range(4):

        increment = 2 * i

        if count == 3:
            increment += 1
        if count == 0:
            increment -= 1

        if action == 2:
            output += data[y_start][x_start: x_start + increment]
            x_start = x_start + increment
        elif action == 0:
            if x_start - increment < 0:
                output += data[y_start][x_start:: -1]
            else:
                output += data[y_start][x_start: x_start - increment: -1]
            x_start = x_start - increment
        elif action == 1:
            if y_start - increment < 0:
                output += [d[x_start] for d in data[y_start:: -1]]
            else:
                output += [d[x_start] for d in data[y_start: y_start - increment: -1]]
            y_start = y_start - increment
        else:
            output += [d[x_start] for d in data[y_start: y_start + increment]]
            y_start = y_start + increment

        action = (action + 1) % 4

print("".join([str(o) for o in output]))