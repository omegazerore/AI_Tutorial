from typing import List

memory_hashmap = {} # key: coordinate, value: smallest_height_diff, shortest_distance


def find_lowest_diff_path(x, y, path: List):

    if (x == n-1) and (y == n-1):
        return 0, 0

    if (x, y) in memory_hashmap:
        return memory_hashmap[(x, y)]

    local_status = []
    path += [(x, y)]

    if ()

    if (y != n - 1) and (x, y+1) not in path: #go right
        min_height_diff_r, min_distance_r = find_lowest_diff_path(x, y+1, path)
        height_diff_r = abs(h[x][y] - h[x][y+1])
        local_status.append((max(height_diff_r, min_height_diff_r), min_distance_r + 1))
    if (x != n - 1) and (x+1, y) not in path: # go down
        min_height_diff_d, min_distance_d = find_lowest_diff_path(x+1, y, path)
        height_diff_d = abs(h[x][y] - h[x+1][y])
        local_status.append((max(height_diff_d, min_height_diff_d), min_distance_d + 1))
    if (y != 0) and ((x < n-1) and (x > 0)) and ((x, y-1) not in path): # go left
        min_height_diff_l, min_distance_l = find_lowest_diff_path(x, y-1, path)
        height_diff_l = abs(h[x][y] - h[x][y-1])
        local_status.append((max(height_diff_l, min_height_diff_l), min_distance_l + 1))
    if (x != 0) and ((y < n-1) and (y > 0)) and (x-1, y) not in path: # go up
        min_height_diff_u, min_distance_u = find_lowest_diff_path(x-1, y, path)
        height_diff_u = abs(h[x][y] - h[x-1][y])
        local_status.append((max(height_diff_u, min_height_diff_u), min_distance_u + 1))

    local_status.sort()

    memory_hashmap[(x, y)] = local_status[0]

    return local_status[0][0], local_status[0][1]


n = int(input())

h = []

for _ in range(n):
    h.append(list(map(int, input().split())))

min_height, min_distance = find_lowest_diff_path(0, 0, path=[])

print(min_height)
print(min_distance)
