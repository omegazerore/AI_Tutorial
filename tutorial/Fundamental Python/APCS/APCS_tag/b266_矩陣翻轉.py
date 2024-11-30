# statement = str(input())

R, C, M = map(int, input().split())

data = [list(map(int, input().split())) for _ in range(R)]

# Inverse the operation
operations = list(map(int, input().split()))[::-1]

# statement = ""
# R, C, M = 3, 2, 3
# data = [[1, 1],[3, 1], [1, 2]]
# operations = [1, 0, 0][::-1]

for m in range(M):
    if operations[m] == 0: # rotation counterclockwise
        data_new = []
        for c in range(C-1, -1, -1):
            data_new.append([data[r][c] for r in range(R)])
        C = R
        data = data_new
        R = len(data)
    else:  # flip
        data_new = []
        for row in data[::-1]:
            data_new.append(row)
        data = data_new

# print(statement)
print(R, C)
for row in data:
    print(*row)

