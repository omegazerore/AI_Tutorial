'''
AC 39ms 4.9MB

'''

data = {}

n, m = map(int, input().split())

idx_start = None
idy_start = None
v_min = 1000000

for idx in range(n):
    values = map(int, input().split())
    for idy, v in enumerate(values):
        data[(idx, idy)] = v
        if v < v_min:
            v_min = v
            idx_start = idx
            idy_start = idy

summation = v_min
del data[(idx_start, idy_start)]

while True:
    candidate_loc = []
    candidate_v = []
    if (idx_start, idy_start+1) in data:
        candidate_loc.append((idx_start, idy_start+1))
        candidate_v.append(data[candidate_loc[-1]])
    if (idx_start, idy_start-1) in data:
        candidate_loc.append((idx_start, idy_start-1))
        candidate_v.append(data[candidate_loc[-1]])
    if (idx_start+1, idy_start) in data:
        candidate_loc.append((idx_start+1, idy_start))
        candidate_v.append(data[candidate_loc[-1]])
    if (idx_start-1, idy_start) in data:
        candidate_loc.append((idx_start-1, idy_start))
        candidate_v.append(data[candidate_loc[-1]])

    if len(candidate_v) == 0:
        break
    else:
        min_value = min(candidate_v)
        loc = candidate_v.index(min_value)
        idx_start, idy_start = candidate_loc[loc]
        summation += min_value
        del data[(idx_start, idy_start)]

print(summation)