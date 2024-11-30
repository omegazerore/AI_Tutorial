R, C, k, m = list(map(int, input().split()))

min_p, max_p = 100 * R * C, 1

city_dict = {}

for r in range(R):
    population = list(map(int, input().split()))
    for c, p in enumerate(population):
        if p != -1:
            city_dict[(r, c)] = p

multiplier = {}

for location in city_dict.keys():
    r, c = location
    n_1 = int((city_dict.get((r+1, c), -1) >= 0))
    n_2 = int((city_dict.get((r-1, c), -1) >= 0))
    n_3 = int((city_dict.get((r, c-1), -1) >= 0))
    n_4 = int((city_dict.get((r, c+1), -1) >= 0))
    multiplier[(r, c)] = n_1 + n_2 + n_3 + n_4

for _ in range(m):
    flow_dict = {}
    for location, v in city_dict.items():
        r, c = location
        flow_dict[(r, c)] = v//k   # math.ceil(v/k)

    city_dict_updated = {}
    for location, v in city_dict.items():
        r, c = location
        f_1 = flow_dict.get((r+1, c), 0)
        f_2 = flow_dict.get((r-1, c), 0)
        f_3 = flow_dict.get((r, c+1), 0)
        f_4 = flow_dict.get((r, c-1), 0)
        city_dict_updated[(r, c)] = v + f_1 + f_2 + f_3 + f_4 - v//k * multiplier[location]

    city_dict = city_dict_updated


for _, v in city_dict.items():

    if v < min_p:
        min_p = v
    if v > max_p:
        max_p = v