'''
Result
84ms, 3.9MB
'''

from collections import defaultdict

C, N = map(int, input().split(" "))

  # in the reverse order
values = []
number_of_coins = []

while True:
    try:
        v = int(input())
        values.append(v)
    except EOFError:
        break

values.sort(reverse=True)
values_dict = {i: v for i, v in enumerate(values)}
residue_dict = defaultdict(int)

def retrieve_minimum_number_of_coins(c_res, location_index):

    quotion_max = c_res//values[location_index]
    residue_min = c_res % values[location_index]

    if location_index == N-1:
        if residue_min == 0:
            return quotion_max
        else:
            return None
    else:
        if residue_min ==0:
            return quotion_max

    total_combinations = []
    for quotion in range(0, quotion_max+1):
        next_residue = residue_min + quotion * values[location_index]
        if tuple([location_index, next_residue]) in residue_dict:
            number_of_coins_res = residue_dict[tuple([location_index, next_residue])]
        else:
            number_of_coins_res = retrieve_minimum_number_of_coins(next_residue, location_index=location_index+1)
            residue_dict[tuple([location_index, next_residue])] = number_of_coins_res
        if number_of_coins_res is not None:
            total_combinations.append(number_of_coins_res + quotion_max - quotion)
    if len(total_combinations) > 0:
        return min(total_combinations)

n = retrieve_minimum_number_of_coins(C, location_index=0)

print(n)


