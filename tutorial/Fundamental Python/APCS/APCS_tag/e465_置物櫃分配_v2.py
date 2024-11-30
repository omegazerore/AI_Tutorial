'''
Current Result: 40%
#8 TLE

'''


from typing import Dict
from copy import copy
import math

M, S, N = map(int, input().split(" "))

user_locker_usage = list(map(int, input().split(" ")))

# M = 20
# S = 14
# N = 5
#
# user_locker_usage = [1, 7, 2, 8, 2]

set_usage_number = set(user_locker_usage)
set_usage_number.discard(0)

unique_usage_number = list(set_usage_number)

locker_usage_bucket = dict({n: user_locker_usage.count(n) for n in unique_usage_number})


def find_smallest_number_of_lockers(current_locker_usage_bucket: Dict, current_locker_demanded: int):

    results = []

    for locker_block, current_locker_usage in current_locker_usage_bucket.items():

        if current_locker_usage > 0:

            if locker_block == current_locker_demanded:
                return locker_block

            if current_locker_demanded % locker_block == 0:
                if current_locker_demanded//locker_block < current_locker_usage:
                    return current_locker_demanded

            next_locker_usage_bucket = copy(current_locker_usage_bucket)
            next_locker_usage_bucket[locker_block] -= 1
            if next_locker_usage_bucket[locker_block] == 0:
                del next_locker_usage_bucket[locker_block]

            # if len(next_locker_usage_bucket) == 1:
            #     return math.ceil(current_locker_demanded/locker_block) * locker_block

            if (current_locker_demanded-locker_block) < 0:
                results.append(locker_block)
            else:
                remaining_buckets_smallest_value = find_smallest_number_of_lockers(next_locker_usage_bucket,
                                                                                   current_locker_demanded=current_locker_demanded-locker_block)
                results.append(locker_block + remaining_buckets_smallest_value)

    result = min(results)

    return result


current_locker_available = M-sum(user_locker_usage)

if S < current_locker_available:
    print(0)
else:
    current_locker_demanded = S - current_locker_available
    result = find_smallest_number_of_lockers(locker_usage_bucket, current_locker_demanded=current_locker_demanded)
    print(result)


