from collections import defaultdict
from typing import List
from copy import copy

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

locker_usage_bucket = [user_locker_usage.count(n) for n in unique_usage_number]

bucket_min_dict = defaultdict(int)


def find_smallest_number_of_lockers(current_locker_usage_bucket: List, current_locker_demanded: int, current_locker_available: int):

    results = []

    if sum(current_locker_usage_bucket) == 0:
        return 0

    if current_locker_available >= S:
        return 0

    for index, current_locker_usage in enumerate(current_locker_usage_bucket):
        if current_locker_usage > 0:
            next_locker_usage_bucket = copy(current_locker_usage_bucket)
            next_locker_usage_bucket[index] -= 1
            next_locker_available = current_locker_available + unique_usage_number[index]

            if (next_locker_available) >= S:
                results.append(unique_usage_number[index])
            else:
                remaining_buckets_smallest_value = bucket_min_dict.get(tuple(next_locker_usage_bucket))
                if remaining_buckets_smallest_value is None:
                    remaining_buckets_smallest_value = find_smallest_number_of_lockers(next_locker_usage_bucket,
                                                                                       current_locker_demanded=current_locker_demanded - unique_usage_number[index],
                                                                                       current_locker_available=next_locker_available)
                results.append(unique_usage_number[index] + remaining_buckets_smallest_value)

    result = min(results)
    bucket_min_dict[tuple(current_locker_usage_bucket)] = result

    return result


result = find_smallest_number_of_lockers(locker_usage_bucket, current_locker_demanded=S, current_locker_available=M-sum(user_locker_usage))

print(result)

