'''
AC 0.4s 9.3MB

'''

import math

N, K = map(int, input().split())
service_points = list(map(int, input().split()))
service_points.sort()

max_service_point = max(service_points)
min_service_point = min(service_points)

max_diameter = max_service_point - min_service_point
min_diameter = 0

def check(diameter):

    k = 1

    coverage = min_service_point + diameter

    for service_point in service_points:
        if service_point <= coverage:
            continue
        else:
            k += 1
            if k > K:
                return False
            else:
                coverage = service_point + diameter

    return True

if K == 1:
    print(max_diameter)
else:
    test_diameter = math.ceil((max_diameter + min_diameter) / 2)
    while (test_diameter != max_diameter):
        if_covered = check(test_diameter)
        if if_covered:
            max_diameter = test_diameter
        else:
            min_diameter = test_diameter

        test_diameter = math.ceil((max_diameter + min_diameter) / 2)

    print(max_diameter)








