'''
AC 95ms 3.8MB
'''

from collections import defaultdict


n, m, k = list(map(int, input().split()))

server_to_city_traffic = []

for i in range(n):
    traffic = list(map(int, input().split()))
    server_to_city_traffic.append(traffic)

plans = []
for i in range(k):
    plans.append(list(map(int, input().split())))

plan_to_cost = []

for plan in plans:
    city_to_city_traffic_hashmap = defaultdict(lambda: defaultdict(int))
    for server, server_location in enumerate(plan):
        for iloc, t in enumerate(server_to_city_traffic[server]):
            city_to_city_traffic_hashmap[server_location][iloc] += t

    cost = 0
    for city_start, traffic_hashmap in city_to_city_traffic_hashmap.items():
        for city_end, traffic in traffic_hashmap.items():
            if city_start == city_end :
                cost += traffic
            else:
                cost += 3 * (min(1000, traffic)) +  2 * (max(0, traffic - 1000))

    plan_to_cost.append(cost)

print(min(plan_to_cost))
