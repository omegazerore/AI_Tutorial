from copy import copy

t = int(input())

counting = 0

ranking_list = []

while True:
    try:
        http, relevance = map(str, input().split(" "))
        ranking_list.append((http, int(relevance)))
        counting += 1
    except EOFError:

        for case in range(t):
            print(f"Case #{case + 1}:")
            sub_list = copy(ranking_list[case * 10: (case+1) * 10])
            max_relevance = 0
            result = []
            for element in sub_list:
                if element[1] > max_relevance:
                    result = [element[0]]
                    max_relevance = element[1]
                elif element[1] == max_relevance:
                    result.append(element[0])
            for element in result:
                print(element)
        break