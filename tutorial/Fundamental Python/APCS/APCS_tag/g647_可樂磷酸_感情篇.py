'''
Linear optimization problem

AC 0.7s, 51.2MB
'''
import math

delta = math.pow(10, -6)

def check_condition_1(t1, t2):

    return (t1 * a1 + t2 * a2) >= n


def check_condition_2(t1, t2):

    return (t1 * b1 + t2 * b2) >= n


def tracing_back(x_0, y_0):

    solutions = []

    reverse_counting = 0

    x = x_0 - reverse_counting
    y = y_0 + reverse_counting

    while (check_condition_1(x, y)) and (check_condition_2(x, y)) and x >= 0:
        solutions.append((x, y, x + y))
        x -= 1
        y += 1

    return solutions

t = int(input())

for _ in range(t):

    a1, b1 = list(map(int, input().split()))
    a2, b2 = list(map(int, input().split()))
    n = int(input())

    x_limit = max(n / a1, n / b1)
    y_limit = max(n / a2, n / b2)

    # condition 1:
    if (a1/a2 <= 1) and (b1/b2 <= 1):
        print(0, math.ceil(y_limit))
    else:
        det = a2 * b1 - b2 * a1
        if det == 0: # two lines are parallel
            if math.ceil(x_limit) < math.ceil(y_limit):
                solutions = tracing_back(math.ceil(x_limit), 0)
                solution = solutions[-1]
                print(solution[0], solution[1])
            else:
                print(0, math.ceil(y_limit))
        else:
            t_2 = ((b1 - a1) * n) // det
            t_1 = ((a2 - b2) * n) // det
            if (t_1 < 0) or (t_2 < 0): # no intersection in the first quadrant
                if math.ceil(x_limit) < math.ceil(y_limit):
                    solutions = tracing_back(math.ceil(x_limit), 0)
                    solution = solutions[-1]
                    print(solution[0], solution[1])
                else:
                    print(0, math.ceil(y_limit))
            else:
                x_limit_floor = math.floor(x_limit)
                x_limit_ceil = math.ceil(x_limit)
                y_limit_floor = math.floor(y_limit)
                y_limit_ceil = math.ceil(y_limit)
                if check_condition_1(t_1, t_2) and check_condition_2(t_1, t_2):
                    print(t_1, t_2)
                else:
                    t_record = [(0, y_limit_ceil, y_limit_ceil)]
                    t_record.extend(tracing_back(x_limit_ceil, 0))

                    if check_condition_1(t_1, t_2+1) and check_condition_2(t_1, t_2+1):
                        solutions = tracing_back(t_1, t_2+1)
                        t_record.append(solutions[-1])
                        t_record.sort(key=lambda x: (x[2], x[0]))
                        print(t_record[0][0], t_record[0][1])
                    elif check_condition_1(t_1 + 1, t_2) and check_condition_2(t_1 + 1, t_2):
                        solutions = tracing_back(t_1 + 1, t_2)
                        t_record.append(solutions[-1])
                        t_record.sort(key=lambda x: (x[2], x[0]))
                        print(t_record[0][0], t_record[0][1])
                    else:
                        solutions = tracing_back(t_1 + 1, t_2 + 1)
                        t_record.append(solutions[-1])
                        t_record.sort(key=lambda x: (x[2], x[0]))
                        print(t_record[0][0], t_record[0][1])
