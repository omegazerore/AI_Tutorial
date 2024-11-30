'''
AC 19ms 3.3MB

'''

def f(x):

    return 2 * x - 3

def g(x, y):

    return 2 * x + y - 7

def h(x, y, z):

    return 3 * x - 2 * y + z

def executation(fuc, arguments):

    if fuc == 'f':
        return [f(arguments[0])] + arguments[1:]
    elif fuc == 'g':
        return [g(*arguments[:2])] + arguments[2:]
    else:
        return [h(*arguments[:3])] + arguments[3:]


input_values = input().split()

arguments = []
for v in input_values[::-1]:
    if v.isalpha():
        arguments = executation(v, arguments)
    else:
        arguments = [int(v)] + arguments

print(int(arguments[0]))
