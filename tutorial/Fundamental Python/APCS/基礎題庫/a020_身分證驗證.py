mapping = {"A": '10', "B": '11', "C": '12', "D": '13', "E": '14',
           "F": '15', "G": '16', "H": '17', "I": '34', "J": '18',
           "K": '19', "L": '20', "M": '21', "N": '22', "O": '35',
           "P": '23', "Q": '24', "R": '25', "S": '26', "T": '27',
           "U": '28', "V": '29', "W": '32', "X": '30', "Y": '31',
           "Z": '33'}

ids = input()

# 1. English Character Transformation

a = mapping[ids[0]]
a_integer_list = []
for b in a:
    a_integer_list.append(int(b))

a_digit_list = []
for b in ids[1:]:
    a_digit_list.append(int(b))

for idx, i in enumerate(range(8, 0, -1)):
    a_digit_list[idx] = a_digit_list[idx] * i

final_list = a_integer_list + a_digit_list

total = sum(final_list)

if total%10 == 0:
    print("real")
else:
    print("fake")

numbers = [int(id) for id in mapping[ids[0]]] + [int(id) for id in ids[1:]]

numbers[1] = numbers[1] * 9

for idx, i in enumerate(range(8, 0, -1)):
    numbers[idx+2] = numbers[idx+2] * i

if sum(numbers) % 10 == 0:
    print("real")
else:
    print("fake")