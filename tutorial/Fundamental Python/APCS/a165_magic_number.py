def iteration(number: str, step: int):

    if step == 1:
        for digit in "123456789":
            if digit not in number:
                iteration(number=number + digit, step=step + 1)
    elif (step >= 2) and (step <= 8):
        if int(number)%step == 0:
            for digit in "123456789":
                if digit not in number:
                    iteration(number = number+digit, step=step+1)
    else:
        if int(number)%step == 0:
            print(number)


for i in "123456789":
    iteration(i, 1)