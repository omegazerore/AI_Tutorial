'''
AC
76ms
'''

while True:
    try:
        N = int(input())

        sections = []

        for _ in range(N):
            L, R = map(int, input().split())
            if L < R:
                sections.append([L, R])

        if len(sections) == 0:
            print(0)
            continue

        sections.sort(reverse=True)

        section_merged = [sections[-1]]
        sections.pop()

        length = 0



        while sections:
            L, R = sections.pop()
            if L > section_merged[-1][1]:
                section_merged.append([L, R])
            else:
                if L <= section_merged[-1][1]:
                    if R > section_merged[-1][1]:
                        section_merged[-1][1] = R

        for section in section_merged:
            length += section[1] - section[0]
        # print(text.replace("入", "出"))
        print(length)

    except EOFError:
        break


