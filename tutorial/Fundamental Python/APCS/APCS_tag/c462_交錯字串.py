'''
AC 86ms 6.6MB
'''

import re

k = int(input())
string = str(input())

# k = 1
# string = 'DDaaAAbbbC'

if len(string) < k:
    print(0)
else:
    regex_upper = re.compile("[0]+")
    regex_lower = re.compile("[1]+")

    case_checklist = [int(s.islower()) for s in string]
    sum_of_case_checklist = sum(case_checklist)

    binary_string = "".join([str(c) for c in case_checklist])

    if sum_of_case_checklist == len(case_checklist):
        print(k)
    elif sum_of_case_checklist == 0:
        print(k)
    else:
        output_list = []
        while len(binary_string) > 0:
            if binary_string[0]=='1':
                match = regex_lower.match(binary_string)
            else:
                match = regex_upper.match(binary_string)

            output_list.append(match.group())
            binary_string = binary_string[match.span()[1]:]

        string_candidates = [""]

        for output in output_list:
            if len(output) == k:
                string_candidates[-1] += output
            else:
                if len(output) < k:
                    string_candidates.append("")
                else:
                    string_candidates[-1] += output[:k]
                    string_candidates.append(output[-k:])

        # if len(output_list[-1]) == k:
        #     string_candidates[-1] += output_list[-1]
        # elif len(output_list[-1]) > k:
        #     string_candidates[-1] += output_list[-1][:k]

        max_length = 0

        for string_candidate in string_candidates:
            if len(string_candidate) > max_length:
                max_length = len(string_candidate)

        print(max_length)




