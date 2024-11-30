"""
最近要開學了!  ( ~~~ 跟題目沒有什麼關係 ) ><



請寫一個程式把所有合法括號匹配方式列出來!



Ex. (())  ,  ((()())) , ()((()))  是合法的匹配方式



      )( , (()))(  , ()(()(  是不合法的匹配方式



     合法匹配的括號 ， 從答案列的開頭到答案列某一點，左括弧次數永遠大於等於右括弧!



    Ex. 合法匹配   ((()()))

    字串 (        左括弧 : 1  >=   右括弧 : 0

    字串 ((        左括弧 : 2  >=   右括弧 : 0

    字串 (((        左括弧 : 3  >=   右括弧 : 0

    字串 ((()        左括弧 : 3  >=   右括弧 : 1

    字串 ((()(        左括弧 : 4  >=   右括弧 : 1

    字串 ((()()        左括弧 : 4  >=   右括弧 : 2

    字串 ((()())        左括弧 : 4  >=   右括弧 : 3

    字串 ((()()))        左括弧 : 4  >=   右括弧 : 4



    Ex. 不合法匹配    (()))(

   字串 (        左括弧 : 1  >=   右括弧 : 0

   字串 ((        左括弧 : 2  >=   右括弧 : 0

   字串 (()        左括弧 : 2  >=   右括弧 : 1

   字串 (())        左括弧 : 2  >=   右括弧 : 2

   字串 (()))        左括弧 : 2  <=   右括弧 : 3

!!! 右括弧次數大於左括弧了!  (()))( 為不合法匹配

"""
from copy import copy

def dynamic_fn(output, parenthesis_dict):

    if parenthesis_dict["("] >  parenthesis_dict[")"]:
        return

    if sum(parenthesis_dict.values()) == 0:
        print(output)

    for str_, number in parenthesis_dict.items():
        if number > 0:
            copy_of_parenthesis_dict = copy(parenthesis_dict)
            copy_of_parenthesis_dict[str_] -= 1
            dynamic_fn(output+str_, copy_of_parenthesis_dict)

while True:
    try:
        N = int(input())
    except EOFError:
        break

    my_dict = {"(": N,
               ")": N}

    dynamic_fn(output="", parenthesis_dict=my_dict)

    print(" ")