"""
 Tiger最近被公司升任为营业部经理，他上任后接受公司交给的第一项任务便是统计并分析公司成立以来的营业情况。

    Tiger拿出了公司的账本，账本上记录了公司成立以来每天的营业额。分析营业情况是一项相当复杂的工作。由于节假日，大减价或者是其他情况的时候，营业额会出现一定的波动，当然一定的波动是能够接受的，但是在某些时候营业额突变得很高或是很低，这就证明公司此时的经营状况出现了问题。经济管理学上定义了一种最小波动值来衡量这种情况：

该天的最小波动值=min{|该天以前某一天的营业额-该天营业额|}

当最小波动值越大时，就说明营业情况越不稳定。

而分析整个公司的从成立到现在营业情况是否稳定，只需要把每一天的最小波动值加起来就可以了。你的任务就是编写一个程序帮助Tiger来计算这一个值。

第一天的最小波动值为第一天的营业额。
"""
n = int(input())

output = 0

series = []

for idx in range(n):
    a_i = int(input())
    series.append((a_i, idx))

series.sort()

for idx, values in enumerate(series):

    if values[1] == 0:
        output += abs(values[0])
        continue

    idx_upper = idx
    idx_lower = idx
    while True:
        idx_upper += 1
        if idx_upper >= len(series):
            upper_diff = 10000000
            break
        else:
            if series[idx_upper][1] < values[1]:
                upper_diff = abs(values[0]-series[idx_upper][0])
                break
    while True:
        idx_lower -= 1
        if idx_lower < 0:
            lower_diff = 10000000
            break
        else:
            if series[idx_lower][1] < values[1]:
                lower_diff = abs(values[0]-series[idx_lower][0])
                break
    output += min(upper_diff, lower_diff)

print(output)


