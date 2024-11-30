'''
謙謙最近迷上棒球，他想自己寫一個簡化的遊戲計分程式。這會讀入隊中每位球員的打擊結果，然後計算出球隊得分。

 這是個簡化版的模擬，假設擊球員打擊結果只有以下情況：

(1) 安打：以1B,2B,3B和HR 分別代表一壘打、二壘打、三壘打和全（四）壘打。

(2) 出局：以 FO,GO和 SO表示。



這個簡化版的規則如下：

(1) 球場上有四個壘包，稱為本壘、一壘、二壘、和三壘、。

(2) 站在本壘握著球棒打球的稱為「擊球員」，站在另外三個壘包的稱為「跑壘員」。

(3) 當擊球員的打擊結果為「安打」時，場上球員（擊球員與跑壘員）可以移動；結果為 「出局」時，跑壘員不動，擊球員離場換下一位擊球員。

(4) 球隊總共有九位球員，依序排列 。比賽開始由第1位開始打擊，當第 i 位球員打擊完畢後，由第 (i+1)位球員擔任擊球員。當第九位球員完畢後，則輪回第一位球員。

(5) 當打出 K 壘打時，場上球員（擊球員和跑壘員）會前進 K 個壘包。從本壘前進一個壘包會移動到一壘，接著是二壘、三壘，最後回到本壘。

(6) 每位球員回到本壘時可得 1分

(7) 每達到三個出局數時，一、二和三壘就會清空（ 跑壘員都得離開） ，重新開始。

請寫出具備這樣功能的程式，計算球隊總得分。

AC 59 ms 4.2MB
'''
import re
from collections import OrderedDict, deque

batting_dict = OrderedDict([(0, deque('1B 1B FO GO 1B'.split())), (1, deque('1B 2B FO FO SO'.split())),
                            (2, deque('SO HR SO 1B'.split())), (3, deque('FO FO FO HR'.split())),
                            (4, deque('1B 1B 1B 1B'.split())), (5, deque('GO GO 3B GO'.split())),
                            (6, deque('1B GO GO SO'.split())), (7, deque('SO GO 2B 2B'.split())),
                            (8, deque('3B GO GO FO'.split()))])

# batting_dict = OrderedDict()

# for count in range(9):
#     input_ = list(map(str, input().split()))
#     batting_dict[count] = deque(input_[1:])

b = 3 # int(input())

out_count = 0

base_occupation = deque()
score = 0

while out_count < b:
    for key in batting_dict.keys():
        result = batting_dict[key].popleft()
        if result in ['FO', 'GO', 'SO']:
            out_count += 1
            if out_count == b:
                break
            if out_count%3 == 0:
                base_occupation = deque()
        else:
            if result in ['1B', '2B', '3B']:
                distance = int(re.match("[\d]{1}", result).group(0))
            else:
                distance = 4

            for i in range(distance):
                if i == 0:
                    base_occupation.appendleft(1)
                else:
                    base_occupation.appendleft(0)

            while len(base_occupation) > 3:
                runner = base_occupation.pop()
                if runner == 1:
                    score += 1

print(score)




