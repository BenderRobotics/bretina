from difflib import *
import itertools
from pprint import pprint as print

txt1 = "Installer"
txt2 = "lnstaller"

seq = SequenceMatcher(lambda x: x is ' ', txt1, txt2)


print("ratio: {}".format(seq.ratio()))
print("qratio: {}".format(seq.quick_ratio()))

# get list of differences
df = ndiff(txt1, txt2)
df = filter(lambda x: x not in ['+  ', '-  '], df)

# get possible allowed substitution
allowed = list()
allowed += list(itertools.permutations(['i', 'l', 'I', '1'], 2))
allowed += list(itertools.permutations(['o', 'O', '0', 'Q'], 2))

print(allowed)

prev_char = None
prev_diff = None
output = []

for d in df:
    # '-': char only in A, '+' char only in B
    if d.startswith('-'):
        diff = -1
    elif d.startswith('+'):
        diff = +1
    else:
        diff = 0
    # take the char only
    char = d[-1]

    # if the difference means that there is different char in A and B (- in one, + in other)
    if (prev_char is not None) and (prev_diff + diff == 0) and (diff != 0):
        # if this combination is allowed, remove last from output buffer and change new one
        # to valid code (starts with space)
        if (char, prev_char) in allowed:
            output.pop()
            d = '  ' + char

    prev_char = char
    prev_diff = diff
    output.append(d)


dfff = filter(lambda x: not x.startswith(' '), output)
print(len(list(dfff)))
print(dfff)
