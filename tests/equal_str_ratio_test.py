import cv2
import sys
import os
import lorem
import random
import difflib
sys.path.insert(0, os.path.abspath('..'))
import bretina

"""
test for detection of the number of cols and its regions
"""


simchars=["vV", "cC", 'o0', 'oO', 'O0', 'o°']

n = 5



def diff_string(a, b):
    """
    Get string diff deltas.

    :param str a:
    :param str b:
    :return: 3 rows of human readable deltas in string
    :rtype list:
    """
    d = difflib.Differ()
    diff = d.compare(a, b)

    l1 = ""
    l2 = ""
    l3 = ""

    for d in diff:
        if d.startswith("-"):
            l1 += d[-1]
            l2 += " "
            l3 += "^"
        elif d.startswith("+"):
            l1 += " "
            l2 += d[-1]
            l3 += "^"
        elif d.startswith(" "):
            l1 += d[-1]
            l2 += d[-1]
            l3 += " "

    return "\n".join((l1, l2, l3))


for _ in range(n):
    s1 = lorem.sentence()
    s2 = f'{s1}'
    s1 = list(s1)
    s2 = list(s2)
    i = random.randint(0, len(s1)-2)
    k = random.randint(0, len(simchars)-2)

    s1[i] = simchars[k][0]
    s2[i] = simchars[k][1]


    s1[i+1] = simchars[k][0]
    s2[i+1] = simchars[k][1]

    i = random.randint(0, len(s1)-2)
    j = random.randint(0, len(s1)-2)
    s1[i] = s1[j]

    s1 = "".join(s1)
    s2 = "".join(s2)


    a, b, c = bretina.equal_str_ratio(s1, s2, simchars, ratio=0.91)


    print(diff_string(s1, s2))
    print(bretina.format_diff(c))
    print(f'................. {a}: {b}')
