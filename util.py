import re
import numpy as np

def extract_nums(line, dtype=int):
    return list(map(dtype, re.findall('\d+', line)))

