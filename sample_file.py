
import random
def getFileSize(path):
    res = 0
    with open(path , "r") as f:
        for line in f:
            res += 1
    return res;


def random_sel_one(path):
    length = 1
    line_selected = 0
    sizes = getFileSize(path)
    # print sizes
    while(length <= sizes):
        if random.randint(0 , sizes * 10) % length == 0:
            line_selected = length
        length += 1

    return line_selected


def random_sel_m(path , m):
    lines = []
    for item in xrange(m):
        lines.append(random_sel_one(path))

    #  lines
    print lines
    content = []
    with open(path , "r") as f:
        for idx , line in enumerate(f):
            if idx in lines:
                content.append(line)

    return content


if __name__ == "__main__":
    m = 10
    content = random_sel_m("test.dat" , m)
    for item in content:
        print item

