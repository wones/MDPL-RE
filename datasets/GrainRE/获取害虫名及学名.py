import re

#定义害虫名字与学名数据结构
class CName:
    def __init__(self,name,xname):
        self.name = name
        self.xname = xname

#匹配中文字符串
def reMacth(string):
    PATTERN = r'[\u4e00-\u9fa5]+'
    pattern = re.compile(PATTERN)
    p = pattern.findall((string))
    num = len(p)
    if num == 1:
        return p[0]
    else:
        if p[1] == '学名待定':
            return p[0]
        return p[1]

#匹配中文数字
def reMacthNum(string):
    PATTERN = r'[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+'
    pattern = re.compile(PATTERN)
    p = pattern.findall((string))
    if len(p) == 0:
        p = ''
    else:
        p = ''.join(p)
    return p

#匹配英文字符串
def reMacthEnglish(string):
    PATTERN = r'[a-zA-Z|Ö|Ü]+'
    pattern = re.compile(PATTERN)
    p = pattern.findall((string))
    if len(p) == 0:
        p = '学名待定'
    else:
        p = ' '.join(p)
    return p

#读取文件中的害虫名字及学名
def getAllNamePro():
    nameList = []
    with open('害虫名字及学名.txt','r',encoding='utf-8') as f:
        for name in f.readlines():
            chinesName = reMacth(name)
            # chinesNum = reMacthNum(chinesName)
            # index = len(chinesNum)
            englishName = reMacthEnglish(name)
            # if chinesName[-1] == '目' or chinesName[-1] =='科':
            #     chinesName = chinesName[index:]
            cname = CName(chinesName,englishName)
            nameList.append(cname)
        return nameList


if __name__ == '__main__':
    names = getAllNamePro()
    for name in names:
        print(name.name,name.xname)
    print(len(names))