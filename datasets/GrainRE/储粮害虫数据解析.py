from bs4 import BeautifulSoup
import os
import re

# 展示害虫信息
def display(hcInfo):
    for name,value in hcInfo.items():
        print(name,':',value)

#re匹配字符串
def reMacth(string):
    PATTERN = r'[\u4e00-\u9fa5]+'
    ENPATTERN = r'[\u0041-\u005a|\u0061-\u007a]+'
    pattern = re.compile(PATTERN)
    enpattern = re.compile(ENPATTERN)
    p = pattern.findall((string))
    if len(p) == 0:
        p = ' '.join(enpattern.findall(string))
    else:
        p = ''.join(p)
    return p

#获取害虫的基本信息
def getBaseInfo(bs):
    hcBaseInfo = dict()
    try:
        basicInfo = bs.find('div', class_='basic-info J-basic-info cmn-clearfix')
        namelist = basicInfo.findAll(attrs={'class': 'basicInfo-item name'})
        valuelist = basicInfo.findAll(attrs={'class': 'basicInfo-item value'})
        for name,value in zip(namelist,valuelist):
            hcBaseInfo[reMacth(name.text)] = reMacth(value.text)
    except:
        pass
    return hcBaseInfo

#获取害虫特征信息
def getFeatureInfo(bs):
    features = dict()
    try:
        paraChildren = bs.find(attrs={'class': 'main-content'}).contents
        title = ''
        contents = ''
        for children in paraChildren:
            try:
                classType = children.attrs['class'][0]
                if classType == 'para-title':
                    title = children.h2.text
                    if title == '对比鉴定' or title == '':
                        continue
                if classType == 'para':
                    if title == '对比鉴定' or title == '':
                        continue
                    contents += children.text
                if classType == 'anchor-list':
                    if title == '对比鉴定' or title == '':
                        continue
                    features[title] = contents
                    contents = ''
            except:
                pass
    except:
        pass
    return features

#获取文件列表
def getFilesName():
    return os.listdir(r'.\baidubaikepro')

def save2txt(features:dict):
    with open(r'data1.txt','a+',encoding='utf-8') as f:
        for name, value in features.items():
            f.write(name + ':' + value + '\n')
        f.write('\n')


if __name__ == '__main__':
    filelist=getFilesName()
    for name in filelist:
        bs = BeautifulSoup(open(r"baidubaikepro\\"+name,encoding='utf-8'))
        hcInfo=getBaseInfo(bs)
        display(hcInfo)
        # save2txt(hcInfo)
        features = getFeatureInfo(bs)
        # save2txt(features)
        print(features)
        # print('---\n')

