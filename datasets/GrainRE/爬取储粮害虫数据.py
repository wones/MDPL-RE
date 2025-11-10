#导入相关包
import requests as rq
import time
from 获取害虫名及学名 import getAllNamePro

base_url='https://baike.baidu.com/item/'

headers={
    'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'Accept-Language':'zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7,be;q=0.6',
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36',

}

#通过名字获取百度百科HTML
def getHtml(key):
    url = base_url+key
    html = rq.request(method='get',url=url,headers=headers)
    return html

#将爬取的网页保存到本地
def saveHtml(html,key):
    with open('baidubaikepro/' + key + '.html','w',encoding = 'utf-8') as f:
        f.write(html.text)
        print('保存完成！')

#读取文件中的害虫名字
def getAllName():
    namelist = []
    with open('储粮害虫名字.txt','r',encoding='utf-8') as f:
        for name in f.readlines():
            namelist.append(name[:-1])
    return namelist


if __name__ == '__main__':
    # namelist=getAllName()
    namelist = getAllNamePro()
    for xname in namelist:
        html = getHtml(xname.name)
        saveHtml(html,xname.name)
        time.sleep(5)
    print('Done!')