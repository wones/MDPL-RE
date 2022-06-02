import json
import os

#0.获取文件路径
def getFilePath(dirName,prefixFile):
    dirPath = os.path.abspath(dirName)
    filePath = []
    for suffix in ['dev','test','train']:
        filePath.append(os.path.join(dirPath,'.'.join([prefixFile,suffix])))
    return filePath

#1.获取文件Json格式全部内容
def getJsonContext(filePath):
    with open(filePath,encoding='utf-8') as f:
        allData = json.load(f)
    return allData

#2.生成新的json格式文件
def convertMrcdata2Muldata(msrJsonContext):
    mulData = []
    typeNums = 4
    allNums = len(msrJsonContext)
    start_position = []
    end_position = []
    span_position = []
    span_entity_label = []
    for curNums in range(allNums):
        context = msrJsonContext[curNums]['context']
        start_position += msrJsonContext[curNums]['start_position']
        end_position += msrJsonContext[curNums]['end_position']
        span_position += msrJsonContext[curNums]['span_position']
        span_entity_label += getItemLabel(msrJsonContext[curNums]['entity_label'],len(msrJsonContext[curNums]['span_position']))
        if (curNums + 1) % typeNums == 0:
            mulDataIter = {}
            mulDataIter['context'] = context
            mulDataIter['start_position'] = start_position
            mulDataIter['end_position'] = end_position
            mulDataIter['span_position'] = span_position
            mulDataIter['span_entity_label'] = span_entity_label
            mulData.append(mulDataIter)
            # print(context,start_position,end_position,span_position,span_entity_label)
            # print(mulDataIter)
            start_position = []
            end_position = []
            span_position = []
            span_entity_label = []
    return mulData

#3.获取每条样本的标签
def getItemLabel(valueLabel,labelNums):
    '''
    valueLabel包含：NS NR NT 分别代表：地名 人名 机构名  可由数字分别表示为（1,2,3） 0:其他
    '''
    labelMap = {'NS':1,'NR':2,'NT':3}
    labels = []
    for i in range(labelNums):
        labels.append(labelMap[valueLabel])
    return labels

#4.保存json格式数据文件
def saveJsonData(mulData,dirPath,fileName):
    dirPath = os.path.abspath(dirPath)
    if os.path.exists(dirPath):
        with open(os.path.join(dirPath,fileName),'w',encoding='utf-8') as f:
            json.dump(mulData,f,ensure_ascii=False)
    else:
        os.makedirs(dirPath)
        with open(os.path.join(dirPath,fileName),'w',encoding='utf-8') as f:
            json.dump(mulData,f,ensure_ascii=False)

if __name__=="__main__":
    filePaths = getFilePath('./msr_MSRA','mrc-ner')
    for filePath in filePaths:
        mrcData = getJsonContext(filePath)
        mulData = convertMrcdata2Muldata(mrcData)
        saveJsonData(mulData,'./mul_MSRA','mul-ner.'+filePath.split('.')[-1])
    print('MRC数据转换MUL数据完成！')