import csv, json, time
from prettytable import PrettyTable as pt
from pypinyin import Style, lazy_pinyin


def saveFile(content, filename='cell_info.json'):
    """
    将内容保存至文件，同时保存一份副本
    :param filename: 保存的文件名
    :param content: 需要保存的内容
    :return:
    """
    global date
    with open(filename, 'w') as f:
        f.write(json.dumps(content))
    with open(f"{filename[:-5]}_{date.replace('-','')}.json", 'w') as f:
        f.write(json.dumps(content))


def flowRecord(flowInfo, filename='细胞冻存流水表.csv'):
    """
    将流水记录保存至流水表
    :param filename: 流水表文件名
    :param flowInfo: 需要保存的流水记录，多条记录以列表的形式保存在同一个列表中
    :return:
    """
    # title = ['细胞名称','数量','位置','操作','操作日期','操作人']
    tb = pt(['Cell Name', 'Num', 'Location', 'In/Out', 'Date', 'Operator'])
    for info in flowInfo:
        tb.add_row(info)
    try:
        with open(filename, 'a+', newline='') as f:
            # 保存流水
            csv_writer = csv.writer(f)
            csv_writer.writerows(flowInfo)
            print('>>>>本次流水信息如下：')
            print(tb)
    except Exception as e:
        input(e)


def cellIn(cellName, cellInfo, frozenDate, operator, date):
    with open('cell_info.json', 'r') as f:
        cellDic = json.load(f)

    # 生成otherInfo：冻存人，冻存日期日期
    nameInitial = ''.join(lazy_pinyin(operator, style=Style.FIRST_LETTER)).upper()
    otherInfo = f"{nameInitial}_{frozenDate}"

    flowInfo = []  # 流水记录们

    # 根据冻存数量和起始位置推算每个细胞冻存位置
    for item in cellInfo:
        num = int(item[0])  # 冻存数量
        tmp = item[1].split('-')  # 起始冻存位置[库提,层,孔位]
        try:
            assert len(tmp) == 3
        except:
            print('位置格式错误')
            return
        cellLoca = int(tmp[2])  # 起始孔位
        storage = '-'.join(tmp[0:2])  # storage=库提-层

        # 生成流水记录
        # title = ['细胞名称','数量','位置','操作','操作日期','操作人']
        flow = [cellName, num, item[1], '入库', date, operator]

        # 查询对应位置是否为空
        for i in range(num):
            cellLocation = storage + '-' + str(cellLoca + i)
            try:
                # 确认该位置是否存在细胞
                if cellDic[cellLocation][0] is not None:
                    print(cellLocation + '已存在细胞')
                    return

            except KeyError as key:
                # 确保保存位置的格式正确
                print(f'{key}位置输入有误，请检查输入信息')
                return

        # 逐个对细胞进行入库
        for i in range(num):
            cellLocation = storage + '-' + str(cellLoca)
            cellDic[cellLocation] = [cellName, otherInfo]
            print(f'细胞已入库{cellLocation} : [{cellName},{otherInfo}]')
            cellLoca += 1

        if num > 1:
            flow[2] = flow[2] + '~' + str(cellLoca - 1)
        flowInfo.append(flow)

    flowRecord(flowInfo)
    saveFile(cellDic)


def cellOut(locationList, operator, date):
    """
    根据位置对细胞进行出库操作
    :param locationList: 需要出库的位置列表[位置1，位置2，……]
    :param operator: 操作人，用于记录流水
    :param date: 操作日期，用于记录流水
    :return:
    """
    with open('cell_info.json', 'r') as f:
        cellDic = json.load(f)

    # 查询对应位置的细胞
    for loca in locationList:
        try:
            # 确认该位置是否存在细胞
            if cellDic[loca][0] is None:
                print(f'{loca}无细胞，请检查输入信息')
                return
            else:
                print(f'{loca}对应细胞为：[{cellDic[loca][0]},{cellDic[loca][1]}]')

        except KeyError as key:
            # 确保保存位置的格式正确
            print(f'{key}位置输入有误，请检查输入信息')
            return

    # 确认出库操作
    s = input(">>>请输入y确认出库操作，按任意键取消\n").lower()

    # 细胞出库
    if s == 'y':
        flowInfo = []
        for loca in locationList:
            # 生成流水记录信息
            # title = ['细胞名称','数量','位置','操作','操作日期','操作人']
            flowInfo.append([cellDic[loca][0], '1', loca, '出库', date, operator])
            # 细胞出库
            cellDic[loca] = [None, None]
        # 记录流水信息
        flowRecord(flowInfo)
        # 保存细胞库文件并保存一份副本
        saveFile(cellDic)
        print('出库完成')
    else:
        print("出库已取消")


def cellCheck(searchWord):
    with open('cell_info.json') as f:
        cellInfo = json.load(f)
    tb = pt(['Cell Location', 'Cell Name', 'Other Info'])

    cont = 0
    for key, value in cellInfo.items():
        if value[0] is None:
            value[0] = 'NONE'
        if searchWord in value[0].upper():
            tb.add_row([key, value[0], value[1]])
            cont += 1
    print(tb)
    print(f'>>>>共计{cont}个')


def cellChangeloca(oldLoca, newLoca, operator, date):
    with open('cell_info.json', 'r') as f:
        cellDic = json.load(f)

    tb = pt(['Cell Name', 'Old Location', 'New Location'])
    # 检查对应出库位置是否存在细胞
    for i in range(len(oldLoca)):
        try:
            if cellDic[oldLoca[i]][0] is None:
                # 确认该位置是否存在细胞
                print(f'{oldLoca[i]}无细胞，请检查输入信息')
                return
            elif cellDic[newLoca[i]][0]:
                # 确认该位置是否存在细胞
                print(f'转移后位置“{newLoca[i]}”存在细胞{cellDic[newLoca[i]][0]}，请检查输入信息')
                return
            else:
                tb.add_row([cellDic[oldLoca[i]][0], oldLoca[i], newLoca[i]])

        except KeyError as key:
            # 确保保存位置的格式正确
            print(f'{key}位置输入有误，请检查输入信息')
            return
    print(">>>>转移位置信息如下：")
    print(tb)

    # 确认是否执行操作
    s = input(">>>请输入y确认操作，按任意键取消\n").lower()
    if s == 'y':
        flowInfo = []
        # 细胞出库
        for i in range(len(oldLoca)):
            # 生成流水记录信息['细胞名称','数量','位置','操作','操作日期','操作人']
            flowInfo.append([cellDic[oldLoca[i]][0], '1', oldLoca[i], '出库', date, operator])
            flowInfo.append([cellDic[oldLoca[i]][0], '1', newLoca[i], '入库', date, operator])

            # 细胞出入库
            cellDic[newLoca[i]] = cellDic[oldLoca[i]]
            cellDic[oldLoca[i]] = [None, None]

        # 记录流水信息
        flowRecord(flowInfo)
        # 保存细胞库文件并保存一份副本
        saveFile(cellDic)
        print('记录完成')
    else:
        print("出库已取消")


if __name__ == '__main__':
    while True:

        inOut = input('细胞入库：in\n细胞出库：out\n信息查询：check\n位置转移：change\n输入“q”退出\n').lower()
        date = time.strftime('%Y-%m-%d', time.localtime(time.time()))

        if inOut == 'q':
            break

        elif inOut == 'in':
            cellName = input('>>>>请输入细胞名称（子项目_转染批次_代次_筛选条件）:\n')
            frozenDate = input('>>>>请输入该细胞的冻存日期：\n')
            cellInfo = input('>>>>请输入冻存细胞数量与起始位置(e.g.\n2 C3-1-3: )，空格隔开，多个不连续位置使用英文“,”分隔：\n').split(',')
            # 将冻存数量和位置保存为列表[[数量1，起始位置1],[位置2,起始位置2],……]
            for i in range(len(cellInfo)):
                cellInfo[i] = cellInfo[i].strip().upper()
                cellInfo[i] = cellInfo[i].split(' ')
            operator = input(">>>>请输入操作人姓名：")

            cellIn(cellName, cellInfo, frozenDate, operator, date)

        elif inOut == 'out':
            locationList = input(">>>>请输入出库细胞位置，多个位置使用‘|’分割：\n").strip('|').split('|')
            operator = input(">>>>请输入操作人姓名：")

            cellOut(locationList, operator, date)

        elif inOut == 'check':
            while True:
                search = input('>>>>请输入细胞名称,输入q退出：').upper()
                if search == 'Q':
                    break
                else:
                    cellCheck(search)

        elif inOut == 'change':
            oldLoca = input(">>>>请输入需要转移的细胞的原始位置，多个位置使用‘|’分割：\n").strip('|').split('|')
            newLoca = input(">>>>请输入细胞转移后的位置，多个位置使用‘|’分割：\n").strip('|').split('|')
            if len(oldLoca) == len(newLoca):
                operator = input(">>>>请输入操作人姓名：")

                cellChangeloca(oldLoca, newLoca, operator, date)

            else:
                print("新旧位置数量不同，请核对输入信息。")

        else:
            print(">>>>输入有误，请重新输入")