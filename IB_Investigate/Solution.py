import numpy as np
import pandas as pd


"""
先删去 cover 股票 数量非常少的券商，设定一个值，少于这个值的可以删去
做一个简单的优化问题，研报数量越多越好
先选出每只股票，按照研报数量最多，最深入
"""


class get_solution():

    def __init__(self):
        self.file = pd.read_excel("调研库.xlsx")
        self.qs_cover_low = 5
        self.qs_num = 15
        # self.select_file()


    def select_file(self):
        """
        如果'券商cover股票数'小于 qs_cover_low 则认为此券商没有意义删掉
        """
        file = self.file
        file = file[file['券商cover股票数'] > self.qs_cover_low]
        self.file = file

    def get_num(self):
        """
        将数据按照 '股票简称' 分类，去每个股票中有最大研报数量的券商
        建立哈希表，将每个股票对应需要调研的券商以 dictionary 的方式储存起来
        """
        file = self.file
        qs_num = self.qs_num

        file_deep = file.groupby(['股票简称']).agg({'研报数量':'max'}).reset_index()
        file_deep = pd.merge(file_deep, file.loc[:,['券商', '股票简称', '股票代码', '研报数量', '券商cover股票数']])
        file_deep = file_deep.drop_duplicates(subset=['股票简称'])

        # 建立哈希表，将每个股票对应需要调研的券商以 dictionary 的方式储存起来
        # 每个股票对应的券商是拥有最多研报的券商，即调研最深入的券商
        qs_stock_dict = file_deep.set_index('股票简称')['券商'].to_dict()

        # 按照券商分配股票的数量将不同的券商进行排序
        qs_stat = file_deep.groupby('券商').agg({'股票简称':'count'}).reset_index()
        qs_stat.columns = ['券商', '券商分配股票数量']
        qs_stat = qs_stat.sort_values(['券商分配股票数量'], ascending=False).reset_index().drop(['index'], axis=1)
        qs_stat = pd.merge(qs_stat, file_deep.loc[:, ['券商','券商cover股票数']].drop_duplicates(), on='券商', how='left')

        # 将目标中的需要拜访的券商分成两类，需要拜访的 'qs_stat_more' 和 不需要拜访的 'qs_stat_less'
        # 选择集中拜访 N 个券商， N = qs_num
        qs_stat_less = qs_stat.iloc[qs_num:, :]
        qs_stat_more = qs_stat.iloc[:qs_num, :]
        # qs_u 即 '券商 in use'，需要拜访的券商
        qs_u = list(qs_stat_more['券商'])
        self.qs_u = qs_u


        # 对每个不需要拜访的券商进行循环
        # 将其替换为需要拜访的券商
        for i in range(len(qs_stat_less)):
            qs = qs_stat_less.iloc[i, 0]  # 不需要拜访的券商

            # stock_target 改券商对应的股票
            stock_target = file_deep[file_deep['券商'] == qs]['股票简称']
            for stock in stock_target:
                qs_candidate = file[file['股票简称'] == stock]['券商']

                # 如果有该股票研报的券商在列表中，则选择替换为需要拜访的券商，break
                # 如果有该股票研报的券商都在不需要拜访的券商的列表中，标记为0，之后删除
                for num, qs_c in enumerate(qs_candidate):
                    if qs_c in qs_u:
                        qs_stock_dict[stock] = qs_c
                        # print(stock, qs_c)
                        break
                    if (num == len(qs_candidate)-1) and (qs_c not in qs_u):
                        qs_stock_dict[stock] = 0

        file_new = pd.DataFrame({'股票简称':qs_stock_dict.keys(),  '券商':qs_stock_dict.values()})
        file_new = file_new[file_new['券商'] != 0]
        qs_stat_new = file_new.groupby('券商').agg({'股票简称':'count'}).reset_index()
        qs_stat_new.columns = ['券商', '券商分配股票数量']
        qs_stat_new = qs_stat_new.sort_values(['券商分配股票数量'], ascending=False)
        qs_stat_new.index = list(range(len(qs_stat_new)))
        # qs_stat_new = pd.merge(qs_stat_new, file_deep.loc[:, ['券商','券商cover股票数']].drop_duplicates(), on='券商', how='left')

        self.qs_stock_dict = qs_stock_dict
        self.qs_stat = qs_stat_new


gs = get_solution()
gs.get_num()

# 查看券商分配的统计结果
gs.qs_stat

# 将 dictionary 以 DataFrame 的形式储存起来
dict = gs.qs_stock_dict
file = pd.DataFrame({'股票简称': dict.keys(), '券商': dict.values()})
file = pd.merge(file, gs.file.loc[:, [ '券商', '股票简称', '研报数量']], how='left')
file = file[file['券商'] != 0]
file = file.loc[:, ['券商', '股票简称', '研报数量']].sort_values('券商')

# 将第一个结果 outcome_part1.xlsx 以 excel 的形式输出
file.to_excel("outcome_part1.xlsx", index=False)




def get_gini(ser):
    """
    计算一个 Series 的基尼系数
    """
    proportion = ser/sum(ser)
    return 1 - sum(proportion * proportion)

def get_entropy(ser):
    """
    计算一个 Series 的信息熵
    """
    proportion = ser / sum(ser)
    return - sum(proportion * np.log2(proportion))

def get_mark(qs_stock_dict, file_org,  m, n):
    """
    计算分配方案的权重
    分配的公式为: mark = m * 基尼系数 + n * 研报总数量
    :param qs_stock_dict: 用于储存分配方案的哈希表
    :param file_org: 最原始的DataFrame，用于更新方案
    :param m: 基尼系数之前的系数
    :param n: 研报总数量之前的系数
    :return: file:分配方案的 DataFrame 形式
             qs_stat:分配方案的统计
             mark 本次分配方案的得分
    """
    file = pd.DataFrame({'股票简称': qs_stock_dict.keys(), '券商': qs_stock_dict.values()})
    file = pd.merge(file, file_org.loc[:, [ '券商', '股票简称', '研报数量']], how='left')
    file = file[file['券商'] != 0]
    qs_stat = file.groupby('券商').agg({'股票简称': 'count', '研报数量':'sum'}).reset_index()
    qs_stat.columns = ['券商', '券商分配股票数量', '研报数量']
    qs_stat = qs_stat.sort_values(['券商分配股票数量'], ascending=False).reset_index().drop(['index'], axis=1)
    gini = get_gini(qs_stat['券商分配股票数量'])
    total_num = sum(qs_stat['研报数量'])
    mark = m * gini + n * total_num
    # mark 应该是约高约好
    return file, qs_stat, mark




# 获取变量
qs_u = gs.qs_u
file_2 = gs.file.copy()
dict_2 = gs.qs_stock_dict.copy()
qs_stat_2 = gs.qs_stat.copy()



current_mark = 0     # 用 0 初始化本次方案的分数
epsilon = 1          # 设置 threshold，如果两次方案的差小于 epsilon，则停止更新

# 从分配到的股票最多的券商开始循环
# 计算分配方法分数的时候使用 m = 1000, n = 0，即不考虑研报数量的影响，要求分配方案尽可能平均
for i in range(len(file_2)):

    # qs 本次循环中需要拜访的券商
    qs = qs_stat_2.iloc[i, 0]
    stock_target = file_2[file_2['券商'] == qs]['股票简称']

    # 记录本次调整目标的股票
    for stock in stock_target:

        qs_candidate = file_2[file_2['股票简称'] == stock]['券商']
        for qs_c in qs_candidate:
            file, qs_stat, mark_o = get_mark(dict_2, file_2, 1000, 0)

            # 新选用的券商应该在需要拜访的券商列表 qs_u 中
            if qs_c in qs_u:
                # dict_2[stock]
                test_dict = dict_2.copy()
                test_dict[stock] = qs_c
                file, qs_stat, mark_t = get_mark(test_dict, file_2, 1000, 0)

                # 如果新方案的 mark 大于 旧方案的 mark，则更新方案
                # 一旦更新则停止本次的循环
                if mark_t > mark_o:
                    dict_2 = test_dict
                    print(i, stock, "旧分数：", mark_o, "新分数：", mark_t)
                    break

    # 如果两次循环中的差小于 threshold 则停止循环
    print(qs_stat)
    if abs(current_mark - mark_o) < epsilon:
        break
    current_mark = mark_o


# 查看券商分配的统计结果
qs_stat


# 将 dictionary 以 DataFrame 的形式储存起来
dict = test_dict
file = pd.DataFrame({'股票简称': dict.keys(), '券商': dict.values()})
file = pd.merge(file, gs.file.loc[:, [ '券商', '股票简称', '研报数量']], how='left')
file = file[file['券商'] != 0]
file = file.loc[:, ['券商', '股票简称', '研报数量']].sort_values('券商')

# 将第二个结果 outcome_part2.xlsx 以 excel 的形式输出
file.to_excel("outcome_part2.xlsx", index=False)




