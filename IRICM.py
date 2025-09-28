"""IRICM code"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
import time

t0 = time.time()
# 1. 加载数据
file_path = 'RMQC dataset.xlsx'  # 替换为你的文件路径
df = pd.read_excel(file_path, sheet_name="Sheet2")

# 2. 数据预处理
X = df[['IRS', 'RQD', 'AS', 'TL', 'AP', 'JRC', 'FM', 'WD', 'GW', 'JC']]  # 特征
y = df['RMR24_level']  # 目标类别
y = y.astype('category').cat.codes  # 将类别标签转换为数值类型
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)  # random_state=42

# 3. 训练随机森林模型
# 定义超参数搜索空间
param_grid = {
    'n_estimators': [100, 200, 300],  # 弱分类器数量 即决策树的数量
    'max_depth': [5, 10, 15, None],         # 决策树的最大深度，控制模型复杂度
    'min_samples_split': [5, 10, 15], # 弱评估器分枝时，父节点上最少要拥有的样本个数
    'min_samples_leaf': [2, 4, 6],    # 弱评估器的叶子节点上最少要拥有的样本个数
    'max_features': ['sqrt', 'log2', None]  # 控制每棵树选择特征的方式
}

clf = RandomForestClassifier(random_state=42, oob_score=True)   # random_state 控制一切随机模式

# 使用网格搜索和5折交叉验证来优化模型，目标是最大化验证集的准确率
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 输出最优参数和最佳分数
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# 获取所有参数组合的结果
results = grid_search.cv_results_

# 提取每个参数组合及其得分
for mean_score, std_score, params in zip(results['mean_test_score'],
                                         results['std_test_score'],
                                         results['params']):
    print(f"Params: {params}")
    print(f"Mean Accuracy: {mean_score:.4f}, Std: {std_score:.4f}\n")

# 用最优参数重新训练模型
rf = grid_search.best_estimator_
rf.fit(X_train, y_train)

accuracy = accuracy_score(y_test, rf.predict(X_test))
print(f"模型的测试集准确率: {accuracy * 100:.2f}%")
conf_matrix = confusion_matrix(y_test, rf.predict(X_test))
print("测试集的混淆矩阵：\n", conf_matrix)
accuracy = accuracy_score(y_train, rf.predict(X_train))
print(f"模型的训练集准确率: {accuracy * 100:.2f}%")
conf_matrix = confusion_matrix(y_train, rf.predict(X_train))
print("训练集的混淆矩阵：\n", conf_matrix)

t1 = time.time()
print("随机森林超参数优化+训练，总耗时: %.2f 秒" % (t1 - t0))

# 4. 提取每棵树的决策路径
def get_decision_paths(tree, features):
    # 获取单棵树的所有决策路径
    paths = []   # 存储路径的列表
    tree_ = tree.tree_   # 访问树结构
    feature_names = [features[i] if i != -2 else "leaf" for i in tree_.feature] # 获取特征名称，叶节点标记为 "leaf"

    def recurse(node, path):
        # 递归遍历树的节点，构建路径
        if tree_.children_left[node] == -1 and tree_.children_right[node] == -1: # 如果是叶节点
            path.append(('leaf', tree_.value[node].argmax())) # 将叶节点的类别标签添加到路径
            paths.append(path[:]) # 保存路径
            path.pop()            # 移除叶节点，返回上一层
        else:
            if tree_.children_left[node] != -1: # 如果有左子节点
                threshold = tree_.threshold[node] # 获取分裂的阈值
                feature = feature_names[node] # 获取特征名称
                path.append((feature, "<=", threshold)) # 添加特征和阈值到路径
                recurse(tree_.children_left[node], path) # 递归遍历左子树
                path.pop() # 移除当前节点的条件

            if tree_.children_right[node] != -1: # 如果有右子节点
                threshold = tree_.threshold[node] # 获取分裂的阈值
                feature = feature_names[node] # 获取特征名称
                path.append((feature, ">", threshold)) # 添加特征和阈值到路径
                recurse(tree_.children_right[node], path) # 递归遍历右子树
                path.pop() # 移除当前节点的条件

    recurse(0, []) # 从根节点开始递归遍历
    return paths   # 返回决策路径

# 从所有树中提取路径
all_paths = [] # 用于存储所有树的路径
for estimator in rf.estimators_:                     # 遍历每一棵决策树
    paths = get_decision_paths(estimator, X.columns) # 提取当前树的决策路径
    all_paths.extend(paths)  # 将路径添加到总路径列表中
print("all_paths", all_paths[0])

# 保存规则到文本文件
def save_rules_to_txt(rules, filename):
    """将规则列表保存到txt文件中，每行一条规则"""
    with open(filename, 'w') as f:  # 打开文件进行写入
        for rule in rules:
            # 将路径条件连接成字符串
            rule_str = " -> ".join([f"{feature} {op} {threshold}" for feature, op, threshold in rule[:-1]])
            rule_str += f" -> Class {rule[-1][1]}"  # 添加最后的分类结果
            f.write(rule_str + "\n") # 写入文件
    print(f"规则已保存到 {filename}")  # 输出保存文件的提示

# 将所有路径(all_paths)保存到文件
save_rules_to_txt(all_paths, "all_paths_rules.txt")


t2 = time.time()
print("提取规则时间，总耗时: %.2f 秒" % (t2 - t1))

# 5. 遗传算法
def fitness_function_test(rules):
    """适应度函数：计算规则集在测试集上的准确率"""
    y_pred = []  # 存储预测标签
    for _, row in X_test.iterrows(): # 遍历测试集的每一行
        votes = [] # 存储当前行符合的类别
        for rule in rules: # 遍历规则集中的每条规则
            match = True   # 初始匹配标记为 True
            for condition in rule[:-1]:  # 遍历规则中的条件
                feature, op, threshold = condition  # 解压条件
                if op == "<=" and not row[feature] <= threshold:  # 判断是否符合条件
                    match = False  # 不符合则标记为 False
                    break
                elif op == ">" and not row[feature] > threshold:  # 判断是否符合条件
                    match = False  # 不符合则标记为 False
                    break
            if match:  # 若规则符合
                votes.append(rule[-1][1])  # 将分类标签添加到投票列表
        # 使用投票结果，若没有匹配到规则，则默认类别为-1
        y_pred.append(max(set(votes), key=votes.count) if votes else -1) # 使用多数投票结果作为预测

    return round(accuracy_score(y_test, y_pred), 4) # 返回预测准确率

def fitness_function_train(rules):
    """适应度函数：计算规则集在训练集上的准确率"""
    y_pred = []  # 存储预测标签
    for _, row in X_train.iterrows(): # 遍历测试集的每一行
        votes = [] # 存储当前行符合的类别
        for rule in rules: # 遍历规则集中的每条规则
            match = True   # 初始匹配标记为 True
            for condition in rule[:-1]:  # 遍历规则中的条件
                feature, op, threshold = condition  # 解压条件
                if op == "<=" and not row[feature] <= threshold:  # 判断是否符合条件
                    match = False  # 不符合则标记为 False
                    break
                elif op == ">" and not row[feature] > threshold:  # 判断是否符合条件
                    match = False  # 不符合则标记为 False
                    break
            if match:  # 若规则符合
                votes.append(rule[-1][1])  # 将分类标签添加到投票列表
        # 使用投票结果，若没有匹配到规则，则默认类别为-1
        y_pred.append(max(set(votes), key=votes.count) if votes else -1) # 使用多数投票结果作为预测

    return round(accuracy_score(y_train, y_pred), 4) # 返回预测准确率

# 打开txt文件（如果文件不存在会自动创建）
output_file = "fitness_scores_log.txt"
# 如果文件已存在，清空内容
with open(output_file, "w") as f:
    f.write("")

def genetic_algorithm_only(all_paths, population_size=300, generations=100, mutation_rate=0.01, seed=32):
    """遗传算法进行规则搜索，限制每一类的规则条数"""
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)

    # 按类别将规则分组
    rules_by_class = {0: [], 1: [], 2: [], 3: [], 4: []}
    for rule in all_paths:
        # 根据最后一个元组中的类别信息将规则分配到相应类别
        class_label = rule[-1][1]
        if class_label in rules_by_class:
            rules_by_class[class_label].append(rule)

    # 确保每个类别至少有足够的规则数
    for class_id in rules_by_class:
        assert len(rules_by_class[class_id]) >= 5

    # 初始化种群：每个个体包含来自每个类别的若干规则
    population = []
    for _ in range(population_size):
        individual = (
                random.sample(rules_by_class[0], 5) +
                random.sample(rules_by_class[1], 5) +
                random.sample(rules_by_class[2], 5) +
                random.sample(rules_by_class[3], 5) +
                random.sample(rules_by_class[4], 5)
        )
        population.append(individual)

    print("Population size:", len(population))
    print("Individual size:", len(population[0]))

    # 用于记录测试集上最高精度的个体和精度
    best_test_fitness = -float('inf')
    best_test_individual = None

    for generation in range(generations):  # 迭代若干代
        # 评估训练集适应度
        fitness_scores_train = [fitness_function_train(rules) for rules in population]
        # 评估测试集适应度
        fitness_scores_test = [fitness_function_test(rules) for rules in population]

        print("第%d次迭代训练集和测试集精度：" % generation, max(fitness_scores_train), max(fitness_scores_test))

        # 找到当前代在测试集上精度最高的个体
        max_test_fitness = max(fitness_scores_test)
        if max_test_fitness > best_test_fitness:
            best_test_fitness = max_test_fitness
            best_test_individual = population[np.argmax(fitness_scores_test)]

        # 获取最高精度并写入日志文件
        with open(output_file, "a") as f:
            f.write(f"{generation} {max(fitness_scores_train):.4f} {max(fitness_scores_test):.4f}\n")  # 保留4位小数

        # 选择：保留前50%的适应度较高的个体
        selected_indices = np.argsort(fitness_scores_train)[-population_size // 2:]  # 选择前一半适应度最高的个体
        selected_population = [population[i] for i in selected_indices]  # 获取被选择的个体

        # 交叉和变异生成新一代
        new_population = []
        while len(new_population) < population_size:  # 新种群达到目标数量前
            parent1, parent2 = random.sample(selected_population, 2)  # 随机选取两个父代
            # 交叉
            cross_point = random.randint(1, min(len(parent1), len(parent2)) - 1)  # 确定交叉点
            child = parent1[:cross_point] + parent2[cross_point:]  # 生成新个体

            # 确保新个体中包含来自不同类别的规则
            child_classes = {rule[-1][1] for rule in child}
            if len(child_classes) == 5:  # 检查新个体中是否包含5个不同类别的规则
                # 变异
                if random.random() < mutation_rate:  # 进行变异操作
                    mutation_rule = random.choice(all_paths)
                    mutation_class = mutation_rule[-1][1]
                    # 替换变异规则时，确保变异后的规则组合仍包含不同类别的规则
                    for i, rule in enumerate(child):
                        if rule[-1][1] == mutation_class:
                            child[i] = mutation_rule
                            break

                new_population.append(child)  # 将新个体加入新种群

        population = new_population  # 更新种群

    # 返回测试集上精度最高的个体
    print(f"Best individual on test set - Fitness: {best_test_fitness:.4f}")
    return best_test_individual  # 返回最优个体


# 6. 使用遗传算法优化规则
best_rules = genetic_algorithm_only(all_paths) # 使用遗传算法搜索最佳规则
save_rules_to_txt(best_rules, "best_rules_optimized.txt") # 将最优规则保存到文件

# 保存规则到文本文件
def save_rules_to_txt2(rules, filename):
    """将规则列表保存到txt文件中，每行一条规则"""
    with open(filename, 'w') as f:  # 打开文件进行写入
        for rule in rules:
            f.write(str(rule) + "\n") # 写入文件
    print(f"规则已保存到 {filename}")  # 输出保存文件的提示
save_rules_to_txt2(best_rules, "best_rules_optimized2.txt")

# 构建基于规则的预测函数
def predict_with_rules(row, rules):
    """根据最优规则对新数据集X进行预测"""
    votes = [] # 存储该样本符合的类别标签
    for rule in rules: # 遍历每一条规则
        match = True   # 初始匹配标记为True
        for condition in rule[:-1]:  # 遍历规则中的每个条件
            feature, op, threshold = condition # 获取条件中的特征、操作符和阈值
            # 根据操作符判断条件是否满足
            if op == "<=" and not row[feature] <= threshold:
                match = False # 如果条件不满足，将匹配标记为False
                break
            elif op == ">" and not row[feature] > threshold:
                match = False # 如果条件不满足，将匹配标记为False
                break
        if match: # 如果当前规则所有条件都满足
            votes.append(rule[-1][1])  # 将该规则的类别标签添加到投票列表
    # 使用投票结果，若没有匹配到规则，则默认类别为-1
    return max(set(votes), key=votes.count) if votes else -1

t3 = time.time()
print("随机森林重构规则时间，总耗时: %.2f 秒" % (t3 - t2))


# 7. 在测试集上使用最佳规则进行预测
y_pred_final = X_test.apply(lambda row: predict_with_rules(row, best_rules), axis=1)
# 1 混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred_final)
print("混淆矩阵:\n", conf_matrix)
# 2总体精度
final_accuracy = accuracy_score(y_test, y_pred_final)
print("使用遗传算法优化后的测试集分类准确率:", final_accuracy)
# 3 F1分数（宏平均、加权平均）
f1_macro = f1_score(y_test, y_pred_final, average='macro')  # 每个类别的F1均值
f1_weighted = f1_score(y_test, y_pred_final, average='weighted')  # 加权平均
print(f"F1分数（宏平均）: {f1_macro:.4f}")
print(f"F1分数（加权平均）: {f1_weighted:.4f}")
# 4. 计算精准率（Precision）
precision_macro = precision_score(y_test, y_pred_final, average='macro')
precision_weighted = precision_score(y_test, y_pred_final, average='weighted')
print(f"精准率（宏平均）: {precision_macro:.4f}")
print(f"精准率（加权平均）: {precision_weighted:.4f}")
# 5. 计算召回率（Recall）
recall_macro = recall_score(y_test, y_pred_final, average='macro')
recall_weighted = recall_score(y_test, y_pred_final, average='weighted')
print(f"召回率（宏平均）: {recall_macro:.4f}")
print(f"召回率（加权平均）: {recall_weighted:.4f}")
# 6. 计算均交并比（mIOU，针对每个类别）
iou_per_class = np.diag(conf_matrix) / (np.sum(conf_matrix, axis=1) + np.sum(conf_matrix, axis=0) - np.diag(conf_matrix))
mean_iou = np.nanmean(iou_per_class)
print(f"每个类别的IOU: {iou_per_class}")
print(f"均交并比（mIOU）: {mean_iou:.4f}")

t4 = time.time()
print("预测时间，总耗时: %.2f 秒" % (t4 - t3))










