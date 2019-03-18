'''
回归树类
'''
from copy import copy

from sklearn.datasets import load_boston

from utils.model_selection import train_test_split, get_r2, model_evaluation


class Node(object):
    def __init__(self,score=None):
        '''
        初始化，存储预测值、左右节点、特征节点、特征和分裂点
        :param score:
        '''
        self.score = score
        self.left = None
        self.right = None
        self.feature = None
        self.split = None

class RegressionTree(object):

    # 初始化，存储根节点和树的高度
    def __init__(self):
        self.root = Node()
        self.level = 0

    # 计算分割点的MSE
    def _get_split_mse(self,X,y,idx,feature,split):
        '''

        :param X: 训练数据特征值
        :param y: 训练数据真实回归值
        :param idx: 分裂节点包含的数据索引index
        :param feature: 需要计算的分裂特征
        :param split: 特征分裂值
        :return: 分裂的节点的mse,split,左右子节点的回归均值
        '''
        split_sum = [0,0]        #记录分裂节点左右两边的回归值之和
        split_cnt = [0,0]        #记录分裂节点左右两边的样本数
        split_sqr_sum =[0,0]     #记录分裂节点左右两边的回归值平方和

        for i in idx:
            xi,yi = X[i][feature],y[i]
            if xi<split:
                split_cnt[0] +=1
                split_sum[0] +=yi
                split_sqr_sum[0] +=yi ** 2
            else:
                split_cnt[1] +=1
                split_sum[1] +=yi
                split_sqr_sum[1] +=yi ** 2

        split_avg = [split_sum[0] / split_cnt[0],split_sum[1]/split_cnt[1]]

        #对于左子树，mse:
        # split_sqr_sum[0] + split_avg[0]**2 * split_cnt[0] - 2*split_sum[0]*split_avg[0]
        split_mse = [split_sqr_sum[0]-split_sum[0]*split_avg[0], \
                     split_sqr_sum[1]-split_sum[1]*split_avg[1]]
        return sum(split_mse), split, split_avg


    # 计算最佳分割点：
    # 遍历特征某一列的所有的不重复的点，找出MSE最小的点作为最佳分割点。
    # 如果特征中没有不重复的元素则返回None。
    def _choose_split_point(self, X, y, idx, feature):
        unique = set([X[i][feature] for i in idx])
        if len(unique) == 1:
            return None

        # 当分裂点为最大值或最小值是，效果是一样的，就是都没有分类，所有去掉最小值，减少一次计算
        unique.remove(min(unique))
        mse, split, split_avg = min([self._get_split_mse(X,y,idx,feature,split) for split in unique],
                                    key=lambda x:x[0])
        return mse, feature, split, split_avg


    # 选择最佳分裂特征
    # 遍历所有特征，计算最佳分割点对应的MSE，找出MSE最小的特征、对应的分割点，左右子节点对应的均值和行号。
    # 如果所有的特征都没有不重复元素则返回None
    def _choose_feature(self,X,y,idx):
        # 特征数
        m = len(X[0])
        # 构造所有特征的最佳分裂点函数，并计算出所有特征的最佳分裂值
        split_rets = [x for x in map(lambda x:self._choose_split_point(X,y,idx,x),range(m))
                      if x is not None]

        if split_rets == []:
            return None

        mse, feature, split, split_avg = min(split_rets,key=lambda x:x[0])

        # 对idx样本集按照feature,split划分为左右子树
        idx_split = [[],[]]
        while idx:
            i = idx.pop()
            xi = X[i][feature]
            if xi < split:
                idx_split[0].append(i)
            else:
                idx_split[1].append(i)
        return feature,split,split_avg,idx_split

    # 规则转文字
    # 将规则用文字表达出来，方便我们查看规则。
    def _expr2literal(self,expr):
        feature, op, split = expr
        op = ">=" if op == 1 else '<'
        return "Feature %d %s %.4f"%(feature, op, split)

    # 获取规则
    # 将回归树的所有规则都用文字表达出来，方便我们了解树的全貌。
    # 这里用到了队列+广度优先搜索。有兴趣也可以试试递归或者深度优先搜索。
    def _get_rules(self):
        que = [[self.root,[]]] # 存储节点和节点对应的分裂规则
        self.rules = []

        while que:
            nd,exprs = que.pop(0)

            # 没有左子树也没有右子树
            if not(nd.left or nd.right):
                literals = list(map(self._expr2literal,exprs))
                self.rules.append([literals,nd.score])

            # 如果有左子树
            if nd.left:
                rule_left = copy(exprs)
                # op=-1:表示小于split
                rule_left.append([nd.feature,-1,nd.split])
                que.append([nd.left,rule_left])

            if nd.right:
                rule_right = copy(exprs)
                rule_right.append([nd.feature,1,nd.split])
                que.append([nd.right,rule_right])


    # 模型训练
    # 队列+广度优先搜索，节点的分裂过程中要注意：
    # 1. 树的深度小于<max_depth;
    # 2. 控制分裂的最少样本量min_samples_split;
    # 3. 叶子节点至少有两个不重复的y值；
    # 4. 至少有一个特征是没有重复值的
    def fit(self, X, y, max_depth=5, min_samples_splits=2):
        '''

        :param X:
        :param y:
        :param max_depth: 树的最大深度
        :param min_samples_splits: 分裂的最少样本量
        :return:
        '''

        # 初始化
        self.root.score = sum(y)/len(y)
        idx = list(range(len(y)))
        que = [(self.level+1,self.root,idx)]

        while que:
            level, nd, idx = que.pop(0)
            # 如果已经达到树的最大深度，则终止分裂
            if level > max_depth:
                level -= 1
                break

            # 如果节点的样本量已经小于最小分裂的最少样本量
            # 或者样本的纯度为100%,则跳过这个节点。
            if len(idx) < min_samples_splits or \
                all(map(lambda i:y[idx[0]]==y[i],idx)):
                continue

            # 如果所有特征的特征值都相同，跳过这个节点
            split_ret = self._choose_feature(X,y,idx)
            if split_ret is None:
                continue

            # 更新节点属性
            nd.feature,nd.split,split_avg,idx_split = split_ret
            nd.left = Node(split_avg[0])
            nd.right = Node(split_avg[1])

            que.append([level+1, nd.left, idx_split[0]])
            que.append([level+1, nd.right, idx_split[1]])

        self.level = level
        self._get_rules()

    # 打印规则
    def print_rules(self):
        for i, rule in enumerate(self.rules):
            literals, score = rule
            print("Rule %d:"%i,'|'.join(literals) + ' => split_hat %.4f' % score)

    # 预测一个样本
    def _predict(self,row):
        nd = self.root
        while nd.left and nd.right:
            if row[nd.feature] < nd.split:
                nd = nd.left
            else:
                nd = nd.right
        return nd.score

    # 预测多个样本
    def predict(self,X):
        return [self._predict(xi) for xi in X]


if __name__ == '__main__':
    print("Testing the accuracy of Regression Tree...")
    # 加载数据
    boston = load_boston()
    X = list(boston.data)
    y = list(boston.target)
    X_train, X_test, y_train, y_test = train_test_split(

        X, y, random_state=10)

    # Train model

    reg = RegressionTree()

    reg.fit(X=X_train, y=y_train, max_depth=4)

    # Show rules

    reg.print_rules()

    # Model accuracy

    get_r2(reg, X_test, y_test)
    #model_evaluation(reg, X_test, y_test)





























