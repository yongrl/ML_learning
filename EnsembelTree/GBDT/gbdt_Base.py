from random import choices

from EnsembelTree.decisionTree.RegressionTree import RegressionTree
from utils.utils import run_time


class GradientBoostingBase(object):
    def __init__(self):
        '''
        gbdt基础类
        属性：
            trees{list}: 一维数组存储Regression Tree
            lr{float}: Learning rate
            init_val{float}: 初始化预测值
            fn{function}: 预测函数的包装函数
        '''
        self.trees = None
        self.lr = None
        self.init_val = None
        self.fn = lambda x: NotImplemented

    def _get_init_val(self,y):
        '''
        预测值的初始化函数
        :param y: list{1d} int or float
        :return:
            NotImplemented
        '''
        return NotImplemented

    def _match_node(self,row,tree):
        '''
        搜寻样本所属的叶子节点
        :param row: list{1d} int or float
        :param tree: regression tree
        :return:  regression tree.node
        '''

        nd = tree.root
        while nd.left and nd.right:
            if row[nd.feature]<nd.split:
                nd = nd.left
            else:
                nd = nd.right

        return nd

    def _get_leaves(self,tree):
        '''
        获取叶子节点
        :param tree: regression tree
        :return:
            list{1d} regression_tree.node object
        '''

        nodes = []
        que = [tree.root]
        while que:
            node = que.pop(0)
            if node.left is None or node.right is None:
                nodes.append(node)
                continue
            left_node = node.left
            right_node = node.right
            que.append(left_node)
            que.append(right_node)
        return nodes

    def _divide_regions(self,tree,nodes,X):
        '''
        根据数的节点分裂规则划分X到对应的叶子节点
        :param tree: regression tree
        :param nodes: list{1d} regression_tree.Node
        :param x: list{2d}
        :return:
            dict: {node1:[1,3,5],node2:[2,4,6]...}

        '''
        regions = {node:[] for node in nodes}
        for i,row in enumerate(X):
            node = self._match_node(row,tree)
            regions[node].append(i)

        return regions

    def _get_score(self,idx,y_hat,residuals):
        '''
        计算叶节点的值
        :param idx:
        :param y_hat:
        :param residuals:
        :return:
                NotImplemented
        '''
        print('*****************')
        return NotImplemented

    def _update_score(self,tree,X,y_hat,residuals):
        '''
        更新叶节点的预测值
        :param tree: regression tree
        :param X: list{2d}
        :param y_hat: list{1d}
        :param residuals(残差): list{1d}
        :return:
            None
        '''

        nodes = self._get_leaves(tree)

        regions = self._divide_regions(tree, nodes, X)
        for node, idxs in regions.items():
            node.score = self._get_score(idxs, y_hat, residuals)
        tree._get_rules()


    def _get_residuals(self,y,y_hat):
        '''
        计算残差
        :param y: list{1d}
        :param y_hat: list{1d}
        :return:
            residuals:list{1d}
        '''

        return [yi - self.fn(y_hat_i) for yi,y_hat_i in zip(y,y_hat)]

    def fit(self,X,y,n_estimators,lr,max_depth,min_samples_split,subsample = None):
        '''
        没有实现正则项
        :param X:
        :param y:
        :param n_estimators:
        :param lr:
        :param max_depth:
        :param min_samples_split:
        :param subsample: Subsample rate without replacement,float(0,1)
        :return:
        '''

        self.init_val = self._get_init_val(y)

        # 计算初始预测值y_hat
        n = len(y)
        y_hat = [self.init_val]*n

        # 计算初始残差
        residuals = self._get_residuals(y,y_hat)

        # 训练回归树
        self.trees = []
        self.lr = lr
        for _ in range(n_estimators):
            idx = range(n)
            #是否要采样
            if subsample is not None:
                k = int(subsample * n)
                idx = choices(population=idx,k=k)

            X_sub = [X[i] for i in idx]
            residuals_sub = [residuals[i] for i in idx]
            y_hat_sub = [y_hat[i] for i in idx]

            # 训练一棵回归树
            tree = RegressionTree()
            tree.fit(X_sub, residuals_sub, max_depth, min_samples_split)

            # 并不需要更新叶节点预测值
            #self._update_score(tree,X_sub,y_hat_sub,residuals_sub)

            # 更新预测值
            y_hat = [y_hat_i + lr*res_hat_i for y_hat_i,res_hat_i in zip(y_hat,tree.predict(X))]

            # 更新残差
            residuals = self._get_residuals(y,y_hat)
            self.trees.append(tree)

    def _predict(self,Xi):
        ret = self.init_val + sum(self.lr * tree._predict(Xi)
                                  for tree in self.trees)
        return self.fn(ret)


    def predict(self,X):
        return NotImplemented















