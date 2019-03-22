import matplotlib.pyplot as plt
import numpy as np

from kdTree.kdNode import kdNode

x = np.random.randint(low=0,high=100,size=(20,1))
y = np.random.randint(low=0,high=100,size=(20,1))
data = np.concatenate((x,y),axis=1)



class KDTree(object):
    def __init__(self,data = data,data_id=[]):
        self.data=data
        self.root = kdNode(data_id = data_id)

    def select_split_feature(self,node):
        data = self.data[node.data_id]

        vars = np.var(data,axis=0)
        max_var_feature_index = np.argmax(vars)
        node.split_feature = max_var_feature_index
        return max_var_feature_index

    def select_split_value(self,node):
        f_value = self.data[node.data_id][:,node.split_feature]
        f_value = np.sort(f_value).tolist()
        mid = f_value[len(f_value)//2]
        node.split_value = mid
        return mid

    def buid_tree(self,root):
        if root is None:
            return root

        if len(root.data_id)<=1:
            root.left = None
            root.right = None
        else:
            max_var_feature_index = self.select_split_feature(root)
            mid = self.select_split_value(root)
            left_temp = np.where(self.data[:,max_var_feature_index]<mid)[0].tolist()
            right_temp = np.where(self.data[:,max_var_feature_index]>=mid)[0].tolist()

            left_data_id = list(set(root.data_id).intersection(set(left_temp)))
            right_data_id = list(set(root.data_id).intersection(set(right_temp)))

            root.left = kdNode(data_id=left_data_id)
            root.right = kdNode(data_id=right_data_id)
            self.buid_tree(root.left)
            self.buid_tree(root.right)
        return root

    def plot_kdTree(self):
        data = self.data
        m,n = data.shape
        assert n==2,'Sorry, This program only can plot kdTree example of two features!'
        plt.scatter(data[:,0],data[:,1])
        nodes = [self.root]
        index =0
        while nodes:
            node = nodes.pop(0)
            node_data = self.data[node.data_id]

            xmax = node_data[:,0].max()
            xmin = node_data[:,0].min()

            ymax = node_data[:,1].max()
            ymin = node_data[:,1].min()

            feat = node.split_feature
            value = node.split_value
            if feat == 0:
                plt.vlines(x=value,ymin = ymin,ymax=ymax,colors='g',label=str(index))
            else:
                plt.hlines(y=value,xmin=xmin,xmax=xmax,colors='r',label=str(index))
            index +=1
            print(index,"-",feat,"-",value,'-',len(node.data_id))

            if node.left is not None:
                nodes.append(node.left)
            if node.right is not None:
                nodes.append(node.right)
        plt.show()

print(data)
data_id = np.array(range(data.shape[0])).tolist()
print(data_id)
kd = KDTree(data,data_id)
kd.buid_tree(kd.root)
print(kd.root.left.split_value)
kd.plot_kdTree()


#
# plt.hlines(y=5, xmin=data[:,0].min()-0.5, xmax=data[:,0].max()+0.5,colors='r')
# plt.xticks()
# plt.show()

