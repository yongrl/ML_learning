class kdNode(object):
    def __init__(self,data=None,data_id=[],split_feature=None,split_value=None,
                 left = None,right=None):
        """

        :param data: all data
        :param data_id: data id in this kdnode
        :param split_feature:
        :param split_value:
        :param left:
        :param right:
        """
        self.data_id = data_id
        self.split_feature = split_feature
        self.split_value = split_value
        self.left = left
        self.right = right
