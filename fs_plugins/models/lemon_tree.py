import torch
import sys

class BinaryTreeNode:
    def __init__(self, order=None):
        self.order = order
        self.leftChild = None
        self.rightChild = None
        self.isMainChain = False
        self.leftArray = []
        self.rightArray = []

    def __str__(self, level=0):
        ret = "    "*level+repr(self.order)+"\n"
        if self.leftChild is not None:
            ret += self.leftChild.__str__(level+1)
        if self.rightChild is not None:
            ret += self.rightChild.__str__(level+1)

        return ret

    @staticmethod
    def subtree(layer_size):
        if layer_size == 0:
            return None
        root = BinaryTreeNode()
        stack = [root]
        for i in range(layer_size-1):
            stack_new = []
            for node in stack:
                left = BinaryTreeNode()
                right = BinaryTreeNode()
                stack_new.extend([left, right])
                node.leftChild = left
                node.rightChild = right
            stack = stack_new

        return root
    
    @staticmethod
    def build_tree(dim, left_tree_layer, src_upsample_scale):
        left_sub_tree_size = 2**left_tree_layer - 1
        root = BinaryTreeNode()
        root.leftChild = BinaryTreeNode()
        now = root
        now.isMainChain = True
        main_chain_size = torch.div(dim-2, (2**left_tree_layer), rounding_mode='trunc').long()
        # print(main_chain_size)
        # print(main_chain_size)
        # main_chain_size = (dim-2) // (src_upsample_scale + left_sub_tree_size*src_upsample_scale) * src_upsample_scale
        for i in range(main_chain_size):
            now.rightChild = BinaryTreeNode()
            now = now.rightChild
            now.isMainChain = True
            now.leftChild = BinaryTreeNode.subtree(left_tree_layer)
            # now = now.rightChild
        # now.rightChild = BinaryTreeNode()
        # now.rightChild.isMainChain = True
        return root

    @staticmethod
    def inorder(root, array):

        if root is None:
            return array

        array = BinaryTreeNode.inorder(root.leftChild, array)
        root.order = len(array)
        array.append(root)
        array = BinaryTreeNode.inorder(root.rightChild, array)
        
        if root.leftChild is not None:

            root.leftArray.append(root.leftChild.order)
            root.leftArray.extend(root.leftChild.leftArray), root.leftArray.extend(root.leftChild.rightArray)
        
        if root.rightChild is not None:
            root.rightArray.append(root.rightChild.order)
            
            root.rightArray.extend(root.rightChild.leftArray), root.rightArray.extend(root.rightChild.rightArray)
        # print(root.order, root.rightArray)
        return array


    @staticmethod
    def get_root(dim, left_tree_layer, src_upsample_scale):
        root = BinaryTreeNode.build_tree(dim, left_tree_layer, src_upsample_scale)
        BinaryTreeNode.inorder(root, [])
        return root

    @staticmethod
    def get_mask(dim, left_tree_layer, src_upsample_scale):
        sys.setrecursionlimit(2048+16)
        root = BinaryTreeNode.build_tree(dim, left_tree_layer, src_upsample_scale)
        BinaryTreeNode.inorder(root, [])
        _left_tree_mask = torch.ones([dim, dim])
        _right_tree_mask = torch.ones([dim, dim])
        _stop_mask = torch.zeros(dim)
        _main_chain = torch.zeros(dim)
        stack = [root]
        while len(stack) > 0:
            now = stack.pop()
            if now is None:
                continue
            _left_tree_mask[now.order, now.leftArray] = 0
            _right_tree_mask[now.order, now.rightArray] = 0
            stack.append(now.rightChild)
            stack.append(now.leftChild)
            if now.rightChild is None and now.leftChild is None:
                _stop_mask[now.order] = 1
            if now.isMainChain:
                _main_chain[now.order] = 1
                # if now.order+1<dim:
                #     _main_chain[now.order+1] = 1
        
        _left_tree_mask[1:,0] = 0
        # _right_tree_mask[1:,0] = 0

        return _left_tree_mask, _right_tree_mask, _main_chain


if __name__ == "__main__":

    src = 4
    left_tree_layer = 2
    src_upsample_scale = 2
    dim = src * int(src_upsample_scale* 2**left_tree_layer) + 2
    print(dim)
    print("dim: ", dim)
    root = BinaryTreeNode.get_root(dim,left_tree_layer,src_upsample_scale)
    print(root)

    
    _left_tree_mask, _right_tree_mask, _main_chain = BinaryTreeNode.get_mask(dim,left_tree_layer,src_upsample_scale)
    _right_tree_mask_main = ~(~_right_tree_mask.bool() & _main_chain.bool().unsqueeze(0))

    print("left")
    _left_tree_mask = _left_tree_mask.tolist()
    for i in range(dim):
        print(_left_tree_mask[i])
    print("right")
    _right_tree_mask = _right_tree_mask.tolist()
    for i in range(dim):
        print(_right_tree_mask[i])
    print("stop")
    print(_main_chain.tolist())
    print("right main")
    _right_tree_mask_main = _right_tree_mask_main.float().tolist()
    for i in range(dim):
        print(_right_tree_mask_main[i])

    max_left = 2**left_tree_layer-1
    _max_left_index = torch.arange(0, dim).unsqueeze(0).unsqueeze(-1)
    _max_left_index = _max_left_index + torch.zeros(1, dim, max_left+1, dtype=torch.int64)
    _max_left_index[:,:,0] = 0

    for i in range(1, max_left+1):
        _max_left_index[:,:,i] = torch.where((_max_left_index[:,:,i] - i) < 0, 0, _max_left_index[:,:,i] - i)
    print("max left index")
    print(_max_left_index.size())
    _max_left_index = _max_left_index.squeeze().tolist()

    for i in range(dim):
        print(_max_left_index[i])


