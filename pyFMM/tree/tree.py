
#------------- needed to allow FMMTree type hints inside TreeNode -------------
from __future__ import annotations
#-----------------------------------------------------------------------------

import numpy as np
from typing import List, Optional


class TreeNode:
    """Lightweight octree/quadtree node for FMM. Holds indices into global point arrays."""
    def __init__(self, tree: FMMTree, center: np.ndarray, size: np.ndarray, indices: np.ndarray, level: int = 0):
        self.center = center            # np.array([x,y,z])
        self.size = size                # np.array([sx,sy,sz])
        self.half_width = np.max(size) / 2.0    # float
        self.indices = indices          # np.array of source/target indices inside node
        self.num_points = len(indices)  # 
        self.level = level
        self.children: List[Optional[TreeNode]] = [None]*8
        self.is_leaf = True
        self.tree = tree

        # storage for multipole/local expansions (complex arrays)
        self.M = None   # multipole coefficients
        self.L = None   # local coefficients

class FMMTree:
    """Tree driver to build octree, compute multipoles (P2M/M2M), translate (M2L), and down-pass (L2L)."""
    def __init__(self,
                 center: np.ndarray,
                 size : np.ndarray,
                 points: np.ndarray,
                 charges: np.ndarray,
                 p: int = 4,
                 max_leaf_size: int = 64,
                 max_level: int = 5):
        """
        points: (N,3) source/target coordinates
        charges: (N,) source strengths (if shared source/target)
        """
        self.points = np.asarray(points)
        self.charges = np.asarray(charges)
        self.center = np.asarray(center)
        self.size = np.asarray(size)
        self.max_leaf_size = int(max_leaf_size)
        self.max_level = int(max_level)
        self.root: Optional[TreeNode] = None
        self.node_list : List[TreeNode] = []
        self.p = p  # multipole order


    def build_tree(self):
        root_node = TreeNode(
            tree=self,
            center=self.center,
            size=self.size,
            indices=np.arange(self.points.shape[0]), 
            level=0
        )
        self.root = root_node
        self.make_children_recursively(root_node)


    def make_children_recursively(self, node: TreeNode):
        self.node_list.append(node)
        #-------------- return once max level is reached --------------
        if node.level >= self.max_level:
            return
        #--------------------------------------------------------------
        self.leaf = False

        for i in range(8):
            child_center = node.center + 0.5 * node.half_width * (2 * (i & 1) - 1) * np.array([1, 0, 0]) \
                                       + 0.5 * node.half_width * (2 * ((i >> 1) & 1) - 1) * np.array([0, 1, 0]) \
                                       + 0.5 * node.half_width * (2 * ((i >> 2) & 1) - 1) * np.array([0, 0, 1])

            in_child_mask = np.all( np.abs(self.points[node.indices] - child_center) <= node.half_width / 2.0, axis=1)  
            child_indices = node.indices[in_child_mask]      


            child_node = TreeNode(
                tree=self,
                center=child_center,
                size=node.size / 2.0,
                indices=child_indices,
                level=node.level + 1
            )
            node.children[i] = child_node


        #--------------- free memory ---------------
        # removes indices from non-leaf nodes to save memory
        # NOTE - does not reset num_points as it counts all points in subtree
        node.indices = []  
        #-----------------------------------------------

        
        for child in node.children:
            self.make_children_recursively(child)




