
#------------- needed to allow FMMTree type hints inside TreeNode -------------
from __future__ import annotations
from collections import deque
#-----------------------------------------------------------------------------


#--------------- import utils and moment modules -----------------------------
try:
    # import the utils module so we don't pollute this module's namespace
    from .. import utils as utils
except Exception:
    import utils as utils  # type: ignore
try:
    # import the utils module so we don't pollute this module's namespace
    from .. import moment as moment
except Exception:
    import moment as moment  # type: ignore
#-----------------------------------------------------------------------------

import numpy as np
from typing import List, Optional, Iterable


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
        self.parent: Optional[TreeNode] = None
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


    def build_tree(self, BFS: bool = True):
        """
        Builds the octree structure recursively.
        1. Create root node containing all points.
        2. Recursively subdivide nodes until max level or leaf size reached.
        3. Store nodes in self.node_list in depth-first order.
        """
        #--------------- first create root node ---------------
        root_node = TreeNode(
            tree=self,
            center=self.center,
            size=self.size,
            indices=np.arange(self.points.shape[0]), 
            level=0
        )
        #------------------------------------------------------
        #--------------- set root node ------------------------
        self.root = root_node
        #-------------------------------------------------------
        #--------------- recursively build tree ----------------
        if BFS:
            self.make_children_BFS(root_node)
        else:
            self.make_children_DFS(root_node)
        #-------------------------------------------------------


    def make_child(self, node: TreeNode, child_center: np.ndarray) -> TreeNode:
        """
        Creates a single child node
        """
        # Determine which points belong to this child
        in_child_mask = np.all(
            np.abs(self.points[node.indices] - child_center) <= node.half_width / 2.0,
            axis=1
        )
        child_indices = node.indices[in_child_mask]
        # Create the child node
        child_node = TreeNode(
            tree=self,
            center=child_center,
            size=node.size / 2.0,
            indices=child_indices,
            level=node.level + 1
        )
        child_node.parent = node
        return child_node

    def make_children_BFS(self, node: TreeNode):
        """
        Creates the full octree structure (replaces make_children_recursively),
        but builds level-by-level instead of depth-first recursion.
        """
        #self.node_list = []  # we now fill this BFS-ordered

        # Queue for BFS
        q = deque([node])

        while q:
            node = q.popleft()

            # Add to global list
            self.node_list.append(node)

            # Stop if max level reached
            if node.level >= self.max_level:
                node.is_leaf = True
                continue

            # Split current node
            node.is_leaf = False

            # Create children nodes
            for i in range(8):
                # Compute child center (your original logic preserved)
                child_center = (
                    node.center
                    + 0.5 * node.half_width * (2 * (i & 1) - 1) * np.array([1, 0, 0])
                    + 0.5 * node.half_width * (2 * ((i >> 1) & 1) - 1) * np.array([0, 1, 0])
                    + 0.5 * node.half_width * (2 * ((i >> 2) & 1) - 1) * np.array([0, 0, 1])
                )
                child_node = self.make_child(node, child_center)
                node.children[i] = child_node

                # Add to queue for future splitting
                q.append(child_node)

            # Free memory on internal node
            node.indices = []  # same as your original

    def make_children_DFS(self, node: TreeNode):
        """
        Recursively creates child nodes for the octree.
        """
        #-------------- append node to global list --------------
        #TODO - decide if we want a different ordering here
        #       curently it is depth-first, but maybe we want breadth-first?
        self.node_list.append(node)
        #--------------------------------------------------------
        #-------------- return once max level is reached --------------
        if node.level >= self.max_level:
            return
        #--------------------------------------------------------------
        #--------------- start by setting is_leaf to False ------------
        node.is_leaf = False
        #--------------------------------------------------------------
        #--------------- create children nodes ------------------------
        for i in range(8):
            #--------------- compute child center ----------------
            child_center = node.center + 0.5 * node.half_width * (2 * (i & 1) - 1) * np.array([1, 0, 0]) \
                                       + 0.5 * node.half_width * (2 * ((i >> 1) & 1) - 1) * np.array([0, 1, 0]) \
                                       + 0.5 * node.half_width * (2 * ((i >> 2) & 1) - 1) * np.array([0, 0, 1])
            #----------------------------------------------------
            #--------------- create child node -------------------
            child_node = self.make_child(node, child_center)
            #------------------------------------------------------
            #-------------- update parent child list with new child --------------
            node.children[i] = child_node
            #----------------------------------------------------------------------
        #--------------------------------------------------------------
        #--------------- free memory ---------------
        # removes indices from non-leaf nodes to save memory
        # NOTE - does not reset num_points as it counts all points in subtree
        node.indices = []  
        #-----------------------------------------------
        #--------------- recursively create children from each child node ---------------
        for child in node.children:
            self.make_children_DFS(child)
        #--------------------------------------------------------------------------------


    def _iter_children(self, node) -> Iterable:
        ch = getattr(node, "children", None)
        if not ch:
            return
        for c in ch:
            if c is not None:
                yield c

    def is_near_neighbor(self, node_a: TreeNode, node_b: TreeNode, pad: float = 1.01) -> bool:
        """
        Axis-aligned bounding boxes touch/overlap test.
        pad > 1 expands node_a's half-widths to get a 'near' halo.
        Works in 2D/3D (any dim, really).
        Requires node.center and node.half_width as array-likes.
        """
        c1 = np.asarray(node_a.center, dtype=float)
        c2 = np.asarray(node_b.center, dtype=float)
        h1 = np.asarray(node_a.half_width, dtype=float) * pad
        h2 = np.asarray(node_b.half_width, dtype=float)
        return np.all(np.abs(c1 - c2) <= (h1 + h2))

    def find_near_neighbors(self, node, pad: float = 1.01):
        """
        Build node.neighbors using the parent's neighbors (assumes those exist).
        Includes the node itself in the set.
        """
        neigh = set()
        neigh.add(node)
        #---------- get parent if not None ------------------------
        parent = getattr(node, "parent", None)
        #----------------------------------------------------------
        #--------- check parent if exists ----------------
        if parent is not None:
            #------------ first check siblings (e.i. nodes with same parent) -------------
            for sib in self._iter_children(parent):
                if sib is not node and self.is_near_neighbor(node, sib, pad=pad):
                    neigh.add(sib)
            #-----------------------------------------------------------------------------
            #----------- check parents neighbohrs (Pn) ------------------------
            for Pn in getattr(parent, "neighbors", ()):
                if Pn is parent:
                    continue
                #-------------- if Pn is leaf ----------------
                if Pn.is_leaf:
                    if self.is_near_neighbor(node, Pn, pad=pad):
                        neigh.add(Pn)
                #----------------------------------------------
                #---------- if Pn is not leaf iterate over its children -------------
                else:
                    for cousin in self._iter_children(Pn):
                        if self.is_near_neighbor(node, cousin, pad=pad):
                            neigh.add(cousin)
                #--------------------------------------------------------------------
        node.neighbors = neigh  

    def make_near_neighbors_lists(self, pad: float = 1.01):
        """
        Breadth-first: computes .neighbors for every node in the tree.
        Root gets {root}. Children at depth d+1 use only info from depth d.
        """
        #--------- first create root nbor list -------------
        # only root itself is in this 
        root = self.root
        root.neighbors = {root}
        #----------------------------------------------------
        #------------- construct node queue for BFS -------------
        q = deque([root])
        #--------------------------------------------------------
        #--------- while queue not empty, process next level -------------
        while q:
            #---------- get nodes from current level -------------
            this_level = list(q)
            #-----------------------------------------------------
            #------------- clear queue for next level -------------
            q.clear()
            #------------------------------------------------------
            #---------- compute neighbors for this level -------------
            for node in this_level:
                self.find_near_neighbors(node, pad=pad)
            #------------------------------------------------------
            #---------- gather next level nodes -------------------
            next_level = [child for parent in this_level for child in self._iter_children(parent)]
            q.extend(next_level)
            #-----------------------------------------------------

    def upwards_pass(self, node: TreeNode):
        """
        Computes multipole expansions from leaves up to root (P2M and M2M).
        """
        #---------- recurse on children first -------------
        # Must be done first to ensure finer levels are computed before coarser levels
        for child in node.children:
            self.upwards_pass(child)
        #--------------------------------------------------
        if node.is_leaf:
            #------------ if leaf, compute P2M ot get moments -------------
            X = self.points[node.indices]
            q = self.charges[node.indices]

            X_rel = X - node.center
            X_rel_sphe = utils.cart_to_sphe(X_rel)
            node.M = moment.P2M_sphe(X_rel_sphe, q, p=self.p)
            #--------------------------------------------------------------
        else:
            #----------- else, use M2M to aggregate child moments -------------
            node.M = np.zeros(((self.p+1)**2,), dtype=np.complex128)
            for child in node.children:
                x1 = node.center
                x0 = child.center
                node.M += moment.M2M_sphe(child.M, x0, x1)
            #---------------------------------------------------------------


