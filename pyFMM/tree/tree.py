
#------------- needed to allow FMMTree type hints inside TreeNode -------------
from __future__ import annotations
from collections import deque
import itertools
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
        self.center = np.asarray(center, dtype=float)   # np.array([x,y,z])
        self.size = np.asarray(size, dtype=float)       # np.array([sx,sy,sz])
        # per-dimension half widths (array) — used for collapsed-dimension handling
        self.half_width = self.size / 2.0
        self.indices = indices          # np.array of source/target indices inside node
        self.num_points = len(indices)  # 
        self.level = level
        # allocate children based on how many active dimensions the tree has
        n_active = int(2 ** int(sum(getattr(tree, "do_dimension", [True, True, True]))))
        self.children: List[Optional[TreeNode]] = [None] * n_active
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
                 min_leaf_size: int = 10,
                 max_level: int = 5):
        """
        points: (N,3) source/target coordinates
        charges: (N,) source strengths (if shared source/target)
        """
        self.points = np.asarray(points)
        self.charges = np.asarray(charges)
        self.center = np.asarray(center)
        self.size = np.asarray(size)
        # minimum number of points per leaf; if a child has fewer than
        # this many points the parent may reclaim them during pruning
        self.min_leaf_size = int(min_leaf_size)
        self.max_level = int(max_level)
        self.root: Optional[TreeNode] = None
        self.node_list : List[TreeNode] = []
        self.p = p  # multipole order
        # tolerance for collapsing near-zero dimensions; dimensions with
        # size <= collapse_tol are considered collapsed (ignored)
        self.collapse_tol = 1e-6
        # default do_dimension (all true) — will be updated in build_tree
        self.do_dimension = [True, True, True]


    def build_tree(self, BFS: bool = True):
        """
        Builds the octree structure recursively.
        1. Create root node containing all points.
        2. Recursively subdivide nodes until max level or leaf size reached.
        3. Store nodes in self.node_list in depth-first order.
        """
        # decide which dimensions are active (collapsed dims are ignored)
        sz = np.asarray(self.size, dtype=float)
        do_dim = [bool(s > self.collapse_tol) for s in sz]
        # ensure at least one dimension remains active
        if not any(do_dim):
            raise ValueError(
                "All dimensions are collapsed (size <= collapse_tol); provide a non-zero "
                "size in at least one dimension or decrease collapse_tol."
            )
        self.do_dimension = do_dim

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
            self._make_children_BFS(root_node)
        else:
            self._make_children_DFS(root_node)
        #-------------------------------------------------------


    def _make_child(self, node: TreeNode, child_center: np.ndarray) -> TreeNode:
        """
        Creates a single child node
        """
        # Determine which points belong to this child.
        # Only consider active (non-collapsed) dimensions when testing membership.
        active = np.asarray(self.do_dimension, dtype=bool)
        child_half = node.half_width / 2.0
        diffs = np.abs(self.points[node.indices] - child_center)
        if active.all():
            in_child_mask = np.all(diffs <= child_half, axis=1)
        else:
            # compare only along active dimensions
            in_child_mask = np.all(diffs[:, active] <= child_half[active], axis=1)
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

    def _make_children_BFS(self, node: TreeNode):
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

            # Create children nodes. Use only active dimensions when forming combinations.
            active_dims = [d for d, flag in enumerate(self.do_dimension) if flag]
            n_children = 2 ** len(active_dims)
            # ensure children list has correct length
            node.children = [None] * n_children
            for idx, signs in enumerate(itertools.product([-1, 1], repeat=len(active_dims))):
                child_center = node.center.copy()
                for d, s in zip(active_dims, signs):
                    child_center[d] += 0.5 * node.half_width[d] * s
                child_node = self._make_child(node, child_center)
                node.children[idx] = child_node
                # Add to queue for future splitting
                q.append(child_node)

            # Free memory on internal node
            node.indices = []  # same as your original

    def _make_children_DFS(self, node: TreeNode):
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
        active_dims = [d for d, flag in enumerate(self.do_dimension) if flag]
        n_children = 2 ** len(active_dims)
        node.children = [None] * n_children
        for idx, signs in enumerate(itertools.product([-1, 1], repeat=len(active_dims))):
            child_center = node.center.copy()
            for d, s in zip(active_dims, signs):
                child_center[d] += 0.5 * node.half_width[d] * s
            #----------------------------------------------------
            #--------------- create child node -------------------
            child_node = self._make_child(node, child_center)
            #------------------------------------------------------
            #-------------- update parent child list with new child --------------
            node.children[idx] = child_node
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

    def find_leaf_for_point(self, point) -> Optional[TreeNode]:
        """Return the leaf node that contains `point` or None if point is outside the root.

        `point` can be an iterable or array with 1-3 elements; only active dimensions
        (self.do_dimension) are considered. If the point is outside the root domain
        along any active dimension, returns None.
        """
        if self.root is None:
            return None

        p = np.asarray(point, dtype=float)
        do_dim = np.asarray(self.do_dimension, dtype=bool)
        # allow 1D/2D/3D input matching active dims
        if p.ndim == 0:
            p = np.asarray([p])
        if p.size != do_dim.sum():
            # try to accept full 3-element points by picking active dims
            if p.size == 3:
                full_p = p
            else:
                return None
        else:
            # expand p to full 3-vector using active dims
            full_p = np.zeros(3, dtype=float)
            full_p[do_dim] = p

        # check inside root
        root = self.root
        hw = root.half_width
        if not np.all(np.abs(full_p - root.center)[do_dim] <= hw[do_dim]):
            return None

        node = root
        # descend until a leaf is reached
        while not node.is_leaf:
            found = False
            for child in getattr(node, 'children', ()):
                if child is None:
                    continue
                ch_center = np.asarray(child.center, dtype=float)
                ch_hw = np.asarray(child.half_width, dtype=float)
                if np.all(np.abs(full_p - ch_center)[do_dim] <= ch_hw[do_dim]):
                    node = child
                    found = True
                    break
            if not found:
                # point not found in any child — return None
                return None
        return node

    def _compute_interaction_list(self, node: TreeNode):
        """Compute and return the interaction list for `node`.

        Definition (informal): children of the near-neighbours of node's parent
        which are at the same level as node and are well-separated from node
        (i.e., not near neighbours of node).
        """
        if node is None or node.parent is None:
            return set()

        parent = node.parent
        # ensure neighbor lists exist for parent
        if not hasattr(parent, 'neighbors') or not parent.neighbors:
            # build global neighbor lists (cheap if already built)
            self._make_near_neighbors_lists()

        interaction = set()
        # ensure node.neighbors exists
        if not hasattr(node, 'neighbors') or not node.neighbors:
            self._find_near_neighbors(node)

        for Pn in getattr(parent, 'neighbors', ()):  # type: ignore
            # skip if neighbour has no children
            for child in getattr(Pn, 'children', ()):  # type: ignore
                if child is None:
                    continue
                # only consider children at the same level as node
                if getattr(child, 'level', None) != getattr(node, 'level', None):
                    continue
                # exclude near neighbours of node (we want well-separated boxes only)
                if child in getattr(node, 'neighbors', ()):  # type: ignore
                    continue
                # finally, check adjacency using _is_near_neighbor to be safe
                if not self._is_near_neighbor(child, node):
                    interaction.add(child)

        # store on node for later use
        node.interaction = interaction
        return interaction

    def _compute_all_interaction_lists(self):
        """Compute interaction lists for all nodes in the tree and store them as node.interaction."""
        if not self.node_list:
            self._rebuild_node_list()
        for node in self.node_list:
            node.interaction = self._compute_interaction_list(node)


    # (internal) near-neighbour helper
    def _is_near_neighbor(self, node_a: TreeNode, node_b: TreeNode, pad: float = 1.01) -> bool:
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

    def _find_near_neighbors(self, node, pad: float = 1.01):
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
                if sib is not node and self._is_near_neighbor(node, sib, pad=pad):
                    neigh.add(sib)
            #-----------------------------------------------------------------------------
            #----------- check parents neighbohrs (Pn) ------------------------
            for Pn in getattr(parent, "neighbors", ()):
                if Pn is parent:
                    continue
                #-------------- if Pn is leaf ----------------
                if Pn.is_leaf:
                    if self._is_near_neighbor(node, Pn, pad=pad):
                        neigh.add(Pn)
                #----------------------------------------------
                #---------- if Pn is not leaf iterate over its children -------------
                else:
                    for cousin in self._iter_children(Pn):
                        if self._is_near_neighbor(node, cousin, pad=pad):
                            neigh.add(cousin)
                #--------------------------------------------------------------------
        node.neighbors = neigh  

    def _make_near_neighbors_lists(self, pad: float = 1.01):
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
                self._find_near_neighbors(node, pad=pad)
            #------------------------------------------------------
            #---------- gather next level nodes -------------------
            next_level = [child for parent in this_level for child in self._iter_children(parent)]
            q.extend(next_level)
            #-----------------------------------------------------

    def make_lists(self, pad: float = 1.01):
        """Public helper: compute neighbor lists and interaction lists for the tree."""
        self._make_near_neighbors_lists(pad=pad)
        self._compute_all_interaction_lists()



    def _rebuild_node_list(self):
        """Rebuild self.node_list using BFS from the root."""
        self.node_list = []
        if self.root is None:
            return
        q = deque([self.root])
        while q:
            n = q.popleft()
            self.node_list.append(n)
            for c in getattr(n, "children", ()):  # type: ignore
                if c is not None:
                    q.append(c)



    def construct_moments(self):
        """Construct multipole expansions for all nodes in the tree."""
        if self.root is None:
            return
        self._upwards_pass(self.root)
        self._downwards_pass(self.root)


    def _upwards_pass(self, node: TreeNode):
        """
        Computes multipole expansions from leaves up to root (P2M and M2M).
        """
        #---------- recurse on children first -------------
        # Must be done first to ensure finer levels are computed before coarser levels
        for child in node.children:
            if child is not None:
                self._upwards_pass(child)
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
            #------------------------------------------------------------------


    def _downwards_pass(self, node: TreeNode):
        """
        Creates the full octree structure (replaces make_children_recursively),
        but builds level-by-level instead of depth-first recursion.
        """
        #self.node_list = []  # we now fill this BFS-ordered

        # Queue for BFS
        q = deque([node])
        while q:
            #-------------- get next node ----------------
            node = q.popleft()
            #---------------------------------------------
            #------------------ convert multipole to local for each node in interaction list -------------------
            for i_node in node.interaction:
                if node.L is None:
                    node.L = np.zeros(((self.p+1)**2,), dtype=np.complex128)
                node.L += moment.M2L_sphe(i_node.M, i_node.center, node.center)
            #----------------------------------------------------------------------------------------------------
            #-------------- if this is a leaf continue to next node in queue -------------------
            if node.is_leaf:
                continue
            #----------------------------------------------------------------------------------
            #------ propagate local expansions to children (L2L) ------------------------------------------------
            for child in node.children:
                if child is None:
                    continue
                if node.L is not None:
                    child.L = moment.L2L_sphe(node.L, node.center, child.center)
                q.append(child)
            #----------------------------------------------------------------------------------------------------