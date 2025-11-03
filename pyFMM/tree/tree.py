
#------------- needed to allow FMMTree type hints inside TreeNode -------------
from __future__ import annotations
from collections import deque
import itertools
from platform import node
import warnings
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

try:
    # import the utils module so we don't pollute this module's namespace
    from .. import pot_eval as pot_eval
except Exception:
    import pot_eval as pot_eval  # type: ignore
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
    """
    Tree structure for Fast Multipole Method (FMM).
    Holds global point data and builds the octree/quadtree structure.
    """
    def __init__(self,
                 center: np.ndarray,
                 size : np.ndarray,
                 points: np.ndarray,
                 charges: np.ndarray,
                 p: int = 4,
                 min_leaf_size: int = 10,
                 max_level: int = 5,
                 collapse_tol: float = 1e-6):
        """
        points: (N,3) source/target coordinates
        charges: (N,) source strengths (if shared source/target)
        """

        #----------- the tree itself stores data ---------------------
        self.points = np.asarray(points)
        self.charges = np.asarray(charges)
        #-------------------------------------------------------------
        #-------------- geometric parameters ----------------------
        self.center = np.asarray(center)            # center of the root box
        self.size = np.asarray(size)                # size of the root box
        self.collapse_tol = collapse_tol            # tolerance for collapsing near-zero dimensions
        self.do_dimension = [True, True, True]      # active dimensions
        #-------------------------------------------------------------
        #---------------- tree construction parameters ----------------
        self.min_leaf_size = int(min_leaf_size)     # minimum number of points per leaf - only used during pruning
        self.max_level = int(max_level)             # maximum tree depth
        #-----------------------------------------------------------
        #----------------- multipole parameters -------------------
        self.p = p  # multipole order
        #-----------------------------------------------------------
        #------------- node storage ----------------------
        self.root: Optional[TreeNode] = None
        self.node_list : List[TreeNode] = []
        #-------------------------------------------------


    def build_tree(self, BFS: bool = True):
        """
        Build tree structure down to max_level.
        input:
            BFS : bool - if True, build tree breadth-first; else depth-first
        """

        #------------- check active dimensions --------------------------------
        sz = np.asarray(self.size, dtype=float)
        do_dim = [bool(s > self.collapse_tol) for s in sz]
        #----------------------------------------------------------------------
        #--------------- ensure at least one dimension remains active ---------
        if not any(do_dim):
            raise ValueError(
                "All dimensions are collapsed (size <= collapse_tol); provide a non-zero "
                "size in at least one dimension or decrease collapse_tol."
            )
        #-----------------------------------------------------------------------
        #-------------- store active dimensions --------------------------------
        self.do_dimension = do_dim
        #------------------------------------------------------------------------
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
        input:
            node: TreeNode - the parent node
            child_center: np.ndarray - the center of the child node
        output:
            child_node: TreeNode - the created child node
        """
        #--------------- get half_width of child node ------------------------------
        child_half = node.half_width / 2.0
        #---------------------------------------------------------------------------
        #--------------- determine which points belong to this child ---------------
            #--------- calculate differences between point and child center --------
        diffs = np.abs(self.points[node.indices] - child_center)
            #------------------------------------------------------------------------
            #----create mask for points inside child using only active dimensions ---
        active = np.asarray(self.do_dimension, dtype=bool)
        if active.all():
            in_child_mask = np.all(diffs <= child_half, axis=1)
        else:
            in_child_mask = np.all(diffs[:, active] <= child_half[active], axis=1)
            #---------------------------------------------------------------------------
            #------------- get indices of points inside this child --------------------
        child_indices = node.indices[in_child_mask]
            #---------------------------------------------------------------------------
        #-------------------------------------------------------------------------------
        #------------------ create child node ------------------------------------------
        child_node = TreeNode(
            tree=self,
            center=child_center,
            size=node.size / 2.0,
            indices=child_indices,
            level=node.level + 1
        )
        #----------------------------------------------------------------------------
        #------------------ set parent of child node -------------------------------
        child_node.parent = node
        #----------------------------------------------------------------------------
        return child_node

    def _make_children_BFS(self, node: TreeNode):
        """
        Populates the tree in breadth-first manner - e.i. no recursion.
        input: node: TreeNode - the starting node (usually the root)
        """
        #------------- create queue for BFS -----------------------------------------
        q = deque([node])
        #----------------------------------------------------------------------------
        #------------- while queue not empty, process nodes -------------------------
        while q:
            #----------------- pop next node from queue ------------------------------
            node = q.popleft()
            #--------------------------------------------------------------------------
            #-------------- add node to tree list of nodes ----------------------------
            self.node_list.append(node)
            #--------------------------------------------------------------------------
            #--------------- stop if max level reached --------------------------------
            if node.level >= self.max_level:
                node.is_leaf = True
                continue
            #---------------------------------------------------------------------------
            #--------------- split current node ----------------------------------------
                #------------ first set is_leaf to False ------------------------------
            node.is_leaf = False
                #----------------------------------------------------------------------
                #-- determine number of children based on active dimensions -----------
            active_dims = [d for d, flag in enumerate(self.do_dimension) if flag]
            n_children = 2 ** len(active_dims)
            node.children = [None] * n_children
                #--------------------------------------------------------------------------
                #--------------- create children nodes ---------------------------------
            for idx, signs in enumerate(itertools.product([-1, 1], repeat=len(active_dims))):
                    #--------- compute child center ------------------------------------
                child_center = node.center.copy()
                    #-------------------------------------------------------------------
                    #---------- only adjust active dimensions --------------------------
                for d, s in zip(active_dims, signs):
                    child_center[d] += 0.5 * node.half_width[d] * s
                    #-------------------------------------------------------------------
                    #--------------- create child node -------------------------------
                child_node = self._make_child(node, child_center)
                    #-----------------------------------------------------------------
                    #-------------- update parent child list with new child --------------
                node.children[idx] = child_node
                    #----------------------------------------------------------------------
                    #--------------- add child to queue for future splitting -------------
                q.append(child_node)
                    #-------------------------------------------------------------------
                #---------------------------------------------------------------------------
            #--------------------------------------------------------------------------------
            #--------- delete indices from non-leaf nodes to save memory ------------------
            node.indices = [] 
            #-------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------

    def _make_children_DFS(self, node: TreeNode):
        """
        Recursively creates child nodes for the octree.
        """
        #-------------- append node to global list --------------
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
        """
        Yield children of `node` - a wrapper to avoid repeating code.
        input:
            node: TreeNode - the parent node
        output:
            list of child nodes of `node`
        """
        #----------------- get children -----------------------
        ch = getattr(node, "children", None)
        #------------------------------------------------------
        #--------------- return if no children -------------------
        if not ch:
            return
        #-----------------------------------------------------
        #------------- yield children ------------------------
        for c in ch:
            if c is not None:
                yield c
        #-----------------------------------------------------

    def find_leaf_for_point(self, point) -> Optional[TreeNode]:
        """
        Find the leaf node that contains the point.
        input: 
            point: array-like with 1-3 elements representing a point in space
        output:
            node : TreeNode | None
        """
        #--------------- first check if root exists ----------------
        if self.root is None:
            return None
        #----------------------------------------------------------

        #--------------- convert point to array -------------------
        p = np.asarray(point, dtype=float)
        #----------------------------------------------------------
        #--------------- get active dimensions -------------------
        do_dim = np.asarray(self.do_dimension, dtype=bool)
        #----------------------------------------------------------

        #--------------- handle point dimensionality ----------------
        if p.ndim == 0:
            p = np.asarray([p])
        #------------------------------------------------------------


        #------------- cast point to full 3-vector ----------------------------
            #---------if point dimensionality does not match active dims ------
        if p.size != do_dim.sum(): 
                # ----- try to accept by casting to 3D ------------------------
            if p.size == 3:
                full_p = p
                #--------------------------------------------------------------
                #--------- otherwise return None ------------------------------
            else:
                warnings.warn("Point dimensionality does not match active dimensions; returning None.", UserWarning)
                return None
                #--------------------------------------------------------------
            #------------------------------------------------------------------
            #---------- else fill full 3D point -------------------------------
        else:
            full_p = np.zeros(3, dtype=float)
            full_p[do_dim] = p
            #-------------------------------------------------------------------
        #-----------------------------------------------------------------------

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
            #----------- handle empty leaf case -------------
            if node.is_leaf:
                if node.indices.size == 0:
                    node.M = np.zeros(((self.p+1)**2,), dtype=np.complex128)
                    return
            #------------------------------------------------------------

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
                if child is None or getattr(child, 'M', None) is None:
                    continue
                node.M += moment.M2M_sphe(child.M, child.center, node.center)
            #------------------------------------------------------------------


    def _downwards_pass(self, root: TreeNode):
        """
        Creates the full octree structure (replaces make_children_recursively),
        but builds level-by-level instead of depth-first recursion.
        """
        #self.node_list = []  # we now fill this BFS-ordered

        # Queue for BFS

        root.L = np.zeros(((self.p+1)**2,), dtype=np.complex128)
        q = deque([root])
        while q:
            #-------------- get next node ----------------
            node = q.popleft()
            #if node.L is None:
            #    node.L = np.zeros(((self.p+1)**2,), dtype=np.complex128)
            #---------------------------------------------
            #------------------ convert multipole to local for each node in interaction list -------------------
            for i_node in node.interaction:
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
                #if node.L is not None:
                child.L = moment.L2L_sphe(node.L, node.center, child.center)
                q.append(child)
            #----------------------------------------------------------------------------------------------------

    

    def eval_P(self, eval_points: np.ndarray) -> np.ndarray:
        """
        Evaluate the multipole expansions at given evaluation points.
        """
        all_P = np.zeros(eval_points.shape[0], dtype=float)

        for i, point in enumerate(eval_points):
            #P = 0
            leaf = self.find_leaf_for_point(point)
            if leaf is None:
                raise ValueError(f"Point {point} is outside the tree domain.")
   

            #--------- evaluate local expansion from interaction list -------------
            points_rel = point - leaf.center
            points_rel_sphe = utils.cart_to_sphe(points_rel.reshape(1,3))[0]
            all_P[i] = pot_eval.P_L_sphe(leaf.L, points_rel_sphe)[0]
            #-----------------------------------------------------------------------
            #--------- evaluate direct sum from near neighbors -------------
            near_neighbors = getattr(leaf, 'neighbors', ())
            for n_node in near_neighbors:
                if n_node is None:
                    continue
                src_indices = n_node.indices
                X_src = self.points[src_indices]
                q_src = self.charges[src_indices]

                P_loc = pot_eval.P_direct_cart(X_src, q_src, np.array([point]))[0]
                all_P[i] += P_loc
            #--------------------------------------------------------------

            #all_P[i] = P
        return all_P
