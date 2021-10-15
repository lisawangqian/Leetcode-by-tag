### 145. Binary Tree Postorder Traversal ###
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def dfs(self, root, ans):
        if not root:
            return
        self.dfs(root.left, ans)
        self.dfs(root.right, ans)
        ans.append(root.val)
        
        
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        ans = []
        self.dfs(root, ans)
        return ans


### 94. Binary Tree Inorder Traversal ###
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def dfs(self, root, ans):
        if not root:
            return
        self.dfs(root.left, ans)
        ans.append(root.val)
        self.dfs(root.right, ans)
        
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        ans = []
        self.dfs(root, ans)
        return ans


### 144. Binary Tree Preorder Traversal ###
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def dfs(self, root, ans):
        if not root:
            return
        
        ans.append(root.val)
        self.dfs(root.left, ans)
        self.dfs(root.right, ans)
        
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        ans = []
        self.dfs(root, ans)
        return ans


### 589. N-ary Tree Preorder Traversal ###

# Definition for a Node.
#class Node:
#    def __init__(self, val=None, children=None):
#        self.val = val
#        self.children = children

class Solution:
    def dfs(self, root, ans):
        if not root:
            return
        
        ans.append(root.val)
        for child in root.children:
            self.dfs(child, ans)
        
    def preorder(self, root: 'Node') -> List[int]:
        ans = []
        self.dfs(root, ans)
        return ans


### 590. N-ary Tree Postorder Traversal ###

# Definition for a Node.
#class Node:
#    def __init__(self, val=None, children=None):
#        self.val = val
#        self.children = children

class Solution:
    def dfs(self, root, ans):
        if not root:
            return
        for child in root.children:
            self.dfs(child, ans)
        ans.append(root.val)
        
    def postorder(self, root: 'Node') -> List[int]:
        ans = []
        self.dfs(root, ans)
        return ans  


### 104. Maximum Depth of Binary Tree ###
#1)DFS 
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def dfs(self, root):
        if not root:  
            return 0
        l = self.dfs(root.left)
        r = self.dfs(root.right)
        height = 1 + max(l, r) 
        return height
            
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        return self.dfs(root)
        
#2)Stack
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        stack = []
        if root:
            stack.append((1, root))
        depth = 0
        while stack:
            curr_depth, root = stack.pop()
            if root:
                depth = max(depth, curr_depth)
                stack.append((curr_depth + 1, root.left))
                stack.append((curr_depth + 1, root.right))
                
        return depth


### 543. Diameter of Binary Tree ###
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        def dfs(root):
            nonlocal diameter
            if not root: #length of leaf as 1
                return 0
            l = dfs(root.left)
            r = dfs(root.right)
            diameter = max(diameter, l + r)
            return max(l, r) + 1 
        
        diameter = 0
        dfs(root)
        return diameter


### 110. Balanced Binary Tree ###
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def dfs(self, root): #calculate height, postorder
        if not root:
            return 0
        l = self.dfs(root.left)
        r = self.dfs(root.right)
        return max(l, r) + 1
        
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        if not root:
            return True
        
        return (abs(self.dfs(root.left) - self.dfs(root.right)) <= 1) & self.isBalanced(root.right) & self.isBalanced(root.left)


### 257. Binary Tree Paths ###
#1) DFS preorder
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def dfs(self, root, path): #preorder
        
        if not root:
            return
        path += str(root.val)
        if not root.left and not root.right: #leaf
            self.result.append(path)
        else:
            path += '->'
            self.dfs(root.left, path)
            self.dfs(root.right, path)
        
    def binaryTreePaths(self, root: TreeNode) -> List[str]:
        
        self.result = []
        self.dfs(root, '')
            
        return self.result
        
#2)Stack
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        if not root:
            return []
        
        result = []
        stack = [(str(root.val), root)]
        while stack:
            path, root = stack.pop()  #every level, has its recorded path
            if not root.left and not root.right:
                result.append(path)
            if root.left:  
                stack.append((path + '->' + str(root.left.val), root.left))
            if root.right:
                stack.append((path + '->' + str(root.right.val), root.right))
                             
        return result


### 111. Minimum Depth of Binary Tree ###
#1)recursion
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def dfs(self, root, length): #preorder
        if not root:
            return 
        if not root.left and not root.right: #leaf
            self.ans = min(self.ans, length + 1)
        else:
            self.dfs(root.left, length + 1)
            self.dfs(root.right, length + 1)
            
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        self.ans = float(inf)
        self.dfs(root, 0)
        return self.ans

#2)Stack
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        ans = float(inf)
        stack = [(root, 0)]
        while stack:
            root, length = stack.pop()  #every level, has its recorded path
            if not root.left and not root.right:
                ans = min(ans, length + 1)
            if root.left:  
                stack.append((root.left, length+1))
            if root.right:
                stack.append((root.right, length+1))
        
        return ans


### 112. Path Sum ###
#1)Recursion
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if not root:
            return False
        targetSum -= root.val
        if not root.left and not root.right:
            return (targetSum == 0)
        
        return self.hasPathSum(root.left, targetSum) or self.hasPathSum(root.right, targetSum)

#2)Stack:
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if not root:
            return False
        stack = [(root, targetSum-root.val)]
        while stack:
            node, targetSum = stack.pop()
            if not node.left and not node.right: 
                if targetSum == 0:
                    return True
            if node.left: 
                stack.append((node.left, targetSum - node.left.val))
            if node.right:
                stack.append((node.right, targetSum - node.right.val))
                
        return False


### 226. Invert Binary Tree ###
#1)DFS 
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    
    def invertTree(self, root: TreeNode) -> TreeNode:  #reversed preorder
        if not root:
            return None
        
        root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
        
        return root

#2)Stack
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    
    def invertTree(self, root: TreeNode) -> TreeNode:  #reversed preorder
        if not root:
            return None
        
        stack = []
        stack.append(root)
        while stack:
            node = stack.pop()
            node.left, node.right = node.right, node.left
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
                
        return root


### 101. Symmetric Tree ###
#1)Recursive
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isMirror(self, t1, t2):
        if not t1 and not t2:
            return True
        elif not t1 or not t2:
            return False
        return (t1.val == t2.val) & (self.isMirror(t1.right, t2.left)) & (self.isMirror(t1.left, t2.right))
        
    def isSymmetric(self, root: TreeNode) -> bool:
        return self.isMirror(root, root)  #start from one point root but later will be their subtrees

#2)stack
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        if not root:
            return True
        stack = []
        stack.append([root, root])
        
        while stack:
            t1, t2 = stack.pop()
            if not t1 and t2:
                return False
            if not t2 and t1:
                return False
            if t1 and t2 and t1.val != t2.val:
                return False
            if t1 and t2:
                stack.append([t1.right, t2.left])
                stack.append([t1.left, t2.right])
            
        return True


### 100. Same Tree ###   #similar as 101. Symmetric Tree
#1)Recursion
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q:
            return True
        if not p or not q:
            return False
        if p.val != q.val:
            return False
        
        return self.isSameTree(p.left, q.left) & self.isSameTree(p.right, q.right)

#2)Stack       
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q:
            return True
        if not p or not q:
            return False
        
        stack = [(p, q)]
        
        while stack:
            p, q = stack.pop()
            if not p and not q:
                continue
            if not p or not q:
                return False
            if p.val != q.val:
                return False
            stack.append((p.left, q.left))
            stack.append((p.right, q.right))
            
        return True


### 572. Subtree of Another Tree ###
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSubtree(self, root: TreeNode, subRoot: TreeNode) -> bool:
        if not subRoot:
            return True
        
        def isSameTree(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
            if not p and not q:
                return True
            if not p or not q:
                return False
            if p.val != q.val:
                return False

            return isSameTree(p.left, q.left) & isSameTree(p.right, q.right)

        def dfs(root1, root2):
            if not root1:
                return False
            
            if root1.val == root2.val and isSameTree(root1, root2):
                return True
            
            return dfs(root1.left, root2) or dfs(root1.right, root2)
            
        return dfs(root, subRoot)


### 617. Merge Two Binary Trees ###
#1)Recursion
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root1:
            return root2
        if not root2:
            return root1
        root1.val += root2.val
        root1.left = self.mergeTrees(root1.left, root2.left)
        root1.right = self.mergeTrees(root1.right, root2.right)
        
        return root1

#2)Stack
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root1:
            return root2
        
        stack = [(root1, root2)]
        
        while stack:
            t1, t2 = stack.pop()
            if not t2:
                continue
            t1.val += t2.val
            
            if not t1.left:
                t1.left = t2.left
            else:
                stack.append((t1.left, t2.left))
            
            if not t1.right:
                t1.right = t2.right
            else:
                stack.append((t1.right, t2.right))
                
        return root1


#2)Stack:
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isCousins(self, root: TreeNode, x: int, y: int) -> bool:
        p = []  #parents
        d = []  #depths
        
        stack = [(root, None, 0)]
        while stack:
            root, parent, depth = stack.pop()  #every level, has its recorded path
            if root.val == x or root.val == y:
                p.append(parent)
                d.append(depth)
                if len(p) == 2:
                    break
                else:
                    continue
               
            if root.left:  
                stack.append((root.left, root, depth+1))
            if root.right:
                stack.append((root.right, root, depth+1))
        
        if len(p) == 2 and len(d) == 2 and p[0]!=p[1] and d[0] == d[1]:
            return True

        
        return False


### 993. Cousins in Binary Tree ###
#1)DFS
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def dfs(self, node, parent, depth, targets): #preorder
        if not node:
            return
        if node.val in targets: #targets is set of (x, y)
            self.p.append(parent)
            self.d.append(depth)
            return
        self.dfs(node.left, node, depth + 1, targets)
        self.dfs(node.right, node, depth + 1, targets)
        
    def isCousins(self, root: TreeNode, x: int, y: int) -> bool:
        self.p = []  #parents
        self.d = []  #depths
        
        self.dfs(root, None, 0, set([x, y]))
        #different parents and same depth
        if len(self.p) == 2 and len(self.d) ==2 and self.p[0]!=self.p[1] and self.d[0] == self.d[1]:
            return True
        else:
            return False


### 404. Sum of Left Leaves ###
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sumOfLeftLeaves(self, root: TreeNode) -> int:
        def dfs(root): #preorder (root, left, right)
            nonlocal s
            if root is None:
                return 0
            if root.left is None and root.right is None: #leaf
                return root.val
            
            left = dfs(root.left)
            s+=left
            dfs(root.right)
            return 0  #essential to avoid left no return(None) - s+left error
        
        s = 0
        dfs(root)
        return s


### 637. Average of Levels in Binary Tree ###
#1)dfs
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def averageOfLevels(self, root: TreeNode) -> List[float]:
        info = []
        def dfs(node, depth):  #preorder
            if node:
                if len(info) <= depth: #this depth just started
                    info.append([0, 0])
                info[depth][0] += node.val  #sum of values for depth
                info[depth][1] += 1  #count of element for depth
                dfs(node.left, depth + 1)
                dfs(node.right, depth + 1)
        
        dfs(root, 0)

        return [s/float(c) for s, c in info]
#2)stack

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def averageOfLevels(self, root: TreeNode) -> List[float]:
        info = []
        stack = [(root, 0)]
        while stack:
            root, depth = stack.pop()
            if len(info) <= depth:
                info.append([0, 0])
            
            info[depth][0]+=root.val
            info[depth][1]+=1
            if root.left:
                stack.append((root.left, depth+1))
            if root.right:
                stack.append((root.right, depth+1))
           
        return [s/float(c) for s, c in info]


### 108. Convert Sorted Array to Binary Search Tree ###
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        def helper(left, right):  #preorder  
            if left > right:
                return None
            mid = left + (right-left)//2
            
            root = TreeNode(nums[mid])
            root.left = helper(left, mid-1)
            root.right = helper(mid+1, right)
            return root
        
        return helper(0, len(nums) - 1)


### 938. Range Sum of BST ###
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rangeSumBST(self, root: Optional[TreeNode], low: int, high: int) -> int:
        def dfs(root):
            if not root:
                return
            nonlocal result
            if low <= root.val <= high:
                result+=root.val
            if root.val > low: #binary search tree property->only need to go left
                dfs(root.left)
            if root.val < high: #binary search tree property->only need to go right
                dfs(root.right)
       
        result = 0
        dfs(root)
        return result


### 270. Closest Binary Search Tree Value ###
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def closestValue(self, root: TreeNode, target: float) -> int:
        closest = float('inf')
        
        def dfs(root):
            nonlocal closest
            
            if not root:
                return
            if abs(root.val - target) < abs(closest - target):
                closest = root.val
                
            # Target should be located on left subtree
            if target < root.val:
                dfs(root.left)
            
            # target should be located on right subtree
            if target > root.val:
                dfs(root.right)
        
        dfs(root)
        return closest


### 897. Increasing Order Search Tree ###
#1)Recontruct
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def increasingBST(self, root: TreeNode) -> TreeNode:
        if not root:
            return None
            
        def dfs(root):  #in-order dfs tranverse bst in order
            if not root:
                return
            dfs(root.left)
            result.append(root.val)
            dfs(root.right)
                
        result = []
        dfs(root)
        ans = cur = TreeNode()
        for v in result:
            cur.right = TreeNode(v)
            cur = cur.right
        return ans.right

#2)Operate in place
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def increasingBST(self, root: TreeNode) -> TreeNode:
        if not root:
            return None
            
        def dfs(node, nxt):  #in order tranverse BST in order
            if not node:
                return nxt
            res = dfs(node.left, node)
            node.left = None
            node.right = dfs(node.right, nxt)
            
            return res
        
        return dfs(root, None)


### 671. Second Minimum Node In a Binary Tree ###
#1) DFS
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findSecondMinimumValue(self, root: TreeNode) -> int:
        col = set()
        def dfs(root):  #preorder
            if not root:
                return
            col.add(root.val)
            dfs(root.left)
            dfs(root.right)
                
        dfs(root)
        if len(col) < 2:
            return -1
        return sorted(col)[1]

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findSecondMinimumValue(self, root: TreeNode) -> int:
        col = set()
        stack = [root]
        while stack:
            node = stack.pop()
            curr = node.val
            col.add(curr)
            
            if node.left:
                stack.append(node.left)
                stack.append(node.right)
                  
        if len(col) >= 2:
            return sorted(col)[1]
        else:
            return -1


### 783. Minimum Distance Between BST Nodes ###
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def minDiffInBST(self, root: TreeNode) -> int:
        def dfs(root): #(left, root, right)
            nonlocal ans, prev
            if root:
                dfs(root.left)
                ans = min(ans, root.val - prev)
                prev = root.val
                dfs(root.right)
                
        prev = float('-inf')
        ans = float('inf')
        dfs(root)
        return ans

### 530. Minimum Absolute Difference in BST  same as 783
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
        def dfs(root): #(left, root, right)
            nonlocal ans, prev
            if root:
                dfs(root.left)
                ans = min(ans, root.val - prev)
                prev = root.val
                dfs(root.right)
                
        prev = float('-inf')
        ans = float('inf')
        dfs(root)
        return ans


### 700. Search in a Binary Search Tree ###
#1) recursion
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        if not root:
            return None
        
        if root.val == val:
            return root
        
        elif root.val > val:
            return self.searchBST(root.left, val)
        
        else:
            return self.searchBST(root.right, val)
#2)iteration           
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        
        while root and root.val != val:
            if root.val > val:
                root = root.left
            else:
                root = root.right
                
        return root


### 235. Lowest Common Ancestor of a Binary Search Tree ###
#recursion
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        parent_v = root.val
        p_v = p.val
        q_v = q.val
        
        if p_v > parent_v and q_v > parent_v: #on the right subtree
            return self.lowestCommonAncestor(root.right, p, q)
        elif p_v < parent_v and q_v < parent_v: #on the left subtree
            return self.lowestCommonAncestor(root.left, p, q)
        else:
            return root

#2)iteration
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        
        p_v = p.val
        q_v = q.val
        node = root
        
        while node:
            if p_v > node.val and q_v > node.val: #on the right subtree
                node = node.right
            elif p_v < node.val and q_v < node.val: #on the left subtree
                node = node.left
            else:
                return node
        

### 102. Binary Tree Level Order Traversal ### similar as 637
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        result = []
        def dfs(root, depth):
            if not root:
                return
            if len(result) <= depth:
                result.append([])
            result[depth].append(root.val)
            dfs(root.left, depth+1)
            dfs(root.right, depth+1)
            
        dfs(root,0)
        return result

### 103. Binary Tree Zigzag Level Order Traversal ### similar as 102
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        result = []
        def dfs(root, depth):
            if not root:
                return
            if len(result) <= depth:
                result.append([])
            if depth%2 == 1:
                result[depth].insert(0, root.val)
            else:
                result[depth].append(root.val)
            dfs(root.left, depth+1)
            dfs(root.right, depth+1)
            
        dfs(root,0)
        return result


### 107. Binary Tree Level Order Traversal II ### similar as 102
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrderBottom(self, root: Optional[TreeNode]) -> List[List[int]]:
        result = []
        def dfs(root, depth):
            if not root:
                return
            if len(result) <= depth:
                result.append([])
            result[depth].append(root.val)
            dfs(root.left, depth+1)
            dfs(root.right, depth+1)
            
        dfs(root,0)
        return result[::-1]


### 105. Construct Binary Tree from Preorder and Inorder Traversal ###
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        
        def array_to_tree(left, right):
            nonlocal preorder_index
            if left > right:
                return None
            
            root_value = preorder[preorder_index]
            root = TreeNode(root_value)
            preorder_index+=1
            root.left = array_to_tree(left, inorder_index_map[root_value] - 1)
            root.right = array_to_tree(inorder_index_map[root_value] + 1, right)
            return root
        
        preorder_index = 0
        inorder_index_map = {}
        for index, value in enumerate(inorder):
            inorder_index_map[value] = index
            
            
        return array_to_tree(0, len(preorder)-1)


### 1008. Construct Binary Search Tree from Preorder Traversal ###
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def bstFromPreorder(self, preorder: List[int]) -> Optional[TreeNode]:
        def array_to_tree(left, right):
            nonlocal preorder_index
            if left > right:
                return None
            
            root_value = preorder[preorder_index]
            root = TreeNode(root_value)
            preorder_index+=1
            root.left = array_to_tree(left, inorder_index_map[root_value] - 1)
            root.right = array_to_tree(inorder_index_map[root_value] + 1, right)
            return root
        
        preorder_index = 0
        inorder = sorted(preorder)  #binary search tree property, inorder is ordered
        inorder_index_map = {}
        for index, value in enumerate(inorder):
            inorder_index_map[value] = index
            
            
        return array_to_tree(0, len(preorder)-1)


### 106. Construct Binary Tree from Inorder and Postorder Traversal ###
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        def array_to_tree(left, right):
            if left > right:
                return None
            
            root_value = postorder.pop()
            root = TreeNode(root_value)
            root.right = array_to_tree(inorder_index_map[root_value] + 1, right)
            root.left = array_to_tree(left, inorder_index_map[root_value] - 1)
            
            return root
        
        
        inorder_index_map = {}
        for index, value in enumerate(inorder):
            inorder_index_map[value] = index
            
            
        return array_to_tree(0, len(postorder)-1)


### 889. Construct Binary Tree from Preorder and Postorder Traversal ###
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def constructFromPrePost(self, preorder: List[int], postorder: List[int]) -> TreeNode:
        def array_to_tree(left, right):
            
        	if left > right: return None
             
        	if left == right: return TreeNode(preorder[left])

        	root_value = preorder[left]
        	root = TreeNode(root_value)
            
            # always pick the previous one in postorder;
            #this value will be root_value'right child's root
        	midVal = postorder[postorder_index_map[root_value]-1] 
            #find this value in preorder list position/index as right child's root
        	midIdx = preorder_index_map[midVal] 
            
            # preorder[left + 1] to be the left subtree root
        	root.left  = array_to_tree(left+1, midIdx-1) 
            # preorder[midIdx] to be the right subtree root
        	root.right = array_to_tree(midIdx, right)

        	return root
        
        # construct num:index mapper
        preorder_index_map  = {value:index for index, value in enumerate(preorder)}
        postorder_index_map = {value:index for index, value in enumerate(postorder)}
        return array_to_tree(0, len(preorder)-1)
        

### 114. Flatten Binary Tree to Linked List ###
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        
        def flatten(root):  #root, left, right for preorder
            if not root:
                return None
            if not root.left and not root.right: #leaf
                return root
            
            leftTail = flatten(root.left)  #tail of flattened
            rightTail = flatten(root.right) #tail of flattened
            
            if leftTail:
                leftTail.right = root.right
                root.right = root.left
                root.left = None
                
            return rightTail if rightTail else leftTail
        
        flatten(root)
            

 ### 236. Lowest Common Ancestor of a Binary Tree ###
 # Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        ans = None
        def dfs(root):  #logic true or false
            nonlocal ans
            if not root:
                return False
            mid = (root == p) or (root == q)
            left = dfs(root.left)
            right = dfs(root.right)
            
            if int(left + right + mid) >=2:
                ans = root  #find the ancestor
              
            return mid or left or right
        
        dfs(root)
        return ans


### 222. Count Complete Tree Nodes ###
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def countNodes(self, root: Optional[TreeNode]) -> int:
        ans, last_depth = 0, 0
        def dfs(root, depth):
            nonlocal ans, last_depth
            if not root:
                return
            if not root.left and not root.right:
                last_depth = max(depth, last_depth)
                if depth == last_depth:
                    ans+=1
                
            if root.left:
                dfs(root.left, depth+1)
            if root.right:
                dfs(root.right, depth+1)
        
        dfs(root, 0)        
        
        for i in range(last_depth):
            ans+=2**i
            
        return ans           

#2)binary search
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def compute_depth(self, root):
        d = 0
        while root.left:
            root = root.left
            d+=1
        return d
    
    def exists(self, idx, d, root):
        left, right = 0, 2**d-1
        for _ in range(d):  #depth
            mid =  left + (right -left)//2
            if idx <= mid:
                root = root.left #final node is at the last level
                right = mid
            else:
                root = root.right
                left = mid+1
        return root is not None
                
    def countNodes(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        d = self.compute_depth(root) #get tree depth
        
        if d == 0:
            return 1
        
        left, right = 0, 2**d-1  #the last level potential nodes index from 0
        while left <= right:
            mid = left + (right -left)//2
            if self.exists(mid, d, root):
                left = mid + 1
            else:
                right = mid -1
                
        return (2**d - 1) + left
                

### 129. Sum Root to Leaf Numbers ###
#1) dfs
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sumNumbers(self, root: TreeNode) -> int:
        def dfs(root, curr_num):
            nonlocal root_to_leaf_sum
            
            if not root:
                return
            
            curr_num = 10 * curr_num + root.val
            
            if not root.left and not root.right: #reach leaf
                root_to_leaf_sum += curr_num
                
            dfs(root.left, curr_num)
            dfs(root.right, curr_num)
            
        root_to_leaf_sum = 0
        dfs(root, 0)
        return root_to_leaf_sum

#2) stack
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sumNumbers(self, root: TreeNode) -> int:
        root_to_leaf_sum = 0
        stack = [(root, 0)]
        while stack:
            root, curr_num = stack.pop()
            curr_num = 10 * curr_num + root.val
            if not root.left and not root.right:
                root_to_leaf_sum += curr_num
            if root.left:
                stack.append((root.left, curr_num))
            if root.right:
                stack.append((root.right, curr_num))
       
        return root_to_leaf_sum
        

### 113. Path Sum II ###
#1)DFS backtracking
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:

     def pathSum(self, root: TreeNode, targetSum: int) -> List[List[int]]:
        def dfs(root, curr_sum, path):
            if not root:
                return
            curr_sum += root.val
            path.append(root.val)
            
            if not root.left and not root.right and curr_sum == targetSum:
                result.append(list(path))  #important use list to make it unmutable
            
            dfs(root.left, curr_sum, path)
            dfs(root.right, curr_sum, path)
            
            path.pop()  #import to pop element for backtracking
        
        result = []
        dfs(root, 0, [])
        return result
#2)DFS:
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pathSum(self, root: TreeNode, targetSum: int) -> List[List[int]]:
        def dfs(root, curr_sum, path):
            if not root:
                return
            curr_sum += root.val
            #path.append(root.val)
            
            if not root.left and not root.right and curr_sum == targetSum:
                result.append(list(path))
            if root.left:
                dfs(root.left, curr_sum, path + [root.left.val])
            if root.right:
                dfs(root.right, curr_sum, path + [root.right.val])
            
        if not root:
            return None
        result = []
        dfs(root, 0, [root.val])
        return result
#3)stack
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pathSum(self, root: TreeNode, targetSum: int) -> List[List[int]]:
        if not root:
            return None
        
        result = []
        stack = [(root, 0, [root.val])]
        while stack:
            root, curr_sum, path = stack.pop()
            curr_sum += root.val
            if not root.left and not root.right and curr_sum == targetSum:
                result.append(list(path))
            if root.left:
                stack.append((root.left, curr_sum, path + [root.left.val]))
            if root.right:
                stack.append((root.right, curr_sum, path+ [root.right.val]))
            
        return result


### 437. Path Sum III ###
#dfs backtracking:
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
       
    def pathSum(self, root: TreeNode, targetSum: int) -> int:
        count, h = 0, defaultdict(int)  #dictionary root-node sum: count
        
        def dfs(root, curr_sum):
            nonlocal count
            if not root:
                return
            curr_sum += root.val
            if curr_sum == targetSum: #1) path start from root
                count+=1
            
            #number of times seeing prefix sum: curr_sum - targetSum: root to somepoint
            #somepoint.next(left or right) to current node:curr_sum - (curr_sum-targetSum)
            count += h[curr_sum - targetSum]  #2) path start somewhere middle
            
            h[curr_sum] += 1  #count this prefix sum as a path
            
            dfs(root.left, curr_sum)
            dfs(root.right, curr_sum)
            
            h[curr_sum] -= 1 #backtracking
            
        dfs(root, 0)
        return count

 #2)recursion
 # Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def find_paths(self, root, target):
        if root:
            return int(root.val == target) + self.find_paths(root.left, target-root.val) + self.find_paths(root.right, target-root.val)
        return 0

    def pathSum(self, root: TreeNode, targetSum: int) -> int:
        if root:
            return self.find_paths(root, targetSum) + self.pathSum(root.left, targetSum) + self.pathSum(root.right, targetSum)
        return 0  


### 662. Maximum Width of Binary Tree ###
#1)dfs
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def widthOfBinaryTree(self, root: TreeNode) -> int:
        first_col_index_table = {}
        max_width = 0
        
        def dfs(root, depth, col_index):
            nonlocal max_width
            if not root:
                return
            if depth not in first_col_index_table:
                first_col_index_table[depth] = col_index
              
            max_width = max(max_width, col_index - first_col_index_table[depth] + 1)

            dfs(root.left, depth + 1, 2 * col_index)
            dfs(root.right, depth + 1, 2 * col_index + 1)
            
        dfs(root, 0, 0)
        
        return max_width
            
#      0               1
#  0,      1       2,      3
#0,  1,  2,  3,  4,  5,  6,  7        

#2)stack: BFS
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def widthOfBinaryTree(self, root: TreeNode) -> int:
        if not root:
            return 0
        first_col_index_table = {}
        max_width = 0
        stack = [(root, 0, 0)]
        while stack:
            root, depth, col_index = stack.pop(0)  #important order
            
            if depth not in first_col_index_table:
                first_col_index_table[depth] = col_index
            max_width = max(max_width, col_index - first_col_index_table[depth] + 1)
            if root.left:
                stack.append((root.left, depth+1, 2*col_index))
            if root.right:
                stack.append((root.right, depth+1, 2*col_index+1))
        
        return max_width


### 199. Binary Tree Right Side View ###
#1)dfs
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rightSideView(self, root: TreeNode) -> List[int]:
        depth_vist_table = set()
        result = []
        def dfs(root, depth): #preorder dfs
            if not root:
                return
            if depth not in depth_vist_table:
                result.append(root.val) 
                ldepth_vist_table.add(depth)
            dfs(root.right, depth + 1) #from right to left
            dfs(root.left, depth + 1)
            
        dfs(root, 0)
        
        return result

#2)stack right-left bfs
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rightSideView(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        depth_vist_table = set()
        result = []
        stack = [(root,0)]
        while stack:
            root, depth = stack.pop(0)
            if depth not in depth_vist_table:
                result.append(root.val) 
                depth_vist_table.add(depth)
            if root.right:#from right to left
                stack.append((root.right, depth+1))
            if root.left:
                stack.append((root.left, depth+1))
        
        return result


### 116. Populating Next Right Pointers in Each Node ###
#1)dfs
# Definition for a Node.
#class Node:
#    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
#        self.val = val
#        self.left = left
#        self.right = right
#        self.next = next
class Solution:
    def connect(self, root: 'Node') -> 'Node':
        previousNode = {}
    
    def dfs(root, depth):
            
            if root is None:
                return
            if depth not in previousNode:
                previousNode[depth] = root
            else:
                previousNode[depth].next = root
                previousNode[depth] = root
            
            dfs(root.left, depth + 1)
            dfs(root.right, depth + 1)
        
        curr = root
        dfs(curr, 0)
        
        return root

#1)stack bfs
# Definition for a Node.
#class Node:
#    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
#        self.val = val
#        self.left = left
#        self.right = right
#        self.next = next

class Solution:
    def connect(self, root: 'Node') -> 'Node':
        if not root:
            return None
        previousNode = {}
        curr = root
        stack = [(root, 0)]
        
        while stack:
            root, depth = stack.pop(0)
            if depth not in previousNode:
                previousNode[depth] = root
            else:
                previousNode[depth].next = root
                previousNode[depth] = root
            if root.left:
                stack.append((root.left, depth+1))
                stack.append((root.right, depth+1))
                
        return curr


### 515. Find Largest Value in Each Tree Row ###
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def largestValues(self, root: Optional[TreeNode]) -> List[int]:
        result = {}
        def dfs(root, depth):
            if not root:
                return
            if depth not in result:
                result[depth] = root.val
            else:
                result[depth] = max(root.val, result[depth])
            
            dfs(root.left, depth+1)
            dfs(root.right, depth+1)
        
        dfs(root, 0)    
        return list(result.values())