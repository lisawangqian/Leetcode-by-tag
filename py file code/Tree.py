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
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        def solve(root):
            if not root:
                return 0
            
            if not root.left and not root.right:
                return 1
            
            l = solve(root.left)
            r = solve(root.right)
            
            return 1 + max(l, r)
        
        return solve(root)
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
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        def height(root):
            if not root:
                return 0
            if not root.left and not root.right:
                return 1
            l = height(root.left)
            r = height(root.right)
            return 1 + max(l, r)
        
        def solve(root):
            if not root:
                return True
            if not root.left and not root.right:
                return True
            l = solve(root.left)
            r = solve(root.right)
            return (abs(height(root.left) - height(root.right)) <= 1) and l and r
        
        return solve(root)
            
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
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        result = []
        def solve(root, path):
            if not root:
                return
            if not root.left and not root.right:
                path.append(str(root.val))
                result.append('->'.join(path))
            
            solve(root.left, path + [str(root.val)])
            solve(root.right, path + [str(root.val)])
            
        solve(root, [])
        
        return result
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
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:
        
        def solve(root):
            if not root:
                return 0
            if not root.left and not root.right:
                return 1
            
            l = solve(root.left)
            r = solve(root.right)
            
            if not l:
                return r + 1
            
            if not r:
                return l + 1
            
            return 1 + min(r, l)
        
        return solve(root)
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
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        
        def solve(root, target):
            if not root:
                return False
            
            if not root.left and not root.right:
                return root.val == target
            
            l = solve(root.left, target-root.val)
            
            r = solve(root.right, target-root.val)

            return l or r
        
        
        return solve(root, targetSum)
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
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        def solve(root1, root2):
            if not root1 and not root2:
                return True
            elif not root1:
                return False
            elif not root2:
                return False
            
            c1 = solve(root1.left, root2.right)
            c2 = solve(root1.right, root2.left)
            
            return (root1.val == root2.val) and c1 and c2
        
        return solve(root, root)
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
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        
        def solve(root1, root2):
            if not root1 and not root2:
                return True
            if not root1 or not root2:
                return False
            l = solve(root1.left, root2.left)
            r = solve(root1.right, root2.right)
            return (root1.val == root2.val) and l and r
        
        def dfs(root):
            
            if not root:
                if not subRoot:
                    return True
                else:
                    return False
            return solve(root, subRoot) or dfs(root.left) or dfs(root.right)
        
        
        return dfs(root)
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        
        def solve(root1, root2):
            if not root1 and not root2:
                return True
            elif not root1:
                return False
            elif not root2:
                return False
            
            c1 = solve(root1.left, root2.left)
            c2 = solve(root1.right, root2.right)
            
            return (root1.val == root2.val) and c1 and c2
        
        return solve(p, q)
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

### 449. Serialize and Deserialize BST
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#297 method
class Codec:

    def serialize(self, root: Optional[TreeNode]) -> str:
        """Encodes a tree to a single string.
        """
        container = []
        def rserialize(root):
            if not root:
                return 
            else:
                container.append(str(root.val))
                rserialize(root.left)
                rserialize(root.right)
        rserialize(root)
        return ' '.join(container)

    def deserialize(self, data: str) -> Optional[TreeNode]:
        """Decodes your encoded data to tree.
        """
        value_list = [int(i) for i in data.split(' ') if data]
        def rdeseiralize(value_list, lower, upper):
            
            if not value_list or value_list[0] < lower or value_list[0] > upper:
                return 
            
            val =  value_list.pop(0)
            root = TreeNode(val)
            root.left = rdeseiralize(value_list, lower, val)
            root.right = rdeseiralize(value_list, val, upper)
            return root
        
        return rdeseiralize(value_list,  float('-inf'), float('inf'))
        

# Your Codec object will be instantiated and called as such:
# Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# tree = ser.serialize(root)
# ans = deser.deserialize(tree)
# return ans
#1) 106 method 
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root: TreeNode) -> str:  #binary search tree
        """Encodes a tree to a single string.
        """
        def dfs(root): #postorder
            return dfs(root.left) + dfs(root.right) +  [str(root.val)] if root else []
        return ' '.join(dfs(root))
        

    def deserialize(self, data: str) -> TreeNode:
        """Decodes your encoded data to tree.
        """
        #Construct Binary Tree from Inorder and Postorder Traversal
        def array_to_tree(left, right):  #inorder array positions
            if left > right:
                return None
            
            root_value = postorder.pop()
            root = TreeNode(root_value)
            root.right = array_to_tree(inorder_index_map[root_value] + 1, right)
            root.left = array_to_tree(left, inorder_index_map[root_value] - 1)  
            return root
        
        postorder = [int(x) for x in data.split(' ') if x]
       
        inorder_index_map = {}
        for index, value in enumerate(sorted(postorder)):  #bst property
            inorder_index_map[value] = index
            
        return array_to_tree(0, len(postorder)-1)
        
# Your Codec object will be instantiated and called as such:
# Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# tree = ser.serialize(root)
# ans = deser.deserialize(tree)
# return ans

#2)
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root: TreeNode) -> str:  #binary search tree
        """Encodes a tree to a single string.
        """
        def dfs(root): #postorder
            return dfs(root.left) + dfs(root.right) +  [str(root.val)] if root else []
        return ' '.join(dfs(root))
        

    def deserialize(self, data: str) -> TreeNode:
        """Decodes your encoded data to tree.
        """
        #Construct Binary Tree from Inorder and Postorder Traversal
        def array_to_tree(left, right):  #inorder array positions
            if not postorder or postorder[-1] < left or postorder[-1] > right:
                return None
            
            root_value = postorder.pop()
            root = TreeNode(root_value)  #every node value from this array
            root.right = array_to_tree(root_value, right) #BST construction
            root.left = array_to_tree(left, root_value)  
            return root
        
        postorder = [int(x) for x in data.split(' ') if x]
       
        return array_to_tree(float('-inf'), float('inf'))
        
# Your Codec object will be instantiated and called as such:
# Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# tree = ser.serialize(root)
# ans = deser.deserialize(tree)
# return ans





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
        
        if not root or root == p or root == q:
            return root
        
        l = self.lowestCommonAncestor(root.left, p, q)
        
        r = self.lowestCommonAncestor(root.right, p, q)
        
        if not l or not r: #if only one valid node returns, which means p, q are in the same subtree, 
                           #return that valid node as their LCA
            if l:
                return l
            else:
                return r
            
        return root #if both returns a valid node which means p, q are in different subtrees, 
                    #then root will be their LCA.
        
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


### 1650. Lowest Common Ancestor of a Binary Tree III ###
# Definition for a Node.
#class Node:
#    def __init__(self, val):
#        self.val = val
#        self.left = None
#       self.right = None
#       self.parent = None

class Solution:
    def lowestCommonAncestor(self, p: 'Node', q: 'Node') -> 'Node':
        pval_set = set()
        while p:
            pval_set.add(p)
            p = p.parent
        
        while q and q not in pval_set:
            q = q.parent
        
        
        return q if q else p
        
# Definition for a Node.
#class Node:
#    def __init__(self, val):
#        self.val = val
#        self.left = None
#       self.right = None
#       self.parent = None

class Solution:
    def lowestCommonAncestor(self, p: 'Node', q: 'Node') -> 'Node':
        #no root available, backward tranverse
        pVals = set()
        def traverse_up(root):
            if not root or root in pVals:
                return root
            pVals.add(root)
            return traverse_up(root.parent)
        a = traverse_up(p) #add p's parent... to set
        b =  traverse_up(q) #find common parent
        
        
        return b if b else a



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
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        result = []
        def solve(root, path):
            if not root:
                return
            if not root.left and not root.right:
                path += str(root.val)
                result.append(int(path))
            
            solve(root.left, path + str(root.val))
            solve(root.right, path + str(root.val))
            
        solve(root, '')
        
        return sum(result)
            
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
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        result = []
        
        def solve(root, target, path):
            if not root:
                return
            
            if not root.left and not root.right:
                if root.val == target:
                    path.append(root.val)
                    result.append(list(path))
                else:
                    return
            
            solve(root.left, target-root.val, path + [root.val])
            solve(root.right, target-root.val, path + [root.val])
            
        solve(root, targetSum, [])
        
        return result
            
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
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pathSum(self, root: TreeNode, targetSum: int) -> int:
        
        def solve(root, target):
            if not root:
                return 0
            
            l = solve(root.left, target-root.val)
            r = solve(root.right, target-root.val)
            
            return int(root.val == target) + l + r
        
        if not root:
            return 0
        
        return solve(root, targetSum) + self.pathSum(root.left, targetSum) + self.pathSum(root.right, targetSum)
            
          
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


### 1448. Count Good Nodes in Binary Tree ###
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def goodNodes(self, root: TreeNode) -> int:
        good = 0
        def dfs(root, pathmax):
            nonlocal good
            if not root:
                return
            if root.val >= pathmax:
                good+=1
                pathmax = root.val
          
            dfs(root.left, pathmax)
            dfs(root.right, pathmax)
            
            
        dfs(root, float(-inf))
        return good


##3 314. Binary Tree Vertical Order Traversal ###
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def verticalOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        
        column_table = defaultdict(list)
        min_col = max_col = 0
        
        def dfs(root, row, col):
            if not root:
                return
            nonlocal min_col, max_col
            column_table[col].append((row, root.val))
            min_col = min(min_col, col)
            max_col = max(max_col, col)
            dfs(root.left, row + 1, col - 1)
            dfs(root.right, row + 1, col + 1)
            
        dfs(root, 0, 0)
        result = []
        for col in range(min_col, max_col+1):
            column_table[col].sort(key = lambda x:x[0])
            col_vals = [val for row, val in column_table[col]]
            result.append(col_vals)
            
        return result


### 545. Boundary of Binary Tree ###
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def boundaryOfBinaryTree(self, root: Optional[TreeNode]) -> List[int]:
        def dfs_leftmost(node): #preorder modified
            if not node or (not node.left and not node.right): #None or leaf
                return
            boundary.append(node.val)
            if node.left:
                dfs_leftmost(node.left)
            else:  #need to be else - only no left to this route
                dfs_leftmost(node.right)

        def dfs_leaves(node):  #inorder
            if not node:
                return
            dfs_leaves(node.left)
            if node != root and not node.left and not node.right: #leaft
                boundary.append(node.val)
            dfs_leaves(node.right)

        def dfs_rightmost(node): #postorder modified
            if not node or (not node.left and not node.right):
                return
            if node.right:
                dfs_rightmost(node.right)
            else: #need to be else - only no right to this route
                dfs_rightmost(node.left)
            boundary.append(node.val)

        if not root:
            return []
        boundary = [root.val]
        dfs_leftmost(root.left)
        dfs_leaves(root)
        dfs_rightmost(root.right)
        return boundary


### 1110. Delete Nodes And Return Forest ###
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def delNodes(self, root: TreeNode, to_delete: List[int]) -> List[TreeNode]:
        
        to_delete = set(to_delete)
        result = []

        def dfs(root, is_root):
            if not root: 
                return None
            root_deleted = root.val in to_delete
            if is_root and not root_deleted:
                result.append(root)
            #if current node deleted, child become root
            root.left = dfs(root.left, root_deleted) 
            root.right = dfs(root.right, root_deleted)
            return None if root_deleted else root  #disconnected if deleted
        
        dfs(root, True)
        return result


### 366. Find Leaves of Binary Tree ###
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findLeaves(self, root: Optional[TreeNode]) -> List[List[int]]:
        result = defaultdict(list)
        def dfs(root, layer): #postorder, left, right, root, None is layer0
            if not root:
                return layer  #0 also correct
            left = dfs(root.left, layer)
            right = dfs(root.right, layer)
            #keep the layer be the maximum level of left and right
            layer = max(left, right)
            result[layer].append(root.val)
            
            return layer + 1  #parent layer
        
        dfs(root, 0)
        return result.values()


### 337. House Robber III ###
#1) recursion
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rob(self, root: TreeNode) -> int:
        def dfs(root): #rob root; not rob root
            if not root:
                return 0, 0
            left = dfs(root.left)
            right = dfs(root.right)
            rob = root.val + left[1] + right[1] #can not rob left or right
            not_rob = max(left) + max(right) # can rob left or right but want max
            return (rob, not_rob)
            
        return max(dfs(root))

#2) recursion with memoization
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rob(self, root: TreeNode) -> int:
        rob_p_map = {}
        not_rob_p_map = {}
        def dfs(root, rob_parent): 
            if not root:
                return 0
            if rob_parent:
                if root in rob_p_map:
                    return rob_p_map[root]
                else:
                    left = dfs(root.left, False)
                    right = dfs(root.right, False)
                    rob_p_map[root] = left + right
                    return rob_p_map[root]
            else:
                if root in not_rob_p_map:
                    return not_rob_p_map[root]
                else:
                    rob = root.val + dfs(root.left, True) + dfs(root.right, True) 
                    not_rob = dfs(root.left, False) + dfs(root.right, False)
                    res = max(rob, not_rob)
                    not_rob_p_map[root] = res
                    return res
            
        return dfs(root, False)


### 894. All Possible Full Binary Trees ###
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def allPossibleFBT(self, n: int) -> List[Optional[TreeNode]]:
        remember =  {0: [], 1: [TreeNode(0)]}
        def dfs(N):
            if N not in remember:
                result = []
                for x in range(N):
                    y = N-1-x #left x; right y; root 1; x+y+1 = N
                    for left in dfs(x):  #left has x nodes
                        for right in dfs(y):  #right has y nodes
                            bns = TreeNode(0)  #root, full bnt so both left and right
                            bns.left = left
                            bns.right = right
                            result.append(bns)
                remember[N] = result
            return remember[N]
               
        return dfs(n)


### 427. Construct Quad Tree ###

# Definition for a QuadTree node.
#class Node:
#    def __init__(self, val, isLeaf, topLeft, topRight, bottomLeft, bottomRight):
#        self.val = val
#        self.isLeaf = isLeaf
#        self.topLeft = topLeft
#        self.topRight = topRight
#        self.bottomLeft = bottomLeft
#        self.bottomRight = bottomRight

class Solution:
    def isLeaf(self, grid):
        return all(grid[i][j] == grid[0][0] for i in range(len(grid)) for j in range(len(grid[i])))
        
    def construct(self, grid: List[List[int]]) -> 'Node':
        if not grid: 
            return None
        
        root = Node(True, True, None, None, None, None)
        if self.isLeaf(grid):  #isLeaf
            root.val = (grid[0][0] == 1)
            return root
            
        #not leaf
        root.isLeaf = False
        size = len(grid)
        root.topLeft = self.construct([row[:size//2] for row in grid[:size//2]])
        root.topRight = self.construct([row[size//2:] for row in grid[:size//2]])
        root.bottomLeft = self.construct([row[:size//2] for row in grid[size//2:]])
        root.bottomRight = self.construct([row[size//2:] for row in grid[size//2:]])
        return root


### 1120. Maximum Average Subtree ###
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maximumAverageSubtree(self, root: Optional[TreeNode]) -> float:
        if not root:
            return 0
        
        def dfs(root): #postorder- sum of nodes/number of nodes
            nonlocal avg
            if not root:
                return (0, 0)
            
            sum_left, node_cnt_left = dfs(root.left)
            sum_right, node_cnt_right = dfs(root.right)
            
            total_sum = root.val + sum_left + sum_right
            total_cnt_node = node_cnt_left + node_cnt_right + 1
            
            avg = max(avg, total_sum/total_cnt_node)
            
            return (total_sum, total_cnt_node)
        
        avg = float("-inf")
        dfs(root)
        return avg


### 958. Check Completeness of a Binary Tree ###
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isCompleteTree(self, root: Optional[TreeNode]) -> bool:
        total_node = 0
        max_coord = 0
        def dfs(root, coord): #max coordinate
            nonlocal total_node
            
            if not root:
                return 0
            left = dfs(root.left, 2*coord)
            right = dfs(root.right, 2*coord+1)
            total_node = total_node + 1
            max_coord = max(coord, left, right)
            return max_coord
        
        if not root:
            return True
        max_coord = dfs(root, 1)
        return total_node == max_coord


### 1522. Diameter of N-Ary Tree ###

# Definition for a Node.
#class Node:
#    def __init__(self, val=None, children=None):
#        self.val = val
#        self.children = children if children is not None else []

class Solution:
    def diameter(self, root: 'Node') -> int:
        """
        :type root: 'Node'
        :rtype: int
        """
        def dfs(root):
            nonlocal diameter
            if len(root.children) == 0:
                return 0
            
            max_height_1, max_height_2 = 0, 0  #current root to child path top two heights
            for child in root.children:
                parent_height = dfs(child) + 1
                if parent_height > max_height_1:
                    max_height_1, max_height_2 = parent_height, max_height_1
                elif parent_height > max_height_2:
                    max_height_2 = parent_height
            
            diameter = max(diameter, max_height_1 + max_height_2)
            
            return max_height_1  #height is the longest path from root
        
        diameter = 0
        dfs(root)
        return diameter        


### 652. Find Duplicate Subtrees ###
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findDuplicateSubtrees(self, root: Optional[TreeNode]) -> List[Optional[TreeNode]]:
        ## Post order traversal
        def dfs(root):
            nonlocal result, serial_cnt
            if not root:
                return ""
            serial = dfs(root.left) + "," + dfs(root.right)  +  '.' + str(root.val) 
            if serial not in serial_cnt:
                serial_cnt[serial] = 1
            else:
                serial_cnt[serial] += 1
            
            if serial_cnt[serial] == 2:
                result.append(root)
            
            return serial
        
        result = []
        serial_cnt = {}
        dfs(root)
        return result


### 979. Distribute Coins in Binary Tree ###
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def distributeCoins(self, root: Optional[TreeNode]) -> int:
        def dfs(root): #postorder
            nonlocal ans
            if not root:
                return 0
            left = dfs(root.left)
            right = dfs(root.right)
            ans+= abs(left) +  abs(right)
            return root.val + left + right - 1
        ans = 0
        dfs(root)
        return ans


### 1302. Deepest Leaves Sum ###
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def deepestLeavesSum(self, root: Optional[TreeNode]) -> int:
        result = []
        max_level = -1
        def preorder(root, level):
            nonlocal max_level, result
            if not root:
                return
            
            if level > max_level:
                result = [root.val]
                max_level = level
            elif level == max_level:
                result.append(root.val)
                
            preorder(root.left, level+1)
            preorder(root.right, level+1)
            
        preorder(root, 0)
        
        return sum(result)

#1)dfs
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def deepestLeavesSum(self, root: Optional[TreeNode]) -> int:
        
        def get_level(root): #get level
            if not root:
                return -1
            return max(get_level(root.left), get_level(root.right)) + 1 
        
        def dfs(root, level):
            if not root: #not deepest leaf all 0
                return 0
            if not level: #level =0, deepest leaf return value
                return root.val
            return dfs(root.left, level-1) + dfs(root.right, level-1) 
        
        level = get_level(root)  #start from 0
        return dfs(root, level)
        
#2)stack
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def deepestLeavesSum(self, root: Optional[TreeNode]) -> int:
        deepest_sum = {}
        deepest_depth = 0
        stack = [(root, 0)]
        while stack:
            node, curr_depth = stack.pop()
            if not node.left and not node.right: #leaf
                if deepest_depth < curr_depth:
                    deepest_depth = curr_depth  #started new sum
                if curr_depth not in deepest_sum:
                    deepest_sum[curr_depth] = node.val
                else:
                    deepest_sum[curr_depth] += node.val
            else:
                if node.left:
                    stack.append((node.left, curr_depth + 1))
                if node.right:
                    stack.append((node.right, curr_depth + 1))
        
        return deepest_sum[deepest]


### 1026. Maximum Difference Between Node and Ancestor
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxAncestorDiff(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        result = 0
        def dfs(root, curr_max, curr_min):
            nonlocal result
            if not root:
                return
            result = max(result, abs(curr_max - root.val), abs(curr_min - root.val))
            curr_max = max(curr_max, root.val)
            curr_min = min(curr_min, root.val)
            dfs(root.left, curr_max, curr_min)
            dfs(root.right, curr_max, curr_min)
            
        dfs(root, root.val, root.val)
        
        return result


### 513. Find Bottom Left Tree Value
#1)dfs
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findBottomLeftValue(self, root: Optional[TreeNode]) -> int:
        ans = 0
        count = -1
        def dfs(root, level): #in-order
            nonlocal ans, count
            if not root:
                return 0
            dfs(root.left, level + 1)
            if level > count:
                count = level  #max level
                ans = root.val
            dfs(root.right, level + 1)
        
        dfs(root, 0)
        return ans
#2)stack
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findBottomLeftValue(self, root: Optional[TreeNode]) -> int:
        if not root:
            return
        ans = 0
        max_depth = -1
        stack = [(root, 0)]
        while stack:
            curr_node, level = stack.pop()
            if level > max_depth:
                max_depth = level
                ans = curr_node.val
            if curr_node.right:
                stack.append((curr_node.right, level+1))
            if curr_node.left:
                stack.append((curr_node.left, level+1))
        return ans

### 250. Count Univalue Subtrees 
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def countUnivalSubtrees(self, root: Optional[TreeNode]) -> int:
        count = 0
        def dfs(root, val):
            nonlocal count
            if not root:
                return True
            if all([dfs(root.left, root.val), dfs(root.right, root.val)]):
                count += 1
            else:
                return False
                
            return root.val == val
        
        dfs(root, None)
        
        return count
        

### 156. Binary Tree Upside Down  
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def upsideDownBinaryTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
            
        if not root or not root.left:
            return root
        
        newRoot = self.upsideDownBinaryTree(root.left)  #1)The original left child becomes the new root.
        #new right child; new left child
        root.left.right, root.left.left = root, root.right #2) & 3)
        root.left = root.right = None        
        
        return newRoot #new root so return it


### 98. Validate Binary Search Tree
#1)dfs
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        def dfs(root, low, high):
            if not root:
                return True

            if root.val <= low or root.val >= high:
                return False
            
            return dfs(root.left, low, root.val) and dfs(root.right, root.val, high)
        
        return dfs(root, -math.inf, math.inf)

#2)dfs
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        def dfs(root): #inorder 
            if not root:
                return True
            if not dfs(root.left):
                return False
            if root.val <= self.prev:
                return False
            self.prev = root.val
            if not dfs(root.right):
                return False
            
            return True
        
        self.prev = -math.inf
        
        return dfs(root)

#3)stack
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        if not root:
            return True
        stack = [(root, -math.inf, math.inf)]
        while stack:
            node, lower, upper = stack.pop()
            curr_val = node.val
            if curr_val <= lower or curr_val >= upper:
                return False
            if node.left:
                stack.append((node.left, lower, curr_val))
            if node.right:
                stack.append((node.right, curr_val, upper))
        return True  


 ###173. Binary Search Tree Iterator
 # Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class BSTIterator:

    def __init__(self, root: TreeNode):
        self.nodes_in_order = []
        self.index = -1
        self._dfs(root)
        
    def _dfs(self, root):
        if not root:
            return
        self._dfs(root.left)
        self.nodes_in_order.append(root.val)
        self._dfs(root.right)
        
    def next(self) -> int:
        self.index +=1
        return self.nodes_in_order[self.index]
            

    def hasNext(self) -> bool:
        return self.index + 1 < len(self.nodes_in_order)


# Your BSTIterator object will be instantiated and called as such:
# obj = BSTIterator(root)
# param_1 = obj.next()
# param_2 = obj.hasNext()


### 96. Unique Binary Search Trees
class Solution:
    def numTrees(self, n: int) -> int:
        
        G = [0] * (n+1)
        G[0], G[1] = 1, 1

        for i in range(2, n+1):
            for j in range(1, i+1):
                G[i] += G[j-1] * G[i-j]  #i-j + j-1 = i-1; 1 is root

        return G[n]


### 95. Unique Binary Search Trees II
#1)dfs
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def generateTrees(self, n: int) -> List[TreeNode]:
         
        def dfs(start, end):
            if start > end:
                return [None]
            all_trees = []
            for i in range(start, end+1): #i is root
                left_trees = dfs(start, i-1) #all left subtree 
                right_trees = dfs(i+1, end) #all right subtree
                
                for l in left_trees:
                    for r in right_trees:
                        current_tree = TreeNode(i) #root
                        current_tree.left = l
                        current_tree.right = r
                        all_trees.append(current_tree)
                        
            return all_trees
                        
        return dfs(1, n) if n > 0 else []


### 230. Kth Smallest Element in a BST
#1)recursion
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        
        def dfs(root): #inorder
            nonlocal result
            if not root:
                return 
            dfs(root.left)
            self.count+=1
            if self.count == k:
                result = root.val
            dfs(root.right)
            
        self.count = 0
        result = None
        dfs(root)
        
        return result
#2)dfs
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        def dfs(root):
            if not root:
                return []
            return dfs(root.left) + [root.val] + dfs(root.right)
        return dfs(root)[k-1]
#3)stack
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        stack = []
        while True:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            k-=1
            if k == 0:
                return root.val
            root = root.right       
       

### 99. Recover Binary Search Tree
#1)recursion
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def recoverTree(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        def dfs(root):
            if not root:
                return []
            return dfs(root.left) + [root.val] + dfs(root.right)
        
        def find_two_swapped(nums):  #eg [3, 2, 1]; [1, 3, 2, 4]
            n = len(nums)
            x = y = None
            for i in range(n-1):
                if nums[i+1] < nums[i]:
                    y = nums[i+1]
                    if x is None:
                        x = nums[i]
                    else:
                        break
            return x, y
        
        def dfs_preorder(root, count):
            if root:
                if root.val == x or root.val == y:
                    if root.val == x:
                        root.val = y
                    else:
                        root.val = x
                    count-=1
                    if count == 0:
                        return 
                dfs_preorder(root.left, count)
                dfs_preorder(root.right, count)
                
        nums = dfs(root)
        x, y = find_two_swapped(nums)
        dfs_preorder(root, 2)

#2)dfs
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def recoverTree(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        
        
        def find_two_swapped(root):  #eg [3, 2, 1]; [1, 3, 2, 4]
            nonlocal x, y, prev
            if not root:
                return
            find_two_swapped(root.left)
            if prev and root.val < prev.val:
                y = root
                if x is None:
                    x = prev
                else:
                    return
            prev = root
            find_two_swapped(root.right)
        
        x = y = prev = None
                
        find_two_swapped(root)
        
        x.val, y.val = y.val, x.val
        

##1382. Balance a Binary Search Tree  
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def balanceBST(self, root: TreeNode) -> TreeNode:
        nodes = []
        
        def dfs(root): #inorder to maintain order
            if not root:
                return
            dfs(root.left)
            nodes.append(root)
            dfs(root.right)
        
        def build_balanced_tree(left, right):
            if left>right:
                return None
            mid = left + (right-left)//2
            root = nodes[mid]
            root.left = build_balanced_tree(left, mid-1)
            root.right = build_balanced_tree(mid+1, right)
            return root
        
        dfs(root) #get nodes in order
        return build_balanced_tree(0, len(nodes)-1)


### 333. Largest BST Subtree
#1)recursion
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    
    def largestBSTSubtree(self, root: TreeNode) -> int:
        
        def is_valid_bst(root):
            if not root:
                return True
            
            left_max = find_max(root.left)
            if left_max >= root.val:
                return False
            
            right_min = find_min(root.right)
            if right_min <= root.val:
                return False
            
            return is_valid_bst(root.left) and is_valid_bst(root.right)
        
        def find_max(root):
            if not root:
                return float('-inf')
            return max(root.val, find_max(root.left), find_max(root.right))
        
        def find_min(root):
            if not root:
                return float('inf')
            return min(root.val, find_min(root.left), find_min(root.right))
        
        def count_nodes(root):
            if not root:
                return 0
            return 1 + count_nodes(root.left) + count_nodes(root.right)
        
        if not root:
            return 0
        
        if is_valid_bst(root):
            return count_nodes(root)
        
        return max(self.largestBSTSubtree(root.left), self.largestBSTSubtree(root.right))

#2)recursion2
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    
    def largestBSTSubtree(self, root: TreeNode) -> int:
        
        if root is None:
            return 0
    # lres and rres tells when left and right subtree are valid bst or not
    # lcount, rcount tell total nodes in left and right subtree if left/right is valid
    # if it is not a valid bst just return 0 so parent knows that it can't depend on its children
        def dfs(root):
            nonlocal result
            if not root:
                return True, 0, float('inf'), float('-inf')  # so that I can do this: lmax < root.val < rmin
            l_valid, l_count, lmin, lmax = dfs(root.left)
            r_valid, r_count, rmin, rmax = dfs(root.right)
            if l_valid and r_valid and lmax < root.val < rmin:#still valid bst with root
                valid_count = l_count + r_count + 1
                result = max(result, valid_count)
                return True, valid_count, min(lmin, root.val), max(rmax, root.val) #due to root None case init
            else:
                return False, 0, min(lmin, rmin, root.val), max(lmax, rmax, root.val)
        
        result = 0
        dfs(root)
        return result


### 538. Convert BST to Greater Tree

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def convertBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        def dfs(root): #reverse in-order  reserve ordered
            nonlocal total
            if root:
                dfs(root.right)
                total+=root.val
                root.val = total
                dfs(root.left)
                
        total = 0
        dfs(root)
        return root

#2)stack
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def convertBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        total = 0
        stack = []
        node = root
        
        while stack or node:
            while node:
                stack.append(node)
                node = node.right
            node = stack.pop()
            total += node.val
            node.val = total
            node = node.left
            
        return root


### 285. Inorder Successor in BST
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def inorderSuccessor(self, root: 'TreeNode', p: 'TreeNode') -> 'TreeNode':
        
        successor = None
        
        while root: 
            if p.val >= root.val:
                root = root.right
            else:
                successor = root
                root = root.left
                
        return successor


### 510. Inorder Successor in BST II
"""
# Definition for a Node.
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None
"""

class Solution:
    def inorderSuccessor(self, node: 'Node') -> 'Node':
        #left, root, right  #bst inorder
        #1) node has a right child, then its successor is its leftmost leaf node in the using this right child as root subtree
        #2) node has no right child, then its successor must be the nearest ancestor whose left is the subtree.
        #3) no successor if none of the above two cases hold
        if node.right:
            node = node.right #tree to this right child and now find the leftmost
            while node.left: 
                node = node.left
            return node

        while node.parent: 
            if node.parent.left == node: #if node is its parent left child, then the parent is the result
                return node.parent
            node = node.parent
        return None


### 426. Convert Binary Search Tree to Sorted Doubly Linked List
"""
# Definition for a Node.
class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
"""

class Solution:
    def treeToDoublyList(self, root: 'Node') -> 'Node':
        
        def dfs(root): #inorder
            nonlocal last, first
            if not root:
                return
            dfs(root.left)
            if last:  #move along (record previous node on the path)
                last.right = root
                root.left = last
            else:
                first = root  #record as the first node
            last  = root
            dfs(root.right)
        
        if not root:
            return None
        first, last = None, None
        dfs(root)
        last.right = first
        first.left = last
        return first
        


### 429. N-ary Tree Level Order Traversal
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""
class Solution:
    def levelOrder(self, root: 'Node') -> List[List[int]]:
        result = []
        
        def preorder(root, level):
            if not root:
                return
            if len(result)-1 < level:
                result.append([root.val])
            else:
                result[level].append(root.val)
            
            for each in root.children:
                preorder(each, level + 1)
            
        preorder(root, 0)
        
        return result



### 987. Vertical Order Traversal of a Binary Tree
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        
        column_table = defaultdict(list)
        min_col = max_col = 0
        
        def dfs(root, row, col):
            if not root:
                return
            nonlocal min_col, max_col
            column_table[col].append((row, root.val))
            min_col = min(min_col, col)
            max_col = max(max_col, col)
            dfs(root.left, row + 1, col - 1)
            dfs(root.right, row + 1, col + 1)
            
        dfs(root, 0, 0)
        result = []
        for col in range(min_col, max_col+1):
            column_table[col].sort(key = lambda x:(x[0], x[1]))
            result.append([val for row, val in column_table[col]])
            
        return result



### 951. Flip Equivalent Binary Trees
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def flipEquiv(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        
        def solve(root1, root2):
            if not root1 and not root2:
                return True
            if not root1 or not root2:
                return False
            
            l1 = solve(root1.left, root2.left)
            l2 = solve(root1.left, root2.right)
            r1 = solve(root1.right, root2.right)
            r2 = solve(root1.right, root2.left)

            return (root1.val == root2.val) and ((l1 and r1) or (l2 and r2))
        
        return solve(root1, root2)



### 965. Univalued Binary Tree
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isUnivalTree(self, root: Optional[TreeNode]) -> bool:
        
        vals = []
        def solve(root):
            
            if not root:
                return 
            vals.append(root.val)
            solve(root.left)
            solve(root.right)
            
        solve(root) 
        
        return len(set(vals)) == 1



### 814. Binary Tree Pruning
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pruneTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        
        def checktree(root):
            if not root:
                return False
            
            l = checktree(root.left)
            r = checktree(root.right)
            
            if not l:
                root.left = None
            if not r:
                root.right = None
                
            return (root.val == 1) or l or r
        
        return root if checktree(root) else None






### 1325. Delete Leaves With a Given Value
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def removeLeafNodes(self, root: Optional[TreeNode], target: int) -> Optional[TreeNode]:
        
        def remove(root):
            if not root:
                return None
            
            root.left = remove(root.left)
            root.right = remove(root.right)
            
            return None if root.val == target and not root.left and not root.right else root
                
            
        return root if remove(root) else None
        

### 669. Trim a Binary Search Tree
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def trimBST(self, root: Optional[TreeNode], low: int, high: int) -> Optional[TreeNode]:
        
        def trimtree(root):
            if not root:
                return None
            elif root.val > high:
                root = trimtree(root.left)
            elif root.val < low:
                root = trimtree(root.right)
            else:
                root.left = trimtree(root.left)
                root.right = trimtree(root.right)
            return root
            
        return trimtree(root)
                

### 297. Serialize and Deserialize Binary Tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        container = []
        def rserialize(root):
            if not root:
                container.append('#')
                return
            else:
                container.append(str(root.val))
                rserialize(root.left)
                rserialize(root.right)
        rserialize(root)
        return ' '.join(container)
                
    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        value_list = data.split(' ')
        
        def rdeseiralize(value_list):
            if value_list[0] == '#':
                value_list.pop(0)
                return None
            
            root = TreeNode(value_list[0])
            value_list.pop(0)
            root.left = rdeseiralize(value_list)
            root.right = rdeseiralize(value_list)
            return root
        
        return rdeseiralize(value_list)
        
        
### 536. Construct Binary Tree from String
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def str2tree(self, s: str) -> Optional[TreeNode]:
        def get_number(s, index):
            is_negative = False
            
            if s[index] == '-':
                is_negative = True
                index = index + 1
                
            number = 0
            while index < len(s) and s[index].isdigit():
                number = number * 10 + int(s[index])
                index+=1
                
            return number if not is_negative else -number, index
        
        def construct_tree(s, index):
            if index == len(s):
                return None, index
            
            value, index = get_number(s, index)  #number and next scan index
            root = TreeNode(value)
            
            if index < len(s) and s[index] == '(':
                root.left, index = construct_tree(s, index+1)
            
            if root.left and index < len(s) and s[index] == '(':
                root.right, index = construct_tree(s, index+1)
                
            return root, index + 1 if index < len(s) and s[index] == ')' else index
        
        return construct_tree(s, 0)[0]
                


### 124. Binary Tree Maximum Path Sum
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        def solve(root):
            nonlocal ans
            if not root: 
                return 0
            l = max(0, solve(root.left))
            r = max(0, solve(root.right))
            ans = max(ans, l + r + root.val)  
            return max(l, r) + root.val 
        
        if not root:
            return 0
        ans = float('-inf')
        solve(root)
        return ans


### 687. Longest Univalue Path
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestUnivaluePath(self, root: Optional[TreeNode]) -> int:
        
        def solve(root):
            nonlocal ans
            if not root:
                return 0 
            l = solve(root.left)
            r = solve(root.right)
            
            pl = 0 #edge
            pr = 0 #edge
            if root.left and root.val == root.left.val:
                pl = l+1 #add one edge
                
            if root.right and root.val == root.right.val:
                pr = r + 1  #add one edge
                
            ans = max(ans, pr + pl)
            return max(pr, pl)
        
        if not root:
            return 0
        ans = 0
        solve(root)
        return ans
        


### 508. Most Frequent Subtree Sum
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findFrequentTreeSum(self, root: Optional[TreeNode]) -> List[int]:
        s = defaultdict(int)
        def solve(root):
            if not root:
                return 0
            l = solve(root.left)
            r = solve(root.right)
            total = l + r + root.val
            s[total] += 1
            return total
        
        if not root:
            return []
        
        solve(root)
        f = max(s.values())
        
        return [i for i, v in s.items() if v == f]
            
        

### 968. Binary Tree Cameras
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def minCameraCover(self, root: Optional[TreeNode]) -> int:
        # state 0: Strict ST: all nodes below this node are covered but not this node
        # state 1: Normal ST: all nodes below this node are covered including this node, but no camera here
        # state 2: Camera placed: all nodes below this node are covered including this node, camera here
        def solve(root): #state 0, state 1, state 2
            if not root:
                return 0, 0, 1 
            
            l = solve(root.left)
            r = solve(root.right)
            
            dp_s0 = l[1] + r[1]  #left and right child need to be in state 1
            dp_s1 = min(l[2] + min(r[1:]), r[2] + min(l[1:]))  #1) left child state2; right child state1/2
                                                               #2) right child state2; left child state1/2
            dp_s2 = 1 + min(l) + min(r) #camera here
            
            return dp_s0, dp_s1, dp_s2
        
        return min(solve(root)[1:])