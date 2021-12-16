### Binary Tree Traversal Basic ###

##145. Binary Tree Postorder Traversal
#1) Recursive
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

#2) Iterative
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        result = []
        if not root:
            return result
        stack = [root]
        while stack:  
            root = stack.pop()
            result.append(root.val)
            if root.left:
                stack.append(root.left)
            if root.right:
                stack.append(root.right)
                
        return result[::-1]


## 94. Binary Tree Inorder Traversal
#1) Recursive
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
#2) Iterative
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        result = []
        if not root:
            return result
        curr = root
        stack = []
        
        while curr or stack:
            while curr:
                stack.append(curr)
                curr = curr.left
            curr = stack.pop()
            result.append(curr.val)
            curr = curr.right
            
        return result


## 144. Binary Tree Preorder Traversal
#1) Recursive
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

#2) Iterative
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        result = []
        if not root:
            return result
        stack = [root]
        while stack:
            root = stack.pop()
            result.append(root.val)
            if root.right:
                stack.append(root.right)
            if root.left:
                stack.append(root.left)
                
        return result


## 589. N-ary Tree Preorder Traversal
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


## 590. N-ary Tree Postorder Traversal
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


## 545. Boundary of Binary Tree
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

        def dfs_leaves(node):  #inorder; could be anyorder
            if not node:
                return
            dfs_leaves(node.left)
            if node != root and not node.left and not node.right: #leaft
                boundary.append(node.val)
            dfs_leaves(node.right)

        def dfs_rightmost(node): #postorder modified/reversed preorder
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



### Binary Tree Traversal Preorder ###

## 429. N-ary Tree Level Order Traversal
# Definition for a Node.
# class Node:
#    def __init__(self, val=None, children=None):
#       self.val = val
#       self.children = children
class Solution:
    def levelOrder(self, root: 'Node') -> List[List[int]]:
        result = []
        
        def preorder(root, depth):
            if not root:
                return
            if len(result) <= depth:
                result.append([])
            result[depth].append(root.val)
            for each in root.children:
                preorder(each, depth + 1)
            
        preorder(root, 0)
        return result


## 102. Binary Tree Level Order Traversal
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        result = []
        def preorder(root, depth):
            if not root:
                return
            if len(result) <= depth:
                result.append([])
            result[depth].append(root.val)
            preorder(root.left, depth+1)
            preorder(root.right, depth+1)
            
        preorder(root,0)
        return result


## 103. Binary Tree Zigzag Level Order Traversal
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        result = []
        def preorder(root, depth):
            if not root:
                return
            if len(result) <= depth:
                result.append([])
            if depth%2 == 1:
                result[depth].insert(0, root.val)
            else:
                result[depth].append(root.val)
            preorder(root.left, depth+1)
            preorder(root.right, depth+1)
            
        preorder(root,0)
        return result


## 107. Binary Tree Level Order Traversal II
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrderBottom(self, root: Optional[TreeNode]) -> List[List[int]]:
        result = []
        def preorder(root, depth):
            if not root:
                return
            if len(result) <= depth:
                result.append([])
            result[depth].append(root.val)
            preorder(root.left, depth+1)
            preorder(root.right, depth+1)
            
        preorder(root,0)
        return result[::-1]


## 1302. Deepest Leaves Sum
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


## 314. Binary Tree Vertical Order Traversal
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
        
        def preorder(root, row, col):
            if not root:
                return
            nonlocal min_col, max_col
            column_table[col].append((row, root.val))
            min_col = min(min_col, col)
            max_col = max(max_col, col)
            preorder(root.left, row + 1, col - 1)
            preorder(root.right, row + 1, col + 1)
            
        preorder(root, 0, 0)
        result = []
        for col in range(min_col, max_col+1):
            column_table[col].sort(key = lambda x:x[0])
            col_vals = [val for row, val in column_table[col]]
            result.append(col_vals)
            
        return result


## 987. Vertical Order Traversal of a Binary Tree
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
        
        def preorder(root, row, col):
            if not root:
                return
            nonlocal min_col, max_col
            column_table[col].append((row, root.val))
            min_col = min(min_col, col)
            max_col = max(max_col, col)
            preorder(root.left, row + 1, col - 1)
            preorder(root.right, row + 1, col + 1)
            
        preorder(root, 0, 0)
        result = []
        for col in range(min_col, max_col+1):
            column_table[col].sort(key = lambda x:(x[0], x[1]))
            result.append([val for row, val in column_table[col]])
            
        return result


## 662. Maximum Width of Binary Tree
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


## 199. Binary Tree Right Side View
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        seens = []
        def solve(root, depth):
            if not root:
                return
            if len(seens) <= depth:
                seens.append(root.val)
            solve(root.right, depth+1)
            solve(root.left, depth+1)
        solve(root, 0)
        return seens
            

## 513. Find Bottom Left Tree Value
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findBottomLeftValue(self, root: Optional[TreeNode]) -> int:
        
        left_depth = []
        def solve(root, depth):
            if not root:
                return
            if len(left_depth) <= depth:
                left_depth.append(root.val)
                
            solve(root.left, depth + 1)
            solve(root.right, depth + 1)
            
        solve(root, 0)
        
        return left_depth[-1]
            

## 637. Average of Levels in Binary Tree
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def averageOfLevels(self, root: Optional[TreeNode]) -> List[float]:
        depths = []
        
        def solve(root, level):
            if not root:
                return
            if len(depths) <= level:
                depths.append([])
            
            depths[level].append(root.val)
            solve(root.left, level+1)
            solve(root.right, level+1)
            
        solve(root, 0)
        
        return [sum(v) / len(v) for v in depths]


## 515. Find Largest Value in Each Tree Row
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def largestValues(self, root: Optional[TreeNode]) -> List[int]:
        result = {}
        def solve(root, depth):
            if not root:
                return
            if depth not in result:
                result[depth] = root.val
            else:
                result[depth] = max(root.val, result[depth])
            
            solve(root.left, depth+1)
            solve(root.right, depth+1)
        
        solve(root, 0)    
        return list(result.values())


## 116. Populating Next Right Pointers in Each Node
# Definition for a Node.
#class Node:
#    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
#        self.val = val
#        self.left = left
#        self.right = right
#        self.next = next
class Solution:
    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        
        visited = []
        def solve(root, depth):
            if not root: return
            if len(visited) <= depth:
                visited.append(root)
            else:
                visited[depth].next = root
                visited[depth] = root
                
            solve(root.left, depth+1)
            solve(root.right, depth + 1)
            
            
        curr = root    
        solve(curr, 0)
        
        return root


## 222. Count Complete Tree Nodes
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def countNodes(self, root: Optional[TreeNode]) -> int:
        ans, last_depth = 0, 0
        def solve(root, depth):
            nonlocal ans, last_depth
            if not root:
                return
            if not root.left and not root.right:
                last_depth = max(depth, last_depth)
                if depth == last_depth:
                    ans+=1
            solve(root.left, depth+1)
            solve(root.right, depth+1)
        solve(root, 0)        
        
        for i in range(last_depth):
            ans+=2**i
        return ans 


## 993. Cousins in Binary Tree
 Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isCousins(self, root: Optional[TreeNode], x: int, y: int) -> bool:
        parents = []
        def solve(root, parent, depth):
            if not root:
                return
            if root.val in (x, y):
                parents.append((parent, depth))
            solve(root.left, root, depth + 1)
            solve(root.right, root, depth + 1)
            
        solve(root, None, 0)
        
        if len(parents) == 2 and parents[0][0] != parents[1][0] and parents[0][1] == parents[1][1]:
            return True
        return False


## 965. Univalued Binary Tree
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


## 872. Leaf-Similar Trees
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def leafSimilar(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        
        def getleave(root, leaves):
            if not root:
                return
            if not root.left and not root.right:
                leaves.append(root.val)
                
            getleave(root.left, leaves)
            getleave(root.right, leaves)
            
            
        result1 = []
        getleave(root1, result1)
        result2 = []
        getleave(root2, result2)
        
        return result1 == result2


## 671. Second Minimum Node In a Binary Tree
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findSecondMinimumValue(self, root: TreeNode) -> int:
        col = set()
        def preorder(root):  #preorder
            if not root:
                return
            col.add(root.val)
            preorder(root.left)
            preorder(root.right)
                
        preorder(root)
        if len(col) < 2:
            return -1
        return sorted(col)[1]


## 1026. Maximum Difference Between Node and Ancestor
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
        def preorder(root, curr_max, curr_min):
            nonlocal result
            if not root:
                return
            result = max(result, abs(curr_max - root.val), abs(curr_min - root.val))
            curr_max = max(curr_max, root.val)
            curr_min = min(curr_min, root.val)
            preorder(root.left, curr_max, curr_min)
            preorder(root.right, curr_max, curr_min)
            
        preorder(root, root.val, root.val)
        return result



### Template 1Root/Path Sum ###

## 104. Maximum Depth of Binary Tree
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


## 111. Minimum Depth of Binary Tree
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


## 112. Path Sum
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


## 113. Path Sum II
#1) Recursive
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
            
            if not root.left and not root.right and root.val == target:
                path.append(root.val)
                result.append(list(path))
            
            solve(root.left, target-root.val, path + [root.val])
            solve(root.right, target-root.val, path+ [root.val])
            
            
        solve(root, targetSum, [])
        
        return result

#2) Backtracking
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


## 437. Path Sum III
#1) Recursive
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
            
#2) backtracking
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


## 129. Sum Root to Leaf Numbers
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


## 257. Binary Tree Paths
#1) recursive
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

#2) backtracking
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
            path.append(str(root.val))
            if not root.left and not root.right:
                result.append('->'.join(path))
            solve(root.left, path)
            solve(root.right, path)
            path.pop()
            
        solve(root, [])
        
        return result


## 543. Diameter of Binary Tree
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


## 1522. Diameter of N-Ary Tree
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
        
        def solve(root):
            nonlocal diameter
            if not root: #length of leaf as 1
                return 0
            
            max_height_1, max_height_2 = 0, 0
            for child in root.children:
                height = solve(child)
                if height > max_height_1:
                    max_height_1, max_height_2 = height, max_height_1
                elif height > max_height_2:
                    max_height_2 = height
            
            diameter = max(diameter, max_height_1 + max_height_2) 
           
            return max(max_height_1, max_height_2) + 1 #longest node number, only one side can be used
        
        diameter = 0
        solve(root)
        return diameter


## 124. Binary Tree Maximum Path Sum
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


## 687. Longest Univalue Path
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


## 250. Count Univalue Subtrees
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def countUnivalSubtrees(self, root: Optional[TreeNode]) -> int:
        count = 0
        def solve(root, val):
            nonlocal count
            if not root:
                return True
            if all([solve(root.left, root.val), solve(root.right, root.val)]):
                count += 1
            else:
                return False
                
            return root.val == val
        
        solve(root, None)
        
        return count


## 508. Most Frequent Subtree Sum
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


## 226. Invert Binary Tree
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    
    def invertTree(self, root: TreeNode) -> TreeNode:  #reversed preorder
        
        def solve(root):
            if not root:
                return
            
            l = solve(root.left)
            r = solve(root.right)
            
            root.left = r
            root.right = l
            
            return root
        
        return solve(root)


## 404. Sum of Left Leaves
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sumOfLeftLeaves(self, root: TreeNode) -> int:
        def solve(root): #preorder (root, left, right)
            nonlocal s
            if root is None:
                return 0
            if root.left is None and root.right is None: #leaf
                return root.val
            
            left = solve(root.left)
            s+=left
            solve(root.right)
            return 0  #essential to avoid left no return(None) - s+left error
        
        s = 0
        solve(root)
        return s


## 1448. Count Good Nodes in Binary Tree
## Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def goodNodes(self, root: TreeNode) -> int:
        ans = 0
        def solve(root, max_num):
            nonlocal ans
            if not root:
                return
            
            max_num = max(max_num, root.val)
            if max_num == root.val:
                ans+=1
                
            solve(root.left, max_num)
            solve(root.right, max_num)
            
        solve(root, float('-inf'))
        
        return ans


## 1120. Maximum Average Subtree
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
        
        def solve(root): 
            nonlocal avg
            if not root:
                return (0, 0)
            
            sum_left, node_cnt_left = solve(root.left)
            sum_right, node_cnt_right = solve(root.right)
            
            total_sum = root.val + sum_left + sum_right
            total_cnt_node = node_cnt_left + node_cnt_right + 1
            
            avg = max(avg, total_sum/total_cnt_node)
            
            return (total_sum, total_cnt_node)
        
        avg = float("-inf")
        solve(root)
        return avg


## 652. Find Duplicate Subtrees
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findDuplicateSubtrees(self, root: Optional[TreeNode]) -> List[Optional[TreeNode]]:
        result = []
        serial_cnt = {}
        def solve(root):
            if not root:
                return ""
            l = solve(root.left) 
            r = solve(root.right) 
            serial = l + ',' + r + '.' + str(root.val) 
            
            if serial not in serial_cnt:
                serial_cnt[serial] = 1
            else:
                serial_cnt[serial] += 1
            
            if serial_cnt[serial] == 2:
                result.append(root)
            
            return serial
        
        solve(root)
        return result


## 958. Check Completeness of a Binary Tree
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
        def solve(root, coord): #max coordinate
            nonlocal total_node
            if not root:
                return 0
            left = solve(root.left, 2*coord)
            right = solve(root.right, 2*coord+1)
            total_node = total_node + 1
            max_coord = max(coord, left, right)
            return max_coord
        
        if not root:
            return True
        max_coord = dfs(root, 1)
        return total_node == max_coord


## 366. Find Leaves of Binary Tree
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findLeaves(self, root: Optional[TreeNode]) -> List[List[int]]:
        value_layers = defaultdict(list)
        def solve(root):
            if not root:
                return 0
            l = solve(root.left) 
            r = solve(root.right)
            layer = max(l, r)
            value_layers[layer].append(root.val)
            
            return layer + 1  #parent returned
            
        solve(root)
        return value_layers.values()


## 110. Balanced Binary Tree
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


## 863. All Nodes Distance K in Binary Tree
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        ans = []
        def solve(root):
            if not root:
                return -1  #target not in this root tree
            if root == target:
                collect(root, k)  #collect nodes distance k
                return 0
            
            l = solve(root.left)  #from left child as root node in k distance
            r = solve(root.right)  #from right child
            
            if l>=0:
                if l == k-1:
                    ans.append(root.val)
                collect(root.right, k-l-2)
                return l + 1
            
            if r>=0:
                if r == k-1:
                    ans.append(root.val)
                collect(root.left, k-r-2)
                return r + 1
            
            return -1
        
        def collect(root, d):
            if not root or d<0:
                return
            if d == 0:
                ans.append(root.val)
            collect(root.left, d-1)
            collect(root.right, d-1)
            
        solve(root)
        return ans



### Template 2Roots ###

## 100. Same Tree
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


## 101. Symmetric Tree
 Definition for a binary tree node.
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


## 951. Flip Equivalent Binary Trees
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


## 572. Subtree of Another Tree
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


## 617. Merge Two Binary Trees
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        
        def solve(root1, root2):
            if not root1:
                return root2
            if not root2:
                return root1
            
            root1.val = root1.val + root2.val
            
            root1.left = solve(root1.left, root2.left)
            root1.right = solve(root1.right, root2.right)
            
            return root1
            
        return solve(root1, root2)



### Tree Pruning ###

## 814. Binary Tree Pruning
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


## 669. Trim a Binary Search Tree
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


## 1325. Delete Leaves With a Given Value
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
        

## 1110. Delete Nodes And Return Forest
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



### Lowest Common Ancestor ###

## 236. Lowest Common Ancestor of a Binary Tree
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


## 235. Lowest Common Ancestor of a Binary Search Tree
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        
        if p.val > root.val and q.val > root.val: #on the right subtree
            return self.lowestCommonAncestor(root.right, p, q)
        elif p.val < root.val and q.val < root.val: #on the left subtree
            return self.lowestCommonAncestor(root.left, p, q)
        else:
            return root


## 1650. Lowest Common Ancestor of a Binary Tree III
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



### Construct Binary Tree ###

## 297. Serialize and Deserialize Binary Tree
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


## 536. Construct Binary Tree from String
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
                

## 449. Serialize and Deserialize BST
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


## 105. Construct Binary Tree from Preorder and Inorder Traversal
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        in_index = {v:i for i, v in enumerate(inorder)}
        
        def solve(pre_st, in_st, in_end):
            if pre_st > len(preorder)-1 or in_st > in_end:
                return None
            root = TreeNode(preorder[pre_st])
            
            idx = in_index[preorder[pre_st]]
            
            root.left = solve(pre_st+1, in_st, idx-1)
            root.right = solve(pre_st + (idx-in_st) + 1, idx+1, in_end)
            
            return root
        
        return solve(0, 0, len(inorder)-1)


## 1008. Construct Binary Search Tree from Preorder Traversal
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def bstFromPreorder(self, preorder: List[int]) -> Optional[TreeNode]:
        inorder = sorted(preorder)
        in_index = {v:i for i, v in enumerate(inorder)}
        
        def solve(pre_st, in_st, in_end):
            if pre_st > len(preorder)-1 or in_st > in_end:
                return None
            root = TreeNode(preorder[pre_st])
            
            idx = in_index[preorder[pre_st]]
            
            root.left = solve(pre_st+1, in_st, idx-1)
            root.right = solve(pre_st + (idx-in_st) + 1, idx+1, in_end)
            
            return root
        
        return solve(0, 0, len(inorder)-1)


## 106. Construct Binary Tree from Inorder and Postorder Traversal
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        in_index = {v:i for i, v in enumerate(inorder)}
        
        def solve(in_st, post_st, size):
            if size == 0:
                return None
            val = postorder[post_st + size - 1]
            root = TreeNode(val)
            idx = in_index[val]
            leftsize = idx - in_st
            rightsize = size - leftsize - 1
            root.left = solve(in_st, post_st, leftsize)
            root.right = solve(idx+1, post_st + leftsize, rightsize)
            
            return root
        
        return solve(0, 0, len(inorder))


## 889. Construct Binary Tree from Preorder and Postorder Traversal
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def constructFromPrePost(self, preorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        post_index = {v: i for i, v in enumerate(postorder)}
        
        def solve(pre_st, post_st, size):
            if size == 0:
                return None
            root = TreeNode(preorder[pre_st])
            if size == 1: #important for pre_st+1 not out of bound!
                return root
            
            idx = post_index[preorder[pre_st+1]]
            leftsize = idx - post_st + 1
            rightsize = size - leftsize - 1
            root.left = solve(pre_st + 1, post_st, leftsize)
            root.right = solve(pre_st + 1 + leftsize, idx+1, rightsize)
            
            return root
        
        return solve(0, 0, len(postorder))


## 114. Flatten Binary Tree to Linked List
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


## 156. Binary Tree Upside Down
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


## 427. Construct Quad Tree
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


## 108. Convert Sorted Array to Binary Search Tree
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        
        def bst(left, right):
            
            if left > right:
                return
            
            mid = left + (right-left)//2
            root = TreeNode(nums[mid])
            root.left = bst(left, mid-1)
            root.right = bst(mid+1, right)
            
            return root
        
        return bst(0, len(nums)-1)


## 1382. Balance a Binary Search Tree
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def balanceBST(self, root: TreeNode) -> TreeNode:
        
        nodes = []
        
        def inorder(root): #inorder to maintain order
            if not root:
                return
            inorder(root.left)
            nodes.append(root)
            inorder(root.right)
        
        def build_balanced_tree(left, right):
            if left>right:
                return None
            mid = left + (right-left)//2
            root = nodes[mid]
            root.left = build_balanced_tree(left, mid-1)
            root.right = build_balanced_tree(mid+1, right)
            return root
        
        inorder(root) #get nodes in order
        return build_balanced_tree(0, len(nodes)-1)



### Binary Search Tree Binary Search ###

## 700. Search in a Binary Search Tree
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        
        def solve(root):
            if not root or root.val == val:
                return root
            
            if root.val < val:
                return solve(root.right)
            else:
                return solve(root.left)
            
        return solve(root)


## 701. Insert into a Binary Search Tree
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        
        def solve(root):
            if not root:
                return TreeNode(val)

            if val < root.val:
                root.left = solve(root.left)
            else:
                root.right = solve(root.right)
                
            return root
        
        return solve(root)


## 450. Delete Node in a BST
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        def leftmost(root):
            while root.left:
                  root = root.left
            return root.val
        
        def solve(root, key):
            if not root: return None
            
            if root.val < key:
                root.right = solve(root.right, key)
            elif root.val > key:
                root.left = solve(root.left, key)
            else:
                if not root.left and not root.right: #leaft
                    root = None
                elif not root.left:  #only right child
                    root = root.right
                elif not root.right: #only left child
                    root= root.left
                else:
                    root.val = leftmost(root.right)
                    root.right = solve(root.right, root.val)
                    
            return root
        
        return solve(root, key)


## 938. Range Sum of BST
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rangeSumBST(self, root: Optional[TreeNode], low: int, high: int) -> int:
        ans = 0
        
        def bst(root):
            nonlocal ans
            if not root:
                return 
            if root.val < low:
                bst(root.right)
            elif root.val > high:
                bst(root.left)
            else:
                ans += root.val
                bst(root.left)
                bst(root.right)
                
        bst(root)
        
        return ans


## 270. Closest Binary Search Tree Value
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def closestValue(self, root: Optional[TreeNode], target: float) -> int:
        closest = float('inf')
        result = None
        def bst(root):
            nonlocal closest, result
            if not root:
                return
            if abs(root.val-target) < closest:
                closest = abs(root.val-target)
                result = root.val
            if root.val > target:
                bst(root.left)
            elif root.val < target:
                bst(root.right)
            else:
                return
            
        bst(root)
        return result


## 98. Validate Binary Search Tree
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        prev_val = float('-inf')
        def inorder(root):
            nonlocal prev_val
            if not root:
                return True
            if not inorder(root.left):
                return False
            if root.val <= prev_val:
                return False
            prev_val = root.val
            if not inorder(root.right):
                return False
            
            return True
        
        return inorder(root)



### Binary Search Tree Inorder Traversal ###

## 333. Largest BST Subtree
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def largestBSTSubtree(self, root: Optional[TreeNode]) -> int:
        
        prev_val = float('-inf')
        
        def inorder(root):
            nonlocal prev_val
            if not root:
                return True
            if not inorder(root.left):
                return False
            if root.val <= prev_val:
                return False
            prev_val = root.val
            if not inorder(root.right):
                return False
            return True
        
        def count_nodes(root):
            if not root:
                return 0
            return 1 + count_nodes(root.left) + count_nodes(root.right)
        
        if not root:
            return 0
        
        if inorder(root):
            return count_nodes(root)
        
        return max(self.largestBSTSubtree(root.left), self.largestBSTSubtree(root.right))


## 783. Minimum Distance Between BST Nodes
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


## 530. Minimum Absolute Difference in BST
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


## 230. Kth Smallest Element in a BST
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


## 99. Recover Binary Search Tree
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def recoverTree(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        first, second, prev = None, None, None
        
        def inorder(root):
            nonlocal first, second, prev
            if not root:
                return
            inorder(root.left)
            if prev and prev.val > root.val:
                if not first:
                    first = prev
                second = root
            prev = root
            inorder(root.right)
            
        inorder(root)
        
        first.val, second.val = second.val, first.val


## 501. Find Mode in Binary Search Tree
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findMode(self, root: Optional[TreeNode]) -> List[int]:
        prev_val = float('-inf')
        mode = defaultdict(int)
        def inorder(root):
            nonlocal prev_val
            if not root:
                return
            inorder(root.left)
            if root.val == prev_val:
                mode[root.val] += 1
            else:
                mode[root.val] = 1
            prev_val = root.val
            inorder(root.right)
            
        inorder(root)
        m = max(mode.values())
        return [i for i, v in mode.items() if v == m]


## 897. Increasing Order Search Tree
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def increasingBST(self, root: TreeNode) -> TreeNode:
        ans = prev_node = TreeNode(None)
        
        def inorder(root):
            nonlocal prev_node
            if not root:
                return 
            inorder(root.left)
            root.left = None
            prev_node.right = root
            #bst inorder assign like others in the category
            prev_node = root
            inorder(root.right)
            
        inorder(root)
        return ans.right


## 426. Convert Binary Search Tree to Sorted Doubly Linked List
# Definition for a Node.
# class Node:
#    def __init__(self, val, left=None, right=None):
#        self.val = val
#        self.left = left
#        self.right = right
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
        

## 173. Binary Search Tree Iterator
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


## 538. Convert BST to Greater Tree
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


## 285. Inorder Successor in BST
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def inorderSuccessor(self, root: 'TreeNode', p: 'TreeNode') -> 'Optional[TreeNode]':
        
        prev, result = None, None
        def inorder(root):
            nonlocal prev, result
            if not root: return
            inorder(root.left)
            if prev:
                if prev.val == p.val:
                    result = root
                
            prev = root
            
            inorder(root.right)
            
        inorder(root)
        
        return result


## 510. Inorder Successor in BST II
# Definition for a Node.
# class Node:
#    def __init__(self, val):
#        self.val = val
#        self.left = None
#        self.right = None
#       self.parent = None
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



### Tree DP/Greedy ###

## 96. Unique Binary Search Trees
#1) DP
class Solution:
    def numTrees(self, n: int) -> int:
        
        G = [0] * (n+1)
        G[0], G[1] = 1, 1

        for i in range(2, n+1):
            for j in range(1, i+1):
                G[i] += G[j-1] * G[i-j]  #i-j + j-1 = i-1; 1 is root

        return G[n]

#2) time exceeds limit
class Solution:
    def numTrees(self, n: int) -> int:
        
        def solve(start, end):
            if start > end:
                return 1
            
            all_trees = 0
            for i in range(start, end+1):
                l = solve(start, i-1)  #different combinations
                r = solve(i+1, end)  #different combinations
                
                all_trees += l* r
            
            return all_trees
        
        return solve(1, n) if n else 0


## 95. Unique Binary Search Trees II
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


## 894. All Possible Full Binary Trees
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


## 968. Binary Tree Cameras
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


## 337. House Robber III
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


## 979. Distribute Coins in Binary Tree
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

