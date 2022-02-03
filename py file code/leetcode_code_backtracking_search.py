
### Combination

## 39. Combination Sum
#1)sort
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        ans = []
        
        def dfs(comb, start, target):
            if target == 0:
                ans.append(list(comb))
                return
            
            for i in range(start, len(candidates)):
                if candidates[i] > target: break
                comb.append(candidates[i])
                dfs(comb, i, target-candidates[i])
                comb.pop()
                
        candidates = sorted(candidates)        
        dfs([], 0, target)
        
        return ans

#2)without sort
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        ans = []
        
        def dfs(comb, start, target):
            if target == 0:
                ans.append(list(comb))
                return
            elif target < 0:
                return
            
            for i in range(start, len(candidates)):
                comb.append(candidates[i])
                dfs(comb, i, target-candidates[i])
                comb.pop()
                
        dfs([], 0, target)
        
        return ans


## 40. Combination Sum II
#1)sort
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        
        counter = collections.Counter(candidates)
        nums = sorted(counter.keys())
        ans = []
        def dfs(comb, start, target):
            
            if target == 0:
                ans.append(list(comb))
                return
                
            for i in range(start, len(nums)):
                if nums[i] > target: 
                    break
                if counter[nums[i]] <=0: 
                    continue
                comb.append(nums[i])
                counter[nums[i]]-=1
                dfs(comb, i, target-nums[i])
                comb.pop()
                counter[nums[i]]+=1
                
        dfs([], 0, target)

        return ans

#2)without sort
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        
        counter = collections.Counter(candidates)
        counter = [(k, v) for k, v in counter.items()]
        ans = []
        def dfs(comb, start, target):
            
            if target == 0:
                ans.append(list(comb))
                return
            elif target < 0:
                return
                
            for i in range(start, len(counter)):
                val, cnt = counter[i]
                if cnt <=0: 
                    continue
                comb.append(val)
                counter[i]= val, cnt-1
                dfs(comb, i, target-val)
                comb.pop()
                counter[i] = val, cnt
                
        dfs([], 0, target)

        return ans


## 90. Subsets II
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        counter = collections.Counter(nums)
        counter = [(k, v) for k, v in counter.items()]
        ans = []
        def dfs(comb, start):
            
            if len(comb) <= len(nums):
                ans.append(list(comb))
            
            for i in range(start, len(counter)):
                val, cnt = counter[i]
                if cnt <=0: 
                    continue
                comb.append(val)
                counter[i]= val, cnt-1
                dfs(comb, i)
                comb.pop()
                counter[i] = val, cnt
                
        dfs([], 0)

        return ans

class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        counter = collections.Counter(nums)
        n = len(nums)
        nums = list(counter.keys())
        ans = []
        def dfs(comb, start):
            if len(comb) <= n:
                ans.append(list(comb))
                
            for i in range(start, len(counter)):
                val = nums[i]
                cnt = counter[val]
                if cnt <=0: continue
                comb.append(val)
                counter[val]= cnt-1
                dfs(comb, i)
                comb.pop()
                counter[val]= cnt
                
        dfs([], 0)

        return ans

## 254. Factor Combinations
class Solution:
    def getFactors(self, n: int) -> List[List[int]]:
        
        if n == 1: 
            return []
        ans = []
        
        def dfs(comb, start, target):
            
            if len(comb)>0:
                ans.append(comb + [target])
                
            for i in range(start, int(math.sqrt(target))+1): 
                if target %i == 0:
                    comb.append(i)
                    dfs(comb, i, target//i)
                    comb.pop()
        
        dfs([], 2, n)

## 77. Combinations
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        ans = []
        def dfs(comb, start):
            if len(comb) == k:
                ans.append(list(comb)) 
                return
            for i in range(start, n+1):
                comb.append(i)
                dfs(comb, i+1)
                comb.pop() #restate
                
        dfs([], 1)
        
        return ans


## 216. Combination Sum III
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        ans = []
        
        def dfs(comb, start, target):
            if (target == 0) and len(comb) == k:
                ans.append(list(comb))
                return
            elif target < 0:
                return
            
            for i in range(start, 10):
                comb.append(i)
                dfs(comb, i+1, target-i)
                comb.pop()
                
        dfs([], 1, n)
        
        return ans
    

## 78. Subsets
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        ans = []
        def dfs(comb, start):
            if len(comb) <= len(nums):
                ans.append(list(comb))
            
            for i in range(start, len(nums)):
                comb.append(nums[i])
                dfs(comb, i+1)
                comb.pop()
                
        dfs([], 0)
        
        return ans


## 1239. Maximum Length of a Concatenated String with Unique Characters
class Solution:
    def maxLength(self, arr: List[str]) -> int:
        new_arr = []
        for each in arr:
            if len(set(each)) == len(each):
                new_arr.append(each)
        if len(new_arr) == 0:
            return 0 
        
        ans = []
        longest = 0
        def dfs(comb, start):
            nonlocal longest
            concatenate = ''.join(comb)
            if len(concatenate) == len(set(concatenate)):
                longest = max(longest, len(concatenate))
            else:
                return
            
            for i in range(start, len(new_arr)):
                each = new_arr[i]
                comb.append(each)
                dfs(comb, i + 1)
                comb.pop()
                
        dfs([], 0)
        return longest


## 17. Letter Combinations of a Phone Number
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        letters = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl", 
                   "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"}
        ans = []
        def dfs(comb, start):
            if len(comb) == len(digits):
                ans.append(''.join(comb))
                return
                
            for each in letters[digits[start]]:
                comb.append(each)
                dfs(comb, start+1)
                comb.pop()
        
        if not digits: #corner case
            return []
        
        dfs([], 0)
        
        return ans


## 784. Letter Case Permutation
class Solution:
    def letterCasePermutation(self, s: str) -> List[str]:
        
        ans = []
        def dfs(comb, start):
            if len(comb) == len(s):
                ans.append(''.join(comb))
                return
                
            curr = s[start]
            if not curr.isalpha():
                comb.append(curr)
                dfs(comb, start+1)
                comb.pop()
            else:
                for each in [curr, chr(ord(curr) ^ (1<<5))]:
                    comb.append(each)
                    dfs(comb, start+1)
                    comb.pop()

        dfs([], 0)
        
        return ans
                

## 797. All Paths From Source to Target
class Solution:
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        ans = []
        n = len(graph) - 1
        def dfs(comb, next_node):
            if next_node == n:
                ans.append(list(comb))
                return
            
            for each in graph[next_node]:
                comb.append(each)
                dfs(comb, each)
                comb.pop()
                
                
        dfs([0], 0)
        
        return ans


## 967. Numbers With Same Consecutive Differences
class Solution:
    def numsSameConsecDiff(self, n: int, k: int) -> List[int]:
        ans = []
        
        def dfs(comb):
            if len(comb) == n:
                ans.append(int(''.join(map(str, comb))))
                return
            
            for each in range(0, 10):
                if len(comb) == 0 and each == 0: continue
                if len(comb) > 0 and (abs(comb[-1] - each) != k): continue
                comb.append(each)
                dfs(comb)
                comb.pop()
                
                
        dfs([])
        
        return ans


## 494. Target Sum
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        count = defaultdict(int)
        count[0] = 1 #init
        for x in nums:
            step = defaultdict(int)
            for y in count:
                step[y + x] += count[y]  #'+'
                step[y - x] += count[y]  #'-'
            count = step

        return count[target]
                

## 377. Combination Sum IV
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
       
        nums = sorted(nums)
        dp= [0 for i in range(target+1)]
        dp[0] = 1  #initialization
        for comb_sum in range(1, target + 1):
            s = 0
            for num in nums:
                if comb_sum - num >= 0:
                    s += dp[comb_sum - num]
                else:
                    break
            dp[comb_sum] = s
        return dp[target]



### Permutation

## 46. Permutations
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        result = []
        used = [0 for i in range(len(nums))]
        def dfs(perm):
            if len(perm) == len(nums):
                result.append(list(perm))
                return
            for i in range(0, len(nums)):
                if used[i]: continue
                used[i] = 1
                perm.append(nums[i])
                dfs(perm)
                perm.pop()
                used[i] = 0
                
        dfs([])
        return result


## 526. Beautiful Arrangement
class Solution:
    def countArrangement(self, n: int) -> int:
  
        cnt = 0
        nums = [i for i in range(1, n+1)]
        used = [0 for i in range(1, n+1)]
        def dfs(perm, curr_index):
            nonlocal cnt
            if len(perm) == n:
                print(perm)
                cnt += 1
                return
            for i in range(0, len(nums)):
                if used[i] == 1: continue
                if (nums[i] % curr_index != 0) and (curr_index % nums[i] != 0): continue
                perm.append(nums[i])
                used[i] = 1
                dfs(perm, curr_index +1)
                perm.pop()
                used[i] = 0
                    
        dfs([], 1)
        
        return cnt


## 357. Count Numbers with Unique Digits
class Solution:
    def countNumbersWithUniqueDigits(self, n: int) -> int:
        cnt = 0
        used = [0 for i in range(0, 10)]
        def dfs(perm):
            nonlocal cnt
            if len(perm) <= n:  #if perm = [] count 0 as 1 cnt
                cnt+=1
                if len(perm) == n:
                    return
            
            for i in range(0, 10):
                if len(perm) == 0 and i == 0: continue
                if used[i]: continue
                used[i] = 1
                perm.append(i)
                dfs(perm)
                perm.pop()
                used[i] = 0
        
        if n == 0:
            return 1
        
        dfs([])
        return cnt


## 47. Permutations II
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        result = []
        nums = sorted(nums)
        used = [0 for i in range(len(nums))]
        def dfs(perm):
            
            if len(perm) == len(nums):
                result.append(list(perm))
                return
            for i in range(0, len(nums)):
                if used[i]: continue
                if (i > 0 and nums[i] == nums[i - 1] and not used[i - 1]): continue
                used[i] = 1
                perm.append(nums[i])
                dfs(perm)
                perm.pop()
                used[i] = 0
                
        dfs([])
        return result


## 996. Number of Squareful Arrays
class Solution:
    def numSquarefulPerms(self, nums: List[int]) -> int:
        
        def squareful(x, y):
            s = int(math.sqrt(x + y));
            return s * s == x + y
        
        cnt = 0
        nums = sorted(nums)
        used = [0 for i in range(len(nums))]
        
        def dfs(perm):
            nonlocal cnt
            if len(perm) == len(nums):
                cnt+=1
                return
            
            for i in range(0, len(nums)):
                if used[i]: continue
                if (i > 0 and nums[i] == nums[i - 1] and not used[i - 1]): continue
                if perm and (not squareful(perm[-1], nums[i])): continue
                
                used[i] = 1
                perm.append(nums[i])
                dfs(perm)
                perm.pop()
                used[i] = 0
                
        dfs([])
        return cnt


## 89. Gray Code
class Solution:
    def grayCode(self, n: int) -> List[int]:
        length = 2**n  
        seen=set([0]) #init from 0
        result = None
        
        
        def dfs(comb, start):
            nonlocal result
            if result:  #return any valid n-bit gray code sequence
                return
            if len(comb) == length:
                result = list(comb)
                return
            
            for i in range(n):
                shift = start ^ (1<<i) #get shift 0/1 value in position i(from right to left)
                if shift in seen: continue
                seen.add(shift)
                comb.append(shift)
                dfs(comb, shift)
                seen.remove(shift) #restate
                comb.pop()
        
        dfs([0], 0)
        return result



### Partition

## 93. Restore IP Addresses
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        result = []
        def dfs(comb, start):
            if len(comb) == 4:
                if start == len(s):
                    result.append('.'.join(comb))
                return
            for i in range(start, min(start+3, len(s))):
                if s[start] == '0' and i != start: continue
                if int(s[start:i+1]) > 255: continue
                comb.append(s[start: i+1])
                dfs(comb, i+1)
                comb.pop()
        
        
        dfs([], 0)
        
        return result


## 131. Palindrome Partitioning
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        result = []
        
        def isPalindrome(start, end):
            while start <= end:
                if s[start] != s[end]:
                    return False
                else:
                    start+=1
                    end-=1
            return True
        
        def dfs(comb, start):
            if start == len(s):
                result.append(list(comb))
                
            for i in range(start, len(s)):
                if not isPalindrome(start, i): continue
                comb.append(s[start:i+1])
                dfs(comb, i+1)
                comb.pop()
                    
        
        dfs([], 0)
        return result


## 1593. Split a String Into the Max Number of Unique Substrings
class Solution:
    def maxUniqueSplit(self, s: str) -> int:
        longest = 0
        def dfs(comb, start):
            nonlocal longest
            if start == len(s):
                longest = max(longest, len(comb))
            
            for i in range(start, len(s)):
                substr = s[start: i+1]
                if substr in comb: continue
                comb.append(substr)
                dfs(comb, i+1)
                comb.pop()
        
        dfs([], 0)
        
        return longest


## 842. Split Array into Fibonacci Sequence
class Solution:
    def splitIntoFibonacci(self, num: str) -> List[int]:
        result = None
        
        def dfs(comb, start):
            nonlocal result
            if result: return
            if len(comb) >= 3 and start == len(num):
                result = [int(i) for i in comb]
                return
                
            for i in range(start, len(num)):
                substr = num[start:i+1]
                if len(substr) > 1 and substr[0] == '0': continue
                if int(substr) >= 2**31-1: continue
                if len(comb) >= 2:
                    if int(comb[-2]) + int(comb[-1]) != int(substr): continue
                comb.append(substr)
                dfs(comb, i+1)
                comb.pop()
                    
        
        dfs([], 0)
        return result


## 291. Word Pattern II
class Solution:
    def wordPatternMatch(self, pattern, s):
        """
        :type pattern: str
        :type str: str
        :rtype: bool
        """
        def dfs(start_p, start_s, ptable, stable):
            if start_p == len(pattern) and start_s == len(s):
                return True
            elif start_p == len(pattern) or start_s == len(s):
                return False
            else:
                p, added = pattern[start_p], False
                for i in range(start_s, len(s)):
                    word = s[start_s:i+1]
                    if (p in ptable and ptable[p] != word) or (word in stable and stable[word] != p):
                        continue
                    if p not in ptable and word not in stable:
                        ptable[p], stable[word], added = word, p, True
                    
                    if dfs(start_p+1, i+1, ptable, stable): return True #to escape final False if find True
                    if added:
                        del ptable[p]
                        del stable[word]
            return False
        
        return dfs(0, 0, {}, {})


## 698. Partition to K Equal Sum Subsets
class Solution:
    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        targetSum = sum(nums) 
        if targetSum % k != 0 :
            return False
        else:
            targetSum = targetSum//k
        
        nums = sorted(nums) #work with for loop break to speed up
        used = [0 for i in range(0, len(nums))]
        
        def dfs(cur_sum, start, remaining_k):
            if remaining_k == 1: #find all partitions
                return True
            
            if cur_sum == targetSum: #reach one partition
                return dfs(0, 0, remaining_k - 1)
            
            for i in range(start, len(nums)):
                if used[i]: continue
                if (i > 0 and nums[i] == nums[i - 1] and not used[i - 1]): continue #permutation duplicate
                if cur_sum + nums[i] > targetSum: break  
                cur_sum = cur_sum + nums[i]
                used[i] = 1
                if dfs(cur_sum, i + 1, remaining_k): return True #to escape final False if find True
                cur_sum = cur_sum - nums[i] #restate
                used[i] = 0  #restate
            
            return False 
        
        return dfs(0,  0, k)
        


### DFS1 backtracking

## 22. Generate Parentheses
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        result = []
        def dfs(comb, left, right):
            if len(comb) == 2 * n:
                result.append(''.join(comb)) 
                return
            if left < n:
                comb.append("(")
                dfs(comb, left+1, right)
                comb.pop() #restate
            if right < left: #make it well formed
                comb.append(")")
                dfs(comb, left, right+1)
                comb.pop() #restate
                
        dfs([], 0, 0)
        return result


## 301. Remove Invalid Parentheses
class Solution:
    def removeInvalidParentheses(self, s: str) -> List[str]:
        def isvalid(s):
            count = 0
            for c in s:
                if c == "(":
                    count+=1
                elif c == ")":
                    count-=1
                if count < 0:
                    return False
                
            return count == 0
                
        
        l, r = 0, 0
        for c in s:
            l += (c == '(')
            if l == 0:
                r += (c == ')')
            else:
                l -= (c == ')')
        result = []
        
        
        def dfs(comb, start, left, right):
            curr_str = ''.join(comb)
            if left == 0 and right == 0 and isvalid(curr_str):
                result.append(curr_str)
                return
            
            for i in range(start, len(s)):
                if s[i] != '(' and s[i] != ')': continue
                if i!= start and s[i] == s[i-1]: continue
                    
                if right > 0 and s[i] == ')':
                    comb[i] = ''
                    dfs(comb, i+1, left, right - 1)
                    comb[i] = s[i]
                elif left > 0 and s[i] == '(':
                    comb[i] = ''
                    dfs(comb, i+1, left-1, right)
                    comb[i] = s[i]
                    
        dfs(list(s), 0, l, r)
        
        return result
    
  


### DFS2 fill matrix backtracking

## 37. Sudoku Solver
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        N = 9
        rows_fill = [[0 for i in range(N)] for i in range(N)]
        cols_fill = [[0 for i in range(N)] for i in range(N)]
        boxes_fill = [[0 for i in range(N)] for i in range(N)]
        
        for i in range(N):
            for j in range(N):
                c = board[i][j]               
                if c != '.':
                    n = int(c)                   
                    bi = i // 3
                    bj = j // 3
                    box_key = bi * 3 + bj
                    rows_fill[i][n-1] = 1 #row i with num n
                    cols_fill[j][n-1] = 1 #column j with num n
                    boxes_fill[box_key][n-1] = 1  #box [i,j] with num n top down then left right
                            
        def dfs(i, j):
            if (i == 9): return True  #reach the end of fill (8, 8)
        
            #pointer to the next cell regards of (i, j)
            nj = (j + 1) % 9  #pointer to next column
            ni = i + 1 if nj == 0 else i #start a new if one row reach end
        
            if board[i][j] != '.': 
                return dfs(ni, nj)  #if no fill need, check if solution found
        
            for n in range(1, 10):
                
                bi = i // 3
                bj = j // 3
                box_key = bi * 3 + bj
                
                if rows_fill[i][n-1] or cols_fill[j][n-1] or boxes_fill[box_key][n-1]: continue
                    
                board[i][j] = str(n)
                rows_fill[i][n-1] = 1
                cols_fill[j][n-1] = 1
                boxes_fill[box_key][n-1] = 1
                
                if dfs(ni, nj): return True  #check if solution found
                
                board[i][j] = '.' #backtrack
                rows_fill[i][n-1] = 0 #backtrack
                cols_fill[j][n-1] = 0 #backtrack
                boxes_fill[box_key][n-1] = 0 #backtrack
            
            return False
  
        dfs(0, 0)
        

### 51. N-Queens
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        result = []
        board = [['.' for _ in range(n)] for _ in range(n)]
        cols = [0 for _ in range(n)]
        diag1 = [0 for _ in range(2 * n - 1)]  
        diag2 = [0 for _ in range(2 * n - 1)]
        
        def available(i, j):
            return not cols[j] and not diag1[i + j] and not diag2[j - i + n - 1]
                 
        def updateBoard(i, j, is_put):
            cols[j] = is_put
            diag1[i + j] = is_put
            diag2[j - i + n - 1] = is_put
            board[i][j] = 'Q' if is_put else '.'
                 
        def dfs(i):       
            if i == n:
                result.append([''.join(row) for row in board])
                return
                 
            for j in range(0, n):
                if not available(i, j): continue
                updateBoard(i, j, 1)
                dfs(i + 1)
                updateBoard(i, j, 0) #backtrack
                 
        dfs(0)
                 
        return result
                 
                 
                 
 ## 52. N-Queens II
class Solution:
    def totalNQueens(self, n: int) -> int:
        cnt = 0
        
        board = [['.' for _ in range(n)] for _ in range(n)]
        cols = [0 for _ in range(n)]
        diag1 = [0 for _ in range(2 * n - 1)]  
        diag2 = [0 for _ in range(2 * n - 1)]
        
        def available(i, j):  #col and diag1 and diag2 are all 0 (not filled)
            return not cols[j] and not diag1[i + j] and not diag2[j - i + n - 1]
                 
        def updateBoard(i, j, is_put):
            cols[j] = is_put
            diag1[i + j] = is_put
            diag2[j - i + n - 1] = is_put
            board[i][j] = 'Q' if is_put else '.'
                 
        def dfs(i):   
            nonlocal cnt
            if i == n: #reach 
                cnt+=1
                return
                 
            for j in range(0, n):
                if not available(i, j): continue
                updateBoard(i, j, 1)
                dfs(i + 1)
                updateBoard(i, j, 0) #backtrack
                 
        dfs(0)
        return cnt     
   


### DFS3 word search backtracking

## 79. Word Search
class Solution:

    def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        m = len(board)
        n = len(board[0])
        
        def dfs(start, i, j):
            if start == len(word):
                return True
            
            if i < 0 or j < 0 or i >= m or j >= n or board[i][j] != word[start]:
                return False

            curr = board[i][j]
            board[i][j] = '.'
            found = dfs(start + 1, i + 1, j) or dfs(start + 1, i - 1, j) or dfs(start + 1, i, j + 1) or dfs(start + 1, i, j - 1)
            board[i][j] = curr #backtracking
            
            return found
        
        
        return any([dfs(0, i, j) for i in range(m) for j in range(n)])


class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        
        m = len(board)
        n = len(board[0])
        
        def exist(word):
            return any([dfs(0, i, j, word) for i in range(m) for j in range(n)])
        
        def dfs(start, i, j, word):
            if start == len(word):
                return True
            
            if i < 0 or j < 0 or i >= m or j >= n or board[i][j] != word[start]:
                return False

            curr = board[i][j]
            board[i][j] = '.'
            found = dfs(start + 1, i + 1, j, word) or dfs(start + 1, i - 1, j, word) or dfs(start + 1, i, j + 1, word) or dfs(start + 1, i, j - 1, word)
            board[i][j] = curr #backtracking
            
            return found
        
        
        
        letters = [board[i][j] for i in range(m) for j in range(n)]
        letters = collections.Counter(letters)
        new_words = []           
                   
        for word_ori in words:
            if len(word_ori) > m * n: continue
            word = collections.Counter(word_ori)
            skip = False
            for each, v in word.items():
                if each not in letters.keys():
                    skip = True
                    break;
                elif v > letters[each]:
                    skip = True
                    break; 
            if skip: continue
            new_words.append(word_ori)
        
        result = []
        for word in new_words:    
            if exist(word):
                result.append(word)
                
        return result
        

### DFS4 search

## 241. Different Ways to Add Parentheses
class Solution:
    def diffWaysToCompute(self, expression: str) -> List[int]:
        ops = {'+': lambda x, y: x + y,
               '-': lambda x, y: x - y,
               '*': lambda x, y: x * y}
        
        def dfs(s):
            ans = []
            for i in range(len(s)):
                if s[i] in "+-*": 
                    l = dfs(s[0:i])
                    r = dfs(s[i+1:])
                    for l1 in l:
                        for r1 in r:
                            ans += [ops[s[i]](l1, r1)]
            
            if not ans: #only digits
                return [int(s)]
              
            return ans
        
        return dfs(expression)


## 282. Expression Add Operators
class Solution:
    def addOperators(self, num: str, target: int) -> List[str]:
        INT_MAX = 2**31 - 1
        result = []
        def dfs(expr, start, prev, curr):
            if start == len(num): #reach
                if curr == target:
                    result.append(expr)
                    return
                
            for i in range(1, len(num) - start + 1):
                t = num[start: start+i]
                if t[0] == '0' and len(t) >=2: break #no leading zero
                n = int(t)
                if n > INT_MAX: break
                if start == 0: #the first element
                    dfs(t, start + i, n, n)
                    continue
                dfs(expr + '+' + t, start + i, n, curr + n)
                dfs(expr + '-' +  t, start + i, -n, curr - n)
                dfs(expr + '*'+ t, start + i, prev * n, curr - prev + prev * n) #reorder the priority in *
                
        dfs("", 0, 0, 0)
        return result



### BFS Search

## 127. Word Ladder
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        wordList = set(wordList)
        if endWord not in wordList:
            return 0
        
        l = len(beginWord)
        steps = 1
        queue = [beginWord]
        
        while queue: 
            wordList = wordList - set(queue)
            size = len(queue)
            for _ in range(size):
                word = queue.pop(0)
                for i in range(l):
                    ch = word[i]
                    for t in string.ascii_lowercase:
                        if t == ch: continue
                        newword = word[:i] + t + word[i+1:]
                        if endWord == newword: 
                            return steps + 1
                        if (newword not in wordList):
                            continue
                        wordList.remove(newword)  #increase efficiency because no need to backtrack
                        queue.append(newword)
            steps +=1
                   
        return 0


## 26. Word Ladder II
class Solution:
    def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        
        wordList = set(wordList)
        if endWord not in wordList:
            return []
        
        parents = defaultdict(set)
        l = len(beginWord)
        queue = [beginWord]
        
        while queue: 
            wordList = wordList - set(queue)
            size = len(queue)
            for _ in range(size):
                word = queue.pop(0)
                for i in range(l):
                    ch = word[i]
                    for t in string.ascii_lowercase:
                        if t == ch: continue
                        newword = word[:i] + t + word[i+1:]
                        if (newword not in wordList):
                            continue
                        parents[word].add(newword)
                        queue.append(newword)
       
        def dfs(word, curr):  #dfs backtracking
            if word == endWord:
                result.append(list(curr))
                return
            for p in parents[word]:
                curr.append(p)
                dfs(p, curr)
                curr.pop()
        
        result = [] 
        
        dfs(beginWord, [beginWord])
            
        return result
    
## 752. Open the Lock 
class Solution:
    def openLock(self, deadends: List[str], target: str) -> int:
        wordList = set(deadends)
        beginWord = '0' * len(target)
        if beginWord == target:
            return 0
        if target in wordList or beginWord in wordList:
            return -1
        
        
        l = len(beginWord)
        queue = [beginWord]
        visited = set([beginWord])
        steps = 0
        
        while queue: 
            size = len(queue)
            for _ in range(size):
                word = queue.pop(0)
                for i in range(l):
                    ch = word[i]
                    if ch == '9':
                        moves = ['0', '8']
                    elif ch == '0':
                        moves = ['1', '9']
                    else:
                        moves = [str(int(ch) - 1), str(int(ch) + 1)]
                    for t in moves:
                        newword = word[:i] + t + word[i+1:]
                        if target == newword: 
                            return steps + 1
                        if (newword in wordList or newword in visited):
                            continue
                        queue.append(newword)
                        visited.add(newword)
            steps +=1
                   
        return -1

## 433. Minimum Genetic Mutation
class Solution:
    def minMutation(self, start: str, end: str, bank: List[str]) -> int:
        wordList = set(bank)
        if end not in wordList:
            return -1
        
        l = len(start)
        queue = [start]
        visited = set([start])
        steps = 0
        
        while queue: 
            wordList = wordList - set(queue)
            size = len(queue)
            for _ in range(size):
                word = queue.pop(0)
                for i in range(l):
                    ch = word[i]
                    for t in ['A', 'C', 'G', 'T']:
                        if t == ch: continue
                        newword = word[:i] + t + word[i+1:]
                        if end == newword: 
                            return steps + 1
                        if (newword not in wordList or newword in visited):
                            continue
                        queue.append(newword)
                        wordList.remove(newword)
            steps+=1
                 
        return -1