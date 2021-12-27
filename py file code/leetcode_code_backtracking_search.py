
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
            if result:
                return
            if len(comb) == length:
                result = list(comb)
                return
            
            for i in range(n):
                shift = start ^ (1<<i) #get shift bit value in position i
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
                    
                    if dfs(start_p+1, i+1, ptable, stable):
                        return True
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
        
        nums = sorted(nums)
        used = [0 for i in range(0, len(nums))]
        
        def dfs(cur_sum, start, remaining_k):
            if remaining_k == 1:
                return True
            
            if cur_sum == targetSum:
                return dfs(0, 0, remaining_k - 1)
            
            for i in range(start, len(nums)):
                if used[i]: continue
                if cur_sum + nums[i] > targetSum: break  
                cur_sum = cur_sum + nums[i]
                used[i] = 1
                if dfs(cur_sum, i + 1, remaining_k): return True
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
        
        l, r = 0, 0
        for char in s:  #total invalid r and l to remove
            l += (char == '(')
            if (l == 0):
                r += (char == ')')
            else:
                l -= (char == ')')
        result = []
        
        def isvalid(s):
            count = 0
            for char in s:
                if char == '(': count += 1
                if char == ')': count -= 1
                if count < 0: return False
            
            return count == 0
        
        def dfs(comb, start, left, right):
            curr_str = ''.join(comb)
            if left == 0 and right == 0 and isvalid(curr_str):
                result.append(curr_str)
                return
            
            for i in range(start, len(s)):
                if s[i] != '(' and  s[i] != ')':
                    continue
                if i != start and s[i] == s[i-1]: 
                    continue
                    
                if (r > 0 and s[i] == ')'):
                    comb[i] = ''
                    dfs(comb, i+1, left, right - 1)
                    comb[i] = s[i]
                elif (l > 0 and s[i] == '('):
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
        
        for y in range(N):
            for x in range(N):
                c = board[y][x]               
                if c != '.':
                    n = int(c)                   
                    bx = x // 3
                    by = y // 3
                    rows_fill[y][n-1] = 1 #row i
                    cols_fill[x][n-1] = 1 #column j
                    boxes_fill[by * 3 + bx][n-1] = 1  
                            
        def dfs(y, x):
            if (y == 9): return True
        
            nx = (x + 1) % 9  #pointer to next column
            ny = y + 1 if nx == 0 else y #start a new if one row reach end
        
            if board[y][x] != '.': return dfs(ny, nx)  #if no fill need, check if solution found
        
            for i in range(1, 10):
                
                by = y // 3
                bx = x // 3
                box_key = by * 3 + bx
                if rows_fill[y][i-1] or cols_fill[x][i-1] or boxes_fill[box_key][i-1]: continue
                rows_fill[y][i-1] = 1
                cols_fill[x][i-1] = 1
                boxes_fill[box_key][i-1] = 1
                board[y][x] = str(i)
                if dfs(ny, nx): return True  #check if solution found
                board[y][x] = '.'
                boxes_fill[box_key][i-1] = 0
                cols_fill[x][i-1] = 0
                rows_fill[y][i-1] = 0
            
            return False
  
        dfs(0, 0)


### 51. N-Queens
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        result = []
        board = [['.' for i in range(n)] for i in range(n)]
        cols = [0 for i in range(n)]
        diag1 = [0 for i in range(2 * n - 1)]
        diag2 = [0 for i in range(2 * n - 1)]
        
        def available(y, x):
            return not cols[x] and not diag1[x + y] and not diag2[x - y + n - 1]
                 
        def updateBoard(y, x, is_put):
            cols[x] = is_put
            diag1[x + y] = is_put
            diag2[x - y + n - 1] = is_put
            board[y][x] = 'Q' if is_put else '.'
                 
        def dfs(y):       
            if y == n:
                result.append([''.join(row) for row in board])
                return
                 
            for x in range(0, n):
                if not available(y, x): continue
                updateBoard(y, x, True)
                dfs(y + 1)
                updateBoard(y, x, False) #backtrack
                 
        dfs(0)
                 
        return result
                 
                 
 ## 52. N-Queens II
 # class Solution:
    def totalNQueens(self, n: int) -> int:
        cnt = 0
        board = [['.' for i in range(n)] for i in range(n)]
        cols = [0 for i in range(n)]
        diag1 = [0 for i in range(2 * n - 1)]
        diag2 = [0 for i in range(2 * n - 1)]
        
        def available(y, x):
            return not cols[x] and not diag1[x + y] and not diag2[x - y + n - 1]
                 
        def updateBoard(y, x, is_put):
            cols[x] = is_put
            diag1[x + y] = is_put
            diag2[x - y + n - 1] = is_put
            board[y][x] = 'Q' if is_put else '.'
                 
        def dfs(y):   
            nonlocal cnt
            if y == n:
                cnt+=1
                return
                 
            for x in range(0, n):
                if not available(y, x): continue
                updateBoard(y, x, True)
                dfs(y + 1)
                updateBoard(y, x, False) #backtrack
                 
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
        ROWS = len(board)
        COLS = len(board[0])
        
        def dfs(start, y, x):
            if start == len(word):
                return True
            
            if y < 0 or y == ROWS or x < 0 or x == COLS or board[y][x] != word[start]:
                return False

            curr = board[y][x]
            board[y][x] = '#'
            found = dfs(start + 1, y + 1, x) or dfs(start + 1, y - 1, x) or dfs(start + 1, y, x + 1) or dfs(start + 1, y, x - 1)
            board[y][x] = curr #backtracking
            
            return found
        
        
        return any([dfs(0, y, x) for y in range(ROWS) for x in range(COLS)])


## 212. Word Search II
class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        
        ROWS = len(board)
        COLS = len(board[0])
        
        def exist(word):
            return any([dfs(0, y, x, word) for y in range(ROWS) for x in range(COLS)])
        
        def dfs(start, y, x, word):
            if start == len(word):
                return True
            
            if y < 0 or y == ROWS or x < 0 or x == COLS or board[y][x] != word[start]:
                return False

            curr = board[y][x]
            board[y][x] = '#'
            found = dfs(start + 1, y + 1, x, word) or dfs(start + 1, y - 1, x, word) or dfs(start + 1, y, x + 1, word) or dfs(start + 1, y, x - 1, word)
            board[y][x] = curr #backtracking
            
            return found
        
        
        
        letters = [board[i][j] for i in range(ROWS) for j in range(COLS)]
        letters = collections.Counter(letters)
        new_words = []           
                   
        for word_ori in words:
            if len(word_ori) > ROWS * COLS: continue
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
                    #more advanced way
                    #ans += [ops[s[i]](l, r) for l, r in itertools.product(dfs(s[0:i]), dfs(s[i+1:]))]
            if not ans: 
                ans.append(int(s))
              
            return ans
        
        return dfs(expression)


## 282. Expression Add Operators
class Solution:
    def addOperators(self, num: str, target: int) -> List[str]:
        INT_MAX = 2**31 - 1
        result = []
        def dfs(expr, start, prev, curr):
            if start == len(num):
                if curr == target:
                    result.append(expr)
                    return
                
            for i in range(1, len(num) - start + 1):
                t = num[start: start+i]
                if t[0] == '0' and i>1: break
                n = int(t)
                if n > INT_MAX: break
                if start == 0:
                    dfs(t, i, n, n)
                    continue
                dfs(expr + '+' + t, start + i, n, curr + n)
                dfs(expr + '-' +  t, start + i, -n, curr - n)
                dfs(expr + '*'+ t, start + i, prev * n, curr - prev + prev * n)
                
        dfs("", 0, 0, 0)
        return result
    
        