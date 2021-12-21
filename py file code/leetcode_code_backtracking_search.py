
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
        


### 