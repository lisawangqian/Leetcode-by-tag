###39. Combination Sum
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        result = []
        def backtrack(remain, comb, start):
            if remain == 0:
                result.append(list(comb))  #must use list or o/w object var will change dynamically
                return
            elif remain < 0:
                return
            for i in range(start, len(candidates)):
                comb.append(candidates[i])
                backtrack(remain-candidates[i], comb, i)
                comb.pop() #restate
                
        backtrack(target, [], 0)
        
        return result


### 216. Combination Sum III        

class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        result = []
        
        def backtrack(remain, comb, start):
            if (remain == 0) & (len(comb) == k):
                result.append(list(comb))
                return
            elif (remain < 0) & (len(comb) == k):
                return
            
            for i in range(start, 10):  #1-9
                comb.append(i)
                backtrack(remain-i, comb, i+1)  #unique number
                comb.pop() #restate
        
        backtrack(n, [], 1)
        return result

### 377. Combination Sum IV
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
       
        nums = sorted(nums)
        dp= [0 for i in range(target+1)]
        dp[0] = 1  #initialization
        for comb_sum in range(1, target + 1):
            for num in nums:
                if comb_sum - num >= 0:
                    dp[comb_sum] += dp[comb_sum - num]
                else:
                    break
        return dp[target]


### 78. Subsets
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        result = []
        def backtrack(comb, start):
            result.append(list(comb))  
            if len(comb) >= len(nums):
                return
            for i in range(start, len(nums)):
                comb.append(nums[i])
                backtrack(comb, i+1)
                comb.pop() #restate
                
        backtrack([], 0)
        
        return result


### 77. Combinations
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        result = []
        def backtrack(comb, start):
            if len(comb) == k:
                result.append(list(comb)) 
                return
            for i in range(start, n+1):
                comb.append(i)
                backtrack(comb, i+1)
                comb.pop() #restate
                
        backtrack([], 1)
        
        return result


### 46. Permutations
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        result = []
        def backtrack(first):
            if first == len(nums):
                result.append(list(nums))
                return
            for i in range(first, len(nums)):
                nums[i], nums[first] = nums[first], nums[i] 
                backtrack(first + 1)
                nums[i], nums[first] = nums[first], nums[i]  #restate
                
        backtrack(0)
        return result



### 40. Combination Sum II
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        result = []
        #a number can be NOT used multiple times in a combination
        counter = Counter(candidates)
        counter = [(c, counter[c]) for c in counter]
        def backtrack(remain, comb, start):
            if remain == 0:
                result.append(list(comb))
                return
            elif remain < 0:
                return
            for i in range(start, len(counter)):
                candidate, freq = counter[i]
                if freq <= 0:
                    continue
                comb.append(candidate)
                counter[i] = (candidate, freq-1)
                backtrack(remain-candidate, comb, i) #if remaining freq >=1, could use again
                counter[i] = (candidate, freq)  #restate
                comb.pop()  #restate
        
        backtrack(target, [], 0)        
        return result
        


### 90. Subsets II
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        result = []
        #a number can be NOT used multiple times in a combination
        counter = Counter(nums)
        counter = [(c, counter[c]) for c in counter]
        def backtrack(comb, start):
            result.append(list(comb))
            if len(comb) >= len(nums):
                return
            for i in range(start, len(counter)):
                candidate, freq = counter[i]
                if freq <= 0:
                    continue
                comb.append(candidate)
                counter[i] = (candidate, freq-1)
                backtrack(comb, i) #if remaining freq >=1, could use again
                counter[i] = (candidate, freq)  #restate
                comb.pop()  #restate
        
        backtrack([], 0)        
        return result


### 47. Permutations II
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        result = []
        counter = Counter(nums)
        counter = [(c, counter[c]) for c in counter]
        def backtrack(comb):
            if len(comb) == len(nums):
                result.append(list(comb))
                return
            for i in range(len(counter)):
                candidate, freq = counter[i] 
                if freq <= 0:
                    continue
                comb.append(candidate)
                counter[i] = (candidate, freq-1)
                backtrack(comb)
                comb.pop()
                counter[i] = (candidate, freq)
                
        backtrack([])
        return result


### 22. Generate Parentheses
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        result = []
        def backtrack(comb, left, right):
            if len(comb) == 2 * n:
                result.append(''.join(comb)) 
                return
            if left < n:
                comb.append("(")
                backtrack(comb, left+1, right)
                comb.pop() #restate
            if right < left:
                comb.append(")")
                backtrack(comb, left, right+1)
                comb.pop() #restate
                
        backtrack([], 0, 0)
        return result


### 93. Restore IP Addresses
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        def backtrack(comb, start):
            if len(comb) == 4:
                if start == len(s):
                    result.append('.'.join(comb))
                return
            for i in range(start, min(start+3, len(s))):
                if s[start] == '0' and i > start:
                    continue
                if 0 <= int(s[start:i+1]) <= 255:
                    comb.append(s[start: i+1])
                    backtrack(comb, i+1)
                    comb.pop()
        
        result = []
        backtrack([], 0)
        
        return result


### 17. Letter Combinations of a Phone Number
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        letters = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl", 
                   "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"}
        
        def backtrack(comb, start):
            if len(comb) == len(digits):
                result.append(''.join(comb))
                return
            
            possible_letters = letters[digits[start]]
            for letter in possible_letters:
                comb.append(letter)
                backtrack(comb, start+1)
                comb.pop()
        
        result = []
        
        if len(digits) == 0: #corner case
            return result
        
        backtrack([], 0)
        
        return result
            

### 842. Split Array into Fibonacci Sequence
class Solution:
    def splitIntoFibonacci(self, num: str) -> List[int]:
        def backtrack(comb, start):
            if self.result:
                return
            if comb:
                if len(comb[-1]) > 1 and comb[-1][0] == "0":
                    return
                if int(comb[-1]) > 2 ** 31 - 1:
                    return
                if len(comb) > 2:
                    if int(comb[-3]) + int(comb[-2]) != int(comb[-1]):
                        return
                    if start == len(num):
                        self.result = [int(i) for i in comb]
                        return
            for j in range(start, len(num)):
                comb.append(num[start:j + 1])
                backtrack(comb, j + 1)
                comb.pop()

        self.result = None
        backtrack([], 0)
        return self.result


### 89. Gray Code
class Solution:
    def grayCode(self, n: int) -> List[int]:
        def backtrack(comb, start):
            if self.result: return
            if len(comb) == length:
                self.result= list(comb)
            for i in range(n):
                shift = 1<<i  #shift bit position
                new = start ^ shift #get shift bit value
                if new not in seen:
                    seen.add(new)
                    comb.append(new)
                    backtrack(comb, new)
                    seen.remove(new)
                    comb.pop()
        
        length = 2**n  
        seen=set([0]) #init from 0
        self.result=[]
        backtrack([0], 0)
        return self.result


### 526. Beautiful Arrangement
class Solution:
    def countArrangement(self, n: int) -> int:
        count = 0
        def backtrack(comb, curr_index):
            nonlocal count
            if curr_index == n:
                count += 1
                return
            for num in range(1, n+1):
                if num in comb:
                    continue
                if (num % (curr_index+1) == 0) or ((curr_index+1) % num == 0):
                    comb.append(num)
                    backtrack(comb, curr_index +1)
                    comb.pop()
                    
        backtrack([], 0)
        return count


### 1239. Maximum Length of a Concatenated String with Unique Characters
class Solution:
    def maxLength(self, arr: List[str]) -> int:
        new_arr = []
        for each in arr:
            if len(set(each)) == len(each):
                new_arr.append(each)
        if len(new_arr) == 0:
            return 0
        
        longest = 0
        def backtrack(comb, start):
            nonlocal longest
            concatenate = ''.join(comb)
            
            if len(concatenate) == len(set(concatenate)):
                longest = max(longest, len(concatenate))
                
            for i in range(start, len(new_arr)):
                if len(concatenate) > len(set(concatenate)):
                    break
                comb.append(new_arr[i])
                backtrack(comb, i+1)
                comb.pop()
                
        backtrack([], 0)
        return longest


### 131. Palindrome Partitioning
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        def isPalindrome(start, end):
            while start <= end:
                if s[start] != s[end]:
                    return False
                else:
                    start+=1
                    end-=1
            return True
        
        def backtrack(comb, i):
            if i == len(s):
                result.append(list(comb))
            for j in range(i, len(s)):
                if isPalindrome(i, j):
                    comb.append(s[i:j+1])
                    backtrack(comb, j+1)
                    comb.pop()
                    
        result = []
        backtrack([], 0)
        return result


### 1593. Split a String Into the Max Number of Unique Substrings
class Solution:
    def maxUniqueSplit(self, s: str) -> int:
        def backtrack(comb, i):
            nonlocal longest
            if i == len(s):
                longest = max(longest, len(comb))
            
            for j in range(i, len(s)):
                sub = s[i:j+1]
                if sub not in comb:
                    comb.append(sub)
                    backtrack(comb, j+1)
                    comb.pop()
                    
        longest = 0
        backtrack([], 0)
        
        return longest


### 967. Numbers With Same Consecutive Differences
class Solution:
    def numsSameConsecDiff(self, n: int, k: int) -> List[int]:
        def backtrack(comb):
            if len(comb) == n:
                if comb[0] == 0:
                    return
                s = int(''.join(map(str, comb)))
                result.append(s)
                return
            for i in range(10):                
                if len(comb)==0:
                    comb.append(i)
                    backtrack(comb)
                    comb.pop()
                else:
                    if abs(comb[-1] - i) == k:
                        comb.append(i)
                        backtrack(comb)
                        comb.pop()
        result=[]
        backtrack([])
        return result


### 357. Count Numbers with Unique Digits
class Solution:
    def countNumbersWithUniqueDigits(self, n: int) -> int:
        result = 0
        str_num = [str(i) for i in range(10)]
        def backtrack(count, comb):
            nonlocal result
            if count == n:
                result += 1
                return
            for num in str_num:
                if not comb:
                    comb = comb + num
                    backtrack(count+1, comb)
                    comb = comb[:-1]
                else:
                    nlz_path = str(int(comb)) #to check if num in
                    if (num not in nlz_path) or (nlz_path == "0"):
                        comb = comb + num
                        backtrack(count+1, comb)
                        comb = comb[:-1]
        
        backtrack(0, '')
        return result
        

### 254. Factor Combinations
class Solution:
    def getFactors(self, n: int) -> List[List[int]]:
        if n == 1: 
            return []
        result = []
        def backtrack(comb, start, target):
            if len(comb)>0:
                result.append(comb + [target])
                
            for i in range(start, int(math.sqrt(target))+1): 
                if target %i == 0:
                    comb.append(i)
                    backtrack(comb, i, target//i)
                    comb.pop()
        
        backtrack([], 2, n)
        return result



### 797. All Paths From Source to Target        
class Solution:
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        target = len(graph) - 1
        result = []
        
        def backtrack(comb, currNode):
            if currNode == target: #last node
                result.append(list(comb))
                return
            for nextNode in graph[currNode]:
                comb.append(nextNode)
                backtrack(comb, nextNode)
                comb.pop()
                
        backtrack([0], 0)
        return result
        

        
### 291. Word Pattern II
class Solution:
    def wordPatternMatch(self, pattern, s):
        """
        :type pattern: str
        :type str: str
        :rtype: bool
        """
        def backtrack(i, j, ptable, stable):
            if i == len(pattern) and j == len(s):
                return True
            elif i == len(pattern) or j == len(s):
                return False
            else:
                p, added = pattern[i], False
                for k in range(j, len(s)):
                    word = s[j:k+1]
                    if (p in ptable and ptable[p] != word) or (word in stable and stable[word] != p):
                        continue
                    if p not in ptable and word not in stable:
                        ptable[p], stable[word], added = word, p, True
                    remainder = backtrack(i+1, k+1, ptable, stable)
                    if added:
                        del ptable[p]
                        del stable[word]
                    if remainder:
                        return True
            return False
        
        return backtrack(0, 0, {}, {})


### 698. Partition to K Equal Sum Subsets
class Solution:
    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        targetSum = sum(nums) 
        if targetSum % k != 0 :
            return False
        else:
            targetSum = targetSum//k
            
        nums.sort(reverse=True)
        visited = [False] * len(nums)
        
        def backtrack(remaining_k, cur_sum, next_index):
            if remaining_k == 1:
                return True
            
            if cur_sum == targetSum:
                return backtrack(remaining_k - 1, 0, 0)
            
            for i in range(next_index, len(nums)):
                if not visited[i] and cur_sum + nums[i] <= targetSum:
                    cur_sum = cur_sum + nums[i]
                    visited[i] = True
                    if backtrack(remaining_k, cur_sum, i + 1): #why need to return
                        return True
                    cur_sum = cur_sum - nums[i]
                    visited[i] = False
                    
            return False 
        
        return backtrack(k,  0,  0)
        