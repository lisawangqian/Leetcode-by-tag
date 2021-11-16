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


