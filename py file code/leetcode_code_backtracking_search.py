
### Combination

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
        

    