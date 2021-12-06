import heapq, collections

### 703. Kth Largest Element in a Stream
class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.nums = nums
        if self.nums:
            heapq.heapify(self.nums)
            while len(self.nums) > self.k:  #only maintain k highest
                heapq.heappop(self.nums)    

    def add(self, val: int) -> int:
        heapq.heappush(self.nums, val)
        if len(self.nums) > self.k:
            heapq.heappop(self.nums)
        
        return self.nums[0]

# Your KthLargest object will be instantiated and called as such:
# obj = KthLargest(k, nums)
# param_1 = obj.add(val)



### 1046. Last Stone Weight
class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        stones = [-i for i in stones]
        if stones:
            heapq.heapify(stones)
        
        while len(stones) > 1:
            q1 = heapq.heappop(stones)
            q2 = heapq.heappop(stones)
            if q1 != q2:
                heapq.heappush(stones, q1-q2)
                
        return -heapq.heappop(stones) if stones else 0



### 1464. Maximum Product of Two Elements in an Array
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        nums = [-i for i in nums]
        heapq.heapify(nums)
        q1 = heapq.heappop(nums)
        q2 = heapq.heappop(nums)
        return (q1+1) * (q2+1)



### 506. Relative Ranks
class Solution:
    def findRelativeRanks(self, score: List[int]) -> List[str]:
        ans = {v:i for i, v in enumerate(score)}
        score = [-v for v in score]
        heapq.heapify(score)
        n = len(score)
        for i in range(1, n+1):
            place = str(i)
            if i == 1:
                place = "Gold Medal"
            elif i == 2:
                place = "Silver Medal"
            elif i == 3:
                place = "Bronze Medal"
            ans[-heapq.heappop(score)] = place
       
        return list(ans.values())
            


### 215. Kth Largest Element in an Array (similar as 703)
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        heapq.heapify(nums)
        while len(nums) > k:  #only maintain k highest
            heapq.heappop(nums) 
        return nums[0]


### 1167. Minimum Cost to Connect Sticks
class Solution:
    def connectSticks(self, sticks: List[int]) -> int:
        
        heapq.heapify(sticks)
        cost = 0
        while len(sticks) > 1:
            q1 = heapq.heappop(sticks)
            q2 = heapq.heappop(sticks)
            heapq.heappush(sticks, q1+q2)
            cost+= (q1+q2)
            
        return cost



### 973. K Closest Points to Origin
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        return heapq.nsmallest(k, points, lambda x: x[0]*x[0] + x[1]*x[1])



### 347. Top K Frequent Elements
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = collections.Counter(nums) 
        return heapq.nlargest(k, count, lambda x: count[x])


### 692. Top K Frequent Words
class Solution:
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        count = collections.Counter(words) 
        return heapq.nsmallest(k, count, lambda x: (-count[x], x))


### 451. Sort Characters By Frequency
class Solution:
    def frequencySort(self, s: str) -> str:
        k = len(s)
        count = collections.Counter(s) 
        #return "".join([count[x] * x for x in heapq.nsmallest(k, count, lambda x: (-count[x], x))])
        return "".join([count[x] * x for x in heapq.nlargest(k, count, lambda x: count[x])])
        


### 1642. Furthest Building You Can Reach
class Solution:
    def furthestBuilding(self, heights: List[int], bricks: int, ladders: int) -> int:
        ladder_allocation = []
        
        for i in range(len(heights) - 1):
            climb = heights[i+1] - heights[i] #need to go to i+1
            #jump down; skip
            if climb <= 0:
                continue
            heapq.heappush(ladder_allocation, climb)
            #if ladder is not enough;
            #we will need to take a climb out of ladder_allocations
            #try to use bricks
            if len(ladder_allocation) > ladders:
                bricks -= heapq.heappop(ladder_allocation)
            
            # If this caused bricks to go negative, we can't get to i + 1
            if bricks < 0:
                return i
            
        return len(heights) - 1



### 253. Meeting Rooms II
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        if not intervals:
            return 0
        
        free_rooms = []
        intervals.sort(key = lambda x: x[0])
        heapq.heappush(free_rooms, intervals[0][1]) #ending time
        for start, end in intervals[1:]:
            if free_rooms[0] <= start:  #room freeup
                heapq.heappop(free_rooms)
            heapq.heappush(free_rooms, end)
            
        return len(free_rooms)
            


### 378. Kth Smallest Element in a Sorted Matrix
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        N = len(matrix)
        minHeap = []  #each row start from first one
        for r in range(min(k, N)):
            minHeap.append((matrix[r][0], r, 0))
        heapq.heapify(minHeap)
        
        while k>0:  
            element, r, c = heapq.heappop(minHeap) #find current smallest
            if c < N-1: #scan the row and add the column index
                heapq.heappush(minHeap, (matrix[r][c+1], r, c+1))
            k-=1
            
        return element
                
        

### 767. Reorganize String
class Solution:
    def reorganizeString(self, s: str) -> str:
        if not s:
            return ""
        
        count = collections.Counter(s)
        count = [(-value,key) for key,value in count.items()]
        heapq.heapify(count)
        
        prev_a, prev_b = heapq.heappop(count)  #biggest count
        res = prev_b
        prev_a += 1 #value is negated for sorting
        while count:
            a, b = heapq.heappop(count)
            res += b
            a += 1 #value is negated for sorting
            if prev_a < 0:  #value is negated for sorting
                heapq.heappush(count, (prev_a, prev_b))
            prev_a, prev_b = a, b
            
        if len(res) != len(s): return ""
        return res        
        


            

