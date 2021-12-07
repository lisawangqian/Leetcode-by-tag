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
        


### 1405. Longest Happy String
class Solution:
    def longestDiverseString(self, a: int, b: int, c: int) -> str:
        count = []
        if a != 0:
            heapq.heappush(count, (-a, 'a'))
        if b != 0:
            heapq.heappush(count, (-b, 'b'))
        if c != 0:
            heapq.heappush(count, (-c, 'c'))
        s = []
        while count:
            first, char1 = heapq.heappop(count) # char with most rest numbers
            if len(s) >= 2 and s[-1] == s[-2] == char1: # check whether this char is the same with previous two
                if not count: # if there is no other choice, just return
                    break
                second, char2 = heapq.heappop(count) # char with second most rest numbers
                s.append(char2)
                second += 1 #negate for largest count
                if second < 0: 
                    heapq.heappush(count, (second, char2))
                heapq.heappush(count, (first, char1)) # also need to put this part back to heap
            else:
			#  situation that this char can be directly added to answer
                s.append(char1)
                first += 1 #negate for largest count
                if first < 0:
                    heapq.heappush(count, (first, char1))
        
        return ''.join(s)
            


### 373. Find K Pairs with Smallest Sums          
class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        res = []
        if not nums1 or not nums2 or not k:
            return res
        
        pairs = []
        visited = set()
        
        heapq.heappush(pairs, (nums1[0] + nums2[0], 0, 0))
        
        visited.add((0, 0))
        
        while len(res) < k and pairs:
            _, i, j = heapq.heappop(pairs)
            res.append([nums1[i], nums2[j]])
            
            if i+1 < len(nums1) and (i+1, j) not in visited:
                heapq.heappush(pairs, (nums1[i+1] + nums2[j], i+1, j))
                visited.add((i+1, j))
            
            if j+1 < len(nums2) and (i, j+1) not in visited:
                heapq.heappush(pairs, (nums1[i] + nums2[j+1], i, j+1))
                visited.add((i, j+1))
                
        return res



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



### 1834. Single-Threaded CPU
class Solution:
    def getOrder(self, tasks: List[List[int]]) -> List[int]:
        res = []
        #start_time, process_time, original_index
        tasks = sorted([(t[0], t[1], i) for i, t in enumerate(tasks)])
        i = 0
        process = []
        start = tasks[0][0]
        while len(res) < len(tasks):
            while (i < len(tasks)) and (tasks[i][0] <= start):
                #by process_time, original_index
                heapq.heappush(process, (tasks[i][1], tasks[i][2])) 
                i += 1
            if process:
                t_process, original_index = heapq.heappop(process)
                start += t_process  #so next round check starting time <= this
                res.append(original_index)
            elif i < len(tasks): #idle
                start = tasks[i][0]
                
        return res



### 1353. Maximum Number of Events That Can Be Attended
class Solution:
    def maxEvents(self, events: List[List[int]]) -> int:
        if not events:
            return 0
        
        end_time = [] # store end time of open events
        events.sort(key = lambda x: x[0])
        events = sorted([(t[0], t[1]) for  t in events])
        count = 0
        i, n  = 0, len(events)
        cur_day = 0 #current day
        
        while i < n or end_time:
            # Iterate over events that have 
            # starting <= current day
            # And add their end days to min heap
            while i < n and events[i][0] <= cur_day:
                heapq.heappush(end_time, events[i][1])
                i += 1
            # No day exists in heap, 
            # Pick current starting day
            if end_time:
                # Pop the earliest ending event
                heapq.heappop(end_time) #attending this today
                # Increase count
                count += 1
                # Increase current day (to mark one day of event attended)
                cur_day += 1
                # If current day has exceeded any 
                # event end time, we cant attend it, 
                # Pop it
                while end_time and cur_day > end_time[0]:
                    heapq.heappop(end_time)
            elif i < n:
                cur_day = events[i][0]
                
        return count



### 355. Design Twitter
class Twitter:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.timer = itertools.count(step=-1)  #smaller, latest
        self.tweets = collections.defaultdict(collections.deque)
        self.followees = collections.defaultdict(set)

    def postTweet(self, userId: int, tweetId: int) -> None:
        """
        Compose a new tweet.
        """
        #collections.deque: appendleft;  itertools.count: next
        #timer is the sorting criteria and smallest (negate so latest)
        self.tweets[userId].appendleft((next(self.timer), tweetId))

    def getNewsFeed(self, userId: int) -> List[int]:
        """
        Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent.
        """
        # | - combination of userId (convert to set) and userId followees (set)
        tweets = heapq.merge(*(self.tweets[u] for u in (self.followees[userId] | {userId}) ))
        return [t for _, t in itertools.islice(tweets, 10)]
        

    def follow(self, followerId: int, followeeId: int) -> None:
        """
        Follower follows a followee. If the operation is invalid, it should be a no-op.
        """
        self.followees[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        """
        Follower unfollows a followee. If the operation is invalid, it should be a no-op.
        """
        self.followees[followerId].discard(followeeId)
        

# Your Twitter object will be instantiated and called as such:
# obj = Twitter()
# obj.postTweet(userId,tweetId)
# param_2 = obj.getNewsFeed(userId)
# obj.follow(followerId,followeeId)
# obj.unfollow(followerId,followeeId)