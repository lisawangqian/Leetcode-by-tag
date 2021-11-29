### 20. Valid Parentheses

class Solution:
    def isValid(self, s: str) -> bool:
        
        stack = []  #last-in first-out 
        mapping = {"}":"{", "]":"[", ")":"("}
        for char in s:
            if char in mapping:
                top_element = stack.pop() if stack else "#"
                if mapping[char] != top_element:
                    return False
            else:
                stack.append(char)
                
        return (not stack)


### 1047. Remove All Adjacent Duplicates In String
class Solution:
    def removeDuplicates(self, s: str) -> str:
        stack = []
        for char in s:
            if stack and char == stack[-1]:
                stack.pop()
            else:
                stack.append(char)
        return ''.join(stack)
        

### 1021. Remove Outermost Parentheses
class Solution:
    def removeOuterParentheses(self, s: str) -> str:
        stack = []
        bd = []
        if not s:
            return ""
        
        for i, char in enumerate(s):
            if char == "(":
                stack.append(char)
            else:
                stack.pop()
                if not stack:
                    bd.append(i)
        prev = 0
        ans = ""
        for i in range(len(bd)):
            curr = s[prev+1:bd[i]]
            ans += curr
            prev = bd[i] + 1
        return ans


### 1614. Maximum Nesting Depth of the Parentheses

class Solution:
    def maxDepth(self, s: str) -> int:
        stack = []
        depth = 0
        for c in s:
            if c == "(":
                stack.append(c)
                depth = max(depth, len(stack))
            if c == ")":
                stack.pop()

        return depth


### 682. Baseball Game
class Solution:
    def calPoints(self, ops: List[str]) -> int:
        stack = []
        for each in ops:
            if each.isdigit() or each.startswith('-'):
                stack.append(int(each))
            elif each == '+':
                stack.append(stack[-1] + stack[-2])
            elif each == 'D':
                stack.append(2*stack[-1])
            else:
                stack.pop()
                
        if stack:
            return sum(stack)
        return 0
        

### 844. Backspace String Compare
class Solution:
    def backspaceCompare(self, s: str, t: str) -> bool:
        
        stack_s = []
        for c in s:
            if c == '#' and stack_s:
                stack_s.pop()
            elif c == "#":
                continue
            else:
                stack_s.append(c)
                
        stack_t = []
        for c in t:
            if c == '#' and stack_t:
                stack_t.pop()
            elif c == "#":
                continue
            else:
                stack_t.append(c)
                
        return ''.join(stack_s) == ''.join(stack_t)
                


### 496. Next Greater Element I     
class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        dic_map = {}
        stack = []
        for i in range(len(nums2)):
            while stack and nums2[i] > stack[-1]:
                dic_map[stack.pop()] = nums2[i] #element map with next bigger
            stack.append(nums2[i])  #stack matain element with no next bigger
            
        while stack:
            dic_map[stack.pop()] = -1
            
        res = []
        for num in nums1:
            res.append(dic_map[num])
        return res  



### 1475. Final Prices With a Special Discount in a Shop
class Solution:
    def finalPrices(self, prices: List[int]) -> List[int]:
        
        stack = []
        for i, p in enumerate(prices):
            #find previous not discounted (in stack) but will be discouted by current price
            while stack and prices[stack[-1]] >= p: 
                last = stack.pop()
                prices[last] = prices[last] - p
            stack.append(i)
                 
        return prices
        


###155. Min Stack
class MinStack:

    def __init__(self):
        self.stack = []
        
    def push(self, val: int) -> None:
        if not self.stack:
            self.stack.append((val, val))
        else:
            self.stack.append((val, min(self.stack[-1][1], val)))
    
    def pop(self) -> None:
        if self.stack:
            self.stack.pop()
        
    def top(self) -> int:
        if self.stack:
            return self.stack[-1][0]
        
    def getMin(self) -> int:
        if self.stack:
            return self.stack[-1][1]

# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()

### 716. Max Stack
class MaxStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        
    def push(self, x: int) -> None:
        if not self.stack:
            self.stack.append((x, x))
        else:
            self.stack.append((x, max(x, self.stack[-1][1])))
        
    def pop(self) -> int:
        if self.stack:
            return self.stack.pop()[0]
        
    def top(self) -> int:
        if self.stack:
            return self.stack[-1][0]

    def peekMax(self) -> int:
        if self.stack:
            return self.stack[-1][1]
    
    def popMax(self) -> int:
        if self.stack:
            max_v = self.stack[-1][1]
            redo_c = []

            while self.stack[-1][0] != max_v:
                redo_c.append(self.pop())
            self.pop()  #remove max element
            
            for each in reversed(redo_c):
                self.push(each)
            
            return max_v
        
# Your MaxStack object will be instantiated and called as such:
# obj = MaxStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.peekMax()
# param_5 = obj.popMax()

### 232. Implement Queue using Stacks
class MyQueue:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.stack1 = []
        self.stack2 = []
        
    def push(self, x: int) -> None:
        """
        Push element x to the back of queue.
        """
        self.stack1.append(x)

    def pop(self) -> int:
        """
        Removes the element from in front of queue and returns that element.
        """
        self.peek()
        return self.stack2.pop()

    def peek(self) -> int:
        """
        Get the front element.
        """
        if not self.stack2: #tail of stack2 will be stack1 head
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        return self.stack2[-1]
        

    def empty(self) -> bool:
        """
        Returns whether the queue is empty.
        """
        return not self.stack1 and not self.stack2
        
# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()


### 225. Implement Stack using Queues
class MyStack:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.q1 = deque()
        self.q2 = deque()
        self._top = None
        
    def push(self, x: int) -> None:
        """
        Push element x onto stack.
        """
        self.q1.append(x)
        self._top = x
        

    def pop(self) -> int:
        """
        Removes the element on top of the stack and returns that element.
        """
        while len(self.q1) > 1:
            self._top = self.q1.popleft()
            self.q2.append(self._top)
            
        result = self.q1.popleft()
        self.q1, self.q2 = self.q2, self.q1
        return result
        

    def top(self) -> int:
        """
        Get the top element.
        """
        return self._top

    def empty(self) -> bool:
        """
        Returns whether the stack is empty.
        """
        return len(self.q1) == 0


# Your MyStack object will be instantiated and called as such:
# obj = MyStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.empty()



### 1190. Reverse Substrings Between Each Pair of Parentheses
class Solution:
    def reverseParentheses(self, s: str) -> str:
        stack = ['']
        for c in s:
            if c == '(':
                stack.append('')
            elif c == ')':  #reverse
                add = stack.pop()[::-1]
                stack[-1] += add
            else:
                stack[-1] += c
        
        return stack.pop()



### 394. Decode String
class Solution:
    def decodeString(self, s: str) -> str:
        stack = [''] 
        stackcnt = []
        k = 0
        for c in s:
            if c.isdigit():
                k = k * 10 + int(c)
            elif c == '[': #re-init 
                stackcnt.append(k)
                k = 0 
                stack.append('')  
            elif c == ']': #begin calculate 
                add = stack.pop()
                cnt = stackcnt.pop()
                stack[-1] += add * cnt
            else:  
                stack[-1] += c
        
        return stack.pop()



### 456. 132 Pattern
class Solution:
    def find132pattern(self, nums: List[int]) -> bool:
        if len(nums) < 3:
            return False
       
        min_array = [-1] * len(nums)
        min_array[0] = nums[0]
        for i in range(1, len(nums)):
            min_array[i] = min(min_array[i-1], nums[i])
        
        stack = []
        for j in range(len(nums)-1, -1, -1): #backward to find p3
            if nums[j] == min_array[j]: #if itself is minimum, p1
                continue
            while stack and stack[-1] <= min_array[j]: #stack top to main current p2, min_array[j]: current p1
                stack.pop()
            if stack and stack[-1] < nums[j]: #find p3
                return True
            stack.append(nums[j])

        return False


### 503. Next Greater Element II
class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        
        stack, res = [], [-1] * len(nums)
        for _ in range(2): #because it is circle so repeat twice
            for i in range(len(nums)-1, -1, -1):
                while stack and (nums[stack[-1]] <= nums[i]):
                    stack.pop()
                if stack and (res[i] == -1):
                    res[i] = nums[stack[-1]]
                stack.append(i) 
                
        return res