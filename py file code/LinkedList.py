### 206. Reverse Linked List ###

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev_node = None
        curr_node = head
        while curr_node:
            next_node = curr_node.next
            curr_node.next = prev_node
            prev_node = curr_node
            curr_node = next_node 
            
        return prev_node


### 141. Linked List Cycle ###

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        if not head:
            return False
        slow = head
        fast = head.next
        while slow != fast:
            if (not fast) or (not fast.next):
                return False
            slow = slow.next
            fast = fast.next.next
            
        return True


### 83. Remove Duplicates from Sorted List ###

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        curr = head
        while curr and curr.next:
            if curr.next.val == curr.val:
                curr.next = curr.next.next
            else:
                curr = curr.next
        return head


### 234. Palindrome Linked List ###

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        if not head:
            return True
        first_half_end = self.end_of_first_half(head)
        second_half_start = self.reverseList(first_half_end.next)
        
        first_position = head
        second_position = second_half_start
        while second_position:
            if second_position.val != first_position.val:
                return False
            second_position = second_position.next
            first_position = first_position.next
        return True
        
    def end_of_first_half(self, head):
        slow, fast = head, head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        return slow  
        
    def reverseList(self, head):
        prev_node = None
        curr_node = head
        while curr_node:
            next_node = curr_node.next
            curr_node.next = prev_node
            prev_node = curr_node
            curr_node = next_node
        return prev_node


### 203. Remove Linked List Elements ###
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        if not head:
            return head
        senti_node = ListNode() 
        senti_node.next = head
        prev_node = senti_node
        while prev_node.next:
            if prev_node.next.val == val:
                prev_node.next = prev_node.next.next
            else:
                prev_node = prev_node.next
        return senti_node.next


### 237. Delete Node in a Linked List ###
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        node.val = node.next.val
        node.next = node.next.next


### 876. Middle of the Linked List ###
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return head
        slow, fast = head, head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        if fast.next:
            return slow.next
        else:
            return slow


### 160. Intersection of Two Linked Lists ###
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        #one pointer goes to A first and then restart from B
        #the other pointer goes to B first then restart from A
        #they will meet in the intersect druing their repeat from the other path
        pA, pB = headA, headB
        while pA != pB:
            pA = headB if not pA else pA.next
            pB = headA if not pB else pB.next
            
        return pA


### 21. Merge Two Sorted Lists ###
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        prehead = ListNode()
        node = prehead
        while l1 and l2:
            if l1.val <= l2.val:
                node.next = l1
                l1 = l1.next
            else:
                node.next = l2
                l2 = l2.next
            node = node.next
        node.next = l1 if l1 else l2
        
        return prehead.next


### 1290. Convert Binary Number in a Linked List to Integer ###
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def getDecimalValue(self, head: ListNode) -> int:
        ans = 0
        while head:
            ans = ans*2 + head.val
            head = head.next
        return ans
        

### 1474. Delete N Nodes After M Nodes of a Linked List ###
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteNodes(self, head: ListNode, m: int, n: int) -> ListNode:
        count = 0
        curr_node = head
        while curr_node:
            count = 0
            prev_node = curr_node #init
            while curr_node and count < m:
                prev_node = curr_node  #m nodes
                curr_node = curr_node.next
                count+=1
            temp = curr_node #begin to delete node from here m+1
            count = 0
            while temp and count < n:
                temp = temp.next #start to include after n nodes
                count+=1
            if prev_node: #m mode delete n and continue
                prev_node.next = temp
            curr_node = temp #next round from here
            
        return head


### 92. Reverse Linked List II ###
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseBetween(self, head: ListNode, left: int, right: int) -> ListNode:
        if not head:
            return None
        
        senti_node = ListNode()
        senti_node.next = head
        curr_node, prev_node = senti_node.next, senti_node
        
        while left > 1:
            prev_node = curr_node   #final state previous node of left position node
            curr_node = curr_node.next  #final state left position
            left, right = left-1, right-1
            
        tail_node, connect_node = curr_node, prev_node
        #start as from left = 1
        
        while right:  #reverse the linked list from right position to left position
            next_node = curr_node.next
            curr_node.next = prev_node
            prev_node = curr_node   #final state head of reversed link list
            curr_node = next_node  #next node in the linked list
            right-=1  
            
        connect_node.next = prev_node
        #original left position node
        tail_node.next = curr_node
        
        return senti_node.next
            

### 43. Reorder List ###
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reorderList(self, head: ListNode) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        if not head:
            return
        
        slow, fast = head, head
        while fast and fast.next: #slow is the middle(later if tied or center)
            slow = slow.next
            fast = fast.next.next
        
        prev_node, curr_node = None, slow
        
        while curr_node: #reverse list
            next_node = curr_node.next
            curr_node.next = prev_node
            prev_node = curr_node #final head of reverse list
            curr_node = next_node
            
        l1, l2 = head, prev_node
        
        while l2.next: #l2 is same or 1 longer than l1
            temp = l1.next
            l1.next = l2
            l1 = temp #l1 move to next
            
            temp = l2.next
            l2.next = l1
            l2 = temp #l2 move to next


### 82. Remove Duplicates from Sorted List II ###
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        sentinel = ListNode(0, head)  #put a sentinel before head to avoid edge case
        prev_node, curr_node = sentinel, head
        
        while curr_node:
            if curr_node.next and curr_node.val == curr_node.next.val:
                #until next val different
                while curr_node.next and curr_node.val == curr_node.next.val:
                    curr_node = curr_node.next
                
                prev_node.next = curr_node.next
            else:
                prev_node = prev_node.next
            curr_node = curr_node.next
            
        return sentinel.next


### 147. Insertion Sort List ###
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def insertionSortList(self, head: ListNode) -> ListNode:
        result = ListNode()  #empty link list to maintain sorted part but with an sentinel node
        curr = head
        
        while curr: #iterate over each node
            prev = result #back to start of maintained sorted list
            #until prev.next is prev.next >= curr.val -> prev, curr, prev.next after insertion
            while prev.next and prev.next.val < curr.val:
                prev = prev.next   #find curr position after prev
                
            next_node = curr.next  #tempory container of next iterated node
            
            #link the curr to prev.next in the maintained sorted list
            curr.next = prev.next 
            
            #link the prev to curr in the maintained sorted list
            prev.next = curr 
            
            curr = next_node #move node
                
                
        return result.next


### 19. Remove Nth Node From End of List ###
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        prehead = ListNode(None)
        prehead.next = head
        length = 0
        curr = head
        while curr:
            curr = curr.next
            length+=1
        remove = length - n
        prev, curr = prehead, head
        while curr and remove:
            remove -= 1
            prev = curr
            curr = curr.next
        prev.next = curr.next
        return prehead.next


### 148. Sort List ###
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        def getSize(head):
            length = 0
            while head:
                length+=1
                head = head.next
                
            return length
        
        def split(head, size):
            for i in range(size - 1):
                if not head:
                    break
                head = head.next  #node end point at size 
                
            if not head:
                return None
            
            next_start, head.next = head.next, None #disconnect node end point at size
            
            return next_start
        
        def merge(l1, l2, prehead):
            tail = prehead
            while l1 and l2:
                if l1.val <= l2.val:
                    tail.next, l1 = l1, l1.next
                else:
                    tail.next, l2 = l2, l2.next
                tail = tail.next
            
            tail.next = l1 if l1 else l2
            while tail.next:
                tail = tail.next  #find tail
                
            return tail
        
        #sort begin
        if not head or not head.next:
            return head
        
        total_length = getSize(head)
        prehead = ListNode(0)
        prehead.next = head
        start, prehead_start, size = None, None, 1 #init
        
        while size < total_length:
            prehead_start = prehead
            start = prehead_start.next
            while start:
                left = start
                #return is next_start
                right = split(left, size) #start from left, size 
                start = split(right, size)#start from right, size
                #return is merge list tail
                prehead_start = merge(left, right, prehead_start)
            size*=2
            
            
        return prehead.next


### 86. Partition List ###   ###same method as 328. Odd Even Linked List
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def partition(self, head: ListNode, x: int) -> ListNode:
        #one use to move pointer and the other maintain the head of linklist        
        before = before_head = ListNode(0) #sentinel  
        #one use to move pointer and the other maintain the head of linklist
        after = after_head = ListNode(0)  #sentinel
        
        while head:
            if head.val < x:
                before.next = head
                before = before.next
            else:
                after.next = head
                after = after.next
            head = head.next
        after.next = None
        before.next = after_head.next #move 2nd sentinel node
        
        return before_head.next


### 328. Odd Even Linked List ###  ### 86. Partition List
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def oddEvenList(self, head: ListNode) -> ListNode:
        oddsHead = odds = ListNode() #sentinel node
        evensHead = evens = ListNode() #sentinel node
        isOdd = True #current index odd or not
        while head:
            if isOdd:
                odds.next = head
                odds = odds.next
            else:
                evens.next = head
                evens = evens.next
            isOdd = not isOdd
            head = head.next
            
        evens.next = None  #tail
        odds.next = evensHead.next #connect
        return oddsHead.next


### 61. Rotate List ###
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        
        if not head or not head.next:
            return head
        
        #get list length
        old_tail = head
        length = 1
        #need to check next because need to stop at last node (not null)
        while old_tail.next: 
            old_tail = old_tail.next
            length+=1
        
        old_tail.next = head  #tail need to connect to head to be circle
        
        #find new tail (length-k-1)th node
        #new head: (length-k)th node
        new_tail = head
        k = k % length #remainder
        for i in range(length-k-1):
            new_tail = new_tail.next #stop at the tail
            
        new_head = new_tail.next  #because it is a circle now
        new_tail.next = None #disconnect circle
        
        return new_head


### 142. Linked List Cycle II ###
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        
        if head is None:
            return None
        
        slow, fast = head, head
        intersect = None
        while fast and fast.next: #get intersect node
            slow = slow.next
            fast = fast.next.next
            if fast == slow:
                intersect = slow
                break
        
        if intersect is None:
            return None
        
        ptr1, ptr2 = head, intersect #two pointers
        while ptr1 != ptr2:
            ptr1 = ptr1.next
            ptr2 = ptr2.next
            
        return ptr1
        

### 138. Copy List with Random Pointer ###
"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""
class Solution:
    def __init__(self):
        self.visited = {}
        
    def getCloneNode(self, node):
        if node:
            if node in self.visited:
                return self.visited[node]
            else:
                self.visited[node] = Node(node.val, None, None)
                return self.visited[node]
        return None
                
    def copyRandomList(self, head: 'Node') -> 'Node':
        
        if not head:
            return head
        
        old_node = head
        new_node = Node(old_node.val, None, None)
        self.visited[old_node] = new_node
        
        while old_node:
            new_node.random = self.getCloneNode(old_node.random)
            new_node.next = self.getCloneNode(old_node.next)
            
            old_node = old_node.next
            new_node = new_node.next
            
        return self.visited[head]  


### 24. Swap Nodes in Pairs ###
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prehead = ListNode()
        prehead.next = head
        prev, curr = prehead, head
        while curr and curr.next:
            next_node = curr.next.next
            curr, curr.next = curr.next, curr
            prev.next = curr
            curr.next.next = next_node
            curr = curr.next.next
            prev = prev.next.next
            
        return prehead.next


### 725. Split Linked List in Parts ###
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def splitListToParts(self, head: ListNode, k: int) -> List[ListNode]:
        length = 1
        curr = head
        while curr and curr.next:
            length+=1
            curr = curr.next
            
        width, remainder = length//k, length%k
       
        result = []
        curr = head #init
        for i in range(k):  #k parts
            add = 0
            if remainder > 0:
                add = 1  #number of elements in one part
            remainder-=1
            
            newhead = curr
            j = 0
            while curr and (j < width + add - 1):  #final state is the tail of current part
                curr = curr.next
                j+=1
            if curr:  #disconnect current part tail and also point to newhead of next part
                curr.next, curr = None, curr.next
                
            result.append(newhead) #every iteration in for loop, newhead pointer
            
        return result
        

### 1669. Merge In Between Linked Lists ###
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeInBetween(self, list1: ListNode, a: int, b: int, list2: ListNode) -> ListNode:
        prehead = ListNode()
        prehead.next = list1
        
        b = b - a + 1
        while list1 and a > 1:
            list1 = list1.next #tail before a
            a-=1
        tail = list1
        while list1 and b >= 0:
            list1 = list1.next  #head after b
            b-=1
            
        tail.next = list2
        while list2 and list2.next:
            list2 = list2.next
        list2.next = list1
        
        return  prehead.next


### 2. Add Two Numbers ###
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        prehead = ListNode()
        curr = prehead
        carry = 0  #current sum carry to next
        while l1 or l2:
            x, y = 0, 0
            if l1:
                x = l1.val
            if l2:
                y = l2.val
            s = carry + x + y
            carry = s//10  #for next
            curr.next = ListNode(s%10)
            curr = curr.next
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next
                
        if carry > 0:
            curr.next = ListNode(carry)
            
        return prehead.next
                

### 445. Add Two Numbers II ###  ### 206. Reverse Linked List + 2. Add Two Numbers 
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        prev_node = None
        curr_node = head
        while curr_node:
            next_node = curr_node.next
            curr_node.next = prev_node
            prev_node = curr_node
            curr_node = next_node
        head = prev_node
        return head  #from head
    
   
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        l1 = self.reverseList(l1)
        l2 = self.reverseList(l2)
        prehead = ListNode()
        curr = prehead
        carry = 0
        while l1 or l2:
            x, y = 0, 0
            if l1:
                x = l1.val
            if l2:
                y = l2.val
            s = carry + x + y
            carry = s//10
            curr.next = ListNode(s%10)
            curr = curr.next
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next
                
        if carry > 0:
            curr.next = ListNode(carry)
            
        return self.reverseList(prehead.next)


### 708. Insert into a Sorted Circular Linked List ###
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, next=None):
        self.val = val
        self.next = next
"""

class Solution:
    def insert(self, head: 'Node', insertVal: int) -> 'Node':
        if not head:
            head = Node(insertVal)
            head.next = head
            return head
        
        prev, curr = head, head.next
        findinsert = False
        while curr and (curr != head):
            if prev.val <= insertVal <= curr.val:  #case 1, only one senario
                findinsert = True
            elif curr.val < prev.val:   # case 2
                if insertVal >= prev.val or insertVal <= curr.val: #two senarios
                    findinsert = True
            if findinsert:  #condition to return
                prev.next = Node(insertVal, curr)
                return head 
            prev, curr = curr, curr.next
            
        prev.next = Node(insertVal, curr)    #case 3 curr == head circle back         
        return head


### 1171. Remove Zero Sum Consecutive Nodes from Linked List ###
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeZeroSumSublists(self, head: ListNode) -> ListNode:  #greedy skip
        prefix = 0
        seen = {}
        seen[0] = prehead = ListNode(0)
        prehead.next = head
        while head:
            prefix += head.val
            #if there is updated from node1 to nodeN means the nodes between 2- N sum 0
            #node1.next = nodeN.next (2-N will be deleted)
            seen[prefix] = head  #mark prefix sum and its latest node(could be updated) 
            head = head.next
        head = prehead
        prefix = 0
        while head:
            prefix += head.val
            #when generating hashmap:
            #if there is updated from node1 to nodeN (same prefix) means the nodes between 2- N sum 0
            #node1.next = nodeN.next (2-N will be deleted)
            #here head is node1, seen[prefix] is nodeN
            head.next = seen[prefix].next
            head = head.next
        return prehead.next


### 817. Linked List Components ###
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def numComponents(self, head: ListNode, nums: List[int]) -> int:
        setG = set(nums)
        component = 0
        while head:
            if head.val in setG and (head.next == None or head.next.val not in setG): #one disconnected comp
                component += 1
            head = head.next
        return component


### 1836. Remove Duplicates From an Unsorted Linked List ###
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteDuplicatesUnsorted(self, head: ListNode) -> ListNode:
        if not head:
            return None
        
        count = {}
        curr = head
        while curr:
            if curr.val in count:
                count[curr.val] +=1
            else:
                count[curr.val] = 1
            curr = curr.next
            
        prehead = ListNode()
        prehead.next = head
        prev, curr = prehead, head
        
        while curr:
            if count[curr.val] > 1:
                prev.next = curr.next #delete curr
                curr = curr.next 
            else:
                prev, curr = curr, curr.next
            
        return prehead.next
        
        
###3 1721. Swapping Nodes in a Linked List ###
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def swapNodes(self, head: ListNode, k: int) -> ListNode:
        prehead = pre_right = pre_left = ListNode(next=head) #sentinel
        right = left = head
        for i in range(k-1):
            #final state is pre_left: (k-1)th node (1 index original list)
            #left: (k)th node (1 index original list)
            pre_left, left = left, left.next 
            #left at k     

        null_checker = left  #move (length - k) steps to reach end

        while null_checker.next: #move (length - k) steps to reach end
            #right at length - k + 1
            pre_right, right = right, right.next #move
            null_checker = null_checker.next

        if left == right:  #corner case swap the middle node means no swap
            return head
        #swap left and right
        pre_left.next, pre_right.next = right, left
        left.next, right.next = right.next, left.next
        return prehead.next       


### 707. Design Linked List ###
class MyLinkedList:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.head = ListNode() #sentinel
        self.size = 0

    def get(self, index: int) -> int:
        """
        Get the value of the index-th node in the linked list. If the index is invalid, return -1.
        """
        if index >= self.size or index < 0:  #index and size diff 1
            return -1
        else:
            curr = self.head.next
            for i in range(index):
                curr = curr.next
            
            return curr.val
        
    def addAtHead(self, val: int) -> None:
        """
        Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list.
        """
        self.addAtIndex(0, val)
        

    def addAtTail(self, val: int) -> None:
        """
        Append a node of value val to the last element of the linked list.
        """
        self.addAtIndex(self.size, val)  #self.size is the original tail None, add before None
        
    def addAtIndex(self, index: int, val: int) -> None:
        """
        Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted.
        """
        if index > self.size:  # == self.size is added before None, add after tail
            return 
        if index < 0:
            return
        
        self.size += 1
        predecessor = self.head
        for i in range(index): #final is the node before index i due to sentinel
            predecessor = predecessor.next  
            
        to_add = ListNode(val)
        to_add.next = predecessor.next #original index i
        predecessor.next = to_add
            
    def deleteAtIndex(self, index: int) -> None:
        """
        Delete the index-th node in the linked list, if the index is valid.
        """
        if index >= self.size or index < 0:
            return 
        self.size -= 1
        predecessor = self.head
        for i in range(index): #final is the node before index i due to sentinel
            predecessor = predecessor.next
        predecessor.next = predecessor.next.next
            
# Your MyLinkedList object will be instantiated and called as such:
# obj = MyLinkedList()
# param_1 = obj.get(index)
# obj.addAtHead(val)
# obj.addAtTail(val)
# obj.addAtIndex(index,val)
# obj.deleteAtIndex(index)


### 146. LRU Cache ### Optional linkedlist design
class Node:
    def __init__(self, k, v):
        self.key = k
        self.val = v
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.dic = dict()  #node dictionary by key
        self.head = Node(0, 0)  #sentinal head
        self.tail = Node(0, 0)  #sentinal tail
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key):
        if key in self.dic:
            node = self.dic[key]
            #touched so move the node from somewhere to tail as frequent user
            self._remove(node)
            self._add(node) 
            return node.val
        return -1

    def put(self, key, value):
        if key in self.dic: #need to update value
            self._remove(self.dic[key]) #remove the node
        new_node = Node(key, value) #new node value
        self._add(new_node)  #add the new node to tail
        self.dic[key] = new_node
        if len(self.dic) > self.capacity: #if reach capacity, remove head(after head sentinel)
            rem_node = self.head.next
            self._remove(rem_node)
            del self.dic[rem_node.key]

    def _remove(self, node): #remove a node from the linked list
        p = node.prev
        n = node.next
        p.next = n
        n.prev = p

    def _add(self, node): #add the node to the tail(before tail sentinel)
        p = self.tail.prev
        p.next = node
        self.tail.prev = node
        node.prev = p
        node.next = self.tail

# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)


### 109. Convert Sorted List to Binary Search Tree ### Binary Search and Binary Tree
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def helperBST(self, array, l, r):
        if l > r:
            return None
        mid = l + (r-l)//2
        root = TreeNode(array[mid])
        if l == r:  #base case
            return root
        root.left = self.helperBST(array, l, mid-1)  #left subtree
        root.right = self.helperBST(array, mid+1, r) #right subtree
        return root
    
    def sortedListToBST(self, head):
        """
        :type head: ListNode
        :rtype: TreeNode
        """
        array = []
        p = head
        while p:
            array.append(p.val)
            p = p.next
        
        return self.helperBST(array, 0, len(array)-1)
            
            
### 430. Flatten a Multilevel Doubly Linked List ###
"""
# Definition for a Node.
class Node:
    def __init__(self, val, prev, next, child):
        self.val = val
        self.prev = prev
        self.next = next
        self.child = child
"""

class Solution:
    def flatten_dfs(self, prev, curr):
        ##### return the tail of the flatten list #####
        if not curr: #curr is None, prev is tail
            return prev
        #link two node prev and curr 
        curr.prev = prev  
        prev.next = curr
        
        temp = curr.next  #curr.next will be changed after recursive call of following
        tail = self.flatten_dfs(curr, curr.child)  #flatten curr and curr.child with curr.next = curr.child
        curr.child = None  #original child need to be clear
        
        #after flatten curr and its child, the list will flatten with curr.next
        return self.flatten_dfs(tail, temp) 
        
    def flatten(self, head: 'Node') -> 'Node':
        #child - left node;  next - right node of a binary tree -> traverse the tree in    preorder DFS
        if not head:
            return head
        
        prehead = Node(None, None, head, None)
        self.flatten_dfs(prehead, head)
        
        prehead.next.prev = None #detach sentinel from real head
        
        return prehead.next