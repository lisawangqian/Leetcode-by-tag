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