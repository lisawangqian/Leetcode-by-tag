### DFS 

## 1466. Reorder Routes to Make All Paths Lead to the City Zero
class Solution:
    def minReorder(self, n: int, connections: List[List[int]]) -> int:
        roads = set()
        g = [[] for i in range(n)]
        for edge in connections:
            roads.add((edge[0], edge[1]))
            g[edge[0]].append(edge[1])
            g[edge[1]].append(edge[0])
        
        result = 0    
        visited = set()    
        def dfs(v, parent):
            nonlocal result
            if v in visited:
                return
            visited.add(v)
            result += (parent, v) in roads
            for nxt in g[v]:
                if nxt not in visited:
                    dfs(nxt, v)
                    
        dfs(0, -1)
        return result


## 1376. Time Needed to Inform All Employees
class Solution:
    def numOfMinutes(self, n: int, headID: int, manager: List[int], informTime: List[int]) -> int:
        subordinates = collections.defaultdict(list)
        for i, v in enumerate(manager):  #manager root and its children
            subordinates[v].append(i)
        
        
        def dfs(manager):
            if manager not in subordinates.keys():
                return 0
            return max([dfs(subordinate) + informTime[manager] for subordinate in subordinates[manager]])
        
        
        return dfs(headID)



### DFS CC ##

## 841. Keys and Rooms
#1)Recursion
class Solution:
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        
        visited = set()
        
        def dfs(v):
            if v in visited:
                return
            visited.add(v)
            for room in rooms[v]:
                if room not in visited:
                    dfs(room)
                    
        dfs(0)
        
        return len(visited) == len(rooms)
#2)Iterative
class Solution:
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        
        
        stack = [0]
        visited = set([0])
        while stack:
            v = stack.pop()
            for room in rooms[v]:
                if room not in visited:
                    stack.append(room)
                    visited.add(room)
        
        return len(rooms) == len(visited)


## 1319. Number of Operations to Make Network Connected
class Solution:
    def makeConnected(self, n: int, connections: List[List[int]]) -> int:
        
        if len(connections) < n-1:
            return -1
        
        g = [[] for i in range(n)]
        for edge in connections:
            g[edge[0]].append(edge[1])
            g[edge[1]].append(edge[0])
            
        visited = set()
        def dfs(v):
            if v in visited:
                return
            visited.add(v)
            for u in g[v]:
                if u not in visited:
                    dfs(u)
                    
        cnt = 0
        for i in range(n):
            if i in visited: continue
            dfs(i)
            cnt+=1
           
            
        return cnt - 1


## 323. Number of Connected Components in an Undirected Graph
class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        g = [[] for i in range(n)]
        for edge in edges:
            g[edge[0]].append(edge[1])
            g[edge[1]].append(edge[0])
            
        visited = set()
        def dfs(v):
            if v in visited:
                return
            visited.add(v)
            for u in g[v]:
                if u not in visited:
                    dfs(u)
                    
        cnt = 0
        for i in range(n):
            if i in visited: continue
            dfs(i)
            cnt+=1
           
            
        return cnt


## 1202. Smallest String With Swaps
#1)DFS
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        g = [[] for i in range(len(s))]
        s = list(s)
        for pair in pairs:
            g[pair[0]].append(pair[1])
            g[pair[1]].append(pair[0])
        visited = set()
        
        
        def dfs(v):
            if v in visited:
                return
            visited.add(v)
            idx.append(v)
            tmp.append(s[v])
            for nxt in g[v]:
                if nxt not in visited:
                    dfs(nxt)
      
        for i in range(len(s)):
            if i in visited: continue
            idx = []
            tmp = []
            dfs(i)
            tmp = sorted(tmp)
            idx = sorted(idx)
            for k in range(len(idx)):
                s[idx[k]] = tmp[k]
        
        return ''.join(s)

#2)Union
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        def find(v):
            if v not in p:
                p[v] = v
            if p[v] != v:
                p[v] = find(p[v])
            return p[v]
        
        def union(u, v):
            p[find(u)] = find(v)
        
        p = {}
        for edge in pairs:
            union(edge[0], edge[1])
        
        s = list(s)
        result, m = [], defaultdict(list)    
        for i in range(len(s)): 
            m[find(i)].append(s[i])
        
        for comp_id in m.keys(): 
            m[comp_id].sort(reverse=True) #reverse sort char in the list
            
        for i in range(len(s)): 
            result.append(m[find(i)].pop())
            
        return ''.join(result)



### DFS Grid ###

## 200. Number of Islands
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])
        
        def dfs(i, j):
            if i < 0 or j < 0 or i >= m or j >= n or grid[i][j] == '0':
                return
            grid[i][j] = '0'
            dfs(i + 1, j)
            dfs(i - 1, j)
            dfs(i, j + 1)
            dfs(i, j - 1)
        
        cnt = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '0': continue
                dfs(i, j)
                cnt+=1
                
        return cnt


## 1905. Count Sub Islands
class Solution:
    def countSubIslands(self, grid1: List[List[int]], grid2: List[List[int]]) -> int:
        m, n = len(grid2), len(grid2[0])

        def dfs(i, j):
            if i < 0 or j < 0 or i >= m or j >= n or grid2[i][j] == 0: 
                return
            
            grid2[i][j] = 0
            dfs(i + 1, j)
            dfs(i - 1, j)
            dfs(i, j + 1)
            dfs(i, j - 1)
        
        #erase nonsubland    
        for i in range(m):
            for j in range(n):
                if grid2[i][j]==1 and grid1[i][j]==0:
                    dfs(i,j)
                    
        cnt = 0
        for i in range(m):
            for j in range(n):
                if grid2[i][j] == 0: continue
                dfs(i, j)
                cnt+=1
                
        return cnt


## 1254. Number of Closed Islands
class Solution:
    def closedIsland(self, grid: List[List[int]]) -> int:
        
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        
        def dfs(i, j, val):
            if i < 0 or j < 0 or i >= m or j >= n or grid[i][j] == 1:
                return
            grid[i][j] = val
            dfs(i, j+1, val)
            dfs(i+1, j, val)
            dfs(i-1, j, val)
            dfs(i, j-1, val)
        
        #edge island including its connected comp is not closed islands so be replace by water
        for i in range(m):
            for j in range(n):
                if (i == 0 or j == 0 or i == m-1 or j == n-1) and grid[i][j] == 0:
                    dfs(i, j, 1)
                
        cnt = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 0:
                    dfs(i, j, 1)
                    cnt += 1
                    
        return cnt


## 695. Max Area of Island
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        
        def dfs(i, j):
            if i < 0 or j < 0 or i >= m or j >= n or grid[i][j] == 0:
                return 0
            grid[i][j] = 0
            return 1 + dfs(i + 1, j) + dfs(i - 1, j) + dfs(i, j + 1) + dfs(i, j - 1)
        
        
        max_area = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 0: continue
                area = dfs(i, j)
                max_area = max(max_area, area)
                
        return max_area


##827. Making A Large Island
class Solution:
    def largestIsland(self, grid: List[List[int]]) -> int:
        
        m, n = len(grid), len(grid[0])
        
        def dfs(i, j):
            nonlocal color
            if i < 0 or j < 0 or i >= m or j >= n or grid[i][j] != 1:
                return 0
            grid[i][j] = color
            return 1 + dfs(i + 1, j) + dfs(i - 1, j) + dfs(i, j + 1) + dfs(i, j - 1)
        
        def getColor(i, j):
            if i < 0 or j < 0 or i >= m or j >= n:
                return 0
            else:
                return grid[i][j]
            
        color_area = defaultdict(int)
        color = 1
        max_area = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1: 
                    color+=1
                    color_area[color] = dfs(i, j)
                    max_area = max(max_area, color_area[color])
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 0:
                    area = 1
                    #set
                    for color in set([getColor(i, j - 1), getColor(i, j + 1), getColor(i - 1, j), getColor(i + 1, j)]):
                        area += color_area[color]
                        
                    max_area = max(max_area, area)
                    
        return max_area


## 130. Surrounded Regions
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        if not board or not board[0]:
            return
        
        m, n = len(board), len(board[0])
        
        def dfs(i, j):
            if board[i][j] != 'O':
                return 
            board[i][j] = 'E'
            if i < m-1:
                dfs(i + 1, j)
            if i > 0:
                dfs(i - 1, j)
            if j < n-1 :
                dfs(i, j + 1)
            if j > 0:
                dfs(i, j - 1)
                
        borders = [(i, j) for i in range(m) for j in [0, n-1]] + [(i, j) for i in [0, m-1] for j in range(n)]
        
        for i, j in borders:
            if board[i][j] == 'O':
                dfs(i, j)
                
        for i in range(m):
            for j in range(n):
                if board[i][j] == 'O': board[i][j] = 'X'
                elif board[i][j] == 'E': board[i][j] = 'O'


## 694. Number of Distinct Islands
class Solution:
    def numDistinctIslands(self, grid: List[List[int]]) -> int:
        
        m, n = len(grid), len(grid[0])
        
        def current_island_is_unique():
            unique = True
            if len(unique_islands) > 0:
                for other_island in unique_islands:
                    if len(other_island) == len(current_island):
                        length = len(other_island)
                        for cell_1, cell_2 in zip(current_island, other_island):
                            print(cell_1, cell_2)
                            if cell_1 == cell_2:
                                length-=1
                            else:
                                break
                            print(length)
                        if length == 0:
                            unique = False
            
            return unique
        
        def dfs(i, j):
            nonlocal row_origin, col_origin
            if i < 0 or j < 0 or i >= m or j >= n or grid[i][j] == 0:
                return
            grid[i][j] = 0
            current_island.append((i - row_origin, j - col_origin))
            dfs(i + 1, j)
            dfs(i - 1, j)
            dfs(i, j + 1)
            dfs(i, j - 1)
            
        
        unique_islands = []
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 0: continue
                current_island = []
                row_origin = i
                col_origin = j
                dfs(i, j)
                if len(current_island) > 0 and current_island_is_unique():
                    unique_islands.append(current_island)
          
        return len(unique_islands)


## 733. Flood Fill
class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
        m, n = len(image), len(image[0])
        
        def dfs(i, j, val):
            if i < 0 or j < 0 or i >= m or j >= n or image[i][j] !=  val:
                return
            
            image[i][j] = newColor
            dfs(i + 1, j, val)
            dfs(i - 1, j, val)
            dfs(i, j + 1, val)
            dfs(i, j - 1, val)
        
        
        val = image[sr][sc]
        if val != newColor:
            dfs(sr, sc, val)
        
        return image


## 463. Island Perimeter
class Solution:
    def islandPerimeter(self, grid: List[List[int]]) -> int:
        
        total = 0
        m, n  = len(grid), len(grid[0])
        
        def dfs(i, j, visited):
            nonlocal total
            edges = 4
            for delta_i, delta_j in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                next_i, next_j = i + delta_i, j + delta_j
                if 0 <= next_i < m and 0 <= next_j < n and grid[next_i][next_j] == 1:
                    edges -= 1
                    if (next_i, next_j) not in visited:
                        visited.add((next_i, next_j))
                        dfs(next_i, next_j, visited)
            total += edges
        
        #used visited to mark visit rather than 0/1 in map because 1 in map is needed for count measure as well
        visited = set()
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    visited.add((i, j))  
                    dfs(i, j, visited)
                    return total
        return 0


## 417. Pacific Atlantic Water Flow
class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        if not heights or not heights[0]: 
            return []
        
        # Initialize variables, including sets used to keep track of visited cells
        m, n = len(heights), len(heights[0])
        
        def dfs(i, j, reachable):
            for delta_i, delta_j in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                next_i, next_j = i + delta_i, j + delta_j
                if 0 <= next_i < m and 0 <= next_j < n and (next_i, next_j) not in reachable and heights[next_i][next_j] >= heights[i][j]:
                    reachable.add((next_i, next_j))
                    dfs(next_i, next_j, reachable)
         
        pacific_reachable = set()
        atlantic_reachable = set()
        for i in range(m):
            pacific_reachable.add((i, 0))
            dfs(i, 0, pacific_reachable)
            atlantic_reachable.add((i, n - 1))
            dfs(i, n - 1, atlantic_reachable)
        
        for i in range(n):
            pacific_reachable.add((0, i))
            dfs(0, i, pacific_reachable)
            atlantic_reachable.add((m - 1, i))
            dfs(m - 1, i, atlantic_reachable)
            
        return list(pacific_reachable & atlantic_reachable)


## 947. Most Stones Removed with Same Row or Column
class Solution:
    def removeStones(self, stones: List[List[int]]) -> int:
        rows, cols = defaultdict(list), defaultdict(list)
        for i,j in stones:
            rows[i].append(j)
            cols[j].append(i)
            
        def dfs(i, j):
            for jj in rows[i]:
                if (i,jj) not in seen:
                    seen.add((i,jj))
                    dfs(i, jj)
            for ii in cols[j]:
                if (ii,j) not in seen:
                    seen.add((ii,j))
                    dfs(ii,j)
                    
        islands = 0
        seen = set()
        for i,j in stones:
            if (i,j) not in seen:
                seen.add((i,j))
                dfs(i, j)
                islands +=1
                
        return len(stones)-islands


## 286. Walls and Gates: TEL
class Solution:
    def wallsAndGates(self, rooms: List[List[int]]) -> None:
        """
        Do not return anything, modify rooms in-place instead.
        """
        
        m, n = len(rooms), len(rooms[0])
        
        def dfs(i, j, steps):
            if i < 0 or j < 0 or i >= m or j >= n or rooms[i][j] < steps:
                return
            
            rooms[i][j] = steps
            dfs(i + 1, j, steps + 1)
            dfs(i - 1, j, steps + 1)
            dfs(i, j + 1, steps + 1)
            dfs(i, j - 1, steps + 1)
                
        for i in range(m):
            for j in range(n):
                if rooms[i][j] == 0: #start from each gate
                    dfs(i, j, 0)


## 547. Number of Provinces
class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        
        def dfs(curr):
            for i in range(n):
                if isConnected[curr][i] == 1:
                    isConnected[curr][i] = isConnected[i][curr] = 0
                    dfs(i)
        
        n = len(isConnected)
        cnt = 0
        for i in range(n):
            if isConnected[i][i] == 1:
                dfs(i)
                cnt += 1
                
        return cnt

#union
class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        def find(v):
            if v not in p:
                p[v] = v
            if p[v] != v:
                p[v] = find(p[v])
            return p[v]
        
        def union(u, v):
            p[find(u)] = find(v)
        
        p = {}
        n = len(isConnected)
        for i in range(n-1):
            for j in range(i+1, n):
                if isConnected[i][j] == 1:
                    union(i, j)
        
        m = defaultdict(list)    
        for i in range(n): 
            m[find(i)].append(i)
            
        return len(m)



### Union Find ###
   
## 839. Similar String Groups
class Solution:
    def numSimilarGroups(self, strs: List[str]) -> int:
        def find(v):
            if v not in p:
                p[v] = v
            if p[v] != v:
                p[v] = find(p[v])
            return p[v]
        
        def union(u, v):
            p[find(u)] = find(v)
        
        def similar(x, y):
            return sum(a != b for a, b in zip(x, y)) <= 2
        
        p= {}
        n = len(strs)
        for i in range(n-1):
            for j in range(i+1, n):
                if similar(strs[i], strs[j]):
                    union(strs[i], strs[j])
                    
        return len({find(x) for x in strs})


## 952. Largest Component Size by Common Factor
class Solution:
    def largestComponentSize(self, nums: List[int]) -> int:
        
        def find(v):
            if v not in p:
                p[v] = v
            if p[v] != v:
                p[v] = find(p[v])
            return p[v]
        
        def union(u, v):
            p[find(u)] = find(v)
            
        p = {}   
        for num in nums:
            for i in range(2, int(math.sqrt(num))+1):
                if num % i == 0:
                    union(num, i)
                    union(num,  int(num/i))
        
        result = []
        for num in nums:
            result.append(find(num))
            
        result = Counter(result)
        
        return max(result.values())


## 990. Satisfiability of Equality Equations
class Solution:
    def equationsPossible(self, equations: List[str]) -> bool:
        def find(v):
            if v not in p:
                p[v] = v
            if p[v] != v:
                p[v] = find(p[v])
            return p[v]
        
        def union(u, v):
            p[find(u)] = find(v)
            
        p = {}  
        new = []
        for each in equations:
            if each[1] == '=':
                union(each[0], each[3])
            else:
                new.append(each)
                
        for each in new:
            if find(each[0]) == find(each[3]):
                return False
            
        return True


## 721. Accounts Merge
class Solution:
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        def find(v):
            if v not in p:
                p[v] = v
            if p[v] != v:
                p[v] = find(p[v])
            return p[v]
        
        def union(u, v):
            p[find(u)] = find(v)
            
        p = {}    
        ownership = {}
        for i, emails in enumerate(accounts):
            for email in emails[1:]:
                if email in ownership:
                    union(i, ownership[email])
                ownership[email] = i
        
        # Append emails to correct index
        result = collections.defaultdict(list)
        for email, owner in ownership.items():
            result[find(owner)].append(email)
        
        return [[accounts[i][0]] + sorted(emails) for i, emails in result.items()]


## 737. Sentence Similarity II
class Solution:
    def areSentencesSimilarTwo(self, sentence1: List[str], sentence2: List[str], similarPairs: List[List[str]]) -> bool:
        if len(sentence1) != len(sentence2): return False
        
        def find(v):
            if v not in p:
                p[v] = v
            if p[v] != v:
                p[v] = find(p[v])
            return p[v]
        
        def union(u, v):
            p[find(u)] = find(v)
            
        
        p = {}
        for a, b in similarPairs:
            union(a, b)
            
        for i in range(len(sentence1)):
            if find(sentence1[i]) != find(sentence2[i]):
                return False
        return True
                

## 959. Regions Cut By Slashes  
class Solution:
    def regionsBySlashes(self, grid: List[str]) -> int:
        def find(v):
            if v not in p:
                p[v] = v
            if p[v] != v:
                p[v] = find(p[v])
            return p[v]
        
        def union(u, v):
            p[find(u)] = find(v)
            
        p = {}    
        n = len(grid)
        
        for r, row in enumerate(grid):
            for c, val in enumerate(row):
                index = 4 * (r * n + c)
                #inner connection
                if val == '/':
                    union(index + 0, index + 3)
                    union(index + 1, index + 2)
                elif val == '\\':
                    union(index + 0, index + 1)
                    union(index + 2, index + 3)
                elif val == ' ':
                    union(index + 0, index + 1)
                    union(index + 1, index + 2)
                    union(index + 2, index + 3)
                
                #inter connection
                if (r + 1 < n):
                    union(index + 2, index + 4 * n + 0)
                if (c + 1 < n):
                    union(index + 1, index + 4 + 3)
        
        result = set()
        for i in range(4 * n * n):
            result.add(find(i))
        return len(result) 


## 399. Evaluate Division
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        #p is dictionary: key: u, value: (v, w) -> u/v = w
        def find(v):
            if v not in p:
                p[v] = (v, 1)
            if v != p[v][0]:
                pv, pw = find(p[v][0])
                p[v] = (pv, p[v][1] * pw)
            return p[v]

        def union(u, v, w):
            ru, wu = find(u)
            rv, wv = find(v)
            if ru != rv:
                p[ru] = (rv, w * wv/ wu)
            
        def divide(u, v):
            ru, wu = find(u)
            rv, wv = find(v)
            if ru != rv: return -1.0
            return wu / wv
            
        p = {}
        for (u, v), w in zip(equations, values):       
            union(u, v, w)
            
        return [divide(x, y) if x in p and y in p else -1 for x, y in queries]
        

 ## 128. Longest Consecutive Sequence
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        def find(v):
            if v not in p:
                p[v] = v
            if p[v] != v:
                p[v] = find(p[v])
            return p[v]
        
        def union(u, v):
            pu, pv = find(u), find(v)
            if pu != pv:
                if rank[pu] >= pv:
                    p[pv] = pu
                    rank[pu] += 1
                else:
                    p[pu] = pv
                    rank[pv] += 1
                    
                    
        if not nums:
            return 0 # corner case
        
        # first pass is initialize parent and rank for all num in nums
        p = {}
        nums = set(nums)
        rank = {i:0 for i in nums}
        
        for num in nums:
            if num-1 in nums:
                union(num-1, num)
            if num+1 in nums:
                union(num+1, num)
                
        result = collections.defaultdict(list)
        for num in nums:
            result[find(num)].append(num)
        return max([len(l) for l in result.values()])             
        

## 684. Redundant Connection
class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        def find(v):
            if v not in p:
                p[v] = v
                rank[v] = 0
            if p[v] != v:
                p[v] = find(p[v])
            return p[v]
        
        def union(u, v):
            pu, pv = find(u), find(v)
            if rank[pu] > rank[pv]:
                p[pv] = pu
            elif rank[pv] > rank[pu]:
                p[pu] = pv
            else:
                p[pv] = pu
                rank[pu] += 1
                  
                    
       
        p = {}
        rank = {}
        
        for u, v in edges:
            if find(u) == find(v):
                return[u, v]
            else:
                union(u, v)
            
        return []


## 685. Redundant Connection II
class Solution:
    def findRedundantDirectedConnection(self, edges: List[List[int]]) -> List[int]:
        def find(v):
            if v not in p:
                p[v] = v
                rank[v] = 0
            if p[v] != v:
                p[v] = find(p[v])
            return p[v]
        
        def union(u, v):
            pu, pv = find(u), find(v)
            if rank[pu] > rank[pv]:
                p[pv] = pu
            elif rank[pv] > rank[pu]:
                p[pu] = pv
            else:
                p[pv] = pu
                rank[pu] += 1
                  
        
        p = {}
        ans1 = None
        ans2 = None
        dup_p = False
        
        for e in edges:
            u, v = e
            if v in p:  #v has duplicate parents
                ans1 = [p[v], v]
                ans2 = [u, v]
                dup_p = True
                e[0] = e[1] = -1  #second one
            else:
                p[v] = u
        
        p = {}
        rank = {}
        for u, v in edges:
            if u < 0: continue  #second one 
            if find(u) == find(v):
                return ans1 if dup_p else [u, v]  #if case 2.2 else case 1
            else:
                union(u, v)
                
        return ans2  #case 2.1


## 1559. Detect Cycles in 2D Grid

class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        def find(v):
            if v not in p:
                p[v] = v
                rank[v] = 0
            if p[v] != v:
                p[v] = find(p[v])
            return p[v]
        
        def union(u, v):
            pu, pv = find(u), find(v)
            if rank[pu] > rank[pv]:
                p[pv] = pu
            elif rank[pv] > rank[pu]:
                p[pu] = pv
            else:
                p[pv] = pu
                rank[pu] += 1
                
                
        m, n = len(grid), len(grid[0])
        p = {}
        rank = {}
        
        for i, row in enumerate(grid):
            for j, letter in enumerate(row):
                if i > 0 and j > 0 and grid[i-1][j] == grid[i][j-1] == letter:
                    if find((i-1, j)) == find((i, j-1)):
                        return True
                if i > 0 and grid[i-1][j] == letter:
                    union((i, j), (i-1, j))
                if j > 0 and grid[i][j-1] == letter:
                    union((i, j), (i, j -1))
                
        return False



### DFS Topological Sort

## 207. Course Schedule
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        def dfs(cur):
            if visit[cur] == 1: return True #cycle
            if visit[cur] == 2: return False
        
            visit[cur] = 1
            for each in graph[cur]:
                if dfs(each): return True
            visit[cur] = 2
            return False
        
        
        graph = defaultdict(list)
        for b, a in prerequisites:
            graph[a].append(b)
        
        #1 == visiting, 2 = visited
        visit = defaultdict(int)
        for i in range(numCourses):
            if dfs(i): return False #there is a cycle
        
        return True


## 210. Course Schedule II
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        def dfs(cur):
            if visit[cur] == 1: return True #cycle
            if visit[cur] == 2: return False
        
            visit[cur] = 1
            for each in graph[cur]:
                if dfs(each): return True
            visit[cur] = 2
            result.append(cur)
            return False
        
        
        graph = defaultdict(list)
        for b, a in prerequisites:
            graph[a].append(b)
        
        #1 == visiting, 2 = visited
        visit = defaultdict(int)
        result = []
        for i in range(numCourses):
            if dfs(i): return [] #there is a cycle
        
        return result[::-1]


## 269. Alien Dictionary
class Solution:
    def alienOrder(self, words: List[str]) -> str:
        def dfs(cur):
            if visit[cur] == 1: return True  #cycle
            if visit[cur] == 2: return False
            visit[cur] = 1
            for each in graph[cur]:
                if dfs(each): return True #cycle
            
            visit[cur] = 2
            result.append(cur)
            return False
        
        result = []
        graph = {c : [] for word in words for c in word}  #important
        visit = defaultdict(int)
        for first_word, second_word in zip(words, words[1:]):
            for s, t in zip(first_word, second_word):
                if s!=t:
                    graph[s].append(t)
                    break
            else:
                if len(first_word) > len(second_word): #not valid
                    return ''
        
        for each in graph:
            if dfs(each): return ""
        
        return ''.join(result[::-1])


## 802. Find Eventual Safe States
class Solution:
    def eventualSafeNodes(self, graph: List[List[int]]) -> List[int]:
        #{UNKNOWN:0, VISITING:1, SAFE:2, UNSAFE:3}
        def dfs(cur):
            if visit[cur] == 1: #cycle
                visit[cur] = 3
                return 3 
            if visit[cur] != 0: #safe or unsafe
                return visit[cur]
        
            visit[cur] = 1
            for each in g[cur]:
                if dfs(each) == 3: 
                    visit[cur] = 3
                    return 3
            visit[cur] = 2
            return 2
        
        
        g = defaultdict(list)
        for i, nodes in enumerate(graph):
            g[i] = nodes
        
        result = []
        visit = defaultdict(int)
        for i in range(len(graph)):
            if dfs(i) == 2:
                result.append(i)
        
        return sorted(result)


## 310. Minimum Height Trees
class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        if n <=2:
            return [i for i in range(n)]
        
        g = defaultdict(list)
        
        for start, end in edges:
            g[start].append(end)
            g[end].append(start)
            
        leaves = []
        
        for i in range(n):
            if len(g[i]) == 1: #leaf, only neighbor is parent
                leaves.append(i)
                
        remaining_nodes = n
        while remaining_nodes > 2:
            remaining_nodes -= len(leaves)
            new_leaves = []
            while leaves:
                leaf = leaves.pop()
                parent = g[leaf].pop()
                g[parent].remove(leaf)
                del g[leaf]
                if len(g[parent]) == 1:
                    new_leaves.append(parent)
            leaves = list(new_leaves)
            
        return list(g.keys())
            


### Bipartition

## 785. Is Graph Bipartite?           
class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        def dfs(cur, color):
            colors[cur] = color
            for each in g[cur]:
                if (colors[each] == color): return False #not bipartition      
                if (colors[each] == 0)  and not dfs(each, -color): return False #not bipartition
            
            return True
        
        g = {i: [] for i in range(len(graph))}
        for i,  nodes in enumerate(graph):
            g[i] = nodes

        colors = defaultdict(int)  
        for i in range(len(graph)): #0: unknown, 1: red, -1: blue
            if colors[i] == 0 and not dfs(i, 1):
                return False
        return True


## 886. Possible Bipartition 
class Solution:
    def possibleBipartition(self, n: int, dislikes: List[List[int]]) -> bool:
        def dfs(cur, color):
            colors[cur] = color
            for each in g[cur]:
                if (colors[each] == color): return False #not bipartition      
                if (colors[each] == 0)  and not dfs(each, -color): return False #not bipartition
            
            return True
        
        g = {i: [] for i in range(n)}
        for u, v in dislikes:
            g[u-1].append(v-1)
            g[v-1].append(u-1)

        colors = defaultdict(int)  
        for i in range(n): #0: unknown, 1: red, -1: blue
            if colors[i] == 0 and not dfs(i, 1):
                return False
        return True
            

## 1042. Flower Planting With No Adjacent  
class Solution:
    def gardenNoAdj(self, n: int, paths: List[List[int]]) -> List[int]:
        g = defaultdict(list)
        for u, v in paths:
            g[u-1].append(v-1)
            g[v-1].append(u-1)
            
        colors = [0]* n  #not known 
        for node in range(n): #garden
            nei_colors = []
            for neighbor in g[node]:
                nei_colors.append(colors[neighbor])
            for k in range(1, 5):
                if k not in nei_colors:
                    colors[node] = k
                    break
        return colors



### BFS Grid

## 542. 01 Matrix
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        
        m, n = len(mat), len(mat[0])
        visited = [[0 for i in range(n)] for i in range(m)]
        result = [[math.inf for i in range(n)] for i in range(m)]
        queue = []
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 0:
                    queue.append((i,j))
                    visited[i][j] = 1
         
        dirs = [0, -1, 0, 1, 0]
        steps = 0
        
        while queue:
            size = len(queue)
            for _ in range(size):
                i, j = queue.pop(0)
                result[i][j] = steps
                for k in range(4):
                    x = i + dirs[k]
                    y = j + dirs[k+1]
                    if x <0 or x>=m or y<0 or y>=n or visited[x][y]: continue
                    visited[x][y] = 1
                    queue.append((x, y))
            
            steps+=1
            
        return result


## 1926. Nearest Exit from Entrance in Maze
class Solution:
    def nearestExit(self, maze: List[List[str]], entrance: List[int]) -> int:
                
        m, n = len(maze), len(maze[0])
        
        queue = []
        queue.append((entrance[0], entrance[1]))
        maze[entrance[0]][entrance[1]] = '#'
        dirs = [0, -1, 0, 1, 0]
        steps = 0
        
        while queue:
            size = len(queue)
            for _ in range(size):
                i, j = queue.pop(0)
                for k in range(4):
                    x = i + dirs[k]
                    y = j + dirs[k+1]
                    if x <0 or x>=m or y<0 or y>=n or maze[x][y] != '.': continue
                    if (x == 0) or (x == m-1) or (y==0) or (y == n-1):
                        return steps + 1
                    maze[x][y] = '#'
                    queue.append((x, y))
            
            steps+=1
            
        return -1

## 490. The Maze
class Solution:
    def hasPath(self, maze: List[List[int]], start: List[int], destination: List[int]) -> bool:
        m, n = len(maze), len(maze[0])
        e0, e1 = destination[0], destination[1]
        
        queue = []
        queue.append((start[0], start[1]))
        maze[start[0]][start[1]] = 2
        dirs = [0, -1, 0, 1, 0]
        
        while queue:
            size = len(queue)
            for _ in range(size):
                i, j = queue.pop(0)
                
                for k in range(4):
                    x = i + dirs[k]
                    y = j + dirs[k+1]
                    if x <0 or x>=m or y<0 or y>=n or maze[x][y] != 0: continue #visited or wall
                    while 0 <= x < m and 0 <= y < n and maze[x][y] != 1: #not a wall continue to roll
                        x += dirs[k]
                        y += dirs[k+1]
                    x -= dirs[k]
                    y -= dirs[k+1]
                    if maze[x][y] != 0: continue
                    if (x == e0) and (y == e1): return True
                    maze[x][y] = 2  #visited
                    queue.append((x, y))
            
        return False


## 505. The Maze II
class Solution:
    def shortestDistance(self, maze: List[List[int]], start: List[int], destination: List[int]) -> int:
        m, n = len(maze), len(maze[0])
        queue = []
        queue.append((start[0], start[1]))
        inf = 2**31-1
        distance = [[inf for i in range(n)] for i in range(m)]
        distance[start[0]][start[1]] = 0
        dirs = [0, -1, 0, 1, 0]
        while queue:
            size = len(queue)
            for _ in range(size):
                i, j = queue.pop(0)
                
                for k in range(4):
                    x = i + dirs[k]
                    y = j + dirs[k+1]
                    if x <0 or x>=m or y<0 or y>=n or maze[x][y] != 0: continue #visited or wall
                        
                    steps = 0
                    while 0 <= x < m and 0 <= y < n and maze[x][y] != 1: #not a wall continue to roll
                        x += dirs[k]
                        y += dirs[k+1]
                        steps+=1 #path length
                    x -= dirs[k]
                    y -= dirs[k+1]
                    
                    if distance[i][j] + steps < distance[x][y]:
                        distance[x][y] = distance[i][j] + steps
                        queue.append((x, y))
            
            
        return distance[destination[0]][destination[1]] if distance[destination[0]][destination[1]] !=inf else -1


## 1162. As Far from Land as Possible
class Solution:
    def maxDistance(self, grid: List[List[int]]) -> int:
        n = len(grid)
        queue = [(i,j) for i in range(n) for j in range(n) if grid[i][j] == 1]  #layer 0
        if len(queue) == n * n or len(queue) == 0: #all 1 or all 0
            return -1
        
        dirs = [0, -1, 0, 1, 0]
        steps = 0
        while queue:
            size = len(queue)
            for _ in range(size):
                i, j = queue.pop(0)
                for k in range(4):
                    x = i + dirs[k]
                    y = j + dirs[k+1]
                    if x <0 or x>=n or y<0 or y>=n or grid[x][y] != 0: continue
                    grid[x][y] = 1
                    queue.append((x, y))
            steps+=1
            
        return steps-1


## 286. Walls and Gates
class Solution:
    def wallsAndGates(self, rooms: List[List[int]]) -> None:
        """
        Do not return anything, modify rooms in-place instead.
        """
        
        m, n = len(rooms), len(rooms[0])
        
        queue = [(i,j) for i in range(m) for j in range(n) if rooms[i][j] == 0]  #layer 0
        inf = 2**31-1
        
        dirs = [0, -1, 0, 1, 0]
        steps = 0
        while queue:
            size = len(queue)
            for _ in range(size):
                i, j = queue.pop(0)
                for k in range(4):
                    x = i + dirs[k]
                    y = j + dirs[k+1]
                    if x <0 or x>=m or y<0 or y>=n or rooms[x][y] != inf: continue
                    rooms[x][y] = steps + 1
                    queue.append((x, y))
            steps+=1


## 994. Rotting Oranges
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        queue = []
        rotten, fresh = 0, 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 2:
                    queue.append((i,j))
                    rotten+=1
                elif grid[i][j] == 1:
                    rotten +=1
                    fresh+=1
        if fresh == 0:
            return 0
        
        dirs = [0, -1, 0, 1, 0]
        steps = 0
        
        while queue:
            size = len(queue)
            for _ in range(size):
                i, j = queue.pop(0)
                rotten-=1
                if rotten == 0:
                    return steps
                for k in range(4):
                    x = i + dirs[k]
                    y = j + dirs[k+1]
                    if x <0 or x>=m or y<0 or y>=n or grid[x][y] != 1: continue
                    grid[x][y] = 2
                    queue.append((x, y))
            steps+=1  #bfs layer marker
            
        return -1


## 909. Snakes and Ladders
class Solution:
    def snakesAndLadders(self, board: List[List[int]]) -> int:
        def label_to_position(label):
            r, c = divmod(label-1, n)
            if r % 2 == 0:
                return n-1-r, c
            else:
                return n-1-r, n-1-c
        
        n = len(board)
        visited = set()
        queue = []
        queue.append(1)
        visited.add(1)
        step = 0
        while queue:
            size = len(queue)
            for _ in range(size):
                label = queue.pop(0)
                r, c = label_to_position(label)
                if board[r][c] != -1:
                    label = board[r][c]
                if label == n*n:
                    return step
                for x in range(1, 7): #[curr + 1, min(curr + 6, n**2)]
                    new_label = label + x
                    if new_label > n*n or new_label in visited: continue
                    queue.append(new_label)
                    visited.add(new_label)
            step+=1
            
        return -1

## 1091. Shortest Path in Binary Matrix
class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        n = len(grid)
        if n == 1 and grid[0][0] == 0: return 1
        if grid[0][0] != 0 or grid[n-1][n-1] != 0: return -1
        
        queue = []
        queue.append((0, 0))
        grid[0][0] = 2
        dirs = [(0, -1), (-1, 0), (1, 0), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
        steps = 1
        
        while queue:
            size = len(queue)
            for _ in range(size):
                i, j = queue.pop(0)
                for di, dj in dirs:
                    x = i + di
                    y = j + dj
                    if x <0 or x>=n or y<0 or y>=n or grid[x][y] != 0: continue
                    if (x == n-1) and (y == n-1):
                        return steps + 1
                    grid[x][y] = 2
                    queue.append((x, y))
            
            steps+=1
            
        return -1


##529. Minesweeper
class Solution:
    def updateBoard(self, board: List[List[str]], click: List[int]) -> List[List[str]]:
        def numBombsAdj(i, j):
            count = 0 
            for x, y in dirs: 
                if 0 <= i + x < m and 0 <= j + y < n and board[i+x][y+j] == "M": count += 1 
            return count 
        
        
        m, n  = len(board), len(board[0])
        s0, s1 = click
        
        if board[s0][s1] == "M":
            board[s0][s1] = "X"
        
        dirs = [(0, -1), (-1, 0), (1, 0), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
        queue = []
        queue.append((s0, s1))
        visited = set((s0, s1))
        
        while queue:
            size = len(queue)
            for _ in range(size):
                i, j = queue.pop(0)
                if board[i][j] == "E":
                    bombsNextTo = numBombsAdj(i, j)
                    board[i][j] = "B" if bombsNextTo == 0 else str(bombsNextTo)
                    if bombsNextTo == 0:
                        for di, dj in dirs:
                            x = i + di
                            y = j + dj
                            if x <0 or x>=m or y<0 or y>=n or (x, y) in visited: continue
                            queue.append((x, y))
                            visited.add((x, y))
        return board 


## 675. Cut Off Trees for Golf Event
class Solution:
    def cutOffTree(self, forest: List[List[int]]) -> int:
        def bfs(sx, sy, tx, ty):
            visited = [[0 for i in range(n)] for i in range(m)]
            queue = [(sx, sy)]
            visited[sx][sy] = 1
            
            dirs = [0, -1, 0, 1, 0]
            steps = 0

            while queue:
                size = len(queue)
                for _ in range(size):
                    i, j = queue.pop(0)
                    if (i == tx) and (j == ty): return steps
                    for k in range(4):
                        x = i + dirs[k]
                        y = j + dirs[k+1]
                        if x <0 or x>=m or y<0 or y>=n or visited[x][y] or not forest[x][y]: continue
                        visited[x][y] = 1
                        queue.append((x, y))
                steps+=1  #bfs layer marker
            return -1
        
        m, n = len(forest), len(forest[0])
        trees = []
        for i in range(m):
            for j in range(n):
                if (forest[i][j] > 1):
                    trees.append((forest[i][j], i, j))
        
        trees.sort(key = lambda k: k[0])
        sx, sy = 0, 0
        total_steps = 0
        for _, tx, ty in trees:
            steps = bfs(sx, sy, tx, ty)
            if (steps == -1): return -1
            forest[tx][ty] = 1
            total_steps += steps
            sx, sy = tx, ty
            
        return total_steps
            

## 934. Shortest Bridge
class Solution:
    def shortestBridge(self, grid: List[List[int]]) -> int:
        def dfs(i, j):
            if i < 0 or j < 0 or i >= n or j >= n or grid[i][j] != 1: return
            print(i, j)
            grid[i][j] = 2  #visited
            queue.append((i, j))
            dfs(i + 1, j)
            dfs(i - 1, j)
            dfs(i, j + 1)
            dfs(i, j - 1)
            
        n = len(grid)
        queue = []
        for i in range(n):
            for j in range(n):
                if (grid[i][j] == 1):
                    dfs(i, j)
                    break
            else: 
                continue
            break
                             
        dirs = [0, -1, 0, 1, 0]
        steps = 0
        while queue:
            size = len(queue)
            for _ in range(size):
                i, j = queue.pop(0)
                for k in range(4):
                    x = i + dirs[k]
                    y = j + dirs[k+1]
                    if x <0 or x>=n or y<0 or y>=n or grid[x][y] == 2: continue
                    if grid[x][y] == 1: 
                        return steps
                    grid[x][y] = 2
                    queue.append((x, y))
            steps+=1  #bfs layer marker
            
        return -1


### BFS shortest path

## 847. Shortest Path Visiting All Nodes
class Solution:
    def shortestPathLength(self, graph: List[List[int]]) -> int:
        n = len(graph)
        if n == 1:
            return 0
        
        # used to check whether all nodes have been visited (11111...111)
        allVisited = (1 << n) - 1   #sum of masks
        queue = [(i, 1 << i) for i in range(n)]
        
        # encoded_visited in visited_states[node] iff
        # (node, encoded_visited) has been pushed onto the queue
        visited_states = [{1 << i} for i in range(n)]  #set in list
        # states in visited_states will never be pushed onto queue again
        steps = 0
        while queue:
            size = len(queue)
            for _ in range(size):
                currentNode, visited = queue.pop(0)
                if visited == allVisited:
                        return steps
                for nb in graph[currentNode]:
                    new_visited = visited | (1 << nb)  
                    if new_visited in visited_states[nb]: continue
                    visited_states[nb].add(new_visited)
                    queue.append((nb, new_visited))

            steps += 1
        
        return -1


## 864. Shortest Path to Get All Keys
class Solution:
    def shortestPathAllKeys(self, grid: List[str]) -> int:
        m, n = len(grid), len(grid[0])
        allVisited = 0
        visited_states = [[set() for _ in range(n)] for _ in range(m)]
        queue = []
        for i in range(m):
            for j in range(n):
                c = grid[i][j]
                if c == '@':  #start
                    queue.append((i, j, 0))
                    visited_states[i][j].add(0)
                elif c >= 'a' and c <= 'f':
                    allVisited |= (1 << (ord(c) - ord('a')))
        
        dirs = [-1, 0, 1, 0, -1]
        steps = 0
        while queue:
            size = len(queue)
            for _ in range(size):
                i, j, state = queue.pop(0)
                if state == allVisited: return steps
                for k in range(4):
                    x = i + dirs[k]
                    y = j + dirs[k+1]
                    new_state = state  #important to reassign within for loop
                    if x < 0 or x >= m or y < 0 or y >= n: continue
                    c = grid[x][y]
                    if c == '#': continue
                        
                    if c in string.ascii_uppercase: #lock 
                        if state & (1 << (ord(c) - ord('A'))) == 0: #no key revealed
                            continue
                    if (c in string.ascii_lowercase): 
                        new_state = state | (1 << (ord(c) - ord('a')))
                    
                    if new_state in visited_states[x][y]: continue
                    queue.append((x, y, new_state))
                    visited_states[x][y].add(new_state)
            
            steps += 1
        
        return -1


## 298. Maximum Candies You Can Get from Boxes
class Solution:
    def maxCandies(self, status: List[int], candies: List[int], keys: List[List[int]], containedBoxes: List[List[int]], initialBoxes: List[int]) -> int:
        
        n = len(status)
        found = [0 for i in range(n)]
        queue = []
        for b in initialBoxes:
            found[b] = 1
            if status[b] == 1:
                queue.append(b)
        
        result = 0
        while queue:
            b = queue.pop(0)
            result += candies[b]
            for t in containedBoxes[b]:
                found[t] = 1
                if status[t] == 1:
                    queue.append(t)
            for t in keys[b]:
                if status[t] == 0 and found[t] == 1:
                    queue.append(t)
                status[t] = 1
                
        return result