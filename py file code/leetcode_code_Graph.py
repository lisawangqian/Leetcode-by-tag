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
        
             
        