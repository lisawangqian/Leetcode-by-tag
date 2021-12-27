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
        s = list(s)
        
        n = len(s)
        p = [i for i in range(n)]
        def find(v):
            if p[v] != v:
                p[v] = find(p[v])
            return p[v]
        def union(u, v):
            p[find(u)] = find(v)
        
        for edge in pairs:
            union(edge[0], edge[1])
        
        
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
        n = len(isConnected)
        p = [i for i in range(n)]
        def find(v):
            if p[v] != v:
                p[v] = find(p[v])
            return p[v]
        
        def union(u, v):
            p[find(u)] = find(v)
        
        for i in range(n-1):
            for j in range(i+1, n):
                if isConnected[i][j] == 1:
                    union(i, j)
        
        m = defaultdict(list)    
        for i in range(n): 
            m[find(i)].append(i)
            
        return len(m)


