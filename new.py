import collections


def registerLoginLogout(logs):
    answer = []
    userinfo = {} #username: (pw, status) 0: online 1: offline
    def register(user, password):
        if user in userinfo.keys():
            answer.append("Username already exists")
        else:
            userinfo[user] = [password, "0"]
            answer.append("Registered Successfully")
    def login(user, password):
        if user not in userinfo.keys():
            answer.append("Login Unsuccessful")
        else:
            current = userinfo[user]
            if current[1] == "1" or current[0] != password:
                answer.append("Login Unsuccessful")
            else:
                current[1] = "1"
                userinfo[user] = current
                answer.append("Logged In Successfully")
    def logout(user):
        if user not in userinfo:
            answer.append("Login Unsuccessful")
        else:
            current = userinfo[user]
            if current[1] == "0":
                answer.append("Logout Unsuccessful")
            else:
                current[1] = "0"
                userinfo[user] = current
                answer.append("Logged Out Successfully")
    for s in logs:
        input = s.split(" ")
        op = input[0]
        user = input[1]
        if len(input) == 3:
            password = input[2]
        else:
            password = ""
        # Task of register
        if op == "register":
            register(user, password)
        # Task of Login
        elif op == "login":
            login(user, password)
        # Task of Logout
        elif op == "logout":
            logout(user)
    return answer

  
def stock(a, k):
    hh = 0
    window = set()
    Maxprice = 0
    for tt in range(len(a)):
        while a[tt] in window:
            window.remove(a[hh])
            hh += 1
        window.add(a[tt])
        if len(window) == k:
            Maxprice = max(Maxprice, sum(window))
    return Maxprice

  
  
def subArrayRanges( nums: List[int]) -> int:
        res = 0
        for l in range(0, len(nums)):
            maxNum = nums[l]
            minNum = nums[l]
            for r in range(l, len(nums)):
                maxNum = max(maxNum, nums[r])
                minNum = min(minNum, nums[r])
                res += (maxNum - minNum)
        return res

      
      
def tempChange(a):
    n = len(a)
    prefix = [0] * n
    prefix[0] = a[0]
    res = 0
    for i in range(1, n):
        prefix[i] = prefix[i - 1] + a[i]
    for day in range(n):
        past = prefix[day]
        future = prefix[-1] - prefix[day] + a[day]
        tmpMax = max(past, future)
        res = max(res, tmpMax)
    return res

  
  
# netprice
def findEarliestMonth(a):
    prefix = [a[0]] * len(a)
    minNet = float("inf")
    res = 0
    for i in range(1, len(a)):
        prefix[i] = prefix[i - 1] + a[i]
    for m in range(len(a) - 1):
        avg1 = prefix[m] // (m + 1)
        avg2 = (prefix[-1] - prefix[m]) // (len(a) - m - 1)
        if abs(avg1 - avg2) < minNet:
            res = m + 1
            minNet = abs(avg1 - avg2)
    return res

  
  
class Solution:
    def maximumElementAfterDecrementingAndRearranging(self, arr: List[int]) -> int:
        arr.sort()
        arr[0] = 1
        for i in range(1, len(arr)):
            if arr[i] - arr[i - 1] > 1:
                arr[i] = arr[i - 1] + 1
        return arr[-1]
      
      
      
      

class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort()
        res = [intervals[0]]
        for i in range(1, len(intervals)):
            if res[-1][1] < intervals[i][0]:
                res.append(intervals[i])
            else:
                #merge
                res[-1][1] = max(res[-1][1], intervals[i][1])
        return res

      
      
      

    def maximumUnits(boxTypes: List[List[int]], truckSize: int) -> int:
        res = 0
        boxTypes = sorted(boxTypes, key = lambda x: -x[1])
        for num, units in boxTypes:
            if num >= truckSize:
                res += truckSize * units
                break
            else:
                res += num * units
                truckSize -= num
        return res

      
      

class Solution:
    def distinctSubarrays(self, nums, k):
        odd = []
        m = len(nums)

        for i, n in enumerate(nums):
            if n % 2:
                odd.append(i)
        res = 0

        for i in range(len(odd) - k + 1):
            left = odd[i]
            right = odd[i + k - 1]

            # find how many numbers on the left of left before another odd number
            if i == 0:
                left_len = left + 1
            else:
                left_len = odd[i] - odd[i - 1]

            # find how many numbers on the right of right before another odd number
            if i == len(odd) - k:
                right_len = m - right
            else:
                right_len = odd[i + k] - right

            # print(left, right, left_len, right_len)
            res += left_len * right_len

        return res

      
      
      
def MaxConsecutiveOnes(arr, m):
    zeroIdx = set()
    hh = 0
    res = []
    MaxLen = 0
    for tt in range(len(arr)):
        if arr[tt] == 0:
            zeroIdx.add(tt)
        if len(zeroIdx) <= m:
            if tt - hh + 1 > MaxLen:
                MaxLen = tt - hh + 1
                res = list(zeroIdx)
        while len(zeroIdx) > m:
            if arr[hh] == 0:
                zeroIdx.remove(hh)
            hh += 1
    return res

  
  
  
  
def choose3Pages(s):
    prefixZero = [0] * len(s)
    prefixOne = [0] * len(s)
    res = 0
    if s[0] == "0":
        prefixZero[0] = 1
    else:
        prefixOne[0] = 1
    for i in range(1, len(s)):
        prefixZero[i] = prefixZero[i - 1]
        prefixOne[i] = prefixOne[i - 1]
        if s[i] == "0":
            prefixZero[i] += 1
        else:
            prefixOne[i] += 1
    # count 010,101
    for i in range(1, len(s) - 1):
        if s[i] == "0":
            res += prefixZero[i] * (prefixZero[-1] - prefixZero[i])
        if s[i] == "1":
            res += prefixOne[i] * (prefixOne[-1] - prefixOne[i])
    return res
  
  
  
  
  

def ConutAZ(s):
    prefixA = [0] * len(s)
    zcount = 0
    res = 0
    if s[0] == "A":
        prefixA[0] = 1
    elif s[0] == "Z":
        zcount += 1
    for i in range(1, len(s)):
        prefixA[i] = prefixA[i - 1]
        if s[i] == "A":
            prefixA[i] += 1
        elif s[i] == "Z":
            zcount += 1
    # choose A or Z
    for i in range(1, len(s)):
        if s[i] == "Z":
            res += prefixA[i]
    return res + max(prefixA[-1], zcount)
  
  
  
  
  

class Solution:
    def kthFactor(self, n: int, k: int) -> int:
        factor = 1
        while factor * factor <= n:
            if n % factor == 0:
                k -= 1
            if k == 0:
                return factor
            factor += 1

        factor -= 1
        if factor * factor == n:
            factor -= 1
        while factor > 0:
            if n % factor == 0:
                k -= 1
            if k == 0:
                return n // factor
            factor -= 1
        return -1

      
      
      
      
def RounterCount(headCount, routerLocation, routerRange):
    buildnum = len(headCount)
    routeinfo = [0] * (buildnum + 1)
    count = 0
    serverBuilding = 0
    for i in range(len(routerLocation)):
        left = max(0, routerLocation[i] - 1 - routerRange[i])
        right = min(buildnum - 1, routerLocation[i] - 1 + routerRange[i])
        routeinfo[left] += 1
        routeinfo[right + 1] -= 1
    for i in range(buildnum):
        count += routeinfo[i]
        if count >= headCount[i]:
            serverBuilding += 1
    return serverBuilding

  
  
  
  
  
def DemolitionOfRobot(lot):
    m, n = len(lot), len(lot[0])
    directs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    MinDis = -1
    # row col dis
    q = collections.deque([(0,0,0)])

    while q:
        row, col, dis = q.popleft()
        if lot[row][col] == 9:
            MinDis = dis
            break
        lot[row][col] = -1

        for d in directs:
            next_row = row + d[0]
            next_col = col + d[1]
            if 0 <= next_row < m and 0 <= next_col < n and (lot[next_row][next_col] == 1 or lot[next_row][next_col] == 9):
                q.append((next_row, next_col, dis + 1))
    return MinDis


  
  
  
  
def minMoves(arr):
    countOnes = 0
    swap = 0
    for n in arr:
        if n == 1:
            countOnes += 1
        if n == 0:
            swap += countOnes

    countZeros = len(arr) - countOnes
    return min(swap, countOnes * countZeros - swap)
  
  
  
  
  

def passwordStrength(s):
    fu = False
    yuan = False
    res = 0
    for c in s:
        if c == "a" or c == "e" or c == "i" or c == "o" or c == "u":
            yuan = True
        else:
            fu = True
        if yuan and fu:
            res += 1
            fu = yuan = False
    return res

  
  
  
  
def deliveryBoxes(boxes):
    count = collections.Counter(boxes)
    res = 0
    for box in count:
        num = count[box]
        if num % 3 == 1 and num % 2 == 1:
            return -1
        deliver3 = num // 3
        deliver2 = (num - deliver3 * 3) // 2
        res += deliver2 + deliver3
    return res
  
  
  
  

def findMaximumSustainableClusterSize(bootingPower, processingPower, powerMax):
    res = 0
    prefix = [0] * len(processingPower)
    prefix[0] = processingPower[0]
    q = collections.deque()

    for i in range(1, len(prefix)):
        prefix[i] = prefix[i - 1] + processingPower[i]

    hh = 0
    for tt in range(len(bootingPower)):
        while q and q[0] < hh:
            q.popleft()
        while q and bootingPower[q[-1]] <= bootingPower[tt]:
            q.pop()
        q.append(tt)
        if bootingPower[q[0]] + (prefix[tt] - prefix[hh] + processingPower[hh]) * (tt - hh + 1) <= powerMax:
            res = max(res, tt - hh + 1)
        else:
            hh += 1
    return res
  
  
  
  

def minSumArrayProduct(nums: List[int]) -> int:
    nums = [0] + nums + [0]
    n = len(nums)
    prefix = [nums[0]] * n
    for i in range(n):
        prefix[i] = prefix[i - 1] + nums[i]

    stk = []
    right_first_small = [-1] * n
    left_first_small = [-1] * n
    for i in range(n):
        while stk and nums[stk[-1]] > nums[i]:
            right_first_small[stk.pop()] = i
        stk.append(i)
    for i in range(n - 1, -1, -1):
        while stk and nums[stk[-1]] > nums[i]:
            left_first_small[stk.pop()] = i
        stk.append(i)

    res = 0
    subarray = set()
    for i in range(1, n - 1):
        left = left_first_small[i]
        right = right_first_small[i]
        for l in range(left + 1, i + 1):
            for r in range(i, right):
               if (l,r) not in subarray:
                   res += (prefix[r] - prefix[l] + nums[l]) * nums[i]
                   subarray.add((l, r))


    return res % (10 ** 9 + 7)
  
  
  
  

def mergePackage(nums: List[int]) -> int:
    stk = []
    res = 0
    for i in range(len(nums) - 1, -1, -1):
        newPackage = nums[i]
        if stk and stk[-1] > nums[i]:
            newPackage += stk.pop()
        stk.append(newPackage)
        res = max(res, newPackage)
    return res
