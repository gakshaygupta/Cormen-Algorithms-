# -*- coding: utf-8 -*-
"""
Created on Thu May 16 16:58:07 2019

@author: Akshay
"""
def exchange(arr,i,j):
    arr[i],arr[j]=arr[j],arr[i]
#insertion sort


def insert(arr):
    l=len(arr)
    print(arr,"\n")
    for i in range(1,l):
        key=arr[i]
        j=i-1
        while j>=0 and arr[j]>key:
            arr[j+1]=arr[j]
            j=j-1
        arr[j+1]=key
    print(arr)    
 #merge sort   
def merge(arr,p,q,r):
    L=arr[p:q+1]
    R=arr[q+1:r+1]
    L.append(10**30)
    R.append(10**30)
    i=0
    j=0
    print(L[:-1],R[:-1],"\n")
    for k in range(p,r+1):
            
        if L[i]<=R[j]:
            arr[k]=L[i]
            i=i+1
        else:
            arr[k]=R[j]
            j=j+1
def merge_sort(arr,p,r):
    q=int((p+r)/2)
    if(p==q and r==p):pass
    else:merge_sort(arr,p,q); merge_sort(arr,q+1,r);merge(arr,p,q,r)
# quick sort
def partition(arr,p,r):
    a=arr[r]
    i=p
    for j in range(p,r):
        if arr[j]<=a:
            b=arr[j]
            arr[j]=arr[i]
            arr[i]=b
            i+=1
    b=arr[r]
    arr[r]=arr[i]
    arr[i]=b  
    return i


def quick_sort(arr,p,r):
            q=partition(arr,p,r)
            if(r==q or p==q):
                pass
            else:
                quick_sort(arr,p,q-1);quick_sort(arr,q+1,r)
                
#heap sort
class HEAPIFIED():
    
    def MAX_HEAPIFY(self,i):
        l=2*i+1
        r=2*i+2
        largest=i
        if(l<self.heap_size and self.arr[l]>self.arr[largest]):
            largest=l
        if(r<self.heap_size and self.arr[r]>self.arr[largest]):
            largest=r
        if(largest!=i):
            exchange(self.arr,i,largest)
            self.MAX_HEAPIFY(largest) 
    def BUILD_MAX_HEAP(self,arr):
        self.arr=arr
        self.heap_size=len(arr)
        for i in  range(int(len(arr)/2)-1,-1,-1):
            print(self.arr)
            self.MAX_HEAPIFY(i)
        print(self.arr)   
    def HEAP_EXTRACT_MAX(self):
        if self.heap_size<1:
            print("heap underflow")
        else:
            maxd=self.arr[0]
            self.arr[0]=self.arr[self.heap_size-1]
            self.heap_size-=1
            self.MAX_HEAPIFY(0)
            return maxd
    def HEAP_INCREASE_KEY(self,i,key):
        if(key<self.arr[i]):
            print("error:-you cant decrease the key")
        else:
            self.arr[i]=key
            
            while(i>0 and self.arr[int((i-1)/2)]<key):
                exchange(self.arr,int((i-1)/2),i)
                i=int((i-1)/2)
    def HEAP_SORT(self):
        for i in range(len(self.arr)-1,0,-1):
            exchange(self.arr,0,i)
            self.heap_size-=1
            self.MAX_HEAPIFY(0)
            
# bubble sort
def bubble(arr):
    for i in range(0,len(arr)):
        for j in range(1,len(arr)-i):
            if(arr[j-1]>arr[j]):
                exchange(arr,j-1,j)

# stack
                
class stack:
    def __init__(self):
        self.top=0
        self.S=[]
    
    def Stack_Empty(self):
       return True if self.top==0 else False
    def Push(self,x):
        self.top+=1
        self.S.append(x)
    def Pop(self):
        if self.Stack_Empty():
            print("underflow")
        else:
            self.top=self.top-1
            return self.S[self.top+1]


#queue

class queue:
    def __init__(self,length):
        self.tail=0
        self.head=0
        self.length
        self.Q=[None]*self.length
    
    def Enqueue(self,x):
        self.Q[self.tail]=x
        if self.tail+1==self.length:
            self.tail=0
        else:
            self.tail+=1
    
    def Dequeue(self):
        x=self.Q[self.head]
        if self.head+1==self.length:
            self.head=0
        else:
            self.head+=1
        return x

# Binary search tree
class Node:
    def __init__(self,val,parrent=None,left_child=None,right_chlid=None):
        self.val = val
        self.parrent = parrent
        self.left_child = left_child
        self.right_chlid = right_chlid

class BST:
    
    def __init__(self):
        self.root = None
    
    def add_node(self,node,r):
        
        if self.root is None:
            self.root=node
        else:
            if r.val>node.val and r.left_child is  None:
                r.left_child = node
            elif r.val>node.val and r.left_child is not None:
               self.add_node(node,r.left_child)       
            if r.val<=node.val and r.right_chlid is  None:
                r.right_chlid = node
            elif r.val<=node.val and r.right_chlid is not None:
               self.add_node(node,r.right_chlid) 
    
    def Inorder_Tree_Walk(self,r):
            if r!= None:
                self.Inorder_Tree_Walk(r.left_child)
                print(r.val)
                self.Inorder_Tree_Walk(r.right_chlid)
    def Preorder_Tree_Walk(self,r):
             if r!= None:
                print(r.val)
                self.Preorder_Tree_Walk(r.left_child)
                self.Preorder_Tree_Walk(r.right_chlid)
    def Postorder_Tree_Walk(self,r):
             if r!= None:
                self.Postorder_Tree_Walk(r.left_child)
                self.Postorder_Tree_Walk(r.right_chlid)
                print(r.val)
    def Tree_Search(self,r,val):
        if r == None or val==r.val:
            return r
        if val<r.val:
            return self.Tree_Search(r.left_child,val)
        return self.Tree_Search(r.right_child,val)
    def Iterative_Tree_Search(self,r,val): 
         while r != None and r.val==val:
             if val < r.val:
                 r = r.left
             else: r = r.right
         return r
    def Tree_Successor(self,x):
        if x.right_child != None:
            return self.Tree_Min(x.right_child)
        y = x.parrent
        while y != None and x == y.right_child:
            x = y
            y = y.parrent
        return y
        while y != None and x == y.right_child:
            x = y
    def Tree_Min(self,z):
        while z.left_child != None:
            z = z.left_child
        return z
    def Tree_Max(self,z):
        while z.right_child != None:
            z = z.right_child
        return z
    def Transplant(self,u,v):
        if u.parrent == None:
            self.root=v
        elif u == u.parrent.left:
            u.parrent.left_child = v
        else:
            u.parrent.right_child = v
        if v != None:
            v.parrent = u.parrent
    def Tree_Delete(self,z):
        if z.left_child == None:
            self.Transplant(z,z.right_child)
        elif z.right_child == None:
            self.Transplant(z,z.left_child)
        else:
            y = self.Tree_Min(z.right_child)
            if y.parrent != z:
                self.Transplant(y,y.right)
                y.right_child = z.right_child
                y.right_child.parrent = y
            self.Transplant(z,y)
            y.left_child = z.left_child
            y.left.parrent = y


#Graphs
            
#BFS
import collections 

class Vertex:
    def __init__(self,val):
        self.d = 10**10
        self.p = None
        self.colour = "white"
        self.val = val
        self.i = 10**10
        self.f = 10**10

# graph maker
class Graph:
    def __init__(self,type):
        
        self.V = set()
        self.E = collections.defaultdict(list)
        self.type = type
        self.weights = {}
    
    def add_edge(self,u,v,weight=1):
            if u not in self.V:
                self.V.add(u)
            if v not in self.V:
                self.V.add(v)
            
            if self.type == "undirected":
                self.E[u].append(v)
                self.E[v].append(u)
                self.weights[(u,v)] = weight
                self.weights[(v,u)] = weight
            else :
                self.E[u].append(v)
                self.weights[(u,v)] = weight

    
    def adj(self,u):
        return self.E[u]
    
    def weight(self,u,v):
        return self.weights[(u,v)]
    
            
        
def BFS(G,s):
    for u in G.V - {s}:
        u.d = 10**10
        u.p = None
        u.colour = "white"
    s.colour = "gray"
    s.d = 0
    s.p = None
    Q = []
    Q.append(s)
    while len(Q) != 0:
        u = Q.pop(0)
        for v in G.adj(u):
           if v.colour == "white":
            v.colour = "gray"
            v.d = u.d + 1
            v.p = u
            Q.append(v)
        u.colour = "black"

            

def Print_Path(G,s,v):
    if v == s:
        print(s.val)
    elif v.p == None:
        print("No path from s to v exist")
    else: 
        Print_Path(G,s,v.p)
        print(v.val)

#DFT
        
def DFS(G,arr):
    for u in G.V:
        u.colour = "white"
        u.p = None 
    time = [0]
    for u in list(G.V):
        if u.colour == 'white':
            DFS_Visit(G,u,time,arr)
def DFS_Visit(G,u,time,arr):
    time[0] = time[0] + 1
    u.i = time[0]
    u.colour = "gray"
    for v in G.adj(u):
        if v.colour == 'white':
            v.p = u
            DFS_Visit(G,v,time,arr)
    u.colour = "black"
    time[0] = time[0] + 1 
    u.f = time[0]
    arr.append(u)
#DFS using stack
def DFS_S(G,arr):
    for u in G.V:
        u.colour = "white"
        u.p = None 
    time = 0
    for u in list(G.V):
        if u.colour == 'white':
            DFS_Stack(G,u,time,arr)
def DFS_Stack(G,u,time,arr):
    time = time + 1
    u.i = time
    u.colour = "gray"
    S = []
    S.append(u)
    while len(S) != 0:
        u = S.pop()
        for v in G.adj(u):
           if v.colour == 'white':
            S.append(v)
            v.p = u
            v.colour = 'gray'
        u.colour = "black"
        time = time + 1 
        u.f = time
        arr.append(u)
def Topological_Sort(G):
        arr = []
        DFS(G,arr)
        arr.reverse()
        return arr
            
def Transpose(G):
    vertex = G.V
    GT = Graph('directed')
    GT.V = vertex
    for v in vertex:
        for u in G.adj(v):
            GT.add_edge(u,v)
    return GT
#strongly connected components
def Strong(G):
    
    def DFS_V(G,u,arr):
        u.colour = "gray"
        for v in G.adj(u):
            if v.colour == 'white':
                DFS_V(G,v,time,arr)
        arr.append(u)
    TS=Topological_Sort(G)
    for u in G.V:
        u.colour = "white"
        u.p = None 
    time = [0]
    S=[]
    GT=Transpose(G)
    while TS:
        i = TS.pop(0)
        if i.colour == 'white':
            arr = []
            DFS_V(GT,i,arr)
            S.append(arr)
    return S


G = Graph('directed')

vertex  = {'B':Vertex('B'),'A':Vertex('A'),'C':Vertex('C'),'E':Vertex('E'),'F':Vertex('F'),'D':Vertex('D')}
G.add_edge(vertex['B'],vertex['C'])

G.add_edge(vertex['C'],vertex['A'])

G.add_edge(vertex['A'],vertex['B'])

G.add_edge(vertex['B'],vertex['E'])

G.add_edge(vertex['E'],vertex['F'])

G.add_edge(vertex['F'],vertex['D'])

G.add_edge(vertex['D'],vertex['E'])

#G_text = Graph("directed")
#
#G_text.add_edge(vertex['B'],vertex['C'])
#
#G_text.add_edge(vertex['C'],vertex['A'])
#
#G_text.add_edge(vertex['B'],vertex['E'])
#
#G_text.add_edge(vertex['E'],vertex['F'])
#
#G_text.add_edge(vertex['F'],vertex['D'])

# disjoinset
class Set_obj:
    def __init__(self,obj):
        self.obj = obj
        self.parrent = None
        self.rank = 0

class Disjoin():
    def __init__(self):
        self.objs= {}
        self.sets = []

    def make_set(self,obj):
        if obj in self.objs:
            print("Already exist")
        else:
            self.objs[obj] = Set_obj(obj)
            self.sets.append(self.objs[obj])
    
    def find_set(self,obj):
        if obj not in self.objs:
            print("Object not in the set")
            return None
        set_obj = self.objs[obj]
        if set_obj.parrent == None:
            return obj
        return self.find_set(set_obj.parrent.obj)
    
    def union(self,obj1,obj2):
        parrent_obj1 = self.find_set(obj1)
        parrent_obj2 = self.find_set(obj2)
        set_p_obj1 = self.objs[parrent_obj1]
        set_p_obj2 = self.objs[parrent_obj2]
        if set_p_obj1.rank > set_p_obj2.rank:
            set_p_obj2.parrent = set_p_obj1
            set_p_obj1.rank += 1
        else:
            set_p_obj1.parrent = set_p_obj2
            set_p_obj2.rank +=1

    def path_compress(self):
        for obj in self.objs:
            temp_set = self.objs[obj]
            while temp_set.parrent:
                temp_set = temp_set.parrent
            self.objs[obj].parrent = temp_set

def MST_Kruskal(G):
    A = set()
    disjoin = Disjoin()
    for v in G.V:
        disjoin.make_set(v)
    edges = sorted([x for x in G.weights.keys()],key = lambda x: G.weights[x])
    for u,v in edges:
        if disjoin.find_set(u) != disjoin.find_set(v):
            A.add((u,v))
            disjoin.union(u,v)
    return A
     
# cycle detection using Disjoin
    
def is_cycle(G):
    disjoin = Disjoin()
    count = 0
    for v in G.V:
        disjoin.make_set(v)
    for u,v in G.weights.keys():
        if disjoin.find_set(u) != disjoin.find_set(v):
            disjoin.union(u,v)
        else:
            count += 1
    
    return count


class HeapQ:
    
    def MIN_HEAPIFY(self,i):
        l=2*i+1
        r=2*i+2
        largest=i
        if(l<self.heap_size and self.arr[l].d<self.arr[largest].d):
            largest=l
        if(r<self.heap_size and self.arr[r].d<self.arr[largest].d):
            largest=r
        if(largest!=i):
            exchange(self.arr,i,largest)
            self.MIN_HEAPIFY(largest) 
    def BUILD_MIN_HEAP(self,arr):
        self.arr=arr
        self.heap_size=len(arr)
        for i in  range(int(len(arr)/2)-1,-1,-1):
            self.MIN_HEAPIFY(i)  
    def HEAP_EXTRACT_MIN(self):
        if self.heap_size<1:
            print("heap underflow")
        else:
            self.MIN_HEAPIFY(0)
            mind=self.arr[0]
            self.arr[0]=self.arr[self.heap_size-1]
            self.heap_size-=1
            return mind
def MST_Prim(G,r):
    for u in G.V:
        u.d = 10**10
        u.p = None
    r.d = 0
    Q = HeapQ()
    Q.BUILD_MIN_HEAP(list(G.V))
    while Q.heap_size != 0:
        u = Q.HEAP_EXTRACT_MIN()
        for v in G.adj(u):
            if v in Q.arr[:Q.heap_size] and G.weight(u,v)<v.d:
                v.p = u
                v.d = G.weight(u,v)
        Q.BUILD_MIN_HEAP(Q.arr[:Q.heap_size])
# weighted
G = Graph('undirected')

vertex  = {'B':Vertex('B'),'A':Vertex('A'),'C':Vertex('C'),'E':Vertex('E'),'F':Vertex('F'),'D':Vertex('D')}
G.add_edge(vertex['B'],vertex['C'],9)

G.add_edge(vertex['C'],vertex['A'],1)

G.add_edge(vertex['A'],vertex['B'],10)

G.add_edge(vertex['B'],vertex['E'],6)

G.add_edge(vertex['E'],vertex['F'],7)

G.add_edge(vertex['F'],vertex['D'],5)

G.add_edge(vertex['D'],vertex['E'],4)


#shortest path 
#bellmen-Ford
def Initialize_Single_Source(G,s):
    for v in G.V:
        v.d = 10**10
        v.p = None
    s.d = 0
def Relax(u,v,G):
    if v.d > u.d +G.weight(u,v):
        v.d = u.d +G.weight(u,v)
        v.p = u
def Bellman_Ford(G,s):
    Initialize_Single_Source(G,s)
    for i in range(0,len(G.V)-1):
        for u,v in G.weights.keys():
            Relax(u,v,G)
    for u,v in G.weights.keys():
        if v.d > u.d +G.weights.keys():
            return False
    return True
        
#matrix multiplication optimization 
def optimizer(P,i,j,memo):
    key = (i,j)
    if key in memo:
        return memo[key]
    else:
        memo[key] = 0
    if i==j or i>j:
        memo[key] = 0
        return 0
    if j-i == 1 and j < len(P):
        memo[key] = P[i-1]*P[i]*P[j]
        return P[i-1]*P[i]*P[j]
    temp = 10**10
    for k in range(i,j):
           temp = min(optimizer(P,i,k,memo)+optimizer(P,k+1,j,memo)+P[i-1]*P[k]*P[j],temp)
    memo[key] = temp
    return temp
def steps(mem,i,j,inde,P,level = 0):
    mi = mem[(i,j)]
    
    if i<j:
        ind = []
        for k in range(i,j):
            if mem[(i,k)]+mem[(k+1,j)]+P[i-1]*P[k]*P[j] == mi:
                ind = [i,k,j]
                print(ind)
                break
        if len(ind) != 0:
            inde.append([ind,level])
            steps(mem,i,k,inde,P,level = level +1)
            steps(mem,k+1,j,inde,P,level = level +1)
          
def optimizer_bottom(P):
     memo = []
     for i in range(0,len(P)):
         memo.append([0]*(len(P)))
     for i in range(1,len(P)):
         for j in range(1,len(P)-1):
             for k in range(1,len(P-1)):
                 if