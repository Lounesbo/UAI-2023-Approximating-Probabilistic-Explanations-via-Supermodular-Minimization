import numpy as np
import sys
sys.path.append('build')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from math import *

# Pysat packages
from pysat.card import *
from pysat.pb import PBEnc
from pysat.formula import WCNF, CNF
from pysat.pb import *
from pysat.examples.rc2 import RC2
from pysat.solvers import Glucose4
from pysat.solvers import Solver, Minisat22

#ours package 
import my_tree as mt
import my_tree

# SAT encoding of "Arenas et al 2022" packages
import utils
import dtree
import gen_dt
import encoder 

# Intermediary functions and error function "M(s"

def W_f(dnf, t, n): # dnf : Orthogonal dnf, t: term, n: dimension "d"  
    w = 0
    def check(C, t_d): # C: term in dnf, t_d: subset of features
        for c in t_d:
            if -c in C:
                return 0
        return len(set(C+t_d))
    for cl in dnf:
        if check(cl, t) != 0:
            w += 2**(n-check(cl, t))
    return w/2**(n-len(t)) # return the number of models of dnf ∧ t divide by 2**(n-size(t))
    

def M_s(phi, t, n): # phi : Orthogonal DNF, t: term, n: number of features  
    w = 0
    def check(C, t_d): # C: term in dnf, t_d: output subset desired
        for c in t_d:
            if -c in C:
                return 0
        return len(set(C+t_d))
    for cl in phi:
        if check(cl, t) != 0:
            w += 2**(n-check(cl, t))
    return w # return the number of models of phi ∧ t

def D_s(Q, l): #help function for greedy dscend 
    O = Q.copy()
    O.remove(l)
    return O
    
    
def Dichotomie(a, b, k, ins, R, e=0.01, time_out=1800): 
    
    error = 1

    def Delta(d, k, R = R, file = 'data_dt'): # SAT calls to use the dichotomous search
        t_solve = 600
        clauses = encoder.generate_encoding(tree_filename=file, instance=ins, delta=d, k=k)
        V = Solver(name='g4', use_timer=True)
        for cl in clauses:
            V.add_clause(cl)
            
        bol = V.solve_limited(assumptions=[-l for l in range(1, len(ins)+1) if l not in R])
        t_v = V.time()
        V.delete()   
        
        return bol

    while error > e:
        d = a + (b - a) / 2

        error = abs(b - a)
        if Delta(d, k):
            a = d           
        else :
            b = d
            
    return d

    
    
def Approx_descent(phi, I_s, n, k=None):     
    """
    Input:
      phi: phi: the oracle of DT "dnf monotone" 
      I_s: subset on variable "or direct reason R_d"
      n: the dimension "d"
      k: the size of probable explanation desiread
    Output:
      subset of variable of size at most k
    """ 
    
    S_k = [] # stock a solution of approx_descent
    S = I_s.copy() # step 1
    i = 1
    if k == None or k > len(I_s):
        k = len(I_s)
    while i <= len(I_s):
        e = S[np.argmin([M_s(phi, D_s(S, l), n=n) for l in S])]
        S_k.append(S.copy())
        S.remove(e)
        i = i+1
    S_k.reverse() # inverse the ordre of the list of solution
    
    return S_k[np.argmin([W_f(dnf=phi, t=C, n=n) for C in S_k[0:k]])]

def Approx_ascendant(phi, I_s, n, k=None):     
    """
    Input:
      phi: phi: the oracle of DT "dnf monotone" 
      I_s: subset on variable "or direct reason R_d"
      n: the dimension "d"
      k: the size of probable explanation desiread
    Output:
      subset of variable of size at most k
    """ 
    S = []
    S_k = [] # stock the solutions for each iteration 
    X = I_s.copy()
    
    if k == None or k > len(I_s):
        k = len(I_s)
        
    M_0 = M_s(phi, [], n)
    c = M_0 - np.min([M_s(phi, [l], n) for l in X]) # compute the curvature "c"
    
    j = k # Because we want a subset of size at most k, we get simply j=k
    #See paper for more detail about the algorithm 2 "section 4.2"        
    
    for i in range(j): 
        e = X[np.argmin([M_s(phi, S+[l], n) for l in X])]
        S = S + [e]
        S_k.append(S)
        X.remove(e)
        
    return S_k[np.argmin([W_f(dnf=phi, t=C, n=n) for C in S_k])]    
    
    
    

