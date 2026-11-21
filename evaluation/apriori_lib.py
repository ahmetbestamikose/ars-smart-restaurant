"""Minimal Apriori (itemsets up to size 3) and rule generation used for the offline evaluation.

Support(A) = n(A)/N, Confidence(A->B) = Support(A u B)/Support(A), Lift = Confidence/Support(B).
A tolerance of 1e-9 is used on both thresholds so that values that are exactly 0.002 or 0.1 on
paper are not lost to floating-point rounding (this gives the 542 rules reported in the article).
"""
import itertools, collections
EPS = 1e-9
def freq(tx,minsup,maxk=3):
    N=len(tx); F={}
    c=collections.Counter(i for t in tx for i in t)
    L={frozenset([i]):v/N for i,v in c.items() if v/N>=minsup-EPS}; F.update(L)
    k=2
    while L and k<=maxk:
        items=sorted({i for s in L for i in s})
        cands=set()
        Ls=list(L)
        for a,b in itertools.combinations(Ls,2):
            u=a|b
            if len(u)==k and all(frozenset(x) in L for x in itertools.combinations(u,k-1)): cands.add(u)
        cnt=collections.Counter()
        for t in tx:
            for cnd in cands:
                if cnd<=t: cnt[cnd]+=1
        L={s:v/N for s,v in cnt.items() if v/N>=minsup-EPS}; F.update(L); k+=1
    return F
def rules(F,minconf,single_conseq=False):
    R=[]
    for s,sup in F.items():
        if len(s)<2: continue
        for r in range(1,len(s)):
            for A in itertools.combinations(s,r):
                A=frozenset(A); B=s-A
                if single_conseq and len(B)!=1: continue
                conf=sup/F[A]
                if conf>=minconf-EPS: R.append((A,B,sup,conf,conf/F[B]))
    return R
