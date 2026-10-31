# host model of the --commit test's cases (index logic of k_comp_pool_dev / k_comp_state / k_comp_commit), to check the expected counts
def run(P0,k,a,commit):
    ck={}; slot0=[None]
    def step(start,toks):
        n=len(toks)
        for kk in range((n+1)//2):
            r=2*kk if start&1 else 2*kk+1
            if r>=n: continue
            grp=(start+r)>>1
            first=slot0[0] if r==0 else toks[r-1]
            ck[grp]=(first,toks[r])
        if ((start+n-1)&1)==0: slot0[0]=toks[n-1]
    real=lambda pos:('t',pos)
    step(0,[real(i) for i in range(P0)])
    p=P0;n=k+1;Pa=p+a
    step(p,[real(p+r) if r<=a else ('rej',p+r) for r in range(n)])
    if commit and Pa%2==0: slot0[0]=real(Pa)   # row a == position Pa
    for t in (Pa+1,Pa+2,Pa+3): step(t,[real(t)])
    return ck,Pa
tot=pred=hit=extra=gate=0
for P0 in (4,5):
  for k in range(1,6):
    for a in range(k+1):
      tot+=1
      ck,Pa=run(P0,k,a,True); ck2,_=run(P0,k,a,False)
      truth={g:(('t',2*g),('t',2*g+1)) for g in range((Pa+4)//2)}
      bad=lambda c:[g for g in range((Pa+4)//2) if c[g]!=truth[g]]
      gate+=bool(bad(ck))
      predicted=(Pa%2==0) and a<k
      b=bad(ck2)
      if predicted: pred+=1; hit+=(b==[(Pa+1)//2])
      else: extra+=bool(b)
print(tot,'cases gate_bad',gate,'predicted',pred,'hit(exactly key row)',hit,'extra',extra)
