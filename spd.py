import sys,time,random
sys.path.insert(0,'.')
from engine.cards import Card,SUITS,RANKS_BY_DECK
from engine.strategy import recommend_for_mode
from engine.tracker import CardTracker
rng=random.Random(2); tr=[];tm=[]
for it in range(300):
    deck=[Card(s,r) for s in SUITS for r in RANKS_BY_DECK[32]]; rng.shuffle(deck)
    trump=rng.choice(SUITS); hand=deck[:6]; table=deck[6:6+rng.randint(0,3)]
    t=CardTracker("2v2"); snap=t.update(hand,table)
    t0=time.perf_counter(); recommend_for_mode(hand,table,trump,"2v2",snap=snap,use_mc=False); tr.append(time.perf_counter()-t0)
    t0=time.perf_counter(); recommend_for_mode(hand,table,trump,"2v2",snap=snap,use_mc=True,budget=0.35); tm.append(time.perf_counter()-t0)
print("instant ms avg/max %.1f/%.1f"%(1000*sum(tr)/len(tr),1000*max(tr)),"| MC ms avg/max %.0f/%.0f"%(1000*sum(tm)/len(tm),1000*max(tm)))
# memory check: play full round, tracker must know every played card
deck=[Card(s,r) for s in SUITS for r in RANKS_BY_DECK[32]]; rng.shuffle(deck)
mine=deck[:8]; t=CardTracker("2v2"); played=[]
for k in range(8):
    trick=[mine[k]]+deck[8+3*k:8+3*k+3]; played+=trick
    for n in range(1,5): snap=t.update(mine[k:],trick[:n])
    t.update(mine[k+1:],[])
snap=t.snapshot(set(),set()); print("tricks_done",t.tricks_done,"unseen",snap["unseen_count"],"expected",32-len(played))
