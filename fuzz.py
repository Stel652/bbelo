import random, sys, inspect
sys.path.insert(0,'.')
from engine.cards import Card, SUITS, RANKS_BY_DECK
from engine.rules import can_play_card
from engine.strategy import recommend_for_mode
print(inspect.signature(recommend_for_mode))
def oracle(card, table, hand, trump):
    if not table: return True
    lead=table[0].suit
    st=lambda c:c.strength(trump)
    same=[c for c in hand if c.suit==lead]
    if same:
        if card.suit!=lead: return False
        if lead==trump:
            hi=max(st(c) for c in table if c.suit==trump)
            up=[c for c in same if st(c)>hi]
            if up: return st(card)>hi
        return True
    tr=[c for c in hand if c.suit==trump]
    if tr:
        if card.suit!=trump: return False
        tt=[c for c in table if c.suit==trump]
        if tt:
            hi=max(st(c) for c in tt); up=[c for c in tr if st(c)>hi]
            if up: return st(card)>hi
        return True
    return True
rng=random.Random(1); bad=0; badrec=0; n=0
for it in range(20000):
    mode=rng.choice(["2v2","1v1","3p"]); deck=[Card(s,r) for s in SUITS for r in RANKS_BY_DECK[32 if mode=="2v2" else 24]]
    rng.shuffle(deck); trump=rng.choice(SUITS)
    k=rng.randint(1,{"2v2":3,"1v1":1,"3p":2}[mode]); hs=rng.randint(1,8 if mode=="2v2" else 6)
    table=deck[:k]; hand=deck[k:k+hs]
    for c in hand:
        a=can_play_card(c,table,hand,trump,True); b=oracle(c,table,hand,trump)
        n+=1
        if a!=b: bad+=1; print("MISMATCH",[str(x) for x in table],[str(x) for x in hand],str(c),a,b) if bad<5 else 0
    try:
        r=recommend_for_mode(hand,table,trump,mode)
        card=r[0] if isinstance(r,tuple) else r
        if card is None or not oracle(card,table,hand,trump):
            badrec+=1
            if badrec<5: print("ILLEGAL REC",[str(x) for x in table],[str(x) for x in hand],trump,r)
    except Exception as e:
        badrec+=1; print("EXC",repr(e)) if badrec<5 else 0
print("checked",n,"rule mismatches",bad,"bad recs",badrec)
