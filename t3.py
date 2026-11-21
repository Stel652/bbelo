import sys,types,cv2,glob,json,collections
sys.path.insert(0,'.')
try: import mss
except ImportError: sys.modules["mss"]=types.SimpleNamespace(mss=lambda:None)
import config
from engine.vision import VisionEngine
v=VisionEngine()
rows=[]
for d in ('/home/claude/w/data','/home/claude/d1/data'):
  for f in sorted(glob.glob(d+'/shots/*_full.jpg')):
    im=cv2.imread(f); H,W=im.shape[:2]
    x,y,w,h=config.scaled_zone(config.ZONE_MY_HAND,W,H); z=im[y:y+h,x:x+w]
    for nm,zn in (('hand',config.ZONE_MY_HAND),('table',config.ZONE_TABLE)):
        x,y,w,h=config.scaled_zone(zn,W,H); z=im[y:y+h,x:x+w]
        sc=None if nm=='hand' else w/float(config.ZONE_TABLE[2])
        hits=v.detect_cards_in_image(z,vmin=105 if nm=='hand' else 185,scale=sc)
        boxes,_=v._find_card_boxes(z,105 if nm=='hand' else 185)
        for c,p in hits:
            if c.suit in('clubs','spades'):
                for b in boxes:
                    if b[0]<=p[0]<=b[0]+b[2] and b[1]<=p[1]<=b[1]+b[3]:
                        crop=z[b[1]:b[1]+b[3],b[0]:b[0]+b[2]]
                        rows.append((f.split('/')[-1][:22],nm,str(c),v._black_suit(crop),v._spade_shape(crop),crop)); break
print(len(rows))
for r in rows[:60]: print(r[0],r[1],r[2],r[3] and (r[3][0],round(r[3][1],2)),r[4] and round(r[4],2))
import pickle; pickle.dump([(r[2],r[3],r[4],r[5]) for r in rows],open('/home/claude/w/rows.pkl','wb'))
