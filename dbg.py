import sys,types,cv2
sys.path.insert(0,'.')
try: import mss
except ImportError: sys.modules["mss"]=types.SimpleNamespace(mss=lambda:None)
import config
from engine.vision import VisionEngine
v=VisionEngine()
base=cv2.imread("tests/screens/s5.png"); W,H=1600,900
img=cv2.resize(base,(W,H),interpolation=cv2.INTER_CUBIC)
x,y,w,h=config.scaled_zone(config.ZONE_MY_HAND_OLD,W,H)
z=img[y:y+h,x:x+w]
hits=v.detect_cards_in_image(z,vmin=105)
print([str(c) for c,_ in hits])
boxes,_=v._find_card_boxes(z,105)
for b in boxes:
    crop=z[b[1]:b[1]+b[3],b[0]:b[0]+b[2]]
    r=v._ranked(crop)[:3]; rp,pp,red=v._idx_patches(crop)
    rs={k:round(max(v._ncc(rp,t) for t in ts),2) for k,ts in v._g_rank.items()}
    print(b[2:],red,[(round(a,2),str(c)) for a,_,_,c in r], sorted(rs.items(),key=lambda t:-t[1])[:3])
