import sys,types,cv2,glob,os
sys.path.insert(0,'.')
try: import mss
except ImportError: sys.modules["mss"]=types.SimpleNamespace(mss=lambda:None)
import config
from engine.vision import VisionEngine
SYM={"hearts":"h","diamonds":"d","clubs":"c","spades":"s"}
A="/mnt/user-data/attachments/43491dc6-18b9-593b-9fc6-376a91e8f3e8/"
T={"2e55c5b6":"Kd 9s Js 9c Jc Qc Kc","9905ec8f":"7c 9c 9h Jh 7s Js Ks","a4b652b9":"Qd Kd 7s 9h 10h Qh 7c Qc","27405f2d":"Jc Qc Ac 9h 7s As 7d Kd",
"3129c4de":"Qh 8c Jc Qc Kc Qd Ad 10s","cdaac949":"9c Jc Kc Ac Jd Ad Js As","74c220c8":"9c Jc Kc Ac Jd Ad Js As","c38a995b":"Jc Kd Ad 10s Ks 9h Jh Qh",
"3337589f":"9c 10d 9s Js Ks","2c8abeae":"9c 10h Qh 9s Js Ks 10d Kd","bea6cd27":"9h 10h Ah 9c Qc 9d Ad As"}
v=VisionEngine(); bad=0
for k,t in T.items():
    f=glob.glob(A+k+"*")[0]; img=cv2.imread(f); H,W=img.shape[:2]
    x,y,w,h=config.scaled_zone(config.ZONE_MY_HAND_OLD,W,H)
    got=[c.rank+SYM[c.suit] for c,_ in v.detect_cards_in_image(img[y:y+h,x:x+w],vmin=105)]
    ok=sorted(got)==sorted(t.split()); bad+=not ok
    print(k,W,"OK" if ok else "FAIL",got if not ok else "", "" if ok else "truth "+t)
print("fail",bad)
