import sys,types,cv2,glob,json,time
sys.path.insert(0,'.')
try: import mss
except ImportError: sys.modules["mss"]=types.SimpleNamespace(mss=lambda:None)
import config
from engine.vision import VisionEngine
v=VisionEngine()
for jf in sorted(glob.glob('../data/shots/*.json')):
    j=json.load(open(jf)); img=cv2.imread(jf[:-5]+'_full.jpg'); H,W=img.shape[:2]
    out={}
    for nm,z in (('hand',config.ZONE_MY_HAND),('table',config.ZONE_TABLE)):
        x,y,w,h=config.scaled_zone(z,W,H); crop=img[y:y+h,x:x+w]
        t=time.time(); hits=v.detect_cards_in_image(crop,vmin=105); out[nm]=([str(c) for c,_ in hits],round(time.time()-t,2))
    print(jf.split('/')[-1],W,H,'\n  truth hand',j['hand'],'\n  got',out)
