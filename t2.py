import sys,types,cv2,glob,json,time
sys.path.insert(0,'.')
try: import mss
except ImportError: sys.modules["mss"]=types.SimpleNamespace(mss=lambda:None)
import config
from engine.vision import VisionEngine
v=VisionEngine()
for jf in sorted(glob.glob('../data/shots/*played*.json'))+sorted(glob.glob('../data/shots/*mismatch*.json')):
    img=cv2.imread(jf[:-5]+'_full.jpg'); H,W=img.shape[:2]
    x,y,w,h=config.scaled_zone(config.ZONE_TABLE,W,H); crop=img[y:y+h,x:x+w]
    res={}
    for vm in (185,165,200,105):
        t=time.time()
        hits=v.detect_cards_in_image(crop,vmin=vm,scale=w/float(config.ZONE_TABLE[2]))
        res[vm]=([(str(c),p) for c,p in hits],round(time.time()-t,2))
    print(jf.split('/')[-1]); [print('  ',k,r) for k,r in res.items()]
