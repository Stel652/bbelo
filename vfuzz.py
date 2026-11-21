import sys, os, types, cv2, numpy as np
sys.path.insert(0,'.')
try: import mss
except ImportError: sys.modules["mss"]=types.SimpleNamespace(mss=lambda:None)
import config
from engine.vision import VisionEngine
sys.path.insert(0,'tests')
from test_vision_screens import TRUTH, NEW, SYM
v=VisionEngine(); rng=np.random.default_rng(0); bad=0; tot=0
def run(img):
    H,W=img.shape[:2]; x,y,w,h=config.scaled_zone(config.ZONE_MY_HAND_OLD,W,H)
    return [c.rank+SYM[c.suit] for c,_ in v.detect_cards_in_image(img[y:y+h,x:x+w],vmin=105)]
for name,truth in [(f"s{i}",t) for i,t in TRUTH.items()]+[(f"n{i}",h) for i,(h,_) in NEW.items()]:
    base=cv2.imread(f"tests/screens/{name}.png")
    for W,H in [(1280,720),(1366,768),(1600,900),(1920,1080),(2560,1440)]:
        img=cv2.resize(base,(W,H),interpolation=cv2.INTER_AREA if W<base.shape[1] else cv2.INTER_CUBIC)
        for mode in ("clean","jpeg","noise","blur"):
            im=img.copy()
            if mode=="jpeg": im=cv2.imdecode(cv2.imencode(".jpg",im,[cv2.IMWRITE_JPEG_QUALITY,70])[1],1)
            if mode=="noise": im=np.clip(im+rng.normal(0,6,im.shape),0,255).astype(np.uint8)
            if mode=="blur": im=cv2.GaussianBlur(im,(3,3),0)
            tot+=1; got=run(im)
            if got!=truth: bad+=1; print(name,W,mode,got,"!=",truth)
print("vision cases",tot,"fail",bad)
