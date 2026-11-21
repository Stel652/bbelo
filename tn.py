import sys,types,cv2,glob
sys.path.insert(0,'.')
try: import mss
except ImportError: sys.modules["mss"]=types.SimpleNamespace(mss=lambda:None)
import config
from engine.vision import VisionEngine
A="/mnt/user-data/attachments/43491dc6-18b9-593b-9fc6-376a91e8f3e8/"
# файл: (козырь, объявил_я)
T={"2e55c5b6":("clubs",True),"9905ec8f":("diamonds",False),"a4b652b9":("clubs",False),"27405f2d":("diamonds",False),"3129c4de":("spades",False),
   "cdaac949":("hearts",True),"c38a995b":("hearts",True),"3337589f":("clubs",True),"2c8abeae":("diamonds",False),"bea6cd27":("spades",False)}
v=VisionEngine(); ok_t=ok_n=0
for k,(tr,me) in T.items():
    img=cv2.imread(A+k+"-image.png"); H,W=img.shape[:2]
    cur={"img":img}
    v.screen_size=lambda W=W,H=H:(W,H)
    v.capture=lambda z,img=img:img[z[1]:z[1]+z[3],z[0]:z[0]+z[2]]
    v._my_name_img=None
    t=v.read_trump_icon(); n=v.declarer_is_me()
    ok_t+= (t==tr); print(k,W,"козырь",t,"(надо",tr+")","✓" if t==tr else "✗","| я выбрал:",n,"(надо",me,")")
print("козыри верно",ok_t,"из",len(T))
