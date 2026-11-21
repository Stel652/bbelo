import sys,types,cv2
sys.path.insert(0,'.')
try: import mss
except ImportError: sys.modules["mss"]=types.SimpleNamespace(mss=lambda:None)
from engine.vision import VisionEngine
SYM={"hearts":"h","diamonds":"d","clubs":"c","spades":"s"}
A="/mnt/user-data/attachments/43491dc6-18b9-593b-9fc6-376a91e8f3e8/"
v=VisionEngine()
# (файл, зона руки x,y,w,h, зона стола x,y,w,h, масштаб карты)
J=[("fa668338",(0,800,790,165),(0,270,700,420),1.55),("e886bd97",(0,725,820,140),(45,235,715,380),1.4),
   ("1ec9c76b",(0,730,750,125),(40,295,650,340),1.35),("333f6d00",(70,820,810,130),(150,375,655,345),1.4)]
for k,hz,tz,sc in J:
    img=cv2.imread(A+k+"-image.jpg")
    x,y,w,h=hz; hand=v.detect_cards_in_image(img[y:y+h,x:x+w],vmin=105)
    x,y,w,h=tz; tab=v.detect_cards_in_image(img[y:y+h,x:x+w],vmin=185,scale=sc)
    print(k,"рука",[c.rank+SYM[c.suit] for c,_ in hand],"стол",[c.rank+SYM[c.suit] for c,_ in tab])
