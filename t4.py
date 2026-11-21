import sys; sys.path.insert(0,'.')
from engine.cards import Card as C
from engine.hit_filters import lead_suit_fix
import tests.test_hit_filters as T
# старые тесты (без pytest)
for n in dir(T):
    if n.startswith('test_'):
        try: getattr(T,n)(); print('ok  ',n)
        except Exception as e: print('FAIL',n,repr(e)[:100])
# новый случай из журнала: стол Q♣, в руке 10♠, козырь ♠ -> НЕ менять
print('trump cut  :',lead_suit_fix([C('spades','10')],[C('clubs','Q')],trump='spades'))
# стабильная карта -> НЕ менять
print('stable     :',lead_suit_fix([C('spades','10')],[C('clubs','Q')],trump='hearts',stable={C('spades','10')}))
# нестабильная, не козырь -> менять как раньше
print('unstable   :',lead_suit_fix([C('spades','10')],[C('clubs','Q')],trump='hearts'))
