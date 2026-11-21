# engine/vision.py
import cv2, numpy as np, mss, os
from typing import List, Optional, Tuple
from .cards import Card
import config

def _imread_unicode(path):
    """Read image files on Windows even when the path contains Unicode."""
    try:
        raw = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(raw, cv2.IMREAD_COLOR) if raw.size else None
    except (OSError, ValueError):
        return None

class VisionEngine:
    SHAPE_MIN_W = 66
    SHAPE_W = 0.02   # вес формы значка (♣/♠) против сравнения с эталонами
    def __init__(self):
        self.templates = self._load_templates()
        self.trump_templates = self._load_trump_templates()
        # Углы карт (ранг + масть): виден даже когда карты перекрываются веером,
        # и не зависит от остальной части карты. Масштабы с запасом.
        self._scaled_templates = []
        self._red = {}
        self._hist = {}
        for (rank, suit), tpl in self.templates.items():
            h, w = tpl.shape[:2]
            corner = tpl[0:int(h*0.40), 0:int(w*0.34)]
            red = self._red_ratio(corner)
            for scale in (0.22, 0.25, 0.28, 0.31, 0.34, 0.37):
                size = (int(corner.shape[1]*scale), int(corner.shape[0]*scale))
                if size[0] >= 6 and size[1] >= 6:
                    self._scaled_templates.append((rank, suit, cv2.resize(corner, size, interpolation=cv2.INTER_AREA), size))
                    self._red[(rank, suit)] = red
        self.allowed_ranks = None   # например {'9','10','J','Q','K','A'} для колоды из 24 карт
        self._sct = mss.mss()
    @staticmethod
    def _red_ratio(img):
        b, g, r = [img[..., i].astype(int) for i in range(3)]
        red = ((r > 140) & (g < 110) & (b < 110)).sum()
        dark = ((r < 90) & (g < 90) & (b < 90)).sum()
        return red / max(1, red + dark)

    def warmup(self):
        """Строит эталоны заранее (раньше это происходило на первом кадре и подвешивало подсказку)."""
        self.detect_cards_in_image(np.zeros((120, 300, 3), np.uint8))

    def screen_size(self):
        mon = self._sct.monitors[1]
        return mon["width"], mon["height"]

    def _load_templates(self):
        t = {}
        d = os.path.join(os.path.dirname(__file__), "..", "assets", "templates")
        if not os.path.exists(d):
            print(f"⚠️ Нет папки {d}")
            return t
        for f in os.listdir(d):
            if not f.endswith(".png") or f.startswith("trump_"):
                continue
            parts = f[:-4].split("_")
            if len(parts) != 2:
                continue
            img = _imread_unicode(os.path.join(d, f))
            if img is not None:
                t[tuple(parts)] = img
        print(f"Загружено шаблонов карт: {len(t)}")
        return t

    def _load_trump_templates(self):
        t = {}
        d = os.path.join(os.path.dirname(__file__), "..", "assets", "templates")
        for suit in ["hearts", "diamonds", "clubs", "spades"]:
            path = os.path.join(d, f"trump_{suit}.png")
            if os.path.exists(path):
                t[suit] = _imread_unicode(path)
        return t

    def capture(self, zone):
        x, y, w, h = zone
        if w <= 1 or h <= 1:
            raise ValueError("Screen zone is unset; configure coordinates in config.py")
        mon = self._sct.monitors[1]
        x=max(0,min(x,mon["width"]-1)); y=max(0,min(y,mon["height"]-1))
        w=max(1,min(w,mon["width"]-x)); h=max(1,min(h,mon["height"]-y))
        img=np.array(self._sct.grab({"top": y, "left": x, "width": w, "height": h}))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def detect_cards(self, zone, dim=False) -> List[Tuple[Card, Tuple[int, int]]]:
        """Template-based card detection with multi-hit NMS.

        Unlike the old implementation, it does not keep only one candidate per
        rank, so two or more cards sharing a rank can be reported. The score is
        conservative because false positives corrupt the game state.
        """
        img = self.capture(zone)
        key = tuple(zone)
        small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (64, 24), interpolation=cv2.INTER_AREA).astype(np.int16)
        prev = getattr(self, "_prev_small", {}).get(key)
        if not hasattr(self, "_same_cnt"):
            self._same_cnt = {}
        if prev is not None and float(np.abs(small - prev).mean()) < 1.5 and key in getattr(self, "_last_out", {}):
            self._same_cnt[key] = self._same_cnt.get(key, 0) + 1
            if self._same_cnt[key] % 4:
                return self._last_out[key]        # кадр не изменился — но каждые 4 кадра перепроверяем (с другим порогом)
            retry = True
        else:
            self._same_cnt[key] = 0
            retry = False
        if not hasattr(self, "_prev_small"):
            self._prev_small, self._last_out = {}, {}
        self._prev_small[key] = small
        vmins = (105, 90, 125) if dim else (185, 165, 200)
        vm = vmins[(self._same_cnt[key] // 4) % 3] if retry else vmins[0]
        hits = self.detect_cards_in_image(img, vmin=vm,
                                          scale=None if dim else zone[2] / float(config.ZONE_TABLE[2]),
                                          hand_scale=(zone[2] / float(config.ZONE_MY_HAND[2])) if dim else None)
        # Если карт стало меньше, чем в прошлый раз (мигание/приглушённые карты) — сразу добираем
        # другими порогами яркости и объединяем, чтобы карта в руке не «исчезала».
        prev_n = getattr(self, "_prev_n", {}).get(key, 0)
        if dim and len(hits) < prev_n and not retry:
            have = {c for c, _ in hits}
            for vm2 in (90, 125):
                try:
                    extra = self.detect_cards_in_image(img, vmin=vm2, hand_scale=zone[2] / float(config.ZONE_MY_HAND[2]))
                except Exception:
                    extra = []
                for c, pos in extra:
                    if c not in have:
                        hits.append((c, pos)); have.add(c)
                if len(hits) >= prev_n:
                    break
        if not hasattr(self, "_prev_n"):
            self._prev_n = {}
        self._prev_n[key] = len(hits)
        out = self._smooth(key, hits)
        self._last_out[key] = out
        return out

    def _smooth(self, key, hits, n=3):
        """Голосование по последним n кадрам: карта считается видимой, если
        есть минимум в половине кадров на том же месте (убирает мигание и случайные ошибки)."""
        from collections import deque, Counter
        hist = self._hist.setdefault(key, deque(maxlen=n))
        hist.append(hits)
        slots = {}
        for frame in hist:
            for card, (cx, cy) in frame:
                k = (round(cx / 30), round(cy / 30))
                slots.setdefault(k, []).append((card, cx, cy))
        out = []
        for votes in slots.values():
            w = Counter()
            for i_, (c_, _, _) in enumerate(votes): w[c_] += 1 + 0.01 * i_    # при равенстве побеждает свежий кадр
            card, cnt = w.most_common(1)[0]
            if cnt >= 1:
                cx = [x for c, x, _ in votes if c == card][-1]; cy = [y for c, _, y in votes if c == card][-1]
                out.append((card, (cx, cy)))
        out = [(c, p) for c, p in out]
        seen, res = set(), []
        for c, p in sorted(out, key=lambda it: (it[1][0], it[1][1])):
            if c not in seen:
                seen.add(c); res.append((c, p))
        return res

    def _detect_corner_templates(self, img):
        if img is None or img.size == 0:
            return []
        candidates = []
        for rank, suit, tpl, size in self._scaled_templates:
            if tpl.shape[0] > img.shape[0] or tpl.shape[1] > img.shape[1]:
                continue
            result = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
            # Keep up to four local maxima per card template, suppressing the
            # neighborhood around each maximum before looking for the next.
            work = result.copy()
            for _ in range(8):
                _, score, _, loc = cv2.minMaxLoc(work)
                if score < config.TEMPLATE_MATCH_THRESHOLD:
                    break
                x, y = loc
                patch = img[y:y+size[1], x:x+size[0]]
                is_red = self._red_ratio(patch) > 0.5
                if is_red != (suit in ("hearts", "diamonds")) and (patch.size and
                        abs(self._red_ratio(patch) - self._red[(rank, suit)]) > 0.5):
                    # цвет не совпал: этот угол другой масти — гасим и идём дальше
                    x1, y1 = max(0, x-size[0]//2), max(0, y-size[1]//2)
                    work[y1:y+size[1]//2+1, x1:x+size[0]//2+1] = -1
                    continue
                candidates.append((float(score), Card(suit, rank),
                                   (x + size[0] // 2, y + size[1] // 2), size))
                x1, y1 = max(0, x-size[0]//2), max(0, y-size[1]//2)
                x2, y2 = min(work.shape[1], x+size[0]//2+1), min(work.shape[0], y+size[1]//2+1)
                work[y1:y2, x1:x2] = -1

        candidates.sort(key=lambda item: item[0], reverse=True)
        found, occupied, used_cards = [], [], set()
        for score, card, center, size in candidates:
            if card in used_cards:
                continue
            cx, cy = center
            # Один угол = одна карта: подавляем соседей в пределах угла.
            if any(abs(cx-ox) < max(size[0], ow)*0.9 and abs(cy-oy) < max(size[1], oh)*0.9
                   for ox, oy, ow, oh in occupied):
                continue
            # Угол не может лежать внутри тела другой карты ниже её индекса
            # (это значки/пипсы в центре карты, а не новая карта).
            if any(ox - ow*0.6 < cx < ox + ow*3.0 and oy + oh*0.9 < cy < oy + oh*2.6
                   for ox, oy, ow, oh in occupied):
                continue
            found.append((card, center))
            occupied.append((cx, cy, size[0], size[1]))
            used_cards.add(card)
        self.last_scores = {str(c): round(s_, 2) for s_, c, *_ in candidates[:12]}
        # Порядок слева направо (рука) — затем сверху вниз.
        found.sort(key=lambda item: (item[1][0], item[1][1]))
        return found


    # ---------- Новый метод: поиск белых карт + разбор индекса (ранг и масть) ----------
    @staticmethod
    def _find_card_boxes(img, vmin=185):
        # vmin=185 — обычные карты; ~105 — рука: карты, которые нельзя сыграть, затемнены игрой
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        white = ((hsv[..., 2] > vmin) & (hsv[..., 1] < 45)).astype(np.uint8) * 255
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes, wide = [], []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w < 24 or h < 34:
                continue
            r = h / w
            if r >= 1.2 and cv2.contourArea(c) < 0.6 * w * h:
                continue
            if 1.2 <= r <= 1.75:
                boxes.append((x, y, w, h))
            elif r < 1.2 and h >= 40:
                # Слипшиеся карты: режем по пустым вертикальным промежуткам,
                # смотря только на верхнюю часть (снизу их может склеивать панель).
                rows = (white[y:y+h, x:x+w] > 0).mean(axis=1) > 0.10
                best, run0 = (0, 0), None
                for i_, v_ in enumerate(list(rows) + [False]):
                    if v_ and run0 is None: run0 = i_
                    if not v_ and run0 is not None:
                        if i_ - run0 > best[1] - best[0]: best = (run0, i_)
                        run0 = None
                y, h = y + best[0], max(1, best[1] - best[0])      # только полоса с картами
                if h < 40:
                    wide.append((x, y, w, h)); continue
                # зазоры между картами = тёмные столбцы фона (лица J/Q/K не белые, поэтому по белому нельзя);
                # полоса 55–86% высоты блока: ниже кнопок Tărț/Bella и выше панели задач, на любом масштабе
                sub = hsv[y + int(h * 0.55): y + int(h * 0.86), x:x + w, 2]
                gap = (sub < 90).mean(axis=0) > 0.85
                segs2, st2 = [], None
                for i2, v2 in enumerate(list(~gap) + [False]):
                    if v2 and st2 is None: st2 = i2
                    if not v2 and st2 is not None:
                        segs2.append((st2, i2)); st2 = None
                if segs2:                                        # ширина карты — по медиане, а не по высоте блока
                    med = float(np.median([b - a for a, b in segs2]))
                    segs2 = [(a, b) for a, b in segs2 if 0.8 * med <= b - a <= 1.25 * med] if h * 0.35 <= med <= h * 0.85 else []
                if len(segs2) >= 2:                              # карты разделены зазорами
                    hc = min(h, int(round(float(np.median([b - a for a, b in segs2])) * 1.43)))
                    for a, b in segs2:
                        boxes.append((x + a, y, b - a, hc))
                    continue
                top = white[y+1:y+max(5, int(h*0.07)), x:x+w] > 0   # белое поле над картой
                prof = top.mean(axis=0) > 0.6
                segs, start = [], None
                for i, v_ in enumerate(list(prof) + [False]):
                    if v_ and start is None: start = i
                    if not v_ and start is not None:
                        segs.append((start, i)); start = None
                segs = [(a, b) for a, b in segs if b - a >= 24]
                if segs and all(1.0 <= (b - a) / max(1, h) * 1.0 * 1.0 + 0.0 or True for a, b in segs):
                    ch_ = int(np.median([(b - a) for a, b in segs]) * 1.43)
                    ch_ = min(ch_, h)
                    avg = np.mean([(b - a) for a, b in segs])
                    if all(abs((b - a) - avg) < avg * 0.35 for a, b in segs) and len(segs) >= 1 and avg < h * 0.9:
                        for a, b in segs:
                            boxes.append((x + a, y, b - a, ch_))
                        continue
                cw_ = h / 1.40                                  # ширина одной карты (70x98)
                n_ = int(round(w / cw_))
                if n_ >= 2 and abs(w / cw_ - n_) < 0.35 and h >= 40:
                    step = w / n_
                    for k in range(n_):                         # карты вплотную без зазоров
                        boxes.append((int(x + k * step), y, int(step), h))
                    continue
                wide.append((x, y, w, h))   # перекрытые карты веером
        return boxes, wide

    @staticmethod
    def _norm(mask, size):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        m = mask[ys.min():ys.max()+1, xs.min():xs.max()+1]
        return (cv2.resize(m, size, interpolation=cv2.INTER_AREA) > 100).astype(np.uint8)

    def _index_parts(self, card_img, x1f=0.19):
        """Из картинки одной карты достаёт маски ранга и масти и цвет (красный?)."""
        h, w = card_img.shape[:2]
        y0, y1, x0, x1 = int(h*0.02), int(h*0.43), int(w*0.015), int(w*x1f)
        strip = card_img[y0:y1, x0:x1]
        if strip.size == 0:
            return None
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        b, g, r = [strip[..., i].astype(int) for i in range(3)]
        ink = ((gray < 170) | ((r > 150) & (g < 120) & (b < 120))).astype(np.uint8) * 255
        # увеличиваем для устойчивости на мелких картах
        k = max(1, int(round(60 / max(1, strip.shape[0]))))
        ink = cv2.resize(ink, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
        ink = (ink > 127).astype(np.uint8) * 255
        n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
        H_, W_ = ink.shape
        comps = []
        for i in range(1, n):
            cx, cy, cw, ch, area = st[i]
            if area < 6 * k:
                continue
            # не берём то, что упирается в границу полосы (край карты, рамка рисунка)
            if cx <= 1 or cy <= 1 or cx + cw >= W_ - 1 or cy + ch >= H_ - 1:
                continue
            if ch > H_ * 0.55 or cw > W_ * 0.95:
                continue
            if x1f < 0.2 and cx + cw / 2 > W_ * 0.62:   # индекс прижат влево; правее — рисунок
                continue
            comps.append((cy, i))
        if len(comps) < 2:
            return None
        comps.sort()
        pip_i = comps[-1][1]
        rank_ids = [i for _, i in comps[:-1]]
        pip = (lab == pip_i).astype(np.uint8) * 255
        rank = np.isin(lab, rank_ids).astype(np.uint8) * 255
        red_pix = ((r > 150) & (g < 120) & (b < 120))
        red_pix = cv2.resize(red_pix.astype(np.uint8), (ink.shape[1], ink.shape[0]), interpolation=cv2.INTER_NEAREST)
        is_red = (red_pix[pip > 0].mean() > 0.5) if (pip > 0).any() else False
        rk, sp = self._norm(rank, (14, 20)), self._norm(pip, (18, 18))
        if rk is None or sp is None:
            return None
        return rk, sp, is_red

    def _build_index_templates(self):
        self._rank_tpl, self._suit_tpl = {}, {}
        for (rank, suit), tpl in self.templates.items():
            pad = cv2.copyMakeBorder(tpl, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=(30, 30, 30))
            boxes, _ = self._find_card_boxes(pad)
            if boxes:
                x, y, w, h = max(boxes, key=lambda b: b[2]*b[3])
                crop = pad[y:y+h, x:x+w]
            else:
                crop = tpl
            parts = self._index_parts(crop, 0.24)
            if parts:
                self._rank_tpl.setdefault(rank, []).append(parts[0])
                self._suit_tpl.setdefault(suit, []).append(parts[1])

    @staticmethod
    def _iou(a, b):
        inter = (a & b).sum(); uni = (a | b).sum()
        return inter / uni if uni else 0.0


    # ---------- Эталоны именно этой игры (снятые со скриншота), фиксированная геометрия ----------
    GAME_CARD = (70, 98)
    RANK_MIN = 0.90      # ниже — ранг НЕ известен (нет эталона): лучше «не вижу», чем чужая карта

    @staticmethod
    def _idx_patches(card_img):
        c = cv2.resize(card_img, VisionEngine.GAME_CARD, interpolation=cv2.INTER_AREA)
        g = 255 - c.min(axis=2)   # красные и чёрные знаки одинаково тёмные
        rank = cv2.resize(g[2:18, 0:13], (26, 32), interpolation=cv2.INTER_CUBIC)
        pip = cv2.resize(g[14:30, 0:13], (26, 32), interpolation=cv2.INTER_CUBIC)
        # красный/чёрный по пипсу: относительно (R - среднее G,B), яркость не важна
        pc = c[14:30, 0:13].astype(int)
        paper = np.percentile(pc.max(axis=2), 90)
        ink = pc.min(axis=2) < paper * 0.62
        red_px = ink & ((pc[..., 2] - (pc[..., 1] + pc[..., 0]) / 2) > 28)
        red = red_px.sum(); dark = max(1, ink.sum() - red_px.sum())
        return rank, pip, red > 0.6 * dark

    def _load_game_idx(self):
        self._g_rank, self._g_suit = {}, {}
        d = os.path.join(os.path.dirname(__file__), "..", "assets", "game_idx")
        if not os.path.isdir(d):
            return
        for f in os.listdir(d):
            img = cv2.imread(os.path.join(d, f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            kind, name = f[:-4].split("_", 1)
            dst = (self._g_rank if kind == "rank" else self._g_suit).setdefault(name.split("#")[0], [])
            dst.append(img)
            dst.append(cv2.GaussianBlur(img, (3, 3), 0.9))     # размытая копия: сжатые/масштабированные кадры

    @staticmethod
    def _ncc(a, b):
        # a — большой патч, b — эталон (центральная часть): ищем лучший сдвиг
        if a.shape == b.shape:
            return float(cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)[0, 0])
        return float(cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED).max())


    @staticmethod
    def _spade_shape(card_img):
        """>0 — значок похож на пики, <0 — на трефы (по выпуклости и ширине верха значка); None если не вышло.
        Дополняет сравнение с эталонами: на уменьшенных/размытых кадрах ♣ и ♠ у шаблонов почти не различаются."""
        if card_img.shape[1] < VisionEngine.SHAPE_MIN_W:
            return None   # карта меньше эталонной (экран <1279 px): форма значка после апскейла врёт, остаются эталоны
        c = cv2.resize(card_img, VisionEngine.GAME_CARD, interpolation=cv2.INTER_AREA)
        g = 255 - c.min(axis=2)
        p = cv2.resize(g[14:30, 0:13], None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
        _, m = cv2.threshold(p, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        comps = [i for i in range(1, n) if st[i, 4] > 150 and st[i, 0] > 0 and st[i, 1] > 0
                 and st[i, 0] + st[i, 2] < m.shape[1] and st[i, 1] + st[i, 3] < m.shape[0]]
        if not comps:
            return None
        i = max(comps, key=lambda i: st[i, 1] + st[i, 3] / 2)
        mm = (lab == i).astype(np.uint8)
        cnt, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(cnt, key=cv2.contourArea)
        sol = cv2.contourArea(cnt) / max(1.0, cv2.contourArea(cv2.convexHull(cnt)))
        x, y, w, h = cv2.boundingRect(cnt)
        top = mm[y:y + max(1, h // 3), x:x + w].sum(axis=1).mean() / max(1, w)
        f = 0.5 * (sol - 0.9525) / 0.0175 + 0.5 * (top - 0.485) / 0.06
        return 0.0 if abs(f) < 0.4 else f   # слабый сигнал формы — шум, решают эталоны

    @staticmethod
    def _black_suit(card_img):
        """Пики или трефы по заполненности значка (у пики он плотнее, у трефы 3 лепестка)."""
        c = cv2.resize(card_img, VisionEngine.GAME_CARD, interpolation=cv2.INTER_AREA)
        g = 255 - c.min(axis=2)
        p = cv2.resize(g[15:30, 0:14], None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        m = (p > 110).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        comps = [i for i in range(1, n) if st[i, 4] > 60]
        if not comps:
            return None
        i = max(comps, key=lambda i: st[i, 1])
        x, y, w, h, a = st[i]
        fill = a / float(w * h)
        return ("spades" if fill > 0.67 else "clubs"), fill

    def _classify_game(self, card_img, relaxed=False, exclude=()):
        if not getattr(self, "_g_rank", None):
            return None
        rank_p, pip_p, red = self._idx_patches(card_img)
        suits = ("hearts", "diamonds") if red else ("clubs", "spades")
        known = [x for x in suits if x in self._g_suit]
        if not known:
            return None
        if len(known) < 2 and max(self._ncc(pip_p, t) for t in self._g_suit[known[0]]) < 0.75:
            return None   # нет эталона второй масти (♥) — берём только уверенное совпадение
        rs = {r: max(self._ncc(rank_p, t) for t in ts) for r, ts in self._g_rank.items()
              if self.allowed_ranks is None or r in self.allowed_ranks}
        ss = {s: max(self._ncc(pip_p, t) for t in ts) for s, ts in self._g_suit.items() if s in suits}
        if not rs or not ss:
            return None
        rank, suit = max(rs, key=rs.get), max(ss, key=ss.get)
        if not red and len(ss) == 2:
            f = self._spade_shape(card_img)   # шаблоны + форма значка (♣/♠ на мелких/размытых кадрах)
            d = ss['spades'] - ss['clubs'] + (self.SHAPE_W * f if f is not None else 0.0)
            suit = 'spades' if d > 0 else 'clubs'
        self.last_scores[f"{rank}{suit[0]}"] = (round(rs[rank], 2), round(ss[suit], 2))
        if relaxed:
            # второй проход: лучшая догадка среди ещё не найденных карт
            cands = sorted(((rs[r] + ss[sx], r, sx) for r in rs for sx in ss
                            if Card(sx, r) not in exclude and rs[r] >= self.RANK_MIN), reverse=True)
            if not cands or cands[0][0] < 1.5:
                return None
            return Card(cands[0][2], cands[0][1])
        if rs[rank] < self.RANK_MIN or ss[suit] < 0.6:
            return None
        # неуверенно, если второй кандидат почти так же хорош
        r2 = sorted(rs.values())[-2] if len(rs) > 1 else 0
        s2 = sorted(ss.values())[-2] if len(ss) > 1 else 0
        if rs[rank] - r2 < 0.08 or (red and ss[suit] - s2 < 0.05):
            return None
        return Card(suit, rank)

    def _ranked(self, card_img):
        """Все кандидаты (сумма, ранг-оценка, масть-оценка, Card) по эталонам игры, лучшие первыми."""
        if not getattr(self, "_g_rank", None):
            return []
        rank_p, pip_p, red = self._idx_patches(card_img)
        suits = ("hearts", "diamonds") if red else ("clubs", "spades")
        rs = {r: max(self._ncc(rank_p, t) for t in ts) for r, ts in self._g_rank.items()
              if self.allowed_ranks is None or r in self.allowed_ranks}
        ss = {sx: max(self._ncc(pip_p, t) for t in ts) for sx, ts in self._g_suit.items() if sx in suits}
        if not red and len(ss) == 2:
            f = self._spade_shape(card_img)
            if f is not None:
                ss = dict(ss); ss['spades'] += self.SHAPE_W / 2 * f; ss['clubs'] -= self.SHAPE_W / 2 * f
        out = [(rs[r] + ss[sx], rs[r], ss[sx], Card(sx, r)) for r in rs for sx in ss]
        out.sort(key=lambda t: t[0], reverse=True)
        return out


    # ---------- Стол: поиск индексов (ранг+масть) по всей картинке — работает для перекрытых карт ----------
    def _glyph_offset(self):
        """Где у карты 70x98 лежит левый верхний угол глyфа ранга (в пикселях карты) — из эталонов игры."""
        if getattr(self, "_goff", None):
            return self._goff
        xs, ys, hs = [], [], []
        for ts in self._g_rank.values():
            for t in ts[::2]:
                m = t > max(60, int(t.max() * 0.55))
                yy, xx = np.where(m)
                if len(xx):
                    xs.append((3 + xx.min()) / 2.0); ys.append(2 + (3 + yy.min()) / 2.0)
                    hs.append((yy.max() - yy.min() + 1) / 2.0)
        self._goff = (float(np.median(xs)), float(np.median(ys)), float(np.median(hs))) if xs else (3.0, 4.0, 10.0)
        return self._goff

    def _corner_scan(self, img, scale):
        """Ищет по всей картинке тёмные «глифы ранга» на белой бумаге, для каждого вырезает карту 70x98
        от предполагаемого левого верхнего угла и распознаёт тем же классификатором, что и руку.
        scale — во сколько раз карты на этом экране больше, чем 70 пикселей в ширину."""
        if not getattr(self, "_g_rank", None) or img is None or img.size == 0:
            return []
        f = 1.0 / max(0.3, scale)
        c = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
        hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        b, g_, r_ = [c[..., i].astype(int) for i in range(3)]
        ink = (c.min(axis=2) < 205).astype(np.uint8)       # тёмное или красное, включая розовые края красных глифов
        paper = ((hsv[..., 2] > 150) & (hsv[..., 1] < 70)).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
        gh = self._glyph_offset()[2]
        comps = []
        for i in range(1, n):
            x, y, w, h, area = st[i]
            if 2 <= w <= 13 and area >= 8:
                if 7 <= h <= 15:
                    comps.append([x, y, x + w, y + h])
                elif 15 < h <= 28:               # ранг слипся со значком масти (красные карты) — берём верхнюю часть
                    comps.append([x, y, x + w, y + int(round(gh))])
        comps.sort()
        merged = []                                   # «10» = два соседних глифа
        for cpt in comps:
            for m in merged:
                if abs(cpt[1] - m[1]) <= 3 and 0 <= cpt[0] - m[2] <= 4 and abs((cpt[3]-cpt[1]) - (m[3]-m[1])) <= 3:
                    m[2] = cpt[2]; m[1] = min(m[1], cpt[1]); m[3] = max(m[3], cpt[3]); break
            else:
                merged.append(list(cpt))
        ox, oy = self._glyph_offset()[:2]
        H_, W_ = c.shape[:2]
        pad = cv2.copyMakeBorder(c, 100, 100, 100, 100, cv2.BORDER_REPLICATE)
        out = []
        for x0, y0, x1, y1 in merged:
            ring = paper[max(0, y0-3):y1+3, max(0, x0-3):x1+3]
            if ring.size == 0 or ring.mean() < 0.45:   # глиф должен лежать на белой бумаге
                continue
            cx0, cy0 = int(round(x0 - ox)), int(round(y0 - oy))
            crop = pad[cy0 + 100: cy0 + 100 + 98, cx0 + 100: cx0 + 100 + 70]
            if crop.shape[0] < 98 or crop.shape[1] < 70:
                continue
            card = self._classify_game(crop)
            if card:
                out.append((card, (int((cx0 + 35) / f), int((cy0 + 49) / f)), crop))
        return out

    def _classify(self, card_img):
        parts = self._index_parts(card_img)
        if not parts:
            return None
        rk, sp, is_red = parts
        suits = ("hearts", "diamonds") if is_red else ("clubs", "spades")
        rscore = {r: max(self._iou(rk, t) for t in ts) for r, ts in self._rank_tpl.items()}
        sscore = {s: max(self._iou(sp, t) for t in self._suit_tpl[s]) for s in suits if s in self._suit_tpl}
        if not rscore or not sscore:
            return None
        rank = max(rscore, key=rscore.get); suit = max(sscore, key=sscore.get)
        if getattr(self, "_g_suit", None):      # масть надёжнее определять эталонами игры + формой значка
            _, pip_p, _red = self._idx_patches(card_img)
            g_ss = {sx: max(self._ncc(pip_p, t) for t in self._g_suit[sx]) for sx in suits if sx in self._g_suit}
            if len(g_ss) == 2:
                if is_red:
                    suit = max(g_ss, key=g_ss.get)
                else:
                    f = self._spade_shape(card_img)
                    suit = 'spades' if g_ss['spades'] - g_ss['clubs'] + (self.SHAPE_W * f if f is not None else 0.0) > 0 else 'clubs'
                sscore[suit] = max(sscore.get(suit, 0), 0.66)
        self.last_scores[f"{rank}{suit[0]}"] = (round(rscore[rank], 2), round(sscore[suit], 2))
        if rscore[rank] < 0.55 or sscore[suit] < 0.65:
            return None
        return Card(suit, rank)

    def detect_cards_in_image(self, img, vmin=185, scale=None, hand_scale=None):
        if img is None or img.size == 0:
            return []
        if not hasattr(self, "_rank_tpl"):
            self._build_index_templates()
            self._load_game_idx()
        self.last_scores = {}
        boxes, wide = self._find_card_boxes(img, vmin)
        self.last_boxes = list(boxes) + list(wide)
        if scale:   # стол: слипшиеся/обрезанные блоки не режем наугад — такие карты находит поиск по индексам
            boxes = [b_ for b_ in boxes if 0.8 * 70 * scale <= b_[2] <= 1.2 * 70 * scale
                     and 0.8 * 98 * scale <= b_[3] <= 1.25 * 98 * scale]
        found, pending = [], []
        for x, y, w, h in boxes:
            crop = img[y:y+h, x:x+w]
            pos = (x + w//2, y + h//2)
            card = self._classify_game(crop) or self._classify(crop)
            if not card:   # индекс закрыт кнопкой/рукой — читаем нижний правый угол (поворот 180°)
                rot = cv2.rotate(crop, cv2.ROTATE_180)
                card = self._classify_game(rot) or self._classify(rot)
                if card: crop = rot
            if card:
                found.append((card, pos, crop))
            else:
                pending.append((crop, pos))
        # одна и та же карта дважды невозможна: у слабее похожего — переоцениваем без неё
        for _ in range(3):
            by = {}
            for it in found: by.setdefault(it[0], []).append(it)
            dups = [v_ for v_ in by.values() if len(v_) > 1]
            if not dups: break
            for group in dups:
                def conf(it):
                    r_ = self._ranked(it[2])
                    return next((t[0] for t in r_ if t[3] == it[0]), 0.0)
                group.sort(key=conf, reverse=True)
                for it in group[1:]:
                    found.remove(it); pending.append((it[2], it[1]))
        for crop, pos in pending:   # второй проход для неуверенных карт
            card = self._classify_game(crop, relaxed=True, exclude={c for c, _, _ in found})
            if not card:
                rot = cv2.rotate(crop, cv2.ROTATE_180)
                card = self._classify_game(rot, relaxed=True, exclude={c for c, _, _ in found})
            if card:
                found.append((card, pos, crop))
        found = [(c, p) for c, p, _ in found]
        if scale:      # стол: добавляем карты, найденные по индексам (в том числе перекрытые)
            have = {c for c, _ in found}
            extra = sorted(self._corner_scan(img, scale), key=lambda t: -self._ranked(t[2])[0][0] if self._ranked(t[2]) else 0)
            for card, pos, _crop in extra:
                if card not in have:
                    found.append((card, pos)); have.add(card)
        if hand_scale and not scale:   # рука: догоняем пропущенные карты поиском по индексам на осветлённой копии
            have = {c for c, _ in found}
            try:
                bright = np.clip(img.astype(np.float32) * 1.7, 0, 255).astype(np.uint8)
                for card, pos, _crop in self._corner_scan(bright, hand_scale):
                    if card not in have:
                        found.append((card, pos)); have.add(card)
            except Exception:
                pass
        # перекрытые карты: запасной метод по углам-шаблонам
        for x, y, w, h in wide:
            for card, (cx, cy) in self._detect_corner_templates(img[y:y+h, x:x+w]):
                if card not in [c for c, _ in found]:
                    found.append((card, (cx + x, cy + y)))
        found.sort(key=lambda it: (it[1][0], it[1][1]))
        return found

    def save_debug(self, zone, path, dim=False):
        """Сохраняет картинку зоны с рамками и подписями — чтобы увидеть, что именно не распознано."""
        img = self.capture(zone)
        hits = self.detect_cards_in_image(img.copy(), vmin=105 if dim else 185)
        out = img.copy()
        for x, y, w, h in getattr(self, "last_boxes", []):
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 160, 255), 1)
        for c, (cx, cy) in hits:
            cv2.putText(out, str(c), (cx - 14, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        ok, buf = cv2.imencode(".png", out)
        if ok:
            buf.tofile(path)
        return [str(c) for c, _ in hits], len(getattr(self, "last_boxes", []))

    def close(self):
        try:
            self._sct.close()
        except Exception:
            pass

    @staticmethod
    def _name_sig(img):
        """Нормализованный «отпечаток» белой надписи (имя игрока): независим от размера и положения."""
        if img is None or img.size == 0:
            return None
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m = ((hsv[..., 2] > 170) & (hsv[..., 1] < 60)).astype(np.uint8)
        ys, xs = np.where(m)
        if len(xs) < 30:
            return None
        sub = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h, w = sub.shape
        if h < 5 or w < 15:
            return None
        return cv2.resize(sub.astype(np.float32), (int(round(w * 20.0 / h)), 20), interpolation=cv2.INTER_AREA)

    @classmethod
    def same_name(cls, a, b, thr=0.60):
        """Совпадают ли две надписи с именем. Устойчиво к разной толщине шрифта (тонкий/жирный):
        сильное размытие + небольшой сдвиг + проверка ширины."""
        sa, sb = cls._name_sig(a), cls._name_sig(b)
        if sa is None or sb is None:
            return None
        wa, wb = sa.shape[1], sb.shape[1]
        if abs(wa - wb) > 0.15 * max(wa, wb):
            return False
        W = 120
        f = lambda m: cv2.GaussianBlur(cv2.resize(m, (W, 20)), (0, 0), 3)
        A, B = f(sa), f(sb)
        u = A.ravel() - A.mean()
        best = -1.0
        for dx in range(-3, 4):
            Bs = np.roll(B, dx, axis=1)
            v = Bs.ravel() - Bs.mean()
            d = float(np.linalg.norm(u) * np.linalg.norm(v))
            if d > 0:
                best = max(best, float(u @ v) / d)
        return best >= thr

    def declarer_is_me(self):
        """True/False — выбрал козырь я или нет; None — имя не прочиталось."""
        sw, sh = self.screen_size()
        a = self.capture(config.scaled_zone(config.ZONE_DECLARER_NAME, sw, sh))
        # моё имя не меняется — отпечаток считаем один раз (экономит захват и обработку каждый кадр)
        if getattr(self, "_my_name_img", None) is None:
            b = self.capture(config.scaled_zone(config.ZONE_MY_NAME, sw, sh))
            if self._name_sig(b) is not None:
                self._my_name_img = b
        else:
            b = self._my_name_img
        return self.same_name(a, b)

    def read_trump_icon(self) -> Optional[str]:
        """Козырь по иконке в белом квадратике слева вверху. Пробует обе раскладки (обычную и с плашкой «14/A»)."""
        sw, sh = self.screen_size()
        for zn in (config.ZONE_TRUMP, getattr(config, "ZONE_TRUMP_ALT", config.ZONE_TRUMP)):
            zone = config.scaled_zone(zn, sw, sh)
            if zone[2] <= 4 or zone[3] <= 4:
                continue
            img = self.capture(zone)
            if img is None or img.size == 0:
                continue
            g = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
            if (g > 225).mean() < 0.35:      # нет белого квадратика — не та раскладка
                continue
            r = self.classify_trump_icon(img)
            if r:
                return r
        return None

    @staticmethod
    def classify_trump_icon(img) -> Optional[str]:
        """Масть козыря по иконке. Цвет -> красные/чёрные; форма (после увеличения до 96px) -> конкретная масть.
        Пики/трефы различаются голосованием 3 признаков (выпуклость контура, ширина на 45% и 15% высоты)."""
        if img is None or img.size == 0 or min(img.shape[:2]) < 6:
            return None
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img[:, :, :3], (96, 96), interpolation=cv2.INTER_CUBIC)
        m = 6
        inner = img[m:-m, m:-m]
        hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
        red = ((hsv[..., 0] < 12) | (hsv[..., 0] > 168)) & (hsv[..., 1] > 90) & (hsv[..., 2] > 90)
        dark = (inner.max(axis=2) < 120)
        is_red = red.sum() >= dark.sum()
        mask = (red if is_red else dark).astype(np.uint8)
        if mask.sum() < 0.05 * mask.size:
            return None
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            return None
        c = max(cs, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        if h < 12 or w < 12:
            return None
        sub = mask[y:y + h, x:x + w]
        width = lambda f: sub[min(h - 1, int(h * f))].sum() / w
        if is_red:
            return "hearts" if width(0.2) > 0.7 else "diamonds"
        hull = cv2.contourArea(cv2.convexHull(c))
        sol = cv2.contourArea(c) / hull if hull else 0
        votes = int(sol > 0.865) + int(width(0.45) > 0.76) + int(width(0.15) < 0.38)
        return "spades" if votes >= 2 else "clubs"

    def read_trump(self) -> Optional[str]:
        # When a dedicated on-screen trump marker zone hasn't been calibrated,
        # use the explicitly configured suit instead of repeatedly matching a 1px placeholder.
        configured = getattr(config, "TRUMP_SUIT", None)
        if configured in ("hearts", "diamonds", "clubs", "spades"):
            return configured
        zone = config.ZONE_TRUMP
        if zone[2] <= 1 or zone[3] <= 1:
            return None
        img = self.capture(zone)
        best_suit, best_val = None, 0
        for suit, tpl in self.trump_templates.items():
            for scale in (0.5, 0.75, 1.0):
                resized = cv2.resize(tpl, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                if resized.shape[0] > img.shape[0] or resized.shape[1] > img.shape[1]:
                    continue
                res = cv2.matchTemplate(img, resized, cv2.TM_CCOEFF_NORMED)
                _, mv, _, _ = cv2.minMaxLoc(res)
                if mv > best_val and mv >= config.TRUMP_MATCH_THRESHOLD:
                    best_val, best_suit = mv, suit
        return best_suit
