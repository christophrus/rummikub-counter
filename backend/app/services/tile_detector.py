"""
Steinerkennung und -segmentierung mit OpenCV.

Findet einzelne Rummikub-Steine im Bild durch mehrere Strategien:
1. Sättigung (weiße Steine haben niedrige Sättigung vs. Holz)
2. Helligkeit (Otsu Threshold)
3. Canny Edge Detection (wenn Steine getrennt liegen)
4. Adaptive Thresholding (Fallback)
Dann zu breite Regionen (zusammenhängende Steine) in Einzelsteine aufteilen.
"""

import cv2
import numpy as np


def detect_tiles(image: np.ndarray) -> list[dict]:
    """
    Erkennt Rummikub-Steine im Bild.

    Probiert mehrere Strategien und nimmt die beste:
    1. Sättigungs-basiert (robust bei hellem UND dunklem Hintergrund)
    2. Helligkeits-basiert (Otsu)
    3. Multi-Threshold (probiert mehrere Helligkeitsschwellen)
    4. Kanten-basiert (Canny, gut bei getrennten Steinen)
    5. Adaptives Thresholding (Fallback)

    Returns:
        Liste von Dicts mit {x, y, w, h, contour} für jeden erkannten Stein.
    """
    img_area = image.shape[0] * image.shape[1]

    # Alle Strategien ausprobieren
    candidates = []

    for strategy_fn in [_detect_by_table_diff, _detect_by_saturation,
                        _detect_by_brightness, _detect_by_multi_threshold,
                        _detect_by_local_otsu,
                        _detect_by_canny, _detect_by_adaptive_threshold]:
        tiles = strategy_fn(image, img_area)
        # Splitting anwenden
        tiles = _split_merged_regions(tiles, image, img_area)
        tiles = _non_max_suppression(tiles, overlap_thresh=0.3)
        candidates.append(tiles)

    # Beste Strategie wählen: die mit den meisten realistischen Steinen
    best_tiles = max(candidates, key=lambda t: _score_tile_set(t, img_area))
    best_score = _score_tile_set(best_tiles, img_area)

    # Tiles aus anderen guten Strategien dazumergen (nur wenn ähnlich gut)
    for other in candidates:
        if other is not best_tiles:
            other_score = _score_tile_set(other, img_area)
            if other_score > best_score * 0.3 and other_score > 5:
                best_tiles = _merge_tile_lists(best_tiles, other)

    best_tiles = _non_max_suppression(best_tiles, overlap_thresh=0.3)

    # Steine mit unrealistischem Seitenverhältnis oder unrealistischer Größe entfernen
    est_w, est_h = _estimate_single_tile_size(best_tiles, img_area, image)
    min_tile_area = est_w * est_h * 0.25
    max_tile_area = est_w * est_h * 3.0
    img_h, img_w = image.shape[:2]
    best_tiles = [t for t in best_tiles
                  if min_tile_area <= t["w"] * t["h"] <= max_tile_area
                  and 0.25 < (t["w"] / t["h"] if t["h"] > 0 else 0) < 1.5
                  and t["y"] > est_h * 0.05
                  and t["y"] + t["h"] < img_h - est_h * 0.05]

    # Vertikales Splitting: Zu hohe Steine (multi-Reihen) aufteilen.
    # Nur auf Steine nahe der geschätzten Einzelsteinbreite anwenden.
    if best_tiles:
        est_w2, est_h2 = _estimate_single_tile_size(best_tiles, img_area, image)
        vert_result = []
        for t in best_tiles:
            if (t["h"] > est_h2 * 1.8
                    and t["w"] < est_w2 * 1.8
                    and t["w"] > est_w2 * 0.4):
                splits = _split_tall_region(t, image, est_w2, est_h2)
                if splits:
                    vert_result.extend(splits)
                    continue
            vert_result.append(t)
        if len(vert_result) > len(best_tiles):
            best_tiles = _non_max_suppression(vert_result, overlap_thresh=0.3)

    # Nach Position sortieren (oben-links nach unten-rechts)
    best_tiles.sort(key=lambda t: (t["y"] // 50, t["x"]))

    return best_tiles


def _score_tile_set(tiles: list[dict], img_area: float) -> float:
    """
    Bewertet eine Menge von Tile-Erkennungen.
    Belohnt:
    - Realistische Anzahl (10-70 Steine)
    - Konsistente Steingröße
    - Realistisches Seitenverhältnis
    Bestraft:
    - Zu viele Steine (>80 = Rauschen)
    - Inkonsistente Größen
    """
    if not tiles:
        return 0

    good_tiles = []
    for t in tiles:
        aspect = t["w"] / t["h"] if t["h"] > 0 else 0
        if 0.25 < aspect < 1.5:
            good_tiles.append(t)

    if not good_tiles:
        return 0

    # Basis-Score: Anzahl guter Steine
    n = len(good_tiles)
    score = n

    # Bonus für perfektes Seitenverhältnis
    for t in good_tiles:
        aspect = t["w"] / t["h"]
        if 0.5 < aspect < 0.85:
            score += 0.5

    # Strafe für zu viele Steine (wahrscheinlich Rauschen)
    if n > 100:
        score *= 0.05
    elif n > 70:
        score *= 0.5

    # Belohnung für konsistente Steingröße (niedrige Varianz)
    if len(good_tiles) >= 5:
        areas = [t["w"] * t["h"] for t in good_tiles]
        mean_area = np.mean(areas)
        std_area = np.std(areas)
        cv = std_area / mean_area if mean_area > 0 else 1.0
        # cv < 0.3 = sehr konsistent, cv > 0.8 = sehr inkonsistent
        if cv < 0.4:
            score *= 1.3
        elif cv > 0.8:
            score *= 0.5

    return score


def _estimate_single_tile_size(tiles: list[dict], img_area: float,
                                image: np.ndarray | None = None) -> tuple[int, int]:
    """
    Schätzt die Größe eines einzelnen Steins anhand der bereits erkannten Steine.
    Returns (estimated_width, estimated_height).
    """
    img_side = int(np.sqrt(img_area))
    max_single = img_side // 5
    min_single = img_side // 25  # Minimale Steingröße (~60px bei 1920er Bild)

    # Sammle Steine die einzeln aussehen (Aspektverhältnis ~0.5-0.9)
    singles = [t for t in tiles
                if 0.4 < (t["w"] / t["h"] if t["h"] > 0 else 0) < 1.0
                and t["w"] < max_single and t["h"] < max_single
                and t["w"] > min_single and t["h"] > min_single]

    if len(singles) >= 3:
        med_w = int(np.median([t["w"] for t in singles]))
        med_h = int(np.median([t["h"] for t in singles]))
        return med_w, med_h

    # Wenn keine Einzelsteine, nutze die Höhe der breitesten Regionen
    if tiles:
        heights = [t["h"] for t in tiles]
        med_h = int(np.median(heights))
        est_w = int(med_h * 0.67)

        # Plausibilitätsprüfung: wenn est_w > 200px bei 1920er Bild,
        # ist die Höhe wahrscheinlich von einem Multi-Reihen-Blob.
        # Versuche Kantenabstand-Analyse als Alternative.
        if image is not None and est_w > 200:
            largest = max(tiles, key=lambda t: t["w"] * t["h"])
            if largest["w"] > max_single:
                edge_est = _estimate_tile_width_from_edges(image, largest)
                if edge_est is not None:
                    return edge_est

        return est_w, med_h

    # Letzter Fallback: Schätze aus Bildfläche
    est_area = img_area * 0.003
    est_h = int(np.sqrt(est_area * 1.5))
    est_w = int(est_h * 0.67)
    return est_w, est_h


def _estimate_tile_width_from_edges(image: np.ndarray,
                                     blob: dict) -> tuple[int, int] | None:
    """
    Schätzt die Steinbreite aus dem Kantenabstand innerhalb eines großen Blobs.
    Nutzt Autokorrelation des binären Sättigungsprofils pro Zeile.
    """
    x, y, w, h = blob["x"], blob["y"], blob["w"], blob["h"]
    roi = image[y:y+h, x:x+w]
    if roi.size == 0 or w < 200:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask = ((s < 70) & (v > 120)).astype(np.uint8) * 255

    # Blob in horizontale Bänder teilen und pro Band Autokorrelation berechnen
    n_bands = max(2, h // 80)
    band_h = h // n_bands
    min_lag = max(80, w // 15)
    max_lag = w // 2

    best_lag = None
    best_corr = 0.0

    for b in range(n_bands):
        by = b * band_h
        bey = min(by + band_h, h)
        band = mask[by:bey, :]
        if band.shape[0] < 20:
            continue

        profile = np.sum(band > 0, axis=0).astype(np.float64)
        if np.std(profile) < 1.0:
            continue

        profile -= np.mean(profile)
        norm = np.dot(profile, profile)
        if norm < 1.0:
            continue

        # Autokorrelation berechnen
        corrs = np.zeros(max_lag - min_lag)
        for lag in range(min_lag, max_lag):
            corrs[lag - min_lag] = np.dot(profile[:-lag], profile[lag:]) / norm

        # Nur lokale Maxima als echte Periodizitäten akzeptieren
        for i in range(1, len(corrs) - 1):
            lag = i + min_lag
            c = corrs[i]
            if c > corrs[i - 1] and c > corrs[i + 1] and c > best_corr:
                best_corr = c
                best_lag = lag

    if best_lag is None or best_corr < 0.1:
        return None

    est_w = best_lag
    est_h = int(est_w * 1.4)
    return est_w, est_h


def _split_merged_regions(tiles: list[dict], image: np.ndarray,
                           img_area: float) -> list[dict]:
    """
    Erkennt zu breite Regionen (mehrere zusammenhängende Steine)
    und teilt sie rekursiv in Einzelsteine auf.
    """
    est_w, est_h = _estimate_single_tile_size(tiles, img_area, image)

    # Mindestbreite für "sieht nach mehreren Steinen aus"
    merge_threshold = est_w * 1.5

    result = []
    for tile in tiles:
        split_tiles = _recursive_split(tile, image, est_w, est_h, merge_threshold, img_area)
        result.extend(split_tiles)

    return result


def _recursive_split(tile: dict, image: np.ndarray, est_w: int, est_h: int,
                      merge_threshold: float, img_area: float,
                      depth: int = 0) -> list[dict]:
    """Teilt einen Tile rekursiv auf bis alle unter dem Threshold sind."""
    if depth > 5 or tile["w"] <= merge_threshold:
        return [tile]

    sub_tiles = _split_wide_region(tile, image, est_w, est_h, img_area)
    if not sub_tiles:
        return [tile]

    # Rekursiv weiter aufteilen falls nötig
    result = []
    for sub in sub_tiles:
        result.extend(_recursive_split(sub, image, est_w, est_h,
                                        merge_threshold, img_area, depth + 1))
    return result


def _split_wide_region(region: dict, image: np.ndarray,
                        est_w: int, est_h: int, img_area: float) -> list[dict]:
    """
    Teilt eine breite Region in einzelne Steine auf.
    Nutzt vertikale Kantenerkennung (Sobel-X) um Grenzen zwischen Steinen zu finden.
    Fällt auf gleichmäßige Aufteilung zurück.
    """
    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
    roi = image[y:y+h, x:x+w]

    if roi.size == 0:
        return []

    n_tiles = max(2, round(w / est_w))

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Vertikale Kanten erkennen (Sobel-X): Grenzen zwischen Steinen
    sobel_x = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
    sobel_abs = np.abs(sobel_x)

    # Vertikales Kantenprofil: Summe der vertikalen Kanten pro Spalte
    # Mittlerer Bereich (um Zahlen oben/unten auszuschließen)
    mid_start = int(h * 0.15)
    mid_end = int(h * 0.85)
    edge_profile = np.sum(sobel_abs[mid_start:mid_end, :], axis=0)

    # Glätten
    kernel_size = max(3, est_w // 10)
    if kernel_size % 2 == 0:
        kernel_size += 1
    smoothed = cv2.GaussianBlur(edge_profile.reshape(1, -1).astype(np.float32),
                                 (kernel_size, 1), 0).flatten()

    # Lokale Maxima im Kantenprofil = Trennstellen
    split_positions = _find_edge_peaks(smoothed, n_tiles, est_w)

    if not split_positions:
        # Fallback: Gleichmäßig aufteilen
        tile_w = w / n_tiles
        split_positions = [int(tile_w * i) for i in range(1, n_tiles)]

    # Sub-tiles erstellen
    sub_tiles = []
    boundaries = [0] + split_positions + [w]

    for i in range(len(boundaries) - 1):
        sx = boundaries[i]
        ex = boundaries[i + 1]
        tw = ex - sx

        if tw < est_w * 0.4:
            continue

        sub_tiles.append({
            "x": x + sx,
            "y": y,
            "w": tw,
            "h": h,
            "area": tw * h,
            "contour": None,
        })

    return sub_tiles if len(sub_tiles) >= 2 else []


def _find_edge_peaks(profile: np.ndarray, n_expected: int,
                      est_w: int) -> list[int]:
    """
    Findet Trennstellen anhand von Peaks im Kantenprofil.
    Peaks = starke vertikale Kanten = Grenzen zwischen Steinen.
    """
    length = len(profile)
    if length < est_w * 2:
        return []

    min_distance = int(est_w * 0.5)
    positions = []

    for i in range(1, n_expected):
        center = int(length * i / n_expected)
        search_start = max(0, center - int(est_w * 0.4))
        search_end = min(length, center + int(est_w * 0.4))

        if search_start >= search_end:
            continue

        segment = profile[search_start:search_end]
        # Peak = stärkste vertikale Kante im Suchbereich
        local_max_idx = search_start + int(np.argmax(segment))

        if not positions or (local_max_idx - positions[-1]) > min_distance:
            positions.append(local_max_idx)

    return positions


def _split_tall_region(region: dict, image: np.ndarray,
                        est_w: int, est_h: int) -> list[dict]:
    """
    Teilt eine vertikal zu hohe Region (mehrere Reihen) in einzelne Reihen auf.
    Nutzt horizontale Kantenerkennung (Sobel-Y) um Grenzen zwischen Reihen zu finden.
    Prüft vorher, ob eine echte horizontale Lücke (dunkler Streifen) zwischen Reihen existiert.
    """
    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
    roi = image[y:y+h, x:x+w]

    if roi.size == 0:
        return []

    # Prüfe ob eine echte horizontale Lücke existiert (z.B. Tisch/Halter zwischen Reihen)
    # Eine echte Lücke = Helligkeit sinkt ab (vom Stein) und steigt wieder an (nächster Stein).
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    row_brightness = np.mean(gray_roi[:, int(w*0.1):int(w*0.9)], axis=1)

    # Glätten
    kernel_size = max(5, est_h // 8)
    if kernel_size % 2 == 0:
        kernel_size += 1
    smoothed_brightness = cv2.GaussianBlur(row_brightness.reshape(-1, 1).astype(np.float32),
                                            (1, kernel_size), 0).flatten()

    # Suche nach einem echten Tal (bright → dark → bright) im mittleren Bereich
    mid_start = int(h * 0.15)
    mid_end = int(h * 0.85)
    mid_profile = smoothed_brightness[mid_start:mid_end]

    if len(mid_profile) < 10:
        return []

    min_pos_mid = int(np.argmin(mid_profile))
    min_val = float(mid_profile[min_pos_mid])

    # Helligkeit oberhalb und unterhalb des Minimums muss deutlich höher sein
    above = mid_profile[:min_pos_mid]
    below = mid_profile[min_pos_mid:]
    if len(above) < 5 or len(below) < 5:
        return []

    max_above = float(np.max(above))
    max_below = float(np.max(below))

    # Beide Seiten müssen deutlich heller sein als das Minimum
    if min_val > max_above * 0.65 or min_val > max_below * 0.65:
        return []

    n_rows = max(2, round(h / est_h))

    # Horizontale Kanten erkennen (Sobel-Y): Grenzen zwischen Reihen
    sobel_y = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
    sobel_abs = np.abs(sobel_y)

    # Horizontales Kantenprofil: Summe der horizontalen Kanten pro Zeile
    mid_start = int(w * 0.1)
    mid_end = int(w * 0.9)
    edge_profile = np.sum(sobel_abs[:, mid_start:mid_end], axis=1)

    # Glätten
    kernel_size2 = max(3, est_h // 10)
    if kernel_size2 % 2 == 0:
        kernel_size2 += 1
    smoothed = cv2.GaussianBlur(edge_profile.reshape(-1, 1).astype(np.float32),
                                 (1, kernel_size2), 0).flatten()

    # Peaks finden
    split_positions = _find_edge_peaks(smoothed, n_rows, est_h)

    if not split_positions:
        # Fallback: Gleichmäßig aufteilen
        row_h = h / n_rows
        split_positions = [int(row_h * i) for i in range(1, n_rows)]

    # Sub-tiles erstellen
    sub_tiles = []
    boundaries = [0] + split_positions + [h]

    for i in range(len(boundaries) - 1):
        sy = boundaries[i]
        ey = boundaries[i + 1]
        th = ey - sy

        if th < est_h * 0.4:
            continue

        sub_tiles.append({
            "x": x,
            "y": y + sy,
            "w": w,
            "h": th,
            "area": w * th,
            "contour": None,
        })

    return sub_tiles if len(sub_tiles) >= 2 else []


def _detect_by_table_diff(image: np.ndarray, img_area: float) -> list[dict]:
    """
    Erkennt Steine durch Vergleich mit der dominanten Tischfarbe.
    Der Tisch ist die häufigste Farbe im Bild. Alles was deutlich abweicht = Stein.
    Robust bei jeder Beleuchtung.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    # Tischfarbe schätzen: Median jedes Kanals (Tisch = größter Bereich)
    table_l = np.median(l_ch)
    table_a = np.median(a_ch)
    table_b = np.median(b_ch)

    # Farbabstand zum Tisch berechnen (euklidisch in LAB)
    diff = np.sqrt(
        (l_ch.astype(float) - table_l) ** 2 +
        (a_ch.astype(float) - table_a) ** 2 +
        (b_ch.astype(float) - table_b) ** 2
    ).astype(np.uint8)

    # Mehrere Schwellen probieren
    best_tiles = []
    best_score = 0

    for thresh_val in range(15, 55, 5):
        _, mask = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)

        # Morphologie: Löcher schließen und Rauschen entfernen
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=2)

        tiles = _extract_tiles_from_mask(mask, img_area)
        score = _score_tile_set(tiles, img_area)

        if score > best_score:
            best_score = score
            best_tiles = tiles

    return best_tiles


def _detect_by_saturation(image: np.ndarray, img_area: float) -> list[dict]:
    """
    Erkennt weiße Steine über niedrige Sättigung im HSV-Raum.
    Weisse Steine haben niedrige Sättigung, Holz hat höhere Sättigung.
    Funktioniert bei hellen UND dunklen Hintergründen.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Weiße Steine: niedrige Sättigung UND hohe Helligkeit
    mask_low_sat = cv2.inRange(s, 0, 70)
    mask_bright = cv2.inRange(v, 120, 255)
    mask = cv2.bitwise_and(mask_low_sat, mask_bright)

    # Morphologie: Löcher horizontal schließen (farbige Zahlen),
    # aber NICHT vertikal um Reihen nicht zu verbinden
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_h, iterations=3)

    # Vertikal nur leicht schließen
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_v, iterations=2)

    # Quadratisches Close für verbleibende Löcher
    kernel_sq = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_sq, iterations=2)

    # Kleine Fragmente entfernen
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=2)

    return _extract_tiles_from_mask(mask, img_area)


def _detect_by_brightness(image: np.ndarray, img_area: float) -> list[dict]:
    """
    Erkennt helle Steine gegen dunklen Hintergrund mit Otsu-Thresholding.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    return _extract_tiles_from_mask(mask, img_area)


def _detect_by_multi_threshold(image: np.ndarray, img_area: float) -> list[dict]:
    """
    Probiert mehrere Helligkeitsschwellen (V-Kanal) und nimmt die beste.
    Robust bei unterschiedlichen Tisch-Helligkeiten.
    Nutzt horizontale Morphologie um Lücken durch farbige Zahlen zu schließen.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    blurred = cv2.GaussianBlur(v_channel, (7, 7), 0)

    best_tiles = []
    best_score = 0

    # Verschiedene Schwellen probieren
    for thresh_val in range(100, 210, 5):
        _, mask = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)

        # Horizontal schließen (farbige Zahlen überbrücken)
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_h, iterations=3)

        # Vertikal schließen (Zahlen-Lücken innerhalb eines Steins füllen)
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_v, iterations=2)

        # Finales Close
        kernel_sq = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_sq, iterations=2)

        # Rauschen entfernen
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=2)

        tiles = _extract_tiles_from_mask(mask, img_area)
        score = _score_tile_set(tiles, img_area)

        if score > best_score:
            best_score = score
            best_tiles = tiles

    return best_tiles


def _detect_by_local_otsu(image: np.ndarray, img_area: float) -> list[dict]:
    """
    Lokale Otsu-Schwellwerte auf horizontalen Streifen.
    Löst das Problem bei ungleichmäßiger Beleuchtung (oben dunkel, Mitte hell).
    Nutzt V-Kanal (Helligkeit) und ignoriert Bildränder (nur Tisch).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_ch = hsv[:, :, 2]
    blurred = cv2.GaussianBlur(v_ch, (7, 7), 0)
    h, w = blurred.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Bild in überlappende Streifen aufteilen
    n_strips = 8
    strip_h = h // n_strips
    overlap = strip_h // 4

    for i in range(n_strips):
        y_start = max(0, i * strip_h - overlap)
        y_end = min(h, (i + 1) * strip_h + overlap)
        strip = blurred[y_start:y_end, :]

        # Otsu nur sinnvoll wenn genug Kontrast im Streifen
        if strip.std() < 15:
            continue

        otsu_thresh, strip_mask = cv2.threshold(strip, 0, 255,
                                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Prüfen ob der Schwellwert sinnvoll ist
        # (Vordergrund sollte 5-80% des Streifens ausmachen)
        fg_ratio = np.sum(strip_mask > 0) / strip_mask.size
        if fg_ratio < 0.05 or fg_ratio > 0.80:
            continue

        # In die Gesamtmaske einfügen (Core-Bereich, ohne Overlap)
        core_start = i * strip_h
        core_end = min(h, (i + 1) * strip_h)
        strip_core_start = core_start - y_start
        strip_core_end = core_end - y_start
        mask[core_start:core_end, :] = strip_mask[strip_core_start:strip_core_end, :]

    # Horizontales Close (farbige Zahlen überbrücken)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_h, iterations=3)

    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_v, iterations=2)

    kernel_sq = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_sq, iterations=2)

    # Rauschen entfernen
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=2)

    return _extract_tiles_from_mask(mask, img_area)


def _detect_by_canny(image: np.ndarray, img_area: float) -> list[dict]:
    """
    Erkennt Steine durch Canny-Kantenerkennung.
    Funktioniert gut wenn Steine einzeln/getrennt liegen.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 30, 100)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tiles = []
    min_area = img_area * 0.0005
    max_area = img_area * 0.5  # Groß, weil ganze Reihen erst später gesplittet werden

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / h if h > 0 else 0
        if aspect < 0.1 or aspect > 10:
            continue

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < 0.6:
            continue

        tiles.append({
            "x": x, "y": y, "w": w, "h": h,
            "area": area, "contour": contour,
        })

    return tiles


def _detect_by_adaptive_threshold(image: np.ndarray, img_area: float) -> list[dict]:
    """
    Erkennt Steine durch adaptives Thresholding.
    Funktioniert gut bei ungleichmäßiger Beleuchtung.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Adaptive Threshold (Steine sind heller als ihre Umgebung)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 51, -10
    )

    # Morphologie
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    thresh = cv2.erode(thresh, kernel_erode, iterations=3)

    return _extract_tiles_from_mask(thresh, img_area)


def _extract_tiles_from_mask(mask: np.ndarray, img_area: float) -> list[dict]:
    """Extrahiert Tile-Kandidaten aus einer Binärmaske."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tiles = []
    min_area = img_area * 0.0005
    max_area = img_area * 0.5  # Groß, weil ganze Reihen erst später gesplittet werden

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Sehr extreme Seitenverhältnisse ausschließen (> 10:1)
        aspect = w / h if h > 0 else 0
        if aspect < 0.1 or aspect > 10:
            continue

        tiles.append({
            "x": x, "y": y, "w": w, "h": h,
            "area": area, "contour": contour,
        })

    return tiles


def _merge_tile_lists(list1: list[dict], list2: list[dict]) -> list[dict]:
    """Merged zwei Tile-Listen, entfernt Duplikate über IoU."""
    merged = list(list1)
    for tile in list2:
        is_dup = False
        for existing in merged:
            if _compute_iou(tile, existing) > 0.3:
                is_dup = True
                break
        if not is_dup:
            merged.append(tile)
    return merged


def _non_max_suppression(tiles: list[dict], overlap_thresh: float = 0.3) -> list[dict]:
    """
    Entfernt überlappende Detektionen (Non-Maximum Suppression).
    Behält die größere Detektion bei Überlappung.
    """
    if not tiles:
        return []

    # Nach Fläche sortieren (größte zuerst)
    tiles_sorted = sorted(tiles, key=lambda t: t["area"], reverse=True)
    keep = []

    for tile in tiles_sorted:
        is_overlap = False
        for kept in keep:
            iou = _compute_iou(tile, kept)
            if iou > overlap_thresh:
                is_overlap = True
                break
        if not is_overlap:
            keep.append(tile)

    return keep


def _compute_iou(box1: dict, box2: dict) -> float:
    """Berechnet Intersection over Union (IoU) zweier Boxen."""
    x1 = max(box1["x"], box2["x"])
    y1 = max(box1["y"], box2["y"])
    x2 = min(box1["x"] + box1["w"], box2["x"] + box2["w"])
    y2 = min(box1["y"] + box1["h"], box2["y"] + box2["h"])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box1["w"] * box1["h"]
    area2 = box2["w"] * box2["h"]
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def draw_detections(image: np.ndarray, tiles: list[dict], results: list[dict] | None = None) -> np.ndarray:
    """
    Zeichnet erkannte Steine und deren Werte ins Bild (Debug-Visualisierung).
    """
    output = image.copy()

    for i, tile in enumerate(tiles):
        color = (0, 255, 0)  # Grün
        label = f"#{i + 1}"

        if results and i < len(results):
            result = results[i]
            number = result.get("number")
            is_joker = result.get("is_joker", False)
            if is_joker:
                label = "J"
                color = (0, 165, 255)  # Orange für Joker
            elif number is not None:
                label = f"{number}"

        cv2.rectangle(output, (tile["x"], tile["y"]),
                      (tile["x"] + tile["w"], tile["y"] + tile["h"]),
                      color, 2)
        cv2.putText(output, label, (tile["x"], tile["y"] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return output
