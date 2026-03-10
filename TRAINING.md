# Rummikub KI-Modell Training – Schritt-für-Schritt Anleitung

## Übersicht

Wir trainieren ein eigenes Deep-Learning-Modell zur Erkennung von Rummikub-Steinen in zwei Phasen:

- **Phase 1: Custom CNN Classifier** – Ein eigenes neuronales Netz, das einzelne Stein-Bilder klassifiziert (Zahlen 1-13 + Joker = 14 Klassen)
- **Phase 2: YOLOv8 Object Detection** – Ein Modell, das Steine im Gesamtbild findet UND klassifiziert (ersetzt OpenCV-Konturfindung + CNN)

---

## Phase 1: Custom CNN Classifier

### Schritt 1: Dataset erstellen

> **Ziel:** Mindestens 20-30 Bilder pro Klasse (14 Klassen = ~300-400 Bilder minimum)

- [ ] **1.1** Rummikub-Steine bereitlegen
- [ ] **1.2** Fotografie-Setup einrichten:
  - Einfarbiger Hintergrund (am besten dunkel)
  - Gleichmäßige Beleuchtung (kein direktes Sonnenlicht)
  - Kamera direkt von oben (Draufsicht)
- [ ] **1.3** Fotos machen – pro Stein mehrere Varianten:
  - Verschiedene Beleuchtungen (heller, dunkler)
  - Leicht unterschiedliche Winkel
  - Verschiedene Abstände/Zoomstufen
- [ ] **1.4** Ordnerstruktur anlegen:
  ```
  backend/dataset/
  ├── raw/                    ← Originalbilder (Gesamtaufnahmen)
  ├── tiles/                  ← Ausgeschnittene Einzelsteine
  │   ├── 1/                  ← Alle Steine mit Zahl 1
  │   ├── 2/
  │   ├── ...
  │   ├── 13/
  │   └── joker/
  ├── train/                  ← Trainingsdaten (80%)
  ├── val/                    ← Validierungsdaten (10%)
  └── test/                   ← Testdaten (10%)
  ```
- [ ] **1.5** Einzelsteine ausschneiden – entweder:
  - **Option A:** Unser Tile-Detector + manuelles Labeln (Tool wird bereitgestellt)
  - **Option B:** Manuell mit einem Bildbearbeitungsprogramm zuschneiden
- [ ] **1.6** Bilder in die richtigen Ordner sortieren (nach Zahl 1-13 + joker)
- [ ] **1.7** Dataset auf Vollständigkeit prüfen:
  - Mindestens 20 Bilder pro Klasse?
  - Alle 14 Klassen vorhanden?
  - Keine falsch zugeordneten Bilder?

### Schritt 2: Data Augmentation

> **Ziel:** Aus wenigen echten Fotos viele Trainingsbilder erzeugen

- [ ] **2.1** Augmentations-Script ausführen (wird bereitgestellt)
  - Rotation (±15°)
  - Helligkeitsänderung (±30%)
  - Spiegelung (horizontal)
  - Leichter Blur
  - Zufälliger Zuschnitt
  - Farbverschiebung
- [ ] **2.2** Prüfen ob augmentierte Bilder realistisch aussehen
- [ ] **2.3** Dataset aufteilen: 80% Train / 10% Validation / 10% Test

### Schritt 3: CNN-Modell definieren

> **Ziel:** Netzwerk-Architektur verstehen und festlegen

- [ ] **3.1** Modell-Architektur studieren:
  ```
  Input (64x64 RGB)
    → Conv2D(32) → BatchNorm → ReLU → MaxPool
    → Conv2D(64) → BatchNorm → ReLU → MaxPool
    → Conv2D(128) → BatchNorm → ReLU → MaxPool
    → Flatten
    → Linear(256) → ReLU → Dropout(0.5)
    → Linear(14)  → Softmax
  ```
  **Lernpunkte:**
  - Conv2D: Filter erkennen Kanten, Formen, Muster
  - BatchNorm: Stabilisiert das Training
  - MaxPool: Reduziert Bildgröße, behält wichtige Features
  - Dropout: Verhindert Overfitting
  - Softmax: Wahrscheinlichkeitsverteilung über 14 Klassen

- [ ] **3.2** PyTorch-Modellklasse implementieren (wird bereitgestellt)
- [ ] **3.3** Modell-Zusammenfassung anschauen (Parameter-Anzahl, Layer)

### Schritt 4: Training

> **Ziel:** Modell trainieren und Ergebnisse beobachten

- [ ] **4.1** Hyperparameter festlegen:
  - Lernrate: `0.001`
  - Batch-Größe: `32`
  - Epochen: `50` (mit Early Stopping)
  - Optimizer: `Adam`
  - Loss: `CrossEntropyLoss`
- [ ] **4.2** Training starten
- [ ] **4.3** Trainingsverlauf beobachten:
  - Loss sinkt? ✓
  - Accuracy steigt? ✓
  - Validation-Loss steigt nicht? (sonst = Overfitting)
- [ ] **4.4** Trainiertes Modell speichern (`model.pth`)
- [ ] **4.5** Lernkurven plotten (Loss + Accuracy über Epochen)

### Schritt 5: Evaluation

> **Ziel:** Wie gut ist das Modell wirklich?

- [ ] **5.1** Testdaten durch das Modell laufen lassen
- [ ] **5.2** Metriken berechnen:
  - Gesamt-Accuracy (Ziel: > 90%)
  - Confusion Matrix (welche Zahlen werden verwechselt?)
  - Pro-Klasse Accuracy
- [ ] **5.3** Fehler analysieren:
  - Welche Steine werden falsch erkannt?
  - Gibt es Muster? (z.B. 6↔9, 1↔7)
- [ ] **5.4** Falls Accuracy < 90%:
  - Mehr Trainingsdaten sammeln
  - Augmentation anpassen
  - Modell-Architektur vergrößern
  - Lernrate reduzieren

### Schritt 6: Integration in die App

> **Ziel:** Trainiertes CNN ersetzt EasyOCR

- [ ] **6.1** Modell-Datei (`model.pth`) in Backend einbinden
- [ ] **6.2** Neuen Service `cnn_classifier.py` erstellen
- [ ] **6.3** Router `analyze.py` umstellen: CNN statt EasyOCR
- [ ] **6.4** Testen: App mit eigenem Modell testen
- [ ] **6.5** Vergleich: Eigenes CNN vs. EasyOCR – was ist besser?

---

## Phase 2: YOLOv8 Object Detection

> Erst starten, wenn Phase 1 abgeschlossen ist!

### Schritt 7: YOLO-Dataset erstellen

> **Ziel:** Bilder mit annotierten Bounding Boxes

- [ ] **7.1** Annotierungstool installieren (z.B. [Label Studio](https://labelstud.io/) oder [CVAT](https://www.cvat.ai/))
- [ ] **7.2** Gesamtbilder (mit mehreren Steinen) annotieren:
  - Bounding Box um jeden Stein zeichnen
  - Klasse zuweisen (1-13, joker)
  - Pro Bild: alle Steine markieren
- [ ] **7.3** Mindestens 100-200 annotierte Gesamtbilder
- [ ] **7.4** Annotationen im YOLO-Format exportieren:
  ```
  # label.txt pro Bild: Klasse x_center y_center width height (normalisiert)
  0 0.45 0.32 0.08 0.12
  7 0.61 0.55 0.09 0.13
  ```
- [ ] **7.5** Dataset-Struktur anlegen:
  ```
  backend/yolo_dataset/
  ├── data.yaml             ← Dataset-Konfiguration
  ├── train/
  │   ├── images/
  │   └── labels/
  ├── val/
  │   ├── images/
  │   └── labels/
  └── test/
      ├── images/
      └── labels/
  ```

### Schritt 8: YOLOv8 trainieren

> **Ziel:** YOLO erkennt Steine und ihre Zahlen in einem Schritt

- [ ] **8.1** Ultralytics installieren (`pip install ultralytics`)
- [ ] **8.2** `data.yaml` konfigurieren
- [ ] **8.3** Training starten:
  ```bash
  yolo detect train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640
  ```
  - `yolov8n.pt` = vortrainiertes Nanomodell (Transfer Learning!)
  - Das Model lernt von seinen COCO-Weights und passt sich an Rummikub an
- [ ] **8.4** Trainingsergebnisse auswerten (mAP, Precision, Recall)
- [ ] **8.5** Bestes Modell speichern (`best.pt`)

### Schritt 9: YOLO in die App integrieren

> **Ziel:** YOLO ersetzt OpenCV-Segmentierung + CNN

- [ ] **9.1** Neuen Service `yolo_detector.py` erstellen
- [ ] **9.2** YOLO erkennt in einem Forward Pass:
  - Wo sind die Steine? (Bounding Boxes)
  - Welche Zahl hat jeder Stein? (Klassifikation)
- [ ] **9.3** Router anpassen: YOLO statt OpenCV + CNN/EasyOCR
- [ ] **9.4** End-to-End Test mit echten Bildern
- [ ] **9.5** Performance-Vergleich: Geschwindigkeit + Genauigkeit

---

## Tipps für gute Trainingsergebnisse

### Beim Fotografieren
- **Konsistenz:** Gleicher Abstand, gleiche Beleuchtung für alle Steine
- **Vielfalt:** Verschiedene Hintergründe, leichte Winkeländerungen
- **Qualität:** Scharfe Bilder, kein Verwackeln
- **Hintergrund:** Einfarbig, kontrastreich zum Stein

### Beim Training
- **Overfitting vermeiden:** Wenn Training-Accuracy hoch, aber Validation-Accuracy niedrig → mehr Daten, mehr Dropout, Data Augmentation
- **Underfitting vermeiden:** Wenn beide Accuracies niedrig → größeres Modell, länger trainieren, Lernrate anpassen
- **Early Stopping:** Training beenden wenn Validation-Loss 5-10 Epochen nicht mehr sinkt
- **Learning Rate Scheduling:** Lernrate nach 20+ Epochen reduzieren

### Tools die helfen
- **TensorBoard:** Trainingsverlauf live beobachten
- **Confusion Matrix:** Zeigt welche Klassen verwechselt werden
- **Grad-CAM:** Visualisiert, worauf das CNN achtet (sehr lehrreich!)

---

## Fortschritt

| Phase | Schritt | Status |
|-------|---------|--------|
| 1     | Dataset erstellen | ⬜ |
| 1     | Data Augmentation | ⬜ |
| 1     | CNN definieren | ⬜ |
| 1     | Training | ⬜ |
| 1     | Evaluation | ⬜ |
| 1     | App-Integration | ⬜ |
| 2     | YOLO-Dataset | ⬜ |
| 2     | YOLO Training | ⬜ |
| 2     | YOLO Integration | ⬜ |
