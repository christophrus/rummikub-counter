"""Drücke Tasten im Fenster und sieh die Key-Codes. Q zum Beenden."""
import cv2

cv2.namedWindow("Key Test", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Key Test", 400, 100)
print("Drücke Tasten im OpenCV-Fenster. Q zum Beenden.")

while True:
    img = __import__("numpy").zeros((100, 400, 3), dtype="uint8")
    cv2.putText(img, "Press keys, check console", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow("Key Test", img)

    key = cv2.waitKeyEx(0)  # waitKeyEx gibt volle Key-Codes
    print(f"  Key code: {key}  (0xFF masked: {key & 0xFF})")

    if key == ord('q'):
        break

cv2.destroyAllWindows()
