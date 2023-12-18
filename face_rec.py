from deepface import DeepFace
import matplotlib.pyplot as plt
import cv2

backend = "opencv"
lokacija = "joe1.png"   #neka od slika koja se nalazi uz python file

slika = cv2.imread(lokacija)
slika = cv2.resize(slika, (720, 640))
gray = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY)

cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
lice = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

rezultat = DeepFace.analyze(img_path=lokacija, actions=["age", "gender"])
print(rezultat)

for (x,y,w,h) in lice:
    cv2.rectangle(slika, (x, y), (x+w, y+h), (0, 255, 0), 2)
    tekst = f"Dob: {rezultat[0]['age']}; Spol: {rezultat[0]['dominant_gender']}"
    cv2.putText(slika, tekst, (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (255, 255, 255), 1)

slika_rgb = cv2.cvtColor(slika, cv2.COLOR_BGR2RGB)

cv2.imshow("lice", slika_rgb)
cv2.waitKey(0)