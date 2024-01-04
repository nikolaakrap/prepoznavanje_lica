from deepface import DeepFace
import matplotlib.pyplot as plt
import cv2

def face_rec_img():
    backend = "opencv"
    lokacija = "kiki3.png"   #neka od slika koja se nalazi uz python file

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

def face_rec_cam():
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    kamera = cv2.VideoCapture(0)
    analiza_gotova = False

    while True:
        _, frame = kamera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lice = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40), flags=cv2.CASCADE_SCALE_IMAGE)

        if not analiza_gotova:
            result = DeepFace.analyze(frame, actions=['age', 'gender'], enforce_detection=False)
            analiza_gotova = True

        for (x,y, w, h) in lice:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            tekst = f"Age:{result[0]['age']}/Gender:{result[0]['dominant_gender']}"
            cv2.putText(frame, tekst, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) == ord("q"):
            break


    kamera.release()
    cv2.destroyAllWindows()

input = int(input("Unesite 1 ako želite prepoznavanje slike, odnosno 2 ako želite učitati kameru: "))
if input == 1:
    face_rec_img()
elif input == 2:
    face_rec_cam()
else:
    print("Pokušajte ponovo!")