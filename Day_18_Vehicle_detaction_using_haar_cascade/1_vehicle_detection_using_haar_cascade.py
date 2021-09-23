import cv2
import imutils

cars_HC_path = "cars.xml"
car_cascade_model = cv2.CascadeClassifier(cars_HC_path)

camera_instance = cv2.VideoCapture(0)

while True:
    success, img = camera_instance.read()
    img = imutils.resize(img, width=600)
    Gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cars_in_image = car_cascade_model.detectMultiScale(Gray_img, 1.1, 1)

    for (x, y, h, w) in cars_in_image:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 1)

    cv2.imshow("Traffic Analyser", img)
    print("---------------------------------------------------")
    total_cars = len(cars_in_image)
    if total_cars > 10:
        print("Traffic level : High")
    elif total_cars >5:
        print("Traffic level : Medium")
    elif total_cars>1:
        print("Traffic level : low")
    else:
        print("No Traffic")

    road_view = f"  {total_cars} cars on road."
    print(road_view)

    key = cv2.waitKey(1)
    if key == 27:
        break

camera_instance.release()
cv2.destroyAllWindows()