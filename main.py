import cv2
import numpy as np

BODY_PARTS = {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
              "Background": 15}
BODY_PAIRS = [["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
              ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
              ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"]]
threshold = 0.5

protoTxt = "pose_deploy_linevec_faster_4_stages.prototxt"
wagi = "pose_iter_160000.caffemodel"

# wczytanie sieci
net = cv2.dnn.readNetFromCaffe(protoTxt, wagi)


def angle_three_points(a, b, c):
    ang = np.degrees(np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0]))
    ang = ang + 360 if ang < 0 else ang
    ang = 360 - ang if ang > 180 else ang
    return ang


def detect_angles(_pairs):
    a1, a2 = _pairs[7]
    b1, b2 = _pairs[5]
    angle_left_shoulder = angle_three_points(a2, a1, b2)
    print(7, 5, angle_left_shoulder)

    # prawa reka
    a1, a2 = _pairs[5]
    b1, b2 = _pairs[7]
    angle_right_shoulder = angle_three_points(a2, b1, b2)
    print(5, 7, angle_right_shoulder)

    # lewe ramie
    a1, a2 = _pairs[3]
    b1, b2 = _pairs[2]
    angle_left_arm = angle_three_points(a2, a1, b1)
    print(3, 2, angle_left_arm)

    # prawe ramie
    a1, a2 = _pairs[5]
    b1, b2 = _pairs[6]
    angle_right_arm = angle_three_points(a1, a2, b2)
    print(5, 6, angle_right_arm)

    return (angle_left_shoulder, angle_right_shoulder, angle_left_arm, angle_right_arm)


def detect_flex(_pairs):
    als, ars, ala, ara = detect_angles(_pairs)

    if als > 45 and ars > 45 and ala < 100 and ara < 100:
        return True

    return False


frame = cv2.imread("flex.jpg")

frameWidth = frame.shape[1]
frameHeight = frame.shape[0]

# rozmiar input
inHeight = 368
inWidth = int((inHeight / frameHeight) * frameWidth)

# przygotowanie obrazu
inpBlob = cv2.dnn.blobFromImage(frame, 1.0/255/2, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

# ustawienie jako input
net.setInput(inpBlob)
# predykcjaqq
output = net.forward()

H = output.shape[2]
W = output.shape[3]

pairs = []

for pairNo, pair in enumerate(BODY_PAIRS):
    # pierwsza część pary
    start = pair[0]
    i = BODY_PARTS[start]

    probMapStart = output[0, i, :, :]
    _, _, _, pointStart = cv2.minMaxLoc(probMapStart)
    xStart = (frameWidth * pointStart[0]) / W
    yStart = (frameHeight * pointStart[1]) / H

    # druga część pary
    end = pair[1]
    j = BODY_PARTS[end]

    probMapEnd = output[0, j, :, :]
    _, _, _, pointEnd = cv2.minMaxLoc(probMapEnd)
    xEnd = (frameWidth * pointEnd[0]) / W
    yEnd = (frameHeight * pointEnd[1]) / H

    # generowanie par linii
    pairs.append(((int(xStart), int(yStart)), (int(xEnd), int(yEnd))))

    # rysowanie linii miedzy parami
    cv2.line(frame, (int(xStart), int(yStart)), (int(xEnd), int(yEnd)), (0, 255, 0), 5)
    cv2.putText(frame, "{}".format(pairNo), (int((xStart + xEnd) / 2), int((yStart + yEnd) / 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3, lineType=cv2.LINE_AA)

print(detect_flex(pairs))

cv2.imshow("test", frame)
cv2.waitKey(0)
