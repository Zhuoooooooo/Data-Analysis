import cv2
import random

cv2.namedWindow("Frame")  #新建一個顯示窗口
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#設定視窗的高和寬
width = cap.get(3)
height = cap.get(4)

sp_w = int(width / 3) #將視窗水平方向切成三等份
sp_h = 100
point = (-1, -1)


def mosaic_effect(img):
    new_img = img.copy()
    h, w, n = img.shape
    size = 10  # 馬賽克大小
    for i in range(size, h - 1 - size, size):
        for j in range(size, w - 1 - size, size):
            i_rand = random.randint(i - size, i)
            j_rand = random.randint(j - size, j)
            new_img[i - size:i + size, j - size:j + size] = img[i_rand, j_rand, :]
    return new_img

#設定滑鼠點擊
def click(event, x_, y_, flags, param):
    global point
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x_, y_)


cv2.setMouseCallback("Frame", click)

font = cv2.FONT_HERSHEY_SIMPLEX
color_black = (0, 0, 0)

while True:
    ret, frame = cap.read()
    #滑鼠點擊的地方
    x, y = point
    #按下第一個按鈕的範圍，轉為邊緣檢測
    if x in range(0, sp_w) and y in range(0, sp_h):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        frame = cv2.Canny(blurred, 10, 150)

    #按下第二個按鈕的範圍，轉為灰階
    elif x in range(sp_w, sp_w * 2) and y in range(0, sp_h):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #按下第三個按鈕範圍，轉為馬賽克
    elif x in range(sp_w * 2, sp_w * 3) and y in range(0, sp_h):
        frame = mosaic_effect(frame)

    #其他範圍則為原畫面
    else:
        frame = frame

    
    frame = cv2.copyMakeBorder(frame, top=sp_h, bottom=0, left=0, right=0, borderType=cv2.BORDER_CONSTANT)
    pos_x = [0, sp_w]
    pos_y = [0, sp_h]

    #設定第一個按鈕的顯示範圍、背景顏色及字體
    cv2.rectangle(frame, (pos_x[0], pos_y[0]), (pos_x[1], pos_y[1]), (220, 220, 220), cv2.FILLED)
    cv2.putText(frame, "Edge Detection",
                (int((pos_x[0] + pos_x[1]) / 2) - 55, int((pos_y[0] + pos_y[1]) / 2)),
                font, 0.5, color_black, 1)

    #設定第二個按鈕的顯示範圍、背景顏色及字體
    pos_x = [sp_w, sp_w * 2]
    cv2.rectangle(frame, (pos_x[0], pos_y[0]), (pos_x[1], pos_y[1]), (210, 210, 210), cv2.FILLED)
    cv2.putText(frame, "Gray",
                (int((pos_x[0] + pos_x[1]) / 2 - 20), int((pos_y[0] + pos_y[1]) / 2)),
                font, 0.5, color_black, 1)

    #設定第三個按鈕的顯示範圍、背景顏色及字體
    pos_x = [sp_w * 2, sp_w * 3]
    cv2.rectangle(frame, (pos_x[0], pos_y[0]), (pos_x[1], pos_y[1]), (200, 200, 200), cv2.FILLED)
    cv2.putText(frame, "Mosaic",
                (int((pos_x[0] + pos_x[1]) / 2 - 25), int((pos_y[0] + pos_y[1]) / 2)),
                font, 0.5, color_black, 1)

    cv2.imshow("Frame", frame)

    #按下 'q' 則break
    key = cv2.waitKey(1)
    if key & 0xFF == ord("q"):
        break