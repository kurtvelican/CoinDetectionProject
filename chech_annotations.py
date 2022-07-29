import cv2
import os
import torch

annotation_path = "data/labels"
img_path = "data/images"

def box_cxcywh_to_xyxy(x_c, y_c, w, h):
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return b


labels = ["5 kurus", "10 kurus", "25 kurus", "50 kurus", "1 lira"]


for img in os.listdir(img_path):
    image = cv2.imread(os.path.join(img_path, img))

    ann_file = img[:-3] + "txt"
    ann_path = os.path.join(annotation_path, ann_file)

    with open(ann_path, "r") as f:
        font = cv2.FONT_HERSHEY_SIMPLEX
        # fontScale
        fontScale = 1

        color = (0, 255, 0)
        # Line thickness of 1 px
        thickness = 2

        for line in f.readlines():
            (l, x, y, w, h) = [float(i) for i in line.split(" ")]
            print(x, y, w, h, l)

            img_w, img_h = (1080, 720)
            x, y, w, h = torch.tensor(box_cxcywh_to_xyxy(x, y, w, h)) * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
            print(x, y, w, h)
            image = cv2.rectangle(image, (int(x.item()), int(y.item())), (int(w.item()), int(h.item())), (0, 0, 255), 3)

            org = (int(x.item()) + 10, int(y.item()) - 10)
            image = cv2.putText(image, labels[int(l)], org, font,
                                fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow(ann_file, image)
    cv2.waitKey(0)

