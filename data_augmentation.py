import albumentations as A
import os
import cv2
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
import torch
annotation_path = "data/labels/"
img_path = "data/images/"

img_list = []
copy_ann_list = []
copy_img_list = []
for img in os.listdir(img_path):
    img_list.append(img)


random_img_ann = [random.choice(img_list) for _ in range(200)]

# resclae_bboxes
def box_cxcywh_to_xyxy(x_c, y_c, w, h):
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return b


for i in range(200):
    img = cv2.imread(os.path.join(img_path + random_img_ann[i]))
    ann_file = random_img_ann[i][:-3] + "txt"
    ann = os.path.join(annotation_path + ann_file)
    copy_ann_list.append(ann)
    copy_img_list.append(img)


# for i in range(200):
#     cv2.imshow(random_img_ann[i], copy_img_list[i])
#     cv2.waitKey(0)

# transform for augmentation
transforms = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.5),
    A.GaussNoise(var_limit=(25.0, 75.0), mean=0.1, p=0.8),
    A.ISONoise(color_shift=(0.1, 0.3), p=0.5)
], bbox_params=A.BboxParams(format='yolo'))

classes = [0, 1, 2, 3, 4]

bboxes = []


for ann in copy_ann_list:
    bbox = []
    with open(os.path.join(ann)) as f:
        for row in f.readlines():
            a = row.strip().split(" ")
            label = int(a[0])
            box = [float(i) for i in a[1:]]
            box.append((classes[label]))
            bbox.append(box)
    bboxes.append(bbox)

transformed_img_list = []
transformed_bbox_list = []

for i in range(200):

    transformed = transforms(image=copy_img_list[i], bboxes=bboxes[i])
    transformed_img = transformed['image']
    transformed_bbox = transformed['bboxes']
    transformed_img_list.append(transformed_img)
    transformed_bbox_list.append(transformed_bbox)

tr = Image.fromarray(transformed_img_list[0])
plt.imshow(tr)
plt.show()

tr_list = []

for i in transformed_img_list:
    i = Image.fromarray(i)
    tr_list.append(i)

j = 1347
for i in tr_list:
    i.save(img_path + "IMG_" + "0" * (6 - len(str(j))) + str(j) + ".jpg")
    j += 1

j = 1347
for i in bboxes:
    f = open(annotation_path + "IMG_" + "0" * (6 - len(str(j))) + str(j) + ".txt", "a")
    for k in i:
        f.write(str(k[4]))
        for s in k[0:4]:
            f.write(" " + str(s))
        f.write("\n")

    j += 1

# tr.save("img.png")
# tr = cv2.imread("img.png")
#
# for i in transformed_bbox_list[0]:
#     print(i)
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     # fontScale
#     fontScale = 1
#
#     color = (0, 255, 0)
#     # Line thickness of 1 px
#     thickness = 2
#     (x, y, w, h, ) = [float(j) for j in i[0:4]]
#     l = i[-1]
#     img_w, img_h = (1080, 720)
#     x, y, w, h = torch.tensor(box_cxcywh_to_xyxy(x, y, w, h)) * torch.tensor([img_w, img_h, img_w, img_h],
#                                                                              dtype=torch.float32)
#     tr = cv2.rectangle(tr, (int(x.item()), int(y.item())), (int(w.item()), int(h.item())), (0, 0, 255), 3)
#
#     org = (int(x.item()) + 10, int(y.item()) - 10)
#     tr = cv2.putText(tr, str(l), org, font, fontScale, color, thickness, cv2.LINE_AA)
#
# plt.imshow(tr)
# plt.show()

