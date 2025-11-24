# utils/visualization_utils.py

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os



def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1],
               color='green', marker='*', s=marker_size,
               edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1],
               color='red', marker='*', s=marker_size,
               edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h,
                               edgecolor='green',
                               facecolor=(0, 0, 0, 0),
                               lw=2))

def show_masks(output_dir,image_name, image, mask, scores, point_coords=None, box_coords=None, input_labels=None,
               borders=True):
    plt.figure(figsize=(10, 10))

    plt.imshow(image)

    show_mask(mask, plt.gca(), borders=borders)
    if point_coords is not None:
        assert input_labels is not None
        show_points(point_coords, input_labels, plt.gca())
    if box_coords is not None:
        show_box(box_coords, plt.gca())

    os.makedirs(f"{output_dir}/images", exist_ok=True)
    output_path = os.path.join(f"{output_dir}/images", image_name)

    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    if 0:
        plt.axis('off')
        plt.show()


def mask_to_polygon(mask, min_contour_area=50):
    mask = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    if hierarchy is None:
        return segmentation
    for i, contour in enumerate(contours):
        contour = contour.flatten().tolist()
        if len(contour) < 6 or cv2.contourArea(contours[i]) < min_contour_area:
            continue
        if hierarchy[0][i][3] == -1:
            segmentation.append(contour)
        else:
            segmentation.append(contour[::-1])
    return segmentation
