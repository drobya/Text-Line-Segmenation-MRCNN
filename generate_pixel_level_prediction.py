import os
import cv2
import numpy as np
from tqdm import tqdm


def generate_pixel_level_prediction(images_dir, labels_dir, out_dir, reassign_labels=True):

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'vis'), exist_ok=True)

    for page in tqdm(os.listdir(images_dir)):
        if page.split('.')[-1] != 'png':
            continue
        img_page = cv2.imread(os.path.join(images_dir, page), 0)
        vis_img = cv2.imread(os.path.join(os.path.join(labels_dir, 'vis'), '{}'.format(page)))
        img_blob_label = cv2.imread(os.path.join(labels_dir, page), 0)

        binary_img = np.zeros_like(img_blob_label)
        binary_img[img_page > 200] = 1

        cv2.imshow('image', cv2.resize(255*binary_img, (600, 800)))
        cv2.imshow('blob', cv2.resize(img_blob_label, (600, 800)))

        pixel_level_label = binary_img * img_blob_label

        vis_pixel_level = np.zeros_like(vis_img)
        vis_pixel_level[binary_img == 1] = vis_img[binary_img == 1]

        cv2.imshow('label', cv2.resize(50*pixel_level_label, (600, 800)))
        cv2.imshow('vis', cv2.resize(vis_pixel_level, (600, 800)))

        cv2.waitKey(1)

        # pixel_level_label = np.zeros_like(img_blob_label)
        #
        # values = np.unique(img_blob_label)
        #
        # for v_i, v in enumerate(values):
        #     if v == 0:
        #         continue
        #
        #
        #     tmp = np.zeros_like(pixel_level_label)
        #     tmp[img_blob_label == v] = 1
        #
        #     pixel_level_label[tmp*binary_img > 0] = v_i+1 if reassign_labels else v
        #
        #
        #     cv2.imshow('over', cv2.resize(100*tmp + img_page, (600, 800)))
        #     cv2.imshow('label', cv2.resize(255*tmp, (600, 800)))
        #     cv2.imshow('pl', cv2.resize(255*pixel_level_label, (600, 800)))
        #     cv2.waitKey()

        cv2.imwrite(os.path.join(out_dir, page), pixel_level_label)
        cv2.imwrite(os.path.join(os.path.join(out_dir, 'vis'), page), vis_pixel_level)


generate_pixel_level_prediction('data/DIVA/CB55/private_test/images', 'new_refined/prediction/blobs/DIVA_CSG863_private_test',
                                'new_refined/prediction/pixel_level/DIVA_CSG863_private_test')

