"""
Mask R-CNN

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 tls.py train --dataset=/path/to/tls/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 tls.py train --dataset=/path/to/tls/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 tls.py train --dataset=/path/to/tls/dataset --weights=imagenet

    # Apply color splash to an image
    python3 tls.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 tls.py splash --weights=last --video=<URL or path to file>
"""

import os
import datetime
import random
import matplotlib.cm as cm
from tqdm import tqdm

import h5py
import numpy as np
import cv2
import skimage.draw
import xml.etree.ElementTree as ET

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Import Mask RCNN
from mrcnn.config import Config
from mrcnn import model as modellib, utils, visualize
from matplotlib import patches, lines
from matplotlib.patches import Polygon

import wandb
from wandb.keras import WandbCallback

wandb.init(project='text-line-segmentation-mrcnn')

# Path to trained weights file
COCO_WEIGHTS_PATH = "mask_rcnn_coco.h5"

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = "logs"

PATCH_SIZE = 350

dataset_name = "AHTE"


############################################################
#  Configurations
############################################################


class TLSConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "text_lines_{}".format(dataset_name)

    NUM_CLASSES = 1 + 1  # Background + text-line

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (64, 128, 256, 512, 1024)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    # TRAIN_ROIS_PER_IMAGE = 60

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class TLSDataset(utils.Dataset):
    def load_pages_and_generate_patches(self, dataset_dir, subset, number_of_patches=5, patch_size=320, SHOW_RESULTS=False):
        xmlns = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2018-07-15"
        b_xmlns_l = []

        # Add text line a class
        self.add_class("text-lines", 1, "text-lines")

        # Train or validation dataset?
        # assert subset in ["train", "val"]

        print('reading {} images'.format(subset))
        images_annotations = []
        for f in tqdm(os.listdir('{}/{}/xml'.format(dataset_dir, subset))):
            name, ext = os.path.splitext(f)
            if not ext == '.xml':
                continue
            tree = ET.parse('{}/{}/xml/{}'.format(dataset_dir, subset, f))
            root = tree.getroot()
            b_xmlns = root.tag.split('PcGts')[0]

            page_e = root.find('{}Page'.format(b_xmlns))

            b_xmlns_l.append(b_xmlns)

            images_annotations.append(page_e)

        print('generating {} masks'.format(subset))

        pages = []
        masks = []
        # Add images
        for page_i, page_e in enumerate(tqdm(images_annotations)):

            b_xmlns = b_xmlns_l[page_i]

            image_path = os.path.join('{}/{}/images'.format(dataset_dir, subset),
                                      '{}.png'.format(page_e.attrib['imageFilename'].split('.')[0]))
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            pages.append(image)

            text_region_es = page_e.findall('{}TextRegion'.format(b_xmlns))
            polygons_txt = []
            for text_region_e in text_region_es:
                polygons_es = text_region_e.findall('{}TextLine'.format(b_xmlns))

                for e in polygons_es:
                    polygons_txt.append(e.find('{}Coords'.format(b_xmlns)).attrib['points'])

            polygons = [np.asarray([np.fromstring(p, sep=',') for p in poly_txt.split(' ')], dtype=np.int) for poly_txt
                        in polygons_txt]

            mask = np.zeros([height, width], dtype=np.uint8)

            for i, p in enumerate(polygons):
                # Get indexes of pixels inside the polygon and set them to 1
                rr, cc = skimage.draw.polygon(p[:, 1], p[:, 0])
                mask[rr, cc] = i

                if SHOW_RESULTS:
                    image[:,:] = 100*mask[:,:]+image[:,:]

                    cv2.imshow('masked', cv2.resize(image, (600,800)))
                    cv2.waitKey()

            masks.append(mask)

        patches_dir = os.path.join(dataset_dir, 'patches')
        patches_subset_dir = os.path.join(patches_dir, subset)

        patches_images_dir = os.path.join(patches_subset_dir, 'patches')
        patches_labels_dir = os.path.join(patches_subset_dir, 'labels')

        if not os.path.exists(patches_dir):
            os.mkdir(patches_dir)
        if not os.path.exists(patches_subset_dir):
            os.mkdir(patches_subset_dir)
        if not os.path.exists(patches_images_dir):
            os.mkdir(patches_images_dir)
        if not os.path.exists(patches_labels_dir):
            os.mkdir(patches_labels_dir)

        print('generating {} patches'.format(subset))

        pbar = tqdm(total=number_of_patches + 1)
        generated_patches_index = 0
        while generated_patches_index < number_of_patches:
            page_number = random.randint(0, len(pages) - 1)

            page = pages[page_number]
            lpage = masks[page_number]
            rows, cols = page.shape
            x = random.randint(0, rows - patch_size)
            y = random.randint(0, cols - patch_size)
            patch = page[x:x + patch_size, y:y + patch_size]

            img_lpatch = lpage[x:x + patch_size, y:y + patch_size]

            values, count = np.unique(img_lpatch, return_counts=True)

            lpatch = np.zeros((patch_size, patch_size, len(values)-1))


            v_i = 0
            for v in values:
                if v == 0:
                    continue
                lpatch[img_lpatch == v, v_i] = 1
                v_i += 1

            if sum([np.count_nonzero(lpatch[:, :, i]) for i in range(lpatch.shape[2])]) < 0.3 * (patch_size ** 2):
                continue

            eles_number = 0
            for t in range(lpatch.shape[2]):
                if np.count_nonzero(lpatch[:, :, t]) > 0:
                    eles_number += 1

            nlpatch = np.zeros((lpatch.shape[0], lpatch.shape[1], eles_number))
            ele_idx = 0
            for t in range(lpatch.shape[2]):
                if np.count_nonzero(lpatch[:, :, t]) > 0:
                    nlpatch[:, :, ele_idx] = lpatch[:, :, t]
                    ele_idx += 1

            patch_name = '{}_{}'.format(subset, generated_patches_index)

            patch_path = os.path.join(patches_images_dir, '{}.png'.format(patch_name))
            patch_label_path = os.path.join(patches_labels_dir, '{}.h5'.format(patch_name))

            h5f = h5py.File(patch_label_path, 'w')
            h5f.create_dataset('text-lines', data=nlpatch)
            h5f.close()
            cv2.imwrite(patch_path, patch)


            if SHOW_RESULTS:
                cv2.imshow('patch', patch)
                for m_i in range(nlpatch.shape[2]):
                    cv2.imshow('m_{}'.format(m_i), nlpatch[:,:, m_i])
                cv2.waitKey()



            self.add_image(
                "text-lines",
                image_id=generated_patches_index,
                path=patch_path,
                width=patch_size, height=patch_size,
                mask_path=patch_label_path)

            generated_patches_index += 1

            pbar.update(1)

        pbar.close()

    def load_from_file(self, dataset_dir, subset, number_to_load=None):

        # Add text line a class
        self.add_class("text-lines", 1, "text-lines")

        # Train or validation dataset?
        #assert subset in ["train", "val"]

        dataset_dir = os.path.join(dataset_dir, subset)
        images_dir_path = os.path.join(dataset_dir, 'patches')
        images_paths = os.listdir(images_dir_path)
        labels_dir_path = os.path.join(dataset_dir, 'labels')

        print('loading patches')
        number_of_patches = len(images_paths)
        if number_to_load is not None:
            number_of_patches = number_to_load

        for i in tqdm(range(number_of_patches)):
            image_name = images_paths[i]
            image_path = os.path.join(images_dir_path, image_name)
            # print(image_path)

            base_name = image_name.split('.')[0]

            label_path = os.path.join(labels_dir_path, '{}.h5'.format(base_name))
            # print(label_path)

            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "text-lines",
                image_id=i,
                path=image_path,
                width=width, height=height,
                mask_path=label_path)

        print('{} images loaded'.format(i))

    def load_mask(self, image_id):

        image_info = self.image_info[image_id]

        if image_info["source"] != "text-lines":
            return super(self.__class__, self).load_mask(image_id)

        h5f = h5py.File(image_info['mask_path'], 'r')
        mask = h5f['text-lines'][:]
        h5f.close()

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "text-lines":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = TLSDataset()
    # dataset_train.load_from_file(args.dataset, 'train')
    dataset_train.load_pages_and_generate_patches(args.dataset, "train", number_of_patches=50000, patch_size=PATCH_SIZE)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = TLSDataset()
    # dataset_val.load_from_file(args.dataset, 'test')
    dataset_val.load_pages_and_generate_patches(args.dataset, "test", number_of_patches=6000, patch_size=PATCH_SIZE)
    dataset_val.prepare()

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='heads', custom_callbacks=[WandbCallback()])

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=100,
                layers="all", custom_callbacks=[WandbCallback()])

    model_path = os.path.join('models', "mask_rcnn_diva_{}_tls.h5".format(dataset_name))
    model.keras_model.save_weights(model_path)




class SlidingWindowGenerator(object):
    def __init__(self, img, patch_size, margin):
        self.img = img
        self.patch_size = patch_size
        self.margin = margin
        self.inner_patch_size = patch_size - 2 * margin
        self._gen = self._generator()


    def __len__(self):
        return (self.img.shape[0]//self.inner_patch_size)*(self.img.shape[1]//self.inner_patch_size)

    def __iter__(self):
        self._gen = self._generator()
        return self

    def __next__(self):
        return next(self._gen)

    def _generator(self):
        padded_img = 255 * np.ones((self.img.shape[0] + 2 * self.margin, self.img.shape[1] + 2 * self.margin, self.img.shape[2]))
        padded_img[self.margin:self.margin + self.img.shape[0], self.margin:self.margin + self.img.shape[1], :] = self.img

        inner_patch_size = self.patch_size - 2 * self.margin

        for i in range(self.margin, self.margin + self.img.shape[0], inner_patch_size):
            for j in range(self.margin, self.margin + self.img.shape[1], inner_patch_size):
                yield i - self.margin, j - self.margin, padded_img[i - self.margin:i + self.patch_size + self.margin,
                                              j - self.margin:j + self.patch_size + self.margin, :]


    def next(self):
        return self.__next__()


class BatchSlidingWindowsGenerator(object):
    def __init__(self, img, patch_size, margin, batch_size):
        self.batch_size = batch_size
        self.sliding_window_generator = SlidingWindowGenerator(img, patch_size, margin)

        self.n_patches = len(self.sliding_window_generator)

        assert self.n_patches % batch_size == 0, '# of patches: {}, should divisible by batch_size {}'.format(self.n_patches, batch_size)

    def __len__(self):
        return self.n_patches//self.batch_size

    def __iter__(self):
        return self

    def __next__(self):
        batch = []
        for i in range(self.batch_size):
            s_w = next(self.sliding_window_generator)
            batch.append(s_w)
        return batch

    def next(self):
        return self.__next__()


def sliding_window_generator(img, patch_size, margin, step_size, padding_value=0):
    padded_img = padding_value * np.ones((img.shape[0] + 2 * margin, img.shape[1] + 2 * margin, img.shape[2]))
    padded_img[margin:margin + img.shape[0], margin:margin + img.shape[1], :] = img

    inner_patch_size = patch_size - 2 * margin

    for i in range(margin, margin + img.shape[0], step_size):
        for j in range(margin, margin + img.shape[1], step_size):
            if i + patch_size + margin > img.shape[0] or j + patch_size + margin > img.shape[1]:
                continue
            yield i - margin, j - margin, padded_img[i:i + patch_size,
                                          j:j + patch_size, :]


def batch_sliding_window_generator(img, patch_size, margin, batch_size):
    sliding_window = sliding_window_generator(img, patch_size, margin)

    batch = []
    for s_w in tqdm(sliding_window):
        batch.append(s_w)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch


def build_score_matrix(masks, patch_masks, x, y, patch_size, margin):
    assert len(masks) > 0, 'there should be at least one mask'

    scores = np.zeros((patch_masks.shape[2], len(masks)))

    for i in range(patch_masks.shape[2]):

        padded_mask = np.zeros((masks[0].shape[0] + 2 * margin, (masks[0].shape[1] + 2 * margin)))
        padded_mask[x:x + patch_size, y:y + patch_size] = patch_masks[:, :, i]

        cv2.imshow('small mask', 255 * patch_masks[:, :, i].astype(np.uint8))
        cv2.imshow('padded page mask', cv2.resize(255 * padded_mask, (300, 400)))
        for j in range(len(masks)):
            p_page_mask = padded_mask[margin:-margin, margin:-margin]
            cv2.imshow('intersection',
                       cv2.resize(255 * ((p_page_mask > 0) & (masks[j] > 0)).astype(np.uint8), (300, 400)))
            intersection = np.sum((p_page_mask > 0) & (masks[j] > 0))

            # print('intersection : {}'.format(intersection))

            # cv2.waitKey(0)

            scores[i, j] = intersection  # should be updated to include line direction/width

    return scores


def predict_pages2(model, page_path=None, pages_dir_path=None, MIN_MASK_SIZE=50, MIN_MATCH_SCORE=0):
    assert page_path or pages_dir_path

    if page_path:
        pages_paths = [page_path]

    elif pages_dir_path:
        _pages_paths = os.listdir(pages_dir_path)
        pages_paths = []

        for p in _pages_paths:
            pages_paths.append(os.path.join(pages_dir_path, p))
    else:
        print('page_path or page_dir_path should be assigned')
        raise

    masks = []

    for p_path in pages_paths:

        image = skimage.io.imread(p_path)

        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]

        margin = PATCH_SIZE // 6
        inner_size = PATCH_SIZE - 2 * margin

        print('patch_size: {}, margin: {}, inner_size:{}'.format(PATCH_SIZE, margin, inner_size))

        sliding_window = sliding_window_generator(image, inner_size, margin)

        for i, j, patch in tqdm(sliding_window):

            p_p = model.detect([patch], verbose=0)[0]

            patch_masks = p_p['masks']

            if len(masks) == 0:
                for m_i in range(patch_masks.shape[2]):
                    if np.sum(patch_masks[margin:-margin, margin:-margin, m_i] > 0) > MIN_MASK_SIZE:
                        mask = np.zeros((image.shape[0], image.shape[1]))
                        mask[i:i + inner_size, j: j + inner_size] = patch_masks[margin:-margin, margin:-margin, m_i]

                        masks.append(mask)
            else:
                from scipy.optimize import linear_sum_assignment
                scores = build_score_matrix(masks, patch_masks, i, j, PATCH_SIZE, margin)
                cost = -1 * scores
                row_ind, col_ind = linear_sum_assignment(cost)

                for m_i in range(patch_masks.shape[2]):
                    assign_i = row_ind == m_i
                    if np.sum(assign_i) > 0 and scores[m_i, col_ind[assign_i]] > MIN_MATCH_SCORE:
                        masks[col_ind[assign_i][0]][i:i + inner_size, j: j + inner_size] = patch_masks[margin:-margin,
                                                                                           margin:-margin, m_i]
                    else:
                        if np.sum(patch_masks[margin:-margin, margin:-margin, m_i] > 0) > MIN_MASK_SIZE:
                            mask = np.zeros((image.shape[0], image.shape[1]))
                            mask[i:i + inner_size, j: j + inner_size] = patch_masks[margin:-margin, margin:-margin, m_i]

                            masks.append(mask)
            disp_image = image.copy()
            for m_i, mask in enumerate(masks):
                import colorsys
                visualize.apply_mask(disp_image, mask, colorsys.hsv_to_rgb(m_i / len(masks), 1, 1.0))

            cv2.rectangle(disp_image, (j, i), (j + inner_size, i + inner_size), (0, 0, 255), 2)
            cv2.rectangle(disp_image, (max(j - margin, 0), max(i - margin, 0)),
                          (min(j + inner_size + margin, image.shape[0]), min(i + inner_size + margin, image.shape[1])),
                          (255, 0, 0), 2)
            cv2.imshow('masks', cv2.resize(disp_image, (600, 800)))
            cv2.waitKey(1)

    for mask in masks:
        cv2.imshow('mask', cv2.resize(255 * mask.astype(np.uint8), (600, 800)))
        cv2.waitKey(0)


def connected_masks(masks, p_page_mask, thresh=1100):
    assert len(masks) > 0, 'there should be at least one mask'

    p_connected_masks = []

    for j in range(len(masks)):
        cv2.imshow('intersection',
                   cv2.resize(255 * ((p_page_mask > 0) & (masks[j] > 0)).astype(np.uint8), (300, 400)))

        intersection = np.sum((p_page_mask > 0) & (masks[j] > 0))

        # print('intersection: {}'.format(intersection))

        if intersection > thresh:
            p_connected_masks.append(j)

    return p_connected_masks


def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # Generate random colors
    colors = colors or visualize.random_colors(N)

    masked_image = image.astype(np.uint8).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            cv2.rectangle(masked_image, (x1, y1), (x2, y2), color, 1)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]

        cv2.putText(masked_image, str(caption), (x1, y1 + 8), cv2.FONT_HERSHEY_SIMPLEX, .5, color, lineType=cv2.LINE_AA)

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = visualize.apply_mask(masked_image, mask, color)

    return masked_image

# 
def predict_pages(model, page_path=None, pages_dir_path=None, MIN_MASK_SIZE=50, MIN_MATCH_SCORE=0, batch_size=20):
    assert page_path or pages_dir_path

    if page_path:
        pages_paths = [page_path]

    elif pages_dir_path:
        _pages_paths = os.listdir(pages_dir_path)
        pages_paths = []

        for p in _pages_paths:
            pages_paths.append(os.path.join(pages_dir_path, p))
    else:
        print('page_path or page_dir_path should be assigned')
        raise

    masks = []

    for p_path in pages_paths:

        image = skimage.io.imread(p_path)

        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]

        margin = PATCH_SIZE // 5
        inner_size = PATCH_SIZE - 2 * margin

        print('patch_size: {}, margin: {}, inner_size:{}'.format(PATCH_SIZE, margin, inner_size))

        batch_sliding_window = BatchSlidingWindowsGenerator(image, inner_size, margin, batch_size)  # batch_sliding_window_generator(image, inner_size, margin, batch_size)

        for s_w_batch in tqdm(batch_sliding_window):

            p_batch = []
            for _,_, patch in s_w_batch:
                p_batch.append(patch)


            batch_p_p = model.detect(p_batch, verbose=0)

            for p_p_i, p_p in enumerate(batch_p_p):
                i, j, patch = s_w_batch[p_p_i]

                disp_patch = display_instances(patch, p_p['rois'], p_p['masks'], p_p['class_ids'], ['BG', 'tls'],
                                               p_p['scores'])
                cv2.imshow('p_patch', disp_patch)

                patch_masks = p_p['masks']

                print("# masks: {}".format(len(masks)))


                if len(masks) == 0:
                    for m_i in range(patch_masks.shape[2]):
                        if np.sum(patch_masks[margin:-margin, margin:-margin, m_i] > 0) > MIN_MASK_SIZE:
                            mask = np.zeros((image.shape[0], image.shape[1]))
                            mask[i:i + inner_size, j: j + inner_size] = patch_masks[margin:-margin, margin:-margin, m_i]

                            masks.append(mask)
                else:

                    for m_i in range(patch_masks.shape[2]):

                        cv2.imshow('considered mask', 255 * patch_masks[:, :, m_i].astype(np.uint8))

                        padded_mask = np.zeros((masks[0].shape[0] + 2 * margin, (masks[0].shape[1] + 2 * margin)))
                        padded_mask[i:i + PATCH_SIZE, j:j + PATCH_SIZE] = patch_masks[:, :, m_i]

                        p_page_mask = padded_mask[margin:-margin, margin:-margin]

                        connected = connected_masks(masks, p_page_mask)

                        if len(connected) > 0:
                            new_mask = np.zeros_like(masks[0])
                            for connected_mask_i in connected:
                                new_mask[masks[connected_mask_i] > 0] = 1

                            new_mask[i:i + inner_size, j: j + inner_size][
                                patch_masks[margin:-margin, margin:-margin, m_i] > 0] = 1

                            cv2.imshow('new mask', cv2.resize(255 * new_mask.astype(np.uint8), (300, 400)))

                            n_masks = []
                            for m_j in range(len(masks)):
                                if m_j not in connected:
                                    n_masks.append(masks[m_j])

                            n_masks.append(new_mask)

                            masks = n_masks

                        elif np.sum(patch_masks[margin:-margin, margin:-margin, m_i] > 0) > MIN_MASK_SIZE:
                            mask = np.zeros((image.shape[0], image.shape[1]))
                            mask[i:i + inner_size, j: j + inner_size] = patch_masks[margin:-margin, margin:-margin, m_i]

                            masks.append(mask)

                disp_image = image.copy()
                for m_i, mask in enumerate(masks):
                    import colorsys
                    visualize.apply_mask(disp_image, mask, colorsys.hsv_to_rgb(m_i / len(masks), 1, 1.0))

                cv2.rectangle(disp_image, (j, i), (j + inner_size, i + inner_size), (0, 0, 255), 2)
                cv2.rectangle(disp_image, (max(j - margin, 0), max(i - margin, 0)),
                              (min(j + inner_size + margin, image.shape[0]), min(i + inner_size + margin, image.shape[1])),
                              (255, 0, 0), 2)
                cv2.imshow('masks', cv2.resize(disp_image, (600, 800)))
                cv2.waitKey(1)

    for mask in masks:
        cv2.imshow('mask', cv2.resize(255 * mask.astype(np.uint8), (600, 800)))
        cv2.waitKey(0)


def p_predict_pages(model, page_path=None, pages_dir_path=None, MIN_MASK_SIZE=200, MIN_MATCH_SCORE=0):
    assert page_path or pages_dir_path

    if page_path:
        pages_paths = [page_path]

    elif pages_dir_path:
        _pages_paths = os.listdir(pages_dir_path)
        pages_paths = []

        for p in _pages_paths:
            pages_paths.append(os.path.join(pages_dir_path, p))
    else:
        print('page_path or page_dir_path should be assigned')
        raise

    masks = []

    for p_path in pages_paths:

        image = skimage.io.imread(p_path)

        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]

        margin = PATCH_SIZE // 5
        inner_size = PATCH_SIZE - 2 * margin

        print('patch_size: {}, margin: {}, inner_size:{}'.format(PATCH_SIZE, margin, inner_size))

        sliding_window = sliding_window_generator(image, inner_size, margin)

        for i, j, patch in tqdm(sliding_window):

            p_p = model.detect([patch], verbose=0)[0]

            disp_patch = display_instances(patch, p_p['rois'], p_p['masks'], p_p['class_ids'], ['BG', 'tls'],
                                           p_p['scores'])
            cv2.imshow('p_patch', disp_patch)

            patch_masks = p_p['masks']

            print("# masks: {}".format(len(masks)))

            if len(masks) == 0:
                for m_i in range(patch_masks.shape[2]):
                    if np.sum(patch_masks[margin:-margin, margin:-margin, m_i] > 0) > MIN_MASK_SIZE:
                        mask = np.zeros((image.shape[0], image.shape[1]))
                        mask[i:i + inner_size, j: j + inner_size] = patch_masks[margin:-margin, margin:-margin, m_i]

                        masks.append(mask)
            else:

                for m_i in range(patch_masks.shape[2]):

                    cv2.imshow('considered mask', 255 * patch_masks[:, :, m_i].astype(np.uint8))

                    padded_mask = np.zeros((masks[0].shape[0] + 2 * margin, (masks[0].shape[1] + 2 * margin)))
                    padded_mask[i:i + PATCH_SIZE, j:j + PATCH_SIZE] = patch_masks[:, :, m_i]

                    p_page_mask = padded_mask[margin:-margin, margin:-margin]

                    connected = connected_masks(masks, p_page_mask)

                    if len(connected) > 0:
                        new_mask = np.zeros_like(masks[0])
                        for connected_mask_i in connected:
                            new_mask[masks[connected_mask_i] > 0] = 1

                        new_mask[i:i + inner_size, j: j + inner_size][
                            patch_masks[margin:-margin, margin:-margin, m_i] > 0] = 1

                        cv2.imshow('new mask', cv2.resize(255 * new_mask.astype(np.uint8), (300, 400)))

                        n_masks = []
                        for m_j in range(len(masks)):
                            if m_j not in connected:
                                n_masks.append(masks[m_j])

                        n_masks.append(new_mask)

                        masks = n_masks

                    elif np.sum(patch_masks[margin:-margin, margin:-margin, m_i] > 0) > MIN_MASK_SIZE:
                        mask = np.zeros((image.shape[0], image.shape[1]))
                        mask[i:i + inner_size, j: j + inner_size] = patch_masks[margin:-margin, margin:-margin, m_i]

                        masks.append(mask)

            disp_image = image.copy()
            for m_i, mask in enumerate(masks):
                import colorsys
                visualize.apply_mask(disp_image, mask, colorsys.hsv_to_rgb(m_i / len(masks), 1, 1.0))

            cv2.rectangle(disp_image, (j, i), (j + inner_size, i + inner_size), (0, 0, 255), 2)
            cv2.rectangle(disp_image, (max(j - margin, 0), max(i - margin, 0)),
                          (min(j + inner_size + margin, image.shape[0]), min(i + inner_size + margin, image.shape[1])),
                          (255, 0, 0), 2)
            cv2.imshow('masks', cv2.resize(disp_image, (600, 800)))
            cv2.waitKey(1)

    for mask in masks:
        cv2.imshow('mask', cv2.resize(255 * mask.astype(np.uint8), (600, 800)))
        cv2.waitKey(0)




def bit_predict_page(model, page_path=None, pages_dir_path=None, MIN_MASK_SIZE=6000, THRESHHOLD=0.4, SHOW_RESULTS=False, SAVE_VID=False):
    assert page_path or pages_dir_path

    if page_path:
        pages_paths = [page_path]

    elif pages_dir_path:
        _pages_paths = os.listdir(pages_dir_path)
        pages_paths = []

        for p in _pages_paths:
            pages_paths.append(os.path.join(pages_dir_path, p))
    else:
        print('page_path or page_dir_path should be assigned')
        raise


    m_colors = {}

    m_ids = {}

    c_m_id = 1

    out_dir = './out/{}'.format(dataset_name)

    os.makedirs(out_dir, exist_ok=True)

    for p_path in pages_paths:

        image = skimage.io.imread(p_path)
        if SHOW_RESULTS:
            cv2.imshow('input image', cv2.resize(image, (600, 800)))

        if SAVE_VID:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            dimensions = (image.shape[1], image.shape[0])
            fps = 2.0
            out = cv2.VideoWriter(os.path.join(out_dir,'{}.mp4'.format(p_path.split('/')[-1].split('.')[0])), fourcc, fps, dimensions)

        output_image = np.zeros_like(image, dtype=np.uint8)

        disp_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]

        step_size = PATCH_SIZE // 7
        margin = PATCH_SIZE // 4
        inner_size = PATCH_SIZE - 2 * margin

        print('patch_size: {}, margin: {}, inner_size:{}'.format(PATCH_SIZE, margin, inner_size))

        sliding_window = sliding_window_generator(image, PATCH_SIZE, margin, step_size)

        for i, j, patch in tqdm(sliding_window, desc='sliding window'):

            if np.sum(np.sum(patch[margin:-margin, margin:-margin])) < 255*MIN_MASK_SIZE:
                continue

            p_p = model.detect([patch], verbose=0)[0]

            disp_patch = display_instances(patch, p_p['rois'], p_p['masks'], p_p['class_ids'], ['BG', 'tls'],
                                           p_p['scores'])

            if SHOW_RESULTS:
                cv2.imshow('patch', patch)
                cv2.imshow('p_patch', disp_patch)

            patch_masks = p_p['masks']

            for m_i in range(patch_masks.shape[2]):

                # print(np.sum(np.sum(patch_masks[:, :, m_i])))

                if np.sum(np.sum(patch_masks[margin:-margin, margin:-margin, m_i])) < MIN_MASK_SIZE//2:
                    continue

                o_p = output_image[i:i + PATCH_SIZE, j:j + PATCH_SIZE] * patch_masks[:, :, m_i]

                values, count = np.unique(o_p, return_counts=True)


                p_id = -1

                if len(values) == 2:
                    if count[values != 0][0] >= MIN_MASK_SIZE:
                        p_id = values[values != 0][0]
                        m_ids[p_id] += 1
                else:
                    max_id = -1
                    max_val = 0
                    for v_i, v in enumerate(values):
                        score = count[v_i]/np.sum(np.sum(patch_masks[:, :, m_i]))
                        # print(score)
                        if v == 0:
                            continue
                        if max_val < score and score >= THRESHHOLD:
                            max_val = score
                            max_id = v
                    p_id = max_id


                if p_id == -1:
                    m_ids[c_m_id] = 1
                    p_id = c_m_id
                    c_m_id += 1

                    m_colors[p_id] = np.random.rand(3)*255


                # print(p_id)


                import colorsys
                output_image[i:i + PATCH_SIZE, j:j + PATCH_SIZE][o_p != 0] = p_id
                output_image[i:i + PATCH_SIZE, j:j + PATCH_SIZE][patch_masks[:, :, m_i] != 0] = p_id
                #
                # c_t = cm.get_cmap('Paired')(p_id)
                # color = np.asarray((c_t[0], c_t[1], c_t[2]))*255

                disp_image[i:i + PATCH_SIZE, j:j + PATCH_SIZE][o_p != 0] = m_colors[p_id]
                disp_image[i:i + PATCH_SIZE, j:j + PATCH_SIZE][patch_masks[:, :, m_i] != 0] = m_colors[p_id]


                # p_id = np.exp2(8-m_i)
                # print(np.exp2(m_i))
                # output_image[i:i + PATCH_SIZE, j:j + PATCH_SIZE] = np.bitwise_or(output_image[i:i + PATCH_SIZE, j:j + PATCH_SIZE],
                #                                                                  (p_id * patch_masks[:, :, m_i]).astype(np.uint8))

            if SAVE_VID:
                out.write(disp_image)

            if SHOW_RESULTS:
                cv2.imshow('output', cv2.resize(disp_image, (600, 800)))
                cv2.imshow('overlay', cv2.resize(0.5*disp_image+image, (600, 800)))


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        cv2.imwrite(os.path.join(out_dir, '{}'.format(p_path.split('/')[-1])), output_image)
        cv2.imwrite(os.path.join(out_dir, 'ann_{}'.format(p_path.split('/')[-1])), disp_image)
        
        if SAVE_VID:
            out.release()

        wandb.log({'ann_{}'.format(p_path.split('/')[-1]): wandb.Image(disp_image),
                   '{}'.format(p_path.split('/')[-1]): wandb.Image(output_image)}) #, 'prediction_video_{}'.format(p_path.split('/')[-1].split('.')[0]): wandb.Video('./out/{}.mp4'.format(p_path.split('/')[-1].split('.')[0]), fps=2, format='mp4')})

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect segment text-lines.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'inference'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/moc/dataset/",
                        help='Directory of the MOC_VML dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--page', required=False,
                        metavar="path or URL to page")
    parser.add_argument('--pages_dir', required=False,
                        metavar="path or pages dir")
    parser.add_argument('--batch_size', required=False, default=1, type=int,
                        metavar="batch size")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "predict":
        assert args.page or args.pages_dir, \
            "Provide --page or --pages_dir to apply prediction on"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)


    # Configurations
    if args.command == "train":
        config = TLSConfig()
    else:
        class InferenceConfig(TLSConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1

            IMAGES_PER_GPU = args.batch_size

            DETECTION_MIN_CONFIDENCE = 0.5


        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "predict":
        # predict_pages(model, page_path=args.page, pages_dir_path=args.pages_dir, batch_size=args.batch_size)
        bit_predict_page(model, page_path=args.page, pages_dir_path=args.pages_dir)

    else:
        print("'{}' is not recognized. ".format(args.command))
