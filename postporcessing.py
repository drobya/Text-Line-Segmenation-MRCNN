import cv2
import numpy as np
import os
import pandas
from tqdm import tqdm
from scipy.signal import find_peaks

SHOW_RESULTS = False


def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def get_mean_text_line_size(page):
    values = np.unique(page)
    sizes = []
    for v in values:
        if v == 0:
            continue

        v_page = np.zeros_like(page)
        v_page[page == v] = 1

        sizes.append(np.sum(np.sum(v_page[page == v])))

    sizes = np.asarray(sizes)

    return sizes.mean()


def get_lines_info(page, alpha=0.7):
    values = np.unique(page)
    lines = pandas.DataFrame()
    min_line_size = alpha * get_mean_text_line_size(page)

    # cv2.imshow('in page', cv2.resize(page, (600, 800)))
    # cv2.waitKey(1)
    # print('min', min_line_size)
    # vis = np.zeros((*page.shape[:2], 3), dtype=np.uint8)
    for v in values:
        if v == 0:
            continue

        v_page = np.zeros_like(page)
        v_page[page == v] = 1

        # vis[page == v] = [0,0, 255]
        # cv2.imshow('vis_lines', cv2.resize(vis, (600, 800)))
        # cv2.waitKey(1)

        area = np.sum(np.sum(v_page[page == v]))

        if area < min_line_size:
            continue

        rmin, rmax, cmin, cmax = bbox2(v_page)

        max_y = rmax
        min_y = rmin

        if SHOW_RESULTS:
            disp_page = np.zeros((*page.shape, 3))
            disp_page[:, :, 0] = v_page
            disp_page[:, :, 1] = v_page
            disp_page[:, :, 2] = v_page

            print(min_y, max_y)

            disp_page = cv2.line(disp_page, (0, min_y), (disp_page.shape[1], min_y), (255, 0, 255), 5)
            disp_page = cv2.line(disp_page, (0, max_y), (disp_page.shape[1], max_y), (255, 0, 0), 7)
            cv2.imshow('contour', cv2.resize(disp_page, (600, 800)))
            cv2.waitKey(0)

            print('min_y ', max_y, 'shape[1]', page.shape[1])

        # if max_y > 0 and min_y < page.shape[1] + 1:
        lines = lines.append({'value': v, 'min_y': min_y, 'max_y': max_y, 'height': max_y - min_y, 'area': area},
                             ignore_index=True)

    lines = lines.sort_values(by='area', ascending=True)

    height_mean = lines['height'].mean()

    print('average line height: {}, max {}, min {}'.format(height_mean, lines['height'].max(), lines['height'].min()))

    return lines


def best_split_point(img, interval=0.15):
    y_hist = np.sum(img, axis=1)

    mid_start = y_hist.shape[0] // 2 - int(y_hist.shape[0] * interval / 2)
    mid_end = y_hist.shape[0] // 2 + int(y_hist.shape[0] * interval / 2)

    mid_hist = y_hist[mid_start:mid_end]

    s_p = np.argmin(mid_hist)

    hist = np.zeros((y_hist.shape[0], int(np.max(y_hist)) + 20))

    for i in range(y_hist.shape[0]):
        hist[i, :y_hist[i]] = 255

    if SHOW_RESULTS:
        cv2.imshow('lines', 255 * (img > 0).astype(np.uint8))
        cv2.imshow('hist', hist)
        cv2.waitKey(1)

    return s_p + mid_start


def line_std(img):
    y_hist = np.sum(img, axis=1)
    return y_hist.std()


def merge_lines(page, margin=0.01, alpha=0.4):
    lines = get_lines_info(page, alpha=alpha)

    refined_page = np.zeros_like(page)
    cline_id = 1

    heights_mean = lines['height'].mean()
    merged = []
    for i, r in tqdm(lines.iterrows(), desc='merging'):
        current_value = r['value']

        if current_value == 0 or current_value in merged:
            continue

        min_y = int(r['min_y']) - int(margin * heights_mean)
        max_y = int(r['max_y']) + int(margin * heights_mean)

        # print(current_value, min_y, max_y)

        curr_line_page = np.zeros_like(page)
        # print('id', cline_id)
        curr_line_page[page == current_value] = cline_id

        if SHOW_RESULTS:
            vis_refined2 = np.zeros((*curr_line_page.shape, 3), dtype=np.uint8)
            values2 = np.unique(curr_line_page)

            # print(values2)

            for v in values2:
                if v == 0:
                    continue
                vis_refined2[curr_line_page == v] = (255 * np.random.rand(3)).astype(np.uint8)
            cv2.imshow('b_currline', cv2.resize(vis_refined2, (600, 800)))

        curr_line_page, c_merged = merge_classes(page, curr_line_page, current_value, min_y, max_y, cline_id)

        merged.extend(c_merged)

        cline_id += 1

        refined_page[curr_line_page > 0] = curr_line_page[curr_line_page > 0]

        if SHOW_RESULTS:

            vis_refined = np.zeros((*refined_page.shape, 3), dtype=np.uint8)
            vis_refined2 = np.zeros((*curr_line_page.shape, 3), dtype=np.uint8)

            values = np.unique(refined_page)
            values2 = np.unique(curr_line_page)

            print(values, values2)

            for v in values2:
                if v == 0:
                    continue
                vis_refined2[curr_line_page == v] = (255 * np.random.rand(3)).astype(np.uint8)

            for v in values:
                if v == 0:
                    continue
                vis_refined[refined_page == v] = (255 * np.random.rand(3)).astype(np.uint8)

            cv2.imshow('refined', cv2.resize(vis_refined, (600, 800)))
            cv2.imshow('currline', cv2.resize(vis_refined2, (600, 800)))
            cv2.imshow('refined_page', cv2.resize(5 * refined_page, (600, 800)))
            cv2.waitKey(0)

    return refined_page


def split_lines(page, margin=0.01, alpha=1.3):
    lines = get_lines_info(page, alpha=0.0)

    refined_page = np.zeros_like(page)
    cline_id = 1

    heights_std = lines['height'].std()
    heights_mean = lines['height'].mean()
    cut_off = heights_mean * alpha

    print('std', heights_std, 'mean', heights_mean, 'cutoff', cut_off)

    for i, r in tqdm(lines.iterrows(), desc='refining'):
        v = r['value']
        h = r['height']

        if v == 0:
            continue

        min_y = int(r['min_y']) - int(margin * heights_mean)
        max_y = int(r['max_y']) + int(margin * heights_mean)

        curr_line_page = np.zeros_like(page)
        curr_line_page[page == v] = cline_id

        if SHOW_RESULTS:
            disp_curr_line = np.zeros((*curr_line_page[min_y:max_y, :].shape, 3))
            disp_curr_line[curr_line_page[min_y:max_y, :] > 0] = [255, 255, 255]
            cv2.imshow('curr_line', cv2.resize(disp_curr_line, (600, 50)))
            cv2.waitKey(1)

        # print('line std', line_std(curr_line_page[min_y:max_y, :]))

        if h > cut_off:
            split_point = best_split_point(curr_line_page[min_y:max_y, :].copy())

            # print('split point: ', split_point)

            if SHOW_RESULTS:
                disp_curr_line = cv2.line(disp_curr_line, (0, split_point), (disp_curr_line.shape[1], split_point),
                                          (0, 0, 255), 2)
                cv2.imshow('split line', disp_curr_line)
                cv2.waitKey(1)

            split_point += min_y

            curr_line_page[min_y:split_point, :][curr_line_page[min_y:split_point, :] > 0] = cline_id

            cline_id += 1

            curr_line_page[split_point:max_y, :][curr_line_page[split_point:max_y, :] > 0] = cline_id

            cline_id += 1
        else:
            curr_line_page[page == v] = cline_id
            cline_id += 1

        refined_page[curr_line_page > 0] = curr_line_page[curr_line_page > 0].copy()

    return refined_page


def merge_classes(page, curr_line_page, curr_line_value, start_y, end_y, cline_id):
    line_img = page[start_y:end_y, :]
    inter_values = np.unique(line_img)
    merged = []

    curr_line_area = np.sum(np.sum(page == curr_line_value))

    for i_v in inter_values:
        if i_v == 0 or i_v == curr_line_value:
            continue

        in_area = np.sum(np.sum(line_img == i_v))
        whole_area = np.sum(np.sum(page == i_v))
        # print('1', in_area / whole_area)
        # print('2', (whole_area - in_area) / curr_line_area)
        if in_area / whole_area > 0.99 and (whole_area - in_area) / curr_line_area < 0.8:
            # print((whole_area - in_area)/curr_line_area)
            curr_line_page[page == i_v] = cline_id
            merged.append(i_v)
            if SHOW_RESULTS:
                disp_curr_line = np.zeros((*curr_line_page[start_y:end_y, :].shape, 3))
                disp_curr_line[curr_line_page[start_y:end_y, :] > 0] = [255, 255, 255]
                disp_curr_line[line_img == i_v] = [0, 0, 255]
                cv2.imshow('merge classes curr_line', cv2.resize(disp_curr_line, (1500, 100)))
                cv2.waitKey(1)

    return curr_line_page, merged


paths = [#'prediction/blobs/DIVA_CSG18_private_test', 'prediction/blobs/DIVA_CB55_private_test',
         #'prediction/blobs/DIVA_CSG863_private_test', 'prediction/blobs/AHTE_test',
         'prediction/pixel_level/CB55_private_test', 'prediction/pixel_level/CSG18_private_test',
         'prediction/pixel_level/CSG863_private_test', 'prediction/pixel_level/AHTE_test']

# paths = ['prediction/pixel_level/AHTE_test', 'prediction/blobs/AHTE_test']

out_dir = 'new_refined'
# prediction_dir = 'prediction/blobs/DIVA_CB55_private_test'
# prediction_dir = 'prediction/pixel_level/CB55_private_test'

for prediction_dir in paths:
    out_path = os.path.join(out_dir, prediction_dir)
    vis_path = os.path.join(out_path, 'vis')
    os.makedirs(vis_path, exist_ok=True)

    for p in os.listdir(prediction_dir):
        if p.split('.')[-1] != 'png':
            continue
        # p = 'e-codices_csg-0018_076_max.png'
        print(p)
        page = cv2.imread(os.path.join(prediction_dir, p), 0)

        values = np.unique(page)

        print('before: # classes: ', len(values) + 1)

        # refined = refine_text_line_segmentation(page)

        vis_refined = np.zeros((*page.shape, 3), dtype=np.uint8)

        values = np.unique(page)

        print('# classes: ', len(values) + 1)

        for v in values:
            if v == 0:
                continue
            vis_refined[page == v] = (255 * np.random.rand(3)).astype(np.uint8)

        cv2.imshow('before refined', cv2.resize(vis_refined, (600, 800)))
        cv2.waitKey(1)

        refined = merge_lines(page)
        refined = split_lines(refined)

        refined = merge_lines(refined, alpha=0.2)

        vis_refined = np.zeros((*refined.shape, 3), dtype=np.uint8)

        values = np.unique(refined)

        print('# classes: ', len(values) + 1)

        for v in values:
            if v == 0:
                continue
            vis_refined[refined == v] = (255 * np.random.rand(3)).astype(np.uint8)

        cv2.imshow('refined', cv2.resize(vis_refined, (600, 800)))
        cv2.waitKey(1)

        cv2.imwrite(os.path.join(out_path, p), refined)
        cv2.imwrite(os.path.join(vis_path, p), vis_refined)

## archive
#
# def refine_text_line_segmentation(page, margin=0.1, alpha=1.5):
#     lines = get_lines_info(page)
#
#     refined_page = np.zeros_like(page)
#     cline_id = 1
#
#     heights_std = lines['height'].std()
#     heights_mean = lines['height'].mean()
#     cut_off = heights_std * alpha
#
#     print('heights mean', heights_mean, 'cutoff ', cut_off)
#
#     for i, r in tqdm(lines.iterrows(), desc='splitting'):
#         v = r['value']
#         h = r['height']
#
#
#         # print('curr height', h)
#
#         min_y = int(r['min_y']) - int(margin * heights_mean)
#         max_y = int(r['max_y']) + int(margin * heights_mean)
#
#         curr_line_page = np.zeros_like(page)
#         curr_line_page[page == v] = 1
#
#         disp_curr_line = 255 * curr_line_page[min_y:max_y, :]
#         cv2.imshow('curr_line', cv2.resize(disp_curr_line, (600, 50)))
#         cv2.waitKey(1)
#
#         if h > heights_mean + cut_off:
#             split_point = best_split_point(curr_line_page[min_y:max_y, :])
#
#             # print('split point: ', split_point)
#
#             disp_curr_line = cv2.line(disp_curr_line, (0, split_point), (disp_curr_line.shape[1], split_point), 255, 2)
#             cv2.imshow('split line', cv2.resize(disp_curr_line, (600, 50)))
#             cv2.waitKey(0)
#
#             split_point += min_y
#
#             curr_line_page[min_y:split_point, :][curr_line_page[min_y:split_point, :] == 1] = cline_id
#
#             curr_line_page = merge_classes(page, curr_line_page, min_y, split_point, cline_id)
#             cline_id += 1
#
#             curr_line_page[split_point:max_y, :][curr_line_page[split_point:max_y, :] == 1] = cline_id
#
#             curr_line_page = merge_classes(page, curr_line_page, split_point, max_y, cline_id)
#             cline_id += 1
#         else:
#             curr_line_page[curr_line_page == 1] = cline_id
#
#             curr_line_page = merge_classes(page, curr_line_page, min_y, max_y, cline_id)
#
#             cline_id += 1
#
#         refined_page += curr_line_page
#
#     return refined_page
