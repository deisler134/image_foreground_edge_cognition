import math
import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np

from data_generator_test import generate_trimap, random_choice, get_alpha_test, generate_trimap_withmask
from model import build_encoder_decoder, build_refinement
from utils import compute_mse_loss, compute_sad_loss
from utils import get_final_output, safe_crop, draw_str


import time

import glob
import os
import imutils

test_images_flag = False

def composite4(fg, bg, a, w, h):
    fg = np.array(fg, np.float32)
    bg_h, bg_w = bg.shape[:2]
    x = 0
    if bg_w > w:
        x = np.random.randint(0, bg_w - w)
    y = 0
    if bg_h > h:
        y = np.random.randint(0, bg_h - h)
    bg = np.array(bg[y:y + h, x:x + w], np.float32)
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a / 255.
    im = alpha * fg + (1 - alpha) * bg
    im = im.astype(np.uint8)
    return im, bg

def composite_alpha(fg,a):
    fg = np.array(fg, np.float32)
    h,w = fg.shape[:2]
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a / 255.
    im = alpha * fg 
    im = im.astype(np.uint8)
    return im

def video_save(videopath, w = 1920, h = 1080, is_rotate = False):
    
#     cam = cv.VideoCapture(videopath)
#     save_path = videopath[:-4] + '_save'  + videopath[-4:]
#     print(save_path)
#                 
    fourcc = cv.VideoWriter_fourcc(*'DIVX')
        
#     width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
#     height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
#     
#     if width > w:
#         width = w
#     if height > h:
#         height = h
        
    if is_rotate:
        saver = cv.VideoWriter(videopath, fourcc, 20, (h, w))
    else:
        saver = cv.VideoWriter(videopath, fourcc, 20, (w, h))
        
    return saver
                
# first use Maskrcnn generator mask video
def video_test(videopath,models_path,videomask_path, is_rotate = False):
    cam = cv.VideoCapture(videopath)
    width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    cam_mask = cv.VideoCapture(videomask_path)
    width_mask = int(cam_mask.get(cv.CAP_PROP_FRAME_WIDTH))
    height_mask = int(cam_mask.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    pretrained_path = models_path       #'models/final.87-0.0372.hdf5'         #'models/final.42-0.0398.hdf5'
    encoder_decoder = build_encoder_decoder()
    final = build_refinement(encoder_decoder)
    final.load_weights(pretrained_path)
    print(final.summary())
    
    tri_videopath = videopath[:-4] + '_tri.mp4'
    tri_video = video_save(tri_videopath, w = width_mask, h = height_mask)
    
    matting_videopath = videopath[:-4] + '_out.mp4'
    matting_video = video_save(matting_videopath, w = width_mask, h = height_mask)
    
    comp_videopath = videopath[:-4] + '_comp.mp4'
    comp_video = video_save(comp_videopath, w = width_mask, h = height_mask)
    
    while(cam.isOpened() and cam_mask.isOpened()):
        start_time = time.time()
        ret, frame = cam.read()
        ret, frame_mask = cam_mask.read()

        if is_rotate :
            frame = imutils.rotate_bound(frame, 90)
            frame_mask = imutils.rotate_bound(frame_mask, 90)
        #             print(frame.shape)
        if frame is None:
            print ('Error image!')
            break
        if frame_mask is None:
            print ('Error mask image!')
            break
                
        bg_h, bg_w = height, width
        print('bg_h, bg_w: ' + str((bg_h, bg_w)))
    
    #         a = get_alpha_test(image_name)
        a = cv.cvtColor(frame_mask, cv.COLOR_BGR2GRAY)
        _,a = cv.threshold(a,240, 255,cv.THRESH_BINARY)
        
        a_h, a_w = height_mask, width_mask
        print('a_h, a_w: ' + str((a_h, a_w)))
    
        alpha = np.zeros((bg_h, bg_w), np.float32)

        alpha[0:a_h, 0:a_w] = a
        trimap = generate_trimap_withmask(alpha)
#         fg = np.array(np.greater_equal(a, 255).astype(np.float32))
#         cv.imshow('test_show',fg)
        different_sizes = [(320, 320), (320, 320), (320, 320), (480, 480), (640, 640)]
        crop_size = random.choice(different_sizes)

    
        bgr_img = frame
        alpha = alpha
        trimap = trimap
    #         cv.imwrite('images/{}_image.png'.format(i), np.array(bgr_img).astype(np.uint8))
    #         cv.imwrite('images/{}_trimap.png'.format(i), np.array(trimap).astype(np.uint8))
    #         cv.imwrite('images/{}_alpha.png'.format(i), np.array(alpha).astype(np.uint8))

    
    #         x_test = np.empty((1, img_rows, img_cols, 4), dtype=np.float32)
    #         x_test[0, :, :, 0:3] = bgr_img / 255.
    #         x_test[0, :, :, 3] = trimap / 255.
        
        x_test = np.empty((1, 320, 320, 4), dtype=np.float32)
        bgr_img1 = cv.resize(bgr_img, (320,320))
        trimap1 = cv.resize(trimap,(320,320))
        x_test[0, :, :, 0:3] = bgr_img1 / 255.
        x_test[0, :, :, 3] = trimap1 / 255.
        
            
    
    
        y_true = np.empty((1, img_rows, img_cols, 2), dtype=np.float32)
        y_true[0, :, :, 0] = alpha / 255.
        y_true[0, :, :, 1] = trimap / 255.
    
    
        y_pred = final.predict(x_test)
        # print('y_pred.shape: ' + str(y_pred.shape))

#         y_pred = np.reshape(y_pred, (img_rows, img_cols))
        y_pred = np.reshape(y_pred, (320, 320))
        print(y_pred.shape)
        y_pred = cv.resize(y_pred,(width,height))
        y_pred = y_pred * 255.0
        cv.imshow('pred',y_pred)
        y_pred = get_final_output(y_pred, trimap)
        
        y_pred = y_pred.astype(np.uint8)
            
            
    
        sad_loss = compute_sad_loss(y_pred, alpha, trimap)
        mse_loss = compute_mse_loss(y_pred, alpha, trimap)
        str_msg = 'sad_loss: %.4f, mse_loss: %.4f, crop_size: %s' % (sad_loss, mse_loss, str(crop_size))
        print(str_msg)
    
        out = y_pred.copy()
        comp = composite_alpha(frame, out)
        draw_str(out, (10, 20), str_msg)
        
        
        trimap_show = np.stack((trimap,trimap,trimap),-1)
        out_show = cv.merge((out, out, out))
#         print(trimap_show.shape,out_show.shape,comp.shape)
        tri_video.write(trimap_show)
        matting_video.write(out_show)
        comp_video.write(comp)
#         cv.imwrite('images/{}_out.png'.format(filename[6:]), out)

#### composite background    
#             sample_bg = sample_bgs[i]
#             bg = cv.imread(os.path.join(bg_test, sample_bg))
#             bh, bw = bg.shape[:2]
#             wratio = img_cols / bw
#             hratio = img_rows / bh
#             ratio = wratio if wratio > hratio else hratio
#             if ratio > 1:
#                 bg = cv.resize(src=bg, dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)), interpolation=cv.INTER_CUBIC)
#     #         im, bg = composite4(bgr_img, bg, y_pred, img_cols, img_rows)
#             im, bg = composite4(bgr_img, bg, y_pred, img_cols, img_rows)
#     #         cv.imwrite('images/{}_compose.png'.format(filename[6:]), im)
#     #         cv.imwrite('images/{}_new_bg.png'.format(i), bg)


        print("Time: {:.2f} s / img".format(time.time() - start_time))
                 
        cv.imshow('out',out)
        cv.imshow('frame',frame)
        cv.imshow('comp',comp)
        cv.imshow('trimap',trimap)
                    
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cam_mask.release()
    tri_video.release()
    matting_video.release()
    comp_video.release()

    cv.destroyAllWindows()

if __name__ == '__main__':
    img_rows, img_cols = 1080,1920          #320, 320
    channel = 4

  
    
    if test_images_flag:
        
        pretrained_path = 'models/final.87-0.0372.hdf5'         #'models/final.42-0.0398.hdf5'
        encoder_decoder = build_encoder_decoder()
        final = build_refinement(encoder_decoder)
        final.load_weights(pretrained_path)
        print(final.summary())

        out_test_path = 'data/merged_test/'
        test_images = [f for f in os.listdir(out_test_path) if
                       os.path.isfile(os.path.join(out_test_path, f)) and f.endswith('.png')]
    #     samples = random.sample(test_images, 100)
        samples = test_images
    
        bg_test = 'data/bg_test/'
        test_bgs = [f for f in os.listdir(bg_test) if
                    os.path.isfile(os.path.join(bg_test, f)) and f.endswith('.png')]
    #     sample_bgs = random.sample(test_bgs, 100)
        sample_bgs = np.random.choice(test_bgs, len(samples))
        
        total_loss = 0.0
        for i in range(len(samples)):
            filename = samples[i]
            image_name = filename.split('.')[0]
    
            print('\nStart processing image: {}'.format(filename))
    
            bgr_img = cv.imread(os.path.join(out_test_path, filename))
            bg_h, bg_w = bgr_img.shape[:2]
            print('bg_h, bg_w: ' + str((bg_h, bg_w)))
            print('image_name:',image_name)
    
    #         a = get_alpha_test(image_name)
            a = get_alpha_test(filename)
            a_h, a_w = a.shape[:2]
            print('a_h, a_w: ' + str((a_h, a_w)))
    
            alpha = np.zeros((bg_h, bg_w), np.float32)
            alpha[0:a_h, 0:a_w] = a
            trimap = generate_trimap_withmask(alpha)
            different_sizes = [(320, 320), (320, 320), (320, 320), (480, 480), (640, 640)]
            crop_size = random.choice(different_sizes)
    #         x, y = random_choice(trimap, crop_size)
    #         print('x, y: ' + str((x, y)))
    
    #         bgr_img = safe_crop(bgr_img, x, y, crop_size)
    #         alpha = safe_crop(alpha, x, y, crop_size)
    #         trimap = safe_crop(trimap, x, y, crop_size)
    
            bgr_img = bgr_img
            alpha = alpha
            trimap = trimap
    #         cv.imwrite('images/{}_image.png'.format(i), np.array(bgr_img).astype(np.uint8))
    #         cv.imwrite('images/{}_trimap.png'.format(i), np.array(trimap).astype(np.uint8))
    #         cv.imwrite('images/{}_alpha.png'.format(i), np.array(alpha).astype(np.uint8))
            cv.imwrite('images/{}_image.png'.format(filename[6:]), np.array(bgr_img).astype(np.uint8))
            cv.imwrite('images/{}_trimap.png'.format(filename[6:]), np.array(trimap).astype(np.uint8))
            cv.imwrite('images/{}_alpha.png'.format(filename[6:]), np.array(alpha).astype(np.uint8))
    
    #         x_test = np.empty((1, img_rows, img_cols, 4), dtype=np.float32)
    #         x_test[0, :, :, 0:3] = bgr_img / 255.
    #         x_test[0, :, :, 3] = trimap / 255.
            
            x_test = np.empty((1, 320, 320, 4), dtype=np.float32)
            bgr_img1 = cv.resize(bgr_img, (320,320))
            trimap1 = cv.resize(trimap,(320,320))
            x_test[0, :, :, 0:3] = bgr_img1 / 255.
            x_test[0, :, :, 3] = trimap1 / 255.
            
    
    
            y_true = np.empty((1, img_rows, img_cols, 2), dtype=np.float32)
            y_true[0, :, :, 0] = alpha / 255.
            y_true[0, :, :, 1] = trimap / 255.
    
    
            y_pred = final.predict(x_test)
            # print('y_pred.shape: ' + str(y_pred.shape))
    
    #         y_pred = np.reshape(y_pred, (img_rows, img_cols))
            y_pred = np.reshape(y_pred, (320, 320))
            print(y_pred.shape)
            y_pred = cv.resize(y_pred,(img_cols,img_rows))
            y_pred = y_pred * 255.0
            y_pred = get_final_output(y_pred, trimap)
            y_pred = y_pred.astype(np.uint8)
            
            
    
            sad_loss = compute_sad_loss(y_pred, alpha, trimap)
            mse_loss = compute_mse_loss(y_pred, alpha, trimap)
            str_msg = 'sad_loss: %.4f, mse_loss: %.4f, crop_size: %s' % (sad_loss, mse_loss, str(crop_size))
            print(str_msg)
    
            out = y_pred.copy()
            draw_str(out, (10, 20), str_msg)
            cv.imwrite('images/{}_out.png'.format(filename[6:]), out)
    
            sample_bg = sample_bgs[i]
            bg = cv.imread(os.path.join(bg_test, sample_bg))
            bh, bw = bg.shape[:2]
            wratio = img_cols / bw
            hratio = img_rows / bh
            ratio = wratio if wratio > hratio else hratio
            if ratio > 1:
                bg = cv.resize(src=bg, dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)), interpolation=cv.INTER_CUBIC)
    #         im, bg = composite4(bgr_img, bg, y_pred, img_cols, img_rows)
            im, bg = composite4(bgr_img, bg, y_pred, img_cols, img_rows)
    #         cv.imwrite('images/{}_compose.png'.format(filename[6:]), im)
    #         cv.imwrite('images/{}_new_bg.png'.format(i), bg)
    
        K.clear_session()
    
    else:
        videopath = "/media/deisler/Data/project/coco/cocodata/Deep-Image-Matting/data/Banchmark - Input - 1920x1080.mp4"
        videomask_path = '/media/deisler/Data/project/coco/cocodata/Deep-Image-Matting/data/Banchmark - Input - 1920x1080_debug_save.mp4'
        models_path = 'models/final.24-0.0181(2nd).hdf5'
        video_test(videopath,models_path,videomask_path)
        
        K.clear_session()
