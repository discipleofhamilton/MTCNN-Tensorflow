# MIT License
#
# Copyright (c) 2017 Baoming Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import argparse
import time
import os
import shutil

import tensorflow as tf
import cv2
import numpy as np

from src.mtcnn import PNet, RNet, ONet
from tools import detect_face, get_model_filenames, detect_face_24net, detect_face_12net

def add_overlays(frame, faces, points, frame_rate, bb_w_scale, bb_h_scale):
    if faces is not None:
        for face in faces:
            face[0] = face[0] * bb_w_scale
            face[1] = face[1] * bb_h_scale
            face[2] = face[2] * bb_w_scale
            face[3] = face[3] * bb_h_scale
            cv2.rectangle(frame,
                          (int(face[0]), int(face[1])), (int(face[2]), int(face[3])),
                          (0, 255, 0), 1)

            cv2.putText(frame, "Person", (int(face[0]), int(face[3])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                        thickness=2, lineType=1)

    if points is not None:
        for point in points:
            for i in range(0, 10, 2):
                point[i]   = point[i] * bb_w_scale
                point[i+1] = point[i+1] * bb_h_scale
                cv2.circle(frame, (int(point[i]), int(point[i + 1])), 2, (0, 255, 0))

    cv2.putText(frame, str(frame_rate) + " fps", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),
                thickness=2, lineType=2)

def main(args):

    detect_totalTime = 0.0
    frameCount = 0

    # Does there need store result images or not
    # If yes, check the directory which store result is existed or not
    # If the directory is existed, delete the directory recursively then recreate the directory.
    if args.save_image:
        output_directory = args.save_image
        print(args.save_image)
        if os.path.exists(output_directory):
            shutil.rmtree(output_directory)
        os.mkdir(output_directory)
        fw = open(os.path.join(output_directory, args.save_bbox_coordinates + '_dets.txt'), 'w')

    # Create 
    # The steps are similiar to "store result images" above.
    if args.save_camera_images is not False:
        source_directory = args.save_camera_images
        if os.path.exists(source_directory):
            shutil.rmtree(source_directory)
        os.mkdir(source_directory)

    with tf.device('/cpu:0'):
        with tf.Graph().as_default():
            config = tf.ConfigProto(allow_soft_placement=True)
            with tf.Session(config=config) as sess:

                file_paths = get_model_filenames(args.model_dir)
                print(file_paths, len(file_paths))

                # The if else statement is to check which type of model user used.
                # if the if condition is true, which means user use separate P-Net, R-Net and O-Net models.
                # In anaconda bash to type the command line which is "python test_camera.py --model_dir model/separate".
                # And there are three folders which are P-Net, R-Net and O-Net in the named separate directory. 
                if len(file_paths) == 3:
                    image_pnet = tf.placeholder(
                        tf.float32, [None, None, None, 3])
                    pnet = PNet({'data': image_pnet}, mode='test')
                    out_tensor_pnet = pnet.get_all_output()

                    image_rnet = tf.placeholder(tf.float32, [None, 24, 24, 3])
                    rnet = RNet({'data': image_rnet}, mode='test')
                    out_tensor_rnet = rnet.get_all_output()

                    image_onet = tf.placeholder(tf.float32, [None, 48, 48, 3])
                    onet = ONet({'data': image_onet}, mode='test')
                    out_tensor_onet = onet.get_all_output()

                    saver_pnet = tf.train.Saver(
                                    [v for v in tf.global_variables()
                                    if v.name[0:5] == "pnet/"])
                    saver_rnet = tf.train.Saver(
                                    [v for v in tf.global_variables()
                                    if v.name[0:5] == "rnet/"])
                    saver_onet = tf.train.Saver(
                                    [v for v in tf.global_variables()
                                    if v.name[0:5] == "onet/"])

                    saver_pnet.restore(sess, file_paths[0])

                    def pnet_fun(img): return sess.run(
                        out_tensor_pnet, feed_dict={image_pnet: img})

                    saver_rnet.restore(sess, file_paths[1])

                    def rnet_fun(img): return sess.run(
                        out_tensor_rnet, feed_dict={image_rnet: img})

                    saver_onet.restore(sess, file_paths[2])

                    def onet_fun(img): return sess.run(
                        out_tensor_onet, feed_dict={image_onet: img})

                else:
                    saver = tf.train.import_meta_graph(file_paths[0])
                    saver.restore(sess, file_paths[1])

                    def pnet_fun(img): return sess.run(
                        ('softmax/Reshape_1:0',
                        'pnet/conv4-2/BiasAdd:0'),
                        feed_dict={
                            'Placeholder:0': img})

                    def rnet_fun(img): return sess.run(
                        ('softmax_1/softmax:0',
                        'rnet/conv5-2/rnet/conv5-2:0'),
                        feed_dict={
                            'Placeholder_1:0': img})

                    def onet_fun(img): return sess.run(
                        ('softmax_2/softmax:0',
                        'onet/conv6-2/onet/conv6-2:0',
                        'onet/conv6-3/onet/conv6-3:0'),
                        feed_dict={
                            'Placeholder_2:0': img})

                video_capture = cv2.VideoCapture(0)
                print(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH), video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

                if video_capture.isOpened() == False:
                    print("ERROR: NO VIDEO STREAM OR NO CAMERA DEVICE.")

                else:

                    print(video_capture.get(cv2.CAP_PROP_FPS))

                    while True:

                        ret, frame = video_capture.read()
                        original_img = frame.copy()

                        if ret:

                            width  = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)*args.resize)
                            height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)*args.resize)
                            resized_image = cv2.resize(frame, (width, height))

                            start_time = time.time()*1000

                            # P-Net + R-Net + O-Net
                            if args.net == "ALL":
                                rectangles, points = detect_face(resized_image, args.minsize,
                                                                pnet_fun, rnet_fun, onet_fun,
                                                                args.threshold, args.factor)

                            # P-Net + R-Net without faces' landmarks
                            elif args.net == "PR":
                                rectangles = detect_face_24net(resized_image, args.minsize, 
                                                                pnet_fun, rnet_fun,
                                                                args.threshold, args.factor)

                            # Only P-Net
                            elif args.net == "P":
                                rectangles = detect_face_12net(resized_image, args.minsize,
                                                                pnet_fun, args.threshold, args.factor)

                            else:
                                print("ERROR: WRONG NET INPUT")

                            end_time = time.time()*1000
                            detect_totalTime = detect_totalTime + (end_time - start_time)

                            if args.net == "ALL":
                                points = np.transpose(points) # The outputs of O-Net which are faces' landmarks
                            else:
                                points = None # the others 

                            add_overlays(frame, rectangles, points, 1000/(end_time - start_time), 1/args.resize, 1/args.resize)
                            cv2.imshow("MTCNN-Tensorflow wangbm", frame)

                            print("ID: {:d}, cost time: {:.1f}ms".format(frameCount, (end_time - start_time))) s

                            if points is not None:
                                for point in points:
                                    for i in range(0, 10, 2):
                                        point[i]   = point[i] * (1/args.resize)
                                        point[i+1] = point[i+1] * (1/args.resize)
                                        print("\tID: {:d}, face landmarks x = {:.1f}, y = {:.1f}".format(int(i/2+1), point[i], point[i+1]))

                            if args.save_image:
                                outputFilePath = os.path.join(output_directory, str(frameCount) + ".jpg")
                                cv2.imwrite(outputFilePath, frame)
                                for rectangle in rectangles:
                                    fw.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(str(frameCount), rectangle[4], rectangle[0], rectangle[1], rectangle[2], rectangle[3]))
                                fw.close()

                            if args.save_camera_images:
                                sourceFilePath = os.path.join(source_directory, str(frameCount) + ".jpg")
                                cv2.imwrite(sourceFilePath, original_img)

                            frameCount = frameCount + 1

                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                cv2.destroyAllWindows()
                                break

                    video_capture.release()
                    detect_average_time = detect_totalTime/frameCount
                    print("*" * 50)
                    print("detection average time: " + str(detect_average_time) + "ms" )
                    print("detection fps: " + str(1000/detect_average_time))

def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str,
                        help='The directory of trained model',
                        default='./save_model/all_in_one/')
    parser.add_argument('--net', type=str, choices=["P", "PR", "ALL"],
                        help='The net of test', default="ALL")
    parser.add_argument(
        '--threshold',
        type=float,
        nargs=3,
        help='Three thresholds for pnet, rnet, onet, respectively.',
        default=[0.8, 0.8, 0.8])
    parser.add_argument('--resize', type=float,
                        help='The resize size of frame to detect.', default=1.0)
    parser.add_argument('--minsize', type=int,
                        help='The minimum size of face to detect.', default=20)
    parser.add_argument('--factor', type=float,
                        help='The scale stride of orginal image', default=0.7)

    parser.add_argument('--save_image', type=str,
                        help='Whether and where to save the result image', default=False)  
    
    parser.add_argument('--save_bbox_coordinates', type=str,
                        help='Whether and where to save coordinates of bouding box', default=False) 

    parser.add_argument('--save_camera_images', type=str,
                        help='Whether and where to save the source images', default=False)               

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
