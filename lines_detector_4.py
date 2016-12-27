#
# Author: Jiri Fajtl
# email: ok1zjf@gmail.com
# Date: 27/12/2016
#
import sys
import time
import numpy as np
import os
import cv2


class RoadLinesDetector:
    def __init__(self):
        self.line_color = (0, 0, 255)
        self.line_thickness = 3
        self.blend_weight = 0.6
        self.show_video = True
        self.debug = True

        self.left_projection_polygon = []
        self.right_projection_polygon = []
        self.left_line_estimate = None
        self.right_line_estimate = None
        self.current_frame = None

        self.wnd_name_img_out = 'detected lines'
        self.wnd_name_img_in = 'input image'

    def init_gui(self):
        cv2.namedWindow(self.wnd_name_img_in, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.wnd_name_img_in, 1200, 800)
        cv2.moveWindow(self.wnd_name_img_in, 0, 0)

        cv2.namedWindow(self.wnd_name_img_out, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.wnd_name_img_out, 1200, 800)
        cv2.moveWindow(self.wnd_name_img_out, 1350, 0)

    def reset(self):
        self.left_line_estimate = None
        self.right_line_estimate = None
        return

    def init_detector(self, video_frame):
        img_height = video_frame.shape[0]
        img_width = video_frame.shape[1]
        self.left_projection_polygon, self.right_projection_polygon = self.get_road_projection_mask(img_width,
                                                                                                    img_height)
        self.left_line_estimate = None
        self.right_line_estimate = None

    def frame_geometry_changed(self, video_frame):
        return (self.current_frame is None or
                video_frame.shape[0] != self.current_frame.shape[0] or
                video_frame.shape[1] != self.current_frame.shape[1])

    def get_road_projection_mask(self, img_width, img_height):
        vertices_left = np.array([[
            (img_width * 0.09, img_height),
            (img_width * 0.42, img_height * 0.62),
            (img_width * 0.51, img_height * 0.62),
            (img_width * 0.34, img_height),
        ]], dtype=np.int32)

        vertices_right = np.array([[
            (img_width * 0.79, img_height),
            (img_width * 0.53, img_height * 0.62),
            (img_width * 0.63, img_height * 0.62),
            (img_width * 1.0, img_height),
        ]], dtype=np.int32)

        return vertices_left, vertices_right

    def threshold_frame(self, video_frame):

        self.current_frame = video_frame

        # Threshold yellow areas
        kernel_size = 5
        blur_clr = cv2.GaussianBlur(video_frame, (kernel_size, kernel_size), 0)
        hsv = cv2.cvtColor(blur_clr, cv2.COLOR_BGR2HSV)

        # Define upper and lower hue level for HSV thresholding
        lower_clr = (20, 100, 100)
        upper_clr = (30, 255, 255)

        mask_yellow = cv2.inRange(hsv, lower_clr, upper_clr)

        kernel = np.ones((7, 3), np.uint8)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)

        # Threshold white areas
        gray = cv2.cvtColor(video_frame, cv2.COLOR_RGB2GRAY)

        kernel_size = 3
        blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 2)

        ret, mask_white = cv2.threshold(blur_gray, 200, 255, cv2.THRESH_BINARY)

        # Combine yellow and white masks
        mask = cv2.bitwise_or(mask_white, mask_yellow)

        if self.debug:
            cv2.imshow('mask_yellow', mask_yellow)
            cv2.imshow('mask_white', mask_white)
            cv2.imshow('mask_lines', mask)

        return mask

    # Creates a gray scale mask image from a given polygon vertices
    def mask_image(self, image, vertices):
        mask = np.zeros_like(image)
        ignore_mask_color = (255)
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image

    def get_mask_image(self, vertices):
        clr_mask = np.zeros_like(self.current_frame)
        cv2.fillPoly(clr_mask, vertices, (255, 0, 0))
        return clr_mask

    # Returns endpoints for a line section begining at the bottom of the image
    # and extending to the image height * 0.61
    # Input is line direction vector (vx,vy) and point on the line x,y
    def extend_line(self, vx, vy, x, y, img_height):
        m = vy / vx
        b = int((-x * m) + y)

        y_offset = int(img_height * 0.61)

        yo = img_height
        xo = (yo - b) / m
        xt = (y_offset - b) / m

        return [(xo, yo), (xt, y_offset)]

    # Fits a line through blobs in the input binary image.
    # Least square error is used to fit the line through the blobs
    # Pixels with values 1 represent points to fit the line through.
    # In order to weight the estimated line to a know prior we render the prior line
    # to the input binary image first. This results in adding points lying on the prior line to the
    # 2D point set generated from the blobs in the input image. The weight the prior line
    # influences the estimated line is controlled by the number of points generated for the prior
    # line. This is achieved but rendering the prior line to the source image with different thickness

    def fit_line(self, image, prior_line):

        prior_line_weight = 1
        if prior_line is not None:
            cv2.line(image, prior_line[0], prior_line[1], 255, prior_line_weight, 4)

        locations = cv2.findNonZero(image)
        [vx, vy, x, y] = cv2.fitLine(locations, cv2.DIST_L2, 0, 0.01, 0.01)

        return self.extend_line(vx, vy, x, y, image.shape[0])

    def render_lines(self, image, left_line, right_line, fps=None):
        lines_image = np.copy(image)
        cv2.line(lines_image, left_line[0], left_line[1], self.line_color, self.line_thickness, cv2.LINE_AA)
        cv2.line(lines_image, right_line[0], right_line[1], self.line_color, self.line_thickness, cv2.LINE_AA)
        image_out = cv2.addWeighted(image, 1 - self.blend_weight, lines_image, self.blend_weight, 0)

        if fps is not None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image_out, "fps: %d" % (int(fps)), (10, 20), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        return image_out

    def detect(self, video_frame):

        if video_frame is None:
            return None

        if self.frame_geometry_changed(video_frame):
            self.init_detector(video_frame)

        binary_img = self.threshold_frame(video_frame)

        masked_left = self.mask_image(binary_img, self.left_projection_polygon)
        masked_right = self.mask_image(binary_img, self.right_projection_polygon)

        self.left_line_estimate = self.fit_line(masked_left, self.left_line_estimate)
        self.right_line_estimate = self.fit_line(masked_right, self.right_line_estimate)

        if self.debug:
            cv2.imshow('masked_edges_left', masked_left)
            cv2.imshow('masked_edges_right', masked_right)

            mask_images_left = self.get_mask_image(self.left_projection_polygon)
            mask_images_right = self.get_mask_image(self.right_projection_polygon)
            masked_image = cv2.bitwise_or(mask_images_left, mask_images_right)

            debug_image = cv2.addWeighted(video_frame, 0.8, masked_image, 1, 0)
            cv2.imshow('masked_image', debug_image)

        return self.left_line_estimate, self.right_line_estimate

    def processImage(self, filename):
        self.reset()
        frame = cv2.imread(filename)
        left_line, right_line = self.detect(frame)
        output_frame = self.render_lines(frame, left_line, right_line)
        return output_frame

    def processImages(self, images_in, path):
        self.init_gui()
        image_id = 0
        num_images = len(images_in)
        while True:

            frame = cv2.imread(path + images_in[image_id])
            print("[" + str(image_id) + "] " + images_in[image_id])

            self.reset()
            left_line, right_line = self.detect(frame)
            output_frame = self.render_lines(frame, left_line, right_line)

            cv2.imshow(self.wnd_name_img_in, frame)
            cv2.imshow(self.wnd_name_img_out, output_frame)

            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

            image_id = (image_id + 1) % num_images

        cv2.destroyAllWindows()

        return

    def processVideo(self, video_in_filename, video_out_filename=None, frame_interval=1):

        if video_out_filename is None:
            path, ext = os.path.splitext(video_in_filename)
            video_out_filename = path + '_annotated' + ext

        cap = cv2.VideoCapture(video_in_filename)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'X264')

        print("Video file: ", video_in_filename)
        print("width: ", width)
        print("height: ", height)
        print("fps: ", fps)

        out = None
        if video_out_filename != "":
            out = cv2.VideoWriter(video_out_filename, fourcc, fps, (width, height))

        if self.show_video:
            self.init_gui()

        frame_id = 0
        self.reset()
        while (cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break

            start = time.time()
            left_line, right_line = self.detect(frame)
            end = time.time()
            fps_now = 1 / (end - start)

            output_frame = self.render_lines(frame, left_line, right_line, fps_now)

            if self.show_video:
                cv2.imshow(self.wnd_name_img_in, frame)
                cv2.imshow(self.wnd_name_img_out, output_frame)

            # write the flipped frame
            if out is not None:
                out.write(output_frame)

            if frame_interval < 0:
                frame_interval = int(1000.0 / fps)

            if cv2.waitKey(frame_interval) & 0xFF == ord('q'):
                break

            sys.stdout.write('Frame: %d   fps (processing): %d\r' % (frame_id, int(fps_now)))
            sys.stdout.flush()

            frame_id += 1

        if out is not None:
            out.release()

        cap.release()
        cv2.destroyAllWindows()

        print("\nDone")

        return


def main():

    data_path = '/home/jiri/Dropbox/Courses/SDC/Term-1/Finding_Lane_Lines/CarND-LaneLines-P1/'

    ld = RoadLinesDetector()
    path = '/home/jiri/Dropbox/Courses/SDC/Term-1/Finding_Lane_Lines/CarND-LaneLines-P1/test_images/'
    images = [
        'yellow_line_left_1.png',
        'yellow_line_left_2.png',
        'yellow_line_left_3.png',
        'yellow_line_left_4.png',
        'solidWhiteCurve.jpg',
        'solidWhiteRight.jpg',
        'solidYellowCurve.jpg',
        'solidYellowCurve2.jpg',
        'solidYellowLeft.jpg',
        'whiteCarLaneSwitch.jpg'
    ]

    ld.processImages(images, path)

    frame_interval = 1
    video_in_filename = data_path+'challenge.mp4'
    video_out_filename = "" # Empty string means do not write out video/image file
    video_out_filename = None
    ld.processVideo(video_in_filename, video_out_filename, frame_interval)

    video_in_filename = data_path+'solidWhiteRight.mp4'
    ld.processVideo(video_in_filename, video_out_filename, frame_interval)

    video_in_filename = data_path + 'solidYellowLeft.mp4'
    ld.processVideo(video_in_filename, video_out_filename, frame_interval)

    return


if __name__ == "__main__":
    main()