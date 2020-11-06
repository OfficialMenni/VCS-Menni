import cv2
import imutils
import numpy as np
import skimage.measure as sk
from constants import limit


class Rectification:
    def __init__(self):
        pass

    def src_pts(self, frame):
        """
             Look for 4 points to match for transformation
             param:
                     frame: sliced image
             return: list of 4 points
           """
        frame_copy = frame.copy()
        color = [255, 255, 255]
        frame_copy = cv2.copyMakeBorder(frame_copy, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=color)
        black_img = np.zeros(frame_copy.shape, dtype=np.uint8)
        frame_copy = cv2.medianBlur(frame_copy, 15)
        reshape = frame_copy.reshape((-1, 3))
        reshape = np.float32(reshape)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        k_cluster = 4
        ret, label, center = cv2.kmeans(reshape, k_cluster, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape(frame_copy.shape)
        res2 = cv2.medianBlur(res2, 27)
        canny = cv2.Canny(res2, 50, 100)
        rho = 1
        theta = np.pi / 180
        threshold = 70
        min_line_length = 0
        max_line_gap = 250
        lines = cv2.HoughLinesP(canny, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
        if lines is None:
            return 0, False
        for i in range(0, len(lines)):
            line = lines[i][0]
            if line[0] == line[2]:
                line[2] += 1
            m = (line[3] - line[1]) / (line[2] - line[0])
            if m < -6 or - 0.7 <= m <= 0.7 or m > 6:
                cv2.line(black_img, (line[0], line[1]), (line[2], line[3]), (255, 255, 255), 4, cv2.LINE_AA)
                line = np.reshape(line, (2, 2))
                [vx, vy, x, y] = cv2.fitLine(line, cv2.DIST_L2, 0, 0.01, 0.01)
                left = int((-x * vy / vx) + y)
                right = int(((frame.shape[1] - x) * vy / vx) + y)
                if left > limit:
                    left = limit
                elif left < - limit:
                    left = - limit
                if right > limit:
                    right = limit
                elif right < - limit:
                    right = - limit
                cv2.line(black_img, (frame.shape[1] - 1, right), (0, left), (255, 255, 255), 2, cv2.LINE_AA)
        black_img = cv2.cvtColor(black_img, cv2.COLOR_BGR2GRAY)
        contours = cv2.findContours(black_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[1:5]
        found = False
        pts = np.zeros((4, 1, 2))
        for c in contours:
            peri = cv2.arcLength(c, True)
            pts = cv2.approxPolyDP(c, 0.01 * peri, True)
            if len(pts) == 4 and cv2.isContourConvex(pts) and peri >= 300:
                found = True
                return pts.sum(axis=1), found
        return pts.sum(axis=1), found

    def order_pts(self, pts):
        """
             Sort points from top-left clockwise
             param:
                     pts: list of points
             return: sorted point's list
           """
        ordered_pts = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        ordered_pts[0] = pts[np.argmin(s)]
        ordered_pts[2] = pts[np.argmax(s)]
        dif = np.diff(pts, axis=1)
        ordered_pts[1] = pts[np.argmin(dif)]
        ordered_pts[3] = pts[np.argmax(dif)]
        return ordered_pts

    def perspective_transform(self, frame):
        """
             Warp image into new space
             param:
                     frame: sliced image

             return: warped image, flag
                     flag == True if warping was performed
                     flag == False otherwise
           """
        points, found = self.src_pts(frame)
        if not found:
            return frame, False
        points = self.order_pts(points)
        (tl, tr, br, bl) = points
        width_bottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_bottom), int(width_top))
        height_right = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_left = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        min_height = min(int(height_right), int(height_left))
        dst_pts = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, min_height - 1], [0, min_height - 1]],
                           dtype=np.float32)
        mat = cv2.getPerspectiveTransform(points, dst_pts)  # Find the transformation matrix
        warped = cv2.warpPerspective(frame, mat, (max_width, min_height))  # Rectify the paint
        entropy = sk.shannon_entropy(warped)  # Entropy-based filter to reduce false rectifications
        if entropy > 3:
            return warped, found
        else:
            found = False
            return frame, found
