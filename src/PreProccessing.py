import cv2
import os
import numpy as np
import pytesseract
from PIL import Image
from fuzzywuzzy import fuzz
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import math
from pylsd import lsd
import argparse
import itertools


class imutils:

    def translate(image, x, y):
        """Define the translation matrix and perform the translation"""
        M = np.float([[1, 0, x], [0, 1, y]])
        shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

        """Return the translated image"""
        return shifted

    def rotate(image, angle, center=None, scale=1.0):
        """Grab the dimensions of the image"""
        (h, w) = image.shape[:2]

        """If the center is None, initialize it as the center of the image"""
        if center is None:
            center = (w / 2, h / 2)

        """Perform the rotation"""
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))

        """Return the rotated image"""
        return rotated

    def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
        """initialize the dimensions of the image to be resized and grab the image size"""
        (h, w) = image.shape[:2]

        """if both the width and height are None, then  return the original image"""
        if width is None and height is None:
            return image

        """check to see if the width is None"""
        if width is None:
            """calculate the ratio of the height and construct the dimensions"""
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            """calculate the ratio of the width and construct the dimensions"""
            r = width / float(w)
            dim = (width, int(h * r))

        """resize the image"""
        resized = cv2.resize(image, dim, interpolation=inter)

        """return the resized image"""
        return resized


class transform:
    def order_points(pts):
        """sort the points based on their x-coordinates"""
        xSorted = pts[np.argsort(pts[:, 0]), :]

        """grab the left-most and right-most points from the sorted x-coordinate points"""
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        """now, sort the left-most coordinates according to their y-coordinates so we can grab the top-left and
        bottom-left points, respectively"""
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        """calculate the Euclidean distance between the top-left and right-most points; by the Pythagorean theorem"""
        D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
        (br, tr) = rightMost[np.argsort(D)[::-1], :]

        """return the coordinates in top-left, top-right, bottom-right and bottom-left order"""
        return np.array([tl, tr, br, bl], dtype="float32")

    def four_point_transform(image, pts):
        """obtain a consistent order of the points and unpack them individually"""
        rect = transform.order_points(pts)
        (tl, tr, br, bl) = rect

        """compute the width of the new image"""
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        """compute the height of the new image"""
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        """construct the set of destination points to obtain a "birds eye view" """
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        """compute the perspective transform matrix and then apply it"""
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        """return the warped image"""
        return warped


class PolygonInteractor(object):
    """
    An polygon editor
    """

    showverts = True
    epsilon = 5  # max pixel distance to count as a vertex hit

    def __init__(self, ax, poly):
        if poly.figure is None:
            raise RuntimeError('You must first add the polygon to a figure or canvas before defining the interactor')
        self.ax = ax
        canvas = poly.figure.canvas
        self.poly = poly

        x, y = zip(*self.poly.xy)
        self.line = Line2D(x, y, marker='o', markerfacecolor='r', animated=True)
        self.ax.add_line(self.line)

        self._ind = None  # the active vert

        canvas.mpl_connect('draw_event', self.draw_callback)
        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('button_release_event', self.button_release_callback)
        canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
        self.canvas = canvas

    def get_poly_points(self):
        return np.asarray(self.poly.xy)

    def draw_callback(self):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)

    def poly_changed(self, poly):
        """this method is called whenever the polygon object is called"""
        # only copy the artist props to the line (except visibility)
        vis = self.line.get_visible()
        Artist.update_from(self.line, poly)
        self.line.set_visible(vis)  # don't use the poly visibility state

    def get_ind_under_point(self, event):
        """get the index of the vertex under point if within epsilon tolerance"""

        # display coords
        xy = np.asarray(self.poly.xy)
        xyt = self.poly.get_transform().transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.sqrt((xt - event.x) ** 2 + (yt - event.y) ** 2)
        indseq = np.nonzero(np.equal(d, np.amin(d)))[0]
        ind = indseq[0]

        if d[ind] >= self.epsilon:
            ind = None

        return ind

    def button_press_callback(self, event):
        """whenever a mouse button is pressed"""
        if not self.showverts:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self._ind = self.get_ind_under_point(event)

    def button_release_callback(self, event):
        """whenever a mouse button is released"""
        if not self.showverts:
            return
        if event.button != 1:
            return
        self._ind = None

    def motion_notify_callback(self, event):
        """on mouse movement"""
        if not self.showverts:
            return
        if self._ind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        x, y = event.xdata, event.ydata

        self.poly.xy[self._ind] = x, y
        if self._ind == 0:
            self.poly.xy[-1] = x, y
        elif self._ind == len(self.poly.xy) - 1:
            self.poly.xy[0] = x, y
        self.line.set_data(zip(*self.poly.xy))

        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)


class Scanner(object):
    def __init__(self, interactive=False, MIN_QUAD_AREA_RATIO=0.25, MAX_QUAD_ANGLE_RANGE=40):
        self.interactive = interactive
        self.MIN_QUAD_AREA_RATIO = MIN_QUAD_AREA_RATIO
        self.MAX_QUAD_ANGLE_RANGE = MAX_QUAD_ANGLE_RANGE

    def filter_corners(self, corners, min_dist=20):
        def predicate(respresentatives, corner):
            return all(dist.euclidean(respresentative, corner) >= min_dist
                       for respresentative in respresentatives)

        filtered_corners = []
        for c in corners:
            if predicate(filtered_corners, c):
                filtered_corners.append(c)
        return filtered_corners

    def angle_between_vectors_degrees(self, u, v):
        return np.degrees(
            math.acos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
        )

    def get_angle(self, p1, p2, p3):
        a = np.radians(np.array(p1))
        b = np.radians(np.array(p2))
        c = np.radians(np.array(p3))

        avec = a - b
        cvec = c - b

        return self.angle_between_vectors_degrees(avec, cvec)

    def angle_range(self, quad):

        tl, tr, br, bl = quad
        ura = self.get_angle(tl[0], tr[0], br[0])
        ula = self.get_angle(bl[0], tl[0], tr[0])
        lra = self.get_angle(tr[0], br[0], bl[0])
        lla = self.get_angle(br[0], bl[0], tl[0])

        angles = [ura, ula, lra, lla]

        return np.ptp(angles)

    def get_corners(self, img):

        lines = lsd(img)

        corners = []
        if lines is not None:
            """separate out the horizontal and vertical lines, and draw them back onto separate canvases"""
            lines = lines.squeeze().astype(np.int32).tolist()
            horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            for line in lines:
                x1, y1, x2, y2, nan = line
                if abs(x2 - x1) > abs(y2 - y1):
                    (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[0])
                    cv2.line(horizontal_lines_canvas, (max(x1 - 5, 0), y1), (min(x2 + 5, img.shape[1] - 1), y2), 255, 2)
                else:
                    (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[1])
                    cv2.line(vertical_lines_canvas, (x1, max(y1 - 5, 0)), (x2, min(y2 + 5, img.shape[0] - 1)), 255, 2)

            lines = []

            """find the horizontal lines (connected-components -> bounding boxes -> final lines)"""
            contours, _ = cv2.findContours(horizontal_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
            horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            for contour in contours:
                contour = contour.reshape((contour.shape[0], contour.shape[2]))
                min_x = np.amin(contour[:, 0], axis=0) + 2
                max_x = np.amax(contour[:, 0], axis=0) - 2
                left_y = int(np.average(contour[contour[:, 0] == min_x][:, 1]))
                right_y = int(np.average(contour[contour[:, 0] == max_x][:, 1]))
                lines.append((min_x, left_y, max_x, right_y))
                cv2.line(horizontal_lines_canvas, (min_x, left_y), (max_x, right_y), 1, 1)
                corners.append((min_x, left_y))
                corners.append((max_x, right_y))

            """find the vertical lines (connected-components -> bounding boxes -> final lines)"""
            contours, _ = cv2.findContours(vertical_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
            vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            for contour in contours:
                contour = contour.reshape((contour.shape[0], contour.shape[2]))
                min_y = np.amin(contour[:, 1], axis=0) + 2
                max_y = np.amax(contour[:, 1], axis=0) - 2
                top_x = int(np.average(contour[contour[:, 1] == min_y][:, 0]))
                bottom_x = int(np.average(contour[contour[:, 1] == max_y][:, 0]))
                lines.append((top_x, min_y, bottom_x, max_y))
                cv2.line(vertical_lines_canvas, (top_x, min_y), (bottom_x, max_y), 1, 1)
                corners.append((top_x, min_y))
                corners.append((bottom_x, max_y))

            """find the corners"""
            corners_y, corners_x = np.where(horizontal_lines_canvas + vertical_lines_canvas == 2)
            corners += zip(corners_x, corners_y)

        """remove corners in close proximity"""
        corners = self.filter_corners(corners)
        return corners

    def is_valid_contour(self, cnt, IM_WIDTH, IM_HEIGHT):

        return (len(cnt) == 4 and cv2.contourArea(cnt) > IM_WIDTH * IM_HEIGHT * self.MIN_QUAD_AREA_RATIO
                and self.angle_range(cnt) < self.MAX_QUAD_ANGLE_RANGE)

    def get_contour(self, rescaled_image):

        """these constants are carefully chosen"""
        MORPH = 9
        CANNY = 84

        IM_HEIGHT, IM_WIDTH, _ = rescaled_image.shape

        """convert the image to grayscale and blur it slightly"""
        gray = cv2.cvtColor(rescaled_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        """dilate helps to remove potential holes between edge segments"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH, MORPH))
        dilated = cv2.dilate(gray, kernel)

        edged = cv2.Canny(dilated, 0, CANNY)
        test_corners = self.get_corners(edged)

        approx_contours = []

        if len(test_corners) >= 4:
            quads = []

            for quad in itertools.combinations(test_corners, 4):
                points = np.array(quad)
                points = transform.order_points(points)
                points = np.array([[p] for p in points], dtype="int32")
                quads.append(points)

            """get top five quadrilaterals by area"""

            """sort candidate quadrilaterals by their angle range, which helps remove outliers"""
            quads = sorted(quads, key=self.angle_range)

            approx = quads[0]
            if self.is_valid_contour(approx, IM_WIDTH, IM_HEIGHT):
                approx_contours.append(approx)

            cv2.drawContours(rescaled_image, [approx], -1, (20, 20, 255), 2)

        (cnts, hierarchy) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        for c in cnts:
            approx = cv2.approxPolyDP(c, 80, True)
            if self.is_valid_contour(approx, IM_WIDTH, IM_HEIGHT):
                approx_contours.append(approx)
                break

        if not approx_contours:
            TOP_RIGHT = (IM_WIDTH, 0)
            BOTTOM_RIGHT = (IM_WIDTH, IM_HEIGHT)
            BOTTOM_LEFT = (0, IM_HEIGHT)
            TOP_LEFT = (0, 0)
            screenCnt = np.array([[TOP_RIGHT], [BOTTOM_RIGHT], [BOTTOM_LEFT], [TOP_LEFT]])

        else:
            screenCnt = max(approx_contours, key=cv2.contourArea)
        return screenCnt.reshape(4, 2)

    def interactive_get_contour(self, screenCnt, rescaled_image):
        poly = Polygon(screenCnt, animated=True, fill=False, color="yellow", linewidth=5)
        fig, ax = plt.subplots()
        ax.add_patch(poly)
        ax.set_title(("Drag the corners of the box to the corners of the document. \n"
                      "Close the window when finished."))
        p = PolygonInteractor(ax, poly)

        new_points = p.get_poly_points()[:4]
        new_points = np.array([[p] for p in new_points], dtype="int32")
        return new_points.reshape(4, 2)

    def scan(self, image_path):
        RESCALED_HEIGHT = 500.0

        image = cv2.imread(image_path)

        assert (image is not None)

        ratio = image.shape[0] / RESCALED_HEIGHT
        orig = image.copy()
        rescaled_image = imutils.resize(image, height=int(RESCALED_HEIGHT))

        screenCnt = self.get_contour(rescaled_image)

        if self.interactive:
            screenCnt = self.interactive_get_contour(screenCnt, rescaled_image)

        wraped = transform.four_point_transform(orig, screenCnt * ratio)

        gray = cv2.cvtColor(wraped, cv2.COLOR_BGR2GRAY)

        sharpen = cv2.GaussianBlur(gray, (3, 3), 3)
        sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)

        thresh = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)

        basename = os.path.basename(image_path)
        cv2.imwrite('scanned.jpg', thresh)
        print("Processed " + basename)


def main(firebase):
    scanner = Scanner()
    scanner.scan('../Test Source/{0}'.format(input("Enter Your Image:   ")))
    if firebase == "1":
        """Input image"""
        img = cv2.imread('scanned.jpg')

        img = cv2.resize(img, (640, 960))

        """Convert image into gray scale mode"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        """Apply binary + OTSU threshold methods"""
        ret, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        """Apply GaussianBlur for the image"""
        blur = cv2.GaussianBlur(threshold, (3, 3), 0)

        r = cv2.selectROI(blur, cv2.WINDOW_FULLSCREEN)
        crop = blur[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

        cv2.imwrite('../data/toNN.png', crop)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
