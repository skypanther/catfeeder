import colorsys
import cv2
import numpy as np
from sklearn.cluster import KMeans


class ColorAnalyzer:
    IMG_SIZE = 320
    kernel_open = np.ones((5, 5))  # for drawing the "open" mask
    kernel_close = np.ones((20, 20))  # for drawing the "closed" mask

    def __init__(self, min_area):
        self.MIN_ORANGE_AREA = min_area

    def set_color_range(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def is_right_cat_visible(self, img, show_windows=False):
        resized_image = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
        img_hsv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
        # create the Mask
        mask = cv2.inRange(
            img_hsv,
            self.lower_bound,
            self.upper_bound)
        # morphology
        mask_open = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            self.kernel_open)
        mask_close = cv2.morphologyEx(
            mask_open,
            cv2.MORPH_CLOSE,
            self.kernel_close)

        mask_final = mask_close
        _, conts, h = cv2.findContours(
            mask_final.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE)

        if show_windows:
            cv2.drawContours(resized_image, conts, -1, (255, 0, 0), 3)
            cv2.imshow("Detected Color Region", resized_image)

        if len(conts):
            for i in range(len(conts)):
                if cv2.contourArea(conts[i]) > self.MIN_ORANGE_AREA:
                    return True
        return False

    def get_dominant_colors(self, img, num_colors=5):
        """
        Find the n dominant colors of a given image
        :param img: OpenCV BGR image
        :param num_colors: How many color "buckets" to find
        :return: 2-Tuple of RGB (not BGR!) and HSV values for the buckets
        """
        height, width, _ = np.shape(img)

        # reshape the image to be a simple list of RGB pixels
        image = img.reshape((height * width, 3))

        # find the 'num_colors' most common colors
        clusters = KMeans(n_clusters=num_colors)
        clusters.fit(image)
        histogram = self._make_histogram(clusters)
        # then sort them, most-common first
        combined = zip(histogram, clusters.cluster_centers_)
        combined = sorted(combined, key=lambda x: x[0], reverse=True)

        rgb_values = []
        hsv_values = []
        for index, rows in enumerate(combined):
            r, g, b = rows[1]
            # rgb values returned as a tuple of integers
            rgb_values.append((int(round(r)), int(round(g)), int(round(b))))
            # need to pass in colors in 0 - 1 range
            h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
            # hsv values are: hue 0-179, saturation & value 0 - 255, as integers
            hsv_values.append((int(round(h * 179)), int(round(s * 255)), int(round(v * 255))))
        return rgb_values, hsv_values

    @staticmethod
    def _make_histogram(cluster):
        """
        Count the number of pixels in each cluster
        :param: KMeans cluster
        :return: numpy histogram
        """
        numLabels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
        hist, _ = np.histogram(cluster.labels_, bins=numLabels)
        hist = hist.astype('float32')
        hist /= hist.sum()
        return hist
