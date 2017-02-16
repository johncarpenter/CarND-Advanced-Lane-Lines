import numpy as np
import cv2
import pickle
import kalman as Kalman
import math

class ImageProcessor:
    def __init__(self, camera = None, use_smoothing = True):
        self.camera = camera
        self.use_smoothing = use_smoothing

        self.mtx = None
        self.dist = None
        self.__load_camera(camera)

    def process_image(self,image, left_line, right_line):
        raw_image = image.copy()

        image = self.__apply_camera(image)

        # Generate HLS Images
        s = self.hls_select(image,thresh=(100, 220))

        # Sobel Images
        ksize=15
        gradx = self.abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(25,110))
        grady = self.abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(25,255))

        # Magnitude Images
        mag_binary = self.mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 512))

        # Direction Images
        dir_binary = self.dir_threshold(image, sobel_kernel=ksize, thresh=(0.7,1.2))

        # Merge the images. Based on lots of trial and error
        combined = np.zeros_like(dir_binary)
        combined[((s == 1))|((dir_binary == 1)&(mag_binary == 1))] = 1

        # Apply Perspective Shift
        warp = self.warp_image(combined)
        warp = self.remove_edge_pixels(warp,thresh=100)

        # Locate the lane markers
        left_line,right_line, out_img  = self.histogram_find_lane(warp, left_line, right_line,render_out = True)

        left = left_line.current_fit
        left_error = left_line.current_fit_error

        right = right_line.current_fit
        right_error = right_line.current_fit_error

        if(self.use_smoothing):
            xl,Pl,rl = left_line.predict(left,left_error)
            residual_l = rl.T*rl

            if(residual_l > 40000):
                left_line.reset()
            else:
                left_line.apply_update(xl,Pl,rl)

            xr,Pr,rr = right_line.predict(right,right_error)
            residual_r = rr.T*rr

            if(residual_r > 40000):
                right_line.reset()
            else:
                right_line.apply_update(xr,Pr,rr)

        overlay = self.draw_lane_poly(warp,left_line,right_line)

        # Remap to drive perspective
        overlay = self.unwarp_image(overlay)

        # Add the overlay to the original image
        final = cv2.addWeighted(raw_image, 1, overlay, 0.3, 0)

        # draw road info
        final = self.draw_road_info(final, left_line, right_line)

        # Add processing overlay
        final = self.draw_processing_inlay(final, out_img, left_line, right_line)

        return final,left_line,right_line


    def draw_processing_inlay(self,final, out_img, left_line, right_line):
        # Add the processing window. Shows Histogram fitting errors
        font = cv2.FONT_HERSHEY_SIMPLEX
        fit_errors = "L: {:0.4f}  R: {:0.4f}".format(np.sum(left_line.current_fit_error**2),np.sum(right_line.current_fit_error**2))
        out_img = cv2.putText(out_img,fit_errors,(10,100), font, 1,(255,255,255),2)
        out_img = cv2.resize(out_img,(320,240))
        x_offset = final.shape[1]- 400
        y_offset = 25
        final[y_offset:y_offset+out_img.shape[0], x_offset:x_offset+out_img.shape[1]] = out_img
        return final

    def draw_road_info(self,img, left_line, right_line):
        # Overlay the Curvature on the images
        l = left_line.curvature()
        r = right_line.curvature()
        display_text = "L Curve: {:.0f}m  R Curve: {:.0f}m".format(l,r)
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = cv2.putText(img,display_text,(10,100), font, 0.7,(255,255,255),2)

        offset = self.calc_center_offset(left_line,right_line)
        left_or_right = 'left' if offset < 0 else 'right'
        offset_text = "Departure: {:.1f}m {} of centre".format(np.abs(offset),left_or_right)
        img = cv2.putText(img,offset_text,(10,140), font, 0.7,(255,255,255),2)
        return img



    def draw_lane_poly(self,img, left_line, right_line, color = (0,255,0)):

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(img).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
        #left_fitx = left[0]*ploty**2 + left[1]*ploty + left[2]
        #right_fitx = right[0]*ploty**2 + right[1]*ploty + right[2]
        left = left_line.get_as_poly()
        left_fitx = left(ploty)
        right = right_line.get_as_poly()
        right_fitx = right(ploty)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), color)
        return color_warp

    def calc_center_offset(self,left_line, right_line):
        center_poly = (left_line.get_as_poly() + right_line.get_as_poly()) / 2
        offset = (720 / 2 - center_poly(719)) * 3.7 / 700
        return offset


    def warp_image(self,img):
        img_size = (img.shape[1], img.shape[0])
        src,dst = self.get_transform_parameters(img)

        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

        return warped

    def unwarp_image(self,img):
        img_size = (img.shape[1], img.shape[0])
        src,dst = self.get_transform_parameters(img)

        M = cv2.getPerspectiveTransform(dst, src)
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

        return warped


    def histogram_find_lane(self,img, left_line, right_line, render_out = False):

        # Using previous measurements to guide histogram search
        guide_left = left_line.get_as_poly()
        guide_right = right_line.get_as_poly()

        if(render_out):
            out_img = cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_GRAY2RGB)

        # Take a histogram of the bottom half of the image
        histogram = np.sum(img[img.shape[0]/2:,:], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)

        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[:midpoint]) + midpoint

        # Current positions to be updated for each window
        if(guide_left):
            leftx_current = int(guide_left(img.shape[0]/4))
        else:
            leftx_current = leftx_base

        if(guide_right):
            rightx_current = int(guide_right(img.shape[0]/4))
        else:
            rightx_current = rightx_base

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(img.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Set the width of the windows +/- margin
        margin = 150
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window+1)*window_height
            win_y_high = img.shape[0] - window*window_height

            center = ((win_y_high - win_y_low) / 2) + win_y_low

            if(guide_left):
                leftx_current = int(guide_left(center))

            if(guide_right):
                rightx_current = int(guide_right(center))

            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            if(render_out):
                cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
                cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            # Only used if no guidance vectors are included
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_line.add_historical_points(leftx,lefty)
        right_line.add_historical_points(rightx,righty)

        leftx = np.concatenate(left_line.historical_x)
        lefty = np.concatenate(left_line.historical_y)

        rightx = np.concatenate(right_line.historical_x)
        righty = np.concatenate(right_line.historical_y)

        # Fit a second order polynomial to each
        left_fit,left_error = np.polyfit(lefty, leftx, 2, cov=True)
        right_fit, right_error  = np.polyfit(righty, rightx, 2, cov=True)

        '''
        Sanity Checks. Should probably be expanded since there are lots of
        details about roads (parallel/size) that can help determine outliers
        '''
        left_sse= np.sum(left_error**2)
        if(left_sse > 0.5 and guide_left):
            left_fit = guide_left

        right_sse = np.sum(right_error**2)
        if(right_sse > 0.5 and guide_right):
            right_fit = guide_right

        # test parallel
        if(guide_left and guide_right):
            delta = np.abs(left_fit-right_fit)

            if(delta[0] > 0.0005 or delta[1] > 0.5):
                # keep the line with the lowest error
                #if(left_sse < right_sse):
                right_fit = guide_right
                #else:
                left_fit = guide_left


        left_line.current_fit = left_fit
        left_line.current_fit_error = left_error

        right_line.current_fit = right_fit
        right_line.current_fit_error = right_error

        if(render_out):
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
            return left_line, right_line, out_img
        else:
            return left_line, right_line


    def read_camera_data(self,filename="camera_pickle.p"):
        #TODO handle missing file
        dist_pickle = pickle.load( open( filename, "rb" ) )
        mtx = dist_pickle["mtx"]
        dist = dist_pickle["dist"]
        return mtx, dist

    def remove_edge_pixels(self, img, thresh=75):
        shape=img.shape
        w=shape[1]
        h=shape[0]
        # left
        cv2.rectangle(img,(0,0),(thresh,h),(0,0,0),thresh)
        #right
        cv2.rectangle(img,(w-thresh,0),(w,h),(0,0,0),thresh)
        return img

    def extract_yellow(self,img):
        """
        Generates an image mask selecting yellow pixels.
        :param img: image with pixels in range 0-255
        :return: Yellow 255 not yellow 0
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, (20, 50, 150), (40, 255, 255))

        return mask

    def hls_select(self,img, thresh=(0, 255)):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]
        binary_output = np.zeros_like(s_channel)
        binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
        return binary_output

    # Define a function that takes an image, gradient orientation,
    # and threshold min / max values.
    def abs_sobel_thresh(self,img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        # Return the result
        return binary_output

    # Define a function to return the magnitude of the gradient
    # for a given sobel kernel size and threshold values
    def mag_thresh(self,img, sobel_kernel=3, mag_thresh=(0, 255)):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255
        gradmag = (gradmag/scale_factor).astype(np.uint8)
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

        # Return the binary image
        return binary_output


    # Define a function to threshold an image for a given range and Sobel kernel
    def dir_threshold(self,img, sobel_kernel=3, thresh=(0, np.pi/2)):
        # Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient direction,
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output =  np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        # Return the binary image
        return binary_output

    def get_transform_parameters(self,img):
        img_size = (img.shape[1], img.shape[0])
        src = np.float32(
            [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
            [((img_size[0] / 6) - 10), img_size[1]],
            [(img_size[0] * 5 / 6) + 60, img_size[1]],
            [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
        dst = np.float32(
            [[(img_size[0] / 4), 0],
            [(img_size[0] / 4), img_size[1]],
            [(img_size[0] * 3 / 4), img_size[1]],
            [(img_size[0] * 3 / 4), 0]])

        return src, dst

    def __load_camera(self,image):
        if(self.camera):
            self.mtx,self.dist = self.read_camera_data(self.camera)
            return image

    def __apply_camera(self,image):
        if(self.camera):
            image = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
            return image
