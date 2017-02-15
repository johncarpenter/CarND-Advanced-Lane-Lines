import numpy as np
import cv2
import pickle
import kalman as Kalman
import math


xml = np.matrix('0. 0. 0.').T
Pl = np.matrix(np.eye(3))*5 # initial uncertainty
Ql = np.matrix(np.eye(3))*0.5 # delta changes

xmr = np.matrix('0. 0. 0.').T
Pr = np.matrix(np.eye(3))*0.5 # initial uncertainty

def process_image_smoothing(image, camera):
    raw_image = image.copy()

    global xml,Pl,xmr,Pr
    if(camera):
        #print("Importing Camera Calibration from {}".format(camera))
        mtx,dist = read_camera_data(camera)
        image = cv2.undistort(image, mtx, dist, None, mtx)
    #else:
        #print("Skipping Camera Calibration")

    s = hls_select(image,thresh=(120, 220))

    ksize=15

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(25,110))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(25,255))

    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 255))

    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.1,1.1))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1)&(s == 1))|((dir_binary == 1)&(mag_binary == 1))] = 1


    warp = warp_image(combined)

    residual_x = -1
    residual_y = -1

    reset_l = False
    reset_r = False

    if(xml.all()):
        guide_left=np.poly1d(xml.flatten().tolist()[0])
    else:
        guide_left=None

    if(xmr.all()):
        guide_right=np.poly1d(xmr.flatten().tolist()[0])
    else:
        guide_right=None


    #try:
    left,right,left_error, right_error, out_img  = histogram_find_lane(warp,render_out = True, guide_left=guide_left, guide_right=guide_right)


    left_error = np.matrix(np.eye(3))*(np.sum(left_error**2))
    right_error = np.matrix(np.eye(3))*(np.sum(right_error**2))


    l = calc_curvature(np.poly1d(left))
    r = calc_curvature(np.poly1d(right))

    #if not xml.all():
    #    xml = np.matrix(left).T
    #else:
    xmlp,Pl,rx = Kalman.kalman_c(xml,Pl,left,left_error)
    residual_x = rx.T*rx

    #if (residual_x <= 100):
        # use measured / filtered value
    xml = xmlp
        # between 100 - 1000 we use the previous measurement instead
    #elif( residual_x > 100 and residual_x <= 1000):
    #    xml,Pl = Kalman.predict(xml,Pl,Q=Ql)
    if( residual_x > 40000 ):
        reset_l = True

    if not xmr.all():
        xmr = np.matrix(right).T
    else:
        #error = np.matrix(np.eye(3))*5
        xmrp,Pr,ry = Kalman.kalman_c(xmr,Pr,right, right_error)
        residual_y = ry.T*ry

        xmr = xmrp
        if( residual_y > 40000):
            reset_r = True


    #except Exception as error:
    #    print("Error processing image:",str(error))

    overlay = draw_lane_poly(warp,xml,xmr)
    overlay = unwarp_image(overlay)

    if(reset_r):
        xmr = np.matrix('0. 0. 0.').T
        Pr = np.matrix(np.eye(3))*0.5

    if(reset_l):
        xml = np.matrix('0. 0. 0.').T
        Pl = np.matrix(np.eye(3))*0.5

    #display_text = "L: {}  R: {}".format(residual_x,residual_y)
    display_text = "L: {}  R: {}".format(l,r)

    final = cv2.addWeighted(raw_image, 1, overlay, 0.3, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    final = cv2.putText(final,display_text,(10,100), font, 0.5,(255,255,255),2)


    fit_errors = "L: {}  R: {}".format(np.sum(left_error**2),np.sum(right_error**2))
    out_img = cv2.putText(out_img,fit_errors,(10,100), font, 1,(255,255,255),2)
    out_img = cv2.resize(out_img,(320,240))
    x_offset = final.shape[1]- 400
    y_offset = 25
    final[y_offset:y_offset+out_img.shape[0], x_offset:x_offset+out_img.shape[1]] = out_img


    return final,xml,xmr



leftx = None
rightx = None
def process_image(image, camera):
    raw_image = image.copy()

    if(camera):
    #    print("Importing Camera Calibration from {}".format(camera))
        mtx,dist = read_camera_data(camera)
        image = cv2.undistort(image, mtx, dist, None, mtx)
    #else:
    #    print("Skipping Camera Calibration")

    s = hls_select(image,thresh=(120, 220))

    ksize=15

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(25,110))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(25,255))

    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 255))

    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.1,1.1))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1)&(s == 1))|((dir_binary == 1)&(mag_binary == 1))] = 1

    warp = warp_image(combined)

    guide_left = leftx if leftx else None
    guide_right = rightx if rightx else None

    left,right,left_error,right_error = histogram_find_lane(warp,guide_left=guide_left, guide_right=guide_right)

    overlay = draw_lane_poly(warp,left,right)
    overlay = unwarp_image(overlay)

    final = cv2.addWeighted(raw_image, 1, overlay, 0.3, 0)
    return final,left,right

def calc_curvature(fit_cr):

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    y = np.array(np.linspace(0, 719, num=10))
    x = np.array([fit_cr(x) for x in y])
    y_eval = np.max(y)

    fit_cr = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)
    curverad = ((1 + (2 * fit_cr[0] * y_eval / 2. + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])

    return curverad

def draw_lane_poly(img, left, right):

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left[0]*ploty**2 + left[1]*ploty + left[2]
    right_fitx = right[0]*ploty**2 + right[1]*ploty + right[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    return color_warp


def get_transform_parameters(img):
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

def warp_image(img):
    img_size = (img.shape[1], img.shape[0])
    src,dst = get_transform_parameters(img)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped

def unwarp_image(img):
    img_size = (img.shape[1], img.shape[0])
    src,dst = get_transform_parameters(img)

    M = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped


def histogram_find_lane(img, render_out = False, guide_left= None, guide_right=None):

    if(render_out):
        out_img = cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_GRAY2RGB)

    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0]/2:,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)

    if(guide_left):
        leftx_base = int(guide_left(img.shape[0]/4))
    else:
        leftx_base = np.argmax(histogram[:midpoint])

    if(guide_right):
        rightx_base = int(guide_right(img.shape[0]/4))
    else:
        rightx_base = np.argmax(histogram[:midpoint]) + midpoint

    # Current positions to be updated for each window
    leftx_current = leftx_base
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

    '''
    Generate additional points along existing line for better polyfit
    '''
    #ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    ploty = np.random.uniform(0, img.shape[0]-1,100)
    if(guide_left):
        left_fitx = guide_left(ploty) + np.random.randint(-50, high=51)
        leftx = np.append(leftx,left_fitx)
        lefty = np.append(lefty,ploty)


    if(guide_right):
        right_fitx = guide_right(ploty)+ np.random.randint(-50, high=51)
        rightx = np.append(rightx,right_fitx)
        righty = np.append(righty,ploty)


    '''
    left_weight = np.ones(len(lefty))
    if(guide_left):
        for y in range(len(lefty)):
            left_weight[y] = (margin - (leftx[y] - guide_left(lefty[y]))) / margin

    right_weight = np.ones(len(righty))
    if(guide_right):
        for y in range(len(righty)):
            right_weight[y] = (margin - (rightx[y] - guide_right(righty[y]))) / margin
    '''

    # Fit a second order polynomial to each
    left_fit,left_error = np.polyfit(lefty, leftx, 2, cov=True)
    right_fit, right_error  = np.polyfit(righty, rightx, 2, cov=True)

    if(np.sum(left_error**2) > 0.5 and guide_left):
        left_fit = guide_left

    if(np.sum(right_error**2) > 0.5 and guide_right):
        right_fit = guide_right


    if(render_out):
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        return left_fit, right_fit, left_error, right_error, out_img
    else:
        return left_fit, right_fit, left_error, right_error

def read_camera_data(filename="camera_pickle.p"):
    #TODO handle missing file
    dist_pickle = pickle.load( open( filename, "rb" ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    return mtx, dist

def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
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
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
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
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
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
