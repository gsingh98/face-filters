import cv2  # Required to get access to the camera for live input
import dlib  # For the landmark detection algroithm
import numpy as np
import math
from numba import njit
import itertools
from numpy.linalg import inv

# TRIGGERS TO SHOW OLDER IMPELEMENTATIONS
TPS_CUSTOM = False
COMPUTE_HOMOGRAPHY = True  # Uses techniques shown in class to compute the homography and place the filter on the face

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./final_project/trained_models/shape_predictor_68_face_landmarks.dat')
cap = cv2.VideoCapture(0)  # Capture video from my webcam
filter_names = ['beard1', 'beard2', 'glasses1']
filter_num = 1
scale_percent = 20

# A function to overlay one image over the other. Returns the output image
# x and y here are the location at which the overlay image needs to be placed
# source: https://stackoverflow.com/questions/40895785/using-opencv-to-overlay-transparent-image-onto-another-image
def overlay_images(background, overlay, x, y):
    # separate the alpha channel from the color channels. The alpha channel with control transparency
    a_channel = overlay[:, :, 3] / 255 # convert from 0-255 to 0.0-1.0
    overlay_updated = overlay[:, :, :3]

    # Generate the mask for every different channel
    rows, cols = a_channel.shape
    a_mask = np.zeros((rows, cols, 3))
    a_mask[:, :, 0] = a_channel
    a_mask[:, :, 1] = a_channel
    a_mask[:, :, 2] = a_channel
    
    rows, cols, channels = overlay.shape
    background_overlap = background[x:x+rows, y:y+cols]
    
    # To create the overlapped we do like a weighted sum in this case
    overlapped = overlay_updated * a_mask + background_overlap * (1 - a_mask)
    
    # overwrite the section of the background image that has been updated
    background[x:x + rows, y:y + cols] = overlapped

    return background


# Function to get a filter along with all it's associated points that will be used to align the filter to the face
# Will also resize by padding the filter. Will be resized to the output size from the camers
def get_filter(file_name):
    # Capture the image from the webcam
    ret, image_original = cap.read()

    w = int(image_original.shape[1] * scale_percent / 100)
    h = int(image_original.shape[0] * scale_percent / 100)

    # These will be the dimensions that our image will be resized to by padding at the end
    dimensions_new = (w, h)

    filter_image = cv2.imread('./final_project/filters/' + file_name + '.png', cv2.IMREAD_UNCHANGED)
    filter_points = []
    filter_points_to_match = list()
    with open('./final_project/filters/' + file_name + '.txt') as f:
        # Get all the points of interest for this particular mask. Should make things extensible
        lines = f.readline()
        points = lines.split(',')

        # Add the required points we need to match on to the list
        for num in points:
           filter_points_to_match.append(int(num) - 1)

        lines = f.readlines()
        for line in lines:
            coordinates = line.split(',')
            filter_points.append([int(coordinates[0]), int(coordinates[1])])

    h_old, w_old, c = filter_image.shape
    h_scale, w_scale = h / h_old, w / w_old

    # Resize the image to match output
    filter_image = cv2.resize(filter_image, (w, h)) 

    # Update the points to match the resized image
    for point in filter_points:
        point[0] = int(point[0] * w_scale)
        point[1] = int(point[1] * h_scale)
    
    # for i, point in enumerate(filter_points):
	#     # Draw the circle to mark the keypoint
    #     cv2.circle(filter_image, (point[0], point[1]), 1, (0, 0, 255), -1)

    # cv2.imshow("Updated filter", filter_image)

    # Return the scaled out values
    return  filter_image, filter_points_to_match, filter_points
            

# This will apply the function U to a given value of r which is the distance
@njit
def U_r(r):
    if r == 0:
        return 0
    return r * r* math.log(r)


# An implementation of a Thin Plate Spline Transformation that will transform points from 1 to 2.
# Assuming: pts1 is the control and pts2 is what we are trying to warp
# It will apply the calculated transoformation on the given mask and return a warped image
# It is assumed that pts1 and pts2 hold points that are in order of matches.
# It is expected for the custom implementation to be slower than the actual impelementation but there are
# ways to be approximating all this.
# Helpful Resources:
# https://elonen.iki.fi/code/tpsdemo/
# https://link.springer.com/content/pdf/10.1007%2F3-540-47977-5_2.pdf
@njit
def TPS_Transformer(pts1, pts2, axis):
    # Make sure the size of the input is the same for both the input points 
    assert(len(pts1) > 0 and len(pts2) > 0)
    assert(len(pts1[0]) == len(pts2[0]))
    n = len(pts1[0])

    # First we need to establish all the control points that we want to bend around
    C = np.zeros((n, 3))
    C[:,0:2] = pts1[0]
    C[:,2] = pts2[0, :, axis]

    # We define P as a matrix to be a set of all the control points that we are already given
    P = np.zeros((n, 3))
    P[:, 0] = 1
    P[:, 1:] = pts1[:]
    P_Trans = P.transpose()

    # Filling up the K matrix
    K = np.zeros((n, n))
    for i in range (0, len(pts1[0])):   # Going over rows
        for j in range (0, len(pts1[0])):   # Going over columns
            dist = np.linalg.norm(C[j, :2] - C[i, :2])
            if dist == 0:
                K[i][j] = 0
            else:
                K[i][j] = U_r(dist)

    # Putting it all gother to create the L matrix
    L = np.zeros((n + 3, n + 3))
    L[:n, :n] = K
    L[:n, n:] = P
    L[n:, :n] = P_Trans

    # # Generate the Y
    Y = np.zeros((n + 3, 1))
    for i in range (0, n):
        Y[i] = C[i, 2]

    return np.linalg.solve(L, Y)


@njit
def TPS_Warp(pts1, pts2, mask): 
    # Finally putting it all together
    Wa_x = TPS_Transformer(pts1, pts2, 1)
    Wa_y = TPS_Transformer(pts1, pts2, 0)
    n = len(pts1[0])

    final_image = np.zeros(mask.shape, dtype=np.uint8)
    # image_map = []

    # rs = np.arange(0, mask.shape[0])
    # cs = np.arange(0, mask.shape[1])
    # for r in itertools.product(rs, cs):
    #     image_map.append([r[0], r[1]])


    # 
    # print(image_map)
    
    # We now have everything to actually warp the given image, which we can now do
    rows, cols, chan = mask.shape
    for row in range (0, rows):
        for col in range (0, cols):
            # x = row
            # y = col
            x = Wa_x[-3][0] + Wa_x[-2][0] * col + Wa_x[-1][0] * row
            for i in range (0, n):
                dist = np.linalg.norm(np.array([pts1[0][i][0] - row, pts1[0][i][1] - col]))
                x += Wa_x[i][0] * U_r(dist)
            y = Wa_y[-3] + Wa_y[-2] * col + Wa_y[-1] * row
            for i in range (0, n):
                dist = np.linalg.norm(np.array([pts1[0][i][0] - row, pts1[0][i][1] - col]))
                # dist = np.linalg.norm(pts1[0][i] - [row, col])
                y += Wa_y[i] * U_r(dist)

            x = int(x)
            y = int(y[0])

            if x > mask.shape[0] - 1 or y > mask.shape[1] - 1 or x < 0 or y < 0:
                final_image[row, col, :] = 0 
            else:
                final_image[row, col, :] = mask[int(x), int(y), :]
            

    # cv2.imshow("mask", mask)
    # cv2.imshow("Warp", final_image)
    return final_image


# Takes a background image with given facial landmark data along with the given filter
# that also has landmark data and projects the filter on to the given image.
# Returns the image with the overlayed mask
# reference: https://medium.com/acmvit/how-to-project-an-image-in-perspective-view-of-a-background-image-opencv-python-d101bdf966bc
def project_mask(background, background_landscape, mask, mask_landscape, points_to_match):
    # Update the pts1 and pts2 with the appropriate points
    # TODO: This is currently only using 2 points to calculate the homography. Need to use more points
    # points_to_match = [0, 7, 9, 16, 48, 51]

    # Create the points that we want to match on
    pts1 = np.zeros((len(points_to_match),2))
    pts2 = np.zeros((len(points_to_match),2))
    matches = list()

    for i, point in enumerate(points_to_match):
        pts1[i] = [background_landscape[point][0], background_landscape[point][1]]
        pts2[i] = [mask_landscape[point][0], mask_landscape[point][1]]
        matches.append(cv2.DMatch(i, i ,0))

    rows, cols, channels = background.shape

    # Need to reshape the arrays due to: https://stackoverflow.com/questions/41536344/problems-when-implementing-opencv-tps-shapetransformer-in-python
    pts1 = pts1.reshape(1, -1, 2)
    pts2 = pts2.reshape(1, -1, 2)

    warped_mask = None

    if COMPUTE_HOMOGRAPHY:
        # Find the homography and apply the tranformation
        h, mask_ret = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
        warped_mask = cv2.warpPerspective(mask, h, (cols, rows))
    else: 
        mask = cv2.resize(mask, (cols, rows))

        # Looked up Thin Plate Spline Shape Transformer: https://stackoverflow.com/questions/70601059/opencv2-thin-plate-splines-apply-transformation-not-working
        if TPS_CUSTOM:
            warped_mask = TPS_Warp(pts1, pts2, mask)
        else:
            splines= cv2.createThinPlateSplineShapeTransformer()
            temp = splines.estimateTransformation(pts1, pts2, matches)
            warped_mask = splines.warpImage(mask) #image warps fine 

    # Now that we have transformed the filter, we can go ahead and overlay it on the background image
    overlayed = overlay_images(background, warped_mask, 0, 0)

    return overlayed


# Image should be tagged already with all the required values that need to be matched up
filter_image, filter_points_to_match, filter_points = get_filter('beard1')

def toggle_filters(event, x, y, flags, param):
    global filter_num, filter_image, filter_points_to_match, filter_points
    if event == cv2.EVENT_LBUTTONDOWN:
        filter_num = (filter_num + 1) % len(filter_names)

        # Image should be tagged already with all the required values that need to be matched up
        filter_image, filter_points_to_match, filter_points = get_filter(filter_names[filter_num])

cv2.namedWindow('Landmark Detection')
cv2.setMouseCallback('Landmark Detection',toggle_filters)

# for i, pt in enumerate(filter_points_to_match):
#     # Draw the circle to mark the keypoint
#     cv2.circle(filter_image, filter_points[pt], 5, (0, 0, 255), -1)
# 
# cv2.imshow("Filt", filter_image)


while True:
    # Capture the image from the webcam
    ret, image_original = cap.read()

    # we need to resize the image and scale it down to 20%
    w = int(image_original.shape[1] * scale_percent / 100)
    h = int(image_original.shape[0] * scale_percent / 100)
    dimensions_new = (w, h)

    image = cv2.resize(image_original, dimensions_new, interpolation = cv2.INTER_AREA)
    

    # Convert the image color to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect the face
    rects = detector(gray, 1)
    # for rect in rects:
	#     # Draw the circle to mark the keypoint
    #     print(rect.tl_corner())
    #     cv2.rectangle(image, [rect.tl_corner().x, rect.tl_corner().y], [rect.br_corner().x,rect.br_corner().y] , (0, 0, 255), 3)

    # Detect landmarks for each face
    for rect in rects:
        # Get the landmark points
        shape = predictor(gray, rect)
	# Convert it to the NumPy Array
        shape_np = np.zeros((68, 2), dtype="int")
        # 0-17 belong to chin
        # 18-22 right eyebrow
        # 23-27 left eyebrow
        for i in range(0, 68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)
        shape = shape_np

        # Display the landmarks
        # for i, (x, y) in enumerate(shape):
	    #     # Draw the circle to mark the keypoint
        #     cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
        
        image =  project_mask(image, shape, filter_image, filter_points, filter_points_to_match) 

    # Display the image
    cv2.imshow('Landmark Detection', image)
    # Press the escape button to terminate the code
    if cv2.waitKey(10) == 27:
        break

cap.release()
