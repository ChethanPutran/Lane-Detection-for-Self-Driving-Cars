#Importing neccessay modules
import cv2
import numpy as np
import matplotlib.pyplot as plt



def makeCoordinates(image,lineParams):
    slope,intercept = lineParams
    
    #Staring point of line is the height of the image
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    
    return np.array([x1,y1,x2,y2])

def averageSlopeIntercept(image,lines):
    
    leftFit = []
    rightFit = []
    
    for line in lines:
        #Getting ends points of the line
        x1, y1, x2, y2 = line.reshape(4)
        
        #Returs slope and y-intercept of line passing through that points
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        
        slope = parameters[0]
        yintercept = parameters[1]
        
        #Lines at left will have negative slope and right will have positive slope
        if slope < 0:
            leftFit.append((slope,yintercept))
        else: 
            rightFit.append((slope,yintercept)) 
            
    leftFitAverage = np.average(leftFit,axis=0)  #Averaging the left line parameters
    rightFitAverage = np.average(rightFit,axis=0)  #Averaging the right line parameters
    
    #Getting single line
    leftLine = makeCoordinates(image,leftFitAverage)
    rightLine = makeCoordinates(image,rightFitAverage)
    
    return np.array([leftLine,rightLine])
    
    
def cannyImage(image):
    #1.Conveting coloured image to grayscale image
    grayScaleImage = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    #2.Reducing noise and smoothening image
    bluredGSImage = cv2.GaussianBlur(grayScaleImage,(5,5),0)
    
    #Determing the edges based on gradient(taking derivative(f(x,y)) x->along width of the message y-> along height)
    canny = cv2.Canny(bluredGSImage,50,150)#(image,low_threshold,hight_threshold)
    
    return canny 

def regioOfInterest(image):
    height = image.shape[0]#No. of rows = height
    polygons = np.array([[(200,height),(1100,height),(550,250)]])#Creating polygon arround the lane
    
    #Creating a black image having same size as that of of test image
    mask = np.zeros_like(image)

    #Filling back image with generated triangle
    cv2.fillPoly(mask,polygons,255)
    
    #Taking bitwise and between image and mask to get the are of interest
    maskedImage = cv2.bitwise_and(image,mask)
    return maskedImage

def displayLines(image,lines):
    lineImage = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(lineImage,(x1,y1),(x2,y2),(255,0,0),10)
    
    return lineImage


""" **** For Single Image ****
    
#Loading test image
image = cv2.imread("test_image.jpg")

#Getting copy of the image
laneImage = np.copy(image)

#Getting canny image
canny = cannyImage(laneImage)

#Displaying image using matplotlib library for getting images dimensional details
# plt.imshow(canny)
# plt.show()

croppedImage = regioOfInterest(canny)

#Creating lines
lines = cv2.HoughLinesP(croppedImage,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)

averagedLines = averageSlopeIntercept(laneImage,lines)

#Adding lines to a black background
lineImage = displayLines(laneImage,averagedLines)

#Comning lines and test image
combinedImage = cv2.addWeighted(laneImage,0.8,lineImage,1,1)

#Displaying image using cv2 library
cv2.imshow('result',combinedImage)
cv2.waitKey(0) """


"""   ***** For Video Frame ******  """
#Capturing video
cap = cv2.VideoCapture("test.mp4") 

#Return true if video Capturing is initialized
while(cap.isOpened()):
    
    try:
        
        _, frame = cap.read()
        
        
        #Loading test image
        image = cv2.imread("test_image.jpg")

        #Getting canny image
        canny = cannyImage(frame)

        #Displaying image using matplotlib library for getting images dimensional details
        # plt.imshow(canny)
        # plt.show()

        croppedImage = regioOfInterest(canny)

        #Creating lines
        lines = cv2.HoughLinesP(croppedImage,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)

        averagedLines = averageSlopeIntercept(frame,lines)

        #Adding lines to a black background
        lineImage = displayLines(frame,averagedLines)

        #Comning lines and test image
        combinedImage = cv2.addWeighted(frame,0.8,lineImage,1,1)

        #Displaying image using cv2 library
        cv2.imshow('result',combinedImage)
    except:
        break
            
    if cv2.waitKey(1) == ord('q'): #Delaying for 1mili second
       break 
       
cap.release()  
cv2.destroyAllWindows()

