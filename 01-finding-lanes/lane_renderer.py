import cv2
import os
import numpy as np

class LaneRenderer:
    def __init__(self):
        pass
        #self.image = cv2.imread(image_path) if image_path else None
    
    def canny(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blur, 50, 150)
    
        return canny
    
    def loadImage(self,image_file_name):
        image_path = os.path.join(os.path.dirname(__file__), image_file_name)
        image = cv2.imread(image_path)        
        return image
    
    def drawLanesOverImage(self,image_file_name):
        image = self.loadImage(image_file_name)
        lane_image = np.copy(image)
        canny_image = self.canny(lane_image)
        cropped_canny_image = self.region_of_interest(canny_image)
        
        lines = cv2.HoughLinesP(cropped_canny_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
        averaged_lines = self.average_slope_intercept(lane_image, lines)
        line_image = self.display_line(lane_image,averaged_lines)
        
        # overlay lines on top of given image
        combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
        
        cv2.imshow("result", combo_image)
        cv2.waitKey(0)
    
    
    def average_slope_intercept(self, image, lines):
        left_fit = [] #coordinates of the averaged lines in the left
        right_fit = [] #coordinates of the averaged lines in the right
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1) #fitting a first degree polynomial to our x and y points and returns a vector of coefficients which describe the x and y intercept
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0: #Left lane lines have negative slop
                left_fit.append((slope, intercept))
            else: #Right lane lines have positive slop
                right_fit.append((slope, intercept))
                
        if len(left_fit) and len(right_fit):
            left_fit_average = np.average(left_fit, axis=0) #to get average over columns, because each list includes slopes in the first column and intercepts in the second column
            right_fit_average = np.average(right_fit, axis=0)
            left_line = self.make_coordinates(image, left_fit_average)
            right_line = self.make_coordinates(image, right_fit_average)
            averaged_lines = [left_line, right_line]
            return averaged_lines

    def make_coordinates(self, image, line_parameters):
        slope, intercept = line_parameters
        y1 = image.shape[0] #image height, because we want to draw the lines from the bottom of the image
        y2 = int(y1 * (3/5)) #we draw the line in the 2/5 bottom of the image
        x1 = int((y1 - intercept)/slope) #y = mx+b ==> x = (y-b)/m
        x2 = int((y2 - intercept)/slope) #y = mx+b ==> x = (y-b)/m

        return np.array([x1, y1, x2, y2])
    
    def display_line(self, image, lines):
        line_image = np.zeros_like(image)
        if lines is not None:
            for x1, y1, x2, y2 in lines:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

        return line_image
    
    def region_of_interest(self, image):
        height = image.shape[0]
        polygons = np.array([
            [(200, height), (1100, height), (550, 250)]
        ])
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, polygons, 255)
        masked_image = cv2.bitwise_and(image, mask)

        return masked_image

    def drawLanesOverVideo(self, vidio_file_name) :
        
        video_path = os.path.join(os.path.dirname(__file__), vidio_file_name)
        cap = cv2.VideoCapture(video_path)
    
        while(cap.isOpened()):
            _, frame = cap.read() #decodes every video frame and return two values. 1st one is a boolean, which we leave it as blank, and the 2nd one is the frame which is an image
            
            #copy of the above algorithm, only replacing the lange image with the current fram in the video
            canny_image = self.canny(frame)
            cropped_image = self.region_of_interest(canny_image)
            lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
            averaged_lines = self.average_slope_intercept(frame, lines)
            line_image = self.display_line(frame, averaged_lines)
            combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

            cv2.imshow("result", combo_image)
            
            #closing video upon pressing the "q" key on the keyboard
            if cv2.waitKey(1) & 0xFF == ord('q'): #waiting 1 milisecond between frames #masking the integer value we get from waitKey to 8 bits to ensure cross platform compatibility of code
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    media_loader = LaneRenderer()
    # media_loader.drawLanesOverImage("test_image.jpg")
    media_loader.drawLanesOverVideo("test_video.mp4")
       
    print("Demo Completed")
    
