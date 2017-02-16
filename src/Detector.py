from Line import Line
from ImageProcessor import ImageProcessor

class Detector:
    def __init__(self,use_smoothing=True, camera = None):
        '''
        Parameters
        use_smoothing: Apply the KF or ignore
        camera: Pickle file for camera calibration
        '''
        self.left_line = Line()

        self.right_line = Line()

        self.camera = camera

        self.use_smoothing = use_smoothing

        self.ImageProcessor = ImageProcessor(camera = camera, use_smoothing= use_smoothing)

    def process_image(self,image):

        image,self.left_line,self.right_line = self.ImageProcessor.process_image(image,self.left_line,self.right_line)

        return image
