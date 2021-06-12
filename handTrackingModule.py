from cv2 import cv2
import mediapipe as mp
import time

class HandDetector():
    """MediaPipe Hands.

    MediaPipe Hands processes an RGB image and returns the hand landmarks and
    handedness (left v.s. right hand) of each detected hand.

    Please refer to https://solutions.mediapipe.dev/hands#python-solution-api for
    usage examples.
    """

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        """Initializes a HandDetector object.

        Args:
            static_image_mode: Whether to treat the input images as a batch of static
                and possibly unrelated images, or a video stream. See details in
                https://solutions.mediapipe.dev/hands#static_image_mode.
            max_num_hands: Maximum number of hands to detect. See details in
                https://solutions.mediapipe.dev/hands#max_num_hands.
            min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for hand
                detection to be considered successful. See details in
                https://solutions.mediapipe.dev/hands#min_detection_confidence.
            min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for the
                hand landmarks to be considered tracked successfully. See details in
                https://solutions.mediapipe.dev/hands#min_tracking_confidence.
        """

        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.mediapipe.python.solutions.drawing_utils
        self.mpHands = mp.solutions.mediapipe.python.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, 
                                        self.maxHands, 
                                        self.detectionCon, 
                                        self.trackCon)
    

    def findHands(self, img, draw=True):
        """Processes an image and returns the image with hand landmarks.

        Args:
            img: An image represented as a numpy ndarray.
            draw: A bool specifying if landmarks need to be drawn.

        Returns:
            A processed image with the hand landmarks drawn (if specified).
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, 
                                               handLms, 
                                               self.mpHands.HAND_CONNECTIONS)

        return img
                

    def findPosition(self, img, handNumber=0, draw=True):
        """Find the coordinates for each landmark in the hand.

        Args:
            img: An image represented as a numpy ndarray.
            handNumber: Select which hand to find coordinates
            draw: A bool specifying if landmarks need to be drawn.

        Returns:
            A list of tuples containing the landmark id and x/y coordinates.
        """
        lmList= []
        
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNumber]

            for id, lm in enumerate(myHand.landmark):
                h, w, _ = img.shape
                x, y = int(lm.x * w), int(lm.y * h)
                lmList.append((id,x,y))

                # Puts a shape on landmark id
                if draw:
                    cv2.circle(img, (x,y), 15, (255,0,255), cv2.FILLED)
        
        return lmList


def main():
    ptime = 0
    ctime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()

        if success:
            img = detector.findHands(img)
        
        lmList = detector.findPosition(img, draw=False)
        if lmList:
            print(lmList[4])

        # Output FPS
        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime
        cv2.putText(img, f'FPS: {fps}', (20,70), cv2.FONT_HERSHEY_PLAIN, 
                    3, (255, 0, 0), 3)

        # Output image
        cv2.imshow('Image', img)
        cv2.waitKey(1)


if __name__=='__main__':
    main()