from cv2 import cv2 
import mediapipe as mp
import handTrackingModule as htm
import numpy as np
import time
import os

# TODO:
#   - Implement check on thumb for left hand
#   - Implement thumbs up case
#   - Responsive display


CAM_WIDTH, CAM_HEIGHT = 640, 480


def get_folder_filenames(folder_path: str) -> list:
    """Retrieves the file names within given directory.

    Args:
        folder_path: A string representing the path to the folder.

    Returns:
        A list of file names (strings) within the given directory.
    """

    folder_path = 'images/right'
    folder_file_list = os.listdir(folder_path)
    folder_file_list.sort()

    return folder_file_list


def read_image_files(folder_path: str, file_type='jpg') -> list:
    """Read in image files with cv2 from folder directory.

    Args:
        folder_path: A string representing the path to the folder.
        folder_file_list: A list of strings representing the files in the folder.
        file_type: A string representing the type of images being read.

    Returns:
        List of images read from folder directory.
    """

    images = []
    folder_file_list = get_folder_filenames(folder_path)

    if file_type == 'png':
        for imgPath in folder_file_list:
            images.append(cv2.imread(f'{folder_path}/{imgPath}', cv2.IMREAD_UNCHANGED))
    else:
        for imgPath in folder_file_list:
            images.append(cv2.imread(f'{folder_path}/{imgPath}'))

    return images


def apply_png_overlay(image, overlay, start_row, start_col, height=200, width=200):
    """Applies an PNG overlay to an RGB image.

    Args:
        image: An RGB image to apply an overlay.
        overlay: An image to be applied as overlay.
        start_row: The starting height of the overlay.
        start_col: The starting width of the overlay.

    Returns:
        A list of file names (strings) within the given directory.
    """

    alpha_s = overlay[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(0, 3):
        image[start_row:start_row+height, start_col:start_col+width, c] = (
            alpha_s * overlay[:, :, c] +
            alpha_l * image[start_row:start_row+height, start_col:start_col+width, c])
    
    return image


def main():
    # Record image drawing for documentation
    record = False
    if record:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

    cap = cv2.VideoCapture(0)
    cap.set(3, CAM_WIDTH)
    cap.set(4, CAM_HEIGHT)

    fingerOverlays = read_image_files('images/right', 'png')
    numberOverlays = read_image_files('images/numbers', 'png')

    detector = htm.HandDetector(detectionCon=0.75)
    finger_tips = (4, 8, 12, 16, 20)
    knuckles = (3, 6, 10, 14, 18)
    previousTime = 0

    print('Press escape to exit') 

    while True:
        success, img = cap.read()

        if success:
            img = detector.findHands(img)
            landmark_list = detector.findPosition(img, draw=False)

            # Check to see if each finger is open or closed
            if landmark_list:
                fingers_open = [0] * 5

                # Check right thumb
                fingertip_width = landmark_list[finger_tips[0]][1]
                knuckle_width = landmark_list[knuckles[0]][1]

                if fingertip_width > knuckle_width:
                    fingers_open[0] = 1

                # Check fingers
                for i in range(1, 5):
                    fingertip_height = landmark_list[finger_tips[i]][2]
                    knuckle_height = landmark_list[knuckles[i]][2]

                    if fingertip_height < knuckle_height:
                        fingers_open[i] = 1
                
                # Apply overlay based on finger count
                fingers_count = sum(fingers_open)
                ol = fingerOverlays[fingers_count]
                ol2 = numberOverlays[fingers_count]
                img = apply_png_overlay(img, ol, 40, 0)
                img = apply_png_overlay(img, ol2, 260, 0)

            # Output FPS
            currentTime = time.time()
            fps = int(1 / (currentTime-previousTime))
            previousTime = currentTime
            cv2.putText(img, f'FPS: {fps}', (20, 30), cv2.FONT_HERSHEY_PLAIN,
                        1, (255, 0, 0), 1)

            # Output image
            if record: 
                out.write(img)
            cv2.imshow('Image', img)
            key = cv2.waitKey(1)
            if key == 27:
                break
    
    cap.release()
    if record: 
        out.release()
    cv2.destroyAllWindows() 

if __name__ == '__main__':
    main()