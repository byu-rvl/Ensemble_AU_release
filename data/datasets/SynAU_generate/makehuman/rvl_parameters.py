import numpy as np
from pathlib import Path
#parameters for the video generation. See the modified code in 7_FACSHuman.py

GENERATE_ON_STARTUP = True
HEADLESS = False
ONLY_ONE_VID_PER_PERSON = True #DO NOT CHANGE. MAKES IT TO SKIP VIDEOS THAT ALREADY EXIST. IF WANT TO RESTART CHANGE SAVE_LOCATION
MAKE_IMAGES_NOT_VIDEOS = True

#BASIC PARAMETERS
SAVE_LOCATION = "out_allPeople/"
Path(SAVE_LOCATION).mkdir(parents=True, exist_ok=True)

ROOT_DATA_PATH = "data/"

DOCUMENTS_DATA_PATH = "Documents/makehuman/v1py3/data/"

DOCUMENTS_DATA_PATH_PARENT = "Documents"

# VIDEO PARAMETERS
NUMBER_VIDEOS_PER_MOTION = 10
NUMBER_MOVEMENTS_AT_ONCE = 2
INCLUDE_BLANK_MOVEMENT = True

# frame controls.
NUMBER_FRAMES = 60
FPS = 30.0 #NOTE: IF YOU CHANGE THIS WILL NEED TO CHANGE VARIOUS CHANGES BELOW ON WHEN TO DO MOTIONS.
START_MOVEMENT_EARLIEST = 0
START_MOVEMENT_LATEST = int(NUMBER_FRAMES / 2)

UP_MOVEMENT_MIN_FRAMES = 5 #this will be the min number of frames it takes to go from blank movement to the full intensity.
DOWN_MOVEMENT_MIN_FRAMES = 5 # this will be min number of frames it takes to go from full intensity to blank.
END_MOVEMENT_BY = NUMBER_FRAMES # will have the movement done by this frame number.
#will randomly select if blink is in frame or not. Generally only blink 12 times per minute. So, only can have one per video.
MIN_BLINK_FRAMES = 7
MAX_BLINK_FRAMES = 13 #7 and 13 decided because blink takes approx. 1/3 of a second. So, give 3 frame buffer either way.

# movement controls.
MIN_INTENSITY = 0.7 # OUT OF 1.0
MAX_INTENSITY = 1.0 # OUT OF 1.0

#camera control
CAMERA_X_MIN = -0.1
CAMERA_X_MAX = 0.1
CAMERA_X_RESOLUTION = 0.05
CAMERA_X_POSSIBLE = list(np.linspace(CAMERA_X_MIN, CAMERA_X_MAX, int((CAMERA_X_MAX - CAMERA_X_MIN) / CAMERA_X_RESOLUTION) + 1))
CAMERA_Y_MIN = -0.15
CAMERA_Y_MAX = 0.1
CAMERA_Y_RESOLUTION = 0.05
CAMERA_Y_POSSIBLE = list(np.linspace(CAMERA_Y_MIN, CAMERA_Y_MAX, int((CAMERA_Y_MAX - CAMERA_Y_MIN) / CAMERA_Y_RESOLUTION) + 1))
CAMERA_Z_MIN = -25
CAMERA_Z_MAX = 25
CAMERA_Z_RESOLUTION = 5
CAMERA_Z_POSSIBLE = list(np.linspace(CAMERA_Z_MIN, CAMERA_Z_MAX, int((CAMERA_Z_MAX - CAMERA_Z_MIN) / CAMERA_Z_RESOLUTION) + 1))
CAMERA_ZOOM_MIN = 4
CAMERA_ZOOM_MAX = 6 #THIS IS AS CLOSE AS i CAN GET WITH 0 AGE.
CAMERA_ZOOM_RESOLUTION = 1
CAMERA_ZOOM_POSSIBLE = list(np.linspace(CAMERA_Z_MIN, CAMERA_Z_MAX, int((CAMERA_Z_MAX - CAMERA_Z_MIN) / CAMERA_Z_RESOLUTION) + 1))

# removed emotions. We don't need these ones. Maybe put back in later as more complicated motions?
# blinking is 43 so removed (so we can have blink be something to do in the video to not be different)
# 26_tongue_down, '19',--tongue. Remove, not tracking the tongue.
# '51', '52', '53', '54', '55', '56', '57', '58', control head rotations. Do not want these ones.
# '61', 'L61', 'R61', '62', 'L62', 'R62', '63', 'L63', 'R63', '64', 'L64', 'R64', '65', '66' are eye movements only. Don't want to track eye movements.
# '38', '39' Nose dilate and nose compress...not necessary to have.
BLINK_AU = "43"
if MAKE_IMAGES_NOT_VIDEOS:
    allAU_useLocallyOnly = ['01_Anger', '01_Langner_Angry', '02_Contempt', '02_Langner_Contemptuous', '03_Disgust', '03_Disgust_a',
                '03_Langner_Disgusted', '04_Embarrassment', '05_Fear', '05_Langner_Fearfull', '06_Happy', '06_Langner_Happy',
                '06_Langner_Happy_A', '06_Langner_Happy_B', '07_Langner_Sadness', '07_Sadness', '08_Langner_Surprised',
                '08_Surprise', '09_Pride', '1', 'L1', 'R1', '2', '2_a', 'L2', 'R2', '4', '4_a', '4_b', '5', 'L5', 'R5',
                '6', '6_a', '6_b', 'L6', 'R6', '7', 'L7', 'R7', '43', 'L43', 'R43', '9', '9_a', 'L9', 'R9', '10', 'L10',
                'R10', '11', 'L11', 'R11', '12', '12_a', '12_b', 'L12', 'R12', '13', 'L13', 'R13', '14', 'R14', 'L14', '15',
                'L15', 'R15', '16', '17', '17_a', '18', '18_a', '20', 'L20', 'R20', '22_25_down', '22_25_up_down',
                '22_25_upper', '23', '24', '28', '28_a', '28_bottom', '25', 'L25', 'R25', '25_a', '26', '26_lip_down',
                '26_tongue_down', '26_tongue_out', '26_a', '61', 'L61', 'R61', '62', 'L62', 'R62', '63', 'L63', 'R63', '64',
                'L64', 'R64', '65', '66', '51', '52', '53', '54', '55', '56', '57', '58', '19', '29', 'L30', 'R30', '31',
                '32', '33', '34', '35', '38', '39', 'both_eye_pupil_contract', 'both_eye_pupil_dilate',
                'left_eye_pupil_contract', 'left_eye_pupil_dilate', 'right_eye_pupil_contract', 'right_eye_pupil_dilate']
    BP4D_AUs_useLocallyOnly = ['1', '2', '4', '6', '7', '10', '12', '14', '15', '17', '23', '24']
    DISFA_AUs_useLocallyOnly = ['1', '2', '4', '6', '9', '12', '25', '26']
    AUs_useLocallyOnly = BP4D_AUs_useLocallyOnly + DISFA_AUs_useLocallyOnly
    DO_NOT_USE_AU = []
    foundAU_useLocallyOnly = []
    for i in allAU_useLocallyOnly:
        if i not in AUs_useLocallyOnly:
            DO_NOT_USE_AU.append(i)
        else:
            foundAU_useLocallyOnly.append(i)
    print("Found AU: ", foundAU_useLocallyOnly, len(foundAU_useLocallyOnly))
else:
    DO_NOT_USE_AU = ['01_Anger', '01_Langner_Angry', '02_Contempt', '02_Langner_Contemptuous', '03_Disgust', '03_Disgust_a',
                 '03_Langner_Disgusted', '04_Embarrassment', '05_Fear', '05_Langner_Fearfull', '06_Happy',
                 '06_Langner_Happy', '06_Langner_Happy_A', '06_Langner_Happy_B', '07_Langner_Sadness', '07_Sadness',
                 '08_Langner_Surprised', '08_Surprise', '09_Pride', "43", 'both_eye_pupil_contract', 'both_eye_pupil_dilate',
                'left_eye_pupil_contract', 'left_eye_pupil_dilate', 'right_eye_pupil_contract', 'right_eye_pupil_dilate',
                 '26_tongue_down', '51', '52', '53', '54', '55', '56', '57', '58', '61', 'L61', 'R61', '62', 'L62',
                 'R62', '63', 'L63', 'R63', '64','L64', 'R64', '65', '66', '38', '39', '19',] #add AU HERE THAT WE SHOULD NOT USE AT ALL.
    
# THE FOLLOWING ARE NOT IMPLEMENTED DUE TO THE FACT THAT IT MAY BE TOO MUCH VARIATION RIGHT OFF THE BAT.
# HEAD_UP_AU = "53"
# HEAD_DOWN_AU = "54"
# HEAD_UP_DOWN_AVAIL_INTENSITY = [0, 10, 20, 30, 40]
# HEAD_TILT_LEFT_AU = "55"
# HEAD_TILT_RIGHT_AU = "56"
# HEAD_TILT_LR_AVAIL_INTENSITY = [0, 10, 20, 30, 40]

# allAU: ['01_Anger', '01_Langner_Angry', '02_Contempt', '02_Langner_Contemptuous', '03_Disgust', '03_Disgust_a',
#           '03_Langner_Disgusted', '04_Embarrassment', '05_Fear', '05_Langner_Fearfull', '06_Happy', '06_Langner_Happy',
#         '06_Langner_Happy_A', '06_Langner_Happy_B', '07_Langner_Sadness', '07_Sadness', '08_Langner_Surprised',
#         '08_Surprise', '09_Pride', '1', 'L1', 'R1', '2', '2_a', 'L2', 'R2', '4', '4_a', '4_b', '5', 'L5', 'R5',
#         '6', '6_a', '6_b', 'L6', 'R6', '7', 'L7', 'R7', '43', 'L43', 'R43', '9', '9_a', 'L9', 'R9', '10', 'L10',
#         'R10', '11', 'L11', 'R11', '12', '12_a', '12_b', 'L12', 'R12', '13', 'L13', 'R13', '14', 'R14', 'L14', '15',
#         'L15', 'R15', '16', '17', '17_a', '18', '18_a', '20', 'L20', 'R20', '22_25_down', '22_25_up_down',
#         '22_25_upper', '23', '24', '28', '28_a', '28_bottom', '25', 'L25', 'R25', '25_a', '26', '26_lip_down',
#         '26_tongue_down', '26_tongue_out', '26_a', '61', 'L61', 'R61', '62', 'L62', 'R62', '63', 'L63', 'R63', '64',
#         'L64', 'R64', '65', '66', '51', '52', '53', '54', '55', '56', '57', '58', '19', '29', 'L30', 'R30', '31',
#         '32', '33', '34', '35', '38', '39', 'both_eye_pupil_contract', 'both_eye_pupil_dilate',
#         'left_eye_pupil_contract', 'left_eye_pupil_dilate', 'right_eye_pupil_contract', 'right_eye_pupil_dilate']

# ADD-ON CONSTANTS
CATEGORY_MODELING = "Modelling"
CATEGORY_EXTRA_ADD_ON = "Geometries"
TASK_BODY_TYPE = "Macro modelling"
TASK_FACS_HUMAN = "FACSHuman 0.1"
TASK_FACE = "Face"
TASK_CLOTHES ="Clothes"
TASK_EYES = "Eyes"
TASK_HAIR = "Hair"
TASK_TEETH = "Teeth"
TASK_EYEBROWS = "Eyebrows"
TASK_EYELASHES = "Eyelashes"
TASK_TONGUE = "Tongue"

CLOTHES_PXY_TYPE = "Clothes"
CARGO_PANTS_FILE = DOCUMENTS_DATA_PATH + "/clothes/cargo_pants/cargo_pants.mhpxy"
HERO_SUIT_FILE = DOCUMENTS_DATA_PATH + "/clothes/hero_suit_4/hero_suit_4.mhpxy"
JEAN_SHORTS_FILE = DOCUMENTS_DATA_PATH + "/clothes/jean_shorts/jean_shorts.mhpxy"
POLO_SHIRT_FILE = DOCUMENTS_DATA_PATH + "/clothes/Polo_t-shirt/Polo_t-shirt.mhpxy"
T_SHIRT_FILE = DOCUMENTS_DATA_PATH + "/clothes/T-shirt/t_shirt_basic_tucked.mhpxy"

MOUSTACHE_FILE = DOCUMENTS_DATA_PATH + "/clothes/Moustache/moustache.mhpxy"
FULL_BEARD_FILE = DOCUMENTS_DATA_PATH + "/clothes/Full_beard/full_beard.mhpxy"

EYELASHES_PXY_TYPE = "Eyelashes"
EYELASHES_01_FILE = DOCUMENTS_DATA_PATH + "/eyelashes/FACSEyeLashes01/FACSEyeLashes01.mhpxy"

TONGUE_PXY_TYPE = "Tongue"
TONGUE_FACS_TONGUE_FILE = DOCUMENTS_DATA_PATH + "/tongue/FACSTongue/FACSTongue.mhpxy"
TONGUE_FACS_TONGUE_BW_FILE = DOCUMENTS_DATA_PATH + "/tongue/FACSTongue_bw/FACSTongue_bw.mhpxy"

TEETH_PXY_TYPE = "Teeth"
TEETH_03_FILE = DOCUMENTS_DATA_PATH + "/teeth/FACSteeth03/FACSTeeth03.mhpxy"
TEETH_03_BW_FILE = DOCUMENTS_DATA_PATH + "/teeth/FACSteeth03_bw/FACSTeeth03_bw.mhpxy"
TEETH_04_FILE = DOCUMENTS_DATA_PATH + "/teeth/FACSteeth04/FACSTeeth04.mhpxy"
TEETH_01_FILE = DOCUMENTS_DATA_PATH + "/teeth/FACSteeth01/FACSTeeth01.mhpxy"
TEETH_01_A_FILE = DOCUMENTS_DATA_PATH + "/teeth/FACSteeth01a/FACSTeeth01a.mhpxy"
TEETH_01_B_FILE = DOCUMENTS_DATA_PATH + "/teeth/FACSteeth01b/FACSTeeth01b.mhpxy"
TEETH_02_FILE = DOCUMENTS_DATA_PATH + "/teeth/FACSteeth02/FACSTeeth02.mhpxy"
TEETH_02_A_FILE = DOCUMENTS_DATA_PATH + "/teeth/FACSteeth02a/FACSTeeth02a.mhpxy"
TEETH_02_B_FILE = DOCUMENTS_DATA_PATH + "/teeth/FACSteeth02b/FACSTeeth02b.mhpxy"

HAIR_PXY_TYPE = "Hair"
HAIR_F01_FILE = ROOT_DATA_PATH + "/hair/fhair01/fhair01.mhpxy"
HAIR_F1_FILE = ROOT_DATA_PATH + "/hair/hair1/hair1.mhpxy"
HAIR_M01_FILE = ROOT_DATA_PATH + "/hair/mhair01/mhair02.mhpxy"
HAIR_M04_FILE = ROOT_DATA_PATH + "/hair/hair_04_boy/hair_04.mhpxy"
HAIR_M05_FILE = ROOT_DATA_PATH + "/hair/hair_05_boy/hair_05.mhpxy"

EYEBROW_PXY_TYPE = "Eyebrows"
EYEBROW_01_FILE = ROOT_DATA_PATH + "/eyebrows/eyebrow1/mind_eyebrows_01.mhpxy"
EYEBROW_02_FILE = ROOT_DATA_PATH + "/eyebrows/eyebrow2/mind_eyebrows_02.mhpxy"
EYEBROW_03_FILE = ROOT_DATA_PATH + "/eyebrows/eyebrow3/mind_eyebrows_03.mhpxy"
EYEBROW_04_FILE = ROOT_DATA_PATH + "/eyebrows/eyebrow4/mind_eyebrows_04.mhpxy"
EYEBROW_05_FILE = ROOT_DATA_PATH + "/eyebrows/eyebrow5/mind_eyebrows_05.mhpxy"
EYEBROW_06_FILE = ROOT_DATA_PATH + "/eyebrows/eyebrow6/mind_eyebrows_06.mhpxy"
EYEBROW_07_FILE = ROOT_DATA_PATH + "/eyebrows/eyebrow7/mind_eyebrows_07.mhpxy"
EYEBROW_08_FILE = ROOT_DATA_PATH + "/eyebrows/eyebrow8/mind_eyebrows_08.mhpxy"
EYEBROW_09_FILE = ROOT_DATA_PATH + "/eyebrows/eyebrow9/mind_eyebrows_09.mhpxy"
EYEBROW_10_FILE = ROOT_DATA_PATH + "/eyebrows/eyebrow10/mind_eyebrows_10.mhpxy"
EYEBROW_11_FILE = ROOT_DATA_PATH + "/eyebrows/eyebrow11/mind_eyebrows_11.mhpxy"
EYEBROW_12_FILE = ROOT_DATA_PATH + "/eyebrows/eyebrow12/mind_eyebrows_12.mhpxy"
EYEBROW_13_FILE = ROOT_DATA_PATH + "/eyebrows/eyebrow13/mind_eyebrows_13.mhpxy"
EYEBROW_14_FILE = ROOT_DATA_PATH + "/eyebrows/eyebrow14/mind_eyebrows_14.mhpxy"


# FACE CONSTANTS
FACE_AGE_SLIDER_INDX = 0
FACE_HEAD_FAT_SLIDER_INDX = 1
FACE_ANGLE_SLIDER_INDX = 2
FACE_OVAL_SLIDER_INDX = 3
FACE_ROUND_SLIDER_INDX = 4
FACE_RECTANGULAR_SLIDER_INDX = 5
FACE_SQUARE_SLIDER_INDX = 6
FACE_TRIANGULAR_SLIDER_INDX = 7
FACE_INVERTED_TRIANGLE_SLIDER_INDX = 8
FACE_DIAMOND_SLIDER_INDX = 9
FACE_SCALE_DEPTH_SLIDER_INDX = 10


# PEOPLE
PEOPLE_NAME_KEY = "NAME"
PEOPLE_CLOTHES_KEY = "CLOTHES"
PEOPLE_FACIAL_HAIR_KEY = "FACIAL HAIR"
PEOPLE_EYELASHES_KEY = "EYELASHES"
PEOPLE_TONGUE_KEY = "TONGUE"
PEOPLE_TEETH_KEY = "TEETH"
PEOPLE_HAIR_KEY = "HAIR"
PEOPLE_EYEBROW_KEY = "EYEBROW"
PEOPLE_GENDER_KEY = "GENDER"
PEOPLE_AGE_KEY = "AGE"
PEOPLE_WEIGHT_KEY = "WEIGHT"
PEOPLE_MUSCLE_KEY = "MUSCLE"
PEOPLE_HEIGHT_KEY = "HEIGHT"
PEOPLE_BODY_PROPORTION_KEY = "BODY_PROPORTION"
PEOPLE_AFRICAN_KEY = "AFRICAN"
PEOPLE_ASIAN_KEY = "ASIAN"
PEOPLE_CAUCASIAN_KEY = "CAUCASIAN"
PEOPLE_HEAD_TYPE_KEY = "HEAD_TYPE"




#TODO items:
# - add ability to do another movement after one movement.