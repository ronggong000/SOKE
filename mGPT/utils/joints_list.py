SMPLX_JOINT_LANDMARK_NAMES = ["pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
    "left_eye_smplhf",
    "right_eye_smplhf",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3", #End of Joint
    "nose", #Start of landmark
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
    "right_eye_brow1",
    "right_eye_brow2",
    "right_eye_brow3",
    "right_eye_brow4",
    "right_eye_brow5",
    "left_eye_brow5",
    "left_eye_brow4",
    "left_eye_brow3",
    "left_eye_brow2",
    "left_eye_brow1",
    "nose1",
    "nose2",
    "nose3",
    "nose4",
    "right_nose_2",
    "right_nose_1",
    "nose_middle",
    "left_nose_1",
    "left_nose_2",
    "right_eye1",
    "right_eye2",
    "right_eye3",
    "right_eye4",
    "right_eye5",
    "right_eye6",
    "left_eye4",
    "left_eye3",
    "left_eye2",
    "left_eye1",
    "left_eye6",
    "left_eye5",
    "right_mouth_1",
    "right_mouth_2",
    "right_mouth_3",
    "mouth_top",
    "left_mouth_3",
    "left_mouth_2",
    "left_mouth_1",
    "left_mouth_5",  # 59 in OpenPose output
    "left_mouth_4",  # 58 in OpenPose output
    "mouth_bottom",
    "right_mouth_4",
    "right_mouth_5",
    "right_lip_1",
    "right_lip_2",
    "lip_top",
    "left_lip_2",
    "left_lip_1",
    "left_lip_3",
    "lip_bottom",
    "right_lip_3",
    "right_contour_1", #start of Face contour
    "right_contour_2",
    "right_contour_3",
    "right_contour_4",
    "right_contour_5",
    "right_contour_6",
    "right_contour_7",
    "right_contour_8",
    "contour_middle",
    "left_contour_8",
    "left_contour_7",
    "left_contour_6",
    "left_contour_5",
    "left_contour_4",
    "left_contour_3",
    "left_contour_2",
    "left_contour_1",
]
SMPLX_JOINT_NAMES = [
    'pelvis','left_hip','right_hip','spine1','left_knee','right_knee','spine2','left_ankle','right_ankle','spine3', 
    'left_foot','right_foot','neck','left_collar','right_collar','head','left_shoulder','right_shoulder','left_elbow', 
    'right_elbow','left_wrist','right_wrist',
    'jaw','left_eye_smplhf','right_eye_smplhf','left_index1','left_index2','left_index3','left_middle1','left_middle2',
    'left_middle3','left_pinky1','left_pinky2','left_pinky3','left_ring1','left_ring2','left_ring3','left_thumb1',
    'left_thumb2','left_thumb3','right_index1','right_index2','right_index3','right_middle1','right_middle2',
    'right_middle3','right_pinky1','right_pinky2','right_pinky3','right_ring1','right_ring2','right_ring3',
    'right_thumb1','right_thumb2','right_thumb3'
]

# 选择的关节索引（排除下半身和面部关节）
SELECTED_JOINT_INDICES = [
    3,  # spine1
    6,  # spine2
    9,  # spine3
    12, # neck
    13, # left_collar
    14, # right_collar
    15, # head
    16, # left_shoulder
    17, # right_shoulder
    18, # left_elbow
    19, # right_elbow
    20, # left_wrist
    21, # right_wrist
    # 左手指关节
    25, 26, 27,  # left_index
    28, 29, 30,  # left_middle
    31, 32, 33,  # left_pinky
    34, 35, 36,  # left_ring
    37, 38, 39,  # left_thumb
    # 右手指关节
    40, 41, 42,  # right_index
    43, 44, 45,  # right_middle
    46, 47, 48,  # right_pinky
    49, 50, 51,  # right_ring
    52, 53, 54,  # right_thumb
]
SELECTED_JOINT_LANDMARK_INDICES = [

    3,  # spine1
    6,  # spine2
    9,  # spine3
    12, # neck
    13, # left_collar
    14, # right_collar
    15, # head
    16, # left_shoulder
    17, # right_shoulder
    18, # left_elbow
    19, # right_elbow
    20, # left_wrist
    21, # right_wrist
    # 左手指关节
    25, 26, 27,  # left_index
    28, 29, 30,  # left_middle
    31, 32, 33,  # left_pinky
    34, 35, 36,  # left_ring
    37, 38, 39,  # left_thumb
    # 右手指关节
    40, 41, 42,  # right_index
    43, 44, 45,  # right_middle
    46, 47, 48,  # right_pinky
    49, 50, 51,  # right_ring
    52, 53, 54,  # right_thumb
    #landmark

    55, #nose
    58, #"right_ear",
    59, #"left_ear",

    66, #"left_thumb"
    67, #"left_index"
    68, #"left_middle"
    69, #"left_ring"    
    70, #"left_pinky"
    71, #"right_thumb"
    72, #"right_index"
    73, #"right_middle"
    74, #"right_ring"
    75, #"right_pinky"
]

SELECTED_JOINT_INDICES_BODY_ONLY = [
    3,  # spine1
    6,  # spine2
    9,  # spine3
    12, # neck
    13, # left_collar
    14, # right_collar
    15, # head
    16, # left_shoulder
    17, # right_shoulder
    18, # left_elbow
    19, # right_elbow
    20, # left_wrist
    21, # right_wrist
]

SELECTED_JOINT_INDICES_HAND_ONLY = [
    # 左手指关节
    25, 26, 27,  # left_index
    28, 29, 30,  # left_middle
    31, 32, 33,  # left_pinky
    34, 35, 36,  # left_ring
    37, 38, 39,  # left_thumb
    # 右手指关节
    40, 41, 42,  # right_index
    43, 44, 45,  # right_middle
    46, 47, 48,  # right_pinky
    49, 50, 51,  # right_ring
    52, 53, 54,  # right_thumb
]

SELECTED_JOINT_LANDMARK_INDICES_BODY_ONLY = [

    3,  # spine1
    6,  # spine2
    9,  # spine3
    12, # neck
    13, # left_collar
    14, # right_collar
    15, # head
    16, # left_shoulder
    17, # right_shoulder
    18, # left_elbow
    19, # right_elbow
    20, # left_wrist
    21, # right_wrist

    #landmark
    55, #nose
    58, #"right_ear",
    59, #"left_ear",
]
SELECTED_JOINT_LANDMARK_INDICES_HAND_ONLY = [
    # 左手指关节
    25, 26, 27,  # left_index
    28, 29, 30,  # left_middle
    31, 32, 33,  # left_pinky
    34, 35, 36,  # left_ring
    37, 38, 39,  # left_thumb
    # 右手指关节
    40, 41, 42,  # right_index
    43, 44, 45,  # right_middle
    46, 47, 48,  # right_pinky
    49, 50, 51,  # right_ring
    52, 53, 54,  # right_thumb
    #landmark
    66, #"left_thumb"
    67, #"left_index"
    68, #"left_middle"
    69, #"left_ring"    
    70, #"left_pinky"
    71, #"right_thumb"
    72, #"right_index"
    73, #"right_middle"
    74, #"right_ring"
    75, #"right_pinky"
]

SELECTED_JOINT_LANDMARK_INDICES_NEIGHBOR_LIST=[

    [1],#     0 3,  # spine1
    [0,2],#     1 6,  # spine2
    [1,3,4,5],#     2 9,  # spine3
    [2,6],#     3 12, # neck
    [2,7],#     4 13, # left_collar
    [2,8],#     5 14, # right_collar
    [3,43,44,45],#     6 15, # head
    [4,9],#     7 16, # left_shoulder
    [5,10],#     8 17, # right_shoulder
    [7,11],#     9 18, # left_elbow
    [8,12],#     10 19, # right_elbow
    [9,13,16,19,22,25],#     11 20, # left_wrist
    [10,28,31,34,37,40],#     12 21, # right_wrist
    
    [11,14],#     13 25, left_index1
    [13,15],# 14 26, left_index2
    [14,47],# 15 27,  # left_index3

    [11,17],#     16 28,left_middle1
    [16,18],#  17 29,left_middle2
    [17,48],#  18 30,  # left_middle3

    [11,20],#     19 31,left_pinky1
    [19,21],#  20 32,left_pinky2
    [20,50],#  21 33,  # left_pinky3

    [11,23],#     22 34,left_ring1
    [22,24],#  23 35, left_ring2
    [23,49],# 24 36,  # left_ring3

    [11,26],#     25 37,left_thumb1
    [25,27],#  26 38,left_thumb2
    [26,46],#  27 39,  # left_thumb3
    #     # 右手指关节
    [12,29],#     28 40,right_index1
    [28,30],# 29 41,right_index2
    [29,52],# 30 42,  # right_index3

    [12,32], # 31 43,right_middle1
    [31,33],  # 32 44,right_middle2
    [32,53],   # 33 45,  # right_middle3

    [12,35], # 34 46,right_pinky1
    [34,36], # 35 47,right_pinky2
    [35,55], #  36 48,  # right_pinky3

    [12,38], # 37 49,right_ring1
    [37,39], # 38 50, right_ring2
    [38,54], # 39 51,  # right_ring3

    [12,41], #40 52, right_thumb1
    [40,42], #41 53, right_thumb2
    [41,51], #42 54,  # right_thumb3
    #     #landmark
    [6],#     43 55, #nose
    [6],#     44 58, #"right_ear",
    [6],#     45 59, #"left_ear",
    [27],#     46 66, #"left_thumb4"
    [15],#     47 67, #"left_index4"
    [18],#     48 68, #"left_middle4"
    [24],#     49 69, #"left_ring4"    
    [21],#     50 70, #"left_pinky4"

    [42],#     51 71, #"right_thumb4"
    [30],#     52 72, #"right_index4"
    [33],#     53 73, #"right_middle4"
    [39],#     54 74, #"right_ring4"
    [36],#     55 75, #"right_pinky4"

]
SELECTED_JOINT_LANDMARK_INDICES_LANDMARK_INDEX=[

    #landmark_index
    43,#     43 55, #nose
    44,#     44 58, #"right_ear",
    45,#     45 59, #"left_ear",
    46,#     46 66, #"left_thumb4"
    47,#     47 67, #"left_index4"
    48,#     48 68, #"left_middle4"
    49,#     49 69, #"left_ring4"    
    50,#     50 70, #"left_pinky4"

    51,#     51 71, #"right_thumb4"
    52,#     52 72, #"right_index4"
    53,#     53 73, #"right_middle4"
    54,#     54 74, #"right_ring4"
    55,#     55 75, #"right_pinky4"
]
SELECTED_JOINT_LANDMARK_BODY_EVAL=[
    0, #3,  # spine1
    1, #6,  # spine2
    2, #9,  # spine3
    3, #12, # neck
    4, #13, # left_collar
    5, #14, # right_collar
    6, #15, # head
    7, #16, # left_shoulder
    8, #17, # right_shoulder
    9, #18, # left_elbow
    10, #19, # right_elbow
    11, #20, # left_wrist
    12, #21, # right_wrist

    43, #55, #nose
    44, #58, #"right_ear",
    45, #59, #"left_ear",
]
SELECTED_JOINT_LANDMARK_LHAND_EVAL=[
    #left hand
    13, #25, left_index1
    14, #26, left_index2
    15, #27,  # left_index3
    
    16, #28,left_middle1
    17, #29,left_middle2
    18, #30,  # left_middle3

    19,# 31,left_pinky1
    20,# 32,left_pinky2
    21,# 33,  # left_pinky3

    22,# 34,left_ring1
    23,# 35, left_ring2
    24,# 36,  # left_ring3

    25,# 37,left_thumb1
    26,# 38,left_thumb2
    27,# 39,  # left_thumb3

    46,# 66, #"left_thumb4"
    47,# 67, #"left_index4"
    48,# 68, #"left_middle4"
    49,# 69, #"left_ring4"    
    50,# 70, #"left_pinky4"
]
SELECTED_JOINT_LANDMARK_RHAND_EVAL=[
    #right hand
    28,# 40,right_index1
    29,# 41,right_index2
    30,# 42,  # right_index3

    31,# 43,right_middle1
    32,# 44,right_middle2
    33,# 45,  # right_middle3

    34,# 46,right_pinky1
    35,# 47,right_pinky2
    36,# 48,  # right_pinky3

    37,# 49,right_ring1
    38,# 50, right_ring2
    39,# 51,  # right_ring3

    40,# 52, right_thumb1
    41,# 53, right_thumb2
    42,# 54,  # right_thumb3

    51, #71, #"right_thumb4"
    52, #72, #"right_index4"
    53,# 73, #"right_middle4"
    54,# 74, #"right_ring4"
    55,# 75, #"right_pinky4"
]



SELECTED_JOINT_INDICES_NEIGHBOR_LIST=[

    [1],#     0 3,  # spine1
    [0,2],#     1 6,  # spine2
    [1,3,4,5],#     2 9,  # spine3
    [2,6],#     3 12, # neck
    [2,7],#     4 13, # left_collar
    [2,8],#     5 14, # right_collar
    [3],#     6 15, # head
    [4,9],#     7 16, # left_shoulder
    [5,10],#     8 17, # right_shoulder
    [7,11],#     9 18, # left_elbow
    [8,12],#     10 19, # right_elbow
    [9,13,16,19,22,25],#     11 20, # left_wrist
    [10,28,31,34,37,40],#     12 21, # right_wrist
    
    [11,14],#     13 25, left_index1
    [13,15],# 14 26, left_index2
    [14],# 15 27,  # left_index3

    [11,17],#     16 28,left_middle1
    [16,18],#  17 29,left_middle2
    [17],#  18 30,  # left_middle3

    [11,20],#     19 31,left_pinky1
    [19,21],#  20 32,left_pinky2
    [20],#  21 33,  # left_pinky3

    [11,23],#     22 34,left_ring1
    [22,24],#  23 35, left_ring2
    [23],# 24 36,  # left_ring3

    [11,26],#     25 37,left_thumb1
    [25,27],#  26 38,left_thumb2
    [26],#  27 39,  # left_thumb3
    #     # 右手指关节
    [12,29],#     28 40,right_index1
    [28,30],# 29 41,right_index2
    [29],# 30 42,  # right_index3

    [12,32], # 31 43,right_middle1
    [31,33],  # 32 44,right_middle2
    [32],   # 33 45,  # right_middle3

    [12,35], # 34 46,right_pinky1
    [34,36], # 35 47,right_pinky2
    [35], #  36 48,  # right_pinky3

    [12,38], # 37 49,right_ring1
    [37,39], # 38 50, right_ring2
    [38], # 39 51,  # right_ring3

    [12,41], #40 52, right_thumb1
    [40,42], #41 53, right_thumb2
    [41], #42 54,  # right_thumb3
    
]