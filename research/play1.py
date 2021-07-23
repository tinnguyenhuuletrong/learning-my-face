import cv2
import numpy as np
import face_recognition
from time import sleep
import pickle
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances


def face_detect(rgb_frame):
    face_locations = face_recognition.face_locations(rgb_frame)

    # no faces
    if len(face_locations) <= 0:
        return [[], []]
    face_encodings = face_recognition.face_encodings(
        rgb_frame, known_face_locations=face_locations, model="large"
    )

    return [face_locations, face_encodings]


def draw_face_rec(frame, face_locations, color=(0, 255, 0)):
    for enc in face_locations:
        (top, right, bottom, left) = enc
        cv2.rectangle(frame, (left, top), (right, bottom), color)
    return frame


def process_video(path, id):

    cap = cv2.VideoCapture(path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (frame_width, frame_height)
    print(size, fps)
    result = cv2.VideoWriter('%s.mp4' % id,
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, size)
    representations = []
    frame_count = 0
    while(True):
        ret, frame = cap.read()
        if ret == True:
            [face_locations, face_encodings] = face_detect(frame)
            if len(face_locations) > 0:
                draw_face_rec(frame, face_locations)
                representations.extend(face_encodings)

            result.write(frame)

            frame_count += 1
            print(frame_count)
        else:
            break

    with open('./%s.pk' % id, "wb") as f:
        pickle.dump(representations, f)

    cap.release()
    result.release()
    cv2.destroyAllWindows()


def load_face_data(id):
    print("id", id)
    representations = []
    with open('./%s.pk' % id, "rb") as f:
        representations = pickle.load(f)
    representations = np.array(representations)

    distance = cosine_distances(representations, representations)
    print('CosineDistance', np.min(distance), np.max(distance))

    distance = euclidean_distances(representations, representations)
    print('EuclideanDistance', np.min(distance), np.max(distance))


def load_model(id):
    with open('./%s.pk' % id, "rb") as f:
        representations = pickle.load(f)
        representations = np.array(representations)
        return representations


def compare(enc, id, debug=True, representations=None):
    if representations is None:
        representations = load_model(id)

    cos_distance = cosine_distances(representations, enc)
    euc_distance = euclidean_distances(representations, enc)
    if debug:
        print("id", id)
        print('\tCosineDistance', np.min(cos_distance), np.max(cos_distance))
        print('\tEuclideanDistance', np.min(
            euc_distance), np.max(euc_distance))

    # https://github.com/serengil/deepface/blob/af13e4558fcc873fc60002a1512b975e97a30813/deepface/commons/distance.py#L24
    return np.max(cos_distance) < 0.07


def test_img(img_path, id):
    print(img_path)
    frame = cv2.imread(img_path)
    [location, enc] = face_detect(frame)
    index = 0
    for face in enc:
        print("-------%d-------" % index)
        index += 1
        res = compare([face], id, debug=True)
        print("\t", res)

# process_video("./my_face.mp4", "1")
# process_video("./test.mp4", "2")
# process_video("./test_trim.mp4", "3")
# load_face_data("1")
# load_face_data("2")
# load_face_data("3")

# --------
# Self test
# --------
# id 1
# CosineDistance 0.0 0.024210203362418747
# EuclideanDistance 0.0 0.3057140925539441
# id 2
# CosineDistance 0.0 0.07679001286737797
# EuclideanDistance 0.0 0.5337827995058241
# id 3
# CosineDistance 0.0 0.019538628657974066
# EuclideanDistance 0.0 0.2660321618995691


# test_img('/Users/admin/Downloads/IMG_3346.JPG', "3")
# test_img('/Users/admin/Downloads/brother.jpg', "3")
# test_img('/Users/admin/Documents/tmp/face-api-playground/tin_an.jpg', "3")
# test_img('/Users/admin/Documents/tmp/face-api-playground/tin_ton.jpg', "3")
# test_img('/Users/admin/Documents/tmp/magick_play/in-doc.jpg', "3")
# test_img('/Users/admin/Desktop/passport/mrz_passport_3.jpeg', "3")
# test_img('/Users/admin/Documents/tmp/python_play/face/dataset/vgg_face_dataset/faces/Recep_Tayyip_Erdogan/Recep_Tayyip_Erdogan_0030.jpg', "3")
