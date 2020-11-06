from matcher import matches
from paint_rectification import Rectification
from paint_retrieval import Retrieve
from constants import *
from util import *
from yolo import Yolo


class serviceLocator():
    """Facade class to expose services"""

    def __init__(self):
        print("Network initialization...")
        self.net = Yolo(names=CFG_DIR + "paintings.names", weights=CFG_DIR + "paintings.weights",
                        cfg=CFG_DIR + "yolov3_custom.cfg")
        self.people_detector = Yolo(names=CFG_DIR + "person.names", weights=CFG_DIR + "yolov3.weights",
                                    cfg=CFG_DIR + "yolov3.cfg")
        self.paint_rectifier = Rectification()
        self.paint_retriever = Retrieve()
        self.m = matches()

    def check_db(self):
        self.paint_retriever.check_db()

    def analyse_video(self, first_frame, cap, skip, topn, outsize):
        permanenza = np.zeros(22)

        if first_frame > cap.shape[0]:
            print("First frame is out of range. Changed to 0.")
            first_frame = 0
        for frame in range(first_frame, cap.shape[0], skip):
            im = cap[frame]
            if im is None:
                continue
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            people = self.people_detector.detect(im)
            bblist = self.net.detect(im)
            # filter for paintings containing people
            people = people_filter(people, bblist)
            img_boxed = im.copy()
            print("****************************** NEW FRAME! {}/{} ******************************".format(frame,
                                                                                                          cap.shape[0]))
            print("PAINTINGS: \n")
            if bblist is not None:  # If at least 1 paint detected
                img_boxed = draw_boxes(img_boxed, bblist, color=(0, 255, 0), label="Painting")
                img_boxed = draw_boxes(img_boxed, people)
                for index, box in enumerate(bblist):
                    y1, y2, x1, x2 = get_bounding_box(box)
                    sliced = im[x1:x2, y1:y2, :]
                    warped_img, found = self.paint_rectifier.perspective_transform(sliced)
                    names, distances = self.paint_retriever.match_img_db(warped_img)
                    distances = format_v(distances)
                    print('Result for frame number {} and paint number {}'.format(frame, index))
                    print_painting(box)
                    if distances[0] == -1:
                        print('Trying to retrieve with no Rectification...')
                        names, distances = self.paint_retriever.match_img_db(sliced)
                        distances = format_v(distances)
                        if distances[0] == -1:
                            print("Image is not in db or score is too low.")
                            show(img_boxed, sliced, warped_img, names, None, flag=1, outsize=outsize)
                        else:
                            print('Retrieved correctly with no Rectification.')
                            for i in range(topn):
                                print('Match {}, {} '.format(names[i], distances[i]))
                            curr_name = self.m.count_matches(names[0], distances[0])
                            show(img_boxed, sliced, None, names, curr_name, flag=0, outsize=outsize)
                    else:
                        for i in range(topn):
                            print('Match {}, {} '.format(names[i], distances[i]))
                        curr_name = self.m.count_matches(names[0], distances[0])
                        show(img_boxed, sliced, warped_img, names, curr_name, flag=0, outsize=outsize)
                    print("------------------------------------------------------------")
            else:
                print("No painting detected in this frame.")
            print("PEOPLE: \n")
            if people:
                img_boxed = draw_boxes(im, people)
                print_people(people)
            else:  # Nothing detected
                print("No people detected in this frame.")
            room = self.m.get_result()
            if room != 0:
                permanenza[room] += 1
            self.m.draw_match(np.argmax(permanenza), people)
        print("We were most of the time in room: {}".format(np.argmax(permanenza)))

    def create_video(self, cap, out_name):
        print("*** Start video writing ***")
        out = cv2.VideoWriter("{}.avi".format(out_name), cv2.VideoWriter_fourcc(*'XVID'), 30., (1280, 720))
        for frame in range(0, cap.shape[0]):
            im = cap[frame]
            if im is None:
                continue
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            people = self.people_detector.detect(im)
            bblist = self.net.detect(im)
            people = people_filter(people, bblist)
            img_boxed = im.copy()
            if bblist is not None:  # If at least 1 paint detection
                img_boxed = draw_boxes(img_boxed, bblist, color=(0, 255, 0), label="Painting")
                img_boxed = draw_boxes(img_boxed, people)
            print("Written frame number: {} / {}".format(frame, cap.shape[0]))
            img_boxed = cv2.resize(img_boxed, (1280, 720))
            out.write(img_boxed.astype('uint8'))
        out.release()
        print("*** Finish video writing ***")
