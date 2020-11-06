import numpy as np
import pandas as pd
import cv2
from constants import *


class matches:

    def __init__(self):
        self.vector = np.zeros(22, dtype=np.float32)
        self.dataframe = pd.read_csv(CFG_DIR + "data.csv")
        self.map = cv2.imread(CFG_DIR + "map.png")
        self.dict = {1: (982, 450), 2: (982, 626), 3: (878, 626), 4: (775, 626), 5: (671, 626), 6: (564, 626),
                     7: (462, 626), 8: (385, 626), 9: (308, 626), 10: (234, 626), 11: (159, 626), 12: (56, 626),
                     13: (56, 450), 14: (56, 325), 15: (56, 155), 16: (75, 30), 17: (180, 30), 18: (263, 30),
                     19: (210, 175), 20: (260, 450), 21: (572, 450), 22: (828, 450)}

    def draw_match(self, room, people):
        """
             Draw detections on the map
             param:
                     room: the detected room
                     people: vector of people (to have ncircles = len(people))
           """
        map_cp = self.map.copy()
        if room != 0:
            pos = self.dict[room]
            cv2.circle(map_cp, pos, 10, (0, 0, 255), thickness=-1)
            cv2.circle(map_cp, (520, 40), 10, (0, 0, 255), thickness=-1)
            cv2.putText(map_cp, "Video maker", (550, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 2)
            if len(people) != 0:
                cv2.circle(map_cp, (520, 70), 5, (255, 0, 0), thickness=-1)
                cv2.putText(map_cp, "Localized people", (550, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 2)
                for idx, x in enumerate(people):
                    new_pos = (pos[0] + 15 * idx, pos[1] + 30)
                    cv2.circle(map_cp, new_pos, 5, (255, 0, 0), thickness=-1)
        else:
            if len(people) == 0:
                cv2.putText(map_cp, "No people detected and can not localize the video maker", (550, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            elif len(people) == 1:
                cv2.putText(map_cp, "Found 1 person, but can not localize it.", (550, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), 2)
            else:
                cv2.putText(map_cp, "Found {} people, but can not localize them.".format(len(people)), (600, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 2)
        cv2.imshow("Map", map_cp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def count_matches(self, name, val):
        """
             Get the match paint-room and paint-name
             param:
                     name: name of retrieved painting
                     val: score of retrieved paiting
            return: name of the retrieved painting
           """
        row = self.dataframe.loc[self.dataframe["Image"] == name]
        room = int(row["Room"])
        img_name = row["Title"]
        self.vector[room] += val
        return img_name[img_name.index[0]]

    def get_result(self):
        """
             Get the current room

        """
        index = np.argmax(self.vector)
        self.vector = np.zeros(22, dtype=np.float32)
        if index == 0:
            print("No idea where I am.")
            return 0
        else:
            print("I'm in room number: {}".format(index))
        return index
