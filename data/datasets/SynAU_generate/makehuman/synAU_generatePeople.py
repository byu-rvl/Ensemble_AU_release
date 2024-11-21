import random


class synAU_generatePeople:
    def __init__(self, numPeople):
        self.numPeople = numPeople
        self.options_clothes = ["CARGO_PANTS_FILE", "JEAN_SHORTS_FILE", "POLO_SHIRT_FILE"]
        self.options_facialHair = ["MOUSTACHE_FILE", "FULL_BEARD_FILE"]
        self.options_eyebrows = ["EYEBROW_01_FILE", "EYEBROW_02_FILE", "EYEBROW_03_FILE", "EYEBROW_04_FILE",
                                 "EYEBROW_05_FILE", "EYEBROW_06_FILE", "EYEBROW_07_FILE", "EYEBROW_08_FILE",
                                 "EYEBROW_09_FILE", "EYEBROW_10_FILE", "EYEBROW_11_FILE", "EYEBROW_12_FILE",
                                 "EYEBROW_13_FILE", "EYEBROW_14_FILE"]
        self.options_female  = 0
        self.options_male = 1
        self.options_male_hair = ["HAIR_M01_FILE", "HAIR_M04_FILE", "HAIR_M05_FILE"]
        self.options_female_hair = ["HAIR_F01_FILE", "HAIR_F1_FILE"]
        self.options_resolution_0_1 = 10

    def listToStr(self, myList):
        listToStr = "["
        for i in range(0, len(myList)):
            if i < len(myList) - 1:
                listToStr += str(myList[i]) + ", "
            else:
                listToStr += str(myList[i])
        listToStr += "]"

        return listToStr

    def selectItems(self, myList, numSelect=1):
        items = random.sample(myList, numSelect)
        return items

    def zeroToOne(self):
        return self.selectItems(list(range(0, self.options_resolution_0_1 + 1)), 1)[0] / self.options_resolution_0_1

    def makePeople(self):
        allPeople = []
        for i in range(0,self.numPeople):
            thisPersonName = "PERSON_" + str(i).zfill(3)
            allPeople.append(thisPersonName)
            personString = thisPersonName + " = {}\n"
            personString += thisPersonName + "[PEOPLE_NAME_KEY] = \"" + str(thisPersonName) + "\"\n"
            personString += thisPersonName + "[PEOPLE_CLOTHES_KEY] = [CARGO_PANTS_FILE, JEAN_SHORTS_FILE, POLO_SHIRT_FILE]\n"
            personString += thisPersonName + "[PEOPLE_EYELASHES_KEY] = [EYELASHES_01_FILE]\n"
            personString += thisPersonName + "[PEOPLE_TONGUE_KEY] = [TONGUE_FACS_TONGUE_FILE]\n"
            personString += thisPersonName + "[PEOPLE_TEETH_KEY] = [TEETH_03_FILE]  # OTHER OPTIONS ABOVE IF HELPFUL\n"
            personString += thisPersonName + "[PEOPLE_EYEBROW_KEY] = [" + self.selectItems(self.options_eyebrows)[0] + "]\n"
            gender = self.selectItems([self.options_female, self.options_male])[0]
            personString += thisPersonName + "[PEOPLE_GENDER_KEY] = " + str(float(gender)) + "\n"
            if gender == self.options_female:
                personString += thisPersonName + "[PEOPLE_FACIAL_HAIR_KEY] = []\n"
                personString += thisPersonName + "[PEOPLE_HAIR_KEY] = [" + self.selectItems(self.options_female_hair)[0] + "]\n"
            else:
                numFacialHair = self.selectItems(list(range(0, len(self.options_facialHair))))[0]
                facialHair = self.selectItems(self.options_facialHair, numFacialHair)

                personString += thisPersonName + "[PEOPLE_FACIAL_HAIR_KEY] = " + self.listToStr(facialHair) + "\n"
                personString += thisPersonName + "[PEOPLE_HAIR_KEY] = [" + self.selectItems(self.options_male_hair)[0] + "]\n"
            age = self.zeroToOne()
            weight = self.zeroToOne()
            muscle = self.zeroToOne()
            height = self.zeroToOne()
            bodyProp = self.zeroToOne()
            personString += thisPersonName + "[PEOPLE_AGE_KEY] = " + str(age) + "# 0.0 TO 1.0 note: 1.0 is for a 70 year old.\n"
            personString += thisPersonName + "[PEOPLE_WEIGHT_KEY] = " + str(weight) + "  # 0.0 TO 1.0\n"
            personString += thisPersonName + "[PEOPLE_MUSCLE_KEY] = " + str(muscle) + "  # 0.0 TO 1.0\n"
            personString += thisPersonName + "[PEOPLE_HEIGHT_KEY] = " + str(height) + "  # 0.0 TO 1.0\n"
            personString += thisPersonName + "[PEOPLE_BODY_PROPORTION_KEY] = " + str(bodyProp) + "  # 0.0 TO 1.0\n"
            race = self.selectItems(list(range(0,3)))[0] #select one of three races.
            if race == 0:
                personString += thisPersonName + "[PEOPLE_AFRICAN_KEY] = 1.0  # 0.0 TO 1.0 NOTE!!!! ONLY SUPPORTED TO DO ONE RACE AT A TIME. OTHERWISE CAN'T GAURENTEE WHAT IT WILL MAKE.\n"
                personString += thisPersonName + "[PEOPLE_ASIAN_KEY] = 0.0  # 0.0 TO 1.0 NOTE!!!! ONLY SUPPORTED TO DO ONE RACE AT A TIME. OTHERWISE CAN'T GAURENTEE WHAT IT WILL MAKE.\n"
                personString += thisPersonName + "[PEOPLE_CAUCASIAN_KEY] = 0.0  # 0.0 TO 1.0 NOTE!!!! ONLY SUPPORTED TO DO ONE RACE AT A TIME. OTHERWISE CAN'T GAURENTEE WHAT IT WILL MAKE.\n"
            elif race == 1:
                personString += thisPersonName + "[PEOPLE_AFRICAN_KEY] = 0.0  # 0.0 TO 1.0 NOTE!!!! ONLY SUPPORTED TO DO ONE RACE AT A TIME. OTHERWISE CAN'T GAURENTEE WHAT IT WILL MAKE.\n"
                personString += thisPersonName + "[PEOPLE_ASIAN_KEY] = 1.0  # 0.0 TO 1.0 NOTE!!!! ONLY SUPPORTED TO DO ONE RACE AT A TIME. OTHERWISE CAN'T GAURENTEE WHAT IT WILL MAKE.\n"
                personString += thisPersonName + "[PEOPLE_CAUCASIAN_KEY] = 0.0  # 0.0 TO 1.0 NOTE!!!! ONLY SUPPORTED TO DO ONE RACE AT A TIME. OTHERWISE CAN'T GAURENTEE WHAT IT WILL MAKE.\n"
            else:
                personString += thisPersonName + "[PEOPLE_AFRICAN_KEY] = 0.0  # 0.0 TO 1.0 NOTE!!!! ONLY SUPPORTED TO DO ONE RACE AT A TIME. OTHERWISE CAN'T GAURENTEE WHAT IT WILL MAKE.\n"
                personString += thisPersonName + "[PEOPLE_ASIAN_KEY] = 0.0  # 0.0 TO 1.0 NOTE!!!! ONLY SUPPORTED TO DO ONE RACE AT A TIME. OTHERWISE CAN'T GAURENTEE WHAT IT WILL MAKE.\n"
                personString += thisPersonName + "[PEOPLE_CAUCASIAN_KEY] = 1.0  # 0.0 TO 1.0 NOTE!!!! ONLY SUPPORTED TO DO ONE RACE AT A TIME. OTHERWISE CAN'T GAURENTEE WHAT IT WILL MAKE.\n"
            headFat = self.zeroToOne()
            ovalFace = self.zeroToOne()
            roundFace = self.zeroToOne()
            rectFace = self.zeroToOne()
            diamondFace = self.zeroToOne()
            personString += thisPersonName + "[PEOPLE_HEAD_TYPE_KEY] = [(FACE_AGE_SLIDER_INDX, " + str(age) + "), (FACE_HEAD_FAT_SLIDER_INDX, " + str(headFat) + ")," \
                                               "(FACE_OVAL_SLIDER_INDX, " + str(ovalFace) + "), (FACE_ROUND_SLIDER_INDX, " + str(roundFace) + ")," \
                                               "(FACE_RECTANGULAR_SLIDER_INDX, " + str(rectFace) + "), (FACE_DIAMOND_SLIDER_INDX," \
                                                                                     "" + str(diamondFace) + ")]  # OTHER OPTIONS ABOVE. FOR DIFFERENT INDEXES. i THINK THESE ARE THE ONES WE WILL USE.\n"

            print(personString)
        print("USE_PEOPLE = " + self.listToStr(allPeople))

if __name__ == "__main__":
    numPeople = 200
    helper = synAU_generatePeople(numPeople)
    helper.makePeople()