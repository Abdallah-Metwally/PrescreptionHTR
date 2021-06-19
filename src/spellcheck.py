from fuzzywuzzy import fuzz
import csv
from collections import defaultdict

"""spellcheck main class"""


class SpellCheck:
    """initialization method"""

    def __init__(self, word_dict_file=None, Drugs_List=None):
        """open the dictionary file"""
        self.file = open(word_dict_file, 'r')
        self.DrugsDict = open(Drugs_List, 'r')

        """load the file data in a variable"""
        data = self.file.read()
        drugs = self.DrugsDict.read()

        """store all the words in a list"""
        data = data.split(",")
        drugs = drugs.split(",")

        """remove all the duplicates in the list"""
        data = set(data)
        drugs = set(drugs)

        """store all the words into a class variable dictionary"""
        self.dictionary = list(data)
        self.drugsDict = list(drugs)

    def check(self, string_to_check):
        """store the string to be checked in a class variable"""
        self.string_to_check = string_to_check

    def find(self, string_to_find):
        """store the string to be checked in a class variable"""
        self.string_to_find = string_to_find

    """this method returns the possible suggestions of the correct words"""

    def suggestions(self):
        """store the words of the string to be checked in a list by using a split function"""
        string_words = self.string_to_check.split(",")

        """a list to store all the possible suggestions"""
        suggestions = []

        """loop over the number of words in the string to be checked"""
        for i in range(len(string_words)):

            """loop over words in the dictionary"""
            for name in self.dictionary:

                """if the fuzzywuzzy returns the matched value greater than 80"""
                if fuzz.ratio(string_words[i].lower(), name.lower()) > 80:
                    # append the dict word to the suggestion list
                    suggestions.append(name)
                else:
                    if string_words[i].lower() in name.lower():
                        suggestions.append(name)

        """return the suggestions list"""
        return suggestions

    """this method returns the corrected string of the given input"""

    def correct(self):
        """store the words of the string to be checked in a list by using a split function"""
        string_words = self.string_to_check.split(",")

        """loop over the number of words in the string to be checked"""
        for i in range(len(string_words)):

            """initialize a maximum probability variable to 0"""
            max_percent = 0
            final = ""
            """loop over the words in the dictionary"""
            for name in self.dictionary:

                """calculate the match probability"""
                percent = fuzz.ratio(string_words[i].lower(), name.lower())

                """if the fuzzywuzzy returns the matched value greater than 80"""
                if percent > 80:

                    """if the matched probability is"""
                    if percent > max_percent:
                        """change the original value with the corrected matched value"""
                        string_words[i] = name

                        """change the max percent to the current matched percent"""
                        max_percent = percent
                        final = name

        """return the corrected string"""
        return final
    ##Dictionary####
    def search(self):
        ToFind = self.string_to_find

        res = ""

        for name in range(len(self.drugsDict)):
            if ToFind != "":
                if ToFind in self.drugsDict[name]:
                    res = res.join(self.drugsDict[name])
                    break
            else:
                break

        return res
    ####DrugList####
    def get(self, string_to_get):
        toGet = string_to_get
        l = len(toGet) + 1
        suf = []
        for name in self.dictionary:
            if toGet != "":
                if len(name) > l:

                    percent = fuzz.ratio(toGet.lower(), name[:l-1].lower())
                    if 80 > percent > 60:
                        print(toGet + name + str(percent))
                        suf.append(name)
        return " ".join(suf)
