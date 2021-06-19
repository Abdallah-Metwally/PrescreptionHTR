from __future__ import division
from __future__ import print_function
import cv2
import editdistance
from DataLoader import DataLoader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess
from WordSegmentation import wordSegmentation, prepareImg
from spellcheck import SpellCheck
import PreProccessing
import os
from num import NUMDECODE
import shutil

class FilePaths:
    """ filenames and paths to data """
    fnCharList = '../model/charList.txt'
    fnAccuracy = '../model/accuracy.txt'
    fnTrain = '../data/'
    fnCorpus = '../data/corpus.txt'
    scanned = 'scanned.jpg'

#####Localization#############
def prescriptionseg(scanned):
    img = cv2.imread(scanned)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    res = []

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        area = cv2.contourArea(c)
        if 400000 > area > 5000:
            if len(approx) == 4:
                currBox = cv2.boundingRect(c)
                x, y, w, h = currBox
                recArea = w * h
                currImg = img[y:y + h, x:x + w]
                if 1.1 > (area / recArea) > 0.9:
                    res.append((currBox, currImg))
    res = sorted(res, key=lambda entry: entry[0][0])

    for (j, w) in enumerate(res):
        (wordBox, wordImg) = w
        x, y, w, h = wordBox
        cv2.imwrite('../presdrug/%d.png' % j, wordImg)


def train(model, loader):
    """train NN"""
    epoch = 0  # number of training epochs since start
    bestCharErrorRate = float('inf')  # best validation character error rate
    noImprovementSince = 0  # number of epochs no improvement of character error rate occurred
    earlyStopping = 5  # stop training after this number of epochs without improvement
    while True:
        epoch += 1
        print('Epoch:', epoch)

        # train
        print('Train NN')
        loader.trainSet()
        while loader.hasNext():
            iterInfo = loader.getIteratorInfo()
            batch = loader.getNext()
            loss = model.trainBatch(batch)
            print('Batch:', iterInfo[0], '/', iterInfo[1], 'Loss:', loss)

        # validate
        charErrorRate = validate(model, loader)

        # if best validation accuracy so far, save model parameters
        if charErrorRate < bestCharErrorRate:
            print('Character error rate improved, save model')
            bestCharErrorRate = charErrorRate
            noImprovementSince = 0
            model.save()
            open(FilePaths.fnAccuracy, 'w').write(
                'Validation character error rate of saved model: %f%%' % (charErrorRate * 100.0))
        else:
            print('Character error rate not improved')
            noImprovementSince += 1

        # stop training if no more improvement in the last x epochs
        if noImprovementSince >= earlyStopping:
            print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
            break


def validate(model, loader):
    """validate NN"""
    print('Validate NN')
    loader.validationSet()
    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0
    numWordTotal = 0
    while loader.hasNext():
        iterInfo = loader.getIteratorInfo()
        print('Batch:', iterInfo[0], '/', iterInfo[1])
        batch = loader.getNext()
        (recognized, _) = model.inferBatch(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
            numWordTotal += 1
            dist = editdistance.eval(recognized[i], batch.gtTexts[i])
            numCharErr += dist
            numCharTotal += len(batch.gtTexts[i])
            print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gtTexts[i] + '"', '->',
                  '"' + recognized[i] + '"')

    # print validation1997 result
    charErrorRate = numCharErr / numCharTotal
    wordAccuracy = numWordOK / numWordTotal
    print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate * 100.0, wordAccuracy * 100.0))
    return charErrorRate

####Modes
def segment(mode, i=None, f=None):
    """reads images from data/ and outputs the word-segmentation to out/"""
    if mode == 1:
        if os.path.exists('../out/toNN.png'):
            shutil.rmtree('../out/toNN.png')
        # read image, prepare it by resizing it to fixed height and converting it to grayscale
        img = prepareImg(cv2.imread('../data/to.png'))

        # execute segmentation with given parameters
        # -kernelSize: size of filter kernel (odd integer)
        # -sigma: standard deviation of Gaussian function used for filter kernel
        # -theta: approximated width/height ratio of words, filter function is distorted by this factor
        # - minArea: ignore word candidates smaller than specified area
        res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=350)

        # write output to 'out/inputFileName' directory
        if not os.path.exists('../out/toNN.png'):
            os.mkdir('../out/toNN.png')

        # iterate over all segmented words
        # print('Segmented into %d words' % len(res))
        for (j, w) in enumerate(res):
            (wordBox, wordImg) = w
            (x, y, w, h) = wordBox
            cv2.imwrite('../out/toNN.png/%d.png' % j, wordImg)  # save word
            cv2.rectangle(img, (x, y), (x + w, y + h), 0, 1)  # draw bounding box in summary image

        # output summary image with bounding boxes around words
        # cv2.imwrite('../out/toNN.png/summary.png', img)
    if mode == 2:
        # read image, prepare it by resizing it to fixed height and converting it to grayscale
        img = prepareImg(cv2.imread('../presdrug/%s' % i))

        # execute segmentation with given parameters
        # -kernelSize: size of filter kernel (odd integer)
        # -sigma: standard deviation of Gaussian function used for filter kernel
        # -theta: approximated width/height ratio of words, filter function is distorted by this factor
        # - minArea: ignore word candidates smaller than specified area
        res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=350)

        # write output to 'out/inputFileName' directory
        if not os.path.exists('../out/toNN.png'):
            os.mkdir('../out/toNN.png')
        if not os.path.exists('../out/toNN.png/%d' % f):
            os.mkdir('../out/toNN.png/%d' % f)

        # iterate over all segmented words
        # print('Segmented into %d words' % len(res))
        for (j, w) in enumerate(res):
            (wordBox, wordImg) = w
            (x, y, w, h) = wordBox
            cv2.imwrite('../out/toNN.png/%d/%d.png' % (f, j), wordImg)  # save word
            cv2.rectangle(img, (x, y), (x + w, y + h), 0, 1)  # draw bounding box in summary image

        # output summary image with bounding boxes around words
        # cv2.imwrite('../out/toNN.png/summary.png', img)


def infer(model, fnImg):
    """recognize text in image provided by file path"""
    img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
    batch = Batch(None, [img])
    (recognized, probability) = model.inferBatch(batch, True)
    # print('Probability:', '"' + str(round(float(probability[0]) * 100, 2)) + '%"')

    return recognized[0]

####Modes
def classify(model, mode, i=None, x=None):
    if mode == 1:
        A = [["", ""], ["", ""], ["", ""], ["", ""], ["", ""]]
        c = 0
        imgFiles = os.listdir('../out/toNN.png')
        for (i, f) in enumerate(sorted(imgFiles)):
            res = infer(model, '../out/toNN.png/%s' % f)
            num = NUMDECODE('../out/toNN.png/%s' % f)
            A[i][0] = res
            A[i][1] = num
            c = i
            # os.remove('../out/toNN.png/%s' % f)
        A = A[:c + 1]
        result = ""
        for i in range(c + 1):
            result += A[i][0] + " "
        spell = SpellCheck("../Drugs List/DrugsList.txt", "../Drugs List/Dictionary.txt")
        spell.check(result[:-1])
        Final = spell.correct()
        if Final == "":
            # print(result[:-1] + " Cannot be Found")
            result = ""
            for i in range(c + 1):
                if i == c:
                    result += A[i][1]
                else:
                    result += A[i][0] + " "
            spell.check(result)
            Final = spell.correct()
        if Final == "":
            # print(result + " Cannot be Found")
            result = ""
            for i in range(c + 1):
                if i == c:
                    result += " " + A[i][1]
                else:
                    result += A[i][0]
            spell.check(result)
            Final = spell.correct()
        if Final == "":
            # print(result + " Cannot be Found")
            result = ""
            result += A[0][0]
            Final = spell.get(result)

        print(Final)
        return Final
    if mode == 2:
        A = [["", ""], ["", ""], ["", ""], ["", ""], ["", ""]]
        c = 0
        imgFiles = os.listdir('../out/toNN.png/%d' % x)
        for (i, f) in enumerate(sorted(imgFiles)):
            res = infer(model, '../out/toNN.png/%d/%s' % (x, f))
            num = NUMDECODE('../out/toNN.png/%d/%s' % (x, f))
            A[i][0] = res
            A[i][1] = num
            c = i
            # os.remove('../out/toNN.png/%s' % f)
        A = A[:c + 1]
        result = ""
        for i in range(c + 1):
            result += A[i][0] + " "
        spell = SpellCheck("../Drugs List/DrugsList.txt", "../Drugs List/Dictionary.txt")
        spell.check(result[:-1])
        Final = spell.correct()
        if Final == "":
            # print(result[:-1] + " Cannot be Found")
            result = ""
            for i in range(c + 1):
                if i == c:
                    result += A[i][1]
                else:
                    result += A[i][0] + " "
            spell.check(result)
            Final = spell.correct()
        if Final == "":
            # print(result + " Cannot be Found")
            result = ""
            for i in range(c + 1):
                if i == c:
                    result += " " + A[i][1]
                else:
                    result += A[i][0]
            spell.check(result)
            Final = spell.correct()
        if Final == "":
            # print(result + " Cannot be Found")
            result = ""
            result += A[0][0]
            Final = spell.get(result)

        print(Final)
        return Final


def test(x):
    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0
    numWordTotal = 0
    true_drug = ["Mucotec 150", "Notussil", "Klacid 500", "Megalase", "Adwiflam", "Comfort", "Omehealth", "Antopral 40",
                 "Motinorm", "Visceralgine", "Buscopan", "Napizole 20", "E-Mox 500", "Levoxin 500", "Picolax",
                 "Spasmo-digestin", "Nexium 40", "Controloc 40", "Spascolon 100", "Fluxopride", "Physiomer", "Cetal",
                 "Simethicone", "Optipred", "Dexatobrin", "Phenadone", "Paracetamol", "Levohistam", "Novactam 750",
                 "Epidron", "Clavimox 457", "Dolo-d", "Megafen-n", "Telfast 120", "Zisrocin 500", "Protozole",
                 "Betadine", "Daktacort", "Gynozol 400", "Lornoxicam", "Dantrelax 50", "Downoprazol 40", "Augmentin",
                 "Alphintern", "Arthrofast 150", "Megamox 457", "Maxilase", "Catafly", "Vitacid C", "Cerebromap",
                 "Escita 10"]
    print("Validating Neural Network")
    imgFiles = os.listdir('D:\Projects\drugs\dataset')
    print('Ground truth -> Recognized')
    for (i, f) in enumerate(sorted(imgFiles)):
        segment(f)
        found, recognized = classify(x, f)
        if found == "":
            recog = infer(x, 'D:/Projects/drugs/dataset/%s' % f)
            spell = SpellCheck("../Drugs List/DrugsList.txt", "../Drugs List/Dictionary.txt")
            spell.check(recog)
            found = spell.correct()
        true = true_drug[i]
        numWordOK += 1 if true == found else 0
        numWordTotal += 1
        dist = editdistance.eval(found, true)
        numCharErr += dist
        numCharTotal += len(true)
        print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + true + '"', '->',
              '"' + found + '"')

    charAccuracyRate = (numCharTotal - numCharErr) / numCharTotal
    wordAccuracy = numWordOK / numWordTotal
    print('Character Accuracy: %f%%. Word Accuracy: %f%%.' % (charAccuracyRate * 100.0, wordAccuracy * 100.0))
    return charAccuracyRate, wordAccuracy


def preproc():
    img = cv2.imread('../data/toNN.png')
    img = cv2.resize(img, (255, 80))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(threshold, (3, 3), 0)
    cv2.imwrite('../data/to.png', blur)


def main():
    """main function"""
    model = Model(open(FilePaths.fnCharList).read(), decoderType=DecoderType.BestPath, mustRestore=True)
    while True:
        r = str(input("Enter Mode:  "))
        print(r)
        if r == '1':
            PreProccessing.main(firebase=r)
            preproc()
            segment(mode=1)
            classify(model, mode=1)
        if r == '2':
            prescription = ["", "", "", "", "", "", "", ""]
            PreProccessing.main(firebase=r)
            if not os.path.exists('../presdrug'):
                os.mkdir('../presdrug')
            prescriptionseg(FilePaths.scanned)
            imgFiles = os.listdir('../presdrug')
            for (i, f) in enumerate(sorted(imgFiles)):
                segment(mode=2, i=f, f=i)
                n = classify(model, mode=2, x=i)
                prescription[i] = n
            prescription = list(dict.fromkeys(prescription))
            print(prescription)
            if os.path.exists('../presdrug'):
                shutil.rmtree('../presdrug')
            if os.path.exists('../out/toNN.png'):
                shutil.rmtree('../out/toNN.png')


if __name__ == '__main__':
    main()
