from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.datasets import mnist
from tensorflow.keras.optimizers import SGD
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
import cv2
import numpy as np


def NUMDECODE(x):
    class CNN:
        @staticmethod
        def build(width, height, depth, total_classes):
            """Initialize the Model"""
            model = Sequential()

            """First CONV => RELU => POOL Layer"""
            model.add(Conv2D(20, (5, 5), padding="same", input_shape=(height, width, depth)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format="channels_last"))

            """Second CONV => RELU => POOL Layer"""
            model.add(Conv2D(50, (5, 5), padding="same"))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format="channels_last"))

            """Third CONV => RELU => POOL Layer"""
            model.add(Conv2D(100, (5, 5), padding="same"))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format="channels_last"))

            """FC => RELU layers"""
            model.add(Flatten())
            model.add(Dense(500))
            model.add(Activation("relu"))

            """Using Softmax Classifier for Linear Classification"""
            model.add(Dense(total_classes))
            model.add(Activation("softmax"))

            return model

    """Load MNIST Dataset"""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    """Reshape Test and Train Images"""
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    """Classify Labels into 10 Classes [0:9]"""
    num_classes = 10
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    """Convert RGB into BINARY[Black&White]"""
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    """Compile Model"""
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    clf = CNN.build(width=28, height=28, depth=1, total_classes=10)
    clf.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

    """To Train The Model for First Time Call train()"""
    def train():
        """Training Model
        Batch Size = 128
        Epochs = 20
        """
        print("Training Model")
        clf.fit(x_train, y_train, batch_size=128, epochs=20, verbose=1, validation_data=(x_test, y_test))
        clf.save('mnist.h5')
        print("Model has been saved Successfully")
        print("Evaluating Accuracy and Loss Function")
        """Calculate Accuracy of The Model"""
        loss, accuracy = clf.evaluate(x_test, y_test, batch_size=128, verbose=1)
        print('Accuracy of Model: {:.2f}%'.format(accuracy * 100))

    """Load pre-saved Model"""
    clf.load_weights('../model/mnist.h5')

    """Create Variable to Save the Result of each Character"""
    result = ""

    """Read Image"""
    image = cv2.imread(x)

    """Add Empty Borders to The Image to resize it"""
    row, col = image.shape[:2]
    bottom = image[row - 2:row, 0:col]
    mean = cv2.mean(bottom)[0]
    image = cv2.copyMakeBorder(image, 40, 40, 40, 40, borderType=cv2.BORDER_CONSTANT, value=[mean, mean, mean])

    """Pre-Process The Image"""
    image = cv2.resize(image, (872, 532))
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    """Get Contours"""
    ctrs, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, ctrs, -1, (0, 255, 0), 2)
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    rects = sorted(rects, key=lambda z: z[0])

    """Load each Contour"""
    for rect in rects:
        x, y, w, h = rect
        """Neglect Noise"""
        if 60 < h < 300 and 50 < w < 300:

            """Get Region of each Number"""
            cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
            leng = int(rect[3] * 1.6)
            pt1 = abs(int(rect[1] + rect[3] // 2 - leng // 2))
            pt2 = abs(int(rect[0] + rect[2] // 2 - leng // 2))
            roi = img[pt1:pt1 + leng, pt2:pt2 + leng]
            """Reshape Number's Region to MNIST Dataset Image Size"""
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            roi = roi.reshape(-1, 28, 28, 1)
            roi = np.array(roi, dtype='float32')
            """Convert Color Scale into Binary"""
            roi /= 255
            """Classify the Number"""
            pred_array = clf.predict(roi)
            pred_array = np.argmax(pred_array)
            """Save Result into String"""
            result += str(pred_array)
            cv2.putText(img, str(pred_array), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 3)

    """Return Final Result"""
    return result
