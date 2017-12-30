# Face recognition with neural networks
# Sunglasses recognition task

from newConx import *
import pylab
from numpy import array

class FaceRecognizer(BackpropNetwork):
    """
    A specialied backprop network for classifying whether images
    contain a face
    """
    def classify(self, output):
        """
        This ensures that that output layer is the correct size for this
        task, and then tests whether the output value is within
        tolerance of 1 (face) or 0 (no face)
        """
        assert len(output) == 1, 'invalid output pattern'

        if output[0] > (1 - self.tolerance):
            return 'face'
        elif output[0] < self.tolerance:
            return 'no face'
        else:
            return '???'

    def evaluate(self):
        """
        For the current set of inputs, tests each one in the network to
        determine its classification, compares this classification to
        the targets, and computes the percent correct.
        """

        if len(self.inputs) == 0:
            print 'no patterns to evaluate'
            return
        correct = 0
        wrong = 0
        wrong_indices = []
        for i in range(len(self.inputs)):
            pattern = self.inputs[i]
            target = self.targets[i]
            output = self.propagate(input=pattern)
            networkAnswer = self.classify(output)
            correctAnswer = self.classify(target)
            if networkAnswer == correctAnswer:
                correct = correct + 1
            else:
                wrong = wrong + 1
                wrong_indices.append(i)
                print 'network classified image #%d (%s) as %s' % \
                      (i, correctAnswer, networkAnswer)
        total = len(self.inputs)
        correctPercent = float(correct) / total * 100
        wrongPercent = float(wrong) / total * 100
        print '%d patterns: %d correct (%.1f%%), %d wrong (%.1f%%)' % \
              (total, correct, correctPercent, wrong, wrongPercent)
        return correctPercent
        
#create the network
n = FaceRecognizer()

#add 3 layers: input size (20 X 20 X 3 = 1200), hidden size 200, output size 1
n.addLayers(1200, 200, 1)

#n.setOrderedInputs(1)

#get the input and target data
n.loadInputsFromFile('inputs.dat')
n.loadTargetsFromFile('targets.dat')

#set the training parameters
n.setEpsilon(0.05)
n.setMomentum(0.0)
n.setReportRate(1)
n.setTolerance(0.35)

# use 80% of dataset for training, 20% for testing
n.splitData(80)

redlist = []
bluelist = []
greenlist = []
for i in range(1200):
    if i%3 == 0:
        redlist.append(i)
    if i%3 == 1:
        bluelist.append(i)
    if i%3 == 2:
        greenlist.append(i)

n.showActivations('input', shape=(20,20), scale=10, units=redlist)
n.showActivations('input', shape=(20,20), scale=10, units=bluelist)
n.showActivations('input', shape=(20,20), scale=10, units=greenlist)

n.showActivations('hidden', shape=(20, 10), scale=10)
n.showActivations('output', scale=100)

n.setSweepReportRate(100)

print "Emotions recognition network is set up"

# swapData
# evaluate() store %correct
# swapData
# train some#
# swapData
# evaluate()
# compare against old %correct
#    stop when old %correct > curr %correct

train_num = 1
oldCorrect = 0.0
currCorrect = 0.0
training_scores = [n.evaluate()]
n.swapData()
currCorrect = n.evaluate()
iteration = 0
test_scores = [currCorrect]
while (currCorrect >= oldCorrect):
    n.swapData()
    n.train(train_num)

    training_scores.append(n.evaluate())

    n.swapData()
    oldCorrect = currCorrect
    currCorrect = n.evaluate()

    test_scores.append(currCorrect)

print "Training Scores\n"
print training_scores
print "\nTest Scores"
print test_scores
print "\n"

# iterations = range(len(training_scores))
# pylab.plot(array(iterations), array(test_scores), 'g-',
#     array(iterations), array(training_scores, 'b-')
# pylab.title("Face Detection Neural Net Scores")
# pylab.xlabel('Iterations')
# pylab.ylabel('Percent Correct')
# pylab.legend(['Test Data','Training Data'])
# pylab.show()