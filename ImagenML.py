from imageai.Prediction.Custom import CustomImagePrediction

prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath("idenprof/models/model_ex-043_acc-0.710938.h5")
prediction.setJsonPath("idenprof/json/model_class.json")
prediction.loadModel(num_objects=2)

predictions, probabilities = prediction.predictImage("image.jpg", result_count=3)

for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)
