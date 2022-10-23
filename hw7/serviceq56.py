import bentoml

from bentoml.io import NumpyNdarray

model_ref = bentoml.sklearn.get("mlzoomcamp_homework:jsi67fslz6txydu5")

model_runner = model_ref.to_runner()

svc = bentoml.Service("mlzoomcamp_classifier", runners = [model_runner])

@svc.api(input = NumpyNdarray(), output = NumpyNdarray())
async def classify(input):
#def classify(input):
    prediction = await model_runner.predict.async_run(input)
    #prediction = model_runner.predict.run(input)
    print(prediction)
    result = prediction[0]
    return result
