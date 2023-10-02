using Microsoft.ML;
using Microsoft.ML.Data;
using DeerClassification.Models;

string folder = Path.Combine(Environment.CurrentDirectory, "./Assets");
string imageFolder = Path.Combine(folder, "Training");
string modelFile = Path.Combine(folder, "model.pb");
string trainingFile = Path.Combine(folder, "model.tsv");
string testingFile = Path.Combine(folder, "test.tsv");

MLContext context = new MLContext();

void DisplayResults(IEnumerable<ImagePrediction> predictionData)
{
	foreach (ImagePrediction prediction in predictionData)
	{
		Console.WriteLine($"Image: {Path.GetFileName(prediction.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score?.Max()} ");
	}
}

void ClassifyImage(MLContext mlContext, ITransformer model)
{
	ImageData image = new ImageData()
	{
		ImagePath = Path.Combine(imageFolder, "test.jpeg")
	};

	PredictionEngine<ImageData, ImagePrediction> engine = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
	ImagePrediction prediction = engine.Predict(image);
	Console.WriteLine($"Image: {Path.GetFileName(image.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score?.Max()} ");
}

ITransformer GenerateModel(MLContext mlContext)
{
	try
	{
		Console.WriteLine("Application Started");
		IEstimator<ITransformer> pipeline =
			mlContext.Transforms.LoadImages("input", imageFolder, nameof(ImageData.ImagePath))
				.Append(mlContext.Transforms.ResizeImages("input", Settings.ImageWidth, Settings.ImageHeight, "input"))
				.Append(mlContext.Transforms.ExtractPixels("input", interleavePixelColors: Settings.ChannelsLast,
					offsetImage: Settings.Mean))
				.Append(mlContext.Model.LoadTensorFlowModel(modelFile)
					.ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2_pre_activation" },
						inputColumnNames: new[] { "input" }, addBatchDimensionInput: true))
				.Append(mlContext.Transforms.Conversion.MapValueToKey("LabelKey", "Label"))
				.Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy("LabelKey",
					"softmax2_pre_activation"))
				.Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabelValue", "PredictedLabel"))
				.AppendCacheCheckpoint(mlContext);

		IDataView training = mlContext.Data.LoadFromTextFile<ImageData>(path: trainingFile, hasHeader: false);
		Console.WriteLine("Loaded files from training file:");
		foreach (ImageData data in mlContext.Data.CreateEnumerable<ImageData>(training, false))
		{
			Console.WriteLine($"Path: {data.ImagePath}, Label: {data.Label}");
		}

		ITransformer model = pipeline.Fit(training);

		IDataView testData = mlContext.Data.LoadFromTextFile<ImageData>(path: testingFile, hasHeader: false);
		IDataView predictions = model.Transform(testData);
		IEnumerable<ImagePrediction> imagePredictionData =
			mlContext.Data.CreateEnumerable<ImagePrediction>(predictions, true);
		DisplayResults(imagePredictionData);

		MulticlassClassificationMetrics metrics =
			mlContext.MulticlassClassification.Evaluate(predictions, "LabelKey", predictedLabelColumnName: "PredictedLabel");

		Console.WriteLine($"LogLoss is: {metrics.LogLoss}");
		Console.WriteLine($"PerClassLogLoss is: {String.Join(" , ", metrics.PerClassLogLoss.Select(c => c))}");

		return model;
	}
	catch (Exception e)
	{
		Console.WriteLine($"{e}");
		return null;
	}
}

ITransformer model = GenerateModel(context);
ClassifyImage(context, model);
struct Settings
{
	public const int ImageWidth = 224;
	public const int ImageHeight = 224;
	public const float Mean = 117;
	public const bool ChannelsLast = true;
};