using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp.Demo.Helpers;
using TorchSharp.Demo.Models;
using TorchSharp.torchvision;
using static TorchSharp.torch;

using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace TorchSharp.Demo.Experiment
{
    public class IrisExperiment
    {
        //static int TotalRow = 150;
        private static int _epochs = 4;
        //private static int _trainBatchSize = (int)(TotalRow*0.8);
        //private static int _testBatchSize = (int)(TotalRow*0.2);

        private readonly static int _logInterval = 10;

        internal static void Run(int epochs, int timeout,string dataset)
        {
            
            _epochs = epochs;

            if (string.IsNullOrEmpty(dataset))
            {
                dataset = "iris";
            }

            var device = cuda.is_available() ? CUDA : CPU;

            Console.WriteLine();
            Console.WriteLine($"\tRunning IRIS with {dataset} on {device.type.ToString()} for {epochs} epochs, terminating after {TimeSpan.FromSeconds(timeout)}.");
            Console.WriteLine();

            var datasetPath = FileHelpers.AppDirectory+$"/../../../../Dataset/{dataset}.csv";

            random.manual_seed(1);

            var cwd = Environment.CurrentDirectory;


            /*
            if (device.type == DeviceType.CUDA)
            {
                _trainBatchSize *= 4;
                _testBatchSize *= 4;
            }
            */

            Console.WriteLine($"\tCreating the model...");

            var model = new IrisModel("model", device);

            //var normImage = transforms.Normalize(new double[] { 0.1307 }, new double[] { 0.3081 }, device: (Device)device);

            Console.WriteLine($"\tPreparing training and test data...");
            Console.WriteLine();

            var dtset = DatasetHelper.LoadAsDataTable(datasetPath, true);
            //dtset.OneHotEncoding("class");

            //var train_label = dtset.Pop(new string[] { "class_Iris-setosa", "class_Iris-versicolor", "class_Iris-virginica" });
            var classes = dtset.ToCategory("class");
            var train_label = dtset.Pop("class");
            dtset.Normalization();
            dtset.Head();
            var train_data = dtset.ToTensors();
            var train = new List<(Tensor,Tensor)>();
            for(int i = 0; i < train_label.Count; i++)
            {
                train.Add((train_data[i],train_label[i]));
            }

            var test_label = train_label.Take(30).ToList();
            var test_data = dtset.GetSubset(1,30).ToTensors();
            var test = new List<(Tensor, Tensor)>();
            for (int i = 0; i < test_label.Count; i++)
            {
                test.Add((test_data[i], test_label[i]));
            }
           
            /*
             var splitted = dtset.Split();
            var train_label = splitted.training.Pop(new string[] { "class_Iris-setosa", "class_Iris-versicolor", "class_Iris-virginica" });
            var train_data = splitted.training.ToTensors();
            var train = new List<(Tensor,Tensor)>();
            for(int i = 0; i < train_label.Count; i++)
            {
                train.Add((train_data[i],train_label[i]));
            }

            var test_label = splitted.test.Pop(new string[] { "class_Iris-setosa", "class_Iris-versicolor", "class_Iris-virginica" });
            var test_data = splitted.test.ToTensors();
            var test = new List<(Tensor, Tensor)>();
            for (int i = 0; i < test_label.Count; i++)
            {
                test.Add((test_data[i], test_label[i]));
            }
             */

            TrainingLoop(dataset, timeout, device, model, train, test);
            
        }

        internal static void TrainingLoop(string dataset, int timeout, Device device, Module model, IEnumerable<(Tensor, Tensor)> train, IEnumerable<(Tensor, Tensor)> test)
        {
            var optimizer = optim.Adam(model.parameters());

            //var optimizer = torch.optim.SGD(model.parameters(), learningRate: 0.01);

            //var scheduler = optim.lr_scheduler.StepLR(optimizer, 1, 0.1);

            var criterion = cross_entropy_loss();//# cross entropy loss


            Stopwatch totalTime = new Stopwatch();
            totalTime.Start();

            for (var epoch = 1; epoch <= _epochs; epoch++)
            {

                //Train(model, optimizer, nll_loss(reduction: Reduction.Mean), device, train, epoch, 10, train.Count());
                //Test(model, nll_loss(reduction: nn.Reduction.Sum), device, test, test.Count());
                Train(model, optimizer, criterion, device, train, epoch, 1, train.Count());
                Test(model, criterion, device, test, test.Count());

                Console.WriteLine($"End-of-epoch memory use: {GC.GetTotalMemory(false)}");

                if (totalTime.Elapsed.TotalSeconds > timeout) break;
            }

            totalTime.Stop();
            Console.WriteLine($"Elapsed time: {totalTime.Elapsed.TotalSeconds:F1} s.");

            Console.WriteLine("Saving model to '{0}'", dataset + ".model.bin");
            model.save(dataset + ".model.bin");
        }

        private static void Train(
            Module model,
            optim.Optimizer optimizer,
            Loss loss,
            Device device,
            IEnumerable<(Tensor, Tensor)> dataLoader,
            int epoch,
            long batchSize,
            long size)
        {
            model.train();

            int batchId = 1;

            Console.WriteLine($"Epoch: {epoch}...");

            foreach (var (data, target) in dataLoader)
            {
                using (var d = torch.NewDisposeScope())
                {
                    optimizer.zero_grad();

                    var prediction = model.forward(data);
                    var output = loss(prediction, target);

                    output.backward();

                    optimizer.step();

                    if (batchId % _logInterval == 0)
                    {
                        Console.WriteLine($"\rTrain: epoch {epoch} [{batchId * batchSize} / {size}] Loss: {output.ToSingle():F4}");
                    }

                    batchId++;

                }
            }
        }

        private static void Test(
            Module model,
            Loss loss,
            Device device,
            IEnumerable<(Tensor, Tensor)> dataLoader,
            long size)
        {
            model.eval();

            double testLoss = 0;
            int correct = 0;

            foreach (var (data, target) in dataLoader)
            {
                using (var d = torch.NewDisposeScope())
                {
                    var prediction = model.forward(data);
                    var output = loss(prediction, target);
                    testLoss += output.ToSingle();

                    correct += prediction.argmax(1).eq(target).sum().ToInt32();
                }
            }

            Console.WriteLine($"Size: {size}, Total: {size}");

            Console.WriteLine($"\rTest set: Average loss {(testLoss / size):F4} | Accuracy {((double)correct / size):P2}");
        }
    }
}
