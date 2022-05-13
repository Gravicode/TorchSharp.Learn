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
        private static int _epochs = 4;

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
                        
            Console.WriteLine($"\tCreating the model...");

            var model = new IrisModel("model", device);

            Console.WriteLine($"\tPreparing training and test data...");
            Console.WriteLine();

            var dtset = DatasetHelper.LoadAsDataTable(datasetPath, true);
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
           
            TrainingLoop(dataset, timeout, device, model, train, test);
            
        }

        internal static void TrainingLoop(string dataset, int timeout, Device device, Module model, IEnumerable<(Tensor, Tensor)> train, IEnumerable<(Tensor, Tensor)> test)
        {
            var optimizer = optim.Adam(model.parameters());

            //var optimizer = torch.optim.SGD(model.parameters(), learningRate: 0.01);

            var criterion = cross_entropy_loss();//# cross entropy loss


            Stopwatch totalTime = new Stopwatch();
            totalTime.Start();

            for (var epoch = 1; epoch <= _epochs; epoch++)
            {

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
