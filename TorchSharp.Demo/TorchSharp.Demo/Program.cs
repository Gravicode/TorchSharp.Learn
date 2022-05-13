// See https://aka.ms/new-console-template for more information
using TorchSharp.Demo.Experiment;

Console.WriteLine("Training IRIS Data with Torch Sharp");

var epochs = 100;
var timeout = 3600;

IrisExperiment.Run(epochs, timeout,"iris");
Console.ReadLine();