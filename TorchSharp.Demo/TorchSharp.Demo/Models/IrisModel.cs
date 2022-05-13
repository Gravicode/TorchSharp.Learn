using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using static TorchSharp.torch;

using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace TorchSharp.Demo.Models
{
    public class IrisModel : Module
    {
        private Module layer1 = Linear(4, 64);
        private Module layer2 = Linear(64, 3);

        private Module relu1 = ReLU();

        //private Module dropout1 = Dropout(0.25);

        private Module logsm = nn.Softmax(1);

        public IrisModel(string name, torch.Device device = null) : base(name)
        {
            RegisterComponents();

            if (device != null && device.type == DeviceType.CUDA)
                this.to(device);
        }

        public override Tensor forward(Tensor input)
        {
            //1 hidden layer

            var l11 = layer1.forward(input);
            var l12 = relu1.forward(l11);
            var l31 = layer2.forward(l12);
            return logsm.forward(l31);
        }
    }
}
