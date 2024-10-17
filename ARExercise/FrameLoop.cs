using Emgu.CV;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ARExercise
{
    public abstract class FrameLoop
    {
        public void Run()
        {
            while (true)
            {
                OnFrame();
                CvInvoke.WaitKey(1);
            }
        }

        public abstract void OnFrame();
    }
}
