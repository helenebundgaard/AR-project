using System.Drawing;

namespace ARExercise
{
    internal class Program
    {
        static void Main(string[] args)
        {

            // Use this to gather intrinsics
            /*
            UtilityAR.CaptureLoop(new Size(4,7),1);
            UtilityAR.CalibrateCamera(new Size(4, 7));
            */
            

            // Marker detection
            MarkerAR markerAR = new MarkerAR();
            markerAR.Run();
        
        }
    }
}