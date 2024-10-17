using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Drawing;

namespace ARExercise
{
    public class MarkerData
    {
        public const int WARPED_MARKER_SIZE = 300;
        public const int MARKER_GRID_COUNT = 6;

        public static readonly VectorOfPoint MARKER_SCREEN_COORDS = new VectorOfPoint(new[] {
            new Point(0, 0),
            new Point(WARPED_MARKER_SIZE, 0),
            new Point(WARPED_MARKER_SIZE, WARPED_MARKER_SIZE),
            new Point(0, WARPED_MARKER_SIZE)
        });

        public static readonly MCvPoint3D32f[][] MARKER_WORLD_COORDS = new[] {
            new MCvPoint3D32f[]{
                new MCvPoint3D32f(0, 0, 0),
                new MCvPoint3D32f(1, 0, 0),
                new MCvPoint3D32f(1, 1, 0),
                new MCvPoint3D32f(0, 1, 0)
            },
            new MCvPoint3D32f[]{
                new MCvPoint3D32f(1, 0, 0),
                new MCvPoint3D32f(1, 1, 0),
                new MCvPoint3D32f(0, 1, 0),
                new MCvPoint3D32f(0, 0, 0)
            },
            new MCvPoint3D32f[]{
                new MCvPoint3D32f(1, 1, 0),
                new MCvPoint3D32f(0, 1, 0),
                new MCvPoint3D32f(0, 0, 0),
                new MCvPoint3D32f(1, 0, 0)
            },
            new MCvPoint3D32f[]{
                new MCvPoint3D32f(0, 1, 0),
                new MCvPoint3D32f(0, 0, 0),
                new MCvPoint3D32f(1, 0, 0),
                new MCvPoint3D32f(1, 1, 0)
            }
        };

        private static MarkerData[] markerList = new[]
        {

            new MarkerData("1","2","triangle", new byte[,] {
                { 0, 0, 0, 0, 0, 0 },
                { 0, 255, 255, 255, 0, 0 },
                { 0, 255, 0, 255, 255, 0 },
                { 0, 0, 255, 0, 0, 0 },
                { 0, 255, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0 },
            }),
            new MarkerData("2","1","triangle", new byte[,] {
                { 0, 0, 0, 0, 0, 0 },
                { 0, 255, 255, 0, 255, 0 },
                { 0, 255, 255, 255, 0, 0 },
                { 0, 0, 0, 0, 255, 0 },
                { 0, 0, 0, 0, 255, 0 },
                { 0, 0, 0, 0, 0, 0 },
            }),
            new MarkerData("3","4","cube", new byte[,] {
                { 0, 0, 0, 0, 0, 0 },
                { 0, 255, 0, 255, 0, 0 },
                { 0, 0, 0, 0, 255, 0 },
                { 0, 255, 255, 0, 255, 0 },
                { 0, 0, 0, 255, 255, 0 },
                { 0, 0, 0, 0, 0, 0 },
            }),
            new MarkerData("4","3","cube", new byte[,] {
                { 0, 0, 0, 0, 0, 0 },
                { 0, 255, 255, 0, 255, 0 },
                { 0, 255, 0, 0, 0, 0 },
                { 0, 255, 0, 255, 255, 0 },
                { 0, 0, 255, 255, 255, 0 },
                { 0, 0, 0, 0, 0, 0 },
            }),
            new MarkerData("5","6","pentagon", new byte[,] {
                { 0, 0, 0, 0, 0, 0 },
                { 0, 0, 255, 255, 0, 0 },
                { 0, 0, 255, 0, 255, 0 },
                { 0, 0, 255, 0, 0, 0 },
                { 0, 0, 255, 255, 255, 0 },
                { 0, 0, 0, 0, 0, 0 },
            }),
            new MarkerData("6","5","pentagon", new byte[,] {
                { 0, 0, 0, 0, 0, 0 },
                { 0, 255, 255, 0, 0, 0 },
                { 0, 0, 0, 0, 255, 0 },
                { 0, 0, 0, 255, 255, 0 },
                { 0, 255, 255, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0 },
            })
        };

        private readonly Matrix<byte>[] markerOrientations = new Matrix<byte>[4];
        private readonly string name;
        private readonly string sibling;
        private readonly string shape;


        private MarkerData(string name, string sibling, string shape, byte[,] markerData)
        {
            markerOrientations[0] = new Matrix<byte>(markerData);

            for (int i = 0; i < 3; i++)
            {
                markerOrientations[i + 1] = new Matrix<byte>(MARKER_GRID_COUNT, MARKER_GRID_COUNT);
                CvInvoke.Rotate(markerOrientations[i], markerOrientations[i + 1], RotateFlags.Rotate90CounterClockwise);
            }

            this.name = name;
            this.sibling = sibling;
            this.shape = shape;
        }

        private int getMarkerOrientation(byte[,] markerData)
        {
            Matrix<byte> tmp = new Matrix<byte>(markerData);
            for (int i = 0; i < markerOrientations.Length; i++)
            {
                if (markerOrientations[i].Equals(tmp))
                    return i;
            }

            return -1;
        }

        public static bool TryFindMarker(byte[,] markerData, out string markerName, out string sibling, out string shape, out int orientation)
        {
            orientation = -1;
            markerName = "";
            sibling = "";
            shape = "";

            foreach (MarkerData md in markerList)
            {
                orientation = md.getMarkerOrientation(markerData);
                if (orientation == -1)
                    continue;

                markerName = md.name;
                sibling = md.sibling;
                shape = md.shape;

                return true;
            }

            return false;
        }
    }
}