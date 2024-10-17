using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Drawing;

namespace ARExercise
{
    public static class UtilityAR
    {
        // Dont change these values
        private static int frameWidth = 800;
        private static int frameHeight = 600;

        /// <summary>
        /// Captures frames from the camera, whenever the user presses a button on the keyboard.<br/>
        /// Stores the images as jpg, naming them capture_0.jpg, capture_1.jpg and so on.<br/>
        /// If images already exist, they are overwritten.
        /// </summary>
        /// <param name="PatternSize">The size of the chessboard pattern to detect</param>
        /// <param name="camIndex">Index of which camera to use for capturing images</param>
        public static void CaptureLoop(Size PatternSize, int camIndex = 0)
        {
            string winName = "Preview";
            CvInvoke.NamedWindow(winName);

            using VideoCapture vcap = new VideoCapture(camIndex);

            vcap.Set(CapProp.FrameWidth, frameWidth);
            vcap.Set(CapProp.FrameHeight, frameHeight);

            int imgIndex = 0;

            while (true)
            {
                Mat frame = new Mat();
                bool frameGrabbed = vcap.Read(frame);
                if (!frameGrabbed)
                {
                    Console.WriteLine("Failed to grab frame");
                    Task.Delay(500).Wait();
                    continue;
                }

                Mat grayFrame = new Mat();
                CvInvoke.CvtColor(frame, grayFrame, ColorConversion.Bgr2Gray);

                Mat binaryFrame = new Mat();
                CvInvoke.Threshold(grayFrame, binaryFrame, 120, 255, ThresholdType.Otsu);

                VectorOfPointF cornerPoints = new VectorOfPointF();
                bool foundChessCorners = CvInvoke.FindChessboardCorners(binaryFrame, PatternSize, cornerPoints,
                    CalibCbType.AdaptiveThresh | CalibCbType.NormalizeImage | CalibCbType.FastCheck);

                Mat chessboardCornersFrame = new Mat();
                frame.CopyTo(chessboardCornersFrame);
                CvInvoke.DrawChessboardCorners(chessboardCornersFrame, PatternSize, cornerPoints, foundChessCorners);

                CvInvoke.SetWindowTitle(winName, $"Preview for frame {imgIndex + 1}");
                CvInvoke.Imshow(winName, chessboardCornersFrame);

                if (CvInvoke.PollKey() != -1)
                    CvInvoke.Imwrite($"capture_{imgIndex++}.jpg", frame);
            }
        }

        /// <summary>
        /// Uses all images named capture_*.jpg for calculating the calibration values.<br/>
        /// Based on <see href="https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html"/>.<br/>
        /// If successful, saves the intrinsics matrix to a file called intrinsics.json.
        /// </summary>
        /// <param name="PatternSize">The size of the chessboard pattern to detect</param>
        /// <param name="showPreview">If true, shows a preview of each image used in the calibration with a drawn overlay of the found chessboard corners</param>
        public static void CalibrateCamera(Size PatternSize, bool showPreview = true)
        {
            string[] images = Directory.GetFiles(Directory.GetCurrentDirectory(), "capture_*.jpg");

            List<MCvPoint3D32f[]> listOfObjectPoints = new List<MCvPoint3D32f[]>();
            List<VectorOfPointF> listOfCornerPoints = new List<VectorOfPointF>();

            MCvPoint3D32f[] objectPoints = GenerateObjectPointsForChessboard(PatternSize);

            if (images.Length == 0)
            {
                Console.WriteLine("No calibration images found");
                return;
            }

            foreach (string image in images)
            {
                Mat frame = CvInvoke.Imread(image);

                Mat grayFrame = new Mat();
                CvInvoke.CvtColor(frame, grayFrame, ColorConversion.Bgr2Gray);

                Mat binaryFrame = new Mat();
                CvInvoke.Threshold(grayFrame, binaryFrame, 120, 255, ThresholdType.Otsu);

                VectorOfPointF cornerPoints = new VectorOfPointF();
                bool foundChessCorners = CvInvoke.FindChessboardCorners(binaryFrame, PatternSize, cornerPoints,
                    CalibCbType.AdaptiveThresh | CalibCbType.NormalizeImage | CalibCbType.FastCheck);

                if (!foundChessCorners)
                {
                    while (showPreview && CvInvoke.WaitKey(1) == -1)
                        CvInvoke.Imshow("Preview", frame);
                    continue;
                }

                CvInvoke.CornerSubPix(binaryFrame, cornerPoints, new Size(11, 11), new Size(-1, -1), new MCvTermCriteria(30, 0.1));

                listOfObjectPoints.Add(objectPoints);
                listOfCornerPoints.Add(cornerPoints);

                if (!showPreview)
                    continue;

                CvInvoke.DrawChessboardCorners(frame, PatternSize, cornerPoints, foundChessCorners);
                while (CvInvoke.WaitKey(1) == -1)
                    CvInvoke.Imshow("Preview", frame);
            }

            Mat intrinsics = new Mat();
            Mat distCoeffs = new Mat();

            Size frameSize = new Size(frameWidth, frameHeight);

            double reprojectionError = CvInvoke.CalibrateCamera(listOfObjectPoints.ToArray(), listOfCornerPoints.Select(x => x.ToArray()).ToArray(),
                frameSize,
                intrinsics,
                distCoeffs,
                CalibType.Default,
                new MCvTermCriteria(30, 0.1),
                out Mat[] rVecs, out Mat[] tVecs);

            using FileStorage fs = new FileStorage("intrinsics.json", FileStorage.Mode.Write);
            fs.Write(intrinsics, "Intrinsics");
            fs.Write(distCoeffs, "DistCoeffs");
        }

        /// <summary>
        /// Reads and returns the intrinsics matrices from the file intrinsics.json.<br/>
        /// See <see cref="CalibrateCamera"/>.
        /// </summary>
        /// <param name="intrinsics">The resulting intrinsics read from the file</param>
        /// <param name="distCoeffs">The resulting distortion coefficients read from the file</param>
        public static void ReadIntrinsicsFromFile(out Matrix<float> intrinsics, out Matrix<float> distCoeffs)
        {
            Mat intrinsicsMat = new Mat();
            Mat distCoeffsMat = new Mat();

            using FileStorage fs = new FileStorage("intrinsics.json", FileStorage.Mode.Read);

            FileNode intrinsicsNode = fs.GetNode("Intrinsics");
            FileNode distCoeffsNode = fs.GetNode("DistCoeffs");

            intrinsicsNode.ReadMat(intrinsicsMat);
            distCoeffsNode.ReadMat(distCoeffsMat);

            intrinsics = new Matrix<float>(3, 3);
            distCoeffs = new Matrix<float>(1, 5);

            intrinsicsMat.ConvertTo(intrinsics, DepthType.Cv32F);
            distCoeffsMat.ConvertTo(distCoeffs, DepthType.Cv32F);
        }

        public static MCvPoint3D32f[] GenerateObjectPointsForChessboard(Size PatternSize)
        {
            List<MCvPoint3D32f> objPoints = new List<MCvPoint3D32f>();
            for (int y = 0; y < PatternSize.Height; y++)
            {
                for (int x = 0; x < PatternSize.Width; x++)
                {
                    objPoints.Add(new MCvPoint3D32f(x, y, 0));
                }
            }

            return objPoints.ToArray();
        }

        /// <summary>
        /// Draws a cube at the origin (0,0) in the world-coordinate.
        /// </summary>
        /// <param name="img">The image to draw the cube onto</param>
        /// <param name="scale">The size of the cube</param>
        /// <param name="projection">the projection-matrix to use for converting world coordinates to screen coordinates</param>
        /// <param name="floorColour">RGB-values of the floor (filled color)</param>
        /// <param name="contourColour">RGB-values of all the lines on the shape</param>
        /// 
        public static void DrawCube(IInputOutputArray img, Matrix<float> projection, float scale = 1, MCvScalar floorColour = default, MCvScalar contourColour = default)
        {
            Matrix<float>[] worldPoints = new[]
            {
                new Matrix<float>(new float[] { 0, 0, 0, 1 }), new Matrix<float>(new float[] { scale, 0, 0, 1 }),
                new Matrix<float>(new float[] { scale, scale, 0, 1 }), new Matrix<float>(new float[] { 0, scale, 0, 1 }),
                new Matrix<float>(new float[] { 0, 0, -scale, 1 }), new Matrix<float>(new float[] { scale, 0, -scale, 1 }),
                new Matrix<float>(new float[] { scale, scale, -scale, 1 }), new Matrix<float>(new float[] { 0, scale, -scale, 1 })
            };

            Point[] screenPoints = worldPoints
                .Select(x => WorldToScreen(x, projection)).ToArray();

            Tuple<int, int>[] lineIndexes = new[] {
                Tuple.Create(0, 1), Tuple.Create(1, 2), // Floor
                Tuple.Create(2, 3), Tuple.Create(3, 0),
                Tuple.Create(4, 5), Tuple.Create(5, 6), // Top
                Tuple.Create(6, 7), Tuple.Create(7, 4),
                Tuple.Create(0, 4), Tuple.Create(1, 5), // Pillars
                Tuple.Create(2, 6), Tuple.Create(3, 7)
            };

            // Draw filled floor
            VectorOfVectorOfPoint floorContour = new VectorOfVectorOfPoint(new VectorOfPoint(screenPoints.Take(4).ToArray()));
            //CvInvoke.DrawContours(img, floorContour, -1, new MCvScalar(0, 255, 0), -3);
            //CvInvoke.DrawContours(img, floorContour, -1, floorColour, -3);


            // Draw top
            foreach (Tuple<int, int> li in lineIndexes.Skip(4).Take(4))
            {
                Point p1 = screenPoints[li.Item1];
                Point p2 = screenPoints[li.Item2];

                //CvInvoke.Line(img, p1, p2, new MCvScalar(255, 0, 0), 3);
                CvInvoke.Line(img, p1, p2, contourColour, 3);

            }

            // Draw pillars
            foreach (Tuple<int, int> li in lineIndexes.Skip(8).Take(4))
            {
                Point p1 = screenPoints[li.Item1];
                Point p2 = screenPoints[li.Item2];

                //CvInvoke.Line(img, p1, p2, new MCvScalar(0, 0, 255), 3);
                CvInvoke.Line(img, p1, p2, contourColour, 3);

            }
        }

        // custom made
        public static void DrawPyramid(IInputOutputArray img, Matrix<float> projection, float scale = 1, MCvScalar floorColour = default, MCvScalar contourColour = default)
        {
            Matrix<float>[] worldPoints = new[]
            {

                new Matrix<float>(new float[] { 0, 0, 0, 1 }), new Matrix<float>(new float[] { scale, 0, 0, 1 }),
                new Matrix<float>(new float[] { scale, scale, 0, 1 }), new Matrix<float>(new float[] { 0, scale, 0, 1 }),
                new Matrix<float>(new float[] { scale / -2, scale / -2, scale, -1 }) // Apex point for the pyramid
            };

            Point[] screenPoints = worldPoints
                .Select(x => WorldToScreen(x, projection)).ToArray();



            Tuple<int, int>[] lineIndexes = new[]
            {
                Tuple.Create(0, 1), Tuple.Create(1, 2), // Base
                Tuple.Create(2, 3), Tuple.Create(3, 0),
                Tuple.Create(0, 4), Tuple.Create(1, 4), // Connecting lines
                Tuple.Create(2, 4), Tuple.Create(3, 4)
            };

            // Draw base
            VectorOfVectorOfPoint baseContour = new VectorOfVectorOfPoint(new VectorOfPoint(screenPoints.Take(4).ToArray()));
            CvInvoke.DrawContours(img, baseContour, -1, floorColour, 5);

            // Draw sides and edges
            foreach (Tuple<int, int> li in lineIndexes)
            {
                Point p1 = screenPoints[li.Item1];
                Point p2 = screenPoints[li.Item2];

                CvInvoke.Line(img, p1, p2, contourColour, 5);
            }
        }
        // custom made
        public static void DrawPentagonCylinder(IInputOutputArray img, Matrix<float> projection, float squareSize, float cylinderHeight, float markerX, float markerY, MCvScalar floorColour = default, MCvScalar contourColour = default)
        {
            int sides = 5; // Number of sides in the pentagon

            // Calculate the scale to fit the pentagon within the square
            float scale = (float)(squareSize / (2 * Math.Cos(Math.PI / sides)));


            // Calculate the 2D coordinates of the pentagon vertices and the top center
            Matrix<float>[] worldPoints = new Matrix<float>[2 * sides];

            for (int i = 0; i < 2 * sides; i++)
            {
                float angle = (float)(i * 2 * Math.PI / (2 * sides));

                float x = scale * (float)Math.Cos(angle) + markerX;
                float y = scale * (float)Math.Sin(angle) + markerY;

                if (i % 2 == 0)
                    worldPoints[i] = new Matrix<float>(new float[] { x, y, 0, 1 }); // Bottom vertices
                else
                    worldPoints[i] = new Matrix<float>(new float[] { x, y, -cylinderHeight, 1 }); // Top vertices adjusted for height
            }

            Point[] screenPoints = worldPoints
                .Select(x => WorldToScreen(x, projection)).ToArray();

            // Draw connecting lines for the cylinder
            for (int i = 0; i < sides; i++)
            {
                CvInvoke.Line(img, screenPoints[i * 2], screenPoints[(i * 2 + 2) % (2 * sides)], contourColour, 3); // Connect bottom to top
                CvInvoke.Line(img, screenPoints[i * 2], screenPoints[(i * 2 + 1)], contourColour, 3); // Connect bottom vertices
                CvInvoke.Line(img, screenPoints[(i * 2 + 1)], screenPoints[(i * 2 + 3) % (2 * sides)], contourColour, 3); // Connect top vertices
            }
        }


        /// <summary>
        /// Converts a homogeneous world coordinate to a screen point
        /// </summary>
        /// <param name="worldPoint">The homogeneous world coordinate</param>
        /// <param name="projection">The projection-matrix to use for converting world coordinates to screen coordinates</param>
        /// <returns>The Point in screen coordinates</returns>
        public static Point WorldToScreen(Matrix<float> worldPoint, Matrix<float> projection)
        {
            Matrix<float> result = projection * worldPoint;
            return new Point((int)(result[0, 0] / result[2, 0]), (int)(result[1, 0] / result[2, 0]));
        }
    }
}