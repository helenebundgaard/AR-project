using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Diagnostics;
using System.Drawing;

namespace ARExercise
{
    public class MarkerAR : FrameLoop
    {
        private VideoCapture vCap;

        private Matrix<float> intrinsics;
        private Matrix<float> distCoeffs;

        private HashSet<string> currentFrameMarkers = new HashSet<string>();

        public MarkerAR()
        {
            vCap = new VideoCapture(1);
            UtilityAR.ReadIntrinsicsFromFile(out intrinsics, out distCoeffs);
        }

        public override void OnFrame()
        {


            Mat frame = new Mat();
            bool frameGrabbed = vCap.Read(frame);
            if (!frameGrabbed)
            {
                Console.Write("Failed to grab frame");
                return;
            }

            Mat grayFrame = new Mat();
            CvInvoke.CvtColor(frame, grayFrame, ColorConversion.Bgr2Gray);

            Mat binaryFrame = new Mat();
            CvInvoke.Threshold(grayFrame, binaryFrame, 175, 255, ThresholdType.Otsu);

            VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
            CvInvoke.FindContours(binaryFrame, contours, null, RetrType.List, ChainApproxMethod.ChainApproxSimple);

            VectorOfVectorOfPoint validContours = GetValidContours(contours);
            VectorOfMat undistortedMarkers = UndistortMarkersFromContours(frame, validContours);

            // Refresh markers for next frame
            currentFrameMarkers.Clear();

            // Pre-pass to store all the detected markers, so that we can later reference this when rendering
            for (int i = 0; i < undistortedMarkers.Size; i++)
            {
                byte[,] centerValues = GetMarkerCenterValues(undistortedMarkers[i]);

                bool markerFound = MarkerData.TryFindMarker(centerValues, out string markerName, out string sibling, out string shape, out int orientIndex);
                if (!markerFound)
                    continue;

                bool success = FindMarkerPerspectiveMatrix(orientIndex, validContours[i], out Matrix<float> worldToScreenMatrix);
                if (!success)
                    continue;

                currentFrameMarkers.Add(markerName);

            }

            Debug.WriteLine(currentFrameMarkers.Count);

            for (int i = 0; i < undistortedMarkers.Size; i++)
            {
                byte[,] centerValues = GetMarkerCenterValues(undistortedMarkers[i]);

                bool markerFound = MarkerData.TryFindMarker(centerValues, out string markerName, out string sibling, out string shape, out int orientIndex);
                if (!markerFound)
                    continue;

                bool success = FindMarkerPerspectiveMatrix(orientIndex, validContours[i], out Matrix<float> worldToScreenMatrix);
                if (!success)
                    continue;

                Matrix<float> originScreen = new Matrix<float>(new float[] { .5f, .5f, 0f, 1 });

                float squareSize = 1f;
                float cylinderHeight = 1f;
                float markerX = 0.5f;
                float markerY = 0.5f;

                // If the marker sibling exsists on the referenced list...
                // ... draw 'true' marker.
                if (currentFrameMarkers.Contains(sibling))
                {

                    switch (shape)
                    {
                        case "triangle":
                            UtilityAR.DrawPyramid(frame, worldToScreenMatrix, scale: 1f, new MCvScalar(0, 155, 0), new MCvScalar(0, 255, 0));
                            break;
                        case "pentagon":
                            UtilityAR.DrawPentagonCylinder(frame, worldToScreenMatrix, squareSize, cylinderHeight, markerX, markerY, new MCvScalar(0, 155, 0), new MCvScalar(0, 255, 0));
                            break;
                        case "cube":
                            UtilityAR.DrawCube(frame, worldToScreenMatrix, scale: 1f, new MCvScalar(0, 155, 0), new MCvScalar(0, 255, 0));
                            break;
                    }


                    CvInvoke.PutText(frame, markerName + "true", UtilityAR.WorldToScreen(originScreen, worldToScreenMatrix), FontFace.HersheyPlain, 1d, new MCvScalar(0, 255, 255), 1);

                }


                // ... else draw 'false' marker
                else
                {
                    switch (shape)
                    {
                        case "triangle":
                            UtilityAR.DrawPyramid(frame, worldToScreenMatrix, scale: 1f, new MCvScalar(0, 155, 155), new MCvScalar(0, 255, 255));
                            break;
                        case "pentagon":
                            UtilityAR.DrawPentagonCylinder(frame, worldToScreenMatrix, squareSize, cylinderHeight, markerX, markerY, new MCvScalar(0, 0, 155), new MCvScalar(0, 0, 255));
                            break;
                        case "cube":
                            UtilityAR.DrawCube(frame, worldToScreenMatrix, scale: 1f, new MCvScalar(155, 0, 0), new MCvScalar(255, 0, 0));
                            break;
                    }
                    CvInvoke.PutText(frame, markerName + "false", UtilityAR.WorldToScreen(originScreen, worldToScreenMatrix), FontFace.HersheyPlain, 1d, new MCvScalar(0, 200, 200), 1);

                }


            }

            CvInvoke.Imshow("Contours", frame);


        }


        /// <summary>
        /// Approximate polygons and discard irrelevant contours
        /// </summary>
        /// <param name="contours">The contours to filter</param>
        /// <returns>The list of valid contours</returns>
        private static VectorOfVectorOfPoint GetValidContours(VectorOfVectorOfPoint contours)
        {
            VectorOfVectorOfPoint validContours = new VectorOfVectorOfPoint();
            for (int i = 0; i < contours.Size; i++)
            {
                VectorOfPoint contour = contours[i];

                // Reduce number of points
                VectorOfPoint approxPoly = new VectorOfPoint();
                CvInvoke.ApproxPolyDP(contour, approxPoly, 4, true);

                // Valid contours have 4 points
                if (approxPoly.Size == 4)
                {
                    double contourLength = CvInvoke.ArcLength(approxPoly, true);
                    double contourArea = CvInvoke.ContourArea(approxPoly, true);

                    // Valid contours must also be within the specified size and correct orientation
                    bool validSize = contourLength > 100 && contourLength < 700;
                    bool validOrientation = contourArea > 0;

                    if (validSize && validOrientation)
                        validContours.Push(approxPoly);
                }
            }

            return validContours;
        }

        /// <summary>
        /// Undistorts markers based on the given contours
        /// </summary>
        /// <param name="image">The image to undistort from</param>
        /// <param name="validContours">The list of contours to warp markers from</param>
        /// <returns>The list of warped (or undistorted) images of markers as matrices</returns>
        private VectorOfMat UndistortMarkersFromContours(Mat image, VectorOfVectorOfPoint validContours)
        {
            VectorOfMat undistortedMarkers = new VectorOfMat();
            for (int i = 0; i < validContours.Size; i++)
            {
                VectorOfPoint contour = validContours[i];
                Mat homography = CvInvoke.FindHomography(contour, MarkerData.MARKER_SCREEN_COORDS, RobustEstimationAlgorithm.Ransac);

                Mat markerContent = new Mat();
                CvInvoke.WarpPerspective(image, markerContent, homography, new Size(MarkerData.WARPED_MARKER_SIZE, MarkerData.WARPED_MARKER_SIZE));

                undistortedMarkers.Push(markerContent);
            }

            return undistortedMarkers;
        }

        /// <summary>
        /// Returns a 2d array with color values for the center of each cell of the marker
        /// </summary>
        /// <param name="warpedMarker">The warped marker to find values from</param>
        private static byte[,] GetMarkerCenterValues(Mat warpedMarker)
        {
            Mat grayMarker = new Mat();
            CvInvoke.CvtColor(warpedMarker, grayMarker, ColorConversion.Bgr2Gray);

            Mat binaryMarker = new Mat();
            CvInvoke.Threshold(grayMarker, binaryMarker, 175, 255, ThresholdType.Otsu);

            int gridSize = MarkerData.WARPED_MARKER_SIZE / MarkerData.MARKER_GRID_COUNT;
            int halfGridSize = gridSize / 2;

            byte[,] centerValues = new byte[MarkerData.MARKER_GRID_COUNT, MarkerData.MARKER_GRID_COUNT];
            for (int y = 0; y < MarkerData.MARKER_GRID_COUNT; y++)
            {
                for (int x = 0; x < MarkerData.MARKER_GRID_COUNT; x++)
                {
                    byte[] centerValue = binaryMarker.GetRawData(new[] {
                            (x * gridSize) + halfGridSize,
                            (y * gridSize) + halfGridSize
                        });

                    centerValues[x, y] = centerValue[0];
                }
            }

            return centerValues;
        }

        /// <summary>
        /// Calculates the projection matrix for the given marker-contour and the given orientation
        /// </summary>
        /// <param name="orientIndex">The index of the orientation of the marker</param>
        /// <param name="contour">The 4 points of the contour for the marker</param>
        /// <param name="worldToScreenMatrix">The resulting projection matrix</param>
        /// <returns>True if the projection matrix could be calculated, otherwise false</returns>
        private bool FindMarkerPerspectiveMatrix(int orientIndex, VectorOfPoint contour, out Matrix<float> worldToScreenMatrix)
        {
            worldToScreenMatrix = null;

            MCvPoint3D32f[] objOrient = MarkerData.MARKER_WORLD_COORDS[orientIndex];
            PointF[] contourPoints = contour.ToArray().Select(x => new PointF(x.X, x.Y)).ToArray();

            Matrix<float> rotationVector = new Matrix<float>(3, 1);
            Matrix<float> translationVector = new Matrix<float>(3, 1);
            bool pnpSolved = CvInvoke.SolvePnP(objOrient, contourPoints, intrinsics, distCoeffs, rotationVector, translationVector);

            if (!pnpSolved)
                return false;

            Matrix<float> rotationMatrix = new Matrix<float>(3, 3);
            CvInvoke.Rodrigues(rotationVector, rotationMatrix);

            float[,] rValues = rotationMatrix.Data;
            float[,] tValues = translationVector.Data;

            Matrix<float> rtMatrix = new Matrix<float>(new float[,] {
                    { rValues[0,0], rValues[0,1], rValues[0,2], tValues[0,0] },
                    { rValues[1,0], rValues[1,1], rValues[1,2], tValues[1,0] },
                    { rValues[2,0], rValues[2,1], rValues[2,2], tValues[2,0] }
                });

            worldToScreenMatrix = intrinsics * rtMatrix;
            return true;
        }
    }
}