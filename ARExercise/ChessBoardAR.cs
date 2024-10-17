using Emgu.CV;
using Emgu.CV.CvEnum;
using System;
using System.Drawing;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Diagnostics;

namespace ARExercise
{
    internal class ChessBoardAR : FrameLoop
    {
        private VideoCapture vCap;
        private Size size;

        public ChessBoardAR()
        {
            vCap = new VideoCapture(1);
            size = new Size(4, 7);
        }

        public override void OnFrame()
        {
            Mat frame = new Mat();
            vCap.Read(frame);

            // 1. Load intrinsic parameters
            Matrix<float> intrinsics;
            Matrix<float> distCoeffs;
            UtilityAR.ReadIntrinsicsFromFile(out intrinsics, out distCoeffs);

            // 2. Specify world-space coordinates
            MCvPoint3D32f[] objectPoints = UtilityAR.GenerateObjectPointsForChessboard(size);

            // 3. Find corresponding coordinates in screen-space
            VectorOfPointF imagePoints = new VectorOfPointF();
            bool foundCorners = CvInvoke.FindChessboardCorners(frame, size, imagePoints);


            if (foundCorners)
            {
                // 4. Solve PnP
                Matrix<float> rotationVector = new Matrix<float>(3, 1);
                Matrix<float> translationVector = new Matrix<float>(3, 1);
                CvInvoke.SolvePnP(objectPoints, imagePoints.ToArray(), intrinsics, distCoeffs, rotationVector, translationVector);

                // 5. Construct Rotation Matrix
                Matrix<float> rotationMatrix = new Matrix<float>(3, 3);
                CvInvoke.Rodrigues(rotationVector, rotationMatrix);

                float[,] rValues = rotationMatrix.Data;
                float[,] tValues = translationVector.Data;

                //Debug.WriteLine("rValues: " + Values[0,0]);

                // Samlet extrinsic matrix
                Matrix<float> rMatrix = new Matrix<float>(new float[,]{
                {rValues[0,0], rValues[0,1], rValues[0,2], tValues[0,0]},
                {rValues[1,0], rValues[1,1], rValues[1,2], tValues[1,0]},
                {rValues[2,0], rValues[2,1], rValues[2,2], tValues[2,0]}
                });


                // Display the frame
                // Intrinsics er 3x3, hvorimod rmatrix er 4x3
                // derfor 'intrinsics * rMatrix', og IKKE 'rMatrix * intrinsics'
               
                // Old method without custom colours
                // UtilityAR.DrawCube(frame, intrinsics * rMatrix);

                // New method with custom colours
                //UtilityAR.DrawTriangle(frame, intrinsics * rMatrix,2, floorColour: new MCvScalar(215, 11, 230), contourColour: new MCvScalar(252, 207, 3));



                // CvInvoke.Imshow("Chessboard AR", frame);
            }
            CvInvoke.Imshow("Chessboard AR", frame);

        }
    }
}