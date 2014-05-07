using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.IO;

using Microsoft.Kinect;
using Microsoft.Kinect.Toolkit;
using Microsoft.Kinect.Toolkit.Controls;
using Microsoft.Kinect.Toolkit.Interaction;

using Emgu.CV;
using Emgu.CV.UI;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.Util;
using Emgu.CV.GPU;


namespace CS682Project
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private KinectSensorChooser sensorChooser;

        short[] depthArray = null;
        short[] smoothDepthArray = null;
        int width = 640;
        int height = 480;
        int widthBound = 639;
        int heightBound = 479;
        int innerBandThreshold = 1;
        int outerBandThreshold = 3;
        int kernelSize = 7;
        
        WriteableBitmap colorImageBitmap = null;

        short[] depthData = null;
        byte[] depthColorImage = null;
        Queue<short[]> myDepthQueue = new Queue<short[]>();
        
        Brush skeletonBrush = new SolidColorBrush(Colors.Red);
        Line[] myBones;
        DepthImagePoint[] myJoints = new DepthImagePoint[2];

        Image<Bgr, Single> CVKinectDepthFrame;
        Image<Bgr, Byte> CVKinectColorFrame;
        

        // Whitney's additions
        private const int ringBufferSize = 5;
        private const int depthKernelSize = 1; //11
        private const double sigmaMult = 7.0;
        private const double degDiv = 1;
        private const double minDotProd = 0.82; //.99
        private const double planeFactor = 1000;
        private const double maxPerpendicular = 1000;
        private const bool useFloodFill = true;
        private const int minGroupSize = 1500;
        private const bool useKmeans = false;

        private DepthImagePixel[] depthPixels;
        private WriteableBitmap depthBitmap = null;

        private int ringBufIdx = 0;
        private int frameSize;
        private bool ready = false;
        private short[] depthRingBuffer;
        private long[] depthRunningSum;
        private byte[] depthRGBPixels;

        private bool isActive = false;
        private Object dpLock = new Object();

        private Image<Bgr, Single> depthImgCV;
        private Image<Bgr, Byte> grayImgCV;
        private Image<Bgr, Single> dxImgCV;
        private Image<Bgr, Single> dyImgCV;
        private Image<Bgr, Single> depthImgOutCV;
        private float[] pixelMag;
        private int[] pixelState;
        private List<int> groupCount;

        Matrix<float> samples;
        Matrix<int>[] finalClusters;
        Matrix<int> compactness;

        private void InitData()
        {
            // save the frame size
            frameSize = sensorChooser.Kinect.DepthStream.FramePixelDataLength;

            // Allocate space to put the depth pixels we'll receive
            depthPixels = new DepthImagePixel[frameSize];

            // This is the bitmap we'll display on-screen
            depthBitmap = new WriteableBitmap(
                                sensorChooser.Kinect.DepthStream.FrameWidth, 
                                sensorChooser.Kinect.DepthStream.FrameHeight, 
                                96.0, 
                                96.0, 
                                PixelFormats.Bgr32, 
                                null);

            // use a ring buffer to keep track of the last N buffers
            depthRingBuffer = new short[frameSize * ringBufferSize];

            // this is the current running sum for the depth ring
            depthRunningSum = new long[frameSize];

            // assign the depth bitmap to the image source
            kinectDepthImage.Source = depthBitmap;

            // Allocate space to put the color pixels we'll create
            depthRGBPixels = new byte[frameSize * sizeof(int)];

            // allocate space for the opencv image
            depthImgCV = new Image<Bgr, Single>(sensorChooser.Kinect.DepthStream.FrameWidth, sensorChooser.Kinect.DepthStream.FrameHeight);
            grayImgCV = new Image<Bgr, Byte>(sensorChooser.Kinect.DepthStream.FrameWidth, sensorChooser.Kinect.DepthStream.FrameHeight);
            dxImgCV = new Image<Bgr, Single>(sensorChooser.Kinect.DepthStream.FrameWidth, sensorChooser.Kinect.DepthStream.FrameHeight);
            dyImgCV = new Image<Bgr, Single>(sensorChooser.Kinect.DepthStream.FrameWidth, sensorChooser.Kinect.DepthStream.FrameHeight);
            depthImgOutCV = new Image<Bgr, Single>(sensorChooser.Kinect.DepthStream.FrameWidth, sensorChooser.Kinect.DepthStream.FrameHeight);

            samples = new Matrix<float>(sensorChooser.Kinect.DepthStream.FrameWidth * sensorChooser.Kinect.DepthStream.FrameHeight, 1, 3);
            finalClusters = new Matrix<int>[20];
            for (int i = 0; i < 20; i++)
                finalClusters[i] = new Matrix<int>(sensorChooser.Kinect.DepthStream.FrameWidth * sensorChooser.Kinect.DepthStream.FrameHeight, 1);
            compactness = new Matrix<int>(20, 1, 1);

            pixelMag = new float[sensorChooser.Kinect.DepthStream.FrameWidth * sensorChooser.Kinect.DepthStream.FrameHeight];
            pixelState = new int[sensorChooser.Kinect.DepthStream.FrameWidth * sensorChooser.Kinect.DepthStream.FrameHeight];
        }

        public MainWindow()
        {
            InitializeComponent();

            Loaded += OnLoaded;
        }

        private void OnLoaded(object sender, RoutedEventArgs e)
        {
            //Detects if a Kinect or Kinect like device is connected and sets up the environment based on the sensors optimal settings.
            this.sensorChooser = new KinectSensorChooser();
            this.sensorChooser.KinectChanged += SensorChooserOnKinectChanged;
            this.sensorChooserUI.KinectSensorChooser = this.sensorChooser;
            this.sensorChooser.Start();

            // initialize our data structures
            InitData();

            //Eventhandlers for the depth, skeletal, and colorstreams. Skeleton is commented out as it isn't needed as of yet.
            this.sensorChooser.Kinect.ColorFrameReady += Kinect_ColorFrameReady;
            this.sensorChooser.Kinect.DepthFrameReady += SensorDepthFrameReady;
            //this.sensorChooser.Kinect.DepthFrameReady += Kinect_DepthFrameReady;
            //this.sensorChooser.Kinect.SkeletonFrameReady += Kinect_SkeletonFrameReady;
        }


        void Kinect_ColorFrameReady(object sender, ColorImageFrameReadyEventArgs e)
        {
            //Code between the hashtags includes the basics for rendering the colorstream.
            //############################################################################################
            using (ColorImageFrame colorFrame = e.OpenColorImageFrame())
            {
                if (colorFrame == null) return;

                byte[] colorData = new byte[colorFrame.PixelDataLength];

                colorFrame.CopyPixelDataTo(colorData);

                if (colorImageBitmap == null)
                {
                    this.colorImageBitmap = new WriteableBitmap(
                        colorFrame.Width,
                        colorFrame.Height,
                        96,
                        96,
                        PixelFormats.Bgr32,
                        null);
                    kinectColorImage.Source = colorImageBitmap;
                }

                this.colorImageBitmap.WritePixels(
                    new Int32Rect(0, 0, colorFrame.Width, colorFrame.Height),
                    colorData,
                    colorFrame.Width * colorFrame.BytesPerPixel, 0
                    );
                //############################################################################################
                
                //The following code demonstrates the conversion of the Kinect Writeable Bitmap to
                //the Image<,>  format handled by the EMGU OpenCV wrapper.

                System.Drawing.Bitmap ColorBitmap = BitmapSourceConverter.ToBitmap(colorImageBitmap);

                CVKinectColorFrame = BitmapSourceConverter.ToOpenCVImage<Bgr, Byte>(ColorBitmap);

                //Once the image has been converted to a Image<Bgr, Byte> a second temporary Image<Gray, Byte>
                //is created on which we can run the various filtering and detection algorithms on.
                Image<Gray, Byte> grayTempCV = CVKinectColorFrame.Convert<Gray, Byte>();

                //Apply Gaussian smoothing with 3x3 kernel and simga = 2.
                grayTempCV._SmoothGaussian(3, 3, 2, 2);

                //Run canny detection to obtain image edges.
                grayTempCV = grayTempCV.Canny(150, 25);//(400, 200);

                //The following code demonstrates box, line, and polyLine detection.
                //Haven't cleaned these functions up yet, currently they accept an Image<Gray, Byte>. Was thinking that depending
                //on the design maybe we pass an Image<Gray, Byte> as the source, as well as the Image<Bgr, Byte> which would also
                //act as the return type. That way the drawing of the various detection elements would be done within the function.

                //####################################################################################################
                //List<MCvBox2D> myBoxes = boxDetect(CannyCV);
                //if (myBoxes != null)
                //{
                //    foreach (MCvBox2D box in myBoxes)
                //        CVKinectColorFrame.Draw(box, new Bgr(System.Drawing.Color.Blue), 2);
                //}


                //LineSegment2D[] myLines = lineDetect(CannyCV);

                //if(myLines != null)
                //{
                //    foreach (LineSegment2D line in myLines)
                //        CVKinectColorFrame.Draw(line, new Bgr(System.Drawing.Color.Green), 2);
                //}


                //Dectect polyLines that form quadralaterals.
                List<System.Drawing.Point[]> myPoly = polyDetect(grayTempCV);

                if (myPoly != null)
                {
                    foreach (System.Drawing.Point[] polyLine in myPoly)
                        CVKinectColorFrame.DrawPolyline(polyLine, true, new Bgr(System.Drawing.Color.Red), 2);
                }
                //##################################################################################################



                //Once we are finished with the gray temp image it needs to be disposed of. 
                grayTempCV.Dispose();


                //Following processing of CV image need to convert back to Windows style bitmap.
                BitmapSource bs = BitmapSourceConverter.ToBitmapSource(CVKinectColorFrame);
                
                //Dispose of the CV color image following the conversion.
                CVKinectColorFrame.Dispose();

                int stride = bs.PixelWidth * (bs.Format.BitsPerPixel / 8);
                byte[] data = new byte[stride * bs.PixelHeight];
                bs.CopyPixels(data, stride, 0);
                colorData = data;

                //Write our converted color pixel data to the original Writeable bitmap.
                this.colorImageBitmap.WritePixels(
                    new Int32Rect(0, 0, colorFrame.Width, colorFrame.Height),
                    colorData,
                    colorFrame.Width * colorFrame.BytesPerPixel, 0
                    );

            }
        }

        
        public List<MCvBox2D> boxDetect(Image<Gray, Byte> myImage)
        {
            List<MCvBox2D> boxlist = new List<MCvBox2D>();
            using (MemStorage storage = new MemStorage())
                for (Contour<System.Drawing.Point> contours = myImage.FindContours(); contours != null; contours = contours.HNext)
                {
                    Contour<System.Drawing.Point> currentContour = contours.ApproxPoly(contours.Perimeter * 0.05, storage);

                    if (contours.Area > 250)
                    {
                        bool isRectangle = true;
                        System.Drawing.Point[] pts = currentContour.ToArray();
                        LineSegment2D[] edges = Emgu.CV.PointCollection.PolyLine(pts, true);

                        Parallel.For(0, edges.Length, myEdges =>
                        {
                            double angle = Math.Abs(edges[(myEdges + 1) % edges.Length].GetExteriorAngleDegree(edges[myEdges]));
                            if (angle < 20 || angle > 120)
                            {
                                isRectangle = false;
                            }
                        });
                        if (isRectangle) boxlist.Add(currentContour.GetMinAreaRect());
                    }
                }
            return boxlist;
        }

        public LineSegment2D[] lineDetect (Image<Gray, Byte> myImage)
        {
            LineSegment2D[] myLines = myImage.HoughLinesBinary(1, (Math.PI / 180), 50, 80, 10)[0];
            return myLines;

        }

        public List<System.Drawing.Point[]> polyDetect(Image<Gray, Byte> myImage)
        {
            
            List<System.Drawing.Point[]> mypts = new List<System.Drawing.Point[]>();
            
            using (MemStorage storage = new MemStorage())

                for (Contour<System.Drawing.Point> contours = myImage.FindContours(CHAIN_APPROX_METHOD.CV_CHAIN_APPROX_SIMPLE, RETR_TYPE.CV_RETR_LIST); contours != null; contours = contours.HNext)
                {
                    Contour<System.Drawing.Point> currentContour = contours.ApproxPoly(contours.Perimeter * 0.1, storage);

                    //Check to see if the contour forms a enclosed quadralateral with a desired minimum area.
                    if (Math.Abs(currentContour.Area) > 150 && currentContour.Convex == true && currentContour.Total == 4)//
                    {
                        System.Drawing.Point[] pts = currentContour.ToArray();
                        mypts.Add(pts);
                    }
                }
            return mypts;
        }

        /// <summary>
        /// Event handler for Kinect sensor's DepthFrameReady event
        /// </summary>
        /// <param name="sender">object sending the event</param>
        /// <param name="e">event arguments</param>
        private void SensorDepthFrameReady(object sender, DepthImageFrameReadyEventArgs e)
        {
            using (DepthImageFrame depthFrame = e.OpenDepthImageFrame())
            {
                if (depthFrame != null)
                {
                    // determine whether to dispatch a thread with this data
                    lock (dpLock)
                    {
                        // if the thread is not active
                        if (!isActive)
                        {
                            isActive = true;
                            // Copy the pixel data from the image to a temporary array
                            depthFrame.CopyDepthImagePixelDataTo(this.depthPixels);
                            
                            //Thread t = new Thread(() => ProcessDepthData(depthFrame.MinDepth, depthFrame.MaxDepth));
                            //Thread t = new Thread(() => ProcessDepthDataAlt(depthFrame.MinDepth, depthFrame.MaxDepth));
                            //Thread t = new Thread(() => ProcessDepthDataAlt2(depthFrame.MinDepth, depthFrame.MaxDepth));
                            Thread t = new Thread(() => ProcessDepthDataAlt3(depthFrame.MinDepth, depthFrame.MaxDepth));
                            t.Start();
                        }
                    }
                }
            }
        }

        private void ProcessDepthData(int minDepth, int maxDepth)
        {
            // Save this frame in our ring buffer
            Parallel.For(0, this.depthPixels.Length, i =>
            {
                // Get the depth for this pixel
                short depth = depthPixels[i].Depth;

                // if we have a full ring buffer, subtract the oldest entry from the running sum
                if (ready)
                    depthRunningSum[i] -= depthRingBuffer[ringBufIdx * frameSize + i];

                // save this pixel for this frame, and update the running sum
                depthRingBuffer[ringBufIdx * frameSize + i] = (depth >= minDepth && depth <= maxDepth) ? depth : (short)0;
                depthRunningSum[i] += depthRingBuffer[ringBufIdx * frameSize + i];
            });

            // increment the ring buffer index
            ringBufIdx = (ringBufIdx + 1) % ringBufferSize;

            // if we have a full ring buffer, enable image processing
            if (!ready && ringBufIdx == 0)
                ready = true;

            // perform image processing if we have a full circular buffer
            if (ready)
            {

                // Convert the depth to RGB
                Parallel.For(0, this.depthRunningSum.Length, i =>
                {
                    long longVal = depthRunningSum[i] / ringBufferSize * 256 / this.sensorChooser.Kinect.DepthStream.MaxDepth;
                    float floatVal = depthRunningSum[i] / ringBufferSize;

                    // Get the depth for this pixel
                    byte depth = (byte)longVal;

                    int x = i / this.sensorChooser.Kinect.DepthStream.FrameWidth;
                    int y = i % this.sensorChooser.Kinect.DepthStream.FrameWidth;

                    // Write out blue byte
                    //this.depthRGBPixels[i * 4] = depth;
                    depthImgCV.Data[x, y, 0] = floatVal;

                    // Write out green byte
                    //this.depthRGBPixels[i * 4 + 1] = depth;
                    depthImgCV.Data[x, y, 1] = floatVal;

                    // Write out red byte                        
                    //this.depthRGBPixels[i * 4 + 2] = depth;
                    depthImgCV.Data[x, y, 2] = floatVal;
                });

                // *******************************************
                //            Start Plane Detection
                // *******************************************

                // first, smooth the image to eliminate noise
                if (depthKernelSize > 1)
                    depthImgCV = depthImgCV.SmoothGaussian(depthKernelSize, depthKernelSize, depthKernelSize * sigmaMult, depthKernelSize * sigmaMult);

                double fx = depthImgCV.Rows / (2.0 * Math.Tan(43/2));
                double fy = depthImgCV.Cols / (2.0 * Math.Tan(57/2));

                //In the class scope:
                Object lockMinMax = new Object();
                int lockCount = 0;

                double[] globalMin = new double[3];
                double[] globalMax = new double[3];

                // next, compute the normal vector at each point (excluding the outermost pixels)
                Parallel.For(0, depthImgCV.Cols * depthImgCV.Rows, i =>
                {
                    double[] yVector = new double[3];
                    double[] xVector = new double[3];
                    double[] crossVector = new double[3];

                    int x = i / depthImgCV.Cols;
                    int y = i % depthImgCV.Cols;

                    // only process the inner pixels
                    if (x > 0 && x < (depthImgCV.Rows - 1) && y > 0 && y < (depthImgCV.Cols - 1))
                    {

                        double x1 = depthImgCV.Data[x, y, 0] * ((double)x - depthImgCV.Rows / 2.0) / fx;
                        double y1 = depthImgCV.Data[x, y, 0] * ((double)y - depthImgCV.Cols / 2.0) / fy;

                        double x2 = depthImgCV.Data[x - 1, y, 0] * (((double)x - 1.0) - depthImgCV.Rows / 2.0) / fx;
                        double y2 = depthImgCV.Data[x - 1, y, 0] * ((double)y - depthImgCV.Cols / 2.0) / fy;

                        // top - bottom: vector pointing in y direction
                        yVector[0] = x2 - x1;
                        yVector[1] = y2 - y1;
                        yVector[2] = depthImgCV.Data[x, y - 1, 0] - depthImgCV.Data[x, y, 0];

                        x2 = depthImgCV.Data[x, y - 1, 0] * ((double)x - depthImgCV.Rows / 2.0) / fx;
                        y2 = depthImgCV.Data[x, y - 1, 0] * (((double)y - 1.0) - depthImgCV.Cols / 2.0) / fy;

                        // left - right: vector pointing in x direction
                        xVector[0] = x2 - x1;
                        xVector[1] = y2 - y1;
                        xVector[2] = depthImgCV.Data[x - 1, y, 0] - depthImgCV.Data[x, y, 0];

                        // compute y cross x
                        crossVector[0] = yVector[1] * xVector[2] - yVector[2] * xVector[1];
                        crossVector[1] = yVector[2] * xVector[0] - yVector[0] * xVector[2];
                        crossVector[2] = yVector[0] * xVector[1] - yVector[1] * xVector[0];

                        // compute magnitude
                        double mag = Math.Sqrt(crossVector[0] * crossVector[0] + crossVector[1] * crossVector[1] + crossVector[2] * crossVector[2]);

                        if (mag > 0)
                        { 
                            // normalize the vector
                            crossVector[0] /= mag;
                            crossVector[1] /= mag;
                            crossVector[2] /= mag;

                            if (degDiv > 0)
                            {
                                // convert to spherical coords & round to nearest degree
                                xVector[0] = Math.Sqrt(crossVector[0] * crossVector[0] + crossVector[1] * crossVector[1] + crossVector[2] * crossVector[2]); // r
                                xVector[1] = Math.Atan2(crossVector[1], crossVector[0]); // theta
                                xVector[2] = Math.Acos(crossVector[2] / xVector[0]); // phi

                                // round degrees down to nearest multiple of degDiv
                                xVector[1] = Math.Floor((xVector[1] * 180.0 / Math.PI) / degDiv) * degDiv * (Math.PI / 180.0);
                                xVector[2] = Math.Floor((xVector[2] * 180.0 / Math.PI) / degDiv) * degDiv * (Math.PI * 180.0);

                                // convert back to cartesian coordinates
                                crossVector[0] = xVector[0] * Math.Cos(xVector[1]) * Math.Sin(xVector[2]);
                                crossVector[1] = xVector[0] * Math.Sin(xVector[1]) * Math.Sin(xVector[2]);
                                crossVector[2] = xVector[0] * Math.Cos(xVector[2]);
                            }

                            depthImgOutCV.Data[x, y, 0] = (float)crossVector[0];
                            depthImgOutCV.Data[x, y, 1] = (float)crossVector[1];
                            depthImgOutCV.Data[x, y, 2] = (float)crossVector[2];

                            double localMin = (crossVector[0] < crossVector[1]) ? crossVector[0] : crossVector[1];
                            localMin = (localMin < crossVector[2]) ? localMin : crossVector[2];

                            double localMax = (crossVector[0] > crossVector[1]) ? crossVector[0] : crossVector[1];
                            localMax = (localMax > crossVector[2]) ? localMax : crossVector[2];

                            // update min & max
                            lock (lockMinMax)
                            {
                                if (lockCount == 0)
                                {
                                    globalMin[0] = crossVector[0];
                                    globalMin[1] = crossVector[1];
                                    globalMin[2] = crossVector[2];

                                    globalMax[0] = crossVector[0];
                                    globalMax[1] = crossVector[1];
                                    globalMax[2] = crossVector[2];

                                    lockCount++;
                                }
                                else
                                {
                                    globalMin[0] = (globalMin[0] < crossVector[0]) ? globalMin[0] : crossVector[0];
                                    globalMin[1] = (globalMin[1] < crossVector[1]) ? globalMin[1] : crossVector[1];
                                    globalMin[2] = (globalMin[2] < crossVector[2]) ? globalMin[2] : crossVector[2];

                                    globalMax[0] = (globalMax[0] > crossVector[0]) ? globalMax[0] : crossVector[0];
                                    globalMax[1] = (globalMax[1] > crossVector[1]) ? globalMax[1] : crossVector[1];
                                    globalMax[2] = (globalMax[2] > crossVector[2]) ? globalMax[2] : crossVector[2];
                                }
                            }
                        }
                        else
                        {
                            depthImgOutCV.Data[x, y, 0] = 0.0f;
                            depthImgOutCV.Data[x, y, 1] = 0.0f;
                            depthImgOutCV.Data[x, y, 2] = 0.0f;

                            // update min & max
                            lock (lockMinMax)
                            {
                                if (lockCount == 0)
                                {
                                    globalMin[0] = 0.0;
                                    globalMin[1] = 0.0;
                                    globalMin[2] = 0.0;

                                    globalMax[0] = 0.0;
                                    globalMax[1] = 0.0;
                                    globalMax[2] = 0.0;

                                    lockCount++;
                                }
                                else
                                {
                                    globalMin[0] = (globalMin[0] < 0.0) ? globalMin[0] : 0.0;
                                    globalMin[1] = (globalMin[1] < 0.0) ? globalMin[1] : 0.0;
                                    globalMin[2] = (globalMin[2] < 0.0) ? globalMin[2] : 0.0;

                                    globalMax[0] = (globalMax[0] > 0.0) ? globalMax[0] : 0.0;
                                    globalMax[1] = (globalMax[1] > 0.0) ? globalMax[1] : 0.0;
                                    globalMax[2] = (globalMax[2] > 0.0) ? globalMax[2] : 0.0;
                                }
                            }
                        }
                    }
                });

                // save the points to the output buffer
                Parallel.For(0, depthImgOutCV.Cols * depthImgOutCV.Rows, i =>
                {
                    int x = i / this.sensorChooser.Kinect.DepthStream.FrameWidth;
                    int y = (this.sensorChooser.Kinect.DepthStream.FrameWidth - 1) - i % this.sensorChooser.Kinect.DepthStream.FrameWidth;

                    // Write out blue byte
                    //this.depthRGBPixels[i * 4] = (byte)(short)((depthImgOutCV.Data[x, y, 0] - globalMin[0]) / (globalMax[0] - globalMin[0]) * 255.0);
                    //this.depthRGBPixels[i * 4] = (byte)(short)(depthImgOutCV.Data[x, y, 0] * 255.0);
                    //this.depthRGBPixels[i * 4] = (byte)0;

                    // Write out green byte
                    this.depthRGBPixels[i * 4 + 1] = (byte)(short)((depthImgOutCV.Data[x, y, 1] - globalMin[1]) / (globalMax[1] - globalMin[1]) * 255.0);
                    //this.depthRGBPixels[i * 4 + 1] = (byte)(short)(depthImgOutCV.Data[x, y, 1] * 255.0);
                    //this.depthRGBPixels[i * 4 + 1] = (byte)0;

                    // Write out red byte                        
                    this.depthRGBPixels[i * 4 + 2] = (byte)(short)((depthImgOutCV.Data[x, y, 2] - globalMin[2]) / (globalMax[2] - globalMin[2]) * 255.0);
                    //this.depthRGBPixels[i * 4 + 2] = (byte)(short)(depthImgOutCV.Data[x, y, 2] * 255.0);
                    //this.depthRGBPixels[i * 4 + 2] = (byte)0;
                });

                // Write the pixel data into our bitmap
                try{
                    this.Dispatcher.Invoke((Action)(() =>
                    {
                        this.depthBitmap.WritePixels(
                            new Int32Rect(0, 0, this.depthBitmap.PixelWidth, this.depthBitmap.PixelHeight),
                            this.depthRGBPixels,
                            this.depthBitmap.PixelWidth * sizeof(int),
                            0);
                    }));
                }
                catch (Exception ex){
                    Console.WriteLine(ex.Message);
                }
            }

            lock (dpLock)
            {
                isActive = false;
            }
        }


        private void ProcessDepthDataAlt(int minDepth, int maxDepth)
        {
            // Save this frame in our ring buffer
            Parallel.For(0, depthPixels.Length, i =>
            {
                // Get the depth for this pixel
                short depth = depthPixels[i].Depth;

                // if we have a full ring buffer, subtract the oldest entry from the running sum
                if (ready)
                    depthRunningSum[i] -= depthRingBuffer[ringBufIdx * frameSize + i];

                // save this pixel for this frame, and update the running sum
                depthRingBuffer[ringBufIdx * frameSize + i] = (depth >= minDepth && depth <= maxDepth) ? depth : (short)0;
                depthRunningSum[i] += depthRingBuffer[ringBufIdx * frameSize + i];
            });

            // increment the ring buffer index
            ringBufIdx = (ringBufIdx + 1) % ringBufferSize;

            // if we have a full ring buffer, enable image processing
            if (!ready && ringBufIdx == 0)
                ready = true;

            // perform image processing if we have a full circular buffer
            if (ready)
            {
                double fx = depthImgCV.Rows / (2.0 * Math.Tan(43 / 2));
                double fy = depthImgCV.Cols / (2.0 * Math.Tan(57 / 2));

                // Convert from pixel space to real world space
                Parallel.For(0, depthRunningSum.Length, i =>
                {
                    float floatVal = depthRunningSum[i] / ringBufferSize;

                    int x = i / sensorChooser.Kinect.DepthStream.FrameWidth;
                    int y = (sensorChooser.Kinect.DepthStream.FrameWidth - 1) - i % sensorChooser.Kinect.DepthStream.FrameWidth;

                    // Write out blue (x-distance) byte
                    depthImgCV.Data[x, y, 0] = (float)(floatVal * ((double)x - (depthImgCV.Rows / 2.0 - 1)) / fx);

                    // Write out green (y-distance) byte
                    depthImgCV.Data[x, y, 1] = (float)(floatVal * ((double)y - (depthImgCV.Cols / 2.0-1)) / fy);

                    // Write out red (z-distance) byte                        
                    depthImgCV.Data[x, y, 2] = floatVal;
                });

                // *******************************************
                //            Start Plane Detection
                // *******************************************

                // first compute x & y derivatives
                Parallel.For(0, 2, i =>
                {
                    if (i == 0)
                        dxImgCV = depthImgCV.Sobel(1, 0, depthKernelSize);
                    else
                        dyImgCV = depthImgCV.Sobel(0, 1, depthKernelSize);
                });

                // next, compute the normal vector at each point (excluding the outermost pixels)
                Parallel.For(0, depthImgCV.Cols * depthImgCV.Rows, i =>
                {
                    int x = i / depthImgCV.Cols;
                    int y = i % depthImgCV.Cols;

                    // compute cross product
                    depthImgOutCV.Data[x, y, 0] = dyImgCV.Data[x, y, 1] * dxImgCV.Data[x, y, 2] - dyImgCV.Data[x, y, 2] * dxImgCV.Data[x, y, 1];
                    depthImgOutCV.Data[x, y, 1] = dyImgCV.Data[x, y, 2] * dxImgCV.Data[x, y, 0] - dyImgCV.Data[x, y, 0] * dxImgCV.Data[x, y, 2];
                    depthImgOutCV.Data[x, y, 2] = dyImgCV.Data[x, y, 0] * dxImgCV.Data[x, y, 1] - dyImgCV.Data[x, y, 1] * dxImgCV.Data[x, y, 0];

                    pixelMag[i] = (float)Math.Sqrt(depthImgOutCV.Data[x, y, 0] * depthImgOutCV.Data[x, y, 0] + depthImgOutCV.Data[x, y, 1] * depthImgOutCV.Data[x, y, 1] + depthImgOutCV.Data[x, y, 2] * depthImgOutCV.Data[x, y, 2]);

                    //samples.Data[i, 0] = depthImgOutCV.Data[x, y, 0];
                    //samples.Data[i, 1] = depthImgOutCV.Data[x, y, 1];
                    //samples.Data[i, 2] = depthImgOutCV.Data[x, y, 2];

                    //if (pixelMag[i] > 0)
                    //{
                    //    // convert to spherical coords
                    //    samples.Data[i, 0] = pixelMag[i]; // r
                    //    samples.Data[i, 1] = (float)Math.Atan2(depthImgOutCV.Data[x, y, 1], depthImgOutCV.Data[x, y, 0]); // theta
                    //    samples.Data[i, 2] = (float)Math.Acos(depthImgOutCV.Data[x, y, 2] / pixelMag[i]); // phi
                    //}
                    //else
                    //{
                    //    samples.Data[i, 0] = 0;
                    //    samples.Data[i, 1] = 0;
                    //    samples.Data[i, 2] = 0;
                    //}

                    //if (mag == 0)
                    //    Console.WriteLine("Found a 0 magnitude vector!!!");

                    // normalize
                    //depthImgOutCV.Data[x, y, 0] /= mag;
                    //depthImgOutCV.Data[x, y, 1] /= mag;
                    //depthImgOutCV.Data[x, y, 2] /= mag;
                });

                if (useFloodFill)
                {

                    //int groupId = OldFill();
                    int groupId = FastFill();

                    int maxColor = (int)Math.Pow(2, 24);

                    // save the points to the output buffer
                    Parallel.For(0, depthImgOutCV.Cols * depthImgOutCV.Rows, i =>
                    {
                        int x = i / sensorChooser.Kinect.DepthStream.FrameWidth;
                        int y = i % sensorChooser.Kinect.DepthStream.FrameWidth;

                        int color = maxColor / groupId * (pixelState[i]);

                        if (color < 0 || groupCount[pixelState[i]] < minGroupSize)
                            color = 0;

                        // Write out blue byte
                        this.depthRGBPixels[i * 4] = (byte)(color & 0xFF);

                        // Write out green byte
                        this.depthRGBPixels[i * 4 + 1] = (byte)((color >> 8) & 0xFF);

                        // Write out red byte                        
                        this.depthRGBPixels[i * 4 + 2] = (byte)((color >> 16) & 0xFF);
                    });
                }
                else if (useKmeans)
                {
                    int k = 30;

                    // compute Kmeans 20 times
                    Parallel.For(0, 1, i =>
                    {
                        MCvTermCriteria term = new MCvTermCriteria(50, 0.5);
                        term.type = TERMCRIT.CV_TERMCRIT_ITER | TERMCRIT.CV_TERMCRIT_EPS;
                        CvInvoke.cvKMeans2(samples, i + k, finalClusters[i], term, 1, IntPtr.Zero, KMeansInitType.PPCenters, IntPtr.Zero, new IntPtr(compactness.Data[i, 0]));
                    });

                    int maxColor = (int)Math.Pow(2, 24);

                    // save the points to the output buffer
                    Parallel.For(0, depthImgOutCV.Cols * depthImgOutCV.Rows, i =>
                    {
                        int x = i / sensorChooser.Kinect.DepthStream.FrameWidth;
                        int y = i % sensorChooser.Kinect.DepthStream.FrameWidth;

                        int color = maxColor / k * finalClusters[0].Data[i, 0];

                        // Write out blue byte
                        this.depthRGBPixels[i * 4] = (byte)(color & 0xFF);

                        // Write out green byte
                        this.depthRGBPixels[i * 4 + 1] = (byte)((color >> 8) & 0xFF);

                        // Write out red byte                        
                        this.depthRGBPixels[i * 4 + 2] = (byte)((color >> 16) & 0xFF);
                    });
                }
                else
                {
                    Parallel.For(0, depthImgOutCV.Cols * depthImgOutCV.Rows, i =>
                    {
                        int x = i / sensorChooser.Kinect.DepthStream.FrameWidth;
                        int y = i % sensorChooser.Kinect.DepthStream.FrameWidth;

                        depthImgOutCV.Data[x, y, 0] *= (255.0f / pixelMag[i]);
                        depthImgOutCV.Data[x, y, 1] *= (255.0f / pixelMag[i]);
                        depthImgOutCV.Data[x, y, 2] *= (255.0f / pixelMag[i]);
                    });

                    Image<Gray, Byte> grayImage = depthImgOutCV.Convert<Gray, Byte>();

                    grayImage._SmoothGaussian(3, 3, 2, 2);

                    grayImage = grayImage.Canny(150, 15);

                    List<System.Drawing.Point[]> myPoly = polyDetect(grayImage);

                    if (myPoly != null)
                    {
                        foreach (System.Drawing.Point[] polyLine in myPoly)
                            depthImgOutCV.DrawPolyline(polyLine, true, new Bgr(System.Drawing.Color.Red), 10);
                    }

                    // save the points to the output buffer
                    Parallel.For(0, depthImgOutCV.Cols * depthImgOutCV.Rows, i =>
                    {
                        int x = i / sensorChooser.Kinect.DepthStream.FrameWidth;
                        int y = i % sensorChooser.Kinect.DepthStream.FrameWidth;

                        // Write out blue byte
                        this.depthRGBPixels[i * 4] = (byte)(short)depthImgOutCV.Data[x, y, 0];

                        // Write out green byte
                        this.depthRGBPixels[i * 4 + 1] = (byte)(short)depthImgOutCV.Data[x, y, 1];

                        // Write out red byte                        
                        this.depthRGBPixels[i * 4 + 2] = (byte)(short)depthImgOutCV.Data[x, y, 2];
                    });
                }

                // Write the pixel data into our bitmap
                try
                {
                    Dispatcher.Invoke((Action)(() =>
                    {
                        depthBitmap.WritePixels(
                            new Int32Rect(0, 0, depthBitmap.PixelWidth, depthBitmap.PixelHeight),
                            depthRGBPixels,
                            depthBitmap.PixelWidth * sizeof(int),
                            0);
                    }));
                }
                catch (Exception ex)
                {
                    Console.WriteLine(ex.Message);
                }
            }

            lock (dpLock)
            {
                isActive = false;
            }
        }


        private void ProcessDepthDataAlt2(int minDepth, int maxDepth)
        {
            // Save this frame in our ring buffer
            Parallel.For(0, depthPixels.Length, i =>
            {
                // Get the depth for this pixel
                short depth = depthPixels[i].Depth;

                // if we have a full ring buffer, subtract the oldest entry from the running sum
                if (ready)
                    depthRunningSum[i] -= depthRingBuffer[ringBufIdx * frameSize + i];

                depth = (short)((depth >= minDepth && depth <= maxDepth) ? depth - (minDepth-1) : 0);

                // save this pixel for this frame, and update the running sum
                depthRingBuffer[ringBufIdx * frameSize + i] = depth;
                depthRunningSum[i] += depthRingBuffer[ringBufIdx * frameSize + i];
            });

            // increment the ring buffer index
            ringBufIdx = (ringBufIdx + 1) % ringBufferSize;

            // if we have a full ring buffer, enable image processing
            if (!ready && ringBufIdx == 0)
                ready = true;

            // perform image processing if we have a full circular buffer
            if (ready)
            {
                // Convert from pixel space to real world space
                Parallel.For(0, depthRunningSum.Length, i =>
                {
                    float floatVal = depthRunningSum[i] / ringBufferSize;

                    int x = i / sensorChooser.Kinect.DepthStream.FrameWidth;
                    int y = (sensorChooser.Kinect.DepthStream.FrameWidth - 1) - i % sensorChooser.Kinect.DepthStream.FrameWidth;

                    // Write out blue (x-distance) byte
                    grayImgCV.Data[x, y, 0] = (byte)(short)(floatVal / (maxDepth-minDepth) * 255);
                    depthImgCV.Data[x, y, 0] = floatVal / (maxDepth - minDepth) * 240;

                    // Write out green (y-distance) byte
                    grayImgCV.Data[x, y, 1] = (byte)(short)(floatVal / (maxDepth - minDepth) * 255);
                    depthImgCV.Data[x, y, 1] = floatVal / (maxDepth - minDepth) * 240;

                    // Write out red (z-distance) byte                        
                    grayImgCV.Data[x, y, 2] = (byte)(short)(floatVal / (maxDepth - minDepth) * 255);
                    depthImgCV.Data[x, y, 2] = floatVal / (maxDepth - minDepth) * 240;
                });

                // *******************************************
                //            Start Plane Detection
                // *******************************************

                // first compute x & y derivatives
                Parallel.For(0, 2, i =>
                {
                    if (i == 0)
                        dxImgCV = depthImgCV.Sobel(1, 0, depthKernelSize).Pow(2);
                    else
                        dyImgCV = depthImgCV.Sobel(0, 1, depthKernelSize).Pow(2);
                });

                depthImgCV = dxImgCV.Add(dyImgCV).Pow(0.5).Dilate(1);

                double[] minVal;
                double[] maxVal;
                System.Drawing.Point[] minLoc;
                System.Drawing.Point[] maxLoc;

                depthImgCV.MinMax(out minVal, out maxVal, out minLoc, out maxLoc);

                //Image<Gray, Byte> grayTempCV = depthImgCV.Convert<Gray, Byte>();

                // Convert from pixel space to real world space
                //Parallel.For(0, depthRunningSum.Length, i =>
                //{
                //    int x = i / sensorChooser.Kinect.DepthStream.FrameWidth;
                //    int y = (sensorChooser.Kinect.DepthStream.FrameWidth - 1) - i % sensorChooser.Kinect.DepthStream.FrameWidth;
                //
                //    if (depthImgCV.Data[x, y, 0] > maxVal[0] / 200)
                //    {
                //        depthImgCV.Data[x, y, 0] = 255;
                //        depthImgCV.Data[x, y, 1] = 255;
                //        depthImgCV.Data[x, y, 2] = 255;
                //        grayTempCV.Data[x, y, 0] = 255;
                //    }
                //    else
                //    {
                //        depthImgCV.Data[x, y, 0] = 0;
                //        depthImgCV.Data[x, y, 1] = 0;
                //        depthImgCV.Data[x, y, 2] = 0;
                //        grayTempCV.Data[x, y, 0] = 0;
                //    }
                //});

                //Apply Gaussian smoothing with 3x3 kernel and simga = 2.
                //grayImgCV._SmoothGaussian(3, 3, 2, 2);

                //Run canny detection to obtain image edges.
                Image<Gray, Byte> grayTempCV = grayImgCV.Canny(35, 25).Dilate(2);

                List<System.Drawing.Point[]> myPoly = polyDetect(grayTempCV);
                
                if (myPoly != null)
                {
                    foreach (System.Drawing.Point[] polyLine in myPoly)
                        grayImgCV.DrawPolyline(polyLine, true, new Bgr(System.Drawing.Color.Red), 2);
                }

                // save the points to the output buffer
                Parallel.For(0, depthImgCV.Cols * depthImgCV.Rows, i =>
                {
                    int x = i / sensorChooser.Kinect.DepthStream.FrameWidth;
                    int y = i % sensorChooser.Kinect.DepthStream.FrameWidth;

                    // Write out blue byte
                    this.depthRGBPixels[i * 4] = (byte)(short)grayImgCV.Data[x, y, 0];

                    // Write out green byte
                    this.depthRGBPixels[i * 4 + 1] = (byte)(short)grayImgCV.Data[x, y, 1];

                    // Write out red byte                        
                    this.depthRGBPixels[i * 4 + 2] = (byte)(short)grayImgCV.Data[x, y, 2];
                });


                // Write the pixel data into our bitmap
                try
                {
                    Dispatcher.Invoke((Action)(() =>
                    {
                        depthBitmap.WritePixels(
                            new Int32Rect(0, 0, depthBitmap.PixelWidth, depthBitmap.PixelHeight),
                            depthRGBPixels,
                            depthBitmap.PixelWidth * sizeof(int),
                            0);
                    }));
                }
                catch (Exception ex)
                {
                    Console.WriteLine(ex.Message);
                }
            }

            lock (dpLock)
            {
                isActive = false;
            }
        }

        private void ProcessDepthDataAlt3(int minDepth, int maxDepth)
        {
            // Save this frame in our ring buffer
            Parallel.For(0, depthPixels.Length, i =>
            {
                // Get the depth for this pixel
                short depth = depthPixels[i].Depth;

                // if we have a full ring buffer, subtract the oldest entry from the running sum
                if (ready)
                    depthRunningSum[i] -= depthRingBuffer[ringBufIdx * frameSize + i];

                // save this pixel for this frame, and update the running sum
                depthRingBuffer[ringBufIdx * frameSize + i] = (depth >= minDepth && depth <= maxDepth) ? depth : (short)0;
                depthRunningSum[i] += depthRingBuffer[ringBufIdx * frameSize + i];
            });

            // increment the ring buffer index
            ringBufIdx = (ringBufIdx + 1) % ringBufferSize;

            // if we have a full ring buffer, enable image processing
            if (!ready && ringBufIdx == 0)
                ready = true;

            // perform image processing if we have a full circular buffer
            if (ready)
            {
                double fx = depthImgCV.Rows / (2.0 * Math.Tan(43 / 2));
                double fy = depthImgCV.Cols / (2.0 * Math.Tan(57 / 2));

                // Convert from pixel space to real world space
                Parallel.For(0, depthRunningSum.Length, i =>
                {
                    float floatVal = depthRunningSum[i] / ringBufferSize;

                    int x = i / sensorChooser.Kinect.DepthStream.FrameWidth;
                    int y = (sensorChooser.Kinect.DepthStream.FrameWidth - 1) - i % sensorChooser.Kinect.DepthStream.FrameWidth;

                    // Write out blue (x-distance) byte
                    depthImgCV.Data[x, y, 0] = (float)(floatVal * ((double)x - (depthImgCV.Rows / 2.0 - 1)) / fx);

                    // Write out green (y-distance) byte
                    depthImgCV.Data[x, y, 1] = (float)(floatVal * ((double)y - (depthImgCV.Cols / 2.0 - 1)) / fy);

                    // Write out red (z-distance) byte                        
                    depthImgCV.Data[x, y, 2] = floatVal;
                });

                // first compute x & y derivatives
                Parallel.For(0, 2, i =>
                {
                    if (i == 0)
                        dxImgCV = depthImgCV.Sobel(1, 0, depthKernelSize);
                    else
                        dyImgCV = depthImgCV.Sobel(0, 1, depthKernelSize);
                });

                float maxMag = -1;

                // next, compute the normal vector at each point (excluding the outermost pixels)
                Parallel.For(0, depthImgCV.Cols * depthImgCV.Rows, i =>
                {
                    int x = i / depthImgCV.Cols;
                    int y = i % depthImgCV.Cols;

                    // compute cross product
                    depthImgOutCV.Data[x, y, 0] = dyImgCV.Data[x, y, 1] * dxImgCV.Data[x, y, 2] - dyImgCV.Data[x, y, 2] * dxImgCV.Data[x, y, 1];
                    depthImgOutCV.Data[x, y, 1] = dyImgCV.Data[x, y, 2] * dxImgCV.Data[x, y, 0] - dyImgCV.Data[x, y, 0] * dxImgCV.Data[x, y, 2];
                    depthImgOutCV.Data[x, y, 2] = dyImgCV.Data[x, y, 0] * dxImgCV.Data[x, y, 1] - dyImgCV.Data[x, y, 1] * dxImgCV.Data[x, y, 0];

                    pixelMag[i] = (float)Math.Sqrt(depthImgOutCV.Data[x, y, 0] * depthImgOutCV.Data[x, y, 0] + depthImgOutCV.Data[x, y, 1] * depthImgOutCV.Data[x, y, 1] + depthImgOutCV.Data[x, y, 2] * depthImgOutCV.Data[x, y, 2]);
                });

                for (int i = 0; i < depthImgCV.Cols * depthImgCV.Rows; ++i)
                    if (pixelMag[i] > maxMag)
                        maxMag = pixelMag[i];

                // next, compute the normal vector at each point (excluding the outermost pixels)
                Parallel.For(0, depthImgCV.Cols * depthImgCV.Rows, i =>
                {
                    int x = i / depthImgCV.Cols;
                    int y = i % depthImgCV.Cols;

                    if (pixelMag[i] > maxMag / 100)
                    {
                        depthImgOutCV.Data[x, y, 0] = 255;
                        depthImgOutCV.Data[x, y, 2] = 255;
                        depthImgOutCV.Data[x, y, 1] = 255;
                    }
                    else
                    {
                        depthImgOutCV.Data[x, y, 0] = 0;
                        depthImgOutCV.Data[x, y, 2] = 0;
                        depthImgOutCV.Data[x, y, 1] = 0;
                    }
                });

                //Run canny detection to obtain image edges.
                //Image<Gray, Byte> grayTempCV = depthImgOutCV.Convert<Gray, Byte>();//.Canny(900, 150);

                //List<System.Drawing.Point[]> myPoly = polyDetect(grayTempCV);
                
                //if (myPoly != null)
                //{
                //    foreach (System.Drawing.Point[] polyLine in myPoly)
                //        depthImgOutCV.DrawPolyline(polyLine, true, new Bgr(System.Drawing.Color.Red), 2);
                //}

                // save the points to the output buffer
                Parallel.For(0, depthImgCV.Cols * depthImgCV.Rows, i =>
                {
                    int x = i / sensorChooser.Kinect.DepthStream.FrameWidth;
                    int y = i % sensorChooser.Kinect.DepthStream.FrameWidth;

                    // Write out blue byte
                    this.depthRGBPixels[i * 4] = (byte)(short)depthImgOutCV.Data[x, y, 0];

                    // Write out green byte
                    this.depthRGBPixels[i * 4 + 1] = (byte)(short)depthImgOutCV.Data[x, y, 1];

                    // Write out red byte                        
                    this.depthRGBPixels[i * 4 + 2] = (byte)(short)depthImgOutCV.Data[x, y, 2];
                });


                // Write the pixel data into our bitmap
                try
                {
                    Dispatcher.Invoke((Action)(() =>
                    {
                        depthBitmap.WritePixels(
                            new Int32Rect(0, 0, depthBitmap.PixelWidth, depthBitmap.PixelHeight),
                            depthRGBPixels,
                            depthBitmap.PixelWidth * sizeof(int),
                            0);
                    }));
                }
                catch (Exception ex)
                {
                    Console.WriteLine(ex.Message);
                }
            }

            lock (dpLock)
            {
                isActive = false;
            }
        }

        private int OldFill()
        {
            groupCount = new List<int>();

            Array.Clear(pixelState, 0, pixelState.Length);

            int groupId = 0;

            // group the points using flood fill algorithm
            for (int i = 0; i < depthImgOutCV.Cols * depthImgOutCV.Rows; i++)
            {
                int x = i / sensorChooser.Kinect.DepthStream.FrameWidth;
                int y = i % sensorChooser.Kinect.DepthStream.FrameWidth;

                // if this point is already assigned, skip it
                if (pixelState[x * sensorChooser.Kinect.DepthStream.FrameWidth + y] > 0)
                    continue;

                groupId++;
                groupCount.Add(1);

                // queue to contain potential group members
                Queue<int[]> pixelQueue = new Queue<int[]>();

                // enqueue the current point as the first point
                pixelQueue.Enqueue(new int[2] { x, y });

                // loop until there are no more points in the queue
                while (pixelQueue.Count > 0)
                {
                    // get the current pixel coordinates
                    int[] pt = pixelQueue.Dequeue();

                    // make sure the point doesn't go out of bounds
                    if (pt[0] < 0 || pt[0] >= sensorChooser.Kinect.DepthStream.FrameHeight || pt[1] < 0 || pt[1] >= sensorChooser.Kinect.DepthStream.FrameWidth)
                        continue;

                    int ptIdx = pt[0] * sensorChooser.Kinect.DepthStream.FrameWidth + pt[1];

                    // get the pixel's state
                    int pState = pixelState[ptIdx];

                    // positive groupId indicates that this point already belongs to our group
                    // negative groupId inficates that we've already visited this point
                    if (Math.Abs(pState) == groupId)
                        continue;

                    // if this pixel is owned by another group
                    if (pState > 0)
                        continue;

                    // calculate the dot product between this point and our reference point
                    float dotProd = depthImgOutCV.Data[x, y, 0] / pixelMag[i] * depthImgOutCV.Data[pt[0], pt[1], 0] / pixelMag[ptIdx] +
                                    depthImgOutCV.Data[x, y, 1] / pixelMag[i] * depthImgOutCV.Data[pt[0], pt[1], 1] / pixelMag[ptIdx] +
                                    depthImgOutCV.Data[x, y, 2] / pixelMag[i] * depthImgOutCV.Data[pt[0], pt[1], 2] / pixelMag[ptIdx];

                    float perp = (depthImgCV.Data[x, y, 0] - depthImgCV.Data[pt[0], pt[1], 0]) * depthImgOutCV.Data[x, y, 0] / pixelMag[i] +
                                 (depthImgCV.Data[x, y, 1] - depthImgCV.Data[pt[0], pt[1], 1]) * depthImgOutCV.Data[x, y, 1] / pixelMag[i] +
                                 (depthImgCV.Data[x, y, 2] - depthImgCV.Data[pt[0], pt[1], 2]) * depthImgOutCV.Data[x, y, 2] / pixelMag[i];

                    // if our normals are close enough
                    if (pixelMag[i] / pixelMag[ptIdx] > 0.75 && pixelMag[i] / pixelMag[ptIdx] < 1.25)// && dotProd > minDotProd)
                    //if (dotProd > minDotProd)
                    // if (dotProd > minDotProd && perp < maxPerpendicular)
                    {
                        // increment the group counter
                        groupCount[groupId - 1]++;

                        // mark this point as belonging to this group with +groupId
                        pixelState[pt[0] * sensorChooser.Kinect.DepthStream.FrameWidth + pt[1]] = groupId;

                        // add the north neighbor
                        pixelQueue.Enqueue(new int[2] { pt[0] - 1, pt[1] });

                        // add the north-east neighbor
                        //pixelQueue.Enqueue(new int[2] { pt[0] - 1, pt[1] + 1 });

                        // add the east neighbor
                        pixelQueue.Enqueue(new int[2] { pt[0], pt[1] + 1 });

                        // add the south-east neighbor
                        //pixelQueue.Enqueue(new int[2] { pt[0] + 1, pt[1] + 1 });

                        // add the south neighbor
                        pixelQueue.Enqueue(new int[2] { pt[0] + 1, pt[1] });

                        // add the south-west neighbor
                        //pixelQueue.Enqueue(new int[2] { pt[0] + 1, pt[1] - 1 });

                        // add the west neighbor
                        pixelQueue.Enqueue(new int[2] { pt[0], pt[1] - 1 });

                        // add the north-west neighbor
                        //pixelQueue.Enqueue(new int[2] { pt[0] - 1, pt[1] - 1 });
                    }
                    //else if (dotProd == 0)
                    //{
                    //    // increment the group counter
                    //    groupCount[groupId - 1]++;
                    //
                    //    // mark this point as belonging to this group with +groupId
                    //    pixelState[pt[0] * sensorChooser.Kinect.DepthStream.FrameWidth + pt[1]] = groupId;
                    //}
                    else
                    {
                        // mark the pixel as visited with -groupId
                        pixelState[pt[0] * sensorChooser.Kinect.DepthStream.FrameWidth + pt[1]] = -groupId;
                    }
                }
            }
            return groupId;
        }

        private int FastFill()
        {
            groupCount = new List<int>();
            groupCount.Add(0); // 0th index represents nothing
            groupCount.Add(0); // 1st index represents points with magnitude == 0

            Array.Clear(pixelState, 0, pixelState.Length);

            int groupId = 1; // 1 is reserved for 0 magnitude points

            // group the points using flood fill algorithm
            for (int i = 0; i < depthImgOutCV.Cols * depthImgOutCV.Rows; i++)
            {
                int x = i / sensorChooser.Kinect.DepthStream.FrameWidth;
                int y = i % sensorChooser.Kinect.DepthStream.FrameWidth;

                // if this point is already assigned, skip it
                if (pixelState[x * sensorChooser.Kinect.DepthStream.FrameWidth + y] > 0)
                    continue;

                // if magnitude is 0, mark as group 1 and skip
                if (pixelMag[x * sensorChooser.Kinect.DepthStream.FrameWidth + y] == 0)
                {
                    // increment the group counter
                    groupCount[1]++;

                    // mark this point as belonging to this group with +groupId
                    pixelState[x * sensorChooser.Kinect.DepthStream.FrameWidth + y] = 1;

                    continue;
                }

                // make this point the first member of a new group
                groupId++;
                groupCount.Add(0);

                // queue to contain potential group members
                Queue<int[]> pixelQueue = new Queue<int[]>();
                Object qLock = new Object();

                // enqueue the current point as the first point
                pixelQueue.Enqueue(new int[2] { x, y });

                // loop until there are no more points in the queue
                while (pixelQueue.Count > 0)
                {
                    // get the current pixel coordinates
                    int[] pt = pixelQueue.Dequeue();

                    // if the current pixel is already assigned, continue
                    if (pixelState[pt[0] * sensorChooser.Kinect.DepthStream.FrameWidth + pt[1]] > 0)
                        continue;

                    // add this pixel to the group
                    pixelState[pt[0] * sensorChooser.Kinect.DepthStream.FrameWidth + pt[1]] = groupId;
                    groupCount[groupId]++;

                    // west and east index counters
                    int w = pt[1];
                    int e = pt[1];

                    // walk east and west to determine our bounds for this line
                    Parallel.For(0, 2, j =>
                    {
                        // walk west
                        if (j == 0)
                        {
                            while (--w >= 0)
                            {
                                int lastIdx = pt[0] * sensorChooser.Kinect.DepthStream.FrameWidth + w + 1;
                                int curIdx = pt[0] * sensorChooser.Kinect.DepthStream.FrameWidth + w;

                                // if magnitude is 0, mark as group 1 & end scan
                                if (pixelMag[curIdx] == 0)
                                {
                                    // increment the group counter
                                    lock (qLock)
                                    {
                                        groupCount[1]++;
                                    }

                                    // mark this point as belonging to this group with +groupId
                                    pixelState[curIdx] = 1;

                                    break;
                                }

                                float dotProd = depthImgOutCV.Data[pt[0], w, 0] / pixelMag[curIdx] * depthImgOutCV.Data[pt[0], w + 1, 0] / pixelMag[lastIdx] +
                                                depthImgOutCV.Data[pt[0], w, 1] / pixelMag[curIdx] * depthImgOutCV.Data[pt[0], w + 1, 1] / pixelMag[lastIdx] +
                                                depthImgOutCV.Data[pt[0], w, 2] / pixelMag[curIdx] * depthImgOutCV.Data[pt[0], w + 1, 2] / pixelMag[lastIdx];

                                // if color matches
                                if (dotProd > minDotProd && (pixelMag[curIdx] / pixelMag[lastIdx] > (1 - planeFactor) && pixelMag[curIdx] / pixelMag[lastIdx] < (1 + planeFactor)))
                                {
                                    // increment the group counter
                                    lock (qLock)
                                    {
                                        groupCount[groupId]++;
                                    }

                                    // mark this point as belonging to this group with +groupId
                                    pixelState[curIdx] = groupId;

                                    // if north neighbor is a match, add them to the queue
                                    if (pt[0] - 1 > 0)
                                    {
                                        lastIdx = (pt[0] - 1) * sensorChooser.Kinect.DepthStream.FrameWidth + w;
                                        dotProd = depthImgOutCV.Data[pt[0], w, 0] / pixelMag[curIdx] * depthImgOutCV.Data[pt[0] - 1, w, 0] / pixelMag[lastIdx] +
                                                  depthImgOutCV.Data[pt[0], w, 1] / pixelMag[curIdx] * depthImgOutCV.Data[pt[0] - 1, w, 1] / pixelMag[lastIdx] +
                                                  depthImgOutCV.Data[pt[0], w, 2] / pixelMag[curIdx] * depthImgOutCV.Data[pt[0] - 1, w, 2] / pixelMag[lastIdx];

                                        // if the north point is a match and is unassigned
                                        if (dotProd > minDotProd && pixelState[lastIdx] == 0 && (pixelMag[curIdx] / pixelMag[lastIdx] > (1 - planeFactor) && pixelMag[curIdx] / pixelMag[lastIdx] < (1 + planeFactor)))
                                        {
                                            // add that point to the queue
                                            lock (qLock)
                                                pixelQueue.Enqueue(new int[2] { pt[0] - 1, w });
                                        }
                                    }

                                    // if south neighbor is a match, add them to the queue
                                    if (pt[0] + 1 < sensorChooser.Kinect.DepthStream.FrameHeight)
                                    {
                                        lastIdx = (pt[0] + 1) * sensorChooser.Kinect.DepthStream.FrameWidth + w;
                                        dotProd = depthImgOutCV.Data[pt[0], w, 0] / pixelMag[curIdx] * depthImgOutCV.Data[pt[0] + 1, w, 0] / pixelMag[lastIdx] +
                                                  depthImgOutCV.Data[pt[0], w, 1] / pixelMag[curIdx] * depthImgOutCV.Data[pt[0] + 1, w, 1] / pixelMag[lastIdx] +
                                                  depthImgOutCV.Data[pt[0], w, 2] / pixelMag[curIdx] * depthImgOutCV.Data[pt[0] + 1, w, 2] / pixelMag[lastIdx];

                                        // if the north point is a match and is unassigned
                                        if (dotProd > minDotProd && pixelState[lastIdx] == 0 && (pixelMag[curIdx] / pixelMag[lastIdx] > (1 - planeFactor) && pixelMag[curIdx] / pixelMag[lastIdx] < (1 + planeFactor)))
                                        {
                                            // add that point to the queue
                                            lock (qLock)
                                                pixelQueue.Enqueue(new int[2] { pt[0] + 1, w });
                                        }
                                    }
                                }
                                // else stop iterating
                                else
                                {
                                    break;
                                }
                            }
                        }
                        // walk east
                        else
                        {
                            while (++e < sensorChooser.Kinect.DepthStream.FrameWidth)
                            {
                                int lastIdx = pt[0] * sensorChooser.Kinect.DepthStream.FrameWidth + e - 1;
                                int curIdx = pt[0] * sensorChooser.Kinect.DepthStream.FrameWidth + e;

                                // if magnitude is 0, mark as group 1 & end scan
                                if (pixelMag[curIdx] == 0)
                                {
                                    // increment the group counter
                                    lock (qLock)
                                    {
                                        groupCount[1]++;
                                    }

                                    // mark this point as belonging to this group with +groupId
                                    pixelState[curIdx] = 1;

                                    break;
                                }

                                float dotProd = depthImgOutCV.Data[pt[0], e, 0] / pixelMag[curIdx] * depthImgOutCV.Data[pt[0], e - 1, 0] / pixelMag[lastIdx] +
                                                depthImgOutCV.Data[pt[0], e, 1] / pixelMag[curIdx] * depthImgOutCV.Data[pt[0], e - 1, 1] / pixelMag[lastIdx] +
                                                depthImgOutCV.Data[pt[0], e, 2] / pixelMag[curIdx] * depthImgOutCV.Data[pt[0], e - 1, 2] / pixelMag[lastIdx];

                                // if color matches
                                if (dotProd > minDotProd && (pixelMag[curIdx] / pixelMag[lastIdx] > (1 - planeFactor) && pixelMag[curIdx] / pixelMag[lastIdx] < (1 + planeFactor)))
                                {
                                    // increment the group counter
                                    lock (qLock)
                                    {
                                        groupCount[groupId]++;
                                    }

                                    // mark this point as belonging to this group with +groupId
                                    pixelState[curIdx] = groupId;

                                    // if north neighbor is a match, add them to the queue
                                    if (pt[0] - 1 > 0)
                                    {
                                        lastIdx = (pt[0] - 1) * sensorChooser.Kinect.DepthStream.FrameWidth + e;
                                        dotProd = depthImgOutCV.Data[pt[0], e, 0] / pixelMag[curIdx] * depthImgOutCV.Data[pt[0] - 1, e, 0] / pixelMag[lastIdx] +
                                                  depthImgOutCV.Data[pt[0], e, 1] / pixelMag[curIdx] * depthImgOutCV.Data[pt[0] - 1, e, 1] / pixelMag[lastIdx] +
                                                  depthImgOutCV.Data[pt[0], e, 2] / pixelMag[curIdx] * depthImgOutCV.Data[pt[0] - 1, e, 2] / pixelMag[lastIdx];

                                        // if the north point is a match and is unassigned
                                        if (dotProd > minDotProd && pixelState[lastIdx] == 0 && (pixelMag[curIdx] / pixelMag[lastIdx] > (1 - planeFactor) && pixelMag[curIdx] / pixelMag[lastIdx] < (1 + planeFactor)))
                                        {
                                            // add that point to the queue
                                            lock (qLock)
                                                pixelQueue.Enqueue(new int[2] { pt[0] - 1, e });
                                        }
                                    }

                                    // if south neighbor is a match, add them to the queue
                                    if (pt[0] + 1 < sensorChooser.Kinect.DepthStream.FrameHeight)
                                    {
                                        lastIdx = (pt[0] + 1) * sensorChooser.Kinect.DepthStream.FrameWidth + e;
                                        dotProd = depthImgOutCV.Data[pt[0], e, 0] / pixelMag[curIdx] * depthImgOutCV.Data[pt[0] + 1, e, 0] / pixelMag[lastIdx] +
                                                  depthImgOutCV.Data[pt[0], e, 1] / pixelMag[curIdx] * depthImgOutCV.Data[pt[0] + 1, e, 1] / pixelMag[lastIdx] +
                                                  depthImgOutCV.Data[pt[0], e, 2] / pixelMag[curIdx] * depthImgOutCV.Data[pt[0] + 1, e, 2] / pixelMag[lastIdx];

                                        // if the north point is a match and is unassigned
                                        if (dotProd > minDotProd && pixelState[lastIdx] == 0 && (pixelMag[curIdx] / pixelMag[lastIdx] > (1 - planeFactor) && pixelMag[curIdx] / pixelMag[lastIdx] < (1 + planeFactor)))
                                        {
                                            // add that point to the queue
                                            lock (qLock)
                                                pixelQueue.Enqueue(new int[2] { pt[0] + 1, e });
                                        }
                                    }
                                }
                                // else stop iterating
                                else
                                {
                                    break;
                                }
                            }
                        }

                    });
                }
            }

            return groupId;
        }

        //Kinect sensor chooser. Honestly its really only impacts the skeleton tracking settings for the most part, as
        //the depth and color frame data and settings are handled same for Xbox and Windows based sensors.
        //Only the Xbox sensor and 3rd party sensors can not do skeletal traking in near mode.

        private void SensorChooserOnKinectChanged(object sender, KinectChangedEventArgs args)
        {
            bool error = false;
            if(args.OldSensor != null)
            {
                try
                {
                    args.OldSensor.ColorStream.Disable();
                    args.OldSensor.DepthStream.Range = DepthRange.Default;
                    args.OldSensor.SkeletonStream.EnableTrackingInNearRange = false;
                    args.OldSensor.DepthStream.Disable();
                    args.OldSensor.SkeletonStream.Disable();
                }
                catch (InvalidOperationException)
                {
                    //KinectSensor might enter an invalid state while enabling/disabling
                    error = true;
                }
            }

            if (args.NewSensor != null)
            {
                TransformSmoothParameters myParam = new TransformSmoothParameters();
                myParam.Smoothing = 0.5f;
                try
                {
                    args.NewSensor.ColorStream.Enable(ColorImageFormat.RgbResolution640x480Fps30);
                    args.NewSensor.DepthStream.Enable(DepthImageFormat.Resolution640x480Fps30);
                    //args.NewSensor.SkeletonStream.Enable(myParam);
                    args.NewSensor.ElevationAngle = 0;

                    try
                    {
                        args.NewSensor.DepthStream.Range = DepthRange.Near;
                        //args.NewSensor.SkeletonStream.EnableTrackingInNearRange = true;
                        //args.NewSensor.SkeletonStream.TrackingMode = SkeletonTrackingMode.Seated;
                        //args.NewSensor.SkeletonStream.Enable(myParam);
                    }
                    catch (InvalidOperationException)
                    {
                        //Non Kinect for Windows device, near mode not supported.
                        args.NewSensor.DepthStream.Range = DepthRange.Default;
                        //args.NewSensor.SkeletonStream.EnableTrackingInNearRange = false;
                        error = true;
                    }
                }
                catch (InvalidOperationException)
                {
                    error = true;
                    //KinectSensor might enter invalid state while enabling/disabling
                }
            }

            //if (!error) KinectRegion.KinectSensor = args.NewSensor;
        }
    }
}
