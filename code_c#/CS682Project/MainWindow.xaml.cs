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

        WriteableBitmap depthImageBitmap = null;
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

        Image<Bgr, Byte> CVKinectDepthFrame;
        Image<Bgr, Byte> CVKinectColorFrame;

        private bool dpActive = false;
        private Object dpLock = new Object();

        private bool clrActive = false;
        private Object clrLock = new Object();

        // Whitney's additions
        private const int ringBufferSize = 1;
        private const int depthKernelSize = 1; //11
        private const double minDotProd = 0.82; //.99
        private const bool useFloodFill = false;
        private const bool useEdgeDetect = false;
        private const bool useCustom = true;
        private const int minGroupSize = 1500;

        private DepthImagePixel[] depthPixels;
        private WriteableBitmap depthBitmap = null;

        private int ringBufIdx = 0;
        private int frameSize;
        private bool ready = false;
        private short[] depthRingBuffer;
        private long[] depthRunningSum;
        private byte[] depthRGBPixels;

        private Image<Bgr, Single> depthImgCV;
        private Image<Bgr, Single> dxImgCV;
        private Image<Bgr, Single> dyImgCV;
        private Image<Bgr, Single> depthImgOutCV;
        private float[] pixelMag;
        private int[] pixelState;
        private List<int> groupCount;

        private byte[] colorRGBData;

        // *************
        // SNS - 04-14
        // *************

        // Plane Tracker to track planes from frame to frame
        static PlaneTracker planetracker = null;

        // read in an overlay image
        private const int MAX_IMAGES = 5;

        Image<Bgr, Byte>[] overlayImageArray = new Image<Bgr, Byte>[2];

        static Image<Bgr, Byte> overlayImage = new Image<Bgr, Byte>(@"C:\Users\wayne\Desktop\ComputerVision\Shapes.jpg");
        static Image<Bgr, Byte> overlayImage2 = new Image<Bgr, Byte>(@"C:\Users\wayne\Desktop\ComputerVision\Shapes1.jpg");

        // make the overlay's "poly" which is just the corners of the image
        System.Drawing.PointF[] overlayPoly =
                    { new System.Drawing.Point(0, 0),
                        new System.Drawing.Point(overlayImage.Size.Width, 0),
                        new System.Drawing.Point(overlayImage.Size.Width, overlayImage.Size.Height),
                        new System.Drawing.Point(0, overlayImage.Size.Height) };
        // *************
        // SNS - 04-14
        // *************

        public MainWindow()
        {
            InitializeComponent();

            Loaded += OnLoaded;
        }

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

            this.colorImageBitmap = new WriteableBitmap(
                sensorChooser.Kinect.ColorStream.FrameWidth,
                sensorChooser.Kinect.ColorStream.FrameHeight,
                96,
                96,
                PixelFormats.Bgr32,
                null);
            kinectColorImage.Source = colorImageBitmap;

            // use a ring buffer to keep track of the last N buffers
            depthRingBuffer = new short[frameSize * ringBufferSize];

            // this is the current running sum for the depth ring
            depthRunningSum = new long[frameSize];

            // assign the depth bitmap to the image source
            kinectDepthImage.Source = depthBitmap;

            // Allocate space to put the color pixels we'll create
            depthRGBPixels = new byte[frameSize * sizeof(int)];
            colorRGBData = new byte[sensorChooser.Kinect.ColorStream.FrameWidth * sensorChooser.Kinect.ColorStream.FrameHeight * sizeof(int)];

            // allocate space for the opencv image
            depthImgCV = new Image<Bgr, Single>(sensorChooser.Kinect.DepthStream.FrameWidth, sensorChooser.Kinect.DepthStream.FrameHeight);
            dxImgCV = new Image<Bgr, Single>(sensorChooser.Kinect.DepthStream.FrameWidth, sensorChooser.Kinect.DepthStream.FrameHeight);
            dyImgCV = new Image<Bgr, Single>(sensorChooser.Kinect.DepthStream.FrameWidth, sensorChooser.Kinect.DepthStream.FrameHeight);
            depthImgOutCV = new Image<Bgr, Single>(sensorChooser.Kinect.DepthStream.FrameWidth, sensorChooser.Kinect.DepthStream.FrameHeight);
            CVKinectColorFrame = new Image<Bgr, Byte>(sensorChooser.Kinect.ColorStream.FrameWidth, sensorChooser.Kinect.ColorStream.FrameHeight);

            pixelMag = new float[sensorChooser.Kinect.DepthStream.FrameWidth * sensorChooser.Kinect.DepthStream.FrameHeight];
            pixelState = new int[sensorChooser.Kinect.DepthStream.FrameWidth * sensorChooser.Kinect.DepthStream.FrameHeight];
            
            // initialize array of overlay images
            overlayImageArray[0] = overlayImage;
            overlayImageArray[1] = overlayImage2;

            // initialize PlaneTracker object and list of planes to track
            List<Plane> planeList = new List<Plane>();
            planetracker = new PlaneTracker(planeList, MAX_IMAGES);

        }

        private void OnLoaded(object sender, RoutedEventArgs e)
        {
            //Detects if a Kinect or Kinect like device is connected and sets up the environment based on the sensors optimal settings.
            this.sensorChooser = new KinectSensorChooser();
            this.sensorChooser.KinectChanged += SensorChooserOnKinectChanged;
            this.sensorChooserUI.KinectSensorChooser = this.sensorChooser;
            this.sensorChooser.Start();

            InitData();

            //Eventhandlers for the depth, skeletal, and colorstreams. Skeleton is commented out as it isn't needed as of yet.
            this.sensorChooser.Kinect.ColorFrameReady += Kinect_ColorFrameReady;
            this.sensorChooser.Kinect.DepthFrameReady += Kinect_DepthFrameReady;
            //this.sensorChooser.Kinect.SkeletonFrameReady += Kinect_SkeletonFrameReady;
        }

        private void Kinect_ColorFrameReady(object sender, ColorImageFrameReadyEventArgs e)
        {
            using (ColorImageFrame colorFrame = e.OpenColorImageFrame())
            {
                if (colorFrame != null)
                {
                    // determine whether to dispatch a thread with this data
                    lock (clrLock)
                    {
                        // if the thread is not active
                        if (!clrActive)
                        {
                            clrActive = true;

                            // Copy the pixel data from the image to a temporary array
                            colorFrame.CopyPixelDataTo(colorRGBData);

                            Thread t = new Thread(() => ProcessColorFrame(colorFrame.BytesPerPixel));
                            t.Start();
                        }
                    }
                }
            }
        }

        private void Kinect_DepthFrameReady(object sender, DepthImageFrameReadyEventArgs e)
        {
            using (DepthImageFrame depthFrame = e.OpenDepthImageFrame())
            {
                if (depthFrame != null)
                {
                    // determine whether to dispatch a thread with this data
                    lock (dpLock)
                    {
                        // if the thread is not active
                        if (!dpActive)
                        {
                            dpActive = true;
                            // Copy the pixel data from the image to a temporary array
                            depthFrame.CopyDepthImagePixelDataTo(depthPixels);

                            Thread t = new Thread(() => ProcessDepthData(depthFrame.MinDepth, depthFrame.MaxDepth));
                            t.Start();
                        }
                    }
                }
            }
        }

        void ProcessColorFrame(int bytesPerPixel)
        {
            System.Diagnostics.Debug.WriteLine("color frame handler");

         
            // Convert from pixel space to real world space
            Parallel.For(0, sensorChooser.Kinect.ColorStream.FrameWidth * sensorChooser.Kinect.ColorStream.FrameHeight, i =>
            {
                int x = i / sensorChooser.Kinect.ColorStream.FrameWidth;
                int y = (sensorChooser.Kinect.ColorStream.FrameWidth - 1) - i % sensorChooser.Kinect.ColorStream.FrameWidth;

                // Write out blue (x-distance) byte
                CVKinectColorFrame.Data[x, y, 0] = colorRGBData[i * 4];

                // Write out green (y-distance) byte
                CVKinectColorFrame.Data[x, y, 1] = colorRGBData[i * 4 + 1];

                // Write out red (z-distance) byte
                CVKinectColorFrame.Data[x, y, 2] = colorRGBData[i * 4 + 2];
            });

            //############################################################################################

            //The following code demonstrates the conversion of the Kinect Writeable Bitmap to
            //the Image<,> format handled by the EMGU OpenCV wrapper.

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

            //Dectect polyLines that form quadralaterals.
            List<System.Drawing.Point[]> myPoly = polyDetect(grayTempCV);

            if (myPoly != null)
            {
                foreach (System.Drawing.Point[] polyLine in myPoly)
                    CVKinectColorFrame.DrawPolyline(polyLine, true, new Bgr(System.Drawing.Color.Red), 2);
            }

            //##################################################################################################

            // **************************************
            // SNS - 04-14
            // **************************************

            // After we find the polylines' vertices we can do the transformation
            // need to have this in PointF versus Point. Blahblahblah
            // System.Diagnostics.Debug.WriteLine("stephanies code");

            System.Diagnostics.Debug.WriteLine("update planes");

            planetracker.UpdatePlanes(myPoly);

            // create the overlay using the planes in plane tracker
            foreach (Plane plane in planetracker.GetPlanes())
            {
                createOverlay(plane.GetPoints(), overlayImageArray[plane.GetOverlayImageIndex() % 2]);
            }
                           
            //Once we are finished with the gray temp image it needs to be disposed of.
            grayTempCV.Dispose();

            // Convert from pixel space to real world space
            Parallel.For(0, sensorChooser.Kinect.ColorStream.FrameWidth * sensorChooser.Kinect.ColorStream.FrameHeight, i =>
            {
                int x = i / sensorChooser.Kinect.ColorStream.FrameWidth;
                int y = i % sensorChooser.Kinect.ColorStream.FrameWidth;

                // Write out blue (x-distance) byte
                colorRGBData[i * 4] = CVKinectColorFrame.Data[x, y, 0];

                // Write out green (y-distance) byte
                colorRGBData[i * 4 + 1] = CVKinectColorFrame.Data[x, y, 1];

                // Write out red (z-distance) byte
                colorRGBData[i * 4 + 2] = CVKinectColorFrame.Data[x, y, 2];
            });

            try
            {
                Dispatcher.Invoke((Action)(() =>
                {
                    //Write our converted color pixel data to the original Writeable bitmap.
                    this.colorImageBitmap.WritePixels(
                        new Int32Rect(0, 0, sensorChooser.Kinect.ColorStream.FrameWidth, sensorChooser.Kinect.ColorStream.FrameHeight),
                        colorRGBData,
                        sensorChooser.Kinect.ColorStream.FrameWidth * bytesPerPixel, 0
                        );
                }));
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }

            lock (clrLock)
            {
                clrActive = false;
            }
        }

        private void ProcessDepthData(int minDepth, int maxDepth)
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

                // *******************************************
                // Start Plane Detection
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
                });

                if (useFloodFill)
                {

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
                else if (useEdgeDetect)
                {

                    Parallel.For(0, depthImgCV.Cols * depthImgCV.Rows, i =>
                    {
                        int x = i / depthImgCV.Cols;
                        int y = i % depthImgCV.Cols;

                        if (useCustom)
                        {
                            if (pixelMag[i] > 1000)
                            {
                                depthImgOutCV.Data[x, y, 0] = 255;
                                depthImgOutCV.Data[x, y, 1] = 255;
                                depthImgOutCV.Data[x, y, 2] = 255;
                            }
                            else
                            {
                                depthImgOutCV.Data[x, y, 0] = 0;
                                depthImgOutCV.Data[x, y, 1] = 0;
                                depthImgOutCV.Data[x, y, 2] = 0;
                            }
                        }
                        else
                        {
                            depthImgOutCV.Data[x, y, 0] = (float)((depthImgOutCV.Data[x, y, 0] / pixelMag[i] + 1) * 255.0 / 2.0);
                            depthImgOutCV.Data[x, y, 1] = (float)((depthImgOutCV.Data[x, y, 2] / pixelMag[i] + 1) * 255.0 / 2.0);
                            depthImgOutCV.Data[x, y, 2] = (float)((depthImgOutCV.Data[x, y, 1] / pixelMag[i] + 1) * 255.0 / 2.0);
                        }

                    });

                    //Run canny detection to obtain image edges.
                    Image<Gray, Byte> grayTempCV;

                    if (useCustom)
                        grayTempCV = depthImgOutCV.Convert<Gray, Byte>();
                    else
                        grayTempCV = depthImgOutCV.Convert<Gray, Byte>().Canny(170, 75).Dilate(1);

                    List<System.Drawing.Point[]> myPoly = polyDetect(grayTempCV);

                    if (myPoly != null)
                    {
                        foreach (System.Drawing.Point[] polyLine in myPoly)
                            depthImgOutCV.DrawPolyline(polyLine, true, new Bgr(System.Drawing.Color.Red), 2);
                    }

                    // save the points to the output buffer
                    Parallel.For(0, depthImgCV.Cols * depthImgCV.Rows, i =>
                    {
                        int x = i / sensorChooser.Kinect.DepthStream.FrameWidth;
                        int y = i % sensorChooser.Kinect.DepthStream.FrameWidth;

                        // Write out blue byte
                        this.depthRGBPixels[i * 4] = (byte)(short)grayTempCV.Data[x, y, 0];

                        // Write out green byte
                        this.depthRGBPixels[i * 4 + 1] = (byte)(short)grayTempCV.Data[x, y, 0];

                        // Write out red byte
                        this.depthRGBPixels[i * 4 + 2] = (byte)(short)grayTempCV.Data[x, y, 0];
                    });
                }
                else
                {
                    Parallel.For(0, depthImgOutCV.Cols * depthImgOutCV.Rows, i =>
                    {
                        int x = i / sensorChooser.Kinect.DepthStream.FrameWidth;
                        int y = i % sensorChooser.Kinect.DepthStream.FrameWidth;

                        depthImgOutCV.Data[x, y, 0] = (float)((depthImgOutCV.Data[x, y, 0] / pixelMag[i] + 1) * 255.0 / 2.0);
                        depthImgOutCV.Data[x, y, 1] = (float)((depthImgOutCV.Data[x, y, 1] / pixelMag[i] + 1) * 255.0 / 2.0);
                        depthImgOutCV.Data[x, y, 2] = (float)((depthImgOutCV.Data[x, y, 2] / pixelMag[i] + 1) * 255.0 / 2.0);
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
                dpActive = false;
            }
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
                                if (dotProd > minDotProd)
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
                                        if (dotProd > minDotProd)
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
                                        if (dotProd > minDotProd)
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
                                if (dotProd > minDotProd)
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
                                        if (dotProd > minDotProd)
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
                                        if (dotProd > minDotProd)
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

        public LineSegment2D[] lineDetect(Image<Gray, Byte> myImage)
        {
            LineSegment2D[] myLines = myImage.HoughLinesBinary(1, (Math.PI / 180), 50, 80, 10)[0];
            return myLines;

        }

        public List<System.Drawing.Point[]> polyDetect(Image<Gray, Byte> myImage)
        {

            List<System.Drawing.Point[]> mypts = new List<System.Drawing.Point[]>();

            // *************
            // SNS - 04-14
            // *************
            List<System.Drawing.Point[]> snsPts = new List<System.Drawing.Point[]>();
            // *************
            // SNS - 04-14
            // *************

            using (MemStorage storage = new MemStorage())

                for (Contour<System.Drawing.Point> contours = myImage.FindContours(CHAIN_APPROX_METHOD.CV_CHAIN_APPROX_SIMPLE, RETR_TYPE.CV_RETR_EXTERNAL); contours != null; contours = contours.HNext)
                {
                    Contour<System.Drawing.Point> currentContour = contours.ApproxPoly(contours.Perimeter * 0.1, storage);

                    //Check to see if the contour forms a enclosed quadralateral with a desired minimum area.
                    if (Math.Abs(currentContour.Area) > 500 && currentContour.Convex == true && currentContour.Total == 4)//
                    {
                        System.Diagnostics.Debug.WriteLine("current contour area " + currentContour.Area);
                      
                        System.Drawing.Point[] pts = currentContour.ToArray();

                        System.Diagnostics.Debug.WriteLine(pts[0] + " " + pts[1] + " " + pts[2] + " " + pts[3]);

                        if (!isSelfIntersecting(pts)) {
                            mypts.Add(pts);
                        }
                    }
                }

            // *************
            // SNS - 04-14
            // *************
            snsPts = OrderPoints(mypts);
            return snsPts;

            //return mypts;
            // *************
            // SNS - 04-14
            // *************
        }

        // *******************
        // SNS Added 04-24-14
        // *******************
        public void createOverlay(System.Drawing.Point[] myPoly, Image<Bgr, Byte> overlayImage)
        {
            System.Drawing.Point[] singlePoly = myPoly;
            System.Drawing.PointF[] singlePolyF = Array.ConvertAll(singlePoly, item => (System.Drawing.PointF)item);

            // Compute the transform matrix
            // GetPerspectiveTransform wants PointF[] arrays
            HomographyMatrix matrixM = CameraCalibration.GetPerspectiveTransform(overlayPoly, singlePolyF);

            // then we need to overlay the transformation onto the original image
            Image<Bgr, Byte> whiteOverlay = new Image<Bgr, Byte>(overlayImage.Size.Width, overlayImage.Size.Height, new Bgr(255, 255, 255));
            Image<Bgr, Byte> mask = new Image<Bgr, Byte>(CVKinectColorFrame.Size.Width, CVKinectColorFrame.Size.Height, new Bgr(0, 0, 0));

            // apply perspective transform to the white image to make a mask
            mask = whiteOverlay.WarpPerspective(matrixM, CVKinectColorFrame.Size.Width, CVKinectColorFrame.Size.Height, Emgu.CV.CvEnum.INTER.CV_INTER_NN, Emgu.CV.CvEnum.WARP.CV_WARP_DEFAULT, new Bgr(0, 0, 0));
     
            // apply warpPerspective to the image we want to warp
            Image<Bgr, Byte> correctedOverlay = overlayImage.WarpPerspective(matrixM, Emgu.CV.CvEnum.INTER.CV_INTER_NN, Emgu.CV.CvEnum.WARP.CV_WARP_DEFAULT, new Bgr(0, 0, 0));
     
            // copy the correctd overlay onto the kinect image using the mask
            correctedOverlay.Copy(CVKinectColorFrame, mask.Convert<Gray, Byte>());

            CVKinectColorFrame.DrawPolyline(myPoly, true, new Bgr(System.Drawing.Color.Red), 2);

            mask.Dispose();
            whiteOverlay.Dispose();
            // correctedOverlay.Dispose();
            // matrixM.Dispose();

            // *************
            // SNS - 04-14
            // *************
        }

        /// <summary>
        /// Returns the list sorted so that the leftmost point is first.
        /// </summary>
        /// <param name="pointsList"></param>
        /// <returns>list of poly points where they start with leftmost point</returns>
        public List<System.Drawing.Point[]> OrderPoints(List<System.Drawing.Point[]> pointsList)
        {

            List<System.Drawing.Point[]> sortedList = new List<System.Drawing.Point[]>();

            foreach (System.Drawing.Point[] points in pointsList)
            {

                var sorted = points.OrderBy(point => point.X).ThenBy(point => point.Y);
                System.Drawing.Point[] sortedPoints = new System.Drawing.Point[4];

                sortedPoints[0] = sorted.ElementAt(0);

                System.Drawing.Point firstPoint = sortedPoints[0];

                // get the index of the starting point in the original list.
                int firstIndex = Array.FindIndex(points, item => item == sortedPoints[0]);

                // now find the other items to add to sortedList
                sortedPoints[1] = points[(firstIndex + 3) % 4];
                sortedPoints[2] = points[(firstIndex + 2) % 4];
                sortedPoints[3] = points[(firstIndex + 1) % 4];

                sortedList.Add(sortedPoints);
            }
            return sortedList;
        }

        public Boolean isSelfIntersecting(System.Drawing.Point[] polygon)
        {
            // return true if actual convex poly
            // note that the OpenCV convex function is undefined for self-intersecting vertices

            // don't care about line segments that share an endpoint (they are going to touch and can't intersect)
            // check if segment 1-2 and segment 3-4 intersect
            double slope12;
            double slope34;
            double b12;
            double b34;
            double x;

            
            // case where line 1 is vertical
            if (polygon[1].X - polygon[0].X == 0)
            {
                // equation is x = X
                x = polygon[1].X;

                // case where line 2 is vertical
                if (polygon[3].X - polygon[2].X == 0)
                {
                    // can't intersect
                    return false;
                }
                else
                {
                    // if this x falls in the range of the other line, return true
                    if ((x > polygon[2].X && x < polygon[3].X) || (x < polygon[2].X && x > polygon[3].X))
                    {
                        return true;
                    }

                }
            }
            else if (polygon[3].X - polygon[2].X == 0)
            {
                // equation is x = X
                x = polygon[3].X;

                // if the first line falls in the range of this one return true
                if ((x > polygon[0].X && x < polygon[1].X) || (x < polygon[0].X && x > polygon[1].X))
                {
                    return true;
                }
            }
            else
            {
                slope12 = (polygon[1].Y - polygon[0].Y) / (polygon[1].X - polygon[0].X);
                slope34 = (polygon[3].Y - polygon[2].Y) / (polygon[3].X - polygon[2].X);

                b12 = polygon[0].Y - slope12 * polygon[0].X;
                b34 = polygon[2].Y - slope34 * polygon[2].X;

                // if they intersect, find the x coord
                if (slope12 != slope34)
                {
                    x = (b34 - b12) / (slope12 - slope34);

                    // check if x coord in range of both lines
                    if ((x > polygon[0].X && x < polygon[1].X) || (x < polygon[0].X && x > polygon[1].X))
                    {
                        return true;
                    }
                }
            }
            return false;
        }


        //Kinect sensor chooser. Honestly its really only impacts the skeleton tracking settings for the most part, as
        //the depth and color frame data and settings are handled same for Xbox and Windows based sensors.
        //Only the Xbox sensor and 3rd party sensors can not do skeletal traking in near mode.

        private void SensorChooserOnKinectChanged(object sender, KinectChangedEventArgs args)
        {
            bool error = false;
            if (args.OldSensor != null)
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