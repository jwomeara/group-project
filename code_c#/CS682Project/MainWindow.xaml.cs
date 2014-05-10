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
        private const bool useFloodFill = true;
        private const bool usedgeDetect = false;
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

        byte[] colorData;


// *************
// SNS - 04-14
// *************

        // Plane Tracker to track planes from frame to frame
        static PlaneTracker planetracker = null;

        // read in an overlay image
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
            colorData = new byte[frameSize];

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
            dxImgCV = new Image<Bgr, Single>(sensorChooser.Kinect.DepthStream.FrameWidth, sensorChooser.Kinect.DepthStream.FrameHeight);
            dyImgCV = new Image<Bgr, Single>(sensorChooser.Kinect.DepthStream.FrameWidth, sensorChooser.Kinect.DepthStream.FrameHeight);
            depthImgOutCV = new Image<Bgr, Single>(sensorChooser.Kinect.DepthStream.FrameWidth, sensorChooser.Kinect.DepthStream.FrameHeight);

            pixelMag = new float[sensorChooser.Kinect.DepthStream.FrameWidth * sensorChooser.Kinect.DepthStream.FrameHeight];
            pixelState = new int[sensorChooser.Kinect.DepthStream.FrameWidth * sensorChooser.Kinect.DepthStream.FrameHeight];
        }

        private void OnLoaded(object sender, RoutedEventArgs e)
        {
            //Detects if a Kinect or Kinect like device is connected and sets up the environment based on the sensors optimal settings.
            this.sensorChooser = new KinectSensorChooser();
            this.sensorChooser.KinectChanged += SensorChooserOnKinectChanged;
            this.sensorChooserUI.KinectSensorChooser = this.sensorChooser;
            this.sensorChooser.Start();

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
                            colorFrame.CopyPixelDataTo(colorData);

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
                new Int32Rect(0, 0, sensorChooser.Kinect.ColorStream.FrameWidth, sensorChooser.Kinect.ColorStream.FrameHeight),
                colorData,
                sensorChooser.Kinect.ColorStream.FrameWidth * bytesPerPixel, 0
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
//               System.Diagnostics.Debug.WriteLine("stephanies code");

            if (myPoly != null && myPoly.Count > 0)
            {
                if (planetracker == null && myPoly.Count >=2)
                {
                    Plane plane0 = new Plane();
                    Plane plane1 = new Plane();
                    plane0.SetPoints(myPoly[0]);
                    plane0.SetOverlayImage(overlayImage);
                    plane1.SetPoints(myPoly[1]);
                    plane1.SetOverlayImage(overlayImage2);

                    // first frame where we found polygons. initialize the plane tracker
                    planetracker = new PlaneTracker(new List<Plane> {plane0, plane1});
                }
                
                if (planetracker != null)
                {
                    // not the first frame. try to update the values in the plane tracker
                    System.Diagnostics.Debug.WriteLine("update planes");

                    planetracker.UpdatePlanes(myPoly);

                    // create the overlay using the planes in plane tracker
                    foreach (Plane plane in planetracker.GetPlanes())
                    {
                        createOverlay(plane.GetPoints(), plane.GetOverlayImage());
                    }
                }
            }

            //Once we are finished with the gray temp image it needs to be disposed of. 
            grayTempCV.Dispose();


            //Following processing of CV image need to convert back to Windows style bitmap.
            BitmapSource bs = BitmapSourceConverter.ToBitmapSource(CVKinectColorFrame);
            System.Diagnostics.Debug.WriteLine("conversion done");
    


            //Dispose of the CV color image following the conversion.
            CVKinectColorFrame.Dispose();

            int stride = bs.PixelWidth * (bs.Format.BitsPerPixel / 8);
            byte[] data = new byte[stride * bs.PixelHeight];
            bs.CopyPixels(data, stride, 0);
            colorData = data;

            //Write our converted color pixel data to the original Writeable bitmap.
            this.colorImageBitmap.WritePixels(
                new Int32Rect(0, 0, sensorChooser.Kinect.ColorStream.FrameWidth, sensorChooser.Kinect.ColorStream.FrameHeight),
                colorData,
                sensorChooser.Kinect.ColorStream.FrameWidth * bytesPerPixel, 0
                );

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
                else if (useEdgeDetect){

                    Parallel.For(0, depthImgCV.Cols * depthImgCV.Rows, i =>
                    {
                        int x = i / depthImgCV.Cols;
                        int y = i % depthImgCV.Cols;

                        if (pixelMag[i] > 1000)
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
                    Image<Gray, Byte> grayTempCV = depthImgOutCV.Convert<Gray, Byte>();//.Canny(900, 150);

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
                        this.depthRGBPixels[i * 4] = (byte)(short)depthImgOutCV.Data[x, y, 0];

                        // Write out green byte
                        this.depthRGBPixels[i * 4 + 1] = (byte)(short)depthImgOutCV.Data[x, y, 1];

                        // Write out red byte                        
                        this.depthRGBPixels[i * 4 + 2] = (byte)(short)depthImgOutCV.Data[x, y, 2];
                    });
                }
                else
                {
                    Parallel.For(0, depthImgOutCV.Cols * depthImgOutCV.Rows, i =>
                    {
                        int x = i / sensorChooser.Kinect.DepthStream.FrameWidth;
                        int y = i % sensorChooser.Kinect.DepthStream.FrameWidth;

                        depthImgOutCV.Data[x, y, 0] = (float)((depthImgOutCV.Data[x, y, 0] / pixelMag[i] + 1)*255.0/2.0);
                        depthImgOutCV.Data[x, y, 1] = (float)((depthImgOutCV.Data[x, y, 1] / pixelMag[i] + 1)*255.0/2.0);
                        depthImgOutCV.Data[x, y, 2] = (float)((depthImgOutCV.Data[x, y, 2] / pixelMag[i] + 1)*255.0/2.0);
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

// *************
// SNS - 04-14
// *************
            List<System.Drawing.Point[]> snsPts = new List<System.Drawing.Point[]>();
// *************
// SNS - 04-14
// *************

            using (MemStorage storage = new MemStorage())
                
                for (Contour<System.Drawing.Point> contours = myImage.FindContours(CHAIN_APPROX_METHOD.CV_CHAIN_APPROX_SIMPLE,RETR_TYPE.CV_RETR_EXTERNAL); contours != null; contours = contours.HNext)
                {
                    Contour<System.Drawing.Point> currentContour = contours.ApproxPoly(contours.Perimeter * 0.1, storage);

                    // SNS - change this back to 150 when done

                    //Check to see if the contour forms a enclosed quadralateral with a desired minimum area.
                    if (Math.Abs(currentContour.Area) > 150 && currentContour.Convex == true && currentContour.Total == 4)//
                    {
                        bool isRectangle = true;
                        System.Drawing.Point[] pts = currentContour.ToArray();
                        mypts.Add(pts);
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
        public void createOverlay(System.Drawing.Point[] myPoly, Image<Bgr, Byte> overlayImage) {
                     
            System.Drawing.Point[] singlePoly = myPoly;
            System.Drawing.PointF[] singlePolyF = Array.ConvertAll(singlePoly, item => (System.Drawing.PointF)item);

//          System.Diagnostics.Debug.WriteLine(singlePolyF[0] + " " + singlePolyF[1] + " " + singlePolyF[2] + " " + singlePolyF[3]);

            // Compute the transform matrix
            // GetPerspectiveTransform wants PointF[] arrays
            HomographyMatrix matrixM = CameraCalibration.GetPerspectiveTransform(overlayPoly, singlePolyF);
                        
            // then we need to overlay the transformation onto the original image
            Image<Bgr, Byte> whiteOverlay = new Image<Bgr, Byte>(overlayImage.Size.Width, overlayImage.Size.Height, new Bgr(255, 255, 255));
            Image<Bgr, Byte> mask = new Image<Bgr, Byte>(CVKinectColorFrame.Size.Width, CVKinectColorFrame.Size.Height, new Bgr(0,0,0));
                       
            // apply perspective transform to the white image to make a mask
            mask = whiteOverlay.WarpPerspective(matrixM, CVKinectColorFrame.Size.Width, CVKinectColorFrame.Size.Height, Emgu.CV.CvEnum.INTER.CV_INTER_NN, Emgu.CV.CvEnum.WARP.CV_WARP_DEFAULT, new Bgr(0, 0, 0));
//            System.Diagnostics.Debug.WriteLine("warp perspective done");

            // mask.Save(@"C:\Users\wayne\Desktop\Mask.jpg");

            // apply warpPerspective to the image we want to warp
            Image<Bgr, Byte> correctedOverlay = overlayImage.WarpPerspective(matrixM, Emgu.CV.CvEnum.INTER.CV_INTER_NN, Emgu.CV.CvEnum.WARP.CV_WARP_DEFAULT, new Bgr(0,0,0));
 //           System.Diagnostics.Debug.WriteLine("warp perspective 2 done");
            // correctedOverlay.Save(@"C:\Users\wayne\Desktop\Corrected.jpg");

            // copy the correctd overlay onto the kinect image using the mask
            correctedOverlay.Copy(CVKinectColorFrame, mask.Convert<Gray, Byte>());
            // CVKinectColorFrame.Save(@"C:\Users\wayne\Desktop\frame.jpg");

            mask.Dispose();
            whiteOverlay.Dispose();
//          correctedOverlay.Dispose();
//          matrixM.Dispose();

// *************
// SNS - 04-14
// *************
        }

        /// <summary>
        /// Returns the list sorted so that the leftmost point is first.
        /// </summary>
        /// <param name="pointsList"></param>
        /// <returns>list of poly points where they start with leftmost point</returns>
        public List<System.Drawing.Point[]> OrderPoints(List<System.Drawing.Point[]> pointsList) {
    
            List<System.Drawing.Point[]> sortedList = new List<System.Drawing.Point[]>();

            foreach (System.Drawing.Point[] points in pointsList) { 

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



        //Not sure if we'll need the skeleton tracking but I included the basic code should you want to play around with it.
        
        //void Kinect_SkeletonFrameReady(object sender, SkeletonFrameReadyEventArgs e)
        //{
            ////string message = "No Skeleton Data";

            //Skeleton[] skeletons = null;
            //myBones = new Line[20];

            //using (SkeletonFrame frame = e.OpenSkeletonFrame())
            //{
            //    if (frame != null)
            //    {
            //        skeletons = new Skeleton[frame.SkeletonArrayLength];
            //        frame.CopySkeletonDataTo(skeletons);
            //    }
            //}

            //if (skeletons == null) return;

            ////canvas.Children.Clear();

            //foreach (Skeleton skeleton in skeletons)
            //{
            //    if (skeleton.TrackingState == SkeletonTrackingState.Tracked)
            //    {

            //        myJoints[0] = findJoint(skeleton.Joints[JointType.HandRight]);
            //        myJoints[1] = findJoint(skeleton.Joints[JointType.HandLeft]);
            //    }
            //}
                   
        //}

        //This function is used for drawing the individual bonelines.
        //Once the line has been returned you just need to add it to the desired canvas.
        //public Line addLine(Joint j1, Joint j2)
        //{
        //    Line boneLine = new Line();
        //    boneLine.Stroke = skeletonBrush;
        //    boneLine.StrokeThickness = 5;

        //    ColorImagePoint j1P = this.sensorChooser.Kinect.CoordinateMapper.MapSkeletonPointToColorPoint(j1.Position, ColorImageFormat.RgbResolution640x480Fps30);


        //    boneLine.X1 = j1P.X / 2;
        //    boneLine.Y1 = j1P.Y / 2;

        //    ColorImagePoint j2P = this.sensorChooser.Kinect.CoordinateMapper.MapSkeletonPointToColorPoint(j2.Position, ColorImageFormat.RgbResolution640x480Fps30);

        //    boneLine.X2 = j2P.X / 2;
        //    boneLine.Y2 = j2P.Y / 2;

        //    return boneLine;
        //}

        
        //The following is manages the depth stream. This is slightly more complicated than the normal implementation as I've
        //included a hole filling, smoothging, and averaging algorithm.
        void Kinect_DepthFrameReady(object sender, DepthImageFrameReadyEventArgs e)
        {
            using (DepthImageFrame depthFrame = e.OpenDepthImageFrame())
            {

                if (depthFrame == null) return;

                if (depthData == null) depthData = new short[depthFrame.PixelDataLength];

                if (depthColorImage == null) depthColorImage = new byte[depthFrame.PixelDataLength * 4];

                depthFrame.CopyPixelDataTo(depthData);

                if (depthArray == null) depthArray = new short[depthData.Length];

                if (smoothDepthArray == null) smoothDepthArray = new short[depthData.Length];

                depthArray = depthData;


                //The following code looks for holes in the depth image and fills them using the mode of surronding pixels.
                Parallel.For(0, 480, depthArrayRowIndex =>
                {
                    // Process each pixel in the row
                    for (int depthArrayColumnIndex = 0; depthArrayColumnIndex < 640; depthArrayColumnIndex++)
                    {

                        var depthIndex = depthArrayColumnIndex + (depthArrayRowIndex * 640);

                        // We are only concerned with eliminating 'white' noise from the data.
                        // We consider any pixel with a depth of 0 as a possible candidate for filtering.
                        if (depthArray[depthIndex] == 0)
                        {
                            // From the depth index, we can determine the X and Y coordinates that the index
                            // will appear in the image. We use this to help us define our filter matrix.
                            int x = depthIndex % 640;
                            int y = (depthIndex - x) / 640;

                            // The filter collection is used to count the frequency of each
                            // depth value in the filter array. This is used later to determine
                            // the statistical mode for possible assignment to the candidate.

                            short[] myfilter = new short[(kernelSize ^ 2) - 1];

                            // The inner and outer band counts are used later to compare against the threshold 
                            // values set in the UI to identify a positive filter result.
                            int innerBandCount = 0;
                            int outerBandCount = 0;

                            // The following loops will loop through a 5 X 5 matrix of pixels surrounding the 
                            // candidate pixel. This defines 2 distinct 'bands' around the candidate pixel.
                            // If any of the pixels in this matrix are non-0, we will accumulate them and count
                            // how many non-0 pixels are in each band. If the number of non-0 pixels breaks the
                            // threshold in either band, then the average of all non-0 pixels in the matrix is applied
                            // to the candidate pixel.
                            for (int yi = -(kernelSize / 2); yi < (kernelSize / 2) + 1; yi++)
                            {
                                for (int xi = -(kernelSize / 2); xi < (kernelSize / 2) + 1; xi++)
                                {
                                    // yi and xi are modifiers that will be subtracted from and added to the
                                    // candidate pixel's x and y coordinates that we calculated earlier. From the
                                    // resulting coordinates, we can calculate the index to be addressed for processing.

                                    // We do not want to consider the candidate
                                    // pixel (xi = 0, yi = 0) in our process at this point.
                                    // We already know that it's 0
                                    if (xi != 0 || yi != 0)
                                    {
                                        // We then create our modified coordinates for each pass
                                        var xSearch = x + xi;
                                        var ySearch = y + yi;

                                        // While the modified coordinates may in fact calculate out to an actual index, it 
                                        // might not be the one we want. Be sure to check
                                        // to make sure that the modified coordinates
                                        // match up with our image bounds.
                                        if (xSearch >= 0 && xSearch <= widthBound && ySearch >= 0 && ySearch <= heightBound)
                                        {
                                            var index = xSearch + (ySearch * width);
                                            // We only want to look for non-0 values
                                            if (depthArray[index] != 0)
                                            {

                                                // We want to find count the frequency of each depth
                                                for (int i = 0; i < (kernelSize ^ 2) - 1; i++)
                                                {
                                                    myfilter[i] = depthArray[index];
                                                }

                                                // We will then determine which band the non-0 pixel
                                                // was found in, and increment the band counters.
                                                if (yi != (kernelSize / 2) && yi != -(kernelSize / 2) && xi != (kernelSize / 2) && xi != -(kernelSize / 2))
                                                    innerBandCount++;
                                                else
                                                    outerBandCount++;
                                            }
                                        }
                                    }
                                }
                            }



                            // Once we have determined our inner and outer band non-zero counts, and 
                            // accumulated all of those values, we can compare it against the threshold
                            // to determine if our candidate pixel will be changed to the
                            // statistical mode of the non-zero surrounding pixels.
                            if (innerBandCount >= innerBandThreshold || outerBandCount >= outerBandThreshold)
                            {
                                short mode = myfilter.GroupBy(v => v).OrderByDescending(g => g.Count()).First(g => g.Key != 0).Key;
                                //short mode = myfilter.GroupBy(v => v).OrderByDescending(g => g.Count()).First().Key;
                                smoothDepthArray[depthIndex] = mode;
                            }
                        }
                        else
                        {
                            // If the pixel is not zero, we will keep the original depth.
                            smoothDepthArray[depthIndex] = depthArray[depthIndex];
                        }
                    }
                });


                //The following code creates a queue of depth frames which will be used to create a moving average of the depth.
                //This provides a more consistent value of depth as the IR tends to be noisy and fluctuate over time.

                myDepthQueue.Enqueue(smoothDepthArray);
                //myDepthQueue.Enqueue(depthData);

                //Currently set to hold five frames. Any more and things slow down.
                while (myDepthQueue.Count > 5) myDepthQueue.Dequeue();


                int[] sumDepthArray = new int[depthData.Length];

                short[] avg_depthData = new short[depthData.Length];

                int denom = 0;
                int count = 1;

                foreach (var item in myDepthQueue)
                {
                    Parallel.For(0, 480, depthRow =>
                    {
                        for (int depthCol = 0; depthCol < 640; depthCol++)
                        {
                            var index = depthCol + (depthRow * 640);

                            sumDepthArray[index] += item[index] * count;
                        }
                    });

                    denom += count;
                    count++;
                }


                Parallel.For(0, depthData.Length, i =>
                {
                    avg_depthData[i] = (short)(int)(sumDepthArray[i] / denom);
                });


                //Unlike colorFrame data which is already in a BGR format, the depth data contains additional bits of data used
                //to store player index information. This must be removed in order to display the depth data.
                Parallel.For(0, 480, depthRow =>
                {
                    for (int depthCol = 0; depthCol < 640; depthCol++)
                    {
                        var depthIndex = depthCol + (depthRow * 640);
                        var index = depthIndex * 4;

                        int depthValue = depthData[depthIndex] >> 3;

                        byte depthByte = (byte)(255 - ((int)depthValue >> 4));
                        depthColorImage[index] = depthByte; //Blue
                        depthColorImage[index + 1] = depthByte; //Green
                        depthColorImage[index + 2] = depthByte; //Red
                        depthColorImage[index + 3] = 0; //Alpha
                    }
                });

                //The rest is similar to the colorFrame data and is used to write the depth pixel data to the depth Writeable bitmap.
                if (depthImageBitmap == null)
                {
                    this.depthImageBitmap = new WriteableBitmap(
                        depthFrame.Width,
                        depthFrame.Height,
                        96,
                        96,
                        PixelFormats.Bgr32,
                        null);
                    kinectDepthImage.Source = depthImageBitmap;
                }


                this.depthImageBitmap.WritePixels(
                    new Int32Rect(0, 0, depthFrame.Width, depthFrame.Height),
                    depthColorImage,
                    depthFrame.Width * 4, 0
                    );
            }
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
