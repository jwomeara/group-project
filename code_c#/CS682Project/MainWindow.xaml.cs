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
        private const int ringBufferSize = 10;

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
        private Image<Bgr, Single> dxImgCV;
        private Image<Bgr, Single> dyImgCV;

        private void InitData()
        {
            // save the frame size
            this.frameSize = this.sensorChooser.Kinect.DepthStream.FramePixelDataLength;

            // Allocate space to put the depth pixels we'll receive
            this.depthPixels = new DepthImagePixel[frameSize];

            // This is the bitmap we'll display on-screen
            this.depthBitmap = new WriteableBitmap(
                                this.sensorChooser.Kinect.DepthStream.FrameWidth, 
                                this.sensorChooser.Kinect.DepthStream.FrameHeight, 
                                96.0, 
                                96.0, 
                                PixelFormats.Bgr32, 
                                null);

            // use a ring buffer to keep track of the last N buffers
            this.depthRingBuffer = new short[frameSize * ringBufferSize];

            // this is the current running sum for the depth ring
            this.depthRunningSum = new long[frameSize];

            // assign the depth bitmap to the image source
            kinectDepthImage.Source = depthBitmap;

            // Allocate space to put the color pixels we'll create
            this.depthRGBPixels = new byte[frameSize * sizeof(int)];

            // allocate space for the opencv image
            this.depthImgCV = new Image<Bgr, Single>(this.sensorChooser.Kinect.DepthStream.FrameWidth, this.sensorChooser.Kinect.DepthStream.FrameHeight);
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
                
                for (Contour<System.Drawing.Point> contours = myImage.FindContours(CHAIN_APPROX_METHOD.CV_CHAIN_APPROX_SIMPLE,RETR_TYPE.CV_RETR_EXTERNAL); contours != null; contours = contours.HNext)
                {
                    Contour<System.Drawing.Point> currentContour = contours.ApproxPoly(contours.Perimeter * 0.1, storage);

                    //Check to see if the contour forms a enclosed quadralateral with a desired minimum area.
                    if (Math.Abs(currentContour.Area) > 150 && currentContour.Convex == true && currentContour.Total == 4)//
                    {
                        bool isRectangle = true;
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

                            Thread t = new Thread(() => ProcessDepthData(depthFrame.MinDepth, depthFrame.MaxDepth));
                            t.Start();
                        }
                    }
                }
            }
        }

        private void ProcessDepthData(int minDepth, int maxDepth)
        {
            // Save this frame in our ring buffer
            for (int i = 0; i < this.depthPixels.Length; ++i)
            {
                // Get the depth for this pixel
                short depth = depthPixels[i].Depth;

                // if we have a full ring buffer, subtract the oldest entry from the running sum
                if (ready)
                    depthRunningSum[i] -= depthRingBuffer[ringBufIdx * frameSize + i];

                // save this pixel for this frame, and update the running sum
                depthRingBuffer[ringBufIdx * frameSize + i] = (depth >= minDepth && depth <= maxDepth) ? depth : (short)0;
                depthRunningSum[i] += depthRingBuffer[ringBufIdx * frameSize + i];
            }

            // increment the ring buffer index
            ringBufIdx = (ringBufIdx + 1) % ringBufferSize;

            // if we have a full ring buffer, enable image processing
            if (!ready && ringBufIdx == 0)
                ready = true;

            // perform image processing if we have a full circular buffer
            if (ready)
            {

                // Convert the depth to RGB
                for (int i = 0; i < this.depthRunningSum.Length; ++i)
                {
                    long longVal = depthRunningSum[i] / ringBufferSize * 256 / this.sensorChooser.Kinect.DepthStream.MaxDepth;
                    float floatVal = depthRunningSum[i] / ringBufferSize;

                    // Get the depth for this pixel
                    byte depth = (byte)longVal;

                    int x = i / this.sensorChooser.Kinect.DepthStream.FrameWidth;
                    int y = i % this.sensorChooser.Kinect.DepthStream.FrameWidth;

                    // Write out blue byte
                    this.depthRGBPixels[i * 4] = depth;
                    depthImgCV.Data[x, y, 0] = floatVal;

                    // Write out green byte
                    this.depthRGBPixels[i * 4 + 1] = depth;
                    depthImgCV.Data[x, y, 1] = floatVal;

                    // Write out red byte                        
                    this.depthRGBPixels[i * 4 + 2] = depth;
                    depthImgCV.Data[x, y, 2] = floatVal;
                }

                // OpenCV processing here!
                depthImgCV = depthImgCV.SmoothGaussian(5, 5, 34.3, 34.3);

                dxImgCV = depthImgCV.Sobel(1, 0, 3).Pow(2);
                dyImgCV = depthImgCV.Sobel(0, 1, 3).Pow(2);
                dxImgCV.Add(dyImgCV).Pow(0.5);

                // save the points to the output buffer
                for (int i = 0; i < dxImgCV.Cols*dxImgCV.Rows; ++i)
                {
                    int x = i / this.sensorChooser.Kinect.DepthStream.FrameWidth;
                    int y = i % this.sensorChooser.Kinect.DepthStream.FrameWidth;

                    // Write out blue byte
                    this.depthRGBPixels[i * 4] = (byte)(short)(dxImgCV.Data[x, y, 0]/10000);

                    // Write out green byte
                    this.depthRGBPixels[i * 4 + 1] = (byte)(short)(dxImgCV.Data[x, y, 1]/10000);

                    // Write out red byte                        
                    this.depthRGBPixels[i * 4 + 2] = (byte)(short)(dxImgCV.Data[x, y, 2]/10000);
                }

                // Write the pixel data into our bitmap
                this.Dispatcher.Invoke((Action)(() =>
                {
                    this.depthBitmap.WritePixels(
                        new Int32Rect(0, 0, this.depthBitmap.PixelWidth, this.depthBitmap.PixelHeight),
                        this.depthRGBPixels,
                        this.depthBitmap.PixelWidth * sizeof(int),
                        0);
                }));
            }

            lock (dpLock)
            {
                isActive = false;
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
