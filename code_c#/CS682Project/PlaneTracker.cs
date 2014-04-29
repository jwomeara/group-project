using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Emgu.CV;
using Emgu.CV.UI;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.Util;
using Emgu.CV.GPU;

namespace CS682Project
{
    class PlaneTracker
    {

        private List<Plane> planes;

        // assume the user has associated images with identified planes in the gui. we send these to the tracker constructor.
        public PlaneTracker(List<Plane> planes)
        {
            this.planes = planes;
        }

        public List<Plane> GetPlanes()
        {
            return this.planes;
        }

        /// <summary>
        ///  takes all polygons just detected in the current frame. we haven't id'd them as planes of interest yet.
        ///  no return type. implicitly updates planes given to it.
        /// </summary>
        /// <param name="polys">polygons detected in the current frame</param>
        public void UpdatePlanes(List<System.Drawing.Point[]> polys)
        {
            // compare each plane being tracked to the current detected polys
            for (int i = 0; i < this.planes.Count; i++)
            {
                double bestDistance = -1;
                int bestIndex = -1;

                // 1. calculate centroid of the plane
                System.Drawing.PointF planeCentroid = calculateCentroid(this.planes.ElementAt(i).GetPoints());

                System.Diagnostics.Debug.WriteLine("plane index: " + i);
                System.Diagnostics.Debug.WriteLine("plane centroid: " + planeCentroid.X + " " + planeCentroid.Y);
        
                for (int j = 0; j < polys.Count; j++)
                {
                    // 1.calculate centroid of the poly
                    System.Drawing.PointF polyCentroid = new System.Drawing.Point();
                    polyCentroid = calculateCentroid(polys.ElementAt(j));

                    System.Diagnostics.Debug.WriteLine("poly index: " + j);
                    System.Diagnostics.Debug.WriteLine("poly centroid: " + polyCentroid.X + " " + polyCentroid.Y);
        

                    //2. calculate euclidean distance between poly and plane centroids
                    double distance = Math.Sqrt(
                        Math.Pow((planeCentroid.X - polyCentroid.X), 2) + 
                        Math.Pow((planeCentroid.Y - polyCentroid.Y), 2)
                        );

                    if (bestDistance == -1)
                    {
                        bestDistance = distance;
                        bestIndex = j;
                    }
                    else
                    {
                        if (distance < bestDistance)
                        {
                            bestDistance = distance;
                            bestIndex = j;
                        }
                    }
                }

            // when the id is made, update the points in the corresponding plane that is being tracked using bestDistance and bestIndex
            // first check if bestDistance is less than threshold (we'll say less than 50 pix for now)
            // TODO -- SNS to work on the threshold.
                double thresholdDistance = 50;

                System.Diagnostics.Debug.WriteLine("best distance: " + bestDistance);

                if (bestDistance < thresholdDistance)
                {
                    System.Diagnostics.Debug.WriteLine("i and bestIndex: " + i + " " + bestIndex);
                    this.planes.ElementAt(i).SetPoints(polys.ElementAt(bestIndex));        
                }
            }
        }

        /// <summary>
        /// calculateCentroid method finds the centroid of the polygon assuming equal mass at four points
        /// </summary>
        /// <param name="points">points of the polygon (should have four points, in sequence)</param>
        /// <returns> centroid in PointF form</returns>
         private System.Drawing.PointF calculateCentroid(System.Drawing.Point[] points)
        {
            float cX = -1;
            float cY = -1;
            float sumX = 0;
            float sumY = 0;
            float area = 0;
            
            foreach (System.Drawing.Point point in points) {
                sumX += point.X;
                sumY += point.Y;
            }

            cX = sumX / 4;
            cY = sumY / 4;

            return new System.Drawing.PointF(cX, cY);
        }
    }
}
