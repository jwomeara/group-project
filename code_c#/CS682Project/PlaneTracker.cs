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
        private int imageCounter = 0;
        private int maxImages = 20;
       

        // assume the user has associated images with identified planes in the gui. we send these to the tracker constructor.
        public PlaneTracker(List<Plane> planes, int maxImages) {
            this.planes = planes;
            this.maxImages = maxImages;
        }

        public List<Plane> GetPlanes() {
            return this.planes;
        }

        /// <summary>
        ///  takes all polygons just detected in the current frame. we haven't id'd them as planes of interest yet.
        ///  no return type. implicitly updates planes given to it.
        /// </summary>
        /// <param name="polys">polygons detected in the current frame</param>
        public void UpdatePlanes(List<System.Drawing.Point[]> polys) {

            List<Plane> trackedPlanes = new List<Plane>();
            int deathCount = 10;

            // check if the plane has reached its time limit of not having a matching polygon over n frames. if so, remove.
            foreach (Plane plane in this.planes)
            {
                if (plane.GetDeathClock() < deathCount)
                {
                    trackedPlanes.Add(plane);
                }
            }

            this.planes = trackedPlanes;

            // figure out which polygons match planes currently in plane tracker
            Dictionary<int, Plane> matches = MapPolygonsToPlanes(polys);

            // if a polygon matches a plane, update the plane.
            foreach (KeyValuePair<int, Plane> entry in matches)
            {
                // poly is new. add it to the plane list
                if (entry.Value == null) {
                    AddPlane(polys[entry.Key]);
                }
                // polygon has plane match. update the corresponding plane    
                else
                {
                    entry.Value.SetPoints(polys[entry.Key]);
                    entry.Value.SetDeathClock(0);
                }
            }

            // if no polygon matches a plane, increment the death counter of the plane
            foreach (Plane plane in this.planes) {
                if (!matches.ContainsValue(plane)) {
                    plane.SetDeathClock(plane.GetDeathClock() + 1);
                }

            }
        }

        /// <summary>
        /// Add a polygon to the plane tracker's list of planes to track
        /// Give this new plane coordinates and an image
        /// Increment the image counter
        /// </summary>
        /// <param name="polygon"></param>
        private void AddPlane(System.Drawing.Point[] polygon)
        {
            Plane newPlane = new Plane();
            newPlane.SetPoints(polygon);
            newPlane.SetOverlayImageIndex(imageCounter);
            this.planes.Add(newPlane);
            imageCounter++;
        }

        /// <summary>
        /// create a dictionary of polygon/plane pairs for each frame.
        /// for the current frame, for each polygon, find the closest matching plane.
        /// </summary>
        /// <param name="polygons"></param>
        /// <returns></returns>
        private Dictionary<int, Plane> MapPolygonsToPlanes(List<System.Drawing.Point[]> polygons) {

            Dictionary<int, Plane> matches = new Dictionary<int, Plane>();

            // make a dictionary of current frame polygons
            for (int i = 0; i < polygons.Count; i++)
            {
                matches.Add(i, null);
            }

            // loop through the planes
            for (int i = 0; i < this.planes.Count; i++)
            {
                // calculate centroid of the plane
                System.Drawing.PointF planeCentroid = calculateCentroid(this.planes.ElementAt(i).GetPoints());

                // calculate the distance between the plane and centroid
                double bestDistance = -1;
                int bestIndex = -1;

                // loop through the polygons
                for (int j = 0; j < polygons.Count; j++)
                {
                    System.Drawing.PointF polyCentroid = calculateCentroid(polygons[j]);

                    double distance = calculateEuclideanDistance(planeCentroid, polyCentroid);

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

                if (bestIndex != -1) {
                    Plane previousMatchPlane = matches[bestIndex];
                    System.Drawing.PointF polyCentroid = calculateCentroid(polygons[bestIndex]);
                    
                    // this polygon has already been chosen by a plane as a potential match
                    if (previousMatchPlane != null)
                    {
                        //compare the distance
                        System.Drawing.PointF previousCentroid = calculateCentroid(previousMatchPlane.GetPoints());
                        double previousDistance = calculateEuclideanDistance(previousCentroid, polyCentroid);
                        
                        // if bestDistance is now better update the plane that goes with the polygon
                        if (bestDistance < previousDistance)
                        {
                            matches[bestIndex] = this.planes.ElementAt(i);
                        }
                    }
                    else
                    {
                        // check if the matched centroid is within in the plane's bounding polygon (so it is relatively close)
                        if (pointPolygonTest(this.planes.ElementAt(i).GetPoints(), polyCentroid) != -1)
                        {
                            matches[bestIndex] = this.planes.ElementAt(i);
                        }
                      
                    }
                }
            }
            return matches;
        }


        /// <summary>
        /// Return a key given a value in 1:1 dictionary. -1 if no match
        /// </summary>
        /// <param name="plane"></param>
        /// <returns></returns>
        private int GetKeyByValue(Dictionary<int, Plane> dictionary, Plane plane)
        {
            foreach (KeyValuePair<int, Plane> entry in dictionary)
            {
                if (entry.Value == plane)
                {
                    return entry.Key;
                }
            }
            return -1;
        }

        /// <summary>
        /// calculateEuclideanDistance finds eucl distance between two points
        /// </summary>
        /// <param name="point1"></param>
        /// <param name="point2"></param>
        /// <returns></returns>
        private double calculateEuclideanDistance(System.Drawing.PointF point1, System.Drawing.PointF point2)
        {
            double distance = Math.Sqrt(
                Math.Pow((point1.X - point2.X), 2) + Math.Pow((point1.Y - point2.Y), 2));
            return distance;
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

        /// <summary>
        /// Takes the polygon vertices and the point.  Determines where the point lies.
        /// This is a direct port of the opencv code on github b/c i couldn't get the cvInvoke version working
        /// </summary>
        /// <param name="polyCoords">vertices of the polygon to use</param>
        /// <param name="point">point to check on</param>
        /// <returns>-1 if not in or on, 0 if on, 1 if in</returns>
         private int pointPolygonTest(System.Drawing.Point[] polygonVertices, System.Drawing.PointF point)
         {
             int result = 0;
             int counter = 0;
         
             System.Drawing.PointF v0, v;
         
             v = polygonVertices[polygonVertices.Count() - 1];

             for (int i = 0; i < polygonVertices.Count(); i++)
             {
                double dist;
                v0 = v;
                v = polygonVertices[i];

                if((v0.Y <= point.Y && v.Y <= point.Y) ||
                   (v0.Y > point.Y && v.Y > point.Y) ||
                   (v0.X < point.X && v.X < point.X) )
                {
                    if (point.Y == v.Y && (point.X == v.X || (point.Y == v0.Y &&
                        ((v0.X <= point.X && point.X <= v.X) || (v.X <= point.X && point.X <= v0.X)))))
                    {
                        return 0;
                    }
                    continue;
                }

                dist = (double)(point.Y - v0.Y)*(v.X - v0.X) - (double)(point.X - v0.X)*(v.Y - v0.Y);
               
                if( dist == 0 ) {
                    return 0;
                }
                
                if( v.Y < v0.Y) {
                    dist = -dist;
                }

                // counter += dist > 0;
                 if (dist > 0) {
                     counter = counter + 1;
                 }
             }
             result = (counter % 2 == 0) ? -1 : 1;

             return result;
         }
    }
}
