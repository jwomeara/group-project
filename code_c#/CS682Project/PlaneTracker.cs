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
        private int maxImages = 0;

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
           // track best matches as we go through as <polyIndex, List<planeIndex, distance>>
            Dictionary<int, List<Tuple<int, double>>> matches = new Dictionary<int, List<Tuple<int, double>>>();

            List<Plane> trackedPlanes = new List<Plane>();

            // if we don't find a match after this many frames, remove the plane
            int deathCount = 30;

            // calculate all the centroids for the polygons in the current frame
            List<System.Drawing.PointF> polyCentroids = new List<System.Drawing.PointF>();
            
            for (int i = 0; i < polys.Count; i++)
            {
                System.Drawing.PointF polyCentroid = new System.Drawing.Point();
                polyCentroid = calculateCentroid(polys.ElementAt(i));
                polyCentroids.Add(polyCentroid);
            }

            // check if the plane has reached its limit of not having a matching polygon. if so, remove it
            foreach (Plane plane in this.planes)
            {
                if (plane.GetDeathClock() < deathCount)
                {
                    trackedPlanes.Add(plane);
                }
            }

            this.planes = trackedPlanes;

            // compare each plane being tracked to each of the polygons in the current frame
            for (int i = 0; i < this.planes.Count; i++)
            {
                double bestDistance = -1;
                int bestIndex = -1;
                System.Drawing.PointF bestCentroid;

                // 1. calculate centroid of the plane
                System.Drawing.PointF planeCentroid = calculateCentroid(this.planes.ElementAt(i).GetPoints());

                // compare to each of the polygons in the current frame
                for (int j = 0; j < polys.Count; j++)
                {
                    //2. calculate euclidean distance between poly and plane centroids
                    double distance = Math.Sqrt(
                        Math.Pow((planeCentroid.X - polyCentroids[j].X), 2) + 
                        Math.Pow((planeCentroid.Y - polyCentroids[j].Y), 2)
                        );

                    if (bestDistance == -1) {
                        bestDistance = distance;
                        bestIndex = j;
                        bestCentroid = polyCentroids[j];
                    }
                    else {
                        if (distance < bestDistance)
                        {
                            bestDistance = distance;
                            bestIndex = j;
                            bestCentroid = polyCentroids[j];
                        }
                    }
                } // end of polys loop

                 // when a potential plane/poly match is made, we track it. bestPolyIndex, List[(planeIndex, bestDistance)]
                System.Diagnostics.Debug.WriteLine("best index: " + bestIndex);

                if (bestIndex != -1)
                {
                    if (!matches.ContainsKey(bestIndex))
                    {
                        // add the bestIndex key to matches
                        matches.Add(bestIndex, new List<Tuple<int, double>>());
                    }
                    matches[bestIndex].Add(Tuple.Create<int, double>(i, bestDistance));
                }

            } // end of planes for loop

            // now we need to check the dictionary to find if any polys get associated with multiple planes
            foreach (KeyValuePair<int, List<Tuple<int, double>>> entry in matches)
            {
                int polyIndex = entry.Key;
                List<Tuple<int, double>> candidates = entry.Value;

                // case 1. only one plane matched this polygon
                if (candidates.Count == 1)
                {
                    System.Diagnostics.Debug.WriteLine("plane/poly check");
                    System.Diagnostics.Debug.WriteLine(this.planes.Count);
                  

                    // check if the polygon's centroid is inside or incident to the plane. if it is consider it a match
                    if (pointPolygonTest(this.planes.ElementAt(candidates[0].Item1).GetPoints(), polyCentroids[polyIndex]) > -1)
                    {
                        this.planes.ElementAt(candidates[0].Item1).SetPoints(polys.ElementAt(polyIndex));
                        this.planes.ElementAt(candidates[0].Item1).SetDeathClock(0);
                    }
                    // otherwise we won't use as a match. increment the death clock and keep current set of points for this plane.
                    else
                    {
                        this.planes.ElementAt(candidates[0].Item1).SetDeathClock(this.planes.ElementAt(candidates[0].Item1).GetDeathClock() + 1);
                    }
                }
                    // case 2. multiple things in the list. find smallest distance match.
                    else
                    {
                        int smallestDistanceIndex = 0;

                        // find the smallest distance of the candidates
                        for (int i = 0; i < candidates.Count; i++)
                        {
                            double smallestDistance = candidates[smallestDistanceIndex].Item2;
                            double currentDistance = candidates[i].Item2;

                            if (currentDistance <= smallestDistance)
                            {
                                smallestDistanceIndex = i;
                            }
                        }

                        // given the smallest distance, update the plane that it goes with and increment deathclock for the rest of planes
                        for (int i = 0; i < candidates.Count; i++)
                        {
                            int planeIndex = candidates[i].Item1;

                            // set the smallest distance polygon
                            if (i == smallestDistanceIndex)
                            {

                                // check if the polygon's centroid is inside or incident to the plane. if it is consider it a match
                                if (pointPolygonTest(this.planes.ElementAt(planeIndex).GetPoints(), polyCentroids[polyIndex]) > -1)
                                {
                                    this.planes.ElementAt(planeIndex).SetPoints(polys.ElementAt(polyIndex));
                                    this.planes.ElementAt(planeIndex).SetDeathClock(0);
                                }
                                // otherwise we won't use as a match. increment the death clock and keep current set of points for this plane.
                                else
                                {
                                    this.planes.ElementAt(candidates[0].Item1).SetDeathClock(this.planes.ElementAt(candidates[0].Item1).GetDeathClock() + 1);
                                }
                            }
                            // otherwise, there was a polygon closer than this one. increment the death clock and keep current set of points for the plane
                            else
                            {
                                this.planes.ElementAt(planeIndex).SetDeathClock(this.planes.ElementAt(planeIndex).GetDeathClock() + 1);
                            }
                        }
                    } // end of multiple matches else
                } // end of dictionary check

            // for any polygons in the current frame that do not match a plane, start tracking them 
            for (int i = 0; i < polys.Count; i++)
            {
                // no potential match.  Add to the plane list for tracking if we have images available.
                if (!matches.ContainsKey(i) && (imageCounter < maxImages))
                {
                    Plane newPlane = new Plane();
                    newPlane.SetPoints(polys[i]);
                    newPlane.SetOverlayImageIndex(imageCounter);
                    this.planes.Add(newPlane);
                    imageCounter++;
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
