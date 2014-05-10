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
    class Plane
    {
        private int overlayImageIndex;
        private System.Drawing.Point[] points;
        private int deathClock = 0;


        public int GetDeathClock() {
            return deathClock;
        }

	    public int GetOverlayImageIndex() {
		    return overlayImageIndex;
	    }
	
	    public System.Drawing.Point[] GetPoints() {
		    return points;
	    }
	
	    public void SetDeathClock(int value) {
		    this.deathClock = value;
	    }
	
	    public void SetOverlayImageIndex(int overlayImageIndex) {
		    this.overlayImageIndex = overlayImageIndex;
	    }
	
	    public void SetPoints(System.Drawing.Point[] points) {
		    this.points = points;
	    }


    }
}
