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
        private Image<Bgr, Byte> overlayImage;
        private System.Drawing.Point[] points;
        private int deathClock = 0;


        public int GetDeathClock() {
            return deathClock;
        }

	    public Image<Bgr, Byte> GetOverlayImage() {
		    return overlayImage;
	    }
	
	    public System.Drawing.Point[] GetPoints() {
		    return points;
	    }
	
	    public void SetDeathClock(int value) {
		    this.deathClock = value;
	    }
	
	    public void SetOverlayImage(Image<Bgr, Byte> overlayImage) {
		    this.overlayImage = overlayImage;
	    }
	
	    public void SetPoints(System.Drawing.Point[] points) {
		    this.points = points;
	    }


    }
}
