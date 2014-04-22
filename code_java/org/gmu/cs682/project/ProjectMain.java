/*****************************************************************************
*                                                                            *
*  OpenNI 1.x Alpha                                                          *
*  Copyright (C) 2012 PrimeSense Ltd.                                        *
*                                                                            *
*  This file is part of OpenNI.                                              *
*                                                                            *
*  Licensed under the Apache License, Version 2.0 (the "License");           *
*  you may not use this file except in compliance with the License.          *
*  You may obtain a copy of the License at                                   *
*                                                                            *
*      http://www.apache.org/licenses/LICENSE-2.0                            *
*                                                                            *
*  Unless required by applicable law or agreed to in writing, software       *
*  distributed under the License is distributed on an "AS IS" BASIS,         *
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
*  See the License for the specific language governing permissions and       *
*  limitations under the License.                                            *
*                                                                            *
*****************************************************************************/
package org.gmu.cs682.project;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.openni.*;

import java.nio.ShortBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.awt.*;
import java.awt.image.*;

class ProjectMain extends Component {

	class DepthPixel implements Comparable<DepthPixel> {
	    int index;
	    float value;
	    float direction;
	    int group;

	    public DepthPixel(int index, float value) {
	        this.index = index;
	        this.value = value;
	    }
	    
	    @Override
	    public int compareTo(DepthPixel o) {
	        return value < o.value ? -1 : value > o.value ? 1 : 0;
	    }
	}
	
    /**
	 * 
	 */
    private static final long serialVersionUID = 1L;
    private OutArg<ScriptNode> scriptNode;
    private Context context;
    private DepthGenerator depthGen;
    private byte[] imgbytes;
    private float histogram[];

    private BufferedImage bimg;
    int width, height;

    private final String SAMPLE_XML_FILE = "./config/Config.xml";    
    public ProjectMain() {
    	
        try {
            scriptNode = new OutArg<ScriptNode>();
            context = Context.createFromXmlFile(SAMPLE_XML_FILE, scriptNode);

            depthGen = DepthGenerator.create(context);
            DepthMetaData depthMD = depthGen.getMetaData();

            histogram = new float[10000];
            width = depthMD.getFullXRes();
            height = depthMD.getFullYRes();
            
            imgbytes = new byte[width*height];
            
            DataBufferByte dataBuffer = new DataBufferByte(imgbytes, width*height);
            Raster raster = Raster.createPackedRaster(dataBuffer, width, height, 8, null);
            bimg = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
            bimg.setData(raster);

        } catch (GeneralException e) {
            e.printStackTrace();
            System.exit(1);
        }
    }
    
    private void calcHist(DepthMetaData depthMD)
    {
        // reset
        for (int i = 0; i < histogram.length; ++i)
            histogram[i] = 0;
        
        ShortBuffer depth = depthMD.getData().createShortBuffer();
        depth.rewind();

        int points = 0;
        while(depth.remaining() > 0)
        {
            short depthVal = depth.get();
            if (depthVal != 0)
            {
                histogram[depthVal]++;
                points++;
            }
        }
        
        for (int i = 1; i < histogram.length; i++)
        {
            histogram[i] += histogram[i-1];
        }

        if (points > 0)
        {
            for (int i = 1; i < histogram.length; i++)
            {
                histogram[i] = (int)(256 * (1.0f - (histogram[i] / (float)points)));
            }
        }
    }


    void updateDepth()
    {
        try {
            DepthMetaData depthMD = depthGen.getMetaData();

            context.waitAnyUpdateAll();
            
            calcHist(depthMD);
            ShortBuffer depth = depthMD.getData().createShortBuffer();
            depth.rewind();
            
            Mat image = Mat.zeros(height, width, CvType.CV_32F);
            float floatBuff[] = new float[width*height];
            image.get(0, 0, floatBuff);
            
            while(depth.remaining() > 0)
            {
                int pos = depth.position();
                floatBuff[pos] = (float)depth.get();
            }
            
            image.put(0, 0, floatBuff);
            
            //Imgproc.GaussianBlur(image, image, new Size(3,3), 0, 0, Imgproc.BORDER_DEFAULT);
            
            Mat dx = Mat.zeros(height, width, CvType.CV_32F);
            Mat dy = Mat.zeros(height, width, CvType.CV_32F);
            
            Imgproc.Sobel(image, dx, CvType.CV_32F, 1, 0);
            Imgproc.Sobel(image, dy, CvType.CV_32F, 0, 1);
            
            Core.magnitude(dx, dy, image);
            
            
            image.put(0, 0, floatBuff);
            ArrayList<DepthPixel> pixelList = new ArrayList<DepthPixel>();
            for (int i = 0; i < height*width; i++)
            	pixelList.add(new DepthPixel(i, floatBuff[i]));
            
            // sort pixels by their gradient value
            Collections.sort(pixelList);
            
            // group pixels according to the range threshold
            float thresh = 30.0f;
            float lastValue = pixelList.get(0).value;
            int group = 0;
            for (int i = 0; i < pixelList.size(); i++){
            	if (pixelList.get(i).value - lastValue > thresh){
            		pixelList.get(i).group = ++group;
            	}
            	else{
            		pixelList.get(i).group = group;
            	}
            	lastValue = pixelList.get(i).value;
            }
            
            System.out.println("There are [" + group + "] planes!");
            
            int stride = (int)(255.0/(float)group);
            for (int i = 0; i < pixelList.size(); i++){
            	imgbytes[pixelList.get(i).index] = (byte)(short)((float)(stride*pixelList.get(i).group));
            }
            
        } catch (GeneralException e) {
            e.printStackTrace();
        }
    }


    public Dimension getPreferredSize() {
        return new Dimension(width, height);
    }

    public void paint(Graphics g) {
        DataBufferByte dataBuffer = new DataBufferByte(imgbytes, width*height);
        Raster raster = Raster.createPackedRaster(dataBuffer, width, height, 8, null);
        bimg.setData(raster);

        g.drawImage(bimg, 0, 0, null);
    }
}

