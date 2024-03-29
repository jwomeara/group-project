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

import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

import javax.swing.JFrame;

public class ProjectApp
{

    private ProjectMain main;
    private boolean shouldRun = true;
    private JFrame frame;

    public ProjectApp (JFrame frame)
    {
        {
        	this.frame = frame;
        	frame.addKeyListener(new KeyListener()
    		{
    			@Override
    			public void keyTyped(KeyEvent arg0) {}
    			@Override
    			public void keyReleased(KeyEvent arg0) {}
    			@Override
    			public void keyPressed(KeyEvent arg0) {
    				if (arg0.getKeyCode() == KeyEvent.VK_ESCAPE)
    				{
    					shouldRun = false;
    				}
    			}
    		});
        }
    }

    public static void main(String s[]) {
        JFrame f = new JFrame("CS682 Project");
        f.addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent e) {System.exit(0);}
        });
        ProjectApp app = new ProjectApp(f);
        
        app.main = new ProjectMain();
        f.add("Center", app.main);
        f.pack();
        f.setVisible(true);
        
        app.run();
    }

    void run() {
        while(shouldRun) {
            main.updateDepth();
            main.repaint();
        }
        frame.dispose();
    }
    
}
