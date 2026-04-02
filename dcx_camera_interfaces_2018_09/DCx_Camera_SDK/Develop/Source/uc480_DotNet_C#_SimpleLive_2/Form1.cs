using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace SimpleLive
{
    public partial class uc480_DotNet_Simple_Live : Form
    {
        private uc480.Camera Camera;
        IntPtr displayHandle = IntPtr.Zero;
        private bool bLive          = false;

        public uc480_DotNet_Simple_Live()
        {
            InitializeComponent();

            displayHandle = DisplayWindow.Handle;
            InitCamera();
        }

        private void InitCamera()
        {
            Camera = new uc480.Camera();//Use only the empty constructor, the one with cameraID has a bug

            uc480.Defines.Status statusRet = 0;

            // Open Camera
            statusRet = Camera.Init();//You can specify a particular cameraId here if you want to open a specific camera

            if (statusRet != uc480.Defines.Status.SUCCESS)
            {
                MessageBox.Show("Camera initializing failed");
            }

            // Allocate Memory
            Int32 s32MemID;
            statusRet = Camera.Memory.Allocate(out s32MemID, true);
            if (statusRet != uc480.Defines.Status.SUCCESS)
            {
                MessageBox.Show("Allocate Memory failed");
            }

            // Start Live Video
            statusRet = Camera.Acquisition.Capture();
            if (statusRet != uc480.Defines.Status.SUCCESS)
            {
                MessageBox.Show("Start Live Video failed");
            }
            else
            {
                bLive = true;
            }

            // Connect Event
            Camera.EventFrame += onFrameEvent;

            CB_Auto_Gain_Balance.Enabled = Camera.AutoFeatures.Software.Gain.Supported;
            CB_Auto_White_Balance.Enabled = Camera.AutoFeatures.Software.WhiteBalance.Supported;
        }

        private void onFrameEvent(object sender, EventArgs e)
        {
            uc480.Camera Camera = sender as uc480.Camera;

            Int32 s32MemID;
            Camera.Memory.GetActive(out s32MemID);

            Camera.Display.Render(s32MemID, displayHandle, uc480.Defines.DisplayRenderMode.FitToWindow);
        }
        
        private void Button_Live_Video_Click(object sender, EventArgs e)
        {
            // Open Camera and Start Live Video
            if (Camera.Acquisition.Capture() == uc480.Defines.Status.SUCCESS)
            {
                bLive = true;
            }
        }

        private void Button_Stop_Video_Click(object sender, EventArgs e)
        {
            // Stop Live Video
            if (Camera.Acquisition.Stop() == uc480.Defines.Status.SUCCESS)
            {
                bLive = false;
            }
        }
        
        private void Button_Freeze_Video_Click(object sender, EventArgs e)
        {
            if (Camera.Acquisition.Freeze() == uc480.Defines.Status.SUCCESS)
            {
                bLive = false;
            }
        }

        private void CB_Auto_White_Balance_CheckedChanged(object sender, EventArgs e)
        {
            Camera.AutoFeatures.Software.WhiteBalance.SetEnable(CB_Auto_White_Balance.Checked);
        }

        private void CB_Auto_Gain_Balance_CheckedChanged(object sender, EventArgs e)
        {
            if (CB_Auto_Gain_Balance.Checked)
            {
                Camera.AutoFeatures.Software.Gain.SetEnable(true);
            }
            else
            {
                Camera.AutoFeatures.Software.Gain.SetEnable(false);
            }
        }

        private void Button_Load_Parameter_Click(object sender, EventArgs e)
        {
            Camera.Acquisition.Stop();

            Camera.Parameter.Load("");

            if (bLive == true)
            {
                Camera.Acquisition.Capture();
            }
        }

        private void Button_Exit_Prog_Click(object sender, EventArgs e)
        {
            // Close the Camera
            Camera.Exit();
            Close();
        }
    }
}
