using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace DotnetCockpit
{
    public partial class MainForm : Form
    {

        #region Variables

        // our camera class
        uc480.Camera m_Camera;

        uc480.Defines.DisplayRenderMode m_RenderMode; // render mode
        Boolean m_IsLive; // saves the capture state
        Int32 m_s32FrameCount; 
        private Timer m_UpdateTimer = new Timer();

        #endregion

        public MainForm()
        {
            InitializeComponent();

            // Check Runtime Version
            Version verMin = new Version(3, 5);
            Boolean bOk = false;
            foreach (Version ver in InstalledDotNetVersions())
            {
                if (ver >= verMin)
                {
                    bOk = true;
                    break;
                }
            }

            if (!bOk)
            {
                this.Load += CloseOnStart;
            }

            pictureBoxDisplay.SizeMode = PictureBoxSizeMode.CenterImage;
            this.pictureBoxDisplay.Width = panelDisplay.Width;
            this.pictureBoxDisplay.Height = panelDisplay.Height;

            // initialize camera object
            // camera is not opened here
            m_Camera = new uc480.Camera();

            m_IsLive = false;
            m_RenderMode = uc480.Defines.DisplayRenderMode.FitToWindow;

            m_UpdateTimer.Interval = 100;
            m_UpdateTimer.Tick += OnUpdateControls;
        }

        private void CloseOnStart(object sender, EventArgs e)
        {
            MessageBox.Show(".NET Runtime Version 3.5.0 is required", "Runtime Error");
            this.Close();
        }

        public static System.Collections.ObjectModel.Collection<Version> InstalledDotNetVersions()
        {
            System.Collections.ObjectModel.Collection<Version> versions = new System.Collections.ObjectModel.Collection<Version>();
            Microsoft.Win32.RegistryKey NDPKey = Microsoft.Win32.Registry.LocalMachine.OpenSubKey(@"SOFTWARE\Microsoft\NET Framework Setup\NDP");
            if (NDPKey != null)
            {
                string[] subkeys = NDPKey.GetSubKeyNames();
                foreach (string subkey in subkeys)
                {
                    GetDotNetVersion(NDPKey.OpenSubKey(subkey), subkey, versions);
                    GetDotNetVersion(NDPKey.OpenSubKey(subkey).OpenSubKey("Client"), subkey, versions);
                    GetDotNetVersion(NDPKey.OpenSubKey(subkey).OpenSubKey("Full"), subkey, versions);
                }
            }
            return versions;
        }

        private static void GetDotNetVersion(Microsoft.Win32.RegistryKey parentKey, string subVersionName, System.Collections.ObjectModel.Collection<Version> versions)
        {
            if (parentKey != null)
            {
                string installed = Convert.ToString(parentKey.GetValue("Install"));
                if (installed == "1")
                {
                    string version = Convert.ToString(parentKey.GetValue("Version"));
                    if (string.IsNullOrEmpty(version))
                    {
                        if (subVersionName.StartsWith("v"))
                            version = subVersionName.Substring(1);
                        else
                            version = subVersionName;
                    }

                    Version ver = new Version(version);

                    if (!versions.Contains(ver))
                        versions.Add(ver);
                }
            }
        }

        private void OnUpdateControls(object sender, EventArgs e)
        {
            // we update here our statusbar 
            Double dFramerate;
            m_Camera.Timing.Framerate.GetCurrentFps(out dFramerate);
            toolStripStatusLabelFPS.Text = "Fps: " + dFramerate.ToString("00.00");

            uc480.Types.CaptureStatus captureStatus;
            m_Camera.Information.GetCaptureStatus(out captureStatus);

            toolStripStatusLabelFailed.Text = "Failed: " + captureStatus.Total;
            toolStripStatusLabelFrameCount.Text = "Frames: " + m_s32FrameCount;
        }

        private uc480.Defines.Status initCamera()
        {
            CameraChoose chooseForm = new CameraChoose();
            uc480.Defines.Status statusRet = uc480.Defines.Status.NO_SUCCESS;
            
            if (chooseForm.ShowDialog() == DialogResult.OK)
            {
                statusRet = m_Camera.Init(chooseForm.DeviceID | (Int32)uc480.Defines.DeviceEnumeration.UseDeviceID, pictureBoxDisplay.Handle);
                if (statusRet != uc480.Defines.Status.SUCCESS)
                {
                    MessageBox.Show("Initializing the camera failed");
                    return statusRet;
                }
               
                statusRet = m_Camera.Memory.Allocate();
                if (statusRet != uc480.Defines.Status.SUCCESS)
                {
                    MessageBox.Show("Allocating memory failed");
                    return statusRet;
                }

                // set event
                m_Camera.EventFrame += onFrameEvent;

                // reset framecount
                m_s32FrameCount = 0;

                // start update timer for our statusbar
                m_UpdateTimer.Start();

                uc480.Types.SensorInfo sensorInfo;
                m_Camera.Information.GetSensorInfo(out sensorInfo);

                pictureBoxDisplay.SizeMode = PictureBoxSizeMode.Normal;
                toolStripStatusLabelCamera.Text = sensorInfo.SensorName;    
            }

            return statusRet;
        }

        private void UpdateToolbar()
        {
            toolStripButton1To1.Enabled = m_Camera.IsOpened;
            toolStripButton1To2.Enabled = m_Camera.IsOpened;
            toolStripButtonFitToWnd.Enabled = m_Camera.IsOpened;

            // general function to update all mainwindow controls
            if (m_Camera.IsOpened)
            {
                // check if directrender or dib mode is active
                uc480.Defines.DisplayMode displayMode;
                m_Camera.Display.Mode.Get(out displayMode);
    
                toolStripButton1To2.Enabled = displayMode == uc480.Defines.DisplayMode.DiB;
            }

            // enable/disable controls
            toolStripButtonAES.Enabled = m_Camera.IsOpened ? m_Camera.AutoFeatures.Software.Shutter.Supported : false;
            toolStripButtonAWB.Enabled = m_Camera.IsOpened ? m_Camera.AutoFeatures.Software.WhiteBalance.Supported : false;
            toolStripButtonExit.Enabled = m_Camera.IsOpened;
            toolStripButtonFreerun.Enabled = m_Camera.IsOpened;
            toolStripButtonSettings.Enabled = m_Camera.IsOpened;
            toolStripButtonSnapshot.Enabled = m_Camera.IsOpened;
            toolStripButtonSaveImage.Enabled = m_Camera.IsOpened;
            toolStripButtonLoadImage.Enabled = m_Camera.IsOpened;
            toolStripButtonVideoRec.Enabled = m_Camera.IsOpened;

            toolStripButtonOpenFreerun.Enabled = !m_Camera.IsOpened;
            toolStripButtonOpenStop.Enabled = !m_Camera.IsOpened;
            
            toolStripButtonFreerun.Checked = m_Camera.IsOpened ? m_IsLive : false;
            toolStripButtonVideoRec.Checked = m_Camera.IsOpened ? m_Camera.Video.Running : false;

            toolStripMenuItemCloseCamera.Enabled = m_Camera.IsOpened;
            toolStripMenuItemLoadImage.Enabled = m_Camera.IsOpened;
            toolStripMenuItemLoadParameterFromEEPROM.Enabled = m_Camera.IsOpened;
            toolStripMenuItemLoadParameterFromFile.Enabled = m_Camera.IsOpened;
            toolStripMenuItemOpenCamera.Enabled = !m_Camera.IsOpened;
            toolStripMenuItemOpenCameraLive.Enabled = !m_Camera.IsOpened;
            toolStripMenuItemSaveImage.Enabled = m_Camera.IsOpened;
            toolStripMenuItemSaveParameterToEEPROM.Enabled = m_Camera.IsOpened;
            toolStripMenuItemSaveParameterToFile.Enabled = m_Camera.IsOpened;

            toolStripMenuItemSnapshot.Enabled = m_Camera.IsOpened;
            toolStripMenuItemFreerun.Enabled = m_Camera.IsOpened;
            toolStripMenuItemFreerun.Checked = m_Camera.IsOpened ? m_IsLive : false;

            toolStripMenuItem1To1.Enabled = m_Camera.IsOpened;
            toolStripMenuItem1To2.Enabled = m_Camera.IsOpened;
            toolStripMenuItemFitToWindow.Enabled = m_Camera.IsOpened;

            toolStripMenuItemMirrorRightLeft.Enabled = m_Camera.IsOpened;
            toolStripMenuItemMirrorUpDown.Enabled = m_Camera.IsOpened;
            toolStripMenuItemCrosshair.Enabled = m_Camera.IsOpened;
            toolStripMenuItemTimestamp.Enabled = m_Camera.IsOpened;
            toolStripMenuItemApiErrorReport.Enabled = m_Camera.IsOpened;

            toolStripButtonFitToWnd.Checked = false;
            toolStripButton1To1.Checked = false;
            toolStripButton1To2.Checked = false;

            toolStripMenuItemFitToWindow.Checked = false;
            toolStripMenuItem1To1.Checked = false;
            toolStripMenuItem1To2.Checked = false;

            toolStripStatusLabelFPS.Visible = m_Camera.IsOpened;
            toolStripStatusLabelFailed.Visible = m_Camera.IsOpened;
            toolStripStatusLabelFrameCount.Visible = m_Camera.IsOpened;
            toolStripStatusLabelCamera.Visible = m_Camera.IsOpened;

            if (m_Camera.IsOpened)
            {
                // update render mode
                switch (m_RenderMode)
                {
                    case uc480.Defines.DisplayRenderMode.FitToWindow:
                        toolStripButtonFitToWnd.Checked = true;
                        toolStripMenuItemFitToWindow.Checked = true;
                        break;

                    case uc480.Defines.DisplayRenderMode.Normal:
                        toolStripButton1To1.Checked = true;
                        toolStripMenuItem1To1.Checked = true;
                        break;

                    case uc480.Defines.DisplayRenderMode.DownScale_1_2:
                        toolStripButton1To2.Checked = true;
                        toolStripMenuItem1To2.Checked = true;
                        break;
                }

                // update 
                Boolean isEnabled;
                m_Camera.AutoFeatures.Software.WhiteBalance.GetEnable(out isEnabled);
                toolStripButtonAWB.Checked = isEnabled;

                m_Camera.AutoFeatures.Software.Shutter.GetEnable(out isEnabled);
                toolStripButtonAES.Checked = isEnabled;
            }
            else
            {
                toolStripMenuItemCrosshair.Checked = false;
                toolStripMenuItemTimestamp.Checked = false;

                toolStripButtonAES.Checked = false;
                toolStripButtonAWB.Checked = false;
            }
        }

        private void onFrameEvent(object sender, EventArgs e)
        {
            // convert sender object to our camera object
            uc480.Camera camera = sender as uc480.Camera;

            if (camera.IsOpened)
            {
                uc480.Defines.DisplayMode mode;
                camera.Display.Mode.Get(out mode);

                // only display in dib mode
                if (mode == uc480.Defines.DisplayMode.DiB)
                {
                    Int32 s32MemID;
                    camera.Memory.GetActive(out s32MemID);
                    camera.Memory.Lock(s32MemID);

                    // do any drawings?
                    if (toolStripMenuItemTimestamp.Checked || toolStripMenuItemCrosshair.Checked)
                    {
                        Bitmap bitmap;
                        m_Camera.Memory.ToBitmap(s32MemID, out bitmap);

                        if (bitmap != null && bitmap.PixelFormat != System.Drawing.Imaging.PixelFormat.Format8bppIndexed)
                        {
                            Graphics graphics = Graphics.FromImage(bitmap);
                            DoDrawing(ref graphics, s32MemID);

                            graphics.Dispose();
                            bitmap.Dispose();
                        }
                    }
                    camera.Memory.Unlock(s32MemID);
                    camera.Display.Render(s32MemID, m_RenderMode);
                }

                ++m_s32FrameCount;
            }
        }

        #region Toolbar Events

        private void DoDrawing(ref Graphics graphics, Int32 s32MemID)
        {
            if (toolStripMenuItemCrosshair.Checked)
            {
                // get image size
                System.Drawing.Rectangle rect;
                m_Camera.Size.AOI.Get(out rect);

                graphics.DrawLine(new Pen(Color.Red), rect.Width / 2, 0, rect.Width / 2, rect.Height);
                graphics.DrawLine(new Pen(Color.Red), 0, rect.Height / 2, rect.Width, rect.Height / 2);
            }

            if (toolStripMenuItemTimestamp.Checked && s32MemID != 0)
            {
                uc480.Types.ImageInfo imageInfo;
                m_Camera.Information.GetImageInfo(s32MemID, out imageInfo);

                graphics.DrawString(imageInfo.TimestampSystem.ToString(), new Font("Courier", 16), Brushes.Red, 0, 0);
            }
        }

        private void toolStripButtonOpenFreerun_Click(object sender, EventArgs e)
        {
            uc480.Defines.Status statusRet;
            statusRet = initCamera();

            if (statusRet == uc480.Defines.Status.SUCCESS)
            {
                // start capture
                statusRet = m_Camera.Acquisition.Capture();
                if (statusRet != uc480.Defines.Status.SUCCESS)
                {
                    MessageBox.Show("Starting live video failed");
                }
                else
                {
                    // everything is ok
                    m_IsLive = true;
                    UpdateToolbar();
                }
            }

            // cleanup on any camera error
            if (statusRet != uc480.Defines.Status.SUCCESS && m_Camera.IsOpened)
            {
                m_Camera.Exit();
            }
        }

        private void toolStripButtonOpenStop_Click(object sender, EventArgs e)
        {
            uc480.Defines.Status statusRet;
            statusRet = initCamera();

            if (statusRet == uc480.Defines.Status.SUCCESS)
            {
                // start capture
                statusRet = m_Camera.Acquisition.Freeze();
                if (statusRet != uc480.Defines.Status.SUCCESS)
                {
                    MessageBox.Show("Starting live video failed");
                }
                else
                {
                    // everything is ok
                    m_IsLive = false;
                    UpdateToolbar();
                }
            }

            // cleanup on any camera error
            if (statusRet != uc480.Defines.Status.SUCCESS && m_Camera.IsOpened)
            {
                m_Camera.Exit();
            }
        }

        private void toolStripButtonExit_Click(object sender, EventArgs e)
        {
            m_UpdateTimer.Stop();

            m_IsLive = false;
            m_Camera.Exit();

            UpdateToolbar();

            // set correct display size
            this.pictureBoxDisplay.Width = panelDisplay.Width;
            this.pictureBoxDisplay.Height = panelDisplay.Height;
            this.pictureBoxDisplay.Anchor = AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right | AnchorStyles.Top;

            pictureBoxDisplay.Invalidate();
            pictureBoxDisplay.SizeMode = PictureBoxSizeMode.CenterImage;

            m_RenderMode = uc480.Defines.DisplayRenderMode.FitToWindow;
        }

        private void toolStripButtonFitToWnd_Click(object sender, EventArgs e)
        {
            // render mode == Fit to window
            m_RenderMode = uc480.Defines.DisplayRenderMode.FitToWindow;

            this.pictureBoxDisplay.Width = panelDisplay.Width;
            this.pictureBoxDisplay.Height = panelDisplay.Height;

            this.pictureBoxDisplay.Anchor = AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right | AnchorStyles.Top;

            // direct render
            uc480.Defines.DisplayMode mode;
            m_Camera.Display.Mode.Get(out mode);
            if (mode != uc480.Defines.DisplayMode.DiB)
            {
                m_Camera.DirectRenderer.SetScaling(true);
            }

            UpdateToolbar();
        }

        private void toolStripButton1To1_Click(object sender, EventArgs e)
        {
            // render mode == 1 to 1
            m_RenderMode = uc480.Defines.DisplayRenderMode.Normal;

            this.pictureBoxDisplay.Anchor = AnchorStyles.Top | AnchorStyles.Left;

            // get image size
            System.Drawing.Rectangle rect;
            m_Camera.Size.AOI.Get(out rect);

            this.pictureBoxDisplay.Width = rect.Width;
            this.pictureBoxDisplay.Height = rect.Height;

            // direct render
            uc480.Defines.DisplayMode mode;
            m_Camera.Display.Mode.Get(out mode);
            if (mode != uc480.Defines.DisplayMode.DiB)
            {
                m_Camera.DirectRenderer.SetScaling(false);
            }

            UpdateToolbar();
        }

        private void toolStripButton1To2_Click(object sender, EventArgs e)
        {
            // render mode == 1 to 2
            m_RenderMode = uc480.Defines.DisplayRenderMode.DownScale_1_2;

            this.pictureBoxDisplay.Anchor = AnchorStyles.Top | AnchorStyles.Left;

            // get image size
            System.Drawing.Rectangle rect;
            m_Camera.Size.AOI.Get(out rect);

            this.pictureBoxDisplay.Width = rect.Width / 2;
            this.pictureBoxDisplay.Height = rect.Height / 2;

            UpdateToolbar();
        }

        #endregion

        private void toolStripButtonFreerun_Click(object sender, EventArgs e)
        {
            uc480.Defines.Status statusRet;
            if (toolStripButtonFreerun.Checked)
            {
                // start capture
                statusRet = m_Camera.Acquisition.Capture();
                m_IsLive = true;
            }
            else
            {
                statusRet = m_Camera.Acquisition.Stop();
                m_IsLive = false;
            }

            UpdateToolbar();
        }

        private void toolStripButtonSnapshot_Click(object sender, EventArgs e)
        {
            uc480.Defines.Status statusRet;

            // start capture
            statusRet = m_Camera.Acquisition.Freeze();
            m_IsLive = false;

            UpdateToolbar();
        }

        private void OnDisplayChanged(object sender, EventArgs e)
        {
            uc480.Defines.DisplayMode displayMode;
            m_Camera.Display.Mode.Get(out displayMode);

            // set scaling options
            if (displayMode != uc480.Defines.DisplayMode.DiB)
            {
                if (m_RenderMode == uc480.Defines.DisplayRenderMode.DownScale_1_2)
                {
                    m_RenderMode = uc480.Defines.DisplayRenderMode.Normal;

                    this.pictureBoxDisplay.Anchor = AnchorStyles.Top | AnchorStyles.Left;

                    // get image size
                    System.Drawing.Rectangle rect;
                    m_Camera.Size.AOI.Get(out rect);

                    this.pictureBoxDisplay.Width = rect.Width;
                    this.pictureBoxDisplay.Height = rect.Height;
                }
                else
                {
                    m_Camera.DirectRenderer.SetScaling(m_RenderMode == uc480.Defines.DisplayRenderMode.FitToWindow);
                }

                // update drawings
                toolStripMenuItemCrosshair_Click(null, EventArgs.Empty);
            }
            else
            {
                if (m_RenderMode != uc480.Defines.DisplayRenderMode.FitToWindow)
                {
                    this.pictureBoxDisplay.Anchor = AnchorStyles.Top | AnchorStyles.Left;

                    // get image size
                    System.Drawing.Rectangle rect;
                    m_Camera.Size.AOI.Get(out rect);

                    if (m_RenderMode != uc480.Defines.DisplayRenderMode.Normal)
                    {

                        this.pictureBoxDisplay.Width = rect.Width / 2;
                        this.pictureBoxDisplay.Height = rect.Height / 2;
                    }
                    else
                    {
                        this.pictureBoxDisplay.Width = rect.Width;
                        this.pictureBoxDisplay.Height = rect.Height;
                    }
                }
            }

            UpdateToolbar();
        }

        private void toolStripButtonSettings_Click(object sender, EventArgs e)
        {
            uc480.Types.SensorInfo sensorInfo;
            m_Camera.Information.GetSensorInfo(out sensorInfo);

            if (sensorInfo.SensorID != uc480.Defines.Sensor.XS &&
                sensorInfo.SensorID != uc480.Defines.Sensor.UI1008_C)
            {
                // avoid multiple instances
                SettingsForm settingsForm = new SettingsForm(m_Camera);
                settingsForm.FormatControl.DisplayChanged += OnDisplayChanged;
                settingsForm.ShowDialog();

                Boolean isEnabled;

                // check if any autofeature is enabled
                m_Camera.AutoFeatures.Software.WhiteBalance.GetEnable(out isEnabled);
                toolStripButtonAWB.Checked = isEnabled;

                m_Camera.AutoFeatures.Software.Shutter.GetEnable(out isEnabled);
                toolStripButtonAES.Checked = isEnabled;

                UpdateToolbar();
            }
            else
            {
                MessageBox.Show("Settings are not supported with this type of camera!");
            }
        }

        private void toolStripButtonAES_Click(object sender, EventArgs e)
        {
            Boolean isEnabled;
            m_Camera.AutoFeatures.Software.Shutter.GetEnable(out isEnabled);
            
            m_Camera.AutoFeatures.Software.Shutter.SetEnable(!isEnabled);
            toolStripButtonAES.Checked = !isEnabled;
        }

        private void toolStripButtonAWB_Click(object sender, EventArgs e)
        {
            Boolean isEnabled;
            m_Camera.AutoFeatures.Software.WhiteBalance.GetEnable(out isEnabled);

            m_Camera.AutoFeatures.Software.WhiteBalance.SetEnable(!isEnabled);
            toolStripButtonAWB.Checked = !isEnabled;
        }

        private void toolStripButtonSaveImage_Click(object sender, EventArgs e)
        {
            m_Camera.Image.Save("");
        }

        private void toolStripButtonLoadImage_Click(object sender, EventArgs e)
        {
            uc480.Defines.Status statusRet;
            statusRet = m_Camera.Image.Load("");

            if (statusRet == uc480.Defines.Status.SUCCESS)
            {
                // update drawing
                onFrameEvent(m_Camera, EventArgs.Empty);
            }
        }

        private void toolStripButtonAbout_Click(object sender, EventArgs e)
        {
            AboutBox aboutBox = new AboutBox();
            aboutBox.ShowDialog();
        }

        private void toolStripMenuItemLoadParameterFromFile_Click(object sender, EventArgs e)
        {
            m_Camera.Parameter.Load("");

            if (m_IsLive)
            {
                m_Camera.Acquisition.Stop();
            }

            // adjust our memory
            Int32 []memList;
            m_Camera.Memory.GetList(out memList);
            m_Camera.Memory.Free(memList);

            // allocate new standard memory
            m_Camera.Memory.Allocate();

            if (m_IsLive)
            {
                m_Camera.Acquisition.Capture();
            }

            UpdateToolbar();
        }

        private void toolStripMenuItemLoadParameterFromEEPROM_Click(object sender, EventArgs e)
        {
            m_Camera.Parameter.Load();

            // adjust our memory
            Int32[] memList;
            m_Camera.Memory.GetList(out memList);
            m_Camera.Memory.Free(memList);

            // allocate new standard memory
            m_Camera.Memory.Allocate();

            if (m_IsLive)
            {
                m_Camera.Acquisition.Capture();
            }

            UpdateToolbar();
        }

        private void toolStripMenuItemSaveParameterToFile_Click(object sender, EventArgs e)
        {
            m_Camera.Parameter.Save("");
        }

        private void toolStripMenuItemSaveParameterToEEPROM_Click(object sender, EventArgs e)
        {
            m_Camera.Parameter.Save();
        }

        private void toolStripMenuItemExit_Click(object sender, EventArgs e)
        {
            Close();
        }

        private void toolStripMenuItemMirrorRightLeft_Click(object sender, EventArgs e)
        {
            m_Camera.RopEffect.Set(uc480.Defines.RopEffectMode.LeftRight, toolStripMenuItemMirrorRightLeft.Checked);
        }

        private void toolStripMenuItemMirrorUpDown_Click(object sender, EventArgs e)
        {
            m_Camera.RopEffect.Set(uc480.Defines.RopEffectMode.UpDown, toolStripMenuItemMirrorUpDown.Checked);
        }

        private void toolStripMenuItemApiErrorReport_Click(object sender, EventArgs e)
        {
            m_Camera.Information.SetEnableErrorReport(toolStripMenuItemApiErrorReport.Checked);
        }

        private void viewToolStripMenuItem_DropDownOpening(object sender, EventArgs e)
        {
            if (m_Camera.IsOpened)
            {
                uc480.Defines.Status statusRet;

                // update selected ropeffect
                uc480.Defines.RopEffectMode ropMode;
                statusRet = m_Camera.RopEffect.Get(out ropMode);

                toolStripMenuItemMirrorRightLeft.Checked = (ropMode & uc480.Defines.RopEffectMode.LeftRight) == uc480.Defines.RopEffectMode.LeftRight;
                toolStripMenuItemMirrorUpDown.Checked = (ropMode & uc480.Defines.RopEffectMode.UpDown) == uc480.Defines.RopEffectMode.UpDown;

                uc480.Defines.DisplayMode displayMode;
                statusRet = m_Camera.Display.Mode.Get(out displayMode);

                // directrenderer
                toolStripMenuItemTimestamp.Enabled = displayMode == uc480.Defines.DisplayMode.DiB;
            }
        }

        private void toolStripMenuItemCrosshair_Click(object sender, EventArgs e)
        {
            uc480.Defines.Status statusRet;

            uc480.Defines.DisplayMode mode;
            statusRet = m_Camera.Display.Mode.Get(out mode);

            // directrenderer
            if (mode != uc480.Defines.DisplayMode.DiB)
            {
                statusRet = m_Camera.DirectRenderer.SetWindowHandle(pictureBoxDisplay.Handle.ToInt32());
                statusRet = m_Camera.DirectRenderer.Overlay.SetVisible(toolStripMenuItemCrosshair.Checked);

                Graphics graphics;
                statusRet = m_Camera.DirectRenderer.Overlay.GetGraphics(out graphics);

                if (statusRet == uc480.Defines.Status.SUCCESS && graphics != null)
                {
                    DoDrawing(ref graphics, 0);
                    statusRet = m_Camera.DirectRenderer.Overlay.SetGraphics(ref graphics);
                }
            }
        }

        private void toolStripMenuItemFreerun_Click(object sender, EventArgs e)
        {
            uc480.Defines.Status statusRet;
            if (toolStripMenuItemFreerun.Checked)
            {
                // start capture
                statusRet = m_Camera.Acquisition.Capture();
                m_IsLive = true;
            }
            else
            {
                statusRet = m_Camera.Acquisition.Stop();
                m_IsLive = false;
            }

            UpdateToolbar();
        }

        private void toolStripButtonVideoRec_Click(object sender, EventArgs e)
        {
            uc480.Defines.Status statusRet = uc480.Defines.Status.SUCCESS;

            if (toolStripButtonVideoRec.Checked)
            {
                SaveFileDialog fileDialog = new SaveFileDialog();
                fileDialog.Filter = "Video file (*.avi)|*.avi";
                fileDialog.DefaultExt = "avi";
                if (fileDialog.ShowDialog() == DialogResult.OK)
                {
                    statusRet = m_Camera.Video.Start(fileDialog.FileName);
                    if (statusRet != uc480.Defines.Status.SUCCESS)
                    {
                        MessageBox.Show("Could not start video recording");
                        toolStripButtonVideoRec.Checked = false;
                    }
                }
                else
                {
                    toolStripButtonVideoRec.Checked = false;
                }
            }
            else
            {
                statusRet = m_Camera.Video.Stop();
            }
        }
    }
}
