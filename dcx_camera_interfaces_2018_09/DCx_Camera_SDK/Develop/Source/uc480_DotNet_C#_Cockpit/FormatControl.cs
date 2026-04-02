using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace DotnetCockpit
{
    public partial class FormatControl : IControl
    {
        public event EventHandler<EventArgs> DisplayChanged;

        public FormatControl(uc480.Camera camera) 
            : base(camera)
        {
            InitializeComponent();
        }

        public FormatControl()
        {
            InitializeComponent();
        }

        public override void OnControlFocusActive()
        {
            InitPixelformat();
            UpdateDisplay();
        }

        public override void OnControlFocusLost()
        {
        }

        private void tabControlFormat_SelectedIndexChanged(object sender, EventArgs e)
        {
            switch (tabControlFormat.SelectedIndex)
            {
                case 0:
                    InitPixelformat();
                    break;
                case 1:
                    UpdateDisplay();
                    break;
            }
        }

        private void InitPixelformat()
        {
            uc480.Defines.Status statusRet;
            uc480.Types.SensorInfo sensorInfo;

            uc480.Defines.DisplayMode displayMode;
            m_Camera.Display.Mode.Get(out displayMode);

            foreach (Control control in this.tabPagePixelformat.Controls)
            {
                control.Enabled = displayMode == uc480.Defines.DisplayMode.DiB;
            }

            if (displayMode != uc480.Defines.DisplayMode.DiB)
            {
                MessageBox.Show("Please change the display mode to DiB to change the pixel format");
                return;
            }

            statusRet = m_Camera.Information.GetSensorInfo(out sensorInfo);

            if (sensorInfo.SensorColorMode == uc480.Defines.SensorColorMode.Monochrome)
            {
                radioButtonGray8Normal.Enabled = true;
                radioButtonGray8High.Enabled = false;

                radioButtonGray12Normal.Enabled = true;
                radioButtonGray12High.Enabled = false;

                radioButtonGray16Normal.Enabled = true;
                radioButtonGray16High.Enabled = false;
            }
            else if (sensorInfo.SensorColorMode == uc480.Defines.SensorColorMode.Bayer)
            {
                radioButtonGray8Normal.Enabled = isColorModeSupported(uc480.Defines.ColorMode.Mono8, uc480.Defines.ColorConvertMode.Software3X3);
                radioButtonGray8High.Enabled = isColorModeSupported(uc480.Defines.ColorMode.Mono8, uc480.Defines.ColorConvertMode.Software5X5);
                radioButtonGray8Hardware.Enabled = isColorModeSupported(uc480.Defines.ColorMode.Mono8, uc480.Defines.ColorConvertMode.Hardware3X3);

                radioButtonGray12Normal.Enabled = isColorModeSupported(uc480.Defines.ColorMode.Mono12, uc480.Defines.ColorConvertMode.Software3X3);
                radioButtonGray12High.Enabled = isColorModeSupported(uc480.Defines.ColorMode.Mono12, uc480.Defines.ColorConvertMode.Software5X5);
                radioButtonGray12Hardware.Enabled = isColorModeSupported(uc480.Defines.ColorMode.Mono12, uc480.Defines.ColorConvertMode.Hardware3X3);

                radioButtonGray16Normal.Enabled = isColorModeSupported(uc480.Defines.ColorMode.Mono16, uc480.Defines.ColorConvertMode.Software3X3);
                radioButtonGray16High.Enabled = isColorModeSupported(uc480.Defines.ColorMode.Mono16, uc480.Defines.ColorConvertMode.Software5X5);
                radioButtonGray16Hardware.Enabled = isColorModeSupported(uc480.Defines.ColorMode.Mono16, uc480.Defines.ColorConvertMode.Hardware3X3);
            }

            radioButtonRaw8Normal.Enabled = isColorModeSupported(uc480.Defines.ColorMode.SensorRaw8, uc480.Defines.ColorConvertMode.Software3X3);
            radioButtonRaw8High.Enabled = isColorModeSupported(uc480.Defines.ColorMode.SensorRaw8, uc480.Defines.ColorConvertMode.Software5X5);
            radioButtonGray8Hardware.Enabled = isColorModeSupported(uc480.Defines.ColorMode.SensorRaw8, uc480.Defines.ColorConvertMode.Hardware3X3);

            radioButtonRaw12Normal.Enabled = isColorModeSupported(uc480.Defines.ColorMode.SensorRaw12, uc480.Defines.ColorConvertMode.Software3X3);
            radioButtonRaw12High.Enabled = isColorModeSupported(uc480.Defines.ColorMode.SensorRaw12, uc480.Defines.ColorConvertMode.Software5X5);
            radioButtonRaw12Hardware.Enabled = isColorModeSupported(uc480.Defines.ColorMode.SensorRaw12, uc480.Defines.ColorConvertMode.Hardware3X3);

            radioButtonRaw16Normal.Enabled = isColorModeSupported(uc480.Defines.ColorMode.SensorRaw16, uc480.Defines.ColorConvertMode.Software3X3);
            radioButtonRaw16High.Enabled = isColorModeSupported(uc480.Defines.ColorMode.SensorRaw16, uc480.Defines.ColorConvertMode.Software5X5);
            radioButtonRaw16Hardware.Enabled = isColorModeSupported(uc480.Defines.ColorMode.SensorRaw16, uc480.Defines.ColorConvertMode.Hardware3X3);

            radioButtonRGB24Normal.Enabled = isColorModeSupported(uc480.Defines.ColorMode.RGB8Packed, uc480.Defines.ColorConvertMode.Software3X3);
            radioButtonRGB24High.Enabled = isColorModeSupported(uc480.Defines.ColorMode.RGB8Packed, uc480.Defines.ColorConvertMode.Software5X5);
            radioButtonRGB24Hardware.Enabled = isColorModeSupported(uc480.Defines.ColorMode.RGB8Packed, uc480.Defines.ColorConvertMode.Hardware3X3);

            radioButtonRGB32Normal.Enabled = isColorModeSupported(uc480.Defines.ColorMode.RGBA8Packed, uc480.Defines.ColorConvertMode.Software3X3);
            radioButtonRGB32High.Enabled = isColorModeSupported(uc480.Defines.ColorMode.RGBA8Packed, uc480.Defines.ColorConvertMode.Software5X5);
            radioButtonRGB32Hardware.Enabled = isColorModeSupported(uc480.Defines.ColorMode.RGBA8Packed, uc480.Defines.ColorConvertMode.Hardware3X3);

            // select active
            uc480.Defines.ColorMode colorMode;
            uc480.Defines.ColorConvertMode convertMode;

            statusRet = m_Camera.PixelFormat.Get(out colorMode);
            statusRet = m_Camera.Color.Converter.Get(colorMode, out convertMode);


            radioButtonGray8Normal.Checked = colorMode == uc480.Defines.ColorMode.Mono8 && convertMode == uc480.Defines.ColorConvertMode.Software3X3;
            radioButtonGray8High.Checked = colorMode == uc480.Defines.ColorMode.Mono8 && convertMode == uc480.Defines.ColorConvertMode.Software5X5;
            radioButtonGray8Hardware.Checked = colorMode == uc480.Defines.ColorMode.Mono8 && convertMode == uc480.Defines.ColorConvertMode.Hardware3X3;

            radioButtonGray12Normal.Checked = colorMode == uc480.Defines.ColorMode.Mono12 && convertMode == uc480.Defines.ColorConvertMode.Software3X3;
            radioButtonGray12High.Checked = colorMode == uc480.Defines.ColorMode.Mono12 && convertMode == uc480.Defines.ColorConvertMode.Software5X5;
            radioButtonGray12Hardware.Checked = colorMode == uc480.Defines.ColorMode.Mono12 && convertMode == uc480.Defines.ColorConvertMode.Hardware3X3;

            radioButtonGray16Normal.Checked = colorMode == uc480.Defines.ColorMode.Mono16 && convertMode == uc480.Defines.ColorConvertMode.Software3X3;
            radioButtonGray16High.Checked = colorMode == uc480.Defines.ColorMode.Mono16 && convertMode == uc480.Defines.ColorConvertMode.Software5X5;
            radioButtonGray16Hardware.Checked = colorMode == uc480.Defines.ColorMode.Mono16 && convertMode == uc480.Defines.ColorConvertMode.Hardware3X3;


            radioButtonRaw8Normal.Checked = colorMode == uc480.Defines.ColorMode.SensorRaw8 && convertMode == uc480.Defines.ColorConvertMode.Software3X3;
            radioButtonRaw8High.Checked = colorMode == uc480.Defines.ColorMode.SensorRaw8 && convertMode == uc480.Defines.ColorConvertMode.Software5X5;
            radioButtonRaw8Hardware.Checked = colorMode == uc480.Defines.ColorMode.SensorRaw8 && convertMode == uc480.Defines.ColorConvertMode.Hardware3X3;

            radioButtonRaw12Normal.Checked = colorMode == uc480.Defines.ColorMode.SensorRaw12 && convertMode == uc480.Defines.ColorConvertMode.Software3X3;
            radioButtonRaw12High.Checked = colorMode == uc480.Defines.ColorMode.SensorRaw12 && convertMode == uc480.Defines.ColorConvertMode.Software5X5;
            radioButtonRaw12Hardware.Checked = colorMode == uc480.Defines.ColorMode.SensorRaw12 && convertMode == uc480.Defines.ColorConvertMode.Hardware3X3;

            radioButtonRaw16Normal.Checked = colorMode == uc480.Defines.ColorMode.SensorRaw16 && convertMode == uc480.Defines.ColorConvertMode.Software3X3;
            radioButtonRaw16High.Checked = colorMode == uc480.Defines.ColorMode.SensorRaw16 && convertMode == uc480.Defines.ColorConvertMode.Software5X5;
            radioButtonRaw16Hardware.Checked = colorMode == uc480.Defines.ColorMode.SensorRaw16 && convertMode == uc480.Defines.ColorConvertMode.Hardware3X3;


            radioButtonRGB24Normal.Checked = colorMode == uc480.Defines.ColorMode.BGR8Packed && convertMode == uc480.Defines.ColorConvertMode.Software3X3;
            radioButtonRGB24High.Checked = colorMode == uc480.Defines.ColorMode.BGR8Packed && convertMode == uc480.Defines.ColorConvertMode.Software5X5;
            radioButtonRGB24Hardware.Checked = colorMode == uc480.Defines.ColorMode.BGR8Packed && convertMode == uc480.Defines.ColorConvertMode.Hardware3X3;

            radioButtonRGB32Normal.Checked = colorMode == uc480.Defines.ColorMode.BGRA8Packed && convertMode == uc480.Defines.ColorConvertMode.Software3X3;
            radioButtonRGB32High.Checked = colorMode == uc480.Defines.ColorMode.BGRA8Packed && convertMode == uc480.Defines.ColorConvertMode.Software5X5;
            radioButtonRGB32Hardware.Checked = colorMode == uc480.Defines.ColorMode.BGRA8Packed && convertMode == uc480.Defines.ColorConvertMode.Hardware3X3;

        }

        private Boolean isColorModeSupported(uc480.Defines.ColorMode colorMode, uc480.Defines.ColorConvertMode colorConvertMode)
        {
            uc480.Defines.ColorConvertMode converterMode = 0;
            uc480.Defines.Status statusRet;

            statusRet = m_Camera.Color.Converter.GetSupported(colorMode, out converterMode);
            if (statusRet == uc480.Defines.Status.SUCCESS && converterMode != uc480.Defines.ColorConvertMode.None)
            {
                if (uc480.Defines.ColorConvertMode.Software3X3 == colorConvertMode)
                {
                    return (converterMode & uc480.Defines.ColorConvertMode.Software3X3) != uc480.Defines.ColorConvertMode.None ? true : false;
                }
                if (uc480.Defines.ColorConvertMode.Software5X5 == colorConvertMode)
                {
                    return (converterMode & uc480.Defines.ColorConvertMode.Software5X5) != uc480.Defines.ColorConvertMode.None ? true : false;
                }
                if (uc480.Defines.ColorConvertMode.Hardware3X3 == colorConvertMode)
                {
                    return (converterMode & uc480.Defines.ColorConvertMode.Hardware3X3) != uc480.Defines.ColorConvertMode.None ? true : false;
                }
            }

            return false;
        }

        private void onPixelFormat_CheckedChanged(object sender, EventArgs e)
        {
            uc480.Defines.ColorMode colorMode = 0;
            uc480.Defines.ColorConvertMode convertMode = 0;

            if (radioButtonRaw8Normal.Checked)
            {
                colorMode = uc480.Defines.ColorMode.SensorRaw8;
                convertMode = uc480.Defines.ColorConvertMode.Software3X3;
            }
            else if (radioButtonRaw8High.Checked)
            {
                colorMode = uc480.Defines.ColorMode.SensorRaw8;
                convertMode = uc480.Defines.ColorConvertMode.Software5X5;
            }
            else if (radioButtonRaw8Hardware.Checked)
            {
                colorMode = uc480.Defines.ColorMode.SensorRaw8;
                convertMode = uc480.Defines.ColorConvertMode.Hardware3X3;
            }
            else if (radioButtonRaw12Normal.Checked)
            {
                colorMode = uc480.Defines.ColorMode.SensorRaw12;
                convertMode = uc480.Defines.ColorConvertMode.Software3X3;
            }
            else if (radioButtonRaw12High.Checked)
            {
                colorMode = uc480.Defines.ColorMode.SensorRaw12;
                convertMode = uc480.Defines.ColorConvertMode.Software5X5;
            }
            else if (radioButtonRaw12Hardware.Checked)
            {
                colorMode = uc480.Defines.ColorMode.SensorRaw12;
                convertMode = uc480.Defines.ColorConvertMode.Hardware3X3;
            }
            else if (radioButtonRaw16Normal.Checked)
            {
                colorMode = uc480.Defines.ColorMode.SensorRaw16;
                convertMode = uc480.Defines.ColorConvertMode.Software3X3;
            }
            else if (radioButtonRaw16High.Checked)
            {
                colorMode = uc480.Defines.ColorMode.SensorRaw16;
                convertMode = uc480.Defines.ColorConvertMode.Software5X5;
            }
            else if (radioButtonRaw16Hardware.Checked)
            {
                colorMode = uc480.Defines.ColorMode.SensorRaw16;
                convertMode = uc480.Defines.ColorConvertMode.Hardware3X3;
            }
            else if (radioButtonGray8Normal.Checked)
            {
                colorMode = uc480.Defines.ColorMode.Mono8;
                convertMode = uc480.Defines.ColorConvertMode.Software3X3;
            }
            else if (radioButtonGray8High.Checked)
            {
                colorMode = uc480.Defines.ColorMode.Mono8;
                convertMode = uc480.Defines.ColorConvertMode.Software5X5;
            }
            else if (radioButtonGray8Hardware.Checked)
            {
                colorMode = uc480.Defines.ColorMode.Mono8;
                convertMode = uc480.Defines.ColorConvertMode.Hardware3X3;
            }
            else if (radioButtonGray12Normal.Checked)
            {
                colorMode = uc480.Defines.ColorMode.Mono12;
                convertMode = uc480.Defines.ColorConvertMode.Software3X3;
            }
            else if (radioButtonGray12High.Checked)
            {
                colorMode = uc480.Defines.ColorMode.Mono12;
                convertMode = uc480.Defines.ColorConvertMode.Software5X5;
            }
            else if (radioButtonGray12Hardware.Checked)
            {
                colorMode = uc480.Defines.ColorMode.Mono12;
                convertMode = uc480.Defines.ColorConvertMode.Hardware3X3;
            }
            else if (radioButtonGray16Normal.Checked)
            {
                colorMode = uc480.Defines.ColorMode.Mono16;
                convertMode = uc480.Defines.ColorConvertMode.Software3X3;
            }
            else if (radioButtonGray16High.Checked)
            {
                colorMode = uc480.Defines.ColorMode.Mono16;
                convertMode = uc480.Defines.ColorConvertMode.Software5X5;
            }
            else if (radioButtonGray16Hardware.Checked)
            {
                colorMode = uc480.Defines.ColorMode.Mono16;
                convertMode = uc480.Defines.ColorConvertMode.Hardware3X3;
            }
            else if (radioButtonRGB24Normal.Checked)
            {
                colorMode = uc480.Defines.ColorMode.BGR8Packed;
                convertMode = uc480.Defines.ColorConvertMode.Software3X3;
            }
            else if (radioButtonRGB24High.Checked)
            {
                colorMode = uc480.Defines.ColorMode.BGR8Packed;
                convertMode = uc480.Defines.ColorConvertMode.Software5X5;
            }
            else if (radioButtonRGB24Hardware.Checked)
            {
                colorMode = uc480.Defines.ColorMode.BGR8Packed;
                convertMode = uc480.Defines.ColorConvertMode.Hardware3X3;
            }
            else if (radioButtonRGB32Normal.Checked)
            {
                colorMode = uc480.Defines.ColorMode.BGRA8Packed;
                convertMode = uc480.Defines.ColorConvertMode.Software3X3;
            }
            else if (radioButtonRGB32High.Checked)
            {
                colorMode = uc480.Defines.ColorMode.BGRA8Packed;
                convertMode = uc480.Defines.ColorConvertMode.Software5X5;
            }
            else if (radioButtonRGB32Hardware.Checked)
            {
                colorMode = uc480.Defines.ColorMode.BGRA8Packed;
                convertMode = uc480.Defines.ColorConvertMode.Hardware3X3;
            }

            uc480.Defines.Status statusRet;

            Boolean isLive;
            statusRet = m_Camera.Acquisition.HasStarted(out isLive);

            if (isLive)
            {
                statusRet = m_Camera.Acquisition.Stop(uc480.Defines.DeviceParameter.Wait);
            }

            statusRet = m_Camera.PixelFormat.Set(colorMode);
            statusRet = m_Camera.Color.Converter.Set(colorMode, convertMode);

            // memory reallocation
            Int32[] memList;
            statusRet = m_Camera.Memory.GetList(out memList);
            statusRet = m_Camera.Memory.Free(memList);
            statusRet = m_Camera.Memory.Allocate();

            if (isLive)
            {
                statusRet = m_Camera.Acquisition.Capture();
            }
        }

        private void UpdateDisplay()
        {
            uc480.Defines.DisplayMode displayMode;

            // check supported
            m_Camera.DirectRenderer.GetSupported(out displayMode);

            radioButtonDisplayOpenGL.Enabled = (displayMode & uc480.Defines.DisplayMode.OpenGL) == uc480.Defines.DisplayMode.OpenGL;
            radioButtonDisplayDirect3D.Enabled = (displayMode & uc480.Defines.DisplayMode.Direct3D) == uc480.Defines.DisplayMode.Direct3D;

            m_Camera.Display.Mode.Get(out displayMode);

            switch (displayMode)
            {
                case uc480.Defines.DisplayMode.DiB:
                    radioButtonDisplayDIB.Checked = true;
                    break;
                case uc480.Defines.DisplayMode.Direct3D:
                    radioButtonDisplayDirect3D.Checked = true;
                    break;
                case uc480.Defines.DisplayMode.OpenGL:
                    radioButtonDisplayOpenGL.Checked = true;
                    break;
            }
        }

        private void radioButtonDisplayDIB_CheckedChanged(object sender, EventArgs e)
        {
            if (radioButtonDisplayDIB.Checked && radioButtonDisplayDIB.Focused)
            {
                uc480.Defines.Status statusRet;
                uc480.Defines.DisplayMode displayMode = uc480.Defines.DisplayMode.DiB;

                statusRet = m_Camera.Display.Mode.Set(displayMode);

                // inform our main form
                if (DisplayChanged != null)
                {
                    DisplayChanged(this, EventArgs.Empty);
                }
            }
        }

        private void radioButtonDisplayDirect3D_CheckedChanged(object sender, EventArgs e)
        {
            if (radioButtonDisplayDirect3D.Checked && radioButtonDisplayDirect3D.Focused)
            {
                uc480.Defines.Status statusRet;

                statusRet = m_Camera.Display.Mode.Set(uc480.Defines.DisplayMode.Direct3D);

                if (statusRet != uc480.Defines.Status.SUCCESS)
                {
                    MessageBox.Show("Setting display mode failed");
                    radioButtonDisplayDIB.Checked = true;
                }
                else
                {
                    // inform our main form
                    if (DisplayChanged != null)
                    {
                        DisplayChanged(this, EventArgs.Empty);
                    }
                }
            }
        }

        private void radioButtonDisplayOpenGL_CheckedChanged(object sender, EventArgs e)
        {
            if (radioButtonDisplayOpenGL.Checked && radioButtonDisplayOpenGL.Focused)
            {
                uc480.Defines.Status statusRet;

                statusRet = m_Camera.Display.Mode.Set(uc480.Defines.DisplayMode.OpenGL);

                if (statusRet != uc480.Defines.Status.SUCCESS)
                {
                    MessageBox.Show("Setting display mode failed");
                    radioButtonDisplayDIB.Checked = true;
                }
                else
                {
                    // inform our main form
                    if (DisplayChanged != null)
                    {
                        DisplayChanged(this, EventArgs.Empty);
                    }
                }
            }
        }
    }
}
