using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace DotnetCockpit
{
    public class IControl : UserControl
    {
        protected uc480.Camera m_Camera;

        public IControl()
        {

        }

        public IControl(uc480.Camera camera)
        {
            m_Camera = camera;
        }

        public void SetCameraObject(uc480.Camera camera)
        {
            m_Camera = camera;
        }

        public virtual void OnControlFocusActive()
        {
        }

        public virtual void OnControlFocusLost()
        {
        }
    }
}
