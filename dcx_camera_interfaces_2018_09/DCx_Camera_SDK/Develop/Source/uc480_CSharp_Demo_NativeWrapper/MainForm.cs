//==============================================================================//
//                                                                              //
//	C# - Interfaceclass for uc480 Camera family								    //
//                                                                              //
//  Copyright (C) 2010                                                          //
//	Thorlabs GmbH									                            //
//  Hans-Boeckler-Str. 6                                                        //
//  D-85221 Dachau                                                              //
//                                                                              //
//  The information in this document is subject to change without               //
//  notice and should not be construed as a commitment by Thorlabs GmbH.        //
//  Thorlabs GmbH does not assume any responsibility for any errors             //
//  that may appear in this document.                                           //
//                                                                              //
//  This document, or source code, is provided solely as an example             //
//  of how to utilize Thorlabs software libraries in a sample application.      //
//  Thorlabs GmbH does not assume any responsibility for the use or             //
//  reliability of any portion of this document or the described software.      //
//                                                                              //
//  General permission to copy or modify, but not for profit, is hereby         //
//  granted,  provided that the above copyright notice is included and          //
//  included and reference made to the fact that reproduction privileges        //
//  were granted by Thorlabs GmbH.                                              //
//                                                                              //
//  Thorlabs cannot assume any responsibility for the use, or misuse, of any    //
//  portion or misuse, of any portion of this software for other than its       //
//  intended diagnostic purpose in calibrating and testing Thorlabs manufactured//
//  image processing boards and software.                                       //
//                                                                              //
//==============================================================================//


using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;
using System.Runtime.InteropServices;


namespace uc480_CSharp_Demo
{
	/// <summary>
	/// Zusammenfassung für Form1.
	/// </summary>
	public class MainForm : System.Windows.Forms.Form
	{
		private System.Windows.Forms.Button btnInit;
		private System.Windows.Forms.Button btnExit;
		private System.Windows.Forms.PictureBox DisplayWindow;
		/// <summary>
		/// Erforderliche Designervariable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		private uc480		m_uc480;
		private bool		m_bLive;
		private bool		m_bDrawing;
		private int			m_RenderMode;

		// uc480 images
		private const int IMAGE_COUNT	= 4;
		private struct UC480IMAGE
		{
			public IntPtr pMemory;
			public int MemID;
			public int nSeqNum;
		}
		private UC480IMAGE[] m_Uc480Images;
		private IntPtr m_pCurMem;

		private System.Windows.Forms.Button btnFreeze;
		private System.Windows.Forms.PictureBox pictureBox1;
		private System.Windows.Forms.GroupBox ImageAcquisition;
		private System.Windows.Forms.Button btnQuit;
		private System.Windows.Forms.Button btnLive;
		private System.Windows.Forms.ListBox listBoxInfo;
		private System.Windows.Forms.RadioButton radioNormal;
		private System.Windows.Forms.RadioButton radioFitToWindow;
		private System.Windows.Forms.GroupBox groupBox1;
		private ToolTip toolTipColor = new ToolTip();


		public MainForm()
		{
			// dialog controls
			InitializeComponent();

			// init variables
			m_bLive = false;
			m_bDrawing = false;
			m_RenderMode = uc480.IS_RENDER_NORMAL;

            // init our uc480 object
			m_uc480 = new uc480();

			// enable static messages ( no open camera is needed )		
			m_uc480.EnableMessage( uc480.IS_NEW_DEVICE, this.Handle.ToInt32() );
			m_uc480.EnableMessage( uc480.IS_DEVICE_REMOVAL, this.Handle.ToInt32() );
						
			// Set up the delays for the ToolTip.
			toolTipColor.AutoPopDelay = 5000;
			toolTipColor.InitialDelay = 1000;
			toolTipColor.ReshowDelay = 500;
			toolTipColor.ShowAlways = true;
			toolTipColor.SetToolTip(this.DisplayWindow, "");

			// update listbox info
			UpdateInfos();

            // init our image struct and alloc marshall pointers for the uc480 memory
			m_Uc480Images = new UC480IMAGE[IMAGE_COUNT];
			int nLoop = 0;
			for (nLoop = 0; nLoop < IMAGE_COUNT; nLoop++)
			{
				m_Uc480Images[nLoop].pMemory = Marshal.AllocCoTaskMem(4);	// create marshal object pointers
				m_Uc480Images[nLoop].MemID	= 0;
				m_Uc480Images[nLoop].nSeqNum	= 0;
			}	
		}

		/// <summary>
		/// Die verwendeten Ressourcen bereinigen.
		/// </summary>
		protected override void Dispose( bool disposing )
		{
			if( disposing )
			{
				if (components != null) 
				{
					components.Dispose();
				}
			}
			base.Dispose( disposing );
		}

		#region Vom Windows Form-Designer generierter Code
		/// <summary>
		/// Erforderliche Methode für die Designerunterstützung. 
		/// Der Inhalt der Methode darf nicht mit dem Code-Editor geändert werden.
		/// </summary>
		private void InitializeComponent()
		{
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(MainForm));
            this.btnInit = new System.Windows.Forms.Button();
            this.btnExit = new System.Windows.Forms.Button();
            this.btnFreeze = new System.Windows.Forms.Button();
            this.ImageAcquisition = new System.Windows.Forms.GroupBox();
            this.btnLive = new System.Windows.Forms.Button();
            this.btnQuit = new System.Windows.Forms.Button();
            this.listBoxInfo = new System.Windows.Forms.ListBox();
            this.radioNormal = new System.Windows.Forms.RadioButton();
            this.radioFitToWindow = new System.Windows.Forms.RadioButton();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.pictureBox1 = new System.Windows.Forms.PictureBox();
            this.DisplayWindow = new System.Windows.Forms.PictureBox();
            this.ImageAcquisition.SuspendLayout();
            this.groupBox1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.DisplayWindow)).BeginInit();
            this.SuspendLayout();
            // 
            // btnInit
            // 
            this.btnInit.Location = new System.Drawing.Point(8, 16);
            this.btnInit.Name = "btnInit";
            this.btnInit.Size = new System.Drawing.Size(152, 32);
            this.btnInit.TabIndex = 3;
            this.btnInit.Text = "Init camera";
            this.btnInit.Click += new System.EventHandler(this.btnInit_Click);
            // 
            // btnExit
            // 
            this.btnExit.Enabled = false;
            this.btnExit.Location = new System.Drawing.Point(8, 56);
            this.btnExit.Name = "btnExit";
            this.btnExit.Size = new System.Drawing.Size(152, 32);
            this.btnExit.TabIndex = 2;
            this.btnExit.Text = "Exit camera";
            this.btnExit.Click += new System.EventHandler(this.btnExit_Click);
            // 
            // btnFreeze
            // 
            this.btnFreeze.Enabled = false;
            this.btnFreeze.Location = new System.Drawing.Point(16, 24);
            this.btnFreeze.Name = "btnFreeze";
            this.btnFreeze.Size = new System.Drawing.Size(120, 32);
            this.btnFreeze.TabIndex = 6;
            this.btnFreeze.Text = "FreezeVideo";
            this.btnFreeze.Click += new System.EventHandler(this.btnFreeze_Click);
            // 
            // ImageAcquisition
            // 
            this.ImageAcquisition.Controls.Add(this.btnLive);
            this.ImageAcquisition.Controls.Add(this.btnFreeze);
            this.ImageAcquisition.Location = new System.Drawing.Point(8, 96);
            this.ImageAcquisition.Name = "ImageAcquisition";
            this.ImageAcquisition.Size = new System.Drawing.Size(152, 104);
            this.ImageAcquisition.TabIndex = 8;
            this.ImageAcquisition.TabStop = false;
            this.ImageAcquisition.Text = "Image Acquisition";
            // 
            // btnLive
            // 
            this.btnLive.Enabled = false;
            this.btnLive.Location = new System.Drawing.Point(16, 64);
            this.btnLive.Name = "btnLive";
            this.btnLive.Size = new System.Drawing.Size(120, 32);
            this.btnLive.TabIndex = 7;
            this.btnLive.Text = "Live video";
            this.btnLive.Click += new System.EventHandler(this.btnLive_Click);
            // 
            // btnQuit
            // 
            this.btnQuit.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.btnQuit.Location = new System.Drawing.Point(8, 456);
            this.btnQuit.Name = "btnQuit";
            this.btnQuit.Size = new System.Drawing.Size(152, 32);
            this.btnQuit.TabIndex = 9;
            this.btnQuit.Text = "Exit";
            this.btnQuit.Click += new System.EventHandler(this.btnQuit_Click);
            // 
            // listBoxInfo
            // 
            this.listBoxInfo.Location = new System.Drawing.Point(8, 288);
            this.listBoxInfo.Name = "listBoxInfo";
            this.listBoxInfo.Size = new System.Drawing.Size(152, 160);
            this.listBoxInfo.TabIndex = 10;
            // 
            // radioNormal
            // 
            this.radioNormal.Checked = true;
            this.radioNormal.Location = new System.Drawing.Point(16, 24);
            this.radioNormal.Name = "radioNormal";
            this.radioNormal.Size = new System.Drawing.Size(104, 16);
            this.radioNormal.TabIndex = 11;
            this.radioNormal.TabStop = true;
            this.radioNormal.Text = "Normal";
            this.radioNormal.CheckedChanged += new System.EventHandler(this.radioNormal_CheckedChanged);
            // 
            // radioFitToWindow
            // 
            this.radioFitToWindow.Location = new System.Drawing.Point(16, 40);
            this.radioFitToWindow.Name = "radioFitToWindow";
            this.radioFitToWindow.Size = new System.Drawing.Size(104, 16);
            this.radioFitToWindow.TabIndex = 12;
            this.radioFitToWindow.Text = "Fit to window";
            this.radioFitToWindow.CheckedChanged += new System.EventHandler(this.radioFitToWindow_CheckedChanged);
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.radioNormal);
            this.groupBox1.Controls.Add(this.radioFitToWindow);
            this.groupBox1.Location = new System.Drawing.Point(8, 208);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(152, 72);
            this.groupBox1.TabIndex = 13;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Rendermode";
            // 
            // pictureBox1
            // 
            this.pictureBox1.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.pictureBox1.BackColor = System.Drawing.SystemColors.Window;
            this.pictureBox1.Image = global::uc480_CSharp_Demo.Properties.Resources.Thorlabs_Logo;
            this.pictureBox1.Location = new System.Drawing.Point(8, 496);
            this.pictureBox1.Name = "pictureBox1";
            this.pictureBox1.Size = new System.Drawing.Size(152, 57);
            this.pictureBox1.SizeMode = System.Windows.Forms.PictureBoxSizeMode.CenterImage;
            this.pictureBox1.TabIndex = 7;
            this.pictureBox1.TabStop = false;
            // 
            // DisplayWindow
            // 
            this.DisplayWindow.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom)
                        | System.Windows.Forms.AnchorStyles.Left)
                        | System.Windows.Forms.AnchorStyles.Right)));
            this.DisplayWindow.BackColor = System.Drawing.SystemColors.Window;
            this.DisplayWindow.BorderStyle = System.Windows.Forms.BorderStyle.Fixed3D;
            this.DisplayWindow.Location = new System.Drawing.Point(168, 0);
            this.DisplayWindow.Name = "DisplayWindow";
            this.DisplayWindow.Size = new System.Drawing.Size(720, 558);
            this.DisplayWindow.TabIndex = 5;
            this.DisplayWindow.TabStop = false;
            this.DisplayWindow.MouseMove += new System.Windows.Forms.MouseEventHandler(this.DisplayWindow_MouseMove);
            this.DisplayWindow.MouseHover += new System.EventHandler(this.DisplayWindow_MouseHover);
            this.DisplayWindow.Paint += new System.Windows.Forms.PaintEventHandler(this.DisplayWindow_Paint);
            // 
            // MainForm
            // 
            this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
            this.ClientSize = new System.Drawing.Size(888, 558);
            this.Controls.Add(this.groupBox1);
            this.Controls.Add(this.listBoxInfo);
            this.Controls.Add(this.btnQuit);
            this.Controls.Add(this.ImageAcquisition);
            this.Controls.Add(this.pictureBox1);
            this.Controls.Add(this.DisplayWindow);
            this.Controls.Add(this.btnInit);
            this.Controls.Add(this.btnExit);
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.Name = "MainForm";
            this.Text = "uc480 CSharp demo";
            this.Paint += new System.Windows.Forms.PaintEventHandler(this.MainForm_Paint);
            this.Closing += new System.ComponentModel.CancelEventHandler(this.MainForm_Closing);
            this.MouseHover += new System.EventHandler(this.MainForm_MouseHover);
            this.ImageAcquisition.ResumeLayout(false);
            this.groupBox1.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.DisplayWindow)).EndInit();
            this.ResumeLayout(false);

		}
		#endregion

		/// <summary>
		/// Der Haupteinstiegspunkt für die Anwendung.
		/// </summary>
		[STAThread]
		static void Main() 
		{
			Application.Run(new MainForm());
		}


		// -----------------  DrawImage  -------------------------
		//
		void DrawImage()
		{
			m_bDrawing = true;
			// draw current memory if a camera is opened
			if ( m_uc480.IsOpen() )
			{
				int num = 0;
				IntPtr pMem = new IntPtr();
				IntPtr pLast = new IntPtr();
				m_uc480.GetActSeqBuf( ref num, ref pMem, ref pLast );
				if ( pLast.ToInt32() == 0 )
				{
					m_bDrawing = false;
					return;
				}

				int nLastID = GetImageID( pLast );
				int nLastNum = GetImageNum( pLast );
				m_uc480.LockSeqBuf( nLastNum, pLast );

				m_pCurMem = pLast;		// remember current buffer for our tootip ctrl

				m_uc480.RenderBitmap( nLastID, DisplayWindow.Handle.ToInt32(), m_RenderMode );

				m_uc480.UnlockSeqBuf( nLastNum, pLast );
			}
			m_bDrawing = false;
		}


		// -----------------  UpdateInfos  -------------------------
		//
		private void UpdateInfos()
		{
			listBoxInfo.Items.Clear();

			int ver = uc480.GetDLLVersion();
			string strText;
            strText = String.Format("uc480 SDK Version: {0}.{1}.{2}", (ver >> 24), (ver >> 16 & 0xff), (ver & 0xffff));
			listBoxInfo.Items.Add(strText);

			int nrOfCameras = 0;
			uc480.GetNumberOfCameras( ref nrOfCameras );
			strText = String.Format("Connected cameras: {0}", nrOfCameras );
			listBoxInfo.Items.Add(strText);

			// camera infos
			if ( m_uc480.IsOpen() )
			{
				listBoxInfo.Items.Add("");
				
				// Sensorinfo
				uc480.SENSORINFO sensorInfo = new uc480.SENSORINFO();
				m_uc480.GetSensorInfo( ref sensorInfo );
				listBoxInfo.Items.Add("Sensor: " + sensorInfo.strSensorName);

				// Camerainfo
				uc480.CAMINFO cameraInfo = new uc480.CAMINFO();
				m_uc480.GetCameraInfo( ref cameraInfo );
				listBoxInfo.Items.Add("CameraInfo:");
				listBoxInfo.Items.Add("   SerNo: " + cameraInfo.SerNo);
				listBoxInfo.Items.Add("   Date: " + cameraInfo.Date);
				listBoxInfo.Items.Add("   Version: " + cameraInfo.Version);
				strText = String.Format("   Camera ID: {0}", cameraInfo.id );
				listBoxInfo.Items.Add(strText);

				// Memory board query
				listBoxInfo.Items.Add("");
				strText = m_uc480.IsMemoryBoardConnected() ? "Memoryboard connected" : "No Memoryboard connected";
				listBoxInfo.Items.Add(strText);
			}
		}

		// ------------------------  GetImageID -------------------------------
		//
		int GetImageID( IntPtr pBuffer )
		{
			// get image id for a given memory
			if ( !m_uc480.IsOpen() )
				return 0;

			int i = 0;
			for ( i=0; i<IMAGE_COUNT; i++)
				if ( m_Uc480Images[i].pMemory == pBuffer )
					return m_Uc480Images[i].MemID;
			return 0;
		}
		
		
		// ------------------------  GetImageNum -------------------------------
		//
		int GetImageNum( IntPtr pBuffer )
		{
			// get number of sequence for a given memory
			if ( !m_uc480.IsOpen() )
				return 0;

			int i = 0;
			for ( i=0; i<IMAGE_COUNT; i++)
				if ( m_Uc480Images[i].pMemory == pBuffer )
					return m_Uc480Images[i].nSeqNum;

			return 0;
		}


		// ------------------------  btnInit_Click -------------------------------
		//
		private void btnInit_Click(object sender, System.EventArgs e)
		{
			// if opened before, close now
			if ( m_uc480.IsOpen() )
			{
				m_uc480.ExitCamera();
			}

			// open a camera
			int nRet = m_uc480.InitCamera (0, DisplayWindow.Handle.ToInt32());
			if (nRet == uc480.IS_STARTER_FW_UPLOAD_NEEDED)
			{
				/************************************************************************************************/
				/*                                                                                              */
				/*  If the camera returns with "IS_STARTER_FW_UPLOAD_NEEDED", an upload of a new firmware       */
				/*  is necessary. This upload can take several seconds. We recommend to check the required      */
				/*  time with the function is_GetDuration().                                                    */
				/*                                                                                              */
				/*  In this case, the camera can only be opened if the flag "IS_ALLOW_STARTER_FW_UPLOAD"        */ 
				/*  is "OR"-ed to m_hCam. This flag allows an automatic upload of the firmware.                 */
				/*                                                                                              */                        
				/************************************************************************************************/
				
				uint nUploadTime = 25000;
				m_uc480.GetDuration (uc480.IS_SE_STARTER_FW_UPLOAD, ref nUploadTime);

				String Str;
				Str = "This camera requires a new firmware. The upload will take about " + nUploadTime / 1000 + " seconds. Please wait ...";
                MessageBox.Show(Str, "uc480");

				nRet = m_uc480.InitCamera (0 | uc480.IS_ALLOW_STARTER_FW_UPLOAD, DisplayWindow.Handle.ToInt32());
			}
			
			if (nRet != uc480.IS_SUCCESS)
			{
                MessageBox.Show("Init failed", "uc480");
				return;
			}

            uc480.SENSORINFO sensorInfo = new uc480.SENSORINFO();
            m_uc480.GetSensorInfo(ref sensorInfo);

            // Set the image size
            int x = 0;
            int y = 0;
            unsafe
            {
                int nAOISupported = -1;
                IntPtr pnAOISupported = (IntPtr)((uint*)&nAOISupported);
                bool bAOISupported = true;

                // check if an arbitrary AOI is supported
                //if (m_uc480.ImageFormat(uc480.IMGFRMT_CMD_GET_ARBITRARY_AOI_SUPPORTED, pnAOISupported, 4) == uc480.IS_SUCCESS)
                //{
                //    bAOISupported = (nAOISupported != 0);
                //}

                // If an arbitrary AOI is supported -> take maximum sensor size
                if (bAOISupported)
                {
                    x = sensorInfo.nMaxWidth;
                    y = sensorInfo.nMaxHeight;
                }
                // Take the image size of the current image format
                else
                {
                    x = m_uc480.SetImageSize(uc480.IS_GET_IMAGE_SIZE_X, 0);
                    y = m_uc480.SetImageSize(uc480.IS_GET_IMAGE_SIZE_Y, 0);
                }

                m_uc480.SetImageSize(x, y);
            }
            
           	// alloc images
			m_uc480.ClearSequence();
			int nLoop = 0;
			for (nLoop = 0; nLoop < IMAGE_COUNT; nLoop++)
			{
				// alloc memory
				m_uc480.AllocImageMem(x, y, 32, ref m_Uc480Images[nLoop].pMemory, ref m_Uc480Images[nLoop].MemID );
				// add our memory to the sequence
				m_uc480.AddToSequence( m_Uc480Images[nLoop].pMemory, m_Uc480Images[nLoop].MemID );
				// set sequence number
				m_Uc480Images[nLoop].nSeqNum	= nLoop + 1;
			}

			m_uc480.SetColorMode( uc480.IS_SET_CM_RGB32 );
			m_uc480.EnableMessage( uc480.IS_FRAME, this.Handle.ToInt32() );

			btnInit.Enabled		= false;
			btnExit.Enabled		= true;
			btnLive.Enabled		= true;
			btnFreeze.Enabled	= true;
			
			UpdateInfos();

			// free image
			if ( DisplayWindow.Image != null )
			{
				DisplayWindow.Image.Dispose();
				DisplayWindow.Image = null;
			}

			// capture a single image
			m_uc480.FreezeVideo(uc480.IS_WAIT);
			Refresh();
		}

		// ------------------------  btnExit_Click -------------------------------
		//
		private void btnExit_Click(object sender, System.EventArgs e)
		{
			btnInit.Enabled		= true;
			btnExit.Enabled		= false;
			btnFreeze.Enabled	= false;
			btnLive.Enabled		= false;
			m_bLive				= false;
			btnLive.Text		= "Start Live";
			
			// release marshal object pointers
			int nLoop = 0;
			for (nLoop = 0; nLoop < IMAGE_COUNT; nLoop++)
				Marshal.FreeCoTaskMem(m_Uc480Images[nLoop].pMemory);
			m_uc480.ExitCamera();
			
			Refresh();

			UpdateInfos();
		}


		// ------------------------  Start Live Video -------------------------------
		//
		private void btnLive_Click(object sender, System.EventArgs e)
		{
			if ( m_bLive )
			{
				if ( m_uc480.StopLiveVideo( uc480.IS_WAIT ) == uc480.IS_SUCCESS )
					m_bLive = false;
				else
					MessageBox.Show("Capture Video failed!");
			}
			else
			{
				
				if ( m_uc480.CaptureVideo( uc480.IS_WAIT ) == uc480.IS_SUCCESS )
					m_bLive = true;
				else
					MessageBox.Show("Capture Video failed!");
			}

			// set controls
			btnFreeze.Enabled = !m_bLive;
			btnLive.Text = m_bLive ?  "Stop Live" : "Start Live";
		}


		// ------------------------  btnFreeze_Click  -------------------------------
		//
		private void btnFreeze_Click(object sender, System.EventArgs e)
		{
			// capture a single image
			if ( m_uc480.FreezeVideo(uc480.IS_WAIT) != uc480.IS_SUCCESS )
				MessageBox.Show("Error freeze image");
		}


		// ------------------------  Quit application  -------------------------------
		//
		private void btnQuit_Click(object sender, System.EventArgs e)
		{
			Close();
		}




		// ------------------------  MainForm_Closing  -------------------------------
		//
		private void MainForm_Closing(object sender, System.ComponentModel.CancelEventArgs e)
		{
			if (m_uc480.IsOpen())
			{
				// release marshal object pointers
				int nLoop = 0;
				for (nLoop = 0; nLoop < IMAGE_COUNT; nLoop++)
					Marshal.FreeCoTaskMem(m_Uc480Images[nLoop].pMemory);
				m_uc480.ExitCamera();
			}
		}



		// ------------------------  DisplayWindow_MouseMove  -------------------------------
		//
		private void DisplayWindow_MouseMove(object sender, System.Windows.Forms.MouseEventArgs e)
		{
			// if there is a open camera, show tooltip ctrl
			if ( m_uc480.IsOpen() )
			{
				string strText;
				int width = 0, height = 0, bitspp = 0, pitch = 0, bytespp = 0;
				m_uc480.InquireImageMem( m_pCurMem, GetImageID(m_pCurMem), ref width, ref height, ref bitspp, ref pitch );
				bytespp = (bitspp+1)/8;

				int pos = pitch*e.Y + bytespp*e.X;
				if ( pos < pitch * height && e.X < width )
				{			
					int b = Marshal.ReadByte( m_pCurMem, pos );
					int g = Marshal.ReadByte( m_pCurMem, pos + 1);
					int r = Marshal.ReadByte( m_pCurMem, pos + 2);
					strText = String.Format(" R:{0} \n G:{1} \n B:{2} ", r, g, b);
					toolTipColor.SetToolTip(this.DisplayWindow, strText);			
				}
				else
					toolTipColor.SetToolTip(DisplayWindow, "");

			}
			else
				toolTipColor.SetToolTip(DisplayWindow, "");
		}

		// ------------------------  DisplayWindow_MouseHover  -------------------------------
		//
		private void DisplayWindow_MouseHover(object sender, System.EventArgs e)
		{
			toolTipColor.SetToolTip(DisplayWindow, "");
		}


		// ------------------------  MainForm_MouseHover  -------------------------------
		//
		private void MainForm_MouseHover(object sender, System.EventArgs e)
		{
			toolTipColor.SetToolTip(DisplayWindow, "");
		}
	
		
		// ------------------------  DisplayWindow_Paint  -------------------------------
		//
		private void DisplayWindow_Paint(object sender, System.Windows.Forms.PaintEventArgs e)
		{
			if ( !m_bDrawing )
				DrawImage();
		}


		// ------------------------  MainForm_Paint  -------------------------------
		//
		private void MainForm_Paint(object sender, System.Windows.Forms.PaintEventArgs e)
		{
		}


		// ------------------------  WndProc  -------------------------------
		//
		[System.Security.Permissions.PermissionSet(System.Security.Permissions.SecurityAction.Demand, Name="FullTrust")]
		protected override void WndProc(ref Message m) 
		{
			// Listen for operating system messages
			switch (m.Msg)
			{
                // uc480 Message
				case uc480.IS_UC480_MESSAGE:
					HandleUc480Message( m.WParam.ToInt32(), m.LParam.ToInt32() );
					break;                
			}
			base.WndProc(ref m);
		}

        // ------------------------  HandleUc480Message  -------------------------------
		//
		void HandleUc480Message( int wParam, int lParam )
		{
			switch (wParam)
			{
				case uc480.IS_FRAME:
					if ( !m_bDrawing )
						DrawImage();
					break;

				case uc480.IS_DEVICE_REMOVAL:
				case uc480.IS_NEW_DEVICE:
					UpdateInfos();
					break;
			}
		}


        // ------------------------  HandleUc480Message  -------------------------------
		//
		private void radioNormal_CheckedChanged(object sender, System.EventArgs e)
		{
			// set render mode
			m_RenderMode = uc480.IS_RENDER_NORMAL;
			Refresh();
		}


        // ------------------------  HandleUc480Message  -------------------------------
		//
		private void radioFitToWindow_CheckedChanged(object sender, System.EventArgs e)
		{
			// set render mode
			m_RenderMode = uc480.IS_RENDER_FIT_TO_WINDOW;
		}


	}
}
