//==========================================================================//
//                                                                          //
//  Copyright (C) 2004 - 2014                                               //
//  OEM uc480		                                                        //
//                                                                          //
//  The information in this document is subject to change without           //
//  notice and should not be construed as a commitment.						//
//  We do not assume any responsibility for any errors						//
//  that may appear in this document.                                       //
//                                                                          //
//  This document, or source code, is provided solely as an example         //
//  of how to utilize uc480 libraries in a sample application.			    //
//  We do not assume any responsibility for the use or						//
//  reliability of any portion of this document or the described software.  //
//                                                                          //
//  General permission to copy or modify, but not for profit, is hereby     //
//  granted,  provided that the above copyright notice is included and      //
//  reference made to the fact that reproduction privileges were granted.	//
//                                                                          //
//==========================================================================//

// uc480AcquireDlg.cpp : implementation file
//

#include "stdafx.h"
#include "uc480Acquire.h"
#include "uc480AcquireDlg.h"
#include ".\uc480acquiredlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CAboutDlg dialog used for App About

class CAboutDlg : public CDialog
{
public:
	CAboutDlg();

// Dialog Data
	//{{AFX_DATA(CAboutDlg)
	enum { IDD = IDD_ABOUTBOX };
	//}}AFX_DATA

	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CAboutDlg)
	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
	//}}AFX_VIRTUAL

// Implementation
protected:
	//{{AFX_MSG(CAboutDlg)
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialog(CAboutDlg::IDD)
{
	//{{AFX_DATA_INIT(CAboutDlg)
	//}}AFX_DATA_INIT
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(CAboutDlg)
	//}}AFX_DATA_MAP
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialog)
	//{{AFX_MSG_MAP(CAboutDlg)
		// No message handlers
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// Cuc480AcquireDlg dialog

Cuc480AcquireDlg::Cuc480AcquireDlg(CWnd* pParent /*=NULL*/)
	: CDialog(Cuc480AcquireDlg::IDD, pParent)
{
	//{{AFX_DATA_INIT(Cuc480AcquireDlg)
		// NOTE: the ClassWizard will add member initialization here
	//}}AFX_DATA_INIT
	// Note that LoadIcon does not require a subsequent DestroyIcon in Win32
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void Cuc480AcquireDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(Cuc480AcquireDlg)
		// NOTE: the ClassWizard will add DDX and DDV calls here
	//}}AFX_DATA_MAP
}

BEGIN_MESSAGE_MAP(Cuc480AcquireDlg, CDialog)
	//{{AFX_MSG_MAP(Cuc480AcquireDlg)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON_EXIT, OnButtonExit)
	ON_BN_CLICKED(IDC_BUTTON_ACQUIRE, OnButtonAcquire)
	ON_BN_CLICKED(IDC_BUTTON_LOAD_PARAMETER, OnBnClickedButtonLoadParameter)
	//}}AFX_MSG_MAP
    ON_WM_CLOSE()
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// Cuc480AcquireDlg message handlers

BOOL Cuc480AcquireDlg::OnInitDialog()
{
	CDialog::OnInitDialog();

	// Add "About..." menu item to system menu.

	// IDM_ABOUTBOX must be in the system command range.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		CString strAboutMenu;
		strAboutMenu.LoadString(IDS_ABOUTBOX);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// Set the icon for this dialog.  The framework does this automatically
	//  when the application's main window is not a dialog
	SetIcon(m_hIcon, TRUE);			// Set big icon
	//SetIcon(m_hIcon, FALSE);		// Set small icon

	m_hG = 0;
	

	m_hWnd = GetDlgItem( IDC_DISPLAY )->m_hWnd; // set display window handle
	RECT rc;									// get gemoetry of display window
	::GetClientRect(m_hWnd, &rc);
	m_nSizeX = 640;		//rc.right - rc.left;	// set video width  to fit into display window
	m_nSizeY = 480;		//rc.bottom - rc.top;	// set video height to fit into display window

	OpenCamera();		// open a camera handle

	return true;

}

void Cuc480AcquireDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialog::OnSysCommand(nID, lParam);
	}
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void Cuc480AcquireDlg::OnPaint() 
{
	if (IsIconic())
	{
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, (WPARAM) dc.GetSafeHdc(), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Draw the icon
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialog::OnPaint();
	}
}

// The system calls this to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR Cuc480AcquireDlg::OnQueryDragIcon()
{
	return (HCURSOR) m_hIcon;
}




///////////////////////////////////////////////////////////////////////////////
//
// METHOD Cuc480AcquireDlg::OnButtonAcquire() 
//
// DESCRIPTION: - acquire a single frame
//				- display frame
//
///////////////////////////////////////////////////////////////////////////////
void Cuc480AcquireDlg::OnButtonAcquire() 
{
	if ( m_hG == 0 )
		OpenCamera();

	if ( m_hG != 0 )
	{
		is_FreezeVideo( m_hG, IS_WAIT );
		is_RenderBitmap( m_hG, m_lMemoryId, m_hWnd, IS_RENDER_FIT_TO_WINDOW );
	}
}




///////////////////////////////////////////////////////////////////////////////
//
// METHOD Cuc480AcquireDlg::OnBnClickedButtonLoadParameter() 
//
// DESCRIPTION: - loads parameters from an ini file
//				- reallocates the memory
//
///////////////////////////////////////////////////////////////////////////////
void Cuc480AcquireDlg::OnBnClickedButtonLoadParameter()
{
	if ( m_hG == 0 )
		OpenCamera();

	if ( m_hG != 0 )
	{
		if( is_LoadParameters( m_hG, NULL ) == IS_SUCCESS )
		{
			// realloc image mem with actual sizes and depth.
			is_FreeImageMem( m_hG, m_pcImageMemory, m_lMemoryId );
			m_nSizeX = is_SetImageSize( m_hG, IS_GET_IMAGE_SIZE_X, 0 );
			m_nSizeY = is_SetImageSize( m_hG, IS_GET_IMAGE_SIZE_Y, 0 );
			switch( is_SetColorMode( m_hG, IS_GET_COLOR_MODE ) )
			{
			case IS_SET_CM_RGB32:
				m_nBitsPerPixel = 32;
				break;
			case IS_SET_CM_RGB24:
				m_nBitsPerPixel = 24;
				break;
			case IS_SET_CM_RGB16:
			case IS_SET_CM_RGB15:
			case IS_SET_CM_UYVY:
				m_nBitsPerPixel = 16;
				break;
			case IS_SET_CM_Y8:
			case IS_SET_CM_BAYER:
			default:
				m_nBitsPerPixel = 8;
				break;
			}

			// memory initialization
			is_AllocImageMem( m_hG,
							m_nSizeX,
							m_nSizeY,
							m_nBitsPerPixel,
							&m_pcImageMemory,
							&m_lMemoryId);
			is_SetImageMem(m_hG, m_pcImageMemory, m_lMemoryId );	// set memory active

			// display initialization
			is_SetImageSize(m_hG, m_nSizeX, m_nSizeY );
		}
	}
}



///////////////////////////////////////////////////////////////////////////////
//
// METHOD Cuc480AcquireDlg::OnButtonExit() 
//
// DESCRIPTION: - exit the camera
//				- quit application
//
///////////////////////////////////////////////////////////////////////////////
void Cuc480AcquireDlg::OnButtonExit() 
{
	if( m_hG != 0 )
	{
		//free old image mem.
		is_FreeImageMem( m_hG, m_pcImageMemory, m_lMemoryId );
		is_ExitCamera( m_hG );
        m_hG = NULL;
	}
    PostQuitMessage(0);
}






///////////////////////////////////////////////////////////////////////////////
//
// METHOD Cuc480AcquireDlg::OpenCamera() 
//
// DESCRIPTION: - Opens a handle to a connected camera
//
///////////////////////////////////////////////////////////////////////////////
bool Cuc480AcquireDlg::OpenCamera()
{
	if ( m_hG != 0 )
    {
		//free old image mem.
		is_FreeImageMem( m_hG, m_pcImageMemory, m_lMemoryId );
		is_ExitCamera( m_hG );
    }

	// init camera
	m_hG = (HCAM) 0;							// open next camera
	m_Ret = is_InitCamera( &m_hG, NULL );		// init camera - no window handle for live required
	
	if( m_Ret == IS_SUCCESS )
	{
		// retrieve original image size
		SENSORINFO sInfo;
		is_GetSensorInfo( m_hG, &sInfo );

        GetMaxImageSize(&m_nSizeX, &m_nSizeY);

		// setup the color depth to the current windows setting
		is_GetColorDepth(m_hG, &m_nBitsPerPixel, &m_nColorMode);
		is_SetColorMode(m_hG, m_nColorMode);

		// memory initialization
		 is_AllocImageMem(	m_hG,
							m_nSizeX,
							m_nSizeY,
							m_nBitsPerPixel,
							&m_pcImageMemory,
							&m_lMemoryId);
		 is_SetImageMem(m_hG, m_pcImageMemory, m_lMemoryId );	// set memory active


		// display initialization
		is_SetImageSize(m_hG, m_nSizeX, m_nSizeY );
		is_SetDisplayMode(m_hG, IS_SET_DM_DIB);

		/*
		// enable the dialog based error report
		m_Ret = is_SetErrorReport(m_hG, IS_ENABLE_ERR_REP); //IS_DISABLE_ERR_REP);
		if( m_Ret != IS_SUCCESS )
		{
			AfxMessageBox( "ERROR: Can not enable the automatic error report!" , MB_ICONEXCLAMATION, 0 );
			return false;
		}
		*/
	}
  else
  {
	AfxMessageBox( "ERROR: Can not open a camera!" , MB_ICONEXCLAMATION, 0 );
	return false;
  }

  return true;
}



void Cuc480AcquireDlg::OnClose()
{
	if( m_hG != 0 )
	{
		//free old image mem.
		is_FreeImageMem( m_hG, m_pcImageMemory, m_lMemoryId );
		is_ExitCamera( m_hG );
        m_hG = NULL;
	}

    CDialog::OnClose();
}


void Cuc480AcquireDlg::GetMaxImageSize(INT *pnSizeX, INT *pnSizeY)
{
    // Check if the camera supports an arbitrary AOI
    INT nAOISupported = 0;
    BOOL bAOISupported = TRUE;
    if (is_ImageFormat(m_hG,
                       IMGFRMT_CMD_GET_ARBITRARY_AOI_SUPPORTED, 
                       (void*)&nAOISupported, 
                       sizeof(nAOISupported)) == IS_SUCCESS)
    {
        bAOISupported = (nAOISupported != 0);
    }

    if (bAOISupported)
    {
        // Get maximum image size
	    SENSORINFO sInfo;
	    is_GetSensorInfo (m_hG, &sInfo);
	    *pnSizeX = sInfo.nMaxWidth;
	    *pnSizeY = sInfo.nMaxHeight;
    }
    else
    {
        // Get image size of the current format
        *pnSizeX = is_SetImageSize(m_hG, IS_GET_IMAGE_SIZE_X, 0);
        *pnSizeY = is_SetImageSize(m_hG, IS_GET_IMAGE_SIZE_Y, 0);
    }
}