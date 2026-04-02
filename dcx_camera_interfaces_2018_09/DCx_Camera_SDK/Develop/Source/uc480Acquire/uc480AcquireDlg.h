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

// uc480AcquireDlg.h : header file
//

#if !defined(AFX_UC480ACQUIRE_H__478B93A8_52DA_4AEA_A524_E79E9C2B6601__INCLUDED_)
#define AFX_UC480ACQUIRE_H__478B93A8_52DA_4AEA_A524_E79E9C2B6601__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "uc480.h"
#include "uc480_deprecated.h"

/////////////////////////////////////////////////////////////////////////////
// Cuc480AcquireDlg dialog

class Cuc480AcquireDlg : public CDialog
{
// Construction
public:
	Cuc480AcquireDlg(CWnd* pParent = NULL);	// standard constructor

// Dialog Data
	//{{AFX_DATA(Cuc480AcquireDlg)
	enum { IDD = IDD_UC480ACQUIRE_DIALOG };
		// NOTE: the ClassWizard will add data members here
	//}}AFX_DATA

	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(Cuc480AcquireDlg)
	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support
	//}}AFX_VIRTUAL

// Implementation
protected:
	HICON	m_hIcon;

	// camera varibles
	HCAM	m_hG;			// handle to frame grabber
	HWND	m_hWnd;			// handle to diplay window
	INT		m_Ret;			// return value for SDK functions
	INT		m_nColorMode;	// Y8/RGB16/RGB24/REG32
	INT		m_nBitsPerPixel;// number of bits needed store one pixel
	INT		m_nSizeX;		// width of video 
	INT		m_nSizeY;		// height of video
	INT		m_lMemoryId;	// grabber memory - buffer ID
	char*	m_pcImageMemory;// grabber memory - pointer to buffer


	bool OpenCamera();
    void GetMaxImageSize(INT *pnSizeX, INT *pnSizeY);

	// Generated message map functions
	//{{AFX_MSG(Cuc480AcquireDlg)
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	afx_msg void OnButtonExit();
	afx_msg void OnButtonAcquire();
	afx_msg void OnBnClickedButtonLoadParameter();
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
public:
    afx_msg void OnClose();
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_UC480ACQUIRE_H__478B93A8_52DA_4AEA_A524_E79E9C2B6601__INCLUDED_)
