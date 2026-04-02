// uc480Acquire.h : main header file for the uc480Acquire application
//

#if !defined(AFX_uc480Acquire_H__37478BE3_A3F6_4D4B_8CD0_BE0712A58367__INCLUDED_)
#define AFX_uc480Acquire_H__37478BE3_A3F6_4D4B_8CD0_BE0712A58367__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#ifndef __AFXWIN_H__
	#error include 'stdafx.h' before including this file for PCH
#endif

#include "resource.h"		// main symbols

/////////////////////////////////////////////////////////////////////////////
// Cuc480AcquireApp:
// See uc480Acquire.cpp for the implementation of this class
//

class Cuc480AcquireApp : public CWinApp
{
public:
	Cuc480AcquireApp();

// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(Cuc480AcquireApp)
	public:
	virtual BOOL InitInstance();
	//}}AFX_VIRTUAL

// Implementation

	//{{AFX_MSG(Cuc480AcquireApp)
		// NOTE - the ClassWizard will add and remove member functions here.
		//    DO NOT EDIT what you see in these blocks of generated code !
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};


/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_uc480Acquire_H__37478BE3_A3F6_4D4B_8CD0_BE0712A58367__INCLUDED_)
