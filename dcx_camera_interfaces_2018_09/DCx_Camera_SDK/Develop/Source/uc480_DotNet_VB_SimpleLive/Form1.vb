
Public Class uEye_DotNet_Simple_Live

    Dim WithEvents camera As New uc480.Camera()
    Dim DisplayHandle As IntPtr
    Dim bLive As Boolean


    Private Sub uEye_DotNet_Simple_Live_Load(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles MyBase.Load

        DisplayHandle = DisplayWindow.Handle
        CameraInit()

    End Sub


    Private Sub EventHandler() Handles camera.EventFrame

        'Event
        Dim s32MemID As Int32
        camera.Memory.GetActive(s32MemID)
        camera.Display.Render(s32MemID, DisplayHandle, uc480.Defines.DisplayRenderMode.FitToWindow)

    End Sub


    Private Sub CameraInit()

        Dim s32MemID As Int32
        Dim statusRet As uc480.Defines.Status

        'Open Camera
        statusRet = camera.Init()
        If (statusRet <> uc480.Defines.Status.Success) Then

            MessageBox.Show("Camera initializing failed")

        End If

        'Allocate Memory
        statusRet = camera.Memory.Allocate(s32MemID, True)
        If (statusRet <> uc480.Defines.Status.Success) Then

            MessageBox.Show("Allocate Memory failed")

        End If

        'Start Live
        statusRet = camera.Acquisition.Capture()
        If (statusRet <> uc480.Defines.Status.Success) Then

            MessageBox.Show("Start Live Video failed")

        Else

            bLive = True

        End If

        CB_Auto_Gain_Balance.Enabled = camera.AutoFeatures.Software.Gain.Supported
        CB_Auto_White_Balance.Enabled = camera.AutoFeatures.Software.WhiteBalance.Supported

    End Sub


    Private Sub Button_Live_Video_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles Button_Live_Video.Click

        'Start Live Video
        If (camera.Acquisition.Capture() = uc480.Defines.Status.Success) Then

            bLive = True

        End If

    End Sub


    Private Sub Button_Stop_Video_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles Button_Stop_Video.Click

        'Stop Live Video
        If (camera.Acquisition.Stop() = uc480.Defines.Status.Success) Then

            bLive = False

        End If

    End Sub


    Private Sub Button_Freeze_Video_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles Button_Freeze_Video.Click

        'Freeze Video
        If (camera.Acquisition.Freeze() = uc480.Defines.Status.Success) Then

            bLive = False

        End If

    End Sub


    Private Sub Button_Load_Parameter_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles Button_Load_Parameter.Click

        'Load Parameters
        camera.Acquisition.Stop()
        camera.Parameter.Load("")

        If (bLive = True) Then

            If (camera.Acquisition.Capture() = uc480.Defines.Status.Success) Then

                bLive = True

            End If

        End If

    End Sub


    Private Sub CB_Auto_White_Balance_CheckedChanged(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles CB_Auto_White_Balance.CheckedChanged

        'CheckBox Auto White Balance
        camera.AutoFeatures.Software.WhiteBalance.SetEnable(CB_Auto_White_Balance.Checked)

    End Sub


    Private Sub CB_Auto_Gain_Balance_CheckedChanged(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles CB_Auto_Gain_Balance.CheckedChanged

        'CheckBox Auto Gain Balance
        camera.AutoFeatures.Software.Gain.SetEnable(CB_Auto_Gain_Balance.Checked)

    End Sub


    Private Sub Button_Exit_Prog_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles Button_Exit_Prog.Click

        'Close the Camera
        camera.Exit()

        'Program Exit
        Close()

    End Sub

End Class
