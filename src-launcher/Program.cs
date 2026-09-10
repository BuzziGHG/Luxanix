using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Net.Sockets;
using System.Threading;
using System.Windows.Forms;

namespace LuxanixLauncher
{
    public class LauncherForm : Form
    {
        private Label lblTitle;
        private Label lblSubtitle;
        private Label lblStatus;
        private ProgressBar progressBar;
        private Button btnOpenBrowser;
        private Button btnExit;
        private NotifyIcon trayIcon;
        private Process serverProcess = null;
        private string appDir;

        public LauncherForm()
        {
            appDir = AppDomain.CurrentDomain.BaseDirectory;
            InitializeComponent();
            StartApplicationFlow();
        }

        private void InitializeComponent()
        {
            this.Text = "Luxanix Studio — RTX Video Remaster";
            this.Size = new Size(540, 310);
            this.StartPosition = FormStartPosition.CenterScreen;
            this.FormBorderStyle = FormBorderStyle.FixedDialog;
            this.MaximizeBox = false;
            this.BackColor = Color.FromArgb(11, 13, 16);
            this.ForeColor = Color.FromArgb(228, 233, 240);

            lblTitle = new Label()
            {
                Text = "⚡ LUXANIX STUDIO",
                Font = new Font("Segoe UI", 16, FontStyle.Bold),
                ForeColor = Color.FromArgb(118, 185, 0),
                Location = new Point(24, 20),
                AutoSize = true
            };

            lblSubtitle = new Label()
            {
                Text = "AI Raytracing & Photorealistic Video Remaster (NVIDIA RTX)",
                Font = new Font("Segoe UI", 9.5f, FontStyle.Regular),
                ForeColor = Color.FromArgb(148, 163, 184),
                Location = new Point(26, 52),
                AutoSize = true
            };

            lblStatus = new Label()
            {
                Text = "Initialisiere GPU-Umgebung...",
                Font = new Font("Segoe UI", 10, FontStyle.Regular),
                ForeColor = Color.FromArgb(200, 210, 220),
                Location = new Point(26, 95),
                Size = new Size(470, 45)
            };

            progressBar = new ProgressBar()
            {
                Location = new Point(26, 145),
                Size = new Size(470, 24),
                Style = ProgressBarStyle.Marquee,
                MarqueeAnimationSpeed = 30
            };

            btnOpenBrowser = new Button()
            {
                Text = "🌐 Studio im Browser öffnen",
                Font = new Font("Segoe UI", 10, FontStyle.Bold),
                BackColor = Color.FromArgb(118, 185, 0),
                ForeColor = Color.Black,
                FlatStyle = FlatStyle.Flat,
                Location = new Point(26, 195),
                Size = new Size(250, 42),
                Visible = false,
                Cursor = Cursors.Hand
            };
            btnOpenBrowser.FlatAppearance.BorderSize = 0;
            btnOpenBrowser.Click += (s, e) => OpenBrowser();

            btnExit = new Button()
            {
                Text = "Beenden",
                Font = new Font("Segoe UI", 9.5f),
                BackColor = Color.FromArgb(30, 36, 45),
                ForeColor = Color.FromArgb(220, 225, 230),
                FlatStyle = FlatStyle.Flat,
                Location = new Point(366, 195),
                Size = new Size(130, 42),
                Cursor = Cursors.Hand
            };
            btnExit.FlatAppearance.BorderSize = 0;
            btnExit.Click += (s, e) => this.Close();

            trayIcon = new NotifyIcon()
            {
                Text = "Luxanix Studio",
                Visible = false
            };
            try
            {
                trayIcon.Icon = SystemIcons.Application;
            }
            catch { }

            trayIcon.DoubleClick += (s, e) =>
            {
                this.Show();
                this.WindowState = FormWindowState.Normal;
            };

            this.Controls.Add(lblTitle);
            this.Controls.Add(lblSubtitle);
            this.Controls.Add(lblStatus);
            this.Controls.Add(progressBar);
            this.Controls.Add(btnOpenBrowser);
            this.Controls.Add(btnExit);

            this.FormClosing += LauncherForm_FormClosing;
        }

        private void StartApplicationFlow()
        {
            Thread t = new Thread(() =>
            {
                try
                {
                    string venvPython = Path.Combine(appDir, ".venv", "Scripts", "python.exe");
                    string appPy = Path.Combine(appDir, "ui", "app.py");

                    if (!File.Exists(venvPython))
                    {
                        UpdateStatus("Ersteinrichtung erforderlich... Starte Installation (Python + PyTorch CUDA)...");
                        string installBat = Path.Combine(appDir, "install.bat");
                        if (File.Exists(installBat))
                        {
                            ProcessStartInfo psiInst = new ProcessStartInfo("cmd.exe", "/c \"" + installBat + "\"")
                            {
                                WorkingDirectory = appDir,
                                UseShellExecute = true
                            };
                            Process pInst = Process.Start(psiInst);
                            pInst.WaitForExit();
                        }
                    }

                    if (!File.Exists(venvPython) || !File.Exists(appPy))
                    {
                        UpdateStatus("Fehler: Python-Umgebung konnte nicht gefunden werden.");
                        return;
                    }

                    UpdateStatus("⚡ Starte Luxanix Studio (NVIDIA CUDA Pipeline)...");

                    ProcessStartInfo psi = new ProcessStartInfo(venvPython, "ui\\app.py")
                    {
                        WorkingDirectory = appDir,
                        CreateNoWindow = true,
                        UseShellExecute = false
                    };
                    serverProcess = Process.Start(psi);

                    // Wait for server port 7860 to become available
                    bool ready = false;
                    for (int i = 0; i < 40; i++)
                    {
                        if (serverProcess.HasExited) break;
                        if (IsPortOpen("127.0.0.1", 7860, 500))
                        {
                            ready = true;
                            break;
                        }
                        Thread.Sleep(500);
                    }

                    if (ready)
                    {
                        UpdateStatus("✅ Luxanix Studio läuft auf http://127.0.0.1:7860");
                        this.Invoke((MethodInvoker)delegate
                        {
                            progressBar.Visible = false;
                            btnOpenBrowser.Visible = true;
                        });
                        OpenBrowser();
                    }
                    else
                    {
                        UpdateStatus("Server gestartet. Öffne Browser...");
                        OpenBrowser();
                    }
                }
                catch (Exception ex)
                {
                    UpdateStatus("Fehler: " + ex.Message);
                }
            });
            t.IsBackground = true;
            t.Start();
        }

        private bool IsPortOpen(string host, int port, int timeoutMs)
        {
            try
            {
                using (var client = new TcpClient())
                {
                    var result = client.BeginConnect(host, port, null, null);
                    var success = result.AsyncWaitHandle.WaitOne(timeoutMs);
                    if (!success) return false;
                    client.EndConnect(result);
                    return true;
                }
            }
            catch
            {
                return false;
            }
        }

        private void UpdateStatus(string text)
        {
            if (this.IsDisposed) return;
            if (this.InvokeRequired)
            {
                this.Invoke((MethodInvoker)delegate { UpdateStatus(text); });
                return;
            }
            lblStatus.Text = text;
        }

        private void OpenBrowser()
        {
            try
            {
                Process.Start(new ProcessStartInfo("http://127.0.0.1:7860") { UseShellExecute = true });
            }
            catch { }
        }

        private void LauncherForm_FormClosing(object sender, FormClosingEventArgs e)
        {
            try
            {
                if (serverProcess != null && !serverProcess.HasExited)
                {
                    // Kill python and child processes
                    ProcessStartInfo psi = new ProcessStartInfo("taskkill", "/F /T /PID " + serverProcess.Id)
                    {
                        CreateNoWindow = true,
                        UseShellExecute = false
                    };
                    Process pKill = Process.Start(psi);
                    if (pKill != null) pKill.WaitForExit(1000);
                }
            }
            catch { }
        }

        [STAThread]
        public static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new LauncherForm());
        }
    }
}
