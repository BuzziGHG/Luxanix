using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.IO.Compression;
using System.Net;
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
                Font = new Font("Segoe UI", 9.5f, FontStyle.Regular),
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
                    // 1. Determine working directory
                    string targetDir = appDir;
                    string appPy = Path.Combine(targetDir, "ui", "app.py");

                    if (!File.Exists(appPy))
                    {
                        string userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
                        string scratchDir = Path.Combine(userProfile, ".gemini", "antigravity", "scratch", "SimRTX-Studio");
                        string localAppDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Luxanix");

                        if (File.Exists(Path.Combine(scratchDir, "ui", "app.py")) && File.Exists(Path.Combine(scratchDir, ".venv", "Scripts", "python.exe")))
                        {
                            targetDir = scratchDir;
                            appPy = Path.Combine(targetDir, "ui", "app.py");
                        }
                        else if (File.Exists(Path.Combine(localAppDir, "ui", "app.py")))
                        {
                            targetDir = localAppDir;
                            appPy = Path.Combine(targetDir, "ui", "app.py");
                        }
                        else
                        {
                            targetDir = localAppDir;
                            appPy = Path.Combine(targetDir, "ui", "app.py");
                        }
                    }

                    // 2. Download application files from GitHub if not found
                    if (!File.Exists(appPy))
                    {
                        UpdateStatus("📥 Lade Luxanix Studio Dateien von GitHub herunter...");
                        if (!Directory.Exists(targetDir))
                        {
                            Directory.CreateDirectory(targetDir);
                        }

                        ServicePointManager.SecurityProtocol = (SecurityProtocolType)3072 | (SecurityProtocolType)768 | (SecurityProtocolType)192;

                        string zipPath = Path.Combine(Path.GetTempPath(), "Luxanix-main.zip");
                        using (WebClient client = new WebClient())
                        {
                            client.Headers.Add("User-Agent", "Luxanix-Installer");
                            client.DownloadFile("https://github.com/BuzziGHG/Luxanix/archive/refs/heads/main.zip", zipPath);
                        }

                        UpdateStatus("📦 Entpacke Anwendungsdateien...");
                        string extractTemp = Path.Combine(Path.GetTempPath(), "Luxanix_Extract_" + Guid.NewGuid().ToString("N"));
                        ZipFile.ExtractToDirectory(zipPath, extractTemp);

                        string sourceDir = Path.Combine(extractTemp, "Luxanix-main");
                        if (!Directory.Exists(sourceDir))
                        {
                            string[] subDirs = Directory.GetDirectories(extractTemp);
                            if (subDirs.Length > 0) sourceDir = subDirs[0];
                        }

                        CopyDirectory(sourceDir, targetDir);

                        try { File.Delete(zipPath); } catch { }
                        try { Directory.Delete(extractTemp, true); } catch { }
                    }

                    // 3. Setup Python 3.11 & PyTorch CUDA environment
                    string venvPython = Path.Combine(targetDir, ".venv", "Scripts", "python.exe");
                    if (!File.Exists(venvPython))
                    {
                        UpdateStatus("⚙️ Richte Python 3.11 & NVIDIA CUDA-Umgebung ein...");

                        string userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
                        string uvExe = Path.Combine(userProfile, ".local", "bin", "uv.exe");
                        if (!File.Exists(uvExe))
                        {
                            uvExe = Path.Combine(userProfile, ".cargo", "bin", "uv.exe");
                        }
                        if (!File.Exists(uvExe))
                        {
                            uvExe = Path.Combine(targetDir, "uv.exe");
                        }

                        if (!File.Exists(uvExe))
                        {
                            UpdateStatus("📥 Lade Paketmanager (uv) herunter...");
                            ProcessStartInfo psiUv = new ProcessStartInfo("powershell.exe", "-ExecutionPolicy Bypass -NoProfile -Command \"irm https://astral.sh/uv/install.ps1 | iex\"")
                            {
                                CreateNoWindow = true,
                                UseShellExecute = false
                            };
                            Process pUv = Process.Start(psiUv);
                            if (pUv != null) pUv.WaitForExit(60000);

                            uvExe = Path.Combine(userProfile, ".local", "bin", "uv.exe");
                        }

                        if (!File.Exists(uvExe))
                        {
                            uvExe = "uv";
                        }

                        UpdateStatus("⚙️ Erstelle isolierte Python 3.11 Umgebung...");
                        RunProcess(uvExe, "venv .venv --python 3.11", targetDir);

                        UpdateStatus("⚡ Installiere PyTorch CUDA 12.4 (NVIDIA RTX Beschleunigung)...");
                        RunProcess(uvExe, "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124", targetDir);

                        UpdateStatus("📦 Installiere Video- & Shader-Bibliotheken...");
                        RunProcess(uvExe, "pip install -r requirements.txt", targetDir);
                    }

                    if (!File.Exists(venvPython) || !File.Exists(appPy))
                    {
                        UpdateStatus("Fehler: Umgebung konnte nicht initialisiert werden.");
                        return;
                    }

                    UpdateStatus("⚡ Starte Luxanix Studio (NVIDIA CUDA Pipeline)...");

                    ProcessStartInfo psi = new ProcessStartInfo(venvPython, "ui\\app.py")
                    {
                        WorkingDirectory = targetDir,
                        CreateNoWindow = true,
                        UseShellExecute = false
                    };
                    serverProcess = Process.Start(psi);

                    // Wait for server port 7860 to become available
                    bool ready = false;
                    for (int i = 0; i < 50; i++)
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

        private static void RunProcess(string filename, string args, string workingDir)
        {
            ProcessStartInfo psi = new ProcessStartInfo(filename, args)
            {
                WorkingDirectory = workingDir,
                CreateNoWindow = true,
                UseShellExecute = false
            };
            Process p = Process.Start(psi);
            if (p != null)
            {
                p.WaitForExit();
            }
        }

        private static void CopyDirectory(string sourceDir, string destDir)
        {
            if (!Directory.Exists(destDir))
            {
                Directory.CreateDirectory(destDir);
            }

            foreach (string file in Directory.GetFiles(sourceDir))
            {
                string destFile = Path.Combine(destDir, Path.GetFileName(file));
                File.Copy(file, destFile, true);
            }

            foreach (string subDir in Directory.GetDirectories(sourceDir))
            {
                string destSubDir = Path.Combine(destDir, Path.GetFileName(subDir));
                CopyDirectory(subDir, destSubDir);
            }
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
