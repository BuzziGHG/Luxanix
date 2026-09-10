using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.IO.Compression;
using System.Net;
using System.Reflection;
using System.Threading;
using System.Windows.Forms;
using Microsoft.Win32;

namespace LuxanixLauncher
{
    public class LauncherForm : Form
    {
        private Label lblTitle;
        private Label lblSubtitle;
        private Label lblStatus;
        private ProgressBar progressBar;
        private Button btnExit;
        private NotifyIcon trayIcon;
        private Process serverProcess = null;

        private string currentExePath;
        private string currentDir;
        private string installDir;
        private bool isInstalled;

        public LauncherForm()
        {
            currentExePath = Process.GetCurrentProcess().MainModule.FileName;
            currentDir = Path.GetDirectoryName(currentExePath).TrimEnd('\\', '/');
            installDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Programs", "Luxanix").TrimEnd('\\', '/');
            isInstalled = string.Equals(currentDir, installDir, StringComparison.OrdinalIgnoreCase);

            InitializeComponent();
            StartApplicationFlow();
        }

        private void InitializeComponent()
        {
            this.Text = isInstalled ? "Luxanix Studio — RTX Video Remaster" : "Luxanix Studio Setup — Permanente Installation";
            this.Size = new Size(540, 270);
            this.StartPosition = FormStartPosition.CenterScreen;
            this.FormBorderStyle = FormBorderStyle.FixedDialog;
            this.MaximizeBox = false;
            this.BackColor = Color.FromArgb(11, 13, 16);
            this.ForeColor = Color.FromArgb(228, 233, 240);

            try
            {
                this.Icon = Icon.ExtractAssociatedIcon(currentExePath);
            }
            catch { }

            lblTitle = new Label()
            {
                Text = isInstalled ? "⚡ LUXANIX STUDIO" : "⚡ LUXANIX STUDIO INSTALLATION",
                Font = new Font("Segoe UI", 15, FontStyle.Bold),
                ForeColor = Color.FromArgb(118, 185, 0),
                Location = new Point(24, 20),
                AutoSize = true
            };

            lblSubtitle = new Label()
            {
                Text = isInstalled ? "AI Raytracing & 8K Remaster (NVIDIA RTX 20/30/40/50 Blackwell)" : "Permanente Installation & Desktop-Integration (RTX 50 Ready)",
                Font = new Font("Segoe UI", 9.5f, FontStyle.Regular),
                ForeColor = Color.FromArgb(148, 163, 184),
                Location = new Point(26, 52),
                AutoSize = true
            };

            lblStatus = new Label()
            {
                Text = isInstalled ? "⚡ Starte Luxanix Studio Desktop-App..." : "📦 Bereite permanente Installation vor...",
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
                MarqueeAnimationSpeed = 25
            };

            btnExit = new Button()
            {
                Text = "Abbrechen",
                Font = new Font("Segoe UI", 9.5f),
                BackColor = Color.FromArgb(30, 36, 45),
                ForeColor = Color.FromArgb(220, 225, 230),
                FlatStyle = FlatStyle.Flat,
                Location = new Point(366, 185),
                Size = new Size(130, 36),
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
                trayIcon.Icon = this.Icon;
            }
            catch { }

            this.Controls.Add(lblTitle);
            this.Controls.Add(lblSubtitle);
            this.Controls.Add(lblStatus);
            this.Controls.Add(progressBar);
            this.Controls.Add(btnExit);

            this.FormClosing += LauncherForm_FormClosing;
        }

        private void StartApplicationFlow()
        {
            Thread t = new Thread(() =>
            {
                try
                {
                    if (!isInstalled)
                    {
                        PerformPermanentInstallation();
                    }
                    else
                    {
                        RunInstalledApp();
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

        private void PerformPermanentInstallation()
        {
            UpdateStatus("📁 Erstelle Installationsverzeichnis: " + installDir);
            if (!Directory.Exists(installDir))
            {
                Directory.CreateDirectory(installDir);
            }

            // 1. Copy executable to target directory
            string targetExe = Path.Combine(installDir, "Luxanix.exe");
            try
            {
                File.Copy(currentExePath, targetExe, true);
            }
            catch { }

            // 2. Determine source of app files (engine, ui, presets, assets)
            string sourceAppDir = null;
            if (File.Exists(Path.Combine(currentDir, "ui", "desktop_app.py")))
            {
                sourceAppDir = currentDir;
            }
            else
            {
                string userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
                string scratchDir = Path.Combine(userProfile, ".gemini", "antigravity", "scratch", "SimRTX-Studio");
                if (File.Exists(Path.Combine(scratchDir, "ui", "desktop_app.py")))
                {
                    sourceAppDir = scratchDir;
                }
            }

            if (sourceAppDir != null)
            {
                UpdateStatus("📦 Kopiere Anwendungsdateien (Engine, UI, Presets, Assets)...");
                CopyDirectory(sourceAppDir, installDir);
            }
            else
            {
                UpdateStatus("📥 Lade Luxanix Studio Dateien von GitHub herunter...");
                DownloadAndExtractGitHub(installDir);
            }

            // 3. Setup Python 3.11 & PyTorch CUDA environment
            string targetVenvPython = Path.Combine(installDir, ".venv", "Scripts", "python.exe");
            if (!File.Exists(targetVenvPython))
            {
                string userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
                string scratchDir = Path.Combine(userProfile, ".gemini", "antigravity", "scratch", "SimRTX-Studio");
                string scratchVenv = Path.Combine(scratchDir, ".venv");

                // If scratch venv exists on this machine, create fast junction to avoid 4GB re-download
                if (Directory.Exists(scratchVenv) && File.Exists(Path.Combine(scratchVenv, "Scripts", "python.exe")))
                {
                    UpdateStatus("⚡ Verknüpfe NVIDIA RTX CUDA Laufzeitumgebung...");
                    string targetVenv = Path.Combine(installDir, ".venv");
                    RunProcess("cmd.exe", "/c mklink /J \"" + targetVenv + "\" \"" + scratchVenv + "\"", installDir);
                }

                // If still missing, install via uv
                if (!File.Exists(targetVenvPython))
                {
                    SetupPythonViaUv(installDir);
                }
            }

            // 4. Create Desktop and Start Menu shortcuts
            UpdateStatus("🔗 Erstelle Desktop- und Startmenü-Verknüpfungen...");
            string desktopLnk = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "Luxanix Studio.lnk");
            string startMenuPrograms = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.StartMenu), "Programs");
            string startMenuLnk = Path.Combine(startMenuPrograms, "Luxanix Studio.lnk");

            CreateShortcut(desktopLnk, targetExe, installDir, "Luxanix Studio — AI Raytracing & 8K Video Remaster");
            CreateShortcut(startMenuLnk, targetExe, installDir, "Luxanix Studio — AI Raytracing & 8K Video Remaster");

            // 5. Register in Windows "Apps & Features"
            UpdateStatus("📝 Registriere Luxanix Studio in Windows...");
            RegisterWindowsApp(installDir, targetExe);

            // 6. Launch installed application and exit installer
            UpdateStatus("✅ Installation erfolgreich! Starte Luxanix Studio...");
            Thread.Sleep(800);

            Process.Start(new ProcessStartInfo(targetExe) { WorkingDirectory = installDir });

            this.Invoke((MethodInvoker)delegate
            {
                Application.Exit();
            });
        }

        private void RunInstalledApp()
        {
            // Ensure shortcuts exist
            string targetExe = Path.Combine(installDir, "Luxanix.exe");
            string desktopLnk = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "Luxanix Studio.lnk");
            string startMenuLnk = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.StartMenu), "Programs", "Luxanix Studio.lnk");
            if (!File.Exists(desktopLnk)) CreateShortcut(desktopLnk, targetExe, installDir, "Luxanix Studio — AI Raytracing & 8K Video Remaster");
            if (!File.Exists(startMenuLnk)) CreateShortcut(startMenuLnk, targetExe, installDir, "Luxanix Studio — AI Raytracing & 8K Video Remaster");

            string venvPython = Path.Combine(installDir, ".venv", "Scripts", "python.exe");
            if (!File.Exists(venvPython))
            {
                SetupPythonViaUv(installDir);
            }

            string desktopAppPy = Path.Combine(installDir, "ui", "desktop_app.py");
            string launchScript = File.Exists(desktopAppPy) ? "ui\\desktop_app.py" : "ui\\app.py";

            UpdateStatus("⚡ Starte Luxanix Studio Desktop-App (NVIDIA RTX Pipeline)...");

            ProcessStartInfo psi = new ProcessStartInfo(venvPython, launchScript)
            {
                WorkingDirectory = installDir,
                CreateNoWindow = true,
                UseShellExecute = false
            };
            serverProcess = Process.Start(psi);

            // Wait for native desktop window to display, then hide launcher splash
            Thread.Sleep(2000);

            if (serverProcess != null && !serverProcess.HasExited)
            {
                this.Invoke((MethodInvoker)delegate
                {
                    this.Hide();
                });
                serverProcess.WaitForExit();
                Application.Exit();
            }
            else
            {
                UpdateStatus("Anwendung beendet.");
            }
        }

        private void SetupPythonViaUv(string targetDir)
        {
            UpdateStatus("⚙️ Richte Python 3.11 & NVIDIA CUDA-Umgebung ein...");
            string userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            string uvExe = Path.Combine(userProfile, ".local", "bin", "uv.exe");
            if (!File.Exists(uvExe)) uvExe = Path.Combine(userProfile, ".cargo", "bin", "uv.exe");
            if (!File.Exists(uvExe)) uvExe = Path.Combine(targetDir, "uv.exe");

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

            if (!File.Exists(uvExe)) uvExe = "uv";

            UpdateStatus("⚙️ Erstelle isolierte Python 3.11 Umgebung...");
            RunProcess(uvExe, "venv .venv --python 3.11", targetDir);

            UpdateStatus("⚡ Installiere PyTorch CUDA 12.4 (NVIDIA RTX Beschleunigung)...");
            RunProcess(uvExe, "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124", targetDir);

            UpdateStatus("📦 Installiere Video- & Shader-Bibliotheken...");
            RunProcess(uvExe, "pip install -r requirements.txt", targetDir);
        }

        private static void DownloadAndExtractGitHub(string destDir)
        {
            ServicePointManager.SecurityProtocol = (SecurityProtocolType)3072 | (SecurityProtocolType)768 | (SecurityProtocolType)192;
            string zipPath = Path.Combine(Path.GetTempPath(), "Luxanix-main.zip");
            using (WebClient client = new WebClient())
            {
                client.Headers.Add("User-Agent", "Luxanix-Installer");
                client.DownloadFile("https://github.com/BuzziGHG/Luxanix/archive/refs/heads/main.zip", zipPath);
            }

            string extractTemp = Path.Combine(Path.GetTempPath(), "Luxanix_Extract_" + Guid.NewGuid().ToString("N"));
            ZipFile.ExtractToDirectory(zipPath, extractTemp);

            string sourceDir = Path.Combine(extractTemp, "Luxanix-main");
            if (!Directory.Exists(sourceDir))
            {
                string[] subDirs = Directory.GetDirectories(extractTemp);
                if (subDirs.Length > 0) sourceDir = subDirs[0];
            }

            CopyDirectory(sourceDir, destDir);
            try { File.Delete(zipPath); } catch { }
            try { Directory.Delete(extractTemp, true); } catch { }
        }

        private static void CreateShortcut(string shortcutPath, string targetPath, string workingDir, string description)
        {
            try
            {
                string dir = Path.GetDirectoryName(shortcutPath);
                if (!Directory.Exists(dir))
                {
                    Directory.CreateDirectory(dir);
                }

                Type shellType = Type.GetTypeFromProgID("WScript.Shell");
                if (shellType == null) return;
                object shell = Activator.CreateInstance(shellType);
                object shortcut = shellType.InvokeMember("CreateShortcut", BindingFlags.InvokeMethod, null, shell, new object[] { shortcutPath });
                Type scType = shortcut.GetType();
                scType.InvokeMember("TargetPath", BindingFlags.SetProperty, null, shortcut, new object[] { targetPath });
                scType.InvokeMember("WorkingDirectory", BindingFlags.SetProperty, null, shortcut, new object[] { workingDir });
                scType.InvokeMember("Description", BindingFlags.SetProperty, null, shortcut, new object[] { description });
                scType.InvokeMember("IconLocation", BindingFlags.SetProperty, null, shortcut, new object[] { targetPath + ",0" });
                scType.InvokeMember("Save", BindingFlags.InvokeMethod, null, shortcut, null);
            }
            catch { }
        }

        private static void RegisterWindowsApp(string targetDir, string exePath)
        {
            try
            {
                using (RegistryKey key = Registry.CurrentUser.CreateSubKey(@"Software\Microsoft\Windows\CurrentVersion\Uninstall\Luxanix Studio"))
                {
                    if (key != null)
                    {
                        key.SetValue("DisplayName", "Luxanix Studio — RTX Video Remaster");
                        key.SetValue("DisplayVersion", "1.1.0");
                        key.SetValue("Publisher", "BuzziGHG");
                        key.SetValue("InstallLocation", targetDir);
                        key.SetValue("DisplayIcon", exePath);
                        key.SetValue("UninstallString", "\"" + exePath + "\" --uninstall");
                        key.SetValue("URLInfoAbout", "https://github.com/BuzziGHG/Luxanix");
                        key.SetValue("NoModify", 1, RegistryValueKind.DWord);
                        key.SetValue("NoRepair", 1, RegistryValueKind.DWord);
                    }
                }
            }
            catch { }
        }

        private static void PerformUninstall()
        {
            DialogResult res = MessageBox.Show(
                "Möchtest du Luxanix Studio wirklich vollständig von deinem PC entfernen?",
                "Luxanix Studio Deinstallation",
                MessageBoxButtons.YesNo,
                MessageBoxIcon.Question
            );

            if (res != DialogResult.Yes) return;

            try
            {
                // 1. Remove Desktop shortcut
                string desktopLnk = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "Luxanix Studio.lnk");
                if (File.Exists(desktopLnk)) File.Delete(desktopLnk);

                // 2. Remove Start Menu shortcut
                string startMenuLnk = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.StartMenu), "Programs", "Luxanix Studio.lnk");
                if (File.Exists(startMenuLnk)) File.Delete(startMenuLnk);

                // 3. Remove Registry entry
                try
                {
                    Registry.CurrentUser.DeleteSubKeyTree(@"Software\Microsoft\Windows\CurrentVersion\Uninstall\Luxanix Studio", false);
                }
                catch { }

                // 4. Schedule directory removal after process exit
                string exePath = Process.GetCurrentProcess().MainModule.FileName;
                string appDir = Path.GetDirectoryName(exePath);

                ProcessStartInfo psi = new ProcessStartInfo("cmd.exe", "/c ping 127.0.0.1 -n 2 > nul & rmdir /s /q \"" + appDir + "\"")
                {
                    CreateNoWindow = true,
                    UseShellExecute = false
                };
                Process.Start(psi);

                MessageBox.Show(
                    "Luxanix Studio wurde erfolgreich deinstalliert.",
                    "Deinstallation abgeschlossen",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Information
                );
            }
            catch (Exception ex)
            {
                MessageBox.Show("Fehler bei der Deinstallation: " + ex.Message, "Fehler", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
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
                string fileName = Path.GetFileName(file);
                if (fileName.StartsWith(".git") || fileName.EndsWith(".zip")) continue;
                string destFile = Path.Combine(destDir, fileName);
                try
                {
                    File.Copy(file, destFile, true);
                }
                catch { }
            }

            foreach (string subDir in Directory.GetDirectories(sourceDir))
            {
                string dirName = Path.GetFileName(subDir);
                if (dirName.Equals(".git", StringComparison.OrdinalIgnoreCase) ||
                    dirName.Equals(".venv", StringComparison.OrdinalIgnoreCase) ||
                    dirName.Equals("test_output", StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }
                string destSubDir = Path.Combine(destDir, dirName);
                CopyDirectory(subDir, destSubDir);
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
        public static void Main(string[] args)
        {
            if (args != null && args.Length > 0 && args[0].ToLower() == "--uninstall")
            {
                PerformUninstall();
                return;
            }

            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new LauncherForm());
        }
    }
}
