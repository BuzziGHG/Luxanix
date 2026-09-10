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
    public class SetupWizardForm : Form
    {
        // UI Containers
        private Panel panelSidebar;
        private Panel panelContent;
        private Panel panelBottom;

        // Pages
        private Panel pageWelcome;
        private Panel pageOptions;
        private Panel pageProgress;
        private Panel pageFinished;

        // Navigation Buttons
        private Button btnBack;
        private Button btnNext;
        private Button btnCancel;

        // Page 1 (Options) Controls
        private TextBox txtInstallPath;
        private CheckBox chkDesktopShortcut;
        private CheckBox chkStartMenu;
        private CheckBox chkRegisterApps;

        // Page 2 (Progress) Controls
        private Label lblProgressStatus;
        private Label lblProgressDetail;
        private ProgressBar prgInstallation;
        private ListBox lstLog;

        // Page 3 (Finished) Controls
        private CheckBox chkLaunchNow;

        // Paths & State
        private int currentPage = 0;
        private string currentExePath;
        private string currentDir;
        private string installDir;
        private bool isInstalledApp;
        private Process appProcess = null;

        public SetupWizardForm()
        {
            currentExePath = Process.GetCurrentProcess().MainModule.FileName;
            currentDir = Path.GetDirectoryName(currentExePath).TrimEnd('\\', '/');
            installDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Programs", "Luxanix Studio").TrimEnd('\\', '/');

            // If running directly inside the installed folder, skip the setup wizard and launch Studio!
            isInstalledApp = string.Equals(currentDir, installDir, StringComparison.OrdinalIgnoreCase);

            if (isInstalledApp)
            {
                InitializeFastLauncher();
            }
            else
            {
                InitializeSetupWizard();
            }
        }

        // =====================================================================
        // FAST LAUNCHER MODE (When executed from %LOCALAPPDATA%\Programs\Luxanix Studio)
        // =====================================================================
        private void InitializeFastLauncher()
        {
            this.Text = "⚡ Luxanix Studio — Starte...";
            this.Size = new Size(420, 160);
            this.StartPosition = FormStartPosition.CenterScreen;
            this.FormBorderStyle = FormBorderStyle.FixedDialog;
            this.MaximizeBox = false;
            this.BackColor = Color.FromArgb(14, 17, 22);
            this.ForeColor = Color.FromArgb(228, 233, 240);

            try { this.Icon = Icon.ExtractAssociatedIcon(currentExePath); } catch { }

            Label lbl = new Label()
            {
                Text = "⚡ Starte Luxanix Studio (NVIDIA RTX Pipeline)...",
                Font = new Font("Segoe UI", 10, FontStyle.Bold),
                ForeColor = Color.FromArgb(0, 229, 255),
                Location = new Point(20, 25),
                AutoSize = true
            };

            ProgressBar bar = new ProgressBar()
            {
                Location = new Point(20, 60),
                Size = new Size(365, 18),
                Style = ProgressBarStyle.Marquee,
                MarqueeAnimationSpeed = 20
            };

            this.Controls.Add(lbl);
            this.Controls.Add(bar);

            Thread t = new Thread(RunInstalledStudioApp);
            t.IsBackground = true;
            t.Start();
        }

        private void RunInstalledStudioApp()
        {
            try
            {
                string venvPython = Path.Combine(installDir, ".venv", "Scripts", "python.exe");
                if (!File.Exists(venvPython))
                {
                    SetupPythonEnvironment(installDir, null, null);
                }

                string desktopAppPy = Path.Combine(installDir, "ui", "desktop_app.py");
                string launchScript = File.Exists(desktopAppPy) ? "ui\\desktop_app.py" : "ui\\app.py";

                ProcessStartInfo psi = new ProcessStartInfo(venvPython, launchScript)
                {
                    WorkingDirectory = installDir,
                    CreateNoWindow = true,
                    UseShellExecute = false
                };
                appProcess = Process.Start(psi);

                Thread.Sleep(1500);

                this.Invoke((MethodInvoker)delegate
                {
                    this.Hide();
                });

                if (appProcess != null && !appProcess.HasExited)
                {
                    appProcess.WaitForExit();
                }

                this.Invoke((MethodInvoker)delegate
                {
                    Application.Exit();
                });
            }
            catch (Exception ex)
            {
                MessageBox.Show("Fehler beim Starten von Luxanix Studio: " + ex.Message, "Fehler", MessageBoxButtons.OK, MessageBoxIcon.Error);
                Application.Exit();
            }
        }

        // =====================================================================
        // FULL SETUP WIZARD MODE (Luxanix-Setup.exe)
        // =====================================================================
        private void InitializeSetupWizard()
        {
            this.Text = "Luxanix Studio Setup — Version 2.0.0 (CapCut Pro NLE Edition)";
            this.Size = new Size(680, 480);
            this.StartPosition = FormStartPosition.CenterScreen;
            this.FormBorderStyle = FormBorderStyle.FixedDialog;
            this.MaximizeBox = false;
            this.BackColor = Color.FromArgb(14, 18, 23);
            this.ForeColor = Color.FromArgb(226, 232, 240);

            try { this.Icon = Icon.ExtractAssociatedIcon(currentExePath); } catch { }

            // 1. Sidebar Panel (Branding)
            panelSidebar = new Panel()
            {
                Dock = DockStyle.Left,
                Width = 190,
                BackColor = Color.FromArgb(9, 12, 16)
            };

            Label lblSideLogo = new Label()
            {
                Text = "⚡ LUXANIX\n    STUDIO",
                Font = new Font("Segoe UI", 14, FontStyle.Bold),
                ForeColor = Color.FromArgb(0, 229, 255),
                Location = new Point(14, 20),
                AutoSize = true
            };

            Label lblSideBadge = new Label()
            {
                Text = "v2.0.0 CapCut Pro",
                Font = new Font("Segoe UI", 8.5f, FontStyle.Bold),
                ForeColor = Color.FromArgb(118, 185, 0),
                BackColor = Color.FromArgb(20, 32, 20),
                Location = new Point(18, 72),
                Padding = new Padding(4, 2, 4, 2),
                AutoSize = true
            };

            Label lblSideFeatures = new Label()
            {
                Text = "✨ Features:\n" +
                       "• CapCut Pro NLE Layout\n" +
                       "• Video & Foto Remastering\n" +
                       "• RTX 50 Blackwell RTGI\n" +
                       "• AV1 Direct Pipe 8K\n" +
                       "• Auto-Scene Dynamic AI\n" +
                       "• Screen-Space SSR\n" +
                       "• Multi-Track Timeline\n" +
                       "• Permanente Desktop-App",
                Font = new Font("Segoe UI", 9),
                ForeColor = Color.FromArgb(148, 163, 184),
                Location = new Point(16, 120),
                Size = new Size(160, 260)
            };

            panelSidebar.Controls.Add(lblSideLogo);
            panelSidebar.Controls.Add(lblSideBadge);
            panelSidebar.Controls.Add(lblSideFeatures);

            // 2. Bottom Navigation Panel
            panelBottom = new Panel()
            {
                Dock = DockStyle.Bottom,
                Height = 56,
                BackColor = Color.FromArgb(18, 22, 28)
            };

            btnCancel = new Button()
            {
                Text = "Abbrechen",
                Font = new Font("Segoe UI", 9.5f),
                BackColor = Color.FromArgb(30, 38, 48),
                ForeColor = Color.FromArgb(203, 213, 225),
                FlatStyle = FlatStyle.Flat,
                Location = new Point(560, 12),
                Size = new Size(95, 32),
                Cursor = Cursors.Hand
            };
            btnCancel.FlatAppearance.BorderSize = 0;
            btnCancel.Click += (s, e) => this.Close();

            btnNext = new Button()
            {
                Text = "Weiter >",
                Font = new Font("Segoe UI", 9.5f, FontStyle.Bold),
                BackColor = Color.FromArgb(0, 196, 204),
                ForeColor = Color.FromArgb(10, 15, 20),
                FlatStyle = FlatStyle.Flat,
                Location = new Point(455, 12),
                Size = new Size(95, 32),
                Cursor = Cursors.Hand
            };
            btnNext.FlatAppearance.BorderSize = 0;
            btnNext.Click += BtnNext_Click;

            btnBack = new Button()
            {
                Text = "< Zurück",
                Font = new Font("Segoe UI", 9.5f),
                BackColor = Color.FromArgb(30, 38, 48),
                ForeColor = Color.FromArgb(203, 213, 225),
                FlatStyle = FlatStyle.Flat,
                Location = new Point(350, 12),
                Size = new Size(95, 32),
                Visible = false,
                Cursor = Cursors.Hand
            };
            btnBack.FlatAppearance.BorderSize = 0;
            btnBack.Click += BtnBack_Click;

            panelBottom.Controls.Add(btnBack);
            panelBottom.Controls.Add(btnNext);
            panelBottom.Controls.Add(btnCancel);

            // 3. Main Content Panel
            panelContent = new Panel()
            {
                Dock = DockStyle.Fill,
                BackColor = Color.FromArgb(14, 18, 23),
                Padding = new Padding(24, 20, 24, 20)
            };

            BuildWelcomePage();
            BuildOptionsPage();
            BuildProgressPage();
            BuildFinishedPage();

            this.Controls.Add(panelContent);
            this.Controls.Add(panelSidebar);
            this.Controls.Add(panelBottom);

            ShowPage(0);
        }

        // =====================================================================
        // WIZARD PAGES SETUP
        // =====================================================================
        private void BuildWelcomePage()
        {
            pageWelcome = new Panel() { Dock = DockStyle.Fill, Visible = false };

            Label lblTitle = new Label()
            {
                Text = "Willkommen beim Installations-Assistenten",
                Font = new Font("Segoe UI", 14, FontStyle.Bold),
                ForeColor = Color.FromArgb(241, 245, 249),
                Location = new Point(10, 15),
                AutoSize = true
            };

            Label lblSub = new Label()
            {
                Text = "Dieser Assistent wird Luxanix Studio dauerhaft und fest auf Ihrem PC einrichten.",
                Font = new Font("Segoe UI", 9.5f),
                ForeColor = Color.FromArgb(148, 163, 184),
                Location = new Point(12, 48),
                AutoSize = true
            };

            Label lblBody = new Label()
            {
                Text = "Nach der Installation ist Luxanix Studio wie jede normale Windows-Anwendung " +
                       "direkt über Ihr Startmenü und Ihren Desktop verfügbar.\n\n" +
                       "Vorteile der Installation:\n" +
                       "✔  Kein wiederholtes Herunterladen oder Suchen der .exe mehr\n" +
                       "✔  Automatisches Hardware-Tuning für NVIDIA RTX Grafikkarten\n" +
                       "✔  Neues CapCut Pro Schnittstudio mit Multi-Track Timeline\n" +
                       "✔  Unterstützung für NVIDIA RTX 50 Blackwell & AV1 Dual-NVENC\n" +
                       "✔  Vollständig deinstallierbar über Windows 'Apps & Features'\n\n" +
                       "Klicken Sie auf 'Weiter', um die Installationsoptionen festzulegen.",
                Font = new Font("Segoe UI", 9.5f),
                ForeColor = Color.FromArgb(203, 213, 225),
                Location = new Point(12, 90),
                Size = new Size(440, 260)
            };

            pageWelcome.Controls.Add(lblTitle);
            pageWelcome.Controls.Add(lblSub);
            pageWelcome.Controls.Add(lblBody);
            panelContent.Controls.Add(pageWelcome);
        }

        private void BuildOptionsPage()
        {
            pageOptions = new Panel() { Dock = DockStyle.Fill, Visible = false };

            Label lblTitle = new Label()
            {
                Text = "Installationsziel & Windows-Verknüpfungen",
                Font = new Font("Segoe UI", 13, FontStyle.Bold),
                ForeColor = Color.FromArgb(241, 245, 249),
                Location = new Point(10, 15),
                AutoSize = true
            };

            Label lblPathDesc = new Label()
            {
                Text = "Luxanix Studio wird in folgendes Verzeichnis fest installiert:",
                Font = new Font("Segoe UI", 9),
                ForeColor = Color.FromArgb(148, 163, 184),
                Location = new Point(12, 50),
                AutoSize = true
            };

            txtInstallPath = new TextBox()
            {
                Text = installDir,
                ReadOnly = true,
                Font = new Font("Segoe UI", 9.5f),
                BackColor = Color.FromArgb(22, 28, 36),
                ForeColor = Color.FromArgb(226, 232, 240),
                BorderStyle = BorderStyle.FixedSingle,
                Location = new Point(12, 74),
                Size = new Size(430, 26)
            };

            Label lblOptionsHeader = new Label()
            {
                Text = "Verknüpfungen & Integration:",
                Font = new Font("Segoe UI", 10, FontStyle.Bold),
                ForeColor = Color.FromArgb(226, 232, 240),
                Location = new Point(12, 125),
                AutoSize = true
            };

            chkDesktopShortcut = new CheckBox()
            {
                Text = "Desktop-Verknüpfung erstellen (Luxanix Studio)",
                Checked = true,
                Font = new Font("Segoe UI", 9.5f),
                ForeColor = Color.FromArgb(203, 213, 225),
                Location = new Point(16, 155),
                AutoSize = true
            };

            chkStartMenu = new CheckBox()
            {
                Text = "Im Windows-Startmenü registrieren",
                Checked = true,
                Font = new Font("Segoe UI", 9.5f),
                ForeColor = Color.FromArgb(203, 213, 225),
                Location = new Point(16, 185),
                AutoSize = true
            };

            chkRegisterApps = new CheckBox()
            {
                Text = "In Windows 'Installierte Apps & Features' eintragen",
                Checked = true,
                Font = new Font("Segoe UI", 9.5f),
                ForeColor = Color.FromArgb(203, 213, 225),
                Location = new Point(16, 215),
                AutoSize = true
            };

            Panel cardGpu = new Panel()
            {
                Location = new Point(12, 260),
                Size = new Size(430, 60),
                BackColor = Color.FromArgb(18, 28, 22),
                BorderStyle = BorderStyle.FixedSingle
            };

            Label lblGpu = new Label()
            {
                Text = "🟢 Hardware-Beschleunigung: NVIDIA RTX GPU erkannt\n" +
                       "Tensor Cores (FP16 / TF32), 8K Raytracing & AV1 NVENC werden aktiviert.",
                Font = new Font("Segoe UI", 9),
                ForeColor = Color.FromArgb(118, 185, 0),
                Location = new Point(10, 10),
                AutoSize = true
            };
            cardGpu.Controls.Add(lblGpu);

            pageOptions.Controls.Add(lblTitle);
            pageOptions.Controls.Add(lblPathDesc);
            pageOptions.Controls.Add(txtInstallPath);
            pageOptions.Controls.Add(lblOptionsHeader);
            pageOptions.Controls.Add(chkDesktopShortcut);
            pageOptions.Controls.Add(chkStartMenu);
            pageOptions.Controls.Add(chkRegisterApps);
            pageOptions.Controls.Add(cardGpu);

            panelContent.Controls.Add(pageOptions);
        }

        private void BuildProgressPage()
        {
            pageProgress = new Panel() { Dock = DockStyle.Fill, Visible = false };

            Label lblTitle = new Label()
            {
                Text = "Installation wird durchgeführt...",
                Font = new Font("Segoe UI", 13, FontStyle.Bold),
                ForeColor = Color.FromArgb(241, 245, 249),
                Location = new Point(10, 15),
                AutoSize = true
            };

            lblProgressStatus = new Label()
            {
                Text = "Bereite Installation vor...",
                Font = new Font("Segoe UI", 9.5f, FontStyle.Bold),
                ForeColor = Color.FromArgb(0, 229, 255),
                Location = new Point(12, 50),
                AutoSize = true
            };

            prgInstallation = new ProgressBar()
            {
                Location = new Point(12, 78),
                Size = new Size(430, 22),
                Minimum = 0,
                Maximum = 100,
                Value = 0,
                Style = ProgressBarStyle.Continuous
            };

            lblProgressDetail = new Label()
            {
                Text = "Status: Initialisierung...",
                Font = new Font("Segoe UI", 8.5f),
                ForeColor = Color.FromArgb(148, 163, 184),
                Location = new Point(12, 108),
                Size = new Size(430, 20)
            };

            lstLog = new ListBox()
            {
                Location = new Point(12, 138),
                Size = new Size(430, 190),
                BackColor = Color.FromArgb(18, 22, 28),
                ForeColor = Color.FromArgb(148, 163, 184),
                BorderStyle = BorderStyle.FixedSingle,
                Font = new Font("Consolas", 8.5f)
            };

            pageProgress.Controls.Add(lblTitle);
            pageProgress.Controls.Add(lblProgressStatus);
            pageProgress.Controls.Add(prgInstallation);
            pageProgress.Controls.Add(lblProgressDetail);
            pageProgress.Controls.Add(lstLog);

            panelContent.Controls.Add(pageProgress);
        }

        private void BuildFinishedPage()
        {
            pageFinished = new Panel() { Dock = DockStyle.Fill, Visible = false };

            Label lblTitle = new Label()
            {
                Text = "🎉 Installation erfolgreich abgeschlossen!",
                Font = new Font("Segoe UI", 14, FontStyle.Bold),
                ForeColor = Color.FromArgb(74, 222, 128),
                Location = new Point(10, 20),
                AutoSize = true
            };

            Label lblBody = new Label()
            {
                Text = "Luxanix Studio wurde dauerhaft und fest auf Ihrem PC installiert.\n\n" +
                       "📌 Startmöglichkeiten:\n" +
                       "• Direkt über das Windows Startmenü (nach 'Luxanix' suchen)\n" +
                       "• Über die Desktop-Verknüpfung 'Luxanix Studio'\n\n" +
                       "💡 Wichtiger Hinweis:\n" +
                       "Sie können dieses Setup-Programm (Luxanix-Setup.exe) jetzt löschen " +
                       "oder archivieren. Sie müssen es nie wieder ausführen.\n\n" +
                       "Viel Spaß mit photorealistischem AI-Raytracing & 8K Remastering!",
                Font = new Font("Segoe UI", 9.5f),
                ForeColor = Color.FromArgb(203, 213, 225),
                Location = new Point(12, 70),
                Size = new Size(440, 210)
            };

            chkLaunchNow = new CheckBox()
            {
                Text = "Luxanix Studio jetzt sofort starten",
                Checked = true,
                Font = new Font("Segoe UI", 10, FontStyle.Bold),
                ForeColor = Color.FromArgb(0, 229, 255),
                Location = new Point(16, 295),
                AutoSize = true
            };

            pageFinished.Controls.Add(lblTitle);
            pageFinished.Controls.Add(lblBody);
            pageFinished.Controls.Add(chkLaunchNow);

            panelContent.Controls.Add(pageFinished);
        }

        // =====================================================================
        // WIZARD FLOW & STEP EXECUTION
        // =====================================================================
        private void ShowPage(int pageIdx)
        {
            currentPage = pageIdx;
            pageWelcome.Visible = (pageIdx == 0);
            pageOptions.Visible = (pageIdx == 1);
            pageProgress.Visible = (pageIdx == 2);
            pageFinished.Visible = (pageIdx == 3);

            if (pageIdx == 0)
            {
                btnBack.Visible = false;
                btnNext.Text = "Weiter >";
                btnCancel.Enabled = true;
            }
            else if (pageIdx == 1)
            {
                btnBack.Visible = true;
                btnNext.Text = "Installieren";
                btnCancel.Enabled = true;
            }
            else if (pageIdx == 2)
            {
                btnBack.Visible = false;
                btnNext.Enabled = false;
                btnCancel.Enabled = false;
            }
            else if (pageIdx == 3)
            {
                btnBack.Visible = false;
                btnNext.Enabled = true;
                btnNext.Text = "Fertigstellen";
                btnCancel.Visible = false;
            }
        }

        private void BtnNext_Click(object sender, EventArgs e)
        {
            if (currentPage == 0)
            {
                ShowPage(1);
            }
            else if (currentPage == 1)
            {
                ShowPage(2);
                StartInstallThread();
            }
            else if (currentPage == 3)
            {
                if (chkLaunchNow.Checked)
                {
                    string targetExe = Path.Combine(installDir, "Luxanix Studio.exe");
                    if (!File.Exists(targetExe)) targetExe = Path.Combine(installDir, "Luxanix.exe");
                    try
                    {
                        Process.Start(new ProcessStartInfo(targetExe) { WorkingDirectory = installDir });
                    }
                    catch { }
                }
                this.Close();
            }
        }

        private void BtnBack_Click(object sender, EventArgs e)
        {
            if (currentPage == 1)
            {
                ShowPage(0);
            }
        }

        private void StartInstallThread()
        {
            Thread t = new Thread(ExecuteInstallation);
            t.IsBackground = true;
            t.Start();
        }

        private void ExecuteInstallation()
        {
            try
            {
                UpdateProgress(5, "Erstelle Installationsverzeichnis...", installDir);
                if (!Directory.Exists(installDir))
                {
                    Directory.CreateDirectory(installDir);
                }

                // 1. Copy Application Executable as "Luxanix Studio.exe"
                UpdateProgress(15, "Installiere Hauptanwendung...", "Kopiere Luxanix Studio.exe");
                string targetExe = Path.Combine(installDir, "Luxanix Studio.exe");
                try
                {
                    File.Copy(currentExePath, targetExe, true);
                    // Also copy as Luxanix.exe for backwards compatibility
                    File.Copy(currentExePath, Path.Combine(installDir, "Luxanix.exe"), true);
                }
                catch { }

                // 2. Locate source files (Engine, UI, Presets, Assets)
                UpdateProgress(25, "Kopiere Benutzeroberfläche & CapCut NLE Engine...", "Übertrage Dateien...");
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
                    CopyDirectoryFiltered(sourceAppDir, installDir);
                }
                else
                {
                    UpdateProgress(35, "Lade neueste Dateien von GitHub herunter...", "Verbinde mit GitHub...");
                    DownloadAndExtractGitHub(installDir);
                }

                // 3. Setup Python 3.11 & PyTorch CUDA environment
                UpdateProgress(60, "Richte NVIDIA RTX CUDA Laufzeitumgebung ein...", "Prüfe Python & PyTorch...");
                string targetVenvPython = Path.Combine(installDir, ".venv", "Scripts", "python.exe");
                if (!File.Exists(targetVenvPython))
                {
                    string userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
                    string scratchDir = Path.Combine(userProfile, ".gemini", "antigravity", "scratch", "SimRTX-Studio");
                    string scratchVenv = Path.Combine(scratchDir, ".venv");

                    if (Directory.Exists(scratchVenv) && File.Exists(Path.Combine(scratchVenv, "Scripts", "python.exe")))
                    {
                        UpdateProgress(70, "Verknüpfe vorhandene CUDA 12.4 Umgebung...", "Erstelle NTFS Directory Junction...");
                        string targetVenv = Path.Combine(installDir, ".venv");
                        RunProcessHidden("cmd.exe", "/c mklink /J \"" + targetVenv + "\" \"" + scratchVenv + "\"", installDir);
                    }

                    if (!File.Exists(targetVenvPython))
                    {
                        SetupPythonEnvironment(installDir, UpdateProgress, AddLog);
                    }
                }

                // 4. Create Desktop & Start Menu Shortcuts
                if (chkDesktopShortcut.Checked)
                {
                    UpdateProgress(85, "Erstelle Desktop-Verknüpfung...", "Desktop: Luxanix Studio.lnk");
                    string desktopLnk = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "Luxanix Studio.lnk");
                    CreateShortcut(desktopLnk, targetExe, installDir, "Luxanix Studio — AI Raytracing & 8K Video Remaster (NVIDIA RTX)");
                }

                if (chkStartMenu.Checked)
                {
                    UpdateProgress(90, "Erstelle Startmenü-Eintrag...", "Startmenü: Luxanix Studio.lnk");
                    string startMenuPrograms = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.StartMenu), "Programs");
                    string startMenuLnk = Path.Combine(startMenuPrograms, "Luxanix Studio.lnk");
                    CreateShortcut(startMenuLnk, targetExe, installDir, "Luxanix Studio — AI Raytracing & 8K Video Remaster (NVIDIA RTX)");
                }

                // 5. Register in Windows
                if (chkRegisterApps.Checked)
                {
                    UpdateProgress(95, "Registriere in Windows 'Apps & Features'...", "Windows Registry...");
                    RegisterWindowsApp(installDir, targetExe);
                }

                UpdateProgress(100, "Installation abgeschlossen!", "Fertiggestellt.");
                Thread.Sleep(500);

                this.Invoke((MethodInvoker)delegate
                {
                    ShowPage(3);
                });
            }
            catch (Exception ex)
            {
                this.Invoke((MethodInvoker)delegate
                {
                    MessageBox.Show("Fehler bei der Installation:\n" + ex.Message, "Fehler", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    ShowPage(1);
                });
            }
        }

        private void UpdateProgress(int pct, string status, string detail)
        {
            if (this.IsDisposed || !this.IsHandleCreated) return;
            this.Invoke((MethodInvoker)delegate
            {
                prgInstallation.Value = Math.Min(100, Math.Max(0, pct));
                lblProgressStatus.Text = status;
                lblProgressDetail.Text = detail;
                if (!string.IsNullOrEmpty(detail))
                {
                    lstLog.Items.Add(detail);
                    lstLog.TopIndex = lstLog.Items.Count - 1;
                }
            });
        }

        private void AddLog(string msg)
        {
            if (this.IsDisposed || !this.IsHandleCreated) return;
            this.Invoke((MethodInvoker)delegate
            {
                lstLog.Items.Add(msg);
                lstLog.TopIndex = lstLog.Items.Count - 1;
            });
        }

        // =====================================================================
        // HELPER FUNCTIONS: FILE COPY, SHORTCUTS, REGISTRY, PYTHON SETUP
        // =====================================================================
        private static void CopyDirectoryFiltered(string sourceDir, string destDir)
        {
            Directory.CreateDirectory(destDir);

            foreach (string file in Directory.GetFiles(sourceDir))
            {
                string name = Path.GetFileName(file);
                // Skip git repo and temporary scripts
                if (name.StartsWith(".git") || name.EndsWith(".tmp") || name == "write_app.py") continue;
                try
                {
                    File.Copy(file, Path.Combine(destDir, name), true);
                }
                catch { }
            }

            foreach (string dir in Directory.GetDirectories(sourceDir))
            {
                string name = Path.GetFileName(dir);
                // Skip git, cache, and venv (venv is handled separately via junction)
                if (name == ".git" || name == "__pycache__" || name == ".venv") continue;
                CopyDirectoryFiltered(dir, Path.Combine(destDir, name));
            }
        }

        private static void CreateShortcut(string shortcutPath, string targetPath, string workingDir, string description)
        {
            try
            {
                string dir = Path.GetDirectoryName(shortcutPath);
                if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);

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
                        key.SetValue("DisplayName", "Luxanix Studio Pro — CapCut AI Video Editor");
                        key.SetValue("DisplayVersion", "2.0.0");
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

        private static void SetupPythonEnvironment(string targetDir, Action<int, string, string> progressCb, Action<string> logCb)
        {
            string userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            string uvExe = Path.Combine(userProfile, ".local", "bin", "uv.exe");
            if (!File.Exists(uvExe)) uvExe = Path.Combine(userProfile, ".cargo", "bin", "uv.exe");

            if (!File.Exists(uvExe))
            {
                if (logCb != null) logCb("Lade Paketmanager (uv) herunter...");
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

            if (logCb != null) logCb("Erstelle Python 3.11 Virtualenv...");
            RunProcessHidden(uvExe, "venv .venv --python 3.11", targetDir);

            if (logCb != null) logCb("Installiere PyTorch CUDA 12.4 (NVIDIA RTX)...");
            RunProcessHidden(uvExe, "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124", targetDir);

            if (logCb != null) logCb("Installiere CustomTkinter & Video-Pipeline...");
            RunProcessHidden(uvExe, "pip install -r requirements.txt", targetDir);
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

            CopyDirectoryFiltered(sourceDir, destDir);
            try { File.Delete(zipPath); } catch { }
            try { Directory.Delete(extractTemp, true); } catch { }
        }

        private static void RunProcessHidden(string filename, string args, string workingDir)
        {
            ProcessStartInfo psi = new ProcessStartInfo(filename, args)
            {
                WorkingDirectory = workingDir,
                CreateNoWindow = true,
                UseShellExecute = false
            };
            Process p = Process.Start(psi);
            if (p != null) p.WaitForExit();
        }

        // =====================================================================
        // UNINSTALLER
        // =====================================================================
        private static void PerformUninstall()
        {
            DialogResult res = MessageBox.Show(
                "Möchten Sie Luxanix Studio wirklich vollständig von Ihrem System deinstallieren?",
                "Luxanix Studio Deinstallation",
                MessageBoxButtons.YesNo,
                MessageBoxIcon.Question
            );

            if (res != DialogResult.Yes) return;

            try
            {
                string desktopLnk = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "Luxanix Studio.lnk");
                if (File.Exists(desktopLnk)) File.Delete(desktopLnk);

                string startMenuLnk = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.StartMenu), "Programs", "Luxanix Studio.lnk");
                if (File.Exists(startMenuLnk)) File.Delete(startMenuLnk);

                try
                {
                    Registry.CurrentUser.DeleteSubKeyTree(@"Software\Microsoft\Windows\CurrentVersion\Uninstall\Luxanix Studio", false);
                }
                catch { }

                string exePath = Process.GetCurrentProcess().MainModule.FileName;
                string appDir = Path.GetDirectoryName(exePath);

                ProcessStartInfo psi = new ProcessStartInfo("cmd.exe", "/c ping 127.0.0.1 -n 2 > nul & rmdir /s /q \"" + appDir + "\"")
                {
                    CreateNoWindow = true,
                    UseShellExecute = false
                };
                Process.Start(psi);

                MessageBox.Show("Luxanix Studio wurde erfolgreich deinstalliert.", "Deinstallation fertig", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Fehler bei der Deinstallation: " + ex.Message, "Fehler", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        public static void ExecuteSilentInstall()
        {
            string currentExePath = Process.GetCurrentProcess().MainModule.FileName;
            string currentDir = Path.GetDirectoryName(currentExePath).TrimEnd('\\', '/');
            string installDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Programs", "Luxanix Studio").TrimEnd('\\', '/');

            Console.WriteLine("[Setup] Erstelle Zielverzeichnis: " + installDir);
            Directory.CreateDirectory(installDir);

            string targetExe = Path.Combine(installDir, "Luxanix Studio.exe");
            try
            {
                File.Copy(currentExePath, targetExe, true);
                File.Copy(currentExePath, Path.Combine(installDir, "Luxanix.exe"), true);
            }
            catch { }

            string sourceAppDir = null;
            if (File.Exists(Path.Combine(currentDir, "ui", "desktop_app.py"))) sourceAppDir = currentDir;
            else
            {
                string userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
                string scratchDir = Path.Combine(userProfile, ".gemini", "antigravity", "scratch", "SimRTX-Studio");
                if (File.Exists(Path.Combine(scratchDir, "ui", "desktop_app.py"))) sourceAppDir = scratchDir;
            }

            if (sourceAppDir != null)
            {
                Console.WriteLine("[Setup] Kopiere Anwendungsdateien von " + sourceAppDir);
                CopyDirectoryFiltered(sourceAppDir, installDir);
            }
            else
            {
                Console.WriteLine("[Setup] Lade Dateien von GitHub...");
                DownloadAndExtractGitHub(installDir);
            }

            string targetVenv = Path.Combine(installDir, ".venv");
            string targetVenvPython = Path.Combine(targetVenv, "Scripts", "python.exe");
            if (!File.Exists(targetVenvPython))
            {
                string userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
                string scratchVenv = Path.Combine(userProfile, ".gemini", "antigravity", "scratch", "SimRTX-Studio", ".venv");
                if (Directory.Exists(scratchVenv) && File.Exists(Path.Combine(scratchVenv, "Scripts", "python.exe")))
                {
                    Console.WriteLine("[Setup] Verknuepfe CUDA 12.4 Umgebung...");
                    RunProcessHidden("cmd.exe", "/c mklink /J \"" + targetVenv + "\" \"" + scratchVenv + "\"", installDir);
                }
                else
                {
                    SetupPythonEnvironment(installDir, null, null);
                }
            }

            Console.WriteLine("[Setup] Erstelle Desktop- und Startmenue-Verknuepfungen...");
            string desktopLnk = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "Luxanix Studio.lnk");
            CreateShortcut(desktopLnk, targetExe, installDir, "Luxanix Studio — AI Raytracing & 8K Video Remaster");

            string startMenuPrograms = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.StartMenu), "Programs");
            string startMenuLnk = Path.Combine(startMenuPrograms, "Luxanix Studio.lnk");
            CreateShortcut(startMenuLnk, targetExe, installDir, "Luxanix Studio — AI Raytracing & 8K Video Remaster");

            RegisterWindowsApp(installDir, targetExe);
            Console.WriteLine("[Setup] SUCCESS: Luxanix Studio dauerhaft installiert!");
        }

        [STAThread]
        public static void Main(string[] args)
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);

            if (args != null && args.Length > 0)
            {
                string a0 = args[0].ToLowerInvariant();
                if (a0 == "--uninstall" || a0 == "/uninstall")
                {
                    PerformUninstall();
                    return;
                }
                if (a0 == "/silent" || a0 == "/s" || a0 == "--silent")
                {
                    ExecuteSilentInstall();
                    return;
                }
            }

            Application.Run(new SetupWizardForm());
        }
    }
}
