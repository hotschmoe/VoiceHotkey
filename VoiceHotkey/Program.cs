using System.Runtime.InteropServices;
using System.Text;
using NAudio.Wave;

namespace VoiceHotkey;

class Program
{
    // --- Win32 hotkey ---
    const int WM_HOTKEY = 0x0312;
    const int HOTKEY_ID = 1;
    const uint MOD_CONTROL = 0x0002;
    const uint VK_SPACE = 0x20;

    [DllImport("user32.dll")] static extern bool RegisterHotKey(IntPtr hWnd, int id, uint fsModifiers, uint vk);
    [DllImport("user32.dll")] static extern bool UnregisterHotKey(IntPtr hWnd, int id);
    [DllImport("user32.dll")] static extern int GetMessage(out MSG lpMsg, IntPtr hWnd, uint wMsgFilterMin, uint wMsgFilterMax);
    [DllImport("user32.dll")] static extern bool TranslateMessage(ref MSG lpMsg);
    [DllImport("user32.dll")] static extern IntPtr DispatchMessage(ref MSG lpMsg);

    [StructLayout(LayoutKind.Sequential)]
    struct MSG { public IntPtr hwnd; public uint message; public IntPtr wParam; public IntPtr lParam; public uint time; public POINT pt; }
    [StructLayout(LayoutKind.Sequential)]
    struct POINT { public int x, y; }

    // --- Clipboard ---
    [DllImport("user32.dll")] static extern bool OpenClipboard(IntPtr hWndNewOwner);
    [DllImport("user32.dll")] static extern bool EmptyClipboard();
    [DllImport("user32.dll")] static extern IntPtr SetClipboardData(uint uFormat, IntPtr hMem);
    [DllImport("user32.dll")] static extern bool CloseClipboard();
    const uint CF_UNICODETEXT = 13;
    const uint GMEM_MOVEABLE = 0x0002;
    [DllImport("kernel32.dll")] static extern IntPtr GlobalAlloc(uint uFlags, UIntPtr dwBytes);
    [DllImport("kernel32.dll")] static extern IntPtr GlobalLock(IntPtr hMem);
    [DllImport("kernel32.dll")] static extern bool GlobalUnlock(IntPtr hMem);

    // --- SendInput for Ctrl+V ---
    [DllImport("user32.dll", SetLastError = true)] static extern uint SendInput(uint nInputs, INPUT[] pInputs, int cbSize);
    [DllImport("user32.dll")] static extern uint MapVirtualKey(uint uCode, uint uMapType);
    [DllImport("user32.dll")] static extern short GetAsyncKeyState(int vKey);
    [DllImport("user32.dll")] static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] static extern int GetWindowText(IntPtr hWnd, System.Text.StringBuilder lpString, int nMaxCount);
    [DllImport("user32.dll")] static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint lpdwProcessId);
    const uint INPUT_KEYBOARD = 1;
    const uint KEYEVENTF_KEYUP = 0x0002;
    const uint KEYEVENTF_SCANCODE = 0x0008;
    const uint KEYEVENTF_EXTENDEDKEY = 0x0001;
    const uint KEYEVENTF_UNICODE = 0x0004;
    const uint MAPVK_VK_TO_VSC = 0;
    const ushort VK_CONTROL_KEY = 0x11;
    const ushort VK_LCONTROL = 0xA2;
    const ushort VK_RCONTROL = 0xA3;
    const ushort VK_SHIFT_KEY = 0x10;
    const ushort VK_LSHIFT = 0xA0;
    const ushort VK_RSHIFT = 0xA1;
    const ushort VK_MENU_KEY = 0x12;
    const ushort VK_LMENU = 0xA4;
    const ushort VK_RMENU = 0xA5;
    const ushort VK_LWIN = 0x5B;
    const ushort VK_RWIN = 0x5C;
    const ushort VK_V_KEY = 0x56;

    [StructLayout(LayoutKind.Sequential)]
    struct INPUT { public uint type; public INPUTUNION u; }
    [StructLayout(LayoutKind.Explicit)]
    struct INPUTUNION
    {
        // INPUT contains a native union. MOUSEINPUT is its largest member on
        // 64-bit Windows, so it must be declared even though this app only
        // sends keyboard input. Without it Marshal.SizeOf<INPUT>() is 32
        // instead of the native 40 bytes and SendInput rejects every call.
        [FieldOffset(0)] public MOUSEINPUT mi;
        [FieldOffset(0)] public KEYBDINPUT ki;
        [FieldOffset(0)] public HARDWAREINPUT hi;
    }
    [StructLayout(LayoutKind.Sequential)]
    struct MOUSEINPUT
    {
        public int dx;
        public int dy;
        public uint mouseData;
        public uint dwFlags;
        public uint time;
        public IntPtr dwExtraInfo;
    }
    [StructLayout(LayoutKind.Sequential)]
    struct KEYBDINPUT { public ushort wVk; public ushort wScan; public uint dwFlags; public uint time; public IntPtr dwExtraInfo; }
    [StructLayout(LayoutKind.Sequential)]
    struct HARDWAREINPUT { public uint uMsg; public ushort wParamL; public ushort wParamH; }

    // --- App state ---
    static Transcriber? transcriber;
    static WaveInEvent? waveSource;
    static List<byte>? recordedBytes;
    static volatile bool isRecording;
    static readonly object stateLock = new();

    // --- VAD auto-stop ---
    // Peak amplitude threshold below which a 20 ms frame counts as silence.
    const float VadPeakThreshold = 0.02f;
    // Auto-stop after this much consecutive silence, but only once we've heard speech.
    const int VadSilenceStopMs = 1200;
    // Hard cap so a stuck mic doesn't record forever.
    const int MaxRecordSeconds = 30;
    static int vadSilentMs;
    static bool vadHeardSpeech;
    static DateTime recordStartUtc;

    [STAThread]
    static void Main()
    {
        Console.WriteLine("=== VoiceHotkey ===");
        Console.WriteLine("NPU-powered voice-to-text for Snapdragon X Elite");
        Console.WriteLine();

        string dir = AppContext.BaseDirectory;
        string encoderOnnx = Path.Combine(dir, "models", "encoder_model_merged.onnx");
        string decoderOnnx = Path.Combine(dir, "models", "decoder_model_merged.onnx");
        string tokenizerDir = Path.Combine(dir, "tokenizer");
        string cacheDir = Path.Combine(dir, "qnn_cache");

        foreach (var p in new[] { encoderOnnx, decoderOnnx,
                                   Path.Combine(tokenizerDir, "vocab.json") })
        {
            if (!File.Exists(p)) { Console.WriteLine($"Missing: {p}"); return; }
        }

        Console.WriteLine("Loading Whisper on NPU (first run JIT-compiles graphs, ~30 s; cached runs are <1 s)...");
        try
        {
            transcriber = new Transcriber(encoderOnnx, decoderOnnx, cacheDir, tokenizerDir);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"FAILED: {ex.Message}");
            return;
        }
        Console.WriteLine("Loaded.");

        if (!RegisterHotKey(IntPtr.Zero, HOTKEY_ID, MOD_CONTROL, VK_SPACE))
        {
            Console.WriteLine("ERROR: Could not register Ctrl+Space.");
            return;
        }

        Console.WriteLine();
        Console.WriteLine("Ready. Ctrl+Space = toggle dictation. Ctrl+C = quit.");
        Console.WriteLine();

        Console.CancelKeyPress += (_, e) => { e.Cancel = true; Shutdown(); };

        while (GetMessage(out MSG msg, IntPtr.Zero, 0, 0) > 0)
        {
            if (msg.message == WM_HOTKEY && msg.wParam == (IntPtr)HOTKEY_ID)
            {
                lock (stateLock)
                {
                    if (isRecording) StopAndTranscribe();
                    else StartRecording();
                }
            }
            TranslateMessage(ref msg);
            DispatchMessage(ref msg);
        }
        Shutdown();
    }

    static void Shutdown()
    {
        UnregisterHotKey(IntPtr.Zero, HOTKEY_ID);
        try { waveSource?.StopRecording(); waveSource?.Dispose(); } catch { }
        transcriber?.Dispose();
        Environment.Exit(0);
    }

    static void StartRecording()
    {
        recordedBytes = new List<byte>(LogMel.SampleRate * 2 * MaxRecordSeconds);
        vadSilentMs = 0;
        vadHeardSpeech = false;
        recordStartUtc = DateTime.UtcNow;
        waveSource = new WaveInEvent { WaveFormat = new WaveFormat(LogMel.SampleRate, 16, 1) };
        waveSource.DataAvailable += OnAudioAvailable;
        waveSource.StartRecording();
        isRecording = true;
        Console.ForegroundColor = ConsoleColor.Red;
        Console.Write("[REC] ");
        Console.ResetColor();
        Console.WriteLine("Speak now... (auto-stops on silence; Ctrl+Space to stop early)");
    }

    static void OnAudioAvailable(object? sender, WaveInEventArgs e)
    {
        if (!isRecording || recordedBytes == null) return;

        lock (recordedBytes) recordedBytes.AddRange(new ArraySegment<byte>(e.Buffer, 0, e.BytesRecorded));

        int samples = e.BytesRecorded / 2;
        float peak = 0f;
        for (int i = 0; i < samples; i++)
        {
            short s = (short)(e.Buffer[2 * i] | (e.Buffer[2 * i + 1] << 8));
            float mag = Math.Abs(s / 32768f);
            if (mag > peak) peak = mag;
        }
        int frameMs = (samples * 1000) / LogMel.SampleRate;

        if (peak >= VadPeakThreshold) { vadHeardSpeech = true; vadSilentMs = 0; }
        else if (vadHeardSpeech) vadSilentMs += frameMs;

        bool silenceDone = vadHeardSpeech && vadSilentMs >= VadSilenceStopMs;
        bool hardCap = (DateTime.UtcNow - recordStartUtc).TotalSeconds >= MaxRecordSeconds;
        if (silenceDone || hardCap)
        {
            Task.Run(() =>
            {
                lock (stateLock) { if (isRecording) StopAndTranscribe(); }
            });
        }
    }

    static void StopAndTranscribe()
    {
        isRecording = false;
        waveSource?.StopRecording();
        waveSource?.Dispose();
        waveSource = null;

        byte[] pcm;
        lock (recordedBytes!) pcm = recordedBytes.ToArray();
        recordedBytes = null;

        float[] audio = Pcm16ToFloat(pcm);
        if (audio.Length < LogMel.SampleRate / 4)  // < 0.25 s
        {
            Console.WriteLine("[DONE] (too short)\n"); return;
        }

        var sw = System.Diagnostics.Stopwatch.StartNew();
        string text = transcriber!.Transcribe(audio).Trim();
        sw.Stop();

        Console.ForegroundColor = ConsoleColor.Green;
        Console.Write("[DONE] ");
        Console.ResetColor();
        Console.WriteLine($"({sw.ElapsedMilliseconds} ms, {audio.Length / (float)LogMel.SampleRate:F1} s audio)");

        if (text.Length == 0) { Console.WriteLine("(silence)\n"); return; }
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine(text);
        Console.ResetColor();
        PasteText(text);
        Console.WriteLine();
    }

    static float[] Pcm16ToFloat(byte[] pcm)
    {
        int n = pcm.Length / 2;
        float[] f = new float[n];
        for (int i = 0; i < n; i++)
        {
            short s = (short)(pcm[2 * i] | (pcm[2 * i + 1] << 8));
            f[i] = s / 32768f;
        }
        return f;
    }

    static void PasteText(string text)
    {
        // Put text on the clipboard, with a few retries since clipboard managers
        // can briefly own it.
        bool clipboardSet = false;
        for (int attempt = 0; attempt < 5 && !clipboardSet; attempt++)
        {
            if (OpenClipboard(IntPtr.Zero))
            {
                EmptyClipboard();
                byte[] bytes = Encoding.Unicode.GetBytes(text + "\0");
                IntPtr hGlobal = GlobalAlloc(GMEM_MOVEABLE, (UIntPtr)bytes.Length);
                if (hGlobal != IntPtr.Zero)
                {
                    IntPtr ptr = GlobalLock(hGlobal);
                    Marshal.Copy(bytes, 0, ptr, bytes.Length);
                    GlobalUnlock(hGlobal);
                    if (SetClipboardData(CF_UNICODETEXT, hGlobal) != IntPtr.Zero) clipboardSet = true;
                }
                CloseClipboard();
            }
            if (!clipboardSet) Thread.Sleep(20);
        }
        if (!clipboardSet) { Console.WriteLine("(clipboard unavailable; text not pasted)"); return; }

        // Release any modifier the user might still be holding from the Ctrl+Space
        // toggle. Unicode typing ignores modifiers, but a stray Ctrl-down can still
        // turn typed characters into shortcuts in some apps.
        var releases = new List<INPUT>();
        AddReleaseIfDown(releases, VK_LCONTROL);
        AddReleaseIfDown(releases, VK_RCONTROL);
        AddReleaseIfDown(releases, VK_LSHIFT);
        AddReleaseIfDown(releases, VK_RSHIFT);
        AddReleaseIfDown(releases, VK_LMENU);
        AddReleaseIfDown(releases, VK_RMENU);
        AddReleaseIfDown(releases, VK_LWIN);
        AddReleaseIfDown(releases, VK_RWIN);
        if (releases.Count > 0)
        {
            var arr = releases.ToArray();
            SendInput((uint)arr.Length, arr, Marshal.SizeOf<INPUT>());
        }
        Thread.Sleep(40);

        // Diagnostic: who's actually going to receive this?
        IntPtr fg = GetForegroundWindow();
        var title = new System.Text.StringBuilder(256);
        GetWindowText(fg, title, title.Capacity);
        GetWindowThreadProcessId(fg, out uint pid);
        string procName = "?";
        try { procName = System.Diagnostics.Process.GetProcessById((int)pid).ProcessName; } catch { }
        Console.WriteLine($"[paste] foreground: {procName} \"{title}\" (pid {pid})");

        // Type via KEYEVENTF_UNICODE. This hits the WM_CHAR path directly and works
        // in virtually every Windows text input, including apps that silently drop
        // synthetic Ctrl+V chords.
        Marshal.SetLastPInvokeError(0);
        uint sent = TypeUnicode(text);
        Console.WriteLine($"[paste] SendInput injected {sent}/{text.Length * 2} events");
        if (sent == 0)
        {
            int error = Marshal.GetLastPInvokeError();
            Console.WriteLine($"[paste] SendInput failed with Win32 error {error}. " +
                "If the target app is running as administrator, run VoiceHotkey as administrator too.");
        }
    }

    static uint TypeUnicode(string text)
    {
        var events = new List<INPUT>(text.Length * 2);
        foreach (char c in text)
        {
            if (c == '\n') { events.Add(UnicodeKey('\r', false)); events.Add(UnicodeKey('\r', true)); continue; }
            events.Add(UnicodeKey(c, false));
            events.Add(UnicodeKey(c, true));
        }
        const int batch = 100;
        int size = Marshal.SizeOf<INPUT>();
        uint total = 0;
        for (int i = 0; i < events.Count; i += batch)
        {
            int n = Math.Min(batch, events.Count - i);
            var arr = new INPUT[n];
            events.CopyTo(i, arr, 0, n);
            total += SendInput((uint)n, arr, size);
        }
        return total;
    }

    static INPUT UnicodeKey(char c, bool up) => new()
    {
        type = INPUT_KEYBOARD,
        u = new INPUTUNION
        {
            ki = new KEYBDINPUT
            {
                wVk = 0,
                wScan = c,
                dwFlags = KEYEVENTF_UNICODE | (up ? KEYEVENTF_KEYUP : 0u)
            }
        }
    };

    static void AddReleaseIfDown(List<INPUT> list, ushort vk)
    {
        if ((GetAsyncKeyState(vk) & 0x8000) != 0) list.Add(MakeScanKey(vk, true));
    }

    static INPUT MakeScanKey(ushort vk, bool up)
    {
        ushort scan = (ushort)MapVirtualKey(vk, MAPVK_VK_TO_VSC);
        uint flags = KEYEVENTF_SCANCODE | (up ? KEYEVENTF_KEYUP : 0u);
        // RWIN / RMENU / RCONTROL need the extended-key flag.
        if (vk == VK_RWIN || vk == VK_RMENU || vk == VK_RCONTROL) flags |= KEYEVENTF_EXTENDEDKEY;
        return new INPUT
        {
            type = INPUT_KEYBOARD,
            u = new INPUTUNION { ki = new KEYBDINPUT { wVk = vk, wScan = scan, dwFlags = flags } }
        };
    }
}
