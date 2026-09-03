using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace VoiceHotkey;

/// Whisper-small transcription on the Hexagon NPU via ORT QNN EP.
/// Loads encoder + decoder ONNX (or cached QNN context .onnx_ctx.onnx),
/// runs mel → encoder → greedy decoder loop → BPE text.
class Transcriber : IDisposable
{
    const int NLayer = 12, NHead = 12, NStateDim = 64;  // n_state / n_head
    const int NAudioCtx = 1500;
    const int NTextCtx = 200;

    readonly InferenceSession _encoder;
    readonly InferenceSession _decoder;
    readonly WhisperTokenizer _tokenizer;
    readonly string[] _encoderOutputNames;
    readonly string[] _decoderOutputNames;

    public Transcriber(string encoderOnnx, string decoderOnnx, string cacheDir, string tokenizerDir)
    {
        _tokenizer = new WhisperTokenizer(tokenizerDir);
        Directory.CreateDirectory(cacheDir);

        string exeDir = AppContext.BaseDirectory;
        string epLib = Path.Combine(exeDir, "onnxruntime_providers_qnn.dll");
        string backend = Path.Combine(exeDir, "QnnHtp.dll");

        var env = OrtEnv.Instance();
        try { env.RegisterExecutionProviderLibrary("QNNExecutionProvider", epLib); }
        catch (OnnxRuntimeException) { /* already registered */ }

        OrtEpDevice? npu = null;
        foreach (var d in env.GetEpDevices())
            if (d.EpName == "QNNExecutionProvider" && d.HardwareDevice.Type == OrtHardwareDeviceType.NPU)
            { npu = d; break; }
        if (npu == null) throw new InvalidOperationException("No QNN NPU device found.");

        _encoder = LoadSession(encoderOnnx, cacheDir, env, npu, backend);
        _decoder = LoadSession(decoderOnnx, cacheDir, env, npu, backend);
        _encoderOutputNames = _encoder.OutputMetadata.Keys.ToArray();
        _decoderOutputNames = _decoder.OutputMetadata.Keys.ToArray();
    }

    static InferenceSession LoadSession(string onnxPath, string cacheDir, OrtEnv env, OrtEpDevice npu, string backend)
    {
        string ctxPath = Path.Combine(cacheDir,
            Path.GetFileNameWithoutExtension(onnxPath) + "_ctx.onnx");
        bool useCache = File.Exists(ctxPath);

        var so = new SessionOptions();
        if (!useCache)
        {
            so.AddSessionConfigEntry("ep.context_enable", "1");
            so.AddSessionConfigEntry("ep.context_file_path", ctxPath);
            // ORT QNN 1.24.4 generates an unloadable embedded cache when a
            // model is split into multiple QNN partitions (microsoft/
            // onnxruntime#31977). External mode emits one companion .bin file
            // and reloads correctly.
            so.AddSessionConfigEntry("ep.context_embed_mode", "0");
        }
        var epOptions = new Dictionary<string, string>
        {
            { "backend_path", backend },
            { "htp_performance_mode", "burst" },
        };
        so.AppendExecutionProvider(env, new[] { npu }, epOptions);

        string loadPath = useCache ? ctxPath : onnxPath;
        Console.WriteLine($"  {(useCache ? "cached" : "fresh (will JIT + cache)")}: {Path.GetFileName(loadPath)}");
        return new InferenceSession(loadPath, so);
    }

    /// Transcribe up to 30 s of 16 kHz mono float32 audio.
    public string Transcribe(float[] audio)
    {
        float[] mel = LogMel.Compute(audio);
        using var melTensor = OrtValue.CreateTensorValueFromMemory(mel,
            new long[] { 1, LogMel.NMels, LogMel.NFrames });

        var encInputs = new Dictionary<string, OrtValue> { { "input_features", melTensor } };
        using var encResults = _encoder.Run(new RunOptions(), encInputs, _encoderOutputNames);
        var kCross = encResults[0];  // (12, 12, 64, 1500)
        var vCross = encResults[1];  // (12, 12, 1500, 64)

        // Self-cache buffers (reused across decoder steps)
        int kSelfLen = NLayer * NHead * NStateDim * NTextCtx;
        int vSelfLen = NLayer * NHead * NTextCtx * NStateDim;
        float[] kSelf = new float[kSelfLen];
        float[] vSelf = new float[vSelfLen];

        int[] prefix = _tokenizer.PrefixTokens;
        var tokens = new List<int>(prefix);

        float[] xBuf = new float[1];   // reused as int32 via reinterpret
        float[] offsetBuf = new float[1];
        long[] kSelfShape = { NLayer, NHead, NStateDim, NTextCtx };
        long[] vSelfShape = { NLayer, NHead, NTextCtx, NStateDim };

        int[] xInt = new int[1];
        int[] offsetInt = new int[1];

        float[]? lastLogits = null;

        // Prime cache with all prefix tokens; last call's logits predicts the first content token.
        for (int i = 0; i < prefix.Length; i++)
        {
            lastLogits = RunDecoderStep(
                prefix[i], i, kCross, vCross,
                kSelf, vSelf, kSelfShape, vSelfShape,
                xInt, offsetInt);
        }

        // Greedy sampling
        while (tokens.Count < NTextCtx)
        {
            int next = ArgMax(lastLogits!);
            if (next == _tokenizer.EndOfText) break;
            tokens.Add(next);
            lastLogits = RunDecoderStep(
                next, tokens.Count - 1, kCross, vCross,
                kSelf, vSelf, kSelfShape, vSelfShape,
                xInt, offsetInt);
        }

        return _tokenizer.Decode(tokens.Skip(prefix.Length));
    }

    float[] RunDecoderStep(
        int tokenId, int position,
        OrtValue kCross, OrtValue vCross,
        float[] kSelf, float[] vSelf,
        long[] kSelfShape, long[] vSelfShape,
        int[] xBuf, int[] offsetBuf)
    {
        xBuf[0] = tokenId;
        offsetBuf[0] = position;

        using var xT = OrtValue.CreateTensorValueFromMemory(xBuf, new long[] { 1 });
        using var offsetT = OrtValue.CreateTensorValueFromMemory(offsetBuf, new long[] { 1 });
        using var kSelfT = OrtValue.CreateTensorValueFromMemory(kSelf, kSelfShape);
        using var vSelfT = OrtValue.CreateTensorValueFromMemory(vSelf, vSelfShape);

        var inputs = new Dictionary<string, OrtValue>
        {
            { "x", xT }, { "offset", offsetT },
            { "k_cache_cross", kCross }, { "v_cache_cross", vCross },
            { "k_cache_self", kSelfT }, { "v_cache_self", vSelfT },
        };

        using var results = _decoder.Run(new RunOptions(), inputs, _decoderOutputNames);
        var logitsTensor = results[0];
        var kNew = results[1];
        var vNew = results[2];

        // Copy updated self-cache back into our persistent buffers
        kNew.GetTensorDataAsSpan<float>().CopyTo(kSelf);
        vNew.GetTensorDataAsSpan<float>().CopyTo(vSelf);

        return logitsTensor.GetTensorDataAsSpan<float>().ToArray();
    }

    static int ArgMax(float[] a)
    {
        int best = 0; float max = a[0];
        for (int i = 1; i < a.Length; i++)
            if (a[i] > max) { max = a[i]; best = i; }
        return best;
    }

    public void Dispose()
    {
        _encoder.Dispose();
        _decoder.Dispose();
    }
}
