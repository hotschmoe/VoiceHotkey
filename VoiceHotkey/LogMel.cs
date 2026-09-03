using MathNet.Numerics.IntegralTransforms;
using System.Numerics;

namespace VoiceHotkey;

/// Whisper-style log-mel spectrogram: 16 kHz PCM → (80, 3000) float32.
static class LogMel
{
    public const int SampleRate = 16000;
    public const int NFft = 400;
    public const int Hop = 160;
    public const int NMels = 80;
    public const int ChunkSeconds = 30;
    public const int NSamples = ChunkSeconds * SampleRate;   // 480_000
    public const int NFrames = NSamples / Hop;               // 3000
    public const int NBins = NFft / 2 + 1;                   // 201

    static readonly float[] HannWindow = BuildHann(NFft);
    static readonly float[,] MelFilter = BuildMelFilter();

    /// Compute log-mel from 30 s of mono float32 audio. Pads or trims to 30 s.
    public static float[] Compute(float[] audio)
    {
        float[] clip = new float[NSamples];
        int n = Math.Min(audio.Length, NSamples);
        Array.Copy(audio, clip, n);

        // Reflect-pad NFft/2 each side (matches torch.stft center=True).
        int pad = NFft / 2;
        float[] padded = new float[clip.Length + 2 * pad];
        Array.Copy(clip, 0, padded, pad, clip.Length);
        for (int i = 0; i < pad; i++)
        {
            padded[pad - 1 - i] = clip[i + 1];
            padded[pad + clip.Length + i] = clip[clip.Length - 2 - i];
        }

        int nFrames = 1 + (padded.Length - NFft) / Hop;
        // Power spectrogram: (nFrames, NBins)
        float[,] power = new float[nFrames, NBins];
        Complex[] fftBuf = new Complex[NFft];

        for (int t = 0; t < nFrames; t++)
        {
            int off = t * Hop;
            for (int i = 0; i < NFft; i++)
                fftBuf[i] = new Complex(padded[off + i] * HannWindow[i], 0);
            Fourier.Forward(fftBuf, FourierOptions.Matlab);  // no scaling, matches numpy rfft magnitudes
            for (int k = 0; k < NBins; k++)
            {
                float re = (float)fftBuf[k].Real;
                float im = (float)fftBuf[k].Imaginary;
                power[t, k] = re * re + im * im;
            }
        }

        // Apply mel filterbank: mel = MelFilter @ power.T  → (NMels, nFrames)
        int T = Math.Min(nFrames, NFrames);
        float[] mel = new float[NMels * NFrames];
        float maxLog = float.NegativeInfinity;
        for (int m = 0; m < NMels; m++)
        {
            for (int t = 0; t < T; t++)
            {
                float s = 0f;
                for (int k = 0; k < NBins; k++) s += MelFilter[m, k] * power[t, k];
                float l = MathF.Log10(MathF.Max(s, 1e-10f));
                mel[m * NFrames + t] = l;
                if (l > maxLog) maxLog = l;
            }
        }

        // Clamp to maxLog - 8 and normalize to (mel + 4) / 4
        float floor = maxLog - 8f;
        for (int i = 0; i < mel.Length; i++)
            mel[i] = (MathF.Max(mel[i], floor) + 4f) / 4f;

        return mel;
    }

    static float[] BuildHann(int n)
    {
        var w = new float[n];
        for (int i = 0; i < n; i++)
            w[i] = 0.5f - 0.5f * MathF.Cos(2f * MathF.PI * i / n);
        return w;
    }

    /// HTK mel filterbank: mel = 2595 * log10(1 + f / 700). Close enough to
    /// openai-whisper's Slaney mel for transcription accuracy.
    static float[,] BuildMelFilter()
    {
        static float HzToMel(float f) => 2595f * MathF.Log10(1f + f / 700f);
        static float MelToHz(float m) => 700f * (MathF.Pow(10f, m / 2595f) - 1f);

        float fmin = 0f, fmax = SampleRate / 2f;
        float mMin = HzToMel(fmin), mMax = HzToMel(fmax);
        var hzPts = new float[NMels + 2];
        for (int i = 0; i < NMels + 2; i++)
            hzPts[i] = MelToHz(mMin + (mMax - mMin) * i / (NMels + 1));

        var bins = new int[NMels + 2];
        for (int i = 0; i < NMels + 2; i++)
            bins[i] = (int)MathF.Floor((NFft + 1) * hzPts[i] / SampleRate);

        var fb = new float[NMels, NBins];
        for (int m = 1; m <= NMels; m++)
        {
            int l = bins[m - 1], c = bins[m], r = bins[m + 1];
            for (int k = l; k < c; k++)
                fb[m - 1, k] = (float)(k - l) / (c - l);
            for (int k = c; k < r; k++)
                fb[m - 1, k] = (float)(r - k) / (r - c);
        }
        return fb;
    }
}
