using System.Text;
using System.Text.Json;

namespace VoiceHotkey;

/// Decode-only Whisper BPE tokenizer. Loads HuggingFace vocab.json +
/// added_tokens.json for openai/whisper-small and converts token IDs → text.
/// Encoding is not needed (we only emit IDs from the decoder).
class WhisperTokenizer
{
    readonly Dictionary<int, string> _idToTok = new();
    readonly HashSet<int> _specialIds = new();
    readonly Dictionary<char, byte> _charToByte;

    public readonly int StartOfTranscript;
    public readonly int LanguageEn;
    public readonly int Transcribe;
    public readonly int NoTimestamps;
    public readonly int EndOfText;

    public int[] PrefixTokens => new[] { StartOfTranscript, LanguageEn, Transcribe, NoTimestamps };

    public WhisperTokenizer(string tokenizerDir)
    {
        _charToByte = BuildByteDecoder();

        LoadVocab(Path.Combine(tokenizerDir, "vocab.json"));
        LoadVocab(Path.Combine(tokenizerDir, "added_tokens.json"), markSpecial: true);

        StartOfTranscript = LookupSpecial("<|startoftranscript|>");
        LanguageEn = LookupSpecial("<|en|>");
        Transcribe = LookupSpecial("<|transcribe|>");
        NoTimestamps = LookupSpecial("<|notimestamps|>");
        EndOfText = LookupSpecial("<|endoftext|>");
    }

    void LoadVocab(string path, bool markSpecial = false)
    {
        using var stream = File.OpenRead(path);
        using var doc = JsonDocument.Parse(stream);
        foreach (var prop in doc.RootElement.EnumerateObject())
        {
            int id = prop.Value.GetInt32();
            _idToTok[id] = prop.Name;
            if (markSpecial) _specialIds.Add(id);
        }
    }

    int LookupSpecial(string name)
    {
        foreach (var kv in _idToTok)
            if (kv.Value == name) return kv.Key;
        throw new InvalidDataException($"Missing special token: {name}");
    }

    public string Decode(IEnumerable<int> ids)
    {
        var sb = new StringBuilder();
        foreach (int id in ids)
        {
            if (_specialIds.Contains(id)) continue;
            if (_idToTok.TryGetValue(id, out var tok)) sb.Append(tok);
        }
        string joined = sb.ToString();

        var bytes = new List<byte>(joined.Length);
        foreach (char c in joined)
        {
            if (_charToByte.TryGetValue(c, out byte b)) bytes.Add(b);
            else bytes.AddRange(Encoding.UTF8.GetBytes(new[] { c }));
        }
        return Encoding.UTF8.GetString(bytes.ToArray());
    }

    /// GPT-2 bytes_to_unicode inverse: map display char → raw byte.
    static Dictionary<char, byte> BuildByteDecoder()
    {
        var bs = new List<int>();
        for (int b = '!'; b <= '~'; b++) bs.Add(b);
        for (int b = '¡'; b <= '¬'; b++) bs.Add(b);
        for (int b = '®'; b <= 'ÿ'; b++) bs.Add(b);
        var cs = new List<int>(bs);
        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            if (bs.Contains(b)) continue;
            bs.Add(b);
            cs.Add(256 + n);
            n++;
        }
        var result = new Dictionary<char, byte>();
        for (int i = 0; i < bs.Count; i++)
            result[(char)cs[i]] = (byte)bs[i];
        return result;
    }
}
