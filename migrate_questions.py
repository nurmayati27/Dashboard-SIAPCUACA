import os
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
from supabase import create_client, Client

# === KONFIG ===
SOURCE_FILE = "Copy of LIST PERTANYAAN.xlsx"
SHEET_NAME  = 0
BATCH_SIZE  = 500

load_dotenv()
SUPABASE_URL = os.environ["SUPABASE_URL"]
SERVICE_KEY  = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
sb: Client = create_client(SUPABASE_URL, SERVICE_KEY)

def norm_q(s: str) -> str:
    s = (s or "").strip().lower()
    return " ".join(s.split())

print("Membaca file Excel...")
df = pd.read_excel(SOURCE_FILE, sheet_name=SHEET_NAME)

# --- normalisasi header ---
df.columns = [c.strip().lower() for c in df.columns]

# --- drop kolom sisa merge ---
drop_cols = [c for c in df.columns if c.startswith("unnamed")]
df.drop(columns=drop_cols, inplace=True, errors="ignore")

# --- rename kolom konten -> schema DB (PAKAI lowercase!) ---
colmap = {
    "list pertanyaan": "question",
    "jawaban": "answer",
    # kolom 'tanggal' TIDAK di-rename; kita deteksi fleksibel di bawah
}
for src, dst in colmap.items():
    if src in df.columns:
        df.rename(columns={src: dst}, inplace=True)

if "question" not in df.columns:
    raise SystemExit("Kolom 'list pertanyaan' tidak ditemukan → 'question' wajib ada.")

# --- bersihkan teks & buang baris kosong ---
df["question"] = df["question"].astype(str).str.strip()
df["answer"]   = df.get("answer", "").astype(str).str.strip()
df = df[df["question"].str.len() > 0].copy()

# === DETEKSI KOLUM TANGGAL + FFILL (atasi Merge & Center) ===
tcol = next((c for c in df.columns if "tanggal" in c), None)
if not tcol:
    raise SystemExit("Kolom 'tanggal' tidak ditemukan di Excel.")

df[tcol] = df[tcol].ffill()

def parse_any_date(v):
    try:
        if pd.isna(v):
            return None
        # Excel serial besar (tanggal)
        if isinstance(v, (int, float)) and v > 40000:
            return pd.Timestamp("1899-12-30") + pd.to_timedelta(v, "D")
        return pd.to_datetime(str(v), errors="coerce")
    except Exception:
        return None

dt = df[tcol].apply(parse_any_date)
df["session_time"] = pd.to_datetime(dt, errors="coerce", utc=True)

# 🔍 Debug print untuk cek hasil ffill dan konversi tanggal
print("\n=== DEBUG: Cek hasil pembacaan tanggal ===")
print(df[[tcol, "session_time", "question"]].head(30))
print("=== (tampilkan 30 baris pertama) ===\n")

# === DEDUP client-side: (normalize(question), date(session_time)) ===
print("Mengambil data existing dari DB untuk dedup...")
existing = sb.table("questions").select("question,session_time").execute()
have = set()
for row in (existing.data or []):
    q = norm_q(row.get("question", ""))
    st = row.get("session_time")
    d = None
    if st:
        try:
            d = datetime.fromisoformat(st.replace("Z","+00:00")).date()
        except Exception:
            d = None
    have.add((q, d))

def key_from_row(q, st):
    qn = norm_q(q)
    d = None
    if pd.notna(st):
        try:
            d = pd.to_datetime(st).date()
        except Exception:
            d = None
    return (qn, d)

df["__key__"] = [key_from_row(q, st) for q, st in zip(df["question"], df["session_time"])]
df_new = df[~df["__key__"].isin(have)].copy().drop(columns=["__key__"], errors="ignore")

print(f"Total baris di file: {len(df)}")
print(f"Baris baru (non-duplikat vs DB): {len(df_new)}")
if df_new.empty:
    print("Tidak ada yang perlu di-insert. Selesai.")
    raise SystemExit(0)

# === Sanitizer waktu (ISO 8601 atau None) ===
def to_iso_or_none(v):
    try:
        ts = pd.to_datetime(v, errors="coerce", utc=True)
    except Exception:
        return None
    if ts is None or pd.isna(ts):
        return None
    return ts.to_pydatetime().isoformat()  # "YYYY-MM-DDTHH:MM:SS+00:00"

# === Build payload: kirim 'session_time' HANYA jika ada ===
records = []
for _, row in df_new.iterrows():
    rec = {
        "question": row["question"],
        "answer": row["answer"],
    }
    iso = to_iso_or_none(row.get("session_time"))
    if iso is not None:
        rec["session_time"] = iso
    records.append(rec)

# Guardrail: pastikan tak ada "NaT"
assert all(r.get("session_time") != "NaT" for r in records if "session_time" in r)

print("records with None/no session_time:", sum("session_time" not in r for r in records))

# === INSERT batch ===
inserted = 0
for i in range(0, len(records), BATCH_SIZE):
    chunk = records[i:i+BATCH_SIZE]
    sb.table("questions").insert(chunk, count="exact").execute()
    inserted += len(chunk)
    print(f"✔️ Insert {inserted}/{len(records)}")

print("🎉 Migrasi selesai.")
