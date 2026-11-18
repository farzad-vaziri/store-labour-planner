"""Download Leeds City Council 'Revised Weekly Data' footfall CSVs from Data Mill North,
UK bank holidays (gov.uk) and Open-Meteo hourly weather for Leeds.
Usage: python scripts/01_download.py <raw_dir>
Raw files stay out of git."""
import re, sys, json, time, urllib.parse, urllib.request, pathlib

RAW = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "raw")
PAGE = "https://datamillnorth.org/dataset/leeds-city-centre-footfall-data"

def get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "store-labour-planner/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

def main():
    (RAW / "footfall").mkdir(parents=True, exist_ok=True)
    html = get(PAGE).decode("utf-8", "ignore")
    urls = sorted(set(re.findall(r'https://datamillnorth\.org/download/2rlld/[^"\'\s<]+', html)))
    revised = [u for u in urls if "revised" in urllib.parse.unquote(u).lower() and u.endswith(".csv")]
    print(f"{len(revised)} revised weekly files")
    for u in revised:
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", urllib.parse.unquote(u.split("/")[-1]))
        out = RAW / "footfall" / f"{u.split('/')[-2]}_{name}"
        if out.exists() and out.stat().st_size > 1000:
            continue
        for attempt in range(3):
            try:
                out.write_bytes(get(u)); break
            except Exception as e:
                print("retry", u, e); time.sleep(2)
    (RAW / "bank_holidays.json").write_bytes(get("https://www.gov.uk/bank-holidays.json"))
    import datetime as dt
    start, end = "2023-10-01", (dt.date.today() - dt.timedelta(days=6)).isoformat()
    w = (f"https://archive-api.open-meteo.com/v1/archive?latitude=53.7997&longitude=-1.5492"
         f"&start_date={start}&end_date={end}&hourly=temperature_2m,precipitation,cloud_cover,wind_speed_10m"
         "&timezone=Europe%2FLondon")
    (RAW / "weather_hourly.json").write_bytes(get(w, 120))
    print("done")

if __name__ == "__main__":
    main()
