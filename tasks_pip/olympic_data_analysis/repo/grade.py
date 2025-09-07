import pandas as pd
from preprocesser import preprocess
import helper

# tiny synthetic dataset
events = pd.DataFrame([
    {"NOC":"USA","Team":"USA","Season":"Summer","Year":2000,"City":"Sydney","Sport":"Athletics","Event":"100m","Name":"A","Medal":"Gold"},
    {"NOC":"USA","Team":"USA","Season":"Summer","Year":2004,"City":"Athens","Sport":"Swimming","Event":"200m","Name":"B","Medal":"Silver"},
    {"NOC":"FRA","Team":"France","Season":"Summer","Year":2004,"City":"Athens","Sport":"Judo","Event":"-66kg","Name":"C","Medal":"Bronze"},
    {"NOC":"FRA","Team":"France","Season":"Winter","Year":2006,"City":"Turin","Sport":"Skiing","Event":"Downhill","Name":"D","Medal":None},
])
regions = pd.DataFrame([
    {"NOC":"USA","region":"USA"},
    {"NOC":"FRA","region":"France"},
])

# preprocess should filter Summer, merge region, add dummy medal cols, dedupe
df = preprocess(events.copy(), regions.copy())
assert set(["Gold","Silver","Bronze"]).issubset(df.columns)
assert "region" in df.columns
assert (df["Season"] == "Summer").all()

# helper sanity checks
years, countries = helper.get_country_year(df)
assert years[0] == "Overall" and "Overall" in countries

over_time = helper.data_over_time(df, "Event", "events")
assert {"year","events"} <= set(over_time.columns)

tally_all = helper.fetch_medal_tally(df, "Overall", "Overall")
assert {"Gold","Silver","Bronze","total"}.issubset(tally_all.columns)

print("OK")