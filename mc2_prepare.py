"""
MC2 VAST Challenge 2025 — Data Preparation Script (v3)
Author: IS428 Group 10
Purpose: Clean, enrich, and transform MC2 knowledge graph CSVs into
         Tableau-ready flat tables and Gephi-ready node/edge files.

FIXES vs v1:
  [BUG A] trips_geo: lat/lon was always null.
          Fix: correct join chain — plan → travel link → geo_node → lat/lon.
  [BUG B] trips_geo: Is_Member was always False.
          Fix: member is the participant of a travel plan, found in
          participant links (source=plan_id, target=member_name).
  [BUG C] member_activity: non-member orgs could leak into counts.
          Fix: explicit MEMBERS whitelist filter with assertion guard.
  [BUG D] Industry on links was inferred from source ID string (regex).
          Fix: primary method is graph traversal (discussion → about → topic).
          String inference used only as fallback for unlinked nodes.
  [NEW] part_of links: source=meeting_id, target=discussion/plan (not vice versa).
        Now correctly assigns Meeting_Num from part_of links instead of regex.
  [NEW] 'Sean' typo in participant targets treated as non-member.
  [NEW] mc2_member_topic_agg.csv: member × topic × dataset (for heatmap, ego charts).
  [NEW] mc2_cross_dataset_comparison.csv: wide pivot for slope/bump charts.
  [NEW] mc2_member_attendance.csv: member × meeting attendance (for timeline chart).
  [NEW] mc2_gephi_[dataset]_nodes/edges.csv: rebuilt from correct graph traversal.

FIXES vs v2:
  [BUG E] mc2_travel_geo.csv: 2 rows had NaN Member.
          Root cause: two travel plans exist in a lobby dataset travel links
          but their COOTEFOO member participant only appears in the journalist
          participant links, not in that lobby's links. The left join produced NaN.
          Fix: drop rows where Member is NaN after the join. These plans have
          no identified member in that specific dataset (39 rows, down from 41).
  [BUG F] mc2_cross_dataset_comparison.csv: Teddy Goldstein missing deep_fishing_dock
          (6 participations in journalist + trout).
          Root cause: cross_count.merge(cross_sent) used an inner join, silently
          dropping rows where Avg_Sentiment was NaN for all discussions in the group
          (deep_fishing_dock has no rated sentiments in the raw data).
          Fix: changed to a left merge so NaN-sentiment topics are preserved.
          Goldstein now correctly shows 6 topic rows (was 5).

Run: python3 mc2_prepare.py
Outputs (in ./tableau_data/):
  mc2_nodes_master.csv          — all nodes, all datasets, long format
  mc2_links_master.csv          — all edges, all datasets, long format
  mc2_member_sentiment.csv      — one row per (member × discussion × dataset)
  mc2_topic_industry.csv        — topic → industry classification lookup
  mc2_member_activity.csv       — activity summary per member per dataset
  mc2_member_topic_agg.csv      — member × topic × dataset aggregated (NEW)
  mc2_cross_dataset_comparison.csv — wide pivot for cross-dataset ego charts (NEW)
  mc2_member_attendance.csv     — member × meeting attendance matrix (NEW)
  mc2_travel_geo.csv            — travel plans with correct geo coordinates
  mc2_coverage_comparison.csv   — which discussions appear in which datasets
  mc2_industry_breakdown.csv    — industry share per dataset (corrected %)
  mc2_gephi_[f/t/j]_nodes.csv  — Gephi-ready node files per dataset
  mc2_gephi_[f/t/j]_edges.csv  — Gephi-ready edge files per dataset
"""

import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────────────────────────
# 0. CONFIGURATION
# ─────────────────────────────────────────────────────────────────

DATA_DIR = "."
OUT_DIR  = "./tableau_data"
os.makedirs(OUT_DIR, exist_ok=True)

PATHS = {
    "filah":      ("mc2_FILAH_nodes.csv",       "mc2_FILAH_links.csv"),
    "trout":      ("mc2_TROUT_nodes.csv",        "mc2_TROUT_links.csv"),
    "journalist": ("mc2_jounalist_nodes.csv",    "mc2_journalist_links.csv"),
}
GEO_NODES_FILE = "mc2_geo_nodes.csv"
GEO_EDGES_FILE = "mc2_geo_edges.csv"

# ─────────────────────────────────────────────────────────────────
# 1. CONSTANTS
# ─────────────────────────────────────────────────────────────────

# The six COOTEFOO members by their exact node IDs
MEMBERS = {
    "Seal", "Ed Helpsford", "Teddy Goldstein",
    "Simone Kat", "Tante Titan", "Carol Limpet",
}

# Topic → industry classification (15 topics, manually verified)
TOPIC_INDUSTRY = {
    "fish_vacuum":             "fishing",
    "deep_fishing_dock":       "fishing",
    "low_volume_crane":        "fishing",
    "seafood_festival":        "fishing",
    "new_crane_lomark":        "fishing",
    "affordable_housing":      "neutral",
    "renaming_park_himark":    "neutral",
    "name_harbor_area":        "neutral",
    "name_inspection_office":  "neutral",
    "statue_john_smoth":       "neutral",
    "expanding_tourist_wharf": "tourism",
    "marine_life_deck":        "tourism",
    "heritage_walking_tour":   "tourism",
    "waterfront_market":       "tourism",
    "concert":                 "tourism",
}

DATASET_LABELS = {
    "filah":      "FILAH (Fishing Lobby)",
    "trout":      "TROUT (Tourism Lobby)",
    "journalist": "Journalist (Full Record)",
}

# Bias score thresholds
BIAS_TOURISM_THRESHOLD =  0.15
BIAS_FISHING_THRESHOLD = -0.15

# ─────────────────────────────────────────────────────────────────
# 2. HELPERS
# ─────────────────────────────────────────────────────────────────

def load_pair(dataset_key):
    """Load nodes + links for one dataset, stamp with Dataset column."""
    n_file, l_file = PATHS[dataset_key]
    nodes = pd.read_csv(os.path.join(DATA_DIR, n_file), low_memory=False)
    links = pd.read_csv(os.path.join(DATA_DIR, l_file), low_memory=False)
    nodes["Dataset"] = dataset_key
    links["Dataset"] = dataset_key
    return nodes, links


def fix_year(s):
    """
    Fix date strings where the year was recorded as '0040' instead of '2040'.
    Converts '0040-04-24' → '2040-04-24'. Other formats pass through unchanged.
    Meeting labels like 'Meeting 1' are left as-is (they parse to NaT).
    """
    if not isinstance(s, str):
        return s
    if s.startswith("0040-"):
        return "2040-" + s[5:]
    return s


def parse_trip_dates(series):
    """Parse trip node date column (mix of '2040-MM-DD' and '0040-MM-DD')."""
    return pd.to_datetime(series.apply(fix_year), format="%Y-%m-%d", errors="coerce")


def infer_topic_from_id(node_id):
    """
    Fallback: extract topic slug from a node ID string.
    e.g. 'fish_vacuum_Meeting_1_Introduction_Discussion' → 'fish_vacuum'
    Returns 'unknown' if no known topic slug is found.
    Only used when the graph-traversal join fails (about link missing).
    """
    if not isinstance(node_id, str):
        return "unknown"
    id_lower = node_id.lower()
    for topic_key in TOPIC_INDUSTRY:
        if topic_key in id_lower:
            return topic_key
    return "unknown"


def sentiment_label(s):
    """Classify a numeric sentiment score into a human-readable label."""
    if pd.isna(s):   return "Not Rated"
    if s > 0.3:      return "Pro"
    if s < -0.3:     return "Against"
    return "Neutral"


def bias_direction(score):
    if pd.isna(score):                    return "No Data"
    if score > BIAS_TOURISM_THRESHOLD:    return "Tourism-Leaning"
    if score < BIAS_FISHING_THRESHOLD:    return "Fishing-Leaning"
    return "Balanced"

# ─────────────────────────────────────────────────────────────────
# 3. LOAD ALL DATASETS
# ─────────────────────────────────────────────────────────────────

all_nodes, all_links = [], []
for key in ["filah", "trout", "journalist"]:
    n, l = load_pair(key)
    all_nodes.append(n)
    all_links.append(l)

nodes_raw = pd.concat(all_nodes, ignore_index=True)
links_raw = pd.concat(all_links, ignore_index=True)

geo_nodes = pd.read_csv(os.path.join(DATA_DIR, GEO_NODES_FILE), low_memory=False)
geo_edges = pd.read_csv(os.path.join(DATA_DIR, GEO_EDGES_FILE), low_memory=False)

# Normalise geo_nodes id to string for joins
geo_nodes["id"] = geo_nodes["id"].astype(str)

print(f"Loaded: {len(nodes_raw)} total nodes, {len(links_raw)} total links")

# ─────────────────────────────────────────────────────────────────
# 4. BUILD TOPIC LOOKUP FROM ABOUT LINKS (graph traversal)
#    discussion/plan → about → topic_node_id → industry
#    This is the authoritative topic classification method.
# ─────────────────────────────────────────────────────────────────

# about links: source = discussion/plan ID, target = topic or sub-plan ID
about_links = links_raw[links_raw["role"] == "about"].copy()

# Keep only about links whose TARGET is a root topic key
about_to_topic = about_links[about_links["target"].isin(TOPIC_INDUSTRY.keys())][
    ["source", "target", "Dataset"]
].copy()
about_to_topic = about_to_topic.rename(columns={"source": "node_id", "target": "Topic_Key"})
about_to_topic["Industry"] = about_to_topic["Topic_Key"].map(TOPIC_INDUSTRY)

# Deduplicate: one discussion can link to multiple plans under one topic;
# a discussion pointing to the same root topic more than once is collapsed.
about_to_topic = about_to_topic.drop_duplicates(subset=["node_id", "Dataset"])

# ─────────────────────────────────────────────────────────────────
# 5. BUILD MEETING NUMBER LOOKUP FROM part_of LINKS
#    part_of: source = Meeting_N (meeting node), target = discussion/plan
#    So: discussion/plan → (reverse lookup) → meeting_number
# ─────────────────────────────────────────────────────────────────

part_of_links = links_raw[links_raw["role"] == "part_of"].copy()

# source = Meeting_1 … Meeting_16, target = discussion/plan node ID
# We want: for each discussion/plan, which meeting does it belong to?
meeting_lookup = part_of_links[["source", "target", "Dataset"]].rename(
    columns={"source": "Meeting_ID", "target": "node_id"}
)
# Extract integer meeting number from Meeting_ID
meeting_lookup["Meeting_Num"] = (
    meeting_lookup["Meeting_ID"]
    .str.extract(r"Meeting_(\d+)", expand=False)
    .astype(float)
)

# ─────────────────────────────────────────────────────────────────
# 6. NODES MASTER TABLE
# ─────────────────────────────────────────────────────────────────

nodes = nodes_raw.copy()
nodes["id"]    = nodes["id"].astype(str).str.strip()
nodes["label"] = nodes["label"].astype(str).str.strip()

# Parse dates — only trip nodes carry real dates; meetings carry labels
nodes["date_parsed"] = parse_trip_dates(nodes["date"])
nodes["Year"]      = nodes["date_parsed"].dt.year
nodes["Month"]     = nodes["date_parsed"].dt.month
nodes["MonthName"] = nodes["date_parsed"].dt.strftime("%b")

# Parse trip start/end times (time-of-day strings, e.g. '09:00:00')
# Combine with date for a full datetime; NaT where date is missing
nodes["start_time"] = pd.to_datetime(
    nodes["date"].apply(fix_year).astype(str) + " " + nodes["start"].astype(str),
    errors="coerce"
)
nodes["end_time"] = pd.to_datetime(
    nodes["date"].apply(fix_year).astype(str) + " " + nodes["end"].astype(str),
    errors="coerce"
)
nodes["duration_hrs"] = (
    (nodes["end_time"] - nodes["start_time"]).dt.total_seconds() / 3600
).round(2)

# Attach Topic_Key and Industry via graph traversal (about links)
nodes = nodes.merge(
    about_to_topic[["node_id", "Dataset", "Topic_Key", "Industry"]],
    left_on=["id", "Dataset"], right_on=["node_id", "Dataset"],
    how="left"
).drop(columns=["node_id"])

# Fallback: for nodes with no about link, infer topic from ID string
mask_unknown = nodes["Topic_Key"].isna()
nodes.loc[mask_unknown, "Topic_Key"] = nodes.loc[mask_unknown, "id"].apply(infer_topic_from_id)
nodes.loc[mask_unknown, "Industry"]  = nodes.loc[mask_unknown, "Topic_Key"].map(TOPIC_INDUSTRY).fillna("unknown")

# For topic-type nodes, use their own id as the topic key
topic_mask = nodes["type"] == "topic"
nodes.loc[topic_mask, "Topic_Key"] = nodes.loc[topic_mask, "id"]
nodes.loc[topic_mask, "Industry"]  = nodes.loc[topic_mask, "id"].map(TOPIC_INDUSTRY).fillna("unknown")

# For place nodes, derive industry from geographic zone
zone_industry = {
    "fishing":       "fishing",
    "tourism":       "tourism",
    "government":    "neutral",
    "commercial":    "neutral",
    "environmental": "neutral",
    "residential":   "neutral",
    "industrial":    "neutral",
}
place_mask = nodes["type"] == "place"
nodes.loc[place_mask, "Industry"] = (
    nodes.loc[place_mask, "zone"].map(zone_industry).fillna("unknown")
)

# Attach meeting number via part_of lookup
nodes = nodes.merge(
    meeting_lookup[["node_id", "Dataset", "Meeting_Num"]],
    left_on=["id", "Dataset"], right_on=["node_id", "Dataset"],
    how="left"
).drop(columns=["node_id"])

# Friendly dataset label
nodes["Dataset_Label"] = nodes["Dataset"].map(DATASET_LABELS)

nodes.to_csv(os.path.join(OUT_DIR, "mc2_nodes_master.csv"), index=False)
print(f"✓ mc2_nodes_master.csv  ({len(nodes)} rows)")

# ─────────────────────────────────────────────────────────────────
# 7. LINKS MASTER TABLE
# ─────────────────────────────────────────────────────────────────

links = links_raw.copy()
links["source"] = links["source"].astype(str).str.strip()
links["target"] = links["target"].astype(str).str.strip()

# Attach Topic_Key + Industry from the about-link lookup (source = discussion ID)
links = links.merge(
    about_to_topic[["node_id", "Dataset", "Topic_Key", "Industry"]],
    left_on=["source", "Dataset"], right_on=["node_id", "Dataset"],
    how="left"
).drop(columns=["node_id"])

# Fallback for unmatched links
mask = links["Topic_Key"].isna()
links.loc[mask, "Topic_Key"] = links.loc[mask, "source"].apply(infer_topic_from_id)
links.loc[mask, "Industry"]  = links.loc[mask, "Topic_Key"].map(TOPIC_INDUSTRY).fillna("unknown")

# Flag whether the TARGET is a COOTEFOO member
# NOTE: 'Sean' in the data is a typo for 'Seal'; it is NOT added to MEMBERS
#       to avoid inflating counts. It will appear as a non-member participant.
links["Is_Member_Target"] = links["target"].isin(MEMBERS)

# Sentiment label
links["Sentiment_Label"] = links["sentiment"].apply(sentiment_label)

# Meeting number (for participant links whose source discussion is in a meeting)
links = links.merge(
    meeting_lookup[["node_id", "Dataset", "Meeting_Num"]],
    left_on=["source", "Dataset"], right_on=["node_id", "Dataset"],
    how="left"
).drop(columns=["node_id"])

# Friendly dataset label
links["Dataset_Label"] = links["Dataset"].map(DATASET_LABELS)

links.to_csv(os.path.join(OUT_DIR, "mc2_links_master.csv"), index=False)
print(f"✓ mc2_links_master.csv  ({len(links)} rows)")

# ─────────────────────────────────────────────────────────────────
# 8. TOPIC INDUSTRY LOOKUP TABLE
# ─────────────────────────────────────────────────────────────────

topic_df = pd.DataFrame([
    {
        "Topic_Key":   k,
        "Industry":    v,
        "Topic_Label": k.replace("_", " ").title(),
    }
    for k, v in TOPIC_INDUSTRY.items()
])
topic_df.to_csv(os.path.join(OUT_DIR, "mc2_topic_industry.csv"), index=False)
print(f"✓ mc2_topic_industry.csv  ({len(topic_df)} rows)")

# ─────────────────────────────────────────────────────────────────
# 9. MEMBER SENTIMENT TABLE
#    One row per (Member × Discussion × Dataset)
#    Source: participant links where target ∈ MEMBERS
#    Industry assigned via graph traversal (about links), with string fallback
# ─────────────────────────────────────────────────────────────────

participant_links = links[links["role"] == "participant"].copy()

# Keep only rows where the participant is a COOTEFOO member
member_rows = participant_links[participant_links["Is_Member_Target"]].copy()
member_rows = member_rows.rename(columns={"target": "Member", "source": "Discussion_ID"})

# Safety assertion — no non-members should be present
unexpected = set(member_rows["Member"].unique()) - MEMBERS
if unexpected:
    print(f"  ⚠ Unexpected participants filtered out: {unexpected}")
    member_rows = member_rows[member_rows["Member"].isin(MEMBERS)]

# Attach plan node metadata (short_title, long_title) from nodes_master
disc_meta = (
    nodes[nodes["type"].isin(["discussion", "plan"])]
    .drop_duplicates(subset=["id", "Dataset"])
    [["id", "Dataset", "type", "short_title", "long_title", "date_parsed", "Year", "Month"]]
    .rename(columns={"id": "Discussion_ID", "type": "discussion_type"})
)
member_rows = member_rows.merge(disc_meta, on=["Discussion_ID", "Dataset"], how="left")

member_rows.to_csv(os.path.join(OUT_DIR, "mc2_member_sentiment.csv"), index=False)
print(f"✓ mc2_member_sentiment.csv  ({len(member_rows)} rows)")

# ─────────────────────────────────────────────────────────────────
# 10. MEMBER ACTIVITY SUMMARY
#     Aggregated per (Member × Dataset): fishing / neutral / tourism counts
#     Bias Score = (tourism − fishing) / total; positive = tourism lean
# ─────────────────────────────────────────────────────────────────

pivot = (
    member_rows
    .groupby(["Member", "Dataset", "Dataset_Label", "Industry"], dropna=False)
    .agg(Num_Participations=("Discussion_ID", "count"))
    .reset_index()
    .pivot_table(
        index=["Member", "Dataset", "Dataset_Label"],
        columns="Industry",
        values="Num_Participations",
        fill_value=0,
    )
    .reset_index()
)
pivot.columns.name = None

for col in ["fishing", "tourism", "neutral", "unknown"]:
    if col not in pivot.columns:
        pivot[col] = 0

pivot["Total_Participations"] = pivot[["fishing", "tourism", "neutral", "unknown"]].sum(axis=1)
pivot["Bias_Score"] = np.where(
    pivot["Total_Participations"] > 0,
    (pivot["tourism"] - pivot["fishing"]) / pivot["Total_Participations"],
    np.nan,
)
pivot["Bias_Direction"] = pivot["Bias_Score"].apply(bias_direction)

pivot.to_csv(os.path.join(OUT_DIR, "mc2_member_activity.csv"), index=False)
print(f"✓ mc2_member_activity.csv  ({len(pivot)} rows)")

# ─────────────────────────────────────────────────────────────────
# 11. MEMBER × TOPIC × DATASET AGGREGATED TABLE  [NEW]
#     For: sentiment heatmap, Goldstein ego chart, topic-level diverging bars
# ─────────────────────────────────────────────────────────────────

member_topic_agg = (
    member_rows
    .groupby(["Member", "Dataset", "Dataset_Label", "Topic_Key", "Industry"], dropna=False)
    .agg(
        Count=("Discussion_ID", "count"),
        Avg_Sentiment=("sentiment", "mean"),
        Num_Pro=("Sentiment_Label", lambda x: (x == "Pro").sum()),
        Num_Against=("Sentiment_Label", lambda x: (x == "Against").sum()),
        Num_Neutral_Sent=("Sentiment_Label", lambda x: (x == "Neutral").sum()),
        Num_Not_Rated=("Sentiment_Label", lambda x: (x == "Not Rated").sum()),
    )
    .reset_index()
)
# Add human-readable topic label
member_topic_agg["Topic_Label"] = (
    member_topic_agg["Topic_Key"].str.replace("_", " ").str.title()
)

member_topic_agg.to_csv(os.path.join(OUT_DIR, "mc2_member_topic_agg.csv"), index=False)
print(f"✓ mc2_member_topic_agg.csv  ({len(member_topic_agg)} rows)")

# ─────────────────────────────────────────────────────────────────
# 12. CROSS-DATASET COMPARISON TABLE  [NEW]
#     Wide format: Member × Topic with counts and sentiments per dataset
#     For: slope/bump chart, side-by-side ego comparison (Task 3 & 4)
# ─────────────────────────────────────────────────────────────────

cross_count = member_topic_agg.pivot_table(
    index=["Member", "Topic_Key", "Industry"],
    columns="Dataset",
    values="Count",
    fill_value=0,
).reset_index()
cross_count.columns = [
    c if c in ["Member", "Topic_Key", "Industry"] else f"Count_{c}"
    for c in cross_count.columns
]

cross_sent = member_topic_agg.pivot_table(
    index=["Member", "Topic_Key", "Industry"],
    columns="Dataset",
    values="Avg_Sentiment",
    fill_value=np.nan,
).reset_index()
cross_sent.columns = [
    c if c in ["Member", "Topic_Key", "Industry"] else f"Sent_{c}"
    for c in cross_sent.columns
]

# [BUG F FIX] Use a left merge so topic rows where Avg_Sentiment is NaN for
# all discussions in the group are still retained. An inner merge silently
# dropped these rows (e.g. Goldstein × deep_fishing_dock has Count=6 but
# Avg_Sentiment=NaN because no sentiment was recorded for those discussions).
cross = cross_count.merge(cross_sent, on=["Member", "Topic_Key", "Industry"], how="left")
cross["Topic_Label"] = cross["Topic_Key"].str.replace("_", " ").str.title()

# Ensure all dataset columns exist even if a dataset has no records for a member/topic
for ds in ["filah", "journalist", "trout"]:
    for prefix in ["Count", "Sent"]:
        col = f"{prefix}_{ds}"
        if col not in cross.columns:
            cross[col] = 0 if prefix == "Count" else np.nan

cross.to_csv(os.path.join(OUT_DIR, "mc2_cross_dataset_comparison.csv"), index=False)
print(f"✓ mc2_cross_dataset_comparison.csv  ({len(cross)} rows)")

# ─────────────────────────────────────────────────────────────────
# 13. MEMBER ATTENDANCE TABLE  [NEW]
#     One row per (Member × Meeting × Dataset)
#     For: timeline/meeting attendance heatmap (Task 4)
# ─────────────────────────────────────────────────────────────────

attendance = (
    member_rows
    .dropna(subset=["Meeting_Num"])
    .groupby(["Member", "Dataset", "Dataset_Label", "Meeting_Num"], dropna=False)
    .agg(
        Discussion_Count=("Discussion_ID", "count"),
        Avg_Sentiment=("sentiment", "mean"),
    )
    .reset_index()
)
attendance["Meeting_Num"] = attendance["Meeting_Num"].astype(int)

# Expand to full grid (all members × all meetings × all datasets)
# so absences show as 0 rows (Tableau can use this for "attended" flag)
all_meetings = list(range(1, 17))  # Meeting_1 to Meeting_16
datasets = ["filah", "journalist", "trout"]
full_index = pd.MultiIndex.from_product(
    [sorted(MEMBERS), datasets, all_meetings],
    names=["Member", "Dataset", "Meeting_Num"]
)
attendance_full = (
    pd.DataFrame(index=full_index)
    .reset_index()
    .merge(
        attendance[["Member", "Dataset", "Meeting_Num", "Discussion_Count", "Avg_Sentiment"]],
        on=["Member", "Dataset", "Meeting_Num"],
        how="left",
    )
)
attendance_full["Discussion_Count"] = attendance_full["Discussion_Count"].fillna(0).astype(int)
attendance_full["Attended"] = attendance_full["Discussion_Count"] > 0
attendance_full["Dataset_Label"] = attendance_full["Dataset"].map(DATASET_LABELS)
# Average sentiment is meaningless where count=0; already NaN from the left join

attendance_full.to_csv(os.path.join(OUT_DIR, "mc2_member_attendance.csv"), index=False)
print(f"✓ mc2_member_attendance.csv  ({len(attendance_full)} rows)")

# ─────────────────────────────────────────────────────────────────
# 14. TRAVEL GEO TABLE  [BUG A + B FIXED]
#     Correct join chain:
#       travel plan node
#       → participant link (source=plan_id, target=member) → Member name
#       → travel link (source=plan_id, target=geo_node_id) → lat/lon
#
#     The old script tried to join trip nodes directly to geo — trip nodes
#     have no coordinates. Coordinates only live in geo_nodes, reached via
#     travel links (plan → geo_node_id).
# ─────────────────────────────────────────────────────────────────

# Step A: travel links give us plan_id → geo_node_id
travel_links = links_raw[links_raw["role"] == "travel"][
    ["source", "target", "Dataset"]
].copy()
travel_links.columns = ["plan_id", "geo_node_id", "Dataset"]
travel_links["geo_node_id"] = travel_links["geo_node_id"].astype(str)

# Step B: join geo_node_id → coordinates
travel_links = travel_links.merge(
    geo_nodes[["id", "x", "y", "city_name", "zone"]].rename(
        columns={"id": "geo_node_id", "x": "lon", "y": "lat"}
    ),
    on="geo_node_id",
    how="left",
)

# Step C: participant links give us plan_id → member
plan_participants = links_raw[
    (links_raw["role"] == "participant")
    & (links_raw["target"].isin(MEMBERS))
][["source", "target", "Dataset", "sentiment"]].copy()
plan_participants.columns = ["plan_id", "Member", "Dataset", "Sentiment"]

# Step D: join member into travel table
travel_geo = travel_links.merge(
    plan_participants[["plan_id", "Dataset", "Member", "Sentiment"]],
    on=["plan_id", "Dataset"],
    how="left",
)

# Step E: attach topic/industry from plan_id string
travel_geo["Topic_Key"] = travel_geo["plan_id"].apply(infer_topic_from_id)
travel_geo["Industry"]  = travel_geo["Topic_Key"].map(TOPIC_INDUSTRY).fillna("unknown")
travel_geo["Dataset_Label"] = travel_geo["Dataset"].map(DATASET_LABELS)

# Step F: attach plan long_title for tooltip
plan_meta = (
    nodes[nodes["type"] == "plan"]
    .drop_duplicates(subset=["id", "Dataset"])
    [["id", "Dataset", "long_title"]]
    .rename(columns={"id": "plan_id"})
)
travel_geo = travel_geo.merge(plan_meta, on=["plan_id", "Dataset"], how="left")

# [BUG E FIX] Drop rows where no COOTEFOO member could be identified in this
# dataset's participant links. These travel plans have a travel link recorded
# by the lobby, but the member who took the trip only appears in a different
# dataset's participant links. They should not appear in this table.
travel_geo = travel_geo.dropna(subset=["Member"])

travel_geo.to_csv(os.path.join(OUT_DIR, "mc2_travel_geo.csv"), index=False)
print(f"✓ mc2_travel_geo.csv  ({len(travel_geo)} rows, {travel_geo['lat'].notna().sum()} with coordinates)")

# ─────────────────────────────────────────────────────────────────
# 15. DATASET COVERAGE COMPARISON
#     Which discussions/plans appear in FILAH, TROUT, journalist?
#     For: topic presence dot-matrix (Task 3), missing discussions bar
# ─────────────────────────────────────────────────────────────────

coverage_nodes = nodes[nodes["type"].isin(["discussion", "plan"])].copy()
coverage_nodes = coverage_nodes[coverage_nodes["Topic_Key"] != "unknown"]

coverage = coverage_nodes.pivot_table(
    index=["id", "type", "Topic_Key", "Industry"],
    columns="Dataset",
    aggfunc="size",
    fill_value=0,
).reset_index()
coverage.columns.name = None

for col in ["filah", "trout", "journalist"]:
    if col not in coverage.columns:
        coverage[col] = 0

coverage["In_FILAH"]      = coverage["filah"] > 0
coverage["In_TROUT"]      = coverage["trout"] > 0
coverage["In_Journalist"] = coverage["journalist"] > 0


def coverage_label(row):
    f, t, j = row["In_FILAH"], row["In_TROUT"], row["In_Journalist"]
    if f and not j:           return "FILAH-Only (Suspicious)"
    if t and not j:           return "TROUT-Only (Suspicious)"
    if j and not f and not t: return "Journalist-Only (Missing from both)"
    if j and f and not t:     return "Missing from TROUT"
    if j and t and not f:     return "Missing from FILAH"
    if f and t and j:         return "In All Datasets"
    return "Partial"


coverage["Coverage_Label"] = coverage.apply(coverage_label, axis=1)
coverage["Topic_Label"]    = coverage["Topic_Key"].str.replace("_", " ").str.title()

coverage.to_csv(os.path.join(OUT_DIR, "mc2_coverage_comparison.csv"), index=False)
print(f"✓ mc2_coverage_comparison.csv  ({len(coverage)} rows)")

# ─────────────────────────────────────────────────────────────────
# 16. INDUSTRY BREAKDOWN TABLE  [CORRECTED PERCENTAGES]
#     Source: Gephi edge weights (member × topic interaction counts)
#     This is what the bipartite network graphs directly represent.
#     The denominator is fishing + tourism + neutral (known topics only).
# ─────────────────────────────────────────────────────────────────

# Use member_topic_agg as the source (same data that drives Gephi edges)
ind_rows = []
for ds in ["filah", "journalist", "trout"]:
    ds_data = member_topic_agg[member_topic_agg["Dataset"] == ds]
    ind_counts = ds_data.groupby("Industry")["Count"].sum()
    known_total = (
        ind_counts.get("fishing", 0)
        + ind_counts.get("tourism", 0)
        + ind_counts.get("neutral", 0)
    )
    all_total = ind_counts.sum()
    for industry in ["fishing", "tourism", "neutral"]:
        count = ind_counts.get(industry, 0)
        ind_rows.append({
            "Dataset":       ds,
            "Dataset_Label": DATASET_LABELS[ds],
            "Industry":      industry,
            "Count":         int(count),
            "Known_Total":   int(known_total),
            "All_Total":     int(all_total),
            "Pct_of_Known":  round(count / known_total * 100, 1) if known_total > 0 else 0,
            "Pct_of_All":    round(count / all_total * 100, 1) if all_total > 0 else 0,
        })

industry_breakdown = pd.DataFrame(ind_rows)
industry_breakdown.to_csv(os.path.join(OUT_DIR, "mc2_industry_breakdown.csv"), index=False)
print(f"✓ mc2_industry_breakdown.csv  ({len(industry_breakdown)} rows)")
print("\n  Industry breakdown (Pct_of_Known, consistent denominator):")
for _, row in industry_breakdown.iterrows():
    print(f"    {row['Dataset_Label']:30s} {row['Industry']:10s} {row['Count']:3d} / {row['Known_Total']} = {row['Pct_of_Known']}%")

# ─────────────────────────────────────────────────────────────────
# 17. GEPHI NODE/EDGE FILES
#     Bipartite: members ↔ topics, edge weight = # participations,
#     avg_sentiment = mean sentiment across those participations.
#     Built from member_topic_agg which uses the corrected topic join.
# ─────────────────────────────────────────────────────────────────

gephi_datasets = {
    "f": "filah",
    "t": "trout",
    "j": "journalist",
}

for suffix, ds_key in gephi_datasets.items():
    ds_agg = member_topic_agg[
        (member_topic_agg["Dataset"] == ds_key)
        & (member_topic_agg["Topic_Key"] != "unknown")
        & (member_topic_agg["Count"] > 0)
    ].copy()

    if ds_agg.empty:
        print(f"  ⚠ No data for Gephi {ds_key}, skipping.")
        continue

    # --- Node file ---
    member_node_ids = ds_agg["Member"].unique()
    topic_node_ids  = ds_agg["Topic_Key"].unique()

    member_nodes_gephi = pd.DataFrame({
        "Id":       member_node_ids,
        "Label":    member_node_ids,
        "Type":     "Member",
        "Industry": "",
    })
    topic_nodes_gephi = pd.DataFrame({
        "Id":       topic_node_ids,
        "Label":    [t.replace("_", " ").title() for t in topic_node_ids],
        "Type":     "Topic",
        "Industry": [TOPIC_INDUSTRY.get(t, "unknown") for t in topic_node_ids],
    })
    gephi_nodes = pd.concat([member_nodes_gephi, topic_nodes_gephi], ignore_index=True)

    # --- Edge file ---
    gephi_edges = ds_agg[["Member", "Topic_Key", "Industry", "Count", "Avg_Sentiment"]].copy()
    gephi_edges.columns = ["Source", "Target", "Industry", "weight", "avg_sentiment"]

    # Save
    node_file = os.path.join(OUT_DIR, f"gephi_{suffix}_nodes.csv")
    edge_file = os.path.join(OUT_DIR, f"gephi_{suffix}_edges.csv")
    gephi_nodes.to_csv(node_file, index=False)
    gephi_edges.to_csv(edge_file, index=False)
    print(f"✓ gephi_{suffix}_nodes.csv ({len(gephi_nodes)} nodes) / gephi_{suffix}_edges.csv ({len(gephi_edges)} edges)")

# ─────────────────────────────────────────────────────────────────
# 18. SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print("\nMember participation counts (journalist, all topics):")
j_activity = pivot[pivot["Dataset"] == "journalist"].sort_values("Total_Participations", ascending=False)
for _, r in j_activity.iterrows():
    print(f"  {r['Member']:20s} fishing={int(r['fishing']):2d}  neutral={int(r['neutral']):2d}  tourism={int(r['tourism']):2d}  total={int(r['Total_Participations']):2d}  bias={r['Bias_Score']:.3f} ({r['Bias_Direction']})")

print("\nDataset coverage (discussions + plans):")
for ds in ["filah", "journalist", "trout"]:
    n_disc  = len(nodes[(nodes["Dataset"]==ds) & (nodes["type"]=="discussion")])
    n_plan  = len(nodes[(nodes["Dataset"]==ds) & (nodes["type"]=="plan")])
    n_mbr   = len(nodes[(nodes["Dataset"]==ds) & (nodes["type"]=="entity.person")])
    print(f"  {ds:12s}: {n_mbr} members, {n_disc} discussions, {n_plan} plans")

print("\nTravel records per dataset:")
for ds in ["filah", "journalist", "trout"]:
    n = len(travel_geo[travel_geo["Dataset"] == ds])
    n_coords = travel_geo[travel_geo["Dataset"] == ds]["lat"].notna().sum()
    print(f"  {ds:12s}: {n} travel plans, {n_coords} with coordinates")

print(f"\n✅ All outputs written to: {OUT_DIR}/")