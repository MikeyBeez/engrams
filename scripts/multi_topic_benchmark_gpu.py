#!/usr/bin/env python3
"""
Multi-Topic Engram Benchmark - GPU VERSION

Forces full GPU placement for consistent results.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
from datetime import datetime
from huggingface_hub import login

# Check GPU
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

token = os.environ.get("HF_TOKEN")
if not token:
    try:
        token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
        login(token=token, add_to_git_credential=False)
    except:
        pass

# =============================================================================
# TOPIC 1: WORLD WAR II
# =============================================================================
WWII_CONTEXT = """
World War II (1939-1945) was the deadliest conflict in human history.

Key dates and events:
- September 1, 1939: Germany invaded Poland, starting the war
- June 6, 1944: D-Day, Allied invasion of Normandy (Operation Overlord)
- May 8, 1945: V-E Day, Germany surrendered
- August 6, 1945: Atomic bomb dropped on Hiroshima
- August 9, 1945: Atomic bomb dropped on Nagasaki
- August 15, 1945: V-J Day, Japan surrendered

Key figures:
- Adolf Hitler: Leader of Nazi Germany
- Winston Churchill: British Prime Minister
- Franklin D. Roosevelt: US President (died April 1945)
- Harry S. Truman: US President who ordered atomic bombs
- Joseph Stalin: Soviet leader
- Benito Mussolini: Italian fascist dictator

Major battles:
- Battle of Stalingrad (1942-43): Turning point on Eastern Front
- Battle of Midway (June 1942): Turning point in Pacific
- Battle of Britain (1940): German air campaign against UK
- Battle of the Bulge (1944-45): Last major German offensive

The Holocaust killed approximately 6 million Jews.
Total deaths estimated at 70-85 million people.
"""

WWII_QUESTIONS = [
    {"q": "When did Germany invade Poland to start World War II?", "a": "september 1, 1939", "keywords": ["september", "1939"]},
    {"q": "What was the date of D-Day?", "a": "june 6, 1944", "keywords": ["june 6", "1944"]},
    {"q": "When did World War II end in Europe?", "a": "may 8, 1945", "keywords": ["may 8", "1945", "v-e day"]},
    {"q": "On what city was the first atomic bomb dropped?", "a": "hiroshima", "keywords": ["hiroshima"]},
    {"q": "Who was the British Prime Minister during most of WWII?", "a": "winston churchill", "keywords": ["churchill"]},
    {"q": "What battle was a turning point on the Eastern Front?", "a": "stalingrad", "keywords": ["stalingrad"]},
    {"q": "How many Jews were killed in the Holocaust?", "a": "6 million", "keywords": ["6 million", "six million"]},
    {"q": "Who was the leader of Nazi Germany?", "a": "adolf hitler", "keywords": ["hitler"]},
    {"q": "What was Operation Overlord?", "a": "d-day normandy invasion", "keywords": ["d-day", "normandy"]},
    {"q": "Which US President ordered the atomic bombs dropped on Japan?", "a": "harry truman", "keywords": ["truman"]},
    {"q": "What was the turning point battle in the Pacific theater?", "a": "midway", "keywords": ["midway"]},
    {"q": "When did Japan surrender?", "a": "august 15, 1945", "keywords": ["august", "1945", "v-j"]},
    {"q": "Who was the Soviet leader during WWII?", "a": "joseph stalin", "keywords": ["stalin"]},
    {"q": "What was the last major German offensive called?", "a": "battle of the bulge", "keywords": ["bulge"]},
    {"q": "When was the atomic bomb dropped on Nagasaki?", "a": "august 9, 1945", "keywords": ["august 9", "1945"]},
    {"q": "What was the German air campaign against Britain called?", "a": "battle of britain", "keywords": ["britain", "battle"]},
    {"q": "Who was the Italian fascist dictator allied with Hitler?", "a": "benito mussolini", "keywords": ["mussolini"]},
    {"q": "Approximately how many total people died in WWII?", "a": "70-85 million", "keywords": ["70", "80", "million"]},
    {"q": "Which US President died in April 1945?", "a": "franklin roosevelt", "keywords": ["roosevelt", "fdr"]},
    {"q": "What years did the Battle of Stalingrad span?", "a": "1942-1943", "keywords": ["1942", "1943"]},
]

# =============================================================================
# TOPIC 2: SOLAR SYSTEM
# =============================================================================
SOLAR_CONTEXT = """
Our Solar System formed approximately 4.6 billion years ago.

The Sun:
- A G-type main-sequence star (yellow dwarf)
- Contains 99.86% of the Solar System's mass
- Surface temperature: about 5,500°C (10,000°F)
- Core temperature: about 15 million°C

The 8 planets (in order from Sun):
1. Mercury: Smallest planet, no atmosphere, extreme temperatures
2. Venus: Hottest planet (462°C), thick CO2 atmosphere, rotates backwards
3. Earth: Only known planet with life, one moon
4. Mars: Red planet, two moons (Phobos and Deimos), thin atmosphere
5. Jupiter: Largest planet, Great Red Spot storm, 95 known moons including Ganymede
6. Saturn: Famous rings made of ice and rock, 146 known moons including Titan
7. Uranus: Tilted on its side (98 degrees), ice giant
8. Neptune: Windiest planet, blue color from methane, moon Triton

Dwarf planets include Pluto, Eris, Ceres, Makemake, and Haumea.
The asteroid belt lies between Mars and Jupiter.
The Kuiper Belt extends beyond Neptune.
"""

SOLAR_QUESTIONS = [
    {"q": "How old is our Solar System?", "a": "4.6 billion years", "keywords": ["4.6", "billion"]},
    {"q": "What type of star is the Sun?", "a": "g-type yellow dwarf", "keywords": ["g-type", "yellow dwarf", "main-sequence"]},
    {"q": "Which planet is the hottest?", "a": "venus", "keywords": ["venus"]},
    {"q": "Which planet is the largest?", "a": "jupiter", "keywords": ["jupiter"]},
    {"q": "What is the name of Mars's two moons?", "a": "phobos and deimos", "keywords": ["phobos", "deimos"]},
    {"q": "What is Jupiter's famous storm called?", "a": "great red spot", "keywords": ["red spot"]},
    {"q": "Which planet has the most prominent ring system?", "a": "saturn", "keywords": ["saturn"]},
    {"q": "What is Saturn's largest moon?", "a": "titan", "keywords": ["titan"]},
    {"q": "Which planet rotates on its side?", "a": "uranus", "keywords": ["uranus"]},
    {"q": "Which planet is the windiest?", "a": "neptune", "keywords": ["neptune"]},
    {"q": "What is the smallest planet?", "a": "mercury", "keywords": ["mercury"]},
    {"q": "Where is the asteroid belt located?", "a": "between mars and jupiter", "keywords": ["mars", "jupiter"]},
    {"q": "What gives Neptune its blue color?", "a": "methane", "keywords": ["methane"]},
    {"q": "How many planets are in our Solar System?", "a": "8", "keywords": ["8", "eight"]},
    {"q": "What is the surface temperature of the Sun?", "a": "5500 celsius", "keywords": ["5500", "5,500", "10000", "10,000"]},
    {"q": "Which planet has a moon named Triton?", "a": "neptune", "keywords": ["neptune"]},
    {"q": "What dwarf planet was formerly the 9th planet?", "a": "pluto", "keywords": ["pluto"]},
    {"q": "What is Jupiter's largest moon?", "a": "ganymede", "keywords": ["ganymede"]},
    {"q": "Which planet rotates backwards compared to most others?", "a": "venus", "keywords": ["venus"]},
    {"q": "What region extends beyond Neptune?", "a": "kuiper belt", "keywords": ["kuiper"]},
]

# =============================================================================
# TOPIC 3: CHEMISTRY - PERIODIC TABLE
# =============================================================================
CHEMISTRY_CONTEXT = """
The Periodic Table organizes elements by atomic number and properties.

Element basics:
- Hydrogen (H): Atomic number 1, lightest element, most abundant in universe
- Helium (He): Atomic number 2, noble gas, used in balloons
- Carbon (C): Atomic number 6, basis of organic chemistry
- Nitrogen (N): Atomic number 7, 78% of Earth's atmosphere
- Oxygen (O): Atomic number 8, essential for respiration
- Gold (Au): Atomic number 79, symbol from Latin "aurum"
- Silver (Ag): Atomic number 47, symbol from Latin "argentum"
- Iron (Fe): Atomic number 26, symbol from Latin "ferrum"
- Copper (Cu): Atomic number 29, symbol from Latin "cuprum"
- Lead (Pb): Atomic number 82, symbol from Latin "plumbum"

Groups and properties:
- Alkali metals (Group 1): Lithium, Sodium, Potassium - highly reactive
- Noble gases (Group 18): Helium, Neon, Argon - chemically inert
- Halogens (Group 17): Fluorine, Chlorine, Bromine, Iodine - reactive nonmetals
- Transition metals: Iron, Copper, Gold, Silver - good conductors

Water (H2O) is made of 2 hydrogen and 1 oxygen.
Table salt (NaCl) is sodium chloride.
The atomic number equals the number of protons.
"""

CHEMISTRY_QUESTIONS = [
    {"q": "What is the lightest element?", "a": "hydrogen", "keywords": ["hydrogen"]},
    {"q": "What is the atomic number of carbon?", "a": "6", "keywords": ["6", "six"]},
    {"q": "What percentage of Earth's atmosphere is nitrogen?", "a": "78%", "keywords": ["78"]},
    {"q": "What is the chemical symbol for gold?", "a": "au", "keywords": ["au"]},
    {"q": "What is the chemical symbol for silver?", "a": "ag", "keywords": ["ag"]},
    {"q": "What is the chemical formula for water?", "a": "h2o", "keywords": ["h2o"]},
    {"q": "What is table salt's chemical name?", "a": "sodium chloride", "keywords": ["sodium chloride", "nacl"]},
    {"q": "Which group of elements are chemically inert?", "a": "noble gases", "keywords": ["noble"]},
    {"q": "What is the atomic number of oxygen?", "a": "8", "keywords": ["8", "eight"]},
    {"q": "What Latin word does the symbol Au come from?", "a": "aurum", "keywords": ["aurum"]},
    {"q": "Which element is the most abundant in the universe?", "a": "hydrogen", "keywords": ["hydrogen"]},
    {"q": "What is the chemical symbol for iron?", "a": "fe", "keywords": ["fe"]},
    {"q": "What group are Lithium, Sodium, and Potassium in?", "a": "alkali metals", "keywords": ["alkali"]},
    {"q": "What is the atomic number of lead?", "a": "82", "keywords": ["82"]},
    {"q": "What does atomic number equal?", "a": "number of protons", "keywords": ["proton"]},
    {"q": "What element has atomic number 2?", "a": "helium", "keywords": ["helium"]},
    {"q": "What is the chemical symbol for copper?", "a": "cu", "keywords": ["cu"]},
    {"q": "Which group contains Fluorine, Chlorine, and Bromine?", "a": "halogens", "keywords": ["halogen"]},
    {"q": "What Latin word does the symbol Pb come from?", "a": "plumbum", "keywords": ["plumbum"]},
    {"q": "What gas is used to fill balloons and makes your voice high?", "a": "helium", "keywords": ["helium"]},
]

# =============================================================================
# TOPIC 4: HUMAN ANATOMY
# =============================================================================
ANATOMY_CONTEXT = """
The human body contains multiple organ systems working together.

Major organs:
- Heart: Pumps blood, about the size of a fist, beats ~100,000 times per day
- Brain: Control center, weighs about 3 pounds (1.4 kg), ~86 billion neurons
- Lungs: Gas exchange, right lung has 3 lobes, left has 2 (space for heart)
- Liver: Largest internal organ, detoxification, produces bile
- Kidneys: Filter blood, produce urine, located near lower back
- Stomach: Digests food with hydrochloric acid
- Small intestine: About 20 feet long, nutrient absorption
- Large intestine: About 5 feet long, water absorption

Skeletal system:
- Adults have 206 bones
- Babies are born with about 270 bones (some fuse together)
- Femur (thigh bone) is the longest and strongest bone
- Stapes (in the ear) is the smallest bone

Circulatory system:
- Red blood cells carry oxygen
- White blood cells fight infection
- Blood vessels: arteries, veins, capillaries
- Average adult has about 5 liters of blood
"""

ANATOMY_QUESTIONS = [
    {"q": "How many bones do adults have?", "a": "206", "keywords": ["206"]},
    {"q": "What is the largest internal organ?", "a": "liver", "keywords": ["liver"]},
    {"q": "How many times does the heart beat per day approximately?", "a": "100000", "keywords": ["100,000", "100000"]},
    {"q": "What is the longest bone in the human body?", "a": "femur", "keywords": ["femur"]},
    {"q": "How much does the human brain weigh approximately?", "a": "3 pounds", "keywords": ["3 pound", "1.4 kg"]},
    {"q": "What is the smallest bone in the human body?", "a": "stapes", "keywords": ["stapes"]},
    {"q": "How long is the small intestine?", "a": "20 feet", "keywords": ["20 feet", "20-foot"]},
    {"q": "How many lobes does the right lung have?", "a": "3", "keywords": ["3", "three"]},
    {"q": "What type of blood cells carry oxygen?", "a": "red blood cells", "keywords": ["red"]},
    {"q": "What acid does the stomach use for digestion?", "a": "hydrochloric acid", "keywords": ["hydrochloric"]},
    {"q": "How many bones are babies born with approximately?", "a": "270", "keywords": ["270"]},
    {"q": "What do the kidneys produce?", "a": "urine", "keywords": ["urine"]},
    {"q": "How many neurons are in the brain approximately?", "a": "86 billion", "keywords": ["86 billion"]},
    {"q": "What does the liver produce for digestion?", "a": "bile", "keywords": ["bile"]},
    {"q": "How many liters of blood does an average adult have?", "a": "5 liters", "keywords": ["5 liter"]},
    {"q": "What type of blood cells fight infection?", "a": "white blood cells", "keywords": ["white"]},
    {"q": "Why does the left lung have fewer lobes than the right?", "a": "space for heart", "keywords": ["heart"]},
    {"q": "What are the three types of blood vessels?", "a": "arteries veins capillaries", "keywords": ["arteries", "veins", "capillaries"]},
    {"q": "Where are the kidneys located?", "a": "lower back", "keywords": ["back", "lower"]},
    {"q": "What is absorbed in the large intestine?", "a": "water", "keywords": ["water"]},
]

# =============================================================================
# TOPIC 5: US GEOGRAPHY
# =============================================================================
GEOGRAPHY_CONTEXT = """
The United States spans North America with diverse geography.

States and capitals:
- California: Sacramento (largest state by population, ~39 million)
- Texas: Austin (second largest by area and population)
- Alaska: Juneau (largest state by area)
- New York: Albany (New York City is largest US city)
- Florida: Tallahassee (third most populous state)
- Hawaii: Honolulu (only island state)

Major geographic features:
- Mississippi River: Longest river in North America (~2,340 miles)
- Colorado River: Carved the Grand Canyon
- Great Lakes: Superior, Michigan, Huron, Erie, Ontario (HOMES mnemonic)
- Lake Superior: Largest Great Lake by surface area
- Rocky Mountains: Major mountain range in western US
- Appalachian Mountains: Eastern mountain range, older and lower
- Mount McKinley (Denali): Highest peak in North America (20,310 feet)
- Death Valley: Lowest point in North America (-282 feet)
- Grand Canyon: In Arizona, carved by Colorado River

The US has 50 states and Washington D.C. as the capital.
The continental US borders Canada and Mexico.
"""

GEOGRAPHY_QUESTIONS = [
    {"q": "What is the capital of California?", "a": "sacramento", "keywords": ["sacramento"]},
    {"q": "What is the largest state by area?", "a": "alaska", "keywords": ["alaska"]},
    {"q": "What is the longest river in North America?", "a": "mississippi", "keywords": ["mississippi"]},
    {"q": "What river carved the Grand Canyon?", "a": "colorado river", "keywords": ["colorado"]},
    {"q": "What is the highest peak in North America?", "a": "denali mount mckinley", "keywords": ["denali", "mckinley"]},
    {"q": "How many states are in the United States?", "a": "50", "keywords": ["50", "fifty"]},
    {"q": "What is the largest Great Lake by surface area?", "a": "lake superior", "keywords": ["superior"]},
    {"q": "What is the capital of Texas?", "a": "austin", "keywords": ["austin"]},
    {"q": "What is the lowest point in North America?", "a": "death valley", "keywords": ["death valley"]},
    {"q": "Which state is the only island state?", "a": "hawaii", "keywords": ["hawaii"]},
    {"q": "What is the capital of New York state?", "a": "albany", "keywords": ["albany"]},
    {"q": "Name the five Great Lakes.", "a": "superior michigan huron erie ontario", "keywords": ["superior", "michigan", "huron", "erie", "ontario"]},
    {"q": "What state is the Grand Canyon located in?", "a": "arizona", "keywords": ["arizona"]},
    {"q": "What is the most populous US state?", "a": "california", "keywords": ["california"]},
    {"q": "What is the capital of Alaska?", "a": "juneau", "keywords": ["juneau"]},
    {"q": "How high is Denali?", "a": "20310 feet", "keywords": ["20,310", "20310"]},
    {"q": "What major mountain range is in the eastern US?", "a": "appalachian", "keywords": ["appalachian"]},
    {"q": "What is the capital of the United States?", "a": "washington dc", "keywords": ["washington", "d.c."]},
    {"q": "What two countries does the continental US border?", "a": "canada and mexico", "keywords": ["canada", "mexico"]},
    {"q": "What is the largest city in the United States?", "a": "new york city", "keywords": ["new york"]},
]

# =============================================================================
# COMPILE ALL TOPICS
# =============================================================================
TOPICS = {
    "wwii": {"context": WWII_CONTEXT, "questions": WWII_QUESTIONS},
    "solar": {"context": SOLAR_CONTEXT, "questions": SOLAR_QUESTIONS},
    "chemistry": {"context": CHEMISTRY_CONTEXT, "questions": CHEMISTRY_QUESTIONS},
    "anatomy": {"context": ANATOMY_CONTEXT, "questions": ANATOMY_QUESTIONS},
    "geography": {"context": GEOGRAPHY_CONTEXT, "questions": GEOGRAPHY_QUESTIONS},
}

STRENGTH = 3.0  # The sweet spot


def extract_engram(model, tokenizer, text, layer_idx=16, num_tokens=32):
    """Extract engram from text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states[layer_idx]
    seq_len = hidden.shape[1]

    chunk_size = max(1, seq_len // num_tokens)
    vectors = []
    for i in range(num_tokens):
        start = i * chunk_size
        end = start + chunk_size if i < num_tokens - 1 else seq_len
        if start >= seq_len:
            vectors.append(hidden[0, -1, :])
        else:
            vectors.append(hidden[0, start:end].mean(dim=0))

    return torch.stack(vectors)


def generate_with_engram(model, tokenizer, question, engram, strength):
    """Generate answer with engram at specified strength."""
    embed = model.get_input_embeddings()

    e_norm = embed.weight.norm(dim=1).mean().item()
    g_norm = engram.norm(dim=1).mean().item()
    scaled = engram * (e_norm / g_norm) * strength

    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    emb = embed(inputs.input_ids)

    combined = torch.cat([scaled.unsqueeze(0).to(emb.dtype), emb], dim=1)

    with torch.no_grad():
        output = model.generate(
            inputs_embeds=combined,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()
    return response


def generate_baseline(model, tokenizer, question):
    """Generate answer without engram."""
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()
    return response


def check_answer(response, keywords):
    """Check if response contains any of the keywords."""
    response_lower = response.lower()
    return any(kw.lower() in response_lower for kw in keywords)


# =============================================================================
# MAIN BENCHMARK
# =============================================================================
print("Loading model with explicit GPU placement...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

# Force GPU placement
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B",
    torch_dtype=torch.float16,
).to("cuda")  # Explicit GPU placement

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Model device: {model.device}")
print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

print("\nExtracting engrams for all topics...")
engrams = {}
for topic_name, topic_data in TOPICS.items():
    engrams[topic_name] = extract_engram(model, tokenizer, topic_data["context"])
    print(f"  {topic_name}: norm = {engrams[topic_name].norm().item():.2f}")

print("\n" + "=" * 100)
print("MULTI-TOPIC ENGRAM BENCHMARK (GPU)")
print("=" * 100)
print(f"Topics: {len(TOPICS)}")
print(f"Questions per topic: 20")
print(f"Total questions: {len(TOPICS) * 20}")
print(f"Engram strength: {STRENGTH}x")
print(f"Conditions: baseline, matched-engram, mismatched-engram")
print()

# Results storage
results = {
    "baseline": {"correct": 0, "total": 0, "by_topic": {}},
    "matched": {"correct": 0, "total": 0, "by_topic": {}},
    "mismatched": {"correct": 0, "total": 0, "by_topic": {}},
}

# Pick a fixed mismatched topic for consistency
topic_list = list(TOPICS.keys())

for topic_idx, (topic_name, topic_data) in enumerate(TOPICS.items()):
    # Pick mismatched engram (next topic in rotation)
    mismatch_topic = topic_list[(topic_idx + 2) % len(topic_list)]
    mismatch_engram = engrams[mismatch_topic]
    matched_engram = engrams[topic_name]

    print(f"\n{'='*100}")
    print(f"TOPIC: {topic_name.upper()} (mismatched engram: {mismatch_topic})")
    print(f"{'='*100}", flush=True)

    results["baseline"]["by_topic"][topic_name] = {"correct": 0, "total": 0}
    results["matched"]["by_topic"][topic_name] = {"correct": 0, "total": 0}
    results["mismatched"]["by_topic"][topic_name] = {"correct": 0, "total": 0}

    for q_data in topic_data["questions"]:
        question = q_data["q"]
        keywords = q_data["keywords"]

        # Baseline
        base_ans = generate_baseline(model, tokenizer, question)
        base_correct = check_answer(base_ans, keywords)
        results["baseline"]["total"] += 1
        results["baseline"]["by_topic"][topic_name]["total"] += 1
        if base_correct:
            results["baseline"]["correct"] += 1
            results["baseline"]["by_topic"][topic_name]["correct"] += 1

        # Matched engram
        match_ans = generate_with_engram(model, tokenizer, question, matched_engram, STRENGTH)
        match_correct = check_answer(match_ans, keywords)
        results["matched"]["total"] += 1
        results["matched"]["by_topic"][topic_name]["total"] += 1
        if match_correct:
            results["matched"]["correct"] += 1
            results["matched"]["by_topic"][topic_name]["correct"] += 1

        # Mismatched engram
        mismatch_ans = generate_with_engram(model, tokenizer, question, mismatch_engram, STRENGTH)
        mismatch_correct = check_answer(mismatch_ans, keywords)
        results["mismatched"]["total"] += 1
        results["mismatched"]["by_topic"][topic_name]["total"] += 1
        if mismatch_correct:
            results["mismatched"]["correct"] += 1
            results["mismatched"]["by_topic"][topic_name]["correct"] += 1

        # Print per-question result
        b = "✓" if base_correct else "✗"
        m = "✓" if match_correct else "✗"
        x = "✓" if mismatch_correct else "✗"
        print(f"  Base={b} Match={m} Mismatch={x} | {question[:60]}...", flush=True)

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 100)
print("SUMMARY BY TOPIC")
print("=" * 100)

print(f"\n{'Topic':<15} {'Baseline':<15} {'Matched':<15} {'Mismatched':<15} {'Match-Base':<12} {'Match-Mis':<12}")
print("-" * 90)

for topic_name in TOPICS.keys():
    base = results["baseline"]["by_topic"][topic_name]
    match = results["matched"]["by_topic"][topic_name]
    mismatch = results["mismatched"]["by_topic"][topic_name]

    base_acc = base["correct"] / base["total"]
    match_acc = match["correct"] / match["total"]
    mismatch_acc = mismatch["correct"] / mismatch["total"]

    delta_base = match_acc - base_acc
    delta_mis = match_acc - mismatch_acc

    print(f"{topic_name:<15} {base_acc:>6.1%} ({base['correct']}/{base['total']})  "
          f"{match_acc:>6.1%} ({match['correct']}/{match['total']})  "
          f"{mismatch_acc:>6.1%} ({mismatch['correct']}/{mismatch['total']})  "
          f"{delta_base:>+6.1%}       {delta_mis:>+6.1%}")

print("-" * 90)

# Overall
base_total = results["baseline"]["correct"] / results["baseline"]["total"]
match_total = results["matched"]["correct"] / results["matched"]["total"]
mismatch_total = results["mismatched"]["correct"] / results["mismatched"]["total"]

print(f"{'OVERALL':<15} {base_total:>6.1%} ({results['baseline']['correct']}/{results['baseline']['total']})  "
      f"{match_total:>6.1%} ({results['matched']['correct']}/{results['matched']['total']})  "
      f"{mismatch_total:>6.1%} ({results['mismatched']['correct']}/{results['mismatched']['total']})  "
      f"{match_total - base_total:>+6.1%}       {match_total - mismatch_total:>+6.1%}")

print("\n" + "=" * 100)
print("ANALYSIS")
print("=" * 100)

if match_total > base_total + 0.05 and match_total > mismatch_total + 0.05:
    print(f"""
HYPOTHESIS CONFIRMED: Domain-specific semantic priming at 3x strength

Key findings:
- Matched engram accuracy: {match_total:.1%}
- Baseline accuracy: {base_total:.1%} (Δ = {match_total - base_total:+.1%})
- Mismatched engram accuracy: {mismatch_total:.1%} (Δ = {match_total - mismatch_total:+.1%})

The matched engram significantly outperforms both baseline and mismatched engrams.
This confirms that at 3x strength:
1. Engram content matters - it's not just noise injection
2. Topic-matched engrams activate relevant knowledge
3. Mismatched engrams may interfere or provide no benefit

This supports using engrams as "semantic retrieval keys" for knowledge
that exists in the model's training data.
""")
elif match_total <= mismatch_total:
    print(f"""
HYPOTHESIS NOT SUPPORTED: No domain specificity advantage

Matched engram ({match_total:.1%}) did not outperform mismatched ({mismatch_total:.1%}).
This suggests even at 3x strength, the mechanism may be:
- General activation perturbation
- Non-semantic priming
- Random variation

More investigation needed.
""")
else:
    print(f"""
PARTIAL SUPPORT: Some domain specificity but not dramatic

Matched engram: {match_total:.1%}
Mismatched engram: {mismatch_total:.1%}
Baseline: {base_total:.1%}

There is some advantage for matched engrams, but the effect is modest.
""")

# Save results
results_file = f"/home/bee/engrams/results/multi_topic_benchmark_gpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(results_file, 'w') as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "strength": STRENGTH,
        "model": "Qwen/Qwen2.5-7B",
        "device": "cuda",
        "results": results,
        "summary": {
            "baseline": base_total,
            "matched": match_total,
            "mismatched": mismatch_total,
        }
    }, f, indent=2)
print(f"\nResults saved to: {results_file}")
