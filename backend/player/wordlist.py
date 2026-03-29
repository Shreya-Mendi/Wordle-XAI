"""
Wordle word list management.
Loads word lists from the Kaggle 'bcruise/wordle-valid-words' dataset CSVs
(data/valid_solutions.csv and data/valid_guesses.csv) when available,
falling back to the bundled list otherwise.
"""

import csv
import os
from pathlib import Path

def _find_repo_root() -> Path:
    """
    Walk up from this file until we find a directory containing data/valid_solutions.csv.
    Works regardless of whether the code is inside a git worktree or a normal checkout.
    """
    current = Path(__file__).resolve().parent
    for _ in range(10):  # max 10 levels up
        if (current / "data" / "valid_solutions.csv").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parents[2]  # fallback: backend/


def _load_from_csv(filename: str) -> list[str]:
    """Load words from a CSV file with a 'word' column."""
    csv_path = _find_repo_root() / "data" / filename
    if not csv_path.exists():
        return []
    words = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row.get("word", "").strip().lower()
            if len(word) == 5 and word.isalpha():
                words.append(word)
    return words

_solutions = _load_from_csv("valid_solutions.csv")
_guesses   = _load_from_csv("valid_guesses.csv")

if _solutions:
    # Use Kaggle dataset: answers as the answer space, all valid words for guessing
    ANSWER_WORDS = sorted(set(_solutions))
    ALL_VALID_WORDS = sorted(set(_solutions) | set(_guesses))
else:
    # Fallback to bundled list
    _BUNDLED = [
    "about", "above", "abuse", "actor", "acute", "admit", "adopt", "adult",
    "after", "again", "agile", "agony", "agree", "ahead", "alarm", "album",
    "alert", "alike", "align", "alive", "alley", "allow", "alone", "along",
    "aloud", "altar", "alter", "angel", "anger", "angle", "angry", "ankle",
    "annex", "anvil", "aorta", "apple", "apply", "apron", "argue", "arise",
    "armed", "armor", "aroma", "arose", "array", "aside", "asset", "atone",
    "attic", "audio", "audit", "augur", "avail", "avoid", "awake", "award",
    "awful", "awoke", "azure", "badly", "bagel", "basic", "basis", "batch",
    "beach", "beard", "beast", "began", "begin", "being", "below", "bench",
    "birth", "black", "blade", "blame", "bland", "blank", "blast", "blaze",
    "bleak", "bleed", "bless", "blind", "block", "blood", "bloom", "blown",
    "blues", "blunt", "board", "boast", "bogus", "bonus", "boost", "boxer",
    "braid", "brain", "brand", "brave", "brawl", "break", "breed", "brick",
    "bride", "brief", "brine", "brink", "brisk", "brood", "brown", "brush",
    "brute", "buddy", "build", "built", "bulky", "bully", "bunch", "burly",
    "burnt", "buyer", "camel", "candy", "canon", "caper", "carry", "carve",
    "cause", "cedar", "chain", "chair", "chalk", "chaos", "chard", "charm",
    "chart", "chase", "check", "cheek", "chess", "chest", "chief", "child",
    "chill", "chord", "civic", "civil", "claim", "clamp", "clasp", "class",
    "clean", "clear", "clerk", "click", "cliff", "climb", "cling", "cloak",
    "clone", "close", "cloud", "clown", "clump", "coach", "comet", "comic",
    "comma", "coral", "corps", "count", "covet", "crack", "craft", "crane",
    "crash", "crazy", "cream", "creek", "creep", "crest", "crime", "crisp",
    "cross", "crowd", "crown", "cruel", "crush", "crust", "crypt", "curly",
    "curve", "cycle", "daily", "dance", "dated", "daunt", "dealt", "death",
    "decay", "decoy", "delay", "delta", "delve", "demon", "dense", "depot",
    "depth", "derby", "detox", "devil", "digit", "dirty", "disco", "diver",
    "dizzy", "dodge", "dogma", "donor", "doubt", "dough", "dowel", "draft",
    "drain", "drama", "drape", "drawn", "dread", "dream", "dregs", "drift",
    "drill", "drink", "drool", "drove", "drown", "drunk", "dryer", "duchy",
    "dunce", "dwarf", "dwell", "dying", "eagle", "early", "earth", "easel",
    "eaten", "ebony", "eight", "elect", "elite", "ember", "empty", "enemy",
    "enjoy", "ensue", "enter", "entry", "envoy", "equal", "equip", "erupt",
    "essay", "evade", "event", "every", "exact", "exert", "exist", "expel",
    "extra", "fable", "fancy", "farce", "fault", "feast", "femur", "fence",
    "feral", "ferry", "fever", "fiber", "field", "fiery", "finch", "fired",
    "first", "fixed", "fizzy", "flame", "flair", "flank", "flare", "flash",
    "flask", "flesh", "flick", "float", "flock", "flood", "floor", "floss",
    "flout", "flown", "flume", "flunk", "flute", "focal", "focus", "foray",
    "forge", "forte", "forum", "fraud", "freak", "fresh", "frisk", "frost",
    "froze", "fruit", "furor", "fused", "fuzzy", "gamut", "gaudy", "gauze",
    "gavel", "ghost", "girth", "given", "gland", "glare", "gleam", "glide",
    "glint", "gloat", "globe", "gloom", "gloss", "glove", "gnome", "going",
    "gorge", "gouge", "grace", "grade", "graft", "grain", "grand", "grant",
    "grape", "grasp", "grass", "grate", "grave", "graze", "greed", "greet",
    "grief", "grill", "grind", "gripe", "groan", "groin", "groom", "gross",
    "grout", "gruel", "gruff", "guard", "guava", "guess", "guest", "guide",
    "guild", "guise", "gusto", "habit", "haste", "haunt", "heart", "heavy",
    "hedge", "heist", "hence", "herbs", "hinge", "hoist", "holly", "homer",
    "honey", "honor", "horse", "hotel", "hound", "hover", "human", "humid",
    "hunch", "hyena", "ideal", "image", "imply", "index", "inert", "infer",
    "input", "inter", "intro", "inure", "irony", "ivory", "jaded", "jaunt",
    "jazzy", "jiffy", "joint", "joust", "judge", "jumbo", "jumpy", "knack",
    "kneel", "knelt", "knife", "knock", "knoll", "known", "kudos", "label",
    "lance", "lanky", "lapel", "lapse", "latch", "latex", "layer", "leach",
    "leaky", "learn", "ledge", "legal", "lemon", "level", "light", "lilac",
    "limit", "liner", "lingo", "llama", "local", "lodge", "logic", "lotus",
    "lover", "lucid", "lucky", "lunar", "lusty", "lyric", "magic", "major",
    "maker", "manor", "maple", "march", "marry", "match", "matte", "maxim",
    "media", "merit", "metal", "micro", "might", "mimic", "mince", "model",
    "mogul", "moist", "money", "moody", "moral", "morph", "motel", "mount",
    "mourn", "mouse", "mouth", "movie", "muddy", "mural", "music", "musty",
    "naive", "navel", "nerve", "never", "niche", "ninja", "noble", "noise",
    "nonce", "occur", "octet", "offer", "often", "optic", "orbit", "order",
    "ought", "ounce", "outdo", "ovary", "ozone", "paint", "pansy", "panic",
    "papal", "paper", "parka", "parse", "patch", "pause", "pearl", "penal",
    "peril", "petty", "phase", "phone", "photo", "picky", "piece", "pilot",
    "pinch", "pixel", "pizza", "place", "plain", "plane", "plant", "plate",
    "plaza", "plead", "plump", "plunk", "plush", "point", "poker", "polar",
    "posit", "pouch", "power", "prank", "press", "price", "pride", "prime",
    "prism", "prize", "probe", "proof", "prose", "proxy", "prude", "prune",
    "psalm", "pulse", "punch", "pupil", "purge", "purse", "queen", "query",
    "quest", "queue", "quick", "quill", "quirk", "quota", "quote", "rabid",
    "radar", "raise", "rally", "ramen", "ranch", "rapid", "raven", "reach",
    "ready", "realm", "rebel", "refer", "refit", "reign", "relax", "relay",
    "relic", "remix", "repay", "repel", "rerun", "reuse", "revel", "rhyme",
    "rider", "ridge", "risky", "rivet", "robot", "rocky", "rogue", "rouge",
    "rough", "round", "rowdy", "royal", "rugby", "ruler", "rupee", "rusty",
    "sadly", "salty", "salve", "salvo", "sandy", "sauce", "sauna", "scale",
    "scarf", "scorn", "scout", "scowl", "scram", "seize", "serif", "serve",
    "setup", "seven", "shall", "shame", "shank", "share", "shark", "sharp",
    "shawl", "shelf", "shell", "shift", "shiny", "shire", "shirt", "shock",
    "shore", "short", "shout", "shove", "shown", "siege", "sight", "sigma",
    "silly", "since", "sixth", "skill", "skimp", "slack", "slain", "slant",
    "slate", "slave", "sleek", "sleet", "slick", "slide", "slime", "slope",
    "sloth", "slump", "smack", "small", "smash", "smear", "smell", "smelt",
    "smile", "smite", "smoke", "snack", "snake", "snare", "snarl", "sneak",
    "sniff", "snore", "snort", "solar", "solve", "south", "space", "spade",
    "spare", "spark", "spasm", "spawn", "spend", "spice", "spill", "spine",
    "spire", "spite", "split", "spoke", "spoon", "sport", "spout", "spray",
    "spree", "sprig", "spunk", "squad", "squat", "squid", "stack", "stain",
    "stale", "stalk", "stall", "stamp", "stand", "stark", "stash", "stave",
    "steak", "steal", "steel", "steep", "steer", "stern", "stick", "stiff",
    "still", "sting", "stink", "stock", "stomp", "stone", "stood", "stool",
    "store", "stork", "storm", "story", "stout", "stove", "strap", "straw",
    "stray", "strip", "strut", "stuck", "study", "stump", "stung", "stunt",
    "style", "suave", "sugar", "suite", "sunny", "super", "surge", "swamp",
    "swarm", "swear", "sweat", "sweep", "sweet", "swept", "swift", "swine",
    "swipe", "swirl", "sword", "swore", "sworn", "syrup", "table", "taffy",
    "taken", "tally", "talon", "tapir", "taunt", "teach", "tease", "tempo",
    "tense", "tepid", "thank", "theme", "thing", "think", "thorn", "those",
    "three", "threw", "throw", "thumb", "tiara", "tiger", "tight", "tilde",
    "timer", "titan", "toast", "token", "tonal", "tonic", "topaz", "topic",
    "torch", "total", "totem", "toxic", "trace", "track", "trade", "trail",
    "train", "trait", "tramp", "trawl", "treat", "trend", "triad", "tried",
    "trill", "tripe", "trite", "troll", "trope", "trout", "trove", "truce",
    "truck", "trump", "trunk", "tryst", "tulip", "tuner", "tunic", "tumor",
    "twang", "tweak", "twice", "twill", "twirl", "twist", "ulcer", "ultra",
    "uncle", "uncut", "undue", "unify", "union", "unity", "until", "upper",
    "upset", "urban", "usage", "usual", "usurp", "utter", "valid", "valor",
    "valve", "vapor", "vault", "vital", "vivid", "voice", "voila", "vouch",
    "wacky", "wager", "waltz", "weary", "wedge", "weigh", "weird", "whale",
    "whack", "wheat", "wheel", "where", "which", "while", "whirl", "white",
    "whole", "whose", "wield", "witty", "world", "worse", "worst", "worth",
    "would", "wrath", "wreck", "wrest", "wring", "wrist", "wrong", "wrote",
    "yacht", "yearn", "yield", "young", "youth", "zebra", "zoned", "adore",
    "abide", "abyss", "acute", "agile", "abbey", "anvil", "apish", "ardor",
    "askew", "atoll", "bayou", "bigot", "blimp", "boxer", "briny", "butch",
    "byway", "cabal", "cache", "cadet", "cairn", "cameo", "cargo", "carol",
    "caste", "caulk", "champ", "chant", "chasm", "cinch", "civic", "cocci",
    "condo", "conga", "covey", "creak", "crimp", "croak", "cupid", "cutie",
    "daily", "daisy", "dated", "depot", "disco", "ditto", "divvy", "dizzy",
    "dodgy", "dowdy", "downy", "dross", "dumpy", "dunno", "dusky", "dusty",
    "dying", "earthy", "edify", "elbow", "elfin", "endow", "envoy", "epoch",
    "epoxy", "erode", "erupt", "etude", "evoke", "exile", "exude", "faddy",
    "fishy", "fitly", "fizzy", "foamy", "foggy", "folly", "foray", "foyer",
    "frail", "franc", "fussy", "gabby", "gaily", "gaudy", "gauze", "gauzy",
    "geeky", "giddy", "gimpy", "glare", "glazy", "glint", "glogg", "golly",
    "gooey", "goofy", "gorge", "gourd", "grail", "gravy", "grimy", "gross",
    "grout", "gruff", "guile", "gulch", "gully", "gummy", "gutsy", "hammy",
    "handy", "happy", "hardy", "hasty", "hatch", "hazel", "heady", "hence",
    "herby", "hippy", "hobby", "homey", "hoppy", "horny", "hotly", "huffy",
    "hulky", "humpy", "husky", "inane", "indie", "inlay", "inner", "irate",
    "itchy", "jaded", "jazzy", "jerky", "jiffy", "jolly", "jokey", "juicy",
    "jumpy", "kinky", "kitty", "knave", "kooky", "larky", "lemon", "lithe",
    "livid", "loopy", "lousy", "lowly", "lumpy", "lusty", "mangy", "mauve",
    "mealy", "meaty", "milky", "minty", "mirky", "misty", "moldy", "muggy",
    "murky", "mushy", "nasty", "natty", "nervy", "nippy", "noisy", "nosey",
    "nutty", "oaken", "occur", "oddly", "offal", "ogler", "oldie", "oomph",
    "outdo", "ovoid", "owing", "oxide", "parch", "pasty", "patsy", "patty",
    "peaky", "perky", "pervy", "petty", "piggy", "piney", "pithy", "plaid",
    "plumb", "plume", "plunk", "plush", "podgy", "pokey", "polka", "poppy",
    "porky", "potty", "pouty", "primp", "privy", "prosy", "pudgy", "puffy",
    "punky", "puppy", "pushy", "quaff", "qualm", "quash", "queer", "quell",
    "rabble", "rainy", "ratty", "rawly", "reedy", "regal", "repro", "ripen",
    "risky", "ritzy", "roomy", "ropey", "ruddy", "rugby", "runty", "rusty",
    "saggy", "sappy", "saucy", "savvy", "scaly", "scone", "seedy", "seepy",
    "shady", "shard", "shine", "sissy", "sketchy", "skimp", "slimy", "slosh",
    "slump", "snaky", "soapy", "soggy", "sooty", "soppy", "spacy", "spiny",
    "spiry", "spoil", "spoof", "spook", "spoon", "sporty", "spray", "stodgy",
    "stuffy", "sulky", "surly", "swampy", "tacky", "tangy", "tardy", "tawny",
    "taxed", "teddy", "tepid", "testy", "tipsy", "titan", "toffy", "touchy",
    "tricky", "trite", "tubby", "tulip", "tummy", "tuque", "tushy", "twisty",
    "ungly", "unity", "unruly", "unzip", "upend", "usher", "vapid", "veiny",
    "vexed", "vigil", "vinyl", "viper", "vivid", "vogue", "voidy", "vomit",
    "waffy", "wanly", "warty", "weedy", "welly", "wimpy", "windy", "wispy",
    "wobbly", "woozy", "wordy", "wormy", "wrath", "wreck",
]

    # Remove duplicates and ensure all are exactly 5 letters
    ANSWER_WORDS = sorted(set(w.lower() for w in _BUNDLED if len(w) == 5))
    ALL_VALID_WORDS = ANSWER_WORDS


def get_word_list() -> list[str]:
    """Return the answer word list (used as the candidate space for the solver)."""
    return list(ANSWER_WORDS)


def get_all_valid_words() -> list[str]:
    """Return all valid guess words (answers + extended guesses from Kaggle dataset)."""
    return list(ALL_VALID_WORDS)


def filter_by_constraints(
    words: list[str],
    green: dict[int, str],
    yellow: dict[int, set[str]],
    gray: set[str],
    must_contain: set[str],
) -> list[str]:
    """
    Filter word list by Wordle constraints.

    Args:
        words: candidate word list
        green: {position: letter} — correct letter at exact position
        yellow: {position: set_of_letters} — letters present but NOT at this position
        gray: set of absent letters
        must_contain: letters that must appear somewhere in the word

    Returns:
        filtered list of still-valid candidate words
    """
    result = []
    for word in words:
        valid = True

        # Check green constraints (correct position)
        for pos, letter in green.items():
            if word[pos] != letter:
                valid = False
                break
        if not valid:
            continue

        # Check gray constraints (absent letters)
        # Only mark absent if not also green/yellow (handles duplicate letters)
        for letter in gray:
            if letter in green.values():
                continue
            if any(letter in letters for letters in yellow.values()):
                # Letter appears in yellow — it IS in the word; gray means
                # there's no extra copy, so count must match
                continue
            if letter in word:
                valid = False
                break
        if not valid:
            continue

        # Check yellow constraints (present but wrong position)
        for pos, letters in yellow.items():
            for letter in letters:
                if word[pos] == letter:  # still at the wrong position
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue

        # Check must_contain (letters seen as yellow must appear somewhere)
        for letter in must_contain:
            if letter not in word:
                valid = False
                break
        if not valid:
            continue

        result.append(word)

    return result


def parse_board_to_constraints(board_rows: list[list[dict]]) -> tuple:
    """
    Convert the analyze API board format into solver constraint triple.

    Args:
        board_rows: list of rows, each row is a list of
                    {"letter": str, "state": str} dicts.
                    state is one of: "correct", "present", "absent", "empty"

    Returns:
        (green, yellow, gray, must_contain) tuple ready for filter_by_constraints
    """
    green: dict[int, str] = {}
    yellow: dict[int, set[str]] = {i: set() for i in range(5)}
    gray: set[str] = set()
    must_contain: set[str] = set()

    for row in board_rows:
        for pos, tile in enumerate(row):
            letter = tile.get("letter", "").lower()
            state = tile.get("state", "empty")

            if not letter or state == "empty":
                continue

            if state == "correct":
                green[pos] = letter
            elif state == "present":
                yellow[pos].add(letter)
                must_contain.add(letter)
            elif state == "absent":
                gray.add(letter)

    return green, yellow, gray, must_contain
