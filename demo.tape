# Socratic Philosophy Dialogue Bot — VHS Demo
#
# Records a terminal GIF showcasing the core features:
#   - Multi-stage LLM pipeline (extraction, contradiction, generation)
#   - Dynamic state injection & claim tracking
#   - Contradiction detection
#   - Socratic constraint enforcement (refuse-and-redirect)
#   - Session management
#
# Prerequisites:
#   - VHS installed: brew install vhs
#   - Python 3.11+ with deps: pip install openai python-dotenv tiktoken
#   - Valid OPENAI_API_KEY in .env
#
# Usage (run from the project root):
#   vhs demo.tape
#
# The Hide/Show blocks skip API wait time so the GIF stays tight (~75s)
# while the actual recording adapts to your GPT-4 latency.
# If turns timeout, increase the WaitTimeout setting.

Output demo.gif
Output demo.webm

# ── Terminal Settings ──────────────────────────────────────

Set Shell "zsh"
Set FontFamily "JetBrains Mono"
Set FontSize 14
Set Width 1200
Set Height 800
Set Padding 20
Set Framerate 30
Set TypingSpeed 30ms
Set Theme "Catppuccin Mocha"
Set WindowBar Colorful
Set WindowBarSize 40
Set Margin 20
Set MarginFill "#1e1e2e"
Set BorderRadius 8
Set WaitTimeout 120s

# ── Hidden Setup ───────────────────────────────────────────
# Activate venv and clear terminal for a clean start.
# Assumes `vhs demo.tape` is run from the project root.

Hide
Type "source venv/bin/activate && clear"
Enter
Sleep 2s
Show

# ══════════════════════════════════════════════════════════════
# 1. LAUNCH
# ══════════════════════════════════════════════════════════════

Type "python main.py"
Sleep 500ms
Enter
Sleep 3s

# ══════════════════════════════════════════════════════════════
# 2. SELECT TOPIC — Free will and determinism
# ══════════════════════════════════════════════════════════════

Type "1"
Sleep 500ms
Enter
Sleep 2s

# ══════════════════════════════════════════════════════════════
# 3. FIRST CLAIM — Hard determinism
#    Pipeline: claim extraction (temp=0) → Socratic response (temp=0.8)
#    Viewer sees: typing → brief spinner → extraction debug + response
# ══════════════════════════════════════════════════════════════

Type "I think free will is an illusion. Everything we do is determined by prior causes — our genes, upbringing, the laws of physics. We never really choose anything."
Sleep 500ms
Enter
Sleep 3s

Hide
Wait /You:/
Show

Sleep 8s

# ══════════════════════════════════════════════════════════════
# 4. INSPECT STATE — Tracked positions from extraction
# ══════════════════════════════════════════════════════════════

Type "/positions"
Sleep 300ms
Enter
Sleep 4s

# ══════════════════════════════════════════════════════════════
# 5. CONTRADICTION — Shift to compatibilism
#    Pipeline: extraction → contradiction detection (temp=0) → generation
#    The debug output shows NEW CONTRADICTIONS detected between
#    the hard determinist and compatibilist positions.
# ══════════════════════════════════════════════════════════════

Type "Actually, I think compatibilism makes more sense. A choice can be both caused and free — as long as it flows from your own desires and reasoning, it counts as free."
Sleep 500ms
Enter
Sleep 3s

Hide
Wait /You:/
Show

Sleep 10s

# ══════════════════════════════════════════════════════════════
# 6. INSPECT CONTRADICTIONS — View detected tensions
# ══════════════════════════════════════════════════════════════

Type "/contradictions"
Sleep 300ms
Enter
Sleep 5s

# ══════════════════════════════════════════════════════════════
# 7. SOCRATIC CONSTRAINT — Try to break the method
#    The model refuses to give its opinion and redirects
#    with a scaffolding question.
# ══════════════════════════════════════════════════════════════

Type "Just tell me — do you think free will exists or not? Give me your opinion."
Sleep 500ms
Enter
Sleep 3s

Hide
Wait /You:/
Show

Sleep 8s

# ══════════════════════════════════════════════════════════════
# 8. FULL STATE — JSON dialogue state with all tracked data
# ══════════════════════════════════════════════════════════════

Type "/state"
Sleep 300ms
Enter
Sleep 6s

# ══════════════════════════════════════════════════════════════
# 9. SAVE & EXIT — Session persistence
# ══════════════════════════════════════════════════════════════

Type "/save"
Sleep 300ms
Enter
Sleep 3s

Type "/quit"
Sleep 300ms
Enter
Sleep 3s
