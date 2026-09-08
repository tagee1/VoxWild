## VoxWild 1.4.0

The biggest release yet — a full audiobook builder, inline speech tags, a clips
library, read-along highlighting, Spanish voices, and a long list of fixes.

---

## New

### 📖 Audiobook mode

A new tab that turns a whole book into a finished audiobook.

Paste or import your text and VoxWild finds the chapters for you — you can edit
them inline before building. Add a title, author, year and cover image, pick a
voice, and press Build.

You get both formats, because different apps want different things:

- **A folder of per-chapter MP3s**, tagged with album, artist, title, track number
  and cover art — works everywhere, including Smart AudioBook Player
- **A single chaptered MP3** with proper chapter markers, read by Voice, VLC, mpv
  and AntennaPod

Every chapter is normalized to a consistent loudness, and a long build checkpoints
as it goes — if something interrupts it, you don't start over.

### 🏷 Inline speech tags

Direct the voice from inside your text. Type `[` in any text box for a searchable
menu, or use the **Tag guide** button.

```
Wait [pause 1s] listen carefully.
That was [loud]completely unacceptable[/loud].
She whispered [quiet]I know what you did[/quiet].
The code is [spell]R4T9[/spell].
Published in [year]1982[/year].
[voice: George]Who's there?[/voice]
```

| Tag | Effect |
|---|---|
| `[pause 1s]` | A pause — also `[pause 500ms]`. No closing tag needed |
| `[slow]` `[fast]` | Slow down or speed up a stretch |
| `[loud]` `[quiet]` | Louder or softer |
| `[spell]` | Read characters one at a time — codes, references |
| `[digits]` | Read a number digit by digit |
| `[year]` | `1982` → "nineteen eighty-two" |
| `[voice: Name]` | Switch narrator mid-sentence (Fast mode) |

Tags work in **both engines** and in **both** the Studio and Dialogue boxes. They
colour green as you type when recognised, orange when not — so a typo shows up
before you generate rather than being read aloud.

### 📁 Clips Library

Save generations into folders instead of losing them off the end of History.
Search, rename, move and export. Deleted clips go to **Recently Deleted** with a
6-second undo and a 30-day grace period.

### 👀 Read-along highlighting

Follow the text as it plays, karaoke style — by **sentence** or by **word**. Off by
default; the switch is in the text panel.

### ▶ Quick preview

Hear just the **first sentence** before committing to a full render. Same voice,
speed and effects as the real thing, so a long job stops being a gamble. Preview
sits next to Generate, or press **Ctrl+Shift+Enter**. Nothing is added to History.

### 🔊 Export loudness + new formats

A new **EXPORT** section in the FX panel normalizes your files to a broadcast
standard on save:

- **Podcast** −16 LUFS · **YouTube** −14 LUFS · **Broadcast** −23 LUFS

Every file in a batch hits the same target, so chapters and episodes are
consistent with each other. Applied only when writing files — what you hear while
editing is untouched.

**FLAC** and **OGG Vorbis** join WAV and MP3.

### 🇪🇸 Spanish voices

Spanish is now available in Fast mode.

### 📊 A progress bar that tells the truth

The old bar jumped to 90% and sat there. The new one shows **one cell per chunk**,
lights them up as real work completes, and gives a time estimate that corrects
itself as it learns your machine's actual speed. The model is warmed at launch, so
your first generation isn't the slow one.

---

## Improved

### Stop actually stops

One press, and everything halts immediately — no second press, no waiting for the
current chunk to finish.

It now works during **AI Enhancement**, where the button used to be greyed out
entirely, and in the **Queue**, **Dialogue** and **Audiobook** tabs, which
previously ignored it. The button reads **"Cancelling…"** so you know it landed.

### Numbers and symbols read the way you'd say them

Recipes, instructions and reference material used to come out garbled. Now:

| You type | VoxWild says |
|---|---|
| `2 1/2 cups` | two and a half cups |
| `3/4` | three quarters |
| `195°F` | one hundred ninety-five degrees Fahrenheit |
| `22-25 minutes` | twenty-two to twenty-five minutes |
| `16:9` | sixteen to nine |
| `555-1234` | five five five, one two three four |
| `3/4/2026` | March fourth twenty twenty-six |
| `3:30 pm` | three thirty pm |

Degree symbols that *look* identical but aren't — `°`, `º`, `˚`, `℃`, `℉` — all work
now. Phone numbers, dates, price ranges and hyphenated words are left alone.

### Dashes pause

A dash — like this one — now inserts a real pause instead of being skipped or read
out as "dash". Works with `--`, en dashes and em dashes.

Pasting from Word or Google Docs used to strip these out before generation.

### Text cleanup everywhere

**Auto-clean** now runs on the Dialogue and Audiobook tabs too, on both paste and
import. Chapter headings are preserved so chapter detection still works.

### A sound when it's done

A chime plays and the taskbar flashes when a long generation finishes, so you can
work elsewhere while it runs. Configurable in Settings, including how long a job
has to be before it notifies you.

---

## Fixed

**Dialogue**
- The **Rename** button for speakers was never actually visible on screen. It now
  appears, tells you how many lines it changed, and keeps your voice assignments
- Dialogue generations could not be cancelled — the Cancel button was the same
  colour as the panel behind it
- Speaker names no longer have to be ALL CAPS
- A line like `See https://example.com` is no longer mistaken for a speaker

**Queue**
- Select several items with **Ctrl+click**; clear the selection with **Escape** or a
  click below the list. Previously the highlight could never be cleared and only
  one item could ever be selected
- **Remove Selected** now removes all of them, not just the first

**Audiobook**
- **Clear** resets everything — chapters, cover, title, author and year — so a second
  book no longer inherits the first one's details. It asks first
- Title, author and year fill themselves in from a labelled header block, which is
  then removed so it isn't read aloud

**History and effects**
- Enhanced clips now show an **Orig** button, so you can hear the original against
  the enhanced version. It was previously being drawn one pixel wide
- Audio FX settings save as you change them, not only on a clean exit
- A cancelled or failed generation no longer chimes or claims "Done in 12s"
- The progress bar's status text no longer gets clipped mid-generation

**Pronunciation dictionary**
- Entries ending in punctuation now work — `i.e.`, `e.g.`, `etc.`, `Dr.`, `C++` all
  silently did nothing before, including two of the built-in defaults
- The column headers described the columns backwards, so anyone reading them
  entered every pair inverted. Corrected, with a tip explaining the trick: spell it
  how it sounds, not how it's written

**Windows and layout**
- Dialogs open centred on high-DPI displays instead of drifting off to one side
- **Settings** and **About** were taller than a 1080p screen allowed, putting Save and
  Close below the bottom edge

**Natural mode**
- Fixed setup failing on clean machines and an oversized model download
- Fixed a crash when running AI Enhancement on multiple clips
- Cancelling a Fast-mode job no longer unloads Natural mode in the background

---

**Updating:** no new dependencies, so the in-app updater will take you straight
there — no reinstall needed.
