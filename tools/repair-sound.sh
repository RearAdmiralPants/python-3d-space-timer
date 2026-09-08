#!/usr/bin/env bash
#
# repair-sound.sh -- diagnose, record, then repair the "audio sink goes dead"
# wedge on this machine.
#
# THE SYMPTOM
#
# Audio stops. Adjusting the volume does nothing, "Test sound" does nothing,
# and the cure is switching sinks (headphones -> speakers -> headphones), after
# which everything the system tried to play comes out at once. Spotify keeps
# showing a running clock throughout.
#
# ROOT CAUSE (found 2026-08-09)
#
# The SoundWire playback PCM enters an xrun loop it cannot recover from. In the
# pipewire journal, thousands per second:
#
#     spa.alsa: hw:sofsoundwirep: (375 suppressed)
#               snd_pcm_avail after recover: Broken pipe
#
# -EPIPE is an xrun; getting it again *immediately after* recovery means the
# recovery does not take. PipeWire spins for minutes, the node eventually
# suspends, the PCM closes, and client streams are left corked with nothing to
# resume them. Switching sinks cures it because that forces a fresh PCM open.
#
# So the states this script detects are the AFTERMATH, not the cause. Both are
# still worth detecting and repairing -- they are what is left to clean up --
# but the fix for the underlying fault is not in here.
#
# Correlates with Moonlight: 7 of 8 xrun episodes since Aug 1 had it running
# within half an hour, and it runs a small fraction of the time. NOT via
# creating sinks (none present at wedge time) and NOT via its split-lock traps
# (258 trap-minutes vs 43 xrun-minutes, only 7 overlapping -- refuted). The
# same error is reported elsewhere specifically under remote-play and game
# streaming workloads, so the mechanism is likely load- or timing-related
# rather than anything Moonlight does to the graph.
#
# It is inspiron-only, and the hardware is why: `hw:sofsoundwirep` is the
# SoundWire PCM (RT711 + 2x RT1308 + RT715 across four master links). dove runs
# identical PipeWire/WirePlumber/Mint but is plain sof-hda-dsp, and has never
# shown it.
#
# Captured mid-wedge twice, every ALSA PCM was `closed` and every sink node was
# healthy -- the damage is done by then and the device has already been closed.
#
# TWO DISTINCT FAILURE MODES, and they need different detection
#
# 1. ORPHANED. The stream lost its sink entirely:
#
#        Sink: 4294967295        <- PA_INVALID_INDEX: attached to NO sink
#        Corked: yes
#
#    and its node had zero links in the graph. Seen 2026-08-07 -- but that was
#    self-inflicted, by `systemctl --user restart wireplumber` during
#    debugging, which destroys and recreates every device node.
#
# 2. STUCK CORKED. The stream is attached to the *correct* default sink and is
#    simply corked forever, so nothing is ever written and the sink stays
#    suspended:
#
#        Sink: 23194             <- correct default sink
#        Corked: yes
#
#    Seen 2026-08-08, organically, triggered by clicking fast-forward in
#    Spotify. This is the mode actually encountered in normal use; mode 1 has
#    only ever been produced deliberately.
#
# Mode 2 cannot be detected from the audio graph alone: a normally paused
# Spotify looks identical. The discriminator is MPRIS over D-Bus -- if the
# player reports PlaybackStatus "Playing" while its stream is corked, the
# client and the server disagree, and that is the wedge. Confirmed against a
# healthy system, where MPRIS "Playing" coincides with `Corked: no`.
#
# WHAT TRIGGERS IT
#
# Mode 1: anything that destroys and recreates sink nodes -- Bluetooth
# connect/disconnect, DP/HDMI sinks coming and going with monitor DPMS (three
# of the five sinks here are HDMI), card profile switches, a wireplumber
# restart.
#
# Mode 2: downstream of the xrun loop above. The seek in Spotify that appeared
# to trigger it was a coincidence of timing -- the PCM had already been failing
# for minutes by then, and an earlier episode that evening (23:23-23:29)
# preceded both the seek and Moonlight's launch.
#
# To check whether an episode is in progress right now:
#
#     journalctl --user -u pipewire --since "-10 min" | grep "Broken pipe"
#
# NOT the timer app. It was not running for the 2026-08-07 instance at all, and
# for 2026-08-08 it was running but held no sink-input (since 8dcde8b it holds
# none unless an alarm is within a second). The app's status is recorded every
# run so each occurrence either implicates it or clears it rather than leaving
# it a standing suspect -- and a stream found while idle is itself a
# regression.
#
# A NOTE ON WHAT "REPAIRED" MEANS
#
# On 2026-08-08 the audio came back while this script ran, before it issued any
# repair -- the run found no orphan and exited without touching anything. So
# either the read-only enumeration below (pactl list / pw-dump / pw-cli) is
# itself enough to knock a stuck stream loose, or it was coincidence. The
# script now probes state before and after the read-only capture and says which
# happened, because that distinction decides whether a gentle fix exists.
#
# USAGE
#
#     tools/repair-sound.sh                 # record, then repair
#     tools/repair-sound.sh --observe-only  # record only, leave it broken
#
# Run it *while the sink is stuck*, before switching sinks by hand -- doing
# that first destroys the evidence.

set -uo pipefail

# Print the whole leading comment block, however long it grows.
usage() { awk 'NR>2 && /^#/ { sub(/^# ?/, ""); print; next } NR>2 { exit }' "$0"; }

# Is the underlying fault active right now? This is the one thing worth
# knowing before anything else: an episode in progress means the PCM is still
# failing and any repair below will not hold.
xrun_episode() {
	journalctl --user -u pipewire --no-pager --since "-15 min" 2>/dev/null \
		| grep -c "snd_pcm_avail after recover"
}

OBSERVE_ONLY=0
for arg in "$@"; do
	case "$arg" in
		--observe-only) OBSERVE_ONLY=1 ;;
		-h|--help) usage; exit 0 ;;
		*) echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
	esac
done

for tool in pactl pw-dump; do
	command -v "$tool" >/dev/null || { echo "missing required tool: $tool" >&2; exit 1; }
done

OUT="${TMPDIR:-/tmp}/audio-wedge-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"

# Everything printed also lands in the report, so a later debugging session can
# read one file instead of reassembling scrollback.
exec > >(tee "$OUT/report.txt") 2>&1

echo "=== repair-sound.sh  $(date -Is) ==="
echo "recording to $OUT"
echo

# --- Client identity ------------------------------------------------------
#
# Naming the broken client needs a join. Spotify's sink-input carries no
# application.name and no application.process.id at all -- only
# `media.name = "audio-src"` and a `Client:` reference -- so the identifying
# properties are read off the client object. Qt's streams do carry
# application.name directly, hence a fallback chain rather than one source.
declare -A CLIENT_NAME CLIENT_PID
load_clients() {
	local idx="" val
	while IFS= read -r line; do
		case "$line" in
			"Client #"*)
				idx="${line#Client #}" ;;
			*"application.name = "*)
				val="${line#*= \"}"; CLIENT_NAME[$idx]="${val%\"}" ;;
			*"application.process.id = "*)
				val="${line#*= \"}"; CLIENT_PID[$idx]="${val%\"}" ;;
		esac
	done < <(pactl list clients 2>/dev/null)
}

# Emits: index <TAB> sink <TAB> corked <TAB> pid <TAB> name
inventory() {
	pactl list sink-inputs 2>/dev/null | awk '
		function flush() {
			if (idx != "") {
				if (name == "?" && media != "") name = media
				printf "%s\t%s\t%s\t%s\t%s\t%s\n", idx, sink, cork, pid, name, client
			}
			idx = ""; sink = "?"; cork = "?"; pid = "?"; name = "?"
			client = "?"; media = ""
		}
		/^Sink Input #/ { flush(); idx = substr($3, 2) }
		/^\tSink: /     { sink = $2 }
		/^\tCorked: /   { cork = $2 }
		/^\tClient: /   { client = $2 }
		/application\.process\.id = / { p = $0; gsub(/.*= "|"$/, "", p); pid = p }
		/application\.name = /        { n = $0; gsub(/.*= "|"$/, "", n); name = n }
		/media\.name = /              { m = $0; gsub(/.*= "|"$/, "", m); media = m }
		END { flush() }
	' | while IFS=$'\t' read -r idx sink cork pid name client; do
		[ "$pid"  = "?" ] && pid="${CLIENT_PID[$client]:-?}"
		[ "$name" = "?" ] && name="${CLIENT_NAME[$client]:-?}"
		# Prefer the client's app name over a generic stream name like
		# "audio-src", which identifies nothing.
		[ -n "${CLIENT_NAME[$client]:-}" ] && name="${CLIENT_NAME[$client]}"
		printf '%s\t%s\t%s\t%s\t%s\n' "$idx" "$sink" "$cork" "$pid" "$name"
	done
}

# --- MPRIS: what the player believes -------------------------------------
#
# The only way to tell "stuck corked" from "paused on purpose". Emits:
#   busname <TAB> pid <TAB> PlaybackStatus
# The PID comes from D-Bus rather than the player, so it can be matched
# against the sink-input's client PID instead of guessing by name.
mpris_players() {
	command -v busctl >/dev/null || return 0
	busctl --user list --no-legend --no-pager 2>/dev/null \
		| awk '$1 ~ /^org\.mpris\.MediaPlayer2\./ { print $1 }' | sort -u \
		| while read -r bus; do
			local status pid
			status=$(busctl --user get-property "$bus" /org/mpris/MediaPlayer2 \
				org.mpris.MediaPlayer2.Player PlaybackStatus 2>/dev/null \
				| sed 's/^s //; s/"//g')
			pid=$(busctl --user call org.freedesktop.DBus /org/freedesktop/DBus \
				org.freedesktop.DBus GetConnectionUnixProcessID s "$bus" 2>/dev/null \
				| awk '{print $2}')
			[ -n "$status" ] && printf '%s\t%s\t%s\n' "$bus" "${pid:-?}" "$status"
		done
}

# Classify the current state. Emits one line per problem found:
#   MODE <TAB> index <TAB> detail
# MODE is ORPHANED or STUCK_CORKED.
classify() {
	local players; players=$(mpris_players)
	inventory | while IFS=$'\t' read -r idx sink cork pid name; do
		if [ "$sink" = "4294967295" ]; then
			printf 'ORPHANED\t%s\t%s (pid %s) has no sink\n' "$idx" "$name" "$pid"
			continue
		fi
		# Stuck corked: this stream's own player says it is playing.
		if [ "$cork" = "yes" ] && [ -n "$players" ]; then
			local claim
			claim=$(printf '%s\n' "$players" \
				| awk -F'\t' -v p="$pid" '$2 == p && $3 == "Playing" { print $1; exit }')
			[ -n "$claim" ] && printf 'STUCK_CORKED\t%s\t%s (pid %s) corked while %s reports Playing\n' \
				"$idx" "$name" "$pid" "$claim"
		fi
	done
}

load_clients

# --- Probe BEFORE the heavy capture --------------------------------------
#
# On 2026-08-08 the wedge cleared during a run that issued no repair at all, so
# the read-only capture itself is a suspect. Snapshotting here and again after
# it is the only way to tell a real fix from a coincidence.
BEFORE=$(classify)

echo "=== timer app ==="
APP_PIDS=$(pgrep -f '[n]aive_timer' 2>/dev/null | grep -vx "$$" | tr '\n' ' ')
APP_PIDS="${APP_PIDS% }"
if [ -n "$APP_PIDS" ]; then
	echo "RUNNING -- pids: $APP_PIDS"
	ps -o pid,lstart,etime,args -p "${APP_PIDS// /,}" 2>/dev/null | sed 's/^/  /'
else
	echo "NOT RUNNING"
fi
if git -C "$(dirname "$0")/.." rev-parse --short HEAD >/dev/null 2>&1; then
	echo "repo HEAD: $(git -C "$(dirname "$0")/.." rev-parse --short HEAD) \
($(git -C "$(dirname "$0")/.." rev-parse --abbrev-ref HEAD))"
fi
echo

echo "=== sink-inputs ==="
printf "  %-7s %-12s %-7s %-8s %s\n" INDEX SINK CORKED PID NAME
inventory | while IFS=$'\t' read -r idx sink cork pid name; do
	mark="   "
	[ "$sink" = "4294967295" ] && mark="!! "
	printf "%s%-7s %-12s %-7s %-8s %s\n" "$mark" "$idx" "$sink" "$cork" "$pid" "$name"
done
echo

echo "=== players (MPRIS) ==="
if [ -n "$(mpris_players)" ]; then
	mpris_players | while IFS=$'\t' read -r bus pid status; do
		printf "  %-8s %-10s %s\n" "$pid" "$status" "$bus"
	done
else
	echo "  (none, or busctl unavailable)"
fi
echo

echo "=== root cause: is the PCM failing? ==="
XRUNS=$(xrun_episode)
if [ "${XRUNS:-0}" -gt 0 ]; then
	echo "  YES -- $XRUNS 'snd_pcm_avail after recover' lines in the last 15 min."
	echo "  The SoundWire PCM is in an unrecoverable xrun loop. Anything below"
	echo "  is aftermath, and a repair may not hold until the loop stops."
	journalctl --user -u pipewire --no-pager --since "-15 min" 2>/dev/null \
		| grep "snd_pcm_avail after recover" | tail -2 | sed 's/^/    /'
	echo "  Moonlight running? $(pgrep -c -i moonlight 2>/dev/null || echo 0) process(es)."
else
	echo "  no -- nothing in the pipewire log in the last 15 min."
	echo "  If audio is dead, the PCM failed earlier and left the mess below."
fi
echo

echo "=== diagnosis ==="
if [ -n "$BEFORE" ]; then
	printf '%s\n' "$BEFORE" | while IFS=$'\t' read -r mode idx detail; do
		echo "  $mode: sink-input $idx -- $detail"
	done
else
	echo "  nothing wrong found"
fi
echo

# Does the timer app hold a stream? Post-8dcde8b it should hold none unless an
# alarm is within ALERT_WARMUP_S, so a hit here is itself a finding.
if [ -n "$APP_PIDS" ]; then
	APP_STREAMS=$(inventory | awk -F'\t' -v pids=" $APP_PIDS " \
		'index(pids, " " $4 " ") { print $1 }' | tr '\n' ' ')
	if [ -n "$APP_STREAMS" ]; then
		echo "NOTE: the timer app holds sink-input(s): $APP_STREAMS"
		echo "      Expected only within ~1s of an alarm. Otherwise a regression."
	else
		echo "the timer app is running and holds no sink-input (expected)"
	fi
	echo
fi

# --- Heavy, read-only capture --------------------------------------------
{
	echo "default-sink: $(pactl get-default-sink 2>/dev/null)"
	echo "pipewire: $(pipewire --version 2>&1 | tail -1)"
	echo "wireplumber: $(wireplumber --version 2>&1 | head -1)"
	echo "uptime: $(uptime)"
	echo
	# Daemon start times matter: pipewire up for weeks while wireplumber
	# restarted minutes ago means the session manager churned the nodes. That
	# is how the 2026-08-07 instance was traced to a manual wireplumber restart.
	echo "audio daemon start times:"
	ps -o pid,lstart,etime,comm -p "$(pgrep -d, -x 'pipewire|wireplumber|pipewire-pulse')" 2>/dev/null
} > "$OUT/context.txt" 2>&1

pactl list        > "$OUT/pactl-list.txt"  2>&1
pw-dump           > "$OUT/pw-dump.json"    2>&1
command -v pw-cli >/dev/null && pw-cli info all > "$OUT/pw-cli-info.txt" 2>&1
cat /proc/asound/card*/pcm*p/sub*/status > "$OUT/alsa-pcm-status.txt" 2>&1

# Sink node ids. PipeWire hands these out monotonically, so comparing against
# an earlier capture shows whether nodes were recreated -- the cheap proxy for
# "device churn happened" without a pw-mon running.
python3 - "$OUT/pw-dump.json" > "$OUT/node-ids.txt" 2>&1 <<'PY'
import json, sys
try:
    objs = json.load(open(sys.argv[1]))
except Exception as exc:
    print("could not parse pw-dump:", exc); raise SystemExit
for o in objs:
    if not o.get("type", "").endswith("Node"):
        continue
    info = o.get("info", {}) or {}
    props = info.get("props", {}) or {}
    cls = props.get("media.class", "")
    if cls in ("Audio/Sink", "Stream/Output/Audio"):
        print(f'{o["id"]:6}  {info.get("state","?"):10} {cls:20} '
              f'{props.get("node.name") or props.get("application.name")}')
PY

# The cause is in the seconds before the wedge, so these windows are what name
# a trigger once a few captures accumulate.
journalctl -k -b --no-pager --since "-30 min" 2>/dev/null \
	| grep -iE "bluetooth|drm|i915|hdmi|sof|sdw|snd_|xrun" > "$OUT/kernel.txt" 2>&1
journalctl --user --no-pager --since "-30 min" \
	-u pipewire -u pipewire-pulse -u wireplumber   > "$OUT/services.txt" 2>&1
journalctl --no-pager --since "-30 min" 2>/dev/null \
	-u bluetooth -u systemd-logind                 > "$OUT/system-events.txt" 2>&1

# --- Did merely looking fix it? ------------------------------------------
load_clients
AFTER=$(classify)
if [ -n "$BEFORE" ] && [ -z "$AFTER" ]; then
	echo "=== !! state changed during the READ-ONLY capture ==="
	echo "The problem cleared before any repair ran. Nothing above this point"
	echo "modifies audio state -- it is all pactl list / pw-dump / pw-cli /"
	echo "journalctl. So enumerating the graph appears to be enough to knock a"
	echo "stuck stream loose, which would mean a gentle fix exists."
	echo "This also happened on 2026-08-08. Two for two makes it a mechanism,"
	echo "not a coincidence -- worth reporting."
	echo
fi

# --- Repair ---------------------------------------------------------------
echo "=== repair ==="
if [ -z "$AFTER" ]; then
	if [ -n "$BEFORE" ]; then
		echo "already resolved (see above); nothing to repair."
	else
		echo "nothing wrong found."
		echo "If audio is dead anyway, this is a failure mode this script does"
		echo "not know about -- keep $OUT and say so; that is new data."
	fi
	echo
	echo "capture saved to $OUT"
	exit 0
fi

if [ "$OBSERVE_ONLY" = "1" ]; then
	echo "--observe-only: leaving it broken."
	printf '%s\n' "$AFTER" | while IFS=$'\t' read -r mode idx detail; do
		echo "  would repair $mode on sink-input $idx"
	done
	echo
	echo "capture saved to $OUT"
	exit 0
fi

DEFAULT_SINK=$(pactl get-default-sink)

# Stage 1, stuck-corked only: ask the player to re-state itself. This targets
# the actual disagreement and is far gentler than touching the graph -- no
# other stream is disturbed. Explicit Pause then Play rather than PlayPause,
# so the end state does not depend on who is right about the current one.
printf '%s\n' "$AFTER" | grep -q '^STUCK_CORKED' && {
	mpris_players | while IFS=$'\t' read -r bus pid status; do
		[ "$status" = "Playing" ] || continue
		echo "  asking $bus to re-state playback (Pause, then Play)"
		busctl --user call "$bus" /org/mpris/MediaPlayer2 \
			org.mpris.MediaPlayer2.Player Pause 2>/dev/null
		sleep 1
		busctl --user call "$bus" /org/mpris/MediaPlayer2 \
			org.mpris.MediaPlayer2.Player Play 2>/dev/null
	done
	sleep 1
	load_clients; AFTER=$(classify)
}

# Stage 2: re-route the stream, which forces a re-link.
if [ -n "$AFTER" ]; then
	printf '%s\n' "$AFTER" | while IFS=$'\t' read -r mode idx detail; do
		echo -n "  moving sink-input $idx -> $DEFAULT_SINK ... "
		pactl move-sink-input "$idx" "$DEFAULT_SINK" 2>/dev/null \
			&& echo "ok" || echo "FAILED"
	done
	sleep 1
	load_clients; AFTER=$(classify)
fi

# Stage 3: bounce the default sink. This is the by-hand cure, and it re-links
# everything rather than one stream, so it is last -- it briefly relocates all
# audio on the machine.
if [ -n "$AFTER" ]; then
	OTHER=$(pactl list short sinks | awk -v d="$DEFAULT_SINK" '$2 != d { print $2; exit }')
	if [ -n "$OTHER" ]; then
		echo "  still stuck; bouncing the default sink via $OTHER"
		pactl set-default-sink "$OTHER"; sleep 1
		pactl set-default-sink "$DEFAULT_SINK"; sleep 1
		load_clients; AFTER=$(classify)
	else
		echo "  still stuck, and no second sink to bounce through"
	fi
fi

echo
if [ -n "$AFTER" ]; then
	echo "RESULT: still stuck --"
	printf '%s\n' "$AFTER" | while IFS=$'\t' read -r mode idx detail; do
		echo "  $mode: sink-input $idx -- $detail"
	done
	echo "Switch sinks in the GUI and note whether that works; if it does,"
	echo "this script's repair is incomplete and that is worth recording."
else
	echo "RESULT: repaired. Play something to confirm it is actually audible."
fi
echo
echo "capture saved to $OUT"
