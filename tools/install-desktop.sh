#!/usr/bin/env bash
#
# Install (or remove) the desktop entry and icon theme files, so the panel and
# menu show the timer's own icon instead of the generic Python one.
#
# Everything lands under ~/.local/share -- no root, no system files touched.
#
#   ./tools/install-desktop.sh              # install
#   ./tools/install-desktop.sh --uninstall  # remove

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_ID="naive-timer"

ICON_SRC="$REPO/src/naive_timer/icons"
DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
APPS="$DATA/applications"
THEME="$DATA/icons/hicolor"

refresh_caches() {
    # Best-effort: the desktop still works without these, it just may take a
    # menu restart to notice the change.
    command -v update-desktop-database >/dev/null && update-desktop-database "$APPS" 2>/dev/null || true
    command -v gtk-update-icon-cache   >/dev/null && gtk-update-icon-cache -f -t "$THEME" 2>/dev/null || true
}

if [ "${1:-}" = "--uninstall" ]; then
    rm -f "$APPS/$APP_ID.desktop"
    find "$THEME" -name "$APP_ID.png" -delete 2>/dev/null || true
    refresh_caches
    echo "removed $APP_ID desktop entry and icons"
    exit 0
fi

if [ ! -d "$ICON_SRC" ]; then
    echo "error: no icons at $ICON_SRC" >&2
    exit 1
fi

# Icons: hicolor wants one file per size directory, named after the app.
count=0
for png in "$ICON_SRC"/timer-icon-*.png; do
    [ -e "$png" ] || continue
    size="${png##*-}"
    size="${size%.png}"
    dest="$THEME/${size}x${size}/apps"
    mkdir -p "$dest"
    cp -f "$png" "$dest/$APP_ID.png"
    count=$((count + 1))
done

# Desktop entry: Exec has to be absolute, so bake in this checkout's launcher.
# --no-panel keeps the dev tuning window out of a menu launch; drop it if you
# want the panel when starting from the menu.
mkdir -p "$APPS"
sed "s|@EXEC@|$REPO/launch.sh --no-panel|" \
    "$REPO/packaging/$APP_ID.desktop.in" >"$APPS/$APP_ID.desktop"
chmod 644 "$APPS/$APP_ID.desktop"

refresh_caches

echo "installed $count icon size(s) into $THEME"
echo "installed $APPS/$APP_ID.desktop"
