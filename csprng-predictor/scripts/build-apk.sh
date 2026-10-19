#!/usr/bin/env bash
# =============================================================================
# scripts/build-apk.sh - SDK-less release build of CSPRNG Predictor
#
# Builds dist/csprng-predictor-v<versionName>.apk without the Android SDK or
# Gradle, using only:
#   * aapt2, zipalign, apksigner     (Ubuntu packages: aapt, zipalign, apksigner)
#   * d8 (preferred) or dx           (Ubuntu package: dalvik-exchange = legacy dx)
#   * a JDK >= 9 (javac --release 8), zip, curl
#   * an API-34 framework jar: $ANDROID_JAR, or the SDK's
#     platforms/android-34/android.jar, or Robolectric's
#     org.robolectric:android-all:14-robolectric-10818077 from Maven Central
#     (downloaded once into tools/android-all-14.jar).
#
# Run from any directory:   ./scripts/build-apk.sh
#
# Environment overrides:
#   ANDROID_JAR                       framework jar to compile/link against
#   ANDROID_HOME / ANDROID_SDK_ROOT   SDK used to find android.jar and build-tools
#   KEYSTORE, KS_ALIAS, KS_PASS       signing key (default: keystore/debug.keystore,
#   KEY_PASS                          androiddebugkey, "android"; KEY_PASS defaults
#                                     to KS_PASS) - set these for a real release key
# =============================================================================
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
fi

START_SECONDS=$SECONDS
step() { printf '\n==> %s\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '    WARNING: %s\n' "$*" >&2; }
die()  { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

# -----------------------------------------------------------------------------
# 1. Paths
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TOOLS="$ROOT/tools"
BUILD="$ROOT/build"
DIST="$ROOT/dist"
MAIN_DIR="$ROOT/app/src/main"
RES_DIR="$MAIN_DIR/res"
JAVA_DIR="$MAIN_DIR/java"
MANIFEST="$MAIN_DIR/AndroidManifest.xml"
CORE_DIR="$JAVA_DIR/com/medonumbworks/csprngpredictor/core"
VERSION_FILE="$ROOT/version.properties"

MIN_SDK=24
TARGET_SDK=34
ROBOLECTRIC_VERSION="14-robolectric-10818077"
ROBOLECTRIC_URL="https://repo1.maven.org/maven2/org/robolectric/android-all/${ROBOLECTRIC_VERSION}/android-all-${ROBOLECTRIC_VERSION}.jar"
SDK_DIR="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-}}"

step "Project"
info "root:     $ROOT"

[[ -f "$MANIFEST" ]] || die "missing $MANIFEST"
[[ -d "$RES_DIR" ]]  || die "missing $RES_DIR"
[[ -d "$JAVA_DIR" ]] || die "missing $JAVA_DIR"
[[ -f "$VERSION_FILE" ]] || die "missing $VERSION_FILE (expected versionCode=... and versionName=...)"

read_prop() {  # read_prop KEY FILE -> value of the last KEY=value line (trimmed)
    local value
    value="$(grep -E "^[[:space:]]*$1[[:space:]]*=" "$2" | tail -n 1 | cut -d= -f2- | tr -d '\r')" || true
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}
VERSION_CODE="$(read_prop versionCode "$VERSION_FILE")"
VERSION_NAME="$(read_prop versionName "$VERSION_FILE")"
[[ "$VERSION_CODE" =~ ^[0-9]+$ ]] || die "versionCode in $VERSION_FILE must be a positive integer (got '$VERSION_CODE')"
[[ "$VERSION_NAME" =~ ^[A-Za-z0-9._+-]+$ ]] || die "versionName in $VERSION_FILE must match [A-Za-z0-9._+-]+ (got '$VERSION_NAME')"
OUT_APK="$DIST/csprng-predictor-v$VERSION_NAME.apk"
info "version:  $VERSION_NAME (code $VERSION_CODE)"
info "output:   $OUT_APK"

# -----------------------------------------------------------------------------
# 2. Framework jar
# -----------------------------------------------------------------------------
step "Framework jar (API $TARGET_SDK)"
if [[ -n "${ANDROID_JAR:-}" ]]; then
    [[ -f "$ANDROID_JAR" ]] || die "ANDROID_JAR is set to '$ANDROID_JAR' but that file does not exist"
    info "using \$ANDROID_JAR"
elif [[ -n "$SDK_DIR" && -f "$SDK_DIR/platforms/android-$TARGET_SDK/android.jar" ]]; then
    ANDROID_JAR="$SDK_DIR/platforms/android-$TARGET_SDK/android.jar"
    info "using the SDK platform jar"
else
    ANDROID_JAR="$TOOLS/android-all-14.jar"
    if [[ -s "$ANDROID_JAR" ]]; then
        info "using cached Robolectric android-all jar"
    else
        command -v curl >/dev/null 2>&1 || die "curl is required to download $ROBOLECTRIC_URL"
        mkdir -p "$TOOLS"
        info "downloading org.robolectric:android-all:$ROBOLECTRIC_VERSION (~130 MB) from Maven Central"
        rm -f "$ANDROID_JAR.part"
        curl -fL --retry 3 -o "$ANDROID_JAR.part" "$ROBOLECTRIC_URL" \
            || die "download failed: $ROBOLECTRIC_URL (set ANDROID_JAR to a local android.jar instead)"
        if command -v sha1sum >/dev/null 2>&1; then
            expected_sha1="$(curl -fsSL --retry 3 "$ROBOLECTRIC_URL.sha1" 2>/dev/null | tr -d '[:space:]' | cut -c1-40)" || expected_sha1=""
            if [[ -n "$expected_sha1" ]]; then
                actual_sha1="$(sha1sum "$ANDROID_JAR.part" | cut -d' ' -f1)"
                if [[ "$actual_sha1" != "$expected_sha1" ]]; then
                    rm -f "$ANDROID_JAR.part"
                    die "checksum mismatch for the downloaded jar (expected $expected_sha1, got $actual_sha1)"
                fi
                info "sha1 verified: $actual_sha1"
            else
                warn "could not fetch the .sha1 file; skipping checksum verification"
            fi
        fi
        mv -f "$ANDROID_JAR.part" "$ANDROID_JAR"
    fi
fi
info "$ANDROID_JAR"

# -----------------------------------------------------------------------------
# 3. Tools
# -----------------------------------------------------------------------------
step "Tools"
find_tool() {  # find_tool NAME... -> first match on PATH, else newest SDK build-tools
    local name dir
    for name in "$@"; do
        if command -v "$name" >/dev/null 2>&1; then
            command -v "$name"
            return 0
        fi
    done
    if [[ -n "$SDK_DIR" && -d "$SDK_DIR/build-tools" ]]; then
        while IFS= read -r dir; do
            for name in "$@"; do
                if [[ -x "$SDK_DIR/build-tools/$dir/$name" ]]; then
                    printf '%s\n' "$SDK_DIR/build-tools/$dir/$name"
                    return 0
                fi
            done
        done < <(ls -1 "$SDK_DIR/build-tools" 2>/dev/null | sort -rV)
    fi
    return 1
}

AAPT2="$(find_tool aapt2)"         || die "aapt2 not found (Ubuntu: sudo apt install aapt)"
ZIPALIGN="$(find_tool zipalign)"   || die "zipalign not found (Ubuntu: sudo apt install zipalign)"
APKSIGNER="$(find_tool apksigner)" || die "apksigner not found (Ubuntu: sudo apt install apksigner)"
if DEX_BIN="$(find_tool d8)"; then
    DEXER="d8"
elif DEX_BIN="$(find_tool dalvik-exchange dx)"; then
    DEXER="dx"
else
    die "no dexer found: install the SDK build-tools (d8) or Ubuntu's dalvik-exchange (dx)"
fi
command -v javac >/dev/null 2>&1 || die "javac not found (Ubuntu: sudo apt install openjdk-21-jdk-headless)"
command -v zip   >/dev/null 2>&1 || die "zip not found (Ubuntu: sudo apt install zip)"
javac --help 2>&1 | grep -q -- '--release' || die "javac does not support --release; use JDK 9 or newer"
AAPT=""
if command -v aapt >/dev/null 2>&1; then AAPT="$(command -v aapt)"; fi

info "aapt2:     $AAPT2 ($("$AAPT2" version 2>&1 | head -n 1))"
info "dexer:     $DEXER -> $DEX_BIN"
info "zipalign:  $ZIPALIGN"
info "apksigner: $APKSIGNER ($("$APKSIGNER" --version 2>/dev/null | head -n 1))"
info "javac:     $(command -v javac) ($(javac -version 2>&1 | grep -v '^Picked up' | head -n 1))"

# -----------------------------------------------------------------------------
# 4. Resources
# -----------------------------------------------------------------------------
mkdir -p "$BUILD" "$DIST"
# Serialise concurrent builds of the same checkout (they share build/).
if command -v flock >/dev/null 2>&1; then
    exec 9>"$BUILD/.build-apk.lock"
    if ! flock -n 9; then
        info "another build-apk.sh is running; waiting for it to finish"
        flock 9
    fi
fi

step "Clean build outputs"
# Only the APK build's own outputs are removed, so a concurrent
# scripts/run-tests.sh (build/test-classes) is not disturbed.
rm -rf "$BUILD/res.zip" "$BUILD/gen" "$BUILD/classes" "$BUILD/dex" \
       "$BUILD/unaligned.apk" "$BUILD/withdex.apk" "$BUILD/aligned.apk" \
       "$BUILD/signed.apk" "$BUILD/signed.apk.idsig" "$BUILD/javap.txt"
mkdir -p "$BUILD/gen" "$BUILD/classes" "$BUILD/dex"
info "$BUILD"

step "aapt2 compile"
"$AAPT2" compile --dir "$RES_DIR" -o "$BUILD/res.zip"
info "$BUILD/res.zip"

step "aapt2 link"
# --replace-version makes version.properties authoritative even when the
# manifest also carries versionCode/versionName.
"$AAPT2" link -o "$BUILD/unaligned.apk" \
    -I "$ANDROID_JAR" \
    --manifest "$MANIFEST" \
    --java "$BUILD/gen" \
    --min-sdk-version "$MIN_SDK" \
    --target-sdk-version "$TARGET_SDK" \
    --version-code "$VERSION_CODE" \
    --version-name "$VERSION_NAME" \
    --replace-version \
    --auto-add-overlay \
    "$BUILD/res.zip"
info "$BUILD/unaligned.apk"

# -----------------------------------------------------------------------------
# 5. Java
# -----------------------------------------------------------------------------
step "Source guards"
if [[ -d "$CORE_DIR" ]]; then
    if core_android="$(grep -rnE '^[[:space:]]*import[[:space:]]+(static[[:space:]]+)?android(x)?\.' "$CORE_DIR" --include='*.java')"; then
        printf '%s\n' "$core_android" >&2
        die "the core package must not import android.* (see docs/SPEC.md section 0)"
    fi
fi
info "core package has no android.* imports"

step "javac --release 8"
SOURCES=()
while IFS= read -r -d '' f; do
    SOURCES+=("$f")
done < <(find "$JAVA_DIR" "$BUILD/gen" -type f -name '*.java' -print0 | sort -z)
[[ ${#SOURCES[@]} -gt 0 ]] || die "no Java sources found under $JAVA_DIR"
info "${#SOURCES[@]} source files (app/src/test is not part of the APK)"
javac --release 8 -Xlint:-options -encoding UTF-8 \
    -cp "$ANDROID_JAR" \
    -d "$BUILD/classes" \
    "${SOURCES[@]}"

CLASS_FILES=()
while IFS= read -r -d '' f; do
    CLASS_FILES+=("$f")
done < <(find "$BUILD/classes" -type f -name '*.class' -print0 | sort -z)
[[ ${#CLASS_FILES[@]} -gt 0 ]] || die "javac produced no class files"
info "${#CLASS_FILES[@]} class files"

step "Bytecode guards"
# dx cannot dex invokedynamic (lambdas / method references) at minSdk 24, and
# these library packages do not exist on API 24 (or are banned by the spec).
if command -v javap >/dev/null 2>&1; then
    javap -c -p "${CLASS_FILES[@]}" > "$BUILD/javap.txt" 2>/dev/null \
        || die "javap failed on the compiled classes"
    if grep -q 'invokedynamic' "$BUILD/javap.txt"; then
        for f in "${CLASS_FILES[@]}"; do
            if javap -c -p "$f" 2>/dev/null | grep -c 'invokedynamic' >/dev/null; then
                printf '    invokedynamic in %s\n' "${f#"$BUILD/classes/"}" >&2
            fi
        done
        die "lambdas / method references found; use anonymous inner classes (docs/SPEC.md section 0)"
    fi
    BANNED_RE='// (Method|InterfaceMethod|Field|class) .*(java/util/stream/|java/time/|java/util/Base64|java/nio/file/|org/json/)'
    if grep -qE "$BANNED_RE" "$BUILD/javap.txt"; then
        for f in "${CLASS_FILES[@]}"; do
            if banned="$(javap -c -p "$f" 2>/dev/null | grep -E "$BANNED_RE")"; then
                printf '    in %s:\n' "${f#"$BUILD/classes/"}" >&2
                printf '%s\n' "$banned" | sed -e 's/^.*\/\/ /        /' | sort -u | head -n 20 >&2
            fi
        done
        die "use of a banned / API 26+ library (java.util.stream, java.time, java.util.Base64, java.nio.file, org.json)"
    fi
    info "no invokedynamic, no banned library references"
else
    warn "javap not found; skipping bytecode guards (the dexer will still reject lambdas)"
fi

# -----------------------------------------------------------------------------
# 6. Dex
# -----------------------------------------------------------------------------
step "Dex ($DEXER)"
if [[ "$DEXER" == "d8" ]]; then
    "$DEX_BIN" --release --min-api "$MIN_SDK" --lib "$ANDROID_JAR" \
        --output "$BUILD/dex" "${CLASS_FILES[@]}"
else
    "$DEX_BIN" --dex --min-sdk-version="$MIN_SDK" \
        --output="$BUILD/dex/classes.dex" "$BUILD/classes"
fi
DEX_FILES=()
while IFS= read -r -d '' f; do
    DEX_FILES+=("$(basename "$f")")
done < <(find "$BUILD/dex" -maxdepth 1 -type f -name 'classes*.dex' -print0 | sort -z)
[[ ${#DEX_FILES[@]} -gt 0 ]] || die "the dexer produced no classes.dex"
info "${DEX_FILES[*]}"

# -----------------------------------------------------------------------------
# 7. Package, align, sign
# -----------------------------------------------------------------------------
step "Add dex to the APK"
cp "$BUILD/unaligned.apk" "$BUILD/withdex.apk"
(cd "$BUILD/dex" && zip -q -X ../withdex.apk "${DEX_FILES[@]}")
info "$BUILD/withdex.apk"

step "zipalign"
"$ZIPALIGN" -f -p 4 "$BUILD/withdex.apk" "$BUILD/aligned.apk"
"$ZIPALIGN" -c -p 4 "$BUILD/aligned.apk" || die "zipalign verification failed"
info "$BUILD/aligned.apk"

step "apksigner sign (v1 + v2 + v3)"
KEYSTORE="${KEYSTORE:-$ROOT/keystore/debug.keystore}"
KS_ALIAS="${KS_ALIAS:-androiddebugkey}"
export KS_PASS="${KS_PASS:-android}"
export KEY_PASS="${KEY_PASS:-$KS_PASS}"
[[ -f "$KEYSTORE" ]] || die "keystore not found: $KEYSTORE"
if [[ "$KEYSTORE" == "$ROOT/keystore/debug.keystore" ]]; then
    info "signing with the DEBUG key (public, not for store distribution)"
fi
info "keystore: $KEYSTORE (alias $KS_ALIAS)"
# Passwords are passed through the environment so they never show up in `ps`.
"$APKSIGNER" sign \
    --ks "$KEYSTORE" \
    --ks-pass env:KS_PASS \
    --ks-key-alias "$KS_ALIAS" \
    --key-pass env:KEY_PASS \
    --v1-signing-enabled true \
    --v2-signing-enabled true \
    --v3-signing-enabled true \
    --out "$BUILD/signed.apk" \
    "$BUILD/aligned.apk" \
    || die "apksigner sign failed (check KEYSTORE / KS_ALIAS / KS_PASS / KEY_PASS)"

step "apksigner verify"
# Verify as if for API 21: with the APK's own minSdk (24) apksigner skips the
# v1 (JAR) signature, which Android 7.0+ ignores; this way all of v1, v2 and v3
# are checked.
"$APKSIGNER" verify --verbose --min-sdk-version 21 "$BUILD/signed.apk" \
    || die "signature verification failed"

# Publish atomically: dist/ never holds a half-written APK.
cp -f "$BUILD/signed.apk" "$OUT_APK.tmp"
mv -f "$OUT_APK.tmp" "$OUT_APK"

if [[ -n "$AAPT" ]]; then
    step "aapt dump badging"
    "$AAPT" dump badging "$OUT_APK" 2>/dev/null \
        | grep -E "^(package|sdkVersion|targetSdkVersion|application-label:|application-icon-|application:|launchable-activity)" \
        | sed 's/^/    /' || warn "aapt dump badging printed nothing"
fi

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
step "Done in $((SECONDS - START_SECONDS))s"
SIZE_BYTES="$(wc -c < "$OUT_APK" | tr -d '[:space:]')"
info "APK:    $OUT_APK"
info "size:   $SIZE_BYTES bytes ($(( (SIZE_BYTES + 1023) / 1024 )) KiB)"
if command -v sha256sum >/dev/null 2>&1; then
    info "sha256: $(sha256sum "$OUT_APK" | cut -d' ' -f1)"
fi
info "install: adb install -r \"$OUT_APK\""
