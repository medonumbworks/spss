#!/usr/bin/env bash
# =============================================================================
# scripts/run-tests.sh - JVM unit tests for the pure-Java core
#
#   1. Source guards over app/src/main/java: no lambdas / method references
#      (grep '\) *->|::'), no java.util.stream / java.time / java.util.Base64 /
#      java.nio.file / org.json / androidx imports, no android.* in core.
#   2. Compiles app/src/main/java/**/core/*.java (javac --release 8, without any
#      Android jar - so the core really is plain Java) and checks the bytecode
#      for invokedynamic.
#   3. Compiles app/src/test/java/**/*.java against the core + JUnit 4.
#   4. Runs org.junit.runner.JUnitCore over every *Test class found.
#
# Run from any directory:
#   ./scripts/run-tests.sh                      all tests
#   ./scripts/run-tests.sh DateUtilTest ...     only the named test classes
#                                               (simple or fully qualified names)
# Exit status: 0 when every guard and test passes, non-zero otherwise.
# JUnit / Hamcrest are downloaded once from Maven Central into tools/.
# =============================================================================
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
fi

step() { printf '\n==> %s\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TOOLS="$ROOT/tools"
MAIN_JAVA="$ROOT/app/src/main/java"
TEST_JAVA="$ROOT/app/src/test/java"
OUT="$ROOT/build/test-classes"
CORE_OUT="$OUT/core"
TEST_OUT="$OUT/test"

MAVEN="https://repo1.maven.org/maven2"
JUNIT_JAR="$TOOLS/junit-4.13.2.jar"
HAMCREST_JAR="$TOOLS/hamcrest-core-1.3.jar"
JUNIT_URL="$MAVEN/junit/junit/4.13.2/junit-4.13.2.jar"
HAMCREST_URL="$MAVEN/org/hamcrest/hamcrest-core/1.3/hamcrest-core-1.3.jar"

command -v javac >/dev/null 2>&1 || die "javac not found (Ubuntu: sudo apt install openjdk-21-jdk-headless)"
command -v java  >/dev/null 2>&1 || die "java not found"

FAILED=0

# -----------------------------------------------------------------------------
# 1. Source guards
# -----------------------------------------------------------------------------
step "Source guards (app/src/main/java)"
if [[ -d "$MAIN_JAVA" ]]; then
    # The dexer used by scripts/build-apk.sh (legacy dx) cannot handle lambdas.
    if hits="$(grep -rnE '\) *->|::' "$MAIN_JAVA" --include='*.java')"; then
        printf '%s\n' "$hits" | sed "s#^$ROOT/#    #" >&2
        printf '    FAIL: lambda / method reference syntax found (use anonymous inner classes)\n' >&2
        FAILED=1
    else
        info "no lambdas / method references"
    fi
    if hits="$(grep -rnE '^[[:space:]]*import[[:space:]]+(static[[:space:]]+)?(java\.util\.stream|java\.time|java\.util\.Base64|java\.nio\.file|org\.json|androidx)\b' "$MAIN_JAVA" --include='*.java')"; then
        printf '%s\n' "$hits" | sed "s#^$ROOT/#    #" >&2
        printf '    FAIL: banned / API 26+ import found (docs/SPEC.md section 0)\n' >&2
        FAILED=1
    else
        info "no banned imports"
    fi
    if hits="$(grep -rnE '^[[:space:]]*import[[:space:]]+(static[[:space:]]+)?android' "$MAIN_JAVA" --include='*.java' \
                 | grep -E '^[^:]*/core/[^:/]*\.java:')"; then
        printf '%s\n' "$hits" | sed "s#^$ROOT/#    #" >&2
        printf '    FAIL: the core package must not import android.*\n' >&2
        FAILED=1
    else
        info "core has no android.* imports"
    fi
else
    info "no $MAIN_JAVA yet"
fi

# -----------------------------------------------------------------------------
# 2. JUnit jars
# -----------------------------------------------------------------------------
fetch() {  # fetch URL DEST
    if [[ -s "$2" ]]; then
        return 0
    fi
    command -v curl >/dev/null 2>&1 || die "curl is required to download $1"
    mkdir -p "$(dirname "$2")"
    info "downloading $1"
    rm -f "$2.part"
    curl -fL --retry 3 -o "$2.part" "$1" || die "download failed: $1"
    mv -f "$2.part" "$2"
}

# -----------------------------------------------------------------------------
# 3. Collect sources
# -----------------------------------------------------------------------------
CORE_SOURCES=()
if [[ -d "$MAIN_JAVA" ]]; then
    while IFS= read -r -d '' f; do
        CORE_SOURCES+=("$f")
    done < <(find "$MAIN_JAVA" -type f -path '*/core/*.java' -print0 | sort -z)
fi
TEST_SOURCES=()
if [[ -d "$TEST_JAVA" ]]; then
    while IFS= read -r -d '' f; do
        TEST_SOURCES+=("$f")
    done < <(find "$TEST_JAVA" -type f -name '*.java' -print0 | sort -z)
fi

rm -rf "$OUT"
mkdir -p "$CORE_OUT" "$TEST_OUT"

# -----------------------------------------------------------------------------
# 4. Compile core
# -----------------------------------------------------------------------------
step "Compile core (javac --release 8, no Android classes on the classpath)"
if [[ ${#CORE_SOURCES[@]} -gt 0 ]]; then
    info "${#CORE_SOURCES[@]} source files"
    javac --release 8 -Xlint:-options -encoding UTF-8 -d "$CORE_OUT" "${CORE_SOURCES[@]}" \
        || die "core compilation failed"
    if command -v javap >/dev/null 2>&1; then
        CORE_CLASSES=()
        while IFS= read -r -d '' f; do
            CORE_CLASSES+=("$f")
        done < <(find "$CORE_OUT" -type f -name '*.class' -print0 | sort -z)
        if [[ ${#CORE_CLASSES[@]} -gt 0 ]]; then
            javap -c -p "${CORE_CLASSES[@]}" > "$OUT/core-javap.txt" 2>/dev/null || die "javap failed"
            if grep -q 'invokedynamic' "$OUT/core-javap.txt"; then
                printf '    FAIL: invokedynamic (lambda / method reference) in compiled core classes\n' >&2
                FAILED=1
            else
                info "bytecode: no invokedynamic"
            fi
        fi
    fi
else
    info "no core sources yet (app/src/main/java/**/core/*.java)"
fi

# -----------------------------------------------------------------------------
# 5. Compile and run tests
# -----------------------------------------------------------------------------
if [[ ${#TEST_SOURCES[@]} -eq 0 ]]; then
    step "Tests"
    info "NOTICE: no test sources under app/src/test/java - nothing to run."
    if [[ $FAILED -ne 0 ]]; then
        die "source guards failed (see above)"
    fi
    exit 0
fi

fetch "$JUNIT_URL" "$JUNIT_JAR"
fetch "$HAMCREST_URL" "$HAMCREST_JAR"
CP="$CORE_OUT:$JUNIT_JAR:$HAMCREST_JAR"

step "Compile tests"
info "${#TEST_SOURCES[@]} source files"
javac --release 8 -Xlint:-options -encoding UTF-8 -cp "$CP" -d "$TEST_OUT" "${TEST_SOURCES[@]}" \
    || die "test compilation failed"

# Discover test classes: top-level *Test.class files (inner classes have a '$').
ALL_TESTS=()
while IFS= read -r -d '' f; do
    rel="${f#"$TEST_OUT/"}"
    rel="${rel%.class}"
    ALL_TESTS+=("${rel//\//.}")
done < <(find "$TEST_OUT" -type f -name '*Test.class' ! -name '*$*' -print0 | sort -z)

TESTS=()
if [[ $# -gt 0 ]]; then
    for wanted in "$@"; do
        found=0
        for cls in "${ALL_TESTS[@]+"${ALL_TESTS[@]}"}"; do
            if [[ "$cls" == "$wanted" || "${cls##*.}" == "$wanted" ]]; then
                TESTS+=("$cls")
                found=1
            fi
        done
        [[ $found -eq 1 ]] || die "no test class named '$wanted'"
    done
else
    TESTS=("${ALL_TESTS[@]+"${ALL_TESTS[@]}"}")
fi

step "Run JUnit"
if [[ ${#TESTS[@]} -eq 0 ]]; then
    info "NOTICE: test sources compiled but no *Test classes were found - nothing to run."
    if [[ $FAILED -ne 0 ]]; then
        die "source guards failed (see above)"
    fi
    exit 0
fi
for cls in "${TESTS[@]}"; do
    info "$cls"
done

set +e
java -cp "$TEST_OUT:$CP" org.junit.runner.JUnitCore "${TESTS[@]}"
JUNIT_STATUS=$?
set -e

echo
if [[ $JUNIT_STATUS -ne 0 ]]; then
    die "JUnit reported failures (exit $JUNIT_STATUS)"
fi
if [[ $FAILED -ne 0 ]]; then
    die "all tests passed, but the source guards failed (see above)"
fi
echo "ALL TESTS PASSED (${#TESTS[@]} test classes)"
