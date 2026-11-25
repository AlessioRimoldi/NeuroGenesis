#!/usr/bin/env bash
# Regenerates the Python protobuf modules from proto/ using the exact protoc
# version the mod build uses (see mod/gradle.properties: protobuf_version).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="$(grep -E '^protobuf_version=' "$ROOT/mod/gradle.properties" | cut -d= -f2)"
find_protoc() {
	find "$HOME/.gradle/caches/modules-2/files-2.1/com.google.protobuf/protoc/$VERSION" \
		-name "protoc-$VERSION-linux-x86_64.exe" 2>/dev/null | head -n1
}
PROTOC="$(find_protoc)"
if [[ -z "$PROTOC" ]]; then
	echo "protoc $VERSION not cached yet; fetching it through Gradle..."
	(cd "$ROOT/mod" && ./gradlew -q generateProto)
	PROTOC="$(find_protoc)"
fi
chmod +x "$PROTOC"
OUT="$ROOT/python/src"
rm -f "$OUT"/neurogenesis/mc/v1/*_pb2.py "$OUT"/neurogenesis/mc/v1/*_pb2.pyi
"$PROTOC" --proto_path="$ROOT/proto" --python_out="$OUT" --pyi_out="$OUT" "$ROOT"/proto/neurogenesis/mc/v1/*.proto
echo "Generated Python protobuf modules with protoc $VERSION into $OUT/neurogenesis/mc/v1"
