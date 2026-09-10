#!/usr/bin/env bash
# Take the deployed meeting-minutes API from localhost-only to a public HTTPS
# endpoint the platform team can call. Run ON THE SERVER as root.
#
#   deploy/go-live.sh minutes.example.com storage.acme.com,cdn.acme.com
#
#   $1  DNS name already pointing at this box (A/AAAA record must resolve here,
#       or Let's Encrypt cannot issue a certificate)
#   $2  comma-separated storage/CDN hostname(s) the platform's signed video URLs
#       use, including any redirect target
#
# Idempotent: safe to re-run after fixing a mistake. Every step verifies itself
# and the script aborts rather than leaving a half-open box.
set -euo pipefail

DNS_NAME="${1:-}"
VIDEO_HOSTS="${2:-}"
ENV_FILE=/etc/coherex-minutes.env
CADDYFILE=/etc/caddy/Caddyfile

die() { echo "ERROR: $*" >&2; exit 1; }
step() { printf '\n=== %s ===\n' "$*"; }

[ -n "$DNS_NAME" ] || die "usage: $0 <dns-name> <storage-hosts>"
[ -n "$VIDEO_HOSTS" ] || die "usage: $0 <dns-name> <storage-hosts>"
[ "$(id -u)" = 0 ] || die "run as root"
[ -f "$ENV_FILE" ] || die "$ENV_FILE missing -- run deploy/README.md steps 1-3 first"

# --- 0. sanity: does the DNS name actually point here? -----------------------
step "Checking DNS"
resolved="$(getent hosts "$DNS_NAME" | awk '{print $1}' | sort -u | paste -sd' ' -)"
[ -n "$resolved" ] || die "$DNS_NAME does not resolve. Create the A record first."
mine="$(ip -4 -o addr show scope global | awk '{split($4,a,"/"); print a[1]}' | paste -sd' ' -)"
echo "  $DNS_NAME -> $resolved"
echo "  this host  -> $mine"
match=no
for r in $resolved; do for m in $mine; do [ "$r" = "$m" ] && match=yes; done; done
[ "$match" = yes ] || die "$DNS_NAME does not point at this host. Certificate issuance will fail."

# --- 1. the settings that must not reach a public endpoint -------------------
step "Locking down video ingestion"
# Both are set by hand during smoke testing; leaving either on once the API is
# reachable turns videoUrl into an SSRF probe of the internal network.
sed -i "s|^COHEREX_VIDEO_ALLOWED_HOSTS=.*|COHEREX_VIDEO_ALLOWED_HOSTS=${VIDEO_HOSTS}|" "$ENV_FILE"
sed -i "s|^COHEREX_MINUTES_ALLOW_PRIVATE_VIDEO_HOSTS=.*|COHEREX_MINUTES_ALLOW_PRIVATE_VIDEO_HOSTS=false|" "$ENV_FILE"
grep -q "^COHEREX_MINUTES_ALLOW_PRIVATE_VIDEO_HOSTS=false" "$ENV_FILE" \
  || die "failed to disable the private-host override"
grep -q "REPLACE_ME" "$ENV_FILE" && die "a REPLACE_ME placeholder is still in $ENV_FILE"
echo "  allowlist: $(grep ^COHEREX_VIDEO_ALLOWED_HOSTS= "$ENV_FILE" | cut -d= -f2)"
echo "  private-address override: disabled"

# --- 2. the model servers stay private --------------------------------------
step "Confirming the model servers are not exposed"
for unit in coherex-vllm coherex-llm; do
  grep -q -- "--host 127.0.0.1" "/etc/systemd/system/$unit.service" \
    || die "$unit is not bound to 127.0.0.1 -- fix it before opening 443"
  echo "  $unit: loopback only"
done

# --- 3. TLS -----------------------------------------------------------------
step "Installing Caddy and issuing a certificate"
if ! command -v caddy >/dev/null; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq debian-keyring \
    debian-archive-keyring apt-transport-https curl gnupg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
    | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  echo "deb [signed-by=/usr/share/keyrings/caddy-stable-archive-keyring.gpg] \
https://dl.cloudsmith.io/public/caddy/stable/deb/debian any-version main" \
    > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq caddy
fi
caddy version | sed 's/^/  /'

install -d /etc/caddy
cat > "$CADDYFILE" <<EOF
${DNS_NAME} {
    encode zstd gzip

    reverse_proxy 127.0.0.1:8080 {
        transport http {
            read_timeout 60s
        }
    }

    header {
        Strict-Transport-Security "max-age=31536000; includeSubDomains"
        X-Content-Type-Options "nosniff"
        Referrer-Policy "no-referrer"
        -Server
    }
}
EOF
caddy validate --config "$CADDYFILE" --adapter caddyfile >/dev/null \
  || die "Caddyfile did not validate"
systemctl enable --now caddy
systemctl reload caddy || systemctl restart caddy
echo "  Caddyfile installed for ${DNS_NAME}"

# --- 4. firewall ------------------------------------------------------------
# SSH first, and verified present, so enabling cannot lock us out.
step "Firewall"
command -v ufw >/dev/null || DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ufw
ufw allow 22/tcp >/dev/null
ufw allow 443/tcp >/dev/null
ufw status | grep -q "22/tcp" || die "SSH rule missing -- refusing to enable the firewall"
ufw --force enable >/dev/null
ufw status verbose | sed 's/^/  /'
# 8000/8001/8080 are deliberately absent: they are loopback-only services.

# --- 5. restart and verify --------------------------------------------------
step "Restarting the API with the new configuration"
systemctl restart coherex-minutes-api coherex-minutes-worker
for _ in $(seq 1 30); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/health)" = 200 ] && break
  sleep 2
done

step "Verifying from the outside"
for _ in $(seq 1 30); do
  code="$(curl -s -o /dev/null -w '%{http_code}' "https://${DNS_NAME}/health" || true)"
  [ "$code" = 200 ] && break
  sleep 4
done
[ "${code:-}" = 200 ] || die "https://${DNS_NAME}/health returned ${code:-no response}. Check: journalctl -u caddy -n 50"
echo "  https://${DNS_NAME}/health -> 200"

unauth="$(curl -s -o /dev/null -w '%{http_code}' "https://${DNS_NAME}/v1/meeting-minutes/x/status")"
[ "$unauth" = 401 ] || die "expected 401 without a bearer token, got $unauth"
echo "  unauthenticated request -> 401"

step "Conformance check against the public endpoint"
KEY="$(grep ^AI_SERVICE_API_KEY= "$ENV_FILE" | cut -d= -f2)"
/opt/coherex-venv/bin/python /opt/coherex/scripts/check_deployed_api.py \
  --base-url "https://${DNS_NAME}" --api-key "$KEY"

cat <<EOF

=== Live ===

Send these to the platform team over a secret channel -- not email or chat
history, and never in the integration guide:

  AI_SERVICE_BASE_URL=https://${DNS_NAME}
  AI_SERVICE_API_KEY=${KEY}

Rotate the key with:
  NEW=\$(openssl rand -hex 32)
  sed -i "s|^AI_SERVICE_API_KEY=.*|AI_SERVICE_API_KEY=\$NEW|" ${ENV_FILE}
  systemctl restart coherex-minutes-api
EOF
