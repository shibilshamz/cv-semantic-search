# Deploying to the VPS

Written for the specific Ubuntu 24.04 host that runs n8n, the trading dashboard
and the job-hunter Flask runner. Adjust ports and paths for anywhere else.

## Before you install anything

This box has 3.8 GB of RAM and three other things already living on it. Check,
don't assume:

```bash
ss -tulpn | grep -E ':(8001|5680)\b'    # must be empty
free -m                                 # note "available" -- this is the baseline
df -h /
ip -4 addr show docker0                 # 172.17.0.1 must be up
docker ps                               # n8n must be on the default bridge
```

**Stop here if** either port is taken or free memory leaves under ~800 MB of
headroom. Port 8000 is `trading-agent` and port 5679 is `job-runner`; 8001 and
5680 were chosen to avoid both.

## Install

```bash
git clone https://github.com/shibilshamz/cv-semantic-search.git /root/cv-semantic-search
cd /root/cv-semantic-search

# Ubuntu 24's system Python is externally managed. A venv, not
# --break-system-packages: this box runs a paying client's pipeline.
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

cp .env.example .env
chmod 600 .env
vi .env                 # ANTHROPIC_API_KEY, and confirm the ports

# On this VPS, use the script: Ubuntu's docker.io ships no Compose v2 plugin,
# and the installed v1 fails against this Docker version with
# "Not supported URL scheme http+docker". The script runs the same container
# with the same flags, and waits for the heartbeat.
bash deploy/chroma-run.sh

# Anywhere with Compose v2, this is equivalent:
#   docker compose up -d
```

Then warm the model cache once, before Supervisor has any reason to care how
long the first request takes:

```bash
.venv/bin/python embedder.py     # downloads ~80 MB, prints its self-test
```

## Index and verify

```bash
.venv/bin/python indexer.py --corpus corpus/synthetic
.venv/bin/python eval/run_eval.py
```

Expect recall@5 of 1.00 and the negative control passing. If the eval fails here
but passed locally, the difference is the environment, not the code -- check that
`CHROMA_PATH` is unset in `.env` so it talks to the container.

## Run it under Supervisor

```bash
cp deploy/cv-search.conf /etc/supervisor/conf.d/
supervisorctl reread && supervisorctl update      # BOTH; reread alone does nothing
supervisorctl status cv-search
curl -s http://172.17.0.1:5680/health
```

## Let n8n reach it

The API listens on the bridge, but ufw still drops container-to-host traffic:
it arrives through the host `INPUT` chain, whose default policy is `DROP`. Until
this rule exists, n8n cannot call `/index` — and neither can it reach any other
host service, which is worth knowing if you have ever assumed otherwise about
`job-runner` on 5679.

```bash
ufw allow in on docker0 from 172.17.0.0/16 to 172.17.0.1 port 5680 proto tcp   comment "n8n container -> cv-search API"

docker exec n8n node -e 'fetch("http://172.17.0.1:5680/health").then(r=>r.text()).then(console.log)'
```

Scoped to the interface, the bridge subnet and one port. Do **not** substitute
`ufw allow 5680` — that opens it to the internet.

## Confirm you did not break anything else

The box has three separate process managers and none of them knows about the
others. Check each:

```bash
systemctl is-active caddy trading-agent ngrok-tunnel supervisor
supervisorctl status
docker ps
curl -I https://72.61.233.142.sslip.io && curl -I http://72.61.233.142
free -m                                  # compare against the baseline above
```

And confirm the API is **not** reachable from outside. Run this from your own
machine, not from the box -- `ufw status` is not evidence:

```powershell
Test-NetConnection 72.61.233.142 -Port 5680    # must fail
Test-NetConnection 72.61.233.142 -Port 8001    # must fail
```

## Updating

`/root/cv-semantic-search` is a clone, so:

```bash
cd /root/cv-semantic-search && git pull && supervisorctl restart cv-search
tail -f /var/log/cv-search.err.log
```

Check `git status` first. Editing files in place works and then silently drifts
from the repo, which is what happened to `/root/job-hunter`.

## Backups

The index lives in the `/root/chroma-data` bind mount. It is rebuildable from
the source CVs, so it is not critical -- but it is also on the same disk as
everything else, which means it is not a backup of anything:

```bash
tar czf /root/chroma-$(date +%Y%m%d).tar.gz /root/chroma-data
# then copy it OFF the box, or it is not a backup
```
