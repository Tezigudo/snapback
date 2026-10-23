# Boot-resume Phase B: moving from observe-only to armed

Status: **BUILT LOCALLY, NOT ARMED, NOT DEPLOYED** (2026-09-29). Branch
`feat/boot-resume-phase-b` off `droplet` @ `704df89`, in a local
worktree. Nothing was pushed, deployed or restarted, and nothing talked
to Binance. Written 2026-09-29 as a plan; read `BOOT_RESUME_PLAN.md` (Phase A) first. This document
supersedes that plan's §5 gate wording.

**What was built** (one commit each, on top of a cherry-pick of Phase A `f83b8a3`, which applied cleanly):

| Item | Built | Where |
|---|---|---|
| Step 0: Phase A onto `droplet` | Yes. The cherry-pick was clean; `qty` survives breakeven's read-modify-write of `active_bracket`. | bot.py |
| C1: pure gate | Yes. `bot_internals.adopt_precheck` + `can_adopt`; `Bot._can_adopt` is now the I/O wrapper. Reason strings are unchanged. | bot_internals.py, bot.py |
| C2: probe | Yes. `tools/boot_resume_probe.py <v1\|donchian\|sol_supertrend>`. It is read-only by construction (whitelist proxies around ccxt *and* the client, `mode=ro` + `query_only` DB). It prints `GATE VERDICT (if armed)` ADOPT/REFUSE/FLATTEN/FLAT (the hypothetical armed-gate result, which is what gets counted; NOT what a restart does), directly followed by `a restart right now would: ...` (what the current config actually does), plus a `gate:` line that is byte-comparable with the OBSERVE reason. | tools/ |
| C3 / B1: tracking seed | Yes. `_seed_adopted_tracking` + the C12 line `adopt: tracking seeded ...`. | bot.py |
| C4 / B2: dedup bar | Yes. `last_entry_bar_ts` is written whenever `_last_signal_ts` advances, and read on the **adopt path only**. The fallback is `floor(entry fill, bar) − 1 bar`, deliberately. A plain floor would suppress the next bar. | bot.py, bot_internals.py |
| C7 / D2: missing qty | Yes. Refuse → flatten. | bot_internals.py |
| C8 / D5: loop guard | Yes. `boot_adopt_log`: adopts 1–3 proceed and the **4th** flattens and alerts (God's "max 3 / 30 min"; the §6 T6 text said "3rd"). | bot_internals.py, bot.py |
| Armed verdict log | Added: `boot-resume ARMED: ADOPT\|FLATTEN ... — reason`. | bot.py |
| Fix pass after the fact-check | Added a **hard arming interlock**: `ARMING_PREREQS_BUILT = False` in bot.py. **Observe mode makes zero exchange calls**: it runs only the offline precheck, and the log says the book check was skipped. The per-bar write now has a 0.5 s busy cap. A malformed `boot_resume` counts as disabled, and flags are strict booleans. Probe guards added: state-DB access is forbidden and client logs carry exception types only. The probe now prints probe time and position age, a loud C5 warning, and the interlock status. | bot.py, state.py, tools/ |
| T9 HALT-after-adopt, T10 kill-after-adopt, T7 retry | Written on `feat/boot-resume-armed-path` (see below). | tests/test_boot_resume_armed_path.py |

### Blockers before arming (all must be done before `ARMING_PREREQS_BUILT` flips)

> **Update 2026-10-13: all 8 items below are BUILT and tested, NOT ARMED.** Branch
> `feat/boot-resume-armed-path` off `droplet` @ `d380405` (local
> worktree). Not pushed or deployed. `ARMING_PREREQS_BUILT` is still
> `False` and v1 still ships `observe_only: true`. The commits are:
> C5 `5e40c2f`, C9 `9754448`, C6 `571035f`, C10+C11 `287c010`, blockers 3+4 `761548e`,
> blocker 5 `a68cb94`, blocker 8 (T9/T10) `7ce6024`, probe view `455dd2f`, exact-root + closing-side
> leg match `9fa95f2`, already-flat flatten `c94238f`, HALT checked before adopting (see the
> HALT-at-boot section below). See "Armed-path build: decisions for God" below.

Arming now needs **two** changes: `observe_only: false` in config, **and** a reviewed code change
that sets `bot.ARMING_PREREQS_BUILT = True`. With only the config line changed, boot logs an ERROR,
alerts, and flattens as before (tested). The constant may flip only once every item below is built
and tested:

1. **C5: identity.** Require `active_bracket.signal_id == latest entry coid root`. Without it, a
   stale `active_bracket` or a manual position that lands inside the ±2% / qty tolerance can be adopted.
2. **C6: re-read before commit.** Re-fetch the position as the last step, and refuse unless side,
   qty and entry are unchanged. `_can_adopt` does two book round-trips after the single read.
3. **SL missing and mark already through the stop.** Adopting "bracket missing but reprotect can
   restore it" when the mark is already past the stop price means reprotect's placement is rejected
   (it would trigger immediately) and it keeps retrying while the position is **unprotected**.
   Whether failed placements count toward `max_replaces_per_position` was not verified here. This
   must flatten instead.
4. **Plain-book SL legs are not scoped to our COID prefix.** `bracket_state` classifies plain
   orders with `reduce_only_bracket_leg`, which does not check the `snap-...` prefix, so a foreign
   or manual reduce-only stop reads as "our SL present". Scope it to the leg's prefix, as algo rows are.
5. **D5 window anchored on the first adopt.** A *slow* crash loop, one crash every 11+ minutes,
   re-anchors every 30 min and adopts forever. Use a sliding window, or a count that does not reset
   on time until the position (signal_id) changes.
6. **C11: alert on every adopt** ("Position RESUMED after restart"). §8 step 9 expects this email.
7. **C9: bounded book retry at boot.** This is the reboot network flap. Without it, the most likely
   reboot outcome is a refusal, then a flatten (safe, but it defeats the feature).
8. T9/T10 (HALT / kill switch after adopt) written and passing.

### Armed-path build: decisions for God (2026-10-13)

1. **Adopt cap never resets on time (`a68cb94`). This overrides D5's literal "3 / 30 min".** The cap is
   at most 3 adopts per position (`signal_id`), and the count resets only when the position changes.
   Why: a window anchored on the first adopt, and a sliding window too, keeps adopting forever if the
   process crashes about every 11 minutes. Cost: a 4th unplanned restart within one position flattens it,
   which is today's behaviour anyway. Planned restarts cannot hit this, because `leg_safe_restart.py`
   requires the leg to be flat. To keep the literal D5 rule, revert `a68cb94`.
2. **Any missing stop now flattens (`761548e`). This is broader than blocker 3's "mark through stop".**
   Blocker 3 asked for a flatten only when the mark is already through the stop; the code refuses to
   adopt whenever the stop is missing. That removes the "bracket missing but reprotect has budget"
   adopt path from §4/§5. A missing TP with the stop present still adopts. The commit is bundled with
   blocker 4.
3. **Unreadable books after the C9 retries, then flatten (red-team #2): undecided.** Today, if the books
   are still unreadable after the retries, the gate refuses and the bot flattens. `close_position` then
   **cancels the stops first** and sends a market close. If that close also fails, which is likely in
   the same outage, the position is left with **no stop** until a later boot. Options:
   - **(a) Status quo: flatten.** Simple, and it is today's behaviour. The risk is that the stop is
     cancelled and the close then fails during the same outage.
   - **(b) Exit without touching the exchange.** In this one case (books unreadable, position open),
     the process exits non-zero and changes nothing, so the resting exchange orders keep protecting
     the position. systemd `Restart=on-failure` retries in 10 s, and the adopt-loop cap still applies.
     Cost: a long outage becomes a restart loop, and it needs its own counter so it cannot loop
     forever.
   - **(c)** Flatten using `sweep_first=False` (close first, sweep after), the way the breakeven close
     does it. This changes the flatten block.

   No code has been written for this. God decides.
4. **HALT present at boot.** Previously the armed boot adopted, which spent an adopt slot and sent a
   RESUMED email, and then the first tick flattened. Now the armed verdict checks `halt_source()`
   first and refuses. The kill switch still applies on the first tick (T10). Checking it before
   adopting would mean passing the boot's equity read into the verdict, and that is not done.
5. **The flatten block changed in one case only (`c94238f`).** That case: `close_position` returns
   None because the exchange is already flat. The block no longer records a $0 `boot_flatten`.
   Instead it sweeps the leg's own orders and sends an alert saying "exit not recorded, reconcile
   via income". The block's sha moved from `c95930f5…` to `a6c5f676…`. Removing the 3 inserted lines
   gives back the droplet sha (tested). Droplet already has this race, and it applies to all legs.
   To keep the flatten block untouched, revert `c94238f`.

**Still NOT ARMED.** It is locked **twice**:

- **Config.** `boot_resume.enabled` defaults to False and `observe_only` defaults to True. Both are
  read as strict booleans. v1's `params.yaml` ships `observe_only: true`, and donchian and sol have
  no block.
- **Code.** `ARMING_PREREQS_BUILT = False`.

Evidence and footprint:

- The flatten block is byte-identical to `704df89` (sha256 `c95930f5…`, pinned in
  `tests/test_boot_resume_flag_off.py`).
- `git diff 704df89 -- bot.py` is 295 insertions / 2 deletions (the 2 are a stale `_open_entry_fill` docstring, outside the flatten); the flatten block is byte-identical (sha test).
- Live side effects with the shipped config, all local, none change an exchange call or trading decision: (1) one `last_entry_bar_ts` meta write per evaluated signal bar on every leg, after evaluation and before sizing/order, sqlite busy-wait 0.5 s per blocked step (bounded, not zero; disk stalls not covered; cannot suppress an entry); (2) `active_bracket` JSON gains a `qty` field (readers: reprotect, breakeven, both read-modify-write); (3) a v1 boot holding a position does one extra local DB read + logs one WARNING/observe line.
  block.
- Observe mode's client call sequence is **identical** to a leg with no `boot_resume` block (tested).
- The only new flag-off side effect is one `meta` upsert (`last_entry_bar_ts`) per evaluated bar on
  every leg. It is best-effort, with a 0.5 s busy cap, and never blocks or delays evaluation beyond that.
- Observe-mode log lines changed. Phase B observe prints
  `boot-resume OBSERVE: PRECHECK OK|WOULD FLATTEN ...`, not the Phase A `WOULD ADOPT` line, because
  it no longer reads the books. The full verdict comes from the probe, which is the counted evidence
  anyway.

**Tests (local `.venv`):**

| Tree | Passed | Skipped | Failed |
|---|---|---|---|
| `704df89` baseline | 438 | 5 | — |
| After Phase A | 463 | 5 | — |
| Phase B, first cut | 548 | 5 | — |
| Phase B, after the fix pass | **570** | 5 | **0** |

**Next:** review. After that, deploy is possible, safe to run observe-only and interlocked (§8 step 3). The blockers above are a separate build.

Line numbers refer to **`droplet` @ `704df89`** unless marked `f83b8a3` (the Phase A branch).
On `main`, the same symbols sit roughly 250 lines lower, and some of them do not exist there at all.

---

## 0. TL;DR

1. **The Phase A arming gate cannot produce evidence as written.** It must be replaced, not just
   run. See §2.
2. **Phase A is not on the droplet**, and its base (`1577c5d`) is now **6 commits** behind `droplet`
   (`66574d7`..`704df89`). 4 of them touch `bot.py`, for 276 insertions. Step 0 is to rebase and deploy it, still observe-only.
3. **Phase A has two real bugs that only show up once you arm it.** They must be fixed before
   `observe_only: false`. Details in §3.
   - B1: an adopted position's exit can be silently dropped.
   - B2: after an adopted position closes, the bot can re-enter on the same signal bar.
4. **Recommended evidence gate:** at least 5 agreeing read-only probe verdicts, taken across at
   least 3 distinct live v1 positions, with no restart and no close. Unit tests and any organic
   observe lines count as supporting evidence. The previous "≥5 deliberate restarts" approach
   would close 5 real trades to prove it.
5. **Scope:** v1 only, on the `droplet` line only. donchian and sol stay fail-closed.

---

## 1. Constraints (these are the design)

- **Flatten stays the default.** Any unknown, exception or mismatch must end in today's flatten.
  Resuming requires affirmative proof. Flattening requires nothing.
- **Only v1 can adopt.** `_can_adopt` requires reprotect to be armed, and only
  `config/params.yaml` (v1) has an armed `reprotect:` block. donchian also fails the
  `place_tp: false` refusal. This stays true in Phase B.
- **The live truth is the `droplet` branch.** The daily-loss breaker, per-leg HALT wiring, SOL
  breakeven and reprotect-armed config all exist there. Anything developed on `main` and
  cherry-picked across is a merge hazard.
- **Instance names vs unit names.** `tools/leg_safe_restart.py` and the probe take the *instance*
  (`v1`, `donchian`, `sol_supertrend`; lsr.py:57-60, bot.py `INSTANCE_PROFILES`). systemd and
  journalctl take the *unit* (`snapback-btc` for v1). The per-leg HALT file is `data/HALT_v1`.
  Passing `snapback-btc` to `leg_safe_restart.py` is a usage error (exit 2).
- **Deliberate restarts cost money.** In observe mode, every restart of a leg that holds a
  position ends in a market close. `tools/leg_safe_restart.py` refuses to restart unless the leg
  is flat (line 122). The sanctioned restart path therefore can never exercise the gate at all.

---

## 2. Why the Phase A gate is unrunnable, and what replaces it

The Phase A gate was: *"≥5 deliberate restarts while holding a position… drive it with deliberate
test restarts on a dry-run leg."* Here is why each route fails, as the code stands on `f83b8a3`:

| Route | Why it produces no usable evidence |
|---|---|
| Dry-run leg | `boot()` checks `if self.dry_run:` **before** `_boot_resume_verdict` (`f83b8a3` bot.py:487-492). A dry-run leg never logs `boot-resume OBSERVE`. It also places no orders, so it has no exchange position, no bracket and no `active_bracket` to evaluate. |
| Deliberate restart, live v1, via `leg_safe_restart.py` | The script refuses because the leg is not flat (exit 3). Nothing is observed. |
| Deliberate raw `systemctl restart`, live v1 | The verdict is logged and then the position is **flattened**. Five restarts means five live v1 trades closed at market: the exact harm this feature exists to prevent, inflicted on purpose. It also bypasses the one sanctioned restart path. |
| Organic restarts (crash, OOM, reboot) | They cost nothing extra, because today's flatten happens anyway. But they are rare: needrestart removed the common trigger on 2026-09-16. They cannot be the gate on their own. |

### Replacement options

| Option | Cost | Benefit | Verdict |
|---|---|---|---|
| **A. Read-only probe.** `tools/boot_resume_probe.py <instance>` computes the adopt verdict against **live** exchange state and a **read-only** state DB, then prints the verdict and every input. No `Bot.boot()`, no restart, no order. | ~1 small refactor (§4 C1) plus one tool. | Exercises the real gate against the real exchange shapes (algo book, COIDs, qty rounding). It can be run on every v1 position, and costs zero trades. | **Recommended. This is the counted evidence.** |
| B. Testnet v1 clone (`BINANCE_ENV=testnet`, `dry_run: false`, reprotect and boot_resume armed) | Needs a unit, keys and a way to force entries. `exchange/binance_client.py:100` calls `ex.set_sandbox_mode(True)`, but it is **unverified** whether current ccxt/Binance still serve USDM futures and `/fapi/v1/openAlgoOrders` on the testnet. Binance has been moving to "demo trading". | The only place the **armed** path, `kill -9` and "bracket fills while down" can be tested end-to-end. | Optional. Do it only if God wants armed-path proof beyond unit tests. Verify testnet algo support first. |
| C. Organic observe lines | Free | These are real boots under real conditions. | Bonus evidence. Each one is read, but none is required. |

What the probe **does not** prove is the boot plumbing itself: that the adopt branch hands the
position to the loop correctly. That part is covered by unit tests (§6) and optionally by the
testnet run (B). The probe proves the **decision**. The unit tests prove the **wiring**.

### The evidence gate (replaces Phase A §5)

Arm only when **all** of the following hold:

1. **≥5 probe verdicts of `ADOPT`, spanning ≥3 distinct v1 positions** (distinct `signal_id`).
   God checks each one independently against the exchange (checklist §8). Any disagreement resets
   the count to zero.
2. **At least one probe per position runs after that position's first 15m bar close.** This
   shows the verdict is stable once the position has aged, not only at entry.
3. **Zero `boot-resume: gate raised` lines** in any log since Phase A deployed.
4. **Every organic `boot-resume OBSERVE` line agrees with God's reading** (bonus evidence, not
   required).
Expected calendar time: v1 makes about 5 entries a month, so 3 distinct positions take roughly
3–5 weeks. A gate open longer than about 8 weeks has stalled rather than being slow.

5. The Phase B code (§4) is **deployed and has run for at least one full v1 position still in
   observe mode**. The flip to armed is then config-only.

### Log lines that constitute evidence

Phase A (observe) boot, same boot, in this order. They appear in the leg's `bot.jsonl` (the
`LOG_FILE` its instance profile sets in `main()`) or in `journalctl -u snapback-btc`:

```
boot-resume OBSERVE: WOULD ADOPT|FLATTEN <side> <qty> @ <entry> — <reason>
Boot found open position <side> qty=... @ .... Flattening (root=<signal_id>).
```

The pairing matters. An `OBSERVE` line with **no** following `Flattening` line means observe
mode changed behaviour, and that is a stop-ship. Failure lines to grep for (each must be zero, or
explained):

```
boot-resume: gate raised — flattening
boot-resume: fetch_open_orders failed
```

Probe output (Option A) must print the same `reason` string that `_can_adopt` returns. This
makes a probe verdict and a boot verdict byte-comparable.

Phase B (armed) adds these lines, which are checked after the first real adopt:

```
Boot found open position ... RESUMING it (root=...)        # log
events: type=boot_adopt                                      # leg state DB
bot_event boot_adopt                                         # consolidate / dashboard
first tick: _detect_bracket_exit snapshot == adopted position (new log line, §4 B1)
```

---

## 3. Bugs in Phase A that bite only once armed

These are harmless in observe mode because the position is flattened anyway. Both become live the
moment `observe_only: false`.

**B1: dropped exit after adoption.** `_detect_bracket_exit` (bot.py:1245) computes
`had_open = prev_side not in ("flat", "unknown")`. `__init__` sets `_last_position_side =
"unknown"` (line ~330), and the Phase A adopt branch does not seed it. Suppose the bracket
fills between `boot()` reading the position and the first tick's snapshot. That window includes
the whole of `boot()` after the read: consolidate pushes, principal logging, and anything that
raises and retries. The first tick then sees `unknown → flat`, which is **not an exit**, and no
`close` row, `exit` event or alert is written. This is the donchian 2026-09-04 dropped-exit shape,
reintroduced through a new door.
**Fix:** on adopt, seed `_last_position_side/entry/qty` from `pos` and `_last_entry_root` from
`root` before returning from `boot()`.

**B2: same-bar re-entry after an adopted position closes.** `_last_signal_ts` is in-memory only
(line 315, checked at 878). After a restart it is `None`, so once the adopted position goes flat,
`_maybe_enter` re-evaluates the **current** last-closed bar. If a v1 position is adopted and
stopped out within the same 15m bar it was entered on, the same signal re-fires and the bot
re-enters.
**Fix:** use the existing persistence slot rather than inventing one. `exchange/state.py:7`
already documents a `last_entry_bar_ts` meta key ("ISO ts of bar bot last considered for entry"),
which is currently unused.
- In `_maybe_enter`, write `last_entry_bar_ts` whenever `_last_signal_ts` advances.
- In `boot()`, initialise `_last_signal_ts` from it on adopt.
- If the key is absent, for positions opened before this ships, fall back to the `fills` row with
  `reason='entry'`, with its `ts` floored to `entry_tf`.
*Note:* the flatten path has the same pre-existing exposure, where a restart within the entry bar
flattens and can then immediately re-enter. That fix is **out of scope** here because it changes
today's default behaviour. It is listed in §9 for a separate decision.

**B3 (identity weaker than the breakeven code's).** `_can_adopt` checks side, a ±2% entry
tolerance and qty, but never that `active_bracket.signal_id == state.latest_entry_coid_root()`.
`_breakeven_step` (line ~1814) already requires that match. The plan's own "stale
`active_bracket` from a previous position" hazard is exactly the case this closes.
**Fix:** add the check, and refuse on mismatch.

**B4: no re-read before commit.** This is H3 from Phase A, still open. `boot()` reads `pos` once,
and `_can_adopt` then does two order-book round-trips.
**Fix:** re-fetch the position as the last step and refuse unless side, qty and entry are
unchanged.

---

## 4. Code changes, per branch

### Branch strategy: decided

- **Build on the droplet line only.** Create `feat/boot-resume-phase-b` from **current
  `droplet` (`704df89`)**. Cherry-pick `f83b8a3` onto it, then add the Phase B commits.
- **Expect a conflict at the `active_bracket` stash write** (droplet bot.py:979, where Phase A
  adds `"qty"`). The SOL breakeven work (`e647707`) now also writes `active_bracket`, at lines
  1873/1893/1908/1925 (`be_moved`, `be_price`, `be_ext`). Confirm two things:
  - Phase A's `qty` survives breakeven's read-modify-write, since it round-trips the whole dict.
  - Nothing in `_can_adopt` is confused by the extra keys.
- **`main`: do nothing.** `main` lacks the daily-loss breaker and the breakeven code, and sits
  about 250 lines offset. Porting boot-resume there before the main/droplet reconciliation just
  creates a third divergent copy. Record that decision in the PR.
- **Verify presence by grep, never with `git merge-base --is-ancestor`.** On the droplet run
  `grep -n "_boot_resume_verdict\|boot_adopt\|_seed_adopted_tracking" bot.py`.

### Phase B code changes (all on the new branch, all fail-closed)

| # | Change | Why |
|---|---|---|
| C1 | Extract the decision into a **pure function** `can_adopt(params, pos, ab_raw, latest_root, open_orders, algo_rows, algo_ok, qty_step, coid_prefix) -> (bool, str)` in `bot_internals.py`. `Bot._can_adopt` becomes the thin wrapper that does the fetches. | The probe tool and the boot must run **the same code**. A probe that re-implements the gate proves nothing. |
| C2 | `tools/boot_resume_probe.py <instance>`: takes an instance (`v1`), resolves its params file and state DB through the same `INSTANCE_PROFILES` mapping `main()` uses (without calling `main()`), builds a `BinanceClient`, and does read-only fetches (position, plain orders, algo orders). It opens the leg's state DB with `sqlite3.connect("file:...?mode=ro", uri=True)` and calls C1. It prints the verdict, reason, and every input (redacting keys). It **never** constructs `Bot` and **never** writes. It must **not** call `main()` or `_setup_logging()`: it writes to stdout only. Otherwise its lines land in the leg's live `bot.jsonl` and get counted as organic observe lines. | This is the counted evidence (§2). |
| C3 | B1 fix: seed the exit-tracking snapshot on adopt. | Otherwise an exit can be dropped. |
| C4 | B2 fix: persist `_last_signal_ts` in the existing (unused) `last_entry_bar_ts` meta key, and seed from it on adopt (fallback: entry fill's bar). | Otherwise the bot can re-enter on the same bar. |
| C5 | B3: require `active_bracket.signal_id == latest_entry_coid_root()`. | Identity parity with breakeven. |
| C6 | B4: re-read the position immediately before committing to adopt. | Closes the adopt/close race. |
| C7 | **Refuse when the stashed `qty` is absent.** Phase A *skips* the qty check for records written before Phase A. | Every position opened before Phase A deploys lacks `qty`. Fail-closed costs, at most, the first position after deploy. **God decision D2.** |
| C8 | **Adopt-loop guard.** Persist `boot_adopt_log = {signal_id, count, first_ts}` in `state.meta`. If the same `signal_id` has been adopted ≥3 times within 30 min, flatten instead. | Without this, a bug that crashes the tick loop *because of* the adopted position becomes a crash→adopt→crash loop, with `Restart=on-failure` / `RestartSec=10`. Reprotect, the time stop and the trend exit would then never run. Today's flatten breaks that loop, and this keeps a breaker for it. |
| C9 | **Bounded retry of unreadable books at boot.** Make up to 3 attempts, 5 s apart, before treating an unreadable plain or algo book as a refusal. | On a **reboot** the network can flap for a few seconds after `network-online.target`. Without retry, the most likely reboot failure is a transient algo read, which turns straight into a market close. After the retries, the result is still flatten. |
| C10 | Seed `_reprotect_capped_alerted` from `reprotect_count >= cap` (H2). | Cosmetic: prevents a duplicate alert. |
| C11 | `send_alert("Position RESUMED after restart", ...)` on adopt. | An unchosen restart while in a position should reach God's inbox, not only the log. |
| C12 | Log `adopt: tracking seeded side=.. entry=.. root=.. last_signal_ts=..` | This is the Phase B evidence line. |

The flatten block must remain **byte-identical**. Enforce it by diffing that block between
`droplet` and the feature branch in review.

**Config (the arming flip itself)** goes in `config/params.yaml` only:
`boot_resume.observe_only: false`. Deploy it as a separate, config-only commit after the gate
passes. The only way it takes effect is a restart, and that goes through `leg_safe_restart.py`
while flat.

---

## 5. Failure modes

| Scenario | What happens under Phase B | Status |
|---|---|---|
| **Exchange open, DB has no matching entry** (a manual position, or a dropped exit then a re-entry) | Empty `active_bracket`, or a `signal_id` ≠ latest root → refuse → flatten. | Covered (Phase A + C5). |
| **Exchange qty ≠ stash** (partial manual close in the app) | qty guard → flatten. | Covered, **if** C7 is taken. Without C7, pre-Phase-A positions skip the check. |
| **Stale `active_bracket` inside the ±2% band** | C5 (root match) plus the qty check. | Covered by C5. |
| **Exchange flat, DB thinks open** | Existing flat path: orphan sweep, no adopt. On the first flat tick, `_maybe_reprotect` clears `active_bracket`. | Unchanged. |
| **Bracket fills while the box is down** | Boot sees flat and sweeps orphans. **The exit is never recorded** (no fill row, no event). | **Pre-existing gap, not introduced here.** Income reconciliation is the backstop. |
| **Bracket fills between the boot read and the first tick** | Without C3: dropped exit. With C3: the normal flat-edge exit is recorded. | Fixed by C3. |
| **Orphan / duplicate reduce-only legs** (a partial reprotect cancel before a crash) | `bracket_state` returns booleans, not counts, so duplicates read as "intact" and the bot adopts. Duplicates are reduce-only and COID-tagged. The flat-edge sweep and `close_position`'s COID cancel both clear them later. | **Accepted.** A duplicate SL is over-protection, not under-protection. Optional hardening: count legs and refuse if more than one SL. **God decision D4.** |
| **SL manually moved in the Binance app** (the COID changes) | `bracket_state` sees our SL missing. Adopt then relies on reprotect, which places **our** SL beside God's. | Pre-existing reprotect behaviour. Documented; not changed here. |
| **Restart mid-bar** | Time stop: age comes from the `fills` table, so it survives. Trend exit (v1 adverse EMA200): recomputed from fetched OHLCV each tick, so it is stateless and survives. Daily anchor: in `state.meta`, survives. Signal dedup: **in-memory**, so B2 applies; fixed by C4. Breakeven: `be_moved` in `active_bracket` (SOL only, and SOL cannot adopt). | Covered after C4. |
| **Reboot** | systemd waits for `network-online.target`. If `fetch_position` raises in `boot()`, the process exits non-zero, systemd retries in 10 s, and no action is taken. A transient order-book failure is handled by C9. | Covered by C9. |
| **HALT file present at boot** | HALT is checked in `loop()` (line 2011), **not** in `boot()`. Order: boot adopts, then the first tick sees HALT → `close_position(close_leg="h")` → exit 0. The end result is still flat. | Acceptable. Test it (§6 T9). Do **not** move the HALT check into `boot()` in this change. |
| **Kill switch breached at boot** | Same pattern: the first tick's `_check_kill_switch` closes the position. | Acceptable. Test it (T10). |
| **Daily-loss breaker latched at boot** | It blocks entries only, and the latch is persisted in `daily_loss_breaker_date`. Adopting is not an entry. | Unaffected. |
| **Crash loop caused by the adopted position** | C8 caps it at 3 adopts per `signal_id` within 30 min, then flattens. | Covered by C8. |
| **Gate raises** | Returns `False`, then flatten (Phase A wrapper). | Covered. |

---

## 6. Test plan (local only)

Run with `.venv/bin/python -m pytest` (the repo venv). **Never `uv run`,
never on the droplet.** Run the full suite on the merged tree (droplet + A + B), not on the feature
commits alone. Record before and after counts. The Phase A message says 392 → 417; the older plan
text says 415. Re-baseline on `704df89` first.

| # | Test | Asserts |
|---|---|---|
| T1 | Adopt, then a bracket fill on the very first tick | An `exit` event and a `close` fill are written (B1 regression). Also run it with C3 reverted and confirm the test fails, which proves the test has teeth. |
| T2 | Adopt, then flat within the entry bar | `_maybe_enter` does **not** re-evaluate that bar (B2). |
| T3 | `active_bracket.signal_id` ≠ latest root | Refuse → flatten. |
| T4 | The position changes between the first and second read | Refuse → flatten. |
| T5 | Stash has no `qty` | Refuse (if D2 = refuse). |
| T6 | Same `signal_id` adopted 3× within 30 min | The 3rd boot flattens. At 31 min, the counter resets. |
| T7 | Algo book unreadable ×2 then readable | Adopt. Unreadable ×3 → flatten. |
| T8 | The pure `can_adopt` and `Bot._can_adopt` give identical `(verdict, reason)` over the whole Phase A fixture set | Probe/boot parity (C1). |
| T9 | HALT present plus an adoptable position | Boot adopts, first loop tick closes with `close_leg="h"`, exit 0. |
| T10 | Kill-switch breach plus an adoptable position | First tick closes with `close_leg="k"`. |
| T11 | Dry-run leg with a position | Unchanged "leaving it alone"; the verdict is never called. |
| T12 | `place_tp: false` (donchian shape) | Refuse (existing test, keep it). |
| T13 | The probe tool against a fixture DB opened `mode=ro` | Any write attempt raises; the probe never imports or constructs `Bot`. |
| T14 | Breakeven-mutated `active_bracket` (`be_moved`, `be_price`) | `can_adopt` still parses it and `qty` is preserved. |
| T15 | Flatten-block byte-identity | Hash of the flatten block's source equals the droplet baseline. |

Reuse the fixture traps recorded in the Phase A commit message:
- A fake algo row without top-level `reduceOnly` is classified as no leg.
- `patch.multiple` only returns mocks for `DEFAULT` kwargs.

---

## 7. Rollback

- **Before the flip (code deployed, still observe-only):** nothing to roll back behaviourally. For
  a code revert, run `git merge --ff-only` to the previous droplet SHA and restart via
  `leg_safe_restart.py` while flat. Never use `reset --hard`.
- **After the flip, leg flat:** set `boot_resume.observe_only: true` in `config/params.yaml` and
  run `tools/leg_safe_restart.py v1`. Takes about 1 minute.
- **After the flip, leg in a position, and adoption is misbehaving:** `leg_safe_restart.py` will
  refuse, correctly. Emergency path:
  1. Run `touch data/HALT_v1`. The next tick flattens and exits 0, and
     `Restart=on-failure` does not resurrect it.
  2. Set `observe_only: true`.
  3. `rm data/HALT_v1`.
  4. `systemctl start snapback-btc`.

  This costs one market close, which is exactly today's behaviour.
- **The kill criterion that triggers a rollback without discussion:** any adopted position whose
  exit is missing from `fills`/`events`, a second `boot_adopt` for the same `signal_id` that God
  cannot explain, or any `gate raised`.

---

## 8. Checklist for God

**Decisions first (§10).** Then:

- [ ] 1. Approve the branch strategy: droplet-line only, `main` untouched (D1).
- [ ] 2. Have `engineer` build `feat/boot-resume-phase-b` from `droplet` `704df89`: cherry-pick
      `f83b8a3` plus C1–C12. Review it with `code-reviewer`. Suite green on the merged tree
      locally (`.venv`), with counts recorded.
- [ ] 3. Deploy to the droplet **with `observe_only: true`** via the normal deploy path. Restart
      v1 only through `tools/leg_safe_restart.py v1` while flat. Verify presence by
      grep, not by `merge-base`.
- [ ] 4. Confirm on the droplet that the boot log shows no `boot-resume` line (the leg was flat),
      and that `grep -c "boot-resume: gate raised"` on the leg log is `0`.
- [ ] 5. **Each time v1 opens a position**, at least once after the first 15m bar closes, run on
      the droplet
      `./.venv/bin/python tools/boot_resume_probe.py v1`. Then **independently**
      check:
      - (a) the Binance app: side, size, entry
      - (b) the Binance app: open orders include a stop **and** a TP for v1
      - (c) the probe's printed `active_bracket.signal_id` equals the latest `entry` fill's COID
        root

      Write down "agree/disagree" next to the probe's `reason`.
- [ ] 6. Tally: ≥5 `ADOPT` verdicts, ≥3 distinct `signal_id`s, 0 disagreements, 0 `gate raised`.
      Any disagreement → stop and send the probe output to `debug-detective`. The count resets.
- [ ] 7. Read every organic `boot-resume OBSERVE` line that appeared, if any. Each must be followed
      by a `Flattening` line in the same boot.
- [ ] 8. Flip `boot_resume.observe_only: false` in `config/params.yaml` (config-only commit).
      Restart via `leg_safe_restart.py` **while flat**.
- [ ] 9. After the first real `boot_adopt`, confirm:
      - the `adopt: tracking seeded` line
      - the alert email
      - the position's eventual exit appearing in `fills` and `events` with the right PnL
      - no duplicate entry on the same bar
- [ ] 10. Keep the rollback commands (§7) somewhere you can reach from your phone.

**Do not:** add `reprotect:` or `boot_resume:` to the donchian or sol configs; run pytest on the
droplet; restart v1 with raw `systemctl restart`; use `git merge-base --is-ancestor` to verify.

---

## 9. What this plan explicitly does NOT do

- It does not extend adoption to donchian (half-bracket untested) or sol (no reprotect, and its
  breakeven + reprotect interaction is unstudied: reprotect would re-place the **original**
  stop, not the `-sb` breakeven stop).
- It does not fix the pre-existing **flatten-path** same-bar re-entry (B2 on the flatten side).
  That needs a separate decision.
- It does not record exits that happen while the process is down. Income reconciliation remains
  the backstop.
- It does not move the HALT check into `boot()`.
- It does not reconcile `main` with `droplet`.
- It corrects two stale references in `BOOT_RESUME_PLAN.md`, noted here only (that file is not
  edited):
  - The implementation commit is **`f83b8a3`**, not `3243f97`.
  - The suite count is **417**, not 415.

---

## 10. Decisions God needs to make

- **D1: branch.** Build on the droplet line only and leave `main` alone. *Recommended: yes.*
- **D2: missing `qty`.** Refuse to adopt positions stashed before Phase A (fail-closed), or skip
  the check. *Recommended: refuse.*
- **D3: evidence route.** Use the read-only probe as the counted gate (recommended), or
  additionally stand up a testnet clone for armed-path proof. Testnet needs its algo-order support
  verified first.
- **D4: duplicate legs.** Accept a duplicate SL as over-protection (recommended), or refuse to
  adopt when more than one SL leg rests.
- **D5: adopt-loop guard thresholds.** 3 adopts / 30 min (recommended), or tighter.


## Decisions — God, 2026-09-29 ("4 defaults")

- D1 build Phase B on the `droplet` line only — **YES**
- D2 refuse to adopt positions opened before Phase A (no stored qty) — **YES, flatten as today**
- D3 read-only probe only, no testnet run — **PROBE ONLY**
- D4 accept a possible duplicate SL on an adopted position — **YES**
- D5 crash-loop guard: max 3 adopts / 30 min, then stop adopting — **YES**
- E1 do NOT wire MAX_CONSECUTIVE_LOSSES as a trading breaker — **AGREED**
- E2 re-comment the constant (and the dead `consecutive_losses` meta key) as UNENFORCED — **YES** (via RISK_REVIEW)
- E3 alert-only streak monitor at 13 (v1) / 9 (donchian) / 7 (sol) — **YES**

Nothing is built yet; implementation needs a separate go and goes through review + tests before any droplet change.
