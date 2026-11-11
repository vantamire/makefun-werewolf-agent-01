// werewolf-agent-01: a deterministic, rule-based Werewolf player.
//
// No model calls at all: every decision is a pure function of the game's view
// (as /observe served it) and MAKEFUN_TRIAL_SEED. That makes the live pass and
// the replay identical: no wall clock, no unseeded randomness, one env request
// at a time, in order.
//
// Strategy, per role:
//   - every role keeps a suspicion score for each living seat, built from the
//     public log: votes and accusations aimed at players whose roles were later
//     revealed, and seer claims that turned out true or false. A seer result of
//     our own is proof and outranks everything.
//   - werewolf: kill the most dangerous villager (a seer claimant first, then
//     players accusing us or our partner), and always agree with the partner's
//     night vote when the partner voted first — the kill only lands when both
//     wolves name the same seat. By day, push votes onto the most-suspected
//     villager and never accuse the partner.
//   - seer: inspect the most-suspected uninspected seat; announce a found wolf
//     at once, naming the seat.
//   - doctor: protect the seer claimant when there is one, else the most active
//     villager (the likeliest wolf target), never repeating last night's seat.
//   - villager: accuse and vote the most-suspected seat.
// Statements come from short deterministic templates, picked by a seeded RNG
// over (trial seed, game, day, round), so the same trial always says the same
// thing at the same table.

import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

type Seat = number;
type Role = 'werewolf' | 'seer' | 'doctor' | 'villager';
type DecisionKind = 'speak' | 'vote' | 'kill' | 'protect' | 'inspect';

interface LogEntry {
  seq: number;
  day: number;
  kind: string;
  seat?: Seat;
  target?: Seat | null;
  text?: string | null;
  round?: number;
  died?: Seat | null;
  role?: string | null;
  tally?: { seat: Seat; votes: number }[];
  [k: string]: unknown;
}

interface Player {
  seat: Seat;
  name: string;
  alive: boolean;
  role: string | null;
}

interface View {
  day: number;
  phase: string;
  you: { seat: Seat; name: string; role: Role; alive: boolean; partner: { seat: Seat; name: string; alive: boolean } | null };
  players: Player[];
  seerResults: { night: number; seat: Seat; werewolf: boolean }[] | null;
  lastProtect: { night: number; seat: Seat | null } | null;
  wolfVotes: { night: number; seat: Seat; target: Seat | null }[] | null;
  log: LogEntry[];
  outcome: { winner: string | null; reason: string; day: number; winners: Seat[] } | null;
}

interface Pending {
  game: number;
  decision: DecisionKind;
  options?: Seat[];
  max_chars?: number;
  view: View;
}

interface Observed {
  games: number;
  finished: number;
  done: boolean;
  pending: Pending[];
}

const isRecord = (v: unknown): v is Record<string, unknown> => v !== null && typeof v === 'object' && !Array.isArray(v);

// ---------------------------------------------------------------- seeded RNG

/**
 * A deterministic PRNG stream named by `label`, drawn from sha256(trial seed,
 * label). Two runs of the same trial draw the same values in the same order,
 * and one stream never moves another, so the order decisions are answered in
 * cannot change what the agent does.
 */
export function rngFor(seed: string, label: string): () => number {
  const h = createHash('sha256').update(seed, 'utf8').update('\0').update(label, 'utf8').digest();
  let a = h.readUInt32BE(0);
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------- statement folding

/** A statement folded to one line the env's way. */
export function foldStatement(text: string): string {
  return text
    .replace(/[\t\n\v\f\r\u0085\u2028\u2029]+/g, ' ')
    .replace(/\p{Cc}/gu, '')
    .trim();
}

const cutTo = (text: string, max: number): string => {
  const folded = foldStatement(text);
  return [...folded].length > max ? [...folded].slice(0, max).join('').trim() : folded;
};

// ---------------------------------------------------------------- game analysis

interface Claim {
  claimant: Seat;
  wolf: Seat | null;
  clear: Seat | null;
}

interface Analysis {
  /** Suspicion per living seat (higher = more likely a wolf), before tie-breaks. */
  suspicion: Map<Seat, number>;
  /** Seats proven to be wolves (own seer results, or revealed roles). */
  knownWolf: Set<Seat>;
  /** Seats proven not to be wolves (own seer results, or revealed roles). */
  knownClear: Set<Seat>;
  /** Seer claims made in speeches, by claimant. */
  claims: Claim[];
  /** How many accusations each seat made against each other seat. */
  accusations: Map<Seat, Map<Seat, number>>;
  /** How many times each seat was accused. */
  accusedCount: Map<Seat, number>;
  /** How many speeches each seat made. */
  speeches: Map<Seat, number>;
  /** Votes cast, in order: {voter, target}. */
  votes: { voter: Seat; target: Seat | null }[];
  /** Votes cast so far in today's voting, per target (the bandwagon signal). */
  tallyToday: Map<Seat, number>;
  /** Name -> seat, for matching name-based accusations. */
  nameToSeat: Map<string, Seat>;
}

const VILLAGE_ROLES: ReadonlySet<string> = new Set(['seer', 'doctor', 'villager']);

/** Find every seat a piece of speech accuses of being a wolf, defends, or names in a seer claim. */
function parseSpeech(text: string, nameToSeat: Map<string, Seat>): { accused: Seat[]; defended: Seat[]; seerClaim: boolean; claimedWolf: Seat | null; claimedClear: Seat | null } {
  const lower = text.toLowerCase();
  const accused = new Set<Seat>();
  const defended = new Set<Seat>();
  const seatRef = (m: RegExpExecArray | null): Seat | null => (m === null ? null : Number(m[1]));
  const bySeat = (re: RegExp): Seat[] => {
    const out: Seat[] = [];
    let m: RegExpExecArray | null;
    const rx = new RegExp(re.source, re.flags.includes('g') ? re.flags : `${re.flags}g`);
    while ((m = rx.exec(lower)) !== null) {
      const s = Number(m[1]);
      if (Number.isInteger(s) && s >= 1 && s <= 8) out.push(s);
    }
    return out;
  };
  for (const s of bySeat(/seat\s+(\d+)\s+is\s+(?:a\s+)?werewolf/g)) accused.add(s);
  for (const s of bySeat(/seat\s+(\d+)\s+is\s+the\s+wolf/g)) accused.add(s);
  for (const s of bySeat(/vote\s+seat\s+(\d+)/g)) accused.add(s);
  for (const s of bySeat(/checked\s+seat\s+(\d+)[^.]*werewolf/g)) accused.add(s);
  for (const s of bySeat(/seat\s+(\d+)\s+is\s+not\s+(?:a\s+)?werewolf/g)) defended.add(s);
  for (const s of bySeat(/checked\s+seat\s+(\d+)[^.]*not\s+(?:a\s+)?werewolf/g)) defended.add(s);
  // Name-based accusations: "Alice is a werewolf", "vote Alice".
  for (const [name, seat] of nameToSeat) {
    const n = name.toLowerCase().replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    if (new RegExp(`\\b${n}\\b[^.]{0,40}?\\bwerewolf\\b`).test(lower)) accused.add(seat);
    if (new RegExp(`\\bvote\\b[^.]{0,20}?\\b${n}\\b`).test(lower)) accused.add(seat);
    if (new RegExp(`\\b${n}\\b[^.]{0,40}?\\bnot\\s+(?:a\\s+)?werewolf\\b`).test(lower)) defended.add(seat);
  }
  const seerClaim = /\bi'?m\s+(?:the\s+)?seer\b/.test(lower) || /\bi\s+am\s+(?:the\s+)?seer\b/.test(lower);
  const claimedWolf = seatRef(/checked\s+seat\s+(\d+)[^.]*werewolf/.exec(lower));
  const claimedClear = seatRef(/checked\s+seat\s+(\d+)[^.]*not\s+(?:a\s+)?werewolf/.exec(lower));
  return { accused: [...accused], defended: [...defended], seerClaim, claimedWolf, claimedClear };
}

/** Everything the view says about the game, reduced to the numbers the decisions need. */
export function analyze(view: View): Analysis {
  const nameToSeat = new Map<string, Seat>();
  for (const p of view.players) nameToSeat.set(p.name, p.seat);
  const knownWolf = new Set<Seat>();
  const knownClear = new Set<Seat>();
  for (const p of view.players) {
    if (p.role === 'werewolf') knownWolf.add(p.seat);
    else if (p.role !== null && VILLAGE_ROLES.has(p.role)) knownClear.add(p.seat);
  }
  for (const r of view.seerResults ?? []) {
    if (r.werewolf) knownWolf.add(r.seat);
    else knownClear.add(r.seat);
  }

  const suspicion = new Map<Seat, number>();
  for (const p of view.players) if (p.alive) suspicion.set(p.seat, 0);
  const claims: Claim[] = [];
  const accusations = new Map<Seat, Map<Seat, number>>();
  const accusedCount = new Map<Seat, number>();
  const speeches = new Map<Seat, number>();
  const votes: { voter: Seat; target: Seat | null }[] = [];
  const tallyToday = new Map<Seat, number>();

  const bump = (seat: Seat, n: number): void => {
    if (suspicion.has(seat)) suspicion.set(seat, (suspicion.get(seat) ?? 0) + n);
  };

  for (const e of view.log) {
    if (e.kind === 'speech' && typeof e.text === 'string' && typeof e.seat === 'number') {
      speeches.set(e.seat, (speeches.get(e.seat) ?? 0) + 1);
      const parsed = parseSpeech(e.text, nameToSeat);
      if (parsed.seerClaim || parsed.claimedWolf !== null || parsed.claimedClear !== null) {
        claims.push({ claimant: e.seat, wolf: parsed.claimedWolf, clear: parsed.claimedClear });
      }
      for (const t of parsed.accused) {
        if (!accusations.has(e.seat)) accusations.set(e.seat, new Map());
        const m = accusations.get(e.seat)!;
        m.set(t, (m.get(t) ?? 0) + 1);
        accusedCount.set(t, (accusedCount.get(t) ?? 0) + 1);
        if (knownClear.has(t)) bump(e.seat, 8); // accused a proven villager
        else if (knownWolf.has(t)) bump(e.seat, -12); // accused a proven wolf
      }
      for (const t of parsed.defended) {
        if (knownWolf.has(t)) bump(e.seat, 10); // defended a proven wolf
        else if (knownClear.has(t)) bump(e.seat, -5); // defended a proven villager
      }
    } else if (e.kind === 'vote' && typeof e.seat === 'number') {
      const target = typeof e.target === 'number' ? e.target : null;
      votes.push({ voter: e.seat, target });
      if (e.day === view.day && target !== null) tallyToday.set(target, (tallyToday.get(target) ?? 0) + 1);
      if (target !== null) {
        if (knownClear.has(target)) bump(e.seat, 15); // voted to exile a proven villager
        else if (knownWolf.has(target)) bump(e.seat, -20); // voted to exile a proven wolf
      }
    }
  }

  // Seer claims judged against revealed roles: a false wolf call is damning,
  // a true one clears the claimant.
  for (const c of claims) {
    if (c.wolf !== null) {
      if (knownClear.has(c.wolf)) bump(c.claimant, 25);
      else if (knownWolf.has(c.wolf)) bump(c.claimant, -30);
    }
    if (c.clear !== null && knownWolf.has(c.clear)) bump(c.claimant, 25);
  }
  // Two seer claimants cannot both be right. When I am the seer, every other
  // claimant is lying; otherwise the claimants split the doubt.
  const claimants = [...new Set(claims.map((c) => c.claimant))];
  if (claimants.length > 1) {
    for (const c of claimants) {
      if (view.you.role === 'seer' && c !== view.you.seat) bump(c, 40);
      else if (view.you.role !== 'seer') bump(c, 5);
    }
  }
  // A seat our own seer results prove is a wolf outranks every heuristic.
  for (const s of knownWolf) if (suspicion.has(s)) suspicion.set(s, 1000);

  return { suspicion, knownWolf, knownClear, claims, accusations, accusedCount, speeches, votes, tallyToday, nameToSeat };
}

/** Living seats, in seat order. */
const living = (view: View): Seat[] => view.players.filter((p) => p.alive).map((p) => p.seat);

const nameOf = (view: View, seat: Seat): string => view.players.find((p) => p.seat === seat)?.name ?? `seat ${seat}`;

/** Pick the highest-scoring candidate; ties fall to the seeded stream (labelled, so order-free). */
function bestOf(candidates: Seat[], score: (seat: Seat) => number, rng: () => number): Seat | null {
  let best: Seat | null = null;
  let bestScore = -Infinity;
  for (const c of candidates) {
    const s = score(c) + rng() * 0.9; // jitter breaks ties deterministically
    if (s > bestScore) {
      bestScore = s;
      best = c;
    }
  }
  return best;
}

// ---------------------------------------------------------------- decisions

/** The vote: the most-suspected living seat that is not us (and not our partner). */
export function chooseVote(view: View, a: Analysis, seed: string, game: number): Seat | null {
  const me = view.you.seat;
  const partner = view.you.partner?.seat ?? null;
  const rng = rngFor(seed, `vote:${game}:${view.day}`);
  let candidates = living(view).filter((s) => s !== me && s !== partner);
  // Never vote a seat proven clear while any unproven seat lives.
  const unproven = candidates.filter((s) => !a.knownClear.has(s));
  if (unproven.length > 0) candidates = unproven;
  if (candidates.length === 0) return null;
  const topSuspicion = Math.max(...candidates.map((s) => a.suspicion.get(s) ?? 0));
  if (topSuspicion <= 0) {
    // No evidence of our own: follow the village's votes so far today. The
    // house players' public votes carry their reads; a lone seeded pick does not.
    const voted = candidates.filter((s) => (a.tallyToday.get(s) ?? 0) > 0);
    if (voted.length > 0) return bestOf(voted, (s) => a.tallyToday.get(s) ?? 0, rng);
  }
  return bestOf(candidates, (s) => a.suspicion.get(s) ?? 0, rng);
}

/** The night kill, as a werewolf. */
export function chooseKill(view: View, a: Analysis, seed: string, game: number, options: Seat[]): Seat | null {
  const me = view.you.seat;
  const partner = view.you.partner?.seat ?? null;
  // The kill only lands when both wolves name the same seat: when the partner
  // already voted tonight, agree with them.
  const tonight = view.wolfVotes?.filter((v) => v.night === view.day) ?? [];
  const partnerVote = partner !== null ? tonight.find((v) => v.seat === partner) : undefined;
  if (partnerVote !== undefined && partnerVote.target !== null && options.includes(partnerVote.target)) return partnerVote.target;

  const rng = rngFor(seed, `kill:${game}:${view.day}`);
  const seerClaimants = new Set(a.claims.map((c) => c.claimant));
  const accusedMe = (s: Seat): boolean => (a.accusations.get(s)?.get(me) ?? 0) > 0 || (partner !== null && (a.accusations.get(s)?.get(partner) ?? 0) > 0);
  const danger = (s: Seat): number =>
    (seerClaimants.has(s) ? 100 : 0) + (accusedMe(s) ? 25 : 0) + (a.speeches.get(s) ?? 0) * 1 + (a.accusations.get(s)?.size ?? 0) * 2;
  const candidates = options.filter((s) => s !== me && s !== partner);
  if (candidates.length === 0) return null;
  return bestOf(candidates, danger, rng);
}

/** The night protect, as the doctor. */
export function chooseProtect(view: View, a: Analysis, seed: string, game: number, options: Seat[]): Seat | null {
  if (options.length === 0) return null;
  const rng = rngFor(seed, `protect:${game}:${view.day}`);
  const seerClaimants = new Set(a.claims.map((c) => c.claimant));
  const value = (s: Seat): number =>
    (seerClaimants.has(s) ? 100 : 0) + (a.speeches.get(s) ?? 0) * 1 + (a.accusations.get(s)?.size ?? 0) * 3 + (a.accusedCount.get(s) ?? 0) * 0.5;
  return bestOf(options, value, rng);
}

/** The night inspect, as the seer. */
export function chooseInspect(view: View, a: Analysis, seed: string, game: number, options: Seat[]): Seat | null {
  if (options.length === 0) return null;
  const rng = rngFor(seed, `inspect:${game}:${view.day}`);
  const inspected = new Set((view.seerResults ?? []).map((r) => r.seat));
  let candidates = options.filter((s) => !inspected.has(s));
  if (candidates.length === 0) candidates = options;
  return bestOf(candidates, (s) => a.suspicion.get(s) ?? 0, rng);
}

// ---------------------------------------------------------------- statements

function reasonFor(view: View, a: Analysis, seat: Seat): string {
  // The strongest evidence against this seat, in words.
  for (const p of view.players) {
    if (p.role !== null && p.alive === false) {
      const voted = a.votes.some((v) => v.voter === seat && v.target === p.seat);
      if (voted && VILLAGE_ROLES.has(p.role)) return `they voted to exile seat ${p.seat} (${p.name}), who turned out a ${p.role}`;
      const accused = (a.accusations.get(seat)?.get(p.seat) ?? 0) > 0;
      if (accused && VILLAGE_ROLES.has(p.role)) return `they accused seat ${p.seat} (${p.name}), who turned out a ${p.role}`;
      if (accused && p.role === 'werewolf') return `they went after seat ${p.seat} (${p.name}) only after the wolf was already exposed`;
    }
  }
  const n = a.accusedCount.get(seat) ?? 0;
  if (n >= 2) return 'more than one of us has already pointed at them';
  if ((a.speeches.get(seat) ?? 0) === 0) return 'they have said almost nothing while the village talks';
  return 'their story does not add up';
}

const pickTemplate = (templates: readonly string[], rng: () => number): string => templates[Math.floor(rng() * templates.length)]!;

/** One statement for the debate, or null to pass (we almost always speak). */
export function chooseSpeak(view: View, a: Analysis, seed: string, game: number): string | null {
  const me = view.you.seat;
  const role = view.you.role;
  const partner = view.you.partner?.seat ?? null;
  const rng = rngFor(seed, `speak:${game}:${view.day}:${view.log.filter((e) => e.kind === 'speech' || e.kind === 'pass').length}`);
  const target = chooseVote(view, a, seed, game);

  // As the seer with a found wolf alive, say so at once: the village needs it.
  if (role === 'seer') {
    const found = [...a.knownWolf].filter((s) => s !== me && living(view).includes(s));
    if (found.length > 0) {
      const s = found[0]!;
      return `I am the seer. I checked seat ${s} (${nameOf(view, s)}) last night and they are a werewolf. Vote seat ${s} out today.`;
    }
  }
  // A claimant's wolf call we have no reason to doubt is worth backing.
  if (role !== 'werewolf') {
    for (const c of a.claims) {
      if (c.wolf !== null && living(view).includes(c.wolf) && !a.knownClear.has(c.wolf) && c.claimant !== me) {
        const claimant = c.claimant;
        // Do not back a claimant our own results contradict.
        if (role === 'seer' && (view.seerResults ?? []).some((r) => r.seat === c.wolf && !r.werewolf)) continue;
        return `Seat ${claimant} says they checked seat ${c.wolf} (${nameOf(view, c.wolf)}) and found a werewolf. That fits what I have seen — vote seat ${c.wolf}.`;
      }
    }
  }
  // Pushed on hard, defend once and turn it back on the accuser's favourite target.
  const pushedOn = (a.accusedCount.get(me) ?? 0) >= 2;
  if (pushedOn && target !== null) {
    return `I am not a wolf. Seat ${target} (${nameOf(view, target)}) is pushing on me to deflect from themself — look at their votes.`;
  }
  if (target === null) return null;
  const reason = reasonFor(view, a, target);
  const name = nameOf(view, target);
  if (role === 'werewolf') {
    return pickTemplate(
      [
        `Seat ${target} (${name}) is my top suspect: ${reason}. Let's vote seat ${target} out.`,
        `I have been watching seat ${target} (${name}). ${reason[0]!.toUpperCase()}${reason.slice(1)}. I am voting seat ${target} today.`,
        `Seat ${target} (${name}) keeps dodging the real question. ${reason[0]!.toUpperCase()}${reason.slice(1)}. Vote seat ${target}.`,
      ],
      rng,
    );
  }
  return pickTemplate(
    [
      `Seat ${target} (${name}) is my top suspect: ${reason}. Let's vote seat ${target} out.`,
      `The evidence points at seat ${target} (${name}) — ${reason}. I am voting seat ${target} today.`,
      `Seat ${target} (${name}) looks worst to me: ${reason}. Vote seat ${target}.`,
    ],
    rng,
  );
}

// ---------------------------------------------------------------- the run

export interface RunOptions {
  envUrl?: string;
  /** The trial seed; MAKEFUN_TRIAL_SEED when omitted. */
  seed?: string;
  brief?: string;
  log?: (line: string) => void;
}

export interface RunSummary {
  decisions: number;
}

interface Reply {
  status: number;
  body: unknown;
}

async function readReply(res: Response): Promise<Reply> {
  const text = await res.text();
  let body: unknown = null;
  try {
    body = JSON.parse(text);
  } catch {
    body = text;
  }
  return { status: res.status, body };
}

/** The pass or abstention for a decision. */
function fallbackFor(p: Pick<Pending, 'game' | 'decision'>): Record<string, unknown> {
  return p.decision === 'speak' ? { game: p.game, decision: 'speak', pass: true } : { game: p.game, decision: p.decision, target: null };
}

/** The /act body for one pending decision: pure rule logic, no model, no clock. */
export function actFor(p: Pending, seed: string): Record<string, unknown> {
  const view = p.view;
  const a = analyze(view);
  const max = p.max_chars ?? 500;
  switch (p.decision) {
    case 'speak': {
      const text = chooseSpeak(view, a, seed, p.game);
      if (text === null) return fallbackFor(p);
      const cut = cutTo(text, max);
      return cut === '' ? fallbackFor(p) : { game: p.game, decision: 'speak', text: cut };
    }
    case 'vote': {
      const t = chooseVote(view, a, seed, p.game);
      return { game: p.game, decision: 'vote', target: t };
    }
    case 'kill': {
      const t = chooseKill(view, a, seed, p.game, p.options ?? []);
      return { game: p.game, decision: 'kill', target: t };
    }
    case 'protect': {
      const t = chooseProtect(view, a, seed, p.game, p.options ?? []);
      return { game: p.game, decision: 'protect', target: t };
    }
    case 'inspect': {
      const t = chooseInspect(view, a, seed, p.game, p.options ?? []);
      return { game: p.game, decision: 'inspect', target: t };
    }
  }
}

export async function run(o: RunOptions = {}): Promise<RunSummary> {
  const env = process.env;
  const envUrl = (o.envUrl ?? env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
  const seed = o.seed ?? env.MAKEFUN_TRIAL_SEED ?? 'werewolf-agent-01';
  const log = o.log ?? ((line: string) => process.stdout.write(`${line}\n`));
  if (envUrl === '') throw new Error('MAKEFUN_ENV_URL is needed');
  if (o.brief === undefined) {
    const file = env.MAKEFUN_PROMPT_FILE ?? '';
    if (file === '') throw new Error('MAKEFUN_PROMPT_FILE is needed');
    readFileSync(file, 'utf8'); // the brief is read, as the harness expects; the rules need no more than the view
  }

  const envPost = async (path: string, body: unknown): Promise<Reply> =>
    readReply(await fetch(`${envUrl}${path}`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) }));

  let decisions = 0;
  for (;;) {
    const r = await envPost('/observe', {});
    if (r.status !== 200 || !isRecord(r.body)) throw new Error(`POST /observe answered ${r.status}`);
    const obs = r.body as unknown as Observed;
    if (obs.done) break;
    // One request at a time, in the order the env lists them: what we send
    // never depends on which reply came back first.
    for (const p of obs.pending) {
      decisions += 1;
      const act = actFor(p, seed);
      let posted = await envPost('/act', act);
      if (posted.status === 400 && JSON.stringify(act) !== JSON.stringify(fallbackFor(p))) {
        posted = await envPost('/act', fallbackFor(p));
      }
      if (posted.status !== 200 && posted.status !== 409) throw new Error(`POST /act answered ${posted.status} for game ${p.game} ${p.decision}`);
    }
  }
  log(`werewolf-agent: ${decisions} decisions, no model calls, trial done`);
  return { decisions };
}

const isMain = process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isMain) {
  try {
    await run();
    process.exitCode = 0;
  } catch (err) {
    process.stderr.write(`werewolf-agent: ${(err as Error).stack ?? String(err)}\n`);
    process.exitCode = 1;
  }
}
