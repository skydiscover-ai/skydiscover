/**
 * SkyDiscover test extension for pi.
 *
 * pi has no hooks.json, so this extension calls the same two scripts under workflow/hooks/ that
 * install.sh wires as hooks for Claude Code and Codex:
 *
 *   clone-reuse guard  ->  pi tool_call(bash)          ->  clone_reuse_guard.py    (exit 2 -> block)
 *   delivery check     ->  pi tool skydiscover_deliver ->  delivery_check.sh  (exit != 0 -> throw)
 *
 * install.sh --agent pi replaces __SKYDISCOVER_HOOKS_DIR__ with this checkout's workflow/hooks
 * directory. Set SKYDISCOVER_HOOKS_DIR to override at runtime.
 */
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import * as path from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

const HOOKS_DIR = process.env.SKYDISCOVER_HOOKS_DIR || "__SKYDISCOVER_HOOKS_DIR__";
const CLONE_GUARD = path.join(HOOKS_DIR, "clone_reuse_guard.py");
const DELIVERY_CHECK = path.join(HOOKS_DIR, "delivery_check.sh");

export default function skydiscoverHooks(pi: ExtensionAPI) {
	// Clone-reuse guard -- the PreToolUse-equivalent. Before any bash command runs, hand the command
	// to the same guard the Claude and Codex hooks use. Exit 2 means "the wiki already grounds this
	// source at this exact commit -- reuse it, do not re-clone", which pi surfaces as a blocked call.
	// Anything else is allowed (exit 0), so the guard can only ever save wasted work.
	//
	// Allow the command if the guard itself cannot run: python3's own "no such file" exit is also 2,
	// so a missing hooks dir would otherwise block every bash command. Only a real guard verdict
	// (exit 2 with the script present and python3 launched) blocks the call.
	pi.on("tool_call", async (event) => {
		if (event.toolName !== "bash") return undefined;
		const command = (event.input?.command as string | undefined) ?? "";
		if (!command) return undefined;
		if (!existsSync(CLONE_GUARD)) return undefined;
		const res = spawnSync("python3", [CLONE_GUARD], {
			input: JSON.stringify({ tool_name: "Bash", tool_input: { command } }),
			encoding: "utf-8",
		});
		if (res.error) return undefined; // python3 missing / spawn failed -> don't block
		if (res.status === 2) {
			const reason = (res.stderr || "").trim();
			return { block: true, reason: reason || "skydiscover reuse guard: reuse the page instead of re-cloning" };
		}
		return undefined;
	});

	// Delivery check -- the TaskCompleted/SubagentStop-equivalent. pi has no task-completion hook, so
	// the impl is "delivered" by calling this tool. Setting SKYDISCOVER_RUN latches the delivery path
	// inside the script, so it always runs the test suite (never a silent skip);
	// a non-zero exit (2 = test violation) is surfaced by THROWING, so the impl can never be marked
	// done unchecked. (pi only marks a tool call errored when execute throws -- the isError field on a
	// normal return is ignored -- so the throw is load-bearing, not decorative. The test output rides
	// the error message, so the agent still sees exactly what failed.) This is the SAME script and the
	// SAME two-sided tests as the Claude and Codex hooks.
	pi.registerTool({
		name: "skydiscover_deliver",
		label: "SkyDiscover delivery check",
		description:
			"Check the delivered impl against the test suite. Call this to mark " +
			"the impl-delivery task done; it fails unless the test is met. SKYDISCOVER_RUN (or the `run` " +
			"argument) must point at the run dir; every test artifact derives from it.",
		promptSnippet: "Run skydiscover_deliver to validate the impl against the test before marking it done.",
		promptGuidelines: [
			"The impl-delivery task is done only when skydiscover_deliver returns PASS.",
			"On FAIL, read the test output, fix the impl, and call skydiscover_deliver again.",
		],
		parameters: Type.Object({
			run: Type.Optional(Type.String({ description: "Run dir, e.g. .skydiscover/<slug> (defaults to $SKYDISCOVER_RUN)." })),
		}),
		async execute(_toolCallId, params) {
			const env = { ...process.env };
			const run = params.run || process.env.SKYDISCOVER_RUN;
			if (!run) {
				// With no run dir, delivery_check.sh gets an empty payload and no
				// enforcement env, so it would exit 0 (nothing to check) and this tool would return
				// PASS having checked nothing. pi has no title backstop (unlike the Claude/Codex
				// hooks), so the run dir is the only delivery signal -- refuse rather than wave through.
				throw new Error(
					"delivery check: cannot check -- no run dir. Pass the `run` argument (e.g. .skydiscover/<slug>) " +
						"or set $SKYDISCOVER_RUN; every test artifact derives from it.",
				);
			}
			env.SKYDISCOVER_RUN = run;
			const res = spawnSync("bash", [DELIVERY_CHECK], { input: "", encoding: "utf-8", env });
			const out = `${res.stdout || ""}${res.stderr || ""}`.trim();
			if (res.status !== 0) {
				// Throw so pi records a real tool error (a returned isError is ignored).
				// The test output is the message, so the agent reads exactly why it failed and retries.
				throw new Error(`delivery check: FAIL (exit ${res.status})\n${out}`);
			}
			return {
				content: [{ type: "text", text: `delivery check: PASS\n${out}` }],
				details: { exitCode: res.status },
			};
		},
	});
}
