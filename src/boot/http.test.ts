import { request } from "node:http";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import type { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";
import {
  HttpSessionRejected,
  startHttpServer,
  type HttpBootOpts,
  type HttpServerHandle,
  type HttpSessionContext,
} from "./http.js";

const INITIALIZE = {
  jsonrpc: "2.0",
  id: 1,
  method: "initialize",
  params: {
    protocolVersion: "2025-06-18",
    capabilities: {},
    clientInfo: { name: "test", version: "0" },
  },
};
const INITIALIZED = { jsonrpc: "2.0", method: "notifications/initialized" };
const LIST_TOOLS = { jsonrpc: "2.0", id: 2, method: "tools/list", params: {} };

// Answers `whoami` with the test-token header its session opened with.
function tokenEchoServer(ctx: HttpSessionContext, onCall?: () => void): Server {
  const token = ctx.headers["test-token"];
  if (typeof token !== "string") throw new HttpSessionRejected("test-token header required");
  const server = new Server({ name: "echo", version: "0" }, { capabilities: { tools: {} } });
  server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: [{ name: "whoami", inputSchema: { type: "object" as const } }],
  }));
  server.setRequestHandler(CallToolRequestSchema, async () => {
    onCall?.();
    return { content: [{ type: "text", text: token }] };
  });
  return server;
}

interface RawResponse {
  status: number;
  headers: Record<string, string | string[] | undefined>;
  body: string;
}

function raw(
  handle: HttpServerHandle,
  opts: {
    method?: string;
    path?: string;
    headers?: Record<string, string>;
    body?: string;
    chunks?: string[];
    chunkDelayMs?: number;
  },
): Promise<RawResponse> {
  return new Promise((resolve, reject) => {
    const req = request(
      {
        host: "127.0.0.1",
        port: handle.port,
        method: opts.method ?? "POST",
        path: opts.path ?? "/mcp",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json, text/event-stream",
          ...opts.headers,
        },
      },
      (res) => {
        let body = "";
        res.setEncoding("utf8");
        res.on("data", (chunk: string) => (body += chunk));
        res.on("end", () => resolve({ status: res.statusCode ?? 0, headers: res.headers, body }));
      },
    );
    req.on("error", reject);
    void (async () => {
      for (const chunk of opts.chunks ?? []) {
        req.write(chunk);
        if (opts.chunkDelayMs) await sleep(opts.chunkDelayMs);
      }
      req.end(opts.body);
    })();
  });
}

// JSON-RPC messages carried in an SSE response body.
function sseMessages(body: string): Array<Record<string, any>> {
  return body
    .split("\n")
    .filter((line) => line.startsWith("data: "))
    .map((line) => JSON.parse(line.slice("data: ".length)));
}

async function openSession(handle: HttpServerHandle, token: string): Promise<string> {
  const res = await raw(handle, {
    headers: { "test-token": token },
    body: JSON.stringify(INITIALIZE),
  });
  expect(res.status).toBe(200);
  const id = res.headers["mcp-session-id"];
  expect(typeof id).toBe("string");
  return id as string;
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

describe("startHttpServer", () => {
  let handle: HttpServerHandle | undefined;

  afterEach(async () => {
    await handle?.close();
    handle = undefined;
  });

  async function start(extra: Partial<HttpBootOpts> = {}): Promise<HttpServerHandle> {
    handle = await startHttpServer({ createServer: tokenEchoServer, ...extra });
    return handle;
  }

  it("serves an SDK client and gives each session its own server", async () => {
    const h = await start();
    const callAs = async (token: string) => {
      const transport = new StreamableHTTPClientTransport(new URL(h.url), {
        requestInit: { headers: { "test-token": token } },
      });
      const client = new Client({ name: "test", version: "0" });
      await client.connect(transport);
      const tools = await client.listTools();
      const result = await client.callTool({ name: "whoami", arguments: {} });
      return { client, transport, tools, result };
    };

    const alice = await callAs("alice");
    const bob = await callAs("bob");
    expect(alice.tools.tools.map((t) => t.name)).toEqual(["whoami"]);
    expect(alice.result.content).toEqual([{ type: "text", text: "alice" }]);
    expect(bob.result.content).toEqual([{ type: "text", text: "bob" }]);
    expect(h.sessionCount()).toBe(2);

    await alice.transport.terminateSession();
    expect(h.sessionCount()).toBe(1);
    await alice.client.close();
    await bob.client.close();
  });

  it("keeps an SDK client's session past initTimeoutMs once the handshake completes", async () => {
    const h = await start({ initTimeoutMs: 50 });
    const transport = new StreamableHTTPClientTransport(new URL(h.url), {
      requestInit: { headers: { "test-token": "alice" } },
    });
    const client = new Client({ name: "test", version: "0" });
    await client.connect(transport);
    await sleep(150);
    expect(h.sessionCount()).toBe(1);
    await client.close();
  });

  it("returns 401 when createServer rejects the session", async () => {
    const h = await start();
    const res = await raw(h, { body: JSON.stringify(INITIALIZE) });
    expect(res.status).toBe(401);
    expect(JSON.parse(res.body).error.message).toBe("test-token header required");
    expect(h.sessionCount()).toBe(0);
  });

  it("refuses a bad Accept or Content-Type before calling createServer", async () => {
    let calls = 0;
    const h = await start({
      createServer: (ctx) => {
        calls += 1;
        return tokenEchoServer(ctx);
      },
    });
    const badAccept = await raw(h, {
      headers: { Accept: "application/json", "test-token": "t" },
      body: JSON.stringify(INITIALIZE),
    });
    expect(badAccept.status).toBe(406);
    const badType = await raw(h, {
      headers: { "Content-Type": "text/plain", "test-token": "t" },
      body: JSON.stringify(INITIALIZE),
    });
    expect(badType.status).toBe(415);
    expect(calls).toBe(0);
    expect(h.sessionCount()).toBe(0);
  });

  it("refuses a Host header outside the allowlist", async () => {
    const h = await start();
    const res = await raw(h, {
      headers: { Host: "attacker.example", "test-token": "t" },
      body: JSON.stringify(INITIALIZE),
    });
    expect(res.status).toBe(403);
  });

  it("accepts allowedHosts overrides", async () => {
    const h = await start({ allowedHosts: ["mcp.internal:9999"] });
    const refused = await raw(h, {
      headers: { "test-token": "t" },
      body: JSON.stringify(INITIALIZE),
    });
    expect(refused.status).toBe(403);
    const accepted = await raw(h, {
      headers: { Host: "mcp.internal:9999", "test-token": "t" },
      body: JSON.stringify(INITIALIZE),
    });
    expect(accepted.status).toBe(200);
  });

  it("refuses any Origin by default", async () => {
    const h = await start();
    const res = await raw(h, {
      headers: { Origin: "http://127.0.0.1", "test-token": "t" },
      body: JSON.stringify(INITIALIZE),
    });
    expect(res.status).toBe(403);
  });

  it("accepts only the Origins in allowedOrigins", async () => {
    const h = await start({ allowedOrigins: ["https://ok.example"] });
    const refused = await raw(h, {
      headers: { Origin: "https://evil.example", "test-token": "t" },
      body: JSON.stringify(INITIALIZE),
    });
    expect(refused.status).toBe(403);
    const accepted = await raw(h, {
      headers: { Origin: "https://ok.example", "test-token": "t" },
      body: JSON.stringify(INITIALIZE),
    });
    expect(accepted.status).toBe(200);
  });

  it("returns 404 for an unknown session id", async () => {
    const h = await start();
    const res = await raw(h, {
      headers: { "mcp-session-id": "no-such-session" },
      body: JSON.stringify(LIST_TOOLS),
    });
    expect(res.status).toBe(404);
  });

  it("requires later requests to repeat the bound headers", async () => {
    const h = await start({ bindHeaders: ["Test-Token"] });
    const id = await openSession(h, "alice");

    for (const token of ["mallory", "alicf"]) {
      const wrong = await raw(h, {
        headers: { "mcp-session-id": id, "test-token": token },
        body: JSON.stringify(LIST_TOOLS),
      });
      expect(wrong.status).toBe(403);
    }

    const missing = await raw(h, {
      headers: { "mcp-session-id": id },
      body: JSON.stringify(LIST_TOOLS),
    });
    expect(missing.status).toBe(403);

    const same = await raw(h, {
      headers: { "mcp-session-id": id, "test-token": "alice" },
      body: JSON.stringify(LIST_TOOLS),
    });
    expect(same.status).toBe(200);
    expect(sseMessages(same.body)[0].result.tools[0].name).toBe("whoami");
  });

  it("answers session requests as SSE, not JSON", async () => {
    const h = await start();
    const id = await openSession(h, "alice");
    const res = await raw(h, {
      headers: { "mcp-session-id": id, "test-token": "alice" },
      body: JSON.stringify(LIST_TOOLS),
    });
    expect(res.headers["content-type"]).toContain("text/event-stream");
  });

  it("applies maxBodyBytes to requests inside a session", async () => {
    const h = await start({ maxBodyBytes: 1024 });
    const id = await openSession(h, "alice");
    const big = JSON.stringify({ ...LIST_TOOLS, params: { pad: "x".repeat(4096) } });
    const res = await raw(h, {
      headers: { "mcp-session-id": id, "test-token": "alice" },
      chunks: [big.slice(0, 2048), big.slice(2048)],
    });
    expect(res.status).toBe(413);
  });

  it("returns 400 for a non-initialize request without a session", async () => {
    const h = await start();
    const post = await raw(h, { body: JSON.stringify(LIST_TOOLS) });
    expect(post.status).toBe(400);
    const get = await raw(h, { method: "GET" });
    expect(get.status).toBe(400);
  });

  it("returns 400 for a body that is not JSON", async () => {
    const h = await start();
    const res = await raw(h, { body: "{not json" });
    expect(res.status).toBe(400);
  });

  it("returns 413 for an initialize body over maxBodyBytes", async () => {
    const h = await start({ maxBodyBytes: 64 });
    const res = await raw(h, {
      headers: { "test-token": "t" },
      body: JSON.stringify(INITIALIZE),
    });
    expect(res.status).toBe(413);
  });

  it("returns 404 outside the MCP path", async () => {
    const h = await start();
    const res = await raw(h, { path: "/other", body: JSON.stringify(INITIALIZE) });
    expect(res.status).toBe(404);
  });

  it("closes a session that never sends notifications/initialized", async () => {
    const h = await start({ initTimeoutMs: 50 });
    await openSession(h, "alice");
    expect(h.sessionCount()).toBe(1);
    await sleep(150);
    expect(h.sessionCount()).toBe(0);
  });

  it("does not run a request whose session closed while its body was arriving", async () => {
    let toolCalls = 0;
    const h = await start({
      initTimeoutMs: 50,
      createServer: (ctx) => tokenEchoServer(ctx, () => (toolCalls += 1)),
    });
    const id = await openSession(h, "alice");
    const call = JSON.stringify({
      jsonrpc: "2.0",
      id: 3,
      method: "tools/call",
      params: { name: "whoami", arguments: {} },
    });
    const res = await raw(h, {
      headers: { "mcp-session-id": id, "test-token": "alice" },
      chunks: [call.slice(0, 10), call.slice(10)],
      chunkDelayMs: 150,
    });
    expect(res.status).toBe(404);
    expect(toolCalls).toBe(0);
  });

  it("calls a consumer's oninitialized with the server as this", async () => {
    let created: Server | undefined;
    let seen: unknown;
    const h = await start({
      createServer: (ctx) => {
        created = tokenEchoServer(ctx);
        created.oninitialized = function (this: unknown) {
          seen = this;
        };
        return created;
      },
    });
    const transport = new StreamableHTTPClientTransport(new URL(h.url), {
      requestInit: { headers: { "test-token": "alice" } },
    });
    const client = new Client({ name: "test", version: "0" });
    await client.connect(transport);
    await vi.waitFor(() => expect(seen).toBeDefined());
    expect(seen).toBe(created);
    await client.close();
  });

  it("routes a rejected async oninitialized to server.onerror", async () => {
    const errors: Error[] = [];
    const h = await start({
      createServer: (ctx) => {
        const server = tokenEchoServer(ctx);
        server.oninitialized = async () => {
          throw new Error("consumer failed");
        };
        server.onerror = (err) => errors.push(err);
        return server;
      },
    });
    const transport = new StreamableHTTPClientTransport(new URL(h.url), {
      requestInit: { headers: { "test-token": "alice" } },
    });
    const client = new Client({ name: "test", version: "0" });
    await client.connect(transport);
    await vi.waitFor(() =>
      expect(errors.some((e) => e.message.includes("consumer failed"))).toBe(true),
    );
    await client.close();
  });

  it("closes the transport when handleRequest throws during initialize", async () => {
    let created: Server | undefined;
    const h = await start({
      createServer: (ctx) => {
        const server = tokenEchoServer(ctx);
        const connect = server.connect.bind(server);
        server.connect = async (transport) => {
          await connect(transport);
          (transport as StreamableHTTPServerTransport).handleRequest = async () => {
            throw new Error("transport failed");
          };
        };
        created = server;
        return server;
      },
    });
    const res = await raw(h, {
      headers: { "test-token": "t" },
      body: JSON.stringify(INITIALIZE),
    });
    expect(res.status).toBe(500);
    expect(created?.transport).toBeUndefined();
    expect(h.sessionCount()).toBe(0);
  });

  it("closes the transport of an initialize that ends without a registered session", async () => {
    let created: Server | undefined;
    let closing: Promise<void> | undefined;
    const h = await start({
      createServer: (ctx) => {
        const server = tokenEchoServer(ctx);
        const connect = server.connect.bind(server);
        // Shutdown starts between connect() and the session registering.
        server.connect = async (transport) => {
          closing = h.close();
          return connect(transport);
        };
        created = server;
        return server;
      },
    });
    await raw(h, {
      headers: { "test-token": "t" },
      body: JSON.stringify(INITIALIZE),
    }).catch(() => undefined);
    await closing;
    handle = undefined;
    await vi.waitFor(() => expect(created?.transport).toBeUndefined());
    expect(h.sessionCount()).toBe(0);
  });

  it("keeps a session that completed the handshake past initTimeoutMs", async () => {
    const h = await start({ initTimeoutMs: 50 });
    const id = await openSession(h, "alice");
    const ack = await raw(h, {
      headers: { "mcp-session-id": id, "test-token": "alice" },
      body: JSON.stringify(INITIALIZED),
    });
    expect(ack.status).toBe(202);
    await sleep(150);
    expect(h.sessionCount()).toBe(1);
  });

  it("closes sessions idle longer than idleTimeoutMs", async () => {
    const h = await start({ idleTimeoutMs: 50 });
    await openSession(h, "alice");
    expect(h.sessionCount()).toBe(1);
    await sleep(200);
    expect(h.sessionCount()).toBe(0);
  });

  it("keeps idle sessions when idleTimeoutMs is unset", async () => {
    const h = await start();
    await openSession(h, "alice");
    await sleep(100);
    expect(h.sessionCount()).toBe(1);
  });

  it("closes open sessions and the listener on close()", async () => {
    const h = await start();
    await openSession(h, "alice");
    await Promise.all([h.close(), h.close()]);
    handle = undefined;
    expect(h.sessionCount()).toBe(0);
    await expect(raw(h, { body: JSON.stringify(INITIALIZE) })).rejects.toThrow();
  });

  it("does not register a session whose createServer finishes after close()", async () => {
    let release!: () => void;
    const gate = new Promise<void>((resolve) => (release = resolve));
    let entered!: () => void;
    const factoryEntered = new Promise<void>((resolve) => (entered = resolve));
    let created: Server | undefined;
    const h = await start({
      createServer: async (ctx) => {
        entered();
        await gate;
        created = tokenEchoServer(ctx);
        return created;
      },
    });

    const pending = raw(h, {
      headers: { "test-token": "t" },
      body: JSON.stringify(INITIALIZE),
    }).catch((err: unknown) => err);
    await factoryEntered;
    await h.close();
    handle = undefined;
    release();
    await pending;
    await sleep(20);

    expect(created).toBeDefined();
    expect(created?.transport).toBeUndefined();
    expect(h.sessionCount()).toBe(0);
  });

  it("removes both signal handlers on close()", async () => {
    const term = process.listenerCount("SIGTERM");
    const int = process.listenerCount("SIGINT");
    const h = await start({ onShutdown: () => {} });
    expect(process.listenerCount("SIGTERM")).toBe(term + 1);
    expect(process.listenerCount("SIGINT")).toBe(int + 1);
    await h.close();
    handle = undefined;
    expect(process.listenerCount("SIGTERM")).toBe(term);
    expect(process.listenerCount("SIGINT")).toBe(int);
  });

  it("runs the signal shutdown once when the signal repeats", async () => {
    const exit = vi.spyOn(process, "exit").mockImplementation(() => undefined as never);
    try {
      let shutdowns = 0;
      await start({ onShutdown: () => void (shutdowns += 1) });
      // Called directly: emitting a real SIGTERM would also reach the
      // test runner's own handlers.
      const handler = process.listeners("SIGTERM").at(-1) as (signal: NodeJS.Signals) => void;
      handler("SIGTERM");
      handler("SIGTERM");
      await vi.waitFor(() => expect(exit).toHaveBeenCalled());
      handle = undefined;
      expect(shutdowns).toBe(1);
      expect(exit).toHaveBeenCalledTimes(1);
    } finally {
      exit.mockRestore();
    }
  });
});
