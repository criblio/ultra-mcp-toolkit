// Streamable-HTTP MCP-server startup helper.
//
// One process serves many MCP clients. Each session gets its own
// `Server` from `createServer`, because an SDK `Server` connects to a
// single transport.
//
// A local HTTP listener also needs checks that stdio never did:
//   1. Host and Origin allowlists. Without them a web page can reach a
//      loopback server through DNS rebinding.
//   2. `bindHeaders`. Each request in a session must repeat the
//      credential headers the session was opened with, so a leaked
//      session id alone cannot act as that client.
//   3. A body limit on every POST. The SDK transport buffers request
//      bodies without one.
//
// Responses use SSE, the SDK default. In SDK 1.29.0 the JSON response
// mode never removes its per-request stream entry, so a long-lived
// session would grow without bound.

import { randomUUID, timingSafeEqual } from "node:crypto";
import {
  createServer as createListener,
  type IncomingHttpHeaders,
  type IncomingMessage,
  type ServerResponse,
} from "node:http";
import type { AddressInfo } from "node:net";
import type { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { isInitializeRequest } from "@modelcontextprotocol/sdk/types.js";

export interface HttpSessionContext {
  // Headers of the initialize request that opened the session.
  headers: IncomingHttpHeaders;
}

// Throw from `createServer` to refuse a session, e.g. when the
// credential headers are missing. The client gets a 401 with `message`.
export class HttpSessionRejected extends Error {
  constructor(message: string) {
    super(message);
    this.name = "HttpSessionRejected";
  }
}

export interface HttpBootOpts {
  // Called once per session, on its initialize request.
  createServer: (ctx: HttpSessionContext) => Server | Promise<Server>;
  // Default "127.0.0.1". Use "0.0.0.0" only inside a container whose
  // port is published on the host's loopback.
  host?: string;
  // Default 0: the OS picks a free port (see the returned handle).
  port?: number;
  // Default "/mcp".
  path?: string;
  // Exact Host header values to accept. Default: localhost, 127.0.0.1
  // and [::1] at the bound port.
  allowedHosts?: string[];
  // Exact Origin header values to accept. Default: none. MCP clients
  // other than browsers send no Origin, so any Origin is refused.
  allowedOrigins?: string[];
  // Header names (case-insensitive) whose initialize-request values
  // every later request in the session must repeat exactly.
  bindHeaders?: string[];
  // Close a session whose client has not sent `notifications/initialized`
  // this long after initialize. Default 30 s. 0 disables.
  initTimeoutMs?: number;
  // Close sessions with no request for this long. Default 0: never.
  // Long-lived agent sessions can sit idle for hours between calls.
  idleTimeoutMs?: number;
  // Largest request body accepted. Default 4 MiB.
  maxBodyBytes?: number;
  // One-line stderr banner; the helper appends the listen URL.
  banner?: string;
  // Optional cleanup on SIGINT/SIGTERM, after sessions and the listener
  // are closed. Errors are swallowed.
  onShutdown?: (signal: NodeJS.Signals) => Promise<void> | void;
}

export interface HttpServerHandle {
  url: string;
  port: number;
  sessionCount(): number;
  // Idempotent. Also removes the signal handlers.
  close(): Promise<void>;
}

interface Session {
  transport: StreamableHTTPServerTransport;
  bound: Map<string, string>;
  lastSeen: number;
}

class HttpError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

const DEFAULT_MAX_BODY_BYTES = 4 * 1024 * 1024;
const DEFAULT_INIT_TIMEOUT_MS = 30_000;
const MAX_REAP_INTERVAL_MS = 60_000;

export async function startHttpServer(opts: HttpBootOpts): Promise<HttpServerHandle> {
  const host = opts.host ?? "127.0.0.1";
  const path = opts.path ?? "/mcp";
  const maxBodyBytes = opts.maxBodyBytes ?? DEFAULT_MAX_BODY_BYTES;
  const initTimeoutMs = opts.initTimeoutMs ?? DEFAULT_INIT_TIMEOUT_MS;
  const bindHeaders = (opts.bindHeaders ?? []).map((name) => name.toLowerCase());
  const allowedOrigins = new Set(opts.allowedOrigins ?? []);
  const sessions = new Map<string, Session>();
  let allowedHosts = new Set<string>();
  let closing = false;

  async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
    if (closing) throw new HttpError(503, "Server is shutting down");
    const { pathname } = new URL(req.url ?? "/", "http://placeholder");
    if (pathname !== path) throw new HttpError(404, "Not found");
    if (!allowedHosts.has((req.headers.host ?? "").toLowerCase())) {
      throw new HttpError(403, "Host not allowed");
    }
    const origin = req.headers.origin;
    if (origin !== undefined && !allowedOrigins.has(origin)) {
      throw new HttpError(403, "Origin not allowed");
    }

    const sessionId = headerValue(req.headers, "mcp-session-id");
    if (sessionId !== undefined) {
      const session = sessions.get(sessionId);
      if (!session) throw new HttpError(404, "Session not found");
      if (!matchesBound(session.bound, req.headers)) {
        throw new HttpError(403, "Request headers do not match the session");
      }
      session.lastSeen = Date.now();
      const body = req.method === "POST" ? await readJson(req, maxBodyBytes) : undefined;
      // The session can close while the body is still arriving.
      if (sessions.get(sessionId) !== session) throw new HttpError(404, "Session not found");
      await session.transport.handleRequest(req, res, body);
      return;
    }

    if (req.method !== "POST") throw new HttpError(400, "Missing Mcp-Session-Id header");
    // Checked here as well as in the SDK so a request it would refuse
    // never reaches `createServer`.
    checkPostHeaders(req.headers);
    const body = await readJson(req, maxBodyBytes);
    if (!isInitializeRequest(body)) {
      throw new HttpError(400, "A request without Mcp-Session-Id must be an initialize request");
    }

    let server: Server;
    try {
      server = await opts.createServer({ headers: req.headers });
    } catch (err) {
      if (err instanceof HttpSessionRejected) throw new HttpError(401, err.message);
      throw err;
    }

    const bound = new Map(bindHeaders.map((name) => [name, headerValue(req.headers, name) ?? ""]));
    let initialized = false;
    let initTimer: NodeJS.Timeout | undefined;
    const transport = new StreamableHTTPServerTransport({
      sessionIdGenerator: () => randomUUID(),
      onsessioninitialized: (id) => {
        if (closing) return;
        sessions.set(id, { transport, bound, lastSeen: Date.now() });
        if (initTimeoutMs > 0) {
          initTimer = setTimeout(() => {
            if (!initialized) void transport.close().catch(() => {});
          }, initTimeoutMs);
          initTimer.unref();
        }
      },
    });
    // Set before connect(): Protocol.connect() keeps this handler and
    // calls it ahead of its own cleanup.
    transport.onclose = () => {
      clearTimeout(initTimer);
      const id = transport.sessionId;
      if (id !== undefined && sessions.get(id)?.transport === transport) sessions.delete(id);
    };
    const previousOnInitialized = server.oninitialized;
    server.oninitialized = () => {
      initialized = true;
      clearTimeout(initTimer);
      // Returned so the SDK routes a rejected async handler to onerror.
      return previousOnInitialized?.call(server);
    };

    await server.connect(transport);
    try {
      await transport.handleRequest(req, res, body);
    } finally {
      // The SDK can answer an initialize with an error after connect()
      // without registering a session, and a shutdown can start while the
      // request is in flight. Either way nobody else will close it.
      const id = transport.sessionId;
      if (id === undefined || sessions.get(id)?.transport !== transport) {
        await transport.close().catch(() => {});
      }
    }
  }

  const listener = createListener((req, res) => {
    handle(req, res).catch((err: unknown) => {
      if (res.headersSent) {
        res.end();
        return;
      }
      if (err instanceof HttpError) {
        sendError(res, err.status, err.message);
        return;
      }
      process.stderr.write(`mcp http: ${err instanceof Error ? err.message : String(err)}\n`);
      sendError(res, 500, "Internal error");
    });
  });

  await new Promise<void>((resolve, reject) => {
    listener.once("error", reject);
    listener.listen(opts.port ?? 0, host, () => {
      listener.off("error", reject);
      resolve();
    });
  });
  const port = (listener.address() as AddressInfo).port;
  allowedHosts = new Set(
    (opts.allowedHosts ?? [`localhost:${port}`, `127.0.0.1:${port}`, `[::1]:${port}`]).map((h) =>
      h.toLowerCase(),
    ),
  );
  const url = `http://${host.includes(":") ? `[${host}]` : host}:${port}${path}`;

  let reaper: NodeJS.Timeout | undefined;
  const idleTimeoutMs = opts.idleTimeoutMs ?? 0;
  if (idleTimeoutMs > 0) {
    reaper = setInterval(
      () => {
        const cutoff = Date.now() - idleTimeoutMs;
        for (const [id, session] of sessions) {
          if (session.lastSeen < cutoff) {
            sessions.delete(id);
            void session.transport.close().catch(() => {});
          }
        }
      },
      Math.min(idleTimeoutMs, MAX_REAP_INTERVAL_MS),
    );
    reaper.unref();
  }

  let signalHandler: ((signal: NodeJS.Signals) => void) | undefined;
  let closed: Promise<void> | undefined;

  function close(): Promise<void> {
    closing = true;
    closed ??= (async () => {
      if (reaper) clearInterval(reaper);
      if (signalHandler) {
        process.off("SIGINT", signalHandler);
        process.off("SIGTERM", signalHandler);
      }
      const open = [...sessions.values()];
      sessions.clear();
      await Promise.allSettled(open.map((session) => session.transport.close()));
      await new Promise<void>((resolve) => {
        listener.close(() => resolve());
        listener.closeAllConnections();
      });
    })();
    return closed;
  }

  if (opts.banner) process.stderr.write(`${opts.banner} on ${url}\n`);

  if (opts.onShutdown) {
    const onShutdown = opts.onShutdown;
    let shuttingDown = false;
    signalHandler = (signal) => {
      if (shuttingDown) return;
      shuttingDown = true;
      void (async () => {
        try {
          await close();
          await onShutdown(signal);
        } catch {
          // Best-effort: we're shutting down anyway, don't block exit.
        }
        process.exit(0);
      })();
    };
    process.on("SIGINT", signalHandler);
    process.on("SIGTERM", signalHandler);
  }

  return { url, port, sessionCount: () => sessions.size, close };
}

function headerValue(headers: IncomingHttpHeaders, name: string): string | undefined {
  const value = headers[name];
  return Array.isArray(value) ? value.join(", ") : value;
}

function checkPostHeaders(headers: IncomingHttpHeaders): void {
  const accept = headerValue(headers, "accept") ?? "";
  if (!accept.includes("application/json") || !accept.includes("text/event-stream")) {
    throw new HttpError(406, "Client must accept both application/json and text/event-stream");
  }
  if (!(headerValue(headers, "content-type") ?? "").includes("application/json")) {
    throw new HttpError(415, "Content-Type must be application/json");
  }
}

function matchesBound(bound: Map<string, string>, headers: IncomingHttpHeaders): boolean {
  for (const [name, expected] of bound) {
    const actual = Buffer.from(headerValue(headers, name) ?? "");
    const wanted = Buffer.from(expected);
    if (actual.length !== wanted.length || !timingSafeEqual(actual, wanted)) return false;
  }
  return true;
}

async function readJson(req: IncomingMessage, maxBytes: number): Promise<unknown> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    size += (chunk as Buffer).length;
    if (size > maxBytes) throw new HttpError(413, "Request body too large");
    chunks.push(chunk as Buffer);
  }
  try {
    return JSON.parse(Buffer.concat(chunks).toString("utf8"));
  } catch {
    throw new HttpError(400, "Request body is not valid JSON");
  }
}

function sendError(res: ServerResponse, status: number, message: string): void {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ jsonrpc: "2.0", error: { code: -32000, message }, id: null }));
}
