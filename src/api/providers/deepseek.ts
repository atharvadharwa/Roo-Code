// deepseek.ts
import axios from "axios"
import fs from "fs"
import { open as fsOpen } from "fs/promises"

import type { ApiHandlerOptions } from "../../shared/api"
import { BaseProvider } from "./base-provider"
import { convertToR1Format } from "../transform/r1-format"
import { getModelParams } from "../transform/model-params"
import { deepSeekModels, deepSeekDefaultModelId, openAiModelInfoSaneDefaults } from "@roo-code/types"
import type { ApiStream, ApiStreamUsageChunk } from "../transform/stream"
import type { SingleCompletionHandler, ApiHandlerCreateMessageMetadata } from "../index"
import type { ModelInfo } from "@roo-code/types"

const DEFAULT_CA_BUNDLE_PATH = "C:\\Users\\45168789\\Documents\\combined_certificates.crt"
const DEFAULT_URL = "http://hk120164337.hc.cloud.hk.hsbc:12214/v1/chat/completions"
const DEFAULT_MODEL = "o3-2025-04-16"
const DEFAULT_TIMEOUT = 120_000 // ms

function makeOutputChannel() {
  try {
    // dynamic require to avoid hard dependency in non-vscode env
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const vscode = require("vscode")
    const ch = vscode.window.createOutputChannel("Deepseek Debug")
    return {
      appendLine: (s: string) => ch.appendLine(s),
      show: (preserveFocus = false) => ch.show(preserveFocus),
    }
  } catch {
    return {
      appendLine: (s: string) => {
        const t = new Date().toISOString()
        // eslint-disable-next-line no-console
        console.log(`[Deepseek Debug ${t}] ${s}`)
      },
      show: (_preserveFocus = false) => undefined,
    }
  }
}

async function readCertHead(caPath: string, maxBytes = 1024): Promise<string> {
  const fd = await fsOpen(caPath, "r")
  try {
    const buffer = Buffer.alloc(Math.max(1, maxBytes))
    const { bytesRead } = await fd.read(buffer, 0, maxBytes, 0)
    return buffer.slice(0, bytesRead).toString("utf8").replace(/\r?\n/g, "\\n")
  } finally {
    await fd.close()
  }
}

/**
 * DeepSeekHandler
 * - non-streaming (single POST)
 * - uses convertToR1Format for messages
 * - logs request/response + cert head to "Deepseek Debug"
 */
export class DeepSeekHandler extends BaseProvider implements SingleCompletionHandler {
  protected options: ApiHandlerOptions
  private output = makeOutputChannel()
  private caBundlePath: string
  private baseUrl: string
  private modelId: string

  constructor(options: ApiHandlerOptions) {
    super()
    this.options = options

    // Allow overrides but default to the values you provided
    // @ts-ignore - keep tolerant access for different casing conventions
    this.caBundlePath = (options.deepSeekCaBundlePath as string) ?? (options.deepseekCaBundlePath as string) ?? DEFAULT_CA_BUNDLE_PATH
    // @ts-ignore
    this.baseUrl = (options.deepSeekBaseUrl as string) ?? (options.deepseekBaseUrl as string) ?? DEFAULT_URL
    // @ts-ignore
    this.modelId = (options.apiModelId as string) ?? DEFAULT_MODEL

    // Must set NODE_EXTRA_CA_CERTS before any outgoing TLS/HTTPS operations
    if (this.caBundlePath) {
      if (!fs.existsSync(this.caBundlePath)) {
        const msg = `CA bundle not found at ${this.caBundlePath}`
        this.output.appendLine(msg)
        // keep original behavior (throw) so misconfig surfaces early
        throw new Error(msg)
      }
      process.env["NODE_EXTRA_CA_CERTS"] = this.caBundlePath

      // Read a small portion asynchronously for logging (non-blocking)
      readCertHead(this.caBundlePath, 1024)
        .then((head) => {
          this.output.appendLine(`CA bundle found at: ${this.caBundlePath}`)
          this.output.appendLine(`CA bundle beginning (first 1KB, newlines escaped): ${head}`)
        })
        .catch((err) => {
          this.output.appendLine(`Failed to read CA bundle head: ${String(err)}`)
        })
    }
  }

  /**
   * createMessage must return ApiStream (async generator) — same shape as openai.ts
   *
   * Note: messages is typed as any[] to avoid coupling to an external SDK type here.
   */
  override async *createMessage(
    systemPrompt: string,
    messages: any[],
    metadata?: ApiHandlerCreateMessageMetadata,
  ): ApiStream {
    // Pull API key from options (deepSeekApiKey preferred)
    // @ts-ignore
    const deepseekApiKey = (this.options.deepSeekApiKey as string) ?? (this.options.deepseekApiKey as string)
    if (!deepseekApiKey) {
      const err = "deepSeekApiKey is missing from ApiHandlerOptions"
      this.output.appendLine(err)
      throw new Error(err)
    }

    // Convert to R1 format per your requirement; include system prompt as first user message (same pattern used elsewhere)
    const converted = convertToR1Format([{ role: "user", content: systemPrompt }, ...messages])

    const body = {
      model: this.modelId,
      messages: converted,
    }

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      Authorization: `Bearer ${deepseekApiKey}`,
    }

    // Logging: request
    this.output.appendLine("----- Deepseek Request START -----")
    this.output.appendLine(`URL: ${this.baseUrl}`)
    this.output.appendLine(`Model: ${this.modelId}`)
    this.output.appendLine("Request Headers (to be sent):")
    Object.entries(headers).forEach(([k, v]) => {
      if (k.toLowerCase() === "authorization") {
        const show = typeof v === "string" ? `${v.slice(0, 7)}...` : String(v)
        this.output.appendLine(`${k}: ${show}`)
      } else {
        this.output.appendLine(`${k}: ${v}`)
      }
    })
    try {
      this.output.appendLine(`Request body: ${JSON.stringify(body)}`)
    } catch (err) {
      this.output.appendLine(`Request body (failed stringify): ${String(err)}`)
    }

    // Perform the non-streaming POST
    try {
      const timeout = DEFAULT_TIMEOUT
      const response = await axios.post(this.baseUrl, body, { headers, timeout })

      this.output.appendLine(`Response status code: ${response.status}`)
      this.output.appendLine("Response headers (received):")
      try {
        Object.entries(response.headers || {}).forEach(([k, v]) => {
          this.output.appendLine(`${k}: ${String(v)}`)
        })
      } catch (err) {
        this.output.appendLine(`Failed enumerating response headers: ${String(err)}`)
      }

      // Extract content and usage safely
      const respBody = response.data
      try {
        this.output.appendLine(`Response body: ${JSON.stringify(respBody)}`)
      } catch (err) {
        this.output.appendLine(`Response body (failed stringify): ${String(err)}`)
      }

      const content =
        respBody?.choices?.[0]?.message?.content ??
        respBody?.choices?.[0]?.text ??
        respBody?.result ??
        respBody?.output?.[0]?.content ??
        ""

      const usage = respBody?.usage ?? respBody?.metrics ?? undefined

      // Yield a text chunk (ApiStream expects chunks; this mirrors openai.ts)
      yield {
        type: "text",
        text: content,
      } as unknown as any // keep minimal cast so compiler accepts shape here (we yield chunk object)

      // If usage info present, yield normalized usage chunk using the same processor shape as openai.ts
      if (usage) {
        yield this.processUsageMetrics(usage, this.getModel().info) as unknown as any
      }

      this.output.appendLine("----- Deepseek Request END -----")
    } catch (err: any) {
      // Detailed error logging
      this.output.appendLine("An error occurred during Deepseek request:")
      if (err.response) {
        this.output.appendLine(`Response status: ${err.response.status}`)
        try {
          this.output.appendLine(`Response data: ${JSON.stringify(err.response.data)}`)
        } catch (e) {
          this.output.appendLine(`Response data (raw): ${String(err.response.data)}`)
        }
        if (err.response.headers) {
          this.output.appendLine("Response headers (error):")
          Object.entries(err.response.headers).forEach(([k, v]) => {
            this.output.appendLine(`${k}: ${String(v)}`)
          })
        }
      } else {
        this.output.appendLine(`Error message: ${String(err.message ?? err)}`)
      }
      this.output.appendLine("----- Deepseek Request END (with error) -----")
      throw err
    }
  }

  /**
   * simple completion helper required by SingleCompletionHandler
   */
  async completePrompt(prompt: string): Promise<string> {
    // Build a minimal request using the same model & api key
    // @ts-ignore
    const deepseekApiKey = (this.options.deepSeekApiKey as string) ?? (this.options.deepseekApiKey as string)
    if (!deepseekApiKey) {
      throw new Error("deepSeekApiKey is missing from ApiHandlerOptions")
    }

    const body = {
      model: this.modelId,
      messages: [{ role: "user", content: prompt }],
    }

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      Authorization: `Bearer ${deepseekApiKey}`,
    }

    try {
      const timeout = DEFAULT_TIMEOUT
      const response = await axios.post(this.baseUrl, body, { headers, timeout })
      const respBody = response.data
      const content =
        respBody?.choices?.[0]?.message?.content ??
        respBody?.choices?.[0]?.text ??
        respBody?.result ??
        respBody?.output?.[0]?.content ??
        ""
      return content
    } catch (err: any) {
      if (err instanceof Error) {
        throw new Error(`Deepseek completion error: ${err.message}`)
      }
      throw err
    }
  }

  /**
   * Mirror openai.ts getModel() shape so other code depending on model params continues to work.
   */
  override getModel() {
    const id = (this.options.apiModelId as string) ?? this.modelId ?? deepSeekDefaultModelId
    const info = deepSeekModels[id as keyof typeof deepSeekModels] ?? (openAiModelInfoSaneDefaults as ModelInfo)
    const params = getModelParams({ format: "openai", modelId: id as string, model: info as any, settings: this.options })
    return { id, info, ...params }
  }

  /**
   * Normalize usage metrics for downstream code
   */
  protected processUsageMetrics(usage: any, _modelInfo?: ModelInfo): ApiStreamUsageChunk {
    return {
      type: "usage",
      inputTokens: usage?.prompt_tokens ?? usage?.input_tokens ?? 0,
      outputTokens: usage?.completion_tokens ?? usage?.output_tokens ?? 0,
      cacheWriteTokens:
        usage?.prompt_tokens_details?.cache_miss_tokens ?? usage?.cache_creation_input_tokens ?? undefined,
      cacheReadTokens: usage?.prompt_tokens_details?.cached_tokens ?? usage?.cache_read_input_tokens ?? undefined,
    }
  }
}
