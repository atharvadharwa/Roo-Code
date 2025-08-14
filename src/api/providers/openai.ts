// openai.ts

import { Anthropic } from "@anthropic-ai/sdk"
import OpenAI, { AzureOpenAI } from "openai"
import axios from "axios"
import fs from "fs"
import process from "process"
import https from "https"
import * as vscode from "vscode"

import {
	type ModelInfo,
	azureOpenAiDefaultApiVersion,
	openAiModelInfoSaneDefaults,
	DEEP_SEEK_DEFAULT_TEMPERATURE,
	OPENAI_AZURE_AI_INFERENCE_PATH,
} from "@roo-code/types"

import type { ApiHandlerOptions } from "../../shared/api"

import { XmlMatcher } from "../../utils/xml-matcher"

import { convertToOpenAiMessages } from "../transform/openai-format"
import { convertToR1Format } from "../transform/r1-format"
import { convertToSimpleMessages } from "../transform/simple-format"
import { ApiStream, ApiStreamUsageChunk } from "../transform/stream"
import { getModelParams } from "../transform/model-params"

import { DEFAULT_HEADERS } from "./constants"
import { BaseProvider } from "./base-provider"
import type { SingleCompletionHandler, ApiHandlerCreateMessageMetadata } from "../index"

// Refactored to respect child-class headers (e.g., DeepSeek) and avoid SDK overriding Authorization.
// If getHeaders() provides an Authorization header, we pass apiKey: "" to the OpenAI SDK so it won't inject its own.
export class OpenAiHandler extends BaseProvider implements SingleCompletionHandler {

  protected options: ApiHandlerOptions
  protected client: OpenAI
  protected outputChannel: vscode.OutputChannel

  logDebug(message: string) {
    if (this.outputChannel) {
      this.outputChannel.appendLine(message)
    }
  }

  logDebugMessage(message: string) {
    if (this.outputChannel) {
      this.outputChannel.appendLine(message)
    }
  }

  constructor(options: ApiHandlerOptions) {
    super()
    this.options = options
    this.outputChannel = vscode.window.createOutputChannel("DeepSeek Debug")

    // Determine base URL & apiKey defaults
    let baseURL = options.openAiBaseUrl ?? "https://api.openai.com/v1"
    let apiKey = options.openAiApiKey ?? "not-provided"

    // DeepSeek-specific override (base url, key, CA bundle)
    let httpsAgent: https.Agent | undefined = undefined
    if (options.deepSeekBaseUrl || options.deepSeekApiKey || options.deepSeekCaBundlePath) {
      baseURL = options.deepSeekBaseUrl ?? baseURL
      apiKey = options.deepSeekApiKey ?? apiKey

      if (options.deepSeekCaBundlePath) {
        try {
          this.outputChannel.appendLine(`[Debug] Provided DeepSeek CA bundle path: ${options.deepSeekCaBundlePath}`)
          if (fs.existsSync(options.deepSeekCaBundlePath)) {
            const caContent = fs.readFileSync(options.deepSeekCaBundlePath)
            this.outputChannel.appendLine(`[Debug] CA bundle content (first 100 chars): ${caContent.toString().substring(0, 100)}`)
            httpsAgent = new https.Agent({
              ca: caContent,
              minVersion: "TLSv1.2",
              maxVersion: "TLSv1.3",
            })
            this.outputChannel.appendLine(`[Debug] Using CA bundle for HTTPS requests: ${options.deepSeekCaBundlePath}`)
          } else {
            this.outputChannel.appendLine(`[Debug] CA bundle path does not exist: ${options.deepSeekCaBundlePath}`)
          }
        } catch (err) {
          console.error("Error setting DeepSeek CA bundle:", err)
          this.outputChannel.appendLine(`Error reading CA bundle: ${err}`)
        }
      } else {
        this.outputChannel.appendLine(`[Debug] No CA bundle path provided for DeepSeek`)
      }
    }

    const isAzureAiInference = this._isAzureAiInference(baseURL)
    const urlHost = this._getUrlHost(baseURL)
    const isAzureOpenAi = urlHost === "azure.com" || urlHost.endsWith(".azure.com") || options.openAiUseAzure

    // Use headers from getHeaders(); if Authorization is present, do NOT pass apiKey to SDK.
    const realHeaders = this.getHeaders()
    const hasAuthHeader = !!Object.keys(realHeaders).find(
      (k) => k.toLowerCase() === "authorization",
    )
    const sdkApiKey = hasAuthHeader ? "" : apiKey

    // Construct client with defaultHeaders so our headers are actually sent.
    if (isAzureAiInference) {
      this.client = new OpenAI({
        baseURL,
        apiKey: sdkApiKey,
        defaultHeaders: realHeaders,
        defaultQuery: { "api-version": options.azureApiVersion || "2024-05-01-preview" },
        ...(httpsAgent ? { httpAgent: httpsAgent, httpsAgent } : {}),
      })
    } else if (isAzureOpenAi) {
      this.client = new AzureOpenAI({
        baseURL,
        apiKey: sdkApiKey,
        apiVersion: options.azureApiVersion || azureOpenAiDefaultApiVersion,
        defaultHeaders: realHeaders,
        ...(httpsAgent ? { httpAgent: httpsAgent, httpsAgent } : {}),
      })
    } else {
      this.client = new OpenAI({
        baseURL,
        apiKey: sdkApiKey,
        defaultHeaders: realHeaders,
        ...(httpsAgent ? { httpAgent: httpsAgent, httpsAgent } : {}),
      })
    }

    // Log effective client config summary (redacted)
    const redactedHeaders = { ...realHeaders }
    if (redactedHeaders.Authorization) {
      redactedHeaders.Authorization = "***" + redactedHeaders.Authorization.slice(-6)
    }
    this.outputChannel.appendLine(`[OpenAI] Client initialized with:`)
    this.outputChannel.appendLine(`- baseURL: ${baseURL}`)
    this.outputChannel.appendLine(`- apiKey supplied to SDK: ${sdkApiKey ? "***" + sdkApiKey.slice(-6) : "(empty — using custom Authorization header)"}`)
    this.outputChannel.appendLine(`- defaultHeaders: ${JSON.stringify(redactedHeaders)}`)
  }

  override async *createMessage(
    systemPrompt: string,
    messages: Anthropic.Messages.MessageParam[],
    metadata?: ApiHandlerCreateMessageMetadata,
  ): ApiStream {
    const { info: modelInfo, reasoning } = this.getModel()
    const modelUrl = this.options.openAiBaseUrl ?? ""
    const modelId = this.options.openAiModelId ?? ""
    const enabledR1Format = this.options.openAiR1FormatEnabled ?? false
    const enabledLegacyFormat = this.options.openAiLegacyFormat ?? false
    const isAzureAiInference = this._isAzureAiInference(modelUrl)
    const deepseekReasoner = true
    const ark = modelUrl.includes(".volces.com")

    // Log request details
    this.outputChannel.appendLine("[OpenAI] Request:")
    this.outputChannel.appendLine("URL: " + modelUrl)
    this.outputChannel.appendLine("Model: " + modelId)
    const realHeaders = this.getHeaders()
    const redacted = { ...realHeaders }
    if (redacted.Authorization) redacted.Authorization = "***" + redacted.Authorization.slice(-6)
    this.outputChannel.appendLine("Headers: " + JSON.stringify(redacted))
    if (modelId.includes("o1") || modelId.includes("o3") || modelId.includes("o4")) {
      yield* this.handleO3FamilyMessage(modelId, systemPrompt, messages)
      return
    }

    // Disable streaming for DeepSeek (and any target that doesn't support it)
    const useStreaming = false

    if (useStreaming) {
      let systemMessage: OpenAI.Chat.ChatCompletionSystemMessageParam = {
        role: "system",
        content: systemPrompt,
      }

      let convertedMessages

      
      convertedMessages = convertToR1Format([{ role: "user", content: systemPrompt }, ...messages])
      
      this.outputChannel.appendLine("Payload: " + JSON.stringify({ convertedMessages }))

      

      const isGrokXAI = this._isGrokXAI(this.options.openAiBaseUrl)

      const requestOptions: OpenAI.Chat.Completions.ChatCompletionCreateParamsStreaming = {
        model: modelId,
        temperature:
          this.options.modelTemperature ?? (deepseekReasoner ? DEEP_SEEK_DEFAULT_TEMPERATURE : 0),
        messages: convertedMessages,
        stream: true as const,
        ...(isGrokXAI ? {} : { stream_options: { include_usage: true } }),
        ...(reasoning && reasoning),
      }

      // Add max tokens if needed
      this.addMaxTokensIfNeeded(requestOptions, modelInfo)
    } else {
      // Non-streaming path (used for DeepSeek)
      const systemMessage: OpenAI.Chat.ChatCompletionUserMessageParam = {
        role: "user",
        content: systemPrompt,
      }

      const requestOptions: OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming = {
        model: modelId,
        messages: deepseekReasoner
          ? convertToR1Format([{ role: "user", content: systemPrompt }, ...messages])
          : enabledLegacyFormat
            ? [systemMessage, ...convertToSimpleMessages(messages)]
            : [systemMessage, ...convertToOpenAiMessages(messages)],
      }

      this.outputChannel.appendLine("Payload2: " + JSON.stringify({ requestOptions }))


      // Add max_tokens if needed
      this.addMaxTokensIfNeeded(requestOptions, modelInfo)

      const response = await this.client.chat.completions.create(
        requestOptions,
        this._isAzureAiInference(modelUrl) ? { path: OPENAI_AZURE_AI_INFERENCE_PATH } : {},
      )

      this.outputChannel.appendLine("[OpenAI] Full API response (non-stream): " + JSON.stringify(response))
      yield {
        type: "text",
        text: response.choices[0]?.message.content || "",
      }

      yield this.processUsageMetrics(response.usage, modelInfo)
    }
  }

  protected processUsageMetrics(usage: any, _modelInfo?: ModelInfo): ApiStreamUsageChunk {
    return {
      type: "usage",
      inputTokens: usage?.prompt_tokens || 0,
      outputTokens: usage?.completion_tokens || 0,
      cacheWriteTokens: usage?.cache_creation_input_tokens || undefined,
      cacheReadTokens: usage?.cache_read_input_tokens || undefined,
    }
  }

  // Default headers: parent class merges defaults; child (DeepSeekHandler) may override to return ONLY what it needs.
  getHeaders() {
    return {
      ...DEFAULT_HEADERS,
      ...(this.options.openAiHeaders || {}),
      ...(this.options.openAiApiKey
        ? { Authorization: `Bearer ${this.options.openAiApiKey}` }
        : {}),
    }
  }

  override getModel() {
    // If DeepSeek integration is present, default to deepseek-reasoner unless explicitly set
    let id = this.options.openAiModelId ?? ""
    if (this.options.deepSeekBaseUrl || this.options.deepSeekApiKey || this.options.deepSeekCaBundlePath) {
      id = this.options.apiModelId ?? "deepseek-reasoner"
    }
    const info = this.options.openAiCustomModelInfo ?? openAiModelInfoSaneDefaults
    const params = getModelParams({ format: "openai", modelId: id, model: info, settings: this.options })
    return { id, info, ...params }
  }

  async completePrompt(prompt: string): Promise<string> {
    try {
      const isAzureAiInference = this._isAzureAiInference(this.options.openAiBaseUrl)
      const model = this.getModel()
      const modelInfo = model.info

      // Log request details
      this.outputChannel.appendLine("[OpenAI] completePrompt Request:")
      this.outputChannel.appendLine("URL: " + this.options.openAiBaseUrl)
      this.outputChannel.appendLine("Model: " + model.id)
      const red = { ...(this.options.openAiHeaders || {}) }
      if (red.Authorization) red.Authorization = "***" + red.Authorization.slice(-6)
      this.outputChannel.appendLine("Headers: " + JSON.stringify(red))
      this.outputChannel.appendLine("Payload: " + JSON.stringify({ prompt }))

      const requestOptions: OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming = {
        model: model.id,
        messages: [{ role: "user", content: prompt }],
      }

      // Add max_tokens if needed
      this.addMaxTokensIfNeeded(requestOptions, modelInfo)

      const response = await this.client.chat.completions.create(
        requestOptions,
        isAzureAiInference ? { path: OPENAI_AZURE_AI_INFERENCE_PATH } : {},
      )

      this.outputChannel.appendLine("[OpenAI] completePrompt Response: " + JSON.stringify(response))
      return response.choices[0]?.message.content || ""
    } catch (error) {
      if (error instanceof Error) {
        throw new Error(`OpenAI completion error: ${error.message}`)
      }
      throw error
    }
  }

  private async *handleO3FamilyMessage(
    modelId: string,
    systemPrompt: string,
    messages: Anthropic.Messages.MessageParam[],
  ): ApiStream {
    const modelInfo = this.getModel().info
    const methodIsAzureAiInference = this._isAzureAiInference(this.options.openAiBaseUrl)

    // Log request details
    this.outputChannel.appendLine("[OpenAI] O3Family Request:")
    this.outputChannel.appendLine("URL: " + this.options.openAiBaseUrl)
    this.outputChannel.appendLine("Model: " + modelId)
    const red = { ...(this.options.openAiHeaders || {}) }
    if (red.Authorization) red.Authorization = "***" + red.Authorization.slice(-6)
    this.outputChannel.appendLine("Headers: " + JSON.stringify(red))
    this.outputChannel.appendLine("Payload: " + JSON.stringify({ systemPrompt, messages }))

    if (this.options.openAiStreamingEnabled ?? true) {
      const isGrokXAI = this._isGrokXAI(this.options.openAiBaseUrl)

      const requestOptions: OpenAI.Chat.Completions.ChatCompletionCreateParamsStreaming = {
        model: modelId,
        messages: [
          {
            role: "developer",
            content: `Formatting re-enabled\n${systemPrompt}`,
          },
          ...convertToOpenAiMessages(messages),
        ],
        stream: true,
        ...(isGrokXAI ? {} : { stream_options: { include_usage: true } }),
        reasoning_effort: (modelInfo as any).reasoningEffort,
        temperature: undefined,
      }

      // O3 family: use max_completion_tokens (not deprecated max_tokens)
      this.addMaxTokensIfNeeded(requestOptions, modelInfo)

      const stream = await this.client.chat.completions.create(
        requestOptions,
        methodIsAzureAiInference ? { path: OPENAI_AZURE_AI_INFERENCE_PATH } : {},
      )

      this.outputChannel.appendLine("[OpenAI] O3Family Response stream started.")
      yield* this.handleStreamResponse(stream)
    } else {
      const requestOptions: OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming = {
        model: modelId,
        messages: [
          {
            role: "developer",
            content: `Formatting re-enabled\n${systemPrompt}`,
          },
          ...convertToOpenAiMessages(messages),
        ],
        reasoning_effort: (modelInfo as any).reasoningEffort,
        temperature: undefined,
      }

      // O3 family: use max_completion_tokens
      this.addMaxTokensIfNeeded(requestOptions, modelInfo)

      const response = await this.client.chat.completions.create(
        requestOptions,
        methodIsAzureAiInference ? { path: OPENAI_AZURE_AI_INFERENCE_PATH } : {},
      )

      this.outputChannel.appendLine("[OpenAI] O3Family Response: " + JSON.stringify(response))
      yield {
        type: "text",
        text: response.choices[0]?.message.content || "",
      }
      yield this.processUsageMetrics(response.usage)
    }
  }

  private async *handleStreamResponse(stream: AsyncIterable<OpenAI.Chat.Completions.ChatCompletionChunk>): ApiStream {
    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta
      if (delta?.content) {
        yield {
          type: "text",
          text: delta.content,
        }
      }

      if (chunk.usage) {
        yield {
          type: "usage",
          inputTokens: chunk.usage.prompt_tokens || 0,
          outputTokens: chunk.usage.completion_tokens || 0,
        }
      }
    }
  }

  private _getUrlHost(baseUrl?: string): string {
    try {
      return new URL(baseUrl ?? "").host
    } catch {
      return ""
    }
  }

  private _isGrokXAI(baseUrl?: string): boolean {
    const urlHost = this._getUrlHost(baseUrl)
    return urlHost.includes("x.ai")
  }

  private _isAzureAiInference(baseUrl?: string): boolean {
    const urlHost = this._getUrlHost(baseUrl)
    return urlHost.endsWith(".services.ai.azure.com")
  }

  /**
   * Adds max_completion_tokens to the request body if needed based on provider configuration
   * Note: max_tokens is deprecated in favor of max_completion_tokens as per OpenAI documentation
   * O3 family models handle max_tokens separately in handleO3FamilyMessage
   */
  private addMaxTokensIfNeeded(
    requestOptions:
      | OpenAI.Chat.Completions.ChatCompletionCreateParamsStreaming
      | OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming,
    modelInfo: ModelInfo,
  ): void {
    if (this.options.includeMaxTokens === true) {
      requestOptions.max_completion_tokens = this.options.modelMaxTokens || modelInfo.maxTokens
    }
  }
}

export async function getOpenAiModels(
  baseUrl?: string,
  apiKey?: string,
  openAiHeaders?: Record<string, string>,
  deepSeekCaBundlePath?: string,
) {
  try {
    if (!baseUrl) {
      return []
    }

    const trimmedBaseUrl = baseUrl.trim()
    if (!URL.canParse(trimmedBaseUrl)) {
      return []
    }

    const config: Record<string, any> = {}
    const headers: Record<string, string> = {
      ...DEFAULT_HEADERS,
      ...(openAiHeaders || {}),
    }

    if (apiKey) {
      headers["Authorization"] = `Bearer ${apiKey}`
    }

    if (Object.keys(headers).length > 0) {
      config["headers"] = headers
    }

    if (deepSeekCaBundlePath && fs.existsSync(deepSeekCaBundlePath)) {
      config["httpsAgent"] = new https.Agent({ ca: fs.readFileSync(deepSeekCaBundlePath) })
    }

    const outputChannel = vscode.window.createOutputChannel("DeepSeek Debug")
    outputChannel.appendLine("[OpenAI] getOpenAiModels Request:")
    outputChannel.appendLine(`URL: ${trimmedBaseUrl}/models`)
    const red = { ...headers }
    if (red.Authorization) red.Authorization = "***" + red.Authorization.slice(-6)
    outputChannel.appendLine("Headers: " + JSON.stringify(red))

    const response = await axios.get(`${trimmedBaseUrl}/models`, config)
    const modelsArray = response.data?.data?.map((model: any) => model.id) || []
    outputChannel.appendLine("[OpenAI] getOpenAiModels Response: " + JSON.stringify(response.data))
    return [...new Set<string>(modelsArray)]
  } catch {
    return []
  }
}
