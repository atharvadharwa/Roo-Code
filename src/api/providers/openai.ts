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

// TODO: Rename this to OpenAICompatibleHandler. Also, I think the
// `OpenAINativeHandler` can subclass from this, since it's obviously
// compatible with the OpenAI API. We can also rename it to `OpenAIHandler`.
export class OpenAiHandler extends BaseProvider implements SingleCompletionHandler {

  protected options: ApiHandlerOptions;
  private client: OpenAI;
  protected outputChannel: vscode.OutputChannel

  logDebug(message: string) {
    if (this.outputChannel) {
      this.outputChannel.appendLine(message);
    }
  }

  logDebugMessage(message: string) {
    if (this.outputChannel) {
      this.outputChannel.appendLine(message);
    }
  }

  constructor(options: ApiHandlerOptions) {
 super();
 this.options = options;

 // Initialize headers
 let headers = {
   ...DEFAULT_HEADERS,
   ...(options.openAiHeaders || {}),
 };
 this.outputChannel = vscode.window.createOutputChannel("DeepSeek Debug")

 // Set base URL and API key
 let baseURL = options.openAiBaseUrl ?? "https://api.openai.com/v1";
 let apiKey = options.openAiApiKey ?? "not-provided";

 // Handle DeepSeek-specific settings
 let httpsAgent: https.Agent | undefined = undefined;
 if (options.deepSeekBaseUrl || options.deepSeekApiKey || options.deepSeekCaBundlePath) {
	 baseURL = options.deepSeekBaseUrl ?? "https://api.deepseek.com";
	 apiKey = options.deepSeekApiKey ?? "not-provided";
	 // Set CA bundle for requests (Node.js way)
	 if (options.deepSeekCaBundlePath) {
		 try {
			 this.outputChannel.appendLine(`[Debug] Provided DeepSeek CA bundle path: ${options.deepSeekCaBundlePath}`);
			 if (fs.existsSync(options.deepSeekCaBundlePath)) {
				 httpsAgent = new https.Agent({ ca: fs.readFileSync(options.deepSeekCaBundlePath) });
				 this.outputChannel.appendLine(`[Debug] Using CA bundle for HTTPS requests: ${options.deepSeekCaBundlePath}`);
			 } else {
				 this.outputChannel.appendLine(`[Debug] CA bundle path does not exist: ${options.deepSeekCaBundlePath}`);
			 }
		 } catch (err) {
			 console.error("Error setting DeepSeek CA bundle:", err);
			 this.outputChannel.appendLine(`Error reading CA bundle: ${err}`)
		 }
	 }
 }

	const isAzureAiInference = this._isAzureAiInference(baseURL);
	const urlHost = this._getUrlHost(baseURL);
	const isAzureOpenAi = urlHost === "azure.com" || urlHost.endsWith(".azure.com") || options.openAiUseAzure;

		if (isAzureAiInference) {
			// Azure AI Inference Service (e.g., for DeepSeek) uses a different path structure
			this.client = new OpenAI({
				baseURL,
				apiKey,
				defaultHeaders: headers,
				defaultQuery: { "api-version": options.azureApiVersion || "2024-05-01-preview" },
				...(httpsAgent ? { httpAgent: httpsAgent, httpsAgent } : {}),
			});
		} else if (isAzureOpenAi) {
			// Azure API shape slightly differs from the core API shape:
			// https://github.com/openai/openai-node?tab=readme-ov-file#microsoft-azure-openai
			this.client = new AzureOpenAI({
				baseURL,
				apiKey,
				apiVersion: options.azureApiVersion || azureOpenAiDefaultApiVersion,
				defaultHeaders: headers,
				...(httpsAgent ? { httpAgent: httpsAgent, httpsAgent } : {}),
			});
		} else {
			this.client = new OpenAI({
				baseURL,
				apiKey,
				defaultHeaders: headers,
				...(httpsAgent ? { httpAgent: httpsAgent, httpsAgent } : {}),
			});
		}
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
		const deepseekReasoner = modelId.includes("deepseek-reasoner") ?? true
		const ark = modelUrl.includes(".volces.com")

		// Log request details
		this.outputChannel.appendLine('[OpenAI] Request:')
		this.outputChannel.appendLine('URL: ' + modelUrl)
		this.outputChannel.appendLine('Model: ' + modelId)
		// Log the real headers used in the request
		let realHeaders = {
		  ...DEFAULT_HEADERS,
		  ...(this.options.openAiHeaders || {}),
		};
		if (this.options.deepSeekBaseUrl || this.options.deepSeekApiKey || this.options.deepSeekCaBundlePath) {
		  realHeaders = {
		    ...realHeaders,
		    "Content-Type": "application/json",
		    "Authorization": `Bearer ${this.options.deepSeekApiKey ?? "not-provided"}`,
		  };
		}
		this.outputChannel.appendLine('Headers: ' + JSON.stringify(realHeaders))
		this.outputChannel.appendLine('Payload: ' + JSON.stringify({ systemPrompt, messages, metadata }))
		if (modelId.includes("o1") || modelId.includes("o3") || modelId.includes("o4")) {
			yield* this.handleO3FamilyMessage(modelId, systemPrompt, messages)
			return
		}

		if (this.options.openAiStreamingEnabled ?? true) {
			let systemMessage: OpenAI.Chat.ChatCompletionSystemMessageParam = {
				role: "system",
				content: systemPrompt,
			}

			let convertedMessages

			if (deepseekReasoner) {
				convertedMessages = convertToR1Format([{ role: "user", content: systemPrompt }, ...messages])
			} else if (ark || enabledLegacyFormat) {
				convertedMessages = [systemMessage, ...convertToSimpleMessages(messages)]
			} else {
				if (modelInfo.supportsPromptCache) {
					systemMessage = {
						role: "system",
						content: [
							{
								type: "text",
								text: systemPrompt,
								// @ts-ignore-next-line
								cache_control: { type: "ephemeral" },
							},
						],
					}
				}

				convertedMessages = [systemMessage, ...convertToOpenAiMessages(messages)]

				if (modelInfo.supportsPromptCache) {
					// Note: the following logic is copied from openrouter:
					// Add cache_control to the last two user messages
					// (note: this works because we only ever add one user message at a time, but if we added multiple we'd need to mark the user message before the last assistant message)
					const lastTwoUserMessages = convertedMessages.filter((msg) => msg.role === "user").slice(-2)

					lastTwoUserMessages.forEach((msg) => {
						if (typeof msg.content === "string") {
							msg.content = [{ type: "text", text: msg.content }]
						}

						if (Array.isArray(msg.content)) {
							// NOTE: this is fine since env details will always be added at the end. but if it weren't there, and the user added a image_url type message, it would pop a text part before it and then move it after to the end.
							let lastTextPart = msg.content.filter((part) => part.type === "text").pop()

							if (!lastTextPart) {
								lastTextPart = { type: "text", text: "..." }
								msg.content.push(lastTextPart)
							}

							// @ts-ignore-next-line
							lastTextPart["cache_control"] = { type: "ephemeral" }
						}
					})
				}
			}

			const isGrokXAI = this._isGrokXAI(this.options.openAiBaseUrl)

			const requestOptions: OpenAI.Chat.Completions.ChatCompletionCreateParamsStreaming = {
				model: modelId,
				temperature: this.options.modelTemperature ?? (deepseekReasoner ? DEEP_SEEK_DEFAULT_TEMPERATURE : 0),
				messages: convertedMessages,
				stream: true as const,
				...(isGrokXAI ? {} : { stream_options: { include_usage: true } }),
				...(reasoning && reasoning),
			}

			// Add max_tokens if needed
			this.addMaxTokensIfNeeded(requestOptions, modelInfo)

			const stream = await this.client.chat.completions.create(
				requestOptions,
				isAzureAiInference ? { path: OPENAI_AZURE_AI_INFERENCE_PATH } : {},
			)

			// Log response stream start
			this.outputChannel.appendLine("[OpenAI] Response stream started.")
			const matcher = new XmlMatcher(
				"think",
				(chunk) =>
					({
						type: chunk.matched ? "reasoning" : "text",
						text: chunk.data,
					}) as const,
			)

			let lastUsage
			let allChunks: any[] = []

			for await (const chunk of stream) {
				allChunks.push(chunk)
				const delta = chunk.choices[0]?.delta ?? {}

				// Log each response chunk
				this.outputChannel.appendLine("[OpenAI] Response chunk: " + JSON.stringify(chunk))
				if (delta.content) {
					for (const chunk of matcher.update(delta.content)) {
						yield chunk
					}
				}

				if ("reasoning_content" in delta && delta.reasoning_content) {
					yield {
						type: "reasoning",
						text: (delta.reasoning_content as string | undefined) || "",
					}
				}
				if (chunk.usage) {
					lastUsage = chunk.usage
				}
			}

			// Print the full response stream as JSON
			this.outputChannel.appendLine("[OpenAI] Full API response (stream): " + JSON.stringify(allChunks))

			for (const chunk of matcher.final()) {
				yield chunk
			}

			if (lastUsage) {
				yield this.processUsageMetrics(lastUsage, modelInfo)
			}
		} else {
			// o1 for instance doesnt support streaming, non-1 temp, or system prompt
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

			// Add max_tokens if needed
			this.addMaxTokensIfNeeded(requestOptions, modelInfo)

			const response = await this.client.chat.completions.create(
				requestOptions,
				this._isAzureAiInference(modelUrl) ? { path: OPENAI_AZURE_AI_INFERENCE_PATH } : {},
			)

			// Log response body
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

	getHeaders() {
return {
	 ...DEFAULT_HEADERS,
	 ...(this.options.openAiHeaders || {}),
};
	}

	override getModel() {
// If DeepSeek integration is selected, default to deepseek-reasoner
let id = this.options.openAiModelId ?? "";
if (this.options.deepSeekBaseUrl || this.options.deepSeekApiKey || this.options.deepSeekCaBundlePath) {
	 id = this.options.apiModelId ?? "deepseek-reasoner";
}
const info = this.options.openAiCustomModelInfo ?? openAiModelInfoSaneDefaults;
const params = getModelParams({ format: "openai", modelId: id, model: info, settings: this.options });
return { id, info, ...params };
	}

	async completePrompt(prompt: string): Promise<string> {
		try {
			const isAzureAiInference = this._isAzureAiInference(this.options.openAiBaseUrl)
			const model = this.getModel()
			const modelInfo = model.info

			// Log request details
			this.outputChannel.appendLine('[OpenAI] completePrompt Request:')
			this.outputChannel.appendLine('URL: ' + this.options.openAiBaseUrl)
			this.outputChannel.appendLine('Model: ' + model.id)
			this.outputChannel.appendLine('Headers: ' + JSON.stringify(this.options.openAiHeaders || {}))
			this.outputChannel.appendLine('Payload: ' + JSON.stringify({ prompt }))
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

			// Log response body
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
		this.outputChannel.appendLine('[OpenAI] O3Family Request:')
		this.outputChannel.appendLine('URL: ' + this.options.openAiBaseUrl)
		this.outputChannel.appendLine('Model: ' + modelId)
		this.outputChannel.appendLine('Headers: ' + JSON.stringify(this.options.openAiHeaders || {}))
		this.outputChannel.appendLine('Payload: ' + JSON.stringify({ systemPrompt, messages }))
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
				reasoning_effort: modelInfo.reasoningEffort,
				temperature: undefined,
			}

			// O3 family models do not support the deprecated max_tokens parameter
			// but they do support max_completion_tokens (the modern OpenAI parameter)
			// This allows O3 models to limit response length when includeMaxTokens is enabled
			this.addMaxTokensIfNeeded(requestOptions, modelInfo)

			const stream = await this.client.chat.completions.create(
				requestOptions,
				methodIsAzureAiInference ? { path: OPENAI_AZURE_AI_INFERENCE_PATH } : {},
			)

			// Log response stream start
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
				reasoning_effort: modelInfo.reasoningEffort,
				temperature: undefined,
			}

			// O3 family models do not support the deprecated max_tokens parameter
			// but they do support max_completion_tokens (the modern OpenAI parameter)
			// This allows O3 models to limit response length when includeMaxTokens is enabled
			this.addMaxTokensIfNeeded(requestOptions, modelInfo)

			const response = await this.client.chat.completions.create(
				requestOptions,
				methodIsAzureAiInference ? { path: OPENAI_AZURE_AI_INFERENCE_PATH } : {},
			)

			// Log response body
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
		} catch (error) {
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
		// Only add max_completion_tokens if includeMaxTokens is true
		if (this.options.includeMaxTokens === true) {
			// Use user-configured modelMaxTokens if available, otherwise fall back to model's default maxTokens
			// Using max_completion_tokens as max_tokens is deprecated
			requestOptions.max_completion_tokens = this.options.modelMaxTokens || modelInfo.maxTokens
		}
	}
}

export async function getOpenAiModels(baseUrl?: string, apiKey?: string, openAiHeaders?: Record<string, string>, deepSeekCaBundlePath?: string) {
	try {
		if (!baseUrl) {
			return []
		}

		// Trim whitespace from baseUrl to handle cases where users accidentally include spaces
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

		// If a CA bundle is provided, use it for axios requests
		if (deepSeekCaBundlePath && fs.existsSync(deepSeekCaBundlePath)) {
			config["httpsAgent"] = new https.Agent({ ca: fs.readFileSync(deepSeekCaBundlePath) });
		}

		// Log request details
		const outputChannel = vscode.window.createOutputChannel("DeepSeek Debug")
		outputChannel.appendLine("[OpenAI] getOpenAiModels Request:")
		outputChannel.appendLine(`URL: ${trimmedBaseUrl}/models`)
		outputChannel.appendLine('Headers: ' + JSON.stringify(headers))
		const response = await axios.get(`${trimmedBaseUrl}/models`, config)
		const modelsArray = response.data?.data?.map((model: any) => model.id) || []
		// Log response body
		outputChannel.appendLine("[OpenAI] getOpenAiModels Response: " + JSON.stringify(response.data))
		return [...new Set<string>(modelsArray)]
	} catch (error) {
		return []
	}
}
