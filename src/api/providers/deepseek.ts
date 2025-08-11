import * as vscode from "vscode"
import https from "https"
import http from "http"
import fs from "fs"
import type { ApiHandlerOptions } from "../../shared/api"
import { getModelParams } from "../transform/model-params"
import { deepSeekModels, deepSeekDefaultModelId } from "@roo-code/types"

// Define the required types locally
type ApiRequest = any
type ApiStreamChunk = any
type ModelInfoWithParams = any

export class DeepSeekHandler {
  private options: ApiHandlerOptions
  private caBundlePath?: string
  private agent?: https.Agent
  private outputChannel: vscode.OutputChannel

  constructor(options: ApiHandlerOptions) {
    this.options = options
    this.caBundlePath = options.deepSeekCaBundlePath
    this.outputChannel = vscode.window.createOutputChannel("DeepSeek Debug")

    // Set CA bundle as environment variable
    if (this.caBundlePath) {
      try {
        process.env["REQUESTS_CA_BUNDLE"] = this.caBundlePath
        const ca = fs.readFileSync(this.caBundlePath)
        this.agent = new https.Agent({ ca })
      } catch (error) {
        this.outputChannel.appendLine(`Error reading CA bundle: ${error}`)
      }
    }
  }

  getModel(): ModelInfoWithParams {
    const id = this.options.apiModelId ?? deepSeekDefaultModelId
    const info = deepSeekModels[id as keyof typeof deepSeekModels] || deepSeekModels[deepSeekDefaultModelId]
    const params = getModelParams({ format: "openai", modelId: id, model: info, settings: this.options })
    return { id, info, ...params }
  }

  async *createMessage(request: ApiRequest): AsyncGenerator<ApiStreamChunk> {
    const endpoint = this.options.deepSeekBaseUrl ?? "https://api.deepseek.com"
    const token = this.options.deepSeekApiKey
    const model = this.getModel().id

    if (!token) {
      throw new Error("DeepSeek API token is required")
    }

    // Format headers exactly as in Python sample
    const headers = {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${token}`
    }

    // Format messages according to Python sample
    const messages = (request.messages || []).map((m: any) => ({
      role: m.role,
      content: m.content
    }))

    // Add system prompt if needed
    if (request.systemPrompt) {
      messages.unshift({
        role: "system",
        content: request.systemPrompt
      })
    }

    // Create payload matching Python sample
    const data = {
      model,
      messages
    }

    // Parse endpoint URL to determine protocol
    const url = new URL(endpoint)
    const httpModule = url.protocol === 'https:' ? https : http
    
    const response = await new Promise<http.IncomingMessage>((resolve, reject) => {
      const options: https.RequestOptions = {
        method: 'POST',
        headers
      }
      
      // Only add agent for HTTPS connections
      if (url.protocol === 'https:' && this.agent) {
        options.agent = this.agent
      }

      const req = httpModule.request(url, options, resolve)
      req.on('error', reject)
      req.write(JSON.stringify(data))
      req.end()
    })

    // Capture full response
    let buffer = ''
    for await (const chunk of response) {
      buffer += chunk.toString()
    }

    // Log API response for debugging
    this.outputChannel.show(true)
    this.outputChannel.appendLine("--- DeepSeek API Request ---")
    this.outputChannel.appendLine(`Endpoint: ${endpoint}`)
    this.outputChannel.appendLine("Headers:")
    this.outputChannel.appendLine(JSON.stringify(headers, null, 2))
    this.outputChannel.appendLine("Body:")
    this.outputChannel.appendLine(JSON.stringify(data, null, 2))
    
    this.outputChannel.appendLine("--- DeepSeek API Response ---")
    this.outputChannel.appendLine(`Status Code: ${response.statusCode}`)
    this.outputChannel.appendLine("Headers:")
    this.outputChannel.appendLine(JSON.stringify(response.headers, null, 2))
    this.outputChannel.appendLine("Body:")
    this.outputChannel.appendLine(buffer.substring(0, 2000) + (buffer.length > 2000 ? "..." : ""))
    
    try {
      const json = JSON.parse(buffer)
      this.outputChannel.appendLine("Parsed response:")
      this.outputChannel.appendLine(JSON.stringify(json, null, 2))
      
      // Extract content with robust error handling
      if (json.choices && json.choices.length > 0) {
        const choice = json.choices[0]
        if (choice.message && choice.message.content) {
          yield {
            type: 'content',
            content: choice.message.content
          }
          return
        }
      }
      
      // Handle error responses
      if (json.error && json.error.message) {
        throw new Error(`API Error: ${json.error.message}`)
      }
      
      throw new Error("No valid content found in API response")
    } catch (e) {
      this.outputChannel.appendLine(`Response parse error: ${e}`)
      throw new Error(`Failed to parse API response: ${e.message}`)
    }
  }

  protected processUsageMetrics(usage: any) {
    return {
      type: "usage",
      inputTokens: usage?.prompt_tokens || 0,
      outputTokens: usage?.completion_tokens || 0,
    }
  }
}
