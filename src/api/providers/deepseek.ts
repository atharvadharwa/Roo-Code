import https from 'https';
import http from 'http';
import fs from 'fs';
import type { ApiHandlerOptions } from "../../shared/api";
import { getModelParams } from "../transform/model-params";
import { deepSeekModels, deepSeekDefaultModelId } from "@roo-code/types";

// Define the required types locally
type ApiRequest = any;
type ApiStreamChunk = any;
type ModelInfoWithParams = any;

export class DeepSeekHandler {
  private options: ApiHandlerOptions;
  private caBundlePath?: string;
  private agent?: https.Agent;

  constructor(options: ApiHandlerOptions) {
    this.options = options;
    this.caBundlePath = options.deepSeekCaBundlePath;

    if (this.caBundlePath) {
      try {
        const ca = fs.readFileSync(this.caBundlePath);
        this.agent = new https.Agent({ ca });
      } catch (error) {
        console.error(`Error reading CA bundle: ${error}`);
      }
    }
  }

  getModel(): ModelInfoWithParams {
    const id = this.options.apiModelId ?? deepSeekDefaultModelId;
    const info = deepSeekModels[id as keyof typeof deepSeekModels] || deepSeekModels[deepSeekDefaultModelId];
    const params = getModelParams({ format: "openai", modelId: id, model: info, settings: this.options });
    return { id, info, ...params };
  }

  async *createMessage(request: ApiRequest): AsyncGenerator<ApiStreamChunk> {
    const endpoint = this.options.deepSeekBaseUrl ?? "https://api.deepseek.com";
    const token = this.options.deepSeekApiKey;
    const model = this.getModel().id;

    if (!token) {
      throw new Error("DeepSeek API token is required");
    }

    const headers = {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${token}`
    };

    const messages = (request.messages || []).map((m: any) => ({
      role: m.role,
      content: m.content
    }));

    const data = {
      model,
      messages
    };

    // Parse endpoint URL to determine protocol
    const url = new URL(endpoint);
    const httpModule = url.protocol === 'https:' ? https : http;
    
    const response = await new Promise<http.IncomingMessage>((resolve, reject) => {
      const options: https.RequestOptions = {
        method: 'POST',
        headers
      };
      
      // Only add agent for HTTPS connections
      if (url.protocol === 'https:' && this.agent) {
        options.agent = this.agent;
      }

      const req = httpModule.request(url, options, resolve);
      req.on('error', reject);
      req.write(JSON.stringify(data));
      req.end();
    });

    // Capture full response for debugging
    let buffer = '';
    for await (const chunk of response) {
      buffer += chunk.toString();
    }

    // Log full API response details for debugging
    console.error("DeepSeek API Response Details:");
    console.error("Status Code:", response.statusCode);
    console.error("Headers:", JSON.stringify(response.headers, null, 2));
    console.error("Full Body:", buffer);
    
    // First try parsing as JSON (non-streaming response)
    try {
      const json = JSON.parse(buffer);
      console.error("Parsed non-streaming response:", json);
      
      if (json.choices && json.choices.length > 0) {
        // Use safe navigation to avoid errors
        const message = json.choices[0].message || {};
        const content = message.content;
        
        if (content) {
          yield {
            type: 'content',
            content: content
          };
          return;
        } else {
          console.error("No content found in non-streaming response choices");
        }
      } else {
        console.error("No choices found in non-streaming response");
      }
    } catch (e) {
      console.error("Non-streaming parse error:", e);
    }
    
    // Process as streaming response
    let lines = buffer.split('\n');
    let contentFound = false;
    
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const jsonStr = line.substring(6);
        
        if (jsonStr === '[DONE]') {
          console.error("Received [DONE] marker");
          if (!contentFound) {
            console.error("No content found before [DONE] marker");
          }
          return;
        }
        
        try {
          const json = JSON.parse(jsonStr);
          console.error("Parsed streaming chunk:", json);
          
          if (json.choices && json.choices.length > 0) {
            // Try all possible content locations
            const content =
              json.choices[0].delta?.content ||
              json.choices[0].message?.content ||
              json.choices[0].text;
            
            if (content) {
              contentFound = true;
              yield {
                type: 'content',
                content: content
              };
            } else {
              console.error("No content in choice:", json.choices[0]);
            }
          }
        } catch (e) {
          console.error('Error parsing JSON chunk:', jsonStr, e);
        }
      }
    }

    if (!contentFound) {
      console.error("No content found in streaming response");
      throw new Error("No assistant messages found in API response");
    }
  }

  protected processUsageMetrics(usage: any) {
    return {
      type: "usage",
      inputTokens: usage?.prompt_tokens || 0,
      outputTokens: usage?.completion_tokens || 0
    };
  }
}
