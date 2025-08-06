import https from 'https';
import fs from 'fs';
import type { ApiHandlerOptions } from "../../shared/api";
import { getModelParams } from "../transform/model-params";
import { deepSeekModels, deepSeekDefaultModelId } from "@roo-code/types";

// Define the required types locally
type ApiRequest = any;
type ApiStreamChunk = any;
type ApiStreamResponse = any;
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

  async createMessage(request: ApiRequest, onProgress: (chunk: ApiStreamChunk) => void): Promise<ApiStreamResponse> {
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

    const messages = request.messages.map((m: any) => ({
      role: m.role,
      content: m.content
    }));

    const data = {
      model,
      messages
    };

    return new Promise((resolve, reject) => {
      const req = https.request(endpoint, {
        method: 'POST',
        headers,
        agent: this.agent
      }, (res: any) => {
        let buffer = '';
        
        res.on('data', (chunk: any) => {
          buffer += chunk.toString();
          
          // Process each complete JSON object in the buffer
          let boundary: number;
          while ((boundary = buffer.indexOf('\n')) !== -1) {
            const line = buffer.substring(0, boundary);
            buffer = buffer.substring(boundary + 1);
            
            if (line.startsWith('data: ')) {
              const jsonStr = line.substring(6);
              
              if (jsonStr === '[DONE]') {
                resolve({});
                return;
              }
              
              try {
                const json = JSON.parse(jsonStr);
                if (json.choices && json.choices.length > 0 && json.choices[0].delta?.content) {
                  onProgress({
                    type: 'content',
                    content: json.choices[0].delta.content
                  });
                }
              } catch (e) {
                console.error('Error parsing JSON:', jsonStr, e);
              }
            }
          }
        });

        res.on('end', () => {
          if (buffer.length > 0) {
            try {
              const json = JSON.parse(buffer);
              if (json.choices && json.choices.length > 0 && json.choices[0].delta?.content) {
                onProgress({
                  type: 'content',
                  content: json.choices[0].delta.content
                });
              }
            } catch (e) {
              console.error('Error parsing final JSON:', buffer, e);
            }
          }
          resolve({});
        });
      });

      req.on('error', (error: any) => {
        reject(error);
      });

      req.write(JSON.stringify(data));
      req.end();
    });
  }

  protected processUsageMetrics(usage: any) {
    return {
      type: "usage",
      inputTokens: usage?.prompt_tokens || 0,
      outputTokens: usage?.completion_tokens || 0
    };
  }
}
