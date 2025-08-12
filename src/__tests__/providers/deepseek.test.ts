import { DeepSeekHandler } from '../../api/providers/deepseek';
import { ApiHandlerOptions } from '../../shared/api';

describe('DeepSeekHandler', () => {
  const mockOptions: ApiHandlerOptions = {
    deepSeekApiKey: 'test-api-key',
    apiModelId: 'deepseek-r1-0528',
    deepSeekBaseUrl: 'https://api.deepseek.com'
  };

  it('should format messages correctly with system message and user messages', () => {
    const handler = new DeepSeekHandler(mockOptions);
    const request = {
      systemMessage: 'System instruction',
      messages: [
        { role: 'user', content: 'First message' },
        { role: 'assistant', content: 'Assistant response' },
        { role: 'user', content: 'Follow-up question' }
      ]
    };

    // @ts-expect-error - accessing private method for testing
    const messages = handler.formatMessages(request);
    
    expect(messages).toEqual([
      { role: 'system', content: 'System instruction' },
      { role: 'user', content: 'First message' },
      { role: 'assistant', content: 'Assistant response' },
      { role: 'user', content: 'Follow-up question' }
    ]);
  });

  it('should use prompt when messages array is empty', () => {
    const handler = new DeepSeekHandler(mockOptions);
    const request = {
      systemMessage: 'System instruction',
      prompt: 'User prompt'
    };

    // @ts-expect-error - accessing private method for testing
    const messages = handler.formatMessages(request);
    
    expect(messages).toEqual([
      { role: 'system', content: 'System instruction' },
      { role: 'user', content: 'User prompt' }
    ]);
  });

  it('should use systemPrompt for backward compatibility', () => {
    const handler = new DeepSeekHandler(mockOptions);
    const request = {
      systemPrompt: 'Legacy system instruction',
      messages: [{ content: 'User message' }]
    };

    // @ts-expect-error - accessing private method for testing
    const messages = handler.formatMessages(request);
    
    expect(messages).toEqual([
      { role: 'system', content: 'Legacy system instruction' },
      { role: 'user', content: 'User message' }
    ]);
  });

  it('should throw error when no user content is provided', () => {
    const handler = new DeepSeekHandler(mockOptions);
    const request = { systemMessage: 'System instruction' };

    expect(() => {
      // @ts-expect-error - accessing private method for testing
      handler.formatMessages(request);
    }).toThrow('No user messages or prompt provided');
  });
});