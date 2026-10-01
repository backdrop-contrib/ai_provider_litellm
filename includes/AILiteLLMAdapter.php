<?php

/**
 * @file
 * LiteLLM adapter for accessing 100+ LLM providers.
 *
 * LiteLLM provides a unified API-compatible interface to call models from
 * OpenAI, Anthropic, Azure, Google, AWS Bedrock, Cohere, and 100+ more providers.
 * This adapter connects to a LiteLLM proxy server or directly to compatible endpoints.
 *
 * @see https://docs.litellm.ai/
 */

class AILiteLLMAdapter extends AIAdapterBase {

  use AICompatibleTrait;

  /** @var string Base URL including /v1 suffix. */
  protected $baseUrl;

  /** @var array|null Model metadata, indexed by public proxy alias. */
  protected $modelMetadata;

  /** @var array|null Catalog reused across capability lookups. */
  protected $models;

  /**
   * Constructor.
   *
   * @param string $api_key
   *   API key for authentication (LiteLLM master key or provider key).
   * @param AIApi|null $api
   *   Optional parent API wrapper.
   */
  public function __construct($api_key, ?AIApi $api = NULL) {
    parent::__construct($api_key, $api);

    $base = config_get('ai_provider_litellm.settings', 'base_url') ?: 'http://localhost:4000';
    $base = rtrim($base, '/');
    if (strpos($base, '/v1') === FALSE) {
      $base .= '/v1';
    }
    $this->baseUrl = $base;
  }

  /**
   * {@inheritdoc}
   */
  protected function getDefaultHeaders(): array {
    return [
      'Authorization' => 'Bearer ' . $this->apiKey,
    ];
  }

  /**
   * {@inheritdoc}
   */
  public function getModels(): array {
    if ($this->models !== NULL) {
      return $this->models;
    }
    $models = [];
    try {
      $data = $this->makeRequest($this->baseUrl . '/models', [], [], 'GET', 10);
      foreach ($data['data'] ?? [] as $model) {
        $id = $model['id'] ?? '';
        if ($id) {
          $models[$id] = $id;
        }
      }
      if (!empty($models)) {
        asort($models);
      }
    }
    catch (\Exception $e) {
      watchdog('ai_provider_litellm', 'Failed to fetch models: @error', ['@error' => $e->getMessage()], WATCHDOG_ERROR);
    }
    return $this->models = $models;
  }

  /**
   * {@inheritdoc}
   *
   * Proxy aliases do not describe capability. Use /model/info metadata instead.
   */
  public function getModelsByCapability($capability): array {
    $all_models = $this->getModels();
    $filtered = [];
    $metadata = $this->getModelMetadata();
    $capability = ai_normalize_capability_name($capability);

    foreach ($all_models as $id => $name) {
      $info = $metadata[$id] ?? [];
      $mode = $info['mode'] ?? '';
      $is_match = FALSE;
      switch ($capability) {
        case 'text':
          // Without /model/info metadata keep every alias selectable for chat,
          // as before; manual capabilities can narrow the list.
          $is_match = $metadata ? in_array($mode, ['chat', 'completion'], TRUE) : TRUE;
          break;

        case 'embedding':
        case 'embeddings':
          $is_match = $mode === 'embedding';
          break;

        case 'vision':
          $is_match = $mode === 'chat' && !empty($info['supports_vision']);
          break;

        case 'tool_calling':
          $is_match = $mode === 'chat' && !empty($info['supports_function_calling']);
          break;

        case 'thinking':
          $is_match = $mode === 'chat' && !empty($info['supports_reasoning']);
          break;

        case 'audio':
          $is_match = $mode === 'chat' && !empty($info['supports_audio_input']);
          break;

        case 'image':
        case 'moderation':
        default:
          $is_match = FALSE;
          break;
      }

      if ($is_match) {
        $filtered[$id] = $name;
      }
    }

    backdrop_alter('ai_model_capabilities', $filtered, $capability, $this);
    return $filtered;
  }

  /**
   * Fetch metadata without persisting proxy credentials from /model/info.
   */
  protected function getModelMetadata(): array {
    if ($this->modelMetadata !== NULL) {
      return $this->modelMetadata;
    }
    $cache_key = ai_models_cache_key('litellm', 'model_metadata', [$this->baseUrl, hash('sha256', $this->apiKey)]);
    if ($cached = cache_get($cache_key, 'cache')) {
      return $this->modelMetadata = $cached->data;
    }
    $metadata = [];
    try {
      $url = preg_replace('#/v1$#', '', $this->baseUrl) . '/model/info';
      $data = $this->makeRequest($url, [], [], 'GET', 10);
      foreach ($data['data'] ?? [] as $model) {
        $id = $model['model_name'] ?? '';
        if ($id !== '') {
          $info = array_intersect_key((array) ($model['model_info'] ?? []), array_flip([
            'mode', 'supports_vision', 'supports_function_calling',
            'supports_reasoning', 'supports_audio_input',
          ]));
          // A public alias can route to several deployments. Keep only metadata
          // all of them share, rather than promising a capability at random.
          if (isset($metadata[$id])) {
            $info = array_intersect_assoc($metadata[$id], $info);
          }
          $metadata[$id] = $info;
        }
      }
    }
    catch (\Exception $e) {
      // Do not log the response body: some proxies return credential details.
      watchdog('ai_provider_litellm', 'Model capability metadata is unavailable. Configure manual capabilities or allow access to /model/info.', [], WATCHDOG_WARNING);
    }
    cache_set($cache_key, $metadata, 'cache', REQUEST_TIME + ($metadata ? 21600 : 300));
    return $this->modelMetadata = $metadata;
  }

  /**
   * {@inheritdoc}
   */
  public function completions(string $model, string $prompt, $temperature, $max_tokens = 512, bool $stream_response = FALSE) {
    $payload = [
      'model'       => $model,
      'prompt'      => $prompt,
      'temperature' => (float) $temperature,
      'max_tokens'  => (int) $max_tokens,
    ];

    if ($stream_response) {
      $payload['stream'] = TRUE;
      $options = $this->buildPostOptions($payload, 300);
      return $this->buildStreamingResponse($this->baseUrl . '/completions', $options, function ($data) {
        return $data['choices'][0]['text'] ?? NULL;
      });
    }

    $result = $this->makeRequest($this->baseUrl . '/completions', $payload, [], 'POST', 60);
    return $result['choices'][0]['text'] ?? '';
  }

  /**
   * {@inheritdoc}
   */
  public function chat(string $model, array $messages, $temperature, $max_tokens = 1024, bool $stream_response = FALSE, array $context_extra = []) {
    if (function_exists('backdrop_alter') && empty($context_extra['skip_ai_message_alter'])) {
      $context = [
        'operation' => 'chat',
        'model'     => $model,
        'provider'  => 'litellm',
      ];
      backdrop_alter('ai_chat_messages', $messages, $context);
    }

    $payload = [
      'model'       => $model,
      'messages'    => $messages,
      'temperature' => (float) $temperature,
      'max_tokens'  => (int) $max_tokens,
    ];

    if ($stream_response) {
      $payload['stream'] = TRUE;
      $options = $this->buildPostOptions($payload, 300);
      return $this->buildStreamingResponse($this->baseUrl . '/chat/completions', $options, function ($data) {
        return $data['choices'][0]['delta']['content'] ?? NULL;
      });
    }

    $result = $this->makeRequest($this->baseUrl . '/chat/completions', $payload, [], 'POST', 60);
    return $result['choices'][0]['message']['content'] ?? '';
  }

  /**
   * {@inheritdoc}
   */
  public function images(string $model, string $prompt, string $size, string $response_format, string $quality = 'standard', string $style = 'natural', ?string $output_format = NULL) {
    watchdog('ai_provider_litellm', 'Image generation may not be supported by all LiteLLM endpoints', [], WATCHDOG_WARNING);
    throw new \RuntimeException('Image generation is not supported by this LiteLLM configuration.');
  }

  /**
   * {@inheritdoc}
   */
  public function textToSpeech(string $model, string $input, string $voice, string $response_format) {
    watchdog('ai_provider_litellm', 'Text-to-speech may not be supported by all LiteLLM endpoints', [], WATCHDOG_WARNING);
    throw new \RuntimeException('Text-to-speech is not supported by this LiteLLM configuration.');
  }

  /**
   * {@inheritdoc}
   */
  public function speechToText(string $model, string $file, string $task = 'transcribe', $temperature = 0.4, string $response_format = 'verbose_json') {
    watchdog('ai_provider_litellm', 'Speech-to-text may not be supported by all LiteLLM endpoints', [], WATCHDOG_WARNING);
    throw new \RuntimeException('Speech-to-text is not supported by this LiteLLM configuration.');
  }

  /**
   * {@inheritdoc}
   */
  public function moderation(string $input, string $model = 'omni-moderation-latest'): array {
    watchdog('ai_provider_litellm', 'Moderation may not be supported by all LiteLLM endpoints', [], WATCHDOG_WARNING);
    throw new \RuntimeException('Moderation is not supported by this LiteLLM configuration.');
  }

  /**
   * {@inheritdoc}
   */
  public function embedding(string $input, string $model, bool $log = TRUE): array {
    $start_time = microtime(TRUE);
    $result = $this->embeddings($model, [$input], 'float', $log);
    if (!empty($result['data'][0]['embedding'])) {
      if (isset($this->api) && method_exists($this->api, 'recordLog')) {
        $duration = microtime(TRUE) - $start_time;
        $this->api->recordLog('embedding', $model, ['input' => $input], $result, TRUE, $duration, NULL, !$log);
      }
      return $result['data'][0]['embedding'];
    }

    if (isset($this->api) && method_exists($this->api, 'recordLog')) {
      $duration = microtime(TRUE) - $start_time;
      $error = !empty($result['error']) ? $result['error'] : 'Unknown error';
      $this->api->recordLog('embedding', $model, ['input' => $input], NULL, FALSE, $duration, $error, !$log);
    }
    throw new \RuntimeException(!empty($result['error']) ? $result['error'] : 'Embedding request failed.');
  }

  /**
   * {@inheritdoc}
   *
   * @param bool $log
   *   Passed through to the error logger. FALSE marks a capability probe, whose
   *   expected failures must not reach watchdog.
   */
  public function embeddings(string $model, array $inputs, string $response_format = 'float', bool $log = TRUE): array {
    try {
      $results = [];
      foreach ($inputs as $index => $input) {
        $response = $this->makeRequest($this->baseUrl . '/embeddings', [
          'model' => $model,
          'input' => $input,
        ]);
        $embedding = $response['data'][0]['embedding'] ?? [];
        if (!empty($embedding)) {
          $results[] = [
            'object'    => 'embedding',
            'embedding' => $embedding,
            'index'     => $index,
          ];
        }
      }

      return [
        'object' => 'list',
        'data'   => $results,
        'model'  => $model,
        'usage'  => [
          'prompt_tokens' => count($inputs),
          'total_tokens'  => count($inputs),
        ],
      ];
    }
    catch (\Exception $e) {
      ai_log_embedding_error('ai_provider_litellm', $e->getMessage(), $log);
      throw $e;
    }
  }

  /**
   * {@inheritdoc}
   */
  public function chatWithTools(string $model, array $messages, array $tools, $temperature, $max_tokens = 1024, string $tool_choice = 'auto', array $context_extra = []): array {
    try {
      $payload = [
        'model'       => $model,
        'messages'    => $messages,
        'tools'       => $tools,
        'tool_choice' => $tool_choice,
        'temperature' => (float) $temperature,
      ];
      if ((int) $max_tokens > 0) {
        $payload['max_tokens'] = (int) $max_tokens;
      }
      $result = $this->makeRequest($this->baseUrl . '/chat/completions', $payload, [], 'POST', 60);
      return $this->normalizeToolResponse($result);
    }
    catch (\Exception $e) {
      watchdog('ai_provider_litellm', 'chatWithTools error: @error', ['@error' => $e->getMessage()], WATCHDOG_ERROR);
      throw $e;
    }
  }

  /**
   * Build request options array for a JSON POST (used by streaming paths).
   */
  protected function buildPostOptions(array $body, int $timeout = 60): array {
    return [
      'method'  => 'POST',
      'headers' => array_merge(
        ['Accept' => 'application/json', 'Content-Type' => 'application/json'],
        $this->getDefaultHeaders()
      ),
      'data'    => json_encode($body),
      'timeout' => $timeout,
    ];
  }

}
