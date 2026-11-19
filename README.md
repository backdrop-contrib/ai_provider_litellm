# AI Provider LiteLLM

Connect Backdrop’s AI features to models served through a LiteLLM proxy. Use one proxy address to reach models from several services. Use it for chat, text generation, and search embeddings. Vision and tool calling depend on the model and proxy configuration.

## Requirements

Backdrop CMS 1.x, PHP 8.2 or later, and the AI module. A reachable LiteLLM proxy is required. The wiki explains how to configure authenticated proxies with the Key module.

## Installation

1. Install and enable this module using the official [Backdrop CMS instructions](https://backdropcms.org/user-guide/modules).
2. Open **AI settings** at `admin/config/ai/settings` and enable the provider.
3. Set **LiteLLM Server URL** to your proxy address. The default is `http://localhost:4000`. The adapter accepts a URL with or without `/v1`.

Read the [module wiki](https://github.com/backdrop-contrib/ai_provider_litellm/wiki) for setup details, supported operations, and troubleshooting.

## Issues

Bugs and feature requests should be reported in the [Issue Queue](https://github.com/backdrop-contrib/ai_provider_litellm/issues).

## Current Maintainer

[Justin Keiser](https://github.com/keiserjb)

## Credits

- Created for Backdrop CMS by [Justin Keiser](https://github.com/keiserjb).

- Developed with AI assistance.

## License

This project is GPL v2 software. See the LICENSE.txt file in this directory for complete text.
