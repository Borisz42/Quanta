# QUANTA Plugin Integration Plan for Unsloth Studio

This document outlines the steps to integrate the QUANTA neuro-symbolic memory engine directly into Unsloth Studio's built-in chat UI using the Model Context Protocol (MCP). Since Unsloth Studio natively supports chat and MCP tools, this approach avoids the need for external UI layers like Open WebUI.

## Overview

The integration relies on Unsloth Studio's ability to connect to external tools via the Model Context Protocol. We will configure Unsloth Studio to launch and connect to the local QUANTA MCP server (`src/server/mcp_server.py`) over `stdio`. This allows the chat model running in Unsloth Studio to natively interact with QUANTA memory.

## Architecture

```mermaid
flowchart TD
    User["User"] -->|Chat| StudioUI["Unsloth Studio Chat UI"]
    StudioUI -->|MCP (stdio)| MCPServer["QUANTA MCP Server (src/server/mcp_server.py)"]
    MCPServer -->|Ingest/Query/Verify| QuantaEngine["QUANTA Neuro-Symbolic Engine"]
    StudioUI -->|Local Inference| Qwen["unsloth/Qwen3.5-4B-MTP-GGUF"]
```

## Step-by-Step Configuration

### 1. Locate the Unsloth Studio MCP Configuration

Unsloth Studio uses a configuration file (typically in a JSON format similar to Claude Desktop or Antigravity) to define available MCP servers. You need to locate the `mcp_config.json` or equivalent configuration section for your Unsloth Studio installation.

### 2. Add the QUANTA MCP Server Entry

Add the following configuration block to the Unsloth Studio MCP settings. This tells the studio how to start the QUANTA MCP server as a background process communicating over standard input/output.

```json
{
  "mcpServers": {
    "quanta-memory": {
      "command": "python",
      "args": [
        "/path/to/Quanta/src/server/mcp_server.py"
      ],
      "env": {
        "QUANTA_ALLOW_CPU_OFFLOAD": "1",
        "PYTHONPATH": "/path/to/Quanta/src"
      }
    }
  }
}
```

*Important:* Replace `/path/to/Quanta` with the actual absolute path to your cloned QUANTA repository directory.

### 3. Restart Unsloth Studio

After saving the configuration, restart Unsloth Studio so that it reads the new configuration and initializes the `quanta-memory` MCP connection.

### 4. Verify Tool Availability in Chat

1. Open the Unsloth Studio Chat interface.
2. Check the tools or plugins menu (usually indicated by a puzzle piece or wrench icon).
3. You should see the QUANTA tools listed and available for the model to use:
   - `quanta_ingest_document`: For extracting and saving long-term knowledge.
   - `quanta_query_memory`: For retrieving specific verified facts from the neuro-symbolic graph.
   - `quanta_get_entity_details`: For inspecting the raw vector state of an entity.

### 5. Interactive Usage

You can now interact with the chat naturally. Try prompts like:
* "Please ingest the background story of Chapter 1 using the quanta tools."
* "What did Alice find out about the plasma conduit? Check your quanta memory."

The Qwen model running inside Unsloth Studio will automatically invoke the tools, wait for the JSON-RPC response from the QUANTA MCP server, and incorporate the verified facts into its reply.
