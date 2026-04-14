// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   test_api.cpp
 * @date   21 Jan 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @brief  Simple application to test CausalLM API
 * @bug    No known bugs except for NYI items
 *
 */

#include "causal_lm_api.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {
constexpr const char *COLOR_RESET = "\033[0m";
constexpr const char *COLOR_BOLD = "\033[1m";
constexpr const char *COLOR_CYAN = "\033[36m";
constexpr const char *COLOR_GREEN = "\033[32m";
constexpr const char *COLOR_YELLOW = "\033[33m";
constexpr const char *COLOR_BLUE = "\033[34m";
constexpr const char *COLOR_RED = "\033[31m";
constexpr const char *COLOR_MAGENTA = "\033[35m";
constexpr const char *COLOR_GRAY = "\033[90m";

void printLine(const std::string &s, int length = 80) {
  for (int i = 0; i < length; ++i)
    std::cout << s;
  std::cout << std::endl;
}

void printSection(const std::string &section) {
  std::cout << "\n"
            << COLOR_BOLD << COLOR_BLUE
            << "+-------------------------------------------------------------+"
            << COLOR_RESET << "\n";
  std::cout << COLOR_BOLD << COLOR_BLUE << "|  " << section
            << std::string(58 - section.length(), ' ') << "|" << COLOR_RESET
            << "\n";
  std::cout << COLOR_BOLD << COLOR_BLUE
            << "+-------------------------------------------------------------+"
            << COLOR_RESET << "\n\n";
}

void printSuccess(const std::string &msg) {
  std::cout << COLOR_GREEN << "✓ " << COLOR_BOLD << msg << COLOR_RESET
            << "\n\n";
}

void printError(const std::string &msg) {
  std::cerr << COLOR_RED << "✗ " << COLOR_BOLD << "Error: " << COLOR_RESET
            << msg << "\n";
}

void printWarning(const std::string &msg) {
  std::cout << COLOR_YELLOW << "⚠ " << msg << COLOR_RESET << "\n";
}

void printInfo(const std::string &label, const std::string &value) {
  std::cout << COLOR_CYAN << "  " << label << ":" << COLOR_RESET << " " << value
            << "\n";
}

void printLogo() {
  std::cout << "\n";
  std::cout << COLOR_BOLD << COLOR_MAGENTA;
  std::cout << "  ███╗   ██╗███╗   ██╗████████╗██████╗ \n";
  std::cout << "  ████╗  ██║████╗  ██║╚══██╔══╝██╔══██╗\n";
  std::cout << "  ██╔██╗ ██║██╔██╗ ██║   ██║   ██████╔╝\n";
  std::cout << "  ██║╚██╗██║██║╚██╗██║   ██║   ██╔══██╗\n";
  std::cout << "  ██║ ╚████║██║ ╚████║   ██║   ██║  ██║\n";
  std::cout << "  ╚═╝  ╚═══╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝\n";
  std::cout << COLOR_RESET;
  std::cout << COLOR_BOLD << COLOR_CYAN
            << "  ────────────────────────────────\n";
  std::cout << "      Causal Language Model API\n"
            << "  ────────────────────────────────\n";
  std::cout << COLOR_RESET << "\n";
}

void printUsage(const char *program_name) {
  std::cout << COLOR_YELLOW << "Usage:" << COLOR_RESET << "\n";
  std::cout << "  " << COLOR_BOLD << program_name << COLOR_RESET
            << " <model_name> [prompt] [use_chat_template] [quantization] "
               "[verbose] [--verify-memory|--multi-turn]\n\n";

  std::cout << COLOR_CYAN << "Arguments:" << COLOR_RESET << "\n";
  std::cout << "  model_name        " << COLOR_BOLD << "REQUIRED" << COLOR_RESET
            << "  - Model name (e.g., QWEN3-0.6B, QWEN3-1.7B)\n";
  std::cout << "  prompt            " << COLOR_GREEN << "OPTIONAL"
            << COLOR_RESET
            << "  - Input prompt (default: 'Hello, how are you?')\n";
  std::cout << "  use_chat_template " << COLOR_GREEN << "OPTIONAL"
            << COLOR_RESET << "  - 0/1 or true/false (default: 1)\n";
  std::cout << "  quantization      " << COLOR_GREEN << "OPTIONAL"
            << COLOR_RESET
            << "  - W4A32/W16A16/W8A16/W32A32/UNKNOWN (default: UNKNOWN)\n";
  std::cout << "  verbose           " << COLOR_GREEN << "OPTIONAL"
            << COLOR_RESET << "  - 0/1 or true/false (default: 0)\n\n";

  std::cout << COLOR_CYAN << "Modes:" << COLOR_RESET << "\n";
  std::cout << "  --verify-memory   " << COLOR_GREEN << "OPTIONAL"
            << COLOR_RESET
            << "  - Scripted 2-turn regression test, then reset+recheck\n";
  std::cout << "  --multi-turn      " << COLOR_GREEN << "OPTIONAL"
            << COLOR_RESET
            << "  - Interactive REPL (/reset, /quit supported)\n\n";

  std::cout << COLOR_YELLOW << "Examples:" << COLOR_RESET << "\n";
  std::cout << "  " << COLOR_BOLD << program_name << COLOR_RESET
            << " QWEN3-0.6B \"Tell me a joke\" 1 W4A32\n";
  std::cout << "  " << COLOR_BOLD << program_name << COLOR_RESET
            << " QWEN3-0.6B \"Write a poem\" 1 W32A32 1\n";
  std::cout << "  " << COLOR_BOLD << program_name << COLOR_RESET
            << " QWEN3-0.6B \"\" 1 W4A32 0 --verify-memory\n";
  std::cout << "  " << COLOR_BOLD << program_name << COLOR_RESET
            << " QWEN3-0.6B \"\" 1 W4A32 0 --multi-turn\n\n";
}

/**
 * @brief Run a single inference turn and (optionally) print metrics.
 * @param prompt        Input user prompt for this turn.
 * @param verbose       Whether to stream tokens (matches global verbose flag).
 * @param show_metrics  Whether to dump performance metrics for this turn.
 * @param[out] out_text If non-null, receives the generated assistant text.
 * @return ErrorCode from runModel().
 */
ErrorCode run_turn(const char *prompt, bool verbose, bool show_metrics,
                   std::string *out_text = nullptr) {
  std::cout << COLOR_CYAN << "📝 " << COLOR_RESET << "Input Prompt:\n";
  std::cout << COLOR_BOLD << COLOR_YELLOW << "  " << prompt << COLOR_RESET
            << "\n\n";

  std::cout << COLOR_CYAN << "⚡ " << COLOR_RESET << "Running inference...\n\n";

  const char *outputText = nullptr;

  if (verbose) {
    std::cout << COLOR_CYAN << "💬 " << COLOR_RESET << "Streaming Output:\n";
    std::cout << COLOR_BOLD << COLOR_GRAY;
  }

  ErrorCode err = runModel(prompt, &outputText);

  if (verbose) {
    std::cout << COLOR_RESET << "\n\n";
  }

  if (err != CAUSAL_LM_ERROR_NONE) {
    printError("Failed to run model");
    std::cerr << "  Error code: " << static_cast<int>(err) << "\n";
    return err;
  }

  if (outputText) {
    if (out_text)
      *out_text = outputText;
    std::cout << COLOR_CYAN << "💬 " << COLOR_RESET << "Output:\n";
    std::cout << COLOR_BOLD << COLOR_GREEN << "  ";
    std::string out(outputText);
    size_t pos = 0;
    while (pos < out.length()) {
      size_t newlinePos = out.find('\n', pos);
      if (newlinePos == std::string::npos) {
        newlinePos = out.length();
      }
      std::string line = out.substr(pos, newlinePos - pos);
      std::cout << line;
      if (newlinePos < out.length()) {
        std::cout << "\n  ";
        pos = newlinePos + 1;
      } else {
        pos = out.length();
      }
    }
    std::cout << COLOR_RESET << "\n\n";
  } else {
    printWarning("No output generated");
  }

  if (!show_metrics)
    return err;

  PerformanceMetrics metrics;
  ErrorCode merr = getPerformanceMetrics(&metrics);
  if (merr != CAUSAL_LM_ERROR_NONE) {
    printWarning("Failed to get metrics");
    std::cout << "  Error code: " << static_cast<int>(merr) << "\n";
    return err;
  }

  double prefill_tps =
    metrics.prefill_duration_ms > 0
      ? (metrics.prefill_tokens / metrics.prefill_duration_ms * 1000.0)
      : 0.0;
  double gen_tps =
    metrics.generation_duration_ms > 0
      ? (metrics.generation_tokens / metrics.generation_duration_ms * 1000.0)
      : 0.0;

  std::cout << COLOR_CYAN << "  📊 " << COLOR_RESET << COLOR_BOLD
            << "Prefill" << COLOR_RESET << "  tokens="
            << metrics.prefill_tokens << "  "
            << std::fixed << std::setprecision(2)
            << metrics.prefill_duration_ms << " ms  "
            << COLOR_GREEN << std::setprecision(1) << prefill_tps
            << COLOR_RESET << " tok/s\n";
  std::cout << COLOR_CYAN << "  📊 " << COLOR_RESET << COLOR_BOLD
            << "Generation" << COLOR_RESET << " tokens="
            << metrics.generation_tokens << "  "
            << std::fixed << std::setprecision(2)
            << metrics.generation_duration_ms << " ms  "
            << COLOR_GREEN << std::setprecision(1) << gen_tps
            << COLOR_RESET << " tok/s\n\n";

  return err;
}

/**
 * @brief Lower-case a string in-place.
 */
std::string to_lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

/**
 * @brief Trim ASCII whitespace from both ends.
 */
std::string trim(const std::string &s) {
  size_t b = 0, e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b])))
    ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])))
    --e;
  return s.substr(b, e - b);
}

/**
 * @brief Scripted multi-turn regression: ensure the model remembers
 *        a name across turns, and forgets it after resetConversation().
 * @return 0 on success, non-zero on assertion failure.
 */
int run_verify_memory(bool verbose) {
  printSection("Verify Memory: Multi-Turn Regression");

  const char *turn1 = "My name is Alice. Please remember my name.";
  const char *turn2 = "What is my name?";

  std::cout << COLOR_BOLD << "Turn 1 (set context)" << COLOR_RESET << "\n";
  std::string out1;
  if (run_turn(turn1, verbose, true, &out1) != CAUSAL_LM_ERROR_NONE) {
    printError("Turn 1 failed");
    return 1;
  }

  std::cout << COLOR_BOLD << "Turn 2 (recall)" << COLOR_RESET << "\n";
  std::string out2;
  if (run_turn(turn2, verbose, true, &out2) != CAUSAL_LM_ERROR_NONE) {
    printError("Turn 2 failed");
    return 1;
  }

  bool remembered = to_lower(out2).find("alice") != std::string::npos;
  if (!remembered) {
    printError("Turn 2 output did not contain 'Alice' — multi-turn context "
               "was NOT preserved.");
    return 2;
  }
  printSuccess("Turn 2 recalled 'Alice' — multi-turn context preserved.");

  std::cout << COLOR_BOLD << "Calling resetConversation()..." << COLOR_RESET
            << "\n";
  ErrorCode rerr = resetConversation();
  if (rerr != CAUSAL_LM_ERROR_NONE) {
    printError("resetConversation() failed");
    std::cerr << "  Error code: " << static_cast<int>(rerr) << "\n";
    return 3;
  }
  printSuccess("Conversation state reset.");

  std::cout << COLOR_BOLD << "Turn 3 (post-reset recall, should NOT know)"
            << COLOR_RESET << "\n";
  std::string out3;
  if (run_turn(turn2, verbose, true, &out3) != CAUSAL_LM_ERROR_NONE) {
    printError("Turn 3 failed");
    return 4;
  }

  bool still_remembers = to_lower(out3).find("alice") != std::string::npos;
  if (still_remembers) {
    printError("Turn 3 still contains 'Alice' AFTER reset — "
               "resetConversation() did not clear context.");
    return 5;
  }
  printSuccess("Turn 3 did not assert 'Alice' — reset cleared context.");

  printLine("═", 63);
  std::cout << COLOR_BOLD << COLOR_GREEN
            << "  ✓ verify-memory PASSED" << COLOR_RESET << "\n";
  printLine("═", 63);
  std::cout << "\n";
  return 0;
}

/**
 * @brief Interactive multi-turn REPL.
 *        Commands: /reset (clear context), /quit (exit), EOF -> exit.
 */
int run_multi_turn_repl(bool verbose) {
  printSection("Multi-Turn REPL");
  std::cout << "Commands: " << COLOR_BOLD << "/reset" << COLOR_RESET
            << " clears context, " << COLOR_BOLD << "/quit" << COLOR_RESET
            << " (or EOF) exits.\n\n";

  while (true) {
    std::cout << COLOR_BOLD << COLOR_CYAN << "you> " << COLOR_RESET
              << std::flush;
    std::string line;
    if (!std::getline(std::cin, line)) {
      std::cout << "\n[EOF] exiting.\n";
      return 0;
    }
    std::string cmd = trim(line);
    if (cmd.empty())
      continue;
    if (cmd == "/quit" || cmd == "/exit") {
      std::cout << "Bye.\n";
      return 0;
    }
    if (cmd == "/reset") {
      ErrorCode rerr = resetConversation();
      if (rerr != CAUSAL_LM_ERROR_NONE) {
        printError("resetConversation() failed");
        std::cerr << "  Error code: " << static_cast<int>(rerr) << "\n";
      } else {
        printSuccess("Conversation state reset.");
      }
      continue;
    }
    std::string out;
    ErrorCode err = run_turn(cmd.c_str(), verbose, true, &out);
    if (err != CAUSAL_LM_ERROR_NONE) {
      printError("Turn failed; you may want to /reset.");
    }
  }
  return 0;
}
} // namespace

int main(int argc, char *argv[]) {
  printLogo();

  // Strip mode flags (--verify-memory / --multi-turn) out of argv before
  // the existing positional-argument parser runs, so older invocations
  // remain byte-identical.
  bool mode_verify_memory = false;
  bool mode_multi_turn = false;
  std::vector<char *> filtered_argv;
  filtered_argv.reserve(argc);
  for (int i = 0; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--verify-memory") {
      mode_verify_memory = true;
      continue;
    }
    if (a == "--multi-turn") {
      mode_multi_turn = true;
      continue;
    }
    filtered_argv.push_back(argv[i]);
  }
  if (mode_verify_memory && mode_multi_turn) {
    printError("--verify-memory and --multi-turn are mutually exclusive.");
    return 1;
  }
  argc = static_cast<int>(filtered_argv.size());
  argv = filtered_argv.data();

  if (argc < 2) {
    printSection("ERROR: Missing Required Arguments");
    printUsage(argv[0]);
    return 1;
  }

  const char *model_name = argv[1];
  const char *prompt = (argc >= 3 && argv[2][0] != '\0')
                         ? argv[2]
                         : "Hello, how are you?";
  bool use_chat_template = true;
  if (argc >= 4) {
    use_chat_template =
      (std::string(argv[3]) == "1" || std::string(argv[3]) == "true");
  }

  std::string quant_str = "UNKNOWN";
  ModelQuantizationType quant_type = CAUSAL_LM_QUANTIZATION_UNKNOWN;
  if (argc >= 5) {
    quant_str = std::string(argv[4]);
    if (quant_str == "W4A32")
      quant_type = CAUSAL_LM_QUANTIZATION_W4A32;
    else if (quant_str == "W16A16")
      quant_type = CAUSAL_LM_QUANTIZATION_W16A16;
    else if (quant_str == "W8A16")
      quant_type = CAUSAL_LM_QUANTIZATION_W8A16;
    else if (quant_str == "W32A32")
      quant_type = CAUSAL_LM_QUANTIZATION_W32A32;
  }

  bool verbose = true;
  if (argc >= 6) {
    verbose = (std::string(argv[5]) == "1" || std::string(argv[5]) == "true");
  }

  printSection("Configuration");
  printInfo("Model Name", model_name);
  printInfo("Use Chat Template", use_chat_template ? "true" : "false");
  printInfo("Quantization", quant_str);
  printInfo("Verbose", verbose ? "true" : "false");
  std::cout << "\n";

  printSection("Initialization");
  std::cout << COLOR_CYAN << "⏳ " << COLOR_RESET << "Configuring options...\n";
  Config config;
  config.use_chat_template = use_chat_template;
  config.debug_mode = true;
  config.verbose = verbose;
  ErrorCode err = setOptions(config);
  if (err != CAUSAL_LM_ERROR_NONE) {
    printError("Failed to set options");
    std::cerr << "  Error code: " << static_cast<int>(err) << "\n";
    return 1;
  }
  printSuccess("Options configured successfully");

  printSection("Model Loading");
  std::cout << COLOR_CYAN << "⏳ " << COLOR_RESET
            << "Loading model: " << COLOR_BOLD << model_name << COLOR_RESET
            << "\n";

  // Map string to ModelType
  ModelType model_type = CAUSAL_LM_MODEL_QWEN3_0_6B;
  std::string model_name_str(model_name);
  if (model_name_str == "QWEN3-0.6B") {
    model_type = CAUSAL_LM_MODEL_QWEN3_0_6B;
  } else if (model_name_str == "QWEN3-1.7B") {
    model_type = CAUSAL_LM_MODEL_QWEN3_1_7B;
  } else {
    std::cout << COLOR_YELLOW << "⚠ Warning: Unknown model name '"
              << model_name_str << "'. Defaulting to QWEN3-0.6B." << COLOR_RESET
              << "\n";
  }

  err = loadModel(CAUSAL_LM_BACKEND_CPU, model_type, quant_type);

  if (err != CAUSAL_LM_ERROR_NONE) {
    printError("Failed to load model");
    std::cerr << "  Error code: " << static_cast<int>(err) << "\n";
    return 1;
  }
  printSuccess("Model loaded successfully");

  // Mode dispatch: --verify-memory and --multi-turn run on top of the
  // already-loaded model and exit; otherwise fall back to legacy single-turn.
  if (mode_verify_memory) {
    int rc = run_verify_memory(verbose);
    return rc;
  }
  if (mode_multi_turn) {
    return run_multi_turn_repl(verbose);
  }

  printSection("Inference");
  std::string single_out;
  err = run_turn(prompt, verbose, false, &single_out);
  if (err != CAUSAL_LM_ERROR_NONE) {
    return 1;
  }

  printSection("Performance Metrics");
  PerformanceMetrics metrics;
  err = getPerformanceMetrics(&metrics);
  if (err != CAUSAL_LM_ERROR_NONE) {
    printWarning("Failed to get metrics");
    std::cout << "  Error code: " << static_cast<int>(err) << "\n";
  } else {
    double prefill_tps =
      metrics.prefill_duration_ms > 0
        ? (metrics.prefill_tokens / metrics.prefill_duration_ms * 1000.0)
        : 0.0;
    double gen_tps =
      metrics.generation_duration_ms > 0
        ? (metrics.generation_tokens / metrics.generation_duration_ms * 1000.0)
        : 0.0;

    std::cout << COLOR_CYAN << "  📊 " << COLOR_RESET << COLOR_BOLD
              << "Prefill Stage" << COLOR_RESET << "\n";
    std::cout << COLOR_CYAN << "    Tokens:" << COLOR_RESET << "       "
              << metrics.prefill_tokens << "\n";
    std::cout << COLOR_CYAN << "    Duration:" << COLOR_RESET << "     "
              << std::fixed << std::setprecision(2)
              << metrics.prefill_duration_ms << " ms\n";
    std::cout << COLOR_CYAN << "    Throughput:" << COLOR_RESET << "   "
              << COLOR_BOLD << COLOR_GREEN << std::fixed << std::setprecision(1)
              << prefill_tps << COLOR_RESET << " tokens/sec\n\n";

    std::cout << COLOR_CYAN << "  📊 " << COLOR_RESET << COLOR_BOLD
              << "Generation Stage" << COLOR_RESET << "\n";
    std::cout << COLOR_CYAN << "    Tokens:" << COLOR_RESET << "       "
              << metrics.generation_tokens << "\n";
    std::cout << COLOR_CYAN << "    Duration:" << COLOR_RESET << "     "
              << std::fixed << std::setprecision(2)
              << metrics.generation_duration_ms << " ms\n";
    std::cout << COLOR_CYAN << "    Throughput:" << COLOR_RESET << "   "
              << COLOR_BOLD << COLOR_GREEN << std::fixed << std::setprecision(1)
              << gen_tps << COLOR_RESET << " tokens/sec\n\n";

    std::cout << COLOR_CYAN << "  📊 " << COLOR_RESET << COLOR_BOLD
              << "Total Stats" << COLOR_RESET << "\n";
    std::cout << COLOR_CYAN << "    Init time:" << COLOR_RESET << "    "
              << std::fixed << std::setprecision(2)
              << metrics.initialization_duration_ms << " ms\n";
    std::cout << COLOR_CYAN << "    Duration :" << COLOR_RESET << "    "
              << std::fixed << std::setprecision(2) << metrics.total_duration_ms
              << " ms\n";
    std::cout << COLOR_CYAN << "    Peak Mem:" << COLOR_RESET << "     "
              << metrics.peak_memory_kb / 1024 << " MB\n\n";
  }

  printLine("═", 63);
  std::cout << COLOR_BOLD << COLOR_GREEN << "  ✓ Test completed successfully!"
            << COLOR_RESET << "\n";
  printLine("═", 63);
  std::cout << "\n";

  return 0;
}
