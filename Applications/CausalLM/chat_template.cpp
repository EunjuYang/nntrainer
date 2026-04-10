// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    chat_template.cpp
 * @date    10 Apr 2026
 * @brief   Chat template implementation with mini Jinja2 renderer
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Eunju Yang <ej.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#include "chat_template.h"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace causallm {

// ============================================================================
// Token types for the Jinja2 lexer
// ============================================================================
enum class TokenType {
  TEXT,
  EXPRESSION_START, // {{
  EXPRESSION_END,   // }}
  STATEMENT_START,  // {%
  STATEMENT_END,    // %}
  STRING,
  INTEGER,
  FLOAT,
  IDENTIFIER,
  DOT,
  LBRACKET,
  RBRACKET,
  LPAREN,
  RPAREN,
  PLUS,
  MINUS,
  PERCENT,
  PIPE,
  COMMA,
  EQ,      // ==
  NEQ,     // !=
  ASSIGN,  // =
  NOT,     // not
  AND,     // and
  OR,      // or
  TRUE_LIT,
  FALSE_LIT,
  NONE_LIT,
  IF,
  ELIF,
  ELSE,
  ENDIF,
  FOR,
  IN,
  ENDFOR,
  SET,
  IS,
  END_OF_INPUT,
};

struct Token {
  TokenType type;
  std::string value;
  bool strip_before = false; // {%- or {{-
  bool strip_after = false;  // -%} or -}}
};

// ============================================================================
// Lexer
// ============================================================================
class Lexer {
public:
  explicit Lexer(const std::string &input) : input_(input), pos_(0) {}

  std::vector<Token> tokenize() {
    std::vector<Token> tokens;

    while (pos_ < input_.size()) {
      if (match("{{")) {
        bool strip = false;
        if (pos_ < input_.size() && input_[pos_] == '-') {
          strip = true;
          pos_++;
        }
        Token start;
        start.type = TokenType::EXPRESSION_START;
        start.strip_before = strip;
        tokens.push_back(start);
        skipWhitespace();
        tokenizeInside(tokens, TokenType::EXPRESSION_END);
      } else if (match("{%")) {
        bool strip = false;
        if (pos_ < input_.size() && input_[pos_] == '-') {
          strip = true;
          pos_++;
        }
        Token start;
        start.type = TokenType::STATEMENT_START;
        start.strip_before = strip;
        tokens.push_back(start);
        skipWhitespace();
        tokenizeInside(tokens, TokenType::STATEMENT_END);
      } else {
        std::string text;
        while (pos_ < input_.size()) {
          if ((pos_ + 1 < input_.size()) &&
              ((input_[pos_] == '{' &&
                (input_[pos_ + 1] == '{' || input_[pos_ + 1] == '%')))) {
            break;
          }
          text += input_[pos_++];
        }
        if (!text.empty()) {
          Token t;
          t.type = TokenType::TEXT;
          t.value = text;
          tokens.push_back(t);
        }
      }
    }

    Token eof;
    eof.type = TokenType::END_OF_INPUT;
    tokens.push_back(eof);
    return tokens;
  }

private:
  bool match(const std::string &s) {
    if (pos_ + s.size() <= input_.size() &&
        input_.substr(pos_, s.size()) == s) {
      pos_ += s.size();
      return true;
    }
    return false;
  }

  void skipWhitespace() {
    while (pos_ < input_.size() && std::isspace(input_[pos_]))
      pos_++;
  }

  void tokenizeInside(std::vector<Token> &tokens, TokenType end_type) {
    while (pos_ < input_.size()) {
      skipWhitespace();
      if (pos_ >= input_.size())
        break;

      // Check for closing tag
      if (end_type == TokenType::EXPRESSION_END) {
        if (pos_ + 1 < input_.size() && input_[pos_] == '-' &&
            input_[pos_ + 1] == '}' && pos_ + 2 < input_.size() &&
            input_[pos_ + 2] == '}') {
          pos_ += 3;
          Token end;
          end.type = end_type;
          end.strip_after = true;
          tokens.push_back(end);
          return;
        }
        if (match("}}")) {
          Token end;
          end.type = end_type;
          tokens.push_back(end);
          return;
        }
      } else if (end_type == TokenType::STATEMENT_END) {
        if (pos_ + 1 < input_.size() && input_[pos_] == '-' &&
            input_[pos_ + 1] == '%' && pos_ + 2 < input_.size() &&
            input_[pos_ + 2] == '}') {
          pos_ += 3;
          Token end;
          end.type = end_type;
          end.strip_after = true;
          tokens.push_back(end);
          return;
        }
        if (match("%}")) {
          Token end;
          end.type = end_type;
          tokens.push_back(end);
          return;
        }
      }

      // String literal
      if (input_[pos_] == '\'' || input_[pos_] == '"') {
        tokens.push_back(readString());
        continue;
      }

      // Number
      if (std::isdigit(input_[pos_])) {
        tokens.push_back(readNumber());
        continue;
      }

      // Identifier or keyword
      if (std::isalpha(input_[pos_]) || input_[pos_] == '_') {
        tokens.push_back(readIdentifier());
        continue;
      }

      // Operators and punctuation
      Token t;
      switch (input_[pos_]) {
      case '.':
        t.type = TokenType::DOT;
        t.value = ".";
        pos_++;
        break;
      case '[':
        t.type = TokenType::LBRACKET;
        t.value = "[";
        pos_++;
        break;
      case ']':
        t.type = TokenType::RBRACKET;
        t.value = "]";
        pos_++;
        break;
      case '(':
        t.type = TokenType::LPAREN;
        t.value = "(";
        pos_++;
        break;
      case ')':
        t.type = TokenType::RPAREN;
        t.value = ")";
        pos_++;
        break;
      case '+':
        t.type = TokenType::PLUS;
        t.value = "+";
        pos_++;
        break;
      case '-':
        t.type = TokenType::MINUS;
        t.value = "-";
        pos_++;
        break;
      case '%':
        t.type = TokenType::PERCENT;
        t.value = "%";
        pos_++;
        break;
      case '|':
        t.type = TokenType::PIPE;
        t.value = "|";
        pos_++;
        break;
      case ',':
        t.type = TokenType::COMMA;
        t.value = ",";
        pos_++;
        break;
      case '=':
        if (pos_ + 1 < input_.size() && input_[pos_ + 1] == '=') {
          t.type = TokenType::EQ;
          t.value = "==";
          pos_ += 2;
        } else {
          t.type = TokenType::ASSIGN;
          t.value = "=";
          pos_++;
        }
        break;
      case '!':
        if (pos_ + 1 < input_.size() && input_[pos_ + 1] == '=') {
          t.type = TokenType::NEQ;
          t.value = "!=";
          pos_ += 2;
        } else {
          pos_++;
          continue;
        }
        break;
      default:
        pos_++;
        continue;
      }
      tokens.push_back(t);
    }
  }

  Token readString() {
    char quote = input_[pos_++];
    std::string value;
    while (pos_ < input_.size() && input_[pos_] != quote) {
      if (input_[pos_] == '\\' && pos_ + 1 < input_.size()) {
        pos_++;
        switch (input_[pos_]) {
        case 'n':
          value += '\n';
          break;
        case 't':
          value += '\t';
          break;
        case '\\':
          value += '\\';
          break;
        case '\'':
          value += '\'';
          break;
        case '"':
          value += '"';
          break;
        default:
          value += '\\';
          value += input_[pos_];
          break;
        }
      } else {
        value += input_[pos_];
      }
      pos_++;
    }
    if (pos_ < input_.size())
      pos_++; // skip closing quote

    Token t;
    t.type = TokenType::STRING;
    t.value = value;
    return t;
  }

  Token readNumber() {
    std::string value;
    bool has_dot = false;
    while (pos_ < input_.size() &&
           (std::isdigit(input_[pos_]) || input_[pos_] == '.')) {
      if (input_[pos_] == '.') {
        if (has_dot)
          break;
        has_dot = true;
      }
      value += input_[pos_++];
    }
    Token t;
    t.type = has_dot ? TokenType::FLOAT : TokenType::INTEGER;
    t.value = value;
    return t;
  }

  Token readIdentifier() {
    std::string value;
    while (pos_ < input_.size() &&
           (std::isalnum(input_[pos_]) || input_[pos_] == '_')) {
      value += input_[pos_++];
    }

    Token t;
    t.value = value;

    // Check for keywords
    if (value == "if")
      t.type = TokenType::IF;
    else if (value == "elif")
      t.type = TokenType::ELIF;
    else if (value == "else")
      t.type = TokenType::ELSE;
    else if (value == "endif")
      t.type = TokenType::ENDIF;
    else if (value == "for")
      t.type = TokenType::FOR;
    else if (value == "in")
      t.type = TokenType::IN;
    else if (value == "endfor")
      t.type = TokenType::ENDFOR;
    else if (value == "set")
      t.type = TokenType::SET;
    else if (value == "not")
      t.type = TokenType::NOT;
    else if (value == "and")
      t.type = TokenType::AND;
    else if (value == "or")
      t.type = TokenType::OR;
    else if (value == "is")
      t.type = TokenType::IS;
    else if (value == "true" || value == "True")
      t.type = TokenType::TRUE_LIT;
    else if (value == "false" || value == "False")
      t.type = TokenType::FALSE_LIT;
    else if (value == "none" || value == "None")
      t.type = TokenType::NONE_LIT;
    else
      t.type = TokenType::IDENTIFIER;

    return t;
  }

  std::string input_;
  size_t pos_;
};

// ============================================================================
// AST Nodes
// ============================================================================
struct ASTNode {
  virtual ~ASTNode() = default;
};

using ASTNodePtr = std::shared_ptr<ASTNode>;

struct ExprNode : ASTNode {};
using ExprNodePtr = std::shared_ptr<ExprNode>;

struct TextNode : ASTNode {
  std::string text;
};

struct OutputNode : ASTNode {
  ExprNodePtr expr;
  bool strip_before = false;
  bool strip_after = false;
};

struct IfBranch {
  ExprNodePtr condition; // nullptr for else branch
  std::vector<ASTNodePtr> body;
};

struct IfNode : ASTNode {
  std::vector<IfBranch> branches;
  bool strip_before = false;
  bool strip_after = false;
};

struct ForNode : ASTNode {
  std::string var_name;
  ExprNodePtr iterable;
  std::vector<ASTNodePtr> body;
  bool strip_before = false;
  bool strip_after = false;
};

struct SetNode : ASTNode {
  std::string var_name;
  ExprNodePtr value;
  bool strip_before = false;
  bool strip_after = false;
};

// Expression nodes
struct StringLiteral : ExprNode {
  std::string value;
};

struct IntegerLiteral : ExprNode {
  int value;
};

struct BoolLiteral : ExprNode {
  bool value;
};

struct NoneLiteral : ExprNode {};

struct VariableExpr : ExprNode {
  std::string name;
};

struct AttributeExpr : ExprNode {
  ExprNodePtr object;
  std::string attribute;
};

struct IndexExpr : ExprNode {
  ExprNodePtr object;
  ExprNodePtr index;
};

struct BinaryExpr : ExprNode {
  std::string op; // "+", "==", "!=", "and", "or", "%"
  ExprNodePtr left;
  ExprNodePtr right;
};

struct UnaryExpr : ExprNode {
  std::string op; // "not"
  ExprNodePtr operand;
};

struct FilterExpr : ExprNode {
  ExprNodePtr value;
  std::string filter_name;
};

struct IsDefinedExpr : ExprNode {
  ExprNodePtr value;
};

struct FunctionCallExpr : ExprNode {
  std::string name;
  std::vector<ExprNodePtr> args;
};

// ============================================================================
// Parser
// ============================================================================
class Parser {
public:
  explicit Parser(const std::vector<Token> &tokens) :
    tokens_(tokens), pos_(0) {}

  std::vector<ASTNodePtr> parse() {
    std::vector<ASTNodePtr> nodes;
    parseBody(nodes, {});
    return nodes;
  }

private:
  const Token &current() const { return tokens_[pos_]; }

  const Token &advance() { return tokens_[pos_++]; }

  bool check(TokenType type) const { return current().type == type; }

  bool matchToken(TokenType type) {
    if (check(type)) {
      advance();
      return true;
    }
    return false;
  }

  Token expect(TokenType type) {
    if (!check(type)) {
      throw std::runtime_error(
        "ChatTemplate parser: unexpected token '" + current().value +
        "', expected type " + std::to_string(static_cast<int>(type)));
    }
    return advance();
  }

  void parseBody(std::vector<ASTNodePtr> &nodes,
                 const std::vector<TokenType> &stop_keywords) {
    while (pos_ < tokens_.size() && current().type != TokenType::END_OF_INPUT) {
      if (current().type == TokenType::TEXT) {
        auto node = std::make_shared<TextNode>();
        node->text = current().value;
        nodes.push_back(node);
        advance();
      } else if (current().type == TokenType::EXPRESSION_START) {
        nodes.push_back(parseOutput());
      } else if (current().type == TokenType::STATEMENT_START) {
        // Peek at the keyword after {%
        size_t save = pos_;
        advance(); // skip STATEMENT_START

        // Check if this is a stop keyword
        bool is_stop = false;
        for (auto sk : stop_keywords) {
          if (check(sk)) {
            is_stop = true;
            break;
          }
        }

        if (is_stop) {
          pos_ = save; // rewind
          return;
        }

        pos_ = save; // rewind
        parseStatement(nodes);
      } else {
        advance(); // skip unexpected
      }
    }
  }

  ASTNodePtr parseOutput() {
    auto node = std::make_shared<OutputNode>();
    Token start = expect(TokenType::EXPRESSION_START);
    node->strip_before = start.strip_before;
    node->expr = parseExpression();
    Token end = expect(TokenType::EXPRESSION_END);
    node->strip_after = end.strip_after;
    return node;
  }

  void parseStatement(std::vector<ASTNodePtr> &nodes) {
    Token start = expect(TokenType::STATEMENT_START);
    bool strip_before = start.strip_before;

    if (check(TokenType::IF)) {
      nodes.push_back(parseIf(strip_before));
    } else if (check(TokenType::FOR)) {
      nodes.push_back(parseFor(strip_before));
    } else if (check(TokenType::SET)) {
      nodes.push_back(parseSet(strip_before));
    } else {
      // Unknown statement - skip to end
      while (pos_ < tokens_.size() &&
             current().type != TokenType::STATEMENT_END) {
        advance();
      }
      if (check(TokenType::STATEMENT_END))
        advance();
    }
  }

  ASTNodePtr parseIf(bool strip_before) {
    auto node = std::make_shared<IfNode>();
    node->strip_before = strip_before;

    // Parse: if <expr> %}
    expect(TokenType::IF);
    IfBranch branch;
    branch.condition = parseExpression();
    Token end = expect(TokenType::STATEMENT_END);
    node->strip_after = end.strip_after;

    // Parse body until elif/else/endif
    parseBody(branch.body,
              {TokenType::ELIF, TokenType::ELSE, TokenType::ENDIF});
    node->branches.push_back(branch);

    // Parse elif/else branches
    while (pos_ < tokens_.size() && current().type == TokenType::STATEMENT_START) {
      advance(); // skip {%

      if (check(TokenType::ELIF)) {
        advance(); // skip elif
        IfBranch elif_branch;
        elif_branch.condition = parseExpression();
        expect(TokenType::STATEMENT_END);
        parseBody(elif_branch.body,
                  {TokenType::ELIF, TokenType::ELSE, TokenType::ENDIF});
        node->branches.push_back(elif_branch);
      } else if (check(TokenType::ELSE)) {
        advance(); // skip else
        expect(TokenType::STATEMENT_END);
        IfBranch else_branch;
        else_branch.condition = nullptr; // no condition = else
        parseBody(else_branch.body, {TokenType::ENDIF});
        node->branches.push_back(else_branch);
      } else if (check(TokenType::ENDIF)) {
        advance(); // skip endif
        expect(TokenType::STATEMENT_END);
        break;
      } else {
        break;
      }
    }

    return node;
  }

  ASTNodePtr parseFor(bool strip_before) {
    auto node = std::make_shared<ForNode>();
    node->strip_before = strip_before;

    expect(TokenType::FOR);
    node->var_name = expect(TokenType::IDENTIFIER).value;
    expect(TokenType::IN);
    node->iterable = parseExpression();
    Token end = expect(TokenType::STATEMENT_END);
    node->strip_after = end.strip_after;

    parseBody(node->body, {TokenType::ENDFOR});

    // Consume endfor
    expect(TokenType::STATEMENT_START);
    expect(TokenType::ENDFOR);
    expect(TokenType::STATEMENT_END);

    return node;
  }

  ASTNodePtr parseSet(bool strip_before) {
    auto node = std::make_shared<SetNode>();
    node->strip_before = strip_before;

    expect(TokenType::SET);
    node->var_name = expect(TokenType::IDENTIFIER).value;
    expect(TokenType::ASSIGN);
    node->value = parseExpression();
    Token end = expect(TokenType::STATEMENT_END);
    node->strip_after = end.strip_after;

    return node;
  }

  // Expression parsing with precedence
  ExprNodePtr parseExpression() { return parseOr(); }

  ExprNodePtr parseOr() {
    auto left = parseAnd();
    while (check(TokenType::OR)) {
      advance();
      auto right = parseAnd();
      auto node = std::make_shared<BinaryExpr>();
      node->op = "or";
      node->left = left;
      node->right = right;
      left = node;
    }
    return left;
  }

  ExprNodePtr parseAnd() {
    auto left = parseNot();
    while (check(TokenType::AND)) {
      advance();
      auto right = parseNot();
      auto node = std::make_shared<BinaryExpr>();
      node->op = "and";
      node->left = left;
      node->right = right;
      left = node;
    }
    return left;
  }

  ExprNodePtr parseNot() {
    if (check(TokenType::NOT)) {
      advance();
      auto node = std::make_shared<UnaryExpr>();
      node->op = "not";
      node->operand = parseNot();
      return node;
    }
    return parseComparison();
  }

  ExprNodePtr parseComparison() {
    auto left = parseAddition();

    if (check(TokenType::EQ) || check(TokenType::NEQ)) {
      std::string op = advance().value;
      auto right = parseAddition();
      auto node = std::make_shared<BinaryExpr>();
      node->op = op;
      node->left = left;
      node->right = right;
      return node;
    }

    // "is defined" test
    if (check(TokenType::IS)) {
      advance();
      if (check(TokenType::IDENTIFIER) && current().value == "defined") {
        advance();
        auto node = std::make_shared<IsDefinedExpr>();
        node->value = left;
        return node;
      }
      // "is not defined"
      if (check(TokenType::NOT)) {
        advance();
        if (check(TokenType::IDENTIFIER) && current().value == "defined") {
          advance();
          auto def_node = std::make_shared<IsDefinedExpr>();
          def_node->value = left;
          auto not_node = std::make_shared<UnaryExpr>();
          not_node->op = "not";
          not_node->operand = def_node;
          return not_node;
        }
      }
    }

    return left;
  }

  ExprNodePtr parseAddition() {
    auto left = parseModulo();
    while (check(TokenType::PLUS)) {
      advance();
      auto right = parseModulo();
      auto node = std::make_shared<BinaryExpr>();
      node->op = "+";
      node->left = left;
      node->right = right;
      left = node;
    }
    return left;
  }

  ExprNodePtr parseModulo() {
    auto left = parseFilter();
    while (check(TokenType::PERCENT)) {
      advance();
      auto right = parseFilter();
      auto node = std::make_shared<BinaryExpr>();
      node->op = "%";
      node->left = left;
      node->right = right;
      left = node;
    }
    return left;
  }

  ExprNodePtr parseFilter() {
    auto left = parsePostfix();
    while (check(TokenType::PIPE)) {
      advance();
      std::string filter_name = expect(TokenType::IDENTIFIER).value;
      auto node = std::make_shared<FilterExpr>();
      node->value = left;
      node->filter_name = filter_name;
      left = node;
    }
    return left;
  }

  ExprNodePtr parsePostfix() {
    auto node = parsePrimary();
    while (true) {
      if (check(TokenType::DOT)) {
        advance();
        std::string attr = expect(TokenType::IDENTIFIER).value;
        auto access = std::make_shared<AttributeExpr>();
        access->object = node;
        access->attribute = attr;
        node = access;
      } else if (check(TokenType::LBRACKET)) {
        advance();
        auto index = parseExpression();
        expect(TokenType::RBRACKET);
        auto access = std::make_shared<IndexExpr>();
        access->object = node;
        access->index = index;
        node = access;
      } else {
        break;
      }
    }
    return node;
  }

  ExprNodePtr parsePrimary() {
    if (check(TokenType::STRING)) {
      auto node = std::make_shared<StringLiteral>();
      node->value = advance().value;
      return node;
    }
    if (check(TokenType::INTEGER)) {
      auto node = std::make_shared<IntegerLiteral>();
      node->value = std::stoi(advance().value);
      return node;
    }
    if (check(TokenType::TRUE_LIT)) {
      advance();
      auto node = std::make_shared<BoolLiteral>();
      node->value = true;
      return node;
    }
    if (check(TokenType::FALSE_LIT)) {
      advance();
      auto node = std::make_shared<BoolLiteral>();
      node->value = false;
      return node;
    }
    if (check(TokenType::NONE_LIT)) {
      advance();
      return std::make_shared<NoneLiteral>();
    }
    if (check(TokenType::IDENTIFIER)) {
      std::string name = advance().value;

      // Check for function call
      if (check(TokenType::LPAREN)) {
        advance();
        auto call = std::make_shared<FunctionCallExpr>();
        call->name = name;
        if (!check(TokenType::RPAREN)) {
          call->args.push_back(parseExpression());
          while (check(TokenType::COMMA)) {
            advance();
            call->args.push_back(parseExpression());
          }
        }
        expect(TokenType::RPAREN);
        return call;
      }

      auto node = std::make_shared<VariableExpr>();
      node->name = name;
      return node;
    }
    if (check(TokenType::LPAREN)) {
      advance();
      auto expr = parseExpression();
      expect(TokenType::RPAREN);
      return expr;
    }

    // Fallback: return an empty string literal
    return std::make_shared<StringLiteral>();
  }

  const std::vector<Token> &tokens_;
  size_t pos_;
};

// ============================================================================
// Evaluator
// ============================================================================
class Evaluator {
public:
  explicit Evaluator(const json &context) { scopes_.push_back(context); }

  std::string evaluate(const std::vector<ASTNodePtr> &nodes) {
    std::string result;
    for (size_t i = 0; i < nodes.size(); ++i) {
      std::string chunk = evalNode(nodes[i].get());

      // Handle whitespace stripping
      if (shouldStripBefore(nodes[i].get())) {
        // Strip trailing whitespace from result
        while (!result.empty() &&
               (result.back() == ' ' || result.back() == '\t' ||
                result.back() == '\n' || result.back() == '\r')) {
          result.pop_back();
        }
      }

      result += chunk;

      // Handle strip_after: strip leading whitespace of next text
      if (shouldStripAfter(nodes[i].get()) && i + 1 < nodes.size()) {
        auto *text = dynamic_cast<TextNode *>(nodes[i + 1].get());
        if (text) {
          size_t start = 0;
          while (start < text->text.size() &&
                 (text->text[start] == ' ' || text->text[start] == '\t' ||
                  text->text[start] == '\n' || text->text[start] == '\r')) {
            start++;
          }
          text->text = text->text.substr(start);
        }
      }
    }
    return result;
  }

private:
  bool shouldStripBefore(ASTNode *node) {
    if (auto *o = dynamic_cast<OutputNode *>(node))
      return o->strip_before;
    if (auto *i = dynamic_cast<IfNode *>(node))
      return i->strip_before;
    if (auto *f = dynamic_cast<ForNode *>(node))
      return f->strip_before;
    if (auto *s = dynamic_cast<SetNode *>(node))
      return s->strip_before;
    return false;
  }

  bool shouldStripAfter(ASTNode *node) {
    if (auto *o = dynamic_cast<OutputNode *>(node))
      return o->strip_after;
    if (auto *i = dynamic_cast<IfNode *>(node))
      return i->strip_after;
    if (auto *f = dynamic_cast<ForNode *>(node))
      return f->strip_after;
    if (auto *s = dynamic_cast<SetNode *>(node))
      return s->strip_after;
    return false;
  }

  std::string evalNode(ASTNode *node) {
    if (auto *text = dynamic_cast<TextNode *>(node)) {
      return text->text;
    }
    if (auto *output = dynamic_cast<OutputNode *>(node)) {
      json val = evalExpr(output->expr.get());
      return jsonToString(val);
    }
    if (auto *if_node = dynamic_cast<IfNode *>(node)) {
      return evalIf(if_node);
    }
    if (auto *for_node = dynamic_cast<ForNode *>(node)) {
      return evalFor(for_node);
    }
    if (auto *set_node = dynamic_cast<SetNode *>(node)) {
      json val = evalExpr(set_node->value.get());
      setVariable(set_node->var_name, val);
      return "";
    }
    return "";
  }

  std::string evalIf(IfNode *node) {
    for (auto &branch : node->branches) {
      if (!branch.condition || isTruthy(evalExpr(branch.condition.get()))) {
        return evaluate(branch.body);
      }
    }
    return "";
  }

  std::string evalFor(ForNode *node) {
    json iterable = evalExpr(node->iterable.get());
    if (!iterable.is_array())
      return "";

    std::string result;
    size_t size = iterable.size();

    for (size_t i = 0; i < size; ++i) {
      // Push new scope with loop variable
      json scope;
      scope[node->var_name] = iterable[i];

      // Loop context
      json loop;
      loop["index"] = static_cast<int>(i + 1);
      loop["index0"] = static_cast<int>(i);
      loop["first"] = (i == 0);
      loop["last"] = (i == size - 1);
      loop["length"] = static_cast<int>(size);
      scope["loop"] = loop;

      scopes_.push_back(scope);
      result += evaluate(node->body);
      scopes_.pop_back();
    }

    return result;
  }

  json evalExpr(ExprNode *node) {
    if (auto *str = dynamic_cast<StringLiteral *>(node)) {
      return str->value;
    }
    if (auto *num = dynamic_cast<IntegerLiteral *>(node)) {
      return num->value;
    }
    if (auto *b = dynamic_cast<BoolLiteral *>(node)) {
      return b->value;
    }
    if (dynamic_cast<NoneLiteral *>(node)) {
      return nullptr;
    }
    if (auto *var = dynamic_cast<VariableExpr *>(node)) {
      return lookupVariable(var->name);
    }
    if (auto *attr = dynamic_cast<AttributeExpr *>(node)) {
      json obj = evalExpr(attr->object.get());
      if (obj.is_object() && obj.contains(attr->attribute)) {
        return obj[attr->attribute];
      }
      return nullptr;
    }
    if (auto *idx = dynamic_cast<IndexExpr *>(node)) {
      json obj = evalExpr(idx->object.get());
      json index = evalExpr(idx->index.get());
      if (obj.is_array() && index.is_number_integer()) {
        int i = index.get<int>();
        if (i >= 0 && i < static_cast<int>(obj.size()))
          return obj[i];
      } else if (obj.is_object() && index.is_string()) {
        std::string key = index.get<std::string>();
        if (obj.contains(key))
          return obj[key];
      }
      return nullptr;
    }
    if (auto *bin = dynamic_cast<BinaryExpr *>(node)) {
      return evalBinary(bin);
    }
    if (auto *unary = dynamic_cast<UnaryExpr *>(node)) {
      if (unary->op == "not") {
        return !isTruthy(evalExpr(unary->operand.get()));
      }
    }
    if (auto *filter = dynamic_cast<FilterExpr *>(node)) {
      json val = evalExpr(filter->value.get());
      if (filter->filter_name == "trim" && val.is_string()) {
        std::string s = val.get<std::string>();
        // Trim whitespace
        size_t start = s.find_first_not_of(" \t\n\r");
        size_t end = s.find_last_not_of(" \t\n\r");
        if (start == std::string::npos)
          return "";
        return s.substr(start, end - start + 1);
      }
      if (filter->filter_name == "length") {
        if (val.is_array())
          return static_cast<int>(val.size());
        if (val.is_string())
          return static_cast<int>(val.get<std::string>().size());
        return 0;
      }
      if (filter->filter_name == "tojson") {
        return val.dump();
      }
      return val; // unknown filter, passthrough
    }
    if (auto *def = dynamic_cast<IsDefinedExpr *>(node)) {
      // Check if the variable exists in any scope
      if (auto *var = dynamic_cast<VariableExpr *>(def->value.get())) {
        for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
          if (it->contains(var->name))
            return true;
        }
        return false;
      }
      // For attribute access, check if the parent exists and has the attr
      if (auto *attr = dynamic_cast<AttributeExpr *>(def->value.get())) {
        json obj = evalExpr(attr->object.get());
        return obj.is_object() && obj.contains(attr->attribute);
      }
      return false;
    }
    if (auto *call = dynamic_cast<FunctionCallExpr *>(node)) {
      if (call->name == "raise_exception") {
        std::string msg = "Template error";
        if (!call->args.empty()) {
          json arg = evalExpr(call->args[0].get());
          if (arg.is_string())
            msg = arg.get<std::string>();
        }
        throw std::runtime_error("ChatTemplate: " + msg);
      }
      if (call->name == "namespace") {
        // Return an empty object (simplified namespace support)
        json ns = json::object();
        for (size_t i = 0; i + 1 < call->args.size(); i += 2) {
          // simplified: not fully supported
        }
        return ns;
      }
      return nullptr;
    }
    return nullptr;
  }

  json evalBinary(BinaryExpr *node) {
    json left = evalExpr(node->left.get());
    json right = evalExpr(node->right.get());

    if (node->op == "+") {
      if (left.is_string() && right.is_string()) {
        return left.get<std::string>() + right.get<std::string>();
      }
      if (left.is_number() && right.is_number()) {
        return left.get<int>() + right.get<int>();
      }
      // String + non-string: convert to string
      return jsonToString(left) + jsonToString(right);
    }
    if (node->op == "==") {
      return left == right;
    }
    if (node->op == "!=") {
      return left != right;
    }
    if (node->op == "%") {
      if (left.is_number_integer() && right.is_number_integer()) {
        int r = right.get<int>();
        if (r != 0)
          return left.get<int>() % r;
      }
      return 0;
    }
    if (node->op == "and") {
      return isTruthy(left) && isTruthy(right);
    }
    if (node->op == "or") {
      return isTruthy(left) || isTruthy(right);
    }
    return nullptr;
  }

  bool isTruthy(const json &val) {
    if (val.is_null())
      return false;
    if (val.is_boolean())
      return val.get<bool>();
    if (val.is_number_integer())
      return val.get<int>() != 0;
    if (val.is_string())
      return !val.get<std::string>().empty();
    if (val.is_array())
      return !val.empty();
    if (val.is_object())
      return !val.empty();
    return false;
  }

  std::string jsonToString(const json &val) {
    if (val.is_string())
      return val.get<std::string>();
    if (val.is_null())
      return "";
    if (val.is_boolean())
      return val.get<bool>() ? "True" : "False";
    if (val.is_number_integer())
      return std::to_string(val.get<int>());
    if (val.is_number_float())
      return std::to_string(val.get<double>());
    return val.dump();
  }

  json lookupVariable(const std::string &name) {
    // Search from innermost scope to outermost
    for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
      if (it->contains(name))
        return (*it)[name];
    }
    return nullptr;
  }

  void setVariable(const std::string &name, const json &value) {
    // Set in the current (innermost) scope
    if (!scopes_.empty()) {
      scopes_.back()[name] = value;
    }
  }

  std::vector<json> scopes_;
};

// ============================================================================
// ChatTemplate Implementation
// ============================================================================

ChatTemplate::ChatTemplate() : available_(false) {}

ChatTemplate ChatTemplate::fromFile(const std::string &tokenizer_config_path) {
  ChatTemplate tmpl;

  std::ifstream file(tokenizer_config_path);
  if (!file.is_open()) {
    std::cerr << "[ChatTemplate] Warning: cannot open " << tokenizer_config_path
              << std::endl;
    return tmpl;
  }

  json config;
  try {
    file >> config;
  } catch (const json::parse_error &e) {
    std::cerr << "[ChatTemplate] Warning: JSON parse error in "
              << tokenizer_config_path << ": " << e.what() << std::endl;
    return tmpl;
  }

  // Extract chat_template
  if (config.contains("chat_template")) {
    if (config["chat_template"].is_string()) {
      tmpl.template_str_ = config["chat_template"].get<std::string>();
    } else if (config["chat_template"].is_array()) {
      // Some models have an array of templates; use the first one
      for (const auto &entry : config["chat_template"]) {
        if (entry.is_object() && entry.contains("template")) {
          tmpl.template_str_ = entry["template"].get<std::string>();
          break;
        }
      }
    }
  }

  if (tmpl.template_str_.empty()) {
    return tmpl;
  }

  // Extract bos_token (can be string or object with "content" field)
  if (config.contains("bos_token")) {
    if (config["bos_token"].is_string()) {
      tmpl.bos_token_ = config["bos_token"].get<std::string>();
    } else if (config["bos_token"].is_object() &&
               config["bos_token"].contains("content")) {
      tmpl.bos_token_ = config["bos_token"]["content"].get<std::string>();
    }
  }

  // Extract eos_token
  if (config.contains("eos_token")) {
    if (config["eos_token"].is_string()) {
      tmpl.eos_token_ = config["eos_token"].get<std::string>();
    } else if (config["eos_token"].is_object() &&
               config["eos_token"].contains("content")) {
      tmpl.eos_token_ = config["eos_token"]["content"].get<std::string>();
    }
  }

  tmpl.available_ = true;
  return tmpl;
}

std::string ChatTemplate::apply(const std::vector<ChatMessage> &messages,
                                bool add_generation_prompt) const {
  if (!available_)
    return "";

  // Build context
  json context;
  json msgs = json::array();
  for (const auto &msg : messages) {
    json m;
    m["role"] = msg.role;
    m["content"] = msg.content;
    msgs.push_back(m);
  }
  context["messages"] = msgs;
  context["bos_token"] = bos_token_;
  context["eos_token"] = eos_token_;
  context["add_generation_prompt"] = add_generation_prompt;

  return render(template_str_, context);
}

std::string ChatTemplate::apply(const std::string &user_input,
                                bool add_generation_prompt) const {
  std::vector<ChatMessage> messages = {{"user", user_input}};
  return apply(messages, add_generation_prompt);
}

bool ChatTemplate::isAvailable() const { return available_; }

std::string ChatTemplate::getBosToken() const { return bos_token_; }

std::string ChatTemplate::getEosToken() const { return eos_token_; }

std::string ChatTemplate::render(const std::string &tmpl,
                                 const json &context) const {
  try {
    Lexer lexer(tmpl);
    auto tokens = lexer.tokenize();

    Parser parser(tokens);
    auto ast = parser.parse();

    Evaluator evaluator(context);
    return evaluator.evaluate(ast);
  } catch (const std::exception &e) {
    std::cerr << "[ChatTemplate] Render error: " << e.what() << std::endl;
    return "";
  }
}

} // namespace causallm
