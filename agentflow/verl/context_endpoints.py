"""Tokenizer endpoints for the training model's AgentFlow proxy."""


def register_tokenizer_routes(app, tokenizer):
    from flask import jsonify, request

    @app.route("/tokenize", methods=["POST"])
    @app.route("/v1/tokenize", methods=["POST"])
    def tokenize():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("prompt"), str):
            return jsonify(error="A string prompt is required"), 400
        add_special_tokens = payload.get("add_special_tokens", False)
        if not isinstance(add_special_tokens, bool):
            return jsonify(error="add_special_tokens must be a boolean"), 400
        # This is the same tokenizer already used by the training daemon.
        # Do not send metadata/tokenizer requests to model inference workers.
        tokens = tokenizer.encode(payload["prompt"], add_special_tokens=add_special_tokens)
        return jsonify(count=len(tokens), tokens=tokens)
