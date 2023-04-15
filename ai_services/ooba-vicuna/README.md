# Oobabooga Text Generation Web UI Server with Vicuna

1. Follow setup instructions from [Oobabooga Text Generation Web UI](https://github.com/oobabooga/text-generation-webui)
2. Get the [Vicuna 13B Model](https://github.com/lm-sys/FastChat/#vicuna-weights) and test it with the instructions from the repository above.
3. Set the `AI_MODEL` in your `.env` file to `ooba-vicuna`
4. Run Oobabooga Text Generation Web UI server with the following command in order to work with this.
    ``python3 server.py --model YOUR-VICUNA-MODEL --listen --no-stream``