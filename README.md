# attoGPT

- A more minimal implementation of GPT's transformer architecture than others I have seen.
- Has methods for AdamW, RProp, and SGD optimizers. AdamW seems to be the best for this use case.
- No "tensors"
- Probably not the best written code. I'll clean it up a bit sometime.
- Pure standard library Rust
- Gets down to about 2.5 nats loss now after fixing a bug
- I might add a tokenizer later rather than just having the thing predict bytes.
