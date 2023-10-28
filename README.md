# attoGPT

- A more minimal implementation of GPT architecture than others I have seen.
- No "tensors"
- Probably not the best written code
- Pure standard library Rust
- Currently haven't gotten loss below 3 on the Shakespeare looking dataset I copied from somewhere I forgot. It remains to be seen whether I need a more sophisticated optimizer, to correct something in the implementation, or to train for longer.
- I might add a tokenizer later rather than just having the thing predict bytes.
