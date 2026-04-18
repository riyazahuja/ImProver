/-!
This aggregate examples target used to import several external example
repositories. Some of those repositories now define names that also exist in
the pinned mathlib build, so importing all of them in one module can fail even
though the training/evaluation executable does not use this module.

Keep `Examples` as a buildable placeholder so bare `lake build` works; build
individual external packages directly when their examples are needed.
-/
