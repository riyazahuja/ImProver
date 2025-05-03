/-- `@[reduce_mod_char]` is an attribute that tags lemmas for preprocessing and cleanup in the
`reduce_mod_char` tactic -/
initialize reduceModCharExt : SimpExtension ←
  /-
    ⊢ Lean.Name
  -/
  registerSimpAttr `reduce_mod_char
  /-
    🎉 no goals
  -/
    "lemmas for preprocessing and cleanup in the `reduce_mod_char` tactic"

