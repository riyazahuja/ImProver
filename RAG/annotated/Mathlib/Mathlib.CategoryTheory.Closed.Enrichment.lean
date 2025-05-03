/-- For `C` closed monoidal, build an instance of `C` as a `C`-category -/
scoped instance : EnrichedCategory C C where
  Hom x := (ihom x).obj
  id _ := id _
  comp _ _ _ := comp _ _ _
  assoc _ _ _ _ := assoc _ _ _ _


