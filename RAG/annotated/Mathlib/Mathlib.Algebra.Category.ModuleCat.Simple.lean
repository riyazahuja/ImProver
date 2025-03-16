theorem simple_iff_isSimpleModule : Simple (of R M) ↔ IsSimpleModule R M :=
  (simple_iff_subobject_isSimpleOrder _).trans (subobjectModule (of R M)).isSimpleOrder_iff


theorem simple_iff_isSimpleModule' (M : ModuleCat R) : Simple M ↔ IsSimpleModule R M :=
  (Simple.iff_of_iso (ofSelfIso M).symm).trans simple_iff_isSimpleModule


/-- A simple module is a simple object in the category of modules. -/
instance simple_of_isSimpleModule [IsSimpleModule R M] : Simple (of R M) :=
  simple_iff_isSimpleModule.mpr ‹_›


/-- A simple object in the category of modules is a simple module. -/
instance isSimpleModule_of_simple (M : ModuleCat R) [Simple M] : IsSimpleModule R M :=
  simple_iff_isSimpleModule.mp (Simple.of_iso (ofSelfIso M))


/-- Any `k`-algebra module which is 1-dimensional over `k` is simple. -/
theorem simple_of_finrank_eq_one {k : Type*} [Field k] [Algebra k R] {V : ModuleCat R}
    (h : finrank k V = 1) : Simple V :=
  (simple_iff_isSimpleModule' V).mpr (is_simple_module_of_finrank_eq_one h)

