/-- If `R` is countable, then any `R`-submodule spanned by a countable family of vectors is
countable. -/
instance {ι : Type*} [Countable R] [Countable ι] (v : ι → M) :
    Countable (Submodule.span R (Set.range v)) := by
  refine Set.countable_coe_iff.mpr (Set.Countable.mono ?_ (Set.countable_range
      (fun c : (ι →₀ R) => c.sum fun i _ => (c i) • v i)))
  /-
    M : Type u_1
    R : Type u_2
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    ι : Type u_3
    inst✝¹ : Countable R
    inst✝ : Countable ι
    v : ι → M
    ⊢ HasSubset.Subset (↑(Submodule.span R (Set.range v))) (Set.range fun c => c.s …
  -/
  exact fun _ h => Finsupp.mem_span_range_iff_exists_finsupp.mp (SetLike.mem_coe.mp h)
  /-
    🎉 no goals
  -/


