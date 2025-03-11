lemma finite [Finite pres.G] :
    Module.Finite A M :=
  Finite.of_surjective _ pres.surjective_π


lemma finitePresentation [Finite pres.G] [Finite pres.R] :
    Module.FinitePresentation A M :=
  Module.finitePresentation_of_surjective _ pres.surjective_π (by
    /-
      A : Type u
      inst✝⁴ : Ring A
      M : Type v
      inst✝³ : AddCommGroup M
      inst✝² : Module A M
      pres : Module.Presentation A M
      inst✝¹ : Finite pres.G
      inst✝ : Finite pres.R
      ⊢ (LinearMap.ker pres.π).FG
    -/
    rw [pres.ker_π]
    /-
      A : Type u
      inst✝⁴ : Ring A
      M : Type v
      inst✝³ : AddCommGroup M
      inst✝² : Module A M
      pres : Module.Presentation A M
      inst✝¹ : Finite pres.G
      inst✝ : Finite pres.R
      ⊢ (Submodule.span A (Set.range pres.relation)).FG
    -/
    exact Submodule.fg_span (Set.finite_range _))
    /-
      🎉 no goals
    -/


lemma finitePresentation_iff_exists_presentation :
    Module.FinitePresentation A M ↔
      ∃ (pres : Presentation.{w₀, w₁} A M), Finite pres.G ∧ Finite pres.R := by
  /-
    A : Type u
    inst✝² : Ring A
    M : Type v
    inst✝¹ : AddCommGroup M
    inst✝ : Module A M
    ⊢ Iff (Module.FinitePresentation A M) (Exists fun pres => And (Finite pres.G)  …
  -/
  constructor
    /-
      case mp
      A : Type u
      inst✝² : Ring A
      M : Type v
      inst✝¹ : AddCommGroup M
      inst✝ : Module A M
      ⊢ Module.FinitePresentation A M → Exists fun pres => And (Finite pres.G) (Fini …
    -/
  · intro
    obtain ⟨G : Type w₀, _, var, hG⟩ :=
      Submodule.fg_iff_exists_finite_generating_family.1
        (finite_def.1 (inferInstanceAs (Module.Finite A M)))
    obtain ⟨R : Type w₁, _, relation, hR⟩ :=
      Submodule.fg_iff_exists_finite_generating_family.1
        (Module.FinitePresentation.fg_ker (Finsupp.linearCombination A var) (by
          rw [← LinearMap.range_eq_top, Finsupp.range_linearCombination, hG]))
    exact
     ⟨{ G := G
        R := R
        relation := relation
        var := var
        linearCombination_var_relation := fun r ↦ by
          rw [Submodule.ext_iff] at hR
          exact (hR _).1 (Submodule.subset_span ⟨_, rfl⟩)
        toIsPresentation := by
          rw [Relations.Solution.isPresentation_iff]
          exact ⟨hG, hR.symm⟩ },
        inferInstance, inferInstance⟩
    /-
      case mpr
      A : Type u
      inst✝² : Ring A
      M : Type v
      inst✝¹ : AddCommGroup M
      inst✝ : Module A M
      ⊢ (Exists fun pres => And (Finite pres.G) (Finite pres.R)) → Module.FinitePres …
    -/
  · rintro ⟨pres, _, _⟩
    /-
      case mpr.intro.intro
      A : Type u
      inst✝² : Ring A
      M : Type v
      inst✝¹ : AddCommGroup M
      inst✝ : Module A M
      pres : Module.Presentation A M
      left✝ : Finite pres.G
      right✝ : Finite pres.R
      ⊢ Module.FinitePresentation A M
    -/
    exact pres.finitePresentation
    /-
      🎉 no goals
    -/


