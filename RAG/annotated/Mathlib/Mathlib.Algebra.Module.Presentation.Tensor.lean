/-- The tensor product of systems of linear equations. -/
@[simps]
noncomputable def tensor :
    Relations A where
  G := relations₁.G × relations₂.G
  R := Sum (relations₁.R × relations₂.G) (relations₁.G × relations₂.R)
  relation r := match r with
    | .inl ⟨r₁, g₂⟩ => Finsupp.embDomain (Function.Embedding.sectL relations₁.G g₂)
        (relations₁.relation r₁)
    | .inr ⟨g₁, r₂⟩ => Finsupp.embDomain (Function.Embedding.sectR g₁ relations₂.G)
        (relations₂.relation r₂)


/-- Given solutions in `M₁` and `M₂` to systems of linear equations, this is the obvious
solution to the tensor product of these systems in `M₁ ⊗[A] M₂`. -/
@[simps]
noncomputable def tensor : (relations₁.tensor relations₂).Solution (M₁ ⊗[A] M₂) where
  var := fun ⟨g₁, g₂⟩ => solution₁.var g₁ ⊗ₜ solution₂.var g₂
  linearCombination_var_relation := by
    /-
      A : Type u
      inst✝⁴ : CommRing A
      M₁ : Type v₁
      M₂ : Type v₂
      inst✝³ : AddCommGroup M₁
      inst✝² : AddCommGroup M₂
      inst✝¹ : Module A M₁
      inst✝ : Module A M₂
      relations₁ : Module.Relations A
      relations₂ : Module.Relations A
      solution₁ : relations₁.Solution M₁
      solution₂ : relations₂.Solution M₂
      ⊢ ∀ (r : (relations₁.tensor relations₂).R), Eq ((Finsupp.linearCombination A f …
    -/
    rintro (⟨r₁, g₂⟩ | ⟨g₁, r₂⟩)
      /-
        case inl.mk
        A : Type u
        inst✝⁴ : CommRing A
        M₁ : Type v₁
        M₂ : Type v₂
        inst✝³ : AddCommGroup M₁
        inst✝² : AddCommGroup M₂
        inst✝¹ : Module A M₁
        inst✝ : Module A M₂
        relations₁ : Module.Relations A
        relations₂ : Module.Relations A
        solution₁ : relations₁.Solution M₁
        solution₂ : relations₂.Solution M₂
        r₁ : relations₁.R
        g₂ : relations₂.G
        ⊢ Eq ((Finsupp.linearCombination A fun x => Module.Relations.Solution.tensor.m …
      -/
    · dsimp
      /-
        case inl.mk
        A : Type u
        inst✝⁴ : CommRing A
        M₁ : Type v₁
        M₂ : Type v₂
        inst✝³ : AddCommGroup M₁
        inst✝² : AddCommGroup M₂
        inst✝¹ : Module A M₁
        inst✝ : Module A M₂
        relations₁ : Module.Relations A
        relations₂ : Module.Relations A
        solution₁ : relations₁.Solution M₁
        solution₂ : relations₂.Solution M₂
        r₁ : relations₁.R
        g₂ : relations₂.G
        ⊢ Eq ((Finsupp.linearCombination A fun x => TensorProduct.tmul A (solution₁.va …
      -/
      rw [Finsupp.linearCombination_embDomain]
      exact (solution₁.postcomp (curry (TensorProduct.comm A M₂ M₁).toLinearMap
        (solution₂.var g₂))).linearCombination_var_relation r₁
      /-
        case inr.mk
        A : Type u
        inst✝⁴ : CommRing A
        M₁ : Type v₁
        M₂ : Type v₂
        inst✝³ : AddCommGroup M₁
        inst✝² : AddCommGroup M₂
        inst✝¹ : Module A M₁
        inst✝ : Module A M₂
        relations₁ : Module.Relations A
        relations₂ : Module.Relations A
        solution₁ : relations₁.Solution M₁
        solution₂ : relations₂.Solution M₂
        g₁ : relations₁.G
        r₂ : relations₂.R
        ⊢ Eq ((Finsupp.linearCombination A fun x => Module.Relations.Solution.tensor.m …
      -/
    · dsimp
      /-
        case inr.mk
        A : Type u
        inst✝⁴ : CommRing A
        M₁ : Type v₁
        M₂ : Type v₂
        inst✝³ : AddCommGroup M₁
        inst✝² : AddCommGroup M₂
        inst✝¹ : Module A M₁
        inst✝ : Module A M₂
        relations₁ : Module.Relations A
        relations₂ : Module.Relations A
        solution₁ : relations₁.Solution M₁
        solution₂ : relations₂.Solution M₂
        g₁ : relations₁.G
        r₂ : relations₂.R
        ⊢ Eq ((Finsupp.linearCombination A fun x => TensorProduct.tmul A (solution₁.va …
      -/
      rw [Finsupp.linearCombination_embDomain]
      /-
        case inr.mk
        A : Type u
        inst✝⁴ : CommRing A
        M₁ : Type v₁
        M₂ : Type v₂
        inst✝³ : AddCommGroup M₁
        inst✝² : AddCommGroup M₂
        inst✝¹ : Module A M₁
        inst✝ : Module A M₂
        relations₁ : Module.Relations A
        relations₂ : Module.Relations A
        solution₁ : relations₁.Solution M₁
        solution₂ : relations₂.Solution M₂
        g₁ : relations₁.G
        r₂ : relations₂.R
        ⊢ Eq ((Finsupp.linearCombination A (Function.comp (fun x => TensorProduct.tmul …
      -/
      exact (solution₂.postcomp (curry .id (solution₁.var g₁))).linearCombination_var_relation r₂
      /-
        🎉 no goals
      -/


/-- The tensor product of two modules admits a presentation by generators and relations. -/
noncomputable def isPresentationCoreTensor :
    Solution.IsPresentationCore.{w} (solution₁.tensor solution₂) where
  desc s := uncurry _ _ _ _ (h₁.desc
    { var := fun g₁ ↦ h₂.desc
        { var := fun g₂ ↦ s.var ⟨g₁, g₂⟩
          linearCombination_var_relation := fun r₂ ↦ by
            erw [← Finsupp.linearCombination_embDomain A
              (Function.Embedding.sectR g₁ relations₂.G)]
            /-
              A : Type u
              inst✝⁶ : CommRing A
              M₁ : Type v₁
              M₂ : Type v₂
              inst✝⁵ : AddCommGroup M₁
              inst✝⁴ : AddCommGroup M₂
              inst✝³ : Module A M₁
              inst✝² : Module A M₂
              relations₁ : Module.Relations A
              relations₂ : Module.Relations A
              solution₁ : relations₁.Solution M₁
              solution₂ : relations₂.Solution M₂
              h₁ : solution₁.IsPresentation
              h₂ : solution₂.IsPresentation
              N✝ : Type w
              inst✝¹ : AddCommGroup N✝
              inst✝ : Module A N✝
              s : (relations₁.tensor relations₂).Solution N✝
              g₁ : relations₁.G
              r₂ : relations₂.R
              ⊢ Eq ((Finsupp.linearCombination A s.var) (Finsupp.embDomain (Function.Embeddi …
            -/
            exact s.linearCombination_var_relation (.inr ⟨g₁, r₂⟩) }
            /-
              🎉 no goals
            -/
      linearCombination_var_relation := fun r₁ ↦ h₂.postcomp_injective (by
        /-
          A : Type u
          inst✝⁶ : CommRing A
          M₁ : Type v₁
          M₂ : Type v₂
          inst✝⁵ : AddCommGroup M₁
          inst✝⁴ : AddCommGroup M₂
          inst✝³ : Module A M₁
          inst✝² : Module A M₂
          relations₁ : Module.Relations A
          relations₂ : Module.Relations A
          solution₁ : relations₁.Solution M₁
          solution₂ : relations₂.Solution M₂
          h₁ : solution₁.IsPresentation
          h₂ : solution₂.IsPresentation
          N✝ : Type w
          inst✝¹ : AddCommGroup N✝
          inst✝ : Module A N✝
          s : (relations₁.tensor relations₂).Solution N✝
          r₁ : relations₁.R
          ⊢ Eq (solution₂.postcomp ((Finsupp.linearCombination A fun g₁ => h₂.desc { var …
        -/
        ext g₂
        /-
          case var.h
          A : Type u
          inst✝⁶ : CommRing A
          M₁ : Type v₁
          M₂ : Type v₂
          inst✝⁵ : AddCommGroup M₁
          inst✝⁴ : AddCommGroup M₂
          inst✝³ : Module A M₁
          inst✝² : Module A M₂
          relations₁ : Module.Relations A
          relations₂ : Module.Relations A
          solution₁ : relations₁.Solution M₁
          solution₂ : relations₂.Solution M₂
          h₁ : solution₁.IsPresentation
          h₂ : solution₂.IsPresentation
          N✝ : Type w
          inst✝¹ : AddCommGroup N✝
          inst✝ : Module A N✝
          s : (relations₁.tensor relations₂).Solution N✝
          r₁ : relations₁.R
          g₂ : relations₂.G
          ⊢ Eq ((solution₂.postcomp ((Finsupp.linearCombination A fun g₁ => h₂.desc { va …
        -/
        dsimp
        /-
          case var.h
          A : Type u
          inst✝⁶ : CommRing A
          M₁ : Type v₁
          M₂ : Type v₂
          inst✝⁵ : AddCommGroup M₁
          inst✝⁴ : AddCommGroup M₂
          inst✝³ : Module A M₁
          inst✝² : Module A M₂
          relations₁ : Module.Relations A
          relations₂ : Module.Relations A
          solution₁ : relations₁.Solution M₁
          solution₂ : relations₂.Solution M₂
          h₁ : solution₁.IsPresentation
          h₂ : solution₂.IsPresentation
          N✝ : Type w
          inst✝¹ : AddCommGroup N✝
          inst✝ : Module A N✝
          s : (relations₁.tensor relations₂).Solution N✝
          r₁ : relations₁.R
          g₂ : relations₂.G
          ⊢ Eq (((Finsupp.linearCombination A fun g₁ => h₂.desc { var := fun g₂ => s.var …
        -/
        erw [Finsupp.apply_linearCombination A (LinearMap.applyₗ (solution₂.var g₂))]
        /-
          case var.h
          A : Type u
          inst✝⁶ : CommRing A
          M₁ : Type v₁
          M₂ : Type v₂
          inst✝⁵ : AddCommGroup M₁
          inst✝⁴ : AddCommGroup M₂
          inst✝³ : Module A M₁
          inst✝² : Module A M₂
          relations₁ : Module.Relations A
          relations₂ : Module.Relations A
          solution₁ : relations₁.Solution M₁
          solution₂ : relations₂.Solution M₂
          h₁ : solution₁.IsPresentation
          h₂ : solution₂.IsPresentation
          N✝ : Type w
          inst✝¹ : AddCommGroup N✝
          inst✝ : Module A N✝
          s : (relations₁.tensor relations₂).Solution N✝
          r₁ : relations₁.R
          g₂ : relations₂.G
          ⊢ Eq ((Finsupp.linearCombination A (Function.comp ⇑(LinearMap.applyₗ (solution …
        -/
        have := s.linearCombination_var_relation (.inl ⟨r₁, g₂⟩)
        /-
          case var.h
          A : Type u
          inst✝⁶ : CommRing A
          M₁ : Type v₁
          M₂ : Type v₂
          inst✝⁵ : AddCommGroup M₁
          inst✝⁴ : AddCommGroup M₂
          inst✝³ : Module A M₁
          inst✝² : Module A M₂
          relations₁ : Module.Relations A
          relations₂ : Module.Relations A
          solution₁ : relations₁.Solution M₁
          solution₂ : relations₂.Solution M₂
          h₁ : solution₁.IsPresentation
          h₂ : solution₂.IsPresentation
          N✝ : Type w
          inst✝¹ : AddCommGroup N✝
          inst✝ : Module A N✝
          s : (relations₁.tensor relations₂).Solution N✝
          r₁ : relations₁.R
          g₂ : relations₂.G
          this : Eq ((Finsupp.linearCombination A s.var) ((relations₁.tensor relations₂) …
          ⊢ Eq ((Finsupp.linearCombination A (Function.comp ⇑(LinearMap.applyₗ (solution …
        -/
        erw [Finsupp.linearCombination_embDomain] at this
        /-
          case var.h
          A : Type u
          inst✝⁶ : CommRing A
          M₁ : Type v₁
          M₂ : Type v₂
          inst✝⁵ : AddCommGroup M₁
          inst✝⁴ : AddCommGroup M₂
          inst✝³ : Module A M₁
          inst✝² : Module A M₂
          relations₁ : Module.Relations A
          relations₂ : Module.Relations A
          solution₁ : relations₁.Solution M₁
          solution₂ : relations₂.Solution M₂
          h₁ : solution₁.IsPresentation
          h₂ : solution₂.IsPresentation
          N✝ : Type w
          inst✝¹ : AddCommGroup N✝
          inst✝ : Module A N✝
          s : (relations₁.tensor relations₂).Solution N✝
          r₁ : relations₁.R
          g₂ : relations₂.G
          this : Eq ((Finsupp.linearCombination A (Function.comp s.var ⇑(Function.Embedd …
          ⊢ Eq ((Finsupp.linearCombination A (Function.comp ⇑(LinearMap.applyₗ (solution …
        -/
        convert this
        /-
          case h.e'_2.h.e'_5.h.e'_7
          A : Type u
          inst✝⁶ : CommRing A
          M₁ : Type v₁
          M₂ : Type v₂
          inst✝⁵ : AddCommGroup M₁
          inst✝⁴ : AddCommGroup M₂
          inst✝³ : Module A M₁
          inst✝² : Module A M₂
          relations₁ : Module.Relations A
          relations₂ : Module.Relations A
          solution₁ : relations₁.Solution M₁
          solution₂ : relations₂.Solution M₂
          h₁ : solution₁.IsPresentation
          h₂ : solution₂.IsPresentation
          N✝ : Type w
          inst✝¹ : AddCommGroup N✝
          inst✝ : Module A N✝
          s : (relations₁.tensor relations₂).Solution N✝
          r₁ : relations₁.R
          g₂ : relations₂.G
          this : Eq ((Finsupp.linearCombination A (Function.comp s.var ⇑(Function.Embedd …
          ⊢ Eq (Function.comp ⇑(LinearMap.applyₗ (solution₂.var g₂)) fun g₁ => h₂.desc { …
        -/
        ext g₁
        /-
          case h.e'_2.h.e'_5.h.e'_7.h
          A : Type u
          inst✝⁶ : CommRing A
          M₁ : Type v₁
          M₂ : Type v₂
          inst✝⁵ : AddCommGroup M₁
          inst✝⁴ : AddCommGroup M₂
          inst✝³ : Module A M₁
          inst✝² : Module A M₂
          relations₁ : Module.Relations A
          relations₂ : Module.Relations A
          solution₁ : relations₁.Solution M₁
          solution₂ : relations₂.Solution M₂
          h₁ : solution₁.IsPresentation
          h₂ : solution₂.IsPresentation
          N✝ : Type w
          inst✝¹ : AddCommGroup N✝
          inst✝ : Module A N✝
          s : (relations₁.tensor relations₂).Solution N✝
          r₁ : relations₁.R
          g₂ : relations₂.G
          this : Eq ((Finsupp.linearCombination A (Function.comp s.var ⇑(Function.Embedd …
          g₁ : relations₁.G
          ⊢ Eq (Function.comp (⇑(LinearMap.applyₗ (solution₂.var g₂))) (fun g₁ => h₂.des …
        -/
        simp) })
        /-
          🎉 no goals
        -/
                        /-
                          A : Type u
                          inst✝⁶ : CommRing A
                          M₁ : Type v₁
                          M₂ : Type v₂
                          inst✝⁵ : AddCommGroup M₁
                          inst✝⁴ : AddCommGroup M₂
                          inst✝³ : Module A M₁
                          inst✝² : Module A M₂
                          relations₁ : Module.Relations A
                          relations₂ : Module.Relations A
                          solution₁ : relations₁.Solution M₁
                          solution₂ : relations₂.Solution M₂
                          h₁ : solution₁.IsPresentation
                          h₂ : solution₂.IsPresentation
                          N✝ : Type w
                          inst✝¹ : AddCommGroup N✝
                          inst✝ : Module A N✝
                          x✝ : (relations₁.tensor relations₂).Solution N✝
                          ⊢ Eq ((solution₁.tensor solution₂).postcomp ((fun {N} [AddCommGroup N] [Module …
                        -/
  postcomp_desc _ := by aesop
                        /-
                          🎉 no goals
                        -/
  postcomp_injective h := curry_injective (h₁.postcomp_injective (by
    /-
      A : Type u
      inst✝⁶ : CommRing A
      M₁ : Type v₁
      M₂ : Type v₂
      inst✝⁵ : AddCommGroup M₁
      inst✝⁴ : AddCommGroup M₂
      inst✝³ : Module A M₁
      inst✝² : Module A M₂
      relations₁ : Module.Relations A
      relations₂ : Module.Relations A
      solution₁ : relations₁.Solution M₁
      solution₂ : relations₂.Solution M₂
      h₁ : solution₁.IsPresentation
      h₂ : solution₂.IsPresentation
      N✝ : Type w
      inst✝¹ : AddCommGroup N✝
      inst✝ : Module A N✝
      f✝ f'✝ : LinearMap (RingHom.id A) (TensorProduct A M₁ M₂) N✝
      h : Eq ((solution₁.tensor solution₂).postcomp f✝) ((solution₁.tensor solution₂ …
      ⊢ Eq (solution₁.postcomp (TensorProduct.curry f✝)) (solution₁.postcomp (Tensor …
    -/
    ext g₁ : 2
    /-
      case var.h
      A : Type u
      inst✝⁶ : CommRing A
      M₁ : Type v₁
      M₂ : Type v₂
      inst✝⁵ : AddCommGroup M₁
      inst✝⁴ : AddCommGroup M₂
      inst✝³ : Module A M₁
      inst✝² : Module A M₂
      relations₁ : Module.Relations A
      relations₂ : Module.Relations A
      solution₁ : relations₁.Solution M₁
      solution₂ : relations₂.Solution M₂
      h₁ : solution₁.IsPresentation
      h₂ : solution₂.IsPresentation
      N✝ : Type w
      inst✝¹ : AddCommGroup N✝
      inst✝ : Module A N✝
      f✝ f'✝ : LinearMap (RingHom.id A) (TensorProduct A M₁ M₂) N✝
      h : Eq ((solution₁.tensor solution₂).postcomp f✝) ((solution₁.tensor solution₂ …
      g₁ : relations₁.G
      ⊢ Eq ((solution₁.postcomp (TensorProduct.curry f✝)).var g₁) ((solution₁.postco …
    -/
    refine h₂.postcomp_injective ?_
    /-
      case var.h
      A : Type u
      inst✝⁶ : CommRing A
      M₁ : Type v₁
      M₂ : Type v₂
      inst✝⁵ : AddCommGroup M₁
      inst✝⁴ : AddCommGroup M₂
      inst✝³ : Module A M₁
      inst✝² : Module A M₂
      relations₁ : Module.Relations A
      relations₂ : Module.Relations A
      solution₁ : relations₁.Solution M₁
      solution₂ : relations₂.Solution M₂
      h₁ : solution₁.IsPresentation
      h₂ : solution₂.IsPresentation
      N✝ : Type w
      inst✝¹ : AddCommGroup N✝
      inst✝ : Module A N✝
      f✝ f'✝ : LinearMap (RingHom.id A) (TensorProduct A M₁ M₂) N✝
      h : Eq ((solution₁.tensor solution₂).postcomp f✝) ((solution₁.tensor solution₂ …
      g₁ : relations₁.G
      ⊢ Eq (solution₂.postcomp ((solution₁.postcomp (TensorProduct.curry f✝)).var g₁ …
    -/
    ext g₂
    /-
      case var.h.var.h
      A : Type u
      inst✝⁶ : CommRing A
      M₁ : Type v₁
      M₂ : Type v₂
      inst✝⁵ : AddCommGroup M₁
      inst✝⁴ : AddCommGroup M₂
      inst✝³ : Module A M₁
      inst✝² : Module A M₂
      relations₁ : Module.Relations A
      relations₂ : Module.Relations A
      solution₁ : relations₁.Solution M₁
      solution₂ : relations₂.Solution M₂
      h₁ : solution₁.IsPresentation
      h₂ : solution₂.IsPresentation
      N✝ : Type w
      inst✝¹ : AddCommGroup N✝
      inst✝ : Module A N✝
      f✝ f'✝ : LinearMap (RingHom.id A) (TensorProduct A M₁ M₂) N✝
      h : Eq ((solution₁.tensor solution₂).postcomp f✝) ((solution₁.tensor solution₂ …
      g₁ : relations₁.G
      g₂ : relations₂.G
      ⊢ Eq ((solution₂.postcomp ((solution₁.postcomp (TensorProduct.curry f✝)).var g …
    -/
    exact congr_var h ⟨g₁, g₂⟩))
    /-
      🎉 no goals
    -/


include h₁ h₂ in
lemma IsPresentation.tensor : (solution₁.tensor solution₂).IsPresentation :=
  (isPresentationCoreTensor h₁ h₂).isPresentation


/-- The presentation of the `A`-module `M₁ ⊗[A] M₂` that is deduced from
a presentation of `M₁` and a presentation of `M₂`. -/
@[simps!]
noncomputable def tensor : Presentation A (M₁ ⊗[A] M₂) where
  toSolution := pres₁.toSolution.tensor pres₂.toSolution
  toIsPresentation := pres₁.toIsPresentation.tensor pres₂.toIsPresentation


