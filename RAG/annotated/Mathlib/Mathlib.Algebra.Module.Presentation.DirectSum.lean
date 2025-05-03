/-- The direct sum operations on `Relations A`. Given a family
`relations : ι → Relations A`, the type of generators and relations
in `directSum relations` are the corresponding `Sigma` types. -/
@[simps G R relation]
noncomputable def directSum : Relations A where
  G := Σ i, (relations i).G
  R := Σ i, (relations i).R
  relation := fun ⟨i, r⟩ ↦ Finsupp.embDomain (Function.Embedding.sigmaMk
      (β := fun i ↦ (relations i).G) i) ((relations i).relation r)


/-- Given an `A`-module `N` and a family `relations : ι → Relations A`,
the data of a solution of `Relations.directSum relations` in `N`
is equivalent to the data of a family of solutions of `relations i` in `N`
for all `i`. -/
@[simps]
def directSumEquiv :
    (Relations.directSum relations).Solution N ≃
      ∀ i, (relations i).Solution N where
  toFun s i :=
    { var := fun g ↦ s.var ⟨i, g⟩
      linearCombination_var_relation := fun r ↦ by
        /-
          A : Type u
          inst✝⁵ : Ring A
          ι : Type w
          inst✝⁴ : DecidableEq ι
          relations : ι → Module.Relations A
          M : ι → Type v
          inst✝³ : (i : ι) → AddCommGroup (M i)
          inst✝² : (i : ι) → Module A (M i)
          N : Type v
          inst✝¹ : AddCommGroup N
          inst✝ : Module A N
          s : (Module.Relations.directSum relations).Solution N
          i : ι
          r : (relations i).R
          ⊢ Eq ((Finsupp.linearCombination A fun g => s.var ⟨i, g⟩) ((relations i).relat …
        -/
        rw [← s.linearCombination_var_relation ⟨i, r⟩]
        /-
          A : Type u
          inst✝⁵ : Ring A
          ι : Type w
          inst✝⁴ : DecidableEq ι
          relations : ι → Module.Relations A
          M : ι → Type v
          inst✝³ : (i : ι) → AddCommGroup (M i)
          inst✝² : (i : ι) → Module A (M i)
          N : Type v
          inst✝¹ : AddCommGroup N
          inst✝ : Module A N
          s : (Module.Relations.directSum relations).Solution N
          i : ι
          r : (relations i).R
          ⊢ Eq ((Finsupp.linearCombination A fun g => s.var ⟨i, g⟩) ((relations i).relat …
        -/
        symm
        /-
          A : Type u
          inst✝⁵ : Ring A
          ι : Type w
          inst✝⁴ : DecidableEq ι
          relations : ι → Module.Relations A
          M : ι → Type v
          inst✝³ : (i : ι) → AddCommGroup (M i)
          inst✝² : (i : ι) → Module A (M i)
          N : Type v
          inst✝¹ : AddCommGroup N
          inst✝ : Module A N
          s : (Module.Relations.directSum relations).Solution N
          i : ι
          r : (relations i).R
          ⊢ Eq ((Finsupp.linearCombination A s.var) ((Module.Relations.directSum relatio …
        -/
        apply Finsupp.linearCombination_embDomain }
        /-
          🎉 no goals
        -/
  invFun t :=
    { var := fun ⟨i, g⟩ ↦ (t i).var g
      linearCombination_var_relation := fun ⟨i, r⟩ ↦ by
        /-
          A : Type u
          inst✝⁵ : Ring A
          ι : Type w
          inst✝⁴ : DecidableEq ι
          relations : ι → Module.Relations A
          M : ι → Type v
          inst✝³ : (i : ι) → AddCommGroup (M i)
          inst✝² : (i : ι) → Module A (M i)
          N : Type v
          inst✝¹ : AddCommGroup N
          inst✝ : Module A N
          t : (i : ι) → (relations i).Solution N
          x✝ : (Module.Relations.directSum relations).R
          i : ι
          r : (relations i).R
          ⊢ Eq ((Finsupp.linearCombination A fun x => Module.Relations.Solution.directSu …
        -/
        rw [← (t i).linearCombination_var_relation r]
        /-
          A : Type u
          inst✝⁵ : Ring A
          ι : Type w
          inst✝⁴ : DecidableEq ι
          relations : ι → Module.Relations A
          M : ι → Type v
          inst✝³ : (i : ι) → AddCommGroup (M i)
          inst✝² : (i : ι) → Module A (M i)
          N : Type v
          inst✝¹ : AddCommGroup N
          inst✝ : Module A N
          t : (i : ι) → (relations i).Solution N
          x✝ : (Module.Relations.directSum relations).R
          i : ι
          r : (relations i).R
          ⊢ Eq ((Finsupp.linearCombination A fun x => Module.Relations.Solution.directSu …
        -/
        apply Finsupp.linearCombination_embDomain }
        /-
          🎉 no goals
        -/
  left_inv _ := rfl
  right_inv _ := rfl


/-- Given `solution : ∀ (i : ι), (relations i).Solution (M i)`, this is the
canonical solution of `Relations.directSum relations` in `⨁ i, M i`. -/
def directSum (solution : ∀ (i : ι), (relations i).Solution (M i)) :
    (Relations.directSum relations).Solution (⨁ i, M i) :=
  directSumEquiv.symm (fun i ↦ (solution i).postcomp (lof A ι M i))


@[simp]
lemma directSum_var (solution : ∀ (i : ι), (relations i).Solution (M i))
    (i : ι) (g : (relations i).G) :
    (directSum solution).var ⟨i, g⟩ = lof A ι M i ((solution i).var g) := rfl


/-- The direct sum admits a presentation by generators and relations. -/
noncomputable def directSum.isRepresentationCore :
    Solution.IsPresentationCore.{w'} (directSum solution) where
  desc s := DirectSum.toModule _ _ _ (fun i ↦ (h i).desc (directSumEquiv s i))
                        /-
                          A : Type u
                          inst✝⁷ : Ring A
                          ι : Type w
                          inst✝⁶ : DecidableEq ι
                          relations : ι → Module.Relations A
                          M : ι → Type v
                          inst✝⁵ : (i : ι) → AddCommGroup (M i)
                          inst✝⁴ : (i : ι) → Module A (M i)
                          N : Type v
                          inst✝³ : AddCommGroup N
                          inst✝² : Module A N
                          solution : (i : ι) → (relations i).Solution (M i)
                          h : ∀ (i : ι), (solution i).IsPresentation
                          N✝ : Type w'
                          inst✝¹ : AddCommGroup N✝
                          inst✝ : Module A N✝
                          s : (Module.Relations.directSum relations).Solution N✝
                          ⊢ Eq ((Module.Relations.Solution.directSum solution).postcomp ((fun {N} [AddCo …
                        -/
  postcomp_desc s := by ext ⟨i, g⟩; simp
                                    /-
                                      🎉 no goals
                                    -/
  postcomp_injective h' := by
    /-
      A : Type u
      inst✝⁷ : Ring A
      ι : Type w
      inst✝⁶ : DecidableEq ι
      relations : ι → Module.Relations A
      M : ι → Type v
      inst✝⁵ : (i : ι) → AddCommGroup (M i)
      inst✝⁴ : (i : ι) → Module A (M i)
      N : Type v
      inst✝³ : AddCommGroup N
      inst✝² : Module A N
      solution : (i : ι) → (relations i).Solution (M i)
      h : ∀ (i : ι), (solution i).IsPresentation
      N✝ : Type w'
      inst✝¹ : AddCommGroup N✝
      inst✝ : Module A N✝
      f✝ f'✝ : LinearMap (RingHom.id A) (DirectSum ι fun i => M i) N✝
      h' : Eq ((Module.Relations.Solution.directSum solution).postcomp f✝) ((Module. …
      ⊢ Eq f✝ f'✝
    -/
    ext i : 1
    /-
      case H
      A : Type u
      inst✝⁷ : Ring A
      ι : Type w
      inst✝⁶ : DecidableEq ι
      relations : ι → Module.Relations A
      M : ι → Type v
      inst✝⁵ : (i : ι) → AddCommGroup (M i)
      inst✝⁴ : (i : ι) → Module A (M i)
      N : Type v
      inst✝³ : AddCommGroup N
      inst✝² : Module A N
      solution : (i : ι) → (relations i).Solution (M i)
      h : ∀ (i : ι), (solution i).IsPresentation
      N✝ : Type w'
      inst✝¹ : AddCommGroup N✝
      inst✝ : Module A N✝
      f✝ f'✝ : LinearMap (RingHom.id A) (DirectSum ι fun i => M i) N✝
      h' : Eq ((Module.Relations.Solution.directSum solution).postcomp f✝) ((Module. …
      i : ι
      ⊢ Eq (f✝.comp (DirectSum.lof A ι M i)) (f'✝.comp (DirectSum.lof A ι M i))
    -/
    apply (h i).postcomp_injective
    /-
      case H
      A : Type u
      inst✝⁷ : Ring A
      ι : Type w
      inst✝⁶ : DecidableEq ι
      relations : ι → Module.Relations A
      M : ι → Type v
      inst✝⁵ : (i : ι) → AddCommGroup (M i)
      inst✝⁴ : (i : ι) → Module A (M i)
      N : Type v
      inst✝³ : AddCommGroup N
      inst✝² : Module A N
      solution : (i : ι) → (relations i).Solution (M i)
      h : ∀ (i : ι), (solution i).IsPresentation
      N✝ : Type w'
      inst✝¹ : AddCommGroup N✝
      inst✝ : Module A N✝
      f✝ f'✝ : LinearMap (RingHom.id A) (DirectSum ι fun i => M i) N✝
      h' : Eq ((Module.Relations.Solution.directSum solution).postcomp f✝) ((Module. …
      i : ι
      ⊢ Eq ((solution i).postcomp (f✝.comp (DirectSum.lof A ι M i))) ((solution i).p …
    -/
    ext g
    /-
      case H.var.h
      A : Type u
      inst✝⁷ : Ring A
      ι : Type w
      inst✝⁶ : DecidableEq ι
      relations : ι → Module.Relations A
      M : ι → Type v
      inst✝⁵ : (i : ι) → AddCommGroup (M i)
      inst✝⁴ : (i : ι) → Module A (M i)
      N : Type v
      inst✝³ : AddCommGroup N
      inst✝² : Module A N
      solution : (i : ι) → (relations i).Solution (M i)
      h : ∀ (i : ι), (solution i).IsPresentation
      N✝ : Type w'
      inst✝¹ : AddCommGroup N✝
      inst✝ : Module A N✝
      f✝ f'✝ : LinearMap (RingHom.id A) (DirectSum ι fun i => M i) N✝
      h' : Eq ((Module.Relations.Solution.directSum solution).postcomp f✝) ((Module. …
      i : ι
      g : (relations i).G
      ⊢ Eq (((solution i).postcomp (f✝.comp (DirectSum.lof A ι M i))).var g) (((solu …
    -/
    exact Solution.congr_var h' ⟨i, g⟩
    /-
      🎉 no goals
    -/


include h in
lemma directSum : (directSum solution).IsPresentation :=
  (directSum.isRepresentationCore h).isPresentation


/-- The obvious presentation of the module `⨁ i, M i` that is obtained from
the data of presentations of the module `M i` for each `i`. -/
@[simps! G R relation]
noncomputable def directSum (pres : ∀ (i : ι), Presentation A (M i)) :
    Presentation A (⨁ i, M i) :=
  ofIsPresentation
    (Relations.Solution.IsPresentation.directSum (fun i ↦ (pres i).toIsPresentation))


@[simp]
lemma directSum_var (pres : ∀ (i : ι), Presentation A (M i)) (i : ι) (g : (pres i).G):
    (directSum pres).var ⟨i, g⟩ = lof A ι M i ((pres i).var g) := rfl


/-- The obvious presentation of the module `ι →₀ N` that is deduced from a presentation
of the module `N`. -/
@[simps! G R relation]
noncomputable def finsupp : Presentation A (ι →₀ N) :=
  (directSum (fun (_ : ι) ↦ pres)).ofLinearEquiv (finsuppLequivDFinsupp _).symm


@[simp]
lemma finsupp_var (i : ι) (g : pres.G) :
    (finsupp pres ι).var ⟨i, g⟩ = Finsupp.single i (pres.var g) := by
  /-
    A : Type u
    inst✝⁴ : Ring A
    N : Type v
    inst✝³ : AddCommGroup N
    inst✝² : Module A N
    pres : Module.Presentation A N
    ι : Type w
    inst✝¹ : DecidableEq ι
    inst✝ : DecidableEq N
    i : ι
    g : pres.G
    ⊢ Eq ((pres.finsupp ι).var ⟨i, g⟩) (Finsupp.single i (pres.var g))
  -/
  apply (finsuppLequivDFinsupp A).injective
  /-
    case a
    A : Type u
    inst✝⁴ : Ring A
    N : Type v
    inst✝³ : AddCommGroup N
    inst✝² : Module A N
    pres : Module.Presentation A N
    ι : Type w
    inst✝¹ : DecidableEq ι
    inst✝ : DecidableEq N
    i : ι
    g : pres.G
    ⊢ Eq ((finsuppLequivDFinsupp A) ((pres.finsupp ι).var ⟨i, g⟩)) ((finsuppLequiv …
  -/
  erw [(finsuppLequivDFinsupp A).apply_symm_apply]
  /-
    case a
    A : Type u
    inst✝⁴ : Ring A
    N : Type v
    inst✝³ : AddCommGroup N
    inst✝² : Module A N
    pres : Module.Presentation A N
    ι : Type w
    inst✝¹ : DecidableEq ι
    inst✝ : DecidableEq N
    i : ι
    g : pres.G
    ⊢ Eq ((Module.Presentation.directSum fun x => pres).var ⟨i, g⟩) ((finsuppLequi …
  -/
  rw [directSum_var, finsuppLequivDFinsupp_apply_apply, Finsupp.toDFinsupp_single]
  /-
    case a
    A : Type u
    inst✝⁴ : Ring A
    N : Type v
    inst✝³ : AddCommGroup N
    inst✝² : Module A N
    pres : Module.Presentation A N
    ι : Type w
    inst✝¹ : DecidableEq ι
    inst✝ : DecidableEq N
    i : ι
    g : pres.G
    ⊢ Eq ((DirectSum.lof A ι (fun x => N) i) (pres.var g)) (DFinsupp.single i (pre …
  -/
  rfl
  /-
    🎉 no goals
  -/


