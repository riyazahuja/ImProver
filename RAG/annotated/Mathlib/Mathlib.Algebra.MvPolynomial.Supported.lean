/-- The set of polynomials whose variables are contained in `s` as a `Subalgebra` over `R`. -/
noncomputable def supported (s : Set σ) : Subalgebra R (MvPolynomial σ R) :=
  Algebra.adjoin R (X '' s)


theorem supported_eq_range_rename (s : Set σ) : supported R s = (rename ((↑) : s → σ)).range := by
  /-
    σ : Type u_1
    R : Type u
    inst✝ : CommSemiring R
    s : Set σ
    ⊢ Eq (MvPolynomial.supported R s) (MvPolynomial.rename Subtype.val).range
  -/
  rw [supported, Set.image_eq_range, adjoin_range_eq_range_aeval, rename]
  /-
    σ : Type u_1
    R : Type u
    inst✝ : CommSemiring R
    s : Set σ
    ⊢ Eq (MvPolynomial.aeval fun x => MvPolynomial.X ↑x).range (MvPolynomial.aeval …
  -/
  congr
  /-
    🎉 no goals
  -/


/-- The isomorphism between the subalgebra of polynomials supported by `s` and
`MvPolynomial s R`. -/
noncomputable def supportedEquivMvPolynomial (s : Set σ) : supported R s ≃ₐ[R] MvPolynomial s R :=
  (Subalgebra.equivOfEq _ _ (supported_eq_range_rename s)).trans
    (AlgEquiv.ofInjective (rename ((↑) : s → σ)) (rename_injective _ Subtype.val_injective)).symm


@[simp]
theorem supportedEquivMvPolynomial_symm_C (s : Set σ) (x : R) :
    (supportedEquivMvPolynomial s).symm (C x) = algebraMap R (supported R s) x := by
  /-
    σ : Type u_1
    R : Type u
    inst✝ : CommSemiring R
    s : Set σ
    x : R
    ⊢ Eq ((MvPolynomial.supportedEquivMvPolynomial s).symm (MvPolynomial.C x)) ((a …
  -/
  ext1
  /-
    case a
    σ : Type u_1
    R : Type u
    inst✝ : CommSemiring R
    s : Set σ
    x : R
    ⊢ Eq ↑((MvPolynomial.supportedEquivMvPolynomial s).symm (MvPolynomial.C x)) ↑( …
  -/
  simp [supportedEquivMvPolynomial, MvPolynomial.algebraMap_eq]
  /-
    🎉 no goals
  -/


@[simp]
theorem supportedEquivMvPolynomial_symm_X (s : Set σ) (i : s) :
    (↑((supportedEquivMvPolynomial s).symm (X i : MvPolynomial s R)) : MvPolynomial σ R) = X ↑i :=
     /-
       σ : Type u_1
       R : Type u
       inst✝ : CommSemiring R
       s : Set σ
       i : ↑s
       ⊢ Eq (↑((MvPolynomial.supportedEquivMvPolynomial s).symm (MvPolynomial.X i)))  …
     -/
  by simp [supportedEquivMvPolynomial]
     /-
       🎉 no goals
     -/


theorem mem_supported : p ∈ supported R s ↔ ↑p.vars ⊆ s := by
  classical
  rw [supported_eq_range_rename, AlgHom.mem_range]
  constructor
  · rintro ⟨p, rfl⟩
    refine _root_.trans (Finset.coe_subset.2 (vars_rename _ _)) ?_
    simp
  · intro hs
    exact exists_rename_eq_of_vars_subset_range p ((↑) : s → σ) Subtype.val_injective (by simpa)


theorem supported_eq_vars_subset : (supported R s : Set (MvPolynomial σ R)) = { p | ↑p.vars ⊆ s } :=
  Set.ext fun _ ↦ mem_supported


@[simp]
theorem mem_supported_vars (p : MvPolynomial σ R) : p ∈ supported R (↑p.vars : Set σ) := by
  /-
    σ : Type u_1
    R : Type u
    inst✝ : CommSemiring R
    p : MvPolynomial σ R
    ⊢ Membership.mem (MvPolynomial.supported R ↑p.vars) p
  -/
  rw [mem_supported]
  /-
    🎉 no goals
  -/


theorem supported_eq_adjoin_X : supported R s = Algebra.adjoin R (X '' s) := rfl


@[simp]
theorem supported_univ : supported R (Set.univ : Set σ) = ⊤ := by
  /-
    σ : Type u_1
    R : Type u
    inst✝ : CommSemiring R
    ⊢ Eq (MvPolynomial.supported R Set.univ) Top.top
  -/
  simp [Algebra.eq_top_iff, mem_supported]
  /-
    🎉 no goals
  -/


@[simp]
                                                            /-
                                                              σ : Type u_1
                                                              R : Type u
                                                              inst✝ : CommSemiring R
                                                              ⊢ Eq (MvPolynomial.supported R EmptyCollection.emptyCollection) Bot.bot
                                                            -/
theorem supported_empty : supported R (∅ : Set σ) = ⊥ := by simp [supported_eq_adjoin_X]
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem supported_mono (st : s ⊆ t) : supported R s ≤ supported R t :=
  Algebra.adjoin_mono (Set.image_subset _ st)


@[simp]
theorem X_mem_supported [Nontrivial R] {i : σ} : X i ∈ supported R s ↔ i ∈ s := by
  /-
    σ : Type u_1
    R : Type u
    inst✝¹ : CommSemiring R
    s : Set σ
    inst✝ : Nontrivial R
    i : σ
    ⊢ Iff (Membership.mem (MvPolynomial.supported R s) (MvPolynomial.X i)) (Member …
  -/
  simp [mem_supported]
  /-
    🎉 no goals
  -/


@[simp]
theorem supported_le_supported_iff [Nontrivial R] : supported R s ≤ supported R t ↔ s ⊆ t := by
  /-
    σ : Type u_1
    R : Type u
    inst✝¹ : CommSemiring R
    s t : Set σ
    inst✝ : Nontrivial R
    ⊢ Iff (LE.le (MvPolynomial.supported R s) (MvPolynomial.supported R t)) (HasSu …
  -/
  constructor
    /-
      case mp
      σ : Type u_1
      R : Type u
      inst✝¹ : CommSemiring R
      s t : Set σ
      inst✝ : Nontrivial R
      ⊢ LE.le (MvPolynomial.supported R s) (MvPolynomial.supported R t) → HasSubset. …
    -/
  · intro h i
    /-
      case mp
      σ : Type u_1
      R : Type u
      inst✝¹ : CommSemiring R
      s t : Set σ
      inst✝ : Nontrivial R
      h : LE.le (MvPolynomial.supported R s) (MvPolynomial.supported R t)
      i : σ
      ⊢ Membership.mem s i → Membership.mem t i
    -/
    simpa using @h (X i)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      σ : Type u_1
      R : Type u
      inst✝¹ : CommSemiring R
      s t : Set σ
      inst✝ : Nontrivial R
      ⊢ HasSubset.Subset s t → LE.le (MvPolynomial.supported R s) (MvPolynomial.supp …
    -/
  · exact supported_mono
    /-
      🎉 no goals
    -/


theorem supported_strictMono [Nontrivial R] :
    StrictMono (supported R : Set σ → Subalgebra R (MvPolynomial σ R)) :=
  strictMono_of_le_iff_le fun _ _ ↦ supported_le_supported_iff.symm


theorem exists_restrict_to_vars (R : Type*) [CommRing R] {F : MvPolynomial σ ℤ}
    (hF : ↑F.vars ⊆ s) : ∃ f : (s → R) → R, ∀ x : σ → R, f (x ∘ (↑) : s → R) = aeval x F := by
  /-
    σ : Type u_1
    s : Set σ
    R : Type u_2
    inst✝ : CommRing R
    F : MvPolynomial σ Int
    hF : HasSubset.Subset (↑F.vars) s
    ⊢ Exists fun f => ∀ (x : σ → R), Eq (f (Function.comp x Subtype.val)) ((MvPoly …
  -/
  rw [← mem_supported, supported_eq_range_rename, AlgHom.mem_range] at hF
  /-
    σ : Type u_1
    s : Set σ
    R : Type u_2
    inst✝ : CommRing R
    F : MvPolynomial σ Int
    hF : Exists fun x => Eq ((MvPolynomial.rename Subtype.val) x) F
    ⊢ Exists fun f => ∀ (x : σ → R), Eq (f (Function.comp x Subtype.val)) ((MvPoly …
  -/
  cases' hF with F' hF'
  /-
    case intro
    σ : Type u_1
    s : Set σ
    R : Type u_2
    inst✝ : CommRing R
    F : MvPolynomial σ Int
    F' : MvPolynomial (Subtype fun x => Membership.mem s x) Int
    hF' : Eq ((MvPolynomial.rename Subtype.val) F') F
    ⊢ Exists fun f => ∀ (x : σ → R), Eq (f (Function.comp x Subtype.val)) ((MvPoly …
  -/
  use fun z ↦ aeval z F'
  /-
    case h
    σ : Type u_1
    s : Set σ
    R : Type u_2
    inst✝ : CommRing R
    F : MvPolynomial σ Int
    F' : MvPolynomial (Subtype fun x => Membership.mem s x) Int
    hF' : Eq ((MvPolynomial.rename Subtype.val) F') F
    ⊢ ∀ (x : σ → R), Eq ((fun z => (MvPolynomial.aeval z) F') (Function.comp x Sub …
  -/
  intro x
  /-
    case h
    σ : Type u_1
    s : Set σ
    R : Type u_2
    inst✝ : CommRing R
    F : MvPolynomial σ Int
    F' : MvPolynomial (Subtype fun x => Membership.mem s x) Int
    hF' : Eq ((MvPolynomial.rename Subtype.val) F') F
    x : σ → R
    ⊢ Eq ((fun z => (MvPolynomial.aeval z) F') (Function.comp x Subtype.val)) ((Mv …
  -/
  simp only [← hF', aeval_rename]
  /-
    🎉 no goals
  -/


