theorem Algebra.map_leftMulMatrix_localization {ι : Type*} [Fintype ι] [DecidableEq ι]
    (b : Basis ι R S) (a : S) :
    (algebraMap R Rₘ).mapMatrix (leftMulMatrix b a) =
    leftMulMatrix (b.localizationLocalization Rₘ M Sₘ) (algebraMap S Sₘ a) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹⁴ : CommRing R
    inst✝¹³ : CommRing S
    inst✝¹² : Algebra R S
    Rₘ : Type u_3
    Sₘ : Type u_4
    inst✝¹¹ : CommRing Rₘ
    inst✝¹⁰ : Algebra R Rₘ
    inst✝⁹ : CommRing Sₘ
    inst✝⁸ : Algebra S Sₘ
    M : Submonoid R
    inst✝⁷ : IsLocalization M Rₘ
    inst✝⁶ : IsLocalization (Algebra.algebraMapSubmonoid S M) Sₘ
    inst✝⁵ : Algebra Rₘ Sₘ
    inst✝⁴ : Algebra R Sₘ
    inst✝³ : IsScalarTower R Rₘ Sₘ
    inst✝² : IsScalarTower R S Sₘ
    ι : Type u_5
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    b : Basis ι R S
    a : S
    ⊢ Eq ((algebraMap R Rₘ).mapMatrix ((Algebra.leftMulMatrix b) a)) ((Algebra.lef …
  -/
  ext i j
  simp only [Matrix.map_apply, RingHom.mapMatrix_apply, leftMulMatrix_eq_repr_mul, ← map_mul,
    Basis.localizationLocalization_apply, Basis.localizationLocalization_repr_algebraMap]


/-- Let `S` be an extension of `R` and `Rₘ Sₘ` be localizations at `M` of `R S` respectively.
Then the norm of `a : Sₘ` over `Rₘ` is the norm of `a : S` over `R` if `S` is free as `R`-module.
-/
theorem Algebra.norm_localization [Module.Free R S] [Module.Finite R S] (a : S) :
    Algebra.norm Rₘ (algebraMap S Sₘ a) = algebraMap R Rₘ (Algebra.norm R a) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹⁴ : CommRing R
    inst✝¹³ : CommRing S
    inst✝¹² : Algebra R S
    Rₘ : Type u_3
    Sₘ : Type u_4
    inst✝¹¹ : CommRing Rₘ
    inst✝¹⁰ : Algebra R Rₘ
    inst✝⁹ : CommRing Sₘ
    inst✝⁸ : Algebra S Sₘ
    M : Submonoid R
    inst✝⁷ : IsLocalization M Rₘ
    inst✝⁶ : IsLocalization (Algebra.algebraMapSubmonoid S M) Sₘ
    inst✝⁵ : Algebra Rₘ Sₘ
    inst✝⁴ : Algebra R Sₘ
    inst✝³ : IsScalarTower R Rₘ Sₘ
    inst✝² : IsScalarTower R S Sₘ
    inst✝¹ : Module.Free R S
    inst✝ : Module.Finite R S
    a : S
    ⊢ Eq ((Algebra.norm Rₘ) ((algebraMap S Sₘ) a)) ((algebraMap R Rₘ) ((Algebra.no …
  -/
  cases subsingleton_or_nontrivial R
    /-
      case inl
      R : Type u_1
      S : Type u_2
      inst✝¹⁴ : CommRing R
      inst✝¹³ : CommRing S
      inst✝¹² : Algebra R S
      Rₘ : Type u_3
      Sₘ : Type u_4
      inst✝¹¹ : CommRing Rₘ
      inst✝¹⁰ : Algebra R Rₘ
      inst✝⁹ : CommRing Sₘ
      inst✝⁸ : Algebra S Sₘ
      M : Submonoid R
      inst✝⁷ : IsLocalization M Rₘ
      inst✝⁶ : IsLocalization (Algebra.algebraMapSubmonoid S M) Sₘ
      inst✝⁵ : Algebra Rₘ Sₘ
      inst✝⁴ : Algebra R Sₘ
      inst✝³ : IsScalarTower R Rₘ Sₘ
      inst✝² : IsScalarTower R S Sₘ
      inst✝¹ : Module.Free R S
      inst✝ : Module.Finite R S
      a : S
      h✝ : Subsingleton R
      ⊢ Eq ((Algebra.norm Rₘ) ((algebraMap S Sₘ) a)) ((algebraMap R Rₘ) ((Algebra.no …
    -/
  · haveI : Subsingleton Rₘ := Module.subsingleton R Rₘ
    /-
      case inl
      R : Type u_1
      S : Type u_2
      inst✝¹⁴ : CommRing R
      inst✝¹³ : CommRing S
      inst✝¹² : Algebra R S
      Rₘ : Type u_3
      Sₘ : Type u_4
      inst✝¹¹ : CommRing Rₘ
      inst✝¹⁰ : Algebra R Rₘ
      inst✝⁹ : CommRing Sₘ
      inst✝⁸ : Algebra S Sₘ
      M : Submonoid R
      inst✝⁷ : IsLocalization M Rₘ
      inst✝⁶ : IsLocalization (Algebra.algebraMapSubmonoid S M) Sₘ
      inst✝⁵ : Algebra Rₘ Sₘ
      inst✝⁴ : Algebra R Sₘ
      inst✝³ : IsScalarTower R Rₘ Sₘ
      inst✝² : IsScalarTower R S Sₘ
      inst✝¹ : Module.Free R S
      inst✝ : Module.Finite R S
      a : S
      h✝ : Subsingleton R
      this : Subsingleton Rₘ
      ⊢ Eq ((Algebra.norm Rₘ) ((algebraMap S Sₘ) a)) ((algebraMap R Rₘ) ((Algebra.no …
    -/
    simp [eq_iff_true_of_subsingleton]
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u_1
    S : Type u_2
    inst✝¹⁴ : CommRing R
    inst✝¹³ : CommRing S
    inst✝¹² : Algebra R S
    Rₘ : Type u_3
    Sₘ : Type u_4
    inst✝¹¹ : CommRing Rₘ
    inst✝¹⁰ : Algebra R Rₘ
    inst✝⁹ : CommRing Sₘ
    inst✝⁸ : Algebra S Sₘ
    M : Submonoid R
    inst✝⁷ : IsLocalization M Rₘ
    inst✝⁶ : IsLocalization (Algebra.algebraMapSubmonoid S M) Sₘ
    inst✝⁵ : Algebra Rₘ Sₘ
    inst✝⁴ : Algebra R Sₘ
    inst✝³ : IsScalarTower R Rₘ Sₘ
    inst✝² : IsScalarTower R S Sₘ
    inst✝¹ : Module.Free R S
    inst✝ : Module.Finite R S
    a : S
    h✝ : Nontrivial R
    ⊢ Eq ((Algebra.norm Rₘ) ((algebraMap S Sₘ) a)) ((algebraMap R Rₘ) ((Algebra.no …
  -/
  let b := Module.Free.chooseBasis R S
  /-
    case inr
    R : Type u_1
    S : Type u_2
    inst✝¹⁴ : CommRing R
    inst✝¹³ : CommRing S
    inst✝¹² : Algebra R S
    Rₘ : Type u_3
    Sₘ : Type u_4
    inst✝¹¹ : CommRing Rₘ
    inst✝¹⁰ : Algebra R Rₘ
    inst✝⁹ : CommRing Sₘ
    inst✝⁸ : Algebra S Sₘ
    M : Submonoid R
    inst✝⁷ : IsLocalization M Rₘ
    inst✝⁶ : IsLocalization (Algebra.algebraMapSubmonoid S M) Sₘ
    inst✝⁵ : Algebra Rₘ Sₘ
    inst✝⁴ : Algebra R Sₘ
    inst✝³ : IsScalarTower R Rₘ Sₘ
    inst✝² : IsScalarTower R S Sₘ
    inst✝¹ : Module.Free R S
    inst✝ : Module.Finite R S
    a : S
    h✝ : Nontrivial R
    b : Basis (Module.Free.ChooseBasisIndex R S) R S := Module.Free.chooseBasis R S
    ⊢ Eq ((Algebra.norm Rₘ) ((algebraMap S Sₘ) a)) ((algebraMap R Rₘ) ((Algebra.no …
  -/
  letI := Classical.decEq (Module.Free.ChooseBasisIndex R S)
  rw [Algebra.norm_eq_matrix_det (b.localizationLocalization Rₘ M Sₘ),
    Algebra.norm_eq_matrix_det b, RingHom.map_det, ← Algebra.map_leftMulMatrix_localization]


variable {M} in
/-- The norm of `a : S` in `R` can be computed in `Sₘ`. -/
lemma Algebra.norm_eq_iff [Module.Free R S] [Module.Finite R S] {a : S} {b : R}
    (hM : M ≤ nonZeroDivisors R) : Algebra.norm R a = b ↔
      (Algebra.norm Rₘ) ((algebraMap S Sₘ) a) = algebraMap R Rₘ b :=
  ⟨fun h ↦ h.symm ▸ Algebra.norm_localization _ M _, fun h ↦
    IsLocalization.injective Rₘ hM <| h.symm ▸ (Algebra.norm_localization R M a).symm⟩


/-- Let `S` be an extension of `R` and `Rₘ Sₘ` be localizations at `M` of `R S` respectively.
Then the trace of `a : Sₘ` over `Rₘ` is the trace of `a : S` over `R` if `S` is free as `R`-module.
-/
theorem Algebra.trace_localization [Module.Free R S] [Module.Finite R S] (a : S) :
    Algebra.trace Rₘ Sₘ (algebraMap S Sₘ a) = algebraMap R Rₘ (Algebra.trace R S a) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹⁴ : CommRing R
    inst✝¹³ : CommRing S
    inst✝¹² : Algebra R S
    Rₘ : Type u_3
    Sₘ : Type u_4
    inst✝¹¹ : CommRing Rₘ
    inst✝¹⁰ : Algebra R Rₘ
    inst✝⁹ : CommRing Sₘ
    inst✝⁸ : Algebra S Sₘ
    M : Submonoid R
    inst✝⁷ : IsLocalization M Rₘ
    inst✝⁶ : IsLocalization (Algebra.algebraMapSubmonoid S M) Sₘ
    inst✝⁵ : Algebra Rₘ Sₘ
    inst✝⁴ : Algebra R Sₘ
    inst✝³ : IsScalarTower R Rₘ Sₘ
    inst✝² : IsScalarTower R S Sₘ
    inst✝¹ : Module.Free R S
    inst✝ : Module.Finite R S
    a : S
    ⊢ Eq ((Algebra.trace Rₘ Sₘ) ((algebraMap S Sₘ) a)) ((algebraMap R Rₘ) ((Algebr …
  -/
  cases subsingleton_or_nontrivial R
    /-
      case inl
      R : Type u_1
      S : Type u_2
      inst✝¹⁴ : CommRing R
      inst✝¹³ : CommRing S
      inst✝¹² : Algebra R S
      Rₘ : Type u_3
      Sₘ : Type u_4
      inst✝¹¹ : CommRing Rₘ
      inst✝¹⁰ : Algebra R Rₘ
      inst✝⁹ : CommRing Sₘ
      inst✝⁸ : Algebra S Sₘ
      M : Submonoid R
      inst✝⁷ : IsLocalization M Rₘ
      inst✝⁶ : IsLocalization (Algebra.algebraMapSubmonoid S M) Sₘ
      inst✝⁵ : Algebra Rₘ Sₘ
      inst✝⁴ : Algebra R Sₘ
      inst✝³ : IsScalarTower R Rₘ Sₘ
      inst✝² : IsScalarTower R S Sₘ
      inst✝¹ : Module.Free R S
      inst✝ : Module.Finite R S
      a : S
      h✝ : Subsingleton R
      ⊢ Eq ((Algebra.trace Rₘ Sₘ) ((algebraMap S Sₘ) a)) ((algebraMap R Rₘ) ((Algebr …
    -/
  · haveI : Subsingleton Rₘ := Module.subsingleton R Rₘ
    /-
      case inl
      R : Type u_1
      S : Type u_2
      inst✝¹⁴ : CommRing R
      inst✝¹³ : CommRing S
      inst✝¹² : Algebra R S
      Rₘ : Type u_3
      Sₘ : Type u_4
      inst✝¹¹ : CommRing Rₘ
      inst✝¹⁰ : Algebra R Rₘ
      inst✝⁹ : CommRing Sₘ
      inst✝⁸ : Algebra S Sₘ
      M : Submonoid R
      inst✝⁷ : IsLocalization M Rₘ
      inst✝⁶ : IsLocalization (Algebra.algebraMapSubmonoid S M) Sₘ
      inst✝⁵ : Algebra Rₘ Sₘ
      inst✝⁴ : Algebra R Sₘ
      inst✝³ : IsScalarTower R Rₘ Sₘ
      inst✝² : IsScalarTower R S Sₘ
      inst✝¹ : Module.Free R S
      inst✝ : Module.Finite R S
      a : S
      h✝ : Subsingleton R
      this : Subsingleton Rₘ
      ⊢ Eq ((Algebra.trace Rₘ Sₘ) ((algebraMap S Sₘ) a)) ((algebraMap R Rₘ) ((Algebr …
    -/
    simp [eq_iff_true_of_subsingleton]
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u_1
    S : Type u_2
    inst✝¹⁴ : CommRing R
    inst✝¹³ : CommRing S
    inst✝¹² : Algebra R S
    Rₘ : Type u_3
    Sₘ : Type u_4
    inst✝¹¹ : CommRing Rₘ
    inst✝¹⁰ : Algebra R Rₘ
    inst✝⁹ : CommRing Sₘ
    inst✝⁸ : Algebra S Sₘ
    M : Submonoid R
    inst✝⁷ : IsLocalization M Rₘ
    inst✝⁶ : IsLocalization (Algebra.algebraMapSubmonoid S M) Sₘ
    inst✝⁵ : Algebra Rₘ Sₘ
    inst✝⁴ : Algebra R Sₘ
    inst✝³ : IsScalarTower R Rₘ Sₘ
    inst✝² : IsScalarTower R S Sₘ
    inst✝¹ : Module.Free R S
    inst✝ : Module.Finite R S
    a : S
    h✝ : Nontrivial R
    ⊢ Eq ((Algebra.trace Rₘ Sₘ) ((algebraMap S Sₘ) a)) ((algebraMap R Rₘ) ((Algebr …
  -/
  let b := Module.Free.chooseBasis R S
  /-
    case inr
    R : Type u_1
    S : Type u_2
    inst✝¹⁴ : CommRing R
    inst✝¹³ : CommRing S
    inst✝¹² : Algebra R S
    Rₘ : Type u_3
    Sₘ : Type u_4
    inst✝¹¹ : CommRing Rₘ
    inst✝¹⁰ : Algebra R Rₘ
    inst✝⁹ : CommRing Sₘ
    inst✝⁸ : Algebra S Sₘ
    M : Submonoid R
    inst✝⁷ : IsLocalization M Rₘ
    inst✝⁶ : IsLocalization (Algebra.algebraMapSubmonoid S M) Sₘ
    inst✝⁵ : Algebra Rₘ Sₘ
    inst✝⁴ : Algebra R Sₘ
    inst✝³ : IsScalarTower R Rₘ Sₘ
    inst✝² : IsScalarTower R S Sₘ
    inst✝¹ : Module.Free R S
    inst✝ : Module.Finite R S
    a : S
    h✝ : Nontrivial R
    b : Basis (Module.Free.ChooseBasisIndex R S) R S := Module.Free.chooseBasis R S
    ⊢ Eq ((Algebra.trace Rₘ Sₘ) ((algebraMap S Sₘ) a)) ((algebraMap R Rₘ) ((Algebr …
  -/
  letI := Classical.decEq (Module.Free.ChooseBasisIndex R S)
  rw [Algebra.trace_eq_matrix_trace (b.localizationLocalization Rₘ M Sₘ),
    Algebra.trace_eq_matrix_trace b, ← Algebra.map_leftMulMatrix_localization]
  /-
    case inr
    R : Type u_1
    S : Type u_2
    inst✝¹⁴ : CommRing R
    inst✝¹³ : CommRing S
    inst✝¹² : Algebra R S
    Rₘ : Type u_3
    Sₘ : Type u_4
    inst✝¹¹ : CommRing Rₘ
    inst✝¹⁰ : Algebra R Rₘ
    inst✝⁹ : CommRing Sₘ
    inst✝⁸ : Algebra S Sₘ
    M : Submonoid R
    inst✝⁷ : IsLocalization M Rₘ
    inst✝⁶ : IsLocalization (Algebra.algebraMapSubmonoid S M) Sₘ
    inst✝⁵ : Algebra Rₘ Sₘ
    inst✝⁴ : Algebra R Sₘ
    inst✝³ : IsScalarTower R Rₘ Sₘ
    inst✝² : IsScalarTower R S Sₘ
    inst✝¹ : Module.Free R S
    inst✝ : Module.Finite R S
    a : S
    h✝ : Nontrivial R
    b : Basis (Module.Free.ChooseBasisIndex R S) R S := Module.Free.chooseBasis R S
    this : DecidableEq (Module.Free.ChooseBasisIndex R S) := Classical.decEq (Modu …
    ⊢ Eq ((algebraMap R Rₘ).mapMatrix ((Algebra.leftMulMatrix b) a)).trace ((algeb …
  -/
  exact (AddMonoidHom.map_trace (algebraMap R Rₘ).toAddMonoidHom _).symm
  /-
    🎉 no goals
  -/


theorem Algebra.traceMatrix_localizationLocalization (b : Basis ι R S) :
    Algebra.traceMatrix Rₘ (b.localizationLocalization Rₘ M Sₘ) =
      (algebraMap R Rₘ).mapMatrix (Algebra.traceMatrix R b) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹⁴ : CommRing R
    inst✝¹³ : CommRing S
    inst✝¹² : Algebra R S
    Rₘ : Type u_3
    inst✝¹¹ : CommRing Rₘ
    inst✝¹⁰ : Algebra R Rₘ
    M : Submonoid R
    inst✝⁹ : IsLocalization M Rₘ
    Sₘ : Type u_5
    inst✝⁸ : CommRing Sₘ
    inst✝⁷ : Algebra S Sₘ
    inst✝⁶ : Algebra Rₘ Sₘ
    inst✝⁵ : Algebra R Sₘ
    inst✝⁴ : IsScalarTower R Rₘ Sₘ
    inst✝³ : IsScalarTower R S Sₘ
    inst✝² : IsLocalization (Algebra.algebraMapSubmonoid S M) Sₘ
    ι : Type u_6
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    b : Basis ι R S
    ⊢ Eq (Algebra.traceMatrix Rₘ ⇑(Basis.localizationLocalization Rₘ M Sₘ b)) ((al …
  -/
  have : Module.Finite R S := Module.Finite.of_basis b
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹⁴ : CommRing R
    inst✝¹³ : CommRing S
    inst✝¹² : Algebra R S
    Rₘ : Type u_3
    inst✝¹¹ : CommRing Rₘ
    inst✝¹⁰ : Algebra R Rₘ
    M : Submonoid R
    inst✝⁹ : IsLocalization M Rₘ
    Sₘ : Type u_5
    inst✝⁸ : CommRing Sₘ
    inst✝⁷ : Algebra S Sₘ
    inst✝⁶ : Algebra Rₘ Sₘ
    inst✝⁵ : Algebra R Sₘ
    inst✝⁴ : IsScalarTower R Rₘ Sₘ
    inst✝³ : IsScalarTower R S Sₘ
    inst✝² : IsLocalization (Algebra.algebraMapSubmonoid S M) Sₘ
    ι : Type u_6
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    b : Basis ι R S
    this : Module.Finite R S
    ⊢ Eq (Algebra.traceMatrix Rₘ ⇑(Basis.localizationLocalization Rₘ M Sₘ b)) ((al …
  -/
  have : Module.Free R S := Module.Free.of_basis b
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹⁴ : CommRing R
    inst✝¹³ : CommRing S
    inst✝¹² : Algebra R S
    Rₘ : Type u_3
    inst✝¹¹ : CommRing Rₘ
    inst✝¹⁰ : Algebra R Rₘ
    M : Submonoid R
    inst✝⁹ : IsLocalization M Rₘ
    Sₘ : Type u_5
    inst✝⁸ : CommRing Sₘ
    inst✝⁷ : Algebra S Sₘ
    inst✝⁶ : Algebra Rₘ Sₘ
    inst✝⁵ : Algebra R Sₘ
    inst✝⁴ : IsScalarTower R Rₘ Sₘ
    inst✝³ : IsScalarTower R S Sₘ
    inst✝² : IsLocalization (Algebra.algebraMapSubmonoid S M) Sₘ
    ι : Type u_6
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    b : Basis ι R S
    this✝ : Module.Finite R S
    this : Module.Free R S
    ⊢ Eq (Algebra.traceMatrix Rₘ ⇑(Basis.localizationLocalization Rₘ M Sₘ b)) ((al …
  -/
  ext i j : 2
  simp_rw [RingHom.mapMatrix_apply, Matrix.map_apply, traceMatrix_apply, traceForm_apply,
    Basis.localizationLocalization_apply, ← map_mul]
  /-
    case a
    R : Type u_1
    S : Type u_2
    inst✝¹⁴ : CommRing R
    inst✝¹³ : CommRing S
    inst✝¹² : Algebra R S
    Rₘ : Type u_3
    inst✝¹¹ : CommRing Rₘ
    inst✝¹⁰ : Algebra R Rₘ
    M : Submonoid R
    inst✝⁹ : IsLocalization M Rₘ
    Sₘ : Type u_5
    inst✝⁸ : CommRing Sₘ
    inst✝⁷ : Algebra S Sₘ
    inst✝⁶ : Algebra Rₘ Sₘ
    inst✝⁵ : Algebra R Sₘ
    inst✝⁴ : IsScalarTower R Rₘ Sₘ
    inst✝³ : IsScalarTower R S Sₘ
    inst✝² : IsLocalization (Algebra.algebraMapSubmonoid S M) Sₘ
    ι : Type u_6
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    b : Basis ι R S
    this✝ : Module.Finite R S
    this : Module.Free R S
    i j : ι
    ⊢ Eq ((Algebra.trace Rₘ Sₘ) ((algebraMap S Sₘ) (HMul.hMul (b i) (b j)))) ((alg …
  -/
  exact Algebra.trace_localization R M _
  /-
    🎉 no goals
  -/


/-- Let `S` be an extension of `R` and `Rₘ Sₘ` be localizations at `M` of `R S` respectively. Let
`b` be a `R`-basis of `S`. Then discriminant of the `Rₘ`-basis of `Sₘ` induced by `b` is the
discriminant of `b`.
-/
theorem Algebra.discr_localizationLocalization (b : Basis ι R S) :
    Algebra.discr Rₘ (b.localizationLocalization Rₘ M Sₘ) =
    algebraMap R Rₘ (Algebra.discr R b) := by
  rw [Algebra.discr_def, Algebra.discr_def, RingHom.map_det,
    Algebra.traceMatrix_localizationLocalization]


