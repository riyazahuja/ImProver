/-- `Field.Emb F E` is the type of `F`-algebra homomorphisms from `E` to the algebraic closure
of `E`. -/
abbrev Emb := E →ₐ[F] AlgebraicClosure E


/-- If `E / F` is an algebraic extension, then the (finite) separable degree of `E / F`
is the number of `F`-algebra homomorphisms from `E` to the algebraic closure of `E`,
as a natural number. It is defined to be zero if there are infinitely many of them.
Note that if `E / F` is not algebraic, then this definition makes no mathematical sense. -/
def finSepDegree : ℕ := Nat.card (Emb F E)


instance instInhabitedEmb : Inhabited (Emb F E) := ⟨IsScalarTower.toAlgHom F E _⟩


instance instNeZeroFinSepDegree [FiniteDimensional F E] : NeZero (finSepDegree F E) :=
  ⟨Nat.card_ne_zero.2 ⟨inferInstance, Fintype.finite <| minpoly.AlgHom.fintype _ _ _⟩⟩


/-- A random bijection between `Field.Emb F E` and `Field.Emb F K` when `E` and `K` are isomorphic
as `F`-algebras. -/
def embEquivOfEquiv (i : E ≃ₐ[F] K) :
    Emb F E ≃ Emb F K := AlgEquiv.arrowCongr i <| AlgEquiv.symm <| by
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    K : Type w
    inst✝¹ : Field K
    inst✝ : Algebra F K
    i : AlgEquiv F E K
    ⊢ AlgEquiv F (AlgebraicClosure K) (AlgebraicClosure E)
  -/
  let _ : Algebra E K := i.toAlgHom.toRingHom.toAlgebra
  have : Algebra.IsAlgebraic E K := by
    constructor
    intro x
    have h := isAlgebraic_algebraMap (R := E) (A := K) (i.symm.toAlgHom x)
    rw [show ∀ y : E, (algebraMap E K) y = i.toAlgHom y from fun y ↦ rfl] at h
    simpa only [AlgEquiv.toAlgHom_eq_coe, AlgHom.coe_coe, AlgEquiv.apply_symm_apply] using h
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    K : Type w
    inst✝¹ : Field K
    inst✝ : Algebra F K
    i : AlgEquiv F E K
    x✝ : Algebra E K := (↑i).toAlgebra
    this : Algebra.IsAlgebraic E K
    ⊢ AlgEquiv F (AlgebraicClosure K) (AlgebraicClosure E)
  -/
  apply AlgEquiv.restrictScalars (R := F) (S := E)
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    K : Type w
    inst✝¹ : Field K
    inst✝ : Algebra F K
    i : AlgEquiv F E K
    x✝ : Algebra E K := (↑i).toAlgebra
    this : Algebra.IsAlgebraic E K
    ⊢ AlgEquiv E (AlgebraicClosure K) (AlgebraicClosure E)
  -/
  exact IsAlgClosure.equivOfAlgebraic E K (AlgebraicClosure K) (AlgebraicClosure E)
  /-
    🎉 no goals
  -/


/-- If `E` and `K` are isomorphic as `F`-algebras, then they have the same `Field.finSepDegree`
over `F`. -/
theorem finSepDegree_eq_of_equiv (i : E ≃ₐ[F] K) :
    finSepDegree F E = finSepDegree F K := Nat.card_congr (embEquivOfEquiv F E K i)


@[simp]
theorem finSepDegree_self : finSepDegree F F = 1 := by
  have : Cardinal.mk (Emb F F) = 1 := le_antisymm
    (Cardinal.le_one_iff_subsingleton.2 AlgHom.subsingleton)
    (Cardinal.one_le_iff_ne_zero.2 <| Cardinal.mk_ne_zero _)
  /-
    F : Type u
    inst✝ : Field F
    this : Eq (Cardinal.mk (Field.Emb F F)) 1
    ⊢ Eq (Field.finSepDegree F F) 1
  -/
  rw [finSepDegree, Nat.card, this, Cardinal.one_toNat]
  /-
    🎉 no goals
  -/


@[simp]
theorem finSepDegree_bot : finSepDegree F (⊥ : IntermediateField F E) = 1 := by
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    ⊢ Eq (Field.finSepDegree F (Subtype fun x => Membership.mem Bot.bot x)) 1
  -/
  rw [finSepDegree_eq_of_equiv _ _ _ (botEquiv F E), finSepDegree_self]
  /-
    🎉 no goals
  -/


@[simp]
theorem finSepDegree_bot' : finSepDegree F (⊥ : IntermediateField E K) = finSepDegree F E :=
  finSepDegree_eq_of_equiv _ _ _ ((botEquiv E K).restrictScalars F)


@[simp]
theorem finSepDegree_top : finSepDegree F (⊤ : IntermediateField E K) = finSepDegree F K :=
  finSepDegree_eq_of_equiv _ _ _ ((topEquiv (F := E) (E := K)).restrictScalars F)


/-- A random bijection between `Field.Emb F E` and `E →ₐ[F] K` if `E = F(S)` such that every
element `s` of `S` is integral (= algebraic) over `F` and whose minimal polynomial splits in `K`.
Combined with `Field.instInhabitedEmb`, it can be viewed as a stronger version of
`IntermediateField.nonempty_algHom_of_adjoin_splits`. -/
def embEquivOfAdjoinSplits {S : Set E} (hS : adjoin F S = ⊤)
    (hK : ∀ s ∈ S, IsIntegral F s ∧ Splits (algebraMap F K) (minpoly F s)) :
    Emb F E ≃ (E →ₐ[F] K) :=
  have : Algebra.IsAlgebraic F (⊤ : IntermediateField F E) :=
    (hS ▸ isAlgebraic_adjoin (S := S) fun x hx ↦ (hK x hx).1)
  have halg := (topEquiv (F := F) (E := E)).isAlgebraic
  Classical.choice <| Function.Embedding.antisymm
    (halg.algHomEmbeddingOfSplits (fun _ ↦ splits_of_mem_adjoin F E (S := S) hK (hS ▸ mem_top)) _)
    (halg.algHomEmbeddingOfSplits (fun _ ↦ IsAlgClosed.splits_codomain _) _)


/-- The `Field.finSepDegree F E` is equal to the cardinality of `E →ₐ[F] K`
if `E = F(S)` such that every element
`s` of `S` is integral (= algebraic) over `F` and whose minimal polynomial splits in `K`. -/
theorem finSepDegree_eq_of_adjoin_splits {S : Set E} (hS : adjoin F S = ⊤)
    (hK : ∀ s ∈ S, IsIntegral F s ∧ Splits (algebraMap F K) (minpoly F s)) :
    finSepDegree F E = Nat.card (E →ₐ[F] K) := Nat.card_congr (embEquivOfAdjoinSplits F E K hS hK)


/-- A random bijection between `Field.Emb F E` and `E →ₐ[F] K` when `E / F` is algebraic
and `K / F` is algebraically closed. -/
def embEquivOfIsAlgClosed [Algebra.IsAlgebraic F E] [IsAlgClosed K] :
    Emb F E ≃ (E →ₐ[F] K) :=
  embEquivOfAdjoinSplits F E K (adjoin_univ F E) fun s _ ↦
    ⟨Algebra.IsIntegral.isIntegral s, IsAlgClosed.splits_codomain _⟩


/-- The `Field.finSepDegree F E` is equal to the cardinality of `E →ₐ[F] K` as a natural number,
when `E / F` is algebraic and `K / F` is algebraically closed. -/
@[stacks 09HJ "We use `finSepDegree` to state a more general result."]
theorem finSepDegree_eq_of_isAlgClosed [Algebra.IsAlgebraic F E] [IsAlgClosed K] :
    finSepDegree F E = Nat.card (E →ₐ[F] K) := Nat.card_congr (embEquivOfIsAlgClosed F E K)


/-- If `K / E / F` is a field extension tower, such that `K / E` is algebraic,
then there is a non-canonical bijection
`Field.Emb F E × Field.Emb E K ≃ Field.Emb F K`. A corollary of `algHomEquivSigma`. -/
def embProdEmbOfIsAlgebraic [Algebra E K] [IsScalarTower F E K] [Algebra.IsAlgebraic E K] :
    Emb F E × Emb E K ≃ Emb F K :=
  let e : ∀ f : E →ₐ[F] AlgebraicClosure K,
      @AlgHom E K _ _ _ _ _ f.toRingHom.toAlgebra ≃ Emb E K := fun f ↦
    (@embEquivOfIsAlgClosed E K _ _ _ _ _ f.toRingHom.toAlgebra).symm
  (algHomEquivSigma (A := F) (B := E) (C := K) (D := AlgebraicClosure K) |>.trans
    (Equiv.sigmaEquivProdOfEquiv e) |>.trans <| Equiv.prodCongrLeft <|
      fun _ : Emb E K ↦ AlgEquiv.arrowCongr (@AlgEquiv.refl F E _ _ _) <|
        (IsAlgClosure.equivOfAlgebraic E K (AlgebraicClosure K)
          (AlgebraicClosure E)).restrictScalars F).symm


/-- If the field extension `E / F` is transcendental, then `Field.Emb F E` is infinite. -/
instance infinite_emb_of_transcendental [H : Algebra.Transcendental F E] : Infinite (Emb F E) := by
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    K : Type w
    inst✝¹ : Field K
    inst✝ : Algebra F K
    H : Algebra.Transcendental F E
    ⊢ Infinite (Field.Emb F E)
  -/
  obtain ⟨ι, x, hx⟩ := exists_isTranscendenceBasis' _ (algebraMap F E).injective
  /-
    case intro.intro
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    K : Type w
    inst✝¹ : Field K
    inst✝ : Algebra F K
    H : Algebra.Transcendental F E
    ι : Type v
    x : ι → E
    hx : IsTranscendenceBasis F x
    ⊢ Infinite (Field.Emb F E)
  -/
  have := hx.isAlgebraic_field
  /-
    case intro.intro
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    K : Type w
    inst✝¹ : Field K
    inst✝ : Algebra F K
    H : Algebra.Transcendental F E
    ι : Type v
    x : ι → E
    hx : IsTranscendenceBasis F x
    this : Algebra.IsAlgebraic (Subtype fun x_1 => Membership.mem (IntermediateFie …
    ⊢ Infinite (Field.Emb F E)
  -/
  rw [← (embProdEmbOfIsAlgebraic F (adjoin F (Set.range x)) E).infinite_iff]
  /-
    case intro.intro
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    K : Type w
    inst✝¹ : Field K
    inst✝ : Algebra F K
    H : Algebra.Transcendental F E
    ι : Type v
    x : ι → E
    hx : IsTranscendenceBasis F x
    this : Algebra.IsAlgebraic (Subtype fun x_1 => Membership.mem (IntermediateFie …
    ⊢ Infinite (Prod (Field.Emb F (Subtype fun x_1 => Membership.mem (Intermediate …
  -/
  refine @Prod.infinite_of_left _ _ ?_ _
  /-
    case intro.intro
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    K : Type w
    inst✝¹ : Field K
    inst✝ : Algebra F K
    H : Algebra.Transcendental F E
    ι : Type v
    x : ι → E
    hx : IsTranscendenceBasis F x
    this : Algebra.IsAlgebraic (Subtype fun x_1 => Membership.mem (IntermediateFie …
    ⊢ Infinite (Field.Emb F (Subtype fun x_1 => Membership.mem (IntermediateField. …
  -/
  rw [← (embEquivOfEquiv _ _ _ hx.1.aevalEquivField).infinite_iff]
  /-
    case intro.intro
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    K : Type w
    inst✝¹ : Field K
    inst✝ : Algebra F K
    H : Algebra.Transcendental F E
    ι : Type v
    x : ι → E
    hx : IsTranscendenceBasis F x
    this : Algebra.IsAlgebraic (Subtype fun x_1 => Membership.mem (IntermediateFie …
    ⊢ Infinite (Field.Emb F (FractionRing (MvPolynomial ι F)))
  -/
  obtain ⟨i⟩ := hx.nonempty_iff_transcendental.2 H
  /-
    case intro.intro.intro
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    K : Type w
    inst✝¹ : Field K
    inst✝ : Algebra F K
    H : Algebra.Transcendental F E
    ι : Type v
    x : ι → E
    hx : IsTranscendenceBasis F x
    this : Algebra.IsAlgebraic (Subtype fun x_1 => Membership.mem (IntermediateFie …
    i : ι
    ⊢ Infinite (Field.Emb F (FractionRing (MvPolynomial ι F)))
  -/
  let K := FractionRing (MvPolynomial ι F)
  /-
    case intro.intro.intro
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    K✝ : Type w
    inst✝¹ : Field K✝
    inst✝ : Algebra F K✝
    H : Algebra.Transcendental F E
    ι : Type v
    x : ι → E
    hx : IsTranscendenceBasis F x
    this : Algebra.IsAlgebraic (Subtype fun x_1 => Membership.mem (IntermediateFie …
    i : ι
    K : Type (max u v) := FractionRing (MvPolynomial ι F)
    ⊢ Infinite (Field.Emb F (FractionRing (MvPolynomial ι F)))
  -/
  let i1 := IsScalarTower.toAlgHom F (MvPolynomial ι F) (AlgebraicClosure K)
  have hi1 : Function.Injective i1 := by
    rw [IsScalarTower.coe_toAlgHom', IsScalarTower.algebraMap_eq _ K]
    exact (algebraMap K (AlgebraicClosure K)).injective.comp (IsFractionRing.injective _ _)
  let f (n : ℕ) : Emb F K := IsFractionRing.liftAlgHom
    (g := i1.comp <| MvPolynomial.aeval fun i : ι ↦ MvPolynomial.X i ^ (n + 1)) <| hi1.comp <| by
      simpa [algebraicIndependent_iff_injective_aeval] using
        MvPolynomial.algebraicIndependent_polynomial_aeval_X _
          fun i : ι ↦ (Polynomial.transcendental_X F).pow n.succ_pos
  /-
    case intro.intro.intro
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    K✝ : Type w
    inst✝¹ : Field K✝
    inst✝ : Algebra F K✝
    H : Algebra.Transcendental F E
    ι : Type v
    x : ι → E
    hx : IsTranscendenceBasis F x
    this : Algebra.IsAlgebraic (Subtype fun x_1 => Membership.mem (IntermediateFie …
    i : ι
    K : Type (max u v) := FractionRing (MvPolynomial ι F)
    i1 : AlgHom F (MvPolynomial ι F) (AlgebraicClosure K) := IsScalarTower.toAlgHo …
    hi1 : Function.Injective ⇑i1
    f : Nat → Field.Emb F K := fun n => IsFractionRing.liftAlgHom ⋯
    ⊢ Infinite (Field.Emb F (FractionRing (MvPolynomial ι F)))
  -/
  refine Infinite.of_injective f fun m n h ↦ ?_
  replace h : (MvPolynomial.X i) ^ (m + 1) = (MvPolynomial.X i) ^ (n + 1) := hi1 <| by
    simpa [f, -map_pow] using congr($h (algebraMap _ K (MvPolynomial.X (R := F) i)))
  /-
    case intro.intro.intro
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    K✝ : Type w
    inst✝¹ : Field K✝
    inst✝ : Algebra F K✝
    H : Algebra.Transcendental F E
    ι : Type v
    x : ι → E
    hx : IsTranscendenceBasis F x
    this : Algebra.IsAlgebraic (Subtype fun x_1 => Membership.mem (IntermediateFie …
    i : ι
    K : Type (max u v) := FractionRing (MvPolynomial ι F)
    i1 : AlgHom F (MvPolynomial ι F) (AlgebraicClosure K) := IsScalarTower.toAlgHo …
    hi1 : Function.Injective ⇑i1
    f : Nat → Field.Emb F K := fun n => IsFractionRing.liftAlgHom ⋯
    m n : Nat
    h : Eq (HPow.hPow (MvPolynomial.X i) (HAdd.hAdd m 1)) (HPow.hPow (MvPolynomial …
    ⊢ Eq m n
  -/
  simpa using congr(MvPolynomial.totalDegree $h)
  /-
    🎉 no goals
  -/


/-- If the field extension `E / F` is transcendental, then `Field.finSepDegree F E = 0`, which
actually means that `Field.Emb F E` is infinite (see `Field.infinite_emb_of_transcendental`). -/
theorem finSepDegree_eq_zero_of_transcendental [Algebra.Transcendental F E] :
    finSepDegree F E = 0 := Nat.card_eq_zero_of_infinite


/-- If `K / E / F` is a field extension tower, such that `K / E` is algebraic, then their
separable degrees satisfy the tower law
$[E:F]_s [K:E]_s = [K:F]_s$. See also `Module.finrank_mul_finrank`. -/
@[stacks 09HK "Part 1, `finSepDegree` variant"]
theorem finSepDegree_mul_finSepDegree_of_isAlgebraic
    [Algebra E K] [IsScalarTower F E K] [Algebra.IsAlgebraic E K] :
    finSepDegree F E * finSepDegree E K = finSepDegree F K := by
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : Algebra.IsAlgebraic E K
    ⊢ Eq (HMul.hMul (Field.finSepDegree F E) (Field.finSepDegree E K)) (Field.finS …
  -/
  simpa only [Nat.card_prod] using Nat.card_congr (embProdEmbOfIsAlgebraic F E K)
  /-
    🎉 no goals
  -/


open Classical in
/-- The separable degree `Polynomial.natSepDegree` of a polynomial is a natural number,
defined to be the number of distinct roots of it over its splitting field.
This is similar to `Polynomial.natDegree` but not to `Polynomial.degree`, namely, the separable
degree of `0` is `0`, not negative infinity. -/
def natSepDegree : ℕ := (f.aroots f.SplittingField).toFinset.card


/-- The separable degree of a polynomial is smaller than its degree. -/
theorem natSepDegree_le_natDegree : f.natSepDegree ≤ f.natDegree := by
  /-
    F : Type u
    inst✝ : Field F
    f : Polynomial F
    ⊢ LE.le f.natSepDegree f.natDegree
  -/
  have := f.map (algebraMap F f.SplittingField) |>.card_roots'
  /-
    F : Type u
    inst✝ : Field F
    f : Polynomial F
    this : LE.le (Polynomial.map (algebraMap F f.SplittingField) f).roots.card (Po …
    ⊢ LE.le f.natSepDegree f.natDegree
  -/
  rw [← aroots_def, natDegree_map] at this
  classical
  exact (f.aroots f.SplittingField).toFinset_card_le.trans this


@[simp]
theorem natSepDegree_X_sub_C (x : F) : (X - C x).natSepDegree = 1 := by
  /-
    F : Type u
    inst✝ : Field F
    x : F
    ⊢ Eq (HSub.hSub Polynomial.X (Polynomial.C x)).natSepDegree 1
  -/
  simp only [natSepDegree, aroots_X_sub_C, Multiset.toFinset_singleton, Finset.card_singleton]
  /-
    🎉 no goals
  -/


@[simp]
theorem natSepDegree_X : (X : F[X]).natSepDegree = 1 := by
  /-
    F : Type u
    inst✝ : Field F
    ⊢ Eq Polynomial.X.natSepDegree 1
  -/
  simp only [natSepDegree, aroots_X, Multiset.toFinset_singleton, Finset.card_singleton]
  /-
    🎉 no goals
  -/


/-- A constant polynomial has zero separable degree. -/
theorem natSepDegree_eq_zero (h : f.natDegree = 0) : f.natSepDegree = 0 := by
  /-
    F : Type u
    inst✝ : Field F
    f : Polynomial F
    h : Eq f.natDegree 0
    ⊢ Eq f.natSepDegree 0
  -/
  linarith only [natSepDegree_le_natDegree f, h]
  /-
    🎉 no goals
  -/


@[simp]
theorem natSepDegree_C (x : F) : (C x).natSepDegree = 0 := natSepDegree_eq_zero _ (natDegree_C _)


@[simp]
theorem natSepDegree_zero : (0 : F[X]).natSepDegree = 0 := by
  /-
    F : Type u
    inst✝ : Field F
    ⊢ Eq (Polynomial.natSepDegree 0) 0
  -/
  rw [← C_0, natSepDegree_C]
  /-
    🎉 no goals
  -/


@[simp]
theorem natSepDegree_one : (1 : F[X]).natSepDegree = 0 := by
  /-
    F : Type u
    inst✝ : Field F
    ⊢ Eq (Polynomial.natSepDegree 1) 0
  -/
  rw [← C_1, natSepDegree_C]
  /-
    🎉 no goals
  -/


/-- A non-constant polynomial has non-zero separable degree. -/
theorem natSepDegree_ne_zero (h : f.natDegree ≠ 0) : f.natSepDegree ≠ 0 := by
  /-
    F : Type u
    inst✝ : Field F
    f : Polynomial F
    h : Ne f.natDegree 0
    ⊢ Ne f.natSepDegree 0
  -/
  rw [natSepDegree, ne_eq, Finset.card_eq_zero, ← ne_eq, ← Finset.nonempty_iff_ne_empty]
  /-
    F : Type u
    inst✝ : Field F
    f : Polynomial F
    h : Ne f.natDegree 0
    ⊢ (f.aroots f.SplittingField).toFinset.Nonempty
  -/
  use rootOfSplits _ (SplittingField.splits f) (ne_of_apply_ne _ h)
  classical
  rw [Multiset.mem_toFinset, mem_aroots]
  exact ⟨ne_of_apply_ne _ h, map_rootOfSplits _ (SplittingField.splits f) (ne_of_apply_ne _ h)⟩


/-- A polynomial has zero separable degree if and only if it is constant. -/
theorem natSepDegree_eq_zero_iff : f.natSepDegree = 0 ↔ f.natDegree = 0 :=
  ⟨(natSepDegree_ne_zero f).mtr, natSepDegree_eq_zero f⟩


/-- A polynomial has non-zero separable degree if and only if it is non-constant. -/
theorem natSepDegree_ne_zero_iff : f.natSepDegree ≠ 0 ↔ f.natDegree ≠ 0 :=
  Iff.not <| natSepDegree_eq_zero_iff f


/-- The separable degree of a non-zero polynomial is equal to its degree if and only if
it is separable. -/
theorem natSepDegree_eq_natDegree_iff (hf : f ≠ 0) :
    f.natSepDegree = f.natDegree ↔ f.Separable := by
  classical
  simp_rw [← card_rootSet_eq_natDegree_iff_of_splits hf (SplittingField.splits f),
    rootSet_def, Finset.coe_sort_coe, Fintype.card_coe]
  rfl


/-- If a polynomial is separable, then its separable degree is equal to its degree. -/
theorem natSepDegree_eq_natDegree_of_separable (h : f.Separable) :
    f.natSepDegree = f.natDegree := (natSepDegree_eq_natDegree_iff f h.ne_zero).2 h


variable {f} in
/-- Same as `Polynomial.natSepDegree_eq_natDegree_of_separable`, but enables the use of
dot notation. -/
theorem Separable.natSepDegree_eq_natDegree (h : f.Separable) :
    f.natSepDegree = f.natDegree := natSepDegree_eq_natDegree_of_separable f h


/-- If a polynomial splits over `E`, then its separable degree is equal to
the number of distinct roots of it over `E`. -/
theorem natSepDegree_eq_of_splits [DecidableEq E] (h : f.Splits (algebraMap F E)) :
    f.natSepDegree = (f.aroots E).toFinset.card := by
  classical
  rw [aroots, ← (SplittingField.lift f h).comp_algebraMap, ← map_map,
    roots_map _ ((splits_id_iff_splits _).mpr <| SplittingField.splits f),
    Multiset.toFinset_map, Finset.card_image_of_injective _ (RingHom.injective _), natSepDegree]


variable (E) in
/-- The separable degree of a polynomial is equal to
the number of distinct roots of it over any algebraically closed field. -/
theorem natSepDegree_eq_of_isAlgClosed [DecidableEq E] [IsAlgClosed E] :
    f.natSepDegree = (f.aroots E).toFinset.card :=
  natSepDegree_eq_of_splits f (IsAlgClosed.splits_codomain f)


theorem natSepDegree_map (f : E[X]) (i : E →+* K) : (f.map i).natSepDegree = f.natSepDegree := by
  classical
  let _ := i.toAlgebra
  simp_rw [show i = algebraMap E K by rfl, natSepDegree_eq_of_isAlgClosed (AlgebraicClosure K),
    aroots_def, map_map, ← IsScalarTower.algebraMap_eq]


@[simp]
theorem natSepDegree_C_mul {x : F} (hx : x ≠ 0) :
    (C x * f).natSepDegree = f.natSepDegree := by
  classical
  simp only [natSepDegree_eq_of_isAlgClosed (AlgebraicClosure F), aroots_C_mul _ hx]


@[simp]
theorem natSepDegree_smul_nonzero {x : F} (hx : x ≠ 0) :
    (x • f).natSepDegree = f.natSepDegree := by
  classical
  simp only [natSepDegree_eq_of_isAlgClosed (AlgebraicClosure F), aroots_smul_nonzero _ hx]


@[simp]
theorem natSepDegree_pow {n : ℕ} : (f ^ n).natSepDegree = if n = 0 then 0 else f.natSepDegree := by
  classical
  simp only [natSepDegree_eq_of_isAlgClosed (AlgebraicClosure F), aroots_pow]
  by_cases h : n = 0
  · simp only [h, zero_smul, Multiset.toFinset_zero, Finset.card_empty, ite_true]
  simp only [h, Multiset.toFinset_nsmul _ n h, ite_false]


theorem natSepDegree_pow_of_ne_zero {n : ℕ} (hn : n ≠ 0) :
                                                /-
                                                  F : Type u
                                                  inst✝ : Field F
                                                  f : Polynomial F
                                                  n : Nat
                                                  hn : Ne n 0
                                                  ⊢ Eq (HPow.hPow f n).natSepDegree f.natSepDegree
                                                -/
    (f ^ n).natSepDegree = f.natSepDegree := by simp_rw [natSepDegree_pow, hn, ite_false]
                                                /-
                                                  🎉 no goals
                                                -/


theorem natSepDegree_X_pow {n : ℕ} : (X ^ n : F[X]).natSepDegree = if n = 0 then 0 else 1 := by
  /-
    F : Type u
    inst✝ : Field F
    n : Nat
    ⊢ Eq (HPow.hPow Polynomial.X n).natSepDegree (ite (Eq n 0) 0 1)
  -/
  simp only [natSepDegree_pow, natSepDegree_X]
  /-
    🎉 no goals
  -/


theorem natSepDegree_X_sub_C_pow {x : F} {n : ℕ} :
    ((X - C x) ^ n).natSepDegree = if n = 0 then 0 else 1 := by
  /-
    F : Type u
    inst✝ : Field F
    x : F
    n : Nat
    ⊢ Eq (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C x)) n).natSepDegree (ite …
  -/
  simp only [natSepDegree_pow, natSepDegree_X_sub_C]
  /-
    🎉 no goals
  -/


theorem natSepDegree_C_mul_X_sub_C_pow {x y : F} {n : ℕ} (hx : x ≠ 0) :
    (C x * (X - C y) ^ n).natSepDegree = if n = 0 then 0 else 1 := by
  /-
    F : Type u
    inst✝ : Field F
    x y : F
    n : Nat
    hx : Ne x 0
    ⊢ Eq (HMul.hMul (Polynomial.C x) (HPow.hPow (HSub.hSub Polynomial.X (Polynomia …
  -/
  simp only [natSepDegree_C_mul _ hx, natSepDegree_X_sub_C_pow]
  /-
    🎉 no goals
  -/


theorem natSepDegree_mul (g : F[X]) :
    (f * g).natSepDegree ≤ f.natSepDegree + g.natSepDegree := by
  /-
    F : Type u
    inst✝ : Field F
    f g : Polynomial F
    ⊢ LE.le (HMul.hMul f g).natSepDegree (HAdd.hAdd f.natSepDegree g.natSepDegree)
  -/
  by_cases h : f * g = 0
    /-
      case pos
      F : Type u
      inst✝ : Field F
      f g : Polynomial F
      h : Eq (HMul.hMul f g) 0
      ⊢ LE.le (HMul.hMul f g).natSepDegree (HAdd.hAdd f.natSepDegree g.natSepDegree)
    -/
  · simp only [h, natSepDegree_zero, zero_le]
    /-
      🎉 no goals
    -/
  classical
  simp_rw [natSepDegree_eq_of_isAlgClosed (AlgebraicClosure F), aroots_mul h, Multiset.toFinset_add]
  exact Finset.card_union_le _ _


theorem natSepDegree_mul_eq_iff (g : F[X]) :
    (f * g).natSepDegree = f.natSepDegree + g.natSepDegree ↔ (f = 0 ∧ g = 0) ∨ IsCoprime f g := by
  /-
    F : Type u
    inst✝ : Field F
    f g : Polynomial F
    ⊢ Iff (Eq (HMul.hMul f g).natSepDegree (HAdd.hAdd f.natSepDegree g.natSepDegre …
  -/
  by_cases h : f * g = 0
    /-
      case pos
      F : Type u
      inst✝ : Field F
      f g : Polynomial F
      h : Eq (HMul.hMul f g) 0
      ⊢ Iff (Eq (HMul.hMul f g).natSepDegree (HAdd.hAdd f.natSepDegree g.natSepDegre …
    -/
  · rw [mul_eq_zero] at h
    /-
      case pos
      F : Type u
      inst✝ : Field F
      f g : Polynomial F
      h : Or (Eq f 0) (Eq g 0)
      ⊢ Iff (Eq (HMul.hMul f g).natSepDegree (HAdd.hAdd f.natSepDegree g.natSepDegre …
    -/
    wlog hf : f = 0 generalizing f g
    · simpa only [mul_comm, add_comm, and_comm,
        isCoprime_comm] using this g f h.symm (h.resolve_left hf)
    rw [hf, zero_mul, natSepDegree_zero, zero_add, isCoprime_zero_left, isUnit_iff, eq_comm,
      natSepDegree_eq_zero_iff, natDegree_eq_zero]
    /-
      F : Type u
      inst✝ : Field F
      f✝ f g : Polynomial F
      h : Or (Eq f 0) (Eq g 0)
      hf : Eq f 0
      ⊢ Iff (Exists fun x => Eq (Polynomial.C x) g) (Or (And (Eq 0 0) (Eq g 0)) (Exi …
    -/
    refine ⟨fun ⟨x, h⟩ ↦ ?_, ?_⟩
      /-
        case refine_1
        F : Type u
        inst✝ : Field F
        f✝ f g : Polynomial F
        h✝ : Or (Eq f 0) (Eq g 0)
        hf : Eq f 0
        x✝ : Exists fun x => Eq (Polynomial.C x) g
        x : F
        h : Eq (Polynomial.C x) g
        ⊢ Or (And (Eq 0 0) (Eq g 0)) (Exists fun r => And (IsUnit r) (Eq (Polynomial.C …
      -/
    · by_cases hx : x = 0
        /-
          case pos
          F : Type u
          inst✝ : Field F
          f✝ f g : Polynomial F
          h✝ : Or (Eq f 0) (Eq g 0)
          hf : Eq f 0
          x✝ : Exists fun x => Eq (Polynomial.C x) g
          x : F
          h : Eq (Polynomial.C x) g
          hx : Eq x 0
          ⊢ Or (And (Eq 0 0) (Eq g 0)) (Exists fun r => And (IsUnit r) (Eq (Polynomial.C …
        -/
      · exact .inl ⟨rfl, by rw [← h, hx, map_zero]⟩
        /-
          🎉 no goals
        -/
      /-
        case neg
        F : Type u
        inst✝ : Field F
        f✝ f g : Polynomial F
        h✝ : Or (Eq f 0) (Eq g 0)
        hf : Eq f 0
        x✝ : Exists fun x => Eq (Polynomial.C x) g
        x : F
        h : Eq (Polynomial.C x) g
        hx : Not (Eq x 0)
        ⊢ Or (And (Eq 0 0) (Eq g 0)) (Exists fun r => And (IsUnit r) (Eq (Polynomial.C …
      -/
      exact .inr ⟨x, Ne.isUnit hx, h⟩
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      F : Type u
      inst✝ : Field F
      f✝ f g : Polynomial F
      h : Or (Eq f 0) (Eq g 0)
      hf : Eq f 0
      ⊢ Or (And (Eq 0 0) (Eq g 0)) (Exists fun r => And (IsUnit r) (Eq (Polynomial.C …
    -/
    rintro (⟨-, h⟩ | ⟨x, -, h⟩)
      /-
        case refine_2.inl.intro
        F : Type u
        inst✝ : Field F
        f✝ f g : Polynomial F
        h✝ : Or (Eq f 0) (Eq g 0)
        hf : Eq f 0
        h : Eq g 0
        ⊢ Exists fun x => Eq (Polynomial.C x) g
      -/
    · exact ⟨0, by rw [h, map_zero]⟩
      /-
        🎉 no goals
      -/
    /-
      case refine_2.inr.intro.intro
      F : Type u
      inst✝ : Field F
      f✝ f g : Polynomial F
      h✝ : Or (Eq f 0) (Eq g 0)
      hf : Eq f 0
      x : F
      h : Eq (Polynomial.C x) g
      ⊢ Exists fun x => Eq (Polynomial.C x) g
    -/
    exact ⟨x, h⟩
    /-
      🎉 no goals
    -/
  classical
  simp_rw [natSepDegree_eq_of_isAlgClosed (AlgebraicClosure F), aroots_mul h, Multiset.toFinset_add,
    Finset.card_union_eq_card_add_card, Finset.disjoint_iff_ne, Multiset.mem_toFinset, mem_aroots]
  rw [mul_eq_zero, not_or] at h
  refine ⟨fun H ↦ .inr (isCoprime_of_irreducible_dvd (not_and.2 fun _ ↦ h.2)
    fun u hu ⟨v, hf⟩ ⟨w, hg⟩ ↦ ?_), ?_⟩
  · obtain ⟨x, hx⟩ := IsAlgClosed.exists_aeval_eq_zero
      (AlgebraicClosure F) _ (degree_pos_of_irreducible hu).ne'
    exact H x ⟨h.1, by simpa only [map_mul, hx, zero_mul] using congr(aeval x $hf)⟩
      x ⟨h.2, by simpa only [map_mul, hx, zero_mul] using congr(aeval x $hg)⟩ rfl
  rintro (⟨rfl, rfl⟩ | hc)
  · exact (h.1 rfl).elim
  rintro x hf _ hg rfl
  obtain ⟨u, v, hfg⟩ := hc
  simpa only [map_add, map_mul, map_one, hf.2, hg.2, mul_zero, add_zero,
    zero_ne_one] using congr(aeval x $hfg)


theorem natSepDegree_mul_of_isCoprime (g : F[X]) (hc : IsCoprime f g) :
    (f * g).natSepDegree = f.natSepDegree + g.natSepDegree :=
  (natSepDegree_mul_eq_iff f g).2 (.inr hc)


theorem natSepDegree_le_of_dvd (g : F[X]) (h1 : f ∣ g) (h2 : g ≠ 0) :
    f.natSepDegree ≤ g.natSepDegree := by
  classical
  simp_rw [natSepDegree_eq_of_isAlgClosed (AlgebraicClosure F)]
  exact Finset.card_le_card <| Multiset.toFinset_subset.mpr <|
    Multiset.Le.subset <| roots.le_of_dvd (map_ne_zero h2) <| map_dvd _ h1


/-- If a field `F` is of exponential characteristic `q`, then `Polynomial.expand F (q ^ n) f`
and `f` have the same separable degree. -/
theorem natSepDegree_expand (q : ℕ) [hF : ExpChar F q] {n : ℕ} :
    (expand F (q ^ n) f).natSepDegree = f.natSepDegree := by
  /-
    F : Type u
    inst✝ : Field F
    f : Polynomial F
    q : Nat
    hF : ExpChar F q
    n : Nat
    ⊢ Eq ((Polynomial.expand F (HPow.hPow q n)) f).natSepDegree f.natSepDegree
  -/
  cases' hF with _ _ hprime _
    /-
      case zero
      F : Type u
      inst✝¹ : Field F
      f : Polynomial F
      n : Nat
      inst✝ : CharZero F
      ⊢ Eq ((Polynomial.expand F (HPow.hPow 1 n)) f).natSepDegree f.natSepDegree
    -/
  · simp only [one_pow, expand_one]
    /-
      🎉 no goals
    -/
  /-
    case prime
    F : Type u
    inst✝ : Field F
    f : Polynomial F
    q n : Nat
    hprime : Nat.Prime q
    hchar✝ : CharP F q
    ⊢ Eq ((Polynomial.expand F (HPow.hPow q n)) f).natSepDegree f.natSepDegree
  -/
  haveI := Fact.mk hprime
  classical
  simpa only [natSepDegree_eq_of_isAlgClosed (AlgebraicClosure F), aroots_def, map_expand,
    Fintype.card_coe] using Fintype.card_eq.2
      ⟨(f.map (algebraMap F (AlgebraicClosure F))).rootsExpandPowEquivRoots q n⟩


theorem natSepDegree_X_pow_char_pow_sub_C (q : ℕ) [ExpChar F q] (n : ℕ) (y : F) :
    (X ^ q ^ n - C y).natSepDegree = 1 := by
  /-
    F : Type u
    inst✝¹ : Field F
    q : Nat
    inst✝ : ExpChar F q
    n : Nat
    y : F
    ⊢ Eq (HSub.hSub (HPow.hPow Polynomial.X (HPow.hPow q n)) (Polynomial.C y)).nat …
  -/
  rw [← expand_X, ← expand_C (q ^ n), ← map_sub, natSepDegree_expand, natSepDegree_X_sub_C]
  /-
    🎉 no goals
  -/


variable {f} in
/-- If `g` is a separable contraction of `f`, then the separable degree of `f` is equal to
the degree of `g`. -/
theorem IsSeparableContraction.natSepDegree_eq {g : Polynomial F} {q : ℕ} [ExpChar F q]
    (h : IsSeparableContraction q f g) : f.natSepDegree = g.natDegree := by
  /-
    F : Type u
    inst✝¹ : Field F
    f g : Polynomial F
    q : Nat
    inst✝ : ExpChar F q
    h : Polynomial.IsSeparableContraction q f g
    ⊢ Eq f.natSepDegree g.natDegree
  -/
  obtain ⟨h1, m, h2⟩ := h
  /-
    case intro.intro
    F : Type u
    inst✝¹ : Field F
    f g : Polynomial F
    q : Nat
    inst✝ : ExpChar F q
    h1 : g.Separable
    m : Nat
    h2 : Eq ((Polynomial.expand F (HPow.hPow q m)) g) f
    ⊢ Eq f.natSepDegree g.natDegree
  -/
  rw [← h2, natSepDegree_expand, h1.natSepDegree_eq_natDegree]
  /-
    🎉 no goals
  -/


variable {f} in
/-- If a polynomial has separable contraction, then its separable degree is equal to the degree of
the given separable contraction. -/
theorem HasSeparableContraction.natSepDegree_eq
    {q : ℕ} [ExpChar F q] (hf : f.HasSeparableContraction q) :
    f.natSepDegree = hf.degree := hf.isSeparableContraction.natSepDegree_eq


/-- The separable degree of an irreducible polynomial divides its degree. -/
theorem natSepDegree_dvd_natDegree (h : Irreducible f) :
    f.natSepDegree ∣ f.natDegree := by
  /-
    F : Type u
    inst✝ : Field F
    f : Polynomial F
    h : Irreducible f
    ⊢ Dvd.dvd f.natSepDegree f.natDegree
  -/
  obtain ⟨q, _⟩ := ExpChar.exists F
  /-
    case intro
    F : Type u
    inst✝ : Field F
    f : Polynomial F
    h : Irreducible f
    q : Nat
    h✝ : ExpChar F q
    ⊢ Dvd.dvd f.natSepDegree f.natDegree
  -/
  have hf := h.hasSeparableContraction q
  /-
    case intro
    F : Type u
    inst✝ : Field F
    f : Polynomial F
    h : Irreducible f
    q : Nat
    h✝ : ExpChar F q
    hf : Polynomial.HasSeparableContraction q f
    ⊢ Dvd.dvd f.natSepDegree f.natDegree
  -/
  rw [hf.natSepDegree_eq]
  /-
    case intro
    F : Type u
    inst✝ : Field F
    f : Polynomial F
    h : Irreducible f
    q : Nat
    h✝ : ExpChar F q
    hf : Polynomial.HasSeparableContraction q f
    ⊢ Dvd.dvd hf.degree f.natDegree
  -/
  exact hf.dvd_degree
  /-
    🎉 no goals
  -/


/-- A monic irreducible polynomial over a field `F` of exponential characteristic `q` has
separable degree one if and only if it is of the form `Polynomial.expand F (q ^ n) (X - C y)`
for some `n : ℕ` and `y : F`. -/
theorem natSepDegree_eq_one_iff_of_monic' (q : ℕ) [ExpChar F q] (hm : f.Monic)
    (hi : Irreducible f) : f.natSepDegree = 1 ↔
    ∃ (n : ℕ) (y : F), f = expand F (q ^ n) (X - C y) := by
  /-
    F : Type u
    inst✝¹ : Field F
    f : Polynomial F
    q : Nat
    inst✝ : ExpChar F q
    hm : f.Monic
    hi : Irreducible f
    ⊢ Iff (Eq f.natSepDegree 1) (Exists fun n => Exists fun y => Eq f ((Polynomial …
  -/
  refine ⟨fun h ↦ ?_, fun ⟨n, y, h⟩ ↦ ?_⟩
    /-
      case refine_1
      F : Type u
      inst✝¹ : Field F
      f : Polynomial F
      q : Nat
      inst✝ : ExpChar F q
      hm : f.Monic
      hi : Irreducible f
      h : Eq f.natSepDegree 1
      ⊢ Exists fun n => Exists fun y => Eq f ((Polynomial.expand F (HPow.hPow q n))  …
    -/
  · obtain ⟨g, h1, n, rfl⟩ := hi.hasSeparableContraction q
    have h2 : g.natDegree = 1 := by
      rwa [natSepDegree_expand _ q, h1.natSepDegree_eq_natDegree] at h
    /-
      case refine_1.intro.intro.intro
      F : Type u
      inst✝¹ : Field F
      q : Nat
      inst✝ : ExpChar F q
      g : Polynomial F
      h1 : g.Separable
      n : Nat
      hm : ((Polynomial.expand F (HPow.hPow q n)) g).Monic
      hi : Irreducible ((Polynomial.expand F (HPow.hPow q n)) g)
      h : Eq ((Polynomial.expand F (HPow.hPow q n)) g).natSepDegree 1
      h2 : Eq g.natDegree 1
      ⊢ Exists fun n_1 => Exists fun y => Eq ((Polynomial.expand F (HPow.hPow q n))  …
    -/
    rw [((monic_expand_iff <| expChar_pow_pos F q n).mp hm).eq_X_add_C h2]
    /-
      case refine_1.intro.intro.intro
      F : Type u
      inst✝¹ : Field F
      q : Nat
      inst✝ : ExpChar F q
      g : Polynomial F
      h1 : g.Separable
      n : Nat
      hm : ((Polynomial.expand F (HPow.hPow q n)) g).Monic
      hi : Irreducible ((Polynomial.expand F (HPow.hPow q n)) g)
      h : Eq ((Polynomial.expand F (HPow.hPow q n)) g).natSepDegree 1
      h2 : Eq g.natDegree 1
      ⊢ Exists fun n_1 => Exists fun y => Eq ((Polynomial.expand F (HPow.hPow q n))  …
    -/
    exact ⟨n, -(g.coeff 0), by rw [map_neg, sub_neg_eq_add]⟩
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    F : Type u
    inst✝¹ : Field F
    f : Polynomial F
    q : Nat
    inst✝ : ExpChar F q
    hm : f.Monic
    hi : Irreducible f
    x✝ : Exists fun n => Exists fun y => Eq f ((Polynomial.expand F (HPow.hPow q n …
    n : Nat
    y : F
    h : Eq f ((Polynomial.expand F (HPow.hPow q n)) (HSub.hSub Polynomial.X (Polyn …
    ⊢ Eq f.natSepDegree 1
  -/
  rw [h, natSepDegree_expand _ q, natSepDegree_X_sub_C]
  /-
    🎉 no goals
  -/


/-- A monic irreducible polynomial over a field `F` of exponential characteristic `q` has
separable degree one if and only if it is of the form `X ^ (q ^ n) - C y`
for some `n : ℕ` and `y : F`. -/
theorem natSepDegree_eq_one_iff_of_monic (q : ℕ) [ExpChar F q] (hm : f.Monic)
    (hi : Irreducible f) : f.natSepDegree = 1 ↔ ∃ (n : ℕ) (y : F), f = X ^ q ^ n - C y := by
  /-
    F : Type u
    inst✝¹ : Field F
    f : Polynomial F
    q : Nat
    inst✝ : ExpChar F q
    hm : f.Monic
    hi : Irreducible f
    ⊢ Iff (Eq f.natSepDegree 1) (Exists fun n => Exists fun y => Eq f (HSub.hSub ( …
  -/
  simp_rw [hi.natSepDegree_eq_one_iff_of_monic' q hm, map_sub, expand_X, expand_C]
  /-
    🎉 no goals
  -/


alias natSepDegree_eq_one_iff_of_irreducible' := Irreducible.natSepDegree_eq_one_iff_of_monic'


alias natSepDegree_eq_one_iff_of_irreducible := Irreducible.natSepDegree_eq_one_iff_of_monic


/-- If a monic polynomial of separable degree one splits, then it is of form `(X - C y) ^ m` for
some non-zero natural number `m` and some element `y` of `F`. -/
theorem eq_X_sub_C_pow_of_natSepDegree_eq_one_of_splits (hm : f.Monic)
    (hs : f.Splits (RingHom.id F))
    (h : f.natSepDegree = 1) : ∃ (m : ℕ) (y : F), m ≠ 0 ∧ f = (X - C y) ^ m := by
  classical
  have h1 := eq_prod_roots_of_monic_of_splits_id hm hs
  have h2 := (natSepDegree_eq_of_splits f hs).symm
  rw [h, aroots_def, Algebra.id.map_eq_id, map_id, Multiset.toFinset_card_eq_one_iff] at h2
  obtain ⟨h2, y, h3⟩ := h2
  exact ⟨_, y, h2, by rwa [h3, Multiset.map_nsmul, Multiset.map_singleton, Multiset.prod_nsmul,
    Multiset.prod_singleton] at h1⟩


/-- If a monic irreducible polynomial over a field `F` of exponential characteristic `q` has
separable degree one, then it is of the form `X ^ (q ^ n) - C y` for some natural number `n`,
and some element `y` of `F`, such that either `n = 0` or `y` has no `q`-th root in `F`. -/
theorem eq_X_pow_char_pow_sub_C_of_natSepDegree_eq_one_of_irreducible (q : ℕ) [ExpChar F q]
    (hm : f.Monic) (hi : Irreducible f) (h : f.natSepDegree = 1) : ∃ (n : ℕ) (y : F),
      (n = 0 ∨ y ∉ (frobenius F q).range) ∧ f = X ^ q ^ n - C y := by
  /-
    F : Type u
    inst✝¹ : Field F
    f : Polynomial F
    q : Nat
    inst✝ : ExpChar F q
    hm : f.Monic
    hi : Irreducible f
    h : Eq f.natSepDegree 1
    ⊢ Exists fun n => Exists fun y => And (Or (Eq n 0) (Not (Membership.mem (frobe …
  -/
  obtain ⟨n, y, hf⟩ := (hm.natSepDegree_eq_one_iff_of_irreducible q hi).1 h
  cases id ‹ExpChar F q› with
  | zero =>
    simp_rw [one_pow, pow_one] at hf ⊢
    exact ⟨0, y, .inl rfl, hf⟩
  | prime hq =>
    refine ⟨n, y, (em _).imp id fun hn ⟨z, hy⟩ ↦ ?_, hf⟩
    haveI := expChar_of_injective_ringHom (R := F) C_injective q
    rw [hf, ← Nat.succ_pred hn, pow_succ, pow_mul, ← hy, frobenius_def, map_pow,
      ← sub_pow_expChar] at hi
    exact not_irreducible_pow hq.ne_one hi


/-- If a monic polynomial over a field `F` of exponential characteristic `q` has separable degree
one, then it is of the form `(X ^ (q ^ n) - C y) ^ m` for some non-zero natural number `m`,
some natural number `n`, and some element `y` of `F`, such that either `n = 0` or `y` has no
`q`-th root in `F`. -/
theorem eq_X_pow_char_pow_sub_C_pow_of_natSepDegree_eq_one (q : ℕ) [ExpChar F q] (hm : f.Monic)
    (h : f.natSepDegree = 1) : ∃ (m n : ℕ) (y : F),
      m ≠ 0 ∧ (n = 0 ∨ y ∉ (frobenius F q).range) ∧ f = (X ^ q ^ n - C y) ^ m := by
  obtain ⟨p, hM, hI, hf⟩ := exists_monic_irreducible_factor _ <| not_isUnit_of_natDegree_pos _
    <| Nat.pos_of_ne_zero <| (natSepDegree_ne_zero_iff _).1 (h.symm ▸ Nat.one_ne_zero)
  have hD := (h ▸ natSepDegree_le_of_dvd p f hf hm.ne_zero).antisymm <|
    Nat.pos_of_ne_zero <| (natSepDegree_ne_zero_iff _).2 hI.natDegree_pos.ne'
  /-
    case intro.intro.intro
    F : Type u
    inst✝¹ : Field F
    f : Polynomial F
    q : Nat
    inst✝ : ExpChar F q
    hm : f.Monic
    h : Eq f.natSepDegree 1
    p : Polynomial F
    hM : p.Monic
    hI : Irreducible p
    hf : Dvd.dvd p f
    hD : Eq p.natSepDegree 1
    ⊢ Exists fun m => Exists fun n => Exists fun y => And (Ne m 0) (And (Or (Eq n  …
  -/
  obtain ⟨n, y, H, hp⟩ := hM.eq_X_pow_char_pow_sub_C_of_natSepDegree_eq_one_of_irreducible q hI hD
  /-
    case intro.intro.intro.intro.intro.intro
    F : Type u
    inst✝¹ : Field F
    f : Polynomial F
    q : Nat
    inst✝ : ExpChar F q
    hm : f.Monic
    h : Eq f.natSepDegree 1
    p : Polynomial F
    hM : p.Monic
    hI : Irreducible p
    hf : Dvd.dvd p f
    hD : Eq p.natSepDegree 1
    n : Nat
    y : F
    H : Or (Eq n 0) (Not (Membership.mem (frobenius F q).range y))
    hp : Eq p (HSub.hSub (HPow.hPow Polynomial.X (HPow.hPow q n)) (Polynomial.C y))
    ⊢ Exists fun m => Exists fun n => Exists fun y => And (Ne m 0) (And (Or (Eq n  …
  -/
  have hF := finiteMultiplicity_of_degree_pos_of_monic (degree_pos_of_irreducible hI) hM hm.ne_zero
  classical
  have hne := (multiplicity_pos_of_dvd hf).ne'
  refine ⟨_, n, y, hne, H, ?_⟩
  obtain ⟨c, hf, H⟩ := hF.exists_eq_pow_mul_and_not_dvd
  rw [hf, natSepDegree_mul_of_isCoprime _ c <| IsCoprime.pow_left <|
    (hI.coprime_or_dvd c).resolve_right H, natSepDegree_pow_of_ne_zero _ hne, hD,
    add_right_eq_self, natSepDegree_eq_zero_iff] at h
  simpa only [eq_one_of_monic_natDegree_zero ((hM.pow _).of_mul_monic_left (hf ▸ hm)) h,
    mul_one, ← hp] using hf


/-- A monic polynomial over a field `F` of exponential characteristic `q` has separable degree one
if and only if it is of the form `(X ^ (q ^ n) - C y) ^ m` for some non-zero natural number `m`,
some natural number `n`, and some element `y` of `F`. -/
theorem natSepDegree_eq_one_iff (q : ℕ) [ExpChar F q] (hm : f.Monic) :
    f.natSepDegree = 1 ↔ ∃ (m n : ℕ) (y : F), m ≠ 0 ∧ f = (X ^ q ^ n - C y) ^ m := by
  /-
    F : Type u
    inst✝¹ : Field F
    f : Polynomial F
    q : Nat
    inst✝ : ExpChar F q
    hm : f.Monic
    ⊢ Iff (Eq f.natSepDegree 1) (Exists fun m => Exists fun n => Exists fun y => A …
  -/
  refine ⟨fun h ↦ ?_, fun ⟨m, n, y, hm, h⟩ ↦ ?_⟩
    /-
      case refine_1
      F : Type u
      inst✝¹ : Field F
      f : Polynomial F
      q : Nat
      inst✝ : ExpChar F q
      hm : f.Monic
      h : Eq f.natSepDegree 1
      ⊢ Exists fun m => Exists fun n => Exists fun y => And (Ne m 0) (Eq f (HPow.hPo …
    -/
  · obtain ⟨m, n, y, hm, -, h⟩ := hm.eq_X_pow_char_pow_sub_C_pow_of_natSepDegree_eq_one q h
    /-
      case refine_1.intro.intro.intro.intro.intro
      F : Type u
      inst✝¹ : Field F
      f : Polynomial F
      q : Nat
      inst✝ : ExpChar F q
      hm✝ : f.Monic
      h✝ : Eq f.natSepDegree 1
      m n : Nat
      y : F
      hm : Ne m 0
      h : Eq f (HPow.hPow (HSub.hSub (HPow.hPow Polynomial.X (HPow.hPow q n)) (Polyn …
      ⊢ Exists fun m => Exists fun n => Exists fun y => And (Ne m 0) (Eq f (HPow.hPo …
    -/
    exact ⟨m, n, y, hm, h⟩
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    F : Type u
    inst✝¹ : Field F
    f : Polynomial F
    q : Nat
    inst✝ : ExpChar F q
    hm✝ : f.Monic
    x✝ : Exists fun m => Exists fun n => Exists fun y => And (Ne m 0) (Eq f (HPow. …
    m n : Nat
    y : F
    hm : Ne m 0
    h : Eq f (HPow.hPow (HSub.hSub (HPow.hPow Polynomial.X (HPow.hPow q n)) (Polyn …
    ⊢ Eq f.natSepDegree 1
  -/
  simp_rw [h, natSepDegree_pow, hm, ite_false, natSepDegree_X_pow_char_pow_sub_C]
  /-
    🎉 no goals
  -/


/-- The minimal polynomial of an element of `E / F` of exponential characteristic `q` has
separable degree one if and only if the minimal polynomial is of the form
`Polynomial.expand F (q ^ n) (X - C y)` for some `n : ℕ` and `y : F`. -/
theorem natSepDegree_eq_one_iff_eq_expand_X_sub_C : (minpoly F x).natSepDegree = 1 ↔
    ∃ (n : ℕ) (y : F), minpoly F x = expand F (q ^ n) (X - C y) := by
  /-
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Ring E
    inst✝¹ : IsDomain E
    inst✝ : Algebra F E
    q : Nat
    hF : ExpChar F q
    x : E
    ⊢ Iff (Eq (minpoly F x).natSepDegree 1) (Exists fun n => Exists fun y => Eq (m …
  -/
  refine ⟨fun h ↦ ?_, fun ⟨n, y, h⟩ ↦ ?_⟩
  · have halg : IsIntegral F x := by_contra fun h' ↦ by
      simp only [eq_zero h', natSepDegree_zero, zero_ne_one] at h
    exact (minpoly.irreducible halg).natSepDegree_eq_one_iff_of_monic' q
      (minpoly.monic halg) |>.1 h
  /-
    case refine_2
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Ring E
    inst✝¹ : IsDomain E
    inst✝ : Algebra F E
    q : Nat
    hF : ExpChar F q
    x : E
    x✝ : Exists fun n => Exists fun y => Eq (minpoly F x) ((Polynomial.expand F (H …
    n : Nat
    y : F
    h : Eq (minpoly F x) ((Polynomial.expand F (HPow.hPow q n)) (HSub.hSub Polynom …
    ⊢ Eq (minpoly F x).natSepDegree 1
  -/
  rw [h, natSepDegree_expand _ q, natSepDegree_X_sub_C]
  /-
    🎉 no goals
  -/


/-- The minimal polynomial of an element of `E / F` of exponential characteristic `q` has
separable degree one if and only if the minimal polynomial is of the form
`X ^ (q ^ n) - C y` for some `n : ℕ` and `y : F`. -/
theorem natSepDegree_eq_one_iff_eq_X_pow_sub_C : (minpoly F x).natSepDegree = 1 ↔
    ∃ (n : ℕ) (y : F), minpoly F x = X ^ q ^ n - C y := by
  /-
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Ring E
    inst✝¹ : IsDomain E
    inst✝ : Algebra F E
    q : Nat
    hF : ExpChar F q
    x : E
    ⊢ Iff (Eq (minpoly F x).natSepDegree 1) (Exists fun n => Exists fun y => Eq (m …
  -/
  simp only [minpoly.natSepDegree_eq_one_iff_eq_expand_X_sub_C q, map_sub, expand_X, expand_C]
  /-
    🎉 no goals
  -/


/-- The minimal polynomial of an element `x` of `E / F` of exponential characteristic `q` has
separable degree one if and only if `x ^ (q ^ n) ∈ F` for some `n : ℕ`. -/
theorem natSepDegree_eq_one_iff_pow_mem : (minpoly F x).natSepDegree = 1 ↔
    ∃ n : ℕ, x ^ q ^ n ∈ (algebraMap F E).range := by
  /-
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Ring E
    inst✝¹ : IsDomain E
    inst✝ : Algebra F E
    q : Nat
    hF : ExpChar F q
    x : E
    ⊢ Iff (Eq (minpoly F x).natSepDegree 1) (Exists fun n => Membership.mem (algeb …
  -/
  convert_to _ ↔ ∃ (n : ℕ) (y : F), Polynomial.aeval x (X ^ q ^ n - C y) = 0
    /-
      case h.e'_2.a
      F : Type u
      E : Type v
      inst✝³ : Field F
      inst✝² : Ring E
      inst✝¹ : IsDomain E
      inst✝ : Algebra F E
      q : Nat
      hF : ExpChar F q
      x : E
      ⊢ Iff (Exists fun n => Membership.mem (algebraMap F E).range (HPow.hPow x (HPo …
    -/
  · simp_rw [RingHom.mem_range, map_sub, map_pow, aeval_C, aeval_X, sub_eq_zero, eq_comm]
    /-
      🎉 no goals
    -/
  /-
    case convert_2
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Ring E
    inst✝¹ : IsDomain E
    inst✝ : Algebra F E
    q : Nat
    hF : ExpChar F q
    x : E
    ⊢ Iff (Eq (minpoly F x).natSepDegree 1) (Exists fun n => Exists fun y => Eq (( …
  -/
  refine ⟨fun h ↦ ?_, fun ⟨n, y, h⟩ ↦ ?_⟩
    /-
      case convert_2.refine_1
      F : Type u
      E : Type v
      inst✝³ : Field F
      inst✝² : Ring E
      inst✝¹ : IsDomain E
      inst✝ : Algebra F E
      q : Nat
      hF : ExpChar F q
      x : E
      h : Eq (minpoly F x).natSepDegree 1
      ⊢ Exists fun n => Exists fun y => Eq ((Polynomial.aeval x) (HSub.hSub (HPow.hP …
    -/
  · obtain ⟨n, y, hx⟩ := (minpoly.natSepDegree_eq_one_iff_eq_X_pow_sub_C q).1 h
    /-
      case convert_2.refine_1.intro.intro
      F : Type u
      E : Type v
      inst✝³ : Field F
      inst✝² : Ring E
      inst✝¹ : IsDomain E
      inst✝ : Algebra F E
      q : Nat
      hF : ExpChar F q
      x : E
      h : Eq (minpoly F x).natSepDegree 1
      n : Nat
      y : F
      hx : Eq (minpoly F x) (HSub.hSub (HPow.hPow Polynomial.X (HPow.hPow q n)) (Pol …
      ⊢ Exists fun n => Exists fun y => Eq ((Polynomial.aeval x) (HSub.hSub (HPow.hP …
    -/
    exact ⟨n, y, hx ▸ aeval F x⟩
    /-
      🎉 no goals
    -/
  /-
    case convert_2.refine_2
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Ring E
    inst✝¹ : IsDomain E
    inst✝ : Algebra F E
    q : Nat
    hF : ExpChar F q
    x : E
    x✝ : Exists fun n => Exists fun y => Eq ((Polynomial.aeval x) (HSub.hSub (HPow …
    n : Nat
    y : F
    h : Eq ((Polynomial.aeval x) (HSub.hSub (HPow.hPow Polynomial.X (HPow.hPow q n …
    ⊢ Eq (minpoly F x).natSepDegree 1
  -/
  have hnezero := X_pow_sub_C_ne_zero (expChar_pow_pos F q n) y
  refine ((natSepDegree_le_of_dvd _ _ (minpoly.dvd F x h) hnezero).trans_eq <|
    natSepDegree_X_pow_char_pow_sub_C q n y).antisymm ?_
  /-
    case convert_2.refine_2
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Ring E
    inst✝¹ : IsDomain E
    inst✝ : Algebra F E
    q : Nat
    hF : ExpChar F q
    x : E
    x✝ : Exists fun n => Exists fun y => Eq ((Polynomial.aeval x) (HSub.hSub (HPow …
    n : Nat
    y : F
    h : Eq ((Polynomial.aeval x) (HSub.hSub (HPow.hPow Polynomial.X (HPow.hPow q n …
    hnezero : Ne (HSub.hSub (HPow.hPow Polynomial.X (HPow.hPow q n)) (Polynomial.C …
    ⊢ LE.le 1 (minpoly F x).natSepDegree
  -/
  rw [Nat.one_le_iff_ne_zero, natSepDegree_ne_zero_iff, ← Nat.one_le_iff_ne_zero]
  /-
    case convert_2.refine_2
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Ring E
    inst✝¹ : IsDomain E
    inst✝ : Algebra F E
    q : Nat
    hF : ExpChar F q
    x : E
    x✝ : Exists fun n => Exists fun y => Eq ((Polynomial.aeval x) (HSub.hSub (HPow …
    n : Nat
    y : F
    h : Eq ((Polynomial.aeval x) (HSub.hSub (HPow.hPow Polynomial.X (HPow.hPow q n …
    hnezero : Ne (HSub.hSub (HPow.hPow Polynomial.X (HPow.hPow q n)) (Polynomial.C …
    ⊢ LE.le 1 (minpoly F x).natDegree
  -/
  exact minpoly.natDegree_pos <| IsAlgebraic.isIntegral ⟨_, hnezero, h⟩
  /-
    🎉 no goals
  -/


/-- The minimal polynomial of an element `x` of `E / F` of exponential characteristic `q` has
separable degree one if and only if the minimal polynomial is of the form
`(X - x) ^ (q ^ n)` for some `n : ℕ`. -/
theorem natSepDegree_eq_one_iff_eq_X_sub_C_pow : (minpoly F x).natSepDegree = 1 ↔
    ∃ n : ℕ, (minpoly F x).map (algebraMap F E) = (X - C x) ^ q ^ n := by
  /-
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Ring E
    inst✝¹ : IsDomain E
    inst✝ : Algebra F E
    q : Nat
    hF : ExpChar F q
    x : E
    ⊢ Iff (Eq (minpoly F x).natSepDegree 1) (Exists fun n => Eq (Polynomial.map (a …
  -/
  haveI := expChar_of_injective_algebraMap (algebraMap F E).injective q
  /-
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Ring E
    inst✝¹ : IsDomain E
    inst✝ : Algebra F E
    q : Nat
    hF : ExpChar F q
    x : E
    this : ExpChar E q
    ⊢ Iff (Eq (minpoly F x).natSepDegree 1) (Exists fun n => Eq (Polynomial.map (a …
  -/
  haveI := expChar_of_injective_ringHom (C_injective (R := E)) q
  /-
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Ring E
    inst✝¹ : IsDomain E
    inst✝ : Algebra F E
    q : Nat
    hF : ExpChar F q
    x : E
    this✝ : ExpChar E q
    this : ExpChar (Polynomial E) q
    ⊢ Iff (Eq (minpoly F x).natSepDegree 1) (Exists fun n => Eq (Polynomial.map (a …
  -/
  refine ⟨fun h ↦ ?_, fun ⟨n, h⟩ ↦ (natSepDegree_eq_one_iff_pow_mem q).2 ?_⟩
    /-
      case refine_1
      F : Type u
      E : Type v
      inst✝³ : Field F
      inst✝² : Ring E
      inst✝¹ : IsDomain E
      inst✝ : Algebra F E
      q : Nat
      hF : ExpChar F q
      x : E
      this✝ : ExpChar E q
      this : ExpChar (Polynomial E) q
      h : Eq (minpoly F x).natSepDegree 1
      ⊢ Exists fun n => Eq (Polynomial.map (algebraMap F E) (minpoly F x)) (HPow.hPo …
    -/
  · obtain ⟨n, y, h⟩ := (natSepDegree_eq_one_iff_eq_X_pow_sub_C q).1 h
    /-
      case refine_1.intro.intro
      F : Type u
      E : Type v
      inst✝³ : Field F
      inst✝² : Ring E
      inst✝¹ : IsDomain E
      inst✝ : Algebra F E
      q : Nat
      hF : ExpChar F q
      x : E
      this✝ : ExpChar E q
      this : ExpChar (Polynomial E) q
      h✝ : Eq (minpoly F x).natSepDegree 1
      n : Nat
      y : F
      h : Eq (minpoly F x) (HSub.hSub (HPow.hPow Polynomial.X (HPow.hPow q n)) (Poly …
      ⊢ Exists fun n => Eq (Polynomial.map (algebraMap F E) (minpoly F x)) (HPow.hPo …
    -/
    have hx := congr_arg (Polynomial.aeval x) h.symm
    /-
      case refine_1.intro.intro
      F : Type u
      E : Type v
      inst✝³ : Field F
      inst✝² : Ring E
      inst✝¹ : IsDomain E
      inst✝ : Algebra F E
      q : Nat
      hF : ExpChar F q
      x : E
      this✝ : ExpChar E q
      this : ExpChar (Polynomial E) q
      h✝ : Eq (minpoly F x).natSepDegree 1
      n : Nat
      y : F
      h : Eq (minpoly F x) (HSub.hSub (HPow.hPow Polynomial.X (HPow.hPow q n)) (Poly …
      hx : Eq ((Polynomial.aeval x) (HSub.hSub (HPow.hPow Polynomial.X (HPow.hPow q  …
      ⊢ Exists fun n => Eq (Polynomial.map (algebraMap F E) (minpoly F x)) (HPow.hPo …
    -/
    rw [minpoly.aeval, map_sub, map_pow, aeval_X, aeval_C, sub_eq_zero, eq_comm] at hx
    /-
      case refine_1.intro.intro
      F : Type u
      E : Type v
      inst✝³ : Field F
      inst✝² : Ring E
      inst✝¹ : IsDomain E
      inst✝ : Algebra F E
      q : Nat
      hF : ExpChar F q
      x : E
      this✝ : ExpChar E q
      this : ExpChar (Polynomial E) q
      h✝ : Eq (minpoly F x).natSepDegree 1
      n : Nat
      y : F
      h : Eq (minpoly F x) (HSub.hSub (HPow.hPow Polynomial.X (HPow.hPow q n)) (Poly …
      hx : Eq ((algebraMap F E) y) (HPow.hPow x (HPow.hPow q n))
      ⊢ Exists fun n => Eq (Polynomial.map (algebraMap F E) (minpoly F x)) (HPow.hPo …
    -/
    use n
    rw [h, Polynomial.map_sub, Polynomial.map_pow, map_X, map_C, hx, map_pow,
      ← sub_pow_expChar_pow_of_commute _ _ (commute_X _)]
  /-
    case refine_2
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Ring E
    inst✝¹ : IsDomain E
    inst✝ : Algebra F E
    q : Nat
    hF : ExpChar F q
    x : E
    this✝ : ExpChar E q
    this : ExpChar (Polynomial E) q
    x✝ : Exists fun n => Eq (Polynomial.map (algebraMap F E) (minpoly F x)) (HPow. …
    n : Nat
    h : Eq (Polynomial.map (algebraMap F E) (minpoly F x)) (HPow.hPow (HSub.hSub P …
    ⊢ Exists fun n => Membership.mem (algebraMap F E).range (HPow.hPow x (HPow.hPo …
  -/
  apply_fun constantCoeff at h
  /-
    case refine_2
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Ring E
    inst✝¹ : IsDomain E
    inst✝ : Algebra F E
    q : Nat
    hF : ExpChar F q
    x : E
    this✝ : ExpChar E q
    this : ExpChar (Polynomial E) q
    x✝ : Exists fun n => Eq (Polynomial.map (algebraMap F E) (minpoly F x)) (HPow. …
    n : Nat
    h : Eq (Polynomial.constantCoeff (Polynomial.map (algebraMap F E) (minpoly F x …
    ⊢ Exists fun n => Membership.mem (algebraMap F E).range (HPow.hPow x (HPow.hPo …
  -/
  simp_rw [map_pow, map_sub, constantCoeff_apply, coeff_map, coeff_X_zero, coeff_C_zero] at h
  /-
    case refine_2
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Ring E
    inst✝¹ : IsDomain E
    inst✝ : Algebra F E
    q : Nat
    hF : ExpChar F q
    x : E
    this✝ : ExpChar E q
    this : ExpChar (Polynomial E) q
    x✝ : Exists fun n => Eq (Polynomial.map (algebraMap F E) (minpoly F x)) (HPow. …
    n : Nat
    h : Eq ((algebraMap F E) ((minpoly F x).coeff 0)) (HPow.hPow (HSub.hSub 0 x) ( …
    ⊢ Exists fun n => Membership.mem (algebraMap F E).range (HPow.hPow x (HPow.hPo …
  -/
  rw [zero_sub, neg_pow, neg_one_pow_expChar_pow] at h
  /-
    case refine_2
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Ring E
    inst✝¹ : IsDomain E
    inst✝ : Algebra F E
    q : Nat
    hF : ExpChar F q
    x : E
    this✝ : ExpChar E q
    this : ExpChar (Polynomial E) q
    x✝ : Exists fun n => Eq (Polynomial.map (algebraMap F E) (minpoly F x)) (HPow. …
    n : Nat
    h : Eq ((algebraMap F E) ((minpoly F x).coeff 0)) (HMul.hMul (-1) (HPow.hPow x …
    ⊢ Exists fun n => Membership.mem (algebraMap F E).range (HPow.hPow x (HPow.hPo …
  -/
  exact ⟨n, -(minpoly F x).coeff 0, by rw [map_neg, h, neg_mul, one_mul, neg_neg]⟩
  /-
    🎉 no goals
  -/


/-- The separable degree of `F⟮α⟯ / F` is equal to the separable degree of the
minimal polynomial of `α` over `F`. -/
theorem finSepDegree_adjoin_simple_eq_natSepDegree {α : E} (halg : IsAlgebraic F α) :
    finSepDegree F F⟮α⟯ = (minpoly F α).natSepDegree := by
  have : finSepDegree F F⟮α⟯ = _ := Nat.card_congr
    (algHomAdjoinIntegralEquiv F (K := AlgebraicClosure F⟮α⟯) halg.isIntegral)
  classical
  rw [this, Nat.card_eq_fintype_card, natSepDegree_eq_of_isAlgClosed (E := AlgebraicClosure F⟮α⟯),
    ← Fintype.card_coe]
  simp_rw [Multiset.mem_toFinset]

-- The separable degree of `F⟮α⟯ / F` divides the degree of `F⟮α⟯ / F`.
-- Marked as `private` because it is a special case of `finSepDegree_dvd_finrank`.

private theorem finSepDegree_adjoin_simple_dvd_finrank (α : E) :
    finSepDegree F F⟮α⟯ ∣ finrank F F⟮α⟯ := by
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    α : E
    ⊢ Dvd.dvd (Field.finSepDegree F (Subtype fun x => Membership.mem (Intermediate …
  -/
  by_cases halg : IsAlgebraic F α
    /-
      case pos
      F : Type u
      E : Type v
      inst✝² : Field F
      inst✝¹ : Field E
      inst✝ : Algebra F E
      α : E
      halg : IsAlgebraic F α
      ⊢ Dvd.dvd (Field.finSepDegree F (Subtype fun x => Membership.mem (Intermediate …
    -/
  · rw [finSepDegree_adjoin_simple_eq_natSepDegree F E halg, adjoin.finrank halg.isIntegral]
    /-
      case pos
      F : Type u
      E : Type v
      inst✝² : Field F
      inst✝¹ : Field E
      inst✝ : Algebra F E
      α : E
      halg : IsAlgebraic F α
      ⊢ Dvd.dvd (minpoly F α).natSepDegree (minpoly F α).natDegree
    -/
    exact (minpoly.irreducible halg.isIntegral).natSepDegree_dvd_natDegree
    /-
      🎉 no goals
    -/
  have : finrank F F⟮α⟯ = 0 := finrank_of_infinite_dimensional fun _ ↦
    halg ((AdjoinSimple.isIntegral_gen F α).1 (IsIntegral.of_finite F _)).isAlgebraic
  /-
    case neg
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    α : E
    halg : Not (IsAlgebraic F α)
    this : Eq (Module.finrank F (Subtype fun x => Membership.mem (IntermediateFiel …
    ⊢ Dvd.dvd (Field.finSepDegree F (Subtype fun x => Membership.mem (Intermediate …
  -/
  rw [this]
  /-
    case neg
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    α : E
    halg : Not (IsAlgebraic F α)
    this : Eq (Module.finrank F (Subtype fun x => Membership.mem (IntermediateFiel …
    ⊢ Dvd.dvd (Field.finSepDegree F (Subtype fun x => Membership.mem (Intermediate …
  -/
  exact dvd_zero _
  /-
    🎉 no goals
  -/


/-- The separable degree of `F⟮α⟯ / F` is smaller than the degree of `F⟮α⟯ / F` if `α` is
algebraic over `F`. -/
theorem finSepDegree_adjoin_simple_le_finrank (α : E) (halg : IsAlgebraic F α) :
    finSepDegree F F⟮α⟯ ≤ finrank F F⟮α⟯ := by
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    α : E
    halg : IsAlgebraic F α
    ⊢ LE.le (Field.finSepDegree F (Subtype fun x => Membership.mem (IntermediateFi …
  -/
  haveI := adjoin.finiteDimensional halg.isIntegral
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    α : E
    halg : IsAlgebraic F α
    this : FiniteDimensional F (Subtype fun x => Membership.mem (IntermediateField …
    ⊢ LE.le (Field.finSepDegree F (Subtype fun x => Membership.mem (IntermediateFi …
  -/
  exact Nat.le_of_dvd finrank_pos <| finSepDegree_adjoin_simple_dvd_finrank F E α
  /-
    🎉 no goals
  -/


/-- If `α` is algebraic over `F`, then the separable degree of `F⟮α⟯ / F` is equal to the degree
of `F⟮α⟯ / F` if and only if `α` is a separable element. -/
theorem finSepDegree_adjoin_simple_eq_finrank_iff (α : E) (halg : IsAlgebraic F α) :
    finSepDegree F F⟮α⟯ = finrank F F⟮α⟯ ↔ IsSeparable F α := by
  rw [finSepDegree_adjoin_simple_eq_natSepDegree F E halg, adjoin.finrank halg.isIntegral,
    natSepDegree_eq_natDegree_iff _ (minpoly.ne_zero halg.isIntegral), IsSeparable]


/-- The separable degree of any field extension `E / F` divides the degree of `E / F`. -/
theorem finSepDegree_dvd_finrank : finSepDegree F E ∣ finrank F E := by
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    ⊢ Dvd.dvd (Field.finSepDegree F E) (Module.finrank F E)
  -/
  by_cases hfd : FiniteDimensional F E
    /-
      case pos
      F : Type u
      E : Type v
      inst✝² : Field F
      inst✝¹ : Field E
      inst✝ : Algebra F E
      hfd : FiniteDimensional F E
      ⊢ Dvd.dvd (Field.finSepDegree F E) (Module.finrank F E)
    -/
  · rw [← finSepDegree_top F, ← finrank_top F E]
    refine induction_on_adjoin (fun K : IntermediateField F E ↦ finSepDegree F K ∣ finrank F K)
      (by simp_rw [finSepDegree_bot, IntermediateField.finrank_bot, one_dvd]) (fun L x h ↦ ?_) ⊤
    /-
      case pos
      F : Type u
      E : Type v
      inst✝² : Field F
      inst✝¹ : Field E
      inst✝ : Algebra F E
      hfd : FiniteDimensional F E
      L : IntermediateField F E
      x : E
      h : (fun K => Dvd.dvd (Field.finSepDegree F (Subtype fun x => Membership.mem K …
      ⊢ (fun K => Dvd.dvd (Field.finSepDegree F (Subtype fun x => Membership.mem K x …
    -/
    simp only at h ⊢
    /-
      case pos
      F : Type u
      E : Type v
      inst✝² : Field F
      inst✝¹ : Field E
      inst✝ : Algebra F E
      hfd : FiniteDimensional F E
      L : IntermediateField F E
      x : E
      h : Dvd.dvd (Field.finSepDegree F (Subtype fun x => Membership.mem L x)) (Modu …
      ⊢ Dvd.dvd (Field.finSepDegree F (Subtype fun x_1 => Membership.mem (Intermedia …
    -/
    have hdvd := mul_dvd_mul h <| finSepDegree_adjoin_simple_dvd_finrank L E x
    /-
      case pos
      F : Type u
      E : Type v
      inst✝² : Field F
      inst✝¹ : Field E
      inst✝ : Algebra F E
      hfd : FiniteDimensional F E
      L : IntermediateField F E
      x : E
      h : Dvd.dvd (Field.finSepDegree F (Subtype fun x => Membership.mem L x)) (Modu …
      hdvd : Dvd.dvd (HMul.hMul (Field.finSepDegree F (Subtype fun x => Membership.m …
      ⊢ Dvd.dvd (Field.finSepDegree F (Subtype fun x_1 => Membership.mem (Intermedia …
    -/
    set M := L⟮x⟯
    /-
      case pos
      F : Type u
      E : Type v
      inst✝² : Field F
      inst✝¹ : Field E
      inst✝ : Algebra F E
      hfd : FiniteDimensional F E
      L : IntermediateField F E
      x : E
      h : Dvd.dvd (Field.finSepDegree F (Subtype fun x => Membership.mem L x)) (Modu …
      M : IntermediateField (Subtype fun x => Membership.mem L x) E := IntermediateF …
      hdvd : Dvd.dvd (HMul.hMul (Field.finSepDegree F (Subtype fun x => Membership.m …
      ⊢ Dvd.dvd (Field.finSepDegree F (Subtype fun x => Membership.mem (Intermediate …
    -/
    have := Algebra.IsAlgebraic.of_finite L M
    rwa [finSepDegree_mul_finSepDegree_of_isAlgebraic F L M,
      Module.finrank_mul_finrank F L M] at hdvd
  /-
    case neg
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    hfd : Not (FiniteDimensional F E)
    ⊢ Dvd.dvd (Field.finSepDegree F E) (Module.finrank F E)
  -/
  rw [finrank_of_infinite_dimensional hfd]
  /-
    case neg
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    hfd : Not (FiniteDimensional F E)
    ⊢ Dvd.dvd (Field.finSepDegree F E) 0
  -/
  exact dvd_zero _
  /-
    🎉 no goals
  -/


/-- The separable degree of a finite extension `E / F` is smaller than the degree of `E / F`. -/
@[stacks 09HA "The inequality"]
theorem finSepDegree_le_finrank [FiniteDimensional F E] :
    finSepDegree F E ≤ finrank F E := Nat.le_of_dvd finrank_pos <| finSepDegree_dvd_finrank F E


/-- If `E / F` is a separable extension, then its separable degree is equal to its degree.
When `E / F` is infinite, it means that `Field.Emb F E` has infinitely many elements.
(But the cardinality of `Field.Emb F E` is not equal to `Module.rank F E` in general!) -/
theorem finSepDegree_eq_finrank_of_isSeparable [Algebra.IsSeparable F E] :
    finSepDegree F E = finrank F E := by
  /-
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Algebra.IsSeparable F E
    ⊢ Eq (Field.finSepDegree F E) (Module.finrank F E)
  -/
  wlog hfd : FiniteDimensional F E generalizing E with H
    /-
      case inr
      F : Type u
      E : Type v
      inst✝³ : Field F
      inst✝² : Field E
      inst✝¹ : Algebra F E
      inst✝ : Algebra.IsSeparable F E
      H : ∀ (E : Type v) [inst : Field E] [inst_1 : Algebra F E] [inst_2 : Algebra.I …
      hfd : Not (FiniteDimensional F E)
      ⊢ Eq (Field.finSepDegree F E) (Module.finrank F E)
    -/
  · rw [finrank_of_infinite_dimensional hfd]
    /-
      case inr
      F : Type u
      E : Type v
      inst✝³ : Field F
      inst✝² : Field E
      inst✝¹ : Algebra F E
      inst✝ : Algebra.IsSeparable F E
      H : ∀ (E : Type v) [inst : Field E] [inst_1 : Algebra F E] [inst_2 : Algebra.I …
      hfd : Not (FiniteDimensional F E)
      ⊢ Eq (Field.finSepDegree F E) 0
    -/
    have halg := Algebra.IsSeparable.isAlgebraic F E
    /-
      case inr
      F : Type u
      E : Type v
      inst✝³ : Field F
      inst✝² : Field E
      inst✝¹ : Algebra F E
      inst✝ : Algebra.IsSeparable F E
      H : ∀ (E : Type v) [inst : Field E] [inst_1 : Algebra F E] [inst_2 : Algebra.I …
      hfd : Not (FiniteDimensional F E)
      halg : Algebra.IsAlgebraic F E
      ⊢ Eq (Field.finSepDegree F E) 0
    -/
    obtain ⟨L, h, h'⟩ := exists_lt_finrank_of_infinite_dimensional hfd (finSepDegree F E)
    /-
      case inr.intro.intro
      F : Type u
      E : Type v
      inst✝³ : Field F
      inst✝² : Field E
      inst✝¹ : Algebra F E
      inst✝ : Algebra.IsSeparable F E
      H : ∀ (E : Type v) [inst : Field E] [inst_1 : Algebra F E] [inst_2 : Algebra.I …
      hfd : Not (FiniteDimensional F E)
      halg : Algebra.IsAlgebraic F E
      L : IntermediateField F E
      h : FiniteDimensional F (Subtype fun x => Membership.mem L x)
      h' : LT.lt (Field.finSepDegree F E) (Module.finrank F (Subtype fun x => Member …
      ⊢ Eq (Field.finSepDegree F E) 0
    -/
    have : Algebra.IsSeparable F L := Algebra.isSeparable_tower_bot_of_isSeparable F L E
    /-
      case inr.intro.intro
      F : Type u
      E : Type v
      inst✝³ : Field F
      inst✝² : Field E
      inst✝¹ : Algebra F E
      inst✝ : Algebra.IsSeparable F E
      H : ∀ (E : Type v) [inst : Field E] [inst_1 : Algebra F E] [inst_2 : Algebra.I …
      hfd : Not (FiniteDimensional F E)
      halg : Algebra.IsAlgebraic F E
      L : IntermediateField F E
      h : FiniteDimensional F (Subtype fun x => Membership.mem L x)
      h' : LT.lt (Field.finSepDegree F E) (Module.finrank F (Subtype fun x => Member …
      this : Algebra.IsSeparable F (Subtype fun x => Membership.mem L x)
      ⊢ Eq (Field.finSepDegree F E) 0
    -/
    have := (halg.tower_top L)
    /-
      case inr.intro.intro
      F : Type u
      E : Type v
      inst✝³ : Field F
      inst✝² : Field E
      inst✝¹ : Algebra F E
      inst✝ : Algebra.IsSeparable F E
      H : ∀ (E : Type v) [inst : Field E] [inst_1 : Algebra F E] [inst_2 : Algebra.I …
      hfd : Not (FiniteDimensional F E)
      halg : Algebra.IsAlgebraic F E
      L : IntermediateField F E
      h : FiniteDimensional F (Subtype fun x => Membership.mem L x)
      h' : LT.lt (Field.finSepDegree F E) (Module.finrank F (Subtype fun x => Member …
      this✝ : Algebra.IsSeparable F (Subtype fun x => Membership.mem L x)
      this : Algebra.IsAlgebraic (Subtype fun x => Membership.mem L x) E
      ⊢ Eq (Field.finSepDegree F E) 0
    -/
    have hd := finSepDegree_mul_finSepDegree_of_isAlgebraic F L E
    /-
      case inr.intro.intro
      F : Type u
      E : Type v
      inst✝³ : Field F
      inst✝² : Field E
      inst✝¹ : Algebra F E
      inst✝ : Algebra.IsSeparable F E
      H : ∀ (E : Type v) [inst : Field E] [inst_1 : Algebra F E] [inst_2 : Algebra.I …
      hfd : Not (FiniteDimensional F E)
      halg : Algebra.IsAlgebraic F E
      L : IntermediateField F E
      h : FiniteDimensional F (Subtype fun x => Membership.mem L x)
      h' : LT.lt (Field.finSepDegree F E) (Module.finrank F (Subtype fun x => Member …
      this✝ : Algebra.IsSeparable F (Subtype fun x => Membership.mem L x)
      this : Algebra.IsAlgebraic (Subtype fun x => Membership.mem L x) E
      hd : Eq (HMul.hMul (Field.finSepDegree F (Subtype fun x => Membership.mem L x) …
      ⊢ Eq (Field.finSepDegree F E) 0
    -/
    rw [H L h] at hd
    /-
      case inr.intro.intro
      F : Type u
      E : Type v
      inst✝³ : Field F
      inst✝² : Field E
      inst✝¹ : Algebra F E
      inst✝ : Algebra.IsSeparable F E
      H : ∀ (E : Type v) [inst : Field E] [inst_1 : Algebra F E] [inst_2 : Algebra.I …
      hfd : Not (FiniteDimensional F E)
      halg : Algebra.IsAlgebraic F E
      L : IntermediateField F E
      h : FiniteDimensional F (Subtype fun x => Membership.mem L x)
      h' : LT.lt (Field.finSepDegree F E) (Module.finrank F (Subtype fun x => Member …
      this✝ : Algebra.IsSeparable F (Subtype fun x => Membership.mem L x)
      this : Algebra.IsAlgebraic (Subtype fun x => Membership.mem L x) E
      hd : Eq (HMul.hMul (Module.finrank F (Subtype fun x => Membership.mem L x)) (F …
      ⊢ Eq (Field.finSepDegree F E) 0
    -/
    by_cases hd' : finSepDegree L E = 0
      /-
        case pos
        F : Type u
        E : Type v
        inst✝³ : Field F
        inst✝² : Field E
        inst✝¹ : Algebra F E
        inst✝ : Algebra.IsSeparable F E
        H : ∀ (E : Type v) [inst : Field E] [inst_1 : Algebra F E] [inst_2 : Algebra.I …
        hfd : Not (FiniteDimensional F E)
        halg : Algebra.IsAlgebraic F E
        L : IntermediateField F E
        h : FiniteDimensional F (Subtype fun x => Membership.mem L x)
        h' : LT.lt (Field.finSepDegree F E) (Module.finrank F (Subtype fun x => Member …
        this✝ : Algebra.IsSeparable F (Subtype fun x => Membership.mem L x)
        this : Algebra.IsAlgebraic (Subtype fun x => Membership.mem L x) E
        hd : Eq (HMul.hMul (Module.finrank F (Subtype fun x => Membership.mem L x)) (F …
        hd' : Eq (Field.finSepDegree (Subtype fun x => Membership.mem L x) E) 0
        ⊢ Eq (Field.finSepDegree F E) 0
      -/
    · rw [← hd, hd', mul_zero]
      /-
        🎉 no goals
      -/
    /-
      case neg
      F : Type u
      E : Type v
      inst✝³ : Field F
      inst✝² : Field E
      inst✝¹ : Algebra F E
      inst✝ : Algebra.IsSeparable F E
      H : ∀ (E : Type v) [inst : Field E] [inst_1 : Algebra F E] [inst_2 : Algebra.I …
      hfd : Not (FiniteDimensional F E)
      halg : Algebra.IsAlgebraic F E
      L : IntermediateField F E
      h : FiniteDimensional F (Subtype fun x => Membership.mem L x)
      h' : LT.lt (Field.finSepDegree F E) (Module.finrank F (Subtype fun x => Member …
      this✝ : Algebra.IsSeparable F (Subtype fun x => Membership.mem L x)
      this : Algebra.IsAlgebraic (Subtype fun x => Membership.mem L x) E
      hd : Eq (HMul.hMul (Module.finrank F (Subtype fun x => Membership.mem L x)) (F …
      hd' : Not (Eq (Field.finSepDegree (Subtype fun x => Membership.mem L x) E) 0)
      ⊢ Eq (Field.finSepDegree F E) 0
    -/
    linarith only [h', hd, Nat.le_mul_of_pos_right (finrank F L) (Nat.pos_of_ne_zero hd')]
    /-
      🎉 no goals
    -/
  /-
    F : Type u
    E✝ : Type v
    inst✝⁵ : Field F
    inst✝⁴ : Field E✝
    inst✝³ : Algebra F E✝
    E : Type v
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Algebra.IsSeparable F E
    hfd : FiniteDimensional F E
    ⊢ Eq (Field.finSepDegree F E) (Module.finrank F E)
  -/
  rw [← finSepDegree_top F, ← finrank_top F E]
  refine induction_on_adjoin (fun K : IntermediateField F E ↦ finSepDegree F K = finrank F K)
    (by simp_rw [finSepDegree_bot, IntermediateField.finrank_bot]) (fun L x h ↦ ?_) ⊤
  /-
    F : Type u
    E✝ : Type v
    inst✝⁵ : Field F
    inst✝⁴ : Field E✝
    inst✝³ : Algebra F E✝
    E : Type v
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Algebra.IsSeparable F E
    hfd : FiniteDimensional F E
    L : IntermediateField F E
    x : E
    h : (fun K => Eq (Field.finSepDegree F (Subtype fun x => Membership.mem K x))  …
    ⊢ (fun K => Eq (Field.finSepDegree F (Subtype fun x => Membership.mem K x)) (M …
  -/
  simp only at h ⊢
  have heq : _ * _ = _ * _ := congr_arg₂ (· * ·) h <|
    (finSepDegree_adjoin_simple_eq_finrank_iff L E x (IsAlgebraic.of_finite L x)).2 <|
      IsSeparable.tower_top L (Algebra.IsSeparable.isSeparable F x)
  /-
    F : Type u
    E✝ : Type v
    inst✝⁵ : Field F
    inst✝⁴ : Field E✝
    inst✝³ : Algebra F E✝
    E : Type v
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Algebra.IsSeparable F E
    hfd : FiniteDimensional F E
    L : IntermediateField F E
    x : E
    h : Eq (Field.finSepDegree F (Subtype fun x => Membership.mem L x)) (Module.fi …
    heq : Eq (HMul.hMul (Field.finSepDegree F (Subtype fun x => Membership.mem L x …
    ⊢ Eq (Field.finSepDegree F (Subtype fun x_1 => Membership.mem (IntermediateFie …
  -/
  set M := L⟮x⟯
  /-
    F : Type u
    E✝ : Type v
    inst✝⁵ : Field F
    inst✝⁴ : Field E✝
    inst✝³ : Algebra F E✝
    E : Type v
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Algebra.IsSeparable F E
    hfd : FiniteDimensional F E
    L : IntermediateField F E
    x : E
    h : Eq (Field.finSepDegree F (Subtype fun x => Membership.mem L x)) (Module.fi …
    M : IntermediateField (Subtype fun x => Membership.mem L x) E := IntermediateF …
    heq : Eq (HMul.hMul (Field.finSepDegree F (Subtype fun x => Membership.mem L x …
    ⊢ Eq (Field.finSepDegree F (Subtype fun x => Membership.mem (IntermediateField …
  -/
  have := Algebra.IsAlgebraic.of_finite L M
  rwa [finSepDegree_mul_finSepDegree_of_isAlgebraic F L M,
    Module.finrank_mul_finrank F L M] at heq


alias Algebra.IsSeparable.finSepDegree_eq := finSepDegree_eq_finrank_of_isSeparable


/-- If `E / F` is a finite extension, then its separable degree is equal to its degree if and
only if it is a separable extension. -/
@[stacks 09HA "The equality condition"]
theorem finSepDegree_eq_finrank_iff [FiniteDimensional F E] :
    finSepDegree F E = finrank F E ↔ Algebra.IsSeparable F E :=
  ⟨fun heq ↦ ⟨fun x ↦ by
    /-
      F : Type u
      E : Type v
      inst✝³ : Field F
      inst✝² : Field E
      inst✝¹ : Algebra F E
      inst✝ : FiniteDimensional F E
      heq : Eq (Field.finSepDegree F E) (Module.finrank F E)
      x : E
      ⊢ IsSeparable F x
    -/
    have halg := IsAlgebraic.of_finite F x
    refine (finSepDegree_adjoin_simple_eq_finrank_iff F E x halg).1 <| le_antisymm
      (finSepDegree_adjoin_simple_le_finrank F E x halg) <| le_of_not_lt fun h ↦ ?_
    /-
      F : Type u
      E : Type v
      inst✝³ : Field F
      inst✝² : Field E
      inst✝¹ : Algebra F E
      inst✝ : FiniteDimensional F E
      heq : Eq (Field.finSepDegree F E) (Module.finrank F E)
      x : E
      halg : IsAlgebraic F x
      h : LT.lt (Field.finSepDegree F (Subtype fun x_1 => Membership.mem (Intermedia …
      ⊢ False
    -/
    have := Nat.mul_lt_mul_of_lt_of_le' h (finSepDegree_le_finrank F⟮x⟯ E) Fin.pos'
    rw [finSepDegree_mul_finSepDegree_of_isAlgebraic F F⟮x⟯ E,
      Module.finrank_mul_finrank F F⟮x⟯ E] at this
    /-
      F : Type u
      E : Type v
      inst✝³ : Field F
      inst✝² : Field E
      inst✝¹ : Algebra F E
      inst✝ : FiniteDimensional F E
      heq : Eq (Field.finSepDegree F E) (Module.finrank F E)
      x : E
      halg : IsAlgebraic F x
      h : LT.lt (Field.finSepDegree F (Subtype fun x_1 => Membership.mem (Intermedia …
      this : LT.lt (Field.finSepDegree F E) (Module.finrank F E)
      ⊢ False
    -/
    linarith only [heq, this]⟩, fun _ ↦ finSepDegree_eq_finrank_of_isSeparable F E⟩
    /-
      🎉 no goals
    -/


lemma IntermediateField.isSeparable_of_mem_isSeparable {L : IntermediateField F E}
    [Algebra.IsSeparable F L] {x : E} (h : x ∈ L) : IsSeparable F x := by
  /-
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Field E
    inst✝¹ : Algebra F E
    L : IntermediateField F E
    inst✝ : Algebra.IsSeparable F (Subtype fun x => Membership.mem L x)
    x : E
    h : Membership.mem L x
    ⊢ IsSeparable F x
  -/
  simpa only [IsSeparable, minpoly_eq] using Algebra.IsSeparable.isSeparable F (K := L) ⟨x, h⟩
  /-
    🎉 no goals
  -/


/-- `F⟮x⟯ / F` is a separable extension if and only if `x` is a separable element.
As a consequence, any rational function of `x` is also a separable element. -/
theorem IntermediateField.isSeparable_adjoin_simple_iff_isSeparable {x : E} :
    Algebra.IsSeparable F F⟮x⟯ ↔ IsSeparable F x := by
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    x : E
    ⊢ Iff (Algebra.IsSeparable F (Subtype fun x_1 => Membership.mem (IntermediateF …
  -/
  refine ⟨fun _ ↦ ?_, fun hsep ↦ ?_⟩
    /-
      case refine_1
      F : Type u
      E : Type v
      inst✝² : Field F
      inst✝¹ : Field E
      inst✝ : Algebra F E
      x : E
      x✝ : Algebra.IsSeparable F (Subtype fun x_1 => Membership.mem (IntermediateFie …
      ⊢ IsSeparable F x
    -/
  · exact isSeparable_of_mem_isSeparable F E <| mem_adjoin_simple_self F x
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      F : Type u
      E : Type v
      inst✝² : Field F
      inst✝¹ : Field E
      inst✝ : Algebra F E
      x : E
      hsep : IsSeparable F x
      ⊢ Algebra.IsSeparable F (Subtype fun x_1 => Membership.mem (IntermediateField. …
    -/
  · have h := IsSeparable.isIntegral hsep
    /-
      case refine_2
      F : Type u
      E : Type v
      inst✝² : Field F
      inst✝¹ : Field E
      inst✝ : Algebra F E
      x : E
      hsep : IsSeparable F x
      h : IsIntegral F x
      ⊢ Algebra.IsSeparable F (Subtype fun x_1 => Membership.mem (IntermediateField. …
    -/
    haveI := adjoin.finiteDimensional h
    rwa [← finSepDegree_eq_finrank_iff,
      finSepDegree_adjoin_simple_eq_finrank_iff F E x h.isAlgebraic]


variable {E K} in
/-- If `K / E / F` is an extension tower such that `E / F` is separable,
`x : K` is separable over `E`, then it's also separable over `F`. -/
theorem IsSeparable.of_algebra_isSeparable_of_isSeparable [Algebra E K] [IsScalarTower F E K]
    [Algebra.IsSeparable F E] {x : K} (hsep : IsSeparable E x) : IsSeparable F x := by
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : Algebra.IsSeparable F E
    x : K
    hsep : IsSeparable E x
    ⊢ IsSeparable F x
  -/
  set f := minpoly E x with hf
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : Algebra.IsSeparable F E
    x : K
    hsep : IsSeparable E x
    f : Polynomial E := minpoly E x
    hf : Eq f (minpoly E x)
    ⊢ IsSeparable F x
  -/
  let E' : IntermediateField F E := adjoin F f.coeffs
  haveI : FiniteDimensional F E' :=
    finiteDimensional_adjoin fun x _ ↦ Algebra.IsSeparable.isIntegral F x
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : Algebra.IsSeparable F E
    x : K
    hsep : IsSeparable E x
    f : Polynomial E := minpoly E x
    hf : Eq f (minpoly E x)
    E' : IntermediateField F E := IntermediateField.adjoin F ↑f.coeffs
    this : FiniteDimensional F (Subtype fun x => Membership.mem E' x)
    ⊢ IsSeparable F x
  -/
  let g : E'[X] := f.toSubring E'.toSubring (subset_adjoin F _)
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : Algebra.IsSeparable F E
    x : K
    hsep : IsSeparable E x
    f : Polynomial E := minpoly E x
    hf : Eq f (minpoly E x)
    E' : IntermediateField F E := IntermediateField.adjoin F ↑f.coeffs
    this : FiniteDimensional F (Subtype fun x => Membership.mem E' x)
    g : Polynomial (Subtype fun x => Membership.mem E' x) := f.toSubring E'.toSubr …
    ⊢ IsSeparable F x
  -/
  have h : g.map (algebraMap E' E) = f := f.map_toSubring E'.toSubring (subset_adjoin F _)
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : Algebra.IsSeparable F E
    x : K
    hsep : IsSeparable E x
    f : Polynomial E := minpoly E x
    hf : Eq f (minpoly E x)
    E' : IntermediateField F E := IntermediateField.adjoin F ↑f.coeffs
    this : FiniteDimensional F (Subtype fun x => Membership.mem E' x)
    g : Polynomial (Subtype fun x => Membership.mem E' x) := f.toSubring E'.toSubr …
    h : Eq (Polynomial.map (algebraMap (Subtype fun x => Membership.mem E' x) E) g …
    ⊢ IsSeparable F x
  -/
  clear_value g
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : Algebra.IsSeparable F E
    x : K
    hsep : IsSeparable E x
    f : Polynomial E := minpoly E x
    hf : Eq f (minpoly E x)
    E' : IntermediateField F E := IntermediateField.adjoin F ↑f.coeffs
    this : FiniteDimensional F (Subtype fun x => Membership.mem E' x)
    g : Polynomial (Subtype fun x => Membership.mem E' x)
    h : Eq (Polynomial.map (algebraMap (Subtype fun x => Membership.mem E' x) E) g …
    ⊢ IsSeparable F x
  -/
  have hx : x ∈ restrictScalars F E'⟮x⟯ := mem_adjoin_simple_self _ x
  have hzero : aeval x g = 0 := by
    simpa only [← hf, ← h, aeval_map_algebraMap] using minpoly.aeval E x
  have halg : IsIntegral E' x :=
    isIntegral_trans (R := F) (A := E) _ (IsSeparable.isIntegral hsep) |>.tower_top
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : Algebra.IsSeparable F E
    x : K
    hsep : IsSeparable E x
    f : Polynomial E := minpoly E x
    hf : Eq f (minpoly E x)
    E' : IntermediateField F E := IntermediateField.adjoin F ↑f.coeffs
    this : FiniteDimensional F (Subtype fun x => Membership.mem E' x)
    g : Polynomial (Subtype fun x => Membership.mem E' x)
    h : Eq (Polynomial.map (algebraMap (Subtype fun x => Membership.mem E' x) E) g …
    hx : Membership.mem (IntermediateField.restrictScalars F (IntermediateField.ad …
    hzero : Eq ((Polynomial.aeval x) g) 0
    halg : IsIntegral (Subtype fun x => Membership.mem E' x) x
    ⊢ IsSeparable F x
  -/
  simp only [IsSeparable, ← hf, ← h, separable_map] at hsep
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : Algebra.IsSeparable F E
    x : K
    f : Polynomial E := minpoly E x
    hf : Eq f (minpoly E x)
    E' : IntermediateField F E := IntermediateField.adjoin F ↑f.coeffs
    this : FiniteDimensional F (Subtype fun x => Membership.mem E' x)
    g : Polynomial (Subtype fun x => Membership.mem E' x)
    h : Eq (Polynomial.map (algebraMap (Subtype fun x => Membership.mem E' x) E) g …
    hx : Membership.mem (IntermediateField.restrictScalars F (IntermediateField.ad …
    hzero : Eq ((Polynomial.aeval x) g) 0
    halg : IsIntegral (Subtype fun x => Membership.mem E' x) x
    hsep : g.Separable
    ⊢ IsSeparable F x
  -/
  replace hsep := hsep.of_dvd <| minpoly.dvd E' x hzero
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : Algebra.IsSeparable F E
    x : K
    f : Polynomial E := minpoly E x
    hf : Eq f (minpoly E x)
    E' : IntermediateField F E := IntermediateField.adjoin F ↑f.coeffs
    this : FiniteDimensional F (Subtype fun x => Membership.mem E' x)
    g : Polynomial (Subtype fun x => Membership.mem E' x)
    h : Eq (Polynomial.map (algebraMap (Subtype fun x => Membership.mem E' x) E) g …
    hx : Membership.mem (IntermediateField.restrictScalars F (IntermediateField.ad …
    hzero : Eq ((Polynomial.aeval x) g) 0
    halg : IsIntegral (Subtype fun x => Membership.mem E' x) x
    hsep : (minpoly (Subtype fun x => Membership.mem E' x) x).Separable
    ⊢ IsSeparable F x
  -/
  haveI : Algebra.IsSeparable F E' := Algebra.isSeparable_tower_bot_of_isSeparable F E' E
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : Algebra.IsSeparable F E
    x : K
    f : Polynomial E := minpoly E x
    hf : Eq f (minpoly E x)
    E' : IntermediateField F E := IntermediateField.adjoin F ↑f.coeffs
    this✝ : FiniteDimensional F (Subtype fun x => Membership.mem E' x)
    g : Polynomial (Subtype fun x => Membership.mem E' x)
    h : Eq (Polynomial.map (algebraMap (Subtype fun x => Membership.mem E' x) E) g …
    hx : Membership.mem (IntermediateField.restrictScalars F (IntermediateField.ad …
    hzero : Eq ((Polynomial.aeval x) g) 0
    halg : IsIntegral (Subtype fun x => Membership.mem E' x) x
    hsep : (minpoly (Subtype fun x => Membership.mem E' x) x).Separable
    this : Algebra.IsSeparable F (Subtype fun x => Membership.mem E' x)
    ⊢ IsSeparable F x
  -/
  haveI := (isSeparable_adjoin_simple_iff_isSeparable _ _).2 hsep
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : Algebra.IsSeparable F E
    x : K
    f : Polynomial E := minpoly E x
    hf : Eq f (minpoly E x)
    E' : IntermediateField F E := IntermediateField.adjoin F ↑f.coeffs
    this✝¹ : FiniteDimensional F (Subtype fun x => Membership.mem E' x)
    g : Polynomial (Subtype fun x => Membership.mem E' x)
    h : Eq (Polynomial.map (algebraMap (Subtype fun x => Membership.mem E' x) E) g …
    hx : Membership.mem (IntermediateField.restrictScalars F (IntermediateField.ad …
    hzero : Eq ((Polynomial.aeval x) g) 0
    halg : IsIntegral (Subtype fun x => Membership.mem E' x) x
    hsep : (minpoly (Subtype fun x => Membership.mem E' x) x).Separable
    this✝ : Algebra.IsSeparable F (Subtype fun x => Membership.mem E' x)
    this : Algebra.IsSeparable (Subtype fun x => Membership.mem E' x) (Subtype fun …
    ⊢ IsSeparable F x
  -/
  haveI := adjoin.finiteDimensional halg
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : Algebra.IsSeparable F E
    x : K
    f : Polynomial E := minpoly E x
    hf : Eq f (minpoly E x)
    E' : IntermediateField F E := IntermediateField.adjoin F ↑f.coeffs
    this✝² : FiniteDimensional F (Subtype fun x => Membership.mem E' x)
    g : Polynomial (Subtype fun x => Membership.mem E' x)
    h : Eq (Polynomial.map (algebraMap (Subtype fun x => Membership.mem E' x) E) g …
    hx : Membership.mem (IntermediateField.restrictScalars F (IntermediateField.ad …
    hzero : Eq ((Polynomial.aeval x) g) 0
    halg : IsIntegral (Subtype fun x => Membership.mem E' x) x
    hsep : (minpoly (Subtype fun x => Membership.mem E' x) x).Separable
    this✝¹ : Algebra.IsSeparable F (Subtype fun x => Membership.mem E' x)
    this✝ : Algebra.IsSeparable (Subtype fun x => Membership.mem E' x) (Subtype fu …
    this : FiniteDimensional (Subtype fun x => Membership.mem E' x) (Subtype fun x …
    ⊢ IsSeparable F x
  -/
  haveI : FiniteDimensional F E'⟮x⟯ := FiniteDimensional.trans F E' E'⟮x⟯
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : Algebra.IsSeparable F E
    x : K
    f : Polynomial E := minpoly E x
    hf : Eq f (minpoly E x)
    E' : IntermediateField F E := IntermediateField.adjoin F ↑f.coeffs
    this✝³ : FiniteDimensional F (Subtype fun x => Membership.mem E' x)
    g : Polynomial (Subtype fun x => Membership.mem E' x)
    h : Eq (Polynomial.map (algebraMap (Subtype fun x => Membership.mem E' x) E) g …
    hx : Membership.mem (IntermediateField.restrictScalars F (IntermediateField.ad …
    hzero : Eq ((Polynomial.aeval x) g) 0
    halg : IsIntegral (Subtype fun x => Membership.mem E' x) x
    hsep : (minpoly (Subtype fun x => Membership.mem E' x) x).Separable
    this✝² : Algebra.IsSeparable F (Subtype fun x => Membership.mem E' x)
    this✝¹ : Algebra.IsSeparable (Subtype fun x => Membership.mem E' x) (Subtype f …
    this✝ : FiniteDimensional (Subtype fun x => Membership.mem E' x) (Subtype fun  …
    this : FiniteDimensional F (Subtype fun x_1 => Membership.mem (IntermediateFie …
    ⊢ IsSeparable F x
  -/
  have : Algebra.IsAlgebraic E' E'⟮x⟯ := Algebra.IsSeparable.isAlgebraic _ _
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : Algebra.IsSeparable F E
    x : K
    f : Polynomial E := minpoly E x
    hf : Eq f (minpoly E x)
    E' : IntermediateField F E := IntermediateField.adjoin F ↑f.coeffs
    this✝⁴ : FiniteDimensional F (Subtype fun x => Membership.mem E' x)
    g : Polynomial (Subtype fun x => Membership.mem E' x)
    h : Eq (Polynomial.map (algebraMap (Subtype fun x => Membership.mem E' x) E) g …
    hx : Membership.mem (IntermediateField.restrictScalars F (IntermediateField.ad …
    hzero : Eq ((Polynomial.aeval x) g) 0
    halg : IsIntegral (Subtype fun x => Membership.mem E' x) x
    hsep : (minpoly (Subtype fun x => Membership.mem E' x) x).Separable
    this✝³ : Algebra.IsSeparable F (Subtype fun x => Membership.mem E' x)
    this✝² : Algebra.IsSeparable (Subtype fun x => Membership.mem E' x) (Subtype f …
    this✝¹ : FiniteDimensional (Subtype fun x => Membership.mem E' x) (Subtype fun …
    this✝ : FiniteDimensional F (Subtype fun x_1 => Membership.mem (IntermediateFi …
    this : Algebra.IsAlgebraic (Subtype fun x => Membership.mem E' x) (Subtype fun …
    ⊢ IsSeparable F x
  -/
  have := finSepDegree_mul_finSepDegree_of_isAlgebraic F E' E'⟮x⟯
  rw [finSepDegree_eq_finrank_of_isSeparable F E',
    finSepDegree_eq_finrank_of_isSeparable E' E'⟮x⟯,
    Module.finrank_mul_finrank F E' E'⟮x⟯,
    eq_comm, finSepDegree_eq_finrank_iff F E'⟮x⟯] at this
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : Algebra.IsSeparable F E
    x : K
    f : Polynomial E := minpoly E x
    hf : Eq f (minpoly E x)
    E' : IntermediateField F E := IntermediateField.adjoin F ↑f.coeffs
    this✝⁵ : FiniteDimensional F (Subtype fun x => Membership.mem E' x)
    g : Polynomial (Subtype fun x => Membership.mem E' x)
    h : Eq (Polynomial.map (algebraMap (Subtype fun x => Membership.mem E' x) E) g …
    hx : Membership.mem (IntermediateField.restrictScalars F (IntermediateField.ad …
    hzero : Eq ((Polynomial.aeval x) g) 0
    halg : IsIntegral (Subtype fun x => Membership.mem E' x) x
    hsep : (minpoly (Subtype fun x => Membership.mem E' x) x).Separable
    this✝⁴ : Algebra.IsSeparable F (Subtype fun x => Membership.mem E' x)
    this✝³ : Algebra.IsSeparable (Subtype fun x => Membership.mem E' x) (Subtype f …
    this✝² : FiniteDimensional (Subtype fun x => Membership.mem E' x) (Subtype fun …
    this✝¹ : FiniteDimensional F (Subtype fun x_1 => Membership.mem (IntermediateF …
    this✝ : Algebra.IsAlgebraic (Subtype fun x => Membership.mem E' x) (Subtype fu …
    this : Algebra.IsSeparable F (Subtype fun x_1 => Membership.mem (IntermediateF …
    ⊢ IsSeparable F x
  -/
  change Algebra.IsSeparable F (restrictScalars F E'⟮x⟯) at this
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : Algebra.IsSeparable F E
    x : K
    f : Polynomial E := minpoly E x
    hf : Eq f (minpoly E x)
    E' : IntermediateField F E := IntermediateField.adjoin F ↑f.coeffs
    this✝⁵ : FiniteDimensional F (Subtype fun x => Membership.mem E' x)
    g : Polynomial (Subtype fun x => Membership.mem E' x)
    h : Eq (Polynomial.map (algebraMap (Subtype fun x => Membership.mem E' x) E) g …
    hx : Membership.mem (IntermediateField.restrictScalars F (IntermediateField.ad …
    hzero : Eq ((Polynomial.aeval x) g) 0
    halg : IsIntegral (Subtype fun x => Membership.mem E' x) x
    hsep : (minpoly (Subtype fun x => Membership.mem E' x) x).Separable
    this✝⁴ : Algebra.IsSeparable F (Subtype fun x => Membership.mem E' x)
    this✝³ : Algebra.IsSeparable (Subtype fun x => Membership.mem E' x) (Subtype f …
    this✝² : FiniteDimensional (Subtype fun x => Membership.mem E' x) (Subtype fun …
    this✝¹ : FiniteDimensional F (Subtype fun x_1 => Membership.mem (IntermediateF …
    this✝ : Algebra.IsAlgebraic (Subtype fun x => Membership.mem E' x) (Subtype fu …
    this : Algebra.IsSeparable F (Subtype fun x_1 => Membership.mem (IntermediateF …
    ⊢ IsSeparable F x
  -/
  exact isSeparable_of_mem_isSeparable F K hx
  /-
    🎉 no goals
  -/


/-- If `E / F` and `K / E` are both separable extensions, then `K / F` is also separable. -/
@[stacks 09HB]
theorem Algebra.IsSeparable.trans [Algebra E K] [IsScalarTower F E K]
    [Algebra.IsSeparable F E] [Algebra.IsSeparable E K] : Algebra.IsSeparable F K :=
  ⟨fun x ↦ IsSeparable.of_algebra_isSeparable_of_isSeparable F
    (Algebra.IsSeparable.isSeparable E x)⟩


/-- If `x` and `y` are both separable elements, then `F⟮x, y⟯ / F` is a separable extension.
As a consequence, any rational function of `x` and `y` is also a separable element. -/
theorem IntermediateField.isSeparable_adjoin_pair_of_isSeparable {x y : E}
    (hx : IsSeparable F x) (hy : IsSeparable F y) :
    Algebra.IsSeparable F F⟮x, y⟯ := by
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    x y : E
    hx : IsSeparable F x
    hy : IsSeparable F y
    ⊢ Algebra.IsSeparable F (Subtype fun x_1 => Membership.mem (IntermediateField. …
  -/
  rw [← adjoin_simple_adjoin_simple]
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    x y : E
    hx : IsSeparable F x
    hy : IsSeparable F y
    ⊢ Algebra.IsSeparable F (Subtype fun x_1 => Membership.mem (IntermediateField. …
  -/
  replace hy := IsSeparable.tower_top F⟮x⟯ hy
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    x y : E
    hx : IsSeparable F x
    hy : IsSeparable (Subtype fun x_1 => Membership.mem (IntermediateField.adjoin  …
    ⊢ Algebra.IsSeparable F (Subtype fun x_1 => Membership.mem (IntermediateField. …
  -/
  rw [← isSeparable_adjoin_simple_iff_isSeparable] at hx hy
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    x y : E
    hx : Algebra.IsSeparable F (Subtype fun x_1 => Membership.mem (IntermediateFie …
    hy : Algebra.IsSeparable (Subtype fun x_1 => Membership.mem (IntermediateField …
    ⊢ Algebra.IsSeparable F (Subtype fun x_1 => Membership.mem (IntermediateField. …
  -/
  exact Algebra.IsSeparable.trans F F⟮x⟯ F⟮x⟯⟮y⟯
  /-
    🎉 no goals
  -/


/-- Any element `x` of `F` is a separable element of `E / F` when embedded into `E`. -/
theorem isSeparable_algebraMap (x : F) : IsSeparable F ((algebraMap F E) x) := by
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    x : F
    ⊢ IsSeparable F ((algebraMap F E) x)
  -/
  rw [IsSeparable, minpoly.algebraMap_eq (algebraMap F E).injective]
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    x : F
    ⊢ (minpoly F x).Separable
  -/
  exact Algebra.IsSeparable.isSeparable F x
  /-
    🎉 no goals
  -/


/-- If `x` and `y` are both separable elements, then `x * y` is also a separable element. -/
theorem isSeparable_mul {x y : E} (hx : IsSeparable F x) (hy : IsSeparable F y) :
    IsSeparable F (x * y) :=
  haveI := isSeparable_adjoin_pair_of_isSeparable F E hx hy
  isSeparable_of_mem_isSeparable F E <| F⟮x, y⟯.mul_mem (subset_adjoin F _ (.inl rfl))
    (subset_adjoin F _ (.inr rfl))


/-- If `x` and `y` are both separable elements, then `x + y` is also a separable element. -/
theorem isSeparable_add {x y : E} (hx : IsSeparable F x) (hy : IsSeparable F y) :
    IsSeparable F (x + y) :=
  haveI := isSeparable_adjoin_pair_of_isSeparable F E hx hy
  isSeparable_of_mem_isSeparable F E <| F⟮x, y⟯.add_mem (subset_adjoin F _ (.inl rfl))
    (subset_adjoin F _ (.inr rfl))


/-- If `x` is a separable elements, then `-x` is also a separable element. -/
theorem isSeparable_neg {x : E} (hx : IsSeparable F x) :
    IsSeparable F (-x) :=
  haveI := (isSeparable_adjoin_simple_iff_isSeparable F E).2 hx
  isSeparable_of_mem_isSeparable F E <| F⟮x⟯.neg_mem <| mem_adjoin_simple_self F x


/-- If `x` and `y` are both separable elements, then `x - y` is also a separable element. -/
theorem isSeparable_sub {x y : E} (hx : IsSeparable F x) (hy : IsSeparable F y) :
    IsSeparable F (x - y) :=
  haveI := isSeparable_adjoin_pair_of_isSeparable F E hx hy
  isSeparable_of_mem_isSeparable F E <| F⟮x, y⟯.sub_mem (subset_adjoin F _ (.inl rfl))
    (subset_adjoin F _ (.inr rfl))


/-- If `x` is a separable element, then `x⁻¹` is also a separable element. -/
theorem isSeparable_inv {x : E} (hx : IsSeparable F x) : IsSeparable F x⁻¹ :=
  haveI := (isSeparable_adjoin_simple_iff_isSeparable F E).2 hx
  isSeparable_of_mem_isSeparable F E <| F⟮x⟯.inv_mem <| mem_adjoin_simple_self F x


/-- A field is a perfect field (which means that any irreducible polynomial is separable)
if and only if every separable degree one polynomial splits. -/
theorem perfectField_iff_splits_of_natSepDegree_eq_one (F : Type*) [Field F] :
    PerfectField F ↔ ∀ f : F[X], f.natSepDegree = 1 → f.Splits (RingHom.id F) := by
  /-
    F : Type u_1
    inst✝ : Field F
    ⊢ Iff (PerfectField F) (∀ (f : Polynomial F), Eq f.natSepDegree 1 → Polynomial …
  -/
  refine ⟨fun ⟨h⟩ f hf ↦ or_iff_not_imp_left.2 fun hn g hg hd ↦ ?_, fun h ↦ ?_⟩
    /-
      case refine_1
      F : Type u_1
      inst✝ : Field F
      x✝ : PerfectField F
      f : Polynomial F
      hf : Eq f.natSepDegree 1
      h : ∀ {f : Polynomial F}, Irreducible f → f.Separable
      hn : Not (Eq (Polynomial.map (RingHom.id F) f) 0)
      g : Polynomial F
      hg : Irreducible g
      hd : Dvd.dvd g (Polynomial.map (RingHom.id F) f)
      ⊢ Eq g.degree 1
    -/
  · rw [map_id] at hn hd
    /-
      case refine_1
      F : Type u_1
      inst✝ : Field F
      x✝ : PerfectField F
      f : Polynomial F
      hf : Eq f.natSepDegree 1
      h : ∀ {f : Polynomial F}, Irreducible f → f.Separable
      hn : Not (Eq f 0)
      g : Polynomial F
      hg : Irreducible g
      hd : Dvd.dvd g f
      ⊢ Eq g.degree 1
    -/
    have := natSepDegree_le_of_dvd g f hd hn
    /-
      case refine_1
      F : Type u_1
      inst✝ : Field F
      x✝ : PerfectField F
      f : Polynomial F
      hf : Eq f.natSepDegree 1
      h : ∀ {f : Polynomial F}, Irreducible f → f.Separable
      hn : Not (Eq f 0)
      g : Polynomial F
      hg : Irreducible g
      hd : Dvd.dvd g f
      this : LE.le g.natSepDegree f.natSepDegree
      ⊢ Eq g.degree 1
    -/
    rw [hf, (h hg).natSepDegree_eq_natDegree] at this
    exact (degree_eq_iff_natDegree_eq_of_pos one_pos).2 <| this.antisymm <|
      natDegree_pos_iff_degree_pos.2 (degree_pos_of_irreducible hg)
  /-
    case refine_2
    F : Type u_1
    inst✝ : Field F
    h : ∀ (f : Polynomial F), Eq f.natSepDegree 1 → Polynomial.Splits (RingHom.id  …
    ⊢ PerfectField F
  -/
  obtain ⟨p, _⟩ := ExpChar.exists F
  haveI := PerfectRing.ofSurjective F p fun x ↦ by
    obtain ⟨y, hy⟩ := exists_root_of_splits _
      (h _ (pow_one p ▸ natSepDegree_X_pow_char_pow_sub_C p 1 x))
      ((degree_X_pow_sub_C (expChar_pos F p) x).symm ▸ Nat.cast_pos.2 (expChar_pos F p)).ne'
    exact ⟨y, by rwa [← eval, eval_sub, eval_pow, eval_X, eval_C, sub_eq_zero] at hy⟩
  /-
    case refine_2.intro
    F : Type u_1
    inst✝ : Field F
    h : ∀ (f : Polynomial F), Eq f.natSepDegree 1 → Polynomial.Splits (RingHom.id  …
    p : Nat
    h✝ : ExpChar F p
    this : PerfectRing F p
    ⊢ PerfectField F
  -/
  exact PerfectRing.toPerfectField F p
  /-
    🎉 no goals
  -/


variable {E K} in
theorem PerfectField.splits_of_natSepDegree_eq_one [PerfectField K] {f : E[X]}
    (i : E →+* K) (hf : f.natSepDegree = 1) : f.Splits i :=
  (splits_id_iff_splits _).mp <| (perfectField_iff_splits_of_natSepDegree_eq_one K).mp ‹_› _
    (natSepDegree_map K f i ▸ hf)

