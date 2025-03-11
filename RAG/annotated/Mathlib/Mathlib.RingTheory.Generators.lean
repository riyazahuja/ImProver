/-- A family of generators of a `R`-algebra `S` consists of
1. `vars`: The type of variables.
2. `val : vars → S`: The assignment of each variable to a value in `S`.
3. `σ`: A section of `R[X] → S`. -/
structure Algebra.Generators where
  /-- The type of variables. -/
  vars : Type w
  /-- The assignment of each variable to a value in `S`. -/
  val : vars → S
  /-- A section of `R[X] → S`. -/
  σ' : S → MvPolynomial vars R
  aeval_val_σ' : ∀ s, aeval val (σ' s) = s
  /-- An `R[X]`-algebra instance on `S`. The default is the one induced by the map `R[X] → S`,
  but this causes a diamond if there is an existing instance. -/
  algebra : Algebra (MvPolynomial vars R) S := (aeval val).toAlgebra
  algebraMap_eq :
    algebraMap (MvPolynomial vars R) S = aeval (R := R) val := by rfl


/-- The polynomial ring wrt a family of generators. -/
protected
abbrev Ring : Type (max w u) := MvPolynomial P.vars R


/-- The designated section of wrt a family of generators. -/
def σ : S → P.Ring := P.σ'


/-- See Note [custom simps projection] -/
def Simps.σ : S → P.Ring := P.σ


@[simp]
lemma aeval_val_σ (s) : aeval P.val (P.σ s) = s := P.aeval_val_σ' s


noncomputable instance {R₀} [CommRing R₀] [Algebra R₀ R] [Algebra R₀ S] [IsScalarTower R₀ R S] :
    IsScalarTower R₀ P.Ring S := IsScalarTower.of_algebraMap_eq' <|
  P.algebraMap_eq ▸ ((aeval (R := R) P.val).comp_algebraMap_of_tower R₀).symm


@[simp]
lemma algebraMap_apply (x) : algebraMap P.Ring S x = aeval (R := R) P.val x := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Algebra.Generators R S
    x : P.Ring
    ⊢ Eq ((algebraMap P.Ring S) x) ((MvPolynomial.aeval P.val) x)
  -/
  simp [algebraMap_eq]
  /-
    🎉 no goals
  -/


@[simp]
lemma σ_smul (x y) : P.σ x • y = x * y := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Algebra.Generators R S
    x y : S
    ⊢ Eq (HSMul.hSMul (P.σ x) y) (HMul.hMul x y)
  -/
  rw [Algebra.smul_def, algebraMap_apply, aeval_val_σ]
  /-
    🎉 no goals
  -/


lemma σ_injective : P.σ.Injective := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Algebra.Generators R S
    ⊢ Function.Injective P.σ
  -/
  intro x y e
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Algebra.Generators R S
    x y : S
    e : Eq (P.σ x) (P.σ y)
    ⊢ Eq x y
  -/
  rw [← P.aeval_val_σ x, ← P.aeval_val_σ y, e]
  /-
    🎉 no goals
  -/


lemma algebraMap_surjective : Function.Surjective (algebraMap P.Ring S) :=
  (⟨_, P.algebraMap_apply _ ▸ P.aeval_val_σ ·⟩)


/-- Construct `Generators` from an assignment `I → S` such that `R[X] → S` is surjective. -/
@[simps val, simps (config := .lemmasOnly) vars]
noncomputable
def ofSurjective {vars} (val : vars → S) (h : Function.Surjective (aeval (R := R) val)) :
    Generators R S where
  vars := vars
  val := val
  σ' x := (h x).choose
  aeval_val_σ' x := (h x).choose_spec


/-- If `algebraMap R S` is surjective, the empty type generates `S`. -/
noncomputable def ofSurjectiveAlgebraMap (h : Function.Surjective (algebraMap R S)) :
    Generators.{w} R S :=
  ofSurjective PEmpty.elim <| fun s ↦ by
    /-
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      P : Algebra.Generators R S
      h : Function.Surjective ⇑(algebraMap R S)
      s : S
      ⊢ Exists fun a => Eq ((MvPolynomial.aeval PEmpty.elim) a) s
    -/
    use C (h s).choose
    /-
      case h
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      P : Algebra.Generators R S
      h : Function.Surjective ⇑(algebraMap R S)
      s : S
      ⊢ Eq ((MvPolynomial.aeval PEmpty.elim) (MvPolynomial.C ⋯.choose)) s
    -/
    simp [(h s).choose_spec]
    /-
      🎉 no goals
    -/


/-- The canonical generators for `R` as an `R`-algebra. -/
noncomputable def id : Generators.{w} R R := ofSurjectiveAlgebraMap <| by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Algebra.Generators R S
    ⊢ Function.Surjective ⇑(algebraMap R R)
  -/
  rw [id.map_eq_id]
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Algebra.Generators R S
    ⊢ Function.Surjective ⇑(RingHom.id R)
  -/
  exact RingHomSurjective.is_surjective
  /-
    🎉 no goals
  -/


/-- Construct `Generators` from an assignment `I → S` such that `R[X] → S` is surjective. -/
noncomputable
def ofAlgHom {I} (f : MvPolynomial I R →ₐ[R] S) (h : Function.Surjective f) :
    Generators R S :=
                           /-
                             R : Type u
                             S : Type v
                             inst✝² : CommRing R
                             inst✝¹ : CommRing S
                             inst✝ : Algebra R S
                             P : Algebra.Generators R S
                             I : Type ?u.20061
                             f : AlgHom R (MvPolynomial I R) S
                             h : Function.Surjective ⇑f
                             ⊢ Function.Surjective ⇑(MvPolynomial.aeval (Function.comp (⇑f) MvPolynomial.X))
                           -/
  ofSurjective (f ∘ X) (by rwa [show aeval (f ∘ X) = f by ext; simp])
                           /-
                             🎉 no goals
                           -/


/-- Construct `Generators` from a family of generators of `S`. -/
noncomputable
def ofSet {s : Set S} (hs : Algebra.adjoin R s = ⊤) : Generators R S := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Algebra.Generators R S
    s : Set S
    hs : Eq (Algebra.adjoin R s) Top.top
    ⊢ Algebra.Generators R S
  -/
  refine ofSurjective (Subtype.val : s → S) ?_
  rwa [← AlgHom.range_eq_top, ← Algebra.adjoin_range_eq_range_aeval,
    Subtype.range_coe_subtype, Set.setOf_mem_eq]


variable (R S) in
/-- The `Generators` containing the whole algebra, which induces the canonical map  `R[S] → S`. -/
@[simps]
noncomputable
def self : Generators R S where
  vars := S
  val := _root_.id
  σ' := X
  aeval_val_σ' := aeval_X _


/-- The extension `R[X₁,...,Xₙ] → S` given a family of generators. -/
@[simps]
noncomputable
def toExtension : Extension R S where
  Ring := P.Ring
  σ := P.σ
                     /-
                       R : Type u
                       S : Type v
                       inst✝² : CommRing R
                       inst✝¹ : CommRing S
                       inst✝ : Algebra R S
                       P : Algebra.Generators R S
                       ⊢ ∀ (x : S), Eq ((algebraMap P.Ring S) (P.σ x)) x
                     -/
  algebraMap_σ := by simp
                     /-
                       🎉 no goals
                     -/


/-- If `S` is the localization of `R` away from `r`, we obtain a canonical generator mapping
to the inverse of `r`. -/
@[simps val, simps (config := .lemmasOnly) vars σ]
noncomputable
def localizationAway : Generators R S where
  vars := Unit
  val _ := IsLocalization.Away.invSelf r
  σ' s :=
    letI a : R := (IsLocalization.Away.sec r s).1
    letI n : ℕ := (IsLocalization.Away.sec r s).2
    C a * X () ^ n
  aeval_val_σ' s := by
    /-
      R : Type u
      S : Type v
      inst✝³ : CommRing R
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      P : Algebra.Generators R S
      r : R
      inst✝ : IsLocalization.Away r S
      s : S
      ⊢ Eq ((MvPolynomial.aeval fun x => IsLocalization.Away.invSelf r) ((fun s => H …
    -/
    rw [map_mul, algHom_C, map_pow, aeval_X]
    /-
      R : Type u
      S : Type v
      inst✝³ : CommRing R
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      P : Algebra.Generators R S
      r : R
      inst✝ : IsLocalization.Away r S
      s : S
      ⊢ Eq (HMul.hMul ((algebraMap R S) (IsLocalization.Away.sec r s).1) (HPow.hPow  …
    -/
    simp only [← IsLocalization.Away.sec_spec, map_pow, IsLocalization.Away.invSelf]
    /-
      R : Type u
      S : Type v
      inst✝³ : CommRing R
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      P : Algebra.Generators R S
      r : R
      inst✝ : IsLocalization.Away r S
      s : S
      ⊢ Eq (HMul.hMul (HMul.hMul s (HPow.hPow ((algebraMap R S) r) (IsLocalization.A …
    -/
    rw [← IsLocalization.mk'_pow, one_pow, ← IsLocalization.mk'_one (M := Submonoid.powers r) S r]
    /-
      R : Type u
      S : Type v
      inst✝³ : CommRing R
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      P : Algebra.Generators R S
      r : R
      inst✝ : IsLocalization.Away r S
      s : S
      ⊢ Eq (HMul.hMul (HMul.hMul s (HPow.hPow (IsLocalization.mk' S r 1) (IsLocaliza …
    -/
    rw [← IsLocalization.mk'_pow, one_pow, mul_assoc, ← IsLocalization.mk'_mul]
    /-
      R : Type u
      S : Type v
      inst✝³ : CommRing R
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      P : Algebra.Generators R S
      r : R
      inst✝ : IsLocalization.Away r S
      s : S
      ⊢ Eq (HMul.hMul s (IsLocalization.mk' S (HMul.hMul (HPow.hPow r (IsLocalizatio …
    -/
    rw [mul_one, one_mul, IsLocalization.mk'_pow]
    /-
      R : Type u
      S : Type v
      inst✝³ : CommRing R
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      P : Algebra.Generators R S
      r : R
      inst✝ : IsLocalization.Away r S
      s : S
      ⊢ Eq (HMul.hMul s (HPow.hPow (IsLocalization.mk' S r ⟨r, ⋯⟩) (IsLocalization.A …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- Given two families of generators `S[X] → T` and `R[Y] → S`,
we may construct the family of generators `R[X, Y] → T`. -/
@[simps val, simps (config := .lemmasOnly) vars σ]
noncomputable
def comp (Q : Generators S T) (P : Generators R S) : Generators R T where
  vars := Q.vars ⊕ P.vars
  val := Sum.elim Q.val (algebraMap S T ∘ P.val)
  σ' x := (Q.σ x).sum (fun n r ↦ rename Sum.inr (P.σ r) * monomial (n.mapDomain Sum.inl) 1)
  aeval_val_σ' s := by
    have (x : P.Ring) : aeval (algebraMap S T ∘ P.val) x = algebraMap S T (aeval P.val x) := by
      rw [map_aeval, aeval_def, coe_eval₂Hom, ← IsScalarTower.algebraMap_eq, Function.comp_def]
    /-
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      P✝ : Algebra.Generators R S
      T : Type ?u.45146
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      s : T
      this : ∀ (x : P.Ring), Eq ((MvPolynomial.aeval (Function.comp (⇑(algebraMap S  …
      ⊢ Eq ((MvPolynomial.aeval (Sum.elim Q.val (Function.comp (⇑(algebraMap S T)) P …
    -/
    conv_rhs => rw [← Q.aeval_val_σ s, ← (Q.σ s).sum_single]
    simp only [map_finsupp_sum, map_mul, aeval_rename, Sum.elim_comp_inr, this, aeval_val_σ,
      aeval_monomial, map_one, Finsupp.prod_mapDomain_index_inj Sum.inl_injective, Sum.elim_inl,
      one_mul, single_eq_monomial]


variable (S) in
/-- If `R → S → T` is a tower of algebras, a family of generators `R[X] → T`
gives a family of generators `S[X] → T`. -/
@[simps val, simps (config := .lemmasOnly) vars]
noncomputable
def extendScalars (P : Generators R T) : Generators S T where
  vars := P.vars
  val := P.val
  σ' x := map (algebraMap R S) (P.σ x)
                       /-
                         R : Type u
                         S : Type v
                         inst✝⁶ : CommRing R
                         inst✝⁵ : CommRing S
                         inst✝⁴ : Algebra R S
                         P✝ : Algebra.Generators R S
                         T : Type ?u.92830
                         inst✝³ : CommRing T
                         inst✝² : Algebra R T
                         inst✝¹ : Algebra S T
                         inst✝ : IsScalarTower R S T
                         P : Algebra.Generators R T
                         s : T
                         ⊢ Eq ((MvPolynomial.aeval P.val) ((fun x => (MvPolynomial.map (algebraMap R S) …
                       -/
  aeval_val_σ' s := by simp [@aeval_def S, ← IsScalarTower.algebraMap_eq, ← @aeval_def R]
                       /-
                         🎉 no goals
                       -/


/-- If `P` is a family of generators of `S` over `R` and `T` is an `R`-algebra, we
obtain a natural family of generators of `T ⊗[R] S` over `T`. -/
@[simps! val, simps! (config := .lemmasOnly) vars]
noncomputable
def baseChange {T} [CommRing T] [Algebra R T] (P : Generators R S) : Generators T (T ⊗[R] S) := by
  /-
    R : Type u
    S : Type v
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    P✝ : Algebra.Generators R S
    T✝ : Type ?u.94501
    inst✝⁵ : CommRing T✝
    inst✝⁴ : Algebra R T✝
    inst✝³ : Algebra S T✝
    inst✝² : IsScalarTower R S T✝
    T : Type ?u.95083
    inst✝¹ : CommRing T
    inst✝ : Algebra R T
    P : Algebra.Generators R S
    ⊢ Algebra.Generators T (TensorProduct R T S)
  -/
  apply Generators.ofSurjective (fun x ↦ 1 ⊗ₜ[R] P.val x)
  /-
    R : Type u
    S : Type v
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    P✝ : Algebra.Generators R S
    T✝ : Type ?u.94501
    inst✝⁵ : CommRing T✝
    inst✝⁴ : Algebra R T✝
    inst✝³ : Algebra S T✝
    inst✝² : IsScalarTower R S T✝
    T : Type ?u.95083
    inst✝¹ : CommRing T
    inst✝ : Algebra R T
    P : Algebra.Generators R S
    ⊢ Function.Surjective ⇑(MvPolynomial.aeval fun x => TensorProduct.tmul R 1 (P. …
  -/
  intro x
  induction x using TensorProduct.induction_on with
  | zero => exact ⟨0, map_zero _⟩
  | tmul a b =>
    let X := P.σ b
    use a • MvPolynomial.map (algebraMap R T) X
    simp only [LinearMapClass.map_smul, X, aeval_map_algebraMap]
    have : ∀ y : P.Ring,
      aeval (fun x ↦ (1 ⊗ₜ[R] P.val x : T ⊗[R] S)) y = 1 ⊗ₜ aeval (fun x ↦ P.val x) y := by
      intro y
      induction y using MvPolynomial.induction_on with
      | h_C a =>
        rw [aeval_C, aeval_C, TensorProduct.algebraMap_apply, algebraMap_eq_smul_one, smul_tmul,
          algebraMap_eq_smul_one]
      | h_add p q hp hq => simp [map_add, tmul_add, hp, hq]
      | h_X p i hp => simp [hp]
    rw [this, P.aeval_val_σ, smul_tmul', smul_eq_mul, mul_one]
  | add x y ex ey =>
    obtain ⟨a, ha⟩ := ex
    obtain ⟨b, hb⟩ := ey
    use (a + b)
    rw [map_add, ha, hb]


/-- Given a commuting square
R --→ P = R[X] ---→ S
|                   |
↓                   ↓
R' -→ P' = R'[X'] → S
A hom between `P` and `P'` is an assignment `I → P'` such that the arrows commute.
Also see `Algebra.Generators.Hom.equivAlgHom`.
-/
@[ext]
structure Hom where
  /-- The assignment of each variable in `I` to a value in `P' = R'[X']`. -/
  val : P.vars → P'.Ring
  aeval_val : ∀ i, aeval P'.val (val i) = algebraMap S S' (P.val i)


/-- A hom between two families of generators gives
an algebra homomorphism between the polynomial rings. -/
noncomputable
def Hom.toAlgHom (f : Hom P P') : P.Ring →ₐ[R] P'.Ring := MvPolynomial.aeval f.val


variable [Algebra R S'] [IsScalarTower R R' S'] [IsScalarTower R S S'] in
@[simp]
lemma Hom.algebraMap_toAlgHom (f : Hom P P') (x) : MvPolynomial.aeval P'.val (f.toAlgHom x) =
    algebraMap S S' (MvPolynomial.aeval P.val x) := by
  suffices ((MvPolynomial.aeval P'.val).restrictScalars R).comp f.toAlgHom =
      (IsScalarTower.toAlgHom R S S').comp (MvPolynomial.aeval P.val) from
    DFunLike.congr_fun this x
  /-
    R : Type u
    S : Type v
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    P : Algebra.Generators R S
    R' : Type u_1
    S' : Type u_2
    inst✝⁷ : CommRing R'
    inst✝⁶ : CommRing S'
    inst✝⁵ : Algebra R' S'
    P' : Algebra.Generators R' S'
    inst✝⁴ : Algebra R R'
    inst✝³ : Algebra S S'
    inst✝² : Algebra R S'
    inst✝¹ : IsScalarTower R R' S'
    inst✝ : IsScalarTower R S S'
    f : P.Hom P'
    x : P.Ring
    ⊢ Eq ((AlgHom.restrictScalars R (MvPolynomial.aeval P'.val)).comp f.toAlgHom)  …
  -/
  apply MvPolynomial.algHom_ext
  /-
    case hf
    R : Type u
    S : Type v
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    P : Algebra.Generators R S
    R' : Type u_1
    S' : Type u_2
    inst✝⁷ : CommRing R'
    inst✝⁶ : CommRing S'
    inst✝⁵ : Algebra R' S'
    P' : Algebra.Generators R' S'
    inst✝⁴ : Algebra R R'
    inst✝³ : Algebra S S'
    inst✝² : Algebra R S'
    inst✝¹ : IsScalarTower R R' S'
    inst✝ : IsScalarTower R S S'
    f : P.Hom P'
    x : P.Ring
    ⊢ ∀ (i : P.vars), Eq (((AlgHom.restrictScalars R (MvPolynomial.aeval P'.val)). …
  -/
  intro i
  /-
    case hf
    R : Type u
    S : Type v
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    P : Algebra.Generators R S
    R' : Type u_1
    S' : Type u_2
    inst✝⁷ : CommRing R'
    inst✝⁶ : CommRing S'
    inst✝⁵ : Algebra R' S'
    P' : Algebra.Generators R' S'
    inst✝⁴ : Algebra R R'
    inst✝³ : Algebra S S'
    inst✝² : Algebra R S'
    inst✝¹ : IsScalarTower R R' S'
    inst✝ : IsScalarTower R S S'
    f : P.Hom P'
    x : P.Ring
    i : P.vars
    ⊢ Eq (((AlgHom.restrictScalars R (MvPolynomial.aeval P'.val)).comp f.toAlgHom) …
  -/
  simp [Hom.toAlgHom]
  /-
    🎉 no goals
  -/


@[simp]
lemma Hom.toAlgHom_X (f : Hom P P') (i) : f.toAlgHom (.X i) = f.val i :=
  MvPolynomial.aeval_X f.val i


lemma Hom.toAlgHom_C (f : Hom P P') (r) : f.toAlgHom (.C r) = .C (algebraMap _ _ r) :=
  MvPolynomial.aeval_C f.val r


lemma Hom.toAlgHom_monomial (f : Generators.Hom P P') (v r) :
    f.toAlgHom (monomial v r) = r • v.prod (f.val · ^ ·) := by
  /-
    R : Type u
    S : Type v
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : Algebra R S
    P : Algebra.Generators R S
    R' : Type u_1
    S' : Type u_2
    inst✝⁴ : CommRing R'
    inst✝³ : CommRing S'
    inst✝² : Algebra R' S'
    P' : Algebra.Generators R' S'
    inst✝¹ : Algebra R R'
    inst✝ : Algebra S S'
    f : P.Hom P'
    v : Finsupp P.vars Nat
    r : R
    ⊢ Eq (f.toAlgHom ((MvPolynomial.monomial v) r)) (HSMul.hSMul r (v.prod fun x1  …
  -/
  rw [toAlgHom, aeval_monomial, Algebra.smul_def]
  /-
    🎉 no goals
  -/


variable [Algebra R S'] [IsScalarTower R R' S'] [IsScalarTower R S S'] in
/-- Giving a hom between two families of generators is equivalent to
giving an algebra homomorphism between the polynomial rings. -/
@[simps]
noncomputable
def Hom.equivAlgHom :
    Hom P P' ≃ { f : P.Ring →ₐ[R] P'.Ring //
      ∀ x, aeval P'.val (f x) = algebraMap S S' (aeval P.val x) } where
  toFun f := ⟨f.toAlgHom, f.algebraMap_toAlgHom⟩
                                              /-
                                                R : Type u
                                                S : Type v
                                                inst✝¹⁷ : CommRing R
                                                inst✝¹⁶ : CommRing S
                                                inst✝¹⁵ : Algebra R S
                                                P : Algebra.Generators R S
                                                R' : Type ?u.178536
                                                S' : Type ?u.178539
                                                inst✝¹⁴ : CommRing R'
                                                inst✝¹³ : CommRing S'
                                                inst✝¹² : Algebra R' S'
                                                P' : Algebra.Generators R' S'
                                                R'' : Type ?u.178873
                                                S'' : Type ?u.178876
                                                inst✝¹¹ : CommRing R''
                                                inst✝¹⁰ : CommRing S''
                                                inst✝⁹ : Algebra R'' S''
                                                P'' : Algebra.Generators R'' S''
                                                inst✝⁸ : Algebra R R'
                                                inst✝⁷ : Algebra R' R''
                                                inst✝⁶ : Algebra R' S''
                                                inst✝⁵ : Algebra S S'
                                                inst✝⁴ : Algebra S' S''
                                                inst✝³ : Algebra S S''
                                                inst✝² : Algebra R S'
                                                inst✝¹ : IsScalarTower R R' S'
                                                inst✝ : IsScalarTower R S S'
                                                f : Subtype fun f => ∀ (x : P.Ring), Eq ((MvPolynomial.aeval P'.val) (f x)) (( …
                                                i : P.vars
                                                ⊢ Eq ((MvPolynomial.aeval P'.val) ((fun i => ↑f (MvPolynomial.X i)) i)) ((alge …
                                              -/
  invFun f := ⟨fun i ↦ f.1 (.X i), fun i ↦ by simp [f.2]⟩
                                              /-
                                                🎉 no goals
                                              -/
                   /-
                     R : Type u
                     S : Type v
                     inst✝¹⁷ : CommRing R
                     inst✝¹⁶ : CommRing S
                     inst✝¹⁵ : Algebra R S
                     P : Algebra.Generators R S
                     R' : Type ?u.178536
                     S' : Type ?u.178539
                     inst✝¹⁴ : CommRing R'
                     inst✝¹³ : CommRing S'
                     inst✝¹² : Algebra R' S'
                     P' : Algebra.Generators R' S'
                     R'' : Type ?u.178873
                     S'' : Type ?u.178876
                     inst✝¹¹ : CommRing R''
                     inst✝¹⁰ : CommRing S''
                     inst✝⁹ : Algebra R'' S''
                     P'' : Algebra.Generators R'' S''
                     inst✝⁸ : Algebra R R'
                     inst✝⁷ : Algebra R' R''
                     inst✝⁶ : Algebra R' S''
                     inst✝⁵ : Algebra S S'
                     inst✝⁴ : Algebra S' S''
                     inst✝³ : Algebra S S''
                     inst✝² : Algebra R S'
                     inst✝¹ : IsScalarTower R R' S'
                     inst✝ : IsScalarTower R S S'
                     f : P.Hom P'
                     ⊢ Eq ((fun f => { val := fun i => ↑f (MvPolynomial.X i), aeval_val := ⋯ }) ((f …
                   -/
  left_inv f := by ext; simp
                        /-
                          🎉 no goals
                        -/
                    /-
                      R : Type u
                      S : Type v
                      inst✝¹⁷ : CommRing R
                      inst✝¹⁶ : CommRing S
                      inst✝¹⁵ : Algebra R S
                      P : Algebra.Generators R S
                      R' : Type ?u.178536
                      S' : Type ?u.178539
                      inst✝¹⁴ : CommRing R'
                      inst✝¹³ : CommRing S'
                      inst✝¹² : Algebra R' S'
                      P' : Algebra.Generators R' S'
                      R'' : Type ?u.178873
                      S'' : Type ?u.178876
                      inst✝¹¹ : CommRing R''
                      inst✝¹⁰ : CommRing S''
                      inst✝⁹ : Algebra R'' S''
                      P'' : Algebra.Generators R'' S''
                      inst✝⁸ : Algebra R R'
                      inst✝⁷ : Algebra R' R''
                      inst✝⁶ : Algebra R' S''
                      inst✝⁵ : Algebra S S'
                      inst✝⁴ : Algebra S' S''
                      inst✝³ : Algebra S S''
                      inst✝² : Algebra R S'
                      inst✝¹ : IsScalarTower R R' S'
                      inst✝ : IsScalarTower R S S'
                      f : Subtype fun f => ∀ (x : P.Ring), Eq ((MvPolynomial.aeval P'.val) (f x)) (( …
                      ⊢ Eq ((fun f => ⟨f.toAlgHom, ⋯⟩) ((fun f => { val := fun i => ↑f (MvPolynomial …
                    -/
  right_inv f := by ext; simp
                         /-
                           🎉 no goals
                         -/


/-- The hom from `P` to `P'` given by the designated section of `P'`. -/
@[simps]
                                                                         /-
                                                                           R : Type u
                                                                           S : Type v
                                                                           inst✝¹⁴ : CommRing R
                                                                           inst✝¹³ : CommRing S
                                                                           inst✝¹² : Algebra R S
                                                                           P : Algebra.Generators R S
                                                                           R' : Type ?u.189334
                                                                           S' : Type ?u.189337
                                                                           inst✝¹¹ : CommRing R'
                                                                           inst✝¹⁰ : CommRing S'
                                                                           inst✝⁹ : Algebra R' S'
                                                                           P' : Algebra.Generators R' S'
                                                                           R'' : Type ?u.189671
                                                                           S'' : Type ?u.189674
                                                                           inst✝⁸ : CommRing R''
                                                                           inst✝⁷ : CommRing S''
                                                                           inst✝⁶ : Algebra R'' S''
                                                                           P'' : Algebra.Generators R'' S''
                                                                           inst✝⁵ : Algebra R R'
                                                                           inst✝⁴ : Algebra R' R''
                                                                           inst✝³ : Algebra R' S''
                                                                           inst✝² : Algebra S S'
                                                                           inst✝¹ : Algebra S' S''
                                                                           inst✝ : Algebra S S''
                                                                           x : P.vars
                                                                           ⊢ Eq ((MvPolynomial.aeval P'.val) (Function.comp P'.σ (Function.comp (⇑(algebr …
                                                                         -/
def defaultHom : Hom P P' := ⟨P'.σ ∘ algebraMap S S' ∘ P.val, fun x ↦ by simp⟩
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


instance : Inhabited (Hom P P') := ⟨defaultHom P P'⟩


/-- The identity hom. -/
@[simps]
                                                       /-
                                                         R : Type u
                                                         S : Type v
                                                         inst✝¹⁴ : CommRing R
                                                         inst✝¹³ : CommRing S
                                                         inst✝¹² : Algebra R S
                                                         P : Algebra.Generators R S
                                                         R' : Type ?u.196416
                                                         S' : Type ?u.196419
                                                         inst✝¹¹ : CommRing R'
                                                         inst✝¹⁰ : CommRing S'
                                                         inst✝⁹ : Algebra R' S'
                                                         P' : Algebra.Generators R' S'
                                                         R'' : Type ?u.196753
                                                         S'' : Type ?u.196756
                                                         inst✝⁸ : CommRing R''
                                                         inst✝⁷ : CommRing S''
                                                         inst✝⁶ : Algebra R'' S''
                                                         P'' : Algebra.Generators R'' S''
                                                         inst✝⁵ : Algebra R R'
                                                         inst✝⁴ : Algebra R' R''
                                                         inst✝³ : Algebra R' S''
                                                         inst✝² : Algebra S S'
                                                         inst✝¹ : Algebra S' S''
                                                         inst✝ : Algebra S S''
                                                         ⊢ ∀ (i : P.vars), Eq ((MvPolynomial.aeval P.val) (MvPolynomial.X i)) ((algebra …
                                                       -/
protected noncomputable def Hom.id : Hom P P := ⟨X, by simp⟩
                                                       /-
                                                         🎉 no goals
                                                       -/


@[simp]
                                                                   /-
                                                                     R : Type u
                                                                     S : Type v
                                                                     inst✝² : CommRing R
                                                                     inst✝¹ : CommRing S
                                                                     inst✝ : Algebra R S
                                                                     P : Algebra.Generators R S
                                                                     ⊢ Eq (Algebra.Generators.Hom.id P).toAlgHom (AlgHom.id R P.Ring)
                                                                   -/
lemma Hom.toAlgHom_id : Hom.toAlgHom (.id P) = AlgHom.id _ _ := by ext1; simp
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


/-- The composition of two homs. -/
@[simps]
noncomputable def Hom.comp [IsScalarTower R' R'' S''] [IsScalarTower R' S' S'']
    [IsScalarTower S S' S''] (f : Hom P' P'') (g : Hom P P') : Hom P P'' where
  val x := aeval f.val (g.val x)
  aeval_val x := by
    /-
      R : Type u
      S : Type v
      inst✝¹⁷ : CommRing R
      inst✝¹⁶ : CommRing S
      inst✝¹⁵ : Algebra R S
      P : Algebra.Generators R S
      R' : Type ?u.206404
      S' : Type ?u.206407
      inst✝¹⁴ : CommRing R'
      inst✝¹³ : CommRing S'
      inst✝¹² : Algebra R' S'
      P' : Algebra.Generators R' S'
      R'' : Type ?u.206741
      S'' : Type ?u.206744
      inst✝¹¹ : CommRing R''
      inst✝¹⁰ : CommRing S''
      inst✝⁹ : Algebra R'' S''
      P'' : Algebra.Generators R'' S''
      inst✝⁸ : Algebra R R'
      inst✝⁷ : Algebra R' R''
      inst✝⁶ : Algebra R' S''
      inst✝⁵ : Algebra S S'
      inst✝⁴ : Algebra S' S''
      inst✝³ : Algebra S S''
      inst✝² : IsScalarTower R' R'' S''
      inst✝¹ : IsScalarTower R' S' S''
      inst✝ : IsScalarTower S S' S''
      f : P'.Hom P''
      g : P.Hom P'
      x : P.vars
      ⊢ Eq ((MvPolynomial.aeval P''.val) ((fun x => (MvPolynomial.aeval f.val) (g.va …
    -/
    simp only
    /-
      R : Type u
      S : Type v
      inst✝¹⁷ : CommRing R
      inst✝¹⁶ : CommRing S
      inst✝¹⁵ : Algebra R S
      P : Algebra.Generators R S
      R' : Type ?u.206404
      S' : Type ?u.206407
      inst✝¹⁴ : CommRing R'
      inst✝¹³ : CommRing S'
      inst✝¹² : Algebra R' S'
      P' : Algebra.Generators R' S'
      R'' : Type ?u.206741
      S'' : Type ?u.206744
      inst✝¹¹ : CommRing R''
      inst✝¹⁰ : CommRing S''
      inst✝⁹ : Algebra R'' S''
      P'' : Algebra.Generators R'' S''
      inst✝⁸ : Algebra R R'
      inst✝⁷ : Algebra R' R''
      inst✝⁶ : Algebra R' S''
      inst✝⁵ : Algebra S S'
      inst✝⁴ : Algebra S' S''
      inst✝³ : Algebra S S''
      inst✝² : IsScalarTower R' R'' S''
      inst✝¹ : IsScalarTower R' S' S''
      inst✝ : IsScalarTower S S' S''
      f : P'.Hom P''
      g : P.Hom P'
      x : P.vars
      ⊢ Eq ((MvPolynomial.aeval P''.val) ((MvPolynomial.aeval f.val) (g.val x))) ((a …
    -/
    rw [IsScalarTower.algebraMap_apply S S' S'', ← g.aeval_val]
    induction g.val x using MvPolynomial.induction_on with
    | h_C r => simp [← IsScalarTower.algebraMap_apply]
    | h_add x y hx hy => simp only [map_add, hx, hy]
    | h_X p i hp => simp only [map_mul, hp, aeval_X, aeval_val]


@[simp]
lemma Hom.comp_id [Algebra R S'] [IsScalarTower R R' S'] [IsScalarTower R S S'] (f : Hom P P') :
                                /-
                                  R : Type u
                                  S : Type v
                                  inst✝¹⁰ : CommRing R
                                  inst✝⁹ : CommRing S
                                  inst✝⁸ : Algebra R S
                                  P : Algebra.Generators R S
                                  R' : Type u_2
                                  S' : Type u_1
                                  inst✝⁷ : CommRing R'
                                  inst✝⁶ : CommRing S'
                                  inst✝⁵ : Algebra R' S'
                                  P' : Algebra.Generators R' S'
                                  inst✝⁴ : Algebra R R'
                                  inst✝³ : Algebra S S'
                                  inst✝² : Algebra R S'
                                  inst✝¹ : IsScalarTower R R' S'
                                  inst✝ : IsScalarTower R S S'
                                  f : P.Hom P'
                                  ⊢ Eq (f.comp (Algebra.Generators.Hom.id P)) f
                                -/
    f.comp (Hom.id P) = f := by ext; simp
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
lemma Hom.id_comp [Algebra S S'] (f : Hom P P') : (Hom.id P').comp f = f := by
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    P : Algebra.Generators R S
    R' : Type u_2
    S' : Type u_1
    inst✝³ : CommRing R'
    inst✝² : CommRing S'
    inst✝¹ : Algebra R' S'
    P' : Algebra.Generators R' S'
    inst✝ : Algebra S S'
    f : P.Hom P'
    ⊢ Eq ((Algebra.Generators.Hom.id P').comp f) f
  -/
  ext; simp [Hom.id, aeval_X_left]
       /-
         🎉 no goals
       -/


@[simp]
lemma Hom.toAlgHom_comp_apply
    [Algebra R R''] [IsScalarTower R R' R''] [IsScalarTower R' R'' S'']
    [IsScalarTower R' S' S''] [IsScalarTower S S' S'']
    (f : Hom P P') (g : Hom P' P'') (x) :
    (g.comp f).toAlgHom x = g.toAlgHom (f.toAlgHom x) := by
  induction x using MvPolynomial.induction_on with
  | h_C r => simp only [← MvPolynomial.algebraMap_eq, AlgHom.map_algebraMap]
  | h_add x y hx hy => simp only [map_add, hx, hy]
  | h_X p i hp => simp only [map_mul, hp, toAlgHom_X, comp_val]; rfl


/-- Given families of generators `X ⊆ T` over `S` and `Y ⊆ S` over `R`,
there is a map of generators `R[Y] → R[X, Y]`. -/
@[simps]
noncomputable
def toComp (Q : Generators S T) (P : Generators R S) : Hom P (Q.comp P) where
  val i := X (.inr i)
                    /-
                      R : Type u
                      S : Type v
                      inst✝¹⁸ : CommRing R
                      inst✝¹⁷ : CommRing S
                      inst✝¹⁶ : Algebra R S
                      P✝ : Algebra.Generators R S
                      R' : Type ?u.255765
                      S' : Type ?u.255768
                      inst✝¹⁵ : CommRing R'
                      inst✝¹⁴ : CommRing S'
                      inst✝¹³ : Algebra R' S'
                      P' : Algebra.Generators R' S'
                      R'' : Type ?u.256102
                      S'' : Type ?u.256105
                      inst✝¹² : CommRing R''
                      inst✝¹¹ : CommRing S''
                      inst✝¹⁰ : Algebra R'' S''
                      P'' : Algebra.Generators R'' S''
                      inst✝⁹ : Algebra R R'
                      inst✝⁸ : Algebra R' R''
                      inst✝⁷ : Algebra R' S''
                      inst✝⁶ : Algebra S S'
                      inst✝⁵ : Algebra S' S''
                      inst✝⁴ : Algebra S S''
                      T : Type ?u.257794
                      inst✝³ : CommRing T
                      inst✝² : Algebra R T
                      inst✝¹ : Algebra S T
                      inst✝ : IsScalarTower R S T
                      Q : Algebra.Generators S T
                      P : Algebra.Generators R S
                      i : P.vars
                      ⊢ Eq ((MvPolynomial.aeval (Q.comp P).val) ((fun i => MvPolynomial.X (Sum.inr i …
                    -/
  aeval_val i := by simp
                    /-
                      🎉 no goals
                    -/


lemma toComp_toAlgHom (Q : Generators S T) (P : Generators R S) :
    (Q.toComp P).toAlgHom = rename Sum.inr := rfl


/-- Given families of generators `X ⊆ T` over `S` and `Y ⊆ S` over `R`,
there is a map of generators `R[X, Y] → S[X]`. -/
@[simps]
noncomputable
def ofComp (Q : Generators S T) (P : Generators R S) : Hom (Q.comp P) Q where
  val i := i.elim X (C ∘ P.val)
                    /-
                      R : Type u
                      S : Type v
                      inst✝¹⁸ : CommRing R
                      inst✝¹⁷ : CommRing S
                      inst✝¹⁶ : Algebra R S
                      P✝ : Algebra.Generators R S
                      R' : Type ?u.263736
                      S' : Type ?u.263739
                      inst✝¹⁵ : CommRing R'
                      inst✝¹⁴ : CommRing S'
                      inst✝¹³ : Algebra R' S'
                      P' : Algebra.Generators R' S'
                      R'' : Type ?u.264073
                      S'' : Type ?u.264076
                      inst✝¹² : CommRing R''
                      inst✝¹¹ : CommRing S''
                      inst✝¹⁰ : Algebra R'' S''
                      P'' : Algebra.Generators R'' S''
                      inst✝⁹ : Algebra R R'
                      inst✝⁸ : Algebra R' R''
                      inst✝⁷ : Algebra R' S''
                      inst✝⁶ : Algebra S S'
                      inst✝⁵ : Algebra S' S''
                      inst✝⁴ : Algebra S S''
                      T : Type ?u.265765
                      inst✝³ : CommRing T
                      inst✝² : Algebra R T
                      inst✝¹ : Algebra S T
                      inst✝ : IsScalarTower R S T
                      Q : Algebra.Generators S T
                      P : Algebra.Generators R S
                      i : (Q.comp P).vars
                      ⊢ Eq ((MvPolynomial.aeval Q.val) ((fun i => Sum.elim MvPolynomial.X (Function. …
                    -/
                                /-
                                  🎉 no goals
                                -/
  aeval_val i := by cases i <;> simp
                                /-
                                  🎉 no goals
                                -/


lemma ofComp_toAlgHom_monomial_sumElim (Q : Generators S T) (P : Generators R S) (v₁ v₂ a) :
    (Q.ofComp P).toAlgHom (monomial (Finsupp.sumElim v₁ v₂) a) =
      monomial v₁ (aeval P.val (monomial v₂ a)) := by
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type u_2
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    v₁ : Finsupp Q.vars Nat
    v₂ : Finsupp P.vars Nat
    a : R
    ⊢ Eq ((Q.ofComp P).toAlgHom ((MvPolynomial.monomial (v₁.sumElim v₂)) a)) ((MvP …
  -/
  erw [Hom.toAlgHom_monomial]
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type u_2
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    v₁ : Finsupp Q.vars Nat
    v₂ : Finsupp P.vars Nat
    a : R
    ⊢ Eq (HSMul.hSMul a ((v₁.sumElim v₂).prod fun x1 x2 => HPow.hPow ((Q.ofComp P) …
  -/
  rw [monomial_eq]
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type u_2
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    v₁ : Finsupp Q.vars Nat
    v₂ : Finsupp P.vars Nat
    a : R
    ⊢ Eq (HSMul.hSMul a ((v₁.sumElim v₂).prod fun x1 x2 => HPow.hPow ((Q.ofComp P) …
  -/
  simp only [MvPolynomial.algebraMap_apply, ofComp_val, aeval_monomial]
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type u_2
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    v₁ : Finsupp Q.vars Nat
    v₂ : Finsupp P.vars Nat
    a : R
    ⊢ Eq (HSMul.hSMul a ((v₁.sumElim v₂).prod fun x1 x2 => HPow.hPow (Sum.elim MvP …
  -/
  rw [Finsupp.prod_sumElim]
  simp only [Function.comp_def, Sum.elim_inl, Sum.elim_inr, ← map_pow, ← map_finsupp_prod,
    C_mul, Algebra.smul_def, MvPolynomial.algebraMap_apply, mul_assoc]
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type u_2
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    v₁ : Finsupp Q.vars Nat
    v₂ : Finsupp P.vars Nat
    a : R
    ⊢ Eq (HMul.hMul (MvPolynomial.C ((algebraMap R S) a)) (HMul.hMul (v₁.prod fun  …
  -/
  nth_rw 2 [mul_comm]
  /-
    🎉 no goals
  -/


lemma toComp_toAlgHom_monomial (Q : Generators S T) (P : Generators R S) (j a) :
    (Q.toComp P).toAlgHom (monomial j a) =
      monomial (Finsupp.sumElim 0 j) a := by
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type u_2
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    j : Finsupp P.vars Nat
    a : R
    ⊢ Eq ((Q.toComp P).toAlgHom ((MvPolynomial.monomial j) a)) ((MvPolynomial.mono …
  -/
  convert rename_monomial _ _ _
  /-
    case h.e'_3.h.e'_5.h.h.e'_4.h.h.e
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type u_2
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    j : Finsupp P.vars Nat
    a : R
    e_1✝ : Eq (LinearMap (RingHom.id R) R (MvPolynomial (Sum Q.vars P.vars) R)) (L …
    e_2✝ : Eq (Sum Q.vars P.vars) (Q.comp P).vars
    ⊢ Eq (Finsupp.sumElim 0) (Finsupp.mapDomain Sum.inr)
  -/
  ext f (i₁ | i₂) <;>
    /-
      case h.e'_3.h.e'_5.h.h.e'_4.h.h.e.h.h.inl
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type u_2
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      j : Finsupp P.vars Nat
      a : R
      e_1✝ : Eq (LinearMap (RingHom.id R) R (MvPolynomial (Sum Q.vars P.vars) R)) (L …
      e_2✝ : Eq (Sum Q.vars P.vars) (Q.comp P).vars
      f : Finsupp P.vars Nat
      i₁ : Q.vars
      ⊢ Eq ((Finsupp.sumElim 0 f) (Sum.inl i₁)) ((Finsupp.mapDomain Sum.inr f) (Sum. …
    -/
    /-
      🎉 no goals
    -/
    simp [Finsupp.mapDomain_notin_range, Finsupp.mapDomain_apply Sum.inr_injective]
    /-
      🎉 no goals
    -/


/-- Given families of generators `X ⊆ T`, there is a map `R[X] → S[X]`. -/
@[simps]
noncomputable
def toExtendScalars (P : Generators R T) : Hom P (P.extendScalars S) where
  val := X
                    /-
                      R : Type u
                      S : Type v
                      inst✝¹⁸ : CommRing R
                      inst✝¹⁷ : CommRing S
                      inst✝¹⁶ : Algebra R S
                      P✝ : Algebra.Generators R S
                      R' : Type ?u.312028
                      S' : Type ?u.312031
                      inst✝¹⁵ : CommRing R'
                      inst✝¹⁴ : CommRing S'
                      inst✝¹³ : Algebra R' S'
                      P' : Algebra.Generators R' S'
                      R'' : Type ?u.312365
                      S'' : Type ?u.312368
                      inst✝¹² : CommRing R''
                      inst✝¹¹ : CommRing S''
                      inst✝¹⁰ : Algebra R'' S''
                      P'' : Algebra.Generators R'' S''
                      inst✝⁹ : Algebra R R'
                      inst✝⁸ : Algebra R' R''
                      inst✝⁷ : Algebra R' S''
                      inst✝⁶ : Algebra S S'
                      inst✝⁵ : Algebra S' S''
                      inst✝⁴ : Algebra S S''
                      T : Type ?u.314057
                      inst✝³ : CommRing T
                      inst✝² : Algebra R T
                      inst✝¹ : Algebra S T
                      inst✝ : IsScalarTower R S T
                      P : Algebra.Generators R T
                      i : P.vars
                      ⊢ Eq ((MvPolynomial.aeval (Algebra.Generators.extendScalars S P).val) (MvPolyn …
                    -/
  aeval_val i := by simp
                    /-
                      🎉 no goals
                    -/


variable {P P'} in
/-- Reinterpret a hom between generators as a hom between extensions. -/
@[simps]
noncomputable
def Hom.toExtensionHom [Algebra R S'] [IsScalarTower R R' S'] [IsScalarTower R S S']
    (f : P.Hom P') : P.toExtension.Hom P'.toExtension where
  toRingHom := f.toAlgHom.toRingHom
                               /-
                                 R : Type u
                                 S : Type v
                                 inst✝²¹ : CommRing R
                                 inst✝²⁰ : CommRing S
                                 inst✝¹⁹ : Algebra R S
                                 P : Algebra.Generators R S
                                 R' : Type ?u.318974
                                 S' : Type ?u.318977
                                 inst✝¹⁸ : CommRing R'
                                 inst✝¹⁷ : CommRing S'
                                 inst✝¹⁶ : Algebra R' S'
                                 P' : Algebra.Generators R' S'
                                 R'' : Type ?u.319311
                                 S'' : Type ?u.319314
                                 inst✝¹⁵ : CommRing R''
                                 inst✝¹⁴ : CommRing S''
                                 inst✝¹³ : Algebra R'' S''
                                 P'' : Algebra.Generators R'' S''
                                 inst✝¹² : Algebra R R'
                                 inst✝¹¹ : Algebra R' R''
                                 inst✝¹⁰ : Algebra R' S''
                                 inst✝⁹ : Algebra S S'
                                 inst✝⁸ : Algebra S' S''
                                 inst✝⁷ : Algebra S S''
                                 T : Type ?u.321003
                                 inst✝⁶ : CommRing T
                                 inst✝⁵ : Algebra R T
                                 inst✝⁴ : Algebra S T
                                 inst✝³ : IsScalarTower R S T
                                 inst✝² : Algebra R S'
                                 inst✝¹ : IsScalarTower R R' S'
                                 inst✝ : IsScalarTower R S S'
                                 f : P.Hom P'
                                 x : R
                                 ⊢ Eq (f.toAlgHom.toRingHom ((algebraMap R P.toExtension.Ring) x)) ((algebraMap …
                               -/
  toRingHom_algebraMap x := by simp
                               /-
                                 🎉 no goals
                               -/
                               /-
                                 R : Type u
                                 S : Type v
                                 inst✝²¹ : CommRing R
                                 inst✝²⁰ : CommRing S
                                 inst✝¹⁹ : Algebra R S
                                 P : Algebra.Generators R S
                                 R' : Type ?u.318974
                                 S' : Type ?u.318977
                                 inst✝¹⁸ : CommRing R'
                                 inst✝¹⁷ : CommRing S'
                                 inst✝¹⁶ : Algebra R' S'
                                 P' : Algebra.Generators R' S'
                                 R'' : Type ?u.319311
                                 S'' : Type ?u.319314
                                 inst✝¹⁵ : CommRing R''
                                 inst✝¹⁴ : CommRing S''
                                 inst✝¹³ : Algebra R'' S''
                                 P'' : Algebra.Generators R'' S''
                                 inst✝¹² : Algebra R R'
                                 inst✝¹¹ : Algebra R' R''
                                 inst✝¹⁰ : Algebra R' S''
                                 inst✝⁹ : Algebra S S'
                                 inst✝⁸ : Algebra S' S''
                                 inst✝⁷ : Algebra S S''
                                 T : Type ?u.321003
                                 inst✝⁶ : CommRing T
                                 inst✝⁵ : Algebra R T
                                 inst✝⁴ : Algebra S T
                                 inst✝³ : IsScalarTower R S T
                                 inst✝² : Algebra R S'
                                 inst✝¹ : IsScalarTower R R' S'
                                 inst✝ : IsScalarTower R S S'
                                 f : P.Hom P'
                                 x : P.toExtension.Ring
                                 ⊢ Eq ((algebraMap P'.toExtension.Ring S') (f.toAlgHom.toRingHom x)) ((algebraM …
                               -/
  algebraMap_toRingHom x := by simp
                               /-
                                 🎉 no goals
                               -/


@[simp]
                                                                       /-
                                                                         R : Type u
                                                                         S : Type v
                                                                         inst✝² : CommRing R
                                                                         inst✝¹ : CommRing S
                                                                         inst✝ : Algebra R S
                                                                         P : Algebra.Generators R S
                                                                         ⊢ Eq (Algebra.Generators.Hom.id P).toExtensionHom (Algebra.Extension.Hom.id P. …
                                                                       -/
lemma Hom.toExtensionHom_id : Hom.toExtensionHom (.id P) = .id _ := by ext; simp
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


@[simp]
lemma Hom.toExtensionHom_comp [Algebra R S'] [IsScalarTower R S S']
    [Algebra R R''] [Algebra R S''] [IsScalarTower R R'' S'']
    [IsScalarTower R S S''] [IsScalarTower R' R'' S''] [IsScalarTower R' S' S'']
    [IsScalarTower S S' S''] [IsScalarTower R R' R''] [IsScalarTower R R' S']
    (f : P'.Hom P'') (g : P.Hom P') :
                                                                             /-
                                                                               R : Type u
                                                                               S : Type v
                                                                               inst✝²⁵ : CommRing R
                                                                               inst✝²⁴ : CommRing S
                                                                               inst✝²³ : Algebra R S
                                                                               P : Algebra.Generators R S
                                                                               R' : Type u_4
                                                                               S' : Type u_1
                                                                               inst✝²² : CommRing R'
                                                                               inst✝²¹ : CommRing S'
                                                                               inst✝²⁰ : Algebra R' S'
                                                                               P' : Algebra.Generators R' S'
                                                                               R'' : Type u_2
                                                                               S'' : Type u_3
                                                                               inst✝¹⁹ : CommRing R''
                                                                               inst✝¹⁸ : CommRing S''
                                                                               inst✝¹⁷ : Algebra R'' S''
                                                                               P'' : Algebra.Generators R'' S''
                                                                               inst✝¹⁶ : Algebra R R'
                                                                               inst✝¹⁵ : Algebra R' R''
                                                                               inst✝¹⁴ : Algebra R' S''
                                                                               inst✝¹³ : Algebra S S'
                                                                               inst✝¹² : Algebra S' S''
                                                                               inst✝¹¹ : Algebra S S''
                                                                               inst✝¹⁰ : Algebra R S'
                                                                               inst✝⁹ : IsScalarTower R S S'
                                                                               inst✝⁸ : Algebra R R''
                                                                               inst✝⁷ : Algebra R S''
                                                                               inst✝⁶ : IsScalarTower R R'' S''
                                                                               inst✝⁵ : IsScalarTower R S S''
                                                                               inst✝⁴ : IsScalarTower R' R'' S''
                                                                               inst✝³ : IsScalarTower R' S' S''
                                                                               inst✝² : IsScalarTower S S' S''
                                                                               inst✝¹ : IsScalarTower R R' R''
                                                                               inst✝ : IsScalarTower R R' S'
                                                                               f : P'.Hom P''
                                                                               g : P.Hom P'
                                                                               ⊢ Eq (f.comp g).toExtensionHom (f.toExtensionHom.comp g.toExtensionHom)
                                                                             -/
    toExtensionHom (f.comp g) = f.toExtensionHom.comp g.toExtensionHom := by ext; simp
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


/-- The kernel of a presentation. -/
noncomputable abbrev ker : Ideal P.Ring := P.toExtension.ker


lemma ker_eq_ker_aeval_val : P.ker = RingHom.ker (aeval P.val) := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Algebra.Generators R S
    ⊢ Eq P.ker (RingHom.ker (MvPolynomial.aeval P.val))
  -/
  simp only [ker, Extension.ker, toExtension_Ring, algebraMap_eq]
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Algebra.Generators R S
    ⊢ Eq (RingHom.ker ↑(MvPolynomial.aeval P.val)) (RingHom.ker (MvPolynomial.aeva …
  -/
  rfl
  /-
    🎉 no goals
  -/


variable {P} in
                                                                       /-
                                                                         R : Type u
                                                                         S : Type v
                                                                         inst✝² : CommRing R
                                                                         inst✝¹ : CommRing S
                                                                         inst✝ : Algebra R S
                                                                         P : Algebra.Generators R S
                                                                         x : P.Ring
                                                                         hx : Membership.mem P.ker x
                                                                         ⊢ Eq ((MvPolynomial.aeval P.val) x) 0
                                                                       -/
lemma aeval_val_eq_zero {x} (hx : x ∈ P.ker) : aeval P.val x = 0 := by rwa [← algebraMap_apply]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


lemma map_toComp_ker (Q : Generators S T) (P : Generators R S) :
    P.ker.map (Q.toComp P).toAlgHom.toRingHom = RingHom.ker (Q.ofComp P).toAlgHom := by
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type u_2
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    ⊢ Eq (Ideal.map (Q.toComp P).toAlgHom.toRingHom P.ker) (RingHom.ker (Q.ofComp  …
  -/
  letI : DecidableEq (Q.vars →₀ ℕ) := Classical.decEq _
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type u_2
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    this : DecidableEq (Finsupp Q.vars Nat) := Classical.decEq (Finsupp Q.vars Nat)
    ⊢ Eq (Ideal.map (Q.toComp P).toAlgHom.toRingHom P.ker) (RingHom.ker (Q.ofComp  …
  -/
  apply le_antisymm
    /-
      case a
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type u_2
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      this : DecidableEq (Finsupp Q.vars Nat) := Classical.decEq (Finsupp Q.vars Nat)
      ⊢ LE.le (Ideal.map (Q.toComp P).toAlgHom.toRingHom P.ker) (RingHom.ker (Q.ofCo …
    -/
  · rw [Ideal.map_le_iff_le_comap]
    /-
      case a
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type u_2
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      this : DecidableEq (Finsupp Q.vars Nat) := Classical.decEq (Finsupp Q.vars Nat)
      ⊢ LE.le P.ker (Ideal.comap (Q.toComp P).toAlgHom.toRingHom (RingHom.ker (Q.ofC …
    -/
    rintro x (hx : algebraMap P.Ring S x = 0)
    have : (Q.ofComp P).toAlgHom.comp (Q.toComp P).toAlgHom = IsScalarTower.toAlgHom R _ _ := by
      ext1; simp
    simp only [comp_vars, AlgHom.toRingHom_eq_coe, Ideal.mem_comap, RingHom.coe_coe,
      RingHom.mem_ker, ← AlgHom.comp_apply, this, IsScalarTower.toAlgHom_apply]
    /-
      case a
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type u_2
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      this✝ : DecidableEq (Finsupp Q.vars Nat) := Classical.decEq (Finsupp Q.vars Nat)
      x : P.Ring
      hx : Eq ((algebraMap P.Ring S) x) 0
      this : Eq ((Q.ofComp P).toAlgHom.comp (Q.toComp P).toAlgHom) (IsScalarTower.to …
      ⊢ Eq ((algebraMap P.Ring Q.Ring) x) 0
    -/
    rw [IsScalarTower.algebraMap_apply P.Ring S, hx, map_zero]
    /-
      🎉 no goals
    -/
    /-
      case a
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type u_2
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      this : DecidableEq (Finsupp Q.vars Nat) := Classical.decEq (Finsupp Q.vars Nat)
      ⊢ LE.le (RingHom.ker (Q.ofComp P).toAlgHom) (Ideal.map (Q.toComp P).toAlgHom.t …
    -/
  · rintro x (h₂ : (Q.ofComp P).toAlgHom x = 0)
    let e : ((Q.comp P).vars →₀ ℕ) ≃+ (Q.vars →₀ ℕ) × (P.vars →₀ ℕ) :=
      Finsupp.sumFinsuppAddEquivProdFinsupp
    suffices ∑ v ∈ (support x).map e, (monomial (e.symm v)) (coeff (e.symm v) x) ∈
        Ideal.map (Q.toComp P).toAlgHom.toRingHom P.ker by
      simpa only [AlgHom.toRingHom_eq_coe, Finset.sum_map, Equiv.coe_toEmbedding,
        EquivLike.coe_coe, AddEquiv.symm_apply_apply, support_sum_monomial_coeff] using this
    /-
      case a
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type u_2
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      this : DecidableEq (Finsupp Q.vars Nat) := Classical.decEq (Finsupp Q.vars Nat)
      x : (Q.comp P).Ring
      h₂ : Eq ((Q.ofComp P).toAlgHom x) 0
      e : AddEquiv (Finsupp (Q.comp P).vars Nat) (Prod (Finsupp Q.vars Nat) (Finsupp …
      ⊢ Membership.mem (Ideal.map (Q.toComp P).toAlgHom.toRingHom P.ker) ((Finset.ma …
    -/
    rw [← Finset.sum_fiberwise_of_maps_to (fun i ↦ Finset.mem_image_of_mem Prod.fst)]
    /-
      case a
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type u_2
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      this : DecidableEq (Finsupp Q.vars Nat) := Classical.decEq (Finsupp Q.vars Nat)
      x : (Q.comp P).Ring
      h₂ : Eq ((Q.ofComp P).toAlgHom x) 0
      e : AddEquiv (Finsupp (Q.comp P).vars Nat) (Prod (Finsupp Q.vars Nat) (Finsupp …
      ⊢ Membership.mem (Ideal.map (Q.toComp P).toAlgHom.toRingHom P.ker) ((Finset.im …
    -/
    refine sum_mem fun i hi ↦ ?_
    convert_to monomial (e.symm (i, 0)) 1 * (Q.toComp P).toAlgHom.toRingHom
      (∑ j ∈ ((support x).map e.toEmbedding).filter (fun x ↦ x.1 = i),
        monomial j.2 (coeff (e.symm j) x)) ∈ _
      /-
        case h.e'_1
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        T : Type u_2
        inst✝³ : CommRing T
        inst✝² : Algebra R T
        inst✝¹ : Algebra S T
        inst✝ : IsScalarTower R S T
        Q : Algebra.Generators S T
        P : Algebra.Generators R S
        this : DecidableEq (Finsupp Q.vars Nat) := Classical.decEq (Finsupp Q.vars Nat)
        x : (Q.comp P).Ring
        h₂ : Eq ((Q.ofComp P).toAlgHom x) 0
        e : AddEquiv (Finsupp (Q.comp P).vars Nat) (Prod (Finsupp Q.vars Nat) (Finsupp …
        i : Finsupp Q.vars Nat
        hi : Membership.mem (Finset.image Prod.fst (Finset.map (↑e).toEmbedding (MvPol …
        ⊢ Eq ((Finset.filter (fun i_1 => Eq i_1.1 i) (Finset.map (↑e).toEmbedding (MvP …
      -/
    · rw [map_sum, Finset.mul_sum]
      /-
        case h.e'_1
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        T : Type u_2
        inst✝³ : CommRing T
        inst✝² : Algebra R T
        inst✝¹ : Algebra S T
        inst✝ : IsScalarTower R S T
        Q : Algebra.Generators S T
        P : Algebra.Generators R S
        this : DecidableEq (Finsupp Q.vars Nat) := Classical.decEq (Finsupp Q.vars Nat)
        x : (Q.comp P).Ring
        h₂ : Eq ((Q.ofComp P).toAlgHom x) 0
        e : AddEquiv (Finsupp (Q.comp P).vars Nat) (Prod (Finsupp Q.vars Nat) (Finsupp …
        i : Finsupp Q.vars Nat
        hi : Membership.mem (Finset.image Prod.fst (Finset.map (↑e).toEmbedding (MvPol …
        ⊢ Eq ((Finset.filter (fun i_1 => Eq i_1.1 i) (Finset.map (↑e).toEmbedding (MvP …
      -/
      refine Finset.sum_congr rfl fun j hj ↦ ?_
      /-
        case h.e'_1
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        T : Type u_2
        inst✝³ : CommRing T
        inst✝² : Algebra R T
        inst✝¹ : Algebra S T
        inst✝ : IsScalarTower R S T
        Q : Algebra.Generators S T
        P : Algebra.Generators R S
        this : DecidableEq (Finsupp Q.vars Nat) := Classical.decEq (Finsupp Q.vars Nat)
        x : (Q.comp P).Ring
        h₂ : Eq ((Q.ofComp P).toAlgHom x) 0
        e : AddEquiv (Finsupp (Q.comp P).vars Nat) (Prod (Finsupp Q.vars Nat) (Finsupp …
        i : Finsupp Q.vars Nat
        hi : Membership.mem (Finset.image Prod.fst (Finset.map (↑e).toEmbedding (MvPol …
        j : Prod (Finsupp Q.vars Nat) (Finsupp P.vars Nat)
        hj : Membership.mem (Finset.filter (fun x => Eq x.1 i) (Finset.map e.toEmbeddi …
        ⊢ Eq ((MvPolynomial.monomial (e.symm j)) (MvPolynomial.coeff (e.symm j) x)) (H …
      -/
      obtain rfl := (Finset.mem_filter.mp hj).2
      /-
        case h.e'_1
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        T : Type u_2
        inst✝³ : CommRing T
        inst✝² : Algebra R T
        inst✝¹ : Algebra S T
        inst✝ : IsScalarTower R S T
        Q : Algebra.Generators S T
        P : Algebra.Generators R S
        this : DecidableEq (Finsupp Q.vars Nat) := Classical.decEq (Finsupp Q.vars Nat)
        x : (Q.comp P).Ring
        h₂ : Eq ((Q.ofComp P).toAlgHom x) 0
        e : AddEquiv (Finsupp (Q.comp P).vars Nat) (Prod (Finsupp Q.vars Nat) (Finsupp …
        j : Prod (Finsupp Q.vars Nat) (Finsupp P.vars Nat)
        hi : Membership.mem (Finset.image Prod.fst (Finset.map (↑e).toEmbedding (MvPol …
        hj : Membership.mem (Finset.filter (fun x => Eq x.1 j.1) (Finset.map e.toEmbed …
        ⊢ Eq ((MvPolynomial.monomial (e.symm j)) (MvPolynomial.coeff (e.symm j) x)) (H …
      -/
      obtain ⟨i, j⟩ := j
      /-
        case h.e'_1.mk
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        T : Type u_2
        inst✝³ : CommRing T
        inst✝² : Algebra R T
        inst✝¹ : Algebra S T
        inst✝ : IsScalarTower R S T
        Q : Algebra.Generators S T
        P : Algebra.Generators R S
        this : DecidableEq (Finsupp Q.vars Nat) := Classical.decEq (Finsupp Q.vars Nat)
        x : (Q.comp P).Ring
        h₂ : Eq ((Q.ofComp P).toAlgHom x) 0
        e : AddEquiv (Finsupp (Q.comp P).vars Nat) (Prod (Finsupp Q.vars Nat) (Finsupp …
        i : Finsupp Q.vars Nat
        j : Finsupp P.vars Nat
        hi : Membership.mem (Finset.image Prod.fst (Finset.map (↑e).toEmbedding (MvPol …
        hj : Membership.mem (Finset.filter (fun x => Eq x.1 { fst := i, snd := j }.1)  …
        ⊢ Eq ((MvPolynomial.monomial (e.symm { fst := i, snd := j })) (MvPolynomial.co …
      -/
      clear hj hi
      have : (Q.toComp P).toAlgHom (monomial j (coeff (e.symm (i, j)) x)) =
          monomial (e.symm (0, j)) (coeff (e.symm (i, j)) x) :=
        toComp_toAlgHom_monomial ..
      simp only [AlgHom.toRingHom_eq_coe, monomial_zero', RingHom.coe_coe, algHom_C,
          MvPolynomial.algebraMap_eq, this]
      /-
        case h.e'_1.mk
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        T : Type u_2
        inst✝³ : CommRing T
        inst✝² : Algebra R T
        inst✝¹ : Algebra S T
        inst✝ : IsScalarTower R S T
        Q : Algebra.Generators S T
        P : Algebra.Generators R S
        this✝ : DecidableEq (Finsupp Q.vars Nat) := Classical.decEq (Finsupp Q.vars Nat)
        x : (Q.comp P).Ring
        h₂ : Eq ((Q.ofComp P).toAlgHom x) 0
        e : AddEquiv (Finsupp (Q.comp P).vars Nat) (Prod (Finsupp Q.vars Nat) (Finsupp …
        i : Finsupp Q.vars Nat
        j : Finsupp P.vars Nat
        this : Eq ((Q.toComp P).toAlgHom ((MvPolynomial.monomial j) (MvPolynomial.coef …
        ⊢ Eq ((MvPolynomial.monomial (e.symm { fst := i, snd := j })) (MvPolynomial.co …
      -/
      rw [monomial_mul, ← map_add, Prod.mk_add_mk, add_zero, zero_add, one_mul]
      /-
        🎉 no goals
      -/
      /-
        case a.convert_4
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        T : Type u_2
        inst✝³ : CommRing T
        inst✝² : Algebra R T
        inst✝¹ : Algebra S T
        inst✝ : IsScalarTower R S T
        Q : Algebra.Generators S T
        P : Algebra.Generators R S
        this : DecidableEq (Finsupp Q.vars Nat) := Classical.decEq (Finsupp Q.vars Nat)
        x : (Q.comp P).Ring
        h₂ : Eq ((Q.ofComp P).toAlgHom x) 0
        e : AddEquiv (Finsupp (Q.comp P).vars Nat) (Prod (Finsupp Q.vars Nat) (Finsupp …
        i : Finsupp Q.vars Nat
        hi : Membership.mem (Finset.image Prod.fst (Finset.map (↑e).toEmbedding (MvPol …
        ⊢ Membership.mem (Ideal.map (Q.toComp P).toAlgHom.toRingHom P.ker) (HMul.hMul  …
      -/
    · apply Ideal.mul_mem_left
      /-
        case a.convert_4.a
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        T : Type u_2
        inst✝³ : CommRing T
        inst✝² : Algebra R T
        inst✝¹ : Algebra S T
        inst✝ : IsScalarTower R S T
        Q : Algebra.Generators S T
        P : Algebra.Generators R S
        this : DecidableEq (Finsupp Q.vars Nat) := Classical.decEq (Finsupp Q.vars Nat)
        x : (Q.comp P).Ring
        h₂ : Eq ((Q.ofComp P).toAlgHom x) 0
        e : AddEquiv (Finsupp (Q.comp P).vars Nat) (Prod (Finsupp Q.vars Nat) (Finsupp …
        i : Finsupp Q.vars Nat
        hi : Membership.mem (Finset.image Prod.fst (Finset.map (↑e).toEmbedding (MvPol …
        ⊢ Membership.mem (Ideal.map (Q.toComp P).toAlgHom.toRingHom P.ker) ((Q.toComp  …
      -/
      refine Ideal.mem_map_of_mem _ ?_
      /-
        case a.convert_4.a
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        T : Type u_2
        inst✝³ : CommRing T
        inst✝² : Algebra R T
        inst✝¹ : Algebra S T
        inst✝ : IsScalarTower R S T
        Q : Algebra.Generators S T
        P : Algebra.Generators R S
        this : DecidableEq (Finsupp Q.vars Nat) := Classical.decEq (Finsupp Q.vars Nat)
        x : (Q.comp P).Ring
        h₂ : Eq ((Q.ofComp P).toAlgHom x) 0
        e : AddEquiv (Finsupp (Q.comp P).vars Nat) (Prod (Finsupp Q.vars Nat) (Finsupp …
        i : Finsupp Q.vars Nat
        hi : Membership.mem (Finset.image Prod.fst (Finset.map (↑e).toEmbedding (MvPol …
        ⊢ Membership.mem P.ker ((Finset.filter (fun x => Eq x.1 i) (Finset.map e.toEmb …
      -/
      simp only [ker_eq_ker_aeval_val, AddEquiv.toEquiv_eq_coe, RingHom.mem_ker, map_sum]
      /-
        case a.convert_4.a
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        T : Type u_2
        inst✝³ : CommRing T
        inst✝² : Algebra R T
        inst✝¹ : Algebra S T
        inst✝ : IsScalarTower R S T
        Q : Algebra.Generators S T
        P : Algebra.Generators R S
        this : DecidableEq (Finsupp Q.vars Nat) := Classical.decEq (Finsupp Q.vars Nat)
        x : (Q.comp P).Ring
        h₂ : Eq ((Q.ofComp P).toAlgHom x) 0
        e : AddEquiv (Finsupp (Q.comp P).vars Nat) (Prod (Finsupp Q.vars Nat) (Finsupp …
        i : Finsupp Q.vars Nat
        hi : Membership.mem (Finset.image Prod.fst (Finset.map (↑e).toEmbedding (MvPol …
        ⊢ Eq ((Finset.filter (fun x => Eq x.1 i) (Finset.map (↑e).toEmbedding (MvPolyn …
      -/
      rw [← coeff_zero i, ← h₂]
      /-
        case a.convert_4.a
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        T : Type u_2
        inst✝³ : CommRing T
        inst✝² : Algebra R T
        inst✝¹ : Algebra S T
        inst✝ : IsScalarTower R S T
        Q : Algebra.Generators S T
        P : Algebra.Generators R S
        this : DecidableEq (Finsupp Q.vars Nat) := Classical.decEq (Finsupp Q.vars Nat)
        x : (Q.comp P).Ring
        h₂ : Eq ((Q.ofComp P).toAlgHom x) 0
        e : AddEquiv (Finsupp (Q.comp P).vars Nat) (Prod (Finsupp Q.vars Nat) (Finsupp …
        i : Finsupp Q.vars Nat
        hi : Membership.mem (Finset.image Prod.fst (Finset.map (↑e).toEmbedding (MvPol …
        ⊢ Eq ((Finset.filter (fun x => Eq x.1 i) (Finset.map (↑e).toEmbedding (MvPolyn …
      -/
      clear h₂ hi
      have (x : (Q.comp P).Ring) : (Function.support fun a ↦ if a.1 = i then aeval P.val
          (monomial a.2 (coeff (e.symm a) x)) else 0) ⊆ ((support x).map e).toSet := by
        rw [← Set.compl_subset_compl]
        intro j
        obtain ⟨j, rfl⟩ := e.surjective j
        simp_all
      /-
        case a.convert_4.a
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        T : Type u_2
        inst✝³ : CommRing T
        inst✝² : Algebra R T
        inst✝¹ : Algebra S T
        inst✝ : IsScalarTower R S T
        Q : Algebra.Generators S T
        P : Algebra.Generators R S
        this✝ : DecidableEq (Finsupp Q.vars Nat) := Classical.decEq (Finsupp Q.vars Nat)
        x : (Q.comp P).Ring
        e : AddEquiv (Finsupp (Q.comp P).vars Nat) (Prod (Finsupp Q.vars Nat) (Finsupp …
        i : Finsupp Q.vars Nat
        this : ∀ (x : (Q.comp P).Ring), HasSubset.Subset (Function.support fun a => it …
        ⊢ Eq ((Finset.filter (fun x => Eq x.1 i) (Finset.map (↑e).toEmbedding (MvPolyn …
      -/
      rw [Finset.sum_filter, ← finsum_eq_sum_of_support_subset _ (this x)]
      induction x using MvPolynomial.induction_on' with
      | h1 v a =>
        rw [finsum_eq_sum_of_support_subset _ (this _), ← Finset.sum_filter]
        obtain ⟨v, rfl⟩ := e.symm.surjective v
        erw [ofComp_toAlgHom_monomial_sumElim]
        classical
        simp only [comp_vars, coeff_monomial, ← e.injective.eq_iff,
          map_zero, AddEquiv.apply_symm_apply, apply_ite]
        rw [← apply_ite, Finset.sum_ite_eq]
        simp only [Finset.mem_filter, Finset.mem_map_equiv, AddEquiv.coe_toEquiv_symm, comp_vars,
          mem_support_iff, coeff_monomial, ↓reduceIte, ne_eq, ite_and, ite_not]
        split
        · simp only [zero_smul, coeff_zero, *, map_zero, ite_self]
        · congr
      | h2 p q hp hq =>
        simp only [coeff_add, map_add, ite_add_zero]
        rw [finsum_add_distrib, hp, hq]
        · refine (((support p).map e).finite_toSet.subset ?_)
          convert this p
        · refine (((support q).map e).finite_toSet.subset ?_)
          convert this q


