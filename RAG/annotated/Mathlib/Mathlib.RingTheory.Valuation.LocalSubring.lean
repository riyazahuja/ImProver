/-- Cast a valuation subring to a local subring. -/
def ValuationSubring.toLocalSubring (A : ValuationSubring K) : LocalSubring K where
  toSubring := A.toSubring
  isLocalRing := A.isLocalRing


lemma ValuationSubring.toLocalSubring_injective :
    Function.Injective (ValuationSubring.toLocalSubring (K := K)) :=
  fun _ _ h ↦ ValuationSubring.toSubring_injective congr(($h).toSubring)


lemma LocalSubring.map_maximalIdeal_eq_top_of_isMax {R : LocalSubring K}
    (hR : IsMax R) {S : Subring K} (hS : R.toSubring < S) :
    (maximalIdeal R.toSubring).map (Subring.inclusion hS.le) = ⊤ := by
  /-
    K : Type u_3
    inst✝ : Field K
    R : LocalSubring K
    hR : IsMax R
    S : Subring K
    hS : LT.lt R.toSubring S
    ⊢ Eq (Ideal.map (Subring.inclusion ⋯) (IsLocalRing.maximalIdeal (Subtype fun x …
  -/
  let mR := (maximalIdeal R.toSubring).map (Subring.inclusion hS.le)
  /-
    K : Type u_3
    inst✝ : Field K
    R : LocalSubring K
    hR : IsMax R
    S : Subring K
    hS : LT.lt R.toSubring S
    mR : Ideal (Subtype fun x => Membership.mem S x) := Ideal.map (Subring.inclusi …
    ⊢ Eq (Ideal.map (Subring.inclusion ⋯) (IsLocalRing.maximalIdeal (Subtype fun x …
  -/
  by_contra h_is_not_top
  /-
    K : Type u_3
    inst✝ : Field K
    R : LocalSubring K
    hR : IsMax R
    S : Subring K
    hS : LT.lt R.toSubring S
    mR : Ideal (Subtype fun x => Membership.mem S x) := Ideal.map (Subring.inclusi …
    h_is_not_top : Not (Eq (Ideal.map (Subring.inclusion ⋯) (IsLocalRing.maximalId …
    ⊢ False
  -/
  obtain ⟨M, h_is_max, h_incl⟩ := Ideal.exists_le_maximal _ h_is_not_top
  /-
    case intro.intro
    K : Type u_3
    inst✝ : Field K
    R : LocalSubring K
    hR : IsMax R
    S : Subring K
    hS : LT.lt R.toSubring S
    mR : Ideal (Subtype fun x => Membership.mem S x) := Ideal.map (Subring.inclusi …
    h_is_not_top : Not (Eq (Ideal.map (Subring.inclusion ⋯) (IsLocalRing.maximalId …
    M : Ideal (Subtype fun x => Membership.mem S x)
    h_is_max : M.IsMaximal
    h_incl : LE.le (Ideal.map (Subring.inclusion ⋯) (IsLocalRing.maximalIdeal (Sub …
    ⊢ False
  -/
  let fSₘ : LocalSubring K := LocalSubring.ofPrime S M
  have h_RleSₘ : R ≤ fSₘ := by
    refine ⟨hS.le.trans (LocalSubring.le_ofPrime _ _), ⟨?_⟩⟩
    rintro ⟨a, h_a_inR⟩ h_fa_isUnit
    apply (IsLocalization.AtPrime.isUnit_to_map_iff _ M ⟨a, hS.le h_a_inR⟩).mp at h_fa_isUnit
    by_contra h
    rw [← mem_nonunits_iff, ← mem_maximalIdeal] at h
    apply Ideal.mem_map_of_mem (Subring.inclusion hS.le) at h
    exact h_fa_isUnit (h_incl h)
  have h_RneSₘ : R ≠ fSₘ :=
    fun e ↦ (hS.trans_le (LocalSubring.le_ofPrime S M)).ne congr(($e).toSubring)
  /-
    case intro.intro
    K : Type u_3
    inst✝ : Field K
    R : LocalSubring K
    hR : IsMax R
    S : Subring K
    hS : LT.lt R.toSubring S
    mR : Ideal (Subtype fun x => Membership.mem S x) := Ideal.map (Subring.inclusi …
    h_is_not_top : Not (Eq (Ideal.map (Subring.inclusion ⋯) (IsLocalRing.maximalId …
    M : Ideal (Subtype fun x => Membership.mem S x)
    h_is_max : M.IsMaximal
    h_incl : LE.le (Ideal.map (Subring.inclusion ⋯) (IsLocalRing.maximalIdeal (Sub …
    fSₘ : LocalSubring K := LocalSubring.ofPrime S M
    h_RleSₘ : LE.le R fSₘ
    h_RneSₘ : Ne R fSₘ
    ⊢ False
  -/
  exact h_RneSₘ (hR.eq_of_le h_RleSₘ)
  /-
    🎉 no goals
  -/


@[stacks 00IC]
lemma LocalSubring.mem_of_isMax_of_isIntegral {R : LocalSubring K}
    (hR : IsMax R) {x : K} (hx : IsIntegral R.toSubring x) : x ∈ R.toSubring := by
  /-
    K : Type u_3
    inst✝ : Field K
    R : LocalSubring K
    hR : IsMax R
    x : K
    hx : IsIntegral (Subtype fun x => Membership.mem R.toSubring x) x
    ⊢ Membership.mem R.toSubring x
  -/
  let S := Algebra.adjoin R.toSubring {x}
  /-
    K : Type u_3
    inst✝ : Field K
    R : LocalSubring K
    hR : IsMax R
    x : K
    hx : IsIntegral (Subtype fun x => Membership.mem R.toSubring x) x
    S : Subalgebra (Subtype fun x => Membership.mem R.toSubring x) K := Algebra.ad …
    ⊢ Membership.mem R.toSubring x
  -/
  have : Algebra.IsIntegral R.toSubring S := Algebra.IsIntegral.adjoin (by simpa)
  obtain ⟨Q : Ideal S.toSubring, hQ, e⟩ := Ideal.exists_ideal_over_maximal_of_isIntegral
    (R := R.toSubring) (S := S) (maximalIdeal _) (le_maximalIdeal (by simp [Ideal.eq_top_iff_one]))
  have : R = .ofPrime S.toSubring Q := by
    have hRS : R.toSubring ≤ S.toSubring := fun r hr ↦ algebraMap_mem S ⟨r, hr⟩
    apply hR.eq_of_le ⟨hRS.trans (LocalSubring.le_ofPrime _ _), ⟨?_⟩⟩
    intro r hr
    have := (IsLocalization.AtPrime.isUnit_to_map_iff (R := S.toSubring) _ Q ⟨_, hRS r.2⟩).mp hr
    by_contra h
    rw [← mem_nonunits_iff, ← mem_maximalIdeal, ← e] at h
    exact this h
  /-
    case intro.intro
    K : Type u_3
    inst✝ : Field K
    R : LocalSubring K
    hR : IsMax R
    x : K
    hx : IsIntegral (Subtype fun x => Membership.mem R.toSubring x) x
    S : Subalgebra (Subtype fun x => Membership.mem R.toSubring x) K := Algebra.ad …
    this✝ : Algebra.IsIntegral (Subtype fun x => Membership.mem R.toSubring x) (Su …
    Q : Ideal (Subtype fun x => Membership.mem S.toSubring x)
    hQ : Q.IsMaximal
    e : Eq (Ideal.comap (algebraMap (Subtype fun x => Membership.mem R.toSubring x …
    this : Eq R (LocalSubring.ofPrime S.toSubring Q)
    ⊢ Membership.mem R.toSubring x
  -/
  rw [this]
  /-
    case intro.intro
    K : Type u_3
    inst✝ : Field K
    R : LocalSubring K
    hR : IsMax R
    x : K
    hx : IsIntegral (Subtype fun x => Membership.mem R.toSubring x) x
    S : Subalgebra (Subtype fun x => Membership.mem R.toSubring x) K := Algebra.ad …
    this✝ : Algebra.IsIntegral (Subtype fun x => Membership.mem R.toSubring x) (Su …
    Q : Ideal (Subtype fun x => Membership.mem S.toSubring x)
    hQ : Q.IsMaximal
    e : Eq (Ideal.comap (algebraMap (Subtype fun x => Membership.mem R.toSubring x …
    this : Eq R (LocalSubring.ofPrime S.toSubring Q)
    ⊢ Membership.mem (LocalSubring.ofPrime S.toSubring Q).toSubring x
  -/
  exact LocalSubring.le_ofPrime _ _ (Algebra.self_mem_adjoin_singleton _ _)
  /-
    🎉 no goals
  -/


@[stacks 052K]
lemma ValuationSubring.isMax_toLocalSubring (R : ValuationSubring K) :
    IsMax R.toLocalSubring := by
  /-
    K : Type u_3
    inst✝ : Field K
    R : ValuationSubring K
    ⊢ IsMax R.toLocalSubring
  -/
  intro S hS
  /-
    K : Type u_3
    inst✝ : Field K
    R : ValuationSubring K
    S : LocalSubring K
    hS : LE.le R.toLocalSubring S
    ⊢ LE.le S R.toLocalSubring
  -/
  suffices R.toLocalSubring = S from this.ge
  /-
    K : Type u_3
    inst✝ : Field K
    R : ValuationSubring K
    S : LocalSubring K
    hS : LE.le R.toLocalSubring S
    ⊢ Eq R.toLocalSubring S
  -/
  refine LocalSubring.toSubring_injective (le_antisymm hS.1 ?_)
  /-
    K : Type u_3
    inst✝ : Field K
    R : ValuationSubring K
    S : LocalSubring K
    hS : LE.le R.toLocalSubring S
    ⊢ LE.le S.toSubring R.toLocalSubring.toSubring
  -/
  intro x hx
  /-
    K : Type u_3
    inst✝ : Field K
    R : ValuationSubring K
    S : LocalSubring K
    hS : LE.le R.toLocalSubring S
    x : K
    hx : Membership.mem S.toSubring x
    ⊢ Membership.mem R.toLocalSubring.toSubring x
  -/
  refine (R.2 x).elim id fun h ↦ ?_
  /-
    K : Type u_3
    inst✝ : Field K
    R : ValuationSubring K
    S : LocalSubring K
    hS : LE.le R.toLocalSubring S
    x : K
    hx : Membership.mem S.toSubring x
    h : Membership.mem R.carrier (Inv.inv x)
    ⊢ Membership.mem R.toLocalSubring.toSubring x
  -/
  by_contra h'
  /-
    K : Type u_3
    inst✝ : Field K
    R : ValuationSubring K
    S : LocalSubring K
    hS : LE.le R.toLocalSubring S
    x : K
    hx : Membership.mem S.toSubring x
    h : Membership.mem R.carrier (Inv.inv x)
    h' : Not (Membership.mem R.toLocalSubring.toSubring x)
    ⊢ False
  -/
  have hx0 : x ≠ 0 := by rintro rfl; exact h' (zero_mem R)
  have : IsUnit (Subring.inclusion hS.1 ⟨x⁻¹, h⟩) :=
    isUnit_iff_exists_inv.mpr ⟨⟨x, hx⟩, Subtype.ext (inv_mul_cancel₀ hx0)⟩
  /-
    K : Type u_3
    inst✝ : Field K
    R : ValuationSubring K
    S : LocalSubring K
    hS : LE.le R.toLocalSubring S
    x : K
    hx : Membership.mem S.toSubring x
    h : Membership.mem R.carrier (Inv.inv x)
    h' : Not (Membership.mem R.toLocalSubring.toSubring x)
    hx0 : Ne x 0
    this : IsUnit ((Subring.inclusion ⋯) ⟨Inv.inv x, h⟩)
    ⊢ False
  -/
  obtain ⟨x', hx'⟩ := isUnit_iff_exists_inv.mp (hS.2.1 _ this)
  /-
    case intro
    K : Type u_3
    inst✝ : Field K
    R : ValuationSubring K
    S : LocalSubring K
    hS : LE.le R.toLocalSubring S
    x : K
    hx : Membership.mem S.toSubring x
    h : Membership.mem R.carrier (Inv.inv x)
    h' : Not (Membership.mem R.toLocalSubring.toSubring x)
    hx0 : Ne x 0
    this : IsUnit ((Subring.inclusion ⋯) ⟨Inv.inv x, h⟩)
    x' : Subtype fun x => Membership.mem R.toLocalSubring.toSubring x
    hx' : Eq (HMul.hMul ⟨Inv.inv x, h⟩ x') 1
    ⊢ False
  -/
  have : x'.1 = x := by simpa [Subtype.ext_iff, inv_mul_eq_iff_eq_mul₀ hx0] using hx'
  /-
    case intro
    K : Type u_3
    inst✝ : Field K
    R : ValuationSubring K
    S : LocalSubring K
    hS : LE.le R.toLocalSubring S
    x : K
    hx : Membership.mem S.toSubring x
    h : Membership.mem R.carrier (Inv.inv x)
    h' : Not (Membership.mem R.toLocalSubring.toSubring x)
    hx0 : Ne x 0
    this✝ : IsUnit ((Subring.inclusion ⋯) ⟨Inv.inv x, h⟩)
    x' : Subtype fun x => Membership.mem R.toLocalSubring.toSubring x
    hx' : Eq (HMul.hMul ⟨Inv.inv x, h⟩ x') 1
    this : Eq (↑x') x
    ⊢ False
  -/
  exact h' (this ▸ x'.2)
  /-
    🎉 no goals
  -/


@[stacks 00IB]
lemma LocalSubring.exists_valuationRing_of_isMax {R : LocalSubring K} (hR : IsMax R) :
    ∃ R' : ValuationSubring K, R'.toLocalSubring = R := by
  suffices ∀ x ∉ R.toSubring, x⁻¹ ∈ R.toSubring from
    ⟨⟨R.toSubring, fun x ↦ or_iff_not_imp_left.mpr (this x)⟩, rfl⟩
  /-
    K : Type u_3
    inst✝ : Field K
    R : LocalSubring K
    hR : IsMax R
    ⊢ ∀ (x : K), Not (Membership.mem R.toSubring x) → Membership.mem R.toSubring ( …
  -/
  intros x hx
  /-
    K : Type u_3
    inst✝ : Field K
    R : LocalSubring K
    hR : IsMax R
    x : K
    hx : Not (Membership.mem R.toSubring x)
    ⊢ Membership.mem R.toSubring (Inv.inv x)
  -/
  have hx0 : x ≠ 0 := fun e ↦ hx (e ▸ zero_mem _)
  /-
    K : Type u_3
    inst✝ : Field K
    R : LocalSubring K
    hR : IsMax R
    x : K
    hx : Not (Membership.mem R.toSubring x)
    hx0 : Ne x 0
    ⊢ Membership.mem R.toSubring (Inv.inv x)
  -/
  apply mem_of_isMax_of_isIntegral hR
  /-
    K : Type u_3
    inst✝ : Field K
    R : LocalSubring K
    hR : IsMax R
    x : K
    hx : Not (Membership.mem R.toSubring x)
    hx0 : Ne x 0
    ⊢ IsIntegral (Subtype fun x => Membership.mem R.toSubring x) (Inv.inv x)
  -/
  let S := Algebra.adjoin R.toSubring {x}
  have : R.toSubring < S.toSubring := SetLike.lt_iff_le_and_exists.mpr
    ⟨fun r hr ↦ algebraMap_mem S ⟨r, hr⟩, ⟨x, Algebra.self_mem_adjoin_singleton _ _, hx⟩⟩
  /-
    K : Type u_3
    inst✝ : Field K
    R : LocalSubring K
    hR : IsMax R
    x : K
    hx : Not (Membership.mem R.toSubring x)
    hx0 : Ne x 0
    S : Subalgebra (Subtype fun x => Membership.mem R.toSubring x) K := Algebra.ad …
    this : LT.lt R.toSubring S.toSubring
    ⊢ IsIntegral (Subtype fun x => Membership.mem R.toSubring x) (Inv.inv x)
  -/
  have := map_maximalIdeal_eq_top_of_isMax hR this
  /-
    K : Type u_3
    inst✝ : Field K
    R : LocalSubring K
    hR : IsMax R
    x : K
    hx : Not (Membership.mem R.toSubring x)
    hx0 : Ne x 0
    S : Subalgebra (Subtype fun x => Membership.mem R.toSubring x) K := Algebra.ad …
    this✝ : LT.lt R.toSubring S.toSubring
    this : Eq (Ideal.map (Subring.inclusion ⋯) (IsLocalRing.maximalIdeal (Subtype  …
    ⊢ IsIntegral (Subtype fun x => Membership.mem R.toSubring x) (Inv.inv x)
  -/
  rw [Ideal.eq_top_iff_one] at this
  /-
    K : Type u_3
    inst✝ : Field K
    R : LocalSubring K
    hR : IsMax R
    x : K
    hx : Not (Membership.mem R.toSubring x)
    hx0 : Ne x 0
    S : Subalgebra (Subtype fun x => Membership.mem R.toSubring x) K := Algebra.ad …
    this✝ : LT.lt R.toSubring S.toSubring
    this : Membership.mem (Ideal.map (Subring.inclusion ⋯) (IsLocalRing.maximalIde …
    ⊢ IsIntegral (Subtype fun x => Membership.mem R.toSubring x) (Inv.inv x)
  -/
  obtain ⟨p, hp, hp'⟩ := (Algebra.mem_ideal_map_adjoin _ _).mp this
  /-
    case intro.intro
    K : Type u_3
    inst✝ : Field K
    R : LocalSubring K
    hR : IsMax R
    x : K
    hx : Not (Membership.mem R.toSubring x)
    hx0 : Ne x 0
    S : Subalgebra (Subtype fun x => Membership.mem R.toSubring x) K := Algebra.ad …
    this✝ : LT.lt R.toSubring S.toSubring
    this : Membership.mem (Ideal.map (Subring.inclusion ⋯) (IsLocalRing.maximalIde …
    p : Polynomial (Subtype fun x => Membership.mem R.toSubring x)
    hp : ∀ (i : Nat), Membership.mem (IsLocalRing.maximalIdeal (Subtype fun x => M …
    hp' : Eq ((Polynomial.aeval x) p) ↑1
    ⊢ IsIntegral (Subtype fun x => Membership.mem R.toSubring x) (Inv.inv x)
  -/
  have := IsUnit.invertible (isUnit_iff_ne_zero.mpr hx0)
  have : Polynomial.aeval (⅟x) (p - 1).reverse = 0 := by
    simpa [← Polynomial.aeval_def, hp'] using
      Polynomial.eval₂_reverse_eq_zero_iff (algebraMap R.toSubring K) x (p - 1)
  /-
    case intro.intro
    K : Type u_3
    inst✝ : Field K
    R : LocalSubring K
    hR : IsMax R
    x : K
    hx : Not (Membership.mem R.toSubring x)
    hx0 : Ne x 0
    S : Subalgebra (Subtype fun x => Membership.mem R.toSubring x) K := Algebra.ad …
    this✝² : LT.lt R.toSubring S.toSubring
    this✝¹ : Membership.mem (Ideal.map (Subring.inclusion ⋯) (IsLocalRing.maximalI …
    p : Polynomial (Subtype fun x => Membership.mem R.toSubring x)
    hp : ∀ (i : Nat), Membership.mem (IsLocalRing.maximalIdeal (Subtype fun x => M …
    hp' : Eq ((Polynomial.aeval x) p) ↑1
    this✝ : Invertible x
    this : Eq ((Polynomial.aeval (Invertible.invOf x)) (HSub.hSub p 1).reverse) 0
    ⊢ IsIntegral (Subtype fun x => Membership.mem R.toSubring x) (Inv.inv x)
  -/
  rw [invOf_eq_right_inv (mul_inv_cancel₀ hx0)] at this
  have H : IsUnit ((p - 1).coeff 0) := by
    by_contra h
    simpa using sub_mem (hp 0) h
  /-
    case intro.intro
    K : Type u_3
    inst✝ : Field K
    R : LocalSubring K
    hR : IsMax R
    x : K
    hx : Not (Membership.mem R.toSubring x)
    hx0 : Ne x 0
    S : Subalgebra (Subtype fun x => Membership.mem R.toSubring x) K := Algebra.ad …
    this✝² : LT.lt R.toSubring S.toSubring
    this✝¹ : Membership.mem (Ideal.map (Subring.inclusion ⋯) (IsLocalRing.maximalI …
    p : Polynomial (Subtype fun x => Membership.mem R.toSubring x)
    hp : ∀ (i : Nat), Membership.mem (IsLocalRing.maximalIdeal (Subtype fun x => M …
    hp' : Eq ((Polynomial.aeval x) p) ↑1
    this✝ : Invertible x
    this : Eq ((Polynomial.aeval (Inv.inv x)) (HSub.hSub p 1).reverse) 0
    H : IsUnit ((HSub.hSub p 1).coeff 0)
    ⊢ IsIntegral (Subtype fun x => Membership.mem R.toSubring x) (Inv.inv x)
  -/
  refine ⟨.C (H.unit⁻¹).1 * (p - 1).reverse, ?_, ?_⟩
  · have : (p - 1).natTrailingDegree = 0 := by
      simp only [Polynomial.natTrailingDegree_eq_zero,
        Polynomial.coeff_sub, Polynomial.coeff_one_zero, ne_eq, sub_eq_zero]
      exact .inr fun h ↦ (IsLocalRing.not_mem_maximalIdeal.mpr isUnit_one (h ▸ hp 0))
    rw [Polynomial.Monic.def, Polynomial.leadingCoeff_mul', Polynomial.reverse_leadingCoeff,
      Polynomial.trailingCoeff, this]
      /-
        case intro.intro.refine_1
        K : Type u_3
        inst✝ : Field K
        R : LocalSubring K
        hR : IsMax R
        x : K
        hx : Not (Membership.mem R.toSubring x)
        hx0 : Ne x 0
        S : Subalgebra (Subtype fun x => Membership.mem R.toSubring x) K := Algebra.ad …
        this✝³ : LT.lt R.toSubring S.toSubring
        this✝² : Membership.mem (Ideal.map (Subring.inclusion ⋯) (IsLocalRing.maximalI …
        p : Polynomial (Subtype fun x => Membership.mem R.toSubring x)
        hp : ∀ (i : Nat), Membership.mem (IsLocalRing.maximalIdeal (Subtype fun x => M …
        hp' : Eq ((Polynomial.aeval x) p) ↑1
        this✝¹ : Invertible x
        this✝ : Eq ((Polynomial.aeval (Inv.inv x)) (HSub.hSub p 1).reverse) 0
        H : IsUnit ((HSub.hSub p 1).coeff 0)
        this : Eq (HSub.hSub p 1).natTrailingDegree 0
        ⊢ Eq (HMul.hMul (Polynomial.C ↑(Inv.inv H.unit)).leadingCoeff ((HSub.hSub p 1) …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.refine_1
        K : Type u_3
        inst✝ : Field K
        R : LocalSubring K
        hR : IsMax R
        x : K
        hx : Not (Membership.mem R.toSubring x)
        hx0 : Ne x 0
        S : Subalgebra (Subtype fun x => Membership.mem R.toSubring x) K := Algebra.ad …
        this✝³ : LT.lt R.toSubring S.toSubring
        this✝² : Membership.mem (Ideal.map (Subring.inclusion ⋯) (IsLocalRing.maximalI …
        p : Polynomial (Subtype fun x => Membership.mem R.toSubring x)
        hp : ∀ (i : Nat), Membership.mem (IsLocalRing.maximalIdeal (Subtype fun x => M …
        hp' : Eq ((Polynomial.aeval x) p) ↑1
        this✝¹ : Invertible x
        this✝ : Eq ((Polynomial.aeval (Inv.inv x)) (HSub.hSub p 1).reverse) 0
        H : IsUnit ((HSub.hSub p 1).coeff 0)
        this : Eq (HSub.hSub p 1).natTrailingDegree 0
        ⊢ Ne (HMul.hMul (Polynomial.C ↑(Inv.inv H.unit)).leadingCoeff (HSub.hSub p 1). …
      -/
    · have : p - 1 ≠ 0 := fun e ↦ by simp [e] at H
      /-
        case intro.intro.refine_1
        K : Type u_3
        inst✝ : Field K
        R : LocalSubring K
        hR : IsMax R
        x : K
        hx : Not (Membership.mem R.toSubring x)
        hx0 : Ne x 0
        S : Subalgebra (Subtype fun x => Membership.mem R.toSubring x) K := Algebra.ad …
        this✝⁴ : LT.lt R.toSubring S.toSubring
        this✝³ : Membership.mem (Ideal.map (Subring.inclusion ⋯) (IsLocalRing.maximalI …
        p : Polynomial (Subtype fun x => Membership.mem R.toSubring x)
        hp : ∀ (i : Nat), Membership.mem (IsLocalRing.maximalIdeal (Subtype fun x => M …
        hp' : Eq ((Polynomial.aeval x) p) ↑1
        this✝² : Invertible x
        this✝¹ : Eq ((Polynomial.aeval (Inv.inv x)) (HSub.hSub p 1).reverse) 0
        H : IsUnit ((HSub.hSub p 1).coeff 0)
        this✝ : Eq (HSub.hSub p 1).natTrailingDegree 0
        this : Ne (HSub.hSub p 1) 0
        ⊢ Ne (HMul.hMul (Polynomial.C ↑(Inv.inv H.unit)).leadingCoeff (HSub.hSub p 1). …
      -/
      simpa
      /-
        🎉 no goals
      -/
    /-
      case intro.intro.refine_2
      K : Type u_3
      inst✝ : Field K
      R : LocalSubring K
      hR : IsMax R
      x : K
      hx : Not (Membership.mem R.toSubring x)
      hx0 : Ne x 0
      S : Subalgebra (Subtype fun x => Membership.mem R.toSubring x) K := Algebra.ad …
      this✝² : LT.lt R.toSubring S.toSubring
      this✝¹ : Membership.mem (Ideal.map (Subring.inclusion ⋯) (IsLocalRing.maximalI …
      p : Polynomial (Subtype fun x => Membership.mem R.toSubring x)
      hp : ∀ (i : Nat), Membership.mem (IsLocalRing.maximalIdeal (Subtype fun x => M …
      hp' : Eq ((Polynomial.aeval x) p) ↑1
      this✝ : Invertible x
      this : Eq ((Polynomial.aeval (Inv.inv x)) (HSub.hSub p 1).reverse) 0
      H : IsUnit ((HSub.hSub p 1).coeff 0)
      ⊢ Eq (Polynomial.eval₂ (algebraMap (Subtype fun x => Membership.mem R.toSubrin …
    -/
  · simp [← Polynomial.aeval_def, this]
    /-
      🎉 no goals
    -/


/-- A local subring is maximal with respect to the domination order
  if and only if it is a valuation ring. -/
lemma LocalSubring.isMax_iff {A : LocalSubring K} :
    IsMax A ↔ ∃ B : ValuationSubring K, B.toLocalSubring = A :=
  ⟨exists_valuationRing_of_isMax, fun ⟨B, e⟩ ↦ e ▸ B.isMax_toLocalSubring⟩


@[stacks 00IA]
lemma LocalSubring.exists_le_valuationSubring (A : LocalSubring K) :
    ∃ B : ValuationSubring K, A ≤ B.toLocalSubring := by
  suffices ∃ B, A ≤ B ∧ IsMax B by
    obtain ⟨B, hB, hB'⟩ := this
    obtain ⟨B, rfl⟩ := B.exists_valuationRing_of_isMax hB'
    exact ⟨B, hB⟩
  /-
    K : Type u_3
    inst✝ : Field K
    A : LocalSubring K
    ⊢ Exists fun B => And (LE.le A B) (IsMax B)
  -/
  refine zorn_le_nonempty_Ici₀ _ ?_ _ le_rfl
  /-
    K : Type u_3
    inst✝ : Field K
    A : LocalSubring K
    ⊢ ∀ (c : Set (LocalSubring K)), HasSubset.Subset c (Set.Ici A) → IsChain (fun  …
  -/
  intro s hs H y hys
  /-
    K : Type u_3
    inst✝ : Field K
    A : LocalSubring K
    s : Set (LocalSubring K)
    hs : HasSubset.Subset s (Set.Ici A)
    H : IsChain (fun x1 x2 => LE.le x1 x2) s
    y : LocalSubring K
    hys : Membership.mem s y
    ⊢ Exists fun ub => ∀ (z : LocalSubring K), Membership.mem s z → LE.le z ub
  -/
  have inst : Nonempty s := ⟨⟨y, hys⟩⟩
  /-
    K : Type u_3
    inst✝ : Field K
    A : LocalSubring K
    s : Set (LocalSubring K)
    hs : HasSubset.Subset s (Set.Ici A)
    H : IsChain (fun x1 x2 => LE.le x1 x2) s
    y : LocalSubring K
    hys : Membership.mem s y
    inst : Nonempty ↑s
    ⊢ Exists fun ub => ∀ (z : LocalSubring K), Membership.mem s z → LE.le z ub
  -/
  have hdir := H.directed.mono_comp _ LocalSubring.toSubring_mono
  /-
    K : Type u_3
    inst✝ : Field K
    A : LocalSubring K
    s : Set (LocalSubring K)
    hs : HasSubset.Subset s (Set.Ici A)
    H : IsChain (fun x1 x2 => LE.le x1 x2) s
    y : LocalSubring K
    hys : Membership.mem s y
    inst : Nonempty ↑s
    hdir : Directed LE.le (Function.comp LocalSubring.toSubring fun x => ↑x)
    ⊢ Exists fun ub => ∀ (z : LocalSubring K), Membership.mem s z → LE.le z ub
  -/
  refine ⟨@LocalSubring.mk _ _ (⨆ i : s, i.1.toSubring) ⟨?_⟩, ?_⟩
    /-
      case refine_1
      K : Type u_3
      inst✝ : Field K
      A : LocalSubring K
      s : Set (LocalSubring K)
      hs : HasSubset.Subset s (Set.Ici A)
      H : IsChain (fun x1 x2 => LE.le x1 x2) s
      y : LocalSubring K
      hys : Membership.mem s y
      inst : Nonempty ↑s
      hdir : Directed LE.le (Function.comp LocalSubring.toSubring fun x => ↑x)
      ⊢ ∀ {a b : Subtype fun x => Membership.mem (iSup fun i => (↑i).toSubring) x},  …
    -/
  · intro ⟨a, ha⟩ ⟨b, hb⟩ e
    /-
      case refine_1
      K : Type u_3
      inst✝ : Field K
      A : LocalSubring K
      s : Set (LocalSubring K)
      hs : HasSubset.Subset s (Set.Ici A)
      H : IsChain (fun x1 x2 => LE.le x1 x2) s
      y : LocalSubring K
      hys : Membership.mem s y
      inst : Nonempty ↑s
      hdir : Directed LE.le (Function.comp LocalSubring.toSubring fun x => ↑x)
      a : K
      ha : Membership.mem (iSup fun i => (↑i).toSubring) a
      b : K
      hb : Membership.mem (iSup fun i => (↑i).toSubring) b
      e : Eq (HAdd.hAdd ⟨a, ha⟩ ⟨b, hb⟩) 1
      ⊢ Or (IsUnit ⟨a, ha⟩) (IsUnit ⟨b, hb⟩)
    -/
    obtain ⟨A, haA : a ∈ A.1.toSubring⟩ := (Subring.mem_iSup_of_directed hdir).mp ha
    /-
      case refine_1.intro
      K : Type u_3
      inst✝ : Field K
      A✝ : LocalSubring K
      s : Set (LocalSubring K)
      hs : HasSubset.Subset s (Set.Ici A✝)
      H : IsChain (fun x1 x2 => LE.le x1 x2) s
      y : LocalSubring K
      hys : Membership.mem s y
      inst : Nonempty ↑s
      hdir : Directed LE.le (Function.comp LocalSubring.toSubring fun x => ↑x)
      a : K
      ha : Membership.mem (iSup fun i => (↑i).toSubring) a
      b : K
      hb : Membership.mem (iSup fun i => (↑i).toSubring) b
      e : Eq (HAdd.hAdd ⟨a, ha⟩ ⟨b, hb⟩) 1
      A : Subtype fun a => Membership.mem s a
      haA : Membership.mem (↑A).toSubring a
      ⊢ Or (IsUnit ⟨a, ha⟩) (IsUnit ⟨b, hb⟩)
    -/
    obtain ⟨B, hbB : b ∈ B.1.toSubring⟩ := (Subring.mem_iSup_of_directed hdir).mp hb
    /-
      case refine_1.intro.intro
      K : Type u_3
      inst✝ : Field K
      A✝ : LocalSubring K
      s : Set (LocalSubring K)
      hs : HasSubset.Subset s (Set.Ici A✝)
      H : IsChain (fun x1 x2 => LE.le x1 x2) s
      y : LocalSubring K
      hys : Membership.mem s y
      inst : Nonempty ↑s
      hdir : Directed LE.le (Function.comp LocalSubring.toSubring fun x => ↑x)
      a : K
      ha : Membership.mem (iSup fun i => (↑i).toSubring) a
      b : K
      hb : Membership.mem (iSup fun i => (↑i).toSubring) b
      e : Eq (HAdd.hAdd ⟨a, ha⟩ ⟨b, hb⟩) 1
      A : Subtype fun a => Membership.mem s a
      haA : Membership.mem (↑A).toSubring a
      B : Subtype fun a => Membership.mem s a
      hbB : Membership.mem (↑B).toSubring b
      ⊢ Or (IsUnit ⟨a, ha⟩) (IsUnit ⟨b, hb⟩)
    -/
    obtain ⟨C, hCA, hCB⟩ := hdir A B
    /-
      case refine_1.intro.intro.intro.intro
      K : Type u_3
      inst✝ : Field K
      A✝ : LocalSubring K
      s : Set (LocalSubring K)
      hs : HasSubset.Subset s (Set.Ici A✝)
      H : IsChain (fun x1 x2 => LE.le x1 x2) s
      y : LocalSubring K
      hys : Membership.mem s y
      inst : Nonempty ↑s
      hdir : Directed LE.le (Function.comp LocalSubring.toSubring fun x => ↑x)
      a : K
      ha : Membership.mem (iSup fun i => (↑i).toSubring) a
      b : K
      hb : Membership.mem (iSup fun i => (↑i).toSubring) b
      e : Eq (HAdd.hAdd ⟨a, ha⟩ ⟨b, hb⟩) 1
      A : Subtype fun a => Membership.mem s a
      haA : Membership.mem (↑A).toSubring a
      B : Subtype fun a => Membership.mem s a
      hbB : Membership.mem (↑B).toSubring b
      C : Subtype fun a => Membership.mem s a
      hCA : LE.le (Function.comp LocalSubring.toSubring (fun x => ↑x) A) (Function.c …
      hCB : LE.le (Function.comp LocalSubring.toSubring (fun x => ↑x) B) (Function.c …
      ⊢ Or (IsUnit ⟨a, ha⟩) (IsUnit ⟨b, hb⟩)
    -/
    refine (C.1.2.2 (a := ⟨a, hCA haA⟩) (b := ⟨b, hCB hbB⟩) (Subtype.ext congr(($e).1))).imp ?_ ?_
      /-
        case refine_1.intro.intro.intro.intro.refine_1
        K : Type u_3
        inst✝ : Field K
        A✝ : LocalSubring K
        s : Set (LocalSubring K)
        hs : HasSubset.Subset s (Set.Ici A✝)
        H : IsChain (fun x1 x2 => LE.le x1 x2) s
        y : LocalSubring K
        hys : Membership.mem s y
        inst : Nonempty ↑s
        hdir : Directed LE.le (Function.comp LocalSubring.toSubring fun x => ↑x)
        a : K
        ha : Membership.mem (iSup fun i => (↑i).toSubring) a
        b : K
        hb : Membership.mem (iSup fun i => (↑i).toSubring) b
        e : Eq (HAdd.hAdd ⟨a, ha⟩ ⟨b, hb⟩) 1
        A : Subtype fun a => Membership.mem s a
        haA : Membership.mem (↑A).toSubring a
        B : Subtype fun a => Membership.mem s a
        hbB : Membership.mem (↑B).toSubring b
        C : Subtype fun a => Membership.mem s a
        hCA : LE.le (Function.comp LocalSubring.toSubring (fun x => ↑x) A) (Function.c …
        hCB : LE.le (Function.comp LocalSubring.toSubring (fun x => ↑x) B) (Function.c …
        ⊢ IsUnit ⟨a, ⋯⟩ → IsUnit ⟨a, ha⟩
      -/
    · exact fun h ↦ h.map (Subring.inclusion (le_iSup (fun i : s ↦ i.1.toSubring) C))
      /-
        🎉 no goals
      -/
      /-
        case refine_1.intro.intro.intro.intro.refine_2
        K : Type u_3
        inst✝ : Field K
        A✝ : LocalSubring K
        s : Set (LocalSubring K)
        hs : HasSubset.Subset s (Set.Ici A✝)
        H : IsChain (fun x1 x2 => LE.le x1 x2) s
        y : LocalSubring K
        hys : Membership.mem s y
        inst : Nonempty ↑s
        hdir : Directed LE.le (Function.comp LocalSubring.toSubring fun x => ↑x)
        a : K
        ha : Membership.mem (iSup fun i => (↑i).toSubring) a
        b : K
        hb : Membership.mem (iSup fun i => (↑i).toSubring) b
        e : Eq (HAdd.hAdd ⟨a, ha⟩ ⟨b, hb⟩) 1
        A : Subtype fun a => Membership.mem s a
        haA : Membership.mem (↑A).toSubring a
        B : Subtype fun a => Membership.mem s a
        hbB : Membership.mem (↑B).toSubring b
        C : Subtype fun a => Membership.mem s a
        hCA : LE.le (Function.comp LocalSubring.toSubring (fun x => ↑x) A) (Function.c …
        hCB : LE.le (Function.comp LocalSubring.toSubring (fun x => ↑x) B) (Function.c …
        ⊢ IsUnit ⟨b, ⋯⟩ → IsUnit ⟨b, hb⟩
      -/
    · exact fun h ↦ h.map (Subring.inclusion (le_iSup (fun i : s ↦ i.1.toSubring) C))
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      K : Type u_3
      inst✝ : Field K
      A : LocalSubring K
      s : Set (LocalSubring K)
      hs : HasSubset.Subset s (Set.Ici A)
      H : IsChain (fun x1 x2 => LE.le x1 x2) s
      y : LocalSubring K
      hys : Membership.mem s y
      inst : Nonempty ↑s
      hdir : Directed LE.le (Function.comp LocalSubring.toSubring fun x => ↑x)
      ⊢ ∀ (z : LocalSubring K), Membership.mem s z → LE.le z (LocalSubring.mk (iSup  …
    -/
  · intro A hA
    /-
      case refine_2
      K : Type u_3
      inst✝ : Field K
      A✝ : LocalSubring K
      s : Set (LocalSubring K)
      hs : HasSubset.Subset s (Set.Ici A✝)
      H : IsChain (fun x1 x2 => LE.le x1 x2) s
      y : LocalSubring K
      hys : Membership.mem s y
      inst : Nonempty ↑s
      hdir : Directed LE.le (Function.comp LocalSubring.toSubring fun x => ↑x)
      A : LocalSubring K
      hA : Membership.mem s A
      ⊢ LE.le A (LocalSubring.mk (iSup fun i => (↑i).toSubring))
    -/
    refine ⟨le_iSup (fun i : s ↦ i.1.toSubring) ⟨A, hA⟩, ⟨?_⟩⟩
    /-
      case refine_2
      K : Type u_3
      inst✝ : Field K
      A✝ : LocalSubring K
      s : Set (LocalSubring K)
      hs : HasSubset.Subset s (Set.Ici A✝)
      H : IsChain (fun x1 x2 => LE.le x1 x2) s
      y : LocalSubring K
      hys : Membership.mem s y
      inst : Nonempty ↑s
      hdir : Directed LE.le (Function.comp LocalSubring.toSubring fun x => ↑x)
      A : LocalSubring K
      hA : Membership.mem s A
      ⊢ ∀ (a : Subtype fun x => Membership.mem A.toSubring x), IsUnit ((Subring.incl …
    -/
    rintro ⟨a, haA⟩ h
    /-
      case refine_2.mk
      K : Type u_3
      inst✝ : Field K
      A✝ : LocalSubring K
      s : Set (LocalSubring K)
      hs : HasSubset.Subset s (Set.Ici A✝)
      H : IsChain (fun x1 x2 => LE.le x1 x2) s
      y : LocalSubring K
      hys : Membership.mem s y
      inst : Nonempty ↑s
      hdir : Directed LE.le (Function.comp LocalSubring.toSubring fun x => ↑x)
      A : LocalSubring K
      hA : Membership.mem s A
      a : K
      haA : Membership.mem A.toSubring a
      h : IsUnit ((Subring.inclusion ⋯) ⟨a, haA⟩)
      ⊢ IsUnit ⟨a, haA⟩
    -/
    obtain ⟨⟨b, hb⟩, e⟩ := isUnit_iff_exists_inv.mp h
    /-
      case refine_2.mk.intro.mk
      K : Type u_3
      inst✝ : Field K
      A✝ : LocalSubring K
      s : Set (LocalSubring K)
      hs : HasSubset.Subset s (Set.Ici A✝)
      H : IsChain (fun x1 x2 => LE.le x1 x2) s
      y : LocalSubring K
      hys : Membership.mem s y
      inst : Nonempty ↑s
      hdir : Directed LE.le (Function.comp LocalSubring.toSubring fun x => ↑x)
      A : LocalSubring K
      hA : Membership.mem s A
      a : K
      haA : Membership.mem A.toSubring a
      h : IsUnit ((Subring.inclusion ⋯) ⟨a, haA⟩)
      b : K
      hb : Membership.mem (LocalSubring.mk (iSup fun i => (↑i).toSubring)).toSubring b
      e : Eq (HMul.hMul ((Subring.inclusion ⋯) ⟨a, haA⟩) ⟨b, hb⟩) 1
      ⊢ IsUnit ⟨a, haA⟩
    -/
    obtain ⟨B, hbB : b ∈ B.1.toSubring⟩ := (Subring.mem_iSup_of_directed hdir).mp hb
    /-
      case refine_2.mk.intro.mk.intro
      K : Type u_3
      inst✝ : Field K
      A✝ : LocalSubring K
      s : Set (LocalSubring K)
      hs : HasSubset.Subset s (Set.Ici A✝)
      H : IsChain (fun x1 x2 => LE.le x1 x2) s
      y : LocalSubring K
      hys : Membership.mem s y
      inst : Nonempty ↑s
      hdir : Directed LE.le (Function.comp LocalSubring.toSubring fun x => ↑x)
      A : LocalSubring K
      hA : Membership.mem s A
      a : K
      haA : Membership.mem A.toSubring a
      h : IsUnit ((Subring.inclusion ⋯) ⟨a, haA⟩)
      b : K
      hb : Membership.mem (LocalSubring.mk (iSup fun i => (↑i).toSubring)).toSubring b
      e : Eq (HMul.hMul ((Subring.inclusion ⋯) ⟨a, haA⟩) ⟨b, hb⟩) 1
      B : Subtype fun a => Membership.mem s a
      hbB : Membership.mem (↑B).toSubring b
      ⊢ IsUnit ⟨a, haA⟩
    -/
    obtain ⟨C, hCA, hCB⟩ := H.directed ⟨A, hA⟩ B
    /-
      case refine_2.mk.intro.mk.intro.intro.intro
      K : Type u_3
      inst✝ : Field K
      A✝ : LocalSubring K
      s : Set (LocalSubring K)
      hs : HasSubset.Subset s (Set.Ici A✝)
      H : IsChain (fun x1 x2 => LE.le x1 x2) s
      y : LocalSubring K
      hys : Membership.mem s y
      inst : Nonempty ↑s
      hdir : Directed LE.le (Function.comp LocalSubring.toSubring fun x => ↑x)
      A : LocalSubring K
      hA : Membership.mem s A
      a : K
      haA : Membership.mem A.toSubring a
      h : IsUnit ((Subring.inclusion ⋯) ⟨a, haA⟩)
      b : K
      hb : Membership.mem (LocalSubring.mk (iSup fun i => (↑i).toSubring)).toSubring b
      e : Eq (HMul.hMul ((Subring.inclusion ⋯) ⟨a, haA⟩) ⟨b, hb⟩) 1
      B : Subtype fun a => Membership.mem s a
      hbB : Membership.mem (↑B).toSubring b
      C : Subtype fun a => Membership.mem s a
      hCA : LE.le ((fun x => ↑x) ⟨A, hA⟩) ((fun x => ↑x) C)
      hCB : LE.le ((fun x => ↑x) B) ((fun x => ↑x) C)
      ⊢ IsUnit ⟨a, haA⟩
    -/
    apply hCA.2.1
    /-
      case refine_2.mk.intro.mk.intro.intro.intro.a
      K : Type u_3
      inst✝ : Field K
      A✝ : LocalSubring K
      s : Set (LocalSubring K)
      hs : HasSubset.Subset s (Set.Ici A✝)
      H : IsChain (fun x1 x2 => LE.le x1 x2) s
      y : LocalSubring K
      hys : Membership.mem s y
      inst : Nonempty ↑s
      hdir : Directed LE.le (Function.comp LocalSubring.toSubring fun x => ↑x)
      A : LocalSubring K
      hA : Membership.mem s A
      a : K
      haA : Membership.mem A.toSubring a
      h : IsUnit ((Subring.inclusion ⋯) ⟨a, haA⟩)
      b : K
      hb : Membership.mem (LocalSubring.mk (iSup fun i => (↑i).toSubring)).toSubring b
      e : Eq (HMul.hMul ((Subring.inclusion ⋯) ⟨a, haA⟩) ⟨b, hb⟩) 1
      B : Subtype fun a => Membership.mem s a
      hbB : Membership.mem (↑B).toSubring b
      C : Subtype fun a => Membership.mem s a
      hCA : LE.le ((fun x => ↑x) ⟨A, hA⟩) ((fun x => ↑x) C)
      hCB : LE.le ((fun x => ↑x) B) ((fun x => ↑x) C)
      ⊢ IsUnit ((Subring.inclusion ⋯) ⟨a, haA⟩)
    -/
    exact isUnit_iff_exists_inv.mpr ⟨⟨b, hCB.1 hbB⟩, Subtype.ext congr(($e).1)⟩
    /-
      🎉 no goals
    -/


lemma bijective_rangeRestrict_comp_of_valuationRing [IsDomain R] [ValuationRing R]
    [IsLocalRing S] [Algebra R K] [IsFractionRing R K]
    (f : R →+* S) (g : S →+* K) (h : g.comp f = algebraMap R K) [IsLocalHom f] :
    Function.Bijective (g.rangeRestrict.comp f) := by
  /-
    R : Type u_1
    S : Type u_2
    K : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Field K
    inst✝⁵ : IsDomain R
    inst✝⁴ : ValuationRing R
    inst✝³ : IsLocalRing S
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    f : RingHom R S
    g : RingHom S K
    h : Eq (g.comp f) (algebraMap R K)
    inst✝ : IsLocalHom f
    ⊢ Function.Bijective ⇑(g.rangeRestrict.comp f)
  -/
  refine ⟨?_, ?_⟩
    /-
      case refine_1
      R : Type u_1
      S : Type u_2
      K : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing S
      inst✝⁶ : Field K
      inst✝⁵ : IsDomain R
      inst✝⁴ : ValuationRing R
      inst✝³ : IsLocalRing S
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      f : RingHom R S
      g : RingHom S K
      h : Eq (g.comp f) (algebraMap R K)
      inst✝ : IsLocalHom f
      ⊢ Function.Injective ⇑(g.rangeRestrict.comp f)
    -/
  · exact .of_comp (f := Subtype.val) (by convert (IsFractionRing.injective R K); rw [← h]; rfl)
    /-
      🎉 no goals
    -/
  · let V : ValuationSubring K :=
      ⟨(algebraMap R K).range, ValuationRing.isInteger_or_isInteger R⟩
    suffices LocalSubring.range g ≤ V.toLocalSubring by
      rintro ⟨_, x, rfl⟩
      obtain ⟨y, hy⟩ := this.1 ⟨x, rfl⟩
      exact ⟨y, Subtype.ext (by simpa [← h] using hy)⟩
    /-
      case refine_2
      R : Type u_1
      S : Type u_2
      K : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing S
      inst✝⁶ : Field K
      inst✝⁵ : IsDomain R
      inst✝⁴ : ValuationRing R
      inst✝³ : IsLocalRing S
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      f : RingHom R S
      g : RingHom S K
      h : Eq (g.comp f) (algebraMap R K)
      inst✝ : IsLocalHom f
      V : ValuationSubring K := { toSubring := (algebraMap R K).range, mem_or_inv_me …
      ⊢ LE.le (LocalSubring.range g) V.toLocalSubring
    -/
    apply V.isMax_toLocalSubring
    /-
      case refine_2.a
      R : Type u_1
      S : Type u_2
      K : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing S
      inst✝⁶ : Field K
      inst✝⁵ : IsDomain R
      inst✝⁴ : ValuationRing R
      inst✝³ : IsLocalRing S
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      f : RingHom R S
      g : RingHom S K
      h : Eq (g.comp f) (algebraMap R K)
      inst✝ : IsLocalHom f
      V : ValuationSubring K := { toSubring := (algebraMap R K).range, mem_or_inv_me …
      ⊢ LE.le V.toLocalSubring (LocalSubring.range g)
    -/
    have H : (algebraMap R K).range ≤ g.range := fun x ⟨a, ha⟩ ↦ ⟨f a, by simp [← ha, ← h]⟩
    /-
      case refine_2.a
      R : Type u_1
      S : Type u_2
      K : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing S
      inst✝⁶ : Field K
      inst✝⁵ : IsDomain R
      inst✝⁴ : ValuationRing R
      inst✝³ : IsLocalRing S
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      f : RingHom R S
      g : RingHom S K
      h : Eq (g.comp f) (algebraMap R K)
      inst✝ : IsLocalHom f
      V : ValuationSubring K := { toSubring := (algebraMap R K).range, mem_or_inv_me …
      H : LE.le (algebraMap R K).range g.range
      ⊢ LE.le V.toLocalSubring (LocalSubring.range g)
    -/
    refine ⟨H, ⟨?_⟩⟩
    /-
      case refine_2.a
      R : Type u_1
      S : Type u_2
      K : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing S
      inst✝⁶ : Field K
      inst✝⁵ : IsDomain R
      inst✝⁴ : ValuationRing R
      inst✝³ : IsLocalRing S
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      f : RingHom R S
      g : RingHom S K
      h : Eq (g.comp f) (algebraMap R K)
      inst✝ : IsLocalHom f
      V : ValuationSubring K := { toSubring := (algebraMap R K).range, mem_or_inv_me …
      H : LE.le (algebraMap R K).range g.range
      ⊢ ∀ (a : Subtype fun x => Membership.mem V.toLocalSubring.toSubring x), IsUnit …
    -/
    rintro ⟨_, a, rfl⟩ (ha : IsUnit (M := g.range) ⟨algebraMap R K a, _⟩)
    /-
      case refine_2.a.mk.intro
      R : Type u_1
      S : Type u_2
      K : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing S
      inst✝⁶ : Field K
      inst✝⁵ : IsDomain R
      inst✝⁴ : ValuationRing R
      inst✝³ : IsLocalRing S
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      f : RingHom R S
      g : RingHom S K
      h : Eq (g.comp f) (algebraMap R K)
      inst✝ : IsLocalHom f
      V : ValuationSubring K := { toSubring := (algebraMap R K).range, mem_or_inv_me …
      H : LE.le (algebraMap R K).range g.range
      a : R
      ha : IsUnit ⟨(algebraMap R K) a, ⋯⟩
      ⊢ IsUnit ⟨(algebraMap R K) a, ⋯⟩
    -/
    suffices IsUnit a from this.map (algebraMap R K).rangeRestrict
    /-
      case refine_2.a.mk.intro
      R : Type u_1
      S : Type u_2
      K : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing S
      inst✝⁶ : Field K
      inst✝⁵ : IsDomain R
      inst✝⁴ : ValuationRing R
      inst✝³ : IsLocalRing S
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      f : RingHom R S
      g : RingHom S K
      h : Eq (g.comp f) (algebraMap R K)
      inst✝ : IsLocalHom f
      V : ValuationSubring K := { toSubring := (algebraMap R K).range, mem_or_inv_me …
      H : LE.le (algebraMap R K).range g.range
      a : R
      ha : IsUnit ⟨(algebraMap R K) a, ⋯⟩
      ⊢ IsUnit a
    -/
    apply IsUnit.of_map f
    /-
      case refine_2.a.mk.intro.h
      R : Type u_1
      S : Type u_2
      K : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing S
      inst✝⁶ : Field K
      inst✝⁵ : IsDomain R
      inst✝⁴ : ValuationRing R
      inst✝³ : IsLocalRing S
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      f : RingHom R S
      g : RingHom S K
      h : Eq (g.comp f) (algebraMap R K)
      inst✝ : IsLocalHom f
      V : ValuationSubring K := { toSubring := (algebraMap R K).range, mem_or_inv_me …
      H : LE.le (algebraMap R K).range g.range
      a : R
      ha : IsUnit ⟨(algebraMap R K) a, ⋯⟩
      ⊢ IsUnit (f a)
    -/
    apply (IsLocalHom.of_surjective g.rangeRestrict g.rangeRestrict_surjective).1
    /-
      case refine_2.a.mk.intro.h.a
      R : Type u_1
      S : Type u_2
      K : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing S
      inst✝⁶ : Field K
      inst✝⁵ : IsDomain R
      inst✝⁴ : ValuationRing R
      inst✝³ : IsLocalRing S
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      f : RingHom R S
      g : RingHom S K
      h : Eq (g.comp f) (algebraMap R K)
      inst✝ : IsLocalHom f
      V : ValuationSubring K := { toSubring := (algebraMap R K).range, mem_or_inv_me …
      H : LE.le (algebraMap R K).range g.range
      a : R
      ha : IsUnit ⟨(algebraMap R K) a, ⋯⟩
      ⊢ IsUnit (g.rangeRestrict (f a))
    -/
    convert ha
    /-
      case h.e'_3.h.e'_3
      R : Type u_1
      S : Type u_2
      K : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing S
      inst✝⁶ : Field K
      inst✝⁵ : IsDomain R
      inst✝⁴ : ValuationRing R
      inst✝³ : IsLocalRing S
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      f : RingHom R S
      g : RingHom S K
      h : Eq (g.comp f) (algebraMap R K)
      inst✝ : IsLocalHom f
      V : ValuationSubring K := { toSubring := (algebraMap R K).range, mem_or_inv_me …
      H : LE.le (algebraMap R K).range g.range
      a : R
      ha : IsUnit ⟨(algebraMap R K) a, ⋯⟩
      ⊢ Eq (↑(g.rangeRestrict (f a))) ((algebraMap R K) a)
    -/
    simp [← h]
    /-
      🎉 no goals
    -/


lemma IsLocalRing.exists_factor_valuationRing [IsLocalRing R] (f : R →+* K) :
    ∃ (A : ValuationSubring K) (h : _), IsLocalHom (f.codRestrict A.toSubring h) := by
  /-
    R : Type u_1
    K : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Field K
    inst✝ : IsLocalRing R
    f : RingHom R K
    ⊢ Exists fun A => Exists fun h => IsLocalHom (f.codRestrict A.toSubring h)
  -/
  obtain ⟨B, hB⟩  := (LocalSubring.range f).exists_le_valuationSubring
  /-
    case intro
    R : Type u_1
    K : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Field K
    inst✝ : IsLocalRing R
    f : RingHom R K
    B : ValuationSubring K
    hB : LE.le (LocalSubring.range f) B.toLocalSubring
    ⊢ Exists fun A => Exists fun h => IsLocalHom (f.codRestrict A.toSubring h)
  -/
  refine ⟨B, fun x ↦ hB.1 ⟨x, rfl⟩, ?_⟩
  exact @RingHom.isLocalHom_comp _ _ _ _ _ _ _ _
    hB.2 (.of_surjective _ f.rangeRestrict_surjective)

