/-- The norm of a maximal ideal as an element of `ℝ≥0` is `> 1`  -/
lemma one_lt_norm : 1 < (absNorm v.asIdeal : NNReal) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    v : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
    ⊢ LT.lt 1 ↑(Ideal.absNorm v.asIdeal)
  -/
  norm_cast
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    v : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
    ⊢ LT.lt 1 (Ideal.absNorm v.asIdeal)
  -/
  by_contra! h
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    v : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
    h : LE.le (Ideal.absNorm v.asIdeal) 1
    ⊢ False
  -/
  apply IsPrime.ne_top v.isPrime
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    v : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
    h : LE.le (Ideal.absNorm v.asIdeal) 1
    ⊢ Eq v.asIdeal Top.top
  -/
  rw [← absNorm_eq_one_iff]
  have : 0 < absNorm v.asIdeal := by
    rw [Nat.pos_iff_ne_zero, absNorm_ne_zero_iff]
    exact (v.asIdeal.fintypeQuotientOfFreeOfNeBot v.ne_bot).finite
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    v : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
    h : LE.le (Ideal.absNorm v.asIdeal) 1
    this : LT.lt 0 (Ideal.absNorm v.asIdeal)
    ⊢ Eq (Ideal.absNorm v.asIdeal) 1
  -/
  omega
  /-
    🎉 no goals
  -/


private lemma norm_ne_zero : (absNorm v.asIdeal : NNReal) ≠ 0 := ne_zero_of_lt (one_lt_norm v)


/-- The `v`-adic absolute value on `K` defined as the norm of `v` raised to negative `v`-adic
valuation.-/
noncomputable def vadicAbv : AbsoluteValue K ℝ where
  toFun x := toNNReal (norm_ne_zero v) (v.valuation x)
                     /-
                       K : Type u_1
                       inst✝¹ : Field K
                       inst✝ : NumberField K
                       v : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
                       x✝¹ x✝ : K
                       ⊢ Eq ((fun x => ↑((WithZeroMulInt.toNNReal ⋯) (v.valuation x))) (HMul.hMul x✝¹ …
                     -/
  map_mul' _ _ := by simp only [_root_.map_mul, NNReal.coe_mul]
                     /-
                       🎉 no goals
                     -/
  nonneg' _ := NNReal.zero_le_coe
                   /-
                     K : Type u_1
                     inst✝¹ : Field K
                     inst✝ : NumberField K
                     v : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
                     x✝ : K
                     ⊢ Iff (Eq ({ toFun := fun x => ↑((WithZeroMulInt.toNNReal ⋯) (v.valuation x)), …
                   -/
  eq_zero' _ := by simp only [NNReal.coe_eq_zero, map_eq_zero]
                   /-
                     🎉 no goals
                   -/
  add_le' x y := by
    -- the triangle inequality is implied by the ultrametric one
    apply le_trans _ <| max_le_add_of_nonneg (zero_le ((toNNReal (norm_ne_zero v)) (v.valuation x)))
      (zero_le ((toNNReal (norm_ne_zero v)) (v.valuation y)))
    /-
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      v : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
      x y : K
      ⊢ LE.le ((WithZeroMulInt.toNNReal ⋯) (v.valuation (HAdd.hAdd x y))) (Max.max ( …
    -/
    have h_mono := (toNNReal_strictMono (one_lt_norm v)).monotone
    /-
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      v : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
      x y : K
      h_mono : Monotone ⇑(WithZeroMulInt.toNNReal ⋯)
      ⊢ LE.le ((WithZeroMulInt.toNNReal ⋯) (v.valuation (HAdd.hAdd x y))) (Max.max ( …
    -/
    rw [← h_mono.map_max] --max goes inside withZeroMultIntToNNReal
    /-
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      v : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
      x y : K
      h_mono : Monotone ⇑(WithZeroMulInt.toNNReal ⋯)
      ⊢ LE.le ((WithZeroMulInt.toNNReal ⋯) (v.valuation (HAdd.hAdd x y))) ((WithZero …
    -/
    exact h_mono (v.valuation.map_add x y)
    /-
      🎉 no goals
    -/


theorem vadicAbv_def {x : K} : vadicAbv v x = toNNReal (norm_ne_zero v) (v.valuation x) := rfl


/-- The embedding of a number field inside its completion with respect to `v`. -/
def embedding : K →+* adicCompletion K v :=
  @UniformSpace.Completion.coeRingHom K _ v.adicValued.toUniformSpace _ _


noncomputable instance instRankOneValuedAdicCompletion :
    Valuation.RankOne (valuedAdicCompletion K v).v where
  hom := {
    toFun := toNNReal (norm_ne_zero v)
    map_zero' := rfl
    map_one' := rfl
    map_mul' := MonoidWithZeroHom.map_mul (toNNReal (norm_ne_zero v))
  }
  strictMono' := toNNReal_strictMono (one_lt_norm v)
  nontrivial' := by
    /-
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      v : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
      ⊢ Exists fun r => And (Ne (Valued.v r) 0) (Ne (Valued.v r) 1)
    -/
    rcases Submodule.exists_mem_ne_zero_of_ne_bot v.ne_bot with ⟨x, hx1, hx2⟩
    /-
      case intro.intro
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      v : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
      x : NumberField.RingOfIntegers K
      hx1 : Membership.mem v.asIdeal x
      hx2 : Ne x 0
      ⊢ Exists fun r => And (Ne (Valued.v r) 0) (Ne (Valued.v r) 1)
    -/
    use (x : K)
    /-
      case h
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      v : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
      x : NumberField.RingOfIntegers K
      hx1 : Membership.mem v.asIdeal x
      hx2 : Ne x 0
      ⊢ And (Ne (Valued.v (↑K ↑x)) 0) (Ne (Valued.v (↑K ↑x)) 1)
    -/
    rw [valuedAdicCompletion_eq_valuation' v (x : K)]
    /-
      case h
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      v : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
      x : NumberField.RingOfIntegers K
      hx1 : Membership.mem v.asIdeal x
      hx2 : Ne x 0
      ⊢ And (Ne (v.valuation ↑x) 0) (Ne (v.valuation ↑x) 1)
    -/
    constructor
      /-
        case h.left
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        v : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
        x : NumberField.RingOfIntegers K
        hx1 : Membership.mem v.asIdeal x
        hx2 : Ne x 0
        ⊢ Ne (v.valuation ↑x) 0
      -/
    · simpa only [ne_eq, map_eq_zero, NoZeroSMulDivisors.algebraMap_eq_zero_iff]
      /-
        🎉 no goals
      -/
      /-
        case h.right
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        v : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
        x : NumberField.RingOfIntegers K
        hx1 : Membership.mem v.asIdeal x
        hx2 : Ne x 0
        ⊢ Ne (v.valuation ↑x) 1
      -/
    · apply ne_of_lt
      /-
        case h.right.h
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        v : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
        x : NumberField.RingOfIntegers K
        hx1 : Membership.mem v.asIdeal x
        hx2 : Ne x 0
        ⊢ LT.lt (v.valuation ↑x) 1
      -/
      rw [valuation_eq_intValuationDef, intValuation_lt_one_iff_dvd]
      /-
        case h.right.h
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        v : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
        x : NumberField.RingOfIntegers K
        hx1 : Membership.mem v.asIdeal x
        hx2 : Ne x 0
        ⊢ Dvd.dvd v.asIdeal (Ideal.span (Singleton.singleton x))
      -/
      exact dvd_span_singleton.mpr hx1
      /-
        🎉 no goals
      -/


/-- The `v`-adic completion of `K` is a normed field. -/
noncomputable instance instNormedFieldValuedAdicCompletion : NormedField (adicCompletion K v) :=
  Valued.toNormedField (adicCompletion K v) (WithZero (Multiplicative ℤ))


/-- A finite place of a number field `K` is a place associated to an embedding into a completion
with respect to a maximal ideal. -/
def FinitePlace (K : Type*) [Field K] [NumberField K] :=
  {w : AbsoluteValue K ℝ // ∃ v : HeightOneSpectrum (𝓞 K), place (embedding v) = w}


/-- Return the finite place defined by a maximal ideal `v`. -/
noncomputable def FinitePlace.mk (v : HeightOneSpectrum (𝓞 K)) : FinitePlace K :=
  ⟨place (embedding v), ⟨v, rfl⟩⟩


lemma toNNReal_Valued_eq_vadicAbv (x : K) :
    toNNReal (norm_ne_zero v) (Valued.v (self:=v.adicValued) x) = vadicAbv v x := rfl


/-- The norm of the image after the embedding associated to `v` is equal to the `v`-adic absolute
value. -/
theorem FinitePlace.norm_def (x : K) : ‖embedding v x‖ = vadicAbv v x := by
  simp only [NormedField.toNorm, instNormedFieldValuedAdicCompletion, Valued.toNormedField,
    instFieldAdicCompletion, Valued.norm, Valuation.RankOne.hom, MonoidWithZeroHom.coe_mk,
    ZeroHom.coe_mk, embedding, UniformSpace.Completion.coeRingHom, RingHom.coe_mk, MonoidHom.coe_mk,
    OneHom.coe_mk, Valued.valuedCompletion_apply, toNNReal_Valued_eq_vadicAbv]


/-- The norm of the image after the embedding associated to `v` is equal to the norm of `v` raised
to the power of the `v`-adic valuation. -/
theorem FinitePlace.norm_def' (x : K) : ‖embedding v x‖ = toNNReal (norm_ne_zero v)
    (v.valuation x) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    v : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
    x : K
    ⊢ Eq (Norm.norm ((NumberField.embedding v) x)) ↑((WithZeroMulInt.toNNReal ⋯) ( …
  -/
  rw [norm_def, vadicAbv_def]
  /-
    🎉 no goals
  -/


/-- The norm of the image after the embedding associated to `v` is equal to the norm of `v` raised
to the power of the `v`-adic valuation for integers. -/
theorem FinitePlace.norm_def_int (x : 𝓞 K) : ‖embedding v x‖ = toNNReal (norm_ne_zero v)
    (v.intValuationDef x) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    v : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
    x : NumberField.RingOfIntegers K
    ⊢ Eq (Norm.norm ((NumberField.embedding v) ↑x)) ↑((WithZeroMulInt.toNNReal ⋯)  …
  -/
  rw [norm_def, vadicAbv_def, valuation_eq_intValuationDef]
  /-
    🎉 no goals
  -/


/-- The `v`-adic norm of an integer is at most 1. -/
theorem norm_le_one (x : 𝓞 K) : ‖embedding v x‖ ≤ 1 := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    v : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
    x : NumberField.RingOfIntegers K
    ⊢ LE.le (Norm.norm ((NumberField.embedding v) ↑x)) 1
  -/
  rw [norm_def', NNReal.coe_le_one, toNNReal_le_one_iff (one_lt_norm v)]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    v : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
    x : NumberField.RingOfIntegers K
    ⊢ LE.le (v.valuation ↑x) 1
  -/
  exact valuation_le_one v x
  /-
    🎉 no goals
  -/


/-- The `v`-adic norm of an integer is 1 if and only if it is not in the ideal. -/
theorem norm_eq_one_iff_not_mem (x : 𝓞 K) : ‖(embedding v) x‖ = 1 ↔ x ∉ v.asIdeal := by
  rw [norm_def_int, NNReal.coe_eq_one, toNNReal_eq_one_iff (v.intValuationDef x)
    (norm_ne_zero v) (one_lt_norm v).ne', ← dvd_span_singleton,
    ← intValuation_lt_one_iff_dvd, not_lt]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    v : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
    x : NumberField.RingOfIntegers K
    ⊢ Iff (Eq (v.intValuationDef x) 1) (LE.le 1 (v.intValuationDef x))
  -/
  exact (intValuation_le_one v x).ge_iff_eq.symm
  /-
    🎉 no goals
  -/


/-- The `v`-adic norm of an integer is less than 1 if and only if it is in the ideal. -/
theorem norm_lt_one_iff_mem (x : 𝓞 K) : ‖embedding v x‖ < 1 ↔ x ∈ v.asIdeal := by
  rw [norm_def_int, NNReal.coe_lt_one, toNNReal_lt_one_iff (one_lt_norm v),
    intValuation_lt_one_iff_dvd]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    v : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
    x : NumberField.RingOfIntegers K
    ⊢ Iff (Dvd.dvd v.asIdeal (Ideal.span (Singleton.singleton x))) (Membership.mem …
  -/
  exact dvd_span_singleton
  /-
    🎉 no goals
  -/


instance : FunLike (FinitePlace K) K ℝ where
  coe w x := w.1 x
  coe_injective' _ _ h := Subtype.eq (AbsoluteValue.ext <| congr_fun h)


instance : MonoidWithZeroHomClass (FinitePlace K) K ℝ where
  map_mul w := w.1.map_mul
  map_one w := w.1.map_one
  map_zero w := w.1.map_zero


instance : NonnegHomClass (FinitePlace K) K ℝ where
  apply_nonneg w := w.1.nonneg


@[simp]
theorem apply (v : HeightOneSpectrum (𝓞 K)) (x : K) : mk v x =  ‖embedding v x‖ := rfl


/-- For a finite place `w`, return a maximal ideal `v` such that `w = finite_place v` . -/
noncomputable def maximalIdeal (w : FinitePlace K) : HeightOneSpectrum (𝓞 K) := w.2.choose


@[simp]
theorem mk_maximalIdeal (w : FinitePlace K) : mk (maximalIdeal w) = w := Subtype.ext w.2.choose_spec


@[simp]
theorem norm_embedding_eq (w : FinitePlace K) (x : K) :
    ‖embedding (maximalIdeal w) x‖ = w x := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    w : NumberField.FinitePlace K
    x : K
    ⊢ Eq (Norm.norm ((NumberField.embedding w.maximalIdeal) x)) (w x)
  -/
  conv_rhs => rw [← mk_maximalIdeal w, apply]
  /-
    🎉 no goals
  -/


theorem pos_iff {w : FinitePlace K} {x : K} : 0 < w x ↔ x ≠ 0 := w.1.pos_iff


@[simp]
theorem mk_eq_iff {v₁ v₂ : HeightOneSpectrum (𝓞 K)} : mk v₁ = mk v₂ ↔ v₁ = v₂ := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    v₁ v₂ : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
    ⊢ Iff (Eq (NumberField.FinitePlace.mk v₁) (NumberField.FinitePlace.mk v₂)) (Eq …
  -/
  refine ⟨?_, fun a ↦ by rw [a]⟩
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    v₁ v₂ : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
    ⊢ Eq (NumberField.FinitePlace.mk v₁) (NumberField.FinitePlace.mk v₂) → Eq v₁ v₂
  -/
  contrapose!
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    v₁ v₂ : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
    ⊢ Ne v₁ v₂ → Ne (NumberField.FinitePlace.mk v₁) (NumberField.FinitePlace.mk v₂)
  -/
  intro h
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    v₁ v₂ : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
    h : Ne v₁ v₂
    ⊢ Ne (NumberField.FinitePlace.mk v₁) (NumberField.FinitePlace.mk v₂)
  -/
  rw [DFunLike.ne_iff]
  have ⟨x, hx1, hx2⟩ : ∃ x : 𝓞 K, x ∈ v₁.asIdeal ∧ x ∉ v₂.asIdeal := by
    by_contra! H
    exact h <| HeightOneSpectrum.ext_iff.mpr <| IsMaximal.eq_of_le (isMaximal v₁) IsPrime.ne_top' H
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    v₁ v₂ : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
    h : Ne v₁ v₂
    x : NumberField.RingOfIntegers K
    hx1 : Membership.mem v₁.asIdeal x
    hx2 : Not (Membership.mem v₂.asIdeal x)
    ⊢ Exists fun a => Ne ((NumberField.FinitePlace.mk v₁) a) ((NumberField.FiniteP …
  -/
  use x
  /-
    case h
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    v₁ v₂ : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
    h : Ne v₁ v₂
    x : NumberField.RingOfIntegers K
    hx1 : Membership.mem v₁.asIdeal x
    hx2 : Not (Membership.mem v₂.asIdeal x)
    ⊢ Ne ((NumberField.FinitePlace.mk v₁) ↑x) ((NumberField.FinitePlace.mk v₂) ↑x)
  -/
  simp only [apply]
  /-
    case h
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    v₁ v₂ : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
    h : Ne v₁ v₂
    x : NumberField.RingOfIntegers K
    hx1 : Membership.mem v₁.asIdeal x
    hx2 : Not (Membership.mem v₂.asIdeal x)
    ⊢ Ne (Norm.norm ((NumberField.embedding v₁) ↑x)) (Norm.norm ((NumberField.embe …
  -/
  rw [← norm_lt_one_iff_mem] at hx1
  /-
    case h
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    v₁ v₂ : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
    h : Ne v₁ v₂
    x : NumberField.RingOfIntegers K
    hx1 : LT.lt (Norm.norm ((NumberField.embedding v₁) ↑x)) 1
    hx2 : Not (Membership.mem v₂.asIdeal x)
    ⊢ Ne (Norm.norm ((NumberField.embedding v₁) ↑x)) (Norm.norm ((NumberField.embe …
  -/
  rw [← norm_eq_one_iff_not_mem] at hx2
  /-
    case h
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    v₁ v₂ : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
    h : Ne v₁ v₂
    x : NumberField.RingOfIntegers K
    hx1 : LT.lt (Norm.norm ((NumberField.embedding v₁) ↑x)) 1
    hx2 : Eq (Norm.norm ((NumberField.embedding v₂) ↑x)) 1
    ⊢ Ne (Norm.norm ((NumberField.embedding v₁) ↑x)) (Norm.norm ((NumberField.embe …
  -/
  linarith
  /-
    🎉 no goals
  -/


theorem maximalIdeal_mk (v : HeightOneSpectrum (𝓞 K)) : maximalIdeal (mk v) = v := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    v : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
    ⊢ Eq (NumberField.FinitePlace.mk v).maximalIdeal v
  -/
  rw [← mk_eq_iff, mk_maximalIdeal]
  /-
    🎉 no goals
  -/


/-- The equivalence between finite places and maximal ideals. -/
noncomputable def equivHeightOneSpectrum :
    FinitePlace K ≃ HeightOneSpectrum (𝓞 K) where
  toFun := maximalIdeal
  invFun := mk
  left_inv := mk_maximalIdeal
  right_inv := maximalIdeal_mk


lemma maximalIdeal_injective : (fun w : FinitePlace K ↦ maximalIdeal w).Injective :=
  equivHeightOneSpectrum.injective


lemma maximalIdeal_inj (w₁ w₂ : FinitePlace K) : maximalIdeal w₁ = maximalIdeal w₂ ↔ w₁ = w₂ :=
  equivHeightOneSpectrum.injective.eq_iff


theorem mulSupport_finite_int {x : 𝓞 K} (h_x_nezero : x ≠ 0) :
    (Function.mulSupport fun w : FinitePlace K ↦ w x).Finite := by
  have (w : FinitePlace K) : w x ≠ 1 ↔ w x < 1 :=
    ne_iff_lt_iff_le.mpr <| norm_embedding_eq w x ▸ norm_le_one w.maximalIdeal x
  simp_rw [Function.mulSupport, this, ← norm_embedding_eq, norm_lt_one_iff_mem,
    ← Ideal.dvd_span_singleton]
  have h : {v : HeightOneSpectrum (𝓞 K) | v.asIdeal ∣ span {x}}.Finite := by
    apply Ideal.finite_factors
    simp only [Submodule.zero_eq_bot, ne_eq, span_singleton_eq_bot, h_x_nezero, not_false_eq_true]
  have h_inj : Set.InjOn FinitePlace.maximalIdeal {w | w.maximalIdeal.asIdeal ∣ span {x}} :=
    Function.Injective.injOn maximalIdeal_injective
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : NumberField.RingOfIntegers K
    h_x_nezero : Ne x 0
    this : ∀ (w : NumberField.FinitePlace K), Iff (Ne (w ↑x) 1) (LT.lt (w ↑x) 1)
    h : (setOf fun v => Dvd.dvd v.asIdeal (Ideal.span (Singleton.singleton x))).Fi …
    h_inj : Set.InjOn NumberField.FinitePlace.maximalIdeal (setOf fun w => Dvd.dvd …
    ⊢ (setOf fun x_1 => Dvd.dvd x_1.maximalIdeal.asIdeal (Ideal.span (Singleton.si …
  -/
  refine (h.subset ?_).of_finite_image h_inj
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : NumberField.RingOfIntegers K
    h_x_nezero : Ne x 0
    this : ∀ (w : NumberField.FinitePlace K), Iff (Ne (w ↑x) 1) (LT.lt (w ↑x) 1)
    h : (setOf fun v => Dvd.dvd v.asIdeal (Ideal.span (Singleton.singleton x))).Fi …
    h_inj : Set.InjOn NumberField.FinitePlace.maximalIdeal (setOf fun w => Dvd.dvd …
    ⊢ HasSubset.Subset (Set.image NumberField.FinitePlace.maximalIdeal (setOf fun  …
  -/
  simp only [dvd_span_singleton, Set.image_subset_iff, Set.preimage_setOf_eq, subset_refl]
  /-
    🎉 no goals
  -/


theorem mulSupport_finite {x : K} (h_x_nezero : x ≠ 0) :
    (Function.mulSupport fun w : FinitePlace K ↦ w x).Finite := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : K
    h_x_nezero : Ne x 0
    ⊢ (Function.mulSupport fun w => w x).Finite
  -/
  rcases IsFractionRing.div_surjective (A := 𝓞 K) x with ⟨a, b, hb, rfl⟩
  simp_all only [ne_eq, div_eq_zero_iff, NoZeroSMulDivisors.algebraMap_eq_zero_iff, not_or,
    map_div₀]
  /-
    case intro.intro.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    a b : NumberField.RingOfIntegers K
    hb : Membership.mem (nonZeroDivisors (NumberField.RingOfIntegers K)) b
    h_x_nezero : And (Not (Eq a 0)) (Not (Eq b 0))
    ⊢ (Function.mulSupport fun w => HDiv.hDiv (w ((algebraMap (NumberField.RingOfI …
  -/
  obtain ⟨ha, hb⟩ := h_x_nezero
  /-
    case intro.intro.intro.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    a b : NumberField.RingOfIntegers K
    hb✝ : Membership.mem (nonZeroDivisors (NumberField.RingOfIntegers K)) b
    ha : Not (Eq a 0)
    hb : Not (Eq b 0)
    ⊢ (Function.mulSupport fun w => HDiv.hDiv (w ((algebraMap (NumberField.RingOfI …
  -/
  simp_rw [← RingOfIntegers.coe_eq_algebraMap]
  /-
    case intro.intro.intro.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    a b : NumberField.RingOfIntegers K
    hb✝ : Membership.mem (nonZeroDivisors (NumberField.RingOfIntegers K)) b
    ha : Not (Eq a 0)
    hb : Not (Eq b 0)
    ⊢ (Function.mulSupport fun w => HDiv.hDiv (w ↑a) (w ↑b)).Finite
  -/
  apply ((mulSupport_finite_int ha).union (mulSupport_finite_int hb)).subset
  /-
    case intro.intro.intro.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    a b : NumberField.RingOfIntegers K
    hb✝ : Membership.mem (nonZeroDivisors (NumberField.RingOfIntegers K)) b
    ha : Not (Eq a 0)
    hb : Not (Eq b 0)
    ⊢ HasSubset.Subset (Function.mulSupport fun w => HDiv.hDiv (w ↑a) (w ↑b)) (Uni …
  -/
  intro w
  /-
    case intro.intro.intro.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    a b : NumberField.RingOfIntegers K
    hb✝ : Membership.mem (nonZeroDivisors (NumberField.RingOfIntegers K)) b
    ha : Not (Eq a 0)
    hb : Not (Eq b 0)
    w : NumberField.FinitePlace K
    ⊢ Membership.mem (Function.mulSupport fun w => HDiv.hDiv (w ↑a) (w ↑b)) w → Me …
  -/
  simp only [Function.mem_mulSupport, ne_eq, Set.mem_union]
  /-
    case intro.intro.intro.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    a b : NumberField.RingOfIntegers K
    hb✝ : Membership.mem (nonZeroDivisors (NumberField.RingOfIntegers K)) b
    ha : Not (Eq a 0)
    hb : Not (Eq b 0)
    w : NumberField.FinitePlace K
    ⊢ Not (Eq (HDiv.hDiv (w ↑a) (w ↑b)) 1) → Or (Not (Eq (w ↑a) 1)) (Not (Eq (w ↑b …
  -/
  contrapose!
  /-
    case intro.intro.intro.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    a b : NumberField.RingOfIntegers K
    hb✝ : Membership.mem (nonZeroDivisors (NumberField.RingOfIntegers K)) b
    ha : Not (Eq a 0)
    hb : Not (Eq b 0)
    w : NumberField.FinitePlace K
    ⊢ And (Eq (w ↑a) 1) (Eq (w ↑b) 1) → Eq (HDiv.hDiv (w ↑a) (w ↑b)) 1
  -/
  simp +contextual only [ne_eq, one_ne_zero, not_false_eq_true, div_self, implies_true]
  /-
    🎉 no goals
  -/


lemma equivHeightOneSpectrum_symm_apply (v : HeightOneSpectrum (𝓞 K)) (x : K) :
    (equivHeightOneSpectrum.symm v) x = ‖embedding v x‖ := by
  have : v = (equivHeightOneSpectrum.symm v).maximalIdeal := by
    show v = equivHeightOneSpectrum (equivHeightOneSpectrum.symm v)
    exact (Equiv.apply_symm_apply _ v).symm
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    v : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
    x : K
    this : Eq v (NumberField.FinitePlace.equivHeightOneSpectrum.symm v).maximalIdeal
    ⊢ Eq ((NumberField.FinitePlace.equivHeightOneSpectrum.symm v) x) (Norm.norm (( …
  -/
  convert (norm_embedding_eq (equivHeightOneSpectrum.symm v) x).symm
  /-
    🎉 no goals
  -/


open Ideal in
lemma embedding_mul_absNorm (v : HeightOneSpectrum (𝓞 K)) {x : 𝓞 K} (h_x_nezero : x ≠ 0) :
    ‖(embedding v) x‖ * absNorm (v.maxPowDividing (span {x})) = 1 := by
  rw [maxPowDividing, map_pow, Nat.cast_pow, norm_def, vadicAbv_def,
    WithZeroMulInt.toNNReal_neg_apply _
      (v.valuation.ne_zero_iff.mpr (RingOfIntegers.coe_ne_zero_iff.mpr h_x_nezero))]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    v : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
    x : NumberField.RingOfIntegers K
    h_x_nezero : Ne x 0
    ⊢ Eq (HMul.hMul (↑(HPow.hPow (↑(Ideal.absNorm v.asIdeal)) (Multiplicative.toAd …
  -/
  push_cast
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    v : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
    x : NumberField.RingOfIntegers K
    h_x_nezero : Ne x 0
    ⊢ Eq (HMul.hMul (HPow.hPow (↑(Ideal.absNorm v.asIdeal)) (Multiplicative.toAdd  …
  -/
  rw [← zpow_natCast, ← zpow_add₀ <| mod_cast (zero_lt_one.trans (one_lt_norm v)).ne']
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    v : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
    x : NumberField.RingOfIntegers K
    h_x_nezero : Ne x 0
    ⊢ Eq (HPow.hPow (↑(Ideal.absNorm v.asIdeal)) (HAdd.hAdd (Multiplicative.toAdd  …
  -/
  norm_cast
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    v : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
    x : NumberField.RingOfIntegers K
    h_x_nezero : Ne x 0
    ⊢ Eq (HPow.hPow (↑(Ideal.absNorm v.asIdeal)) (HAdd.hAdd (Multiplicative.toAdd  …
  -/
  rw [zpow_eq_one_iff_right₀ (Nat.cast_nonneg' _) (mod_cast (one_lt_norm v).ne')]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    v : IsDedekindDomain.HeightOneSpectrum (NumberField.RingOfIntegers K)
    x : NumberField.RingOfIntegers K
    h_x_nezero : Ne x 0
    ⊢ Eq (HAdd.hAdd (Multiplicative.toAdd (WithZero.unzero ⋯)) ↑((Associates.mk v. …
  -/
  simp [valuation_eq_intValuationDef, intValuationDef_if_neg, h_x_nezero]
  /-
    🎉 no goals
  -/


