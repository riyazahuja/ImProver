/-- A function is a Schwartz function if it is smooth and all derivatives decay faster than
  any power of `‖x‖`. -/
structure SchwartzMap where
  toFun : E → F
  smooth' : ContDiff ℝ ∞ toFun
  decay' : ∀ k n : ℕ, ∃ C : ℝ, ∀ x, ‖x‖ ^ k * ‖iteratedFDeriv ℝ n toFun x‖ ≤ C


/-- A function is a Schwartz function if it is smooth and all derivatives decay faster than
  any power of `‖x‖`. -/
scoped[SchwartzMap] notation "𝓢(" E ", " F ")" => SchwartzMap E F


instance instFunLike : FunLike 𝓢(E, F) E F where
  coe f := f.toFun
                             /-
                               𝕜 : Type u_1
                               𝕜' : Type u_2
                               D : Type u_3
                               E : Type u_4
                               F : Type u_5
                               G : Type u_6
                               V : Type u_7
                               inst✝³ : NormedAddCommGroup E
                               inst✝² : NormedSpace Real E
                               inst✝¹ : NormedAddCommGroup F
                               inst✝ : NormedSpace Real F
                               f g : SchwartzMap E F
                               h : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
                               ⊢ Eq f g
                             -/
  coe_injective' f g h := by cases f; cases g; congr
                                               /-
                                                 🎉 no goals
                                               -/


/-- All derivatives of a Schwartz function are rapidly decaying. -/
theorem decay (f : 𝓢(E, F)) (k n : ℕ) :
    ∃ C : ℝ, 0 < C ∧ ∀ x, ‖x‖ ^ k * ‖iteratedFDeriv ℝ n f x‖ ≤ C := by
  /-
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : SchwartzMap E F
    k n : Nat
    ⊢ Exists fun C => And (LT.lt 0 C) (∀ (x : E), LE.le (HMul.hMul (HPow.hPow (Nor …
  -/
  rcases f.decay' k n with ⟨C, hC⟩
  /-
    case intro
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : SchwartzMap E F
    k n : Nat
    C : Real
    hC : ∀ (x : E), LE.le (HMul.hMul (HPow.hPow (Norm.norm x) k) (Norm.norm (itera …
    ⊢ Exists fun C => And (LT.lt 0 C) (∀ (x : E), LE.le (HMul.hMul (HPow.hPow (Nor …
  -/
  exact ⟨max C 1, by positivity, fun x => (hC x).trans (le_max_left _ _)⟩
  /-
    🎉 no goals
  -/


/-- Every Schwartz function is smooth. -/
theorem smooth (f : 𝓢(E, F)) (n : ℕ∞) : ContDiff ℝ n f :=
  f.smooth'.of_le (mod_cast le_top)


/-- Every Schwartz function is continuous. -/
@[continuity]
protected theorem continuous (f : 𝓢(E, F)) : Continuous f :=
  (f.smooth 0).continuous


instance instContinuousMapClass : ContinuousMapClass 𝓢(E, F) E F where
  map_continuous := SchwartzMap.continuous


/-- Every Schwartz function is differentiable. -/
protected theorem differentiable (f : 𝓢(E, F)) : Differentiable ℝ f :=
  (f.smooth 1).differentiable rfl.le


/-- Every Schwartz function is differentiable at any point. -/
protected theorem differentiableAt (f : 𝓢(E, F)) {x : E} : DifferentiableAt ℝ f x :=
  f.differentiable.differentiableAt


@[ext]
theorem ext {f g : 𝓢(E, F)} (h : ∀ x, (f : E → F) x = g x) : f = g :=
  DFunLike.ext f g h


/-- Auxiliary lemma, used in proving the more general result `isBigO_cocompact_rpow`. -/
theorem isBigO_cocompact_zpow_neg_nat (k : ℕ) :
    f =O[cocompact E] fun x => ‖x‖ ^ (-k : ℤ) := by
  /-
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : SchwartzMap E F
    k : Nat
    ⊢ Asymptotics.IsBigO (Filter.cocompact E) ⇑f fun x => HPow.hPow (Norm.norm x)  …
  -/
  obtain ⟨d, _, hd'⟩ := f.decay k 0
  /-
    case intro.intro
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : SchwartzMap E F
    k : Nat
    d : Real
    left✝ : LT.lt 0 d
    hd' : ∀ (x : E), LE.le (HMul.hMul (HPow.hPow (Norm.norm x) k) (Norm.norm (iter …
    ⊢ Asymptotics.IsBigO (Filter.cocompact E) ⇑f fun x => HPow.hPow (Norm.norm x)  …
  -/
  simp only [norm_iteratedFDeriv_zero] at hd'
  /-
    case intro.intro
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : SchwartzMap E F
    k : Nat
    d : Real
    left✝ : LT.lt 0 d
    hd' : ∀ (x : E), LE.le (HMul.hMul (HPow.hPow (Norm.norm x) k) (Norm.norm (f x) …
    ⊢ Asymptotics.IsBigO (Filter.cocompact E) ⇑f fun x => HPow.hPow (Norm.norm x)  …
  -/
  simp_rw [Asymptotics.IsBigO, Asymptotics.IsBigOWith]
  /-
    case intro.intro
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : SchwartzMap E F
    k : Nat
    d : Real
    left✝ : LT.lt 0 d
    hd' : ∀ (x : E), LE.le (HMul.hMul (HPow.hPow (Norm.norm x) k) (Norm.norm (f x) …
    ⊢ Exists fun c => Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hM …
  -/
  refine ⟨d, Filter.Eventually.filter_mono Filter.cocompact_le_cofinite ?_⟩
  /-
    case intro.intro
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : SchwartzMap E F
    k : Nat
    d : Real
    left✝ : LT.lt 0 d
    hd' : ∀ (x : E), LE.le (HMul.hMul (HPow.hPow (Norm.norm x) k) (Norm.norm (f x) …
    ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul d (Norm.norm  …
  -/
  refine (Filter.eventually_cofinite_ne 0).mono fun x hx => ?_
  /-
    case intro.intro
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : SchwartzMap E F
    k : Nat
    d : Real
    left✝ : LT.lt 0 d
    hd' : ∀ (x : E), LE.le (HMul.hMul (HPow.hPow (Norm.norm x) k) (Norm.norm (f x) …
    x : E
    hx : Ne x 0
    ⊢ LE.le (Norm.norm (f x)) (HMul.hMul d (Norm.norm (HPow.hPow (Norm.norm x) (Ne …
  -/
  rw [Real.norm_of_nonneg (zpow_nonneg (norm_nonneg _) _), zpow_neg, ← div_eq_mul_inv, le_div_iff₀']
  /-
    case intro.intro
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : SchwartzMap E F
    k : Nat
    d : Real
    left✝ : LT.lt 0 d
    hd' : ∀ (x : E), LE.le (HMul.hMul (HPow.hPow (Norm.norm x) k) (Norm.norm (f x) …
    x : E
    hx : Ne x 0
    ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm x) ↑k) (Norm.norm (f x))) d
  -/
  exacts [hd' x, zpow_pos (norm_pos_iff.mpr hx) _]
  /-
    🎉 no goals
  -/


theorem isBigO_cocompact_rpow [ProperSpace E] (s : ℝ) :
    f =O[cocompact E] fun x => ‖x‖ ^ s := by
  /-
    E : Type u_4
    F : Type u_5
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    f : SchwartzMap E F
    inst✝ : ProperSpace E
    s : Real
    ⊢ Asymptotics.IsBigO (Filter.cocompact E) ⇑f fun x => HPow.hPow (Norm.norm x) s
  -/
  let k := ⌈-s⌉₊
  /-
    E : Type u_4
    F : Type u_5
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    f : SchwartzMap E F
    inst✝ : ProperSpace E
    s : Real
    k : Nat := Nat.ceil (Neg.neg s)
    ⊢ Asymptotics.IsBigO (Filter.cocompact E) ⇑f fun x => HPow.hPow (Norm.norm x) s
  -/
  have hk : -(k : ℝ) ≤ s := neg_le.mp (Nat.le_ceil (-s))
  /-
    E : Type u_4
    F : Type u_5
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    f : SchwartzMap E F
    inst✝ : ProperSpace E
    s : Real
    k : Nat := Nat.ceil (Neg.neg s)
    hk : LE.le (Neg.neg ↑k) s
    ⊢ Asymptotics.IsBigO (Filter.cocompact E) ⇑f fun x => HPow.hPow (Norm.norm x) s
  -/
  refine (isBigO_cocompact_zpow_neg_nat f k).trans ?_
  suffices (fun x : ℝ => x ^ (-k : ℤ)) =O[atTop] fun x : ℝ => x ^ s
    from this.comp_tendsto tendsto_norm_cocompact_atTop
  /-
    E : Type u_4
    F : Type u_5
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    f : SchwartzMap E F
    inst✝ : ProperSpace E
    s : Real
    k : Nat := Nat.ceil (Neg.neg s)
    hk : LE.le (Neg.neg ↑k) s
    ⊢ Asymptotics.IsBigO Filter.atTop (fun x => HPow.hPow x (Neg.neg ↑k)) fun x => …
  -/
  simp_rw [Asymptotics.IsBigO, Asymptotics.IsBigOWith]
  /-
    E : Type u_4
    F : Type u_5
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    f : SchwartzMap E F
    inst✝ : ProperSpace E
    s : Real
    k : Nat := Nat.ceil (Neg.neg s)
    hk : LE.le (Neg.neg ↑k) s
    ⊢ Exists fun c => Filter.Eventually (fun x => LE.le (Norm.norm (HPow.hPow x (N …
  -/
  refine ⟨1, (Filter.eventually_ge_atTop 1).mono fun x hx => ?_⟩
  rw [one_mul, Real.norm_of_nonneg (Real.rpow_nonneg (zero_le_one.trans hx) _),
    Real.norm_of_nonneg (zpow_nonneg (zero_le_one.trans hx) _), ← Real.rpow_intCast, Int.cast_neg,
    Int.cast_natCast]
  /-
    E : Type u_4
    F : Type u_5
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    f : SchwartzMap E F
    inst✝ : ProperSpace E
    s : Real
    k : Nat := Nat.ceil (Neg.neg s)
    hk : LE.le (Neg.neg ↑k) s
    x : Real
    hx : LE.le 1 x
    ⊢ LE.le (HPow.hPow x (Neg.neg ↑k)) (HPow.hPow x s)
  -/
  exact Real.rpow_le_rpow_of_exponent_le hx hk
  /-
    🎉 no goals
  -/


theorem isBigO_cocompact_zpow [ProperSpace E] (k : ℤ) :
    f =O[cocompact E] fun x => ‖x‖ ^ k := by
  /-
    E : Type u_4
    F : Type u_5
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    f : SchwartzMap E F
    inst✝ : ProperSpace E
    k : Int
    ⊢ Asymptotics.IsBigO (Filter.cocompact E) ⇑f fun x => HPow.hPow (Norm.norm x) k
  -/
  simpa only [Real.rpow_intCast] using isBigO_cocompact_rpow f k
  /-
    🎉 no goals
  -/


theorem bounds_nonempty (k n : ℕ) (f : 𝓢(E, F)) :
    ∃ c : ℝ, c ∈ { c : ℝ | 0 ≤ c ∧ ∀ x : E, ‖x‖ ^ k * ‖iteratedFDeriv ℝ n f x‖ ≤ c } :=
  let ⟨M, hMp, hMb⟩ := f.decay k n
  ⟨M, le_of_lt hMp, hMb⟩


theorem bounds_bddBelow (k n : ℕ) (f : 𝓢(E, F)) :
    BddBelow { c | 0 ≤ c ∧ ∀ x, ‖x‖ ^ k * ‖iteratedFDeriv ℝ n f x‖ ≤ c } :=
  ⟨0, fun _ ⟨hn, _⟩ => hn⟩


theorem decay_add_le_aux (k n : ℕ) (f g : 𝓢(E, F)) (x : E) :
    ‖x‖ ^ k * ‖iteratedFDeriv ℝ n ((f : E → F) + (g : E → F)) x‖ ≤
      ‖x‖ ^ k * ‖iteratedFDeriv ℝ n f x‖ + ‖x‖ ^ k * ‖iteratedFDeriv ℝ n g x‖ := by
  /-
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    k n : Nat
    f g : SchwartzMap E F
    x : E
    ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm x) k) (Norm.norm (iteratedFDeriv Real …
  -/
  rw [← mul_add]
  /-
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    k n : Nat
    f g : SchwartzMap E F
    x : E
    ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm x) k) (Norm.norm (iteratedFDeriv Real …
  -/
  refine mul_le_mul_of_nonneg_left ?_ (by positivity)
  /-
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    k n : Nat
    f g : SchwartzMap E F
    x : E
    ⊢ LE.le (Norm.norm (iteratedFDeriv Real n (HAdd.hAdd ⇑f ⇑g) x)) (HAdd.hAdd (No …
  -/
  rw [iteratedFDeriv_add_apply (f.smooth _) (g.smooth _)]
  /-
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    k n : Nat
    f g : SchwartzMap E F
    x : E
    ⊢ LE.le (Norm.norm (HAdd.hAdd (iteratedFDeriv Real n (⇑f) x) (iteratedFDeriv R …
  -/
  exact norm_add_le _ _
  /-
    🎉 no goals
  -/


theorem decay_neg_aux (k n : ℕ) (f : 𝓢(E, F)) (x : E) :
    ‖x‖ ^ k * ‖iteratedFDeriv ℝ n (-f : E → F) x‖ = ‖x‖ ^ k * ‖iteratedFDeriv ℝ n f x‖ := by
  /-
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    k n : Nat
    f : SchwartzMap E F
    x : E
    ⊢ Eq (HMul.hMul (HPow.hPow (Norm.norm x) k) (Norm.norm (iteratedFDeriv Real n  …
  -/
  rw [iteratedFDeriv_neg_apply, norm_neg]
  /-
    🎉 no goals
  -/


theorem decay_smul_aux (k n : ℕ) (f : 𝓢(E, F)) (c : 𝕜) (x : E) :
    ‖x‖ ^ k * ‖iteratedFDeriv ℝ n (c • (f : E → F)) x‖ =
      ‖c‖ * ‖x‖ ^ k * ‖iteratedFDeriv ℝ n f x‖ := by
  rw [mul_comm ‖c‖, mul_assoc, iteratedFDeriv_const_smul_apply (f.smooth _),
    norm_smul c (iteratedFDeriv ℝ n (⇑f) x)]


/-- Helper definition for the seminorms of the Schwartz space. -/
protected def seminormAux (k n : ℕ) (f : 𝓢(E, F)) : ℝ :=
  sInf { c | 0 ≤ c ∧ ∀ x, ‖x‖ ^ k * ‖iteratedFDeriv ℝ n f x‖ ≤ c }


theorem seminormAux_nonneg (k n : ℕ) (f : 𝓢(E, F)) : 0 ≤ f.seminormAux k n :=
  le_csInf (bounds_nonempty k n f) fun _ ⟨hx, _⟩ => hx


theorem le_seminormAux (k n : ℕ) (f : 𝓢(E, F)) (x : E) :
    ‖x‖ ^ k * ‖iteratedFDeriv ℝ n (⇑f) x‖ ≤ f.seminormAux k n :=
  le_csInf (bounds_nonempty k n f) fun _ ⟨_, h⟩ => h x


/-- If one controls the norm of every `A x`, then one controls the norm of `A`. -/
theorem seminormAux_le_bound (k n : ℕ) (f : 𝓢(E, F)) {M : ℝ} (hMp : 0 ≤ M)
    (hM : ∀ x, ‖x‖ ^ k * ‖iteratedFDeriv ℝ n f x‖ ≤ M) : f.seminormAux k n ≤ M :=
  csInf_le (bounds_bddBelow k n f) ⟨hMp, hM⟩


instance instSMul : SMul 𝕜 𝓢(E, F) :=
  ⟨fun c f =>
    { toFun := c • (f : E → F)
      smooth' := (f.smooth _).const_smul c
      decay' := fun k n => by
        /-
          𝕜 : Type u_1
          𝕜' : Type u_2
          D : Type u_3
          E : Type u_4
          F : Type u_5
          G : Type u_6
          V : Type u_7
          inst✝⁹ : NormedAddCommGroup E
          inst✝⁸ : NormedSpace Real E
          inst✝⁷ : NormedAddCommGroup F
          inst✝⁶ : NormedSpace Real F
          inst✝⁵ : NormedField 𝕜
          inst✝⁴ : NormedSpace 𝕜 F
          inst✝³ : SMulCommClass Real 𝕜 F
          inst✝² : NormedField 𝕜'
          inst✝¹ : NormedSpace 𝕜' F
          inst✝ : SMulCommClass Real 𝕜' F
          c : 𝕜
          f : SchwartzMap E F
          k n : Nat
          ⊢ Exists fun C => ∀ (x : E), LE.le (HMul.hMul (HPow.hPow (Norm.norm x) k) (Nor …
        -/
        refine ⟨f.seminormAux k n * (‖c‖ + 1), fun x => ?_⟩
        /-
          𝕜 : Type u_1
          𝕜' : Type u_2
          D : Type u_3
          E : Type u_4
          F : Type u_5
          G : Type u_6
          V : Type u_7
          inst✝⁹ : NormedAddCommGroup E
          inst✝⁸ : NormedSpace Real E
          inst✝⁷ : NormedAddCommGroup F
          inst✝⁶ : NormedSpace Real F
          inst✝⁵ : NormedField 𝕜
          inst✝⁴ : NormedSpace 𝕜 F
          inst✝³ : SMulCommClass Real 𝕜 F
          inst✝² : NormedField 𝕜'
          inst✝¹ : NormedSpace 𝕜' F
          inst✝ : SMulCommClass Real 𝕜' F
          c : 𝕜
          f : SchwartzMap E F
          k n : Nat
          x : E
          ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm x) k) (Norm.norm (iteratedFDeriv Real …
        -/
        have hc : 0 ≤ ‖c‖ := by positivity
        /-
          𝕜 : Type u_1
          𝕜' : Type u_2
          D : Type u_3
          E : Type u_4
          F : Type u_5
          G : Type u_6
          V : Type u_7
          inst✝⁹ : NormedAddCommGroup E
          inst✝⁸ : NormedSpace Real E
          inst✝⁷ : NormedAddCommGroup F
          inst✝⁶ : NormedSpace Real F
          inst✝⁵ : NormedField 𝕜
          inst✝⁴ : NormedSpace 𝕜 F
          inst✝³ : SMulCommClass Real 𝕜 F
          inst✝² : NormedField 𝕜'
          inst✝¹ : NormedSpace 𝕜' F
          inst✝ : SMulCommClass Real 𝕜' F
          c : 𝕜
          f : SchwartzMap E F
          k n : Nat
          x : E
          hc : LE.le 0 (Norm.norm c)
          ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm x) k) (Norm.norm (iteratedFDeriv Real …
        -/
        refine le_trans ?_ ((mul_le_mul_of_nonneg_right (f.le_seminormAux k n x) hc).trans ?_)
          /-
            case refine_1
            𝕜 : Type u_1
            𝕜' : Type u_2
            D : Type u_3
            E : Type u_4
            F : Type u_5
            G : Type u_6
            V : Type u_7
            inst✝⁹ : NormedAddCommGroup E
            inst✝⁸ : NormedSpace Real E
            inst✝⁷ : NormedAddCommGroup F
            inst✝⁶ : NormedSpace Real F
            inst✝⁵ : NormedField 𝕜
            inst✝⁴ : NormedSpace 𝕜 F
            inst✝³ : SMulCommClass Real 𝕜 F
            inst✝² : NormedField 𝕜'
            inst✝¹ : NormedSpace 𝕜' F
            inst✝ : SMulCommClass Real 𝕜' F
            c : 𝕜
            f : SchwartzMap E F
            k n : Nat
            x : E
            hc : LE.le 0 (Norm.norm c)
            ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm x) k) (Norm.norm (iteratedFDeriv Real …
          -/
        · apply Eq.le
          /-
            case refine_1.hab
            𝕜 : Type u_1
            𝕜' : Type u_2
            D : Type u_3
            E : Type u_4
            F : Type u_5
            G : Type u_6
            V : Type u_7
            inst✝⁹ : NormedAddCommGroup E
            inst✝⁸ : NormedSpace Real E
            inst✝⁷ : NormedAddCommGroup F
            inst✝⁶ : NormedSpace Real F
            inst✝⁵ : NormedField 𝕜
            inst✝⁴ : NormedSpace 𝕜 F
            inst✝³ : SMulCommClass Real 𝕜 F
            inst✝² : NormedField 𝕜'
            inst✝¹ : NormedSpace 𝕜' F
            inst✝ : SMulCommClass Real 𝕜' F
            c : 𝕜
            f : SchwartzMap E F
            k n : Nat
            x : E
            hc : LE.le 0 (Norm.norm c)
            ⊢ Eq (HMul.hMul (HPow.hPow (Norm.norm x) k) (Norm.norm (iteratedFDeriv Real n  …
          -/
          rw [mul_comm _ ‖c‖, ← mul_assoc]
          /-
            case refine_1.hab
            𝕜 : Type u_1
            𝕜' : Type u_2
            D : Type u_3
            E : Type u_4
            F : Type u_5
            G : Type u_6
            V : Type u_7
            inst✝⁹ : NormedAddCommGroup E
            inst✝⁸ : NormedSpace Real E
            inst✝⁷ : NormedAddCommGroup F
            inst✝⁶ : NormedSpace Real F
            inst✝⁵ : NormedField 𝕜
            inst✝⁴ : NormedSpace 𝕜 F
            inst✝³ : SMulCommClass Real 𝕜 F
            inst✝² : NormedField 𝕜'
            inst✝¹ : NormedSpace 𝕜' F
            inst✝ : SMulCommClass Real 𝕜' F
            c : 𝕜
            f : SchwartzMap E F
            k n : Nat
            x : E
            hc : LE.le 0 (Norm.norm c)
            ⊢ Eq (HMul.hMul (HPow.hPow (Norm.norm x) k) (Norm.norm (iteratedFDeriv Real n  …
          -/
          exact decay_smul_aux k n f c x
          /-
            🎉 no goals
          -/
          /-
            case refine_2
            𝕜 : Type u_1
            𝕜' : Type u_2
            D : Type u_3
            E : Type u_4
            F : Type u_5
            G : Type u_6
            V : Type u_7
            inst✝⁹ : NormedAddCommGroup E
            inst✝⁸ : NormedSpace Real E
            inst✝⁷ : NormedAddCommGroup F
            inst✝⁶ : NormedSpace Real F
            inst✝⁵ : NormedField 𝕜
            inst✝⁴ : NormedSpace 𝕜 F
            inst✝³ : SMulCommClass Real 𝕜 F
            inst✝² : NormedField 𝕜'
            inst✝¹ : NormedSpace 𝕜' F
            inst✝ : SMulCommClass Real 𝕜' F
            c : 𝕜
            f : SchwartzMap E F
            k n : Nat
            x : E
            hc : LE.le 0 (Norm.norm c)
            ⊢ LE.le (HMul.hMul (SchwartzMap.seminormAux k n f) (Norm.norm c)) (HMul.hMul ( …
          -/
        · apply mul_le_mul_of_nonneg_left _ (f.seminormAux_nonneg k n)
          /-
            𝕜 : Type u_1
            𝕜' : Type u_2
            D : Type u_3
            E : Type u_4
            F : Type u_5
            G : Type u_6
            V : Type u_7
            inst✝⁹ : NormedAddCommGroup E
            inst✝⁸ : NormedSpace Real E
            inst✝⁷ : NormedAddCommGroup F
            inst✝⁶ : NormedSpace Real F
            inst✝⁵ : NormedField 𝕜
            inst✝⁴ : NormedSpace 𝕜 F
            inst✝³ : SMulCommClass Real 𝕜 F
            inst✝² : NormedField 𝕜'
            inst✝¹ : NormedSpace 𝕜' F
            inst✝ : SMulCommClass Real 𝕜' F
            c : 𝕜
            f : SchwartzMap E F
            k n : Nat
            x : E
            hc : LE.le 0 (Norm.norm c)
            ⊢ LE.le (Norm.norm c) (HAdd.hAdd (Norm.norm c) 1)
          -/
          linarith }⟩
          /-
            🎉 no goals
          -/


@[simp]
theorem smul_apply {f : 𝓢(E, F)} {c : 𝕜} {x : E} : (c • f) x = c • f x :=
  rfl


instance instIsScalarTower [SMul 𝕜 𝕜'] [IsScalarTower 𝕜 𝕜' F] : IsScalarTower 𝕜 𝕜' 𝓢(E, F) :=
  ⟨fun a b f => ext fun x => smul_assoc a b (f x)⟩


instance instSMulCommClass [SMulCommClass 𝕜 𝕜' F] : SMulCommClass 𝕜 𝕜' 𝓢(E, F) :=
  ⟨fun a b f => ext fun x => smul_comm a b (f x)⟩


theorem seminormAux_smul_le (k n : ℕ) (c : 𝕜) (f : 𝓢(E, F)) :
    (c • f).seminormAux k n ≤ ‖c‖ * f.seminormAux k n := by
  refine
    (c • f).seminormAux_le_bound k n (mul_nonneg (norm_nonneg _) (seminormAux_nonneg _ _ _))
      fun x => (decay_smul_aux k n f c x).le.trans ?_
  /-
    𝕜 : Type u_1
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    k n : Nat
    c : 𝕜
    f : SchwartzMap E F
    x : E
    ⊢ LE.le (HMul.hMul (HMul.hMul (Norm.norm c) (HPow.hPow (Norm.norm x) k)) (Norm …
  -/
  rw [mul_assoc]
  /-
    𝕜 : Type u_1
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    k n : Nat
    c : 𝕜
    f : SchwartzMap E F
    x : E
    ⊢ LE.le (HMul.hMul (Norm.norm c) (HMul.hMul (HPow.hPow (Norm.norm x) k) (Norm. …
  -/
  exact mul_le_mul_of_nonneg_left (f.le_seminormAux k n x) (norm_nonneg _)
  /-
    🎉 no goals
  -/


instance instNSMul : SMul ℕ 𝓢(E, F) :=
  ⟨fun c f =>
    { toFun := c • (f : E → F)
      smooth' := (f.smooth _).const_smul c
                   /-
                     𝕜 : Type u_1
                     𝕜' : Type u_2
                     D : Type u_3
                     E : Type u_4
                     F : Type u_5
                     G : Type u_6
                     V : Type u_7
                     inst✝⁹ : NormedAddCommGroup E
                     inst✝⁸ : NormedSpace Real E
                     inst✝⁷ : NormedAddCommGroup F
                     inst✝⁶ : NormedSpace Real F
                     inst✝⁵ : NormedField 𝕜
                     inst✝⁴ : NormedSpace 𝕜 F
                     inst✝³ : SMulCommClass Real 𝕜 F
                     inst✝² : NormedField 𝕜'
                     inst✝¹ : NormedSpace 𝕜' F
                     inst✝ : SMulCommClass Real 𝕜' F
                     c : Nat
                     f : SchwartzMap E F
                     ⊢ ∀ (k n : Nat), Exists fun C => ∀ (x : E), LE.le (HMul.hMul (HPow.hPow (Norm. …
                   -/
      decay' := by simpa [← Nat.cast_smul_eq_nsmul ℝ] using ((c : ℝ) • f).decay' }⟩
                   /-
                     🎉 no goals
                   -/


instance instZSMul : SMul ℤ 𝓢(E, F) :=
  ⟨fun c f =>
    { toFun := c • (f : E → F)
      smooth' := (f.smooth _).const_smul c
                   /-
                     𝕜 : Type u_1
                     𝕜' : Type u_2
                     D : Type u_3
                     E : Type u_4
                     F : Type u_5
                     G : Type u_6
                     V : Type u_7
                     inst✝⁹ : NormedAddCommGroup E
                     inst✝⁸ : NormedSpace Real E
                     inst✝⁷ : NormedAddCommGroup F
                     inst✝⁶ : NormedSpace Real F
                     inst✝⁵ : NormedField 𝕜
                     inst✝⁴ : NormedSpace 𝕜 F
                     inst✝³ : SMulCommClass Real 𝕜 F
                     inst✝² : NormedField 𝕜'
                     inst✝¹ : NormedSpace 𝕜' F
                     inst✝ : SMulCommClass Real 𝕜' F
                     c : Int
                     f : SchwartzMap E F
                     ⊢ ∀ (k n : Nat), Exists fun C => ∀ (x : E), LE.le (HMul.hMul (HPow.hPow (Norm. …
                   -/
      decay' := by simpa [← Int.cast_smul_eq_zsmul ℝ] using ((c : ℝ) • f).decay' }⟩
                   /-
                     🎉 no goals
                   -/


instance instZero : Zero 𝓢(E, F) :=
  ⟨{  toFun := fun _ => 0
      smooth' := contDiff_const
                                           /-
                                             𝕜 : Type u_1
                                             𝕜' : Type u_2
                                             D : Type u_3
                                             E : Type u_4
                                             F : Type u_5
                                             G : Type u_6
                                             V : Type u_7
                                             inst✝³ : NormedAddCommGroup E
                                             inst✝² : NormedSpace Real E
                                             inst✝¹ : NormedAddCommGroup F
                                             inst✝ : NormedSpace Real F
                                             x✝² x✝¹ : Nat
                                             x✝ : E
                                             ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm x✝) x✝²) (Norm.norm (iteratedFDeriv R …
                                           -/
      decay' := fun _ _ => ⟨1, fun _ => by simp⟩ }⟩
                                           /-
                                             🎉 no goals
                                           -/


instance instInhabited : Inhabited 𝓢(E, F) :=
  ⟨0⟩


theorem coe_zero : DFunLike.coe (0 : 𝓢(E, F)) = (0 : E → F) :=
  rfl


@[simp]
theorem coeFn_zero : ⇑(0 : 𝓢(E, F)) = (0 : E → F) :=
  rfl


@[simp]
theorem zero_apply {x : E} : (0 : 𝓢(E, F)) x = 0 :=
  rfl


theorem seminormAux_zero (k n : ℕ) : (0 : 𝓢(E, F)).seminormAux k n = 0 :=
                                                             /-
                                                               E : Type u_4
                                                               F : Type u_5
                                                               inst✝³ : NormedAddCommGroup E
                                                               inst✝² : NormedSpace Real E
                                                               inst✝¹ : NormedAddCommGroup F
                                                               inst✝ : NormedSpace Real F
                                                               k n : Nat
                                                               x✝ : E
                                                               ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm x✝) k) (Norm.norm (iteratedFDeriv Rea …
                                                             -/
  le_antisymm (seminormAux_le_bound k n _ rfl.le fun _ => by simp [Pi.zero_def])
                                                             /-
                                                               🎉 no goals
                                                             -/
    (seminormAux_nonneg _ _ _)


instance instNeg : Neg 𝓢(E, F) :=
  ⟨fun f =>
    ⟨-f, (f.smooth _).neg, fun k n =>
      ⟨f.seminormAux k n, fun x => (decay_neg_aux k n f x).le.trans (f.le_seminormAux k n x)⟩⟩⟩


instance instAdd : Add 𝓢(E, F) :=
  ⟨fun f g =>
    ⟨f + g, (f.smooth _).add (g.smooth _), fun k n =>
      ⟨f.seminormAux k n + g.seminormAux k n, fun x =>
        (decay_add_le_aux k n f g x).trans
          (add_le_add (f.le_seminormAux k n x) (g.le_seminormAux k n x))⟩⟩⟩


@[simp]
theorem add_apply {f g : 𝓢(E, F)} {x : E} : (f + g) x = f x + g x :=
  rfl


theorem seminormAux_add_le (k n : ℕ) (f g : 𝓢(E, F)) :
    (f + g).seminormAux k n ≤ f.seminormAux k n + g.seminormAux k n :=
  (f + g).seminormAux_le_bound k n
    (add_nonneg (seminormAux_nonneg _ _ _) (seminormAux_nonneg _ _ _)) fun x =>
    (decay_add_le_aux k n f g x).trans <|
      add_le_add (f.le_seminormAux k n x) (g.le_seminormAux k n x)


instance instSub : Sub 𝓢(E, F) :=
  ⟨fun f g =>
    ⟨f - g, (f.smooth _).sub (g.smooth _), by
      /-
        𝕜 : Type u_1
        𝕜' : Type u_2
        D : Type u_3
        E : Type u_4
        F : Type u_5
        G : Type u_6
        V : Type u_7
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace Real E
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace Real F
        f g : SchwartzMap E F
        ⊢ ∀ (k n : Nat), Exists fun C => ∀ (x : E), LE.le (HMul.hMul (HPow.hPow (Norm. …
      -/
      intro k n
      /-
        𝕜 : Type u_1
        𝕜' : Type u_2
        D : Type u_3
        E : Type u_4
        F : Type u_5
        G : Type u_6
        V : Type u_7
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace Real E
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace Real F
        f g : SchwartzMap E F
        k n : Nat
        ⊢ Exists fun C => ∀ (x : E), LE.le (HMul.hMul (HPow.hPow (Norm.norm x) k) (Nor …
      -/
      refine ⟨f.seminormAux k n + g.seminormAux k n, fun x => ?_⟩
      /-
        𝕜 : Type u_1
        𝕜' : Type u_2
        D : Type u_3
        E : Type u_4
        F : Type u_5
        G : Type u_6
        V : Type u_7
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace Real E
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace Real F
        f g : SchwartzMap E F
        k n : Nat
        x : E
        ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm x) k) (Norm.norm (iteratedFDeriv Real …
      -/
      refine le_trans ?_ (add_le_add (f.le_seminormAux k n x) (g.le_seminormAux k n x))
      /-
        𝕜 : Type u_1
        𝕜' : Type u_2
        D : Type u_3
        E : Type u_4
        F : Type u_5
        G : Type u_6
        V : Type u_7
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace Real E
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace Real F
        f g : SchwartzMap E F
        k n : Nat
        x : E
        ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm x) k) (Norm.norm (iteratedFDeriv Real …
      -/
      rw [sub_eq_add_neg]
      /-
        𝕜 : Type u_1
        𝕜' : Type u_2
        D : Type u_3
        E : Type u_4
        F : Type u_5
        G : Type u_6
        V : Type u_7
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace Real E
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace Real F
        f g : SchwartzMap E F
        k n : Nat
        x : E
        ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm x) k) (Norm.norm (iteratedFDeriv Real …
      -/
      rw [← decay_neg_aux k n g x]
      /-
        𝕜 : Type u_1
        𝕜' : Type u_2
        D : Type u_3
        E : Type u_4
        F : Type u_5
        G : Type u_6
        V : Type u_7
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace Real E
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace Real F
        f g : SchwartzMap E F
        k n : Nat
        x : E
        ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm x) k) (Norm.norm (iteratedFDeriv Real …
      -/
      convert decay_add_le_aux k n f (-g) x⟩⟩
      /-
        🎉 no goals
      -/

-- exact fails with deterministic timeout

@[simp]
theorem sub_apply {f g : 𝓢(E, F)} {x : E} : (f - g) x = f x - g x :=
  rfl


instance instAddCommGroup : AddCommGroup 𝓢(E, F) :=
  DFunLike.coe_injective.addCommGroup _ rfl (fun _ _ => rfl) (fun _ => rfl) (fun _ _ => rfl)
    (fun _ _ => rfl) fun _ _ => rfl


/-- Coercion as an additive homomorphism. -/
def coeHom : 𝓢(E, F) →+ E → F where
  toFun f := f
  map_zero' := coe_zero
  map_add' _ _ := rfl


theorem coe_coeHom : (coeHom E F : 𝓢(E, F) → E → F) = DFunLike.coe :=
  rfl


theorem coeHom_injective : Function.Injective (coeHom E F) := by
  /-
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    ⊢ Function.Injective ⇑(SchwartzMap.coeHom E F)
  -/
  rw [coe_coeHom]
  /-
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    ⊢ Function.Injective DFunLike.coe
  -/
  exact DFunLike.coe_injective
  /-
    🎉 no goals
  -/


instance instModule : Module 𝕜 𝓢(E, F) :=
  coeHom_injective.module 𝕜 (coeHom E F) fun _ _ => rfl


/-- The seminorms of the Schwartz space given by the best constants in the definition of
`𝓢(E, F)`. -/
protected def seminorm (k n : ℕ) : Seminorm 𝕜 𝓢(E, F) :=
  Seminorm.ofSMulLE (SchwartzMap.seminormAux k n) (seminormAux_zero k n) (seminormAux_add_le k n)
    (seminormAux_smul_le k n)


/-- If one controls the seminorm for every `x`, then one controls the seminorm. -/
theorem seminorm_le_bound (k n : ℕ) (f : 𝓢(E, F)) {M : ℝ} (hMp : 0 ≤ M)
    (hM : ∀ x, ‖x‖ ^ k * ‖iteratedFDeriv ℝ n f x‖ ≤ M) : SchwartzMap.seminorm 𝕜 k n f ≤ M :=
  f.seminormAux_le_bound k n hMp hM


/-- If one controls the seminorm for every `x`, then one controls the seminorm.

Variant for functions `𝓢(ℝ, F)`. -/
theorem seminorm_le_bound' (k n : ℕ) (f : 𝓢(ℝ, F)) {M : ℝ} (hMp : 0 ≤ M)
    (hM : ∀ x, |x| ^ k * ‖iteratedDeriv n f x‖ ≤ M) : SchwartzMap.seminorm 𝕜 k n f ≤ M := by
  /-
    𝕜 : Type u_1
    F : Type u_5
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    k n : Nat
    f : SchwartzMap Real F
    M : Real
    hMp : LE.le 0 M
    hM : ∀ (x : Real), LE.le (HMul.hMul (HPow.hPow (abs x) k) (Norm.norm (iterated …
    ⊢ LE.le ((SchwartzMap.seminorm 𝕜 k n) f) M
  -/
  refine seminorm_le_bound 𝕜 k n f hMp ?_
  /-
    𝕜 : Type u_1
    F : Type u_5
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    k n : Nat
    f : SchwartzMap Real F
    M : Real
    hMp : LE.le 0 M
    hM : ∀ (x : Real), LE.le (HMul.hMul (HPow.hPow (abs x) k) (Norm.norm (iterated …
    ⊢ ∀ (x : Real), LE.le (HMul.hMul (HPow.hPow (Norm.norm x) k) (Norm.norm (itera …
  -/
  simpa only [Real.norm_eq_abs, norm_iteratedFDeriv_eq_norm_iteratedDeriv]
  /-
    🎉 no goals
  -/


/-- The seminorm controls the Schwartz estimate for any fixed `x`. -/
theorem le_seminorm (k n : ℕ) (f : 𝓢(E, F)) (x : E) :
    ‖x‖ ^ k * ‖iteratedFDeriv ℝ n f x‖ ≤ SchwartzMap.seminorm 𝕜 k n f :=
  f.le_seminormAux k n x


/-- The seminorm controls the Schwartz estimate for any fixed `x`.

Variant for functions `𝓢(ℝ, F)`. -/
theorem le_seminorm' (k n : ℕ) (f : 𝓢(ℝ, F)) (x : ℝ) :
    |x| ^ k * ‖iteratedDeriv n f x‖ ≤ SchwartzMap.seminorm 𝕜 k n f := by
  /-
    𝕜 : Type u_1
    F : Type u_5
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    k n : Nat
    f : SchwartzMap Real F
    x : Real
    ⊢ LE.le (HMul.hMul (HPow.hPow (abs x) k) (Norm.norm (iteratedDeriv n (⇑f) x))) …
  -/
  have := le_seminorm 𝕜 k n f x
  /-
    𝕜 : Type u_1
    F : Type u_5
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    k n : Nat
    f : SchwartzMap Real F
    x : Real
    this : LE.le (HMul.hMul (HPow.hPow (Norm.norm x) k) (Norm.norm (iteratedFDeriv …
    ⊢ LE.le (HMul.hMul (HPow.hPow (abs x) k) (Norm.norm (iteratedDeriv n (⇑f) x))) …
  -/
  rwa [← Real.norm_eq_abs, ← norm_iteratedFDeriv_eq_norm_iteratedDeriv]
  /-
    🎉 no goals
  -/


theorem norm_iteratedFDeriv_le_seminorm (f : 𝓢(E, F)) (n : ℕ) (x₀ : E) :
    ‖iteratedFDeriv ℝ n f x₀‖ ≤ (SchwartzMap.seminorm 𝕜 0 n) f := by
  /-
    𝕜 : Type u_1
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    f : SchwartzMap E F
    n : Nat
    x₀ : E
    ⊢ LE.le (Norm.norm (iteratedFDeriv Real n (⇑f) x₀)) ((SchwartzMap.seminorm 𝕜 0 …
  -/
  have := SchwartzMap.le_seminorm 𝕜 0 n f x₀
  /-
    𝕜 : Type u_1
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    f : SchwartzMap E F
    n : Nat
    x₀ : E
    this : LE.le (HMul.hMul (HPow.hPow (Norm.norm x₀) 0) (Norm.norm (iteratedFDeri …
    ⊢ LE.le (Norm.norm (iteratedFDeriv Real n (⇑f) x₀)) ((SchwartzMap.seminorm 𝕜 0 …
  -/
  rwa [pow_zero, one_mul] at this
  /-
    🎉 no goals
  -/


theorem norm_pow_mul_le_seminorm (f : 𝓢(E, F)) (k : ℕ) (x₀ : E) :
    ‖x₀‖ ^ k * ‖f x₀‖ ≤ (SchwartzMap.seminorm 𝕜 k 0) f := by
  /-
    𝕜 : Type u_1
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    f : SchwartzMap E F
    k : Nat
    x₀ : E
    ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm x₀) k) (Norm.norm (f x₀))) ((Schwartz …
  -/
  have := SchwartzMap.le_seminorm 𝕜 k 0 f x₀
  /-
    𝕜 : Type u_1
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    f : SchwartzMap E F
    k : Nat
    x₀ : E
    this : LE.le (HMul.hMul (HPow.hPow (Norm.norm x₀) k) (Norm.norm (iteratedFDeri …
    ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm x₀) k) (Norm.norm (f x₀))) ((Schwartz …
  -/
  rwa [norm_iteratedFDeriv_zero] at this
  /-
    🎉 no goals
  -/


theorem norm_le_seminorm (f : 𝓢(E, F)) (x₀ : E) : ‖f x₀‖ ≤ (SchwartzMap.seminorm 𝕜 0 0) f := by
  /-
    𝕜 : Type u_1
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    f : SchwartzMap E F
    x₀ : E
    ⊢ LE.le (Norm.norm (f x₀)) ((SchwartzMap.seminorm 𝕜 0 0) f)
  -/
  have := norm_pow_mul_le_seminorm 𝕜 f 0 x₀
  /-
    𝕜 : Type u_1
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    f : SchwartzMap E F
    x₀ : E
    this : LE.le (HMul.hMul (HPow.hPow (Norm.norm x₀) 0) (Norm.norm (f x₀))) ((Sch …
    ⊢ LE.le (Norm.norm (f x₀)) ((SchwartzMap.seminorm 𝕜 0 0) f)
  -/
  rwa [pow_zero, one_mul] at this
  /-
    🎉 no goals
  -/


/-- The family of Schwartz seminorms. -/
def _root_.schwartzSeminormFamily : SeminormFamily 𝕜 𝓢(E, F) (ℕ × ℕ) :=
  fun m => SchwartzMap.seminorm 𝕜 m.1 m.2


@[simp]
theorem schwartzSeminormFamily_apply (n k : ℕ) :
    schwartzSeminormFamily 𝕜 E F (n, k) = SchwartzMap.seminorm 𝕜 n k :=
  rfl


@[simp]
theorem schwartzSeminormFamily_apply_zero :
    schwartzSeminormFamily 𝕜 E F 0 = SchwartzMap.seminorm 𝕜 0 0 :=
  rfl


/-- A more convenient version of `le_sup_seminorm_apply`.

The set `Finset.Iic m` is the set of all pairs `(k', n')` with `k' ≤ m.1` and `n' ≤ m.2`.
Note that the constant is far from optimal. -/
theorem one_add_le_sup_seminorm_apply {m : ℕ × ℕ} {k n : ℕ} (hk : k ≤ m.1) (hn : n ≤ m.2)
    (f : 𝓢(E, F)) (x : E) :
    (1 + ‖x‖) ^ k * ‖iteratedFDeriv ℝ n f x‖ ≤
      2 ^ m.1 * (Finset.Iic m).sup (fun m => SchwartzMap.seminorm 𝕜 m.1 m.2) f := by
  /-
    𝕜 : Type u_1
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    m : Prod Nat Nat
    k n : Nat
    hk : LE.le k m.1
    hn : LE.le n m.2
    f : SchwartzMap E F
    x : E
    ⊢ LE.le (HMul.hMul (HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) k) (Norm.norm (itera …
  -/
  rw [add_comm, add_pow]
  /-
    𝕜 : Type u_1
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    m : Prod Nat Nat
    k n : Nat
    hk : LE.le k m.1
    hn : LE.le n m.2
    f : SchwartzMap E F
    x : E
    ⊢ LE.le (HMul.hMul ((Finset.range (HAdd.hAdd k 1)).sum fun m => HMul.hMul (HMu …
  -/
  simp only [one_pow, mul_one, Finset.sum_congr, Finset.sum_mul]
  /-
    𝕜 : Type u_1
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    m : Prod Nat Nat
    k n : Nat
    hk : LE.le k m.1
    hn : LE.le n m.2
    f : SchwartzMap E F
    x : E
    ⊢ LE.le ((Finset.range (HAdd.hAdd k 1)).sum fun i => HMul.hMul (HMul.hMul (HPo …
  -/
  norm_cast
  /-
    𝕜 : Type u_1
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    m : Prod Nat Nat
    k n : Nat
    hk : LE.le k m.1
    hn : LE.le n m.2
    f : SchwartzMap E F
    x : E
    ⊢ LE.le ((Finset.range (HAdd.hAdd k 1)).sum fun i => HMul.hMul (HMul.hMul (HPo …
  -/
  rw [← Nat.sum_range_choose m.1]
  /-
    𝕜 : Type u_1
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    m : Prod Nat Nat
    k n : Nat
    hk : LE.le k m.1
    hn : LE.le n m.2
    f : SchwartzMap E F
    x : E
    ⊢ LE.le ((Finset.range (HAdd.hAdd k 1)).sum fun i => HMul.hMul (HMul.hMul (HPo …
  -/
  push_cast
  /-
    𝕜 : Type u_1
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    m : Prod Nat Nat
    k n : Nat
    hk : LE.le k m.1
    hn : LE.le n m.2
    f : SchwartzMap E F
    x : E
    ⊢ LE.le ((Finset.range (HAdd.hAdd k 1)).sum fun i => HMul.hMul (HMul.hMul (HPo …
  -/
  rw [Finset.sum_mul]
  have hk' : Finset.range (k + 1) ⊆ Finset.range (m.1 + 1) := by
    rwa [Finset.range_subset, add_le_add_iff_right]
  /-
    𝕜 : Type u_1
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    m : Prod Nat Nat
    k n : Nat
    hk : LE.le k m.1
    hn : LE.le n m.2
    f : SchwartzMap E F
    x : E
    hk' : HasSubset.Subset (Finset.range (HAdd.hAdd k 1)) (Finset.range (HAdd.hAdd …
    ⊢ LE.le ((Finset.range (HAdd.hAdd k 1)).sum fun i => HMul.hMul (HMul.hMul (HPo …
  -/
  refine le_trans (Finset.sum_le_sum_of_subset_of_nonneg hk' fun _ _ _ => by positivity) ?_
  /-
    𝕜 : Type u_1
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    m : Prod Nat Nat
    k n : Nat
    hk : LE.le k m.1
    hn : LE.le n m.2
    f : SchwartzMap E F
    x : E
    hk' : HasSubset.Subset (Finset.range (HAdd.hAdd k 1)) (Finset.range (HAdd.hAdd …
    ⊢ LE.le ((Finset.range (HAdd.hAdd m.1 1)).sum fun i => HMul.hMul (HMul.hMul (H …
  -/
  gcongr ∑ _i ∈ Finset.range (m.1 + 1), ?_ with i hi
  /-
    case h
    𝕜 : Type u_1
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    m : Prod Nat Nat
    k n : Nat
    hk : LE.le k m.1
    hn : LE.le n m.2
    f : SchwartzMap E F
    x : E
    hk' : HasSubset.Subset (Finset.range (HAdd.hAdd k 1)) (Finset.range (HAdd.hAdd …
    i : Nat
    hi : Membership.mem (Finset.range (HAdd.hAdd m.1 1)) i
    ⊢ LE.le (HMul.hMul (HMul.hMul (HPow.hPow (Norm.norm x) i) ↑(k.choose i)) (Norm …
  -/
  move_mul [(Nat.choose k i : ℝ), (Nat.choose m.1 i : ℝ)]
  /-
    case h
    𝕜 : Type u_1
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    m : Prod Nat Nat
    k n : Nat
    hk : LE.le k m.1
    hn : LE.le n m.2
    f : SchwartzMap E F
    x : E
    hk' : HasSubset.Subset (Finset.range (HAdd.hAdd k 1)) (Finset.range (HAdd.hAdd …
    i : Nat
    hi : Membership.mem (Finset.range (HAdd.hAdd m.1 1)) i
    ⊢ LE.le (HMul.hMul (HMul.hMul (HPow.hPow (Norm.norm x) i) (Norm.norm (iterated …
  -/
  gcongr
    /-
      case h.h₁
      𝕜 : Type u_1
      E : Type u_4
      F : Type u_5
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace Real E
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NormedSpace Real F
      inst✝² : NormedField 𝕜
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : SMulCommClass Real 𝕜 F
      m : Prod Nat Nat
      k n : Nat
      hk : LE.le k m.1
      hn : LE.le n m.2
      f : SchwartzMap E F
      x : E
      hk' : HasSubset.Subset (Finset.range (HAdd.hAdd k 1)) (Finset.range (HAdd.hAdd …
      i : Nat
      hi : Membership.mem (Finset.range (HAdd.hAdd m.1 1)) i
      ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm x) i) (Norm.norm (iteratedFDeriv Real …
    -/
  · apply (le_seminorm 𝕜 i n f x).trans
    /-
      case h.h₁
      𝕜 : Type u_1
      E : Type u_4
      F : Type u_5
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace Real E
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NormedSpace Real F
      inst✝² : NormedField 𝕜
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : SMulCommClass Real 𝕜 F
      m : Prod Nat Nat
      k n : Nat
      hk : LE.le k m.1
      hn : LE.le n m.2
      f : SchwartzMap E F
      x : E
      hk' : HasSubset.Subset (Finset.range (HAdd.hAdd k 1)) (Finset.range (HAdd.hAdd …
      i : Nat
      hi : Membership.mem (Finset.range (HAdd.hAdd m.1 1)) i
      ⊢ LE.le ((SchwartzMap.seminorm 𝕜 i n) f) (((Finset.Iic m).sup fun m => Schwart …
    -/
    apply Seminorm.le_def.1
    exact Finset.le_sup_of_le (Finset.mem_Iic.2 <|
      Prod.mk_le_mk.2 ⟨Finset.mem_range_succ_iff.mp hi, hn⟩) le_rfl
    /-
      case h.h₂.h
      𝕜 : Type u_1
      E : Type u_4
      F : Type u_5
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace Real E
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NormedSpace Real F
      inst✝² : NormedField 𝕜
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : SMulCommClass Real 𝕜 F
      m : Prod Nat Nat
      k n : Nat
      hk : LE.le k m.1
      hn : LE.le n m.2
      f : SchwartzMap E F
      x : E
      hk' : HasSubset.Subset (Finset.range (HAdd.hAdd k 1)) (Finset.range (HAdd.hAdd …
      i : Nat
      hi : Membership.mem (Finset.range (HAdd.hAdd m.1 1)) i
      ⊢ LE.le (k.choose i) (m.1.choose i)
    -/
  · exact mod_cast Nat.choose_le_choose i hk
    /-
      🎉 no goals
    -/


instance instTopologicalSpace : TopologicalSpace 𝓢(E, F) :=
  (schwartzSeminormFamily ℝ E F).moduleFilterBasis.topology'


theorem _root_.schwartz_withSeminorms : WithSeminorms (schwartzSeminormFamily 𝕜 E F) := by
  /-
    𝕜 : Type u_1
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    ⊢ WithSeminorms (schwartzSeminormFamily 𝕜 E F)
  -/
  have A : WithSeminorms (schwartzSeminormFamily ℝ E F) := ⟨rfl⟩
  /-
    𝕜 : Type u_1
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    A : WithSeminorms (schwartzSeminormFamily Real E F)
    ⊢ WithSeminorms (schwartzSeminormFamily 𝕜 E F)
  -/
  rw [SeminormFamily.withSeminorms_iff_nhds_eq_iInf] at A ⊢
  /-
    𝕜 : Type u_1
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    A : Eq (nhds 0) (iInf fun i => Filter.comap (⇑(schwartzSeminormFamily Real E F …
    ⊢ Eq (nhds 0) (iInf fun i => Filter.comap (⇑(schwartzSeminormFamily 𝕜 E F i))  …
  -/
  rw [A]
  /-
    𝕜 : Type u_1
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    A : Eq (nhds 0) (iInf fun i => Filter.comap (⇑(schwartzSeminormFamily Real E F …
    ⊢ Eq (iInf fun i => Filter.comap (⇑(schwartzSeminormFamily Real E F i)) (nhds  …
  -/
  rfl
  /-
    🎉 no goals
  -/


instance instContinuousSMul : ContinuousSMul 𝕜 𝓢(E, F) := by
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    ⊢ ContinuousSMul 𝕜 (SchwartzMap E F)
  -/
  rw [(schwartz_withSeminorms 𝕜 E F).withSeminorms_eq]
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    ⊢ ContinuousSMul 𝕜 (SchwartzMap E F)
  -/
  exact (schwartzSeminormFamily 𝕜 E F).moduleFilterBasis.continuousSMul
  /-
    🎉 no goals
  -/


instance instTopologicalAddGroup : TopologicalAddGroup 𝓢(E, F) :=
  (schwartzSeminormFamily ℝ E F).addGroupFilterBasis.isTopologicalAddGroup


instance instUniformSpace : UniformSpace 𝓢(E, F) :=
  (schwartzSeminormFamily ℝ E F).addGroupFilterBasis.uniformSpace


instance instUniformAddGroup : UniformAddGroup 𝓢(E, F) :=
  (schwartzSeminormFamily ℝ E F).addGroupFilterBasis.uniformAddGroup


instance instLocallyConvexSpace : LocallyConvexSpace ℝ 𝓢(E, F) :=
  (schwartz_withSeminorms ℝ E F).toLocallyConvexSpace


instance instFirstCountableTopology : FirstCountableTopology 𝓢(E, F) :=
  (schwartz_withSeminorms ℝ E F).firstCountableTopology


/-- A function is called of temperate growth if it is smooth and all iterated derivatives are
polynomially bounded. -/
def _root_.Function.HasTemperateGrowth (f : E → F) : Prop :=
  ContDiff ℝ ∞ f ∧ ∀ n : ℕ, ∃ (k : ℕ) (C : ℝ), ∀ x, ‖iteratedFDeriv ℝ n f x‖ ≤ C * (1 + ‖x‖) ^ k


theorem _root_.Function.HasTemperateGrowth.norm_iteratedFDeriv_le_uniform_aux {f : E → F}
    (hf_temperate : f.HasTemperateGrowth) (n : ℕ) :
    ∃ (k : ℕ) (C : ℝ), 0 ≤ C ∧ ∀ N ≤ n, ∀ x : E, ‖iteratedFDeriv ℝ N f x‖ ≤ C * (1 + ‖x‖) ^ k := by
  /-
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : E → F
    hf_temperate : Function.HasTemperateGrowth f
    n : Nat
    ⊢ Exists fun k => Exists fun C => And (LE.le 0 C) (∀ (N : Nat), LE.le N n → ∀  …
  -/
  choose k C f using hf_temperate.2
  /-
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f✝ : E → F
    hf_temperate : Function.HasTemperateGrowth f✝
    n : Nat
    k : Nat → Nat
    C : Nat → Real
    f : ∀ (n : Nat) (x : E), LE.le (Norm.norm (iteratedFDeriv Real n f✝ x)) (HMul. …
    ⊢ Exists fun k => Exists fun C => And (LE.le 0 C) (∀ (N : Nat), LE.le N n → ∀  …
  -/
  use (Finset.range (n + 1)).sup k
  /-
    case h
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f✝ : E → F
    hf_temperate : Function.HasTemperateGrowth f✝
    n : Nat
    k : Nat → Nat
    C : Nat → Real
    f : ∀ (n : Nat) (x : E), LE.le (Norm.norm (iteratedFDeriv Real n f✝ x)) (HMul. …
    ⊢ Exists fun C => And (LE.le 0 C) (∀ (N : Nat), LE.le N n → ∀ (x : E), LE.le ( …
  -/
  let C' := max (0 : ℝ) ((Finset.range (n + 1)).sup' (by simp) C)
  /-
    case h
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f✝ : E → F
    hf_temperate : Function.HasTemperateGrowth f✝
    n : Nat
    k : Nat → Nat
    C : Nat → Real
    f : ∀ (n : Nat) (x : E), LE.le (Norm.norm (iteratedFDeriv Real n f✝ x)) (HMul. …
    C' : Real := Max.max 0 ((Finset.range (HAdd.hAdd n 1)).sup' ⋯ C)
    ⊢ Exists fun C => And (LE.le 0 C) (∀ (N : Nat), LE.le N n → ∀ (x : E), LE.le ( …
  -/
  have hC' : 0 ≤ C' := by simp only [C', le_refl, Finset.le_sup'_iff, true_or, le_max_iff]
  /-
    case h
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f✝ : E → F
    hf_temperate : Function.HasTemperateGrowth f✝
    n : Nat
    k : Nat → Nat
    C : Nat → Real
    f : ∀ (n : Nat) (x : E), LE.le (Norm.norm (iteratedFDeriv Real n f✝ x)) (HMul. …
    C' : Real := Max.max 0 ((Finset.range (HAdd.hAdd n 1)).sup' ⋯ C)
    hC' : LE.le 0 C'
    ⊢ Exists fun C => And (LE.le 0 C) (∀ (N : Nat), LE.le N n → ∀ (x : E), LE.le ( …
  -/
  use C', hC'
  /-
    case right
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f✝ : E → F
    hf_temperate : Function.HasTemperateGrowth f✝
    n : Nat
    k : Nat → Nat
    C : Nat → Real
    f : ∀ (n : Nat) (x : E), LE.le (Norm.norm (iteratedFDeriv Real n f✝ x)) (HMul. …
    C' : Real := Max.max 0 ((Finset.range (HAdd.hAdd n 1)).sup' ⋯ C)
    hC' : LE.le 0 C'
    ⊢ ∀ (N : Nat), LE.le N n → ∀ (x : E), LE.le (Norm.norm (iteratedFDeriv Real N  …
  -/
  intro N hN x
  /-
    case right
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f✝ : E → F
    hf_temperate : Function.HasTemperateGrowth f✝
    n : Nat
    k : Nat → Nat
    C : Nat → Real
    f : ∀ (n : Nat) (x : E), LE.le (Norm.norm (iteratedFDeriv Real n f✝ x)) (HMul. …
    C' : Real := Max.max 0 ((Finset.range (HAdd.hAdd n 1)).sup' ⋯ C)
    hC' : LE.le 0 C'
    N : Nat
    hN : LE.le N n
    x : E
    ⊢ LE.le (Norm.norm (iteratedFDeriv Real N f✝ x)) (HMul.hMul C' (HPow.hPow (HAd …
  -/
  rw [← Finset.mem_range_succ_iff] at hN
  /-
    case right
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f✝ : E → F
    hf_temperate : Function.HasTemperateGrowth f✝
    n : Nat
    k : Nat → Nat
    C : Nat → Real
    f : ∀ (n : Nat) (x : E), LE.le (Norm.norm (iteratedFDeriv Real n f✝ x)) (HMul. …
    C' : Real := Max.max 0 ((Finset.range (HAdd.hAdd n 1)).sup' ⋯ C)
    hC' : LE.le 0 C'
    N : Nat
    hN : Membership.mem (Finset.range n.succ) N
    x : E
    ⊢ LE.le (Norm.norm (iteratedFDeriv Real N f✝ x)) (HMul.hMul C' (HPow.hPow (HAd …
  -/
  refine le_trans (f N x) (mul_le_mul ?_ ?_ (by positivity) hC')
    /-
      case right.refine_1
      E : Type u_4
      F : Type u_5
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace Real F
      f✝ : E → F
      hf_temperate : Function.HasTemperateGrowth f✝
      n : Nat
      k : Nat → Nat
      C : Nat → Real
      f : ∀ (n : Nat) (x : E), LE.le (Norm.norm (iteratedFDeriv Real n f✝ x)) (HMul. …
      C' : Real := Max.max 0 ((Finset.range (HAdd.hAdd n 1)).sup' ⋯ C)
      hC' : LE.le 0 C'
      N : Nat
      hN : Membership.mem (Finset.range n.succ) N
      x : E
      ⊢ LE.le (C N) C'
    -/
  · simp only [C', Finset.le_sup'_iff, le_max_iff]
    /-
      case right.refine_1
      E : Type u_4
      F : Type u_5
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace Real F
      f✝ : E → F
      hf_temperate : Function.HasTemperateGrowth f✝
      n : Nat
      k : Nat → Nat
      C : Nat → Real
      f : ∀ (n : Nat) (x : E), LE.le (Norm.norm (iteratedFDeriv Real n f✝ x)) (HMul. …
      C' : Real := Max.max 0 ((Finset.range (HAdd.hAdd n 1)).sup' ⋯ C)
      hC' : LE.le 0 C'
      N : Nat
      hN : Membership.mem (Finset.range n.succ) N
      x : E
      ⊢ Or (LE.le (C N) 0) (Exists fun b => And (Membership.mem (Finset.range (HAdd. …
    -/
    right
    /-
      case right.refine_1.h
      E : Type u_4
      F : Type u_5
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace Real F
      f✝ : E → F
      hf_temperate : Function.HasTemperateGrowth f✝
      n : Nat
      k : Nat → Nat
      C : Nat → Real
      f : ∀ (n : Nat) (x : E), LE.le (Norm.norm (iteratedFDeriv Real n f✝ x)) (HMul. …
      C' : Real := Max.max 0 ((Finset.range (HAdd.hAdd n 1)).sup' ⋯ C)
      hC' : LE.le 0 C'
      N : Nat
      hN : Membership.mem (Finset.range n.succ) N
      x : E
      ⊢ Exists fun b => And (Membership.mem (Finset.range (HAdd.hAdd n 1)) b) (LE.le …
    -/
    exact ⟨N, hN, rfl.le⟩
    /-
      🎉 no goals
    -/
  /-
    case right.refine_2
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f✝ : E → F
    hf_temperate : Function.HasTemperateGrowth f✝
    n : Nat
    k : Nat → Nat
    C : Nat → Real
    f : ∀ (n : Nat) (x : E), LE.le (Norm.norm (iteratedFDeriv Real n f✝ x)) (HMul. …
    C' : Real := Max.max 0 ((Finset.range (HAdd.hAdd n 1)).sup' ⋯ C)
    hC' : LE.le 0 C'
    N : Nat
    hN : Membership.mem (Finset.range n.succ) N
    x : E
    ⊢ LE.le (HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) (k N)) (HPow.hPow (HAdd.hAdd 1  …
  -/
  gcongr
    /-
      case right.refine_2.ha
      E : Type u_4
      F : Type u_5
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace Real F
      f✝ : E → F
      hf_temperate : Function.HasTemperateGrowth f✝
      n : Nat
      k : Nat → Nat
      C : Nat → Real
      f : ∀ (n : Nat) (x : E), LE.le (Norm.norm (iteratedFDeriv Real n f✝ x)) (HMul. …
      C' : Real := Max.max 0 ((Finset.range (HAdd.hAdd n 1)).sup' ⋯ C)
      hC' : LE.le 0 C'
      N : Nat
      hN : Membership.mem (Finset.range n.succ) N
      x : E
      ⊢ LE.le 1 (HAdd.hAdd 1 (Norm.norm x))
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case right.refine_2.hmn
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f✝ : E → F
    hf_temperate : Function.HasTemperateGrowth f✝
    n : Nat
    k : Nat → Nat
    C : Nat → Real
    f : ∀ (n : Nat) (x : E), LE.le (Norm.norm (iteratedFDeriv Real n f✝ x)) (HMul. …
    C' : Real := Max.max 0 ((Finset.range (HAdd.hAdd n 1)).sup' ⋯ C)
    hC' : LE.le 0 C'
    N : Nat
    hN : Membership.mem (Finset.range n.succ) N
    x : E
    ⊢ LE.le (k N) ((Finset.range (HAdd.hAdd n 1)).sup k)
  -/
  exact Finset.le_sup hN
  /-
    🎉 no goals
  -/


lemma _root_.Function.HasTemperateGrowth.of_fderiv {f : E → F}
    (h'f : Function.HasTemperateGrowth (fderiv ℝ f)) (hf : Differentiable ℝ f) {k : ℕ} {C : ℝ}
    (h : ∀ x, ‖f x‖ ≤ C * (1 + ‖x‖) ^ k) :
    Function.HasTemperateGrowth f := by
  /-
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : E → F
    h'f : Function.HasTemperateGrowth (fderiv Real f)
    hf : Differentiable Real f
    k : Nat
    C : Real
    h : ∀ (x : E), LE.le (Norm.norm (f x)) (HMul.hMul C (HPow.hPow (HAdd.hAdd 1 (N …
    ⊢ Function.HasTemperateGrowth f
  -/
  refine ⟨contDiff_succ_iff_fderiv.2 ⟨hf, by simp, h'f.1⟩ , fun n ↦ ?_⟩
  /-
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : E → F
    h'f : Function.HasTemperateGrowth (fderiv Real f)
    hf : Differentiable Real f
    k : Nat
    C : Real
    h : ∀ (x : E), LE.le (Norm.norm (f x)) (HMul.hMul C (HPow.hPow (HAdd.hAdd 1 (N …
    n : Nat
    ⊢ Exists fun k => Exists fun C => ∀ (x : E), LE.le (Norm.norm (iteratedFDeriv  …
  -/
  rcases n with rfl|m
    /-
      case zero
      E : Type u_4
      F : Type u_5
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace Real F
      f : E → F
      h'f : Function.HasTemperateGrowth (fderiv Real f)
      hf : Differentiable Real f
      k : Nat
      C : Real
      h : ∀ (x : E), LE.le (Norm.norm (f x)) (HMul.hMul C (HPow.hPow (HAdd.hAdd 1 (N …
      ⊢ Exists fun k => Exists fun C => ∀ (x : E), LE.le (Norm.norm (iteratedFDeriv  …
    -/
  · exact ⟨k, C, fun x ↦ by simpa using h x⟩
    /-
      🎉 no goals
    -/
    /-
      case succ
      E : Type u_4
      F : Type u_5
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace Real F
      f : E → F
      h'f : Function.HasTemperateGrowth (fderiv Real f)
      hf : Differentiable Real f
      k : Nat
      C : Real
      h : ∀ (x : E), LE.le (Norm.norm (f x)) (HMul.hMul C (HPow.hPow (HAdd.hAdd 1 (N …
      m : Nat
      ⊢ Exists fun k => Exists fun C => ∀ (x : E), LE.le (Norm.norm (iteratedFDeriv  …
    -/
  · rcases h'f.2 m with ⟨k', C', h'⟩
    /-
      case succ.intro.intro
      E : Type u_4
      F : Type u_5
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace Real F
      f : E → F
      h'f : Function.HasTemperateGrowth (fderiv Real f)
      hf : Differentiable Real f
      k : Nat
      C : Real
      h : ∀ (x : E), LE.le (Norm.norm (f x)) (HMul.hMul C (HPow.hPow (HAdd.hAdd 1 (N …
      m k' : Nat
      C' : Real
      h' : ∀ (x : E), LE.le (Norm.norm (iteratedFDeriv Real m (fderiv Real f) x)) (H …
      ⊢ Exists fun k => Exists fun C => ∀ (x : E), LE.le (Norm.norm (iteratedFDeriv  …
    -/
    refine ⟨k', C', ?_⟩
    /-
      case succ.intro.intro
      E : Type u_4
      F : Type u_5
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace Real F
      f : E → F
      h'f : Function.HasTemperateGrowth (fderiv Real f)
      hf : Differentiable Real f
      k : Nat
      C : Real
      h : ∀ (x : E), LE.le (Norm.norm (f x)) (HMul.hMul C (HPow.hPow (HAdd.hAdd 1 (N …
      m k' : Nat
      C' : Real
      h' : ∀ (x : E), LE.le (Norm.norm (iteratedFDeriv Real m (fderiv Real f) x)) (H …
      ⊢ ∀ (x : E), LE.le (Norm.norm (iteratedFDeriv Real (HAdd.hAdd m 1) f x)) (HMul …
    -/
    simpa [iteratedFDeriv_succ_eq_comp_right] using h'
    /-
      🎉 no goals
    -/


lemma _root_.Function.HasTemperateGrowth.zero :
    Function.HasTemperateGrowth (fun _ : E ↦ (0 : F)) := by
  /-
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    ⊢ Function.HasTemperateGrowth fun x => 0
  -/
  refine ⟨contDiff_const, fun n ↦ ⟨0, 0, fun x ↦ ?_⟩⟩
  /-
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    n : Nat
    x : E
    ⊢ LE.le (Norm.norm (iteratedFDeriv Real n (fun x => 0) x)) (HMul.hMul 0 (HPow. …
  -/
  simp only [iteratedFDeriv_zero_fun, Pi.zero_apply, norm_zero, forall_const]
  /-
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    n : Nat
    x : E
    ⊢ LE.le 0 (HMul.hMul 0 (HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) 0))
  -/
  positivity
  /-
    🎉 no goals
  -/


lemma _root_.Function.HasTemperateGrowth.const (c : F) :
    Function.HasTemperateGrowth (fun _ : E ↦ c) :=
                 /-
                   E : Type u_4
                   F : Type u_5
                   inst✝³ : NormedAddCommGroup E
                   inst✝² : NormedSpace Real E
                   inst✝¹ : NormedAddCommGroup F
                   inst✝ : NormedSpace Real F
                   c : F
                   ⊢ Function.HasTemperateGrowth (fderiv Real fun x => c)
                 -/
                 /-
                   🎉 no goals
                 -/
  .of_fderiv (by simpa using .zero) (differentiable_const c) (k := 0) (C := ‖c‖) (fun x ↦ by simp)
                                                                                             /-
                                                                                               🎉 no goals
                                                                                             -/


lemma _root_.ContinuousLinearMap.hasTemperateGrowth (f : E →L[ℝ] F) :
    Function.HasTemperateGrowth f := by
  /-
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : ContinuousLinearMap (RingHom.id Real) E F
    ⊢ Function.HasTemperateGrowth ⇑f
  -/
  apply Function.HasTemperateGrowth.of_fderiv ?_ f.differentiable (k := 1) (C := ‖f‖) (fun x ↦ ?_)
    /-
      E : Type u_4
      F : Type u_5
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace Real F
      f : ContinuousLinearMap (RingHom.id Real) E F
      ⊢ Function.HasTemperateGrowth (fderiv Real ⇑f)
    -/
  · have : fderiv ℝ f = fun _ ↦ f := by ext1 v; simp only [ContinuousLinearMap.fderiv]
    /-
      E : Type u_4
      F : Type u_5
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace Real F
      f : ContinuousLinearMap (RingHom.id Real) E F
      this : Eq (fderiv Real ⇑f) fun x => f
      ⊢ Function.HasTemperateGrowth (fderiv Real ⇑f)
    -/
    simpa [this] using .const _
    /-
      🎉 no goals
    -/
    /-
      E : Type u_4
      F : Type u_5
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace Real F
      f : ContinuousLinearMap (RingHom.id Real) E F
      x : E
      ⊢ LE.le (Norm.norm (f x)) (HMul.hMul (Norm.norm f) (HPow.hPow (HAdd.hAdd 1 (No …
    -/
  · exact (f.le_opNorm x).trans (by simp [mul_add])
    /-
      🎉 no goals
    -/


/-- A measure `μ` has temperate growth if there is an `n : ℕ` such that `(1 + ‖x‖) ^ (- n)` is
`μ`-integrable. -/
class _root_.MeasureTheory.Measure.HasTemperateGrowth (μ : Measure D) : Prop where
  exists_integrable : ∃ (n : ℕ), Integrable (fun x ↦ (1 + ‖x‖) ^ (- (n : ℝ))) μ


open Classical in
/-- An integer exponent `l` such that `(1 + ‖x‖) ^ (-l)` is integrable if `μ` has
temperate growth. -/
def _root_.MeasureTheory.Measure.integrablePower (μ : Measure D) : ℕ :=
  if h : μ.HasTemperateGrowth then h.exists_integrable.choose else 0


lemma integrable_pow_neg_integrablePower
    (μ : Measure D) [h : μ.HasTemperateGrowth] :
    Integrable (fun x ↦ (1 + ‖x‖) ^ (- (μ.integrablePower : ℝ))) μ := by
  /-
    D : Type u_3
    inst✝¹ : NormedAddCommGroup D
    inst✝ : MeasurableSpace D
    μ : MeasureTheory.Measure D
    h : μ.HasTemperateGrowth
    ⊢ MeasureTheory.Integrable (fun x => HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) (Ne …
  -/
  simpa [Measure.integrablePower, h] using h.exists_integrable.choose_spec
  /-
    🎉 no goals
  -/


instance _root_.MeasureTheory.Measure.IsFiniteMeasure.instHasTemperateGrowth {μ : Measure D}
                                                              /-
                                                                𝕜 : Type u_1
                                                                𝕜' : Type u_2
                                                                D : Type u_3
                                                                E : Type u_4
                                                                F : Type u_5
                                                                G : Type u_6
                                                                V : Type u_7
                                                                inst✝⁵ : NormedAddCommGroup E
                                                                inst✝⁴ : NormedSpace Real E
                                                                inst✝³ : NormedAddCommGroup F
                                                                inst✝² : NormedSpace Real F
                                                                inst✝¹ : NormedAddCommGroup D
                                                                inst✝ : MeasurableSpace D
                                                                μ : MeasureTheory.Measure D
                                                                h : MeasureTheory.IsFiniteMeasure μ
                                                                ⊢ MeasureTheory.Integrable (fun x => HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) (Ne …
                                                              -/
    [h : IsFiniteMeasure μ] : μ.HasTemperateGrowth := ⟨⟨0, by simp⟩⟩
                                                              /-
                                                                🎉 no goals
                                                              -/


variable [NormedSpace ℝ D] [FiniteDimensional ℝ D] [BorelSpace D] in
instance _root_.MeasureTheory.Measure.IsAddHaarMeasure.instHasTemperateGrowth {μ : Measure D}
    [h : μ.IsAddHaarMeasure] : μ.HasTemperateGrowth :=
                        /-
                          𝕜 : Type u_1
                          𝕜' : Type u_2
                          D : Type u_3
                          E : Type u_4
                          F : Type u_5
                          G : Type u_6
                          V : Type u_7
                          inst✝⁸ : NormedAddCommGroup E
                          inst✝⁷ : NormedSpace Real E
                          inst✝⁶ : NormedAddCommGroup F
                          inst✝⁵ : NormedSpace Real F
                          inst✝⁴ : NormedAddCommGroup D
                          inst✝³ : MeasurableSpace D
                          inst✝² : NormedSpace Real D
                          inst✝¹ : FiniteDimensional Real D
                          inst✝ : BorelSpace D
                          μ : MeasureTheory.Measure D
                          h : μ.IsAddHaarMeasure
                          ⊢ MeasureTheory.Integrable (fun x => HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) (Ne …
                        -/
  ⟨⟨finrank ℝ D + 1, by apply integrable_one_add_norm; norm_num⟩⟩
                                                       /-
                                                         🎉 no goals
                                                       -/


/-- Pointwise inequality to control `x ^ k * f` in terms of `1 / (1 + x) ^ l` if one controls both
`f` (with a bound `C₁`) and `x ^ (k + l) * f` (with a bound `C₂`). This will be used to check
integrability of `x ^ k * f x` when `f` is a Schwartz function, and to control explicitly its
integral in terms of suitable seminorms of `f`. -/
lemma pow_mul_le_of_le_of_pow_mul_le {C₁ C₂ : ℝ} {k l : ℕ} {x f : ℝ} (hx : 0 ≤ x) (hf : 0 ≤ f)
    (h₁ : f ≤ C₁) (h₂ : x ^ (k + l) * f ≤ C₂) :
    x ^ k * f ≤ 2 ^ l * (C₁ + C₂) * (1 + x) ^ (- (l : ℝ)) := by
  /-
    C₁ C₂ : Real
    k l : Nat
    x f : Real
    hx : LE.le 0 x
    hf : LE.le 0 f
    h₁ : LE.le f C₁
    h₂ : LE.le (HMul.hMul (HPow.hPow x (HAdd.hAdd k l)) f) C₂
    ⊢ LE.le (HMul.hMul (HPow.hPow x k) f) (HMul.hMul (HMul.hMul (HPow.hPow 2 l) (H …
  -/
  have : 0 ≤ C₂ := le_trans (by positivity) h₂
  have : 2 ^ l * (C₁ + C₂) * (1 + x) ^ (- (l : ℝ)) = ((1 + x) / 2) ^ (-(l : ℝ)) * (C₁ + C₂) := by
    rw [Real.div_rpow (by linarith) zero_le_two]
    simp [div_eq_inv_mul, ← Real.rpow_neg_one, ← Real.rpow_mul]
    ring
  /-
    C₁ C₂ : Real
    k l : Nat
    x f : Real
    hx : LE.le 0 x
    hf : LE.le 0 f
    h₁ : LE.le f C₁
    h₂ : LE.le (HMul.hMul (HPow.hPow x (HAdd.hAdd k l)) f) C₂
    this✝ : LE.le 0 C₂
    this : Eq (HMul.hMul (HMul.hMul (HPow.hPow 2 l) (HAdd.hAdd C₁ C₂)) (HPow.hPow  …
    ⊢ LE.le (HMul.hMul (HPow.hPow x k) f) (HMul.hMul (HMul.hMul (HPow.hPow 2 l) (H …
  -/
  rw [this]
  /-
    C₁ C₂ : Real
    k l : Nat
    x f : Real
    hx : LE.le 0 x
    hf : LE.le 0 f
    h₁ : LE.le f C₁
    h₂ : LE.le (HMul.hMul (HPow.hPow x (HAdd.hAdd k l)) f) C₂
    this✝ : LE.le 0 C₂
    this : Eq (HMul.hMul (HMul.hMul (HPow.hPow 2 l) (HAdd.hAdd C₁ C₂)) (HPow.hPow  …
    ⊢ LE.le (HMul.hMul (HPow.hPow x k) f) (HMul.hMul (HPow.hPow (HDiv.hDiv (HAdd.h …
  -/
  rcases le_total x 1 with h'x|h'x
    /-
      case inl
      C₁ C₂ : Real
      k l : Nat
      x f : Real
      hx : LE.le 0 x
      hf : LE.le 0 f
      h₁ : LE.le f C₁
      h₂ : LE.le (HMul.hMul (HPow.hPow x (HAdd.hAdd k l)) f) C₂
      this✝ : LE.le 0 C₂
      this : Eq (HMul.hMul (HMul.hMul (HPow.hPow 2 l) (HAdd.hAdd C₁ C₂)) (HPow.hPow  …
      h'x : LE.le x 1
      ⊢ LE.le (HMul.hMul (HPow.hPow x k) f) (HMul.hMul (HPow.hPow (HDiv.hDiv (HAdd.h …
    -/
  · gcongr
      /-
        case inl.h₁
        C₁ C₂ : Real
        k l : Nat
        x f : Real
        hx : LE.le 0 x
        hf : LE.le 0 f
        h₁ : LE.le f C₁
        h₂ : LE.le (HMul.hMul (HPow.hPow x (HAdd.hAdd k l)) f) C₂
        this✝ : LE.le 0 C₂
        this : Eq (HMul.hMul (HMul.hMul (HPow.hPow 2 l) (HAdd.hAdd C₁ C₂)) (HPow.hPow  …
        h'x : LE.le x 1
        ⊢ LE.le (HPow.hPow x k) (HPow.hPow (HDiv.hDiv (HAdd.hAdd 1 x) 2) (Neg.neg ↑l))
      -/
    · apply (pow_le_one₀ hx h'x).trans
      /-
        case inl.h₁
        C₁ C₂ : Real
        k l : Nat
        x f : Real
        hx : LE.le 0 x
        hf : LE.le 0 f
        h₁ : LE.le f C₁
        h₂ : LE.le (HMul.hMul (HPow.hPow x (HAdd.hAdd k l)) f) C₂
        this✝ : LE.le 0 C₂
        this : Eq (HMul.hMul (HMul.hMul (HPow.hPow 2 l) (HAdd.hAdd C₁ C₂)) (HPow.hPow  …
        h'x : LE.le x 1
        ⊢ LE.le 1 (HPow.hPow (HDiv.hDiv (HAdd.hAdd 1 x) 2) (Neg.neg ↑l))
      -/
      apply Real.one_le_rpow_of_pos_of_le_one_of_nonpos
        /-
          case inl.h₁.hx1
          C₁ C₂ : Real
          k l : Nat
          x f : Real
          hx : LE.le 0 x
          hf : LE.le 0 f
          h₁ : LE.le f C₁
          h₂ : LE.le (HMul.hMul (HPow.hPow x (HAdd.hAdd k l)) f) C₂
          this✝ : LE.le 0 C₂
          this : Eq (HMul.hMul (HMul.hMul (HPow.hPow 2 l) (HAdd.hAdd C₁ C₂)) (HPow.hPow  …
          h'x : LE.le x 1
          ⊢ LT.lt 0 (HDiv.hDiv (HAdd.hAdd 1 x) 2)
        -/
      · linarith
        /-
          🎉 no goals
        -/
        /-
          case inl.h₁.hx2
          C₁ C₂ : Real
          k l : Nat
          x f : Real
          hx : LE.le 0 x
          hf : LE.le 0 f
          h₁ : LE.le f C₁
          h₂ : LE.le (HMul.hMul (HPow.hPow x (HAdd.hAdd k l)) f) C₂
          this✝ : LE.le 0 C₂
          this : Eq (HMul.hMul (HMul.hMul (HPow.hPow 2 l) (HAdd.hAdd C₁ C₂)) (HPow.hPow  …
          h'x : LE.le x 1
          ⊢ LE.le (HDiv.hDiv (HAdd.hAdd 1 x) 2) 1
        -/
      · linarith
        /-
          🎉 no goals
        -/
        /-
          case inl.h₁.hz
          C₁ C₂ : Real
          k l : Nat
          x f : Real
          hx : LE.le 0 x
          hf : LE.le 0 f
          h₁ : LE.le f C₁
          h₂ : LE.le (HMul.hMul (HPow.hPow x (HAdd.hAdd k l)) f) C₂
          this✝ : LE.le 0 C₂
          this : Eq (HMul.hMul (HMul.hMul (HPow.hPow 2 l) (HAdd.hAdd C₁ C₂)) (HPow.hPow  …
          h'x : LE.le x 1
          ⊢ LE.le (Neg.neg ↑l) 0
        -/
      · simp
        /-
          🎉 no goals
        -/
      /-
        case inl.h₂
        C₁ C₂ : Real
        k l : Nat
        x f : Real
        hx : LE.le 0 x
        hf : LE.le 0 f
        h₁ : LE.le f C₁
        h₂ : LE.le (HMul.hMul (HPow.hPow x (HAdd.hAdd k l)) f) C₂
        this✝ : LE.le 0 C₂
        this : Eq (HMul.hMul (HMul.hMul (HPow.hPow 2 l) (HAdd.hAdd C₁ C₂)) (HPow.hPow  …
        h'x : LE.le x 1
        ⊢ LE.le f (HAdd.hAdd C₁ C₂)
      -/
    · linarith
      /-
        🎉 no goals
      -/
  · calc
    x ^ k * f = x ^ (-(l : ℝ)) * (x ^ (k + l) * f) := by
      rw [← Real.rpow_natCast, ← Real.rpow_natCast, ← mul_assoc, ← Real.rpow_add (by linarith)]
      simp
    _ ≤ ((1 + x) / 2) ^ (-(l : ℝ)) * (C₁ + C₂) := by
      apply mul_le_mul _ _ (by positivity) (by positivity)
      · exact Real.rpow_le_rpow_of_nonpos (by linarith) (by linarith) (by simp)
      · exact h₂.trans (by linarith)


variable [BorelSpace D] [SecondCountableTopology D] in
/-- Given a function such that `f` and `x ^ (k + l) * f` are bounded for a suitable `l`, then
`x ^ k * f` is integrable. The bounds are not relevant for the integrability conclusion, but they
are relevant for bounding the integral in `integral_pow_mul_le_of_le_of_pow_mul_le`. We formulate
the two lemmas with the same set of assumptions for ease of applications. -/
-- We redeclare `E` here to avoid the `NormedSpace ℝ E` typeclass available throughout this file.
lemma integrable_of_le_of_pow_mul_le
    {E : Type*} [NormedAddCommGroup E]
    {μ : Measure D} [μ.HasTemperateGrowth] {f : D → E} {C₁ C₂ : ℝ} {k : ℕ}
    (hf : ∀ x, ‖f x‖ ≤ C₁) (h'f : ∀ x, ‖x‖ ^ (k + μ.integrablePower) * ‖f x‖ ≤ C₂)
    (h''f : AEStronglyMeasurable f μ) :
    Integrable (fun x ↦ ‖x‖ ^ k * ‖f x‖) μ := by
  /-
    D : Type u_3
    inst✝⁵ : NormedAddCommGroup D
    inst✝⁴ : MeasurableSpace D
    inst✝³ : BorelSpace D
    inst✝² : SecondCountableTopology D
    E : Type u_8
    inst✝¹ : NormedAddCommGroup E
    μ : MeasureTheory.Measure D
    inst✝ : μ.HasTemperateGrowth
    f : D → E
    C₁ C₂ : Real
    k : Nat
    hf : ∀ (x : D), LE.le (Norm.norm (f x)) C₁
    h'f : ∀ (x : D), LE.le (HMul.hMul (HPow.hPow (Norm.norm x) (HAdd.hAdd k μ.inte …
    h''f : MeasureTheory.AEStronglyMeasurable f μ
    ⊢ MeasureTheory.Integrable (fun x => HMul.hMul (HPow.hPow (Norm.norm x) k) (No …
  -/
  apply ((integrable_pow_neg_integrablePower μ).const_mul (2 ^ μ.integrablePower * (C₁ + C₂))).mono'
    /-
      case hf
      D : Type u_3
      inst✝⁵ : NormedAddCommGroup D
      inst✝⁴ : MeasurableSpace D
      inst✝³ : BorelSpace D
      inst✝² : SecondCountableTopology D
      E : Type u_8
      inst✝¹ : NormedAddCommGroup E
      μ : MeasureTheory.Measure D
      inst✝ : μ.HasTemperateGrowth
      f : D → E
      C₁ C₂ : Real
      k : Nat
      hf : ∀ (x : D), LE.le (Norm.norm (f x)) C₁
      h'f : ∀ (x : D), LE.le (HMul.hMul (HPow.hPow (Norm.norm x) (HAdd.hAdd k μ.inte …
      h''f : MeasureTheory.AEStronglyMeasurable f μ
      ⊢ MeasureTheory.AEStronglyMeasurable (fun x => HMul.hMul (HPow.hPow (Norm.norm …
    -/
  · exact AEStronglyMeasurable.mul (aestronglyMeasurable_id.norm.pow _) h''f.norm
    /-
      🎉 no goals
    -/
    /-
      case h
      D : Type u_3
      inst✝⁵ : NormedAddCommGroup D
      inst✝⁴ : MeasurableSpace D
      inst✝³ : BorelSpace D
      inst✝² : SecondCountableTopology D
      E : Type u_8
      inst✝¹ : NormedAddCommGroup E
      μ : MeasureTheory.Measure D
      inst✝ : μ.HasTemperateGrowth
      f : D → E
      C₁ C₂ : Real
      k : Nat
      hf : ∀ (x : D), LE.le (Norm.norm (f x)) C₁
      h'f : ∀ (x : D), LE.le (HMul.hMul (HPow.hPow (Norm.norm x) (HAdd.hAdd k μ.inte …
      h''f : MeasureTheory.AEStronglyMeasurable f μ
      ⊢ Filter.Eventually (fun a => LE.le (Norm.norm (HMul.hMul (HPow.hPow (Norm.nor …
    -/
  · filter_upwards with v
    /-
      case h.h
      D : Type u_3
      inst✝⁵ : NormedAddCommGroup D
      inst✝⁴ : MeasurableSpace D
      inst✝³ : BorelSpace D
      inst✝² : SecondCountableTopology D
      E : Type u_8
      inst✝¹ : NormedAddCommGroup E
      μ : MeasureTheory.Measure D
      inst✝ : μ.HasTemperateGrowth
      f : D → E
      C₁ C₂ : Real
      k : Nat
      hf : ∀ (x : D), LE.le (Norm.norm (f x)) C₁
      h'f : ∀ (x : D), LE.le (HMul.hMul (HPow.hPow (Norm.norm x) (HAdd.hAdd k μ.inte …
      h''f : MeasureTheory.AEStronglyMeasurable f μ
      v : D
      ⊢ LE.le (Norm.norm (HMul.hMul (HPow.hPow (Norm.norm v) k) (Norm.norm (f v))))  …
    -/
    simp only [norm_mul, norm_pow, norm_norm]
    /-
      case h.h
      D : Type u_3
      inst✝⁵ : NormedAddCommGroup D
      inst✝⁴ : MeasurableSpace D
      inst✝³ : BorelSpace D
      inst✝² : SecondCountableTopology D
      E : Type u_8
      inst✝¹ : NormedAddCommGroup E
      μ : MeasureTheory.Measure D
      inst✝ : μ.HasTemperateGrowth
      f : D → E
      C₁ C₂ : Real
      k : Nat
      hf : ∀ (x : D), LE.le (Norm.norm (f x)) C₁
      h'f : ∀ (x : D), LE.le (HMul.hMul (HPow.hPow (Norm.norm x) (HAdd.hAdd k μ.inte …
      h''f : MeasureTheory.AEStronglyMeasurable f μ
      v : D
      ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm v) k) (Norm.norm (f v))) (HMul.hMul ( …
    -/
    apply pow_mul_le_of_le_of_pow_mul_le (norm_nonneg _) (norm_nonneg _) (hf v) (h'f v)
    /-
      🎉 no goals
    -/


/-- Given a function such that `f` and `x ^ (k + l) * f` are bounded for a suitable `l`, then
one can bound explicitly the integral of `x ^ k * f`. -/
-- We redeclare `E` here to avoid the `NormedSpace ℝ E` typeclass available throughout this file.
lemma integral_pow_mul_le_of_le_of_pow_mul_le
    {E : Type*} [NormedAddCommGroup E]
    {μ : Measure D} [μ.HasTemperateGrowth] {f : D → E} {C₁ C₂ : ℝ} {k : ℕ}
    (hf : ∀ x, ‖f x‖ ≤ C₁) (h'f : ∀ x, ‖x‖ ^ (k + μ.integrablePower) * ‖f x‖ ≤ C₂) :
    ∫ x, ‖x‖ ^ k * ‖f x‖ ∂μ ≤ 2 ^ μ.integrablePower *
      (∫ x, (1 + ‖x‖) ^ (- (μ.integrablePower : ℝ)) ∂μ) * (C₁ + C₂) := by
  /-
    D : Type u_3
    inst✝³ : NormedAddCommGroup D
    inst✝² : MeasurableSpace D
    E : Type u_8
    inst✝¹ : NormedAddCommGroup E
    μ : MeasureTheory.Measure D
    inst✝ : μ.HasTemperateGrowth
    f : D → E
    C₁ C₂ : Real
    k : Nat
    hf : ∀ (x : D), LE.le (Norm.norm (f x)) C₁
    h'f : ∀ (x : D), LE.le (HMul.hMul (HPow.hPow (Norm.norm x) (HAdd.hAdd k μ.inte …
    ⊢ LE.le (MeasureTheory.integral μ fun x => HMul.hMul (HPow.hPow (Norm.norm x)  …
  -/
  rw [← integral_mul_left, ← integral_mul_right]
  /-
    D : Type u_3
    inst✝³ : NormedAddCommGroup D
    inst✝² : MeasurableSpace D
    E : Type u_8
    inst✝¹ : NormedAddCommGroup E
    μ : MeasureTheory.Measure D
    inst✝ : μ.HasTemperateGrowth
    f : D → E
    C₁ C₂ : Real
    k : Nat
    hf : ∀ (x : D), LE.le (Norm.norm (f x)) C₁
    h'f : ∀ (x : D), LE.le (HMul.hMul (HPow.hPow (Norm.norm x) (HAdd.hAdd k μ.inte …
    ⊢ LE.le (MeasureTheory.integral μ fun x => HMul.hMul (HPow.hPow (Norm.norm x)  …
  -/
  apply integral_mono_of_nonneg
    /-
      case hf
      D : Type u_3
      inst✝³ : NormedAddCommGroup D
      inst✝² : MeasurableSpace D
      E : Type u_8
      inst✝¹ : NormedAddCommGroup E
      μ : MeasureTheory.Measure D
      inst✝ : μ.HasTemperateGrowth
      f : D → E
      C₁ C₂ : Real
      k : Nat
      hf : ∀ (x : D), LE.le (Norm.norm (f x)) C₁
      h'f : ∀ (x : D), LE.le (HMul.hMul (HPow.hPow (Norm.norm x) (HAdd.hAdd k μ.inte …
      ⊢ (MeasureTheory.ae μ).EventuallyLE 0 fun a => HMul.hMul (HPow.hPow (Norm.norm …
    -/
  · filter_upwards with v using by positivity
    /-
      🎉 no goals
    -/
    /-
      case hgi
      D : Type u_3
      inst✝³ : NormedAddCommGroup D
      inst✝² : MeasurableSpace D
      E : Type u_8
      inst✝¹ : NormedAddCommGroup E
      μ : MeasureTheory.Measure D
      inst✝ : μ.HasTemperateGrowth
      f : D → E
      C₁ C₂ : Real
      k : Nat
      hf : ∀ (x : D), LE.le (Norm.norm (f x)) C₁
      h'f : ∀ (x : D), LE.le (HMul.hMul (HPow.hPow (Norm.norm x) (HAdd.hAdd k μ.inte …
      ⊢ MeasureTheory.Integrable (fun a => HMul.hMul (HMul.hMul (HPow.hPow 2 μ.integ …
    -/
  · exact ((integrable_pow_neg_integrablePower μ).const_mul _).mul_const _
    /-
      🎉 no goals
    -/
  /-
    case h
    D : Type u_3
    inst✝³ : NormedAddCommGroup D
    inst✝² : MeasurableSpace D
    E : Type u_8
    inst✝¹ : NormedAddCommGroup E
    μ : MeasureTheory.Measure D
    inst✝ : μ.HasTemperateGrowth
    f : D → E
    C₁ C₂ : Real
    k : Nat
    hf : ∀ (x : D), LE.le (Norm.norm (f x)) C₁
    h'f : ∀ (x : D), LE.le (HMul.hMul (HPow.hPow (Norm.norm x) (HAdd.hAdd k μ.inte …
    ⊢ (MeasureTheory.ae μ).EventuallyLE (fun a => HMul.hMul (HPow.hPow (Norm.norm  …
  -/
  filter_upwards with v
  exact (pow_mul_le_of_le_of_pow_mul_le (norm_nonneg _) (norm_nonneg _) (hf v) (h'f v)).trans
    (le_of_eq (by ring))


/-- Create a semilinear map between Schwartz spaces.

Note: This is a helper definition for `mkCLM`. -/
def mkLM (A : (D → E) → F → G) (hadd : ∀ (f g : 𝓢(D, E)) (x), A (f + g) x = A f x + A g x)
    (hsmul : ∀ (a : 𝕜) (f : 𝓢(D, E)) (x), A (a • f) x = σ a • A f x)
    (hsmooth : ∀ f : 𝓢(D, E), ContDiff ℝ ∞ (A f))
    (hbound : ∀ n : ℕ × ℕ, ∃ (s : Finset (ℕ × ℕ)) (C : ℝ), 0 ≤ C ∧ ∀ (f : 𝓢(D, E)) (x : F),
      ‖x‖ ^ n.fst * ‖iteratedFDeriv ℝ n.snd (A f) x‖ ≤ C * s.sup (schwartzSeminormFamily 𝕜 D E) f) :
    𝓢(D, E) →ₛₗ[σ] 𝓢(F, G) where
  toFun f :=
    { toFun := A f
      smooth' := hsmooth f
      decay' := by
        /-
          𝕜 : Type u_1
          𝕜' : Type u_2
          D : Type u_3
          E : Type u_4
          F : Type u_5
          G : Type u_6
          V : Type u_7
          inst✝¹³ : NormedAddCommGroup E
          inst✝¹² : NormedSpace Real E
          inst✝¹¹ : NormedAddCommGroup F
          inst✝¹⁰ : NormedSpace Real F
          inst✝⁹ : NormedField 𝕜
          inst✝⁸ : NormedField 𝕜'
          inst✝⁷ : NormedAddCommGroup D
          inst✝⁶ : NormedSpace Real D
          inst✝⁵ : NormedSpace 𝕜 E
          inst✝⁴ : SMulCommClass Real 𝕜 E
          inst✝³ : NormedAddCommGroup G
          inst✝² : NormedSpace Real G
          inst✝¹ : NormedSpace 𝕜' G
          inst✝ : SMulCommClass Real 𝕜' G
          σ : RingHom 𝕜 𝕜'
          A : (D → E) → F → G
          hadd : ∀ (f g : SchwartzMap D E) (x : F), Eq (A (HAdd.hAdd ⇑f ⇑g) x) (HAdd.hAd …
          hsmul : ∀ (a : 𝕜) (f : SchwartzMap D E) (x : F), Eq (A (HSMul.hSMul a ⇑f) x) ( …
          hsmooth : ∀ (f : SchwartzMap D E), ContDiff Real (↑Top.top) (A ⇑f)
          hbound : ∀ (n : Prod Nat Nat), Exists fun s => Exists fun C => And (LE.le 0 C) …
          f : SchwartzMap D E
          ⊢ ∀ (k n : Nat), Exists fun C => ∀ (x : F), LE.le (HMul.hMul (HPow.hPow (Norm. …
        -/
        intro k n
        /-
          𝕜 : Type u_1
          𝕜' : Type u_2
          D : Type u_3
          E : Type u_4
          F : Type u_5
          G : Type u_6
          V : Type u_7
          inst✝¹³ : NormedAddCommGroup E
          inst✝¹² : NormedSpace Real E
          inst✝¹¹ : NormedAddCommGroup F
          inst✝¹⁰ : NormedSpace Real F
          inst✝⁹ : NormedField 𝕜
          inst✝⁸ : NormedField 𝕜'
          inst✝⁷ : NormedAddCommGroup D
          inst✝⁶ : NormedSpace Real D
          inst✝⁵ : NormedSpace 𝕜 E
          inst✝⁴ : SMulCommClass Real 𝕜 E
          inst✝³ : NormedAddCommGroup G
          inst✝² : NormedSpace Real G
          inst✝¹ : NormedSpace 𝕜' G
          inst✝ : SMulCommClass Real 𝕜' G
          σ : RingHom 𝕜 𝕜'
          A : (D → E) → F → G
          hadd : ∀ (f g : SchwartzMap D E) (x : F), Eq (A (HAdd.hAdd ⇑f ⇑g) x) (HAdd.hAd …
          hsmul : ∀ (a : 𝕜) (f : SchwartzMap D E) (x : F), Eq (A (HSMul.hSMul a ⇑f) x) ( …
          hsmooth : ∀ (f : SchwartzMap D E), ContDiff Real (↑Top.top) (A ⇑f)
          hbound : ∀ (n : Prod Nat Nat), Exists fun s => Exists fun C => And (LE.le 0 C) …
          f : SchwartzMap D E
          k n : Nat
          ⊢ Exists fun C => ∀ (x : F), LE.le (HMul.hMul (HPow.hPow (Norm.norm x) k) (Nor …
        -/
        rcases hbound ⟨k, n⟩ with ⟨s, C, _, h⟩
        /-
          case intro.intro.intro
          𝕜 : Type u_1
          𝕜' : Type u_2
          D : Type u_3
          E : Type u_4
          F : Type u_5
          G : Type u_6
          V : Type u_7
          inst✝¹³ : NormedAddCommGroup E
          inst✝¹² : NormedSpace Real E
          inst✝¹¹ : NormedAddCommGroup F
          inst✝¹⁰ : NormedSpace Real F
          inst✝⁹ : NormedField 𝕜
          inst✝⁸ : NormedField 𝕜'
          inst✝⁷ : NormedAddCommGroup D
          inst✝⁶ : NormedSpace Real D
          inst✝⁵ : NormedSpace 𝕜 E
          inst✝⁴ : SMulCommClass Real 𝕜 E
          inst✝³ : NormedAddCommGroup G
          inst✝² : NormedSpace Real G
          inst✝¹ : NormedSpace 𝕜' G
          inst✝ : SMulCommClass Real 𝕜' G
          σ : RingHom 𝕜 𝕜'
          A : (D → E) → F → G
          hadd : ∀ (f g : SchwartzMap D E) (x : F), Eq (A (HAdd.hAdd ⇑f ⇑g) x) (HAdd.hAd …
          hsmul : ∀ (a : 𝕜) (f : SchwartzMap D E) (x : F), Eq (A (HSMul.hSMul a ⇑f) x) ( …
          hsmooth : ∀ (f : SchwartzMap D E), ContDiff Real (↑Top.top) (A ⇑f)
          hbound : ∀ (n : Prod Nat Nat), Exists fun s => Exists fun C => And (LE.le 0 C) …
          f : SchwartzMap D E
          k n : Nat
          s : Finset (Prod Nat Nat)
          C : Real
          left✝ : LE.le 0 C
          h : ∀ (f : SchwartzMap D E) (x : F), LE.le (HMul.hMul (HPow.hPow (Norm.norm x) …
          ⊢ Exists fun C => ∀ (x : F), LE.le (HMul.hMul (HPow.hPow (Norm.norm x) k) (Nor …
        -/
        exact ⟨C * (s.sup (schwartzSeminormFamily 𝕜 D E)) f, h f⟩ }
        /-
          🎉 no goals
        -/
  map_add' f g := ext (hadd f g)
  map_smul' a f := ext (hsmul a f)


/-- Create a continuous semilinear map between Schwartz spaces.

For an example of using this definition, see `fderivCLM`. -/
def mkCLM [RingHomIsometric σ] (A : (D → E) → F → G)
    (hadd : ∀ (f g : 𝓢(D, E)) (x), A (f + g) x = A f x + A g x)
    (hsmul : ∀ (a : 𝕜) (f : 𝓢(D, E)) (x), A (a • f) x = σ a • A f x)
    (hsmooth : ∀ f : 𝓢(D, E), ContDiff ℝ ∞ (A f))
    (hbound : ∀ n : ℕ × ℕ, ∃ (s : Finset (ℕ × ℕ)) (C : ℝ), 0 ≤ C ∧ ∀ (f : 𝓢(D, E)) (x : F),
      ‖x‖ ^ n.fst * ‖iteratedFDeriv ℝ n.snd (A f) x‖ ≤ C * s.sup (schwartzSeminormFamily 𝕜 D E) f) :
    𝓢(D, E) →SL[σ] 𝓢(F, G) where
  cont := by
    /-
      𝕜 : Type u_1
      𝕜' : Type u_2
      D : Type u_3
      E : Type u_4
      F : Type u_5
      G : Type u_6
      V : Type u_7
      inst✝¹⁴ : NormedAddCommGroup E
      inst✝¹³ : NormedSpace Real E
      inst✝¹² : NormedAddCommGroup F
      inst✝¹¹ : NormedSpace Real F
      inst✝¹⁰ : NormedField 𝕜
      inst✝⁹ : NormedField 𝕜'
      inst✝⁸ : NormedAddCommGroup D
      inst✝⁷ : NormedSpace Real D
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : SMulCommClass Real 𝕜 E
      inst✝⁴ : NormedAddCommGroup G
      inst✝³ : NormedSpace Real G
      inst✝² : NormedSpace 𝕜' G
      inst✝¹ : SMulCommClass Real 𝕜' G
      σ : RingHom 𝕜 𝕜'
      inst✝ : RingHomIsometric σ
      A : (D → E) → F → G
      hadd : ∀ (f g : SchwartzMap D E) (x : F), Eq (A (HAdd.hAdd ⇑f ⇑g) x) (HAdd.hAd …
      hsmul : ∀ (a : 𝕜) (f : SchwartzMap D E) (x : F), Eq (A (HSMul.hSMul a ⇑f) x) ( …
      hsmooth : ∀ (f : SchwartzMap D E), ContDiff Real (↑Top.top) (A ⇑f)
      hbound : ∀ (n : Prod Nat Nat), Exists fun s => Exists fun C => And (LE.le 0 C) …
      ⊢ Continuous (SchwartzMap.mkLM A hadd hsmul hsmooth hbound).toFun
    -/
    change Continuous (mkLM A hadd hsmul hsmooth hbound : 𝓢(D, E) →ₛₗ[σ] 𝓢(F, G))
    refine
      Seminorm.continuous_from_bounded (schwartz_withSeminorms 𝕜 D E)
        (schwartz_withSeminorms 𝕜' F G) _ fun n => ?_
    /-
      𝕜 : Type u_1
      𝕜' : Type u_2
      D : Type u_3
      E : Type u_4
      F : Type u_5
      G : Type u_6
      V : Type u_7
      inst✝¹⁴ : NormedAddCommGroup E
      inst✝¹³ : NormedSpace Real E
      inst✝¹² : NormedAddCommGroup F
      inst✝¹¹ : NormedSpace Real F
      inst✝¹⁰ : NormedField 𝕜
      inst✝⁹ : NormedField 𝕜'
      inst✝⁸ : NormedAddCommGroup D
      inst✝⁷ : NormedSpace Real D
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : SMulCommClass Real 𝕜 E
      inst✝⁴ : NormedAddCommGroup G
      inst✝³ : NormedSpace Real G
      inst✝² : NormedSpace 𝕜' G
      inst✝¹ : SMulCommClass Real 𝕜' G
      σ : RingHom 𝕜 𝕜'
      inst✝ : RingHomIsometric σ
      A : (D → E) → F → G
      hadd : ∀ (f g : SchwartzMap D E) (x : F), Eq (A (HAdd.hAdd ⇑f ⇑g) x) (HAdd.hAd …
      hsmul : ∀ (a : 𝕜) (f : SchwartzMap D E) (x : F), Eq (A (HSMul.hSMul a ⇑f) x) ( …
      hsmooth : ∀ (f : SchwartzMap D E), ContDiff Real (↑Top.top) (A ⇑f)
      hbound : ∀ (n : Prod Nat Nat), Exists fun s => Exists fun C => And (LE.le 0 C) …
      n : Prod Nat Nat
      ⊢ Exists fun s => Exists fun C => LE.le ((schwartzSeminormFamily 𝕜' F G n).com …
    -/
    rcases hbound n with ⟨s, C, hC, h⟩
    /-
      case intro.intro.intro
      𝕜 : Type u_1
      𝕜' : Type u_2
      D : Type u_3
      E : Type u_4
      F : Type u_5
      G : Type u_6
      V : Type u_7
      inst✝¹⁴ : NormedAddCommGroup E
      inst✝¹³ : NormedSpace Real E
      inst✝¹² : NormedAddCommGroup F
      inst✝¹¹ : NormedSpace Real F
      inst✝¹⁰ : NormedField 𝕜
      inst✝⁹ : NormedField 𝕜'
      inst✝⁸ : NormedAddCommGroup D
      inst✝⁷ : NormedSpace Real D
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : SMulCommClass Real 𝕜 E
      inst✝⁴ : NormedAddCommGroup G
      inst✝³ : NormedSpace Real G
      inst✝² : NormedSpace 𝕜' G
      inst✝¹ : SMulCommClass Real 𝕜' G
      σ : RingHom 𝕜 𝕜'
      inst✝ : RingHomIsometric σ
      A : (D → E) → F → G
      hadd : ∀ (f g : SchwartzMap D E) (x : F), Eq (A (HAdd.hAdd ⇑f ⇑g) x) (HAdd.hAd …
      hsmul : ∀ (a : 𝕜) (f : SchwartzMap D E) (x : F), Eq (A (HSMul.hSMul a ⇑f) x) ( …
      hsmooth : ∀ (f : SchwartzMap D E), ContDiff Real (↑Top.top) (A ⇑f)
      hbound : ∀ (n : Prod Nat Nat), Exists fun s => Exists fun C => And (LE.le 0 C) …
      n : Prod Nat Nat
      s : Finset (Prod Nat Nat)
      C : Real
      hC : LE.le 0 C
      h : ∀ (f : SchwartzMap D E) (x : F), LE.le (HMul.hMul (HPow.hPow (Norm.norm x) …
      ⊢ Exists fun s => Exists fun C => LE.le ((schwartzSeminormFamily 𝕜' F G n).com …
    -/
    refine ⟨s, ⟨C, hC⟩, fun f => ?_⟩
    /-
      case intro.intro.intro
      𝕜 : Type u_1
      𝕜' : Type u_2
      D : Type u_3
      E : Type u_4
      F : Type u_5
      G : Type u_6
      V : Type u_7
      inst✝¹⁴ : NormedAddCommGroup E
      inst✝¹³ : NormedSpace Real E
      inst✝¹² : NormedAddCommGroup F
      inst✝¹¹ : NormedSpace Real F
      inst✝¹⁰ : NormedField 𝕜
      inst✝⁹ : NormedField 𝕜'
      inst✝⁸ : NormedAddCommGroup D
      inst✝⁷ : NormedSpace Real D
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : SMulCommClass Real 𝕜 E
      inst✝⁴ : NormedAddCommGroup G
      inst✝³ : NormedSpace Real G
      inst✝² : NormedSpace 𝕜' G
      inst✝¹ : SMulCommClass Real 𝕜' G
      σ : RingHom 𝕜 𝕜'
      inst✝ : RingHomIsometric σ
      A : (D → E) → F → G
      hadd : ∀ (f g : SchwartzMap D E) (x : F), Eq (A (HAdd.hAdd ⇑f ⇑g) x) (HAdd.hAd …
      hsmul : ∀ (a : 𝕜) (f : SchwartzMap D E) (x : F), Eq (A (HSMul.hSMul a ⇑f) x) ( …
      hsmooth : ∀ (f : SchwartzMap D E), ContDiff Real (↑Top.top) (A ⇑f)
      hbound : ∀ (n : Prod Nat Nat), Exists fun s => Exists fun C => And (LE.le 0 C) …
      n : Prod Nat Nat
      s : Finset (Prod Nat Nat)
      C : Real
      hC : LE.le 0 C
      h : ∀ (f : SchwartzMap D E) (x : F), LE.le (HMul.hMul (HPow.hPow (Norm.norm x) …
      f : SchwartzMap D E
      ⊢ LE.le ((fun f => ⇑f) ((schwartzSeminormFamily 𝕜' F G n).comp (SchwartzMap.mk …
    -/
    exact (mkLM A hadd hsmul hsmooth hbound f).seminorm_le_bound 𝕜' n.1 n.2 (by positivity) (h f)
    /-
      🎉 no goals
    -/
  toLinearMap := mkLM A hadd hsmul hsmooth hbound


/-- Define a continuous semilinear map from Schwartz space to a normed space. -/
def mkCLMtoNormedSpace [RingHomIsometric σ] (A : 𝓢(D, E) → G)
    (hadd : ∀ (f g : 𝓢(D, E)), A (f + g) = A f + A g)
    (hsmul : ∀ (a : 𝕜) (f : 𝓢(D, E)), A (a • f) = σ a • A f)
    (hbound : ∃ (s : Finset (ℕ × ℕ)) (C : ℝ), 0 ≤ C ∧ ∀ (f : 𝓢(D, E)),
      ‖A f‖ ≤ C * s.sup (schwartzSeminormFamily 𝕜 D E) f) :
    𝓢(D, E) →SL[σ] G where
  toLinearMap :=
    { toFun := (A ·)
      map_add' := hadd
      map_smul' := hsmul }
  cont := by
    /-
      𝕜 : Type u_1
      𝕜' : Type u_2
      D : Type u_3
      E : Type u_4
      F : Type u_5
      G : Type u_6
      V : Type u_7
      inst✝¹⁴ : NormedAddCommGroup E
      inst✝¹³ : NormedSpace Real E
      inst✝¹² : NormedAddCommGroup F
      inst✝¹¹ : NormedSpace Real F
      inst✝¹⁰ : NormedField 𝕜
      inst✝⁹ : NormedField 𝕜'
      inst✝⁸ : NormedAddCommGroup D
      inst✝⁷ : NormedSpace Real D
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : SMulCommClass Real 𝕜 E
      inst✝⁴ : NormedAddCommGroup G
      inst✝³ : NormedSpace Real G
      inst✝² : NormedSpace 𝕜' G
      inst✝¹ : SMulCommClass Real 𝕜' G
      σ : RingHom 𝕜 𝕜'
      inst✝ : RingHomIsometric σ
      A : SchwartzMap D E → G
      hadd : ∀ (f g : SchwartzMap D E), Eq (A (HAdd.hAdd f g)) (HAdd.hAdd (A f) (A g))
      hsmul : ∀ (a : 𝕜) (f : SchwartzMap D E), Eq (A (HSMul.hSMul a f)) (HSMul.hSMul …
      hbound : Exists fun s => Exists fun C => And (LE.le 0 C) (∀ (f : SchwartzMap D …
      ⊢ Continuous { toFun := fun x => A x, map_add' := hadd, map_smul' := hsmul }.t …
    -/
    change Continuous (LinearMap.mk _ _)
    /-
      𝕜 : Type u_1
      𝕜' : Type u_2
      D : Type u_3
      E : Type u_4
      F : Type u_5
      G : Type u_6
      V : Type u_7
      inst✝¹⁴ : NormedAddCommGroup E
      inst✝¹³ : NormedSpace Real E
      inst✝¹² : NormedAddCommGroup F
      inst✝¹¹ : NormedSpace Real F
      inst✝¹⁰ : NormedField 𝕜
      inst✝⁹ : NormedField 𝕜'
      inst✝⁸ : NormedAddCommGroup D
      inst✝⁷ : NormedSpace Real D
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : SMulCommClass Real 𝕜 E
      inst✝⁴ : NormedAddCommGroup G
      inst✝³ : NormedSpace Real G
      inst✝² : NormedSpace 𝕜' G
      inst✝¹ : SMulCommClass Real 𝕜' G
      σ : RingHom 𝕜 𝕜'
      inst✝ : RingHomIsometric σ
      A : SchwartzMap D E → G
      hadd : ∀ (f g : SchwartzMap D E), Eq (A (HAdd.hAdd f g)) (HAdd.hAdd (A f) (A g))
      hsmul : ∀ (a : 𝕜) (f : SchwartzMap D E), Eq (A (HSMul.hSMul a f)) (HSMul.hSMul …
      hbound : Exists fun s => Exists fun C => And (LE.le 0 C) (∀ (f : SchwartzMap D …
      ⊢ Continuous ⇑{ toFun := fun x => A x, map_add' := hadd, map_smul' := hsmul }
    -/
    apply Seminorm.cont_withSeminorms_normedSpace G (schwartz_withSeminorms 𝕜 D E)
    /-
      case hf
      𝕜 : Type u_1
      𝕜' : Type u_2
      D : Type u_3
      E : Type u_4
      F : Type u_5
      G : Type u_6
      V : Type u_7
      inst✝¹⁴ : NormedAddCommGroup E
      inst✝¹³ : NormedSpace Real E
      inst✝¹² : NormedAddCommGroup F
      inst✝¹¹ : NormedSpace Real F
      inst✝¹⁰ : NormedField 𝕜
      inst✝⁹ : NormedField 𝕜'
      inst✝⁸ : NormedAddCommGroup D
      inst✝⁷ : NormedSpace Real D
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : SMulCommClass Real 𝕜 E
      inst✝⁴ : NormedAddCommGroup G
      inst✝³ : NormedSpace Real G
      inst✝² : NormedSpace 𝕜' G
      inst✝¹ : SMulCommClass Real 𝕜' G
      σ : RingHom 𝕜 𝕜'
      inst✝ : RingHomIsometric σ
      A : SchwartzMap D E → G
      hadd : ∀ (f g : SchwartzMap D E), Eq (A (HAdd.hAdd f g)) (HAdd.hAdd (A f) (A g))
      hsmul : ∀ (a : 𝕜) (f : SchwartzMap D E), Eq (A (HSMul.hSMul a f)) (HSMul.hSMul …
      hbound : Exists fun s => Exists fun C => And (LE.le 0 C) (∀ (f : SchwartzMap D …
      ⊢ Exists fun s => Exists fun C => LE.le ((normSeminorm 𝕜' G).comp { toFun := f …
    -/
    rcases hbound with ⟨s, C, hC, h⟩
    /-
      case hf.intro.intro.intro
      𝕜 : Type u_1
      𝕜' : Type u_2
      D : Type u_3
      E : Type u_4
      F : Type u_5
      G : Type u_6
      V : Type u_7
      inst✝¹⁴ : NormedAddCommGroup E
      inst✝¹³ : NormedSpace Real E
      inst✝¹² : NormedAddCommGroup F
      inst✝¹¹ : NormedSpace Real F
      inst✝¹⁰ : NormedField 𝕜
      inst✝⁹ : NormedField 𝕜'
      inst✝⁸ : NormedAddCommGroup D
      inst✝⁷ : NormedSpace Real D
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : SMulCommClass Real 𝕜 E
      inst✝⁴ : NormedAddCommGroup G
      inst✝³ : NormedSpace Real G
      inst✝² : NormedSpace 𝕜' G
      inst✝¹ : SMulCommClass Real 𝕜' G
      σ : RingHom 𝕜 𝕜'
      inst✝ : RingHomIsometric σ
      A : SchwartzMap D E → G
      hadd : ∀ (f g : SchwartzMap D E), Eq (A (HAdd.hAdd f g)) (HAdd.hAdd (A f) (A g))
      hsmul : ∀ (a : 𝕜) (f : SchwartzMap D E), Eq (A (HSMul.hSMul a f)) (HSMul.hSMul …
      s : Finset (Prod Nat Nat)
      C : Real
      hC : LE.le 0 C
      h : ∀ (f : SchwartzMap D E), LE.le (Norm.norm (A f)) (HMul.hMul C ((s.sup (sch …
      ⊢ Exists fun s => Exists fun C => LE.le ((normSeminorm 𝕜' G).comp { toFun := f …
    -/
    exact ⟨s, ⟨C, hC⟩, h⟩
    /-
      🎉 no goals
    -/


/-- The map applying a vector to Hom-valued Schwartz function as a continuous linear map. -/
protected def evalCLM (m : E) : 𝓢(E, E →L[ℝ] F) →L[𝕜] 𝓢(E, F) :=
  mkCLM (fun f x => f x m) (fun _ _ _ => rfl) (fun _ _ _ => rfl)
    (fun f => ContDiff.clm_apply f.2 contDiff_const) <| by
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    m : E
    ⊢ ∀ (n : Prod Nat Nat), Exists fun s => Exists fun C => And (LE.le 0 C) (∀ (f  …
  -/
  rintro ⟨k, n⟩
  /-
    case mk
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    m : E
    k n : Nat
    ⊢ Exists fun s => Exists fun C => And (LE.le 0 C) (∀ (f : SchwartzMap E (Conti …
  -/
  use {(k, n)}, ‖m‖, norm_nonneg _
  /-
    case right
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    m : E
    k n : Nat
    ⊢ ∀ (f : SchwartzMap E (ContinuousLinearMap (RingHom.id Real) E F)) (x : E), L …
  -/
  intro f x
  refine le_trans
    (mul_le_mul_of_nonneg_left (norm_iteratedFDeriv_clm_apply_const f.2 (mod_cast le_top))
      (by positivity)) ?_
  /-
    case right
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    m : E
    k n : Nat
    f : SchwartzMap E (ContinuousLinearMap (RingHom.id Real) E F)
    x : E
    ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm x) { fst := k, snd := n }.1) (HMul.hM …
  -/
  move_mul [‖m‖]
  /-
    case right
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    m : E
    k n : Nat
    f : SchwartzMap E (ContinuousLinearMap (RingHom.id Real) E F)
    x : E
    ⊢ LE.le (HMul.hMul (HMul.hMul (HPow.hPow (Norm.norm x) { fst := k, snd := n }. …
  -/
  gcongr ?_ * ‖m‖
  /-
    case right.h
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    m : E
    k n : Nat
    f : SchwartzMap E (ContinuousLinearMap (RingHom.id Real) E F)
    x : E
    ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm x) { fst := k, snd := n }.1) (Norm.no …
  -/
  simp only [Finset.sup_singleton, schwartzSeminormFamily_apply, le_seminorm]
  /-
    🎉 no goals
  -/


/-- The map `f ↦ (x ↦ B (f x) (g x))` as a continuous `𝕜`-linear map on Schwartz space,
where `B` is a continuous `𝕜`-linear map and `g` is a function of temperate growth. -/
def bilinLeftCLM (B : E →L[𝕜] F →L[𝕜] G) {g : D → F} (hg : g.HasTemperateGrowth) :
    𝓢(D, E) →L[𝕜] 𝓢(D, G) := by
  refine mkCLM (fun f x => B (f x) (g x))
    (fun _ _ _ => by
      simp only [map_add, add_left_inj, Pi.add_apply, eq_self_iff_true,
        ContinuousLinearMap.add_apply])
    (fun _ _ _ => by
      simp only [smul_apply, map_smul, ContinuousLinearMap.coe_smul', Pi.smul_apply,
        RingHom.id_apply])
    (fun f => (B.bilinearRestrictScalars ℝ).isBoundedBilinearMap.contDiff.comp
      (f.smooth'.prod hg.1)) ?_
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAlgebra Real 𝕜
    inst✝⁶ : NormedAddCommGroup D
    inst✝⁵ : NormedSpace Real D
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : NormedSpace Real G
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    g : D → F
    hg : Function.HasTemperateGrowth g
    ⊢ ∀ (n : Prod Nat Nat), Exists fun s => Exists fun C => And (LE.le 0 C) (∀ (f  …
  -/
  rintro ⟨k, n⟩
  /-
    case mk
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAlgebra Real 𝕜
    inst✝⁶ : NormedAddCommGroup D
    inst✝⁵ : NormedSpace Real D
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : NormedSpace Real G
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    g : D → F
    hg : Function.HasTemperateGrowth g
    k n : Nat
    ⊢ Exists fun s => Exists fun C => And (LE.le 0 C) (∀ (f : SchwartzMap D E) (x  …
  -/
  rcases hg.norm_iteratedFDeriv_le_uniform_aux n with ⟨l, C, hC, hgrowth⟩
  use
    Finset.Iic (l + k, n), ‖B‖ * ((n : ℝ) + (1 : ℝ)) * n.choose (n / 2) * (C * 2 ^ (l + k)),
    by positivity
  /-
    case right
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAlgebra Real 𝕜
    inst✝⁶ : NormedAddCommGroup D
    inst✝⁵ : NormedSpace Real D
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : NormedSpace Real G
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    g : D → F
    hg : Function.HasTemperateGrowth g
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    hgrowth : ∀ (N : Nat), LE.le N n → ∀ (x : D), LE.le (Norm.norm (iteratedFDeriv …
    ⊢ ∀ (f : SchwartzMap D E) (x : D), LE.le (HMul.hMul (HPow.hPow (Norm.norm x) { …
  -/
  intro f x
  /-
    case right
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAlgebra Real 𝕜
    inst✝⁶ : NormedAddCommGroup D
    inst✝⁵ : NormedSpace Real D
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : NormedSpace Real G
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    g : D → F
    hg : Function.HasTemperateGrowth g
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    hgrowth : ∀ (N : Nat), LE.le N n → ∀ (x : D), LE.le (Norm.norm (iteratedFDeriv …
    f : SchwartzMap D E
    x : D
    ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm x) { fst := k, snd := n }.1) (Norm.no …
  -/
  have hxk : 0 ≤ ‖x‖ ^ k := by positivity
  /-
    case right
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAlgebra Real 𝕜
    inst✝⁶ : NormedAddCommGroup D
    inst✝⁵ : NormedSpace Real D
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : NormedSpace Real G
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    g : D → F
    hg : Function.HasTemperateGrowth g
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    hgrowth : ∀ (N : Nat), LE.le N n → ∀ (x : D), LE.le (Norm.norm (iteratedFDeriv …
    f : SchwartzMap D E
    x : D
    hxk : LE.le 0 (HPow.hPow (Norm.norm x) k)
    ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm x) { fst := k, snd := n }.1) (Norm.no …
  -/
  simp_rw [← ContinuousLinearMap.bilinearRestrictScalars_apply_apply ℝ B]
  have hnorm_mul :=
    ContinuousLinearMap.norm_iteratedFDeriv_le_of_bilinear (B.bilinearRestrictScalars ℝ)
    f.smooth' hg.1 x (n := n) (mod_cast le_top)
  /-
    case right
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAlgebra Real 𝕜
    inst✝⁶ : NormedAddCommGroup D
    inst✝⁵ : NormedSpace Real D
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : NormedSpace Real G
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    g : D → F
    hg : Function.HasTemperateGrowth g
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    hgrowth : ∀ (N : Nat), LE.le N n → ∀ (x : D), LE.le (Norm.norm (iteratedFDeriv …
    f : SchwartzMap D E
    x : D
    hxk : LE.le 0 (HPow.hPow (Norm.norm x) k)
    hnorm_mul : LE.le (Norm.norm (iteratedFDeriv Real n (fun y => ((ContinuousLine …
    ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm x) k) (Norm.norm (iteratedFDeriv Real …
  -/
  refine le_trans (mul_le_mul_of_nonneg_left hnorm_mul hxk) ?_
  /-
    case right
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAlgebra Real 𝕜
    inst✝⁶ : NormedAddCommGroup D
    inst✝⁵ : NormedSpace Real D
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : NormedSpace Real G
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    g : D → F
    hg : Function.HasTemperateGrowth g
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    hgrowth : ∀ (N : Nat), LE.le N n → ∀ (x : D), LE.le (Norm.norm (iteratedFDeriv …
    f : SchwartzMap D E
    x : D
    hxk : LE.le 0 (HPow.hPow (Norm.norm x) k)
    hnorm_mul : LE.le (Norm.norm (iteratedFDeriv Real n (fun y => ((ContinuousLine …
    ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm x) k) (HMul.hMul (Norm.norm (Continuo …
  -/
  rw [ContinuousLinearMap.norm_bilinearRestrictScalars]
  /-
    case right
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAlgebra Real 𝕜
    inst✝⁶ : NormedAddCommGroup D
    inst✝⁵ : NormedSpace Real D
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : NormedSpace Real G
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    g : D → F
    hg : Function.HasTemperateGrowth g
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    hgrowth : ∀ (N : Nat), LE.le N n → ∀ (x : D), LE.le (Norm.norm (iteratedFDeriv …
    f : SchwartzMap D E
    x : D
    hxk : LE.le 0 (HPow.hPow (Norm.norm x) k)
    hnorm_mul : LE.le (Norm.norm (iteratedFDeriv Real n (fun y => ((ContinuousLine …
    ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm x) k) (HMul.hMul (Norm.norm B) ((Fins …
  -/
  move_mul [← ‖B‖]
  /-
    case right
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAlgebra Real 𝕜
    inst✝⁶ : NormedAddCommGroup D
    inst✝⁵ : NormedSpace Real D
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : NormedSpace Real G
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    g : D → F
    hg : Function.HasTemperateGrowth g
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    hgrowth : ∀ (N : Nat), LE.le N n → ∀ (x : D), LE.le (Norm.norm (iteratedFDeriv …
    f : SchwartzMap D E
    x : D
    hxk : LE.le 0 (HPow.hPow (Norm.norm x) k)
    hnorm_mul : LE.le (Norm.norm (iteratedFDeriv Real n (fun y => ((ContinuousLine …
    ⊢ LE.le (HMul.hMul (HMul.hMul (Norm.norm B) (HPow.hPow (Norm.norm x) k)) ((Fin …
  -/
  simp_rw [mul_assoc ‖B‖]
  /-
    case right
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAlgebra Real 𝕜
    inst✝⁶ : NormedAddCommGroup D
    inst✝⁵ : NormedSpace Real D
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : NormedSpace Real G
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    g : D → F
    hg : Function.HasTemperateGrowth g
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    hgrowth : ∀ (N : Nat), LE.le N n → ∀ (x : D), LE.le (Norm.norm (iteratedFDeriv …
    f : SchwartzMap D E
    x : D
    hxk : LE.le 0 (HPow.hPow (Norm.norm x) k)
    hnorm_mul : LE.le (Norm.norm (iteratedFDeriv Real n (fun y => ((ContinuousLine …
    ⊢ LE.le (HMul.hMul (Norm.norm B) (HMul.hMul (HPow.hPow (Norm.norm x) k) ((Fins …
  -/
  gcongr _ * ?_
  /-
    case right.h
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAlgebra Real 𝕜
    inst✝⁶ : NormedAddCommGroup D
    inst✝⁵ : NormedSpace Real D
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : NormedSpace Real G
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    g : D → F
    hg : Function.HasTemperateGrowth g
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    hgrowth : ∀ (N : Nat), LE.le N n → ∀ (x : D), LE.le (Norm.norm (iteratedFDeriv …
    f : SchwartzMap D E
    x : D
    hxk : LE.le 0 (HPow.hPow (Norm.norm x) k)
    hnorm_mul : LE.le (Norm.norm (iteratedFDeriv Real n (fun y => ((ContinuousLine …
    ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm x) k) ((Finset.range (HAdd.hAdd n 1)) …
  -/
  rw [Finset.mul_sum]
  /-
    case right.h
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAlgebra Real 𝕜
    inst✝⁶ : NormedAddCommGroup D
    inst✝⁵ : NormedSpace Real D
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : NormedSpace Real G
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    g : D → F
    hg : Function.HasTemperateGrowth g
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    hgrowth : ∀ (N : Nat), LE.le N n → ∀ (x : D), LE.le (Norm.norm (iteratedFDeriv …
    f : SchwartzMap D E
    x : D
    hxk : LE.le 0 (HPow.hPow (Norm.norm x) k)
    hnorm_mul : LE.le (Norm.norm (iteratedFDeriv Real n (fun y => ((ContinuousLine …
    ⊢ LE.le ((Finset.range (HAdd.hAdd n 1)).sum fun i => HMul.hMul (HPow.hPow (Nor …
  -/
  have : (∑ _x ∈ Finset.range (n + 1), (1 : ℝ)) = n + 1 := by simp
  /-
    case right.h
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAlgebra Real 𝕜
    inst✝⁶ : NormedAddCommGroup D
    inst✝⁵ : NormedSpace Real D
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : NormedSpace Real G
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    g : D → F
    hg : Function.HasTemperateGrowth g
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    hgrowth : ∀ (N : Nat), LE.le N n → ∀ (x : D), LE.le (Norm.norm (iteratedFDeriv …
    f : SchwartzMap D E
    x : D
    hxk : LE.le 0 (HPow.hPow (Norm.norm x) k)
    hnorm_mul : LE.le (Norm.norm (iteratedFDeriv Real n (fun y => ((ContinuousLine …
    this : Eq ((Finset.range (HAdd.hAdd n 1)).sum fun _x => 1) (HAdd.hAdd (↑n) 1)
    ⊢ LE.le ((Finset.range (HAdd.hAdd n 1)).sum fun i => HMul.hMul (HPow.hPow (Nor …
  -/
  simp_rw [mul_assoc ((n : ℝ) + 1)]
  /-
    case right.h
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAlgebra Real 𝕜
    inst✝⁶ : NormedAddCommGroup D
    inst✝⁵ : NormedSpace Real D
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : NormedSpace Real G
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    g : D → F
    hg : Function.HasTemperateGrowth g
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    hgrowth : ∀ (N : Nat), LE.le N n → ∀ (x : D), LE.le (Norm.norm (iteratedFDeriv …
    f : SchwartzMap D E
    x : D
    hxk : LE.le 0 (HPow.hPow (Norm.norm x) k)
    hnorm_mul : LE.le (Norm.norm (iteratedFDeriv Real n (fun y => ((ContinuousLine …
    this : Eq ((Finset.range (HAdd.hAdd n 1)).sum fun _x => 1) (HAdd.hAdd (↑n) 1)
    ⊢ LE.le ((Finset.range (HAdd.hAdd n 1)).sum fun i => HMul.hMul (HPow.hPow (Nor …
  -/
  rw [← this, Finset.sum_mul]
  /-
    case right.h
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAlgebra Real 𝕜
    inst✝⁶ : NormedAddCommGroup D
    inst✝⁵ : NormedSpace Real D
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : NormedSpace Real G
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    g : D → F
    hg : Function.HasTemperateGrowth g
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    hgrowth : ∀ (N : Nat), LE.le N n → ∀ (x : D), LE.le (Norm.norm (iteratedFDeriv …
    f : SchwartzMap D E
    x : D
    hxk : LE.le 0 (HPow.hPow (Norm.norm x) k)
    hnorm_mul : LE.le (Norm.norm (iteratedFDeriv Real n (fun y => ((ContinuousLine …
    this : Eq ((Finset.range (HAdd.hAdd n 1)).sum fun _x => 1) (HAdd.hAdd (↑n) 1)
    ⊢ LE.le ((Finset.range (HAdd.hAdd n 1)).sum fun i => HMul.hMul (HPow.hPow (Nor …
  -/
  refine Finset.sum_le_sum fun i hi => ?_
  /-
    case right.h
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAlgebra Real 𝕜
    inst✝⁶ : NormedAddCommGroup D
    inst✝⁵ : NormedSpace Real D
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : NormedSpace Real G
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    g : D → F
    hg : Function.HasTemperateGrowth g
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    hgrowth : ∀ (N : Nat), LE.le N n → ∀ (x : D), LE.le (Norm.norm (iteratedFDeriv …
    f : SchwartzMap D E
    x : D
    hxk : LE.le 0 (HPow.hPow (Norm.norm x) k)
    hnorm_mul : LE.le (Norm.norm (iteratedFDeriv Real n (fun y => ((ContinuousLine …
    this : Eq ((Finset.range (HAdd.hAdd n 1)).sum fun _x => 1) (HAdd.hAdd (↑n) 1)
    i : Nat
    hi : Membership.mem (Finset.range (HAdd.hAdd n 1)) i
    ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm x) k) (HMul.hMul (HMul.hMul (↑(n.choo …
  -/
  simp only [one_mul]
  /-
    case right.h
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAlgebra Real 𝕜
    inst✝⁶ : NormedAddCommGroup D
    inst✝⁵ : NormedSpace Real D
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : NormedSpace Real G
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    g : D → F
    hg : Function.HasTemperateGrowth g
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    hgrowth : ∀ (N : Nat), LE.le N n → ∀ (x : D), LE.le (Norm.norm (iteratedFDeriv …
    f : SchwartzMap D E
    x : D
    hxk : LE.le 0 (HPow.hPow (Norm.norm x) k)
    hnorm_mul : LE.le (Norm.norm (iteratedFDeriv Real n (fun y => ((ContinuousLine …
    this : Eq ((Finset.range (HAdd.hAdd n 1)).sum fun _x => 1) (HAdd.hAdd (↑n) 1)
    i : Nat
    hi : Membership.mem (Finset.range (HAdd.hAdd n 1)) i
    ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm x) k) (HMul.hMul (HMul.hMul (↑(n.choo …
  -/
  move_mul [(Nat.choose n i : ℝ), (Nat.choose n (n / 2) : ℝ)]
  /-
    case right.h
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAlgebra Real 𝕜
    inst✝⁶ : NormedAddCommGroup D
    inst✝⁵ : NormedSpace Real D
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : NormedSpace Real G
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    g : D → F
    hg : Function.HasTemperateGrowth g
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    hgrowth : ∀ (N : Nat), LE.le N n → ∀ (x : D), LE.le (Norm.norm (iteratedFDeriv …
    f : SchwartzMap D E
    x : D
    hxk : LE.le 0 (HPow.hPow (Norm.norm x) k)
    hnorm_mul : LE.le (Norm.norm (iteratedFDeriv Real n (fun y => ((ContinuousLine …
    this : Eq ((Finset.range (HAdd.hAdd n 1)).sum fun _x => 1) (HAdd.hAdd (↑n) 1)
    i : Nat
    hi : Membership.mem (Finset.range (HAdd.hAdd n 1)) i
    ⊢ LE.le (HMul.hMul (HMul.hMul (HMul.hMul (HPow.hPow (Norm.norm x) k) (Norm.nor …
  -/
  gcongr ?_ * ?_
  /-
    case right.h.h₁
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAlgebra Real 𝕜
    inst✝⁶ : NormedAddCommGroup D
    inst✝⁵ : NormedSpace Real D
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : NormedSpace Real G
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    g : D → F
    hg : Function.HasTemperateGrowth g
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    hgrowth : ∀ (N : Nat), LE.le N n → ∀ (x : D), LE.le (Norm.norm (iteratedFDeriv …
    f : SchwartzMap D E
    x : D
    hxk : LE.le 0 (HPow.hPow (Norm.norm x) k)
    hnorm_mul : LE.le (Norm.norm (iteratedFDeriv Real n (fun y => ((ContinuousLine …
    this : Eq ((Finset.range (HAdd.hAdd n 1)).sum fun _x => 1) (HAdd.hAdd (↑n) 1)
    i : Nat
    hi : Membership.mem (Finset.range (HAdd.hAdd n 1)) i
    ⊢ LE.le (HMul.hMul (HMul.hMul (HPow.hPow (Norm.norm x) k) (Norm.norm (iterated …
  -/
  swap
    /-
      case right.h.h₂
      𝕜 : Type u_1
      𝕜' : Type u_2
      D : Type u_3
      E : Type u_4
      F : Type u_5
      G : Type u_6
      V : Type u_7
      inst✝¹² : NormedAddCommGroup E
      inst✝¹¹ : NormedSpace Real E
      inst✝¹⁰ : NormedAddCommGroup F
      inst✝⁹ : NormedSpace Real F
      inst✝⁸ : NontriviallyNormedField 𝕜
      inst✝⁷ : NormedAlgebra Real 𝕜
      inst✝⁶ : NormedAddCommGroup D
      inst✝⁵ : NormedSpace Real D
      inst✝⁴ : NormedAddCommGroup G
      inst✝³ : NormedSpace Real G
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : NormedSpace 𝕜 G
      B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
      g : D → F
      hg : Function.HasTemperateGrowth g
      k n l : Nat
      C : Real
      hC : LE.le 0 C
      hgrowth : ∀ (N : Nat), LE.le N n → ∀ (x : D), LE.le (Norm.norm (iteratedFDeriv …
      f : SchwartzMap D E
      x : D
      hxk : LE.le 0 (HPow.hPow (Norm.norm x) k)
      hnorm_mul : LE.le (Norm.norm (iteratedFDeriv Real n (fun y => ((ContinuousLine …
      this : Eq ((Finset.range (HAdd.hAdd n 1)).sum fun _x => 1) (HAdd.hAdd (↑n) 1)
      i : Nat
      hi : Membership.mem (Finset.range (HAdd.hAdd n 1)) i
      ⊢ LE.le ↑(n.choose i) ↑(n.choose (HDiv.hDiv n 2))
    -/
  · norm_cast
    /-
      case right.h.h₂
      𝕜 : Type u_1
      𝕜' : Type u_2
      D : Type u_3
      E : Type u_4
      F : Type u_5
      G : Type u_6
      V : Type u_7
      inst✝¹² : NormedAddCommGroup E
      inst✝¹¹ : NormedSpace Real E
      inst✝¹⁰ : NormedAddCommGroup F
      inst✝⁹ : NormedSpace Real F
      inst✝⁸ : NontriviallyNormedField 𝕜
      inst✝⁷ : NormedAlgebra Real 𝕜
      inst✝⁶ : NormedAddCommGroup D
      inst✝⁵ : NormedSpace Real D
      inst✝⁴ : NormedAddCommGroup G
      inst✝³ : NormedSpace Real G
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : NormedSpace 𝕜 G
      B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
      g : D → F
      hg : Function.HasTemperateGrowth g
      k n l : Nat
      C : Real
      hC : LE.le 0 C
      hgrowth : ∀ (N : Nat), LE.le N n → ∀ (x : D), LE.le (Norm.norm (iteratedFDeriv …
      f : SchwartzMap D E
      x : D
      hxk : LE.le 0 (HPow.hPow (Norm.norm x) k)
      hnorm_mul : LE.le (Norm.norm (iteratedFDeriv Real n (fun y => ((ContinuousLine …
      this : Eq ((Finset.range (HAdd.hAdd n 1)).sum fun _x => 1) (HAdd.hAdd (↑n) 1)
      i : Nat
      hi : Membership.mem (Finset.range (HAdd.hAdd n 1)) i
      ⊢ LE.le (n.choose i) (n.choose (HDiv.hDiv n 2))
    -/
    exact i.choose_le_middle n
    /-
      🎉 no goals
    -/
  /-
    case right.h.h₁
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAlgebra Real 𝕜
    inst✝⁶ : NormedAddCommGroup D
    inst✝⁵ : NormedSpace Real D
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : NormedSpace Real G
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    g : D → F
    hg : Function.HasTemperateGrowth g
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    hgrowth : ∀ (N : Nat), LE.le N n → ∀ (x : D), LE.le (Norm.norm (iteratedFDeriv …
    f : SchwartzMap D E
    x : D
    hxk : LE.le 0 (HPow.hPow (Norm.norm x) k)
    hnorm_mul : LE.le (Norm.norm (iteratedFDeriv Real n (fun y => ((ContinuousLine …
    this : Eq ((Finset.range (HAdd.hAdd n 1)).sum fun _x => 1) (HAdd.hAdd (↑n) 1)
    i : Nat
    hi : Membership.mem (Finset.range (HAdd.hAdd n 1)) i
    ⊢ LE.le (HMul.hMul (HMul.hMul (HPow.hPow (Norm.norm x) k) (Norm.norm (iterated …
  -/
  specialize hgrowth (n - i) (by simp only [tsub_le_self]) x
  /-
    case right.h.h₁
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAlgebra Real 𝕜
    inst✝⁶ : NormedAddCommGroup D
    inst✝⁵ : NormedSpace Real D
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : NormedSpace Real G
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    g : D → F
    hg : Function.HasTemperateGrowth g
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    f : SchwartzMap D E
    x : D
    hxk : LE.le 0 (HPow.hPow (Norm.norm x) k)
    hnorm_mul : LE.le (Norm.norm (iteratedFDeriv Real n (fun y => ((ContinuousLine …
    this : Eq ((Finset.range (HAdd.hAdd n 1)).sum fun _x => 1) (HAdd.hAdd (↑n) 1)
    i : Nat
    hi : Membership.mem (Finset.range (HAdd.hAdd n 1)) i
    hgrowth : LE.le (Norm.norm (iteratedFDeriv Real (HSub.hSub n i) g x)) (HMul.hM …
    ⊢ LE.le (HMul.hMul (HMul.hMul (HPow.hPow (Norm.norm x) k) (Norm.norm (iterated …
  -/
  refine le_trans (mul_le_mul_of_nonneg_left hgrowth (by positivity)) ?_
  /-
    case right.h.h₁
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAlgebra Real 𝕜
    inst✝⁶ : NormedAddCommGroup D
    inst✝⁵ : NormedSpace Real D
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : NormedSpace Real G
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    g : D → F
    hg : Function.HasTemperateGrowth g
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    f : SchwartzMap D E
    x : D
    hxk : LE.le 0 (HPow.hPow (Norm.norm x) k)
    hnorm_mul : LE.le (Norm.norm (iteratedFDeriv Real n (fun y => ((ContinuousLine …
    this : Eq ((Finset.range (HAdd.hAdd n 1)).sum fun _x => 1) (HAdd.hAdd (↑n) 1)
    i : Nat
    hi : Membership.mem (Finset.range (HAdd.hAdd n 1)) i
    hgrowth : LE.le (Norm.norm (iteratedFDeriv Real (HSub.hSub n i) g x)) (HMul.hM …
    ⊢ LE.le (HMul.hMul (HMul.hMul (HPow.hPow (Norm.norm x) k) (Norm.norm (iterated …
  -/
  move_mul [C]
  /-
    case right.h.h₁
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAlgebra Real 𝕜
    inst✝⁶ : NormedAddCommGroup D
    inst✝⁵ : NormedSpace Real D
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : NormedSpace Real G
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    g : D → F
    hg : Function.HasTemperateGrowth g
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    f : SchwartzMap D E
    x : D
    hxk : LE.le 0 (HPow.hPow (Norm.norm x) k)
    hnorm_mul : LE.le (Norm.norm (iteratedFDeriv Real n (fun y => ((ContinuousLine …
    this : Eq ((Finset.range (HAdd.hAdd n 1)).sum fun _x => 1) (HAdd.hAdd (↑n) 1)
    i : Nat
    hi : Membership.mem (Finset.range (HAdd.hAdd n 1)) i
    hgrowth : LE.le (Norm.norm (iteratedFDeriv Real (HSub.hSub n i) g x)) (HMul.hM …
    ⊢ LE.le (HMul.hMul (HMul.hMul (HMul.hMul (HPow.hPow (Norm.norm x) k) (Norm.nor …
  -/
  gcongr ?_ * C
  /-
    case right.h.h₁.h
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAlgebra Real 𝕜
    inst✝⁶ : NormedAddCommGroup D
    inst✝⁵ : NormedSpace Real D
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : NormedSpace Real G
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    g : D → F
    hg : Function.HasTemperateGrowth g
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    f : SchwartzMap D E
    x : D
    hxk : LE.le 0 (HPow.hPow (Norm.norm x) k)
    hnorm_mul : LE.le (Norm.norm (iteratedFDeriv Real n (fun y => ((ContinuousLine …
    this : Eq ((Finset.range (HAdd.hAdd n 1)).sum fun _x => 1) (HAdd.hAdd (↑n) 1)
    i : Nat
    hi : Membership.mem (Finset.range (HAdd.hAdd n 1)) i
    hgrowth : LE.le (Norm.norm (iteratedFDeriv Real (HSub.hSub n i) g x)) (HMul.hM …
    ⊢ LE.le (HMul.hMul (HMul.hMul (HPow.hPow (Norm.norm x) k) (Norm.norm (iterated …
  -/
  rw [Finset.mem_range_succ_iff] at hi
  /-
    case right.h.h₁.h
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAlgebra Real 𝕜
    inst✝⁶ : NormedAddCommGroup D
    inst✝⁵ : NormedSpace Real D
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : NormedSpace Real G
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    g : D → F
    hg : Function.HasTemperateGrowth g
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    f : SchwartzMap D E
    x : D
    hxk : LE.le 0 (HPow.hPow (Norm.norm x) k)
    hnorm_mul : LE.le (Norm.norm (iteratedFDeriv Real n (fun y => ((ContinuousLine …
    this : Eq ((Finset.range (HAdd.hAdd n 1)).sum fun _x => 1) (HAdd.hAdd (↑n) 1)
    i : Nat
    hi : LE.le i n
    hgrowth : LE.le (Norm.norm (iteratedFDeriv Real (HSub.hSub n i) g x)) (HMul.hM …
    ⊢ LE.le (HMul.hMul (HMul.hMul (HPow.hPow (Norm.norm x) k) (Norm.norm (iterated …
  -/
  change i ≤ (l + k, n).snd at hi
  /-
    case right.h.h₁.h
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAlgebra Real 𝕜
    inst✝⁶ : NormedAddCommGroup D
    inst✝⁵ : NormedSpace Real D
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : NormedSpace Real G
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    g : D → F
    hg : Function.HasTemperateGrowth g
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    f : SchwartzMap D E
    x : D
    hxk : LE.le 0 (HPow.hPow (Norm.norm x) k)
    hnorm_mul : LE.le (Norm.norm (iteratedFDeriv Real n (fun y => ((ContinuousLine …
    this : Eq ((Finset.range (HAdd.hAdd n 1)).sum fun _x => 1) (HAdd.hAdd (↑n) 1)
    i : Nat
    hgrowth : LE.le (Norm.norm (iteratedFDeriv Real (HSub.hSub n i) g x)) (HMul.hM …
    hi : LE.le i { fst := HAdd.hAdd l k, snd := n }.2
    ⊢ LE.le (HMul.hMul (HMul.hMul (HPow.hPow (Norm.norm x) k) (Norm.norm (iterated …
  -/
  refine le_trans ?_ (one_add_le_sup_seminorm_apply le_rfl hi f x)
  /-
    case right.h.h₁.h
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAlgebra Real 𝕜
    inst✝⁶ : NormedAddCommGroup D
    inst✝⁵ : NormedSpace Real D
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : NormedSpace Real G
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    g : D → F
    hg : Function.HasTemperateGrowth g
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    f : SchwartzMap D E
    x : D
    hxk : LE.le 0 (HPow.hPow (Norm.norm x) k)
    hnorm_mul : LE.le (Norm.norm (iteratedFDeriv Real n (fun y => ((ContinuousLine …
    this : Eq ((Finset.range (HAdd.hAdd n 1)).sum fun _x => 1) (HAdd.hAdd (↑n) 1)
    i : Nat
    hgrowth : LE.le (Norm.norm (iteratedFDeriv Real (HSub.hSub n i) g x)) (HMul.hM …
    hi : LE.le i { fst := HAdd.hAdd l k, snd := n }.2
    ⊢ LE.le (HMul.hMul (HMul.hMul (HPow.hPow (Norm.norm x) k) (Norm.norm (iterated …
  -/
  rw [pow_add]
  /-
    case right.h.h₁.h
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAlgebra Real 𝕜
    inst✝⁶ : NormedAddCommGroup D
    inst✝⁵ : NormedSpace Real D
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : NormedSpace Real G
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    g : D → F
    hg : Function.HasTemperateGrowth g
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    f : SchwartzMap D E
    x : D
    hxk : LE.le 0 (HPow.hPow (Norm.norm x) k)
    hnorm_mul : LE.le (Norm.norm (iteratedFDeriv Real n (fun y => ((ContinuousLine …
    this : Eq ((Finset.range (HAdd.hAdd n 1)).sum fun _x => 1) (HAdd.hAdd (↑n) 1)
    i : Nat
    hgrowth : LE.le (Norm.norm (iteratedFDeriv Real (HSub.hSub n i) g x)) (HMul.hM …
    hi : LE.le i { fst := HAdd.hAdd l k, snd := n }.2
    ⊢ LE.le (HMul.hMul (HMul.hMul (HPow.hPow (Norm.norm x) k) (Norm.norm (iterated …
  -/
  move_mul [(1 + ‖x‖) ^ l]
  /-
    case right.h.h₁.h
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAlgebra Real 𝕜
    inst✝⁶ : NormedAddCommGroup D
    inst✝⁵ : NormedSpace Real D
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : NormedSpace Real G
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    g : D → F
    hg : Function.HasTemperateGrowth g
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    f : SchwartzMap D E
    x : D
    hxk : LE.le 0 (HPow.hPow (Norm.norm x) k)
    hnorm_mul : LE.le (Norm.norm (iteratedFDeriv Real n (fun y => ((ContinuousLine …
    this : Eq ((Finset.range (HAdd.hAdd n 1)).sum fun _x => 1) (HAdd.hAdd (↑n) 1)
    i : Nat
    hgrowth : LE.le (Norm.norm (iteratedFDeriv Real (HSub.hSub n i) g x)) (HMul.hM …
    hi : LE.le i { fst := HAdd.hAdd l k, snd := n }.2
    ⊢ LE.le (HMul.hMul (HMul.hMul (HPow.hPow (Norm.norm x) k) (Norm.norm (iterated …
  -/
  gcongr
  /-
    case right.h.h₁.h.h.h.hab
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAlgebra Real 𝕜
    inst✝⁶ : NormedAddCommGroup D
    inst✝⁵ : NormedSpace Real D
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : NormedSpace Real G
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    g : D → F
    hg : Function.HasTemperateGrowth g
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    f : SchwartzMap D E
    x : D
    hxk : LE.le 0 (HPow.hPow (Norm.norm x) k)
    hnorm_mul : LE.le (Norm.norm (iteratedFDeriv Real n (fun y => ((ContinuousLine …
    this : Eq ((Finset.range (HAdd.hAdd n 1)).sum fun _x => 1) (HAdd.hAdd (↑n) 1)
    i : Nat
    hgrowth : LE.le (Norm.norm (iteratedFDeriv Real (HSub.hSub n i) g x)) (HMul.hM …
    hi : LE.le i { fst := HAdd.hAdd l k, snd := n }.2
    ⊢ LE.le (Norm.norm x) (HAdd.hAdd 1 (Norm.norm x))
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Composition with a function on the right is a continuous linear map on Schwartz space
provided that the function is temperate and growths polynomially near infinity. -/
def compCLM {g : D → E} (hg : g.HasTemperateGrowth)
    (hg_upper : ∃ (k : ℕ) (C : ℝ), ∀ x, ‖x‖ ≤ C * (1 + ‖g x‖) ^ k) : 𝓢(E, F) →L[𝕜] 𝓢(D, F) := by
  refine mkCLM (fun f x => f (g x))
    (fun _ _ _ => by simp only [add_left_inj, Pi.add_apply, eq_self_iff_true]) (fun _ _ _ => rfl)
    (fun f => f.smooth'.comp hg.1) ?_
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup D
    inst✝² : NormedSpace Real D
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    g : D → E
    hg : Function.HasTemperateGrowth g
    hg_upper : Exists fun k => Exists fun C => ∀ (x : D), LE.le (Norm.norm x) (HMu …
    ⊢ ∀ (n : Prod Nat Nat), Exists fun s => Exists fun C => And (LE.le 0 C) (∀ (f  …
  -/
  rintro ⟨k, n⟩
  /-
    case mk
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup D
    inst✝² : NormedSpace Real D
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    g : D → E
    hg : Function.HasTemperateGrowth g
    hg_upper : Exists fun k => Exists fun C => ∀ (x : D), LE.le (Norm.norm x) (HMu …
    k n : Nat
    ⊢ Exists fun s => Exists fun C => And (LE.le 0 C) (∀ (f : SchwartzMap E F) (x  …
  -/
  rcases hg.norm_iteratedFDeriv_le_uniform_aux n with ⟨l, C, hC, hgrowth⟩
  /-
    case mk.intro.intro.intro
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup D
    inst✝² : NormedSpace Real D
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    g : D → E
    hg : Function.HasTemperateGrowth g
    hg_upper : Exists fun k => Exists fun C => ∀ (x : D), LE.le (Norm.norm x) (HMu …
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    hgrowth : ∀ (N : Nat), LE.le N n → ∀ (x : D), LE.le (Norm.norm (iteratedFDeriv …
    ⊢ Exists fun s => Exists fun C => And (LE.le 0 C) (∀ (f : SchwartzMap E F) (x  …
  -/
  rcases hg_upper with ⟨kg, Cg, hg_upper'⟩
  have hCg : 1 ≤ 1 + Cg := by
    refine le_add_of_nonneg_right ?_
    specialize hg_upper' 0
    rw [norm_zero] at hg_upper'
    exact nonneg_of_mul_nonneg_left hg_upper' (by positivity)
  /-
    case mk.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup D
    inst✝² : NormedSpace Real D
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    g : D → E
    hg : Function.HasTemperateGrowth g
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    hgrowth : ∀ (N : Nat), LE.le N n → ∀ (x : D), LE.le (Norm.norm (iteratedFDeriv …
    kg : Nat
    Cg : Real
    hg_upper' : ∀ (x : D), LE.le (Norm.norm x) (HMul.hMul Cg (HPow.hPow (HAdd.hAdd …
    hCg : LE.le 1 (HAdd.hAdd 1 Cg)
    ⊢ Exists fun s => Exists fun C => And (LE.le 0 C) (∀ (f : SchwartzMap E F) (x  …
  -/
  let k' := kg * (k + l * n)
  /-
    case mk.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup D
    inst✝² : NormedSpace Real D
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    g : D → E
    hg : Function.HasTemperateGrowth g
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    hgrowth : ∀ (N : Nat), LE.le N n → ∀ (x : D), LE.le (Norm.norm (iteratedFDeriv …
    kg : Nat
    Cg : Real
    hg_upper' : ∀ (x : D), LE.le (Norm.norm x) (HMul.hMul Cg (HPow.hPow (HAdd.hAdd …
    hCg : LE.le 1 (HAdd.hAdd 1 Cg)
    k' : Nat := HMul.hMul kg (HAdd.hAdd k (HMul.hMul l n))
    ⊢ Exists fun s => Exists fun C => And (LE.le 0 C) (∀ (f : SchwartzMap E F) (x  …
  -/
  use Finset.Iic (k', n), (1 + Cg) ^ (k + l * n) * ((C + 1) ^ n * n ! * 2 ^ k'), by positivity
  /-
    case right
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup D
    inst✝² : NormedSpace Real D
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    g : D → E
    hg : Function.HasTemperateGrowth g
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    hgrowth : ∀ (N : Nat), LE.le N n → ∀ (x : D), LE.le (Norm.norm (iteratedFDeriv …
    kg : Nat
    Cg : Real
    hg_upper' : ∀ (x : D), LE.le (Norm.norm x) (HMul.hMul Cg (HPow.hPow (HAdd.hAdd …
    hCg : LE.le 1 (HAdd.hAdd 1 Cg)
    k' : Nat := HMul.hMul kg (HAdd.hAdd k (HMul.hMul l n))
    ⊢ ∀ (f : SchwartzMap E F) (x : D), LE.le (HMul.hMul (HPow.hPow (Norm.norm x) { …
  -/
  intro f x
  /-
    case right
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup D
    inst✝² : NormedSpace Real D
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    g : D → E
    hg : Function.HasTemperateGrowth g
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    hgrowth : ∀ (N : Nat), LE.le N n → ∀ (x : D), LE.le (Norm.norm (iteratedFDeriv …
    kg : Nat
    Cg : Real
    hg_upper' : ∀ (x : D), LE.le (Norm.norm x) (HMul.hMul Cg (HPow.hPow (HAdd.hAdd …
    hCg : LE.le 1 (HAdd.hAdd 1 Cg)
    k' : Nat := HMul.hMul kg (HAdd.hAdd k (HMul.hMul l n))
    f : SchwartzMap E F
    x : D
    ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm x) { fst := k, snd := n }.1) (Norm.no …
  -/
  let seminorm_f := ((Finset.Iic (k', n)).sup (schwartzSeminormFamily 𝕜 _ _)) f
  have hg_upper'' : (1 + ‖x‖) ^ (k + l * n) ≤ (1 + Cg) ^ (k + l * n) * (1 + ‖g x‖) ^ k' := by
    rw [pow_mul, ← mul_pow]
    gcongr
    rw [add_mul]
    refine add_le_add ?_ (hg_upper' x)
    nth_rw 1 [← one_mul (1 : ℝ)]
    gcongr
    apply one_le_pow₀
    simp only [le_add_iff_nonneg_right, norm_nonneg]
  have hbound :
    ∀ i, i ≤ n → ‖iteratedFDeriv ℝ i f (g x)‖ ≤ 2 ^ k' * seminorm_f / (1 + ‖g x‖) ^ k' := by
    intro i hi
    have hpos : 0 < (1 + ‖g x‖) ^ k' := by positivity
    rw [le_div_iff₀' hpos]
    change i ≤ (k', n).snd at hi
    exact one_add_le_sup_seminorm_apply le_rfl hi _ _
  have hgrowth' : ∀ N : ℕ, 1 ≤ N → N ≤ n →
      ‖iteratedFDeriv ℝ N g x‖ ≤ ((C + 1) * (1 + ‖x‖) ^ l) ^ N := by
    intro N hN₁ hN₂
    refine (hgrowth N hN₂ x).trans ?_
    rw [mul_pow]
    have hN₁' := (lt_of_lt_of_le zero_lt_one hN₁).ne'
    gcongr
    · exact le_trans (by simp [hC]) (le_self_pow₀ (by simp [hC]) hN₁')
    · refine le_self_pow₀ (one_le_pow₀ ?_) hN₁'
      simp only [le_add_iff_nonneg_right, norm_nonneg]
  /-
    case right
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup D
    inst✝² : NormedSpace Real D
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    g : D → E
    hg : Function.HasTemperateGrowth g
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    hgrowth : ∀ (N : Nat), LE.le N n → ∀ (x : D), LE.le (Norm.norm (iteratedFDeriv …
    kg : Nat
    Cg : Real
    hg_upper' : ∀ (x : D), LE.le (Norm.norm x) (HMul.hMul Cg (HPow.hPow (HAdd.hAdd …
    hCg : LE.le 1 (HAdd.hAdd 1 Cg)
    k' : Nat := HMul.hMul kg (HAdd.hAdd k (HMul.hMul l n))
    f : SchwartzMap E F
    x : D
    seminorm_f : Real := ((Finset.Iic { fst := k', snd := n }).sup (schwartzSemino …
    hg_upper'' : LE.le (HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) (HAdd.hAdd k (HMul.h …
    hbound : ∀ (i : Nat), LE.le i n → LE.le (Norm.norm (iteratedFDeriv Real i (⇑f) …
    hgrowth' : ∀ (N : Nat), LE.le 1 N → LE.le N n → LE.le (Norm.norm (iteratedFDer …
    ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm x) { fst := k, snd := n }.1) (Norm.no …
  -/
  have := norm_iteratedFDeriv_comp_le f.smooth' hg.1 (mod_cast le_top) x hbound hgrowth'
  have hxk : ‖x‖ ^ k ≤ (1 + ‖x‖) ^ k :=
    pow_le_pow_left₀ (norm_nonneg _) (by simp only [zero_le_one, le_add_iff_nonneg_left]) _
  /-
    case right
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup D
    inst✝² : NormedSpace Real D
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    g : D → E
    hg : Function.HasTemperateGrowth g
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    hgrowth : ∀ (N : Nat), LE.le N n → ∀ (x : D), LE.le (Norm.norm (iteratedFDeriv …
    kg : Nat
    Cg : Real
    hg_upper' : ∀ (x : D), LE.le (Norm.norm x) (HMul.hMul Cg (HPow.hPow (HAdd.hAdd …
    hCg : LE.le 1 (HAdd.hAdd 1 Cg)
    k' : Nat := HMul.hMul kg (HAdd.hAdd k (HMul.hMul l n))
    f : SchwartzMap E F
    x : D
    seminorm_f : Real := ((Finset.Iic { fst := k', snd := n }).sup (schwartzSemino …
    hg_upper'' : LE.le (HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) (HAdd.hAdd k (HMul.h …
    hbound : ∀ (i : Nat), LE.le i n → LE.le (Norm.norm (iteratedFDeriv Real i (⇑f) …
    hgrowth' : ∀ (N : Nat), LE.le 1 N → LE.le N n → LE.le (Norm.norm (iteratedFDer …
    this : LE.le (Norm.norm (iteratedFDeriv Real n (Function.comp f.toFun g) x)) ( …
    hxk : LE.le (HPow.hPow (Norm.norm x) k) (HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) …
    ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm x) { fst := k, snd := n }.1) (Norm.no …
  -/
  refine le_trans (mul_le_mul hxk this (by positivity) (by positivity)) ?_
  have rearrange :
    (1 + ‖x‖) ^ k *
        (n ! * (2 ^ k' * seminorm_f / (1 + ‖g x‖) ^ k') * ((C + 1) * (1 + ‖x‖) ^ l) ^ n) =
      (1 + ‖x‖) ^ (k + l * n) / (1 + ‖g x‖) ^ k' *
        ((C + 1) ^ n * n ! * 2 ^ k' * seminorm_f) := by
    rw [mul_pow, pow_add, ← pow_mul]
    ring
  /-
    case right
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup D
    inst✝² : NormedSpace Real D
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    g : D → E
    hg : Function.HasTemperateGrowth g
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    hgrowth : ∀ (N : Nat), LE.le N n → ∀ (x : D), LE.le (Norm.norm (iteratedFDeriv …
    kg : Nat
    Cg : Real
    hg_upper' : ∀ (x : D), LE.le (Norm.norm x) (HMul.hMul Cg (HPow.hPow (HAdd.hAdd …
    hCg : LE.le 1 (HAdd.hAdd 1 Cg)
    k' : Nat := HMul.hMul kg (HAdd.hAdd k (HMul.hMul l n))
    f : SchwartzMap E F
    x : D
    seminorm_f : Real := ((Finset.Iic { fst := k', snd := n }).sup (schwartzSemino …
    hg_upper'' : LE.le (HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) (HAdd.hAdd k (HMul.h …
    hbound : ∀ (i : Nat), LE.le i n → LE.le (Norm.norm (iteratedFDeriv Real i (⇑f) …
    hgrowth' : ∀ (N : Nat), LE.le 1 N → LE.le N n → LE.le (Norm.norm (iteratedFDer …
    this : LE.le (Norm.norm (iteratedFDeriv Real n (Function.comp f.toFun g) x)) ( …
    hxk : LE.le (HPow.hPow (Norm.norm x) k) (HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) …
    rearrange : Eq (HMul.hMul (HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) k) (HMul.hMul …
    ⊢ LE.le (HMul.hMul (HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) k) (HMul.hMul (HMul. …
  -/
  rw [rearrange]
  /-
    case right
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup D
    inst✝² : NormedSpace Real D
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    g : D → E
    hg : Function.HasTemperateGrowth g
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    hgrowth : ∀ (N : Nat), LE.le N n → ∀ (x : D), LE.le (Norm.norm (iteratedFDeriv …
    kg : Nat
    Cg : Real
    hg_upper' : ∀ (x : D), LE.le (Norm.norm x) (HMul.hMul Cg (HPow.hPow (HAdd.hAdd …
    hCg : LE.le 1 (HAdd.hAdd 1 Cg)
    k' : Nat := HMul.hMul kg (HAdd.hAdd k (HMul.hMul l n))
    f : SchwartzMap E F
    x : D
    seminorm_f : Real := ((Finset.Iic { fst := k', snd := n }).sup (schwartzSemino …
    hg_upper'' : LE.le (HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) (HAdd.hAdd k (HMul.h …
    hbound : ∀ (i : Nat), LE.le i n → LE.le (Norm.norm (iteratedFDeriv Real i (⇑f) …
    hgrowth' : ∀ (N : Nat), LE.le 1 N → LE.le N n → LE.le (Norm.norm (iteratedFDer …
    this : LE.le (Norm.norm (iteratedFDeriv Real n (Function.comp f.toFun g) x)) ( …
    hxk : LE.le (HPow.hPow (Norm.norm x) k) (HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) …
    rearrange : Eq (HMul.hMul (HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) k) (HMul.hMul …
    ⊢ LE.le (HMul.hMul (HDiv.hDiv (HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) (HAdd.hAd …
  -/
  have hgxk' : 0 < (1 + ‖g x‖) ^ k' := by positivity
  /-
    case right
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup D
    inst✝² : NormedSpace Real D
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    g : D → E
    hg : Function.HasTemperateGrowth g
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    hgrowth : ∀ (N : Nat), LE.le N n → ∀ (x : D), LE.le (Norm.norm (iteratedFDeriv …
    kg : Nat
    Cg : Real
    hg_upper' : ∀ (x : D), LE.le (Norm.norm x) (HMul.hMul Cg (HPow.hPow (HAdd.hAdd …
    hCg : LE.le 1 (HAdd.hAdd 1 Cg)
    k' : Nat := HMul.hMul kg (HAdd.hAdd k (HMul.hMul l n))
    f : SchwartzMap E F
    x : D
    seminorm_f : Real := ((Finset.Iic { fst := k', snd := n }).sup (schwartzSemino …
    hg_upper'' : LE.le (HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) (HAdd.hAdd k (HMul.h …
    hbound : ∀ (i : Nat), LE.le i n → LE.le (Norm.norm (iteratedFDeriv Real i (⇑f) …
    hgrowth' : ∀ (N : Nat), LE.le 1 N → LE.le N n → LE.le (Norm.norm (iteratedFDer …
    this : LE.le (Norm.norm (iteratedFDeriv Real n (Function.comp f.toFun g) x)) ( …
    hxk : LE.le (HPow.hPow (Norm.norm x) k) (HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) …
    rearrange : Eq (HMul.hMul (HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) k) (HMul.hMul …
    hgxk' : LT.lt 0 (HPow.hPow (HAdd.hAdd 1 (Norm.norm (g x))) k')
    ⊢ LE.le (HMul.hMul (HDiv.hDiv (HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) (HAdd.hAd …
  -/
  rw [← div_le_iff₀ hgxk'] at hg_upper''
  have hpos : (0 : ℝ) ≤ (C + 1) ^ n * n ! * 2 ^ k' * seminorm_f := by
    have : 0 ≤ seminorm_f := apply_nonneg _ _
    positivity
  /-
    case right
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup D
    inst✝² : NormedSpace Real D
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    g : D → E
    hg : Function.HasTemperateGrowth g
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    hgrowth : ∀ (N : Nat), LE.le N n → ∀ (x : D), LE.le (Norm.norm (iteratedFDeriv …
    kg : Nat
    Cg : Real
    hg_upper' : ∀ (x : D), LE.le (Norm.norm x) (HMul.hMul Cg (HPow.hPow (HAdd.hAdd …
    hCg : LE.le 1 (HAdd.hAdd 1 Cg)
    k' : Nat := HMul.hMul kg (HAdd.hAdd k (HMul.hMul l n))
    f : SchwartzMap E F
    x : D
    seminorm_f : Real := ((Finset.Iic { fst := k', snd := n }).sup (schwartzSemino …
    hg_upper'' : LE.le (HDiv.hDiv (HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) (HAdd.hAd …
    hbound : ∀ (i : Nat), LE.le i n → LE.le (Norm.norm (iteratedFDeriv Real i (⇑f) …
    hgrowth' : ∀ (N : Nat), LE.le 1 N → LE.le N n → LE.le (Norm.norm (iteratedFDer …
    this : LE.le (Norm.norm (iteratedFDeriv Real n (Function.comp f.toFun g) x)) ( …
    hxk : LE.le (HPow.hPow (Norm.norm x) k) (HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) …
    rearrange : Eq (HMul.hMul (HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) k) (HMul.hMul …
    hgxk' : LT.lt 0 (HPow.hPow (HAdd.hAdd 1 (Norm.norm (g x))) k')
    hpos : LE.le 0 (HMul.hMul (HMul.hMul (HMul.hMul (HPow.hPow (HAdd.hAdd C 1) n)  …
    ⊢ LE.le (HMul.hMul (HDiv.hDiv (HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) (HAdd.hAd …
  -/
  refine le_trans (mul_le_mul_of_nonneg_right hg_upper'' hpos) ?_
  /-
    case right
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup D
    inst✝² : NormedSpace Real D
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    g : D → E
    hg : Function.HasTemperateGrowth g
    k n l : Nat
    C : Real
    hC : LE.le 0 C
    hgrowth : ∀ (N : Nat), LE.le N n → ∀ (x : D), LE.le (Norm.norm (iteratedFDeriv …
    kg : Nat
    Cg : Real
    hg_upper' : ∀ (x : D), LE.le (Norm.norm x) (HMul.hMul Cg (HPow.hPow (HAdd.hAdd …
    hCg : LE.le 1 (HAdd.hAdd 1 Cg)
    k' : Nat := HMul.hMul kg (HAdd.hAdd k (HMul.hMul l n))
    f : SchwartzMap E F
    x : D
    seminorm_f : Real := ((Finset.Iic { fst := k', snd := n }).sup (schwartzSemino …
    hg_upper'' : LE.le (HDiv.hDiv (HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) (HAdd.hAd …
    hbound : ∀ (i : Nat), LE.le i n → LE.le (Norm.norm (iteratedFDeriv Real i (⇑f) …
    hgrowth' : ∀ (N : Nat), LE.le 1 N → LE.le N n → LE.le (Norm.norm (iteratedFDer …
    this : LE.le (Norm.norm (iteratedFDeriv Real n (Function.comp f.toFun g) x)) ( …
    hxk : LE.le (HPow.hPow (Norm.norm x) k) (HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) …
    rearrange : Eq (HMul.hMul (HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) k) (HMul.hMul …
    hgxk' : LT.lt 0 (HPow.hPow (HAdd.hAdd 1 (Norm.norm (g x))) k')
    hpos : LE.le 0 (HMul.hMul (HMul.hMul (HMul.hMul (HPow.hPow (HAdd.hAdd C 1) n)  …
    ⊢ LE.le (HMul.hMul (HPow.hPow (HAdd.hAdd 1 Cg) (HAdd.hAdd k (HMul.hMul l n)))  …
  -/
  rw [← mul_assoc]
  /-
    🎉 no goals
  -/


@[simp] lemma compCLM_apply {g : D → E} (hg : g.HasTemperateGrowth)
    (hg_upper : ∃ (k : ℕ) (C : ℝ), ∀ x, ‖x‖ ≤ C * (1 + ‖g x‖) ^ k) (f : 𝓢(E, F)) :
    compCLM 𝕜 hg hg_upper f = f ∘ g := rfl


/-- Composition with a function on the right is a continuous linear map on Schwartz space
provided that the function is temperate and antilipschitz. -/
def compCLMOfAntilipschitz {K : ℝ≥0} {g : D → E}
    (hg : g.HasTemperateGrowth) (h'g : AntilipschitzWith K g) :
    𝓢(E, F) →L[𝕜] 𝓢(D, F) := by
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup D
    inst✝² : NormedSpace Real D
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    K : NNReal
    g : D → E
    hg : Function.HasTemperateGrowth g
    h'g : AntilipschitzWith K g
    ⊢ ContinuousLinearMap (RingHom.id 𝕜) (SchwartzMap E F) (SchwartzMap D F)
  -/
  refine compCLM 𝕜 hg ⟨1, K * max 1 ‖g 0‖, fun x ↦ ?_⟩
  calc
  ‖x‖ ≤ K * ‖g x - g 0‖ := by
    rw [← dist_zero_right, ← dist_eq_norm]
    apply h'g.le_mul_dist
  _ ≤ K * (‖g x‖ + ‖g 0‖) := by
    gcongr
    exact norm_sub_le _ _
  _ ≤ K * (‖g x‖ + max 1 ‖g 0‖) := by
    gcongr
    exact le_max_right _ _
  _ ≤ (K * max 1 ‖g 0‖ : ℝ) * (1 + ‖g x‖) ^ 1 := by
    simp only [mul_add, add_comm (K * ‖g x‖), pow_one, mul_one, add_le_add_iff_left]
    gcongr
    exact le_mul_of_one_le_right (by positivity) (le_max_left _ _)


@[simp] lemma compCLMOfAntilipschitz_apply {K : ℝ≥0} {g : D → E} (hg : g.HasTemperateGrowth)
    (h'g : AntilipschitzWith K g) (f : 𝓢(E, F)) :
    compCLMOfAntilipschitz 𝕜 hg h'g f = f ∘ g := rfl


/-- Composition with a continuous linear equiv on the right is a continuous linear map on
Schwartz space. -/
def compCLMOfContinuousLinearEquiv (g : D ≃L[ℝ] E) :
    𝓢(E, F) →L[𝕜] 𝓢(D, F) :=
  compCLMOfAntilipschitz 𝕜 (g.toContinuousLinearMap.hasTemperateGrowth) g.antilipschitz


@[simp] lemma compCLMOfContinuousLinearEquiv_apply (g : D ≃L[ℝ] E) (f : 𝓢(E, F)) :
    compCLMOfContinuousLinearEquiv 𝕜 g f = f ∘ g := rfl


/-- The Fréchet derivative on Schwartz space as a continuous `𝕜`-linear map. -/
def fderivCLM : 𝓢(E, F) →L[𝕜] 𝓢(E, E →L[ℝ] F) :=
  mkCLM (fderiv ℝ) (fun f g _ => fderiv_add f.differentiableAt g.differentiableAt)
    (fun a f _ => fderiv_const_smul f.differentiableAt a)
    (fun f => (contDiff_succ_iff_fderiv.mp f.smooth').2.2) fun ⟨k, n⟩ =>
    ⟨{⟨k, n + 1⟩}, 1, zero_le_one, fun f x => by
      simpa only [schwartzSeminormFamily_apply, Seminorm.comp_apply, Finset.sup_singleton,
        one_smul, norm_iteratedFDeriv_fderiv, one_mul] using f.le_seminorm 𝕜 k (n + 1) x⟩


@[simp]
theorem fderivCLM_apply (f : 𝓢(E, F)) (x : E) : fderivCLM 𝕜 f x = fderiv ℝ f x :=
  rfl


/-- The 1-dimensional derivative on Schwartz space as a continuous `𝕜`-linear map. -/
def derivCLM : 𝓢(ℝ, F) →L[𝕜] 𝓢(ℝ, F) :=
  mkCLM deriv (fun f g _ => deriv_add f.differentiableAt g.differentiableAt)
    (fun a f _ => deriv_const_smul a f.differentiableAt)
    (fun f => (contDiff_succ_iff_deriv.mp f.smooth').2.2) fun ⟨k, n⟩ =>
    ⟨{⟨k, n + 1⟩}, 1, zero_le_one, fun f x => by
      simpa only [Real.norm_eq_abs, Finset.sup_singleton, schwartzSeminormFamily_apply, one_mul,
        norm_iteratedFDeriv_eq_norm_iteratedDeriv, ← iteratedDeriv_succ'] using
        f.le_seminorm' 𝕜 k (n + 1) x⟩


@[simp]
theorem derivCLM_apply (f : 𝓢(ℝ, F)) (x : ℝ) : derivCLM 𝕜 f x = deriv f x :=
  rfl


/-- The partial derivative (or directional derivative) in the direction `m : E` as a
continuous linear map on Schwartz space. -/
def pderivCLM (m : E) : 𝓢(E, F) →L[𝕜] 𝓢(E, F) :=
  (SchwartzMap.evalCLM m).comp (fderivCLM 𝕜)


@[simp]
theorem pderivCLM_apply (m : E) (f : 𝓢(E, F)) (x : E) : pderivCLM 𝕜 m f x = fderiv ℝ f x m :=
  rfl


theorem pderivCLM_eq_lineDeriv (m : E) (f : 𝓢(E, F)) (x : E) :
    pderivCLM 𝕜 m f x = lineDeriv ℝ f x m := by
  /-
    𝕜 : Type u_1
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    m : E
    f : SchwartzMap E F
    x : E
    ⊢ Eq (((SchwartzMap.pderivCLM 𝕜 m) f) x) (lineDeriv Real (⇑f) x m)
  -/
  simp only [pderivCLM_apply, f.differentiableAt.lineDeriv_eq_fderiv]
  /-
    🎉 no goals
  -/


/-- The iterated partial derivative (or directional derivative) as a continuous linear map on
Schwartz space. -/
def iteratedPDeriv {n : ℕ} : (Fin n → E) → 𝓢(E, F) →L[𝕜] 𝓢(E, F) :=
  Nat.recOn n (fun _ => ContinuousLinearMap.id 𝕜 _) fun _ rec x =>
    (pderivCLM 𝕜 (x 0)).comp (rec (Fin.tail x))


@[simp]
theorem iteratedPDeriv_zero (m : Fin 0 → E) (f : 𝓢(E, F)) : iteratedPDeriv 𝕜 m f = f :=
  rfl


@[simp]
theorem iteratedPDeriv_one (m : Fin 1 → E) (f : 𝓢(E, F)) :
    iteratedPDeriv 𝕜 m f = pderivCLM 𝕜 (m 0) f :=
  rfl


theorem iteratedPDeriv_succ_left {n : ℕ} (m : Fin (n + 1) → E) (f : 𝓢(E, F)) :
    iteratedPDeriv 𝕜 m f = pderivCLM 𝕜 (m 0) (iteratedPDeriv 𝕜 (Fin.tail m) f) :=
  rfl


theorem iteratedPDeriv_succ_right {n : ℕ} (m : Fin (n + 1) → E) (f : 𝓢(E, F)) :
    iteratedPDeriv 𝕜 m f = iteratedPDeriv 𝕜 (Fin.init m) (pderivCLM 𝕜 (m (Fin.last n)) f) := by
  induction n with
  | zero =>
    rw [iteratedPDeriv_zero, iteratedPDeriv_one]
    rfl
  -- The proof is `∂^{n + 2} = ∂ ∂^{n + 1} = ∂ ∂^n ∂ = ∂^{n+1} ∂`
  | succ n IH =>
    have hmzero : Fin.init m 0 = m 0 := by simp only [Fin.init_def, Fin.castSucc_zero]
    have hmtail : Fin.tail m (Fin.last n) = m (Fin.last n.succ) := by
      simp only [Fin.tail_def, Fin.succ_last]
    calc
      _ = pderivCLM 𝕜 (m 0) (iteratedPDeriv 𝕜 _ f) := iteratedPDeriv_succ_left _ _ _
      _ = pderivCLM 𝕜 (m 0) ((iteratedPDeriv 𝕜 _) ((pderivCLM 𝕜 _) f)) := by
        congr 1
        exact IH _
      _ = _ := by
        simp only [hmtail, iteratedPDeriv_succ_left, hmzero, Fin.tail_init_eq_init_tail]


theorem iteratedPDeriv_eq_iteratedFDeriv {n : ℕ} {m : Fin n → E} {f : 𝓢(E, F)} {x : E} :
    iteratedPDeriv 𝕜 m f x = iteratedFDeriv ℝ n f x m := by
  induction n generalizing x with
  | zero => simp
  | succ n ih =>
    simp only [iteratedPDeriv_succ_left, iteratedFDeriv_succ_apply_left]
    rw [← fderiv_continuousMultilinear_apply_const_apply]
    · simp [← ih]
    · exact f.smooth'.differentiable_iteratedFDeriv (mod_cast ENat.coe_lt_top n) x



variable (𝕜 μ) in
lemma integral_pow_mul_iteratedFDeriv_le (f : 𝓢(D, V)) (k n : ℕ) :
    ∫ x, ‖x‖ ^ k * ‖iteratedFDeriv ℝ n f x‖ ∂μ ≤ 2 ^ μ.integrablePower *
      (∫ x, (1 + ‖x‖) ^ (- (μ.integrablePower : ℝ)) ∂μ) *
        (SchwartzMap.seminorm 𝕜 0 n f + SchwartzMap.seminorm 𝕜 (k + μ.integrablePower) n f) :=
  integral_pow_mul_le_of_le_of_pow_mul_le (norm_iteratedFDeriv_le_seminorm ℝ _ _)
    (le_seminorm ℝ _ _ _)


variable (μ) in
lemma integrable_pow_mul_iteratedFDeriv
    (f : 𝓢(D, V))
    (k n : ℕ) : Integrable (fun x ↦ ‖x‖ ^ k * ‖iteratedFDeriv ℝ n f x‖) μ :=
  integrable_of_le_of_pow_mul_le (norm_iteratedFDeriv_le_seminorm ℝ _ _) (le_seminorm ℝ _ _ _)
    ((f.smooth ⊤).continuous_iteratedFDeriv (mod_cast le_top)).aestronglyMeasurable


variable (μ) in
lemma integrable_pow_mul (f : 𝓢(D, V))
    (k : ℕ) : Integrable (fun x ↦ ‖x‖ ^ k * ‖f x‖) μ := by
  /-
    D : Type u_3
    V : Type u_7
    inst✝⁶ : NormedAddCommGroup D
    inst✝⁵ : NormedSpace Real D
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : NormedSpace Real V
    inst✝² : MeasurableSpace D
    μ : MeasureTheory.Measure D
    hμ : μ.HasTemperateGrowth
    inst✝¹ : BorelSpace D
    inst✝ : SecondCountableTopology D
    f : SchwartzMap D V
    k : Nat
    ⊢ MeasureTheory.Integrable (fun x => HMul.hMul (HPow.hPow (Norm.norm x) k) (No …
  -/
  convert integrable_pow_mul_iteratedFDeriv μ f k 0 with x
  /-
    case h.e'_6.h.h.e'_6
    D : Type u_3
    V : Type u_7
    inst✝⁶ : NormedAddCommGroup D
    inst✝⁵ : NormedSpace Real D
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : NormedSpace Real V
    inst✝² : MeasurableSpace D
    μ : MeasureTheory.Measure D
    hμ : μ.HasTemperateGrowth
    inst✝¹ : BorelSpace D
    inst✝ : SecondCountableTopology D
    f : SchwartzMap D V
    k : Nat
    x : D
    ⊢ Eq (Norm.norm (f x)) (Norm.norm (iteratedFDeriv Real 0 (⇑f) x))
  -/
  simp
  /-
    🎉 no goals
  -/


lemma integrable (f : 𝓢(D, V)) : Integrable f μ :=
  (f.integrable_pow_mul μ 0).mono f.continuous.aestronglyMeasurable
                                      /-
                                        D : Type u_3
                                        V : Type u_7
                                        inst✝⁶ : NormedAddCommGroup D
                                        inst✝⁵ : NormedSpace Real D
                                        inst✝⁴ : NormedAddCommGroup V
                                        inst✝³ : NormedSpace Real V
                                        inst✝² : MeasurableSpace D
                                        μ : MeasureTheory.Measure D
                                        hμ : μ.HasTemperateGrowth
                                        inst✝¹ : BorelSpace D
                                        inst✝ : SecondCountableTopology D
                                        f : SchwartzMap D V
                                        x✝ : D
                                        ⊢ LE.le (Norm.norm (f x✝)) (Norm.norm (HMul.hMul (HPow.hPow (Norm.norm x✝) 0)  …
                                      -/
    (Eventually.of_forall (fun _ ↦ by simp))
                                      /-
                                        🎉 no goals
                                      -/


variable (𝕜 μ) in
/-- The integral as a continuous linear map from Schwartz space to the codomain. -/
def integralCLM : 𝓢(D, V) →L[𝕜] V := by
  refine mkCLMtoNormedSpace (∫ x, · x ∂μ)
    (fun f g ↦ integral_add f.integrable g.integrable) (integral_smul · ·) ?_
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : RCLike 𝕜
    inst✝⁷ : NormedAddCommGroup D
    inst✝⁶ : NormedSpace Real D
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : NormedSpace Real V
    inst✝³ : NormedSpace 𝕜 V
    inst✝² : MeasurableSpace D
    μ : MeasureTheory.Measure D
    hμ : μ.HasTemperateGrowth
    inst✝¹ : BorelSpace D
    inst✝ : SecondCountableTopology D
    ⊢ Exists fun s => Exists fun C => And (LE.le 0 C) (∀ (f : SchwartzMap D V), LE …
  -/
  rcases hμ.exists_integrable with ⟨n, h⟩
  /-
    case intro
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : RCLike 𝕜
    inst✝⁷ : NormedAddCommGroup D
    inst✝⁶ : NormedSpace Real D
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : NormedSpace Real V
    inst✝³ : NormedSpace 𝕜 V
    inst✝² : MeasurableSpace D
    μ : MeasureTheory.Measure D
    hμ : μ.HasTemperateGrowth
    inst✝¹ : BorelSpace D
    inst✝ : SecondCountableTopology D
    n : Nat
    h : MeasureTheory.Integrable (fun x => HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) ( …
    ⊢ Exists fun s => Exists fun C => And (LE.le 0 C) (∀ (f : SchwartzMap D V), LE …
  -/
  let m := (n, 0)
  /-
    case intro
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : RCLike 𝕜
    inst✝⁷ : NormedAddCommGroup D
    inst✝⁶ : NormedSpace Real D
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : NormedSpace Real V
    inst✝³ : NormedSpace 𝕜 V
    inst✝² : MeasurableSpace D
    μ : MeasureTheory.Measure D
    hμ : μ.HasTemperateGrowth
    inst✝¹ : BorelSpace D
    inst✝ : SecondCountableTopology D
    n : Nat
    h : MeasureTheory.Integrable (fun x => HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) ( …
    m : Prod Nat Nat := { fst := n, snd := 0 }
    ⊢ Exists fun s => Exists fun C => And (LE.le 0 C) (∀ (f : SchwartzMap D V), LE …
  -/
  use Finset.Iic m, 2 ^ n * ∫ x : D, (1 + ‖x‖) ^ (- (n : ℝ)) ∂μ
  /-
    case h
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : RCLike 𝕜
    inst✝⁷ : NormedAddCommGroup D
    inst✝⁶ : NormedSpace Real D
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : NormedSpace Real V
    inst✝³ : NormedSpace 𝕜 V
    inst✝² : MeasurableSpace D
    μ : MeasureTheory.Measure D
    hμ : μ.HasTemperateGrowth
    inst✝¹ : BorelSpace D
    inst✝ : SecondCountableTopology D
    n : Nat
    h : MeasureTheory.Integrable (fun x => HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) ( …
    m : Prod Nat Nat := { fst := n, snd := 0 }
    ⊢ And (LE.le 0 (HMul.hMul (HPow.hPow 2 n) (MeasureTheory.integral μ fun x => H …
  -/
  refine ⟨by positivity, fun f ↦ (norm_integral_le_integral_norm f).trans ?_⟩
  have h' : ∀ x, ‖f x‖ ≤ (1 + ‖x‖) ^ (-(n : ℝ)) *
      (2 ^ n * ((Finset.Iic m).sup (fun m' => SchwartzMap.seminorm 𝕜 m'.1 m'.2) f)) := by
    intro x
    rw [rpow_neg (by positivity), ← div_eq_inv_mul, le_div_iff₀' (by positivity), rpow_natCast]
    simpa using one_add_le_sup_seminorm_apply (m := m) (k := n) (n := 0) le_rfl le_rfl f x
  /-
    case h
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : RCLike 𝕜
    inst✝⁷ : NormedAddCommGroup D
    inst✝⁶ : NormedSpace Real D
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : NormedSpace Real V
    inst✝³ : NormedSpace 𝕜 V
    inst✝² : MeasurableSpace D
    μ : MeasureTheory.Measure D
    hμ : μ.HasTemperateGrowth
    inst✝¹ : BorelSpace D
    inst✝ : SecondCountableTopology D
    n : Nat
    h : MeasureTheory.Integrable (fun x => HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) ( …
    m : Prod Nat Nat := { fst := n, snd := 0 }
    f : SchwartzMap D V
    h' : ∀ (x : D), LE.le (Norm.norm (f x)) (HMul.hMul (HPow.hPow (HAdd.hAdd 1 (No …
    ⊢ LE.le (MeasureTheory.integral μ fun a => Norm.norm (f a)) (HMul.hMul (HMul.h …
  -/
  apply (integral_mono (by simpa using f.integrable_pow_mul μ 0) _ h').trans
    /-
      case h
      𝕜 : Type u_1
      𝕜' : Type u_2
      D : Type u_3
      E : Type u_4
      F : Type u_5
      G : Type u_6
      V : Type u_7
      inst✝¹² : NormedAddCommGroup E
      inst✝¹¹ : NormedSpace Real E
      inst✝¹⁰ : NormedAddCommGroup F
      inst✝⁹ : NormedSpace Real F
      inst✝⁸ : RCLike 𝕜
      inst✝⁷ : NormedAddCommGroup D
      inst✝⁶ : NormedSpace Real D
      inst✝⁵ : NormedAddCommGroup V
      inst✝⁴ : NormedSpace Real V
      inst✝³ : NormedSpace 𝕜 V
      inst✝² : MeasurableSpace D
      μ : MeasureTheory.Measure D
      hμ : μ.HasTemperateGrowth
      inst✝¹ : BorelSpace D
      inst✝ : SecondCountableTopology D
      n : Nat
      h : MeasureTheory.Integrable (fun x => HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) ( …
      m : Prod Nat Nat := { fst := n, snd := 0 }
      f : SchwartzMap D V
      h' : ∀ (x : D), LE.le (Norm.norm (f x)) (HMul.hMul (HPow.hPow (HAdd.hAdd 1 (No …
      ⊢ LE.le (MeasureTheory.integral μ fun a => HMul.hMul (HPow.hPow (HAdd.hAdd 1 ( …
    -/
  · rw [integral_mul_right, ← mul_assoc, mul_comm (2 ^ n)]
    /-
      case h
      𝕜 : Type u_1
      𝕜' : Type u_2
      D : Type u_3
      E : Type u_4
      F : Type u_5
      G : Type u_6
      V : Type u_7
      inst✝¹² : NormedAddCommGroup E
      inst✝¹¹ : NormedSpace Real E
      inst✝¹⁰ : NormedAddCommGroup F
      inst✝⁹ : NormedSpace Real F
      inst✝⁸ : RCLike 𝕜
      inst✝⁷ : NormedAddCommGroup D
      inst✝⁶ : NormedSpace Real D
      inst✝⁵ : NormedAddCommGroup V
      inst✝⁴ : NormedSpace Real V
      inst✝³ : NormedSpace 𝕜 V
      inst✝² : MeasurableSpace D
      μ : MeasureTheory.Measure D
      hμ : μ.HasTemperateGrowth
      inst✝¹ : BorelSpace D
      inst✝ : SecondCountableTopology D
      n : Nat
      h : MeasureTheory.Integrable (fun x => HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) ( …
      m : Prod Nat Nat := { fst := n, snd := 0 }
      f : SchwartzMap D V
      h' : ∀ (x : D), LE.le (Norm.norm (f x)) (HMul.hMul (HPow.hPow (HAdd.hAdd 1 (No …
      ⊢ LE.le (HMul.hMul (HMul.hMul (MeasureTheory.integral μ fun a => HPow.hPow (HA …
    -/
    rfl
    /-
      🎉 no goals
    -/
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    D : Type u_3
    E : Type u_4
    F : Type u_5
    G : Type u_6
    V : Type u_7
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : RCLike 𝕜
    inst✝⁷ : NormedAddCommGroup D
    inst✝⁶ : NormedSpace Real D
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : NormedSpace Real V
    inst✝³ : NormedSpace 𝕜 V
    inst✝² : MeasurableSpace D
    μ : MeasureTheory.Measure D
    hμ : μ.HasTemperateGrowth
    inst✝¹ : BorelSpace D
    inst✝ : SecondCountableTopology D
    n : Nat
    h : MeasureTheory.Integrable (fun x => HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) ( …
    m : Prod Nat Nat := { fst := n, snd := 0 }
    f : SchwartzMap D V
    h' : ∀ (x : D), LE.le (Norm.norm (f x)) (HMul.hMul (HPow.hPow (HAdd.hAdd 1 (No …
    ⊢ MeasureTheory.Integrable (fun i => HMul.hMul (HPow.hPow (HAdd.hAdd 1 (Norm.n …
  -/
  apply h.mul_const
  /-
    🎉 no goals
  -/


variable (𝕜) in
@[simp]
                                                                              /-
                                                                                𝕜 : Type u_1
                                                                                D : Type u_3
                                                                                V : Type u_7
                                                                                inst✝⁸ : RCLike 𝕜
                                                                                inst✝⁷ : NormedAddCommGroup D
                                                                                inst✝⁶ : NormedSpace Real D
                                                                                inst✝⁵ : NormedAddCommGroup V
                                                                                inst✝⁴ : NormedSpace Real V
                                                                                inst✝³ : NormedSpace 𝕜 V
                                                                                inst✝² : MeasurableSpace D
                                                                                μ : MeasureTheory.Measure D
                                                                                hμ : μ.HasTemperateGrowth
                                                                                inst✝¹ : BorelSpace D
                                                                                inst✝ : SecondCountableTopology D
                                                                                f : SchwartzMap D V
                                                                                ⊢ Eq ((SchwartzMap.integralCLM 𝕜 μ) f) (MeasureTheory.integral μ fun x => f x)
                                                                              -/
lemma integralCLM_apply (f : 𝓢(D, V)) : integralCLM 𝕜 μ f = ∫ x, f x ∂μ := by rfl
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


instance instBoundedContinuousMapClass : BoundedContinuousMapClass 𝓢(E, F) E F where
  __ := instContinuousMapClass
  map_bounded := fun f ↦ ⟨2 * (SchwartzMap.seminorm ℝ 0 0) f,
    (BoundedContinuousFunction.dist_le_two_norm' (norm_le_seminorm ℝ f))⟩


/-- Schwartz functions as bounded continuous functions -/
def toBoundedContinuousFunction (f : 𝓢(E, F)) : E →ᵇ F :=
  BoundedContinuousFunction.ofNormedAddCommGroup f (SchwartzMap.continuous f)
    (SchwartzMap.seminorm ℝ 0 0 f) (norm_le_seminorm ℝ f)


@[simp]
theorem toBoundedContinuousFunction_apply (f : 𝓢(E, F)) (x : E) :
    f.toBoundedContinuousFunction x = f x :=
  rfl


/-- Schwartz functions as continuous functions -/
def toContinuousMap (f : 𝓢(E, F)) : C(E, F) :=
  f.toBoundedContinuousFunction.toContinuousMap


/-- The inclusion map from Schwartz functions to bounded continuous functions as a continuous linear
map. -/
def toBoundedContinuousFunctionCLM : 𝓢(E, F) →L[𝕜] E →ᵇ F :=
                                                     /-
                                                       𝕜 : Type u_1
                                                       𝕜' : Type u_2
                                                       D : Type u_3
                                                       E : Type u_4
                                                       F : Type u_5
                                                       G : Type u_6
                                                       V : Type u_7
                                                       inst✝⁶ : NormedAddCommGroup E
                                                       inst✝⁵ : NormedSpace Real E
                                                       inst✝⁴ : NormedAddCommGroup F
                                                       inst✝³ : NormedSpace Real F
                                                       inst✝² : RCLike 𝕜
                                                       inst✝¹ : NormedSpace 𝕜 F
                                                       inst✝ : SMulCommClass Real 𝕜 F
                                                       ⊢ ∀ (f g : SchwartzMap E F), Eq (HAdd.hAdd f g).toBoundedContinuousFunction (H …
                                                     -/
  mkCLMtoNormedSpace toBoundedContinuousFunction (by intro f g; ext; exact add_apply)
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
        /-
          𝕜 : Type u_1
          𝕜' : Type u_2
          D : Type u_3
          E : Type u_4
          F : Type u_5
          G : Type u_6
          V : Type u_7
          inst✝⁶ : NormedAddCommGroup E
          inst✝⁵ : NormedSpace Real E
          inst✝⁴ : NormedAddCommGroup F
          inst✝³ : NormedSpace Real F
          inst✝² : RCLike 𝕜
          inst✝¹ : NormedSpace 𝕜 F
          inst✝ : SMulCommClass Real 𝕜 F
          ⊢ ∀ (a : 𝕜) (f : SchwartzMap E F), Eq (HSMul.hSMul a f).toBoundedContinuousFun …
        -/
    (by intro a f; ext; exact smul_apply)
                        /-
                          🎉 no goals
                        -/
    (⟨{0}, 1, zero_le_one, by
      /-
        𝕜 : Type u_1
        𝕜' : Type u_2
        D : Type u_3
        E : Type u_4
        F : Type u_5
        G : Type u_6
        V : Type u_7
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedSpace Real E
        inst✝⁴ : NormedAddCommGroup F
        inst✝³ : NormedSpace Real F
        inst✝² : RCLike 𝕜
        inst✝¹ : NormedSpace 𝕜 F
        inst✝ : SMulCommClass Real 𝕜 F
        ⊢ ∀ (f : SchwartzMap E F), LE.le (Norm.norm f.toBoundedContinuousFunction) (HM …
      -/
      simpa [BoundedContinuousFunction.norm_le (apply_nonneg _ _)] using norm_le_seminorm 𝕜 ⟩)
      /-
        🎉 no goals
      -/


@[simp]
theorem toBoundedContinuousFunctionCLM_apply (f : 𝓢(E, F)) (x : E) :
    toBoundedContinuousFunctionCLM 𝕜 E F f x = f x :=
  rfl


/-- The Dirac delta distribution -/
def delta (x : E) : 𝓢(E, F) →L[𝕜] F :=
  (BoundedContinuousFunction.evalCLM 𝕜 x).comp (toBoundedContinuousFunctionCLM 𝕜 E F)


@[simp]
theorem delta_apply (x₀ : E) (f : 𝓢(E, F)) : delta 𝕜 F x₀ f = f x₀ :=
  rfl


/-- Integrating against the Dirac measure is equal to the delta distribution. -/
@[simp]
                                                                                         /-
                                                                                           𝕜 : Type u_1
                                                                                           E : Type u_4
                                                                                           F : Type u_5
                                                                                           inst✝¹⁰ : NormedAddCommGroup E
                                                                                           inst✝⁹ : NormedSpace Real E
                                                                                           inst✝⁸ : NormedAddCommGroup F
                                                                                           inst✝⁷ : NormedSpace Real F
                                                                                           inst✝⁶ : RCLike 𝕜
                                                                                           inst✝⁵ : NormedSpace 𝕜 F
                                                                                           inst✝⁴ : SMulCommClass Real 𝕜 F
                                                                                           inst✝³ : MeasurableSpace E
                                                                                           inst✝² : BorelSpace E
                                                                                           inst✝¹ : SecondCountableTopology E
                                                                                           inst✝ : CompleteSpace F
                                                                                           x : E
                                                                                           ⊢ Eq (SchwartzMap.integralCLM 𝕜 (MeasureTheory.Measure.dirac x)) (SchwartzMap. …
                                                                                         -/
theorem integralCLM_dirac_eq_delta (x : E) : integralCLM 𝕜 (dirac x) = delta 𝕜 F x := by aesop
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/


instance instZeroAtInftyContinuousMapClass : ZeroAtInftyContinuousMapClass 𝓢(E, F) E F where
  __ := instContinuousMapClass
  zero_at_infty := by
    /-
      𝕜 : Type u_1
      𝕜' : Type u_2
      D : Type u_3
      E : Type u_4
      F : Type u_5
      G : Type u_6
      V : Type u_7
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      inst✝ : ProperSpace E
      ⊢ ∀ (f : SchwartzMap E F), Filter.Tendsto (⇑f) (Filter.cocompact E) (nhds 0)
    -/
    intro f
    /-
      𝕜 : Type u_1
      𝕜' : Type u_2
      D : Type u_3
      E : Type u_4
      F : Type u_5
      G : Type u_6
      V : Type u_7
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      inst✝ : ProperSpace E
      f : SchwartzMap E F
      ⊢ Filter.Tendsto (⇑f) (Filter.cocompact E) (nhds 0)
    -/
    apply zero_at_infty_of_norm_le
    /-
      case h
      𝕜 : Type u_1
      𝕜' : Type u_2
      D : Type u_3
      E : Type u_4
      F : Type u_5
      G : Type u_6
      V : Type u_7
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      inst✝ : ProperSpace E
      f : SchwartzMap E F
      ⊢ ∀ (ε : Real), LT.lt 0 ε → Exists fun r => ∀ (x : E), LT.lt r (Norm.norm x) → …
    -/
    intro ε hε
    /-
      case h
      𝕜 : Type u_1
      𝕜' : Type u_2
      D : Type u_3
      E : Type u_4
      F : Type u_5
      G : Type u_6
      V : Type u_7
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      inst✝ : ProperSpace E
      f : SchwartzMap E F
      ε : Real
      hε : LT.lt 0 ε
      ⊢ Exists fun r => ∀ (x : E), LT.lt r (Norm.norm x) → LT.lt (Norm.norm (f x)) ε
    -/
    use (SchwartzMap.seminorm ℝ 1 0) f / ε
    /-
      case h
      𝕜 : Type u_1
      𝕜' : Type u_2
      D : Type u_3
      E : Type u_4
      F : Type u_5
      G : Type u_6
      V : Type u_7
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      inst✝ : ProperSpace E
      f : SchwartzMap E F
      ε : Real
      hε : LT.lt 0 ε
      ⊢ ∀ (x : E), LT.lt (HDiv.hDiv ((SchwartzMap.seminorm Real 1 0) f) ε) (Norm.nor …
    -/
    intro x hx
    /-
      case h
      𝕜 : Type u_1
      𝕜' : Type u_2
      D : Type u_3
      E : Type u_4
      F : Type u_5
      G : Type u_6
      V : Type u_7
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      inst✝ : ProperSpace E
      f : SchwartzMap E F
      ε : Real
      hε : LT.lt 0 ε
      x : E
      hx : LT.lt (HDiv.hDiv ((SchwartzMap.seminorm Real 1 0) f) ε) (Norm.norm x)
      ⊢ LT.lt (Norm.norm (f x)) ε
    -/
    rw [div_lt_iff₀ hε] at hx
    have hxpos : 0 < ‖x‖ := by
      rw [norm_pos_iff]
      intro hxzero
      simp only [hxzero, norm_zero, zero_mul, ← not_le] at hx
      exact hx (apply_nonneg (SchwartzMap.seminorm ℝ 1 0) f)
    /-
      case h
      𝕜 : Type u_1
      𝕜' : Type u_2
      D : Type u_3
      E : Type u_4
      F : Type u_5
      G : Type u_6
      V : Type u_7
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      inst✝ : ProperSpace E
      f : SchwartzMap E F
      ε : Real
      hε : LT.lt 0 ε
      x : E
      hx : LT.lt ((SchwartzMap.seminorm Real 1 0) f) (HMul.hMul (Norm.norm x) ε)
      hxpos : LT.lt 0 (Norm.norm x)
      ⊢ LT.lt (Norm.norm (f x)) ε
    -/
    have := norm_pow_mul_le_seminorm ℝ f 1 x
    /-
      case h
      𝕜 : Type u_1
      𝕜' : Type u_2
      D : Type u_3
      E : Type u_4
      F : Type u_5
      G : Type u_6
      V : Type u_7
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      inst✝ : ProperSpace E
      f : SchwartzMap E F
      ε : Real
      hε : LT.lt 0 ε
      x : E
      hx : LT.lt ((SchwartzMap.seminorm Real 1 0) f) (HMul.hMul (Norm.norm x) ε)
      hxpos : LT.lt 0 (Norm.norm x)
      this : LE.le (HMul.hMul (HPow.hPow (Norm.norm x) 1) (Norm.norm (f x))) ((Schwa …
      ⊢ LT.lt (Norm.norm (f x)) ε
    -/
    rw [pow_one, ← le_div_iff₀' hxpos] at this
    /-
      case h
      𝕜 : Type u_1
      𝕜' : Type u_2
      D : Type u_3
      E : Type u_4
      F : Type u_5
      G : Type u_6
      V : Type u_7
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      inst✝ : ProperSpace E
      f : SchwartzMap E F
      ε : Real
      hε : LT.lt 0 ε
      x : E
      hx : LT.lt ((SchwartzMap.seminorm Real 1 0) f) (HMul.hMul (Norm.norm x) ε)
      hxpos : LT.lt 0 (Norm.norm x)
      this : LE.le (Norm.norm (f x)) (HDiv.hDiv ((SchwartzMap.seminorm Real 1 0) f)  …
      ⊢ LT.lt (Norm.norm (f x)) ε
    -/
    apply lt_of_le_of_lt this
    /-
      case h
      𝕜 : Type u_1
      𝕜' : Type u_2
      D : Type u_3
      E : Type u_4
      F : Type u_5
      G : Type u_6
      V : Type u_7
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      inst✝ : ProperSpace E
      f : SchwartzMap E F
      ε : Real
      hε : LT.lt 0 ε
      x : E
      hx : LT.lt ((SchwartzMap.seminorm Real 1 0) f) (HMul.hMul (Norm.norm x) ε)
      hxpos : LT.lt 0 (Norm.norm x)
      this : LE.le (Norm.norm (f x)) (HDiv.hDiv ((SchwartzMap.seminorm Real 1 0) f)  …
      ⊢ LT.lt (HDiv.hDiv ((SchwartzMap.seminorm Real 1 0) f) (Norm.norm x)) ε
    -/
    rwa [div_lt_iff₀' hxpos]
    /-
      🎉 no goals
    -/


/-- Schwartz functions as continuous functions vanishing at infinity. -/
def toZeroAtInfty (f : 𝓢(E, F)) : C₀(E, F) where
  toFun := f
  zero_at_infty' := zero_at_infty f


@[simp] theorem toZeroAtInfty_apply (f : 𝓢(E, F)) (x : E) : f.toZeroAtInfty x = f x :=
  rfl


@[simp] theorem toZeroAtInfty_toBCF (f : 𝓢(E, F)) :
    f.toZeroAtInfty.toBCF = f.toBoundedContinuousFunction :=
  rfl


/-- The inclusion map from Schwartz functions to continuous functions vanishing at infinity as a
continuous linear map. -/
def toZeroAtInftyCLM : 𝓢(E, F) →L[𝕜] C₀(E, F) :=
                                       /-
                                         𝕜 : Type u_1
                                         𝕜' : Type u_2
                                         D : Type u_3
                                         E : Type u_4
                                         F : Type u_5
                                         G : Type u_6
                                         V : Type u_7
                                         inst✝⁷ : NormedAddCommGroup E
                                         inst✝⁶ : NormedSpace Real E
                                         inst✝⁵ : NormedAddCommGroup F
                                         inst✝⁴ : NormedSpace Real F
                                         inst✝³ : ProperSpace E
                                         inst✝² : RCLike 𝕜
                                         inst✝¹ : NormedSpace 𝕜 F
                                         inst✝ : SMulCommClass Real 𝕜 F
                                         ⊢ ∀ (f g : SchwartzMap E F), Eq (HAdd.hAdd f g).toZeroAtInfty (HAdd.hAdd f.toZ …
                                       -/
  mkCLMtoNormedSpace toZeroAtInfty (by intro f g; ext; exact add_apply)
                                                       /-
                                                         🎉 no goals
                                                       -/
        /-
          𝕜 : Type u_1
          𝕜' : Type u_2
          D : Type u_3
          E : Type u_4
          F : Type u_5
          G : Type u_6
          V : Type u_7
          inst✝⁷ : NormedAddCommGroup E
          inst✝⁶ : NormedSpace Real E
          inst✝⁵ : NormedAddCommGroup F
          inst✝⁴ : NormedSpace Real F
          inst✝³ : ProperSpace E
          inst✝² : RCLike 𝕜
          inst✝¹ : NormedSpace 𝕜 F
          inst✝ : SMulCommClass Real 𝕜 F
          ⊢ ∀ (a : 𝕜) (f : SchwartzMap E F), Eq (HSMul.hSMul a f).toZeroAtInfty (HSMul.h …
        -/
    (by intro a f; ext; exact smul_apply)
                        /-
                          🎉 no goals
                        -/
    (⟨{0}, 1, zero_le_one, by simpa [← ZeroAtInftyContinuousMap.norm_toBCF_eq_norm,
      BoundedContinuousFunction.norm_le (apply_nonneg _ _)] using norm_le_seminorm 𝕜 ⟩)


@[simp] theorem toZeroAtInftyCLM_apply (f : 𝓢(E, F)) (x : E) : toZeroAtInftyCLM 𝕜 E F f x = f x :=
  rfl


