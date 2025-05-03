/-- Two absolute values `f, g` on `R` with values in `ℝ` are *equivalent* if there exists
a positive real constant `c` such that for all `x : R`, `(f x)^c = g x`. -/
def Equiv (f g : AbsoluteValue R ℝ) : Prop :=
  ∃ c : ℝ, 0 < c ∧ (f · ^ c) = g


/-- Equivalence of absolute values is reflexive. -/
lemma equiv_refl (f : AbsoluteValue R ℝ) : Equiv f f :=
  ⟨1, Real.zero_lt_one, funext fun x ↦ Real.rpow_one (f x)⟩


/-- Equivalence of absolute values is symmetric. -/
lemma equiv_symm {f g : AbsoluteValue R ℝ} (hfg : Equiv f g) : Equiv g f := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f g : AbsoluteValue R Real
    hfg : f.Equiv g
    ⊢ g.Equiv f
  -/
  rcases hfg with ⟨c, hcpos, h⟩
  /-
    case intro.intro
    R : Type u_1
    inst✝ : Semiring R
    f g : AbsoluteValue R Real
    c : Real
    hcpos : LT.lt 0 c
    h : Eq (fun x => HPow.hPow (f x) c) ⇑g
    ⊢ g.Equiv f
  -/
  refine ⟨1 / c, one_div_pos.mpr hcpos, ?_⟩
  /-
    case intro.intro
    R : Type u_1
    inst✝ : Semiring R
    f g : AbsoluteValue R Real
    c : Real
    hcpos : LT.lt 0 c
    h : Eq (fun x => HPow.hPow (f x) c) ⇑g
    ⊢ Eq (fun x => HPow.hPow (g x) (HDiv.hDiv 1 c)) ⇑f
  -/
  simp [← h, Real.rpow_rpow_inv (apply_nonneg f _) (ne_of_lt hcpos).symm]
  /-
    🎉 no goals
  -/


/-- Equivalence of absolute values is transitive. -/
lemma equiv_trans {f g k : AbsoluteValue R ℝ} (hfg : Equiv f g) (hgk : Equiv g k) :
    Equiv f k := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f g k : AbsoluteValue R Real
    hfg : f.Equiv g
    hgk : g.Equiv k
    ⊢ f.Equiv k
  -/
  rcases hfg with ⟨c, hcPos, hfg⟩
  /-
    case intro.intro
    R : Type u_1
    inst✝ : Semiring R
    f g k : AbsoluteValue R Real
    hgk : g.Equiv k
    c : Real
    hcPos : LT.lt 0 c
    hfg : Eq (fun x => HPow.hPow (f x) c) ⇑g
    ⊢ f.Equiv k
  -/
  rcases hgk with ⟨d, hdPos, hgk⟩
  /-
    case intro.intro.intro.intro
    R : Type u_1
    inst✝ : Semiring R
    f g k : AbsoluteValue R Real
    c : Real
    hcPos : LT.lt 0 c
    hfg : Eq (fun x => HPow.hPow (f x) c) ⇑g
    d : Real
    hdPos : LT.lt 0 d
    hgk : Eq (fun x => HPow.hPow (g x) d) ⇑k
    ⊢ f.Equiv k
  -/
  refine ⟨c * d, mul_pos hcPos hdPos, ?_⟩
  /-
    case intro.intro.intro.intro
    R : Type u_1
    inst✝ : Semiring R
    f g k : AbsoluteValue R Real
    c : Real
    hcPos : LT.lt 0 c
    hfg : Eq (fun x => HPow.hPow (f x) c) ⇑g
    d : Real
    hdPos : LT.lt 0 d
    hgk : Eq (fun x => HPow.hPow (g x) d) ⇑k
    ⊢ Eq (fun x => HPow.hPow (f x) (HMul.hMul c d)) ⇑k
  -/
  simp [← hgk, ← hfg, Real.rpow_mul (apply_nonneg f _)]
  /-
    🎉 no goals
  -/


/-- An absolute value is equivalent to the trivial iff it is trivial itself. -/
@[simp]
lemma eq_trivial_of_equiv_trivial [DecidablePred fun x : R ↦ x = 0] [NoZeroDivisors R]
    (f : AbsoluteValue R ℝ) :
    f.Equiv .trivial ↔ f = .trivial := by
  /-
    R : Type u_1
    inst✝² : Semiring R
    inst✝¹ : DecidablePred fun x => Eq x 0
    inst✝ : NoZeroDivisors R
    f : AbsoluteValue R Real
    ⊢ Iff (f.Equiv AbsoluteValue.trivial) (Eq f AbsoluteValue.trivial)
  -/
  refine ⟨fun ⟨c, hc₀, hc⟩ ↦ ext fun x ↦ ?_, fun H ↦ H ▸ equiv_refl f⟩
  /-
    R : Type u_1
    inst✝² : Semiring R
    inst✝¹ : DecidablePred fun x => Eq x 0
    inst✝ : NoZeroDivisors R
    f : AbsoluteValue R Real
    x✝ : f.Equiv AbsoluteValue.trivial
    c : Real
    hc₀ : LT.lt 0 c
    hc : Eq (fun x => HPow.hPow (f x) c) ⇑AbsoluteValue.trivial
    x : R
    ⊢ Eq (f x) (AbsoluteValue.trivial x)
  -/
  apply_fun (· x) at hc
  /-
    R : Type u_1
    inst✝² : Semiring R
    inst✝¹ : DecidablePred fun x => Eq x 0
    inst✝ : NoZeroDivisors R
    f : AbsoluteValue R Real
    x✝ : f.Equiv AbsoluteValue.trivial
    c : Real
    hc₀ : LT.lt 0 c
    x : R
    hc : Eq ((fun x => HPow.hPow (f x) c) x) (AbsoluteValue.trivial x)
    ⊢ Eq (f x) (AbsoluteValue.trivial x)
  -/
  rcases eq_or_ne x 0 with rfl | hx
    /-
      case inl
      R : Type u_1
      inst✝² : Semiring R
      inst✝¹ : DecidablePred fun x => Eq x 0
      inst✝ : NoZeroDivisors R
      f : AbsoluteValue R Real
      x✝ : f.Equiv AbsoluteValue.trivial
      c : Real
      hc₀ : LT.lt 0 c
      hc : Eq ((fun x => HPow.hPow (f x) c) 0) (AbsoluteValue.trivial 0)
      ⊢ Eq (f 0) (AbsoluteValue.trivial 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u_1
      inst✝² : Semiring R
      inst✝¹ : DecidablePred fun x => Eq x 0
      inst✝ : NoZeroDivisors R
      f : AbsoluteValue R Real
      x✝ : f.Equiv AbsoluteValue.trivial
      c : Real
      hc₀ : LT.lt 0 c
      x : R
      hc : Eq ((fun x => HPow.hPow (f x) c) x) (AbsoluteValue.trivial x)
      hx : Ne x 0
      ⊢ Eq (f x) (AbsoluteValue.trivial x)
    -/
  · simp only [ne_eq, hx, not_false_eq_true, trivial_apply] at hc ⊢
    /-
      case inr
      R : Type u_1
      inst✝² : Semiring R
      inst✝¹ : DecidablePred fun x => Eq x 0
      inst✝ : NoZeroDivisors R
      f : AbsoluteValue R Real
      x✝ : f.Equiv AbsoluteValue.trivial
      c : Real
      hc₀ : LT.lt 0 c
      x : R
      hx : Ne x 0
      hc : Eq (HPow.hPow (f x) c) 1
      ⊢ Eq (f x) 1
    -/
    exact (Real.rpow_left_inj (f.nonneg x) zero_le_one hc₀.ne').mp <| (Real.one_rpow c).symm ▸ hc
    /-
      🎉 no goals
    -/


instance : Setoid (AbsoluteValue R ℝ) where
  r := Equiv
  iseqv := {
    refl := equiv_refl
    symm := equiv_symm
    trans := equiv_trans
  }


