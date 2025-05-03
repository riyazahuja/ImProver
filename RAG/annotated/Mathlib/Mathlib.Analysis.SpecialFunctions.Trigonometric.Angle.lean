/-- The type of angles -/
def Angle : Type :=
  AddCircle (2 * π)


instance : NormedAddCommGroup Angle :=
  inferInstanceAs (NormedAddCommGroup (AddCircle (2 * π)))

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/10754): added due to missing instances due to no deriving

instance : Inhabited Angle :=
  inferInstanceAs (Inhabited (AddCircle (2 * π)))

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/10754): added due to missing instances due to no deriving
-- also, without this, a plain `QuotientAddGroup.mk`
-- causes coerced terms to be of type `ℝ ⧸ AddSubgroup.zmultiples (2 * π)`

/-- The canonical map from `ℝ` to the quotient `Angle`. -/
@[coe]
protected def coe (r : ℝ) : Angle := QuotientAddGroup.mk r


instance : Coe ℝ Angle := ⟨Angle.coe⟩


instance : CircularOrder Real.Angle :=
                                             /-
                                               ⊢ LT.lt 0 (HMul.hMul 2 Real.pi)
                                             -/
  QuotientAddGroup.circularOrder (hp' := ⟨by norm_num [pi_pos]⟩)
                                             /-
                                               🎉 no goals
                                             -/



@[continuity]
theorem continuous_coe : Continuous ((↑) : ℝ → Angle) :=
  continuous_quotient_mk'


/-- Coercion `ℝ → Angle` as an additive homomorphism. -/
def coeHom : ℝ →+ Angle :=
  QuotientAddGroup.mk' _


@[simp]
theorem coe_coeHom : (coeHom : ℝ → Angle) = ((↑) : ℝ → Angle) :=
  rfl


/-- An induction principle to deduce results for `Angle` from those for `ℝ`, used with
`induction θ using Real.Angle.induction_on`. -/
@[elab_as_elim]
protected theorem induction_on {p : Angle → Prop} (θ : Angle) (h : ∀ x : ℝ, p x) : p θ :=
  Quotient.inductionOn' θ h


@[simp]
theorem coe_zero : ↑(0 : ℝ) = (0 : Angle) :=
  rfl


@[simp]
theorem coe_add (x y : ℝ) : ↑(x + y : ℝ) = (↑x + ↑y : Angle) :=
  rfl


@[simp]
theorem coe_neg (x : ℝ) : ↑(-x : ℝ) = -(↑x : Angle) :=
  rfl


@[simp]
theorem coe_sub (x y : ℝ) : ↑(x - y : ℝ) = (↑x - ↑y : Angle) :=
  rfl


theorem coe_nsmul (n : ℕ) (x : ℝ) : ↑(n • x : ℝ) = n • (↑x : Angle) :=
  rfl


theorem coe_zsmul (z : ℤ) (x : ℝ) : ↑(z • x : ℝ) = z • (↑x : Angle) :=
  rfl


@[simp, norm_cast]
theorem natCast_mul_eq_nsmul (x : ℝ) (n : ℕ) : ↑((n : ℝ) * x) = n • (↑x : Angle) := by
  /-
    x : Real
    n : Nat
    ⊢ Eq (↑(HMul.hMul (↑n) x)) (HSMul.hSMul n ↑x)
  -/
  simpa only [nsmul_eq_mul] using coeHom.map_nsmul x n
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem intCast_mul_eq_zsmul (x : ℝ) (n : ℤ) : ↑((n : ℝ) * x : ℝ) = n • (↑x : Angle) := by
  /-
    x : Real
    n : Int
    ⊢ Eq (↑(HMul.hMul (↑n) x)) (HSMul.hSMul n ↑x)
  -/
  simpa only [zsmul_eq_mul] using coeHom.map_zsmul x n
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-05-25")] alias coe_nat_mul_eq_nsmul := natCast_mul_eq_nsmul

@[deprecated (since := "2024-05-25")] alias coe_int_mul_eq_zsmul := intCast_mul_eq_zsmul


theorem angle_eq_iff_two_pi_dvd_sub {ψ θ : ℝ} : (θ : Angle) = ψ ↔ ∃ k : ℤ, θ - ψ = 2 * π * k := by
  simp only [QuotientAddGroup.eq, AddSubgroup.zmultiples_eq_closure,
    AddSubgroup.mem_closure_singleton, zsmul_eq_mul', (sub_eq_neg_add _ _).symm, eq_comm]
  -- Porting note: added `rw`, `simp [Angle.coe, QuotientAddGroup.eq]` doesn't fire otherwise
  /-
    ψ θ : Real
    ⊢ Iff (Eq ↑ψ ↑θ) (Exists fun k => Eq (HSub.hSub θ ψ) (HMul.hMul (HMul.hMul 2 R …
  -/
  rw [Angle.coe, Angle.coe, QuotientAddGroup.eq]
  simp only [AddSubgroup.zmultiples_eq_closure,
    AddSubgroup.mem_closure_singleton, zsmul_eq_mul', (sub_eq_neg_add _ _).symm, eq_comm]


@[simp]
theorem coe_two_pi : ↑(2 * π : ℝ) = (0 : Angle) :=
                                       /-
                                         ⊢ Eq (HSub.hSub (HMul.hMul 2 Real.pi) 0) (HMul.hMul (HMul.hMul 2 Real.pi) ↑1)
                                       -/
  angle_eq_iff_two_pi_dvd_sub.2 ⟨1, by rw [sub_zero, Int.cast_one, mul_one]⟩
                                       /-
                                         🎉 no goals
                                       -/


@[simp]
theorem neg_coe_pi : -(π : Angle) = π := by
  /-
    ⊢ Eq (Neg.neg ↑Real.pi) ↑Real.pi
  -/
  rw [← coe_neg, angle_eq_iff_two_pi_dvd_sub]
  /-
    ⊢ Exists fun k => Eq (HSub.hSub (Neg.neg Real.pi) Real.pi) (HMul.hMul (HMul.hM …
  -/
  use -1
  /-
    case h
    ⊢ Eq (HSub.hSub (Neg.neg Real.pi) Real.pi) (HMul.hMul (HMul.hMul 2 Real.pi) ↑( …
  -/
  simp [two_mul, sub_eq_add_neg]
  /-
    🎉 no goals
  -/


@[simp]
theorem two_nsmul_coe_div_two (θ : ℝ) : (2 : ℕ) • (↑(θ / 2) : Angle) = θ := by
  /-
    θ : Real
    ⊢ Eq (HSMul.hSMul 2 ↑(HDiv.hDiv θ 2)) ↑θ
  -/
  rw [← coe_nsmul, two_nsmul, add_halves]
  /-
    🎉 no goals
  -/


@[simp]
theorem two_zsmul_coe_div_two (θ : ℝ) : (2 : ℤ) • (↑(θ / 2) : Angle) = θ := by
  /-
    θ : Real
    ⊢ Eq (HSMul.hSMul 2 ↑(HDiv.hDiv θ 2)) ↑θ
  -/
  rw [← coe_zsmul, two_zsmul, add_halves]
  /-
    🎉 no goals
  -/


theorem two_nsmul_neg_pi_div_two : (2 : ℕ) • (↑(-π / 2) : Angle) = π := by
  /-
    ⊢ Eq (HSMul.hSMul 2 ↑(HDiv.hDiv (Neg.neg Real.pi) 2)) ↑Real.pi
  -/
  rw [two_nsmul_coe_div_two, coe_neg, neg_coe_pi]
  /-
    🎉 no goals
  -/


theorem two_zsmul_neg_pi_div_two : (2 : ℤ) • (↑(-π / 2) : Angle) = π := by
  /-
    ⊢ Eq (HSMul.hSMul 2 ↑(HDiv.hDiv (Neg.neg Real.pi) 2)) ↑Real.pi
  -/
  rw [two_zsmul, ← two_nsmul, two_nsmul_neg_pi_div_two]
  /-
    🎉 no goals
  -/


theorem sub_coe_pi_eq_add_coe_pi (θ : Angle) : θ - π = θ + π := by
  /-
    θ : Real.Angle
    ⊢ Eq (HSub.hSub θ ↑Real.pi) (HAdd.hAdd θ ↑Real.pi)
  -/
  rw [sub_eq_add_neg, neg_coe_pi]
  /-
    🎉 no goals
  -/


@[simp]
                                                           /-
                                                             ⊢ Eq (HSMul.hSMul 2 ↑Real.pi) 0
                                                           -/
theorem two_nsmul_coe_pi : (2 : ℕ) • (π : Angle) = 0 := by simp [← natCast_mul_eq_nsmul]
                                                           /-
                                                             🎉 no goals
                                                           -/


@[simp]
                                                           /-
                                                             ⊢ Eq (HSMul.hSMul 2 ↑Real.pi) 0
                                                           -/
theorem two_zsmul_coe_pi : (2 : ℤ) • (π : Angle) = 0 := by simp [← intCast_mul_eq_zsmul]
                                                           /-
                                                             🎉 no goals
                                                           -/


@[simp]
                                                           /-
                                                             ⊢ Eq (HAdd.hAdd ↑Real.pi ↑Real.pi) 0
                                                           -/
theorem coe_pi_add_coe_pi : (π : Real.Angle) + π = 0 := by rw [← two_nsmul, two_nsmul_coe_pi]
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem zsmul_eq_iff {ψ θ : Angle} {z : ℤ} (hz : z ≠ 0) :
    z • ψ = z • θ ↔ ∃ k : Fin z.natAbs, ψ = θ + (k : ℕ) • (2 * π / z : ℝ) :=
  QuotientAddGroup.zmultiples_zsmul_eq_zsmul_iff hz


theorem nsmul_eq_iff {ψ θ : Angle} {n : ℕ} (hz : n ≠ 0) :
    n • ψ = n • θ ↔ ∃ k : Fin n, ψ = θ + (k : ℕ) • (2 * π / n : ℝ) :=
  QuotientAddGroup.zmultiples_nsmul_eq_nsmul_iff hz


theorem two_zsmul_eq_iff {ψ θ : Angle} : (2 : ℤ) • ψ = (2 : ℤ) • θ ↔ ψ = θ ∨ ψ = θ + ↑π := by
  /-
    ψ θ : Real.Angle
    ⊢ Iff (Eq (HSMul.hSMul 2 ψ) (HSMul.hSMul 2 θ)) (Or (Eq ψ θ) (Eq ψ (HAdd.hAdd θ …
  -/
  have : Int.natAbs 2 = 2 := rfl
  rw [zsmul_eq_iff two_ne_zero, this, Fin.exists_fin_two, Fin.val_zero,
    Fin.val_one, zero_smul, add_zero, one_smul, Int.cast_two,
    mul_div_cancel_left₀ (_ : ℝ) two_ne_zero]


theorem two_nsmul_eq_iff {ψ θ : Angle} : (2 : ℕ) • ψ = (2 : ℕ) • θ ↔ ψ = θ ∨ ψ = θ + ↑π := by
  /-
    ψ θ : Real.Angle
    ⊢ Iff (Eq (HSMul.hSMul 2 ψ) (HSMul.hSMul 2 θ)) (Or (Eq ψ θ) (Eq ψ (HAdd.hAdd θ …
  -/
  simp_rw [← natCast_zsmul, Nat.cast_ofNat, two_zsmul_eq_iff]
  /-
    🎉 no goals
  -/


theorem two_nsmul_eq_zero_iff {θ : Angle} : (2 : ℕ) • θ = 0 ↔ θ = 0 ∨ θ = π := by
  /-
    θ : Real.Angle
    ⊢ Iff (Eq (HSMul.hSMul 2 θ) 0) (Or (Eq θ 0) (Eq θ ↑Real.pi))
  -/
                               /-
                                 🎉 no goals
                               -/
  convert two_nsmul_eq_iff <;> simp
                               /-
                                 🎉 no goals
                               -/


theorem two_nsmul_ne_zero_iff {θ : Angle} : (2 : ℕ) • θ ≠ 0 ↔ θ ≠ 0 ∧ θ ≠ π := by
  /-
    θ : Real.Angle
    ⊢ Iff (Ne (HSMul.hSMul 2 θ) 0) (And (Ne θ 0) (Ne θ ↑Real.pi))
  -/
  rw [← not_or, ← two_nsmul_eq_zero_iff]
  /-
    🎉 no goals
  -/


theorem two_zsmul_eq_zero_iff {θ : Angle} : (2 : ℤ) • θ = 0 ↔ θ = 0 ∨ θ = π := by
  /-
    θ : Real.Angle
    ⊢ Iff (Eq (HSMul.hSMul 2 θ) 0) (Or (Eq θ 0) (Eq θ ↑Real.pi))
  -/
  simp_rw [two_zsmul, ← two_nsmul, two_nsmul_eq_zero_iff]
  /-
    🎉 no goals
  -/


theorem two_zsmul_ne_zero_iff {θ : Angle} : (2 : ℤ) • θ ≠ 0 ↔ θ ≠ 0 ∧ θ ≠ π := by
  /-
    θ : Real.Angle
    ⊢ Iff (Ne (HSMul.hSMul 2 θ) 0) (And (Ne θ 0) (Ne θ ↑Real.pi))
  -/
  rw [← not_or, ← two_zsmul_eq_zero_iff]
  /-
    🎉 no goals
  -/


theorem eq_neg_self_iff {θ : Angle} : θ = -θ ↔ θ = 0 ∨ θ = π := by
  /-
    θ : Real.Angle
    ⊢ Iff (Eq θ (Neg.neg θ)) (Or (Eq θ 0) (Eq θ ↑Real.pi))
  -/
  rw [← add_eq_zero_iff_eq_neg, ← two_nsmul, two_nsmul_eq_zero_iff]
  /-
    🎉 no goals
  -/


theorem ne_neg_self_iff {θ : Angle} : θ ≠ -θ ↔ θ ≠ 0 ∧ θ ≠ π := by
  /-
    θ : Real.Angle
    ⊢ Iff (Ne θ (Neg.neg θ)) (And (Ne θ 0) (Ne θ ↑Real.pi))
  -/
  rw [← not_or, ← eq_neg_self_iff.not]
  /-
    🎉 no goals
  -/


                                                                   /-
                                                                     θ : Real.Angle
                                                                     ⊢ Iff (Eq (Neg.neg θ) θ) (Or (Eq θ 0) (Eq θ ↑Real.pi))
                                                                   -/
theorem neg_eq_self_iff {θ : Angle} : -θ = θ ↔ θ = 0 ∨ θ = π := by rw [eq_comm, eq_neg_self_iff]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


theorem neg_ne_self_iff {θ : Angle} : -θ ≠ θ ↔ θ ≠ 0 ∧ θ ≠ π := by
  /-
    θ : Real.Angle
    ⊢ Iff (Ne (Neg.neg θ) θ) (And (Ne θ 0) (Ne θ ↑Real.pi))
  -/
  rw [← not_or, ← neg_eq_self_iff.not]
  /-
    🎉 no goals
  -/


theorem two_nsmul_eq_pi_iff {θ : Angle} : (2 : ℕ) • θ = π ↔ θ = (π / 2 : ℝ) ∨ θ = (-π / 2 : ℝ) := by
  /-
    θ : Real.Angle
    ⊢ Iff (Eq (HSMul.hSMul 2 θ) ↑Real.pi) (Or (Eq θ ↑(HDiv.hDiv Real.pi 2)) (Eq θ  …
  -/
  have h : (π : Angle) = ((2 : ℕ) • (π / 2 : ℝ):) := by rw [two_nsmul, add_halves]
  /-
    θ : Real.Angle
    h : Eq ↑Real.pi ↑(HSMul.hSMul 2 (HDiv.hDiv Real.pi 2))
    ⊢ Iff (Eq (HSMul.hSMul 2 θ) ↑Real.pi) (Or (Eq θ ↑(HDiv.hDiv Real.pi 2)) (Eq θ  …
  -/
  nth_rw 1 [h]
  /-
    θ : Real.Angle
    h : Eq ↑Real.pi ↑(HSMul.hSMul 2 (HDiv.hDiv Real.pi 2))
    ⊢ Iff (Eq (HSMul.hSMul 2 θ) ↑(HSMul.hSMul 2 (HDiv.hDiv Real.pi 2))) (Or (Eq θ  …
  -/
  rw [coe_nsmul, two_nsmul_eq_iff]
  -- Porting note: `congr` didn't simplify the goal of iff of `Or`s
  /-
    θ : Real.Angle
    h : Eq ↑Real.pi ↑(HSMul.hSMul 2 (HDiv.hDiv Real.pi 2))
    ⊢ Iff (Or (Eq θ ↑(HDiv.hDiv Real.pi 2)) (Eq θ (HAdd.hAdd ↑(HDiv.hDiv Real.pi 2 …
  -/
  convert Iff.rfl
  rw [add_comm, ← coe_add, ← sub_eq_zero, ← coe_sub, neg_div, ← neg_sub, sub_neg_eq_add, add_assoc,
    add_halves, ← two_mul, coe_neg, coe_two_pi, neg_zero]


theorem two_zsmul_eq_pi_iff {θ : Angle} : (2 : ℤ) • θ = π ↔ θ = (π / 2 : ℝ) ∨ θ = (-π / 2 : ℝ) := by
  /-
    θ : Real.Angle
    ⊢ Iff (Eq (HSMul.hSMul 2 θ) ↑Real.pi) (Or (Eq θ ↑(HDiv.hDiv Real.pi 2)) (Eq θ  …
  -/
  rw [two_zsmul, ← two_nsmul, two_nsmul_eq_pi_iff]
  /-
    🎉 no goals
  -/


theorem cos_eq_iff_coe_eq_or_eq_neg {θ ψ : ℝ} :
    cos θ = cos ψ ↔ (θ : Angle) = ψ ∨ (θ : Angle) = -ψ := by
  /-
    θ ψ : Real
    ⊢ Iff (Eq (Real.cos θ) (Real.cos ψ)) (Or (Eq ↑θ ↑ψ) (Eq (↑θ) (Neg.neg ↑ψ)))
  -/
  constructor
    /-
      case mp
      θ ψ : Real
      ⊢ Eq (Real.cos θ) (Real.cos ψ) → Or (Eq ↑θ ↑ψ) (Eq (↑θ) (Neg.neg ↑ψ))
    -/
  · intro Hcos
    rw [← sub_eq_zero, cos_sub_cos, mul_eq_zero, mul_eq_zero, neg_eq_zero,
      eq_false (two_ne_zero' ℝ), false_or, sin_eq_zero_iff, sin_eq_zero_iff] at Hcos
    /-
      case mp
      θ ψ : Real
      Hcos : Or (Exists fun n => Eq (HMul.hMul (↑n) Real.pi) (HDiv.hDiv (HAdd.hAdd θ …
      ⊢ Or (Eq ↑θ ↑ψ) (Eq (↑θ) (Neg.neg ↑ψ))
    -/
    rcases Hcos with (⟨n, hn⟩ | ⟨n, hn⟩)
      /-
        case mp.inl.intro
        θ ψ : Real
        n : Int
        hn : Eq (HMul.hMul (↑n) Real.pi) (HDiv.hDiv (HAdd.hAdd θ ψ) 2)
        ⊢ Or (Eq ↑θ ↑ψ) (Eq (↑θ) (Neg.neg ↑ψ))
      -/
    · right
      /-
        case mp.inl.intro.h
        θ ψ : Real
        n : Int
        hn : Eq (HMul.hMul (↑n) Real.pi) (HDiv.hDiv (HAdd.hAdd θ ψ) 2)
        ⊢ Eq (↑θ) (Neg.neg ↑ψ)
      -/
      rw [eq_div_iff_mul_eq (two_ne_zero' ℝ), ← sub_eq_iff_eq_add] at hn
      rw [← hn, coe_sub, eq_neg_iff_add_eq_zero, sub_add_cancel, mul_assoc, intCast_mul_eq_zsmul,
        mul_comm, coe_two_pi, zsmul_zero]
      /-
        case mp.inr.intro
        θ ψ : Real
        n : Int
        hn : Eq (HMul.hMul (↑n) Real.pi) (HDiv.hDiv (HSub.hSub θ ψ) 2)
        ⊢ Or (Eq ↑θ ↑ψ) (Eq (↑θ) (Neg.neg ↑ψ))
      -/
    · left
      /-
        case mp.inr.intro.h
        θ ψ : Real
        n : Int
        hn : Eq (HMul.hMul (↑n) Real.pi) (HDiv.hDiv (HSub.hSub θ ψ) 2)
        ⊢ Eq ↑θ ↑ψ
      -/
      rw [eq_div_iff_mul_eq (two_ne_zero' ℝ), eq_sub_iff_add_eq] at hn
      rw [← hn, coe_add, mul_assoc, intCast_mul_eq_zsmul, mul_comm, coe_two_pi, zsmul_zero,
        zero_add]
    /-
      case mpr
      θ ψ : Real
      ⊢ Or (Eq ↑θ ↑ψ) (Eq (↑θ) (Neg.neg ↑ψ)) → Eq (Real.cos θ) (Real.cos ψ)
    -/
  · rw [angle_eq_iff_two_pi_dvd_sub, ← coe_neg, angle_eq_iff_two_pi_dvd_sub]
    /-
      case mpr
      θ ψ : Real
      ⊢ Or (Exists fun k => Eq (HSub.hSub θ ψ) (HMul.hMul (HMul.hMul 2 Real.pi) ↑k)) …
    -/
    rintro (⟨k, H⟩ | ⟨k, H⟩)
    · rw [← sub_eq_zero, cos_sub_cos, H, mul_assoc 2 π k, mul_div_cancel_left₀ _ (two_ne_zero' ℝ),
        mul_comm π _, sin_int_mul_pi, mul_zero]
    rw [← sub_eq_zero, cos_sub_cos, ← sub_neg_eq_add, H, mul_assoc 2 π k,
      mul_div_cancel_left₀ _ (two_ne_zero' ℝ), mul_comm π _, sin_int_mul_pi, mul_zero,
      zero_mul]


theorem sin_eq_iff_coe_eq_or_add_eq_pi {θ ψ : ℝ} :
    sin θ = sin ψ ↔ (θ : Angle) = ψ ∨ (θ : Angle) + ψ = π := by
  /-
    θ ψ : Real
    ⊢ Iff (Eq (Real.sin θ) (Real.sin ψ)) (Or (Eq ↑θ ↑ψ) (Eq (HAdd.hAdd ↑θ ↑ψ) ↑Rea …
  -/
  constructor
    /-
      case mp
      θ ψ : Real
      ⊢ Eq (Real.sin θ) (Real.sin ψ) → Or (Eq ↑θ ↑ψ) (Eq (HAdd.hAdd ↑θ ↑ψ) ↑Real.pi)
    -/
  · intro Hsin
    /-
      case mp
      θ ψ : Real
      Hsin : Eq (Real.sin θ) (Real.sin ψ)
      ⊢ Or (Eq ↑θ ↑ψ) (Eq (HAdd.hAdd ↑θ ↑ψ) ↑Real.pi)
    -/
    rw [← cos_pi_div_two_sub, ← cos_pi_div_two_sub] at Hsin
    /-
      case mp
      θ ψ : Real
      Hsin : Eq (Real.cos (HSub.hSub (HDiv.hDiv Real.pi 2) θ)) (Real.cos (HSub.hSub  …
      ⊢ Or (Eq ↑θ ↑ψ) (Eq (HAdd.hAdd ↑θ ↑ψ) ↑Real.pi)
    -/
    cases' cos_eq_iff_coe_eq_or_eq_neg.mp Hsin with h h
      /-
        case mp.inl
        θ ψ : Real
        Hsin : Eq (Real.cos (HSub.hSub (HDiv.hDiv Real.pi 2) θ)) (Real.cos (HSub.hSub  …
        h : Eq ↑(HSub.hSub (HDiv.hDiv Real.pi 2) θ) ↑(HSub.hSub (HDiv.hDiv Real.pi 2) ψ)
        ⊢ Or (Eq ↑θ ↑ψ) (Eq (HAdd.hAdd ↑θ ↑ψ) ↑Real.pi)
      -/
    · left
      /-
        case mp.inl.h
        θ ψ : Real
        Hsin : Eq (Real.cos (HSub.hSub (HDiv.hDiv Real.pi 2) θ)) (Real.cos (HSub.hSub  …
        h : Eq ↑(HSub.hSub (HDiv.hDiv Real.pi 2) θ) ↑(HSub.hSub (HDiv.hDiv Real.pi 2) ψ)
        ⊢ Eq ↑θ ↑ψ
      -/
      rw [coe_sub, coe_sub] at h
      /-
        case mp.inl.h
        θ ψ : Real
        Hsin : Eq (Real.cos (HSub.hSub (HDiv.hDiv Real.pi 2) θ)) (Real.cos (HSub.hSub  …
        h : Eq (HSub.hSub ↑(HDiv.hDiv Real.pi 2) ↑θ) (HSub.hSub ↑(HDiv.hDiv Real.pi 2) …
        ⊢ Eq ↑θ ↑ψ
      -/
      exact sub_right_inj.1 h
      /-
        🎉 no goals
      -/
    /-
      case mp.inr
      θ ψ : Real
      Hsin : Eq (Real.cos (HSub.hSub (HDiv.hDiv Real.pi 2) θ)) (Real.cos (HSub.hSub  …
      h : Eq (↑(HSub.hSub (HDiv.hDiv Real.pi 2) θ)) (Neg.neg ↑(HSub.hSub (HDiv.hDiv  …
      ⊢ Or (Eq ↑θ ↑ψ) (Eq (HAdd.hAdd ↑θ ↑ψ) ↑Real.pi)
    -/
    right
    rw [coe_sub, coe_sub, eq_neg_iff_add_eq_zero, add_sub, sub_add_eq_add_sub, ← coe_add,
      add_halves, sub_sub, sub_eq_zero] at h
    /-
      case mp.inr.h
      θ ψ : Real
      Hsin : Eq (Real.cos (HSub.hSub (HDiv.hDiv Real.pi 2) θ)) (Real.cos (HSub.hSub  …
      h : Eq (↑Real.pi) (HAdd.hAdd ↑θ ↑ψ)
      ⊢ Eq (HAdd.hAdd ↑θ ↑ψ) ↑Real.pi
    -/
    exact h.symm
    /-
      🎉 no goals
    -/
    /-
      case mpr
      θ ψ : Real
      ⊢ Or (Eq ↑θ ↑ψ) (Eq (HAdd.hAdd ↑θ ↑ψ) ↑Real.pi) → Eq (Real.sin θ) (Real.sin ψ)
    -/
  · rw [angle_eq_iff_two_pi_dvd_sub, ← eq_sub_iff_add_eq, ← coe_sub, angle_eq_iff_two_pi_dvd_sub]
    /-
      case mpr
      θ ψ : Real
      ⊢ Or (Exists fun k => Eq (HSub.hSub θ ψ) (HMul.hMul (HMul.hMul 2 Real.pi) ↑k)) …
    -/
    rintro (⟨k, H⟩ | ⟨k, H⟩)
    · rw [← sub_eq_zero, sin_sub_sin, H, mul_assoc 2 π k, mul_div_cancel_left₀ _ (two_ne_zero' ℝ),
        mul_comm π _, sin_int_mul_pi, mul_zero, zero_mul]
    have H' : θ + ψ = 2 * k * π + π := by
      rwa [← sub_add, sub_add_eq_add_sub, sub_eq_iff_eq_add, mul_assoc, mul_comm π _, ←
        mul_assoc] at H
    rw [← sub_eq_zero, sin_sub_sin, H', add_div, mul_assoc 2 _ π,
      mul_div_cancel_left₀ _ (two_ne_zero' ℝ), cos_add_pi_div_two, sin_int_mul_pi, neg_zero,
      mul_zero]


theorem cos_sin_inj {θ ψ : ℝ} (Hcos : cos θ = cos ψ) (Hsin : sin θ = sin ψ) : (θ : Angle) = ψ := by
  /-
    θ ψ : Real
    Hcos : Eq (Real.cos θ) (Real.cos ψ)
    Hsin : Eq (Real.sin θ) (Real.sin ψ)
    ⊢ Eq ↑θ ↑ψ
  -/
  cases' cos_eq_iff_coe_eq_or_eq_neg.mp Hcos with hc hc; · exact hc
                                                           /-
                                                             🎉 no goals
                                                           -/
  /-
    case inr
    θ ψ : Real
    Hcos : Eq (Real.cos θ) (Real.cos ψ)
    Hsin : Eq (Real.sin θ) (Real.sin ψ)
    hc : Eq (↑θ) (Neg.neg ↑ψ)
    ⊢ Eq ↑θ ↑ψ
  -/
  cases' sin_eq_iff_coe_eq_or_add_eq_pi.mp Hsin with hs hs; · exact hs
                                                              /-
                                                                🎉 no goals
                                                              -/
  /-
    case inr.inr
    θ ψ : Real
    Hcos : Eq (Real.cos θ) (Real.cos ψ)
    Hsin : Eq (Real.sin θ) (Real.sin ψ)
    hc : Eq (↑θ) (Neg.neg ↑ψ)
    hs : Eq (HAdd.hAdd ↑θ ↑ψ) ↑Real.pi
    ⊢ Eq ↑θ ↑ψ
  -/
  rw [eq_neg_iff_add_eq_zero, hs] at hc
  /-
    case inr.inr
    θ ψ : Real
    Hcos : Eq (Real.cos θ) (Real.cos ψ)
    Hsin : Eq (Real.sin θ) (Real.sin ψ)
    hc : Eq (↑Real.pi) 0
    hs : Eq (HAdd.hAdd ↑θ ↑ψ) ↑Real.pi
    ⊢ Eq ↑θ ↑ψ
  -/
  obtain ⟨n, hn⟩ : ∃ n, n • _ = _ := QuotientAddGroup.leftRel_apply.mp (Quotient.exact' hc)
  rw [← neg_one_mul, add_zero, ← sub_eq_zero, zsmul_eq_mul, ← mul_assoc, ← sub_mul, mul_eq_zero,
    eq_false (ne_of_gt pi_pos), or_false, sub_neg_eq_add, ← Int.cast_zero, ← Int.cast_one,
    ← Int.cast_ofNat, ← Int.cast_mul, ← Int.cast_add, Int.cast_inj] at hn
  /-
    case inr.inr.intro
    θ ψ : Real
    Hcos : Eq (Real.cos θ) (Real.cos ψ)
    Hsin : Eq (Real.sin θ) (Real.sin ψ)
    hc : Eq (↑Real.pi) 0
    hs : Eq (HAdd.hAdd ↑θ ↑ψ) ↑Real.pi
    n : Int
    hn : Eq (HAdd.hAdd (HMul.hMul n 2) 1) 0
    ⊢ Eq ↑θ ↑ψ
  -/
  have : (n * 2 + 1) % (2 : ℤ) = 0 % (2 : ℤ) := congr_arg (· % (2 : ℤ)) hn
  /-
    case inr.inr.intro
    θ ψ : Real
    Hcos : Eq (Real.cos θ) (Real.cos ψ)
    Hsin : Eq (Real.sin θ) (Real.sin ψ)
    hc : Eq (↑Real.pi) 0
    hs : Eq (HAdd.hAdd ↑θ ↑ψ) ↑Real.pi
    n : Int
    hn : Eq (HAdd.hAdd (HMul.hMul n 2) 1) 0
    this : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul n 2) 1) 2) (HMod.hMod 0 2)
    ⊢ Eq ↑θ ↑ψ
  -/
  rw [add_comm, Int.add_mul_emod_self] at this
  /-
    case inr.inr.intro
    θ ψ : Real
    Hcos : Eq (Real.cos θ) (Real.cos ψ)
    Hsin : Eq (Real.sin θ) (Real.sin ψ)
    hc : Eq (↑Real.pi) 0
    hs : Eq (HAdd.hAdd ↑θ ↑ψ) ↑Real.pi
    n : Int
    hn : Eq (HAdd.hAdd (HMul.hMul n 2) 1) 0
    this : Eq (HMod.hMod 1 2) (HMod.hMod 0 2)
    ⊢ Eq ↑θ ↑ψ
  -/
  exact absurd this one_ne_zero
  /-
    🎉 no goals
  -/


/-- The sine of a `Real.Angle`. -/
def sin (θ : Angle) : ℝ :=
  sin_periodic.lift θ


@[simp]
theorem sin_coe (x : ℝ) : sin (x : Angle) = Real.sin x :=
  rfl


@[continuity]
theorem continuous_sin : Continuous sin :=
  Real.continuous_sin.quotient_liftOn' _


/-- The cosine of a `Real.Angle`. -/
def cos (θ : Angle) : ℝ :=
  cos_periodic.lift θ


@[simp]
theorem cos_coe (x : ℝ) : cos (x : Angle) = Real.cos x :=
  rfl


@[continuity]
theorem continuous_cos : Continuous cos :=
  Real.continuous_cos.quotient_liftOn' _


theorem cos_eq_real_cos_iff_eq_or_eq_neg {θ : Angle} {ψ : ℝ} :
    cos θ = Real.cos ψ ↔ θ = ψ ∨ θ = -ψ := by
  /-
    θ : Real.Angle
    ψ : Real
    ⊢ Iff (Eq θ.cos (Real.cos ψ)) (Or (Eq θ ↑ψ) (Eq θ (Neg.neg ↑ψ)))
  -/
  induction θ using Real.Angle.induction_on
  /-
    case h
    ψ x✝ : Real
    ⊢ Iff (Eq (↑x✝).cos (Real.cos ψ)) (Or (Eq ↑x✝ ↑ψ) (Eq (↑x✝) (Neg.neg ↑ψ)))
  -/
  exact cos_eq_iff_coe_eq_or_eq_neg
  /-
    🎉 no goals
  -/


theorem cos_eq_iff_eq_or_eq_neg {θ ψ : Angle} : cos θ = cos ψ ↔ θ = ψ ∨ θ = -ψ := by
  /-
    θ ψ : Real.Angle
    ⊢ Iff (Eq θ.cos ψ.cos) (Or (Eq θ ψ) (Eq θ (Neg.neg ψ)))
  -/
  induction ψ using Real.Angle.induction_on
  /-
    case h
    θ : Real.Angle
    x✝ : Real
    ⊢ Iff (Eq θ.cos (↑x✝).cos) (Or (Eq θ ↑x✝) (Eq θ (Neg.neg ↑x✝)))
  -/
  exact cos_eq_real_cos_iff_eq_or_eq_neg
  /-
    🎉 no goals
  -/


theorem sin_eq_real_sin_iff_eq_or_add_eq_pi {θ : Angle} {ψ : ℝ} :
    sin θ = Real.sin ψ ↔ θ = ψ ∨ θ + ψ = π := by
  /-
    θ : Real.Angle
    ψ : Real
    ⊢ Iff (Eq θ.sin (Real.sin ψ)) (Or (Eq θ ↑ψ) (Eq (HAdd.hAdd θ ↑ψ) ↑Real.pi))
  -/
  induction θ using Real.Angle.induction_on
  /-
    case h
    ψ x✝ : Real
    ⊢ Iff (Eq (↑x✝).sin (Real.sin ψ)) (Or (Eq ↑x✝ ↑ψ) (Eq (HAdd.hAdd ↑x✝ ↑ψ) ↑Real …
  -/
  exact sin_eq_iff_coe_eq_or_add_eq_pi
  /-
    🎉 no goals
  -/


theorem sin_eq_iff_eq_or_add_eq_pi {θ ψ : Angle} : sin θ = sin ψ ↔ θ = ψ ∨ θ + ψ = π := by
  /-
    θ ψ : Real.Angle
    ⊢ Iff (Eq θ.sin ψ.sin) (Or (Eq θ ψ) (Eq (HAdd.hAdd θ ψ) ↑Real.pi))
  -/
  induction ψ using Real.Angle.induction_on
  /-
    case h
    θ : Real.Angle
    x✝ : Real
    ⊢ Iff (Eq θ.sin (↑x✝).sin) (Or (Eq θ ↑x✝) (Eq (HAdd.hAdd θ ↑x✝) ↑Real.pi))
  -/
  exact sin_eq_real_sin_iff_eq_or_add_eq_pi
  /-
    🎉 no goals
  -/


@[simp]
                                             /-
                                               ⊢ Eq (Real.Angle.sin 0) 0
                                             -/
theorem sin_zero : sin (0 : Angle) = 0 := by rw [← coe_zero, sin_coe, Real.sin_zero]
                                             /-
                                               🎉 no goals
                                             -/


                                               /-
                                                 ⊢ Eq (↑Real.pi).sin 0
                                               -/
theorem sin_coe_pi : sin (π : Angle) = 0 := by rw [sin_coe, Real.sin_pi]
                                               /-
                                                 🎉 no goals
                                               -/


theorem sin_eq_zero_iff {θ : Angle} : sin θ = 0 ↔ θ = 0 ∨ θ = π := by
  /-
    θ : Real.Angle
    ⊢ Iff (Eq θ.sin 0) (Or (Eq θ 0) (Eq θ ↑Real.pi))
  -/
  nth_rw 1 [← sin_zero]
  /-
    θ : Real.Angle
    ⊢ Iff (Eq θ.sin (Real.Angle.sin 0)) (Or (Eq θ 0) (Eq θ ↑Real.pi))
  -/
  rw [sin_eq_iff_eq_or_add_eq_pi]
  /-
    θ : Real.Angle
    ⊢ Iff (Or (Eq θ 0) (Eq (HAdd.hAdd θ 0) ↑Real.pi)) (Or (Eq θ 0) (Eq θ ↑Real.pi))
  -/
  simp
  /-
    🎉 no goals
  -/


theorem sin_ne_zero_iff {θ : Angle} : sin θ ≠ 0 ↔ θ ≠ 0 ∧ θ ≠ π := by
  /-
    θ : Real.Angle
    ⊢ Iff (Ne θ.sin 0) (And (Ne θ 0) (Ne θ ↑Real.pi))
  -/
  rw [← not_or, ← sin_eq_zero_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem sin_neg (θ : Angle) : sin (-θ) = -sin θ := by
  /-
    θ : Real.Angle
    ⊢ Eq (Neg.neg θ).sin (Neg.neg θ.sin)
  -/
  induction θ using Real.Angle.induction_on
  /-
    case h
    x✝ : Real
    ⊢ Eq (Neg.neg ↑x✝).sin (Neg.neg (↑x✝).sin)
  -/
  exact Real.sin_neg _
  /-
    🎉 no goals
  -/


theorem sin_antiperiodic : Function.Antiperiodic sin (π : Angle) := by
  /-
    ⊢ Function.Antiperiodic Real.Angle.sin ↑Real.pi
  -/
  intro θ
  /-
    θ : Real.Angle
    ⊢ Eq (HAdd.hAdd θ ↑Real.pi).sin (Neg.neg θ.sin)
  -/
  induction θ using Real.Angle.induction_on
  /-
    case h
    x✝ : Real
    ⊢ Eq (HAdd.hAdd ↑x✝ ↑Real.pi).sin (Neg.neg (↑x✝).sin)
  -/
  exact Real.sin_antiperiodic _
  /-
    🎉 no goals
  -/


@[simp]
theorem sin_add_pi (θ : Angle) : sin (θ + π) = -sin θ :=
  sin_antiperiodic θ


@[simp]
theorem sin_sub_pi (θ : Angle) : sin (θ - π) = -sin θ :=
  sin_antiperiodic.sub_eq θ


@[simp]
                                             /-
                                               ⊢ Eq (Real.Angle.cos 0) 1
                                             -/
theorem cos_zero : cos (0 : Angle) = 1 := by rw [← coe_zero, cos_coe, Real.cos_zero]
                                             /-
                                               🎉 no goals
                                             -/


                                                /-
                                                  ⊢ Eq (↑Real.pi).cos (-1)
                                                -/
theorem cos_coe_pi : cos (π : Angle) = -1 := by rw [cos_coe, Real.cos_pi]
                                                /-
                                                  🎉 no goals
                                                -/


@[simp]
theorem cos_neg (θ : Angle) : cos (-θ) = cos θ := by
  /-
    θ : Real.Angle
    ⊢ Eq (Neg.neg θ).cos θ.cos
  -/
  induction θ using Real.Angle.induction_on
  /-
    case h
    x✝ : Real
    ⊢ Eq (Neg.neg ↑x✝).cos (↑x✝).cos
  -/
  exact Real.cos_neg _
  /-
    🎉 no goals
  -/


theorem cos_antiperiodic : Function.Antiperiodic cos (π : Angle) := by
  /-
    ⊢ Function.Antiperiodic Real.Angle.cos ↑Real.pi
  -/
  intro θ
  /-
    θ : Real.Angle
    ⊢ Eq (HAdd.hAdd θ ↑Real.pi).cos (Neg.neg θ.cos)
  -/
  induction θ using Real.Angle.induction_on
  /-
    case h
    x✝ : Real
    ⊢ Eq (HAdd.hAdd ↑x✝ ↑Real.pi).cos (Neg.neg (↑x✝).cos)
  -/
  exact Real.cos_antiperiodic _
  /-
    🎉 no goals
  -/


@[simp]
theorem cos_add_pi (θ : Angle) : cos (θ + π) = -cos θ :=
  cos_antiperiodic θ


@[simp]
theorem cos_sub_pi (θ : Angle) : cos (θ - π) = -cos θ :=
  cos_antiperiodic.sub_eq θ


theorem cos_eq_zero_iff {θ : Angle} : cos θ = 0 ↔ θ = (π / 2 : ℝ) ∨ θ = (-π / 2 : ℝ) := by
  /-
    θ : Real.Angle
    ⊢ Iff (Eq θ.cos 0) (Or (Eq θ ↑(HDiv.hDiv Real.pi 2)) (Eq θ ↑(HDiv.hDiv (Neg.ne …
  -/
  rw [← cos_pi_div_two, ← cos_coe, cos_eq_iff_eq_or_eq_neg, ← coe_neg, ← neg_div]
  /-
    🎉 no goals
  -/


theorem sin_add (θ₁ θ₂ : Real.Angle) : sin (θ₁ + θ₂) = sin θ₁ * cos θ₂ + cos θ₁ * sin θ₂ := by
  /-
    θ₁ θ₂ : Real.Angle
    ⊢ Eq (HAdd.hAdd θ₁ θ₂).sin (HAdd.hAdd (HMul.hMul θ₁.sin θ₂.cos) (HMul.hMul θ₁. …
  -/
  induction θ₁ using Real.Angle.induction_on
  /-
    case h
    θ₂ : Real.Angle
    x✝ : Real
    ⊢ Eq (HAdd.hAdd (↑x✝) θ₂).sin (HAdd.hAdd (HMul.hMul (↑x✝).sin θ₂.cos) (HMul.hM …
  -/
  induction θ₂ using Real.Angle.induction_on
  /-
    case h.h
    x✝¹ x✝ : Real
    ⊢ Eq (HAdd.hAdd ↑x✝¹ ↑x✝).sin (HAdd.hAdd (HMul.hMul (↑x✝¹).sin (↑x✝).cos) (HMu …
  -/
  exact Real.sin_add _ _
  /-
    🎉 no goals
  -/


theorem cos_add (θ₁ θ₂ : Real.Angle) : cos (θ₁ + θ₂) = cos θ₁ * cos θ₂ - sin θ₁ * sin θ₂ := by
  /-
    θ₁ θ₂ : Real.Angle
    ⊢ Eq (HAdd.hAdd θ₁ θ₂).cos (HSub.hSub (HMul.hMul θ₁.cos θ₂.cos) (HMul.hMul θ₁. …
  -/
  induction θ₂ using Real.Angle.induction_on
  /-
    case h
    θ₁ : Real.Angle
    x✝ : Real
    ⊢ Eq (HAdd.hAdd θ₁ ↑x✝).cos (HSub.hSub (HMul.hMul θ₁.cos (↑x✝).cos) (HMul.hMul …
  -/
  induction θ₁ using Real.Angle.induction_on
  /-
    case h.h
    x✝¹ x✝ : Real
    ⊢ Eq (HAdd.hAdd ↑x✝ ↑x✝¹).cos (HSub.hSub (HMul.hMul (↑x✝).cos (↑x✝¹).cos) (HMu …
  -/
  exact Real.cos_add _ _
  /-
    🎉 no goals
  -/


@[simp]
theorem cos_sq_add_sin_sq (θ : Real.Angle) : cos θ ^ 2 + sin θ ^ 2 = 1 := by
  /-
    θ : Real.Angle
    ⊢ Eq (HAdd.hAdd (HPow.hPow θ.cos 2) (HPow.hPow θ.sin 2)) 1
  -/
  induction θ using Real.Angle.induction_on
  /-
    case h
    x✝ : Real
    ⊢ Eq (HAdd.hAdd (HPow.hPow (↑x✝).cos 2) (HPow.hPow (↑x✝).sin 2)) 1
  -/
  exact Real.cos_sq_add_sin_sq _
  /-
    🎉 no goals
  -/


theorem sin_add_pi_div_two (θ : Angle) : sin (θ + ↑(π / 2)) = cos θ := by
  /-
    θ : Real.Angle
    ⊢ Eq (HAdd.hAdd θ ↑(HDiv.hDiv Real.pi 2)).sin θ.cos
  -/
  induction θ using Real.Angle.induction_on
  /-
    case h
    x✝ : Real
    ⊢ Eq (HAdd.hAdd ↑x✝ ↑(HDiv.hDiv Real.pi 2)).sin (↑x✝).cos
  -/
  exact Real.sin_add_pi_div_two _
  /-
    🎉 no goals
  -/


theorem sin_sub_pi_div_two (θ : Angle) : sin (θ - ↑(π / 2)) = -cos θ := by
  /-
    θ : Real.Angle
    ⊢ Eq (HSub.hSub θ ↑(HDiv.hDiv Real.pi 2)).sin (Neg.neg θ.cos)
  -/
  induction θ using Real.Angle.induction_on
  /-
    case h
    x✝ : Real
    ⊢ Eq (HSub.hSub ↑x✝ ↑(HDiv.hDiv Real.pi 2)).sin (Neg.neg (↑x✝).cos)
  -/
  exact Real.sin_sub_pi_div_two _
  /-
    🎉 no goals
  -/


theorem sin_pi_div_two_sub (θ : Angle) : sin (↑(π / 2) - θ) = cos θ := by
  /-
    θ : Real.Angle
    ⊢ Eq (HSub.hSub (↑(HDiv.hDiv Real.pi 2)) θ).sin θ.cos
  -/
  induction θ using Real.Angle.induction_on
  /-
    case h
    x✝ : Real
    ⊢ Eq (HSub.hSub ↑(HDiv.hDiv Real.pi 2) ↑x✝).sin (↑x✝).cos
  -/
  exact Real.sin_pi_div_two_sub _
  /-
    🎉 no goals
  -/


theorem cos_add_pi_div_two (θ : Angle) : cos (θ + ↑(π / 2)) = -sin θ := by
  /-
    θ : Real.Angle
    ⊢ Eq (HAdd.hAdd θ ↑(HDiv.hDiv Real.pi 2)).cos (Neg.neg θ.sin)
  -/
  induction θ using Real.Angle.induction_on
  /-
    case h
    x✝ : Real
    ⊢ Eq (HAdd.hAdd ↑x✝ ↑(HDiv.hDiv Real.pi 2)).cos (Neg.neg (↑x✝).sin)
  -/
  exact Real.cos_add_pi_div_two _
  /-
    🎉 no goals
  -/


theorem cos_sub_pi_div_two (θ : Angle) : cos (θ - ↑(π / 2)) = sin θ := by
  /-
    θ : Real.Angle
    ⊢ Eq (HSub.hSub θ ↑(HDiv.hDiv Real.pi 2)).cos θ.sin
  -/
  induction θ using Real.Angle.induction_on
  /-
    case h
    x✝ : Real
    ⊢ Eq (HSub.hSub ↑x✝ ↑(HDiv.hDiv Real.pi 2)).cos (↑x✝).sin
  -/
  exact Real.cos_sub_pi_div_two _
  /-
    🎉 no goals
  -/


theorem cos_pi_div_two_sub (θ : Angle) : cos (↑(π / 2) - θ) = sin θ := by
  /-
    θ : Real.Angle
    ⊢ Eq (HSub.hSub (↑(HDiv.hDiv Real.pi 2)) θ).cos θ.sin
  -/
  induction θ using Real.Angle.induction_on
  /-
    case h
    x✝ : Real
    ⊢ Eq (HSub.hSub ↑(HDiv.hDiv Real.pi 2) ↑x✝).cos (↑x✝).sin
  -/
  exact Real.cos_pi_div_two_sub _
  /-
    🎉 no goals
  -/


theorem abs_sin_eq_of_two_nsmul_eq {θ ψ : Angle} (h : (2 : ℕ) • θ = (2 : ℕ) • ψ) :
    |sin θ| = |sin ψ| := by
  /-
    θ ψ : Real.Angle
    h : Eq (HSMul.hSMul 2 θ) (HSMul.hSMul 2 ψ)
    ⊢ Eq (abs θ.sin) (abs ψ.sin)
  -/
  rw [two_nsmul_eq_iff] at h
  /-
    θ ψ : Real.Angle
    h : Or (Eq θ ψ) (Eq θ (HAdd.hAdd ψ ↑Real.pi))
    ⊢ Eq (abs θ.sin) (abs ψ.sin)
  -/
  rcases h with (rfl | rfl)
    /-
      case inl
      θ : Real.Angle
      ⊢ Eq (abs θ.sin) (abs θ.sin)
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case inr
      ψ : Real.Angle
      ⊢ Eq (abs (HAdd.hAdd ψ ↑Real.pi).sin) (abs ψ.sin)
    -/
  · rw [sin_add_pi, abs_neg]
    /-
      🎉 no goals
    -/


theorem abs_sin_eq_of_two_zsmul_eq {θ ψ : Angle} (h : (2 : ℤ) • θ = (2 : ℤ) • ψ) :
    |sin θ| = |sin ψ| := by
  /-
    θ ψ : Real.Angle
    h : Eq (HSMul.hSMul 2 θ) (HSMul.hSMul 2 ψ)
    ⊢ Eq (abs θ.sin) (abs ψ.sin)
  -/
  simp_rw [two_zsmul, ← two_nsmul] at h
  /-
    θ ψ : Real.Angle
    h : Eq (HSMul.hSMul 2 θ) (HSMul.hSMul 2 ψ)
    ⊢ Eq (abs θ.sin) (abs ψ.sin)
  -/
  exact abs_sin_eq_of_two_nsmul_eq h
  /-
    🎉 no goals
  -/


theorem abs_cos_eq_of_two_nsmul_eq {θ ψ : Angle} (h : (2 : ℕ) • θ = (2 : ℕ) • ψ) :
    |cos θ| = |cos ψ| := by
  /-
    θ ψ : Real.Angle
    h : Eq (HSMul.hSMul 2 θ) (HSMul.hSMul 2 ψ)
    ⊢ Eq (abs θ.cos) (abs ψ.cos)
  -/
  rw [two_nsmul_eq_iff] at h
  /-
    θ ψ : Real.Angle
    h : Or (Eq θ ψ) (Eq θ (HAdd.hAdd ψ ↑Real.pi))
    ⊢ Eq (abs θ.cos) (abs ψ.cos)
  -/
  rcases h with (rfl | rfl)
    /-
      case inl
      θ : Real.Angle
      ⊢ Eq (abs θ.cos) (abs θ.cos)
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case inr
      ψ : Real.Angle
      ⊢ Eq (abs (HAdd.hAdd ψ ↑Real.pi).cos) (abs ψ.cos)
    -/
  · rw [cos_add_pi, abs_neg]
    /-
      🎉 no goals
    -/


theorem abs_cos_eq_of_two_zsmul_eq {θ ψ : Angle} (h : (2 : ℤ) • θ = (2 : ℤ) • ψ) :
    |cos θ| = |cos ψ| := by
  /-
    θ ψ : Real.Angle
    h : Eq (HSMul.hSMul 2 θ) (HSMul.hSMul 2 ψ)
    ⊢ Eq (abs θ.cos) (abs ψ.cos)
  -/
  simp_rw [two_zsmul, ← two_nsmul] at h
  /-
    θ ψ : Real.Angle
    h : Eq (HSMul.hSMul 2 θ) (HSMul.hSMul 2 ψ)
    ⊢ Eq (abs θ.cos) (abs ψ.cos)
  -/
  exact abs_cos_eq_of_two_nsmul_eq h
  /-
    🎉 no goals
  -/


@[simp]
theorem coe_toIcoMod (θ ψ : ℝ) : ↑(toIcoMod two_pi_pos ψ θ) = (θ : Angle) := by
  /-
    θ ψ : Real
    ⊢ Eq ↑(toIcoMod Real.two_pi_pos ψ θ) ↑θ
  -/
  rw [angle_eq_iff_two_pi_dvd_sub]
  /-
    θ ψ : Real
    ⊢ Exists fun k => Eq (HSub.hSub (toIcoMod Real.two_pi_pos ψ θ) θ) (HMul.hMul ( …
  -/
  refine ⟨-toIcoDiv two_pi_pos ψ θ, ?_⟩
  /-
    θ ψ : Real
    ⊢ Eq (HSub.hSub (toIcoMod Real.two_pi_pos ψ θ) θ) (HMul.hMul (HMul.hMul 2 Real …
  -/
  rw [toIcoMod_sub_self, zsmul_eq_mul, mul_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem coe_toIocMod (θ ψ : ℝ) : ↑(toIocMod two_pi_pos ψ θ) = (θ : Angle) := by
  /-
    θ ψ : Real
    ⊢ Eq ↑(toIocMod Real.two_pi_pos ψ θ) ↑θ
  -/
  rw [angle_eq_iff_two_pi_dvd_sub]
  /-
    θ ψ : Real
    ⊢ Exists fun k => Eq (HSub.hSub (toIocMod Real.two_pi_pos ψ θ) θ) (HMul.hMul ( …
  -/
  refine ⟨-toIocDiv two_pi_pos ψ θ, ?_⟩
  /-
    θ ψ : Real
    ⊢ Eq (HSub.hSub (toIocMod Real.two_pi_pos ψ θ) θ) (HMul.hMul (HMul.hMul 2 Real …
  -/
  rw [toIocMod_sub_self, zsmul_eq_mul, mul_comm]
  /-
    🎉 no goals
  -/


/-- Convert a `Real.Angle` to a real number in the interval `Ioc (-π) π`. -/
def toReal (θ : Angle) : ℝ :=
  (toIocMod_periodic two_pi_pos (-π)).lift θ


theorem toReal_coe (θ : ℝ) : (θ : Angle).toReal = toIocMod two_pi_pos (-π) θ :=
  rfl


theorem toReal_coe_eq_self_iff {θ : ℝ} : (θ : Angle).toReal = θ ↔ -π < θ ∧ θ ≤ π := by
  /-
    θ : Real
    ⊢ Iff (Eq (↑θ).toReal θ) (And (LT.lt (Neg.neg Real.pi) θ) (LE.le θ Real.pi))
  -/
  rw [toReal_coe, toIocMod_eq_self two_pi_pos]
  /-
    θ : Real
    ⊢ Iff (Membership.mem (Set.Ioc (Neg.neg Real.pi) (HAdd.hAdd (Neg.neg Real.pi)  …
  -/
  ring_nf
  /-
    θ : Real
    ⊢ Iff (Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) θ) (And (LT.lt (Neg. …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem toReal_coe_eq_self_iff_mem_Ioc {θ : ℝ} : (θ : Angle).toReal = θ ↔ θ ∈ Set.Ioc (-π) π := by
  /-
    θ : Real
    ⊢ Iff (Eq (↑θ).toReal θ) (Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) θ)
  -/
  rw [toReal_coe_eq_self_iff, ← Set.mem_Ioc]
  /-
    🎉 no goals
  -/


theorem toReal_injective : Function.Injective toReal := by
  /-
    ⊢ Function.Injective Real.Angle.toReal
  -/
  intro θ ψ h
  /-
    θ ψ : Real.Angle
    h : Eq θ.toReal ψ.toReal
    ⊢ Eq θ ψ
  -/
  induction θ using Real.Angle.induction_on
  /-
    case h
    ψ : Real.Angle
    x✝ : Real
    h : Eq (↑x✝).toReal ψ.toReal
    ⊢ Eq (↑x✝) ψ
  -/
  induction ψ using Real.Angle.induction_on
  simpa [toReal_coe, toIocMod_eq_toIocMod, zsmul_eq_mul, mul_comm _ (2 * π), ←
    angle_eq_iff_two_pi_dvd_sub, eq_comm] using h


@[simp]
theorem toReal_inj {θ ψ : Angle} : θ.toReal = ψ.toReal ↔ θ = ψ :=
  toReal_injective.eq_iff


@[simp]
theorem coe_toReal (θ : Angle) : (θ.toReal : Angle) = θ := by
  /-
    θ : Real.Angle
    ⊢ Eq (↑θ.toReal) θ
  -/
  induction θ using Real.Angle.induction_on
  /-
    case h
    x✝ : Real
    ⊢ Eq ↑(↑x✝).toReal ↑x✝
  -/
  exact coe_toIocMod _ _
  /-
    🎉 no goals
  -/


theorem neg_pi_lt_toReal (θ : Angle) : -π < θ.toReal := by
  /-
    θ : Real.Angle
    ⊢ LT.lt (Neg.neg Real.pi) θ.toReal
  -/
  induction θ using Real.Angle.induction_on
  /-
    case h
    x✝ : Real
    ⊢ LT.lt (Neg.neg Real.pi) (↑x✝).toReal
  -/
  exact left_lt_toIocMod _ _ _
  /-
    🎉 no goals
  -/


theorem toReal_le_pi (θ : Angle) : θ.toReal ≤ π := by
  /-
    θ : Real.Angle
    ⊢ LE.le θ.toReal Real.pi
  -/
  induction θ using Real.Angle.induction_on
  /-
    case h
    x✝ : Real
    ⊢ LE.le (↑x✝).toReal Real.pi
  -/
  convert toIocMod_le_right two_pi_pos _ _
  /-
    case h.e'_4
    x✝ : Real
    ⊢ Eq Real.pi (HAdd.hAdd (Neg.neg Real.pi) (HMul.hMul 2 Real.pi))
  -/
  ring
  /-
    🎉 no goals
  -/


theorem abs_toReal_le_pi (θ : Angle) : |θ.toReal| ≤ π :=
  abs_le.2 ⟨(neg_pi_lt_toReal _).le, toReal_le_pi _⟩


theorem toReal_mem_Ioc (θ : Angle) : θ.toReal ∈ Set.Ioc (-π) π :=
  ⟨neg_pi_lt_toReal _, toReal_le_pi _⟩


@[simp]
theorem toIocMod_toReal (θ : Angle) : toIocMod two_pi_pos (-π) θ.toReal = θ.toReal := by
  /-
    θ : Real.Angle
    ⊢ Eq (toIocMod Real.two_pi_pos (Neg.neg Real.pi) θ.toReal) θ.toReal
  -/
  induction θ using Real.Angle.induction_on
  /-
    case h
    x✝ : Real
    ⊢ Eq (toIocMod Real.two_pi_pos (Neg.neg Real.pi) (↑x✝).toReal) (↑x✝).toReal
  -/
  rw [toReal_coe]
  /-
    case h
    x✝ : Real
    ⊢ Eq (toIocMod Real.two_pi_pos (Neg.neg Real.pi) (toIocMod Real.two_pi_pos (Ne …
  -/
  exact toIocMod_toIocMod _ _ _ _
  /-
    🎉 no goals
  -/


@[simp]
theorem toReal_zero : (0 : Angle).toReal = 0 := by
  /-
    ⊢ Eq (Real.Angle.toReal 0) 0
  -/
  rw [← coe_zero, toReal_coe_eq_self_iff]
  /-
    ⊢ And (LT.lt (Neg.neg Real.pi) 0) (LE.le 0 Real.pi)
  -/
  exact ⟨Left.neg_neg_iff.2 Real.pi_pos, Real.pi_pos.le⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem toReal_eq_zero_iff {θ : Angle} : θ.toReal = 0 ↔ θ = 0 := by
  /-
    θ : Real.Angle
    ⊢ Iff (Eq θ.toReal 0) (Eq θ 0)
  -/
  nth_rw 1 [← toReal_zero]
  /-
    θ : Real.Angle
    ⊢ Iff (Eq θ.toReal (Real.Angle.toReal 0)) (Eq θ 0)
  -/
  exact toReal_inj
  /-
    🎉 no goals
  -/


@[simp]
theorem toReal_pi : (π : Angle).toReal = π := by
  /-
    ⊢ Eq (↑Real.pi).toReal Real.pi
  -/
  rw [toReal_coe_eq_self_iff]
  /-
    ⊢ And (LT.lt (Neg.neg Real.pi) Real.pi) (LE.le Real.pi Real.pi)
  -/
  exact ⟨Left.neg_lt_self Real.pi_pos, le_refl _⟩
  /-
    🎉 no goals
  -/


@[simp]
                                                                  /-
                                                                    θ : Real.Angle
                                                                    ⊢ Iff (Eq θ.toReal Real.pi) (Eq θ ↑Real.pi)
                                                                  -/
theorem toReal_eq_pi_iff {θ : Angle} : θ.toReal = π ↔ θ = π := by rw [← toReal_inj, toReal_pi]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem pi_ne_zero : (π : Angle) ≠ 0 := by
  /-
    ⊢ Ne (↑Real.pi) 0
  -/
  rw [← toReal_injective.ne_iff, toReal_pi, toReal_zero]
  /-
    ⊢ Ne Real.pi 0
  -/
  exact Real.pi_ne_zero
  /-
    🎉 no goals
  -/


@[simp]
theorem toReal_pi_div_two : ((π / 2 : ℝ) : Angle).toReal = π / 2 :=
                                 /-
                                   ⊢ And (LT.lt (Neg.neg Real.pi) (HDiv.hDiv Real.pi 2)) (LE.le (HDiv.hDiv Real.p …
                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
  toReal_coe_eq_self_iff.2 <| by constructor <;> linarith [pi_pos]
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp]
theorem toReal_eq_pi_div_two_iff {θ : Angle} : θ.toReal = π / 2 ↔ θ = (π / 2 : ℝ) := by
  /-
    θ : Real.Angle
    ⊢ Iff (Eq θ.toReal (HDiv.hDiv Real.pi 2)) (Eq θ ↑(HDiv.hDiv Real.pi 2))
  -/
  rw [← toReal_inj, toReal_pi_div_two]
  /-
    🎉 no goals
  -/


@[simp]
theorem toReal_neg_pi_div_two : ((-π / 2 : ℝ) : Angle).toReal = -π / 2 :=
                                 /-
                                   ⊢ And (LT.lt (Neg.neg Real.pi) (HDiv.hDiv (Neg.neg Real.pi) 2)) (LE.le (HDiv.h …
                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
  toReal_coe_eq_self_iff.2 <| by constructor <;> linarith [pi_pos]
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp]
theorem toReal_eq_neg_pi_div_two_iff {θ : Angle} : θ.toReal = -π / 2 ↔ θ = (-π / 2 : ℝ) := by
  /-
    θ : Real.Angle
    ⊢ Iff (Eq θ.toReal (HDiv.hDiv (Neg.neg Real.pi) 2)) (Eq θ ↑(HDiv.hDiv (Neg.neg …
  -/
  rw [← toReal_inj, toReal_neg_pi_div_two]
  /-
    🎉 no goals
  -/


theorem pi_div_two_ne_zero : ((π / 2 : ℝ) : Angle) ≠ 0 := by
  /-
    ⊢ Ne (↑(HDiv.hDiv Real.pi 2)) 0
  -/
  rw [← toReal_injective.ne_iff, toReal_pi_div_two, toReal_zero]
  /-
    ⊢ Ne (HDiv.hDiv Real.pi 2) 0
  -/
  exact div_ne_zero Real.pi_ne_zero two_ne_zero
  /-
    🎉 no goals
  -/


theorem neg_pi_div_two_ne_zero : ((-π / 2 : ℝ) : Angle) ≠ 0 := by
  /-
    ⊢ Ne (↑(HDiv.hDiv (Neg.neg Real.pi) 2)) 0
  -/
  rw [← toReal_injective.ne_iff, toReal_neg_pi_div_two, toReal_zero]
  /-
    ⊢ Ne (HDiv.hDiv (Neg.neg Real.pi) 2) 0
  -/
  exact div_ne_zero (neg_ne_zero.2 Real.pi_ne_zero) two_ne_zero
  /-
    🎉 no goals
  -/


theorem abs_toReal_coe_eq_self_iff {θ : ℝ} : |(θ : Angle).toReal| = θ ↔ 0 ≤ θ ∧ θ ≤ π :=
  ⟨fun h => h ▸ ⟨abs_nonneg _, abs_toReal_le_pi _⟩, fun h =>
    (toReal_coe_eq_self_iff.2 ⟨(Left.neg_neg_iff.2 Real.pi_pos).trans_le h.1, h.2⟩).symm ▸
      abs_eq_self.2 h.1⟩


theorem abs_toReal_neg_coe_eq_self_iff {θ : ℝ} : |(-θ : Angle).toReal| = θ ↔ 0 ≤ θ ∧ θ ≤ π := by
  /-
    θ : Real
    ⊢ Iff (Eq (abs (Neg.neg ↑θ).toReal) θ) (And (LE.le 0 θ) (LE.le θ Real.pi))
  -/
  refine ⟨fun h => h ▸ ⟨abs_nonneg _, abs_toReal_le_pi _⟩, fun h => ?_⟩
  /-
    θ : Real
    h : And (LE.le 0 θ) (LE.le θ Real.pi)
    ⊢ Eq (abs (Neg.neg ↑θ).toReal) θ
  -/
  by_cases hnegpi : θ = π; · simp [hnegpi, Real.pi_pos.le]
                             /-
                               🎉 no goals
                             -/
  rw [← coe_neg,
    toReal_coe_eq_self_iff.2
      ⟨neg_lt_neg (lt_of_le_of_ne h.2 hnegpi), (neg_nonpos.2 h.1).trans Real.pi_pos.le⟩,
    abs_neg, abs_eq_self.2 h.1]


theorem abs_toReal_eq_pi_div_two_iff {θ : Angle} :
    |θ.toReal| = π / 2 ↔ θ = (π / 2 : ℝ) ∨ θ = (-π / 2 : ℝ) := by
  rw [abs_eq (div_nonneg Real.pi_pos.le two_pos.le), ← neg_div, toReal_eq_pi_div_two_iff,
    toReal_eq_neg_pi_div_two_iff]


theorem nsmul_toReal_eq_mul {n : ℕ} (h : n ≠ 0) {θ : Angle} :
    (n • θ).toReal = n * θ.toReal ↔ θ.toReal ∈ Set.Ioc (-π / n) (π / n) := by
  /-
    n : Nat
    h : Ne n 0
    θ : Real.Angle
    ⊢ Iff (Eq (HSMul.hSMul n θ).toReal (HMul.hMul (↑n) θ.toReal)) (Membership.mem  …
  -/
  nth_rw 1 [← coe_toReal θ]
  /-
    n : Nat
    h : Ne n 0
    θ : Real.Angle
    ⊢ Iff (Eq (HSMul.hSMul n ↑θ.toReal).toReal (HMul.hMul (↑n) θ.toReal)) (Members …
  -/
  have h' : 0 < (n : ℝ) := mod_cast Nat.pos_of_ne_zero h
  rw [← coe_nsmul, nsmul_eq_mul, toReal_coe_eq_self_iff, Set.mem_Ioc, div_lt_iff₀' h',
    le_div_iff₀' h']


theorem two_nsmul_toReal_eq_two_mul {θ : Angle} :
    ((2 : ℕ) • θ).toReal = 2 * θ.toReal ↔ θ.toReal ∈ Set.Ioc (-π / 2) (π / 2) :=
  mod_cast nsmul_toReal_eq_mul two_ne_zero


theorem two_zsmul_toReal_eq_two_mul {θ : Angle} :
    ((2 : ℤ) • θ).toReal = 2 * θ.toReal ↔ θ.toReal ∈ Set.Ioc (-π / 2) (π / 2) := by
  /-
    θ : Real.Angle
    ⊢ Iff (Eq (HSMul.hSMul 2 θ).toReal (HMul.hMul 2 θ.toReal)) (Membership.mem (Se …
  -/
  rw [two_zsmul, ← two_nsmul, two_nsmul_toReal_eq_two_mul]
  /-
    🎉 no goals
  -/


theorem toReal_coe_eq_self_sub_two_mul_int_mul_pi_iff {θ : ℝ} {k : ℤ} :
    (θ : Angle).toReal = θ - 2 * k * π ↔ θ ∈ Set.Ioc ((2 * k - 1 : ℝ) * π) ((2 * k + 1) * π) := by
  rw [← sub_zero (θ : Angle), ← zsmul_zero k, ← coe_two_pi, ← coe_zsmul, ← coe_sub, zsmul_eq_mul, ←
    mul_assoc, mul_comm (k : ℝ), toReal_coe_eq_self_iff, Set.mem_Ioc]
  /-
    θ : Real
    k : Int
    ⊢ Iff (And (LT.lt (Neg.neg Real.pi) (HSub.hSub θ (HMul.hMul (HMul.hMul 2 ↑k) R …
  -/
  exact ⟨fun h => ⟨by linarith, by linarith⟩, fun h => ⟨by linarith, by linarith⟩⟩
  /-
    🎉 no goals
  -/


theorem toReal_coe_eq_self_sub_two_pi_iff {θ : ℝ} :
    (θ : Angle).toReal = θ - 2 * π ↔ θ ∈ Set.Ioc π (3 * π) := by
  /-
    θ : Real
    ⊢ Iff (Eq (↑θ).toReal (HSub.hSub θ (HMul.hMul 2 Real.pi))) (Membership.mem (Se …
  -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
  convert @toReal_coe_eq_self_sub_two_mul_int_mul_pi_iff θ 1 <;> norm_num
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem toReal_coe_eq_self_add_two_pi_iff {θ : ℝ} :
    (θ : Angle).toReal = θ + 2 * π ↔ θ ∈ Set.Ioc (-3 * π) (-π) := by
  /-
    θ : Real
    ⊢ Iff (Eq (↑θ).toReal (HAdd.hAdd θ (HMul.hMul 2 Real.pi))) (Membership.mem (Se …
  -/
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
  convert @toReal_coe_eq_self_sub_two_mul_int_mul_pi_iff θ (-1) using 2 <;> norm_num
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


theorem two_nsmul_toReal_eq_two_mul_sub_two_pi {θ : Angle} :
    ((2 : ℕ) • θ).toReal = 2 * θ.toReal - 2 * π ↔ π / 2 < θ.toReal := by
  /-
    θ : Real.Angle
    ⊢ Iff (Eq (HSMul.hSMul 2 θ).toReal (HSub.hSub (HMul.hMul 2 θ.toReal) (HMul.hMu …
  -/
  nth_rw 1 [← coe_toReal θ]
  /-
    θ : Real.Angle
    ⊢ Iff (Eq (HSMul.hSMul 2 ↑θ.toReal).toReal (HSub.hSub (HMul.hMul 2 θ.toReal) ( …
  -/
  rw [← coe_nsmul, two_nsmul, ← two_mul, toReal_coe_eq_self_sub_two_pi_iff, Set.mem_Ioc]
  exact
    ⟨fun h => by linarith, fun h =>
      ⟨(div_lt_iff₀' (zero_lt_two' ℝ)).1 h, by linarith [pi_pos, toReal_le_pi θ]⟩⟩


theorem two_zsmul_toReal_eq_two_mul_sub_two_pi {θ : Angle} :
    ((2 : ℤ) • θ).toReal = 2 * θ.toReal - 2 * π ↔ π / 2 < θ.toReal := by
  /-
    θ : Real.Angle
    ⊢ Iff (Eq (HSMul.hSMul 2 θ).toReal (HSub.hSub (HMul.hMul 2 θ.toReal) (HMul.hMu …
  -/
  rw [two_zsmul, ← two_nsmul, two_nsmul_toReal_eq_two_mul_sub_two_pi]
  /-
    🎉 no goals
  -/


theorem two_nsmul_toReal_eq_two_mul_add_two_pi {θ : Angle} :
    ((2 : ℕ) • θ).toReal = 2 * θ.toReal + 2 * π ↔ θ.toReal ≤ -π / 2 := by
  /-
    θ : Real.Angle
    ⊢ Iff (Eq (HSMul.hSMul 2 θ).toReal (HAdd.hAdd (HMul.hMul 2 θ.toReal) (HMul.hMu …
  -/
  nth_rw 1 [← coe_toReal θ]
  /-
    θ : Real.Angle
    ⊢ Iff (Eq (HSMul.hSMul 2 ↑θ.toReal).toReal (HAdd.hAdd (HMul.hMul 2 θ.toReal) ( …
  -/
  rw [← coe_nsmul, two_nsmul, ← two_mul, toReal_coe_eq_self_add_two_pi_iff, Set.mem_Ioc]
  refine
    ⟨fun h => by linarith, fun h =>
      ⟨by linarith [pi_pos, neg_pi_lt_toReal θ], (le_div_iff₀' (zero_lt_two' ℝ)).1 h⟩⟩


theorem two_zsmul_toReal_eq_two_mul_add_two_pi {θ : Angle} :
    ((2 : ℤ) • θ).toReal = 2 * θ.toReal + 2 * π ↔ θ.toReal ≤ -π / 2 := by
  /-
    θ : Real.Angle
    ⊢ Iff (Eq (HSMul.hSMul 2 θ).toReal (HAdd.hAdd (HMul.hMul 2 θ.toReal) (HMul.hMu …
  -/
  rw [two_zsmul, ← two_nsmul, two_nsmul_toReal_eq_two_mul_add_two_pi]
  /-
    🎉 no goals
  -/


@[simp]
theorem sin_toReal (θ : Angle) : Real.sin θ.toReal = sin θ := by
  /-
    θ : Real.Angle
    ⊢ Eq (Real.sin θ.toReal) θ.sin
  -/
  conv_rhs => rw [← coe_toReal θ, sin_coe]
  /-
    🎉 no goals
  -/


@[simp]
theorem cos_toReal (θ : Angle) : Real.cos θ.toReal = cos θ := by
  /-
    θ : Real.Angle
    ⊢ Eq (Real.cos θ.toReal) θ.cos
  -/
  conv_rhs => rw [← coe_toReal θ, cos_coe]
  /-
    🎉 no goals
  -/


theorem cos_nonneg_iff_abs_toReal_le_pi_div_two {θ : Angle} : 0 ≤ cos θ ↔ |θ.toReal| ≤ π / 2 := by
  /-
    θ : Real.Angle
    ⊢ Iff (LE.le 0 θ.cos) (LE.le (abs θ.toReal) (HDiv.hDiv Real.pi 2))
  -/
  nth_rw 1 [← coe_toReal θ]
  /-
    θ : Real.Angle
    ⊢ Iff (LE.le 0 (↑θ.toReal).cos) (LE.le (abs θ.toReal) (HDiv.hDiv Real.pi 2))
  -/
  rw [abs_le, cos_coe]
  /-
    θ : Real.Angle
    ⊢ Iff (LE.le 0 (Real.cos θ.toReal)) (And (LE.le (Neg.neg (HDiv.hDiv Real.pi 2) …
  -/
  refine ⟨fun h => ?_, cos_nonneg_of_mem_Icc⟩
  /-
    θ : Real.Angle
    h : LE.le 0 (Real.cos θ.toReal)
    ⊢ And (LE.le (Neg.neg (HDiv.hDiv Real.pi 2)) θ.toReal) (LE.le θ.toReal (HDiv.h …
  -/
  by_contra hn
  /-
    θ : Real.Angle
    h : LE.le 0 (Real.cos θ.toReal)
    hn : Not (And (LE.le (Neg.neg (HDiv.hDiv Real.pi 2)) θ.toReal) (LE.le θ.toReal …
    ⊢ False
  -/
  rw [not_and_or, not_le, not_le] at hn
  /-
    θ : Real.Angle
    h : LE.le 0 (Real.cos θ.toReal)
    hn : Or (LT.lt θ.toReal (Neg.neg (HDiv.hDiv Real.pi 2))) (LT.lt (HDiv.hDiv Rea …
    ⊢ False
  -/
  refine (not_lt.2 h) ?_
  /-
    θ : Real.Angle
    h : LE.le 0 (Real.cos θ.toReal)
    hn : Or (LT.lt θ.toReal (Neg.neg (HDiv.hDiv Real.pi 2))) (LT.lt (HDiv.hDiv Rea …
    ⊢ LT.lt (Real.cos θ.toReal) 0
  -/
  rcases hn with (hn | hn)
    /-
      case inl
      θ : Real.Angle
      h : LE.le 0 (Real.cos θ.toReal)
      hn : LT.lt θ.toReal (Neg.neg (HDiv.hDiv Real.pi 2))
      ⊢ LT.lt (Real.cos θ.toReal) 0
    -/
  · rw [← Real.cos_neg]
    /-
      case inl
      θ : Real.Angle
      h : LE.le 0 (Real.cos θ.toReal)
      hn : LT.lt θ.toReal (Neg.neg (HDiv.hDiv Real.pi 2))
      ⊢ LT.lt (Real.cos (Neg.neg θ.toReal)) 0
    -/
    refine cos_neg_of_pi_div_two_lt_of_lt (by linarith) ?_
    /-
      case inl
      θ : Real.Angle
      h : LE.le 0 (Real.cos θ.toReal)
      hn : LT.lt θ.toReal (Neg.neg (HDiv.hDiv Real.pi 2))
      ⊢ LT.lt (Neg.neg θ.toReal) (HAdd.hAdd Real.pi (HDiv.hDiv Real.pi 2))
    -/
    linarith [neg_pi_lt_toReal θ]
    /-
      🎉 no goals
    -/
    /-
      case inr
      θ : Real.Angle
      h : LE.le 0 (Real.cos θ.toReal)
      hn : LT.lt (HDiv.hDiv Real.pi 2) θ.toReal
      ⊢ LT.lt (Real.cos θ.toReal) 0
    -/
  · refine cos_neg_of_pi_div_two_lt_of_lt hn ?_
    /-
      case inr
      θ : Real.Angle
      h : LE.le 0 (Real.cos θ.toReal)
      hn : LT.lt (HDiv.hDiv Real.pi 2) θ.toReal
      ⊢ LT.lt θ.toReal (HAdd.hAdd Real.pi (HDiv.hDiv Real.pi 2))
    -/
    linarith [toReal_le_pi θ]
    /-
      🎉 no goals
    -/


theorem cos_pos_iff_abs_toReal_lt_pi_div_two {θ : Angle} : 0 < cos θ ↔ |θ.toReal| < π / 2 := by
  rw [lt_iff_le_and_ne, lt_iff_le_and_ne, cos_nonneg_iff_abs_toReal_le_pi_div_two, ←
    and_congr_right]
  /-
    θ : Real.Angle
    ⊢ LE.le (abs θ.toReal) (HDiv.hDiv Real.pi 2) → Iff (Ne (abs θ.toReal) (HDiv.hD …
  -/
  rintro -
  /-
    θ : Real.Angle
    ⊢ Iff (Ne (abs θ.toReal) (HDiv.hDiv Real.pi 2)) (Ne 0 θ.cos)
  -/
  rw [Ne, Ne, not_iff_not, @eq_comm ℝ 0, abs_toReal_eq_pi_div_two_iff, cos_eq_zero_iff]
  /-
    🎉 no goals
  -/


theorem cos_neg_iff_pi_div_two_lt_abs_toReal {θ : Angle} : cos θ < 0 ↔ π / 2 < |θ.toReal| := by
  /-
    θ : Real.Angle
    ⊢ Iff (LT.lt θ.cos 0) (LT.lt (HDiv.hDiv Real.pi 2) (abs θ.toReal))
  -/
  rw [← not_le, ← not_le, not_iff_not, cos_nonneg_iff_abs_toReal_le_pi_div_two]
  /-
    🎉 no goals
  -/


theorem abs_cos_eq_abs_sin_of_two_nsmul_add_two_nsmul_eq_pi {θ ψ : Angle}
    (h : (2 : ℕ) • θ + (2 : ℕ) • ψ = π) : |cos θ| = |sin ψ| := by
  /-
    θ ψ : Real.Angle
    h : Eq (HAdd.hAdd (HSMul.hSMul 2 θ) (HSMul.hSMul 2 ψ)) ↑Real.pi
    ⊢ Eq (abs θ.cos) (abs ψ.sin)
  -/
  rw [← eq_sub_iff_add_eq, ← two_nsmul_coe_div_two, ← nsmul_sub, two_nsmul_eq_iff] at h
  /-
    θ ψ : Real.Angle
    h : Or (Eq θ (HSub.hSub (↑(HDiv.hDiv Real.pi 2)) ψ)) (Eq θ (HAdd.hAdd (HSub.hS …
    ⊢ Eq (abs θ.cos) (abs ψ.sin)
  -/
                                /-
                                  🎉 no goals
                                -/
  rcases h with (rfl | rfl) <;> simp [cos_pi_div_two_sub]
                                /-
                                  🎉 no goals
                                -/


theorem abs_cos_eq_abs_sin_of_two_zsmul_add_two_zsmul_eq_pi {θ ψ : Angle}
    (h : (2 : ℤ) • θ + (2 : ℤ) • ψ = π) : |cos θ| = |sin ψ| := by
  /-
    θ ψ : Real.Angle
    h : Eq (HAdd.hAdd (HSMul.hSMul 2 θ) (HSMul.hSMul 2 ψ)) ↑Real.pi
    ⊢ Eq (abs θ.cos) (abs ψ.sin)
  -/
  simp_rw [two_zsmul, ← two_nsmul] at h
  /-
    θ ψ : Real.Angle
    h : Eq (HAdd.hAdd (HSMul.hSMul 2 θ) (HSMul.hSMul 2 ψ)) ↑Real.pi
    ⊢ Eq (abs θ.cos) (abs ψ.sin)
  -/
  exact abs_cos_eq_abs_sin_of_two_nsmul_add_two_nsmul_eq_pi h
  /-
    🎉 no goals
  -/


/-- The tangent of a `Real.Angle`. -/
def tan (θ : Angle) : ℝ :=
  sin θ / cos θ


theorem tan_eq_sin_div_cos (θ : Angle) : tan θ = sin θ / cos θ :=
  rfl


@[simp]
theorem tan_coe (x : ℝ) : tan (x : Angle) = Real.tan x := by
  /-
    x : Real
    ⊢ Eq (↑x).tan (Real.tan x)
  -/
  rw [tan, sin_coe, cos_coe, Real.tan_eq_sin_div_cos]
  /-
    🎉 no goals
  -/


@[simp]
                                             /-
                                               ⊢ Eq (Real.Angle.tan 0) 0
                                             -/
theorem tan_zero : tan (0 : Angle) = 0 := by rw [← coe_zero, tan_coe, Real.tan_zero]
                                             /-
                                               🎉 no goals
                                             -/


                                               /-
                                                 ⊢ Eq (↑Real.pi).tan 0
                                               -/
theorem tan_coe_pi : tan (π : Angle) = 0 := by rw [tan_coe, Real.tan_pi]
                                               /-
                                                 🎉 no goals
                                               -/


theorem tan_periodic : Function.Periodic tan (π : Angle) := by
  /-
    ⊢ Function.Periodic Real.Angle.tan ↑Real.pi
  -/
  intro θ
  /-
    θ : Real.Angle
    ⊢ Eq (HAdd.hAdd θ ↑Real.pi).tan θ.tan
  -/
  induction θ using Real.Angle.induction_on
  /-
    case h
    x✝ : Real
    ⊢ Eq (HAdd.hAdd ↑x✝ ↑Real.pi).tan (↑x✝).tan
  -/
  rw [← coe_add, tan_coe, tan_coe]
  /-
    case h
    x✝ : Real
    ⊢ Eq (Real.tan (HAdd.hAdd x✝ Real.pi)) (Real.tan x✝)
  -/
  exact Real.tan_periodic _
  /-
    🎉 no goals
  -/


@[simp]
theorem tan_add_pi (θ : Angle) : tan (θ + π) = tan θ :=
  tan_periodic θ


@[simp]
theorem tan_sub_pi (θ : Angle) : tan (θ - π) = tan θ :=
  tan_periodic.sub_eq θ


@[simp]
theorem tan_toReal (θ : Angle) : Real.tan θ.toReal = tan θ := by
  /-
    θ : Real.Angle
    ⊢ Eq (Real.tan θ.toReal) θ.tan
  -/
  conv_rhs => rw [← coe_toReal θ, tan_coe]
  /-
    🎉 no goals
  -/


theorem tan_eq_of_two_nsmul_eq {θ ψ : Angle} (h : (2 : ℕ) • θ = (2 : ℕ) • ψ) : tan θ = tan ψ := by
  /-
    θ ψ : Real.Angle
    h : Eq (HSMul.hSMul 2 θ) (HSMul.hSMul 2 ψ)
    ⊢ Eq θ.tan ψ.tan
  -/
  rw [two_nsmul_eq_iff] at h
  /-
    θ ψ : Real.Angle
    h : Or (Eq θ ψ) (Eq θ (HAdd.hAdd ψ ↑Real.pi))
    ⊢ Eq θ.tan ψ.tan
  -/
  rcases h with (rfl | rfl)
    /-
      case inl
      θ : Real.Angle
      ⊢ Eq θ.tan θ.tan
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case inr
      ψ : Real.Angle
      ⊢ Eq (HAdd.hAdd ψ ↑Real.pi).tan ψ.tan
    -/
  · exact tan_add_pi _
    /-
      🎉 no goals
    -/


theorem tan_eq_of_two_zsmul_eq {θ ψ : Angle} (h : (2 : ℤ) • θ = (2 : ℤ) • ψ) : tan θ = tan ψ := by
  /-
    θ ψ : Real.Angle
    h : Eq (HSMul.hSMul 2 θ) (HSMul.hSMul 2 ψ)
    ⊢ Eq θ.tan ψ.tan
  -/
  simp_rw [two_zsmul, ← two_nsmul] at h
  /-
    θ ψ : Real.Angle
    h : Eq (HSMul.hSMul 2 θ) (HSMul.hSMul 2 ψ)
    ⊢ Eq θ.tan ψ.tan
  -/
  exact tan_eq_of_two_nsmul_eq h
  /-
    🎉 no goals
  -/


theorem tan_eq_inv_of_two_nsmul_add_two_nsmul_eq_pi {θ ψ : Angle}
    (h : (2 : ℕ) • θ + (2 : ℕ) • ψ = π) : tan ψ = (tan θ)⁻¹ := by
  /-
    θ ψ : Real.Angle
    h : Eq (HAdd.hAdd (HSMul.hSMul 2 θ) (HSMul.hSMul 2 ψ)) ↑Real.pi
    ⊢ Eq ψ.tan (Inv.inv θ.tan)
  -/
  induction θ using Real.Angle.induction_on
  /-
    case h
    ψ : Real.Angle
    x✝ : Real
    h : Eq (HAdd.hAdd (HSMul.hSMul 2 ↑x✝) (HSMul.hSMul 2 ψ)) ↑Real.pi
    ⊢ Eq ψ.tan (Inv.inv (↑x✝).tan)
  -/
  induction ψ using Real.Angle.induction_on
  /-
    case h.h
    x✝¹ x✝ : Real
    h : Eq (HAdd.hAdd (HSMul.hSMul 2 ↑x✝¹) (HSMul.hSMul 2 ↑x✝)) ↑Real.pi
    ⊢ Eq (↑x✝).tan (Inv.inv (↑x✝¹).tan)
  -/
  rw [← smul_add, ← coe_add, ← coe_nsmul, two_nsmul, ← two_mul, angle_eq_iff_two_pi_dvd_sub] at h
  /-
    case h.h
    x✝¹ x✝ : Real
    h : Exists fun k => Eq (HSub.hSub (HMul.hMul 2 (HAdd.hAdd x✝¹ x✝)) Real.pi) (H …
    ⊢ Eq (↑x✝).tan (Inv.inv (↑x✝¹).tan)
  -/
  rcases h with ⟨k, h⟩
  rw [sub_eq_iff_eq_add, ← mul_inv_cancel_left₀ two_ne_zero π, mul_assoc, ← mul_add,
    mul_right_inj' (two_ne_zero' ℝ), ← eq_sub_iff_add_eq', mul_inv_cancel_left₀ two_ne_zero π,
    inv_mul_eq_div, mul_comm] at h
  /-
    case h.h.intro
    x✝¹ x✝ : Real
    k : Int
    h : Eq x✝ (HSub.hSub (HAdd.hAdd (HMul.hMul (↑k) Real.pi) (HDiv.hDiv Real.pi 2) …
    ⊢ Eq (↑x✝).tan (Inv.inv (↑x✝¹).tan)
  -/
  rw [tan_coe, tan_coe, ← tan_pi_div_two_sub, h, add_sub_assoc, add_comm]
  /-
    case h.h.intro
    x✝¹ x✝ : Real
    k : Int
    h : Eq x✝ (HSub.hSub (HAdd.hAdd (HMul.hMul (↑k) Real.pi) (HDiv.hDiv Real.pi 2) …
    ⊢ Eq (Real.tan (HAdd.hAdd (HSub.hSub (HDiv.hDiv Real.pi 2) x✝¹) (HMul.hMul (↑k …
  -/
  exact Real.tan_periodic.int_mul _ _
  /-
    🎉 no goals
  -/


theorem tan_eq_inv_of_two_zsmul_add_two_zsmul_eq_pi {θ ψ : Angle}
    (h : (2 : ℤ) • θ + (2 : ℤ) • ψ = π) : tan ψ = (tan θ)⁻¹ := by
  /-
    θ ψ : Real.Angle
    h : Eq (HAdd.hAdd (HSMul.hSMul 2 θ) (HSMul.hSMul 2 ψ)) ↑Real.pi
    ⊢ Eq ψ.tan (Inv.inv θ.tan)
  -/
  simp_rw [two_zsmul, ← two_nsmul] at h
  /-
    θ ψ : Real.Angle
    h : Eq (HAdd.hAdd (HSMul.hSMul 2 θ) (HSMul.hSMul 2 ψ)) ↑Real.pi
    ⊢ Eq ψ.tan (Inv.inv θ.tan)
  -/
  exact tan_eq_inv_of_two_nsmul_add_two_nsmul_eq_pi h
  /-
    🎉 no goals
  -/


/-- The sign of a `Real.Angle` is `0` if the angle is `0` or `π`, `1` if the angle is strictly
between `0` and `π` and `-1` is the angle is strictly between `-π` and `0`. It is defined as the
sign of the sine of the angle. -/
def sign (θ : Angle) : SignType :=
  SignType.sign (sin θ)


@[simp]
theorem sign_zero : (0 : Angle).sign = 0 := by
  /-
    ⊢ Eq (Real.Angle.sign 0) 0
  -/
  rw [sign, sin_zero, _root_.sign_zero]
  /-
    🎉 no goals
  -/


@[simp]
                                                 /-
                                                   ⊢ Eq (↑Real.pi).sign 0
                                                 -/
theorem sign_coe_pi : (π : Angle).sign = 0 := by rw [sign, sin_coe_pi, _root_.sign_zero]
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp]
theorem sign_neg (θ : Angle) : (-θ).sign = -θ.sign := by
  /-
    θ : Real.Angle
    ⊢ Eq (Neg.neg θ).sign (Neg.neg θ.sign)
  -/
  simp_rw [sign, sin_neg, Left.sign_neg]
  /-
    🎉 no goals
  -/


theorem sign_antiperiodic : Function.Antiperiodic sign (π : Angle) := fun θ => by
  /-
    θ : Real.Angle
    ⊢ Eq (HAdd.hAdd θ ↑Real.pi).sign (Neg.neg θ.sign)
  -/
  rw [sign, sign, sin_add_pi, Left.sign_neg]
  /-
    🎉 no goals
  -/


@[simp]
theorem sign_add_pi (θ : Angle) : (θ + π).sign = -θ.sign :=
  sign_antiperiodic θ


@[simp]
                                                                         /-
                                                                           θ : Real.Angle
                                                                           ⊢ Eq (HAdd.hAdd (↑Real.pi) θ).sign (Neg.neg θ.sign)
                                                                         -/
theorem sign_pi_add (θ : Angle) : ((π : Angle) + θ).sign = -θ.sign := by rw [add_comm, sign_add_pi]
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


@[simp]
theorem sign_sub_pi (θ : Angle) : (θ - π).sign = -θ.sign :=
  sign_antiperiodic.sub_eq θ


@[simp]
theorem sign_pi_sub (θ : Angle) : ((π : Angle) - θ).sign = θ.sign := by
  /-
    θ : Real.Angle
    ⊢ Eq (HSub.hSub (↑Real.pi) θ).sign θ.sign
  -/
  simp [sign_antiperiodic.sub_eq']
  /-
    🎉 no goals
  -/


theorem sign_eq_zero_iff {θ : Angle} : θ.sign = 0 ↔ θ = 0 ∨ θ = π := by
  /-
    θ : Real.Angle
    ⊢ Iff (Eq θ.sign 0) (Or (Eq θ 0) (Eq θ ↑Real.pi))
  -/
  rw [sign, _root_.sign_eq_zero_iff, sin_eq_zero_iff]
  /-
    🎉 no goals
  -/


theorem sign_ne_zero_iff {θ : Angle} : θ.sign ≠ 0 ↔ θ ≠ 0 ∧ θ ≠ π := by
  /-
    θ : Real.Angle
    ⊢ Iff (Ne θ.sign 0) (And (Ne θ 0) (Ne θ ↑Real.pi))
  -/
  rw [← not_or, ← sign_eq_zero_iff]
  /-
    🎉 no goals
  -/


theorem toReal_neg_iff_sign_neg {θ : Angle} : θ.toReal < 0 ↔ θ.sign = -1 := by
  /-
    θ : Real.Angle
    ⊢ Iff (LT.lt θ.toReal 0) (Eq θ.sign (-1))
  -/
  rw [sign, ← sin_toReal, sign_eq_neg_one_iff]
  /-
    θ : Real.Angle
    ⊢ Iff (LT.lt θ.toReal 0) (LT.lt (Real.sin θ.toReal) 0)
  -/
  rcases lt_trichotomy θ.toReal 0 with (h | h | h)
    /-
      case inl
      θ : Real.Angle
      h : LT.lt θ.toReal 0
      ⊢ Iff (LT.lt θ.toReal 0) (LT.lt (Real.sin θ.toReal) 0)
    -/
  · exact ⟨fun _ => Real.sin_neg_of_neg_of_neg_pi_lt h (neg_pi_lt_toReal θ), fun _ => h⟩
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      θ : Real.Angle
      h : Eq θ.toReal 0
      ⊢ Iff (LT.lt θ.toReal 0) (LT.lt (Real.sin θ.toReal) 0)
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
  · exact
      ⟨fun hn => False.elim (h.asymm hn), fun hn =>
        False.elim (hn.not_le (sin_nonneg_of_nonneg_of_le_pi h.le (toReal_le_pi θ)))⟩


theorem toReal_nonneg_iff_sign_nonneg {θ : Angle} : 0 ≤ θ.toReal ↔ 0 ≤ θ.sign := by
  /-
    θ : Real.Angle
    ⊢ Iff (LE.le 0 θ.toReal) (LE.le 0 θ.sign)
  -/
  rcases lt_trichotomy θ.toReal 0 with (h | h | h)
    /-
      case inl
      θ : Real.Angle
      h : LT.lt θ.toReal 0
      ⊢ Iff (LE.le 0 θ.toReal) (LE.le 0 θ.sign)
    -/
  · refine ⟨fun hn => False.elim (h.not_le hn), fun hn => ?_⟩
    /-
      case inl
      θ : Real.Angle
      h : LT.lt θ.toReal 0
      hn : LE.le 0 θ.sign
      ⊢ LE.le 0 θ.toReal
    -/
    rw [toReal_neg_iff_sign_neg.1 h] at hn
    /-
      case inl
      θ : Real.Angle
      h : LT.lt θ.toReal 0
      hn : LE.le 0 (-1)
      ⊢ LE.le 0 θ.toReal
    -/
    exact False.elim (hn.not_lt (by decide))
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      θ : Real.Angle
      h : Eq θ.toReal 0
      ⊢ Iff (LE.le 0 θ.toReal) (LE.le 0 θ.sign)
    -/
  · simp [h, sign, ← sin_toReal]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      θ : Real.Angle
      h : LT.lt 0 θ.toReal
      ⊢ Iff (LE.le 0 θ.toReal) (LE.le 0 θ.sign)
    -/
  · refine ⟨fun _ => ?_, fun _ => h.le⟩
    /-
      case inr.inr
      θ : Real.Angle
      h : LT.lt 0 θ.toReal
      x✝ : LE.le 0 θ.toReal
      ⊢ LE.le 0 θ.sign
    -/
    rw [sign, ← sin_toReal, sign_nonneg_iff]
    /-
      case inr.inr
      θ : Real.Angle
      h : LT.lt 0 θ.toReal
      x✝ : LE.le 0 θ.toReal
      ⊢ LE.le 0 (Real.sin θ.toReal)
    -/
    exact sin_nonneg_of_nonneg_of_le_pi h.le (toReal_le_pi θ)
    /-
      🎉 no goals
    -/


@[simp]
theorem sign_toReal {θ : Angle} (h : θ ≠ π) : SignType.sign θ.toReal = θ.sign := by
  /-
    θ : Real.Angle
    h : Ne θ ↑Real.pi
    ⊢ Eq (SignType.sign θ.toReal) θ.sign
  -/
  rcases lt_trichotomy θ.toReal 0 with (ht | ht | ht)
    /-
      case inl
      θ : Real.Angle
      h : Ne θ ↑Real.pi
      ht : LT.lt θ.toReal 0
      ⊢ Eq (SignType.sign θ.toReal) θ.sign
    -/
  · simp [ht, toReal_neg_iff_sign_neg.1 ht]
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      θ : Real.Angle
      h : Ne θ ↑Real.pi
      ht : Eq θ.toReal 0
      ⊢ Eq (SignType.sign θ.toReal) θ.sign
    -/
  · simp [sign, ht, ← sin_toReal]
    /-
      🎉 no goals
    -/
  · rw [sign, ← sin_toReal, sign_pos ht,
      sign_pos
        (sin_pos_of_pos_of_lt_pi ht ((toReal_le_pi θ).lt_of_ne (toReal_eq_pi_iff.not.2 h)))]


theorem coe_abs_toReal_of_sign_nonneg {θ : Angle} (h : 0 ≤ θ.sign) : ↑|θ.toReal| = θ := by
  /-
    θ : Real.Angle
    h : LE.le 0 θ.sign
    ⊢ Eq (↑(abs θ.toReal)) θ
  -/
  rw [abs_eq_self.2 (toReal_nonneg_iff_sign_nonneg.2 h), coe_toReal]
  /-
    🎉 no goals
  -/


theorem neg_coe_abs_toReal_of_sign_nonpos {θ : Angle} (h : θ.sign ≤ 0) : -↑|θ.toReal| = θ := by
  /-
    θ : Real.Angle
    h : LE.le θ.sign 0
    ⊢ Eq (Neg.neg ↑(abs θ.toReal)) θ
  -/
  rw [SignType.nonpos_iff] at h
  /-
    θ : Real.Angle
    h : Or (Eq θ.sign (-1)) (Eq θ.sign 0)
    ⊢ Eq (Neg.neg ↑(abs θ.toReal)) θ
  -/
  rcases h with (h | h)
    /-
      case inl
      θ : Real.Angle
      h : Eq θ.sign (-1)
      ⊢ Eq (Neg.neg ↑(abs θ.toReal)) θ
    -/
  · rw [abs_of_neg (toReal_neg_iff_sign_neg.2 h), coe_neg, neg_neg, coe_toReal]
    /-
      🎉 no goals
    -/
    /-
      case inr
      θ : Real.Angle
      h : Eq θ.sign 0
      ⊢ Eq (Neg.neg ↑(abs θ.toReal)) θ
    -/
  · rw [sign_eq_zero_iff] at h
    /-
      case inr
      θ : Real.Angle
      h : Or (Eq θ 0) (Eq θ ↑Real.pi)
      ⊢ Eq (Neg.neg ↑(abs θ.toReal)) θ
    -/
                                  /-
                                    🎉 no goals
                                  -/
    rcases h with (rfl | rfl) <;> simp [abs_of_pos Real.pi_pos]
                                  /-
                                    🎉 no goals
                                  -/


theorem eq_iff_sign_eq_and_abs_toReal_eq {θ ψ : Angle} :
    θ = ψ ↔ θ.sign = ψ.sign ∧ |θ.toReal| = |ψ.toReal| := by
  /-
    θ ψ : Real.Angle
    ⊢ Iff (Eq θ ψ) (And (Eq θ.sign ψ.sign) (Eq (abs θ.toReal) (abs ψ.toReal)))
  -/
  refine ⟨?_, fun h => ?_⟩
    /-
      case refine_1
      θ ψ : Real.Angle
      ⊢ Eq θ ψ → And (Eq θ.sign ψ.sign) (Eq (abs θ.toReal) (abs ψ.toReal))
    -/
  · rintro rfl
    /-
      case refine_1
      θ : Real.Angle
      ⊢ And (Eq θ.sign θ.sign) (Eq (abs θ.toReal) (abs θ.toReal))
    -/
    exact ⟨rfl, rfl⟩
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    θ ψ : Real.Angle
    h : And (Eq θ.sign ψ.sign) (Eq (abs θ.toReal) (abs ψ.toReal))
    ⊢ Eq θ ψ
  -/
  rcases h with ⟨hs, hr⟩
  /-
    case refine_2.intro
    θ ψ : Real.Angle
    hs : Eq θ.sign ψ.sign
    hr : Eq (abs θ.toReal) (abs ψ.toReal)
    ⊢ Eq θ ψ
  -/
  rw [abs_eq_abs] at hr
  /-
    case refine_2.intro
    θ ψ : Real.Angle
    hs : Eq θ.sign ψ.sign
    hr : Or (Eq θ.toReal ψ.toReal) (Eq θ.toReal (Neg.neg ψ.toReal))
    ⊢ Eq θ ψ
  -/
  rcases hr with (hr | hr)
    /-
      case refine_2.intro.inl
      θ ψ : Real.Angle
      hs : Eq θ.sign ψ.sign
      hr : Eq θ.toReal ψ.toReal
      ⊢ Eq θ ψ
    -/
  · exact toReal_injective hr
    /-
      🎉 no goals
    -/
    /-
      case refine_2.intro.inr
      θ ψ : Real.Angle
      hs : Eq θ.sign ψ.sign
      hr : Eq θ.toReal (Neg.neg ψ.toReal)
      ⊢ Eq θ ψ
    -/
  · by_cases h : θ = π
      /-
        case pos
        θ ψ : Real.Angle
        hs : Eq θ.sign ψ.sign
        hr : Eq θ.toReal (Neg.neg ψ.toReal)
        h : Eq θ ↑Real.pi
        ⊢ Eq θ ψ
      -/
    · rw [h, toReal_pi, ← neg_eq_iff_eq_neg] at hr
      /-
        case pos
        θ ψ : Real.Angle
        hs : Eq θ.sign ψ.sign
        hr : Eq (Neg.neg Real.pi) ψ.toReal
        h : Eq θ ↑Real.pi
        ⊢ Eq θ ψ
      -/
      exact False.elim ((neg_pi_lt_toReal ψ).ne hr)
      /-
        🎉 no goals
      -/
      /-
        case neg
        θ ψ : Real.Angle
        hs : Eq θ.sign ψ.sign
        hr : Eq θ.toReal (Neg.neg ψ.toReal)
        h : Not (Eq θ ↑Real.pi)
        ⊢ Eq θ ψ
      -/
    · by_cases h' : ψ = π
        /-
          case pos
          θ ψ : Real.Angle
          hs : Eq θ.sign ψ.sign
          hr : Eq θ.toReal (Neg.neg ψ.toReal)
          h : Not (Eq θ ↑Real.pi)
          h' : Eq ψ ↑Real.pi
          ⊢ Eq θ ψ
        -/
      · rw [h', toReal_pi] at hr
        /-
          case pos
          θ ψ : Real.Angle
          hs : Eq θ.sign ψ.sign
          hr : Eq θ.toReal (Neg.neg Real.pi)
          h : Not (Eq θ ↑Real.pi)
          h' : Eq ψ ↑Real.pi
          ⊢ Eq θ ψ
        -/
        exact False.elim ((neg_pi_lt_toReal θ).ne hr.symm)
        /-
          🎉 no goals
        -/
      · rw [← sign_toReal h, ← sign_toReal h', hr, Left.sign_neg, SignType.neg_eq_self_iff,
          _root_.sign_eq_zero_iff, toReal_eq_zero_iff] at hs
        /-
          case neg
          θ ψ : Real.Angle
          hs : Eq ψ 0
          hr : Eq θ.toReal (Neg.neg ψ.toReal)
          h : Not (Eq θ ↑Real.pi)
          h' : Not (Eq ψ ↑Real.pi)
          ⊢ Eq θ ψ
        -/
        rw [hs, toReal_zero, neg_zero, toReal_eq_zero_iff] at hr
        /-
          case neg
          θ ψ : Real.Angle
          hs : Eq ψ 0
          hr : Eq θ 0
          h : Not (Eq θ ↑Real.pi)
          h' : Not (Eq ψ ↑Real.pi)
          ⊢ Eq θ ψ
        -/
        rw [hr, hs]
        /-
          🎉 no goals
        -/


theorem eq_iff_abs_toReal_eq_of_sign_eq {θ ψ : Angle} (h : θ.sign = ψ.sign) :
                                          /-
                                            θ ψ : Real.Angle
                                            h : Eq θ.sign ψ.sign
                                            ⊢ Iff (Eq θ ψ) (Eq (abs θ.toReal) (abs ψ.toReal))
                                          -/
    θ = ψ ↔ |θ.toReal| = |ψ.toReal| := by simpa [h] using @eq_iff_sign_eq_and_abs_toReal_eq θ ψ
                                          /-
                                            🎉 no goals
                                          -/


@[simp]
theorem sign_coe_pi_div_two : (↑(π / 2) : Angle).sign = 1 := by
  /-
    ⊢ Eq (↑(HDiv.hDiv Real.pi 2)).sign 1
  -/
  rw [sign, sin_coe, sin_pi_div_two, sign_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem sign_coe_neg_pi_div_two : (↑(-π / 2) : Angle).sign = -1 := by
  /-
    ⊢ Eq (↑(HDiv.hDiv (Neg.neg Real.pi) 2)).sign (-1)
  -/
  rw [sign, sin_coe, neg_div, Real.sin_neg, sin_pi_div_two, Left.sign_neg, sign_one]
  /-
    🎉 no goals
  -/


theorem sign_coe_nonneg_of_nonneg_of_le_pi {θ : ℝ} (h0 : 0 ≤ θ) (hpi : θ ≤ π) :
    0 ≤ (θ : Angle).sign := by
  /-
    θ : Real
    h0 : LE.le 0 θ
    hpi : LE.le θ Real.pi
    ⊢ LE.le 0 (↑θ).sign
  -/
  rw [sign, sign_nonneg_iff]
  /-
    θ : Real
    h0 : LE.le 0 θ
    hpi : LE.le θ Real.pi
    ⊢ LE.le 0 (↑θ).sin
  -/
  exact sin_nonneg_of_nonneg_of_le_pi h0 hpi
  /-
    🎉 no goals
  -/


theorem sign_neg_coe_nonpos_of_nonneg_of_le_pi {θ : ℝ} (h0 : 0 ≤ θ) (hpi : θ ≤ π) :
    (-θ : Angle).sign ≤ 0 := by
  /-
    θ : Real
    h0 : LE.le 0 θ
    hpi : LE.le θ Real.pi
    ⊢ LE.le (Neg.neg ↑θ).sign 0
  -/
  rw [sign, sign_nonpos_iff, sin_neg, Left.neg_nonpos_iff]
  /-
    θ : Real
    h0 : LE.le 0 θ
    hpi : LE.le θ Real.pi
    ⊢ LE.le 0 (↑θ).sin
  -/
  exact sin_nonneg_of_nonneg_of_le_pi h0 hpi
  /-
    🎉 no goals
  -/


theorem sign_two_nsmul_eq_sign_iff {θ : Angle} :
    ((2 : ℕ) • θ).sign = θ.sign ↔ θ = π ∨ |θ.toReal| < π / 2 := by
  /-
    θ : Real.Angle
    ⊢ Iff (Eq (HSMul.hSMul 2 θ).sign θ.sign) (Or (Eq θ ↑Real.pi) (LT.lt (abs θ.toR …
  -/
  by_cases hpi : θ = π; · simp [hpi]
                          /-
                            🎉 no goals
                          -/
  /-
    case neg
    θ : Real.Angle
    hpi : Not (Eq θ ↑Real.pi)
    ⊢ Iff (Eq (HSMul.hSMul 2 θ).sign θ.sign) (Or (Eq θ ↑Real.pi) (LT.lt (abs θ.toR …
  -/
  rw [or_iff_right hpi]
  /-
    case neg
    θ : Real.Angle
    hpi : Not (Eq θ ↑Real.pi)
    ⊢ Iff (Eq (HSMul.hSMul 2 θ).sign θ.sign) (LT.lt (abs θ.toReal) (HDiv.hDiv Real …
  -/
  refine ⟨fun h => ?_, fun h => ?_⟩
    /-
      case neg.refine_1
      θ : Real.Angle
      hpi : Not (Eq θ ↑Real.pi)
      h : Eq (HSMul.hSMul 2 θ).sign θ.sign
      ⊢ LT.lt (abs θ.toReal) (HDiv.hDiv Real.pi 2)
    -/
  · by_contra hle
    /-
      case neg.refine_1
      θ : Real.Angle
      hpi : Not (Eq θ ↑Real.pi)
      h : Eq (HSMul.hSMul 2 θ).sign θ.sign
      hle : Not (LT.lt (abs θ.toReal) (HDiv.hDiv Real.pi 2))
      ⊢ False
    -/
    rw [not_lt, le_abs, le_neg] at hle
    /-
      case neg.refine_1
      θ : Real.Angle
      hpi : Not (Eq θ ↑Real.pi)
      h : Eq (HSMul.hSMul 2 θ).sign θ.sign
      hle : Or (LE.le (HDiv.hDiv Real.pi 2) θ.toReal) (LE.le θ.toReal (Neg.neg (HDiv …
      ⊢ False
    -/
    have hpi' : θ.toReal ≠ π := by simpa using hpi
    /-
      case neg.refine_1
      θ : Real.Angle
      hpi : Not (Eq θ ↑Real.pi)
      h : Eq (HSMul.hSMul 2 θ).sign θ.sign
      hle : Or (LE.le (HDiv.hDiv Real.pi 2) θ.toReal) (LE.le θ.toReal (Neg.neg (HDiv …
      hpi' : Ne θ.toReal Real.pi
      ⊢ False
    -/
    rcases hle with (hle | hle) <;> rcases hle.eq_or_lt with (heq | hlt)
      /-
        case neg.refine_1.inl.inl
        θ : Real.Angle
        hpi : Not (Eq θ ↑Real.pi)
        h : Eq (HSMul.hSMul 2 θ).sign θ.sign
        hpi' : Ne θ.toReal Real.pi
        hle : LE.le (HDiv.hDiv Real.pi 2) θ.toReal
        heq : Eq (HDiv.hDiv Real.pi 2) θ.toReal
        ⊢ False
      -/
    · rw [← coe_toReal θ, ← heq] at h
      /-
        case neg.refine_1.inl.inl
        θ : Real.Angle
        hpi : Not (Eq θ ↑Real.pi)
        h : Eq (HSMul.hSMul 2 ↑(HDiv.hDiv Real.pi 2)).sign (↑(HDiv.hDiv Real.pi 2)).sign
        hpi' : Ne θ.toReal Real.pi
        hle : LE.le (HDiv.hDiv Real.pi 2) θ.toReal
        heq : Eq (HDiv.hDiv Real.pi 2) θ.toReal
        ⊢ False
      -/
      simp at h
      /-
        🎉 no goals
      -/
    · rw [← sign_toReal hpi, sign_pos (pi_div_two_pos.trans hlt), ← sign_toReal,
        two_nsmul_toReal_eq_two_mul_sub_two_pi.2 hlt, _root_.sign_neg] at h
        /-
          case neg.refine_1.inl.inr
          θ : Real.Angle
          hpi : Not (Eq θ ↑Real.pi)
          h : Eq (-1) 1
          hpi' : Ne θ.toReal Real.pi
          hle : LE.le (HDiv.hDiv Real.pi 2) θ.toReal
          hlt : LT.lt (HDiv.hDiv Real.pi 2) θ.toReal
          ⊢ False
        -/
      · simp at h
        /-
          🎉 no goals
        -/
        /-
          case neg.refine_1.inl.inr
          θ : Real.Angle
          hpi : Not (Eq θ ↑Real.pi)
          h : Eq (SignType.sign (HSub.hSub (HMul.hMul 2 θ.toReal) (HMul.hMul 2 Real.pi)) …
          hpi' : Ne θ.toReal Real.pi
          hle : LE.le (HDiv.hDiv Real.pi 2) θ.toReal
          hlt : LT.lt (HDiv.hDiv Real.pi 2) θ.toReal
          ⊢ LT.lt (HSub.hSub (HMul.hMul 2 θ.toReal) (HMul.hMul 2 Real.pi)) 0
        -/
      · rw [← mul_sub]
        /-
          case neg.refine_1.inl.inr
          θ : Real.Angle
          hpi : Not (Eq θ ↑Real.pi)
          h : Eq (SignType.sign (HSub.hSub (HMul.hMul 2 θ.toReal) (HMul.hMul 2 Real.pi)) …
          hpi' : Ne θ.toReal Real.pi
          hle : LE.le (HDiv.hDiv Real.pi 2) θ.toReal
          hlt : LT.lt (HDiv.hDiv Real.pi 2) θ.toReal
          ⊢ LT.lt (HMul.hMul 2 (HSub.hSub θ.toReal Real.pi)) 0
        -/
        exact mul_neg_of_pos_of_neg two_pos (sub_neg.2 ((toReal_le_pi _).lt_of_ne hpi'))
        /-
          🎉 no goals
        -/
        /-
          case neg.refine_1.inl.inr
          θ : Real.Angle
          hpi : Not (Eq θ ↑Real.pi)
          h : Eq (HSMul.hSMul 2 θ).sign 1
          hpi' : Ne θ.toReal Real.pi
          hle : LE.le (HDiv.hDiv Real.pi 2) θ.toReal
          hlt : LT.lt (HDiv.hDiv Real.pi 2) θ.toReal
          ⊢ Ne (HSMul.hSMul 2 θ) ↑Real.pi
        -/
      · intro he
        /-
          case neg.refine_1.inl.inr
          θ : Real.Angle
          hpi : Not (Eq θ ↑Real.pi)
          h : Eq (HSMul.hSMul 2 θ).sign 1
          hpi' : Ne θ.toReal Real.pi
          hle : LE.le (HDiv.hDiv Real.pi 2) θ.toReal
          hlt : LT.lt (HDiv.hDiv Real.pi 2) θ.toReal
          he : Eq (HSMul.hSMul 2 θ) ↑Real.pi
          ⊢ False
        -/
        simp [he] at h
        /-
          🎉 no goals
        -/
      /-
        case neg.refine_1.inr.inl
        θ : Real.Angle
        hpi : Not (Eq θ ↑Real.pi)
        h : Eq (HSMul.hSMul 2 θ).sign θ.sign
        hpi' : Ne θ.toReal Real.pi
        hle : LE.le θ.toReal (Neg.neg (HDiv.hDiv Real.pi 2))
        heq : Eq θ.toReal (Neg.neg (HDiv.hDiv Real.pi 2))
        ⊢ False
      -/
    · rw [← coe_toReal θ, heq] at h
      /-
        case neg.refine_1.inr.inl
        θ : Real.Angle
        hpi : Not (Eq θ ↑Real.pi)
        h : Eq (HSMul.hSMul 2 ↑(Neg.neg (HDiv.hDiv Real.pi 2))).sign (↑(Neg.neg (HDiv. …
        hpi' : Ne θ.toReal Real.pi
        hle : LE.le θ.toReal (Neg.neg (HDiv.hDiv Real.pi 2))
        heq : Eq θ.toReal (Neg.neg (HDiv.hDiv Real.pi 2))
        ⊢ False
      -/
      simp at h
      /-
        🎉 no goals
      -/
    · rw [← sign_toReal hpi, _root_.sign_neg (hlt.trans (Left.neg_neg_iff.2 pi_div_two_pos)), ←
        sign_toReal] at h
      /-
        case neg.refine_1.inr.inr
        θ : Real.Angle
        hpi : Not (Eq θ ↑Real.pi)
        h : Eq (SignType.sign (HSMul.hSMul 2 θ).toReal) (-1)
        hpi' : Ne θ.toReal Real.pi
        hle : LE.le θ.toReal (Neg.neg (HDiv.hDiv Real.pi 2))
        hlt : LT.lt θ.toReal (Neg.neg (HDiv.hDiv Real.pi 2))
        ⊢ False
      -/
      swap
        /-
          case neg.refine_1.inr.inr
          θ : Real.Angle
          hpi : Not (Eq θ ↑Real.pi)
          h : Eq (HSMul.hSMul 2 θ).sign (-1)
          hpi' : Ne θ.toReal Real.pi
          hle : LE.le θ.toReal (Neg.neg (HDiv.hDiv Real.pi 2))
          hlt : LT.lt θ.toReal (Neg.neg (HDiv.hDiv Real.pi 2))
          ⊢ Ne (HSMul.hSMul 2 θ) ↑Real.pi
        -/
      · intro he
        /-
          case neg.refine_1.inr.inr
          θ : Real.Angle
          hpi : Not (Eq θ ↑Real.pi)
          h : Eq (HSMul.hSMul 2 θ).sign (-1)
          hpi' : Ne θ.toReal Real.pi
          hle : LE.le θ.toReal (Neg.neg (HDiv.hDiv Real.pi 2))
          hlt : LT.lt θ.toReal (Neg.neg (HDiv.hDiv Real.pi 2))
          he : Eq (HSMul.hSMul 2 θ) ↑Real.pi
          ⊢ False
        -/
        simp [he] at h
        /-
          🎉 no goals
        -/
      /-
        case neg.refine_1.inr.inr
        θ : Real.Angle
        hpi : Not (Eq θ ↑Real.pi)
        h : Eq (SignType.sign (HSMul.hSMul 2 θ).toReal) (-1)
        hpi' : Ne θ.toReal Real.pi
        hle : LE.le θ.toReal (Neg.neg (HDiv.hDiv Real.pi 2))
        hlt : LT.lt θ.toReal (Neg.neg (HDiv.hDiv Real.pi 2))
        ⊢ False
      -/
      rw [← neg_div] at hlt
      /-
        case neg.refine_1.inr.inr
        θ : Real.Angle
        hpi : Not (Eq θ ↑Real.pi)
        h : Eq (SignType.sign (HSMul.hSMul 2 θ).toReal) (-1)
        hpi' : Ne θ.toReal Real.pi
        hle : LE.le θ.toReal (Neg.neg (HDiv.hDiv Real.pi 2))
        hlt : LT.lt θ.toReal (HDiv.hDiv (Neg.neg Real.pi) 2)
        ⊢ False
      -/
      rw [two_nsmul_toReal_eq_two_mul_add_two_pi.2 hlt.le, sign_pos] at h
        /-
          case neg.refine_1.inr.inr
          θ : Real.Angle
          hpi : Not (Eq θ ↑Real.pi)
          h : Eq 1 (-1)
          hpi' : Ne θ.toReal Real.pi
          hle : LE.le θ.toReal (Neg.neg (HDiv.hDiv Real.pi 2))
          hlt : LT.lt θ.toReal (HDiv.hDiv (Neg.neg Real.pi) 2)
          ⊢ False
        -/
      · simp at h
        /-
          🎉 no goals
        -/
        /-
          case neg.refine_1.inr.inr
          θ : Real.Angle
          hpi : Not (Eq θ ↑Real.pi)
          h : Eq (SignType.sign (HAdd.hAdd (HMul.hMul 2 θ.toReal) (HMul.hMul 2 Real.pi)) …
          hpi' : Ne θ.toReal Real.pi
          hle : LE.le θ.toReal (Neg.neg (HDiv.hDiv Real.pi 2))
          hlt : LT.lt θ.toReal (HDiv.hDiv (Neg.neg Real.pi) 2)
          ⊢ LT.lt 0 (HAdd.hAdd (HMul.hMul 2 θ.toReal) (HMul.hMul 2 Real.pi))
        -/
      · linarith [neg_pi_lt_toReal θ]
        /-
          🎉 no goals
        -/
  · have hpi' : (2 : ℕ) • θ ≠ π := by
      rw [Ne, two_nsmul_eq_pi_iff, not_or]
      constructor
      · rintro rfl
        simp [pi_pos, div_pos, abs_of_pos] at h
      · rintro rfl
        rw [toReal_neg_pi_div_two] at h
        simp [pi_pos, div_pos, neg_div, abs_of_pos] at h
    /-
      case neg.refine_2
      θ : Real.Angle
      hpi : Not (Eq θ ↑Real.pi)
      h : LT.lt (abs θ.toReal) (HDiv.hDiv Real.pi 2)
      hpi' : Ne (HSMul.hSMul 2 θ) ↑Real.pi
      ⊢ Eq (HSMul.hSMul 2 θ).sign θ.sign
    -/
    rw [abs_lt, ← neg_div] at h
    rw [← sign_toReal hpi, ← sign_toReal hpi', two_nsmul_toReal_eq_two_mul.2 ⟨h.1, h.2.le⟩,
      sign_mul, sign_pos (zero_lt_two' ℝ), one_mul]


theorem sign_two_zsmul_eq_sign_iff {θ : Angle} :
    ((2 : ℤ) • θ).sign = θ.sign ↔ θ = π ∨ |θ.toReal| < π / 2 := by
  /-
    θ : Real.Angle
    ⊢ Iff (Eq (HSMul.hSMul 2 θ).sign θ.sign) (Or (Eq θ ↑Real.pi) (LT.lt (abs θ.toR …
  -/
  rw [two_zsmul, ← two_nsmul, sign_two_nsmul_eq_sign_iff]
  /-
    🎉 no goals
  -/


theorem continuousAt_sign {θ : Angle} (h0 : θ ≠ 0) (hpi : θ ≠ π) : ContinuousAt sign θ :=
  (continuousAt_sign_of_ne_zero (sin_ne_zero_iff.2 ⟨h0, hpi⟩)).comp continuous_sin.continuousAt


theorem _root_.ContinuousOn.angle_sign_comp {α : Type*} [TopologicalSpace α] {f : α → Angle}
    {s : Set α} (hf : ContinuousOn f s) (hs : ∀ z ∈ s, f z ≠ 0 ∧ f z ≠ π) :
    ContinuousOn (sign ∘ f) s := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    f : α → Real.Angle
    s : Set α
    hf : ContinuousOn f s
    hs : ∀ (z : α), Membership.mem s z → And (Ne (f z) 0) (Ne (f z) ↑Real.pi)
    ⊢ ContinuousOn (Function.comp Real.Angle.sign f) s
  -/
  refine (continuousOn_of_forall_continuousAt fun θ hθ => ?_).comp hf (Set.mapsTo_image f s)
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    f : α → Real.Angle
    s : Set α
    hf : ContinuousOn f s
    hs : ∀ (z : α), Membership.mem s z → And (Ne (f z) 0) (Ne (f z) ↑Real.pi)
    θ : Real.Angle
    hθ : Membership.mem (Set.image f s) θ
    ⊢ ContinuousAt Real.Angle.sign θ
  -/
  obtain ⟨z, hz, rfl⟩ := hθ
  /-
    case intro.intro
    α : Type u_1
    inst✝ : TopologicalSpace α
    f : α → Real.Angle
    s : Set α
    hf : ContinuousOn f s
    hs : ∀ (z : α), Membership.mem s z → And (Ne (f z) 0) (Ne (f z) ↑Real.pi)
    z : α
    hz : Membership.mem s z
    ⊢ ContinuousAt Real.Angle.sign (f z)
  -/
  exact continuousAt_sign (hs _ hz).1 (hs _ hz).2
  /-
    🎉 no goals
  -/


/-- Suppose a function to angles is continuous on a connected set and never takes the values `0`
or `π` on that set. Then the values of the function on that set all have the same sign. -/
theorem sign_eq_of_continuousOn {α : Type*} [TopologicalSpace α] {f : α → Angle} {s : Set α}
    {x y : α} (hc : IsConnected s) (hf : ContinuousOn f s) (hs : ∀ z ∈ s, f z ≠ 0 ∧ f z ≠ π)
    (hx : x ∈ s) (hy : y ∈ s) : (f y).sign = (f x).sign :=
  (hc.image _ (hf.angle_sign_comp hs)).isPreconnected.subsingleton (Set.mem_image_of_mem _ hy)
    (Set.mem_image_of_mem _ hx)


