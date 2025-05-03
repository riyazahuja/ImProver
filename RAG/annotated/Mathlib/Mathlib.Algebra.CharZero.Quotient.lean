/-- `z • r` is a multiple of `p` iff `r` is `pk/z` above a multiple of `p`, where `0 ≤ k < |z|`. -/
theorem zsmul_mem_zmultiples_iff_exists_sub_div {r : R} {z : ℤ} (hz : z ≠ 0) :
    z • r ∈ AddSubgroup.zmultiples p ↔
      ∃ k : Fin z.natAbs, r - (k : ℕ) • (p / z : R) ∈ AddSubgroup.zmultiples p := by
  /-
    R : Type u_1
    inst✝¹ : DivisionRing R
    inst✝ : CharZero R
    p r : R
    z : Int
    hz : Ne z 0
    ⊢ Iff (Membership.mem (AddSubgroup.zmultiples p) (HSMul.hSMul z r)) (Exists fu …
  -/
  rw [AddSubgroup.mem_zmultiples_iff]
  /-
    R : Type u_1
    inst✝¹ : DivisionRing R
    inst✝ : CharZero R
    p r : R
    z : Int
    hz : Ne z 0
    ⊢ Iff (Exists fun k => Eq (HSMul.hSMul k p) (HSMul.hSMul z r)) (Exists fun k = …
  -/
  simp_rw [AddSubgroup.mem_zmultiples_iff, div_eq_mul_inv, ← smul_mul_assoc, eq_sub_iff_add_eq]
  /-
    R : Type u_1
    inst✝¹ : DivisionRing R
    inst✝ : CharZero R
    p r : R
    z : Int
    hz : Ne z 0
    ⊢ Iff (Exists fun k => Eq (HSMul.hSMul k p) (HSMul.hSMul z r)) (Exists fun k = …
  -/
  have hz' : (z : R) ≠ 0 := Int.cast_ne_zero.mpr hz
  /-
    R : Type u_1
    inst✝¹ : DivisionRing R
    inst✝ : CharZero R
    p r : R
    z : Int
    hz : Ne z 0
    hz' : Ne (↑z) 0
    ⊢ Iff (Exists fun k => Eq (HSMul.hSMul k p) (HSMul.hSMul z r)) (Exists fun k = …
  -/
  conv_rhs => simp (config := { singlePass := true }) only [← (mul_right_injective₀ hz').eq_iff]
  simp_rw [← zsmul_eq_mul, smul_add, ← mul_smul_comm, zsmul_eq_mul (z : R)⁻¹, mul_inv_cancel₀ hz',
    mul_one, ← natCast_zsmul, smul_smul, ← add_smul]
  /-
    R : Type u_1
    inst✝¹ : DivisionRing R
    inst✝ : CharZero R
    p r : R
    z : Int
    hz : Ne z 0
    hz' : Ne (↑z) 0
    ⊢ Iff (Exists fun k => Eq (HSMul.hSMul k p) (HSMul.hSMul z r)) (Exists fun k = …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝¹ : DivisionRing R
      inst✝ : CharZero R
      p r : R
      z : Int
      hz : Ne z 0
      hz' : Ne (↑z) 0
      ⊢ (Exists fun k => Eq (HSMul.hSMul k p) (HSMul.hSMul z r)) → Exists fun k => E …
    -/
  · rintro ⟨k, h⟩
    /-
      case mp.intro
      R : Type u_1
      inst✝¹ : DivisionRing R
      inst✝ : CharZero R
      p r : R
      z : Int
      hz : Ne z 0
      hz' : Ne (↑z) 0
      k : Int
      h : Eq (HSMul.hSMul k p) (HSMul.hSMul z r)
      ⊢ Exists fun k => Exists fun k_1 => Eq (HSMul.hSMul (HAdd.hAdd (HMul.hMul z k_ …
    -/
    simp_rw [← h]
    /-
      case mp.intro
      R : Type u_1
      inst✝¹ : DivisionRing R
      inst✝ : CharZero R
      p r : R
      z : Int
      hz : Ne z 0
      hz' : Ne (↑z) 0
      k : Int
      h : Eq (HSMul.hSMul k p) (HSMul.hSMul z r)
      ⊢ Exists fun k_1 => Exists fun k_2 => Eq (HSMul.hSMul (HAdd.hAdd (HMul.hMul z  …
    -/
    refine ⟨⟨(k % z).toNat, ?_⟩, k / z, ?_⟩
      /-
        case mp.intro.refine_1
        R : Type u_1
        inst✝¹ : DivisionRing R
        inst✝ : CharZero R
        p r : R
        z : Int
        hz : Ne z 0
        hz' : Ne (↑z) 0
        k : Int
        h : Eq (HSMul.hSMul k p) (HSMul.hSMul z r)
        ⊢ LT.lt (HMod.hMod k z).toNat z.natAbs
      -/
    · rw [← Int.ofNat_lt, Int.toNat_of_nonneg (Int.emod_nonneg _ hz)]
      /-
        case mp.intro.refine_1
        R : Type u_1
        inst✝¹ : DivisionRing R
        inst✝ : CharZero R
        p r : R
        z : Int
        hz : Ne z 0
        hz' : Ne (↑z) 0
        k : Int
        h : Eq (HSMul.hSMul k p) (HSMul.hSMul z r)
        ⊢ LT.lt (HMod.hMod k z) ↑z.natAbs
      -/
      exact (Int.emod_lt _ hz).trans_eq (Int.abs_eq_natAbs _)
      /-
        🎉 no goals
      -/
    /-
      case mp.intro.refine_2
      R : Type u_1
      inst✝¹ : DivisionRing R
      inst✝ : CharZero R
      p r : R
      z : Int
      hz : Ne z 0
      hz' : Ne (↑z) 0
      k : Int
      h : Eq (HSMul.hSMul k p) (HSMul.hSMul z r)
      ⊢ Eq (HSMul.hSMul (HAdd.hAdd (HMul.hMul z (HDiv.hDiv k z)) ↑↑⟨(HMod.hMod k z). …
    -/
    rw [Fin.val_mk, Int.toNat_of_nonneg (Int.emod_nonneg _ hz)]
    /-
      case mp.intro.refine_2
      R : Type u_1
      inst✝¹ : DivisionRing R
      inst✝ : CharZero R
      p r : R
      z : Int
      hz : Ne z 0
      hz' : Ne (↑z) 0
      k : Int
      h : Eq (HSMul.hSMul k p) (HSMul.hSMul z r)
      ⊢ Eq (HSMul.hSMul (HAdd.hAdd (HMul.hMul z (HDiv.hDiv k z)) (HMod.hMod k z)) p) …
    -/
    nth_rewrite 3 [← Int.ediv_add_emod k z]
    /-
      case mp.intro.refine_2
      R : Type u_1
      inst✝¹ : DivisionRing R
      inst✝ : CharZero R
      p r : R
      z : Int
      hz : Ne z 0
      hz' : Ne (↑z) 0
      k : Int
      h : Eq (HSMul.hSMul k p) (HSMul.hSMul z r)
      ⊢ Eq (HSMul.hSMul (HAdd.hAdd (HMul.hMul z (HDiv.hDiv k z)) (HMod.hMod k z)) p) …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      inst✝¹ : DivisionRing R
      inst✝ : CharZero R
      p r : R
      z : Int
      hz : Ne z 0
      hz' : Ne (↑z) 0
      ⊢ (Exists fun k => Exists fun k_1 => Eq (HSMul.hSMul (HAdd.hAdd (HMul.hMul z k …
    -/
  · rintro ⟨k, n, h⟩
    /-
      case mpr.intro.intro
      R : Type u_1
      inst✝¹ : DivisionRing R
      inst✝ : CharZero R
      p r : R
      z : Int
      hz : Ne z 0
      hz' : Ne (↑z) 0
      k : Fin z.natAbs
      n : Int
      h : Eq (HSMul.hSMul (HAdd.hAdd (HMul.hMul z n) ↑↑k) p) (HSMul.hSMul z r)
      ⊢ Exists fun k => Eq (HSMul.hSMul k p) (HSMul.hSMul z r)
    -/
    exact ⟨_, h⟩
    /-
      🎉 no goals
    -/


theorem nsmul_mem_zmultiples_iff_exists_sub_div {r : R} {n : ℕ} (hn : n ≠ 0) :
    n • r ∈ AddSubgroup.zmultiples p ↔
      ∃ k : Fin n, r - (k : ℕ) • (p / n : R) ∈ AddSubgroup.zmultiples p := by
  rw [← natCast_zsmul r, zsmul_mem_zmultiples_iff_exists_sub_div (Int.natCast_ne_zero.mpr hn),
    Int.cast_natCast]
  /-
    R : Type u_1
    inst✝¹ : DivisionRing R
    inst✝ : CharZero R
    p r : R
    n : Nat
    hn : Ne n 0
    ⊢ Iff (Exists fun k => Membership.mem (AddSubgroup.zmultiples p) (HSub.hSub r  …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem zmultiples_zsmul_eq_zsmul_iff {ψ θ : R ⧸ AddSubgroup.zmultiples p} {z : ℤ} (hz : z ≠ 0) :
    z • ψ = z • θ ↔ ∃ k : Fin z.natAbs, ψ = θ + ((k : ℕ) • (p / z) : R) := by
  /-
    R : Type u_1
    inst✝¹ : DivisionRing R
    inst✝ : CharZero R
    p : R
    ψ θ : HasQuotient.Quotient R (AddSubgroup.zmultiples p)
    z : Int
    hz : Ne z 0
    ⊢ Iff (Eq (HSMul.hSMul z ψ) (HSMul.hSMul z θ)) (Exists fun k => Eq ψ (HAdd.hAd …
  -/
  induction ψ using Quotient.inductionOn
  /-
    case h
    R : Type u_1
    inst✝¹ : DivisionRing R
    inst✝ : CharZero R
    p : R
    θ : HasQuotient.Quotient R (AddSubgroup.zmultiples p)
    z : Int
    hz : Ne z 0
    a✝ : R
    ⊢ Iff (Eq (HSMul.hSMul z (Quotient.mk (QuotientAddGroup.leftRel (AddSubgroup.z …
  -/
  induction θ using Quotient.inductionOn
  -- Porting note: Introduced Zp notation to shorten lines
  /-
    case h.h
    R : Type u_1
    inst✝¹ : DivisionRing R
    inst✝ : CharZero R
    p : R
    z : Int
    hz : Ne z 0
    a✝¹ a✝ : R
    ⊢ Iff (Eq (HSMul.hSMul z (Quotient.mk (QuotientAddGroup.leftRel (AddSubgroup.z …
  -/
  let Zp := AddSubgroup.zmultiples p
  /-
    case h.h
    R : Type u_1
    inst✝¹ : DivisionRing R
    inst✝ : CharZero R
    p : R
    z : Int
    hz : Ne z 0
    a✝¹ a✝ : R
    Zp : AddSubgroup R := AddSubgroup.zmultiples p
    ⊢ Iff (Eq (HSMul.hSMul z (Quotient.mk (QuotientAddGroup.leftRel (AddSubgroup.z …
  -/
  have : (Quotient.mk _ : R → R ⧸ Zp) = ((↑) : R → R ⧸ Zp) := rfl
  /-
    case h.h
    R : Type u_1
    inst✝¹ : DivisionRing R
    inst✝ : CharZero R
    p : R
    z : Int
    hz : Ne z 0
    a✝¹ a✝ : R
    Zp : AddSubgroup R := AddSubgroup.zmultiples p
    this : Eq (Quotient.mk (QuotientAddGroup.leftRel Zp)) QuotientAddGroup.mk
    ⊢ Iff (Eq (HSMul.hSMul z (Quotient.mk (QuotientAddGroup.leftRel (AddSubgroup.z …
  -/
  simp only [Zp, this]
  simp_rw [← QuotientAddGroup.mk_zsmul, ← QuotientAddGroup.mk_add,
    QuotientAddGroup.eq_iff_sub_mem, ← smul_sub, ← sub_sub]
  /-
    case h.h
    R : Type u_1
    inst✝¹ : DivisionRing R
    inst✝ : CharZero R
    p : R
    z : Int
    hz : Ne z 0
    a✝¹ a✝ : R
    Zp : AddSubgroup R := AddSubgroup.zmultiples p
    this : Eq (Quotient.mk (QuotientAddGroup.leftRel Zp)) QuotientAddGroup.mk
    ⊢ Iff (Membership.mem (AddSubgroup.zmultiples p) (HSMul.hSMul z (HSub.hSub a✝¹ …
  -/
  exact AddSubgroup.zsmul_mem_zmultiples_iff_exists_sub_div hz
  /-
    🎉 no goals
  -/


theorem zmultiples_nsmul_eq_nsmul_iff {ψ θ : R ⧸ AddSubgroup.zmultiples p} {n : ℕ} (hz : n ≠ 0) :
    n • ψ = n • θ ↔ ∃ k : Fin n, ψ = θ + (k : ℕ) • (p / n : R) := by
  rw [← natCast_zsmul ψ, ← natCast_zsmul θ,
    zmultiples_zsmul_eq_zsmul_iff (Int.natCast_ne_zero.mpr hz), Int.cast_natCast]
  /-
    R : Type u_1
    inst✝¹ : DivisionRing R
    inst✝ : CharZero R
    p : R
    ψ θ : HasQuotient.Quotient R (AddSubgroup.zmultiples p)
    n : Nat
    hz : Ne n 0
    ⊢ Iff (Exists fun k => Eq ψ (HAdd.hAdd θ ↑(HSMul.hSMul (↑k) (HDiv.hDiv p ↑n))) …
  -/
  rfl
  /-
    🎉 no goals
  -/


