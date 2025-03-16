/-- Auxiliary definition that unifies
 the totalised inverse formal power series `(_)⁻¹` and
 the inverse formal power series that depends on
 an inverse of the constant coefficient `invOfUnit`. -/
protected noncomputable def inv.aux (a : R) (φ : MvPowerSeries σ R) : MvPowerSeries σ R
  | n =>
    letI := Classical.decEq σ
    if n = 0 then a
    else
      -a *
        ∑ x ∈ antidiagonal n, if _ : x.2 < n then coeff R x.1 φ * inv.aux a φ x.2 else 0
termination_by n => n


theorem coeff_inv_aux [DecidableEq σ] (n : σ →₀ ℕ) (a : R) (φ : MvPowerSeries σ R) :
    coeff R n (inv.aux a φ) =
      if n = 0 then a
      else
        -a *
          ∑ x ∈ antidiagonal n, if x.2 < n then coeff R x.1 φ * coeff R x.2 (inv.aux a φ) else 0 :=
  show inv.aux a φ n = _ by
    /-
      σ : Type u_1
      R : Type u_2
      inst✝¹ : Ring R
      inst✝ : DecidableEq σ
      n : Finsupp σ Nat
      a : R
      φ : MvPowerSeries σ R
      ⊢ Eq (MvPowerSeries.inv.aux a φ n) (ite (Eq n 0) a (HMul.hMul (Neg.neg a) ((Fi …
    -/
    cases Subsingleton.elim ‹DecidableEq σ› (Classical.decEq σ)
    /-
      case refl
      σ : Type u_1
      R : Type u_2
      inst✝ : Ring R
      n : Finsupp σ Nat
      a : R
      φ : MvPowerSeries σ R
      ⊢ Eq (MvPowerSeries.inv.aux a φ n) (ite (Eq n 0) a (HMul.hMul (Neg.neg a) ((Fi …
    -/
    rw [inv.aux]
    /-
      case refl
      σ : Type u_1
      R : Type u_2
      inst✝ : Ring R
      n : Finsupp σ Nat
      a : R
      φ : MvPowerSeries σ R
      ⊢ Eq (ite (Eq n 0) a (HMul.hMul (Neg.neg a) ((Finset.HasAntidiagonal.antidiago …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- A multivariate formal power series is invertible if the constant coefficient is invertible. -/
def invOfUnit (φ : MvPowerSeries σ R) (u : Rˣ) : MvPowerSeries σ R :=
  inv.aux (↑u⁻¹) φ


theorem coeff_invOfUnit [DecidableEq σ] (n : σ →₀ ℕ) (φ : MvPowerSeries σ R) (u : Rˣ) :
    coeff R n (invOfUnit φ u) =
      if n = 0 then ↑u⁻¹
      else
        -↑u⁻¹ *
          ∑ x ∈ antidiagonal n,
            if x.2 < n then coeff R x.1 φ * coeff R x.2 (invOfUnit φ u) else 0 := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝¹ : Ring R
    inst✝ : DecidableEq σ
    n : Finsupp σ Nat
    φ : MvPowerSeries σ R
    u : Units R
    ⊢ Eq ((MvPowerSeries.coeff R n) (φ.invOfUnit u)) (ite (Eq n 0) (↑(Inv.inv u))  …
  -/
  convert coeff_inv_aux n (↑u⁻¹) φ
  /-
    🎉 no goals
  -/


@[simp]
theorem constantCoeff_invOfUnit (φ : MvPowerSeries σ R) (u : Rˣ) :
    constantCoeff σ R (invOfUnit φ u) = ↑u⁻¹ := by
  classical
  rw [← coeff_zero_eq_constantCoeff_apply, coeff_invOfUnit, if_pos rfl]


@[simp]
theorem mul_invOfUnit (φ : MvPowerSeries σ R) (u : Rˣ) (h : constantCoeff σ R φ = u) :
    φ * invOfUnit φ u = 1 :=
  ext fun n =>
    letI := Classical.decEq (σ →₀ ℕ)
    if H : n = 0 then by
      /-
        σ : Type u_1
        R : Type u_2
        inst✝ : Ring R
        φ : MvPowerSeries σ R
        u : Units R
        h : Eq ((MvPowerSeries.constantCoeff σ R) φ) ↑u
        n : Finsupp σ Nat
        this : DecidableEq (Finsupp σ Nat) := Classical.decEq (Finsupp σ Nat)
        H : Eq n 0
        ⊢ Eq ((MvPowerSeries.coeff R n) (HMul.hMul φ (φ.invOfUnit u))) ((MvPowerSeries …
      -/
      rw [H]
      /-
        σ : Type u_1
        R : Type u_2
        inst✝ : Ring R
        φ : MvPowerSeries σ R
        u : Units R
        h : Eq ((MvPowerSeries.constantCoeff σ R) φ) ↑u
        n : Finsupp σ Nat
        this : DecidableEq (Finsupp σ Nat) := Classical.decEq (Finsupp σ Nat)
        H : Eq n 0
        ⊢ Eq ((MvPowerSeries.coeff R 0) (HMul.hMul φ (φ.invOfUnit u))) ((MvPowerSeries …
      -/
      simp [coeff_mul, support_single_ne_zero, h]
      /-
        🎉 no goals
      -/
    else by
      classical
      have : ((0 : σ →₀ ℕ), n) ∈ antidiagonal n := by rw [mem_antidiagonal, zero_add]
      rw [coeff_one, if_neg H, coeff_mul, ← Finset.insert_erase this,
        Finset.sum_insert (Finset.not_mem_erase _ _), coeff_zero_eq_constantCoeff_apply, h,
        coeff_invOfUnit, if_neg H, neg_mul, mul_neg, Units.mul_inv_cancel_left, ←
        Finset.insert_erase this, Finset.sum_insert (Finset.not_mem_erase _ _),
        Finset.insert_erase this, if_neg (not_lt_of_ge <| le_rfl), zero_add, add_comm, ←
        sub_eq_add_neg, sub_eq_zero, Finset.sum_congr rfl]
      rintro ⟨i, j⟩ hij
      rw [Finset.mem_erase, mem_antidiagonal] at hij
      cases' hij with h₁ h₂
      subst n
      rw [if_pos]
      suffices (0 : _) + j < i + j by simpa
      apply add_lt_add_right
      constructor
      · intro s
        exact Nat.zero_le _
      · intro H
        apply h₁
        suffices i = 0 by simp [this]
        ext1 s
        exact Nat.eq_zero_of_le_zero (H s)

-- TODO : can one prove equivalence?

@[simp]
theorem invOfUnit_mul (φ : MvPowerSeries σ R) (u : Rˣ) (h : constantCoeff σ R φ = u) :
    invOfUnit φ u * φ = 1 := by
  rw [← mul_cancel_right_mem_nonZeroDivisors (r := φ.invOfUnit u), mul_assoc, one_mul,
    mul_invOfUnit _ _ h, mul_one]
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Ring R
    φ : MvPowerSeries σ R
    u : Units R
    h : Eq ((MvPowerSeries.constantCoeff σ R) φ) ↑u
    ⊢ Membership.mem (nonZeroDivisors (MvPowerSeries σ R)) (φ.invOfUnit u)
  -/
  apply mem_nonZeroDivisors_of_constantCoeff
  /-
    case hφ
    σ : Type u_1
    R : Type u_2
    inst✝ : Ring R
    φ : MvPowerSeries σ R
    u : Units R
    h : Eq ((MvPowerSeries.constantCoeff σ R) φ) ↑u
    ⊢ Membership.mem (nonZeroDivisors R) ((MvPowerSeries.constantCoeff σ R) (φ.inv …
  -/
  simp only [constantCoeff_invOfUnit, IsUnit.mem_nonZeroDivisors (Units.isUnit u⁻¹)]
  /-
    🎉 no goals
  -/


theorem isUnit_iff_constantCoeff {φ : MvPowerSeries σ R} :
    IsUnit φ ↔ IsUnit (constantCoeff σ R φ) := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Ring R
    φ : MvPowerSeries σ R
    ⊢ Iff (IsUnit φ) (IsUnit ((MvPowerSeries.constantCoeff σ R) φ))
  -/
  constructor
    /-
      case mp
      σ : Type u_1
      R : Type u_2
      inst✝ : Ring R
      φ : MvPowerSeries σ R
      ⊢ IsUnit φ → IsUnit ((MvPowerSeries.constantCoeff σ R) φ)
    -/
  · exact IsUnit.map _
    /-
      🎉 no goals
    -/
    /-
      case mpr
      σ : Type u_1
      R : Type u_2
      inst✝ : Ring R
      φ : MvPowerSeries σ R
      ⊢ IsUnit ((MvPowerSeries.constantCoeff σ R) φ) → IsUnit φ
    -/
  · intro ⟨u, hu⟩
    /-
      case mpr
      σ : Type u_1
      R : Type u_2
      inst✝ : Ring R
      φ : MvPowerSeries σ R
      u : Units R
      hu : Eq (↑u) ((MvPowerSeries.constantCoeff σ R) φ)
      ⊢ IsUnit φ
    -/
    exact ⟨⟨_, φ.invOfUnit u, mul_invOfUnit φ u hu.symm, invOfUnit_mul φ u hu.symm⟩, rfl⟩
    /-
      🎉 no goals
    -/


/-- Multivariate formal power series over a local ring form a local ring. -/
instance [IsLocalRing R] : IsLocalRing (MvPowerSeries σ R) :=
  IsLocalRing.of_isUnit_or_isUnit_one_sub_self <| by
    /-
      σ : Type u_1
      R : Type u_2
      inst✝¹ : CommRing R
      inst✝ : IsLocalRing R
      ⊢ ∀ (a : MvPowerSeries σ R), Or (IsUnit a) (IsUnit (HSub.hSub 1 a))
    -/
    intro φ
    obtain ⟨u, h⟩ | ⟨u, h⟩ := IsLocalRing.isUnit_or_isUnit_one_sub_self (constantCoeff σ R φ) <;>
        [left; right] <;>
        /-
          case inl.intro.h
          σ : Type u_1
          R : Type u_2
          inst✝¹ : CommRing R
          inst✝ : IsLocalRing R
          φ : MvPowerSeries σ R
          u : Units R
          h : Eq (↑u) ((MvPowerSeries.constantCoeff σ R) φ)
          ⊢ IsUnit φ
        -/
        /-
          case inl.intro.h
          σ : Type u_1
          R : Type u_2
          inst✝¹ : CommRing R
          inst✝ : IsLocalRing R
          φ : MvPowerSeries σ R
          u : Units R
          h : Eq (↑u) ((MvPowerSeries.constantCoeff σ R) φ)
          ⊢ Eq ((MvPowerSeries.constantCoeff σ R) φ) ↑u
        -/
        /-
          🎉 no goals
        -/
        /-
          case inr.intro.h
          σ : Type u_1
          R : Type u_2
          inst✝¹ : CommRing R
          inst✝ : IsLocalRing R
          φ : MvPowerSeries σ R
          u : Units R
          h : Eq (↑u) (HSub.hSub 1 ((MvPowerSeries.constantCoeff σ R) φ))
          ⊢ Eq ((MvPowerSeries.constantCoeff σ R) (HSub.hSub 1 φ)) ↑u
        -/
        simpa using h.symm
        /-
          🎉 no goals
        -/

-- TODO(jmc): once adic topology lands, show that this is complete

/-- The map between multivariate formal power series over the same indexing set
 induced by a local ring hom `A → B` is local -/
@[instance]
theorem map.isLocalHom : IsLocalHom (map σ f) :=
  ⟨by
    /-
      σ : Type u_1
      R : Type u_2
      S : Type u_3
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      f : RingHom R S
      inst✝ : IsLocalHom f
      ⊢ ∀ (a : MvPowerSeries σ R), IsUnit ((MvPowerSeries.map σ f) a) → IsUnit a
    -/
    rintro φ ⟨ψ, h⟩
    /-
      case intro
      σ : Type u_1
      R : Type u_2
      S : Type u_3
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      f : RingHom R S
      inst✝ : IsLocalHom f
      φ : MvPowerSeries σ R
      ψ : Units (MvPowerSeries σ S)
      h : Eq (↑ψ) ((MvPowerSeries.map σ f) φ)
      ⊢ IsUnit φ
    -/
    replace h := congr_arg (constantCoeff σ S) h
    /-
      case intro
      σ : Type u_1
      R : Type u_2
      S : Type u_3
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      f : RingHom R S
      inst✝ : IsLocalHom f
      φ : MvPowerSeries σ R
      ψ : Units (MvPowerSeries σ S)
      h : Eq ((MvPowerSeries.constantCoeff σ S) ↑ψ) ((MvPowerSeries.constantCoeff σ  …
      ⊢ IsUnit φ
    -/
    rw [constantCoeff_map] at h
    /-
      case intro
      σ : Type u_1
      R : Type u_2
      S : Type u_3
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      f : RingHom R S
      inst✝ : IsLocalHom f
      φ : MvPowerSeries σ R
      ψ : Units (MvPowerSeries σ S)
      h : Eq ((MvPowerSeries.constantCoeff σ S) ↑ψ) (f ((MvPowerSeries.constantCoeff …
      ⊢ IsUnit φ
    -/
    have : IsUnit (constantCoeff σ S ↑ψ) := isUnit_constantCoeff _ ψ.isUnit
    /-
      case intro
      σ : Type u_1
      R : Type u_2
      S : Type u_3
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      f : RingHom R S
      inst✝ : IsLocalHom f
      φ : MvPowerSeries σ R
      ψ : Units (MvPowerSeries σ S)
      h : Eq ((MvPowerSeries.constantCoeff σ S) ↑ψ) (f ((MvPowerSeries.constantCoeff …
      this : IsUnit ((MvPowerSeries.constantCoeff σ S) ↑ψ)
      ⊢ IsUnit φ
    -/
    rw [h] at this
    /-
      case intro
      σ : Type u_1
      R : Type u_2
      S : Type u_3
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      f : RingHom R S
      inst✝ : IsLocalHom f
      φ : MvPowerSeries σ R
      ψ : Units (MvPowerSeries σ S)
      h : Eq ((MvPowerSeries.constantCoeff σ S) ↑ψ) (f ((MvPowerSeries.constantCoeff …
      this : IsUnit (f ((MvPowerSeries.constantCoeff σ R) φ))
      ⊢ IsUnit φ
    -/
    rcases isUnit_of_map_unit f _ this with ⟨c, hc⟩
    /-
      case intro.intro
      σ : Type u_1
      R : Type u_2
      S : Type u_3
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      f : RingHom R S
      inst✝ : IsLocalHom f
      φ : MvPowerSeries σ R
      ψ : Units (MvPowerSeries σ S)
      h : Eq ((MvPowerSeries.constantCoeff σ S) ↑ψ) (f ((MvPowerSeries.constantCoeff …
      this : IsUnit (f ((MvPowerSeries.constantCoeff σ R) φ))
      c : Units R
      hc : Eq (↑c) ((MvPowerSeries.constantCoeff σ R) φ)
      ⊢ IsUnit φ
    -/
    exact isUnit_of_mul_eq_one φ (invOfUnit φ c) (mul_invOfUnit φ c hc.symm)⟩
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-10")]
alias map.isLocalRingHom := map.isLocalHom


/-- The inverse `1/f` of a multivariable power series `f` over a field -/
protected def inv (φ : MvPowerSeries σ k) : MvPowerSeries σ k :=
  inv.aux (constantCoeff σ k φ)⁻¹ φ


instance : Inv (MvPowerSeries σ k) :=
  ⟨MvPowerSeries.inv⟩


theorem coeff_inv [DecidableEq σ] (n : σ →₀ ℕ) (φ : MvPowerSeries σ k) :
    coeff k n φ⁻¹ =
      if n = 0 then (constantCoeff σ k φ)⁻¹
      else
        -(constantCoeff σ k φ)⁻¹ *
          ∑ x ∈ antidiagonal n, if x.2 < n then coeff k x.1 φ * coeff k x.2 φ⁻¹ else 0 :=
  coeff_inv_aux n _ φ


@[simp]
theorem constantCoeff_inv (φ : MvPowerSeries σ k) :
    constantCoeff σ k φ⁻¹ = (constantCoeff σ k φ)⁻¹ := by
  classical
  rw [← coeff_zero_eq_constantCoeff_apply, coeff_inv, if_pos rfl]


theorem inv_eq_zero {φ : MvPowerSeries σ k} : φ⁻¹ = 0 ↔ constantCoeff σ k φ = 0 :=
               /-
                 σ : Type u_1
                 k : Type u_3
                 inst✝ : Field k
                 φ : MvPowerSeries σ k
                 h : Eq (Inv.inv φ) 0
                 ⊢ Eq ((MvPowerSeries.constantCoeff σ k) φ) 0
               -/
  ⟨fun h => by simpa using congr_arg (constantCoeff σ k) h, fun h =>
               /-
                 🎉 no goals
               -/
    ext fun n => by
      classical
      rw [coeff_inv]
      split_ifs <;>
        simp only [h, map_zero, zero_mul, inv_zero, neg_zero]⟩


@[simp]
theorem zero_inv : (0 : MvPowerSeries σ k)⁻¹ = 0 := by
  /-
    σ : Type u_1
    k : Type u_3
    inst✝ : Field k
    ⊢ Eq (Inv.inv 0) 0
  -/
  rw [inv_eq_zero, constantCoeff_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem invOfUnit_eq (φ : MvPowerSeries σ k) (h : constantCoeff σ k φ ≠ 0) :
    invOfUnit φ (Units.mk0 _ h) = φ⁻¹ :=
  rfl


@[simp]
theorem invOfUnit_eq' (φ : MvPowerSeries σ k) (u : Units k) (h : constantCoeff σ k φ = u) :
    invOfUnit φ u = φ⁻¹ := by
  /-
    σ : Type u_1
    k : Type u_3
    inst✝ : Field k
    φ : MvPowerSeries σ k
    u : Units k
    h : Eq ((MvPowerSeries.constantCoeff σ k) φ) ↑u
    ⊢ Eq (φ.invOfUnit u) (Inv.inv φ)
  -/
  rw [← invOfUnit_eq φ (h.symm ▸ u.ne_zero)]
  /-
    σ : Type u_1
    k : Type u_3
    inst✝ : Field k
    φ : MvPowerSeries σ k
    u : Units k
    h : Eq ((MvPowerSeries.constantCoeff σ k) φ) ↑u
    ⊢ Eq (φ.invOfUnit u) (φ.invOfUnit (Units.mk0 ((MvPowerSeries.constantCoeff σ k …
  -/
  apply congrArg (invOfUnit φ)
  /-
    σ : Type u_1
    k : Type u_3
    inst✝ : Field k
    φ : MvPowerSeries σ k
    u : Units k
    h : Eq ((MvPowerSeries.constantCoeff σ k) φ) ↑u
    ⊢ Eq u (Units.mk0 ((MvPowerSeries.constantCoeff σ k) φ) ⋯)
  -/
  rw [Units.ext_iff]
  /-
    σ : Type u_1
    k : Type u_3
    inst✝ : Field k
    φ : MvPowerSeries σ k
    u : Units k
    h : Eq ((MvPowerSeries.constantCoeff σ k) φ) ↑u
    ⊢ Eq ↑u ↑(Units.mk0 ((MvPowerSeries.constantCoeff σ k) φ) ⋯)
  -/
  exact h.symm
  /-
    🎉 no goals
  -/


@[simp]
protected theorem mul_inv_cancel (φ : MvPowerSeries σ k) (h : constantCoeff σ k φ ≠ 0) :
                      /-
                        σ : Type u_1
                        k : Type u_3
                        inst✝ : Field k
                        φ : MvPowerSeries σ k
                        h : Ne ((MvPowerSeries.constantCoeff σ k) φ) 0
                        ⊢ Eq (HMul.hMul φ (Inv.inv φ)) 1
                      -/
    φ * φ⁻¹ = 1 := by rw [← invOfUnit_eq φ h, mul_invOfUnit φ (Units.mk0 _ h) rfl]
                      /-
                        🎉 no goals
                      -/


@[simp]
protected theorem inv_mul_cancel (φ : MvPowerSeries σ k) (h : constantCoeff σ k φ ≠ 0) :
                      /-
                        σ : Type u_1
                        k : Type u_3
                        inst✝ : Field k
                        φ : MvPowerSeries σ k
                        h : Ne ((MvPowerSeries.constantCoeff σ k) φ) 0
                        ⊢ Eq (HMul.hMul (Inv.inv φ) φ) 1
                      -/
    φ⁻¹ * φ = 1 := by rw [mul_comm, φ.mul_inv_cancel h]
                      /-
                        🎉 no goals
                      -/


protected theorem eq_mul_inv_iff_mul_eq {φ₁ φ₂ φ₃ : MvPowerSeries σ k}
    (h : constantCoeff σ k φ₃ ≠ 0) : φ₁ = φ₂ * φ₃⁻¹ ↔ φ₁ * φ₃ = φ₂ :=
               /-
                 σ : Type u_1
                 k✝ : Type u_3
                 inst✝ : Field k✝
                 φ₁ φ₂ φ₃ : MvPowerSeries σ k✝
                 h : Ne ((MvPowerSeries.constantCoeff σ k✝) φ₃) 0
                 k : Eq φ₁ (HMul.hMul φ₂ (Inv.inv φ₃))
                 ⊢ Eq (HMul.hMul φ₁ φ₃) φ₂
               -/
  ⟨fun k => by simp [k, mul_assoc, MvPowerSeries.inv_mul_cancel _ h], fun k => by
               /-
                 🎉 no goals
               -/
    /-
      σ : Type u_1
      k✝ : Type u_3
      inst✝ : Field k✝
      φ₁ φ₂ φ₃ : MvPowerSeries σ k✝
      h : Ne ((MvPowerSeries.constantCoeff σ k✝) φ₃) 0
      k : Eq (HMul.hMul φ₁ φ₃) φ₂
      ⊢ Eq φ₁ (HMul.hMul φ₂ (Inv.inv φ₃))
    -/
    simp [← k, mul_assoc, MvPowerSeries.mul_inv_cancel _ h]⟩
    /-
      🎉 no goals
    -/


protected theorem eq_inv_iff_mul_eq_one {φ ψ : MvPowerSeries σ k} (h : constantCoeff σ k ψ ≠ 0) :
                              /-
                                σ : Type u_1
                                k : Type u_3
                                inst✝ : Field k
                                φ ψ : MvPowerSeries σ k
                                h : Ne ((MvPowerSeries.constantCoeff σ k) ψ) 0
                                ⊢ Iff (Eq φ (Inv.inv ψ)) (Eq (HMul.hMul φ ψ) 1)
                              -/
    φ = ψ⁻¹ ↔ φ * ψ = 1 := by rw [← MvPowerSeries.eq_mul_inv_iff_mul_eq h, one_mul]
                              /-
                                🎉 no goals
                              -/


protected theorem inv_eq_iff_mul_eq_one {φ ψ : MvPowerSeries σ k} (h : constantCoeff σ k ψ ≠ 0) :
                              /-
                                σ : Type u_1
                                k : Type u_3
                                inst✝ : Field k
                                φ ψ : MvPowerSeries σ k
                                h : Ne ((MvPowerSeries.constantCoeff σ k) ψ) 0
                                ⊢ Iff (Eq (Inv.inv ψ) φ) (Eq (HMul.hMul φ ψ) 1)
                              -/
    ψ⁻¹ = φ ↔ φ * ψ = 1 := by rw [eq_comm, MvPowerSeries.eq_inv_iff_mul_eq_one h]
                              /-
                                🎉 no goals
                              -/


@[simp]
protected theorem mul_inv_rev (φ ψ : MvPowerSeries σ k) :
    (φ * ψ)⁻¹ = ψ⁻¹ * φ⁻¹ := by
  /-
    σ : Type u_1
    k : Type u_3
    inst✝ : Field k
    φ ψ : MvPowerSeries σ k
    ⊢ Eq (Inv.inv (HMul.hMul φ ψ)) (HMul.hMul (Inv.inv ψ) (Inv.inv φ))
  -/
  by_cases h : constantCoeff σ k (φ * ψ) = 0
    /-
      case pos
      σ : Type u_1
      k : Type u_3
      inst✝ : Field k
      φ ψ : MvPowerSeries σ k
      h : Eq ((MvPowerSeries.constantCoeff σ k) (HMul.hMul φ ψ)) 0
      ⊢ Eq (Inv.inv (HMul.hMul φ ψ)) (HMul.hMul (Inv.inv ψ) (Inv.inv φ))
    -/
  · rw [inv_eq_zero.mpr h]
    /-
      case pos
      σ : Type u_1
      k : Type u_3
      inst✝ : Field k
      φ ψ : MvPowerSeries σ k
      h : Eq ((MvPowerSeries.constantCoeff σ k) (HMul.hMul φ ψ)) 0
      ⊢ Eq 0 (HMul.hMul (Inv.inv ψ) (Inv.inv φ))
    -/
    simp only [map_mul, mul_eq_zero] at h
    -- we don't have `NoZeroDivisors (MvPowerSeries σ k)` yet,
    /-
      case pos
      σ : Type u_1
      k : Type u_3
      inst✝ : Field k
      φ ψ : MvPowerSeries σ k
      h : Or (Eq ((MvPowerSeries.constantCoeff σ k) φ) 0) (Eq ((MvPowerSeries.consta …
      ⊢ Eq 0 (HMul.hMul (Inv.inv ψ) (Inv.inv φ))
    -/
                          /-
                            🎉 no goals
                          -/
    cases' h with h h <;> simp [inv_eq_zero.mpr h]
                          /-
                            🎉 no goals
                          -/
    /-
      case neg
      σ : Type u_1
      k : Type u_3
      inst✝ : Field k
      φ ψ : MvPowerSeries σ k
      h : Not (Eq ((MvPowerSeries.constantCoeff σ k) (HMul.hMul φ ψ)) 0)
      ⊢ Eq (Inv.inv (HMul.hMul φ ψ)) (HMul.hMul (Inv.inv ψ) (Inv.inv φ))
    -/
  · rw [MvPowerSeries.inv_eq_iff_mul_eq_one h]
    /-
      case neg
      σ : Type u_1
      k : Type u_3
      inst✝ : Field k
      φ ψ : MvPowerSeries σ k
      h : Not (Eq ((MvPowerSeries.constantCoeff σ k) (HMul.hMul φ ψ)) 0)
      ⊢ Eq (HMul.hMul (HMul.hMul (Inv.inv ψ) (Inv.inv φ)) (HMul.hMul φ ψ)) 1
    -/
    simp only [not_or, map_mul, mul_eq_zero] at h
    rw [← mul_assoc, mul_assoc _⁻¹, MvPowerSeries.inv_mul_cancel _ h.left, mul_one,
      MvPowerSeries.inv_mul_cancel _ h.right]


instance : InvOneClass (MvPowerSeries σ k) :=
  { inferInstanceAs (One (MvPowerSeries σ k)),
    inferInstanceAs (Inv (MvPowerSeries σ k)) with
    inv_one := by
      /-
        σ : Type u_1
        R : Type u_2
        k : Type u_3
        inst✝ : Field k
        ⊢ Eq (Inv.inv 1) 1
      -/
      rw [MvPowerSeries.inv_eq_iff_mul_eq_one, mul_one]
      /-
        σ : Type u_1
        R : Type u_2
        k : Type u_3
        inst✝ : Field k
        ⊢ Ne ((MvPowerSeries.constantCoeff σ k) 1) 0
      -/
      simp }
      /-
        🎉 no goals
      -/


@[simp]
theorem C_inv (r : k) : (C σ k r)⁻¹ = C σ k r⁻¹ := by
  /-
    σ : Type u_1
    k : Type u_3
    inst✝ : Field k
    r : k
    ⊢ Eq (Inv.inv ((MvPowerSeries.C σ k) r)) ((MvPowerSeries.C σ k) (Inv.inv r))
  -/
  rcases eq_or_ne r 0 with (rfl | hr)
    /-
      case inl
      σ : Type u_1
      k : Type u_3
      inst✝ : Field k
      ⊢ Eq (Inv.inv ((MvPowerSeries.C σ k) 0)) ((MvPowerSeries.C σ k) (Inv.inv 0))
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    σ : Type u_1
    k : Type u_3
    inst✝ : Field k
    r : k
    hr : Ne r 0
    ⊢ Eq (Inv.inv ((MvPowerSeries.C σ k) r)) ((MvPowerSeries.C σ k) (Inv.inv r))
  -/
  rw [MvPowerSeries.inv_eq_iff_mul_eq_one, ← map_mul, inv_mul_cancel₀ hr, map_one]
  /-
    case inr
    σ : Type u_1
    k : Type u_3
    inst✝ : Field k
    r : k
    hr : Ne r 0
    ⊢ Ne ((MvPowerSeries.constantCoeff σ k) ((MvPowerSeries.C σ k) r)) 0
  -/
  simpa using hr
  /-
    🎉 no goals
  -/


@[simp]
theorem X_inv (s : σ) : (X s : MvPowerSeries σ k)⁻¹ = 0 := by
  /-
    σ : Type u_1
    k : Type u_3
    inst✝ : Field k
    s : σ
    ⊢ Eq (Inv.inv (MvPowerSeries.X s)) 0
  -/
  rw [inv_eq_zero, constantCoeff_X]
  /-
    🎉 no goals
  -/


@[simp]
theorem smul_inv (r : k) (φ : MvPowerSeries σ k) : (r • φ)⁻¹ = r⁻¹ • φ⁻¹ := by
  /-
    σ : Type u_1
    k : Type u_3
    inst✝ : Field k
    r : k
    φ : MvPowerSeries σ k
    ⊢ Eq (Inv.inv (HSMul.hSMul r φ)) (HSMul.hSMul (Inv.inv r) (Inv.inv φ))
  -/
  simp [smul_eq_C_mul, mul_comm]
  /-
    🎉 no goals
  -/


