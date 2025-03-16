/-- A subgroup `H` of `G` is *saturated* if for all `n : ℕ` and `g : G` with `g^n ∈ H`
we have `n = 0` or `g ∈ H`. -/
@[to_additive
      "An additive subgroup `H` of `G` is *saturated* if for all `n : ℕ` and `g : G` with `n•g ∈ H`
      we have `n = 0` or `g ∈ H`."]
def Saturated (H : Subgroup G) : Prop :=
  ∀ ⦃n g⦄, g ^ n ∈ H → n = 0 ∨ g ∈ H


@[to_additive]
theorem saturated_iff_npow {H : Subgroup G} :
    Saturated H ↔ ∀ (n : ℕ) (g : G), g ^ n ∈ H → n = 0 ∨ g ∈ H :=
  Iff.rfl


@[to_additive]
theorem saturated_iff_zpow {H : Subgroup G} :
    Saturated H ↔ ∀ (n : ℤ) (g : G), g ^ n ∈ H → n = 0 ∨ g ∈ H := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    ⊢ Iff H.Saturated (∀ (n : Int) (g : G), Membership.mem H (HPow.hPow g n) → Or  …
  -/
  constructor
    /-
      case mp
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      ⊢ H.Saturated → ∀ (n : Int) (g : G), Membership.mem H (HPow.hPow g n) → Or (Eq …
    -/
  · intros hH n g hgn
    /-
      case mp
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      hH : H.Saturated
      n : Int
      g : G
      hgn : Membership.mem H (HPow.hPow g n)
      ⊢ Or (Eq n 0) (Membership.mem H g)
    -/
    induction' n with n n
      /-
        case mp.ofNat
        G : Type u_1
        inst✝ : Group G
        H : Subgroup G
        hH : H.Saturated
        g : G
        n : Nat
        hgn : Membership.mem H (HPow.hPow g (Int.ofNat n))
        ⊢ Or (Eq (Int.ofNat n) 0) (Membership.mem H g)
      -/
    · simp only [Int.natCast_eq_zero, Int.ofNat_eq_coe, zpow_natCast] at hgn ⊢
      /-
        case mp.ofNat
        G : Type u_1
        inst✝ : Group G
        H : Subgroup G
        hH : H.Saturated
        g : G
        n : Nat
        hgn : Membership.mem H (HPow.hPow g n)
        ⊢ Or (Eq n 0) (Membership.mem H g)
      -/
      exact hH hgn
      /-
        🎉 no goals
      -/
    · suffices g ^ (n + 1) ∈ H by
        refine (hH this).imp ?_ id
        simp only [IsEmpty.forall_iff, Nat.succ_ne_zero]
      /-
        case mp.negSucc
        G : Type u_1
        inst✝ : Group G
        H : Subgroup G
        hH : H.Saturated
        g : G
        n : Nat
        hgn : Membership.mem H (HPow.hPow g (Int.negSucc n))
        ⊢ Membership.mem H (HPow.hPow g (HAdd.hAdd n 1))
      -/
      simpa only [inv_mem_iff, zpow_negSucc] using hgn
      /-
        🎉 no goals
      -/
    /-
      case mpr
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      ⊢ (∀ (n : Int) (g : G), Membership.mem H (HPow.hPow g n) → Or (Eq n 0) (Member …
    -/
  · intro h n g hgn
    /-
      case mpr
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      h : ∀ (n : Int) (g : G), Membership.mem H (HPow.hPow g n) → Or (Eq n 0) (Membe …
      n : Nat
      g : G
      hgn : Membership.mem H (HPow.hPow g n)
      ⊢ Or (Eq n 0) (Membership.mem H g)
    -/
    specialize h n g
    /-
      case mpr
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      n : Nat
      g : G
      hgn : Membership.mem H (HPow.hPow g n)
      h : Membership.mem H (HPow.hPow g ↑n) → Or (Eq (↑n) 0) (Membership.mem H g)
      ⊢ Or (Eq n 0) (Membership.mem H g)
    -/
    simp only [Int.natCast_eq_zero, zpow_natCast] at h
    /-
      case mpr
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      n : Nat
      g : G
      hgn : Membership.mem H (HPow.hPow g n)
      h : Membership.mem H (HPow.hPow g n) → Or (Eq n 0) (Membership.mem H g)
      ⊢ Or (Eq n 0) (Membership.mem H g)
    -/
    apply h hgn
    /-
      🎉 no goals
    -/


theorem ker_saturated {A₁ A₂ : Type*} [AddCommGroup A₁] [AddCommGroup A₂] [NoZeroSMulDivisors ℕ A₂]
    (f : A₁ →+ A₂) : f.ker.Saturated := by
  /-
    A₁ : Type u_1
    A₂ : Type u_2
    inst✝² : AddCommGroup A₁
    inst✝¹ : AddCommGroup A₂
    inst✝ : NoZeroSMulDivisors Nat A₂
    f : AddMonoidHom A₁ A₂
    ⊢ f.ker.Saturated
  -/
  intro n g hg
  /-
    A₁ : Type u_1
    A₂ : Type u_2
    inst✝² : AddCommGroup A₁
    inst✝¹ : AddCommGroup A₂
    inst✝ : NoZeroSMulDivisors Nat A₂
    f : AddMonoidHom A₁ A₂
    n : Nat
    g : A₁
    hg : Membership.mem f.ker (HSMul.hSMul n g)
    ⊢ Or (Eq n 0) (Membership.mem f.ker g)
  -/
  simpa only [f.mem_ker, nsmul_eq_smul, f.map_nsmul, smul_eq_zero] using hg
  /-
    🎉 no goals
  -/


