@[simp] lemma zero_mem_center : (0 : M₀) ∈ center M₀ where
               /-
                 M₀ : Type u_1
                 inst✝ : MulZeroClass M₀
                 x✝ : M₀
                 ⊢ Eq (HMul.hMul 0 x✝) (HMul.hMul x✝ 0)
               -/
  comm _ := by rw [zero_mul, mul_zero]
               /-
                 🎉 no goals
               -/
                       /-
                         M₀ : Type u_1
                         inst✝ : MulZeroClass M₀
                         x✝¹ x✝ : M₀
                         ⊢ Eq (HMul.hMul 0 (HMul.hMul x✝¹ x✝)) (HMul.hMul (HMul.hMul 0 x✝¹) x✝)
                       -/
  left_assoc _ _ := by rw [zero_mul, zero_mul, zero_mul]
                       /-
                         🎉 no goals
                       -/
                      /-
                        M₀ : Type u_1
                        inst✝ : MulZeroClass M₀
                        x✝¹ x✝ : M₀
                        ⊢ Eq (HMul.hMul (HMul.hMul x✝¹ 0) x✝) (HMul.hMul x✝¹ (HMul.hMul 0 x✝))
                      -/
  mid_assoc _ _ := by rw [mul_zero, zero_mul, mul_zero]
                      /-
                        🎉 no goals
                      -/
                        /-
                          M₀ : Type u_1
                          inst✝ : MulZeroClass M₀
                          x✝¹ x✝ : M₀
                          ⊢ Eq (HMul.hMul (HMul.hMul x✝¹ x✝) 0) (HMul.hMul x✝¹ (HMul.hMul x✝ 0))
                        -/
  right_assoc _ _ := by rw [mul_zero, mul_zero, mul_zero]
                        /-
                          🎉 no goals
                        -/


                                                                    /-
                                                                      M₀ : Type u_1
                                                                      inst✝ : MulZeroClass M₀
                                                                      s : Set M₀
                                                                      ⊢ Membership.mem s.centralizer 0
                                                                    -/
@[simp] lemma zero_mem_centralizer : (0 : M₀) ∈ centralizer s := by simp [mem_centralizer_iff]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


lemma center_units_subset : center G₀ˣ ⊆ ((↑) : G₀ˣ → G₀) ⁻¹' center G₀ := by
  /-
    G₀ : Type u_2
    inst✝ : GroupWithZero G₀
    ⊢ HasSubset.Subset (Set.center (Units G₀)) (Set.preimage Units.val (Set.center …
  -/
  simp_rw [subset_def, mem_preimage, _root_.Semigroup.mem_center_iff]
  /-
    G₀ : Type u_2
    inst✝ : GroupWithZero G₀
    ⊢ ∀ (x : Units G₀), (∀ (g : Units G₀), Eq (HMul.hMul g x) (HMul.hMul x g)) → ∀ …
  -/
  intro u hu a
  /-
    G₀ : Type u_2
    inst✝ : GroupWithZero G₀
    u : Units G₀
    hu : ∀ (g : Units G₀), Eq (HMul.hMul g u) (HMul.hMul u g)
    a : G₀
    ⊢ Eq (HMul.hMul a ↑u) (HMul.hMul (↑u) a)
  -/
  obtain rfl | ha := eq_or_ne a 0
    /-
      case inl
      G₀ : Type u_2
      inst✝ : GroupWithZero G₀
      u : Units G₀
      hu : ∀ (g : Units G₀), Eq (HMul.hMul g u) (HMul.hMul u g)
      ⊢ Eq (HMul.hMul 0 ↑u) (HMul.hMul (↑u) 0)
    -/
  · rw [zero_mul, mul_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr
      G₀ : Type u_2
      inst✝ : GroupWithZero G₀
      u : Units G₀
      hu : ∀ (g : Units G₀), Eq (HMul.hMul g u) (HMul.hMul u g)
      a : G₀
      ha : Ne a 0
      ⊢ Eq (HMul.hMul a ↑u) (HMul.hMul (↑u) a)
    -/
  · exact congr_arg Units.val <| hu <| Units.mk0 a ha
    /-
      🎉 no goals
    -/


/-- In a group with zero, the center of the units is the preimage of the center. -/
lemma center_units_eq : center G₀ˣ = ((↑) : G₀ˣ → G₀) ⁻¹' center G₀ :=
  center_units_subset.antisymm subset_center_units


@[simp] lemma inv_mem_centralizer₀ (ha : a ∈ centralizer s) : a⁻¹ ∈ centralizer s := by
  /-
    G₀ : Type u_2
    inst✝ : GroupWithZero G₀
    s : Set G₀
    a : G₀
    ha : Membership.mem s.centralizer a
    ⊢ Membership.mem s.centralizer (Inv.inv a)
  -/
  obtain rfl | ha₀ := eq_or_ne a 0
    /-
      case inl
      G₀ : Type u_2
      inst✝ : GroupWithZero G₀
      s : Set G₀
      ha : Membership.mem s.centralizer 0
      ⊢ Membership.mem s.centralizer (Inv.inv 0)
    -/
  · rw [inv_zero]
    /-
      case inl
      G₀ : Type u_2
      inst✝ : GroupWithZero G₀
      s : Set G₀
      ha : Membership.mem s.centralizer 0
      ⊢ Membership.mem s.centralizer 0
    -/
    exact zero_mem_centralizer
    /-
      🎉 no goals
    -/
    /-
      case inr
      G₀ : Type u_2
      inst✝ : GroupWithZero G₀
      s : Set G₀
      a : G₀
      ha : Membership.mem s.centralizer a
      ha₀ : Ne a 0
      ⊢ Membership.mem s.centralizer (Inv.inv a)
    -/
  · rintro c hc
    /-
      case inr
      G₀ : Type u_2
      inst✝ : GroupWithZero G₀
      s : Set G₀
      a : G₀
      ha : Membership.mem s.centralizer a
      ha₀ : Ne a 0
      c : G₀
      hc : Membership.mem s c
      ⊢ Eq (HMul.hMul c (Inv.inv a)) (HMul.hMul (Inv.inv a) c)
    -/
    rw [mul_inv_eq_iff_eq_mul₀ ha₀, mul_assoc, eq_inv_mul_iff_mul_eq₀ ha₀, ha c hc]
    /-
      🎉 no goals
    -/


@[simp] lemma div_mem_centralizer₀ (ha : a ∈ centralizer s) (hb : b ∈ centralizer s) :
    a / b ∈ centralizer s := by
  /-
    G₀ : Type u_2
    inst✝ : GroupWithZero G₀
    s : Set G₀
    a b : G₀
    ha : Membership.mem s.centralizer a
    hb : Membership.mem s.centralizer b
    ⊢ Membership.mem s.centralizer (HDiv.hDiv a b)
  -/
  simpa only [div_eq_mul_inv] using mul_mem_centralizer ha (inv_mem_centralizer₀ hb)
  /-
    🎉 no goals
  -/


@[deprecated inv_mem_center (since := "2024-06-17")]
theorem inv_mem_center₀ (ha : a ∈ Set.center G₀) : a⁻¹ ∈ Set.center G₀ :=
  inv_mem_center ha


@[deprecated div_mem_center (since := "2024-06-17")]
theorem div_mem_center₀ (ha : a ∈ Set.center G₀) (hb : b ∈ Set.center G₀) : a / b ∈ Set.center G₀ :=
  div_mem_center ha hb


