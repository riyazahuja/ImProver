local infixl:50 " ~ᵤ " => Associated


/-- `toGCDMonoid` constructs a GCD monoid out of a unique factorization domain. -/
noncomputable def UniqueFactorizationMonoid.toGCDMonoid (α : Type*) [CancelCommMonoidWithZero α]
    [UniqueFactorizationMonoid α] : GCDMonoid α where
  gcd a b := Quot.out (Associates.mk a ⊓ Associates.mk b : Associates α)
  lcm a b := Quot.out (Associates.mk a ⊔ Associates.mk b : Associates α)
  gcd_dvd_left a b := by
    /-
      α✝ : Type u_1
      α : Type u_2
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a b : α
      ⊢ Dvd.dvd ((fun a b => Quot.out (Min.min (Associates.mk a) (Associates.mk b))) …
    -/
    rw [← mk_dvd_mk, Associates.quot_out, congr_fun₂ dvd_eq_le]
    /-
      α✝ : Type u_1
      α : Type u_2
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a b : α
      ⊢ LE.le (Min.min (Associates.mk a) (Associates.mk b)) (Associates.mk a)
    -/
    exact inf_le_left
    /-
      🎉 no goals
    -/
  gcd_dvd_right a b := by
    /-
      α✝ : Type u_1
      α : Type u_2
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a b : α
      ⊢ Dvd.dvd ((fun a b => Quot.out (Min.min (Associates.mk a) (Associates.mk b))) …
    -/
    rw [← mk_dvd_mk, Associates.quot_out, congr_fun₂ dvd_eq_le]
    /-
      α✝ : Type u_1
      α : Type u_2
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a b : α
      ⊢ LE.le (Min.min (Associates.mk a) (Associates.mk b)) (Associates.mk b)
    -/
    exact inf_le_right
    /-
      🎉 no goals
    -/
  dvd_gcd {a b c} hac hab := by
    rw [← mk_dvd_mk, Associates.quot_out, congr_fun₂ dvd_eq_le, le_inf_iff,
      mk_le_mk_iff_dvd, mk_le_mk_iff_dvd]
    /-
      α✝ : Type u_1
      α : Type u_2
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a b c : α
      hac : Dvd.dvd a c
      hab : Dvd.dvd a b
      ⊢ And (Dvd.dvd a c) (Dvd.dvd a b)
    -/
    exact ⟨hac, hab⟩
    /-
      🎉 no goals
    -/
                        /-
                          α✝ : Type u_1
                          α : Type u_2
                          inst✝¹ : CancelCommMonoidWithZero α
                          inst✝ : UniqueFactorizationMonoid α
                          a : α
                          ⊢ Eq ((fun a b => Quot.out (Max.max (Associates.mk a) (Associates.mk b))) 0 a) 0
                        -/
  lcm_zero_left a := by simp
                        /-
                          🎉 no goals
                        -/
                         /-
                           α✝ : Type u_1
                           α : Type u_2
                           inst✝¹ : CancelCommMonoidWithZero α
                           inst✝ : UniqueFactorizationMonoid α
                           a : α
                           ⊢ Eq ((fun a b => Quot.out (Max.max (Associates.mk a) (Associates.mk b))) a 0) 0
                         -/
  lcm_zero_right a := by simp
                         /-
                           🎉 no goals
                         -/
  gcd_mul_lcm a b := by
    rw [← mk_eq_mk_iff_associated, ← Associates.mk_mul_mk, ← associated_iff_eq, Associates.quot_out,
      Associates.quot_out, mul_comm, sup_mul_inf, Associates.mk_mul_mk]


/-- `toNormalizedGCDMonoid` constructs a GCD monoid out of a normalization on a
  unique factorization domain. -/
noncomputable def UniqueFactorizationMonoid.toNormalizedGCDMonoid (α : Type*)
    [CancelCommMonoidWithZero α] [UniqueFactorizationMonoid α] [NormalizationMonoid α] :
    NormalizedGCDMonoid α :=
  { ‹NormalizationMonoid α› with
    gcd := fun a b => (Associates.mk a ⊓ Associates.mk b).out
    lcm := fun a b => (Associates.mk a ⊔ Associates.mk b).out
    gcd_dvd_left := fun a b => (out_dvd_iff a (Associates.mk a ⊓ Associates.mk b)).2 <| inf_le_left
    gcd_dvd_right := fun a b =>
      (out_dvd_iff b (Associates.mk a ⊓ Associates.mk b)).2 <| inf_le_right
    dvd_gcd := fun {a} {b} {c} hac hab =>
      show a ∣ (Associates.mk c ⊓ Associates.mk b).out by
        /-
          α✝ : Type u_1
          α : Type u_2
          inst✝² : CancelCommMonoidWithZero α
          inst✝¹ : UniqueFactorizationMonoid α
          inst✝ : NormalizationMonoid α
          a b c : α
          hac : Dvd.dvd a c
          hab : Dvd.dvd a b
          ⊢ Dvd.dvd a (Min.min (Associates.mk c) (Associates.mk b)).out
        -/
        rw [dvd_out_iff, le_inf_iff, mk_le_mk_iff_dvd, mk_le_mk_iff_dvd]
        /-
          α✝ : Type u_1
          α : Type u_2
          inst✝² : CancelCommMonoidWithZero α
          inst✝¹ : UniqueFactorizationMonoid α
          inst✝ : NormalizationMonoid α
          a b c : α
          hac : Dvd.dvd a c
          hab : Dvd.dvd a b
          ⊢ And (Dvd.dvd a c) (Dvd.dvd a b)
        -/
        exact ⟨hac, hab⟩
        /-
          🎉 no goals
        -/
                                                                    /-
                                                                      α✝ : Type u_1
                                                                      α : Type u_2
                                                                      inst✝² : CancelCommMonoidWithZero α
                                                                      inst✝¹ : UniqueFactorizationMonoid α
                                                                      inst✝ : NormalizationMonoid α
                                                                      a : α
                                                                      ⊢ Eq (Max.max Top.top (Associates.mk a)).out 0
                                                                    -/
    lcm_zero_left := fun a => show (⊤ ⊔ Associates.mk a).out = 0 by simp
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
      /-
        α✝ : Type u_1
        α : Type u_2
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : UniqueFactorizationMonoid α
        inst✝ : NormalizationMonoid α
        a b : α
        ⊢ Associated (HMul.hMul ((fun a b => (Min.min (Associates.mk a) (Associates.mk …
      -/
                                                                     /-
                                                                       α✝ : Type u_1
                                                                       α : Type u_2
                                                                       inst✝² : CancelCommMonoidWithZero α
                                                                       inst✝¹ : UniqueFactorizationMonoid α
                                                                       inst✝ : NormalizationMonoid α
                                                                       a : α
                                                                       ⊢ Eq (Max.max (Associates.mk a) Top.top).out 0
                                                                     -/
      /-
        α✝ : Type u_1
        α : Type u_2
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : UniqueFactorizationMonoid α
        inst✝ : NormalizationMonoid α
        a b : α
        ⊢ Associated (normalize (HMul.hMul a b)) (HMul.hMul a b)
      -/
    lcm_zero_right := fun a => show (Associates.mk a ⊔ ⊤).out = 0 by simp
      /-
        🎉 no goals
      -/
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
    gcd_mul_lcm := fun a b => by
      rw [← out_mul, mul_comm, sup_mul_inf, mk_mul_mk, out_mk]
      exact normalize_associated (a * b)
                                   /-
                                     α✝ : Type u_1
                                     α : Type u_2
                                     inst✝² : CancelCommMonoidWithZero α
                                     inst✝¹ : UniqueFactorizationMonoid α
                                     inst✝ : NormalizationMonoid α
                                     a b : α
                                     ⊢ Eq (normalize (GCDMonoid.gcd a b)) (GCDMonoid.gcd a b)
                                   -/
    normalize_gcd := fun a b => by apply normalize_out _
                                   /-
                                     🎉 no goals
                                   -/
                                   /-
                                     α✝ : Type u_1
                                     α : Type u_2
                                     inst✝² : CancelCommMonoidWithZero α
                                     inst✝¹ : UniqueFactorizationMonoid α
                                     inst✝ : NormalizationMonoid α
                                     a b : α
                                     ⊢ Eq (normalize (GCDMonoid.lcm a b)) (GCDMonoid.lcm a b)
                                   -/
    normalize_lcm := fun a b => by apply normalize_out _ }
                                   /-
                                     🎉 no goals
                                   -/


instance (α) [CancelCommMonoidWithZero α] [UniqueFactorizationMonoid α] :
    Nonempty (NormalizedGCDMonoid α) := by
  /-
    α✝ : Type u_1
    α : Type u_2
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    ⊢ Nonempty (NormalizedGCDMonoid α)
  -/
  letI := UniqueFactorizationMonoid.normalizationMonoid (α := α)
  /-
    α✝ : Type u_1
    α : Type u_2
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    this : NormalizationMonoid α := UniqueFactorizationMonoid.normalizationMonoid
    ⊢ Nonempty (NormalizedGCDMonoid α)
  -/
  classical exact ⟨UniqueFactorizationMonoid.toNormalizedGCDMonoid α⟩
  /-
    🎉 no goals
  -/


