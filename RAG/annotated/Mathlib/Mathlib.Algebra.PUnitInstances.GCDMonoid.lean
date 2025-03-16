instance normalizedGCDMonoid : NormalizedGCDMonoid PUnit where
  gcd _ _ := unit
  lcm _ _ := unit
  normUnit _ := 1
  normUnit_zero := rfl
                     /-
                       ⊢ ∀ {a b : PUnit.{?u.2 + 1}}, Ne a 0 → Ne b 0 → Eq ((fun x => 1) (HMul.hMul a  …
                     -/
  normUnit_mul := by intros; rfl
                             /-
                               🎉 no goals
                             -/
                           /-
                             ⊢ ∀ (u : Units PUnit.{?u.2 + 1}), Eq ((fun x => 1) ↑u) (Inv.inv u)
                           -/
  normUnit_coe_units := by intros; rfl
                                   /-
                                     🎉 no goals
                                   -/
                                /-
                                  x✝¹ x✝ : PUnit.{?u.2 + 1}
                                  ⊢ Eq x✝¹ (HMul.hMul ((fun x x => PUnit.unit) x✝¹ x✝) PUnit.unit)
                                -/
  gcd_dvd_left _ _ := ⟨unit, by subsingleton⟩
                                /-
                                  🎉 no goals
                                -/
                                 /-
                                   x✝¹ x✝ : PUnit.{?u.2 + 1}
                                   ⊢ Eq x✝ (HMul.hMul ((fun x x => PUnit.unit) x✝¹ x✝) PUnit.unit)
                                 -/
  gcd_dvd_right _ _ := ⟨unit, by subsingleton⟩
                                 /-
                                   🎉 no goals
                                 -/
                                   /-
                                     x✝⁴ x✝³ x✝² : PUnit.{?u.2 + 1}
                                     x✝¹ : Dvd.dvd x✝⁴ x✝²
                                     x✝ : Dvd.dvd x✝⁴ x✝³
                                     ⊢ Eq ((fun x x => PUnit.unit) x✝² x✝³) (HMul.hMul x✝⁴ PUnit.unit)
                                   -/
  dvd_gcd {_ _} _ _ _ := ⟨unit, by subsingleton⟩
                                   /-
                                     🎉 no goals
                                   -/
                            /-
                              x✝¹ x✝ : PUnit.{?u.2 + 1}
                              ⊢ Eq (HMul.hMul (HMul.hMul ((fun x x => PUnit.unit) x✝¹ x✝) ((fun x x => PUnit …
                            -/
  gcd_mul_lcm _ _ := ⟨1, by subsingleton⟩
                            /-
                              🎉 no goals
                            -/
                      /-
                        ⊢ ∀ (a : PUnit.{?u.2 + 1}), Eq ((fun x x => PUnit.unit) 0 a) 0
                      -/
  lcm_zero_left := by intros; rfl
                              /-
                                🎉 no goals
                              -/
                       /-
                         ⊢ ∀ (a : PUnit.{?u.2 + 1}), Eq ((fun x x => PUnit.unit) a 0) 0
                       -/
  lcm_zero_right := by intros; rfl
                               /-
                                 🎉 no goals
                               -/
                      /-
                        ⊢ ∀ (a b : PUnit.{?u.2 + 1}), Eq (normalize (GCDMonoid.gcd a b)) (GCDMonoid.gc …
                      -/
  normalize_gcd := by intros; rfl
                              /-
                                🎉 no goals
                              -/
                      /-
                        ⊢ ∀ (a b : PUnit.{?u.2 + 1}), Eq (normalize (GCDMonoid.lcm a b)) (GCDMonoid.lc …
                      -/
  normalize_lcm := by intros; rfl
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem gcd_eq {x y : PUnit} : gcd x y = unit :=
  rfl


@[simp]
theorem lcm_eq {x y : PUnit} : lcm x y = unit :=
  rfl


@[simp]
theorem norm_unit_eq {x : PUnit} : normUnit x = 1 :=
  rfl


