instance instCommMonoid : CommMonoid ℤ where
  mul_comm := Int.mul_comm
  mul_one := Int.mul_one
  one_mul := Int.one_mul
  npow n x := x ^ n
  npow_zero _ := rfl
  npow_succ _ _ := rfl
  mul_assoc := Int.mul_assoc


instance instAddCommGroup : AddCommGroup ℤ where
  add_comm := Int.add_comm
  add_assoc := Int.add_assoc
  add_zero := Int.add_zero
  zero_add := Int.zero_add
  neg_add_cancel := Int.add_left_neg
  nsmul := (·*·)
  nsmul_zero := Int.zero_mul
  nsmul_succ n x :=
    show (n + 1 : ℤ) * x = n * x + x
       /-
         n : Nat
         x : Int
         ⊢ Eq (HMul.hMul (HAdd.hAdd (↑n) 1) x) (HAdd.hAdd (HMul.hMul (↑n) x) x)
       -/
    by rw [Int.add_mul, Int.one_mul]
       /-
         🎉 no goals
       -/
  zsmul := (·*·)
  zsmul_zero' := Int.zero_mul
  zsmul_succ' m n := by
    /-
      m : Nat
      n : Int
      ⊢ Eq ((fun x1 x2 => HMul.hMul x1 x2) (↑m.succ) n) (HAdd.hAdd ((fun x1 x2 => HM …
    -/
    simp only [ofNat_succ, Int.add_mul, Int.add_comm, Int.one_mul]
    /-
      🎉 no goals
    -/
                       /-
                         m : Nat
                         n : Int
                         ⊢ Eq ((fun x1 x2 => HMul.hMul x1 x2) (Int.negSucc m) n) (Neg.neg ((fun x1 x2 = …
                       -/
  zsmul_neg' m n := by simp only [negSucc_coe, ofNat_succ, Int.neg_mul]
                       /-
                         🎉 no goals
                       -/
  sub_eq_add_neg _ _ := Int.sub_eq_add_neg


                                                         /-
                                                           ⊢ AddCommMonoid Int
                                                         -/
instance instAddCommMonoid    : AddCommMonoid ℤ    := by infer_instance
                                                         /-
                                                           🎉 no goals
                                                         -/

                                                         /-
                                                           ⊢ AddMonoid Int
                                                         -/
instance instAddMonoid        : AddMonoid ℤ        := by infer_instance
                                                         /-
                                                           🎉 no goals
                                                         -/

                                                         /-
                                                           ⊢ Monoid Int
                                                         -/
instance instMonoid           : Monoid ℤ           := by infer_instance
                                                         /-
                                                           🎉 no goals
                                                         -/

                                                         /-
                                                           ⊢ CommSemigroup Int
                                                         -/
instance instCommSemigroup    : CommSemigroup ℤ    := by infer_instance
                                                         /-
                                                           🎉 no goals
                                                         -/

                                                         /-
                                                           ⊢ Semigroup Int
                                                         -/
instance instSemigroup        : Semigroup ℤ        := by infer_instance
                                                         /-
                                                           🎉 no goals
                                                         -/

                                                         /-
                                                           ⊢ AddGroup Int
                                                         -/
instance instAddGroup         : AddGroup ℤ         := by infer_instance
                                                         /-
                                                           🎉 no goals
                                                         -/

                                                         /-
                                                           ⊢ AddCommSemigroup Int
                                                         -/
instance instAddCommSemigroup : AddCommSemigroup ℤ := by infer_instance
                                                         /-
                                                           🎉 no goals
                                                         -/

                                                         /-
                                                           ⊢ AddSemigroup Int
                                                         -/
instance instAddSemigroup     : AddSemigroup ℤ     := by infer_instance
                                                         /-
                                                           🎉 no goals
                                                         -/


lemma toAdd_pow (a : Multiplicative ℤ) (b : ℕ) : (a ^ b).toAdd = a.toAdd * b := mul_comm _ _


lemma toAdd_zpow (a : Multiplicative ℤ) (b : ℤ) : (a ^ b).toAdd = a.toAdd * b := mul_comm _ _


@[simp] lemma ofAdd_mul (a b : ℤ) : ofAdd (a * b) = ofAdd a ^ b := (toAdd_zpow ..).symm


lemma units_natAbs (u : ℤˣ) : natAbs u = 1 :=
  Units.ext_iff.1 <|
    Nat.units_eq_one
                                 /-
                                   u : Units Int
                                   ⊢ Eq (HMul.hMul (↑u).natAbs (↑(Inv.inv u)).natAbs) 1
                                 -/
      ⟨natAbs u, natAbs ↑u⁻¹, by rw [← natAbs_mul, Units.mul_inv]; rfl, by
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
        /-
          u : Units Int
          ⊢ Eq (HMul.hMul (↑(Inv.inv u)).natAbs (↑u).natAbs) 1
        -/
        rw [← natAbs_mul, Units.inv_mul]; rfl⟩
                                          /-
                                            🎉 no goals
                                          -/


@[simp] lemma natAbs_of_isUnit (hu : IsUnit u) : natAbs u = 1 := units_natAbs hu.unit


lemma isUnit_eq_one_or (hu : IsUnit u) : u = 1 ∨ u = -1 := by
  /-
    u : Int
    hu : IsUnit u
    ⊢ Or (Eq u 1) (Eq u (-1))
  -/
  simpa only [natAbs_of_isUnit hu] using natAbs_eq u
  /-
    🎉 no goals
  -/


lemma isUnit_ne_iff_eq_neg (hu : IsUnit u) (hv : IsUnit v) : u ≠ v ↔ u = -v := by
  /-
    u v : Int
    hu : IsUnit u
    hv : IsUnit v
    ⊢ Iff (Ne u v) (Eq u (Neg.neg v))
  -/
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/
  obtain rfl | rfl := isUnit_eq_one_or hu <;> obtain rfl | rfl := isUnit_eq_one_or hv <;> decide
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


lemma isUnit_eq_or_eq_neg (hu : IsUnit u) (hv : IsUnit v) : u = v ∨ u = -v :=
  or_iff_not_imp_left.2 (isUnit_ne_iff_eq_neg hu hv).1


lemma isUnit_iff : IsUnit u ↔ u = 1 ∨ u = -1 := by
  /-
    u : Int
    ⊢ Iff (IsUnit u) (Or (Eq u 1) (Eq u (-1)))
  -/
  refine ⟨fun h ↦ isUnit_eq_one_or h, fun h ↦ ?_⟩
  /-
    u : Int
    h : Or (Eq u 1) (Eq u (-1))
    ⊢ IsUnit u
  -/
  rcases h with (rfl | rfl)
    /-
      case inl
      ⊢ IsUnit 1
    -/
  · exact isUnit_one
    /-
      🎉 no goals
    -/
    /-
      case inr
      ⊢ IsUnit (-1)
    -/
  · exact ⟨⟨-1, -1, by decide, by decide⟩, rfl⟩
    /-
      🎉 no goals
    -/


lemma eq_one_or_neg_one_of_mul_eq_one (h : u * v = 1) : u = 1 ∨ u = -1 :=
  isUnit_iff.1 (isUnit_of_mul_eq_one u v h)


lemma eq_one_or_neg_one_of_mul_eq_one' (h : u * v = 1) : u = 1 ∧ v = 1 ∨ u = -1 ∧ v = -1 := by
  /-
    u v : Int
    h : Eq (HMul.hMul u v) 1
    ⊢ Or (And (Eq u 1) (Eq v 1)) (And (Eq u (-1)) (Eq v (-1)))
  -/
  have h' : v * u = 1 := mul_comm u v ▸ h
  /-
    u v : Int
    h : Eq (HMul.hMul u v) 1
    h' : Eq (HMul.hMul v u) 1
    ⊢ Or (And (Eq u 1) (Eq v 1)) (And (Eq u (-1)) (Eq v (-1)))
  -/
  obtain rfl | rfl := eq_one_or_neg_one_of_mul_eq_one h <;>
      /-
        case inl
        v : Int
        h : Eq (HMul.hMul 1 v) 1
        h' : Eq (HMul.hMul v 1) 1
        ⊢ Or (And (Eq 1 1) (Eq v 1)) (And (Eq 1 (-1)) (Eq v (-1)))
      -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
      obtain rfl | rfl := eq_one_or_neg_one_of_mul_eq_one h' <;> tauto
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


lemma eq_of_mul_eq_one (h : u * v = 1) : u = v :=
  (eq_one_or_neg_one_of_mul_eq_one' h).elim
    (and_imp.2 (·.trans ·.symm)) (and_imp.2 (·.trans ·.symm))


lemma mul_eq_one_iff_eq_one_or_neg_one : u * v = 1 ↔ u = 1 ∧ v = 1 ∨ u = -1 ∧ v = -1 := by
  /-
    u v : Int
    ⊢ Iff (Eq (HMul.hMul u v) 1) (Or (And (Eq u 1) (Eq v 1)) (And (Eq u (-1)) (Eq  …
  -/
  refine ⟨eq_one_or_neg_one_of_mul_eq_one', fun h ↦ Or.elim h (fun H ↦ ?_) fun H ↦ ?_⟩ <;>
    /-
      case refine_1
      u v : Int
      h : Or (And (Eq u 1) (Eq v 1)) (And (Eq u (-1)) (Eq v (-1)))
      H : And (Eq u 1) (Eq v 1)
      ⊢ Eq (HMul.hMul u v) 1
    -/
                               /-
                                 🎉 no goals
                               -/
    obtain ⟨rfl, rfl⟩ := H <;> rfl
                               /-
                                 🎉 no goals
                               -/


lemma eq_one_or_neg_one_of_mul_eq_neg_one' (h : u * v = -1) : u = 1 ∧ v = -1 ∨ u = -1 ∧ v = 1 := by
  /-
    u v : Int
    h : Eq (HMul.hMul u v) (-1)
    ⊢ Or (And (Eq u 1) (Eq v (-1))) (And (Eq u (-1)) (Eq v 1))
  -/
  obtain rfl | rfl := isUnit_eq_one_or (IsUnit.mul_iff.mp (Int.isUnit_iff.mpr (Or.inr h))).1
    /-
      case inl
      v : Int
      h : Eq (HMul.hMul 1 v) (-1)
      ⊢ Or (And (Eq 1 1) (Eq v (-1))) (And (Eq 1 (-1)) (Eq v 1))
    -/
  · exact Or.inl ⟨rfl, one_mul v ▸ h⟩
    /-
      🎉 no goals
    -/
    /-
      case inr
      v : Int
      h : Eq (HMul.hMul (-1) v) (-1)
      ⊢ Or (And (Eq (-1) 1) (Eq v (-1))) (And (Eq (-1) (-1)) (Eq v 1))
    -/
  · simpa [Int.neg_mul] using h
    /-
      🎉 no goals
    -/


lemma mul_eq_neg_one_iff_eq_one_or_neg_one : u * v = -1 ↔ u = 1 ∧ v = -1 ∨ u = -1 ∧ v = 1 := by
  /-
    u v : Int
    ⊢ Iff (Eq (HMul.hMul u v) (-1)) (Or (And (Eq u 1) (Eq v (-1))) (And (Eq u (-1) …
  -/
  refine ⟨eq_one_or_neg_one_of_mul_eq_neg_one', fun h ↦ Or.elim h (fun H ↦ ?_) fun H ↦ ?_⟩ <;>
    /-
      case refine_1
      u v : Int
      h : Or (And (Eq u 1) (Eq v (-1))) (And (Eq u (-1)) (Eq v 1))
      H : And (Eq u 1) (Eq v (-1))
      ⊢ Eq (HMul.hMul u v) (-1)
    -/
                               /-
                                 🎉 no goals
                               -/
    obtain ⟨rfl, rfl⟩ := H <;> rfl
                               /-
                                 🎉 no goals
                               -/


                                                           /-
                                                             u : Int
                                                             ⊢ Iff (IsUnit u) (Eq u.natAbs 1)
                                                           -/
lemma isUnit_iff_natAbs_eq : IsUnit u ↔ u.natAbs = 1 := by simp [natAbs_eq_iff, isUnit_iff]
                                                           /-
                                                             🎉 no goals
                                                           -/


alias ⟨IsUnit.natAbs_eq, _⟩ := isUnit_iff_natAbs_eq

-- Porting note: `rw` didn't work on `natAbs_ofNat`, so had to change to `simp`,
-- presumably because `(n : ℤ)` is `Nat.cast` and not just `ofNat`

@[norm_cast]
                                                             /-
                                                               n : Nat
                                                               ⊢ Iff (IsUnit ↑n) (IsUnit n)
                                                             -/
lemma ofNat_isUnit {n : ℕ} : IsUnit (n : ℤ) ↔ IsUnit n := by simp [isUnit_iff_natAbs_eq]
                                                             /-
                                                               🎉 no goals
                                                             -/


lemma isUnit_mul_self (hu : IsUnit u) : u * u = 1 :=
  (isUnit_eq_one_or hu).elim (fun h ↦ h.symm ▸ rfl) fun h ↦ h.symm ▸ rfl


lemma isUnit_add_isUnit_eq_isUnit_add_isUnit {a b c d : ℤ} (ha : IsUnit a) (hb : IsUnit b)
    (hc : IsUnit c) (hd : IsUnit d) : a + b = c + d ↔ a = c ∧ b = d ∨ a = d ∧ b = c := by
  /-
    a b c d : Int
    ha : IsUnit a
    hb : IsUnit b
    hc : IsUnit c
    hd : IsUnit d
    ⊢ Iff (Eq (HAdd.hAdd a b) (HAdd.hAdd c d)) (Or (And (Eq a c) (Eq b d)) (And (E …
  -/
  rw [isUnit_iff] at ha hb hc hd
  /-
    a b c d : Int
    ha : Or (Eq a 1) (Eq a (-1))
    hb : Or (Eq b 1) (Eq b (-1))
    hc : Or (Eq c 1) (Eq c (-1))
    hd : Or (Eq d 1) (Eq d (-1))
    ⊢ Iff (Eq (HAdd.hAdd a b) (HAdd.hAdd c d)) (Or (And (Eq a c) (Eq b d)) (And (E …
  -/
  aesop
  /-
    🎉 no goals
  -/


lemma eq_one_or_neg_one_of_mul_eq_neg_one (h : u * v = -1) : u = 1 ∨ u = -1 :=
  Or.elim (eq_one_or_neg_one_of_mul_eq_neg_one' h) (fun H => Or.inl H.1) fun H => Or.inr H.1


@[simp] lemma emod_two_ne_one : ¬n % 2 = 1 ↔ n % 2 = 0 := by
  /-
    n : Int
    ⊢ Iff (Not (Eq (HMod.hMod n 2) 1)) (Eq (HMod.hMod n 2) 0)
  -/
                                                /-
                                                  🎉 no goals
                                                -/
  cases' emod_two_eq_zero_or_one n with h h <;> simp [h]
                                                /-
                                                  🎉 no goals
                                                -/


@[simp] lemma one_emod_two : (1 : Int) % 2 = 1 := rfl

-- `EuclideanDomain.mod_eq_zero` uses (2 ∣ n) as normal form

@[local simp] lemma emod_two_ne_zero : ¬n % 2 = 0 ↔ n % 2 = 1 := by
  /-
    n : Int
    ⊢ Iff (Not (Eq (HMod.hMod n 2) 0)) (Eq (HMod.hMod n 2) 1)
  -/
                                                /-
                                                  🎉 no goals
                                                -/
  cases' emod_two_eq_zero_or_one n with h h <;> simp [h]
                                                /-
                                                  🎉 no goals
                                                -/


lemma even_iff : Even n ↔ n % 2 = 0 where
                         /-
                           n : Int
                           x✝ : Even n
                           m : Int
                           hm : Eq n (HAdd.hAdd m m)
                           ⊢ Eq (HMod.hMod n 2) 0
                         -/
  mp := fun ⟨m, hm⟩ ↦ by simp [← Int.two_mul, hm]
                         /-
                           🎉 no goals
                         -/
                                                      /-
                                                        n : Int
                                                        h : Eq (HMod.hMod n 2) 0
                                                        ⊢ Eq (HAdd.hAdd (HMod.hMod n 2) (HMul.hMul 2 (HDiv.hDiv n 2))) (HAdd.hAdd (HDi …
                                                      -/
  mpr h := ⟨n / 2, (emod_add_ediv n 2).symm.trans (by simp [← Int.two_mul, h])⟩
                                                      /-
                                                        🎉 no goals
                                                      -/


                                               /-
                                                 n : Int
                                                 ⊢ Iff (Not (Even n)) (Eq (HMod.hMod n 2) 1)
                                               -/
lemma not_even_iff : ¬Even n ↔ n % 2 = 1 := by rw [even_iff, emod_two_ne_zero]
                                               /-
                                                 🎉 no goals
                                               -/


@[simp] lemma two_dvd_ne_zero : ¬2 ∣ n ↔ n % 2 = 1 :=
  (even_iff_exists_two_nsmul _).symm.not.trans not_even_iff


instance : DecidablePred (Even : ℤ → Prop) := fun _ ↦ decidable_of_iff _ even_iff.symm


/-- `IsSquare` can be decided on `ℤ` by checking against the square root. -/
instance : DecidablePred (IsSquare : ℤ → Prop) :=
  fun m ↦ decidable_of_iff' (sqrt m * sqrt m = m) <| by
    /-
      u v m✝ n m : Int
      ⊢ Iff (IsSquare m) (Eq (HMul.hMul (Int.sqrt m) (Int.sqrt m)) m)
    -/
    simp_rw [← exists_mul_self m, IsSquare, eq_comm]
    /-
      🎉 no goals
    -/


                                                 /-
                                                   ⊢ Not (Even 1)
                                                 -/
@[simp] lemma not_even_one : ¬Even (1 : ℤ) := by simp [even_iff]
                                                 /-
                                                   🎉 no goals
                                                 -/


@[parity_simps] lemma even_add : Even (m + n) ↔ (Even m ↔ Even n) := by
  /-
    m n : Int
    ⊢ Iff (Even (HAdd.hAdd m n)) (Iff (Even m) (Even n))
  -/
  cases' emod_two_eq_zero_or_one m with h₁ h₁ <;>
  /-
    case inl
    m n : Int
    h₁ : Eq (HMod.hMod m 2) 0
    ⊢ Iff (Even (HAdd.hAdd m n)) (Iff (Even m) (Even n))
  -/
  cases' emod_two_eq_zero_or_one n with h₂ h₂ <;>
  /-
    case inl.inl
    m n : Int
    h₁ : Eq (HMod.hMod m 2) 0
    h₂ : Eq (HMod.hMod n 2) 0
    ⊢ Iff (Even (HAdd.hAdd m n)) (Iff (Even m) (Even n))
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  simp [even_iff, h₁, h₂, Int.add_emod, one_add_one_eq_two, emod_self]
  /-
    🎉 no goals
  -/


                                                                 /-
                                                                   n : Int
                                                                   ⊢ Not (Dvd.dvd 2 (HAdd.hAdd (HMul.hMul 2 n) 1))
                                                                 -/
lemma two_not_dvd_two_mul_add_one (n : ℤ) : ¬2 ∣ 2 * n + 1 := by simp [add_emod]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[parity_simps]
                                                        /-
                                                          m n : Int
                                                          ⊢ Iff (Even (HSub.hSub m n)) (Iff (Even m) (Even n))
                                                        -/
lemma even_sub : Even (m - n) ↔ (Even m ↔ Even n) := by simp [sub_eq_add_neg, parity_simps]
                                                        /-
                                                          🎉 no goals
                                                        -/


                                                                  /-
                                                                    n : Int
                                                                    ⊢ Iff (Even (HAdd.hAdd n 1)) (Not (Even n))
                                                                  -/
@[parity_simps] lemma even_add_one : Even (n + 1) ↔ ¬Even n := by simp [even_add]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


                                                                  /-
                                                                    n : Int
                                                                    ⊢ Iff (Even (HSub.hSub n 1)) (Not (Even n))
                                                                  -/
@[parity_simps] lemma even_sub_one : Even (n - 1) ↔ ¬Even n := by simp [even_sub]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[parity_simps] lemma even_mul : Even (m * n) ↔ Even m ∨ Even n := by
  /-
    m n : Int
    ⊢ Iff (Even (HMul.hMul m n)) (Or (Even m) (Even n))
  -/
  cases' emod_two_eq_zero_or_one m with h₁ h₁ <;>
  /-
    case inl
    m n : Int
    h₁ : Eq (HMod.hMod m 2) 0
    ⊢ Iff (Even (HMul.hMul m n)) (Or (Even m) (Even n))
  -/
  cases' emod_two_eq_zero_or_one n with h₂ h₂ <;>
  /-
    case inl.inl
    m n : Int
    h₁ : Eq (HMod.hMod m 2) 0
    h₂ : Eq (HMod.hMod n 2) 0
    ⊢ Iff (Even (HMul.hMul m n)) (Or (Even m) (Even n))
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  simp [even_iff, h₁, h₂, Int.mul_emod]
  /-
    🎉 no goals
  -/


@[parity_simps] lemma even_pow {n : ℕ} : Even (m ^ n) ↔ Even m ∧ n ≠ 0 := by
  /-
    m : Int
    n : Nat
    ⊢ Iff (Even (HPow.hPow m n)) (And (Even m) (Ne n 0))
  -/
                  /-
                    🎉 no goals
                  -/
  induction n <;> simp [*, even_mul, pow_succ]; tauto
                                                /-
                                                  🎉 no goals
                                                -/


lemma even_pow' {n : ℕ} (h : n ≠ 0) : Even (m ^ n) ↔ Even m := even_pow.trans <| and_iff_left h


@[simp, norm_cast] lemma even_coe_nat (n : ℕ) : Even (n : ℤ) ↔ Even n := by
  /-
    n : Nat
    ⊢ Iff (Even ↑n) (Even n)
  -/
  rw_mod_cast [even_iff, Nat.even_iff]
  /-
    🎉 no goals
  -/


lemma two_mul_ediv_two_of_even : Even n → 2 * (n / 2) = n :=
  fun h ↦ Int.mul_ediv_cancel' ((even_iff_exists_two_nsmul _).mp h)


lemma ediv_two_mul_two_of_even : Even n → n / 2 * 2 = n :=
  fun h ↦ Int.ediv_mul_cancel ((even_iff_exists_two_nsmul _).mp h)

-- Here are examples of how `parity_simps` can be used with `Int`.

lemma zsmul_int_int (a b : ℤ) : a • b = a * b := rfl


lemma zsmul_int_one (n : ℤ) : n • (1 : ℤ) = n := mul_one _

