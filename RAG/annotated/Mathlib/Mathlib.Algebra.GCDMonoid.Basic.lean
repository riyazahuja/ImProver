/-- Normalization monoid: multiplying with `normUnit` gives a normal form for associated
elements. -/
class NormalizationMonoid (α : Type*) [CancelCommMonoidWithZero α] where
  /-- `normUnit` assigns to each element of the monoid a unit of the monoid. -/
  normUnit : α → αˣ
  /-- The proposition that `normUnit` maps `0` to the identity. -/
  normUnit_zero : normUnit 0 = 1
  /-- The proposition that `normUnit` respects multiplication of non-zero elements. -/
  normUnit_mul : ∀ {a b}, a ≠ 0 → b ≠ 0 → normUnit (a * b) = normUnit a * normUnit b
  /-- The proposition that `normUnit` maps units to their inverses. -/
  normUnit_coe_units : ∀ u : αˣ, normUnit u = u⁻¹


@[simp]
theorem normUnit_one : normUnit (1 : α) = 1 :=
  normUnit_coe_units 1


/-- Chooses an element of each associate class, by multiplying by `normUnit` -/
def normalize : α →*₀ α where
  toFun x := x * normUnit x
  map_zero' := by
    /-
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : NormalizationMonoid α
      ⊢ Eq ((fun x => HMul.hMul x ↑(NormalizationMonoid.normUnit x)) 0) 0
    -/
    simp only [normUnit_zero]
    /-
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : NormalizationMonoid α
      ⊢ Eq (HMul.hMul 0 ↑1) 0
    -/
    exact mul_one (0 : α)
    /-
      🎉 no goals
    -/
                 /-
                   α : Type u_1
                   inst✝¹ : CancelCommMonoidWithZero α
                   inst✝ : NormalizationMonoid α
                   ⊢ Eq ({ toFun := fun x => HMul.hMul x ↑(NormalizationMonoid.normUnit x), map_z …
                 -/
  map_one' := by dsimp only; rw [normUnit_one, one_mul]; rfl
                                                         /-
                                                           🎉 no goals
                                                         -/
  map_mul' x y :=
                                   /-
                                     α : Type u_1
                                     inst✝¹ : CancelCommMonoidWithZero α
                                     inst✝ : NormalizationMonoid α
                                     x y : α
                                     hx : Eq x 0
                                     ⊢ Eq ({ toFun := fun x => HMul.hMul x ↑(NormalizationMonoid.normUnit x), map_z …
                                   -/
    (by_cases fun hx : x = 0 => by dsimp only; rw [hx, zero_mul, zero_mul, zero_mul]) fun hx =>
                                               /-
                                                 🎉 no goals
                                               -/
                                     /-
                                       α : Type u_1
                                       inst✝¹ : CancelCommMonoidWithZero α
                                       inst✝ : NormalizationMonoid α
                                       x y : α
                                       hx : Not (Eq x 0)
                                       hy : Eq y 0
                                       ⊢ Eq ({ toFun := fun x => HMul.hMul x ↑(NormalizationMonoid.normUnit x), map_z …
                                     -/
      (by_cases fun hy : y = 0 => by dsimp only; rw [hy, mul_zero, zero_mul, mul_zero]) fun hy => by
                                                 /-
                                                   🎉 no goals
                                                 -/
        /-
          α : Type u_1
          inst✝¹ : CancelCommMonoidWithZero α
          inst✝ : NormalizationMonoid α
          x y : α
          hx : Not (Eq x 0)
          hy : Not (Eq y 0)
          ⊢ Eq ({ toFun := fun x => HMul.hMul x ↑(NormalizationMonoid.normUnit x), map_z …
        -/
        simp only [normUnit_mul hx hy, Units.val_mul]; simp only [mul_assoc, mul_left_comm y]
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem associated_normalize (x : α) : Associated x (normalize x) :=
  ⟨_, rfl⟩


theorem normalize_associated (x : α) : Associated (normalize x) x :=
  (associated_normalize _).symm


theorem associated_normalize_iff {x y : α} : Associated x (normalize y) ↔ Associated x y :=
  ⟨fun h => h.trans (normalize_associated y), fun h => h.trans (associated_normalize y)⟩


theorem normalize_associated_iff {x y : α} : Associated (normalize x) y ↔ Associated x y :=
  ⟨fun h => (associated_normalize _).trans h, fun h => (normalize_associated _).trans h⟩


theorem Associates.mk_normalize (x : α) : Associates.mk (normalize x) = Associates.mk x :=
  Associates.mk_eq_mk_iff_associated.2 (normalize_associated _)


theorem normalize_apply (x : α) : normalize x = x * normUnit x :=
  rfl


theorem normalize_zero : normalize (0 : α) = 0 :=
  normalize.map_zero


theorem normalize_one : normalize (1 : α) = 1 :=
  normalize.map_one


                                                                   /-
                                                                     α : Type u_1
                                                                     inst✝¹ : CancelCommMonoidWithZero α
                                                                     inst✝ : NormalizationMonoid α
                                                                     u : Units α
                                                                     ⊢ Eq (normalize ↑u) 1
                                                                   -/
theorem normalize_coe_units (u : αˣ) : normalize (u : α) = 1 := by simp [normalize_apply]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


theorem normalize_eq_zero {x : α} : normalize x = 0 ↔ x = 0 :=
  ⟨fun hx => (associated_zero_iff_eq_zero x).1 <| hx ▸ associated_normalize _, by
    /-
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : NormalizationMonoid α
      x : α
      ⊢ Eq x 0 → Eq (normalize x) 0
    -/
    rintro rfl; exact normalize_zero⟩
                /-
                  🎉 no goals
                -/


theorem normalize_eq_one {x : α} : normalize x = 1 ↔ IsUnit x :=
  ⟨fun hx => isUnit_iff_exists_inv.2 ⟨_, hx⟩, fun ⟨u, hu⟩ => hu ▸ normalize_coe_units u⟩

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11083): quite slow. Improve performance?

@[simp]
theorem normUnit_mul_normUnit (a : α) : normUnit (a * normUnit a) = 1 := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : NormalizationMonoid α
    a : α
    ⊢ Eq (NormalizationMonoid.normUnit (HMul.hMul a ↑(NormalizationMonoid.normUnit …
  -/
  nontriviality α using Subsingleton.elim a 0
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : NormalizationMonoid α
    a : α
    a✝ : Nontrivial α
    ⊢ Eq (NormalizationMonoid.normUnit (HMul.hMul a ↑(NormalizationMonoid.normUnit …
  -/
  obtain rfl | h := eq_or_ne a 0
    /-
      case inl
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : NormalizationMonoid α
      a✝ : Nontrivial α
      ⊢ Eq (NormalizationMonoid.normUnit (HMul.hMul 0 ↑(NormalizationMonoid.normUnit …
    -/
  · rw [normUnit_zero, zero_mul, normUnit_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : NormalizationMonoid α
      a : α
      a✝ : Nontrivial α
      h : Ne a 0
      ⊢ Eq (NormalizationMonoid.normUnit (HMul.hMul a ↑(NormalizationMonoid.normUnit …
    -/
  · rw [normUnit_mul h (Units.ne_zero _), normUnit_coe_units, mul_inv_eq_one]
    /-
      🎉 no goals
    -/


@[simp]
                                                                             /-
                                                                               α : Type u_1
                                                                               inst✝¹ : CancelCommMonoidWithZero α
                                                                               inst✝ : NormalizationMonoid α
                                                                               x : α
                                                                               ⊢ Eq (normalize (normalize x)) (normalize x)
                                                                             -/
theorem normalize_idem (x : α) : normalize (normalize x) = normalize x := by simp [normalize_apply]
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


theorem normalize_eq_normalize {a b : α} (hab : a ∣ b) (hba : b ∣ a) :
    normalize a = normalize b := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : NormalizationMonoid α
    a b : α
    hab : Dvd.dvd a b
    hba : Dvd.dvd b a
    ⊢ Eq (normalize a) (normalize b)
  -/
  nontriviality α
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : NormalizationMonoid α
    a b : α
    hab : Dvd.dvd a b
    hba : Dvd.dvd b a
    a✝ : Nontrivial α
    ⊢ Eq (normalize a) (normalize b)
  -/
  rcases associated_of_dvd_dvd hab hba with ⟨u, rfl⟩
  /-
    case intro
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : NormalizationMonoid α
    a : α
    a✝ : Nontrivial α
    u : Units α
    hab : Dvd.dvd a (HMul.hMul a ↑u)
    hba : Dvd.dvd (HMul.hMul a ↑u) a
    ⊢ Eq (normalize a) (normalize (HMul.hMul a ↑u))
  -/
  refine by_cases (by rintro rfl; simp only [zero_mul]) fun ha : a ≠ 0 => ?_
  suffices a * ↑(normUnit a) = a * ↑u * ↑(normUnit a) * ↑u⁻¹ by
    simpa only [normalize_apply, mul_assoc, normUnit_mul ha u.ne_zero, normUnit_coe_units]
  calc
    a * ↑(normUnit a) = a * ↑(normUnit a) * ↑u * ↑u⁻¹ := (Units.mul_inv_cancel_right _ _).symm
    _ = a * ↑u * ↑(normUnit a) * ↑u⁻¹ := by rw [mul_right_comm a]


theorem normalize_eq_normalize_iff {x y : α} : normalize x = normalize y ↔ x ∣ y ∧ y ∣ x :=
  ⟨fun h => ⟨Units.dvd_mul_right.1 ⟨_, h.symm⟩, Units.dvd_mul_right.1 ⟨_, h⟩⟩, fun ⟨hxy, hyx⟩ =>
    normalize_eq_normalize hxy hyx⟩


theorem dvd_antisymm_of_normalize_eq {a b : α} (ha : normalize a = a) (hb : normalize b = b)
    (hab : a ∣ b) (hba : b ∣ a) : a = b :=
  ha ▸ hb ▸ normalize_eq_normalize hab hba


theorem Associated.eq_of_normalized
    {a b : α} (h : Associated a b) (ha : normalize a = a) (hb : normalize b = b) :
    a = b :=
  dvd_antisymm_of_normalize_eq ha hb h.dvd h.dvd'


@[simp]
theorem dvd_normalize_iff {a b : α} : a ∣ normalize b ↔ a ∣ b :=
  Units.dvd_mul_right


@[simp]
theorem normalize_dvd_iff {a b : α} : normalize a ∣ b ↔ a ∣ b :=
  Units.mul_right_dvd


/-- Maps an element of `Associates` back to the normalized element of its associate class -/
protected def out : Associates α → α :=
  (Quotient.lift (normalize : α → α)) fun a _ ⟨_, hu⟩ =>
    hu ▸ normalize_eq_normalize ⟨_, rfl⟩ (Units.mul_right_dvd.2 <| dvd_refl a)


@[simp]
theorem out_mk (a : α) : (Associates.mk a).out = normalize a :=
  rfl


@[simp]
theorem out_one : (1 : Associates α).out = 1 :=
  normalize_one


theorem out_mul (a b : Associates α) : (a * b).out = a.out * b.out :=
  Quotient.inductionOn₂ a b fun _ _ => by
    /-
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : NormalizationMonoid α
      a b : Associates α
      x✝¹ x✝ : α
      ⊢ Eq (HMul.hMul (Quotient.mk (Associated.setoid α) x✝¹) (Quotient.mk (Associat …
    -/
    simp only [Associates.quotient_mk_eq_mk, out_mk, mk_mul_mk, normalize.map_mul]
    /-
      🎉 no goals
    -/


theorem dvd_out_iff (a : α) (b : Associates α) : a ∣ b.out ↔ Associates.mk a ≤ b :=
  Quotient.inductionOn b <| by
    /-
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : NormalizationMonoid α
      a : α
      b : Associates α
      ⊢ ∀ (a_1 : α), Iff (Dvd.dvd a (Associates.out (Quotient.mk (Associated.setoid  …
    -/
    simp [Associates.out_mk, Associates.quotient_mk_eq_mk, mk_le_mk_iff_dvd]
    /-
      🎉 no goals
    -/


theorem out_dvd_iff (a : α) (b : Associates α) : b.out ∣ a ↔ b ≤ Associates.mk a :=
  Quotient.inductionOn b <| by
    /-
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : NormalizationMonoid α
      a : α
      b : Associates α
      ⊢ ∀ (a_1 : α), Iff (Dvd.dvd (Associates.out (Quotient.mk (Associated.setoid α) …
    -/
    simp [Associates.out_mk, Associates.quotient_mk_eq_mk, mk_le_mk_iff_dvd]
    /-
      🎉 no goals
    -/


@[simp]
theorem out_top : (⊤ : Associates α).out = 0 :=
  normalize_zero


@[simp]
theorem normalize_out (a : Associates α) : normalize a.out = a.out :=
  Quotient.inductionOn a normalize_idem


@[simp]
theorem mk_out (a : Associates α) : Associates.mk a.out = a :=
  Quotient.inductionOn a mk_normalize


theorem out_injective : Function.Injective (Associates.out : _ → α) :=
  Function.LeftInverse.injective mk_out


/-- GCD monoid: a `CancelCommMonoidWithZero` with `gcd` (greatest common divisor) and
`lcm` (least common multiple) operations, determined up to a unit. The type class focuses on `gcd`
and we derive the corresponding `lcm` facts from `gcd`.
-/
class GCDMonoid (α : Type*) [CancelCommMonoidWithZero α] where
  /-- The greatest common divisor between two elements. -/
  gcd : α → α → α
  /-- The least common multiple between two elements. -/
  lcm : α → α → α
  /-- The GCD is a divisor of the first element. -/
  gcd_dvd_left : ∀ a b, gcd a b ∣ a
  /-- The GCD is a divisor of the second element. -/
  gcd_dvd_right : ∀ a b, gcd a b ∣ b
  /-- Any common divisor of both elements is a divisor of the GCD. -/
  dvd_gcd : ∀ {a b c}, a ∣ c → a ∣ b → a ∣ gcd c b
  /-- The product of two elements is `Associated` with the product of their GCD and LCM. -/
  gcd_mul_lcm : ∀ a b, Associated (gcd a b * lcm a b) (a * b)
  /-- `0` is left-absorbing. -/
  lcm_zero_left : ∀ a, lcm 0 a = 0
  /-- `0` is right-absorbing. -/
  lcm_zero_right : ∀ a, lcm a 0 = 0


/-- Normalized GCD monoid: a `CancelCommMonoidWithZero` with normalization and `gcd`
(greatest common divisor) and `lcm` (least common multiple) operations. In this setting `gcd` and
`lcm` form a bounded lattice on the associated elements where `gcd` is the infimum, `lcm` is the
supremum, `1` is bottom, and `0` is top. The type class focuses on `gcd` and we derive the
corresponding `lcm` facts from `gcd`.
-/
class NormalizedGCDMonoid (α : Type*) [CancelCommMonoidWithZero α] extends NormalizationMonoid α,
  GCDMonoid α where
  /-- The GCD is normalized to itself. -/
  normalize_gcd : ∀ a b, normalize (gcd a b) = gcd a b
  /-- The LCM is normalized to itself. -/
  normalize_lcm : ∀ a b, normalize (lcm a b) = lcm a b


instance [NormalizationMonoid α] : Nonempty (NormalizationMonoid α) := ⟨‹_›⟩

instance [GCDMonoid α] : Nonempty (GCDMonoid α) := ⟨‹_›⟩

instance [NormalizedGCDMonoid α] : Nonempty (NormalizedGCDMonoid α) := ⟨‹_›⟩

instance [h : Nonempty (NormalizedGCDMonoid α)] : Nonempty (NormalizationMonoid α) :=
  h.elim fun _ ↦ inferInstance

instance [h : Nonempty (NormalizedGCDMonoid α)] : Nonempty (GCDMonoid α) :=
  h.elim fun _ ↦ inferInstance


theorem gcd_isUnit_iff_isRelPrime [GCDMonoid α] {a b : α} :
    IsUnit (gcd a b) ↔ IsRelPrime a b :=
  ⟨fun h _ ha hb ↦ isUnit_of_dvd_unit (dvd_gcd ha hb) h, (· (gcd_dvd_left a b) (gcd_dvd_right a b))⟩


@[simp]
theorem normalize_gcd [NormalizedGCDMonoid α] : ∀ a b : α, normalize (gcd a b) = gcd a b :=
  NormalizedGCDMonoid.normalize_gcd


theorem gcd_mul_lcm [GCDMonoid α] : ∀ a b : α, Associated (gcd a b * lcm a b) (a * b) :=
  GCDMonoid.gcd_mul_lcm


theorem dvd_gcd_iff [GCDMonoid α] (a b c : α) : a ∣ gcd b c ↔ a ∣ b ∧ a ∣ c :=
  Iff.intro (fun h => ⟨h.trans (gcd_dvd_left _ _), h.trans (gcd_dvd_right _ _)⟩) fun ⟨hab, hac⟩ =>
    dvd_gcd hab hac


theorem gcd_comm [NormalizedGCDMonoid α] (a b : α) : gcd a b = gcd b a :=
  dvd_antisymm_of_normalize_eq (normalize_gcd _ _) (normalize_gcd _ _)
    (dvd_gcd (gcd_dvd_right _ _) (gcd_dvd_left _ _))
    (dvd_gcd (gcd_dvd_right _ _) (gcd_dvd_left _ _))


theorem gcd_comm' [GCDMonoid α] (a b : α) : Associated (gcd a b) (gcd b a) :=
  associated_of_dvd_dvd (dvd_gcd (gcd_dvd_right _ _) (gcd_dvd_left _ _))
    (dvd_gcd (gcd_dvd_right _ _) (gcd_dvd_left _ _))


theorem gcd_assoc [NormalizedGCDMonoid α] (m n k : α) : gcd (gcd m n) k = gcd m (gcd n k) :=
  dvd_antisymm_of_normalize_eq (normalize_gcd _ _) (normalize_gcd _ _)
    (dvd_gcd ((gcd_dvd_left (gcd m n) k).trans (gcd_dvd_left m n))
      (dvd_gcd ((gcd_dvd_left (gcd m n) k).trans (gcd_dvd_right m n)) (gcd_dvd_right (gcd m n) k)))
    (dvd_gcd
      (dvd_gcd (gcd_dvd_left m (gcd n k)) ((gcd_dvd_right m (gcd n k)).trans (gcd_dvd_left n k)))
      ((gcd_dvd_right m (gcd n k)).trans (gcd_dvd_right n k)))


theorem gcd_assoc' [GCDMonoid α] (m n k : α) : Associated (gcd (gcd m n) k) (gcd m (gcd n k)) :=
  associated_of_dvd_dvd
    (dvd_gcd ((gcd_dvd_left (gcd m n) k).trans (gcd_dvd_left m n))
      (dvd_gcd ((gcd_dvd_left (gcd m n) k).trans (gcd_dvd_right m n)) (gcd_dvd_right (gcd m n) k)))
    (dvd_gcd
      (dvd_gcd (gcd_dvd_left m (gcd n k)) ((gcd_dvd_right m (gcd n k)).trans (gcd_dvd_left n k)))
      ((gcd_dvd_right m (gcd n k)).trans (gcd_dvd_right n k)))


instance [NormalizedGCDMonoid α] : Std.Commutative (α := α) gcd where
  comm := gcd_comm


instance [NormalizedGCDMonoid α] : Std.Associative (α := α) gcd where
  assoc := gcd_assoc


theorem gcd_eq_normalize [NormalizedGCDMonoid α] {a b c : α} (habc : gcd a b ∣ c)
    (hcab : c ∣ gcd a b) : gcd a b = normalize c :=
  normalize_gcd a b ▸ normalize_eq_normalize habc hcab


@[simp]
theorem gcd_zero_left [NormalizedGCDMonoid α] (a : α) : gcd 0 a = normalize a :=
  gcd_eq_normalize (gcd_dvd_right 0 a) (dvd_gcd (dvd_zero _) (dvd_refl a))


theorem gcd_zero_left' [GCDMonoid α] (a : α) : Associated (gcd 0 a) a :=
  associated_of_dvd_dvd (gcd_dvd_right 0 a) (dvd_gcd (dvd_zero _) (dvd_refl a))


@[simp]
theorem gcd_zero_right [NormalizedGCDMonoid α] (a : α) : gcd a 0 = normalize a :=
  gcd_eq_normalize (gcd_dvd_left a 0) (dvd_gcd (dvd_refl a) (dvd_zero _))


theorem gcd_zero_right' [GCDMonoid α] (a : α) : Associated (gcd a 0) a :=
  associated_of_dvd_dvd (gcd_dvd_left a 0) (dvd_gcd (dvd_refl a) (dvd_zero _))


@[simp]
theorem gcd_eq_zero_iff [GCDMonoid α] (a b : α) : gcd a b = 0 ↔ a = 0 ∧ b = 0 :=
  Iff.intro
    (fun h => by
      /-
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : GCDMonoid α
        a b : α
        h : Eq (GCDMonoid.gcd a b) 0
        ⊢ And (Eq a 0) (Eq b 0)
      -/
      let ⟨ca, ha⟩ := gcd_dvd_left a b
      /-
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : GCDMonoid α
        a b : α
        h : Eq (GCDMonoid.gcd a b) 0
        ca : α
        ha : Eq a (HMul.hMul (GCDMonoid.gcd a b) ca)
        ⊢ And (Eq a 0) (Eq b 0)
      -/
      let ⟨cb, hb⟩ := gcd_dvd_right a b
      /-
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : GCDMonoid α
        a b : α
        h : Eq (GCDMonoid.gcd a b) 0
        ca : α
        ha : Eq a (HMul.hMul (GCDMonoid.gcd a b) ca)
        cb : α
        hb : Eq b (HMul.hMul (GCDMonoid.gcd a b) cb)
        ⊢ And (Eq a 0) (Eq b 0)
      -/
      rw [h, zero_mul] at ha hb
      /-
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : GCDMonoid α
        a b : α
        h : Eq (GCDMonoid.gcd a b) 0
        ca : α
        ha : Eq a 0
        cb : α
        hb : Eq b 0
        ⊢ And (Eq a 0) (Eq b 0)
      -/
      exact ⟨ha, hb⟩)
      /-
        🎉 no goals
      -/
    fun ⟨ha, hb⟩ => by
    /-
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : GCDMonoid α
      a b : α
      x✝ : And (Eq a 0) (Eq b 0)
      ha : Eq a 0
      hb : Eq b 0
      ⊢ Eq (GCDMonoid.gcd a b) 0
    -/
    rw [ha, hb, ← zero_dvd_iff]
    /-
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : GCDMonoid α
      a b : α
      x✝ : And (Eq a 0) (Eq b 0)
      ha : Eq a 0
      hb : Eq b 0
      ⊢ Dvd.dvd 0 (GCDMonoid.gcd 0 0)
    -/
                      /-
                        🎉 no goals
                      -/
    apply dvd_gcd <;> rfl
                      /-
                        🎉 no goals
                      -/


@[simp]
theorem gcd_one_left [NormalizedGCDMonoid α] (a : α) : gcd 1 a = 1 :=
  dvd_antisymm_of_normalize_eq (normalize_gcd _ _) normalize_one (gcd_dvd_left _ _) (one_dvd _)


@[simp]
theorem isUnit_gcd_one_left [GCDMonoid α] (a : α) : IsUnit (gcd 1 a) :=
  isUnit_of_dvd_one (gcd_dvd_left _ _)


                                                                           /-
                                                                             α : Type u_1
                                                                             inst✝¹ : CancelCommMonoidWithZero α
                                                                             inst✝ : GCDMonoid α
                                                                             a : α
                                                                             ⊢ Associated (GCDMonoid.gcd 1 a) 1
                                                                           -/
theorem gcd_one_left' [GCDMonoid α] (a : α) : Associated (gcd 1 a) 1 := by simp
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


@[simp]
theorem gcd_one_right [NormalizedGCDMonoid α] (a : α) : gcd a 1 = 1 :=
  dvd_antisymm_of_normalize_eq (normalize_gcd _ _) normalize_one (gcd_dvd_right _ _) (one_dvd _)


@[simp]
theorem isUnit_gcd_one_right [GCDMonoid α] (a : α) : IsUnit (gcd a 1) :=
  isUnit_of_dvd_one (gcd_dvd_right _ _)


                                                                            /-
                                                                              α : Type u_1
                                                                              inst✝¹ : CancelCommMonoidWithZero α
                                                                              inst✝ : GCDMonoid α
                                                                              a : α
                                                                              ⊢ Associated (GCDMonoid.gcd a 1) 1
                                                                            -/
theorem gcd_one_right' [GCDMonoid α] (a : α) : Associated (gcd a 1) 1 := by simp
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


theorem gcd_dvd_gcd [GCDMonoid α] {a b c d : α} (hab : a ∣ b) (hcd : c ∣ d) : gcd a c ∣ gcd b d :=
  dvd_gcd ((gcd_dvd_left _ _).trans hab) ((gcd_dvd_right _ _).trans hcd)


protected theorem Associated.gcd [GCDMonoid α]
    {a₁ a₂ b₁ b₂ : α} (ha : Associated a₁ a₂) (hb : Associated b₁ b₂) :
    Associated (gcd a₁ b₁) (gcd a₂ b₂) :=
  associated_of_dvd_dvd (gcd_dvd_gcd ha.dvd hb.dvd) (gcd_dvd_gcd ha.dvd' hb.dvd')


@[simp]
theorem gcd_same [NormalizedGCDMonoid α] (a : α) : gcd a a = normalize a :=
  gcd_eq_normalize (gcd_dvd_left _ _) (dvd_gcd (dvd_refl a) (dvd_refl a))


@[simp]
theorem gcd_mul_left [NormalizedGCDMonoid α] (a b c : α) :
    gcd (a * b) (a * c) = normalize a * gcd b c :=
                /-
                  α : Type u_1
                  inst✝¹ : CancelCommMonoidWithZero α
                  inst✝ : NormalizedGCDMonoid α
                  a b c : α
                  ⊢ Eq a 0 → Eq (GCDMonoid.gcd (HMul.hMul a b) (HMul.hMul a c)) (HMul.hMul (norm …
                -/
  (by_cases (by rintro rfl; simp only [zero_mul, gcd_zero_left, normalize_zero]))
                            /-
                              🎉 no goals
                            -/
    fun ha : a ≠ 0 =>
                                                              /-
                                                                α : Type u_1
                                                                inst✝¹ : CancelCommMonoidWithZero α
                                                                inst✝ : NormalizedGCDMonoid α
                                                                a b c : α
                                                                ha : Ne a 0
                                                                this : Eq (GCDMonoid.gcd (HMul.hMul a b) (HMul.hMul a c)) (normalize (HMul.hMu …
                                                                ⊢ Eq (GCDMonoid.gcd (HMul.hMul a b) (HMul.hMul a c)) (HMul.hMul (normalize a)  …
                                                              -/
    suffices gcd (a * b) (a * c) = normalize (a * gcd b c) by simpa
                                                              /-
                                                                🎉 no goals
                                                              -/
    let ⟨d, eq⟩ := dvd_gcd (dvd_mul_right a b) (dvd_mul_right a c)
    gcd_eq_normalize
      (eq.symm ▸ mul_dvd_mul_left a
        (show d ∣ gcd b c from
          dvd_gcd ((mul_dvd_mul_iff_left ha).1 <| eq ▸ gcd_dvd_left _ _)
            ((mul_dvd_mul_iff_left ha).1 <| eq ▸ gcd_dvd_right _ _)))
      (dvd_gcd (mul_dvd_mul_left a <| gcd_dvd_left _ _) (mul_dvd_mul_left a <| gcd_dvd_right _ _))


theorem gcd_mul_left' [GCDMonoid α] (a b c : α) :
    Associated (gcd (a * b) (a * c)) (a * gcd b c) := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : GCDMonoid α
    a b c : α
    ⊢ Associated (GCDMonoid.gcd (HMul.hMul a b) (HMul.hMul a c)) (HMul.hMul a (GCD …
  -/
  obtain rfl | ha := eq_or_ne a 0
    /-
      case inl
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : GCDMonoid α
      b c : α
      ⊢ Associated (GCDMonoid.gcd (HMul.hMul 0 b) (HMul.hMul 0 c)) (HMul.hMul 0 (GCD …
    -/
  · simp only [zero_mul, gcd_zero_left']
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : GCDMonoid α
    a b c : α
    ha : Ne a 0
    ⊢ Associated (GCDMonoid.gcd (HMul.hMul a b) (HMul.hMul a c)) (HMul.hMul a (GCD …
  -/
  obtain ⟨d, eq⟩ := dvd_gcd (dvd_mul_right a b) (dvd_mul_right a c)
  /-
    case inr.intro
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : GCDMonoid α
    a b c : α
    ha : Ne a 0
    d : α
    eq : Eq (GCDMonoid.gcd (HMul.hMul a b) (HMul.hMul a c)) (HMul.hMul a d)
    ⊢ Associated (GCDMonoid.gcd (HMul.hMul a b) (HMul.hMul a c)) (HMul.hMul a (GCD …
  -/
  apply associated_of_dvd_dvd
    /-
      case inr.intro.hab
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : GCDMonoid α
      a b c : α
      ha : Ne a 0
      d : α
      eq : Eq (GCDMonoid.gcd (HMul.hMul a b) (HMul.hMul a c)) (HMul.hMul a d)
      ⊢ Dvd.dvd (GCDMonoid.gcd (HMul.hMul a b) (HMul.hMul a c)) (HMul.hMul a (GCDMon …
    -/
  · rw [eq]
    /-
      case inr.intro.hab
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : GCDMonoid α
      a b c : α
      ha : Ne a 0
      d : α
      eq : Eq (GCDMonoid.gcd (HMul.hMul a b) (HMul.hMul a c)) (HMul.hMul a d)
      ⊢ Dvd.dvd (HMul.hMul a d) (HMul.hMul a (GCDMonoid.gcd b c))
    -/
    apply mul_dvd_mul_left
    exact
      dvd_gcd ((mul_dvd_mul_iff_left ha).1 <| eq ▸ gcd_dvd_left _ _)
        ((mul_dvd_mul_iff_left ha).1 <| eq ▸ gcd_dvd_right _ _)
    /-
      case inr.intro.hba
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : GCDMonoid α
      a b c : α
      ha : Ne a 0
      d : α
      eq : Eq (GCDMonoid.gcd (HMul.hMul a b) (HMul.hMul a c)) (HMul.hMul a d)
      ⊢ Dvd.dvd (HMul.hMul a (GCDMonoid.gcd b c)) (GCDMonoid.gcd (HMul.hMul a b) (HM …
    -/
  · exact dvd_gcd (mul_dvd_mul_left a <| gcd_dvd_left _ _) (mul_dvd_mul_left a <| gcd_dvd_right _ _)
    /-
      🎉 no goals
    -/


@[simp]
theorem gcd_mul_right [NormalizedGCDMonoid α] (a b c : α) :
                                                      /-
                                                        α : Type u_1
                                                        inst✝¹ : CancelCommMonoidWithZero α
                                                        inst✝ : NormalizedGCDMonoid α
                                                        a b c : α
                                                        ⊢ Eq (GCDMonoid.gcd (HMul.hMul b a) (HMul.hMul c a)) (HMul.hMul (GCDMonoid.gcd …
                                                      -/
    gcd (b * a) (c * a) = gcd b c * normalize a := by simp only [mul_comm, gcd_mul_left]
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp]
theorem gcd_mul_right' [GCDMonoid α] (a b c : α) :
    Associated (gcd (b * a) (c * a)) (gcd b c * a) := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : GCDMonoid α
    a b c : α
    ⊢ Associated (GCDMonoid.gcd (HMul.hMul b a) (HMul.hMul c a)) (HMul.hMul (GCDMo …
  -/
  simp only [mul_comm, gcd_mul_left']
  /-
    🎉 no goals
  -/


theorem gcd_eq_left_iff [NormalizedGCDMonoid α] (a b : α) (h : normalize a = a) :
    gcd a b = a ↔ a ∣ b :=
  (Iff.intro fun eq => eq ▸ gcd_dvd_right _ _) fun hab =>
    dvd_antisymm_of_normalize_eq (normalize_gcd _ _) h (gcd_dvd_left _ _) (dvd_gcd (dvd_refl a) hab)


theorem gcd_eq_right_iff [NormalizedGCDMonoid α] (a b : α) (h : normalize b = b) :
                              /-
                                α : Type u_1
                                inst✝¹ : CancelCommMonoidWithZero α
                                inst✝ : NormalizedGCDMonoid α
                                a b : α
                                h : Eq (normalize b) b
                                ⊢ Iff (Eq (GCDMonoid.gcd a b) b) (Dvd.dvd b a)
                              -/
    gcd a b = b ↔ b ∣ a := by simpa only [gcd_comm a b] using gcd_eq_left_iff b a h
                              /-
                                🎉 no goals
                              -/


theorem gcd_dvd_gcd_mul_left [GCDMonoid α] (m n k : α) : gcd m n ∣ gcd (k * m) n :=
  gcd_dvd_gcd (dvd_mul_left _ _) dvd_rfl


theorem gcd_dvd_gcd_mul_right [GCDMonoid α] (m n k : α) : gcd m n ∣ gcd (m * k) n :=
  gcd_dvd_gcd (dvd_mul_right _ _) dvd_rfl


theorem gcd_dvd_gcd_mul_left_right [GCDMonoid α] (m n k : α) : gcd m n ∣ gcd m (k * n) :=
  gcd_dvd_gcd dvd_rfl (dvd_mul_left _ _)


theorem gcd_dvd_gcd_mul_right_right [GCDMonoid α] (m n k : α) : gcd m n ∣ gcd m (n * k) :=
  gcd_dvd_gcd dvd_rfl (dvd_mul_right _ _)


theorem Associated.gcd_eq_left [NormalizedGCDMonoid α] {m n : α} (h : Associated m n) (k : α) :
    gcd m k = gcd n k :=
  dvd_antisymm_of_normalize_eq (normalize_gcd _ _) (normalize_gcd _ _) (gcd_dvd_gcd h.dvd dvd_rfl)
    (gcd_dvd_gcd h.symm.dvd dvd_rfl)


theorem Associated.gcd_eq_right [NormalizedGCDMonoid α] {m n : α} (h : Associated m n) (k : α) :
    gcd k m = gcd k n :=
  dvd_antisymm_of_normalize_eq (normalize_gcd _ _) (normalize_gcd _ _) (gcd_dvd_gcd dvd_rfl h.dvd)
    (gcd_dvd_gcd dvd_rfl h.symm.dvd)


theorem dvd_gcd_mul_of_dvd_mul [GCDMonoid α] {m n k : α} (H : k ∣ m * n) : k ∣ gcd k m * n :=
  (dvd_gcd (dvd_mul_right _ n) H).trans (gcd_mul_right' n k m).dvd


theorem dvd_gcd_mul_iff_dvd_mul [GCDMonoid α] {m n k : α} : k ∣ gcd k m * n ↔ k ∣ m * n :=
  ⟨fun h => h.trans (mul_dvd_mul (gcd_dvd_right k m) dvd_rfl), dvd_gcd_mul_of_dvd_mul⟩


theorem dvd_mul_gcd_of_dvd_mul [GCDMonoid α] {m n k : α} (H : k ∣ m * n) : k ∣ m * gcd k n := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : GCDMonoid α
    m n k : α
    H : Dvd.dvd k (HMul.hMul m n)
    ⊢ Dvd.dvd k (HMul.hMul m (GCDMonoid.gcd k n))
  -/
  rw [mul_comm] at H ⊢
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : GCDMonoid α
    m n k : α
    H : Dvd.dvd k (HMul.hMul n m)
    ⊢ Dvd.dvd k (HMul.hMul (GCDMonoid.gcd k n) m)
  -/
  exact dvd_gcd_mul_of_dvd_mul H
  /-
    🎉 no goals
  -/


theorem dvd_mul_gcd_iff_dvd_mul [GCDMonoid α] {m n k : α} : k ∣ m * gcd k n ↔ k ∣ m * n :=
  ⟨fun h => h.trans (mul_dvd_mul dvd_rfl (gcd_dvd_right k n)), dvd_mul_gcd_of_dvd_mul⟩


/-- Represent a divisor of `m * n` as a product of a divisor of `m` and a divisor of `n`.

Note: In general, this representation is highly non-unique.

See `Nat.prodDvdAndDvdOfDvdProd` for a constructive version on `ℕ`. -/
instance [h : Nonempty (GCDMonoid α)] : DecompositionMonoid α where
  primal k m n H := by
    /-
      α : Type u_1
      inst✝ : CancelCommMonoidWithZero α
      h : Nonempty (GCDMonoid α)
      k m n : α
      H : Dvd.dvd k (HMul.hMul m n)
      ⊢ Exists fun a₁ => Exists fun a₂ => And (Dvd.dvd a₁ m) (And (Dvd.dvd a₂ n) (Eq …
    -/
    cases h
    /-
      case intro
      α : Type u_1
      inst✝ : CancelCommMonoidWithZero α
      k m n : α
      H : Dvd.dvd k (HMul.hMul m n)
      val✝ : GCDMonoid α
      ⊢ Exists fun a₁ => Exists fun a₂ => And (Dvd.dvd a₁ m) (And (Dvd.dvd a₂ n) (Eq …
    -/
    by_cases h0 : gcd k m = 0
      /-
        case pos
        α : Type u_1
        inst✝ : CancelCommMonoidWithZero α
        k m n : α
        H : Dvd.dvd k (HMul.hMul m n)
        val✝ : GCDMonoid α
        h0 : Eq (GCDMonoid.gcd k m) 0
        ⊢ Exists fun a₁ => Exists fun a₂ => And (Dvd.dvd a₁ m) (And (Dvd.dvd a₂ n) (Eq …
      -/
    · rw [gcd_eq_zero_iff] at h0
      /-
        case pos
        α : Type u_1
        inst✝ : CancelCommMonoidWithZero α
        k m n : α
        H : Dvd.dvd k (HMul.hMul m n)
        val✝ : GCDMonoid α
        h0 : And (Eq k 0) (Eq m 0)
        ⊢ Exists fun a₁ => Exists fun a₂ => And (Dvd.dvd a₁ m) (And (Dvd.dvd a₂ n) (Eq …
      -/
      rcases h0 with ⟨rfl, rfl⟩
      /-
        case pos.intro
        α : Type u_1
        inst✝ : CancelCommMonoidWithZero α
        n : α
        val✝ : GCDMonoid α
        H : Dvd.dvd 0 (HMul.hMul 0 n)
        ⊢ Exists fun a₁ => Exists fun a₂ => And (Dvd.dvd a₁ 0) (And (Dvd.dvd a₂ n) (Eq …
      -/
      exact ⟨0, n, dvd_refl 0, dvd_refl n, by simp⟩
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        inst✝ : CancelCommMonoidWithZero α
        k m n : α
        H : Dvd.dvd k (HMul.hMul m n)
        val✝ : GCDMonoid α
        h0 : Not (Eq (GCDMonoid.gcd k m) 0)
        ⊢ Exists fun a₁ => Exists fun a₂ => And (Dvd.dvd a₁ m) (And (Dvd.dvd a₂ n) (Eq …
      -/
    · obtain ⟨a, ha⟩ := gcd_dvd_left k m
      /-
        case neg.intro
        α : Type u_1
        inst✝ : CancelCommMonoidWithZero α
        k m n : α
        H : Dvd.dvd k (HMul.hMul m n)
        val✝ : GCDMonoid α
        h0 : Not (Eq (GCDMonoid.gcd k m) 0)
        a : α
        ha : Eq k (HMul.hMul (GCDMonoid.gcd k m) a)
        ⊢ Exists fun a₁ => Exists fun a₂ => And (Dvd.dvd a₁ m) (And (Dvd.dvd a₂ n) (Eq …
      -/
      refine ⟨gcd k m, a, gcd_dvd_right _ _, ?_, ha⟩
      /-
        case neg.intro
        α : Type u_1
        inst✝ : CancelCommMonoidWithZero α
        k m n : α
        H : Dvd.dvd k (HMul.hMul m n)
        val✝ : GCDMonoid α
        h0 : Not (Eq (GCDMonoid.gcd k m) 0)
        a : α
        ha : Eq k (HMul.hMul (GCDMonoid.gcd k m) a)
        ⊢ Dvd.dvd a n
      -/
      rw [← mul_dvd_mul_iff_left h0, ← ha]
      /-
        case neg.intro
        α : Type u_1
        inst✝ : CancelCommMonoidWithZero α
        k m n : α
        H : Dvd.dvd k (HMul.hMul m n)
        val✝ : GCDMonoid α
        h0 : Not (Eq (GCDMonoid.gcd k m) 0)
        a : α
        ha : Eq k (HMul.hMul (GCDMonoid.gcd k m) a)
        ⊢ Dvd.dvd k (HMul.hMul (GCDMonoid.gcd k m) n)
      -/
      exact dvd_gcd_mul_of_dvd_mul H
      /-
        🎉 no goals
      -/


theorem gcd_mul_dvd_mul_gcd [GCDMonoid α] (k m n : α) : gcd k (m * n) ∣ gcd k m * gcd k n := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : GCDMonoid α
    k m n : α
    ⊢ Dvd.dvd (GCDMonoid.gcd k (HMul.hMul m n)) (HMul.hMul (GCDMonoid.gcd k m) (GC …
  -/
  obtain ⟨m', n', hm', hn', h⟩ := exists_dvd_and_dvd_of_dvd_mul (gcd_dvd_right k (m * n))
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : GCDMonoid α
    k m n m' n' : α
    hm' : Dvd.dvd m' m
    hn' : Dvd.dvd n' n
    h : Eq (GCDMonoid.gcd k (HMul.hMul m n)) (HMul.hMul m' n')
    ⊢ Dvd.dvd (GCDMonoid.gcd k (HMul.hMul m n)) (HMul.hMul (GCDMonoid.gcd k m) (GC …
  -/
  replace h : gcd k (m * n) = m' * n' := h
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : GCDMonoid α
    k m n m' n' : α
    hm' : Dvd.dvd m' m
    hn' : Dvd.dvd n' n
    h : Eq (GCDMonoid.gcd k (HMul.hMul m n)) (HMul.hMul m' n')
    ⊢ Dvd.dvd (GCDMonoid.gcd k (HMul.hMul m n)) (HMul.hMul (GCDMonoid.gcd k m) (GC …
  -/
  rw [h]
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : GCDMonoid α
    k m n m' n' : α
    hm' : Dvd.dvd m' m
    hn' : Dvd.dvd n' n
    h : Eq (GCDMonoid.gcd k (HMul.hMul m n)) (HMul.hMul m' n')
    ⊢ Dvd.dvd (HMul.hMul m' n') (HMul.hMul (GCDMonoid.gcd k m) (GCDMonoid.gcd k n))
  -/
  have hm'n' : m' * n' ∣ k := h ▸ gcd_dvd_left _ _
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : GCDMonoid α
    k m n m' n' : α
    hm' : Dvd.dvd m' m
    hn' : Dvd.dvd n' n
    h : Eq (GCDMonoid.gcd k (HMul.hMul m n)) (HMul.hMul m' n')
    hm'n' : Dvd.dvd (HMul.hMul m' n') k
    ⊢ Dvd.dvd (HMul.hMul m' n') (HMul.hMul (GCDMonoid.gcd k m) (GCDMonoid.gcd k n))
  -/
  apply mul_dvd_mul
    /-
      case intro.intro.intro.intro.a
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : GCDMonoid α
      k m n m' n' : α
      hm' : Dvd.dvd m' m
      hn' : Dvd.dvd n' n
      h : Eq (GCDMonoid.gcd k (HMul.hMul m n)) (HMul.hMul m' n')
      hm'n' : Dvd.dvd (HMul.hMul m' n') k
      ⊢ Dvd.dvd m' (GCDMonoid.gcd k m)
    -/
  · have hm'k : m' ∣ k := (dvd_mul_right m' n').trans hm'n'
    /-
      case intro.intro.intro.intro.a
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : GCDMonoid α
      k m n m' n' : α
      hm' : Dvd.dvd m' m
      hn' : Dvd.dvd n' n
      h : Eq (GCDMonoid.gcd k (HMul.hMul m n)) (HMul.hMul m' n')
      hm'n' : Dvd.dvd (HMul.hMul m' n') k
      hm'k : Dvd.dvd m' k
      ⊢ Dvd.dvd m' (GCDMonoid.gcd k m)
    -/
    exact dvd_gcd hm'k hm'
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.a
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : GCDMonoid α
      k m n m' n' : α
      hm' : Dvd.dvd m' m
      hn' : Dvd.dvd n' n
      h : Eq (GCDMonoid.gcd k (HMul.hMul m n)) (HMul.hMul m' n')
      hm'n' : Dvd.dvd (HMul.hMul m' n') k
      ⊢ Dvd.dvd n' (GCDMonoid.gcd k n)
    -/
  · have hn'k : n' ∣ k := (dvd_mul_left n' m').trans hm'n'
    /-
      case intro.intro.intro.intro.a
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : GCDMonoid α
      k m n m' n' : α
      hm' : Dvd.dvd m' m
      hn' : Dvd.dvd n' n
      h : Eq (GCDMonoid.gcd k (HMul.hMul m n)) (HMul.hMul m' n')
      hm'n' : Dvd.dvd (HMul.hMul m' n') k
      hn'k : Dvd.dvd n' k
      ⊢ Dvd.dvd n' (GCDMonoid.gcd k n)
    -/
    exact dvd_gcd hn'k hn'
    /-
      🎉 no goals
    -/


theorem gcd_pow_right_dvd_pow_gcd [GCDMonoid α] {a b : α} {k : ℕ} :
    gcd a (b ^ k) ∣ gcd a b ^ k := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : GCDMonoid α
    a b : α
    k : Nat
    ⊢ Dvd.dvd (GCDMonoid.gcd a (HPow.hPow b k)) (HPow.hPow (GCDMonoid.gcd a b) k)
  -/
  by_cases hg : gcd a b = 0
    /-
      case pos
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : GCDMonoid α
      a b : α
      k : Nat
      hg : Eq (GCDMonoid.gcd a b) 0
      ⊢ Dvd.dvd (GCDMonoid.gcd a (HPow.hPow b k)) (HPow.hPow (GCDMonoid.gcd a b) k)
    -/
  · rw [gcd_eq_zero_iff] at hg
    /-
      case pos
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : GCDMonoid α
      a b : α
      k : Nat
      hg : And (Eq a 0) (Eq b 0)
      ⊢ Dvd.dvd (GCDMonoid.gcd a (HPow.hPow b k)) (HPow.hPow (GCDMonoid.gcd a b) k)
    -/
    rcases hg with ⟨rfl, rfl⟩
    exact
      (gcd_zero_left' (0 ^ k : α)).dvd.trans
        (pow_dvd_pow_of_dvd (gcd_zero_left' (0 : α)).symm.dvd _)
  · induction k with
    | zero => rw [pow_zero, pow_zero]; exact (gcd_one_right' a).dvd
    | succ k hk =>
      rw [pow_succ', pow_succ']
      trans gcd a b * gcd a (b ^ k)
      · exact gcd_mul_dvd_mul_gcd a b (b ^ k)
      · exact (mul_dvd_mul_iff_left hg).mpr hk


theorem gcd_pow_left_dvd_pow_gcd [GCDMonoid α] {a b : α} {k : ℕ} : gcd (a ^ k) b ∣ gcd a b ^ k :=
  calc
    gcd (a ^ k) b ∣ gcd b (a ^ k) := (gcd_comm' _ _).dvd
    _ ∣ gcd b a ^ k := gcd_pow_right_dvd_pow_gcd
    _ ∣ gcd a b ^ k := pow_dvd_pow_of_dvd (gcd_comm' _ _).dvd _


theorem pow_dvd_of_mul_eq_pow [GCDMonoid α] {a b c d₁ d₂ : α} (ha : a ≠ 0) (hab : IsUnit (gcd a b))
    {k : ℕ} (h : a * b = c ^ k) (hc : c = d₁ * d₂) (hd₁ : d₁ ∣ a) : d₁ ^ k ≠ 0 ∧ d₁ ^ k ∣ a := by
  have h1 : IsUnit (gcd (d₁ ^ k) b) := by
    apply isUnit_of_dvd_one
    trans gcd d₁ b ^ k
    · exact gcd_pow_left_dvd_pow_gcd
    · apply IsUnit.dvd
      apply IsUnit.pow
      apply isUnit_of_dvd_one
      apply dvd_trans _ hab.dvd
      apply gcd_dvd_gcd hd₁ (dvd_refl b)
  have h2 : d₁ ^ k ∣ a * b := by
    use d₂ ^ k
    rw [h, hc]
    exact mul_pow d₁ d₂ k
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : GCDMonoid α
    a b c d₁ d₂ : α
    ha : Ne a 0
    hab : IsUnit (GCDMonoid.gcd a b)
    k : Nat
    h : Eq (HMul.hMul a b) (HPow.hPow c k)
    hc : Eq c (HMul.hMul d₁ d₂)
    hd₁ : Dvd.dvd d₁ a
    h1 : IsUnit (GCDMonoid.gcd (HPow.hPow d₁ k) b)
    h2 : Dvd.dvd (HPow.hPow d₁ k) (HMul.hMul a b)
    ⊢ And (Ne (HPow.hPow d₁ k) 0) (Dvd.dvd (HPow.hPow d₁ k) a)
  -/
  rw [mul_comm] at h2
  have h3 : d₁ ^ k ∣ a := by
    apply (dvd_gcd_mul_of_dvd_mul h2).trans
    rw [h1.mul_left_dvd]
  have h4 : d₁ ^ k ≠ 0 := by
    intro hdk
    rw [hdk] at h3
    apply absurd (zero_dvd_iff.mp h3) ha
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : GCDMonoid α
    a b c d₁ d₂ : α
    ha : Ne a 0
    hab : IsUnit (GCDMonoid.gcd a b)
    k : Nat
    h : Eq (HMul.hMul a b) (HPow.hPow c k)
    hc : Eq c (HMul.hMul d₁ d₂)
    hd₁ : Dvd.dvd d₁ a
    h1 : IsUnit (GCDMonoid.gcd (HPow.hPow d₁ k) b)
    h2 : Dvd.dvd (HPow.hPow d₁ k) (HMul.hMul b a)
    h3 : Dvd.dvd (HPow.hPow d₁ k) a
    h4 : Ne (HPow.hPow d₁ k) 0
    ⊢ And (Ne (HPow.hPow d₁ k) 0) (Dvd.dvd (HPow.hPow d₁ k) a)
  -/
  exact ⟨h4, h3⟩
  /-
    🎉 no goals
  -/


theorem exists_associated_pow_of_mul_eq_pow [GCDMonoid α] {a b c : α} (hab : IsUnit (gcd a b))
    {k : ℕ} (h : a * b = c ^ k) : ∃ d : α, Associated (d ^ k) a := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : GCDMonoid α
    a b c : α
    hab : IsUnit (GCDMonoid.gcd a b)
    k : Nat
    h : Eq (HMul.hMul a b) (HPow.hPow c k)
    ⊢ Exists fun d => Associated (HPow.hPow d k) a
  -/
  cases subsingleton_or_nontrivial α
    /-
      case inl
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : GCDMonoid α
      a b c : α
      hab : IsUnit (GCDMonoid.gcd a b)
      k : Nat
      h : Eq (HMul.hMul a b) (HPow.hPow c k)
      h✝ : Subsingleton α
      ⊢ Exists fun d => Associated (HPow.hPow d k) a
    -/
  · use 0
    /-
      case h
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : GCDMonoid α
      a b c : α
      hab : IsUnit (GCDMonoid.gcd a b)
      k : Nat
      h : Eq (HMul.hMul a b) (HPow.hPow c k)
      h✝ : Subsingleton α
      ⊢ Associated (HPow.hPow 0 k) a
    -/
    rw [Subsingleton.elim a (0 ^ k)]
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : GCDMonoid α
    a b c : α
    hab : IsUnit (GCDMonoid.gcd a b)
    k : Nat
    h : Eq (HMul.hMul a b) (HPow.hPow c k)
    h✝ : Nontrivial α
    ⊢ Exists fun d => Associated (HPow.hPow d k) a
  -/
  by_cases ha : a = 0
    /-
      case pos
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : GCDMonoid α
      a b c : α
      hab : IsUnit (GCDMonoid.gcd a b)
      k : Nat
      h : Eq (HMul.hMul a b) (HPow.hPow c k)
      h✝ : Nontrivial α
      ha : Eq a 0
      ⊢ Exists fun d => Associated (HPow.hPow d k) a
    -/
  · use 0
    /-
      case h
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : GCDMonoid α
      a b c : α
      hab : IsUnit (GCDMonoid.gcd a b)
      k : Nat
      h : Eq (HMul.hMul a b) (HPow.hPow c k)
      h✝ : Nontrivial α
      ha : Eq a 0
      ⊢ Associated (HPow.hPow 0 k) a
    -/
    obtain rfl | hk := eq_or_ne k 0
      /-
        case h.inl
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : GCDMonoid α
        a b c : α
        hab : IsUnit (GCDMonoid.gcd a b)
        h✝ : Nontrivial α
        ha : Eq a 0
        h : Eq (HMul.hMul a b) (HPow.hPow c 0)
        ⊢ Associated (HPow.hPow 0 0) a
      -/
    · simp [ha] at h
      /-
        🎉 no goals
      -/
      /-
        case h.inr
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : GCDMonoid α
        a b c : α
        hab : IsUnit (GCDMonoid.gcd a b)
        k : Nat
        h : Eq (HMul.hMul a b) (HPow.hPow c k)
        h✝ : Nontrivial α
        ha : Eq a 0
        hk : Ne k 0
        ⊢ Associated (HPow.hPow 0 k) a
      -/
    · rw [ha, zero_pow hk]
      /-
        🎉 no goals
      -/
  /-
    case neg
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : GCDMonoid α
    a b c : α
    hab : IsUnit (GCDMonoid.gcd a b)
    k : Nat
    h : Eq (HMul.hMul a b) (HPow.hPow c k)
    h✝ : Nontrivial α
    ha : Not (Eq a 0)
    ⊢ Exists fun d => Associated (HPow.hPow d k) a
  -/
  by_cases hb : b = 0
    /-
      case pos
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : GCDMonoid α
      a b c : α
      hab : IsUnit (GCDMonoid.gcd a b)
      k : Nat
      h : Eq (HMul.hMul a b) (HPow.hPow c k)
      h✝ : Nontrivial α
      ha : Not (Eq a 0)
      hb : Eq b 0
      ⊢ Exists fun d => Associated (HPow.hPow d k) a
    -/
  · use 1
    /-
      case h
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : GCDMonoid α
      a b c : α
      hab : IsUnit (GCDMonoid.gcd a b)
      k : Nat
      h : Eq (HMul.hMul a b) (HPow.hPow c k)
      h✝ : Nontrivial α
      ha : Not (Eq a 0)
      hb : Eq b 0
      ⊢ Associated (HPow.hPow 1 k) a
    -/
    rw [one_pow]
    /-
      case h
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : GCDMonoid α
      a b c : α
      hab : IsUnit (GCDMonoid.gcd a b)
      k : Nat
      h : Eq (HMul.hMul a b) (HPow.hPow c k)
      h✝ : Nontrivial α
      ha : Not (Eq a 0)
      hb : Eq b 0
      ⊢ Associated 1 a
    -/
    apply (associated_one_iff_isUnit.mpr hab).symm.trans
    /-
      case h
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : GCDMonoid α
      a b c : α
      hab : IsUnit (GCDMonoid.gcd a b)
      k : Nat
      h : Eq (HMul.hMul a b) (HPow.hPow c k)
      h✝ : Nontrivial α
      ha : Not (Eq a 0)
      hb : Eq b 0
      ⊢ Associated (GCDMonoid.gcd a b) a
    -/
    rw [hb]
    /-
      case h
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : GCDMonoid α
      a b c : α
      hab : IsUnit (GCDMonoid.gcd a b)
      k : Nat
      h : Eq (HMul.hMul a b) (HPow.hPow c k)
      h✝ : Nontrivial α
      ha : Not (Eq a 0)
      hb : Eq b 0
      ⊢ Associated (GCDMonoid.gcd a 0) a
    -/
    exact gcd_zero_right' a
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : GCDMonoid α
    a b c : α
    hab : IsUnit (GCDMonoid.gcd a b)
    k : Nat
    h : Eq (HMul.hMul a b) (HPow.hPow c k)
    h✝ : Nontrivial α
    ha : Not (Eq a 0)
    hb : Not (Eq b 0)
    ⊢ Exists fun d => Associated (HPow.hPow d k) a
  -/
  obtain rfl | hk := k.eq_zero_or_pos
    /-
      case neg.inl
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : GCDMonoid α
      a b c : α
      hab : IsUnit (GCDMonoid.gcd a b)
      h✝ : Nontrivial α
      ha : Not (Eq a 0)
      hb : Not (Eq b 0)
      h : Eq (HMul.hMul a b) (HPow.hPow c 0)
      ⊢ Exists fun d => Associated (HPow.hPow d 0) a
    -/
  · use 1
    /-
      case h
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : GCDMonoid α
      a b c : α
      hab : IsUnit (GCDMonoid.gcd a b)
      h✝ : Nontrivial α
      ha : Not (Eq a 0)
      hb : Not (Eq b 0)
      h : Eq (HMul.hMul a b) (HPow.hPow c 0)
      ⊢ Associated (HPow.hPow 1 0) a
    -/
    rw [pow_zero] at h ⊢
    /-
      case h
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : GCDMonoid α
      a b c : α
      hab : IsUnit (GCDMonoid.gcd a b)
      h✝ : Nontrivial α
      ha : Not (Eq a 0)
      hb : Not (Eq b 0)
      h : Eq (HMul.hMul a b) 1
      ⊢ Associated 1 a
    -/
    use Units.mkOfMulEqOne _ _ h
    /-
      case h
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : GCDMonoid α
      a b c : α
      hab : IsUnit (GCDMonoid.gcd a b)
      h✝ : Nontrivial α
      ha : Not (Eq a 0)
      hb : Not (Eq b 0)
      h : Eq (HMul.hMul a b) 1
      ⊢ Eq (HMul.hMul 1 ↑(Units.mkOfMulEqOne a b h)) a
    -/
    rw [Units.val_mkOfMulEqOne, one_mul]
    /-
      🎉 no goals
    -/
  have hc : c ∣ a * b := by
    rw [h]
    exact dvd_pow_self _ hk.ne'
  /-
    case neg.inr
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : GCDMonoid α
    a b c : α
    hab : IsUnit (GCDMonoid.gcd a b)
    k : Nat
    h : Eq (HMul.hMul a b) (HPow.hPow c k)
    h✝ : Nontrivial α
    ha : Not (Eq a 0)
    hb : Not (Eq b 0)
    hk : GT.gt k 0
    hc : Dvd.dvd c (HMul.hMul a b)
    ⊢ Exists fun d => Associated (HPow.hPow d k) a
  -/
  obtain ⟨d₁, d₂, hd₁, hd₂, hc⟩ := exists_dvd_and_dvd_of_dvd_mul hc
  /-
    case neg.inr.intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : GCDMonoid α
    a b c : α
    hab : IsUnit (GCDMonoid.gcd a b)
    k : Nat
    h : Eq (HMul.hMul a b) (HPow.hPow c k)
    h✝ : Nontrivial α
    ha : Not (Eq a 0)
    hb : Not (Eq b 0)
    hk : GT.gt k 0
    hc✝ : Dvd.dvd c (HMul.hMul a b)
    d₁ d₂ : α
    hd₁ : Dvd.dvd d₁ a
    hd₂ : Dvd.dvd d₂ b
    hc : Eq c (HMul.hMul d₁ d₂)
    ⊢ Exists fun d => Associated (HPow.hPow d k) a
  -/
  use d₁
  /-
    case h
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : GCDMonoid α
    a b c : α
    hab : IsUnit (GCDMonoid.gcd a b)
    k : Nat
    h : Eq (HMul.hMul a b) (HPow.hPow c k)
    h✝ : Nontrivial α
    ha : Not (Eq a 0)
    hb : Not (Eq b 0)
    hk : GT.gt k 0
    hc✝ : Dvd.dvd c (HMul.hMul a b)
    d₁ d₂ : α
    hd₁ : Dvd.dvd d₁ a
    hd₂ : Dvd.dvd d₂ b
    hc : Eq c (HMul.hMul d₁ d₂)
    ⊢ Associated (HPow.hPow d₁ k) a
  -/
  obtain ⟨h0₁, ⟨a', ha'⟩⟩ := pow_dvd_of_mul_eq_pow ha hab h hc hd₁
  /-
    case h.intro.intro
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : GCDMonoid α
    a b c : α
    hab : IsUnit (GCDMonoid.gcd a b)
    k : Nat
    h : Eq (HMul.hMul a b) (HPow.hPow c k)
    h✝ : Nontrivial α
    ha : Not (Eq a 0)
    hb : Not (Eq b 0)
    hk : GT.gt k 0
    hc✝ : Dvd.dvd c (HMul.hMul a b)
    d₁ d₂ : α
    hd₁ : Dvd.dvd d₁ a
    hd₂ : Dvd.dvd d₂ b
    hc : Eq c (HMul.hMul d₁ d₂)
    h0₁ : Ne (HPow.hPow d₁ k) 0
    a' : α
    ha' : Eq a (HMul.hMul (HPow.hPow d₁ k) a')
    ⊢ Associated (HPow.hPow d₁ k) a
  -/
  rw [mul_comm] at h hc
  /-
    case h.intro.intro
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : GCDMonoid α
    a b c : α
    hab : IsUnit (GCDMonoid.gcd a b)
    k : Nat
    h : Eq (HMul.hMul b a) (HPow.hPow c k)
    h✝ : Nontrivial α
    ha : Not (Eq a 0)
    hb : Not (Eq b 0)
    hk : GT.gt k 0
    hc✝ : Dvd.dvd c (HMul.hMul a b)
    d₁ d₂ : α
    hd₁ : Dvd.dvd d₁ a
    hd₂ : Dvd.dvd d₂ b
    hc : Eq c (HMul.hMul d₂ d₁)
    h0₁ : Ne (HPow.hPow d₁ k) 0
    a' : α
    ha' : Eq a (HMul.hMul (HPow.hPow d₁ k) a')
    ⊢ Associated (HPow.hPow d₁ k) a
  -/
  rw [(gcd_comm' a b).isUnit_iff] at hab
  /-
    case h.intro.intro
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : GCDMonoid α
    a b c : α
    hab : IsUnit (GCDMonoid.gcd b a)
    k : Nat
    h : Eq (HMul.hMul b a) (HPow.hPow c k)
    h✝ : Nontrivial α
    ha : Not (Eq a 0)
    hb : Not (Eq b 0)
    hk : GT.gt k 0
    hc✝ : Dvd.dvd c (HMul.hMul a b)
    d₁ d₂ : α
    hd₁ : Dvd.dvd d₁ a
    hd₂ : Dvd.dvd d₂ b
    hc : Eq c (HMul.hMul d₂ d₁)
    h0₁ : Ne (HPow.hPow d₁ k) 0
    a' : α
    ha' : Eq a (HMul.hMul (HPow.hPow d₁ k) a')
    ⊢ Associated (HPow.hPow d₁ k) a
  -/
  obtain ⟨h0₂, ⟨b', hb'⟩⟩ := pow_dvd_of_mul_eq_pow hb hab h hc hd₂
  /-
    case h.intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : GCDMonoid α
    a b c : α
    hab : IsUnit (GCDMonoid.gcd b a)
    k : Nat
    h : Eq (HMul.hMul b a) (HPow.hPow c k)
    h✝ : Nontrivial α
    ha : Not (Eq a 0)
    hb : Not (Eq b 0)
    hk : GT.gt k 0
    hc✝ : Dvd.dvd c (HMul.hMul a b)
    d₁ d₂ : α
    hd₁ : Dvd.dvd d₁ a
    hd₂ : Dvd.dvd d₂ b
    hc : Eq c (HMul.hMul d₂ d₁)
    h0₁ : Ne (HPow.hPow d₁ k) 0
    a' : α
    ha' : Eq a (HMul.hMul (HPow.hPow d₁ k) a')
    h0₂ : Ne (HPow.hPow d₂ k) 0
    b' : α
    hb' : Eq b (HMul.hMul (HPow.hPow d₂ k) b')
    ⊢ Associated (HPow.hPow d₁ k) a
  -/
  rw [ha', hb', hc, mul_pow] at h
  have h' : a' * b' = 1 := by
    apply (mul_right_inj' h0₁).mp
    rw [mul_one]
    apply (mul_right_inj' h0₂).mp
    rw [← h]
    rw [mul_assoc, mul_comm a', ← mul_assoc _ b', ← mul_assoc b', mul_comm b']
  /-
    case h.intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : GCDMonoid α
    a b c : α
    hab : IsUnit (GCDMonoid.gcd b a)
    k : Nat
    h✝ : Nontrivial α
    ha : Not (Eq a 0)
    hb : Not (Eq b 0)
    hk : GT.gt k 0
    hc✝ : Dvd.dvd c (HMul.hMul a b)
    d₁ d₂ : α
    hd₁ : Dvd.dvd d₁ a
    hd₂ : Dvd.dvd d₂ b
    hc : Eq c (HMul.hMul d₂ d₁)
    h0₁ : Ne (HPow.hPow d₁ k) 0
    a' : α
    ha' : Eq a (HMul.hMul (HPow.hPow d₁ k) a')
    h0₂ : Ne (HPow.hPow d₂ k) 0
    b' : α
    h : Eq (HMul.hMul (HMul.hMul (HPow.hPow d₂ k) b') (HMul.hMul (HPow.hPow d₁ k)  …
    hb' : Eq b (HMul.hMul (HPow.hPow d₂ k) b')
    h' : Eq (HMul.hMul a' b') 1
    ⊢ Associated (HPow.hPow d₁ k) a
  -/
  use Units.mkOfMulEqOne _ _ h'
  /-
    case h
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : GCDMonoid α
    a b c : α
    hab : IsUnit (GCDMonoid.gcd b a)
    k : Nat
    h✝ : Nontrivial α
    ha : Not (Eq a 0)
    hb : Not (Eq b 0)
    hk : GT.gt k 0
    hc✝ : Dvd.dvd c (HMul.hMul a b)
    d₁ d₂ : α
    hd₁ : Dvd.dvd d₁ a
    hd₂ : Dvd.dvd d₂ b
    hc : Eq c (HMul.hMul d₂ d₁)
    h0₁ : Ne (HPow.hPow d₁ k) 0
    a' : α
    ha' : Eq a (HMul.hMul (HPow.hPow d₁ k) a')
    h0₂ : Ne (HPow.hPow d₂ k) 0
    b' : α
    h : Eq (HMul.hMul (HMul.hMul (HPow.hPow d₂ k) b') (HMul.hMul (HPow.hPow d₁ k)  …
    hb' : Eq b (HMul.hMul (HPow.hPow d₂ k) b')
    h' : Eq (HMul.hMul a' b') 1
    ⊢ Eq (HMul.hMul (HPow.hPow d₁ k) ↑(Units.mkOfMulEqOne a' b' h')) a
  -/
  rw [Units.val_mkOfMulEqOne, ha']
  /-
    🎉 no goals
  -/


theorem exists_eq_pow_of_mul_eq_pow [GCDMonoid α] [Subsingleton αˣ]
    {a b c : α} (hab : IsUnit (gcd a b)) {k : ℕ} (h : a * b = c ^ k) : ∃ d : α, a = d ^ k :=
  let ⟨d, hd⟩ := exists_associated_pow_of_mul_eq_pow hab h
  ⟨d, (associated_iff_eq.mp hd).symm⟩


theorem gcd_greatest {α : Type*} [CancelCommMonoidWithZero α] [NormalizedGCDMonoid α] {a b d : α}
    (hda : d ∣ a) (hdb : d ∣ b) (hd : ∀ e : α, e ∣ a → e ∣ b → e ∣ d) :
    GCDMonoid.gcd a b = normalize d :=
  haveI h := hd _ (GCDMonoid.gcd_dvd_left a b) (GCDMonoid.gcd_dvd_right a b)
  gcd_eq_normalize h (GCDMonoid.dvd_gcd hda hdb)


theorem gcd_greatest_associated {α : Type*} [CancelCommMonoidWithZero α] [GCDMonoid α] {a b d : α}
    (hda : d ∣ a) (hdb : d ∣ b) (hd : ∀ e : α, e ∣ a → e ∣ b → e ∣ d) :
    Associated d (GCDMonoid.gcd a b) :=
  haveI h := hd _ (GCDMonoid.gcd_dvd_left a b) (GCDMonoid.gcd_dvd_right a b)
  associated_of_dvd_dvd (GCDMonoid.dvd_gcd hda hdb) h


theorem isUnit_gcd_of_eq_mul_gcd {α : Type*} [CancelCommMonoidWithZero α] [GCDMonoid α]
    {x y x' y' : α} (ex : x = gcd x y * x') (ey : y = gcd x y * y') (h : gcd x y ≠ 0) :
    IsUnit (gcd x' y') := by
  /-
    α : Type u_2
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : GCDMonoid α
    x y x' y' : α
    ex : Eq x (HMul.hMul (GCDMonoid.gcd x y) x')
    ey : Eq y (HMul.hMul (GCDMonoid.gcd x y) y')
    h : Ne (GCDMonoid.gcd x y) 0
    ⊢ IsUnit (GCDMonoid.gcd x' y')
  -/
  rw [← associated_one_iff_isUnit]
  /-
    α : Type u_2
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : GCDMonoid α
    x y x' y' : α
    ex : Eq x (HMul.hMul (GCDMonoid.gcd x y) x')
    ey : Eq y (HMul.hMul (GCDMonoid.gcd x y) y')
    h : Ne (GCDMonoid.gcd x y) 0
    ⊢ Associated (GCDMonoid.gcd x' y') 1
  -/
  refine Associated.of_mul_left ?_ (Associated.refl <| gcd x y) h
  /-
    α : Type u_2
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : GCDMonoid α
    x y x' y' : α
    ex : Eq x (HMul.hMul (GCDMonoid.gcd x y) x')
    ey : Eq y (HMul.hMul (GCDMonoid.gcd x y) y')
    h : Ne (GCDMonoid.gcd x y) 0
    ⊢ Associated (HMul.hMul (GCDMonoid.gcd x y) (GCDMonoid.gcd x' y')) (HMul.hMul  …
  -/
  convert (gcd_mul_left' (gcd x y) x' y').symm using 1
  /-
    case h.e'_4
    α : Type u_2
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : GCDMonoid α
    x y x' y' : α
    ex : Eq x (HMul.hMul (GCDMonoid.gcd x y) x')
    ey : Eq y (HMul.hMul (GCDMonoid.gcd x y) y')
    h : Ne (GCDMonoid.gcd x y) 0
    ⊢ Eq (HMul.hMul (GCDMonoid.gcd x y) 1) (GCDMonoid.gcd (HMul.hMul (GCDMonoid.gc …
  -/
  rw [← ex, ← ey, mul_one]
  /-
    🎉 no goals
  -/


theorem extract_gcd {α : Type*} [CancelCommMonoidWithZero α] [GCDMonoid α] (x y : α) :
    ∃ x' y', x = gcd x y * x' ∧ y = gcd x y * y' ∧ IsUnit (gcd x' y') := by
  /-
    α : Type u_2
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : GCDMonoid α
    x y : α
    ⊢ Exists fun x' => Exists fun y' => And (Eq x (HMul.hMul (GCDMonoid.gcd x y) x …
  -/
  by_cases h : gcd x y = 0
    /-
      case pos
      α : Type u_2
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : GCDMonoid α
      x y : α
      h : Eq (GCDMonoid.gcd x y) 0
      ⊢ Exists fun x' => Exists fun y' => And (Eq x (HMul.hMul (GCDMonoid.gcd x y) x …
    -/
  · obtain ⟨rfl, rfl⟩ := (gcd_eq_zero_iff x y).1 h
    /-
      case pos.intro
      α : Type u_2
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : GCDMonoid α
      h : Eq (GCDMonoid.gcd 0 0) 0
      ⊢ Exists fun x' => Exists fun y' => And (Eq 0 (HMul.hMul (GCDMonoid.gcd 0 0) x …
    -/
    simp_rw [← associated_one_iff_isUnit]
    /-
      case pos.intro
      α : Type u_2
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : GCDMonoid α
      h : Eq (GCDMonoid.gcd 0 0) 0
      ⊢ Exists fun x' => Exists fun y' => And (Eq 0 (HMul.hMul (GCDMonoid.gcd 0 0) x …
    -/
    exact ⟨1, 1, by rw [h, zero_mul], by rw [h, zero_mul], gcd_one_left' 1⟩
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_2
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : GCDMonoid α
    x y : α
    h : Not (Eq (GCDMonoid.gcd x y) 0)
    ⊢ Exists fun x' => Exists fun y' => And (Eq x (HMul.hMul (GCDMonoid.gcd x y) x …
  -/
  obtain ⟨x', ex⟩ := gcd_dvd_left x y
  /-
    case neg.intro
    α : Type u_2
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : GCDMonoid α
    x y : α
    h : Not (Eq (GCDMonoid.gcd x y) 0)
    x' : α
    ex : Eq x (HMul.hMul (GCDMonoid.gcd x y) x')
    ⊢ Exists fun x' => Exists fun y' => And (Eq x (HMul.hMul (GCDMonoid.gcd x y) x …
  -/
  obtain ⟨y', ey⟩ := gcd_dvd_right x y
  /-
    case neg.intro.intro
    α : Type u_2
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : GCDMonoid α
    x y : α
    h : Not (Eq (GCDMonoid.gcd x y) 0)
    x' : α
    ex : Eq x (HMul.hMul (GCDMonoid.gcd x y) x')
    y' : α
    ey : Eq y (HMul.hMul (GCDMonoid.gcd x y) y')
    ⊢ Exists fun x' => Exists fun y' => And (Eq x (HMul.hMul (GCDMonoid.gcd x y) x …
  -/
  exact ⟨x', y', ex, ey, isUnit_gcd_of_eq_mul_gcd ex ey h⟩
  /-
    🎉 no goals
  -/


theorem associated_gcd_left_iff [GCDMonoid α] {x y : α} : Associated x (gcd x y) ↔ x ∣ y :=
  ⟨fun hx => hx.dvd.trans (gcd_dvd_right x y),
    fun hxy => associated_of_dvd_dvd (dvd_gcd dvd_rfl hxy) (gcd_dvd_left x y)⟩


theorem associated_gcd_right_iff [GCDMonoid α] {x y : α} : Associated y (gcd x y) ↔ y ∣ x :=
  ⟨fun hx => hx.dvd.trans (gcd_dvd_left x y),
    fun hxy => associated_of_dvd_dvd (dvd_gcd hxy dvd_rfl) (gcd_dvd_right x y)⟩


theorem Irreducible.isUnit_gcd_iff [GCDMonoid α] {x y : α} (hx : Irreducible x) :
    IsUnit (gcd x y) ↔ ¬(x ∣ y) := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : GCDMonoid α
    x y : α
    hx : Irreducible x
    ⊢ Iff (IsUnit (GCDMonoid.gcd x y)) (Not (Dvd.dvd x y))
  -/
  rw [hx.isUnit_iff_not_associated_of_dvd (gcd_dvd_left x y), not_iff_not, associated_gcd_left_iff]
  /-
    🎉 no goals
  -/


theorem Irreducible.gcd_eq_one_iff [NormalizedGCDMonoid α] {x y : α} (hx : Irreducible x) :
    gcd x y = 1 ↔ ¬(x ∣ y) := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : NormalizedGCDMonoid α
    x y : α
    hx : Irreducible x
    ⊢ Iff (Eq (GCDMonoid.gcd x y) 1) (Not (Dvd.dvd x y))
  -/
  rw [← hx.isUnit_gcd_iff, ← normalize_eq_one, NormalizedGCDMonoid.normalize_gcd]
  /-
    🎉 no goals
  -/


lemma gcd_neg' [GCDMonoid α] {a b : α} : Associated (gcd a (-b)) (gcd a b) :=
  Associated.gcd .rfl (.neg_left .rfl)


lemma gcd_neg [NormalizedGCDMonoid α] {a b : α} : gcd a (-b) = gcd a b :=
  gcd_neg'.eq_of_normalized (normalize_gcd _ _) (normalize_gcd _ _)


lemma neg_gcd' [GCDMonoid α] {a b : α} : Associated (gcd (-a) b) (gcd a b) :=
  Associated.gcd (.neg_left .rfl) .rfl


lemma neg_gcd [NormalizedGCDMonoid α] {a b : α} : gcd (-a) b = gcd a b :=
  neg_gcd'.eq_of_normalized (normalize_gcd _ _) (normalize_gcd _ _)


theorem lcm_dvd_iff [GCDMonoid α] {a b c : α} : lcm a b ∣ c ↔ a ∣ c ∧ b ∣ c := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : GCDMonoid α
    a b c : α
    ⊢ Iff (Dvd.dvd (GCDMonoid.lcm a b) c) (And (Dvd.dvd a c) (Dvd.dvd b c))
  -/
  by_cases h : a = 0 ∨ b = 0
    /-
      case pos
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : GCDMonoid α
      a b c : α
      h : Or (Eq a 0) (Eq b 0)
      ⊢ Iff (Dvd.dvd (GCDMonoid.lcm a b) c) (And (Dvd.dvd a c) (Dvd.dvd b c))
    -/
  · rcases h with (rfl | rfl) <;>
      simp +contextual only [iff_def, lcm_zero_left, lcm_zero_right,
        zero_dvd_iff, dvd_zero, eq_self_iff_true, and_true, imp_true_iff]
    /-
      case neg
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : GCDMonoid α
      a b c : α
      h : Not (Or (Eq a 0) (Eq b 0))
      ⊢ Iff (Dvd.dvd (GCDMonoid.lcm a b) c) (And (Dvd.dvd a c) (Dvd.dvd b c))
    -/
  · obtain ⟨h1, h2⟩ := not_or.1 h
    /-
      case neg.intro
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : GCDMonoid α
      a b c : α
      h : Not (Or (Eq a 0) (Eq b 0))
      h1 : Not (Eq a 0)
      h2 : Not (Eq b 0)
      ⊢ Iff (Dvd.dvd (GCDMonoid.lcm a b) c) (And (Dvd.dvd a c) (Dvd.dvd b c))
    -/
    have h : gcd a b ≠ 0 := fun H => h1 ((gcd_eq_zero_iff _ _).1 H).1
    rw [← mul_dvd_mul_iff_left h, (gcd_mul_lcm a b).dvd_iff_dvd_left, ←
      (gcd_mul_right' c a b).dvd_iff_dvd_right, dvd_gcd_iff, mul_comm b c, mul_dvd_mul_iff_left h1,
      mul_dvd_mul_iff_right h2, and_comm]


theorem dvd_lcm_left [GCDMonoid α] (a b : α) : a ∣ lcm a b :=
  (lcm_dvd_iff.1 (dvd_refl (lcm a b))).1


theorem dvd_lcm_right [GCDMonoid α] (a b : α) : b ∣ lcm a b :=
  (lcm_dvd_iff.1 (dvd_refl (lcm a b))).2


theorem lcm_dvd [GCDMonoid α] {a b c : α} (hab : a ∣ b) (hcb : c ∣ b) : lcm a c ∣ b :=
  lcm_dvd_iff.2 ⟨hab, hcb⟩


@[simp]
theorem lcm_eq_zero_iff [GCDMonoid α] (a b : α) : lcm a b = 0 ↔ a = 0 ∨ b = 0 :=
  Iff.intro
    (fun h : lcm a b = 0 => by
      /-
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : GCDMonoid α
        a b : α
        h : Eq (GCDMonoid.lcm a b) 0
        ⊢ Or (Eq a 0) (Eq b 0)
      -/
      have : Associated (a * b) 0 := (gcd_mul_lcm a b).symm.trans <| by rw [h, mul_zero]
      /-
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : GCDMonoid α
        a b : α
        h : Eq (GCDMonoid.lcm a b) 0
        this : Associated (HMul.hMul a b) 0
        ⊢ Or (Eq a 0) (Eq b 0)
      -/
      rwa [← mul_eq_zero, ← associated_zero_iff_eq_zero])
      /-
        🎉 no goals
      -/
        /-
          α : Type u_1
          inst✝¹ : CancelCommMonoidWithZero α
          inst✝ : GCDMonoid α
          a b : α
          ⊢ Or (Eq a 0) (Eq b 0) → Eq (GCDMonoid.lcm a b) 0
        -/
    (by rintro (rfl | rfl) <;> [apply lcm_zero_left; apply lcm_zero_right])
        /-
          🎉 no goals
        -/


@[simp]
theorem normalize_lcm [NormalizedGCDMonoid α] (a b : α) : normalize (lcm a b) = lcm a b :=
  NormalizedGCDMonoid.normalize_lcm a b


theorem lcm_comm [NormalizedGCDMonoid α] (a b : α) : lcm a b = lcm b a :=
  dvd_antisymm_of_normalize_eq (normalize_lcm _ _) (normalize_lcm _ _)
    (lcm_dvd (dvd_lcm_right _ _) (dvd_lcm_left _ _))
    (lcm_dvd (dvd_lcm_right _ _) (dvd_lcm_left _ _))


theorem lcm_comm' [GCDMonoid α] (a b : α) : Associated (lcm a b) (lcm b a) :=
  associated_of_dvd_dvd (lcm_dvd (dvd_lcm_right _ _) (dvd_lcm_left _ _))
    (lcm_dvd (dvd_lcm_right _ _) (dvd_lcm_left _ _))


theorem lcm_assoc [NormalizedGCDMonoid α] (m n k : α) : lcm (lcm m n) k = lcm m (lcm n k) :=
  dvd_antisymm_of_normalize_eq (normalize_lcm _ _) (normalize_lcm _ _)
    (lcm_dvd (lcm_dvd (dvd_lcm_left _ _) ((dvd_lcm_left _ _).trans (dvd_lcm_right _ _)))
      ((dvd_lcm_right _ _).trans (dvd_lcm_right _ _)))
    (lcm_dvd ((dvd_lcm_left _ _).trans (dvd_lcm_left _ _))
      (lcm_dvd ((dvd_lcm_right _ _).trans (dvd_lcm_left _ _)) (dvd_lcm_right _ _)))


theorem lcm_assoc' [GCDMonoid α] (m n k : α) : Associated (lcm (lcm m n) k) (lcm m (lcm n k)) :=
  associated_of_dvd_dvd
    (lcm_dvd (lcm_dvd (dvd_lcm_left _ _) ((dvd_lcm_left _ _).trans (dvd_lcm_right _ _)))
      ((dvd_lcm_right _ _).trans (dvd_lcm_right _ _)))
    (lcm_dvd ((dvd_lcm_left _ _).trans (dvd_lcm_left _ _))
      (lcm_dvd ((dvd_lcm_right _ _).trans (dvd_lcm_left _ _)) (dvd_lcm_right _ _)))


instance [NormalizedGCDMonoid α] : Std.Commutative (α := α) lcm where
  comm := lcm_comm


instance [NormalizedGCDMonoid α] : Std.Associative (α := α) lcm where
  assoc := lcm_assoc


theorem lcm_eq_normalize [NormalizedGCDMonoid α] {a b c : α} (habc : lcm a b ∣ c)
    (hcab : c ∣ lcm a b) : lcm a b = normalize c :=
  normalize_lcm a b ▸ normalize_eq_normalize habc hcab


theorem lcm_dvd_lcm [GCDMonoid α] {a b c d : α} (hab : a ∣ b) (hcd : c ∣ d) : lcm a c ∣ lcm b d :=
  lcm_dvd (hab.trans (dvd_lcm_left _ _)) (hcd.trans (dvd_lcm_right _ _))


protected theorem Associated.lcm [GCDMonoid α]
    {a₁ a₂ b₁ b₂ : α} (ha : Associated a₁ a₂) (hb : Associated b₁ b₂) :
    Associated (lcm a₁ b₁) (lcm a₂ b₂) :=
  associated_of_dvd_dvd (lcm_dvd_lcm ha.dvd hb.dvd) (lcm_dvd_lcm ha.dvd' hb.dvd')


@[simp]
theorem lcm_units_coe_left [NormalizedGCDMonoid α] (u : αˣ) (a : α) : lcm (↑u) a = normalize a :=
  lcm_eq_normalize (lcm_dvd Units.coe_dvd dvd_rfl) (dvd_lcm_right _ _)


@[simp]
theorem lcm_units_coe_right [NormalizedGCDMonoid α] (a : α) (u : αˣ) : lcm a ↑u = normalize a :=
  (lcm_comm a u).trans <| lcm_units_coe_left _ _


@[simp]
theorem lcm_one_left [NormalizedGCDMonoid α] (a : α) : lcm 1 a = normalize a :=
  lcm_units_coe_left 1 a


@[simp]
theorem lcm_one_right [NormalizedGCDMonoid α] (a : α) : lcm a 1 = normalize a :=
  lcm_units_coe_right a 1


@[simp]
theorem lcm_same [NormalizedGCDMonoid α] (a : α) : lcm a a = normalize a :=
  lcm_eq_normalize (lcm_dvd dvd_rfl dvd_rfl) (dvd_lcm_left _ _)


@[simp]
theorem lcm_eq_one_iff [NormalizedGCDMonoid α] (a b : α) : lcm a b = 1 ↔ a ∣ 1 ∧ b ∣ 1 :=
  Iff.intro (fun eq => eq ▸ ⟨dvd_lcm_left _ _, dvd_lcm_right _ _⟩) fun ⟨⟨c, hc⟩, ⟨d, hd⟩⟩ =>
    show lcm (Units.mkOfMulEqOne a c hc.symm : α) (Units.mkOfMulEqOne b d hd.symm) = 1 by
      /-
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : NormalizedGCDMonoid α
        a b : α
        x✝ : And (Dvd.dvd a 1) (Dvd.dvd b 1)
        c : α
        hc : Eq 1 (HMul.hMul a c)
        d : α
        hd : Eq 1 (HMul.hMul b d)
        ⊢ Eq (GCDMonoid.lcm ↑(Units.mkOfMulEqOne a c ⋯) ↑(Units.mkOfMulEqOne b d ⋯)) 1
      -/
      rw [lcm_units_coe_left, normalize_coe_units]
      /-
        🎉 no goals
      -/


@[simp]
theorem lcm_mul_left [NormalizedGCDMonoid α] (a b c : α) :
    lcm (a * b) (a * c) = normalize a * lcm b c :=
                /-
                  α : Type u_1
                  inst✝¹ : CancelCommMonoidWithZero α
                  inst✝ : NormalizedGCDMonoid α
                  a b c : α
                  ⊢ Eq a 0 → Eq (GCDMonoid.lcm (HMul.hMul a b) (HMul.hMul a c)) (HMul.hMul (norm …
                -/
  (by_cases (by rintro rfl; simp only [zero_mul, lcm_zero_left, normalize_zero]))
                            /-
                              🎉 no goals
                            -/
    fun ha : a ≠ 0 =>
                                                              /-
                                                                α : Type u_1
                                                                inst✝¹ : CancelCommMonoidWithZero α
                                                                inst✝ : NormalizedGCDMonoid α
                                                                a b c : α
                                                                ha : Ne a 0
                                                                this : Eq (GCDMonoid.lcm (HMul.hMul a b) (HMul.hMul a c)) (normalize (HMul.hMu …
                                                                ⊢ Eq (GCDMonoid.lcm (HMul.hMul a b) (HMul.hMul a c)) (HMul.hMul (normalize a)  …
                                                              -/
    suffices lcm (a * b) (a * c) = normalize (a * lcm b c) by simpa
                                                              /-
                                                                🎉 no goals
                                                              -/
    have : a ∣ lcm (a * b) (a * c) := (dvd_mul_right _ _).trans (dvd_lcm_left _ _)
    let ⟨_, eq⟩ := this
    lcm_eq_normalize
      (lcm_dvd (mul_dvd_mul_left a (dvd_lcm_left _ _)) (mul_dvd_mul_left a (dvd_lcm_right _ _)))
      (eq.symm ▸
        (mul_dvd_mul_left a <|
          lcm_dvd ((mul_dvd_mul_iff_left ha).1 <| eq ▸ dvd_lcm_left _ _)
            ((mul_dvd_mul_iff_left ha).1 <| eq ▸ dvd_lcm_right _ _)))


@[simp]
theorem lcm_mul_right [NormalizedGCDMonoid α] (a b c : α) :
                                                      /-
                                                        α : Type u_1
                                                        inst✝¹ : CancelCommMonoidWithZero α
                                                        inst✝ : NormalizedGCDMonoid α
                                                        a b c : α
                                                        ⊢ Eq (GCDMonoid.lcm (HMul.hMul b a) (HMul.hMul c a)) (HMul.hMul (GCDMonoid.lcm …
                                                      -/
    lcm (b * a) (c * a) = lcm b c * normalize a := by simp only [mul_comm, lcm_mul_left]
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem lcm_eq_left_iff [NormalizedGCDMonoid α] (a b : α) (h : normalize a = a) :
    lcm a b = a ↔ b ∣ a :=
  (Iff.intro fun eq => eq ▸ dvd_lcm_right _ _) fun hab =>
    dvd_antisymm_of_normalize_eq (normalize_lcm _ _) h (lcm_dvd (dvd_refl a) hab) (dvd_lcm_left _ _)


theorem lcm_eq_right_iff [NormalizedGCDMonoid α] (a b : α) (h : normalize b = b) :
                              /-
                                α : Type u_1
                                inst✝¹ : CancelCommMonoidWithZero α
                                inst✝ : NormalizedGCDMonoid α
                                a b : α
                                h : Eq (normalize b) b
                                ⊢ Iff (Eq (GCDMonoid.lcm a b) b) (Dvd.dvd a b)
                              -/
    lcm a b = b ↔ a ∣ b := by simpa only [lcm_comm b a] using lcm_eq_left_iff b a h
                              /-
                                🎉 no goals
                              -/


theorem lcm_dvd_lcm_mul_left [GCDMonoid α] (m n k : α) : lcm m n ∣ lcm (k * m) n :=
  lcm_dvd_lcm (dvd_mul_left _ _) dvd_rfl


theorem lcm_dvd_lcm_mul_right [GCDMonoid α] (m n k : α) : lcm m n ∣ lcm (m * k) n :=
  lcm_dvd_lcm (dvd_mul_right _ _) dvd_rfl


theorem lcm_dvd_lcm_mul_left_right [GCDMonoid α] (m n k : α) : lcm m n ∣ lcm m (k * n) :=
  lcm_dvd_lcm dvd_rfl (dvd_mul_left _ _)


theorem lcm_dvd_lcm_mul_right_right [GCDMonoid α] (m n k : α) : lcm m n ∣ lcm m (n * k) :=
  lcm_dvd_lcm dvd_rfl (dvd_mul_right _ _)


theorem lcm_eq_of_associated_left [NormalizedGCDMonoid α] {m n : α} (h : Associated m n) (k : α) :
    lcm m k = lcm n k :=
  dvd_antisymm_of_normalize_eq (normalize_lcm _ _) (normalize_lcm _ _) (lcm_dvd_lcm h.dvd dvd_rfl)
    (lcm_dvd_lcm h.symm.dvd dvd_rfl)


theorem lcm_eq_of_associated_right [NormalizedGCDMonoid α] {m n : α} (h : Associated m n) (k : α) :
    lcm k m = lcm k n :=
  dvd_antisymm_of_normalize_eq (normalize_lcm _ _) (normalize_lcm _ _) (lcm_dvd_lcm dvd_rfl h.dvd)
    (lcm_dvd_lcm dvd_rfl h.symm.dvd)


@[deprecated (since := "2024-02-12")] alias GCDMonoid.prime_of_irreducible := Irreducible.prime

@[deprecated (since := "2024-02-12")] alias GCDMonoid.irreducible_iff_prime := irreducible_iff_prime


instance (priority := 100) NormalizationMonoid.ofUniqueUnits : NormalizationMonoid α where
  normUnit _ := 1
  normUnit_zero := rfl
  normUnit_mul _ _ := (mul_one 1).symm
  normUnit_coe_units _ := Subsingleton.elim _ _


instance uniqueNormalizationMonoidOfUniqueUnits : Unique (NormalizationMonoid α) where
  default := .ofUniqueUnits
                                 /-
                                   α : Type u_1
                                   inst✝¹ : CancelCommMonoidWithZero α
                                   inst✝ : Subsingleton (Units α)
                                   x✝ : NormalizationMonoid α
                                   u : α → Units α
                                   normUnit_zero✝ : Eq (u 0) 1
                                   normUnit_mul✝ : ∀ {a b : α}, Ne a 0 → Ne b 0 → Eq (u (HMul.hMul a b)) (HMul.hM …
                                   normUnit_coe_units✝ : ∀ (u_1 : Units α), Eq (u ↑u_1) (Inv.inv u_1)
                                   ⊢ Eq { normUnit := u, normUnit_zero := normUnit_zero✝, normUnit_mul := normUni …
                                 -/
  uniq := fun ⟨u, _, _, _⟩ => by congr; simp [eq_iff_true_of_subsingleton]
                                        /-
                                          🎉 no goals
                                        -/


instance subsingleton_gcdMonoid_of_unique_units : Subsingleton (GCDMonoid α) :=
  ⟨fun g₁ g₂ => by
    have hgcd : g₁.gcd = g₂.gcd := by
      ext a b
      refine associated_iff_eq.mp (associated_of_dvd_dvd ?_ ?_)
      -- Porting note: Lean4 seems to need help specifying `g₁` and `g₂`
      · exact dvd_gcd (@gcd_dvd_left _ _ g₁ _ _) (@gcd_dvd_right _ _ g₁ _ _)
      · exact @dvd_gcd _ _ g₁ _ _ _ (@gcd_dvd_left _ _ g₂ _ _) (@gcd_dvd_right _ _ g₂ _ _)
    have hlcm : g₁.lcm = g₂.lcm := by
      ext a b
      -- Porting note: Lean4 seems to need help specifying `g₁` and `g₂`
      refine associated_iff_eq.mp (associated_of_dvd_dvd ?_ ?_)
      · exact (@lcm_dvd_iff _ _ g₁ ..).mpr ⟨@dvd_lcm_left _ _ g₂ _ _, @dvd_lcm_right _ _ g₂ _ _⟩
      · exact lcm_dvd_iff.mpr ⟨@dvd_lcm_left _ _ g₁ _ _, @dvd_lcm_right _ _ g₁ _ _⟩
    /-
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : Subsingleton (Units α)
      g₁ g₂ : GCDMonoid α
      hgcd : Eq GCDMonoid.gcd GCDMonoid.gcd
      hlcm : Eq GCDMonoid.lcm GCDMonoid.lcm
      ⊢ Eq g₁ g₂
    -/
    cases g₁
    /-
      case mk
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : Subsingleton (Units α)
      g₂ : GCDMonoid α
      gcd✝ lcm✝ : α → α → α
      gcd_dvd_left✝ : ∀ (a b : α), Dvd.dvd (gcd✝ a b) a
      gcd_dvd_right✝ : ∀ (a b : α), Dvd.dvd (gcd✝ a b) b
      dvd_gcd✝ : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd✝ c b)
      gcd_mul_lcm✝ : ∀ (a b : α), Associated (HMul.hMul (gcd✝ a b) (lcm✝ a b)) (HMul …
      lcm_zero_left✝ : ∀ (a : α), Eq (lcm✝ 0 a) 0
      lcm_zero_right✝ : ∀ (a : α), Eq (lcm✝ a 0) 0
      hgcd : Eq GCDMonoid.gcd GCDMonoid.gcd
      hlcm : Eq GCDMonoid.lcm GCDMonoid.lcm
      ⊢ Eq { gcd := gcd✝, lcm := lcm✝, gcd_dvd_left := gcd_dvd_left✝, gcd_dvd_right  …
    -/
    cases g₂
    /-
      case mk.mk
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : Subsingleton (Units α)
      gcd✝¹ lcm✝¹ : α → α → α
      gcd_dvd_left✝¹ : ∀ (a b : α), Dvd.dvd (gcd✝¹ a b) a
      gcd_dvd_right✝¹ : ∀ (a b : α), Dvd.dvd (gcd✝¹ a b) b
      dvd_gcd✝¹ : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd✝¹ c b)
      gcd_mul_lcm✝¹ : ∀ (a b : α), Associated (HMul.hMul (gcd✝¹ a b) (lcm✝¹ a b)) (H …
      lcm_zero_left✝¹ : ∀ (a : α), Eq (lcm✝¹ 0 a) 0
      lcm_zero_right✝¹ : ∀ (a : α), Eq (lcm✝¹ a 0) 0
      gcd✝ lcm✝ : α → α → α
      gcd_dvd_left✝ : ∀ (a b : α), Dvd.dvd (gcd✝ a b) a
      gcd_dvd_right✝ : ∀ (a b : α), Dvd.dvd (gcd✝ a b) b
      dvd_gcd✝ : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd✝ c b)
      gcd_mul_lcm✝ : ∀ (a b : α), Associated (HMul.hMul (gcd✝ a b) (lcm✝ a b)) (HMul …
      lcm_zero_left✝ : ∀ (a : α), Eq (lcm✝ 0 a) 0
      lcm_zero_right✝ : ∀ (a : α), Eq (lcm✝ a 0) 0
      hgcd : Eq GCDMonoid.gcd GCDMonoid.gcd
      hlcm : Eq GCDMonoid.lcm GCDMonoid.lcm
      ⊢ Eq { gcd := gcd✝¹, lcm := lcm✝¹, gcd_dvd_left := gcd_dvd_left✝¹, gcd_dvd_rig …
    -/
    dsimp only at hgcd hlcm
    /-
      case mk.mk
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : Subsingleton (Units α)
      gcd✝¹ lcm✝¹ : α → α → α
      gcd_dvd_left✝¹ : ∀ (a b : α), Dvd.dvd (gcd✝¹ a b) a
      gcd_dvd_right✝¹ : ∀ (a b : α), Dvd.dvd (gcd✝¹ a b) b
      dvd_gcd✝¹ : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd✝¹ c b)
      gcd_mul_lcm✝¹ : ∀ (a b : α), Associated (HMul.hMul (gcd✝¹ a b) (lcm✝¹ a b)) (H …
      lcm_zero_left✝¹ : ∀ (a : α), Eq (lcm✝¹ 0 a) 0
      lcm_zero_right✝¹ : ∀ (a : α), Eq (lcm✝¹ a 0) 0
      gcd✝ lcm✝ : α → α → α
      gcd_dvd_left✝ : ∀ (a b : α), Dvd.dvd (gcd✝ a b) a
      gcd_dvd_right✝ : ∀ (a b : α), Dvd.dvd (gcd✝ a b) b
      dvd_gcd✝ : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd✝ c b)
      gcd_mul_lcm✝ : ∀ (a b : α), Associated (HMul.hMul (gcd✝ a b) (lcm✝ a b)) (HMul …
      lcm_zero_left✝ : ∀ (a : α), Eq (lcm✝ 0 a) 0
      lcm_zero_right✝ : ∀ (a : α), Eq (lcm✝ a 0) 0
      hgcd : Eq gcd✝¹ gcd✝
      hlcm : Eq lcm✝¹ lcm✝
      ⊢ Eq { gcd := gcd✝¹, lcm := lcm✝¹, gcd_dvd_left := gcd_dvd_left✝¹, gcd_dvd_rig …
    -/
    simp only [hgcd, hlcm]⟩
    /-
      🎉 no goals
    -/


instance subsingleton_normalizedGCDMonoid_of_unique_units : Subsingleton (NormalizedGCDMonoid α) :=
  ⟨by
    /-
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : Subsingleton (Units α)
      ⊢ ∀ (a b : NormalizedGCDMonoid α), Eq a b
    -/
    intro a b
    /-
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : Subsingleton (Units α)
      a b : NormalizedGCDMonoid α
      ⊢ Eq a b
    -/
    cases a; rename_i a_norm a_gcd _ _
    /-
      case mk
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : Subsingleton (Units α)
      b : NormalizedGCDMonoid α
      a_norm : NormalizationMonoid α
      a_gcd : GCDMonoid α
      normalize_gcd✝ : ∀ (a b : α), Eq (normalize (GCDMonoid.gcd a b)) (GCDMonoid.gc …
      normalize_lcm✝ : ∀ (a b : α), Eq (normalize (GCDMonoid.lcm a b)) (GCDMonoid.lc …
      ⊢ Eq (NormalizedGCDMonoid.mk normalize_gcd✝ normalize_lcm✝) b
    -/
    cases b; rename_i b_norm b_gcd _ _
    /-
      case mk.mk
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : Subsingleton (Units α)
      a_norm : NormalizationMonoid α
      a_gcd : GCDMonoid α
      normalize_gcd✝¹ : ∀ (a b : α), Eq (normalize (GCDMonoid.gcd a b)) (GCDMonoid.g …
      normalize_lcm✝¹ : ∀ (a b : α), Eq (normalize (GCDMonoid.lcm a b)) (GCDMonoid.l …
      b_norm : NormalizationMonoid α
      b_gcd : GCDMonoid α
      normalize_gcd✝ : ∀ (a b : α), Eq (normalize (GCDMonoid.gcd a b)) (GCDMonoid.gc …
      normalize_lcm✝ : ∀ (a b : α), Eq (normalize (GCDMonoid.lcm a b)) (GCDMonoid.lc …
      ⊢ Eq (NormalizedGCDMonoid.mk normalize_gcd✝¹ normalize_lcm✝¹) (NormalizedGCDMo …
    -/
    have := Subsingleton.elim a_gcd b_gcd
    /-
      case mk.mk
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : Subsingleton (Units α)
      a_norm : NormalizationMonoid α
      a_gcd : GCDMonoid α
      normalize_gcd✝¹ : ∀ (a b : α), Eq (normalize (GCDMonoid.gcd a b)) (GCDMonoid.g …
      normalize_lcm✝¹ : ∀ (a b : α), Eq (normalize (GCDMonoid.lcm a b)) (GCDMonoid.l …
      b_norm : NormalizationMonoid α
      b_gcd : GCDMonoid α
      normalize_gcd✝ : ∀ (a b : α), Eq (normalize (GCDMonoid.gcd a b)) (GCDMonoid.gc …
      normalize_lcm✝ : ∀ (a b : α), Eq (normalize (GCDMonoid.lcm a b)) (GCDMonoid.lc …
      this : Eq a_gcd b_gcd
      ⊢ Eq (NormalizedGCDMonoid.mk normalize_gcd✝¹ normalize_lcm✝¹) (NormalizedGCDMo …
    -/
    subst this
    /-
      case mk.mk
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : Subsingleton (Units α)
      a_norm : NormalizationMonoid α
      a_gcd : GCDMonoid α
      normalize_gcd✝¹ : ∀ (a b : α), Eq (normalize (GCDMonoid.gcd a b)) (GCDMonoid.g …
      normalize_lcm✝¹ : ∀ (a b : α), Eq (normalize (GCDMonoid.lcm a b)) (GCDMonoid.l …
      b_norm : NormalizationMonoid α
      normalize_gcd✝ : ∀ (a b : α), Eq (normalize (GCDMonoid.gcd a b)) (GCDMonoid.gc …
      normalize_lcm✝ : ∀ (a b : α), Eq (normalize (GCDMonoid.lcm a b)) (GCDMonoid.lc …
      ⊢ Eq (NormalizedGCDMonoid.mk normalize_gcd✝¹ normalize_lcm✝¹) (NormalizedGCDMo …
    -/
    have := Subsingleton.elim a_norm b_norm
    /-
      case mk.mk
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : Subsingleton (Units α)
      a_norm : NormalizationMonoid α
      a_gcd : GCDMonoid α
      normalize_gcd✝¹ : ∀ (a b : α), Eq (normalize (GCDMonoid.gcd a b)) (GCDMonoid.g …
      normalize_lcm✝¹ : ∀ (a b : α), Eq (normalize (GCDMonoid.lcm a b)) (GCDMonoid.l …
      b_norm : NormalizationMonoid α
      normalize_gcd✝ : ∀ (a b : α), Eq (normalize (GCDMonoid.gcd a b)) (GCDMonoid.gc …
      normalize_lcm✝ : ∀ (a b : α), Eq (normalize (GCDMonoid.lcm a b)) (GCDMonoid.lc …
      this : Eq a_norm b_norm
      ⊢ Eq (NormalizedGCDMonoid.mk normalize_gcd✝¹ normalize_lcm✝¹) (NormalizedGCDMo …
    -/
    subst this
    /-
      case mk.mk
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : Subsingleton (Units α)
      a_norm : NormalizationMonoid α
      a_gcd : GCDMonoid α
      normalize_gcd✝¹ : ∀ (a b : α), Eq (normalize (GCDMonoid.gcd a b)) (GCDMonoid.g …
      normalize_lcm✝¹ : ∀ (a b : α), Eq (normalize (GCDMonoid.lcm a b)) (GCDMonoid.l …
      normalize_gcd✝ : ∀ (a b : α), Eq (normalize (GCDMonoid.gcd a b)) (GCDMonoid.gc …
      normalize_lcm✝ : ∀ (a b : α), Eq (normalize (GCDMonoid.lcm a b)) (GCDMonoid.lc …
      ⊢ Eq (NormalizedGCDMonoid.mk normalize_gcd✝¹ normalize_lcm✝¹) (NormalizedGCDMo …
    -/
    rfl⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem normUnit_eq_one (x : α) : normUnit x = 1 :=
  rfl


@[simp]
theorem normalize_eq (x : α) : normalize x = x :=
  mul_one x


/-- If a monoid's only unit is `1`, then it is isomorphic to its associates. -/
@[simps]
def associatesEquivOfUniqueUnits : Associates α ≃* α where
  toFun := Associates.out
  invFun := Associates.mk
  left_inv := Associates.mk_out
  right_inv _ := (Associates.out_mk _).trans <| normalize_eq _
  map_mul' := Associates.out_mul


theorem gcd_eq_of_dvd_sub_right {a b c : α} (h : a ∣ b - c) : gcd a b = gcd a c := by
  /-
    α : Type u_1
    inst✝² : CommRing α
    inst✝¹ : IsDomain α
    inst✝ : NormalizedGCDMonoid α
    a b c : α
    h : Dvd.dvd a (HSub.hSub b c)
    ⊢ Eq (GCDMonoid.gcd a b) (GCDMonoid.gcd a c)
  -/
  apply dvd_antisymm_of_normalize_eq (normalize_gcd _ _) (normalize_gcd _ _) <;>
    /-
      case hab
      α : Type u_1
      inst✝² : CommRing α
      inst✝¹ : IsDomain α
      inst✝ : NormalizedGCDMonoid α
      a b c : α
      h : Dvd.dvd a (HSub.hSub b c)
      ⊢ Dvd.dvd (GCDMonoid.gcd a b) (GCDMonoid.gcd a c)
    -/
    rw [dvd_gcd_iff] <;>
    /-
      case hab
      α : Type u_1
      inst✝² : CommRing α
      inst✝¹ : IsDomain α
      inst✝ : NormalizedGCDMonoid α
      a b c : α
      h : Dvd.dvd a (HSub.hSub b c)
      ⊢ And (Dvd.dvd (GCDMonoid.gcd a b) a) (Dvd.dvd (GCDMonoid.gcd a b) c)
    -/
    refine ⟨gcd_dvd_left _ _, ?_⟩
    /-
      case hab
      α : Type u_1
      inst✝² : CommRing α
      inst✝¹ : IsDomain α
      inst✝ : NormalizedGCDMonoid α
      a b c : α
      h : Dvd.dvd a (HSub.hSub b c)
      ⊢ Dvd.dvd (GCDMonoid.gcd a b) c
    -/
  · rcases h with ⟨d, hd⟩
    /-
      case hab.intro
      α : Type u_1
      inst✝² : CommRing α
      inst✝¹ : IsDomain α
      inst✝ : NormalizedGCDMonoid α
      a b c d : α
      hd : Eq (HSub.hSub b c) (HMul.hMul a d)
      ⊢ Dvd.dvd (GCDMonoid.gcd a b) c
    -/
    rcases gcd_dvd_right a b with ⟨e, he⟩
    /-
      case hab.intro.intro
      α : Type u_1
      inst✝² : CommRing α
      inst✝¹ : IsDomain α
      inst✝ : NormalizedGCDMonoid α
      a b c d : α
      hd : Eq (HSub.hSub b c) (HMul.hMul a d)
      e : α
      he : Eq b (HMul.hMul (GCDMonoid.gcd a b) e)
      ⊢ Dvd.dvd (GCDMonoid.gcd a b) c
    -/
    rcases gcd_dvd_left a b with ⟨f, hf⟩
    /-
      case hab.intro.intro.intro
      α : Type u_1
      inst✝² : CommRing α
      inst✝¹ : IsDomain α
      inst✝ : NormalizedGCDMonoid α
      a b c d : α
      hd : Eq (HSub.hSub b c) (HMul.hMul a d)
      e : α
      he : Eq b (HMul.hMul (GCDMonoid.gcd a b) e)
      f : α
      hf : Eq a (HMul.hMul (GCDMonoid.gcd a b) f)
      ⊢ Dvd.dvd (GCDMonoid.gcd a b) c
    -/
    use e - f * d
    /-
      case h
      α : Type u_1
      inst✝² : CommRing α
      inst✝¹ : IsDomain α
      inst✝ : NormalizedGCDMonoid α
      a b c d : α
      hd : Eq (HSub.hSub b c) (HMul.hMul a d)
      e : α
      he : Eq b (HMul.hMul (GCDMonoid.gcd a b) e)
      f : α
      hf : Eq a (HMul.hMul (GCDMonoid.gcd a b) f)
      ⊢ Eq c (HMul.hMul (GCDMonoid.gcd a b) (HSub.hSub e (HMul.hMul f d)))
    -/
    rw [mul_sub, ← he, ← mul_assoc, ← hf, ← hd, sub_sub_cancel]
    /-
      🎉 no goals
    -/
    /-
      case hba
      α : Type u_1
      inst✝² : CommRing α
      inst✝¹ : IsDomain α
      inst✝ : NormalizedGCDMonoid α
      a b c : α
      h : Dvd.dvd a (HSub.hSub b c)
      ⊢ Dvd.dvd (GCDMonoid.gcd a c) b
    -/
  · rcases h with ⟨d, hd⟩
    /-
      case hba.intro
      α : Type u_1
      inst✝² : CommRing α
      inst✝¹ : IsDomain α
      inst✝ : NormalizedGCDMonoid α
      a b c d : α
      hd : Eq (HSub.hSub b c) (HMul.hMul a d)
      ⊢ Dvd.dvd (GCDMonoid.gcd a c) b
    -/
    rcases gcd_dvd_right a c with ⟨e, he⟩
    /-
      case hba.intro.intro
      α : Type u_1
      inst✝² : CommRing α
      inst✝¹ : IsDomain α
      inst✝ : NormalizedGCDMonoid α
      a b c d : α
      hd : Eq (HSub.hSub b c) (HMul.hMul a d)
      e : α
      he : Eq c (HMul.hMul (GCDMonoid.gcd a c) e)
      ⊢ Dvd.dvd (GCDMonoid.gcd a c) b
    -/
    rcases gcd_dvd_left a c with ⟨f, hf⟩
    /-
      case hba.intro.intro.intro
      α : Type u_1
      inst✝² : CommRing α
      inst✝¹ : IsDomain α
      inst✝ : NormalizedGCDMonoid α
      a b c d : α
      hd : Eq (HSub.hSub b c) (HMul.hMul a d)
      e : α
      he : Eq c (HMul.hMul (GCDMonoid.gcd a c) e)
      f : α
      hf : Eq a (HMul.hMul (GCDMonoid.gcd a c) f)
      ⊢ Dvd.dvd (GCDMonoid.gcd a c) b
    -/
    use e + f * d
    /-
      case h
      α : Type u_1
      inst✝² : CommRing α
      inst✝¹ : IsDomain α
      inst✝ : NormalizedGCDMonoid α
      a b c d : α
      hd : Eq (HSub.hSub b c) (HMul.hMul a d)
      e : α
      he : Eq c (HMul.hMul (GCDMonoid.gcd a c) e)
      f : α
      hf : Eq a (HMul.hMul (GCDMonoid.gcd a c) f)
      ⊢ Eq b (HMul.hMul (GCDMonoid.gcd a c) (HAdd.hAdd e (HMul.hMul f d)))
    -/
    rw [mul_add, ← he, ← mul_assoc, ← hf, ← hd, ← add_sub_assoc, add_comm c b, add_sub_cancel_right]
    /-
      🎉 no goals
    -/


theorem gcd_eq_of_dvd_sub_left {a b c : α} (h : a ∣ b - c) : gcd b a = gcd c a := by
  /-
    α : Type u_1
    inst✝² : CommRing α
    inst✝¹ : IsDomain α
    inst✝ : NormalizedGCDMonoid α
    a b c : α
    h : Dvd.dvd a (HSub.hSub b c)
    ⊢ Eq (GCDMonoid.gcd b a) (GCDMonoid.gcd c a)
  -/
  rw [gcd_comm _ a, gcd_comm _ a, gcd_eq_of_dvd_sub_right h]
  /-
    🎉 no goals
  -/


private theorem map_mk_unit_aux [DecidableEq α] {f : Associates α →* α}
    (hinv : Function.RightInverse f Associates.mk) (a : α) :
    a * ↑(Classical.choose (associated_map_mk hinv a)) = f (Associates.mk a) :=
  Classical.choose_spec (associated_map_mk hinv a)


/-- Define `NormalizationMonoid` on a structure from a `MonoidHom` inverse to `Associates.mk`. -/
def normalizationMonoidOfMonoidHomRightInverse [DecidableEq α] (f : Associates α →* α)
    (hinv : Function.RightInverse f Associates.mk) :
    NormalizationMonoid α where
  normUnit a :=
    if a = 0 then 1
    else Classical.choose (Associates.mk_eq_mk_iff_associated.1 (hinv (Associates.mk a)).symm)
  normUnit_zero := if_pos rfl
  normUnit_mul {a b} ha hb := by
    /-
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : DecidableEq α
      f : MonoidHom (Associates α) α
      hinv : Function.RightInverse (⇑f) Associates.mk
      a b : α
      ha : Ne a 0
      hb : Ne b 0
      ⊢ Eq ((fun a => ite (Eq a 0) 1 (Classical.choose ⋯)) (HMul.hMul a b)) (HMul.hM …
    -/
    simp_rw [if_neg (mul_ne_zero ha hb), if_neg ha, if_neg hb, Units.ext_iff, Units.val_mul]
    suffices a * b * ↑(Classical.choose (associated_map_mk hinv (a * b))) =
        a * ↑(Classical.choose (associated_map_mk hinv a)) *
        (b * ↑(Classical.choose (associated_map_mk hinv b))) by
      apply mul_left_cancel₀ (mul_ne_zero ha hb) _
      -- Porting note: original `simpa` fails with `unexpected bound variable #1`
      -- simpa only [mul_assoc, mul_comm, mul_left_comm] using this
      rw [this, mul_assoc, ← mul_assoc _ b, mul_comm _ b, ← mul_assoc, ← mul_assoc,
        mul_assoc (a * b)]
    rw [map_mk_unit_aux hinv a, map_mk_unit_aux hinv (a * b), map_mk_unit_aux hinv b, ←
      MonoidHom.map_mul, Associates.mk_mul_mk]
  normUnit_coe_units u := by
    /-
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : DecidableEq α
      f : MonoidHom (Associates α) α
      hinv : Function.RightInverse (⇑f) Associates.mk
      u : Units α
      ⊢ Eq ((fun a => ite (Eq a 0) 1 (Classical.choose ⋯)) ↑u) (Inv.inv u)
    -/
    nontriviality α
    /-
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : DecidableEq α
      f : MonoidHom (Associates α) α
      hinv : Function.RightInverse (⇑f) Associates.mk
      u : Units α
      a✝ : Nontrivial α
      ⊢ Eq ((fun a => ite (Eq a 0) 1 (Classical.choose ⋯)) ↑u) (Inv.inv u)
    -/
    simp_rw [if_neg (Units.ne_zero u), Units.ext_iff]
    /-
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : DecidableEq α
      f : MonoidHom (Associates α) α
      hinv : Function.RightInverse (⇑f) Associates.mk
      u : Units α
      a✝ : Nontrivial α
      ⊢ Eq ↑(Classical.choose ⋯) ↑(Inv.inv u)
    -/
    apply mul_left_cancel₀ (Units.ne_zero u)
    rw [Units.mul_inv, map_mk_unit_aux hinv u,
      Associates.mk_eq_mk_iff_associated.2 (associated_one_iff_isUnit.2 ⟨u, rfl⟩),
      Associates.mk_one, MonoidHom.map_one]


/-- Define `GCDMonoid` on a structure just from the `gcd` and its properties. -/
noncomputable def gcdMonoidOfGCD [DecidableEq α] (gcd : α → α → α)
    (gcd_dvd_left : ∀ a b, gcd a b ∣ a) (gcd_dvd_right : ∀ a b, gcd a b ∣ b)
    (dvd_gcd : ∀ {a b c}, a ∣ c → a ∣ b → a ∣ gcd c b) : GCDMonoid α :=
  { gcd
    gcd_dvd_left
    gcd_dvd_right
    dvd_gcd := fun {_ _ _} => dvd_gcd
    lcm := fun a b =>
      if a = 0 then 0 else Classical.choose ((gcd_dvd_left a b).trans (Dvd.intro b rfl))
    gcd_mul_lcm := fun a b => by
      -- Porting note (https://github.com/leanprover-community/mathlib4/issues/12129): additional beta reduction needed
      /-
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : DecidableEq α
        gcd : α → α → α
        gcd_dvd_left : ∀ (a b : α), Dvd.dvd (gcd a b) a
        gcd_dvd_right : ∀ (a b : α), Dvd.dvd (gcd a b) b
        dvd_gcd : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd c b)
        a b : α
        ⊢ Associated (HMul.hMul (gcd a b) ((fun a b => ite (Eq a 0) 0 (Classical.choos …
      -/
      beta_reduce
      /-
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : DecidableEq α
        gcd : α → α → α
        gcd_dvd_left : ∀ (a b : α), Dvd.dvd (gcd a b) a
        gcd_dvd_right : ∀ (a b : α), Dvd.dvd (gcd a b) b
        dvd_gcd : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd c b)
        a b : α
        ⊢ Associated (HMul.hMul (gcd a b) (ite (Eq a 0) 0 (Classical.choose ⋯))) (HMul …
      -/
      split_ifs with a0
        /-
          case pos
          α : Type u_1
          inst✝¹ : CancelCommMonoidWithZero α
          inst✝ : DecidableEq α
          gcd : α → α → α
          gcd_dvd_left : ∀ (a b : α), Dvd.dvd (gcd a b) a
          gcd_dvd_right : ∀ (a b : α), Dvd.dvd (gcd a b) b
          dvd_gcd : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd c b)
          a b : α
          a0 : Eq a 0
          ⊢ Associated (HMul.hMul (gcd a b) 0) (HMul.hMul a b)
        -/
      · rw [mul_zero, a0, zero_mul]
        /-
          🎉 no goals
        -/
        /-
          case neg
          α : Type u_1
          inst✝¹ : CancelCommMonoidWithZero α
          inst✝ : DecidableEq α
          gcd : α → α → α
          gcd_dvd_left : ∀ (a b : α), Dvd.dvd (gcd a b) a
          gcd_dvd_right : ∀ (a b : α), Dvd.dvd (gcd a b) b
          dvd_gcd : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd c b)
          a b : α
          a0 : Not (Eq a 0)
          ⊢ Associated (HMul.hMul (gcd a b) (Classical.choose ⋯)) (HMul.hMul a b)
        -/
      · rw [← Classical.choose_spec ((gcd_dvd_left a b).trans (Dvd.intro b rfl))]
        /-
          🎉 no goals
        -/
    lcm_zero_left := fun _ => if_pos rfl
    lcm_zero_right := fun a => by
      -- Porting note (https://github.com/leanprover-community/mathlib4/issues/12129): additional beta reduction needed
      /-
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : DecidableEq α
        gcd : α → α → α
        gcd_dvd_left : ∀ (a b : α), Dvd.dvd (gcd a b) a
        gcd_dvd_right : ∀ (a b : α), Dvd.dvd (gcd a b) b
        dvd_gcd : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd c b)
        a : α
        ⊢ Eq ((fun a b => ite (Eq a 0) 0 (Classical.choose ⋯)) a 0) 0
      -/
      beta_reduce
      /-
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : DecidableEq α
        gcd : α → α → α
        gcd_dvd_left : ∀ (a b : α), Dvd.dvd (gcd a b) a
        gcd_dvd_right : ∀ (a b : α), Dvd.dvd (gcd a b) b
        dvd_gcd : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd c b)
        a : α
        ⊢ Eq (ite (Eq a 0) 0 (Classical.choose ⋯)) 0
      -/
      split_ifs with a0
        /-
          case pos
          α : Type u_1
          inst✝¹ : CancelCommMonoidWithZero α
          inst✝ : DecidableEq α
          gcd : α → α → α
          gcd_dvd_left : ∀ (a b : α), Dvd.dvd (gcd a b) a
          gcd_dvd_right : ∀ (a b : α), Dvd.dvd (gcd a b) b
          dvd_gcd : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd c b)
          a : α
          a0 : Eq a 0
          ⊢ Eq 0 0
        -/
      · rfl
        /-
          🎉 no goals
        -/
      /-
        case neg
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : DecidableEq α
        gcd : α → α → α
        gcd_dvd_left : ∀ (a b : α), Dvd.dvd (gcd a b) a
        gcd_dvd_right : ∀ (a b : α), Dvd.dvd (gcd a b) b
        dvd_gcd : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd c b)
        a : α
        a0 : Not (Eq a 0)
        ⊢ Eq (Classical.choose ⋯) 0
      -/
      have h := (Classical.choose_spec ((gcd_dvd_left a 0).trans (Dvd.intro 0 rfl))).symm
      have a0' : gcd a 0 ≠ 0 := by
        contrapose! a0
        rw [← associated_zero_iff_eq_zero, ← a0]
        exact associated_of_dvd_dvd (dvd_gcd (dvd_refl a) (dvd_zero a)) (gcd_dvd_left _ _)
      /-
        case neg
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : DecidableEq α
        gcd : α → α → α
        gcd_dvd_left : ∀ (a b : α), Dvd.dvd (gcd a b) a
        gcd_dvd_right : ∀ (a b : α), Dvd.dvd (gcd a b) b
        dvd_gcd : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd c b)
        a : α
        a0 : Not (Eq a 0)
        h : Eq (HMul.hMul (gcd a 0) (Classical.choose ⋯)) (HMul.hMul a 0)
        a0' : Ne (gcd a 0) 0
        ⊢ Eq (Classical.choose ⋯) 0
      -/
      apply Or.resolve_left (mul_eq_zero.1 _) a0'
      /-
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : DecidableEq α
        gcd : α → α → α
        gcd_dvd_left : ∀ (a b : α), Dvd.dvd (gcd a b) a
        gcd_dvd_right : ∀ (a b : α), Dvd.dvd (gcd a b) b
        dvd_gcd : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd c b)
        a : α
        a0 : Not (Eq a 0)
        h : Eq (HMul.hMul (gcd a 0) (Classical.choose ⋯)) (HMul.hMul a 0)
        a0' : Ne (gcd a 0) 0
        ⊢ Eq (HMul.hMul (gcd a 0) (Classical.choose ⋯)) 0
      -/
      rw [h, mul_zero] }
      /-
        🎉 no goals
      -/


/-- Define `NormalizedGCDMonoid` on a structure just from the `gcd` and its properties. -/
noncomputable def normalizedGCDMonoidOfGCD [NormalizationMonoid α] [DecidableEq α] (gcd : α → α → α)
    (gcd_dvd_left : ∀ a b, gcd a b ∣ a) (gcd_dvd_right : ∀ a b, gcd a b ∣ b)
    (dvd_gcd : ∀ {a b c}, a ∣ c → a ∣ b → a ∣ gcd c b)
    (normalize_gcd : ∀ a b, normalize (gcd a b) = gcd a b) : NormalizedGCDMonoid α :=
  { (inferInstance : NormalizationMonoid α) with
    gcd
    gcd_dvd_left
    gcd_dvd_right
    dvd_gcd := fun {_ _ _} => dvd_gcd
    normalize_gcd
    lcm := fun a b =>
      if a = 0 then 0
      else Classical.choose (dvd_normalize_iff.2 ((gcd_dvd_left a b).trans (Dvd.intro b rfl)))
    normalize_lcm := fun a b => by
      /-
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : NormalizationMonoid α
        inst✝ : DecidableEq α
        gcd : α → α → α
        gcd_dvd_left : ∀ (a b : α), Dvd.dvd (gcd a b) a
        gcd_dvd_right : ∀ (a b : α), Dvd.dvd (gcd a b) b
        dvd_gcd : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd c b)
        normalize_gcd : ∀ (a b : α), Eq (normalize (gcd a b)) (gcd a b)
        a b : α
        ⊢ Eq (normalize (GCDMonoid.lcm a b)) (GCDMonoid.lcm a b)
      -/
      dsimp [normalize]
      /-
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : NormalizationMonoid α
        inst✝ : DecidableEq α
        gcd : α → α → α
        gcd_dvd_left : ∀ (a b : α), Dvd.dvd (gcd a b) a
        gcd_dvd_right : ∀ (a b : α), Dvd.dvd (gcd a b) b
        dvd_gcd : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd c b)
        normalize_gcd : ∀ (a b : α), Eq (normalize (gcd a b)) (gcd a b)
        a b : α
        ⊢ Eq (HMul.hMul (ite (Eq a 0) 0 (Classical.choose ⋯)) ↑(NormalizationMonoid.no …
      -/
      split_ifs with a0
        /-
          case pos
          α : Type u_1
          inst✝² : CancelCommMonoidWithZero α
          inst✝¹ : NormalizationMonoid α
          inst✝ : DecidableEq α
          gcd : α → α → α
          gcd_dvd_left : ∀ (a b : α), Dvd.dvd (gcd a b) a
          gcd_dvd_right : ∀ (a b : α), Dvd.dvd (gcd a b) b
          dvd_gcd : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd c b)
          normalize_gcd : ∀ (a b : α), Eq (normalize (gcd a b)) (gcd a b)
          a b : α
          a0 : Eq a 0
          ⊢ Eq (HMul.hMul 0 ↑(NormalizationMonoid.normUnit 0)) 0
        -/
      · exact @normalize_zero α _ _
        /-
          🎉 no goals
        -/
      · have := (Classical.choose_spec
          (dvd_normalize_iff.2 ((gcd_dvd_left a b).trans (Dvd.intro b rfl)))).symm
        /-
          case neg
          α : Type u_1
          inst✝² : CancelCommMonoidWithZero α
          inst✝¹ : NormalizationMonoid α
          inst✝ : DecidableEq α
          gcd : α → α → α
          gcd_dvd_left : ∀ (a b : α), Dvd.dvd (gcd a b) a
          gcd_dvd_right : ∀ (a b : α), Dvd.dvd (gcd a b) b
          dvd_gcd : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd c b)
          normalize_gcd : ∀ (a b : α), Eq (normalize (gcd a b)) (gcd a b)
          a b : α
          a0 : Not (Eq a 0)
          this : Eq (HMul.hMul (gcd a b) (Classical.choose ⋯)) (normalize (HMul.hMul a b))
          ⊢ Eq (HMul.hMul (Classical.choose ⋯) ↑(NormalizationMonoid.normUnit (Classical …
        -/
        set l := Classical.choose (dvd_normalize_iff.2 ((gcd_dvd_left a b).trans (Dvd.intro b rfl)))
        /-
          case neg
          α : Type u_1
          inst✝² : CancelCommMonoidWithZero α
          inst✝¹ : NormalizationMonoid α
          inst✝ : DecidableEq α
          gcd : α → α → α
          gcd_dvd_left : ∀ (a b : α), Dvd.dvd (gcd a b) a
          gcd_dvd_right : ∀ (a b : α), Dvd.dvd (gcd a b) b
          dvd_gcd : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd c b)
          normalize_gcd : ∀ (a b : α), Eq (normalize (gcd a b)) (gcd a b)
          a b : α
          a0 : Not (Eq a 0)
          l : α := Classical.choose ⋯
          this : Eq (HMul.hMul (gcd a b) l) (normalize (HMul.hMul a b))
          ⊢ Eq (HMul.hMul l ↑(NormalizationMonoid.normUnit l)) l
        -/
        obtain rfl | hb := eq_or_ne b 0
        -- Porting note: using `simp only` causes the propositions inside `Classical.choose` to
        -- differ, so `set` is unable to produce `l = 0` inside `this`. See
        -- https://leanprover.zulipchat.com/#narrow/stream/287929-mathlib4/topic/
        -- Classical.2Echoose/near/317491179
          /-
            case neg.inl
            α : Type u_1
            inst✝² : CancelCommMonoidWithZero α
            inst✝¹ : NormalizationMonoid α
            inst✝ : DecidableEq α
            gcd : α → α → α
            gcd_dvd_left : ∀ (a b : α), Dvd.dvd (gcd a b) a
            gcd_dvd_right : ∀ (a b : α), Dvd.dvd (gcd a b) b
            dvd_gcd : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd c b)
            normalize_gcd : ∀ (a b : α), Eq (normalize (gcd a b)) (gcd a b)
            a : α
            a0 : Not (Eq a 0)
            l : α := Classical.choose ⋯
            this : Eq (HMul.hMul (gcd a 0) l) (normalize (HMul.hMul a 0))
            ⊢ Eq (HMul.hMul l ↑(NormalizationMonoid.normUnit l)) l
          -/
        · rw [mul_zero a, normalize_zero, mul_eq_zero] at this
          /-
            case neg.inl
            α : Type u_1
            inst✝² : CancelCommMonoidWithZero α
            inst✝¹ : NormalizationMonoid α
            inst✝ : DecidableEq α
            gcd : α → α → α
            gcd_dvd_left : ∀ (a b : α), Dvd.dvd (gcd a b) a
            gcd_dvd_right : ∀ (a b : α), Dvd.dvd (gcd a b) b
            dvd_gcd : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd c b)
            normalize_gcd : ∀ (a b : α), Eq (normalize (gcd a b)) (gcd a b)
            a : α
            a0 : Not (Eq a 0)
            l : α := Classical.choose ⋯
            this : Or (Eq (gcd a 0) 0) (Eq l 0)
            ⊢ Eq (HMul.hMul l ↑(NormalizationMonoid.normUnit l)) l
          -/
          obtain ha | hl := this
            /-
              case neg.inl.inl
              α : Type u_1
              inst✝² : CancelCommMonoidWithZero α
              inst✝¹ : NormalizationMonoid α
              inst✝ : DecidableEq α
              gcd : α → α → α
              gcd_dvd_left : ∀ (a b : α), Dvd.dvd (gcd a b) a
              gcd_dvd_right : ∀ (a b : α), Dvd.dvd (gcd a b) b
              dvd_gcd : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd c b)
              normalize_gcd : ∀ (a b : α), Eq (normalize (gcd a b)) (gcd a b)
              a : α
              a0 : Not (Eq a 0)
              l : α := Classical.choose ⋯
              ha : Eq (gcd a 0) 0
              ⊢ Eq (HMul.hMul l ↑(NormalizationMonoid.normUnit l)) l
            -/
          · apply (a0 _).elim
            /-
              α : Type u_1
              inst✝² : CancelCommMonoidWithZero α
              inst✝¹ : NormalizationMonoid α
              inst✝ : DecidableEq α
              gcd : α → α → α
              gcd_dvd_left : ∀ (a b : α), Dvd.dvd (gcd a b) a
              gcd_dvd_right : ∀ (a b : α), Dvd.dvd (gcd a b) b
              dvd_gcd : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd c b)
              normalize_gcd : ∀ (a b : α), Eq (normalize (gcd a b)) (gcd a b)
              a : α
              a0 : Not (Eq a 0)
              l : α := Classical.choose ⋯
              ha : Eq (gcd a 0) 0
              ⊢ Eq a 0
            -/
            rw [← zero_dvd_iff, ← ha]
            /-
              α : Type u_1
              inst✝² : CancelCommMonoidWithZero α
              inst✝¹ : NormalizationMonoid α
              inst✝ : DecidableEq α
              gcd : α → α → α
              gcd_dvd_left : ∀ (a b : α), Dvd.dvd (gcd a b) a
              gcd_dvd_right : ∀ (a b : α), Dvd.dvd (gcd a b) b
              dvd_gcd : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd c b)
              normalize_gcd : ∀ (a b : α), Eq (normalize (gcd a b)) (gcd a b)
              a : α
              a0 : Not (Eq a 0)
              l : α := Classical.choose ⋯
              ha : Eq (gcd a 0) 0
              ⊢ Dvd.dvd (gcd a 0) a
            -/
      /-
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : NormalizationMonoid α
        inst✝ : DecidableEq α
        gcd : α → α → α
        gcd_dvd_left : ∀ (a b : α), Dvd.dvd (gcd a b) a
        gcd_dvd_right : ∀ (a b : α), Dvd.dvd (gcd a b) b
        dvd_gcd : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd c b)
        normalize_gcd : ∀ (a b : α), Eq (normalize (gcd a b)) (gcd a b)
        a b : α
        ⊢ Associated (HMul.hMul (gcd a b) ((fun a b => ite (Eq a 0) 0 (Classical.choos …
      -/
            exact gcd_dvd_left _ _
      /-
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : NormalizationMonoid α
        inst✝ : DecidableEq α
        gcd : α → α → α
        gcd_dvd_left : ∀ (a b : α), Dvd.dvd (gcd a b) a
        gcd_dvd_right : ∀ (a b : α), Dvd.dvd (gcd a b) b
        dvd_gcd : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd c b)
        normalize_gcd : ∀ (a b : α), Eq (normalize (gcd a b)) (gcd a b)
        a b : α
        ⊢ Associated (HMul.hMul (gcd a b) (ite (Eq a 0) 0 (Classical.choose ⋯))) (HMul …
      -/
            /-
              🎉 no goals
            -/
        /-
          case pos
          α : Type u_1
          inst✝² : CancelCommMonoidWithZero α
          inst✝¹ : NormalizationMonoid α
          inst✝ : DecidableEq α
          gcd : α → α → α
          gcd_dvd_left : ∀ (a b : α), Dvd.dvd (gcd a b) a
          gcd_dvd_right : ∀ (a b : α), Dvd.dvd (gcd a b) b
          dvd_gcd : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd c b)
          normalize_gcd : ∀ (a b : α), Eq (normalize (gcd a b)) (gcd a b)
          a b : α
          a0 : Eq a 0
          ⊢ Associated (HMul.hMul (gcd a b) 0) (HMul.hMul a b)
        -/
            /-
              case neg.inl.inr
              α : Type u_1
              inst✝² : CancelCommMonoidWithZero α
              inst✝¹ : NormalizationMonoid α
              inst✝ : DecidableEq α
              gcd : α → α → α
              gcd_dvd_left : ∀ (a b : α), Dvd.dvd (gcd a b) a
              gcd_dvd_right : ∀ (a b : α), Dvd.dvd (gcd a b) b
              dvd_gcd : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd c b)
              normalize_gcd : ∀ (a b : α), Eq (normalize (gcd a b)) (gcd a b)
              a : α
              a0 : Not (Eq a 0)
              l : α := Classical.choose ⋯
              hl : Eq l 0
              ⊢ Eq (HMul.hMul l ↑(NormalizationMonoid.normUnit l)) l
            -/
        /-
          🎉 no goals
        -/
          · rw [hl, zero_mul]
            /-
              🎉 no goals
            -/
        /-
          case neg
          α : Type u_1
          inst✝² : CancelCommMonoidWithZero α
          inst✝¹ : NormalizationMonoid α
          inst✝ : DecidableEq α
          gcd : α → α → α
          gcd_dvd_left : ∀ (a b : α), Dvd.dvd (gcd a b) a
          gcd_dvd_right : ∀ (a b : α), Dvd.dvd (gcd a b) b
          dvd_gcd : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd c b)
          normalize_gcd : ∀ (a b : α), Eq (normalize (gcd a b)) (gcd a b)
          a b : α
          a0 : Not (Eq a 0)
          ⊢ Associated (normalize (HMul.hMul a b)) (HMul.hMul a b)
        -/
        have h1 : gcd a b ≠ 0 := by
        /-
          🎉 no goals
        -/
          have hab : a * b ≠ 0 := mul_ne_zero a0 hb
          contrapose! hab
          rw [← normalize_eq_zero, ← this, hab, zero_mul]
      /-
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : NormalizationMonoid α
        inst✝ : DecidableEq α
        gcd : α → α → α
        gcd_dvd_left : ∀ (a b : α), Dvd.dvd (gcd a b) a
        gcd_dvd_right : ∀ (a b : α), Dvd.dvd (gcd a b) b
        dvd_gcd : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd c b)
        normalize_gcd : ∀ (a b : α), Eq (normalize (gcd a b)) (gcd a b)
        a : α
        ⊢ Eq ((fun a b => ite (Eq a 0) 0 (Classical.choose ⋯)) a 0) 0
      -/
        /-
          case neg.inr
          α : Type u_1
          inst✝² : CancelCommMonoidWithZero α
          inst✝¹ : NormalizationMonoid α
          inst✝ : DecidableEq α
          gcd : α → α → α
          gcd_dvd_left : ∀ (a b : α), Dvd.dvd (gcd a b) a
          gcd_dvd_right : ∀ (a b : α), Dvd.dvd (gcd a b) b
          dvd_gcd : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd c b)
          normalize_gcd : ∀ (a b : α), Eq (normalize (gcd a b)) (gcd a b)
          a b : α
          a0 : Not (Eq a 0)
          l : α := Classical.choose ⋯
          this : Eq (HMul.hMul (gcd a b) l) (normalize (HMul.hMul a b))
          hb : Ne b 0
          h1 : Ne (gcd a b) 0
          ⊢ Eq (HMul.hMul l ↑(NormalizationMonoid.normUnit l)) l
        -/
      /-
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : NormalizationMonoid α
        inst✝ : DecidableEq α
        gcd : α → α → α
        gcd_dvd_left : ∀ (a b : α), Dvd.dvd (gcd a b) a
        gcd_dvd_right : ∀ (a b : α), Dvd.dvd (gcd a b) b
        dvd_gcd : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd c b)
        normalize_gcd : ∀ (a b : α), Eq (normalize (gcd a b)) (gcd a b)
        a : α
        ⊢ Eq (ite (Eq a 0) 0 (Classical.choose ⋯)) 0
      -/
        have h2 : normalize (gcd a b * l) = gcd a b * l := by rw [this, normalize_idem]
        /-
          case pos
          α : Type u_1
          inst✝² : CancelCommMonoidWithZero α
          inst✝¹ : NormalizationMonoid α
          inst✝ : DecidableEq α
          gcd : α → α → α
          gcd_dvd_left : ∀ (a b : α), Dvd.dvd (gcd a b) a
          gcd_dvd_right : ∀ (a b : α), Dvd.dvd (gcd a b) b
          dvd_gcd : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd c b)
          normalize_gcd : ∀ (a b : α), Eq (normalize (gcd a b)) (gcd a b)
          a : α
          a0 : Eq a 0
          ⊢ Eq 0 0
        -/
        /-
          case neg.inr
          α : Type u_1
          inst✝² : CancelCommMonoidWithZero α
          inst✝¹ : NormalizationMonoid α
          inst✝ : DecidableEq α
          gcd : α → α → α
          gcd_dvd_left : ∀ (a b : α), Dvd.dvd (gcd a b) a
          gcd_dvd_right : ∀ (a b : α), Dvd.dvd (gcd a b) b
          dvd_gcd : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd c b)
          normalize_gcd : ∀ (a b : α), Eq (normalize (gcd a b)) (gcd a b)
          a b : α
          a0 : Not (Eq a 0)
          l : α := Classical.choose ⋯
          this : Eq (HMul.hMul (gcd a b) l) (normalize (HMul.hMul a b))
          hb : Ne b 0
          h1 : Ne (gcd a b) 0
          h2 : Eq (normalize (HMul.hMul (gcd a b) l)) (HMul.hMul (gcd a b) l)
          ⊢ Eq (HMul.hMul l ↑(NormalizationMonoid.normUnit l)) l
        -/
        /-
          🎉 no goals
        -/
      /-
        case neg
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : NormalizationMonoid α
        inst✝ : DecidableEq α
        gcd : α → α → α
        gcd_dvd_left : ∀ (a b : α), Dvd.dvd (gcd a b) a
        gcd_dvd_right : ∀ (a b : α), Dvd.dvd (gcd a b) b
        dvd_gcd : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd c b)
        normalize_gcd : ∀ (a b : α), Eq (normalize (gcd a b)) (gcd a b)
        a : α
        a0 : Not (Eq a 0)
        ⊢ Eq (Classical.choose ⋯) 0
      -/
        rw [← normalize_gcd] at this
        /-
          case neg.inr
          α : Type u_1
          inst✝² : CancelCommMonoidWithZero α
          inst✝¹ : NormalizationMonoid α
          inst✝ : DecidableEq α
          gcd : α → α → α
          gcd_dvd_left : ∀ (a b : α), Dvd.dvd (gcd a b) a
          gcd_dvd_right : ∀ (a b : α), Dvd.dvd (gcd a b) b
          dvd_gcd : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd c b)
          normalize_gcd : ∀ (a b : α), Eq (normalize (gcd a b)) (gcd a b)
          a b : α
          a0 : Not (Eq a 0)
          l : α := Classical.choose ⋯
          this : Eq (HMul.hMul (normalize (gcd a b)) l) (normalize (HMul.hMul a b))
          hb : Ne b 0
          h1 : Ne (gcd a b) 0
          h2 : Eq (normalize (HMul.hMul (gcd a b) l)) (HMul.hMul (gcd a b) l)
          ⊢ Eq (HMul.hMul l ↑(NormalizationMonoid.normUnit l)) l
        -/
        rwa [normalize.map_mul, normalize_gcd, mul_right_inj' h1] at h2
        /-
          🎉 no goals
        -/
    gcd_mul_lcm := fun a b => by
      -- Porting note (https://github.com/leanprover-community/mathlib4/issues/12129): additional beta reduction needed
      beta_reduce
      /-
        case neg
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : NormalizationMonoid α
        inst✝ : DecidableEq α
        gcd : α → α → α
        gcd_dvd_left : ∀ (a b : α), Dvd.dvd (gcd a b) a
        gcd_dvd_right : ∀ (a b : α), Dvd.dvd (gcd a b) b
        dvd_gcd : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd c b)
        normalize_gcd : ∀ (a b : α), Eq (normalize (gcd a b)) (gcd a b)
        a : α
        a0 : Not (Eq (normalize a) 0)
        h : Eq (HMul.hMul (gcd a 0) (Classical.choose ⋯)) (normalize (HMul.hMul a 0))
        gcd0 : Eq (gcd a 0) (normalize a)
        ⊢ Eq (Classical.choose ⋯) 0
      -/
      split_ifs with a0
      /-
        case neg
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : NormalizationMonoid α
        inst✝ : DecidableEq α
        gcd : α → α → α
        gcd_dvd_left : ∀ (a b : α), Dvd.dvd (gcd a b) a
        gcd_dvd_right : ∀ (a b : α), Dvd.dvd (gcd a b) b
        dvd_gcd : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd c b)
        normalize_gcd : ∀ (a b : α), Eq (normalize (gcd a b)) (gcd a b)
        a : α
        a0 : Not (Eq (gcd a 0) 0)
        h : Eq (HMul.hMul (gcd a 0) (Classical.choose ⋯)) (normalize (HMul.hMul a 0))
        gcd0 : Eq (gcd a 0) (normalize a)
        ⊢ Eq (Classical.choose ⋯) 0
      -/
      · rw [mul_zero, a0, zero_mul]
      /-
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : NormalizationMonoid α
        inst✝ : DecidableEq α
        gcd : α → α → α
        gcd_dvd_left : ∀ (a b : α), Dvd.dvd (gcd a b) a
        gcd_dvd_right : ∀ (a b : α), Dvd.dvd (gcd a b) b
        dvd_gcd : ∀ {a b c : α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (gcd c b)
        normalize_gcd : ∀ (a b : α), Eq (normalize (gcd a b)) (gcd a b)
        a : α
        a0 : Not (Eq (gcd a 0) 0)
        h : Eq (HMul.hMul (gcd a 0) (Classical.choose ⋯)) (normalize (HMul.hMul a 0))
        gcd0 : Eq (gcd a 0) (normalize a)
        ⊢ Eq (HMul.hMul (gcd a 0) (Classical.choose ⋯)) 0
      -/
      · rw [←
      /-
        🎉 no goals
      -/
          Classical.choose_spec (dvd_normalize_iff.2 ((gcd_dvd_left a b).trans (Dvd.intro b rfl)))]
        exact normalize_associated (a * b)
    lcm_zero_left := fun _ => if_pos rfl
    lcm_zero_right := fun a => by
      -- Porting note (https://github.com/leanprover-community/mathlib4/issues/12129): additional beta reduction needed
      beta_reduce
      split_ifs with a0
      · rfl
      rw [← normalize_eq_zero] at a0
      have h :=
        (Classical.choose_spec
            (dvd_normalize_iff.2 ((gcd_dvd_left a 0).trans (Dvd.intro 0 rfl)))).symm
      have gcd0 : gcd a 0 = normalize a := by
        rw [← normalize_gcd]
        exact normalize_eq_normalize (gcd_dvd_left _ _) (dvd_gcd (dvd_refl a) (dvd_zero a))
      rw [← gcd0] at a0
      apply Or.resolve_left (mul_eq_zero.1 _) a0
      rw [h, mul_zero, normalize_zero] }


/-- Define `GCDMonoid` on a structure just from the `lcm` and its properties. -/
noncomputable def gcdMonoidOfLCM [DecidableEq α] (lcm : α → α → α)
    (dvd_lcm_left : ∀ a b, a ∣ lcm a b) (dvd_lcm_right : ∀ a b, b ∣ lcm a b)
    (lcm_dvd : ∀ {a b c}, c ∣ a → b ∣ a → lcm c b ∣ a) : GCDMonoid α :=
  let exists_gcd a b := lcm_dvd (Dvd.intro b rfl) (Dvd.intro_left a rfl)
  { lcm
    gcd := fun a b => if a = 0 then b else if b = 0 then a else Classical.choose (exists_gcd a b)
    gcd_mul_lcm := fun a b => by
      -- Porting note (https://github.com/leanprover-community/mathlib4/issues/12129): additional beta reduction needed
      /-
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (HMul.hMul a b) := fun a b => lcm_ …
        a b : α
        ⊢ Associated (HMul.hMul ((fun a b => ite (Eq a 0) b (ite (Eq b 0) a (Classical …
      -/
      beta_reduce
      /-
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (HMul.hMul a b) := fun a b => lcm_ …
        a b : α
        ⊢ Associated (HMul.hMul (ite (Eq a 0) b (ite (Eq b 0) a (Classical.choose ⋯))) …
      -/
      split_ifs with h h_1
        /-
          case pos
          α : Type u_1
          inst✝¹ : CancelCommMonoidWithZero α
          inst✝ : DecidableEq α
          lcm : α → α → α
          dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
          dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
          lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
          exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (HMul.hMul a b) := fun a b => lcm_ …
          a b : α
          h : Eq a 0
          ⊢ Associated (HMul.hMul b (lcm a b)) (HMul.hMul a b)
        -/
      · rw [h, eq_zero_of_zero_dvd (dvd_lcm_left _ _), mul_zero, zero_mul]
        /-
          🎉 no goals
        -/
        /-
          case pos
          α : Type u_1
          inst✝¹ : CancelCommMonoidWithZero α
          inst✝ : DecidableEq α
          lcm : α → α → α
          dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
          dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
          lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
          exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (HMul.hMul a b) := fun a b => lcm_ …
          a b : α
          h : Not (Eq a 0)
          h_1 : Eq b 0
          ⊢ Associated (HMul.hMul a (lcm a b)) (HMul.hMul a b)
        -/
      · rw [h_1, eq_zero_of_zero_dvd (dvd_lcm_right _ _)]
      /-
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (HMul.hMul a b) := fun a b => lcm_ …
        a b : α
        ⊢ Dvd.dvd ((fun a b => ite (Eq a 0) b (ite (Eq b 0) a (Classical.choose ⋯))) a …
      -/
        /-
          🎉 no goals
        -/
      /-
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (HMul.hMul a b) := fun a b => lcm_ …
        a b : α
        ⊢ Dvd.dvd (ite (Eq a 0) b (ite (Eq b 0) a (Classical.choose ⋯))) a
      -/
      /-
        case neg
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (HMul.hMul a b) := fun a b => lcm_ …
        a b : α
        h : Not (Eq a 0)
        h_1 : Not (Eq b 0)
        ⊢ Associated (HMul.hMul (Classical.choose ⋯) (lcm a b)) (HMul.hMul a b)
      -/
        /-
          case pos
          α : Type u_1
          inst✝¹ : CancelCommMonoidWithZero α
          inst✝ : DecidableEq α
          lcm : α → α → α
          dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
          dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
          lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
          exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (HMul.hMul a b) := fun a b => lcm_ …
          a b : α
          h : Eq a 0
          ⊢ Dvd.dvd b a
        -/
      rw [mul_comm, ← Classical.choose_spec (exists_gcd a b)]
        /-
          case pos
          α : Type u_1
          inst✝¹ : CancelCommMonoidWithZero α
          inst✝ : DecidableEq α
          lcm : α → α → α
          dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
          dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
          lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
          exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (HMul.hMul a b) := fun a b => lcm_ …
          a b : α
          h : Eq a 0
          ⊢ Dvd.dvd b 0
        -/
      /-
        🎉 no goals
      -/
        /-
          🎉 no goals
        -/
        /-
          case pos
          α : Type u_1
          inst✝¹ : CancelCommMonoidWithZero α
          inst✝ : DecidableEq α
          lcm : α → α → α
          dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
          dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
          lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
          exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (HMul.hMul a b) := fun a b => lcm_ …
          a b : α
          h : Not (Eq a 0)
          h_1 : Eq b 0
          ⊢ Dvd.dvd a a
        -/
    lcm_zero_left := fun _ => eq_zero_of_zero_dvd (dvd_lcm_left _ _)
        /-
          🎉 no goals
        -/
    lcm_zero_right := fun _ => eq_zero_of_zero_dvd (dvd_lcm_right _ _)
    gcd_dvd_left := fun a b => by
      -- Porting note (https://github.com/leanprover-community/mathlib4/issues/12129): additional beta reduction needed
      beta_reduce
      split_ifs with h h_1
      · rw [h]
        apply dvd_zero
      · exact dvd_rfl
      have h0 : lcm a b ≠ 0 := by
      /-
        case neg
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (HMul.hMul a b) := fun a b => lcm_ …
        a b : α
        h : Not (Eq a 0)
        h_1 : Not (Eq b 0)
        h0 : Ne (lcm a b) 0
        ⊢ Dvd.dvd b (lcm a b)
      -/
        intro con
      /-
        🎉 no goals
      -/
        have h := lcm_dvd (Dvd.intro b rfl) (Dvd.intro_left a rfl)
        rw [con, zero_dvd_iff, mul_eq_zero] at h
      /-
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (HMul.hMul a b) := fun a b => lcm_ …
        a b : α
        ⊢ Dvd.dvd ((fun a b => ite (Eq a 0) b (ite (Eq b 0) a (Classical.choose ⋯))) a …
      -/
        cases h
      /-
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (HMul.hMul a b) := fun a b => lcm_ …
        a b : α
        ⊢ Dvd.dvd (ite (Eq a 0) b (ite (Eq b 0) a (Classical.choose ⋯))) b
      -/
        · exact absurd ‹a = 0› h
        /-
          case pos
          α : Type u_1
          inst✝¹ : CancelCommMonoidWithZero α
          inst✝ : DecidableEq α
          lcm : α → α → α
          dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
          dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
          lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
          exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (HMul.hMul a b) := fun a b => lcm_ …
          a b : α
          h : Eq a 0
          ⊢ Dvd.dvd b b
        -/
        · exact absurd ‹b = 0› h_1
        /-
          🎉 no goals
        -/
        /-
          case pos
          α : Type u_1
          inst✝¹ : CancelCommMonoidWithZero α
          inst✝ : DecidableEq α
          lcm : α → α → α
          dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
          dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
          lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
          exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (HMul.hMul a b) := fun a b => lcm_ …
          a b : α
          h : Not (Eq a 0)
          h_1 : Eq b 0
          ⊢ Dvd.dvd a b
        -/
      rw [← mul_dvd_mul_iff_left h0, ← Classical.choose_spec (exists_gcd a b), mul_comm,
        /-
          case pos
          α : Type u_1
          inst✝¹ : CancelCommMonoidWithZero α
          inst✝ : DecidableEq α
          lcm : α → α → α
          dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
          dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
          lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
          exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (HMul.hMul a b) := fun a b => lcm_ …
          a b : α
          h : Not (Eq a 0)
          h_1 : Eq b 0
          ⊢ Dvd.dvd a 0
        -/
        mul_dvd_mul_iff_right h]
        /-
          🎉 no goals
        -/
      apply dvd_lcm_right
    gcd_dvd_right := fun a b => by
      -- Porting note (https://github.com/leanprover-community/mathlib4/issues/12129): additional beta reduction needed
      beta_reduce
      split_ifs with h h_1
      · exact dvd_rfl
      · rw [h_1]
        apply dvd_zero
      have h0 : lcm a b ≠ 0 := by
      /-
        case neg
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (HMul.hMul a b) := fun a b => lcm_ …
        a b : α
        h : Not (Eq a 0)
        h_1 : Not (Eq b 0)
        h0 : Ne (lcm a b) 0
        ⊢ Dvd.dvd a (lcm a b)
      -/
        intro con
      /-
        🎉 no goals
      -/
        have h := lcm_dvd (Dvd.intro b rfl) (Dvd.intro_left a rfl)
        rw [con, zero_dvd_iff, mul_eq_zero] at h
      /-
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (HMul.hMul a b) := fun a b => lcm_ …
        a b c : α
        ac : Dvd.dvd a c
        ab : Dvd.dvd a b
        ⊢ Dvd.dvd a ((fun a b => ite (Eq a 0) b (ite (Eq b 0) a (Classical.choose ⋯))) …
      -/
        cases h
      /-
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (HMul.hMul a b) := fun a b => lcm_ …
        a b c : α
        ac : Dvd.dvd a c
        ab : Dvd.dvd a b
        ⊢ Dvd.dvd a (ite (Eq c 0) b (ite (Eq b 0) c (Classical.choose ⋯)))
      -/
        · exact absurd ‹a = 0› h
        /-
          case pos
          α : Type u_1
          inst✝¹ : CancelCommMonoidWithZero α
          inst✝ : DecidableEq α
          lcm : α → α → α
          dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
          dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
          lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
          exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (HMul.hMul a b) := fun a b => lcm_ …
          a b c : α
          ac : Dvd.dvd a c
          ab : Dvd.dvd a b
          h : Eq c 0
          ⊢ Dvd.dvd a b
        -/
        · exact absurd ‹b = 0› h_1
        /-
          🎉 no goals
        -/
        /-
          case pos
          α : Type u_1
          inst✝¹ : CancelCommMonoidWithZero α
          inst✝ : DecidableEq α
          lcm : α → α → α
          dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
          dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
          lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
          exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (HMul.hMul a b) := fun a b => lcm_ …
          a b c : α
          ac : Dvd.dvd a c
          ab : Dvd.dvd a b
          h : Not (Eq c 0)
          h_1 : Eq b 0
          ⊢ Dvd.dvd a c
        -/
      rw [← mul_dvd_mul_iff_left h0, ← Classical.choose_spec (exists_gcd a b),
        /-
          🎉 no goals
        -/
        mul_dvd_mul_iff_right h_1]
      apply dvd_lcm_left
    dvd_gcd := fun {a b c} ac ab => by
      -- Porting note (https://github.com/leanprover-community/mathlib4/issues/12129): additional beta reduction needed
      beta_reduce
      split_ifs with h h_1
      · exact ab
      /-
        case neg
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (HMul.hMul a b) := fun a b => lcm_ …
        a b c : α
        ac : Dvd.dvd a c
        ab : Dvd.dvd a b
        h : Not (Eq c 0)
        h_1 : Not (Eq b 0)
        h0 : Ne (lcm c b) 0
        ⊢ Dvd.dvd a (Classical.choose ⋯)
      -/
      · exact ac
      /-
        case neg
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (HMul.hMul a b) := fun a b => lcm_ …
        a b c : α
        ac : Dvd.dvd a c
        ab : Dvd.dvd a b
        h : Not (Eq c 0)
        h_1 : Not (Eq b 0)
        h0 : Ne (lcm c b) 0
        ⊢ Dvd.dvd (HMul.hMul (lcm c b) a) (HMul.hMul c b)
      -/
      have h0 : lcm c b ≠ 0 := by
      /-
        case neg.intro
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (HMul.hMul a b) := fun a b => lcm_ …
        a c : α
        ac : Dvd.dvd a c
        h : Not (Eq c 0)
        d : α
        h_1 : Not (Eq (HMul.hMul a d) 0)
        h0 : Ne (lcm c (HMul.hMul a d)) 0
        ⊢ Dvd.dvd (HMul.hMul (lcm c (HMul.hMul a d)) a) (HMul.hMul c (HMul.hMul a d))
      -/
        intro con
      /-
        case neg.intro
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (HMul.hMul a b) := fun a b => lcm_ …
        a c : α
        ac : Dvd.dvd a c
        h : Not (Eq c 0)
        d : α
        h_1 : Not (Or (Eq a 0) (Eq d 0))
        h0 : Ne (lcm c (HMul.hMul a d)) 0
        ⊢ Dvd.dvd (HMul.hMul (lcm c (HMul.hMul a d)) a) (HMul.hMul c (HMul.hMul a d))
      -/
        have h := lcm_dvd (Dvd.intro b rfl) (Dvd.intro_left c rfl)
      /-
        case neg.intro
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (HMul.hMul a b) := fun a b => lcm_ …
        a c : α
        ac : Dvd.dvd a c
        h : Not (Eq c 0)
        d : α
        h0 : Ne (lcm c (HMul.hMul a d)) 0
        h_1 : And (Ne a 0) (Ne d 0)
        ⊢ Dvd.dvd (HMul.hMul (lcm c (HMul.hMul a d)) a) (HMul.hMul c (HMul.hMul a d))
      -/
        rw [con, zero_dvd_iff, mul_eq_zero] at h
      /-
        case neg.intro
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (HMul.hMul a b) := fun a b => lcm_ …
        a c : α
        ac : Dvd.dvd a c
        h : Not (Eq c 0)
        d : α
        h0 : Ne (lcm c (HMul.hMul a d)) 0
        h_1 : And (Ne a 0) (Ne d 0)
        ⊢ Dvd.dvd (lcm c (HMul.hMul d a)) (HMul.hMul c d)
      -/
        cases h
      /-
        case neg.intro
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (HMul.hMul a b) := fun a b => lcm_ …
        a c : α
        ac : Dvd.dvd a c
        h : Not (Eq c 0)
        d : α
        h0 : Ne (lcm c (HMul.hMul a d)) 0
        h_1 : And (Ne a 0) (Ne d 0)
        ⊢ Dvd.dvd (HMul.hMul d a) (HMul.hMul c d)
      -/
        · exact absurd ‹c = 0› h
      /-
        case neg.intro
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (HMul.hMul a b) := fun a b => lcm_ …
        a c : α
        ac : Dvd.dvd a c
        h : Not (Eq c 0)
        d : α
        h0 : Ne (lcm c (HMul.hMul a d)) 0
        h_1 : And (Ne a 0) (Ne d 0)
        ⊢ Dvd.dvd a c
      -/
        · exact absurd ‹b = 0› h_1
      /-
        🎉 no goals
      -/
      rw [← mul_dvd_mul_iff_left h0, ← Classical.choose_spec (exists_gcd c b)]
      rcases ab with ⟨d, rfl⟩
      rw [mul_eq_zero] at ‹a * d ≠ 0›
      push_neg at h_1
      rw [mul_comm a, ← mul_assoc, mul_dvd_mul_iff_right h_1.1]
      apply lcm_dvd (Dvd.intro d rfl)
      rw [mul_comm, mul_dvd_mul_iff_right h_1.2]
      apply ac }


/-- Define `NormalizedGCDMonoid` on a structure just from the `lcm` and its properties. -/
noncomputable def normalizedGCDMonoidOfLCM [NormalizationMonoid α] [DecidableEq α] (lcm : α → α → α)
    (dvd_lcm_left : ∀ a b, a ∣ lcm a b) (dvd_lcm_right : ∀ a b, b ∣ lcm a b)
    (lcm_dvd : ∀ {a b c}, c ∣ a → b ∣ a → lcm c b ∣ a)
    (normalize_lcm : ∀ a b, normalize (lcm a b) = lcm a b) : NormalizedGCDMonoid α :=
  let exists_gcd a b := dvd_normalize_iff.2 (lcm_dvd (Dvd.intro b rfl) (Dvd.intro_left a rfl))
  { (inferInstance : NormalizationMonoid α) with
    lcm
    gcd := fun a b =>
      if a = 0 then normalize b
      else if b = 0 then normalize a else Classical.choose (exists_gcd a b)
    gcd_mul_lcm := fun a b => by
      /-
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : NormalizationMonoid α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        normalize_lcm : ∀ (a b : α), Eq (normalize (lcm a b)) (lcm a b)
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (normalize (HMul.hMul a b)) := fun …
        a b : α
        ⊢ Associated (HMul.hMul ((fun a b => ite (Eq a 0) (normalize b) (ite (Eq b 0)  …
      -/
      beta_reduce
      /-
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : NormalizationMonoid α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        normalize_lcm : ∀ (a b : α), Eq (normalize (lcm a b)) (lcm a b)
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (normalize (HMul.hMul a b)) := fun …
        a b : α
        ⊢ Associated (HMul.hMul (ite (Eq a 0) (normalize b) (ite (Eq b 0) (normalize a …
      -/
      split_ifs with h h_1
        /-
          case pos
          α : Type u_1
          inst✝² : CancelCommMonoidWithZero α
          inst✝¹ : NormalizationMonoid α
          inst✝ : DecidableEq α
          lcm : α → α → α
          dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
          dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
          lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
          normalize_lcm : ∀ (a b : α), Eq (normalize (lcm a b)) (lcm a b)
          exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (normalize (HMul.hMul a b)) := fun …
          a b : α
          h : Eq a 0
          ⊢ Associated (HMul.hMul (normalize b) (lcm a b)) (HMul.hMul a b)
        -/
      · rw [h, eq_zero_of_zero_dvd (dvd_lcm_left _ _), mul_zero, zero_mul]
        /-
          🎉 no goals
        -/
        /-
          case pos
          α : Type u_1
          inst✝² : CancelCommMonoidWithZero α
          inst✝¹ : NormalizationMonoid α
          inst✝ : DecidableEq α
          lcm : α → α → α
          dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
          dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
          lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
          normalize_lcm : ∀ (a b : α), Eq (normalize (lcm a b)) (lcm a b)
          exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (normalize (HMul.hMul a b)) := fun …
          a b : α
          h : Not (Eq a 0)
          h_1 : Eq b 0
          ⊢ Associated (HMul.hMul (normalize a) (lcm a b)) (HMul.hMul a b)
        -/
      · rw [h_1, eq_zero_of_zero_dvd (dvd_lcm_right _ _), mul_zero, mul_zero]
        /-
          🎉 no goals
        -/
      /-
        case neg
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : NormalizationMonoid α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        normalize_lcm : ∀ (a b : α), Eq (normalize (lcm a b)) (lcm a b)
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (normalize (HMul.hMul a b)) := fun …
        a b : α
        h : Not (Eq a 0)
        h_1 : Not (Eq b 0)
        ⊢ Associated (HMul.hMul (Classical.choose ⋯) (lcm a b)) (HMul.hMul a b)
      -/
      rw [mul_comm, ← Classical.choose_spec (exists_gcd a b)]
      /-
        case neg
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : NormalizationMonoid α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        normalize_lcm : ∀ (a b : α), Eq (normalize (lcm a b)) (lcm a b)
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (normalize (HMul.hMul a b)) := fun …
        a b : α
        h : Not (Eq a 0)
        h_1 : Not (Eq b 0)
        ⊢ Associated (normalize (HMul.hMul a b)) (HMul.hMul a b)
      -/
      exact normalize_associated (a * b)
      /-
        🎉 no goals
      -/
    normalize_lcm
    normalize_gcd := fun a b => by
      /-
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : NormalizationMonoid α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        normalize_lcm : ∀ (a b : α), Eq (normalize (lcm a b)) (lcm a b)
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (normalize (HMul.hMul a b)) := fun …
        a b : α
        ⊢ Eq (normalize (GCDMonoid.gcd a b)) (GCDMonoid.gcd a b)
      -/
      dsimp [normalize]
      /-
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : NormalizationMonoid α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        normalize_lcm : ∀ (a b : α), Eq (normalize (lcm a b)) (lcm a b)
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (normalize (HMul.hMul a b)) := fun …
        a b : α
        ⊢ Eq (HMul.hMul (ite (Eq a 0) (HMul.hMul b ↑(NormalizationMonoid.normUnit b))  …
      -/
      split_ifs with h h_1
        /-
          case pos
          α : Type u_1
          inst✝² : CancelCommMonoidWithZero α
          inst✝¹ : NormalizationMonoid α
          inst✝ : DecidableEq α
          lcm : α → α → α
          dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
          dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
          lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
          normalize_lcm : ∀ (a b : α), Eq (normalize (lcm a b)) (lcm a b)
          exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (normalize (HMul.hMul a b)) := fun …
          a b : α
          h : Eq a 0
          ⊢ Eq (HMul.hMul (HMul.hMul b ↑(NormalizationMonoid.normUnit b)) ↑(Normalizatio …
        -/
      · apply normalize_idem
        /-
          🎉 no goals
        -/
        /-
          case pos
          α : Type u_1
          inst✝² : CancelCommMonoidWithZero α
          inst✝¹ : NormalizationMonoid α
          inst✝ : DecidableEq α
          lcm : α → α → α
          dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
          dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
          lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
          normalize_lcm : ∀ (a b : α), Eq (normalize (lcm a b)) (lcm a b)
          exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (normalize (HMul.hMul a b)) := fun …
          a b : α
          h : Not (Eq a 0)
          h_1 : Eq b 0
          ⊢ Eq (HMul.hMul (HMul.hMul a ↑(NormalizationMonoid.normUnit a)) ↑(Normalizatio …
        -/
      · apply normalize_idem
        /-
          🎉 no goals
        -/
      have h0 : lcm a b ≠ 0 := by
      /-
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : NormalizationMonoid α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        normalize_lcm : ∀ (a b : α), Eq (normalize (lcm a b)) (lcm a b)
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (normalize (HMul.hMul a b)) := fun …
        a b : α
        ⊢ Dvd.dvd ((fun a b => ite (Eq a 0) (normalize b) (ite (Eq b 0) (normalize a)  …
      -/
        intro con
      /-
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : NormalizationMonoid α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        normalize_lcm : ∀ (a b : α), Eq (normalize (lcm a b)) (lcm a b)
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (normalize (HMul.hMul a b)) := fun …
        a b : α
        ⊢ Dvd.dvd (ite (Eq a 0) (normalize b) (ite (Eq b 0) (normalize a) (Classical.c …
      -/
        have h := lcm_dvd (Dvd.intro b rfl) (Dvd.intro_left a rfl)
        /-
          case pos
          α : Type u_1
          inst✝² : CancelCommMonoidWithZero α
          inst✝¹ : NormalizationMonoid α
          inst✝ : DecidableEq α
          lcm : α → α → α
          dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
          dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
          lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
          normalize_lcm : ∀ (a b : α), Eq (normalize (lcm a b)) (lcm a b)
          exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (normalize (HMul.hMul a b)) := fun …
          a b : α
          h : Eq a 0
          ⊢ Dvd.dvd (normalize b) a
        -/
        rw [con, zero_dvd_iff, mul_eq_zero] at h
        /-
          case pos
          α : Type u_1
          inst✝² : CancelCommMonoidWithZero α
          inst✝¹ : NormalizationMonoid α
          inst✝ : DecidableEq α
          lcm : α → α → α
          dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
          dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
          lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
          normalize_lcm : ∀ (a b : α), Eq (normalize (lcm a b)) (lcm a b)
          exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (normalize (HMul.hMul a b)) := fun …
          a b : α
          h : Eq a 0
          ⊢ Dvd.dvd (normalize b) 0
        -/
        cases h
        /-
          🎉 no goals
        -/
        /-
          case pos
          α : Type u_1
          inst✝² : CancelCommMonoidWithZero α
          inst✝¹ : NormalizationMonoid α
          inst✝ : DecidableEq α
          lcm : α → α → α
          dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
          dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
          lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
          normalize_lcm : ∀ (a b : α), Eq (normalize (lcm a b)) (lcm a b)
          exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (normalize (HMul.hMul a b)) := fun …
          a b : α
          h : Not (Eq a 0)
          h_1 : Eq b 0
          ⊢ Dvd.dvd (normalize a) a
        -/
        · exact absurd ‹a = 0› h
        /-
          🎉 no goals
        -/
        · exact absurd ‹b = 0› h_1
      /-
        case neg
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : NormalizationMonoid α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        normalize_lcm : ∀ (a b : α), Eq (normalize (lcm a b)) (lcm a b)
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (normalize (HMul.hMul a b)) := fun …
        a b : α
        h : Not (Eq a 0)
        h_1 : Not (Eq b 0)
        h0 : Ne (lcm a b) 0
        ⊢ Eq (HMul.hMul (Classical.choose ⋯) ↑(NormalizationMonoid.normUnit (Classical …
      -/
      apply mul_left_cancel₀ h0
      /-
        case neg
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : NormalizationMonoid α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        normalize_lcm : ∀ (a b : α), Eq (normalize (lcm a b)) (lcm a b)
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (normalize (HMul.hMul a b)) := fun …
        a b : α
        h : Not (Eq a 0)
        h_1 : Not (Eq b 0)
        h0 : Ne (lcm a b) 0
        ⊢ Eq (HMul.hMul (lcm a b) (HMul.hMul (Classical.choose ⋯) ↑(NormalizationMonoi …
      -/
      refine _root_.trans ?_ (Classical.choose_spec (exists_gcd a b))
      conv_lhs =>
        congr
        rw [← normalize_lcm a b]
      /-
        case neg
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : NormalizationMonoid α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        normalize_lcm : ∀ (a b : α), Eq (normalize (lcm a b)) (lcm a b)
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (normalize (HMul.hMul a b)) := fun …
        a b : α
        h : Not (Eq a 0)
        h_1 : Not (Eq b 0)
        h0 : Ne (lcm a b) 0
        ⊢ Eq (HMul.hMul (normalize (lcm a b)) (HMul.hMul (Classical.choose ⋯) ↑(Normal …
      -/
      /-
        case neg
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : NormalizationMonoid α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        normalize_lcm : ∀ (a b : α), Eq (normalize (lcm a b)) (lcm a b)
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (normalize (HMul.hMul a b)) := fun …
        a b : α
        h : Not (Eq a 0)
        h_1 : Not (Eq b 0)
        h0 : Ne (lcm a b) 0
        ⊢ Dvd.dvd b (lcm a b)
      -/
      erw [← normalize.map_mul, ← Classical.choose_spec (exists_gcd a b), normalize_idem]
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
      /-
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : NormalizationMonoid α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        normalize_lcm : ∀ (a b : α), Eq (normalize (lcm a b)) (lcm a b)
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (normalize (HMul.hMul a b)) := fun …
        a b : α
        ⊢ Dvd.dvd ((fun a b => ite (Eq a 0) (normalize b) (ite (Eq b 0) (normalize a)  …
      -/
    lcm_zero_left := fun _ => eq_zero_of_zero_dvd (dvd_lcm_left _ _)
      /-
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : NormalizationMonoid α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        normalize_lcm : ∀ (a b : α), Eq (normalize (lcm a b)) (lcm a b)
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (normalize (HMul.hMul a b)) := fun …
        a b : α
        ⊢ Dvd.dvd (ite (Eq a 0) (normalize b) (ite (Eq b 0) (normalize a) (Classical.c …
      -/
    lcm_zero_right := fun _ => eq_zero_of_zero_dvd (dvd_lcm_right _ _)
        /-
          case pos
          α : Type u_1
          inst✝² : CancelCommMonoidWithZero α
          inst✝¹ : NormalizationMonoid α
          inst✝ : DecidableEq α
          lcm : α → α → α
          dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
          dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
          lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
          normalize_lcm : ∀ (a b : α), Eq (normalize (lcm a b)) (lcm a b)
          exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (normalize (HMul.hMul a b)) := fun …
          a b : α
          h : Eq a 0
          ⊢ Dvd.dvd (normalize b) b
        -/
    gcd_dvd_left := fun a b => by
        /-
          🎉 no goals
        -/
        /-
          case pos
          α : Type u_1
          inst✝² : CancelCommMonoidWithZero α
          inst✝¹ : NormalizationMonoid α
          inst✝ : DecidableEq α
          lcm : α → α → α
          dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
          dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
          lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
          normalize_lcm : ∀ (a b : α), Eq (normalize (lcm a b)) (lcm a b)
          exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (normalize (HMul.hMul a b)) := fun …
          a b : α
          h : Not (Eq a 0)
          h_1 : Eq b 0
          ⊢ Dvd.dvd (normalize a) b
        -/
      beta_reduce
        /-
          case pos
          α : Type u_1
          inst✝² : CancelCommMonoidWithZero α
          inst✝¹ : NormalizationMonoid α
          inst✝ : DecidableEq α
          lcm : α → α → α
          dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
          dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
          lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
          normalize_lcm : ∀ (a b : α), Eq (normalize (lcm a b)) (lcm a b)
          exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (normalize (HMul.hMul a b)) := fun …
          a b : α
          h : Not (Eq a 0)
          h_1 : Eq b 0
          ⊢ Dvd.dvd (normalize a) 0
        -/
      split_ifs with h h_1
        /-
          🎉 no goals
        -/
      · rw [h]
        apply dvd_zero
      · exact (normalize_associated _).dvd
      have h0 : lcm a b ≠ 0 := by
        intro con
        have h := lcm_dvd (Dvd.intro b rfl) (Dvd.intro_left a rfl)
        rw [con, zero_dvd_iff, mul_eq_zero] at h
        cases h
        · exact absurd ‹a = 0› h
      /-
        case neg
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : NormalizationMonoid α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        normalize_lcm : ∀ (a b : α), Eq (normalize (lcm a b)) (lcm a b)
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (normalize (HMul.hMul a b)) := fun …
        a b : α
        h : Not (Eq a 0)
        h_1 : Not (Eq b 0)
        h0 : Ne (lcm a b) 0
        ⊢ Dvd.dvd a (lcm a b)
      -/
        · exact absurd ‹b = 0› h_1
      /-
        🎉 no goals
      -/
      rw [← mul_dvd_mul_iff_left h0, ← Classical.choose_spec (exists_gcd a b), normalize_dvd_iff,
      /-
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : NormalizationMonoid α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        normalize_lcm : ∀ (a b : α), Eq (normalize (lcm a b)) (lcm a b)
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (normalize (HMul.hMul a b)) := fun …
        a b c : α
        ac : Dvd.dvd a c
        ab : Dvd.dvd a b
        ⊢ Dvd.dvd a ((fun a b => ite (Eq a 0) (normalize b) (ite (Eq b 0) (normalize a …
      -/
        mul_comm, mul_dvd_mul_iff_right h]
      /-
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : NormalizationMonoid α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        normalize_lcm : ∀ (a b : α), Eq (normalize (lcm a b)) (lcm a b)
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (normalize (HMul.hMul a b)) := fun …
        a b c : α
        ac : Dvd.dvd a c
        ab : Dvd.dvd a b
        ⊢ Dvd.dvd a (ite (Eq c 0) (normalize b) (ite (Eq b 0) (normalize c) (Classical …
      -/
      apply dvd_lcm_right
        /-
          case pos
          α : Type u_1
          inst✝² : CancelCommMonoidWithZero α
          inst✝¹ : NormalizationMonoid α
          inst✝ : DecidableEq α
          lcm : α → α → α
          dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
          dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
          lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
          normalize_lcm : ∀ (a b : α), Eq (normalize (lcm a b)) (lcm a b)
          exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (normalize (HMul.hMul a b)) := fun …
          a b c : α
          ac : Dvd.dvd a c
          ab : Dvd.dvd a b
          h : Eq c 0
          ⊢ Dvd.dvd a (normalize b)
        -/
    gcd_dvd_right := fun a b => by
        /-
          🎉 no goals
        -/
        /-
          case pos
          α : Type u_1
          inst✝² : CancelCommMonoidWithZero α
          inst✝¹ : NormalizationMonoid α
          inst✝ : DecidableEq α
          lcm : α → α → α
          dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
          dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
          lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
          normalize_lcm : ∀ (a b : α), Eq (normalize (lcm a b)) (lcm a b)
          exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (normalize (HMul.hMul a b)) := fun …
          a b c : α
          ac : Dvd.dvd a c
          ab : Dvd.dvd a b
          h : Not (Eq c 0)
          h_1 : Eq b 0
          ⊢ Dvd.dvd a (normalize c)
        -/
      beta_reduce
        /-
          🎉 no goals
        -/
      split_ifs with h h_1
      · exact (normalize_associated _).dvd
      · rw [h_1]
        apply dvd_zero
      have h0 : lcm a b ≠ 0 := by
        intro con
        have h := lcm_dvd (Dvd.intro b rfl) (Dvd.intro_left a rfl)
        rw [con, zero_dvd_iff, mul_eq_zero] at h
        cases h
        · exact absurd ‹a = 0› h
        · exact absurd ‹b = 0› h_1
      /-
        case neg
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : NormalizationMonoid α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        normalize_lcm : ∀ (a b : α), Eq (normalize (lcm a b)) (lcm a b)
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (normalize (HMul.hMul a b)) := fun …
        a b c : α
        ac : Dvd.dvd a c
        ab : Dvd.dvd a b
        h : Not (Eq c 0)
        h_1 : Not (Eq b 0)
        h0 : Ne (lcm c b) 0
        ⊢ Dvd.dvd (HMul.hMul (lcm c b) a) (HMul.hMul c b)
      -/
      rw [← mul_dvd_mul_iff_left h0, ← Classical.choose_spec (exists_gcd a b), normalize_dvd_iff,
      /-
        case neg.intro
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : NormalizationMonoid α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        normalize_lcm : ∀ (a b : α), Eq (normalize (lcm a b)) (lcm a b)
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (normalize (HMul.hMul a b)) := fun …
        a c : α
        ac : Dvd.dvd a c
        h : Not (Eq c 0)
        d : α
        h_1 : Not (Eq (HMul.hMul a d) 0)
        h0 : Ne (lcm c (HMul.hMul a d)) 0
        ⊢ Dvd.dvd (HMul.hMul (lcm c (HMul.hMul a d)) a) (HMul.hMul c (HMul.hMul a d))
      -/
        mul_dvd_mul_iff_right h_1]
      /-
        case neg.intro
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : NormalizationMonoid α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        normalize_lcm : ∀ (a b : α), Eq (normalize (lcm a b)) (lcm a b)
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (normalize (HMul.hMul a b)) := fun …
        a c : α
        ac : Dvd.dvd a c
        h : Not (Eq c 0)
        d : α
        h_1 : Not (Or (Eq a 0) (Eq d 0))
        h0 : Ne (lcm c (HMul.hMul a d)) 0
        ⊢ Dvd.dvd (HMul.hMul (lcm c (HMul.hMul a d)) a) (HMul.hMul c (HMul.hMul a d))
      -/
      apply dvd_lcm_left
      /-
        case neg.intro
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : NormalizationMonoid α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        normalize_lcm : ∀ (a b : α), Eq (normalize (lcm a b)) (lcm a b)
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (normalize (HMul.hMul a b)) := fun …
        a c : α
        ac : Dvd.dvd a c
        h : Not (Eq c 0)
        d : α
        h0 : Ne (lcm c (HMul.hMul a d)) 0
        h_1 : And (Ne a 0) (Ne d 0)
        ⊢ Dvd.dvd (HMul.hMul (lcm c (HMul.hMul a d)) a) (HMul.hMul c (HMul.hMul a d))
      -/
    dvd_gcd := fun {a b c} ac ab => by
      /-
        case neg.intro
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : NormalizationMonoid α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        normalize_lcm : ∀ (a b : α), Eq (normalize (lcm a b)) (lcm a b)
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (normalize (HMul.hMul a b)) := fun …
        a c : α
        ac : Dvd.dvd a c
        h : Not (Eq c 0)
        d : α
        h0 : Ne (lcm c (HMul.hMul a d)) 0
        h_1 : And (Ne a 0) (Ne d 0)
        ⊢ Dvd.dvd (lcm c (HMul.hMul d a)) (HMul.hMul c d)
      -/
      beta_reduce
      /-
        case neg.intro
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : NormalizationMonoid α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        normalize_lcm : ∀ (a b : α), Eq (normalize (lcm a b)) (lcm a b)
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (normalize (HMul.hMul a b)) := fun …
        a c : α
        ac : Dvd.dvd a c
        h : Not (Eq c 0)
        d : α
        h0 : Ne (lcm c (HMul.hMul a d)) 0
        h_1 : And (Ne a 0) (Ne d 0)
        ⊢ Dvd.dvd (HMul.hMul d a) (HMul.hMul c d)
      -/
      split_ifs with h h_1
      /-
        case neg.intro
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : NormalizationMonoid α
        inst✝ : DecidableEq α
        lcm : α → α → α
        dvd_lcm_left : ∀ (a b : α), Dvd.dvd a (lcm a b)
        dvd_lcm_right : ∀ (a b : α), Dvd.dvd b (lcm a b)
        lcm_dvd : ∀ {a b c : α}, Dvd.dvd c a → Dvd.dvd b a → Dvd.dvd (lcm c b) a
        normalize_lcm : ∀ (a b : α), Eq (normalize (lcm a b)) (lcm a b)
        exists_gcd : ∀ (a b : α), Dvd.dvd (lcm a b) (normalize (HMul.hMul a b)) := fun …
        a c : α
        ac : Dvd.dvd a c
        h : Not (Eq c 0)
        d : α
        h0 : Ne (lcm c (HMul.hMul a d)) 0
        h_1 : And (Ne a 0) (Ne d 0)
        ⊢ Dvd.dvd a c
      -/
      · apply dvd_normalize_iff.2 ab
      /-
        🎉 no goals
      -/
      · apply dvd_normalize_iff.2 ac
      have h0 : lcm c b ≠ 0 := by
        intro con
        have h := lcm_dvd (Dvd.intro b rfl) (Dvd.intro_left c rfl)
        rw [con, zero_dvd_iff, mul_eq_zero] at h
        cases h
        · exact absurd ‹c = 0› h
        · exact absurd ‹b = 0› h_1
      rw [← mul_dvd_mul_iff_left h0, ←
      Classical.choose_spec
        (dvd_normalize_iff.2 (lcm_dvd (Dvd.intro b rfl) (Dvd.intro_left c rfl))),
      dvd_normalize_iff]
      rcases ab with ⟨d, rfl⟩
      rw [mul_eq_zero] at h_1
      push_neg at h_1
      rw [mul_comm a, ← mul_assoc, mul_dvd_mul_iff_right h_1.1]
      apply lcm_dvd (Dvd.intro d rfl)
      rw [mul_comm, mul_dvd_mul_iff_right h_1.2]
      apply ac }


/-- Define a `GCDMonoid` structure on a monoid just from the existence of a `gcd`. -/
noncomputable def gcdMonoidOfExistsGCD [DecidableEq α]
    (h : ∀ a b : α, ∃ c : α, ∀ d : α, d ∣ a ∧ d ∣ b ↔ d ∣ c) : GCDMonoid α :=
  gcdMonoidOfGCD (fun a b => Classical.choose (h a b))
    (fun a b => ((Classical.choose_spec (h a b) (Classical.choose (h a b))).2 dvd_rfl).1)
    (fun a b => ((Classical.choose_spec (h a b) (Classical.choose (h a b))).2 dvd_rfl).2)
    fun {a b c} ac ab => (Classical.choose_spec (h c b) a).1 ⟨ac, ab⟩


/-- Define a `NormalizedGCDMonoid` structure on a monoid just from the existence of a `gcd`. -/
noncomputable def normalizedGCDMonoidOfExistsGCD [NormalizationMonoid α] [DecidableEq α]
    (h : ∀ a b : α, ∃ c : α, ∀ d : α, d ∣ a ∧ d ∣ b ↔ d ∣ c) : NormalizedGCDMonoid α :=
  normalizedGCDMonoidOfGCD (fun a b => normalize (Classical.choose (h a b)))
    (fun a b =>
      normalize_dvd_iff.2 ((Classical.choose_spec (h a b) (Classical.choose (h a b))).2 dvd_rfl).1)
    (fun a b =>
      normalize_dvd_iff.2 ((Classical.choose_spec (h a b) (Classical.choose (h a b))).2 dvd_rfl).2)
    (fun {a b c} ac ab => dvd_normalize_iff.2 ((Classical.choose_spec (h c b) a).1 ⟨ac, ab⟩))
    fun _ _ => normalize_idem _


/-- Define a `GCDMonoid` structure on a monoid just from the existence of an `lcm`. -/
noncomputable def gcdMonoidOfExistsLCM [DecidableEq α]
    (h : ∀ a b : α, ∃ c : α, ∀ d : α, a ∣ d ∧ b ∣ d ↔ c ∣ d) : GCDMonoid α :=
  gcdMonoidOfLCM (fun a b => Classical.choose (h a b))
    (fun a b => ((Classical.choose_spec (h a b) (Classical.choose (h a b))).2 dvd_rfl).1)
    (fun a b => ((Classical.choose_spec (h a b) (Classical.choose (h a b))).2 dvd_rfl).2)
    fun {a b c} ac ab => (Classical.choose_spec (h c b) a).1 ⟨ac, ab⟩


/-- Define a `NormalizedGCDMonoid` structure on a monoid just from the existence of an `lcm`. -/
noncomputable def normalizedGCDMonoidOfExistsLCM [NormalizationMonoid α] [DecidableEq α]
    (h : ∀ a b : α, ∃ c : α, ∀ d : α, a ∣ d ∧ b ∣ d ↔ c ∣ d) : NormalizedGCDMonoid α :=
  normalizedGCDMonoidOfLCM (fun a b => normalize (Classical.choose (h a b)))
    (fun a b =>
      dvd_normalize_iff.2 ((Classical.choose_spec (h a b) (Classical.choose (h a b))).2 dvd_rfl).1)
    (fun a b =>
      dvd_normalize_iff.2 ((Classical.choose_spec (h a b) (Classical.choose (h a b))).2 dvd_rfl).2)
    (fun {a b c} ac ab => normalize_dvd_iff.2 ((Classical.choose_spec (h c b) a).1 ⟨ac, ab⟩))
    fun _ _ => normalize_idem _


instance (priority := 100) : NormalizedGCDMonoid G₀ where
  normUnit x := if h : x = 0 then 1 else (Units.mk0 x h)⁻¹
  normUnit_zero := dif_pos rfl
  normUnit_mul := fun {x y} x0 y0 => Units.eq_iff.1 (by
    -- Porting note (https://github.com/leanprover-community/mathlib4/issues/12129): additional beta reduction needed
    -- Porting note: `simp` reaches maximum heartbeat
    -- by Units.eq_iff.mp (by simp only [x0, y0, mul_comm])
    /-
      α : Type u_1
      G₀ : Type u_2
      inst✝¹ : CommGroupWithZero G₀
      inst✝ : DecidableEq G₀
      x y : G₀
      x0 : Ne x 0
      y0 : Ne y 0
      ⊢ Eq ↑((fun x => dite (Eq x 0) (fun h => 1) fun h => Inv.inv (Units.mk0 x h))  …
    -/
    beta_reduce
    /-
      α : Type u_1
      G₀ : Type u_2
      inst✝¹ : CommGroupWithZero G₀
      inst✝ : DecidableEq G₀
      x y : G₀
      x0 : Ne x 0
      y0 : Ne y 0
      ⊢ Eq ↑(dite (Eq (HMul.hMul x y) 0) (fun h => 1) fun h => Inv.inv (Units.mk0 (H …
    -/
    split_ifs with h
      /-
        case pos
        α : Type u_1
        G₀ : Type u_2
        inst✝¹ : CommGroupWithZero G₀
        inst✝ : DecidableEq G₀
        x y : G₀
        x0 : Ne x 0
        y0 : Ne y 0
        h : Eq (HMul.hMul x y) 0
        ⊢ Eq ↑1 ↑(HMul.hMul (Inv.inv (Units.mk0 x x0)) (Inv.inv (Units.mk0 y y0)))
      -/
    · rw [mul_eq_zero] at h
      /-
        case pos
        α : Type u_1
        G₀ : Type u_2
        inst✝¹ : CommGroupWithZero G₀
        inst✝ : DecidableEq G₀
        x y : G₀
        x0 : Ne x 0
        y0 : Ne y 0
        h : Or (Eq x 0) (Eq y 0)
        ⊢ Eq ↑1 ↑(HMul.hMul (Inv.inv (Units.mk0 x x0)) (Inv.inv (Units.mk0 y y0)))
      -/
      cases h
        /-
          case pos.inl
          α : Type u_1
          G₀ : Type u_2
          inst✝¹ : CommGroupWithZero G₀
          inst✝ : DecidableEq G₀
          x y : G₀
          x0 : Ne x 0
          y0 : Ne y 0
          h✝ : Eq x 0
          ⊢ Eq ↑1 ↑(HMul.hMul (Inv.inv (Units.mk0 x x0)) (Inv.inv (Units.mk0 y y0)))
        -/
      · exact absurd ‹x = 0› x0
        /-
          🎉 no goals
        -/
        /-
          case pos.inr
          α : Type u_1
          G₀ : Type u_2
          inst✝¹ : CommGroupWithZero G₀
          inst✝ : DecidableEq G₀
          x y : G₀
          x0 : Ne x 0
          y0 : Ne y 0
          h✝ : Eq y 0
          ⊢ Eq ↑1 ↑(HMul.hMul (Inv.inv (Units.mk0 x x0)) (Inv.inv (Units.mk0 y y0)))
        -/
      · exact absurd ‹y = 0› y0
        /-
          🎉 no goals
        -/
      /-
        case neg
        α : Type u_1
        G₀ : Type u_2
        inst✝¹ : CommGroupWithZero G₀
        inst✝ : DecidableEq G₀
        x y : G₀
        x0 : Ne x 0
        y0 : Ne y 0
        h : Not (Eq (HMul.hMul x y) 0)
        ⊢ Eq ↑(Inv.inv (Units.mk0 (HMul.hMul x y) h)) ↑(HMul.hMul (Inv.inv (Units.mk0  …
      -/
    · rw [Units.mk0_mul, mul_inv_rev, mul_comm] )
      /-
        🎉 no goals
      -/
  normUnit_coe_units u := by
    -- Porting note (https://github.com/leanprover-community/mathlib4/issues/12129): additional beta reduction needed
    /-
      α : Type u_1
      G₀ : Type u_2
      inst✝¹ : CommGroupWithZero G₀
      inst✝ : DecidableEq G₀
      u : Units G₀
      ⊢ Eq ((fun x => dite (Eq x 0) (fun h => 1) fun h => Inv.inv (Units.mk0 x h)) ↑ …
    -/
    beta_reduce
    /-
      α : Type u_1
      G₀ : Type u_2
      inst✝¹ : CommGroupWithZero G₀
      inst✝ : DecidableEq G₀
      u : Units G₀
      ⊢ Eq (dite (Eq (↑u) 0) (fun h => 1) fun h => Inv.inv (Units.mk0 (↑u) h)) (Inv. …
    -/
    rw [dif_neg (Units.ne_zero _), Units.mk0_val]
    /-
      🎉 no goals
    -/
  gcd a b := if a = 0 ∧ b = 0 then 0 else 1
  lcm a b := if a = 0 ∨ b = 0 then 0 else 1
  gcd_dvd_left a b := by
    -- Porting note (https://github.com/leanprover-community/mathlib4/issues/12129): additional beta reduction needed
    /-
      α : Type u_1
      G₀ : Type u_2
      inst✝¹ : CommGroupWithZero G₀
      inst✝ : DecidableEq G₀
      a b : G₀
      ⊢ Dvd.dvd ((fun a b => ite (And (Eq a 0) (Eq b 0)) 0 1) a b) a
    -/
    beta_reduce
    /-
      α : Type u_1
      G₀ : Type u_2
      inst✝¹ : CommGroupWithZero G₀
      inst✝ : DecidableEq G₀
      a b : G₀
      ⊢ Dvd.dvd (ite (And (Eq a 0) (Eq b 0)) 0 1) a
    -/
    split_ifs with h
      /-
        case pos
        α : Type u_1
        G₀ : Type u_2
        inst✝¹ : CommGroupWithZero G₀
        inst✝ : DecidableEq G₀
        a b : G₀
        h : And (Eq a 0) (Eq b 0)
        ⊢ Dvd.dvd 0 a
      -/
    · rw [h.1]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        G₀ : Type u_2
        inst✝¹ : CommGroupWithZero G₀
        inst✝ : DecidableEq G₀
        a b : G₀
        h : Not (And (Eq a 0) (Eq b 0))
        ⊢ Dvd.dvd 1 a
      -/
    · exact one_dvd _
      /-
        🎉 no goals
      -/
  gcd_dvd_right a b := by
    -- Porting note (https://github.com/leanprover-community/mathlib4/issues/12129): additional beta reduction needed
    /-
      α : Type u_1
      G₀ : Type u_2
      inst✝¹ : CommGroupWithZero G₀
      inst✝ : DecidableEq G₀
      a b : G₀
      ⊢ Dvd.dvd ((fun a b => ite (And (Eq a 0) (Eq b 0)) 0 1) a b) b
    -/
    beta_reduce
    /-
      α : Type u_1
      G₀ : Type u_2
      inst✝¹ : CommGroupWithZero G₀
      inst✝ : DecidableEq G₀
      a b : G₀
      ⊢ Dvd.dvd (ite (And (Eq a 0) (Eq b 0)) 0 1) b
    -/
    split_ifs with h
      /-
        case pos
        α : Type u_1
        G₀ : Type u_2
        inst✝¹ : CommGroupWithZero G₀
        inst✝ : DecidableEq G₀
        a b : G₀
        h : And (Eq a 0) (Eq b 0)
        ⊢ Dvd.dvd 0 b
      -/
    · rw [h.2]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        G₀ : Type u_2
        inst✝¹ : CommGroupWithZero G₀
        inst✝ : DecidableEq G₀
        a b : G₀
        h : Not (And (Eq a 0) (Eq b 0))
        ⊢ Dvd.dvd 1 b
      -/
    · exact one_dvd _
      /-
        🎉 no goals
      -/
  dvd_gcd := fun {a b c} hac hab => by
    -- Porting note (https://github.com/leanprover-community/mathlib4/issues/12129): additional beta reduction needed
    /-
      α : Type u_1
      G₀ : Type u_2
      inst✝¹ : CommGroupWithZero G₀
      inst✝ : DecidableEq G₀
      a b c : G₀
      hac : Dvd.dvd a c
      hab : Dvd.dvd a b
      ⊢ Dvd.dvd a ((fun a b => ite (And (Eq a 0) (Eq b 0)) 0 1) c b)
    -/
    beta_reduce
    /-
      α : Type u_1
      G₀ : Type u_2
      inst✝¹ : CommGroupWithZero G₀
      inst✝ : DecidableEq G₀
      a b c : G₀
      hac : Dvd.dvd a c
      hab : Dvd.dvd a b
      ⊢ Dvd.dvd a (ite (And (Eq c 0) (Eq b 0)) 0 1)
    -/
    split_ifs with h
      /-
        case pos
        α : Type u_1
        G₀ : Type u_2
        inst✝¹ : CommGroupWithZero G₀
        inst✝ : DecidableEq G₀
        a b c : G₀
        hac : Dvd.dvd a c
        hab : Dvd.dvd a b
        h : And (Eq c 0) (Eq b 0)
        ⊢ Dvd.dvd a 0
      -/
    · apply dvd_zero
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        G₀ : Type u_2
        inst✝¹ : CommGroupWithZero G₀
        inst✝ : DecidableEq G₀
        a b c : G₀
        hac : Dvd.dvd a c
        hab : Dvd.dvd a b
        h : Not (And (Eq c 0) (Eq b 0))
        ⊢ Dvd.dvd a 1
      -/
    · rw [not_and_or] at h
      /-
        case neg
        α : Type u_1
        G₀ : Type u_2
        inst✝¹ : CommGroupWithZero G₀
        inst✝ : DecidableEq G₀
        a b c : G₀
        hac : Dvd.dvd a c
        hab : Dvd.dvd a b
        h : Or (Not (Eq c 0)) (Not (Eq b 0))
        ⊢ Dvd.dvd a 1
      -/
      cases h
        /-
          case neg.inl
          α : Type u_1
          G₀ : Type u_2
          inst✝¹ : CommGroupWithZero G₀
          inst✝ : DecidableEq G₀
          a b c : G₀
          hac : Dvd.dvd a c
          hab : Dvd.dvd a b
          h✝ : Not (Eq c 0)
          ⊢ Dvd.dvd a 1
        -/
      · refine isUnit_iff_dvd_one.mp (isUnit_of_dvd_unit ?_ (IsUnit.mk0 _ ‹c ≠ 0›))
        /-
          case neg.inl
          α : Type u_1
          G₀ : Type u_2
          inst✝¹ : CommGroupWithZero G₀
          inst✝ : DecidableEq G₀
          a b c : G₀
          hac : Dvd.dvd a c
          hab : Dvd.dvd a b
          h✝ : Not (Eq c 0)
          ⊢ Dvd.dvd a c
        -/
        exact hac
        /-
          🎉 no goals
        -/
        /-
          case neg.inr
          α : Type u_1
          G₀ : Type u_2
          inst✝¹ : CommGroupWithZero G₀
          inst✝ : DecidableEq G₀
          a b c : G₀
          hac : Dvd.dvd a c
          hab : Dvd.dvd a b
          h✝ : Not (Eq b 0)
          ⊢ Dvd.dvd a 1
        -/
      · refine isUnit_iff_dvd_one.mp (isUnit_of_dvd_unit ?_ (IsUnit.mk0 _ ‹b ≠ 0›))
        /-
          case neg.inr
          α : Type u_1
          G₀ : Type u_2
          inst✝¹ : CommGroupWithZero G₀
          inst✝ : DecidableEq G₀
          a b c : G₀
          hac : Dvd.dvd a c
          hab : Dvd.dvd a b
          h✝ : Not (Eq b 0)
          ⊢ Dvd.dvd a b
        -/
        exact hab
        /-
          🎉 no goals
        -/
  gcd_mul_lcm a b := by
    /-
      α : Type u_1
      G₀ : Type u_2
      inst✝¹ : CommGroupWithZero G₀
      inst✝ : DecidableEq G₀
      a b : G₀
      ⊢ Associated (HMul.hMul ((fun a b => ite (And (Eq a 0) (Eq b 0)) 0 1) a b) ((f …
    -/
    by_cases ha : a = 0
      /-
        case pos
        α : Type u_1
        G₀ : Type u_2
        inst✝¹ : CommGroupWithZero G₀
        inst✝ : DecidableEq G₀
        a b : G₀
        ha : Eq a 0
        ⊢ Associated (HMul.hMul ((fun a b => ite (And (Eq a 0) (Eq b 0)) 0 1) a b) ((f …
      -/
    · simp only [ha, true_and, true_or, ite_true, mul_zero, zero_mul]
      /-
        case pos
        α : Type u_1
        G₀ : Type u_2
        inst✝¹ : CommGroupWithZero G₀
        inst✝ : DecidableEq G₀
        a b : G₀
        ha : Eq a 0
        ⊢ Associated 0 0
      -/
      exact Associated.refl _
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        G₀ : Type u_2
        inst✝¹ : CommGroupWithZero G₀
        inst✝ : DecidableEq G₀
        a b : G₀
        ha : Not (Eq a 0)
        ⊢ Associated (HMul.hMul ((fun a b => ite (And (Eq a 0) (Eq b 0)) 0 1) a b) ((f …
      -/
    · by_cases hb : b = 0
        /-
          case pos
          α : Type u_1
          G₀ : Type u_2
          inst✝¹ : CommGroupWithZero G₀
          inst✝ : DecidableEq G₀
          a b : G₀
          ha : Not (Eq a 0)
          hb : Eq b 0
          ⊢ Associated (HMul.hMul ((fun a b => ite (And (Eq a 0) (Eq b 0)) 0 1) a b) ((f …
        -/
      · simp only [hb, and_true, or_true, ite_true, mul_zero]
        /-
          case pos
          α : Type u_1
          G₀ : Type u_2
          inst✝¹ : CommGroupWithZero G₀
          inst✝ : DecidableEq G₀
          a b : G₀
          ha : Not (Eq a 0)
          hb : Eq b 0
          ⊢ Associated 0 0
        -/
        exact Associated.refl _
        /-
          🎉 no goals
        -/
      -- Porting note (https://github.com/leanprover-community/mathlib4/issues/12129): additional beta reduction needed
        /-
          case neg
          α : Type u_1
          G₀ : Type u_2
          inst✝¹ : CommGroupWithZero G₀
          inst✝ : DecidableEq G₀
          a b : G₀
          ha : Not (Eq a 0)
          hb : Not (Eq b 0)
          ⊢ Associated (HMul.hMul ((fun a b => ite (And (Eq a 0) (Eq b 0)) 0 1) a b) ((f …
        -/
      · beta_reduce
        /-
          case neg
          α : Type u_1
          G₀ : Type u_2
          inst✝¹ : CommGroupWithZero G₀
          inst✝ : DecidableEq G₀
          a b : G₀
          ha : Not (Eq a 0)
          hb : Not (Eq b 0)
          ⊢ Associated (HMul.hMul (ite (And (Eq a 0) (Eq b 0)) 0 1) (ite (Or (Eq a 0) (E …
        -/
        rw [if_neg (not_and_of_not_left _ ha), one_mul, if_neg (not_or_intro ha hb)]
        /-
          case neg
          α : Type u_1
          G₀ : Type u_2
          inst✝¹ : CommGroupWithZero G₀
          inst✝ : DecidableEq G₀
          a b : G₀
          ha : Not (Eq a 0)
          hb : Not (Eq b 0)
          ⊢ Associated 1 (HMul.hMul a b)
        -/
        exact (associated_one_iff_isUnit.mpr ((IsUnit.mk0 _ ha).mul (IsUnit.mk0 _ hb))).symm
        /-
          🎉 no goals
        -/
  lcm_zero_left _ := if_pos (Or.inl rfl)
  lcm_zero_right _ := if_pos (Or.inr rfl)
  -- `split_ifs` wants to split `normalize`, so handle the cases manually
                                                    /-
                                                      α : Type u_1
                                                      G₀ : Type u_2
                                                      inst✝¹ : CommGroupWithZero G₀
                                                      inst✝ : DecidableEq G₀
                                                      a b : G₀
                                                      h : And (Eq a 0) (Eq b 0)
                                                      ⊢ Eq (normalize (GCDMonoid.gcd a b)) (GCDMonoid.gcd a b)
                                                    -/
                                                    /-
                                                      🎉 no goals
                                                    -/
  normalize_gcd a b := if h : a = 0 ∧ b = 0 then by simp [if_pos h] else by simp [if_neg h]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
                                                    /-
                                                      α : Type u_1
                                                      G₀ : Type u_2
                                                      inst✝¹ : CommGroupWithZero G₀
                                                      inst✝ : DecidableEq G₀
                                                      a b : G₀
                                                      h : Or (Eq a 0) (Eq b 0)
                                                      ⊢ Eq (normalize (GCDMonoid.lcm a b)) (GCDMonoid.lcm a b)
                                                    -/
                                                    /-
                                                      🎉 no goals
                                                    -/
  normalize_lcm a b := if h : a = 0 ∨ b = 0 then by simp [if_pos h] else by simp [if_neg h]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


@[simp]
                                                                              /-
                                                                                G₀ : Type u_2
                                                                                inst✝¹ : CommGroupWithZero G₀
                                                                                inst✝ : DecidableEq G₀
                                                                                a : G₀
                                                                                h0 : Ne a 0
                                                                                ⊢ Eq (↑(NormalizationMonoid.normUnit a)) (Inv.inv a)
                                                                              -/
theorem coe_normUnit {a : G₀} (h0 : a ≠ 0) : (↑(normUnit a) : G₀) = a⁻¹ := by simp [normUnit, h0]
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


                                                                       /-
                                                                         G₀ : Type u_2
                                                                         inst✝¹ : CommGroupWithZero G₀
                                                                         inst✝ : DecidableEq G₀
                                                                         a : G₀
                                                                         h0 : Ne a 0
                                                                         ⊢ Eq (normalize a) 1
                                                                       -/
theorem normalize_eq_one {a : G₀} (h0 : a ≠ 0) : normalize a = 1 := by simp [normalize_apply, h0]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


instance instGCDMonoid : GCDMonoid (Associates α) where
  gcd := Quotient.map₂ gcd fun _ _ (ha : Associated _ _) _ _ (hb : Associated _ _) => ha.gcd hb
  lcm := Quotient.map₂ lcm fun _ _ (ha : Associated _ _) _ _ (hb : Associated _ _) => ha.lcm hb
                     /-
                       α : Type u_1
                       inst✝¹ : CancelCommMonoidWithZero α
                       inst✝ : GCDMonoid α
                       ⊢ ∀ (a b : Associates α), Dvd.dvd (Quotient.map₂ GCDMonoid.gcd ⋯ a b) a
                     -/
  gcd_dvd_left := by rintro ⟨a⟩ ⟨b⟩; exact mk_le_mk_of_dvd (gcd_dvd_left _ _)
                                     /-
                                       🎉 no goals
                                     -/
                      /-
                        α : Type u_1
                        inst✝¹ : CancelCommMonoidWithZero α
                        inst✝ : GCDMonoid α
                        ⊢ ∀ (a b : Associates α), Dvd.dvd (Quotient.map₂ GCDMonoid.gcd ⋯ a b) b
                      -/
  gcd_dvd_right := by rintro ⟨a⟩ ⟨b⟩; exact mk_le_mk_of_dvd (gcd_dvd_right _ _)
                                      /-
                                        🎉 no goals
                                      -/
  dvd_gcd := by
    /-
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : GCDMonoid α
      ⊢ ∀ {a b c : Associates α}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a (Quotient.ma …
    -/
    rintro ⟨a⟩ ⟨b⟩ ⟨c⟩ hac hbc
    /-
      case mk.mk.mk
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : GCDMonoid α
      a✝ : Associates α
      a : α
      b✝ : Associates α
      b : α
      c✝ : Associates α
      c : α
      hac : Dvd.dvd (Quot.mk (⇑(Associated.setoid α)) a) (Quot.mk (⇑(Associated.seto …
      hbc : Dvd.dvd (Quot.mk (⇑(Associated.setoid α)) a) (Quot.mk (⇑(Associated.seto …
      ⊢ Dvd.dvd (Quot.mk (⇑(Associated.setoid α)) a) (Quotient.map₂ GCDMonoid.gcd ⋯  …
    -/
    exact mk_le_mk_of_dvd (dvd_gcd (dvd_of_mk_le_mk hac) (dvd_of_mk_le_mk hbc))
    /-
      🎉 no goals
    -/
  gcd_mul_lcm := by
    /-
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : GCDMonoid α
      ⊢ ∀ (a b : Associates α), Associated (HMul.hMul (Quotient.map₂ GCDMonoid.gcd ⋯ …
    -/
    rintro ⟨a⟩ ⟨b⟩
    /-
      case mk.mk
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : GCDMonoid α
      a✝ : Associates α
      a : α
      b✝ : Associates α
      b : α
      ⊢ Associated (HMul.hMul (Quotient.map₂ GCDMonoid.gcd ⋯ (Quot.mk (⇑(Associated. …
    -/
    rw [associated_iff_eq]
    /-
      case mk.mk
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : GCDMonoid α
      a✝ : Associates α
      a : α
      b✝ : Associates α
      b : α
      ⊢ Eq (HMul.hMul (Quotient.map₂ GCDMonoid.gcd ⋯ (Quot.mk (⇑(Associated.setoid α …
    -/
    exact Quotient.sound <| gcd_mul_lcm _ _
    /-
      🎉 no goals
    -/
                      /-
                        α : Type u_1
                        inst✝¹ : CancelCommMonoidWithZero α
                        inst✝ : GCDMonoid α
                        ⊢ ∀ (a : Associates α), Eq (Quotient.map₂ GCDMonoid.lcm ⋯ 0 a) 0
                      -/
  lcm_zero_left := by rintro ⟨a⟩; exact congr_arg Associates.mk <| lcm_zero_left _
                                  /-
                                    🎉 no goals
                                  -/
                       /-
                         α : Type u_1
                         inst✝¹ : CancelCommMonoidWithZero α
                         inst✝ : GCDMonoid α
                         ⊢ ∀ (a : Associates α), Eq (Quotient.map₂ GCDMonoid.lcm ⋯ a 0) 0
                       -/
  lcm_zero_right := by rintro ⟨a⟩; exact congr_arg Associates.mk <| lcm_zero_right _
                                   /-
                                     🎉 no goals
                                   -/


theorem gcd_mk_mk {a b : α} : gcd (Associates.mk a) (Associates.mk b) = Associates.mk (gcd a b) :=
  rfl

theorem lcm_mk_mk {a b : α} : lcm (Associates.mk a) (Associates.mk b) = Associates.mk (lcm a b) :=
  rfl


