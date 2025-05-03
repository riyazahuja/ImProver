/-- Typeclass for division rounded down. For each `a > 0`, this asserts the existence of a right
adjoint to the map `b ↦ a • b : β → β`. -/
class FloorDiv where
  /-- Flooring division. If `a > 0`, then `b ⌊/⌋ a` is the greatest `c` such that `a • c ≤ b`. -/
  floorDiv : β → α → β
  /-- Do not use this. Use `gc_floorDiv_smul` or `gc_floorDiv_mul` instead. -/
  protected floorDiv_gc ⦃a⦄ : 0 < a → GaloisConnection (a • ·) (floorDiv · a)
  /-- Do not use this. Use `floorDiv_nonpos` instead. -/
  protected floorDiv_nonpos ⦃a⦄ : a ≤ 0 → ∀ b, floorDiv b a = 0
  /-- Do not use this. Use `zero_floorDiv` instead. -/
  protected zero_floorDiv (a) : floorDiv 0 a = 0


/-- Typeclass for division rounded up. For each `a > 0`, this asserts the existence of a left
adjoint to the map `b ↦ a • b : β → β`. -/
class CeilDiv where
  /-- Ceiling division. If `a > 0`, then `b ⌈/⌉ a` is the least `c` such that `b ≤ a • c`. -/
  ceilDiv : β → α → β
  /-- Do not use this. Use `gc_smul_ceilDiv` or `gc_mul_ceilDiv` instead. -/
  protected ceilDiv_gc ⦃a⦄ : 0 < a → GaloisConnection (ceilDiv · a) (a • ·)
  /-- Do not use this. Use `ceilDiv_nonpos` instead. -/
  protected ceilDiv_nonpos ⦃a⦄ : a ≤ 0 → ∀ b, ceilDiv b a = 0
  /-- Do not use this. Use `zero_ceilDiv` instead. -/
  protected zero_ceilDiv (a) : ceilDiv 0 a = 0


@[inherit_doc] infixl:70 " ⌊/⌋ "   => FloorDiv.floorDiv

@[inherit_doc] infixl:70 " ⌈/⌉ "   => CeilDiv.ceilDiv


lemma gc_floorDiv_smul (ha : 0 < a) : GaloisConnection (a • · : β → β) (· ⌊/⌋ a) :=
  FloorDiv.floorDiv_gc ha


@[simp] lemma le_floorDiv_iff_smul_le (ha : 0 < a) : c ≤ b ⌊/⌋ a ↔ a • c ≤ b :=
  (gc_floorDiv_smul ha _ _).symm


@[simp] lemma floorDiv_of_nonpos (ha : a ≤ 0) (b : β) : b ⌊/⌋ a = 0 := FloorDiv.floorDiv_nonpos ha _

                                                      /-
                                                        α : Type u_2
                                                        β : Type u_3
                                                        inst✝³ : OrderedAddCommMonoid α
                                                        inst✝² : OrderedAddCommMonoid β
                                                        inst✝¹ : SMulZeroClass α β
                                                        inst✝ : FloorDiv α β
                                                        b : β
                                                        ⊢ Eq (FloorDiv.floorDiv b 0) 0
                                                      -/
lemma floorDiv_zero (b : β) : b ⌊/⌋ (0 : α) = 0 := by simp
                                                      /-
                                                        🎉 no goals
                                                      -/

@[simp] lemma zero_floorDiv (a : α) : (0 : β) ⌊/⌋ a = 0 := FloorDiv.zero_floorDiv _


lemma smul_floorDiv_le (ha : 0 < a) : a • (b ⌊/⌋ a) ≤ b := (le_floorDiv_iff_smul_le ha).1 le_rfl


lemma gc_smul_ceilDiv (ha : 0 < a) : GaloisConnection (· ⌈/⌉ a) (a • · : β → β) :=
  CeilDiv.ceilDiv_gc ha


@[simp]
lemma ceilDiv_le_iff_le_smul (ha : 0 < a) : b ⌈/⌉ a ≤ c ↔ b ≤ a • c := gc_smul_ceilDiv ha _ _


@[simp] lemma ceilDiv_of_nonpos (ha : a ≤ 0) (b : β) : b ⌈/⌉ a = 0 := CeilDiv.ceilDiv_nonpos ha _

                                                     /-
                                                       α : Type u_2
                                                       β : Type u_3
                                                       inst✝³ : OrderedAddCommMonoid α
                                                       inst✝² : OrderedAddCommMonoid β
                                                       inst✝¹ : SMulZeroClass α β
                                                       inst✝ : CeilDiv α β
                                                       b : β
                                                       ⊢ Eq (CeilDiv.ceilDiv b 0) 0
                                                     -/
lemma ceilDiv_zero (b : β) : b ⌈/⌉ (0 : α) = 0 := by simp
                                                     /-
                                                       🎉 no goals
                                                     -/

@[simp] lemma zero_ceilDiv (a : α) : (0 : β) ⌈/⌉ a = 0 := CeilDiv.zero_ceilDiv _


lemma le_smul_ceilDiv (ha : 0 < a) : b ≤ a • (b ⌈/⌉ a) := (ceilDiv_le_iff_le_smul ha).1 le_rfl


lemma floorDiv_le_ceilDiv : b ⌊/⌋ a ≤ b ⌈/⌉ a := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝⁵ : LinearOrderedAddCommMonoid α
    inst✝⁴ : OrderedAddCommMonoid β
    inst✝³ : SMulZeroClass α β
    inst✝² : PosSMulReflectLE α β
    inst✝¹ : FloorDiv α β
    inst✝ : CeilDiv α β
    a : α
    b : β
    ⊢ LE.le (FloorDiv.floorDiv b a) (CeilDiv.ceilDiv b a)
  -/
  obtain ha | ha := le_or_lt a 0
    /-
      case inl
      α : Type u_2
      β : Type u_3
      inst✝⁵ : LinearOrderedAddCommMonoid α
      inst✝⁴ : OrderedAddCommMonoid β
      inst✝³ : SMulZeroClass α β
      inst✝² : PosSMulReflectLE α β
      inst✝¹ : FloorDiv α β
      inst✝ : CeilDiv α β
      a : α
      b : β
      ha : LE.le a 0
      ⊢ LE.le (FloorDiv.floorDiv b a) (CeilDiv.ceilDiv b a)
    -/
  · simp [ha]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_2
      β : Type u_3
      inst✝⁵ : LinearOrderedAddCommMonoid α
      inst✝⁴ : OrderedAddCommMonoid β
      inst✝³ : SMulZeroClass α β
      inst✝² : PosSMulReflectLE α β
      inst✝¹ : FloorDiv α β
      inst✝ : CeilDiv α β
      a : α
      b : β
      ha : LT.lt 0 a
      ⊢ LE.le (FloorDiv.floorDiv b a) (CeilDiv.ceilDiv b a)
    -/
  · exact le_of_smul_le_smul_left ((smul_floorDiv_le ha).trans <| le_smul_ceilDiv ha) ha
    /-
      🎉 no goals
    -/


@[simp] lemma floorDiv_one [Nontrivial α] (b : β) : b ⌊/⌋ (1 : α) = b :=
                                    /-
                                      α : Type u_2
                                      β : Type u_3
                                      inst✝⁴ : OrderedSemiring α
                                      inst✝³ : OrderedAddCommMonoid β
                                      inst✝² : MulActionWithZero α β
                                      inst✝¹ : FloorDiv α β
                                      inst✝ : Nontrivial α
                                      b c : β
                                      ⊢ Iff (LE.le c (FloorDiv.floorDiv b 1)) (LE.le c b)
                                    -/
  eq_of_forall_le_iff <| fun c ↦ by simp [zero_lt_one' α]
                                    /-
                                      🎉 no goals
                                    -/


@[simp] lemma smul_floorDiv [PosSMulMono α β] [PosSMulReflectLE α β] (ha : 0 < a) (b : β) :
    a • b ⌊/⌋ a = b :=
                            /-
                              α : Type u_2
                              β : Type u_3
                              inst✝⁵ : OrderedSemiring α
                              inst✝⁴ : OrderedAddCommMonoid β
                              inst✝³ : MulActionWithZero α β
                              inst✝² : FloorDiv α β
                              a : α
                              inst✝¹ : PosSMulMono α β
                              inst✝ : PosSMulReflectLE α β
                              ha : LT.lt 0 a
                              b : β
                              ⊢ ∀ (c : β), Iff (LE.le c (FloorDiv.floorDiv (HSMul.hSMul a b) a)) (LE.le c b)
                            -/
  eq_of_forall_le_iff <| by simp [smul_le_smul_iff_of_pos_left, ha]
                            /-
                              🎉 no goals
                            -/


@[simp] lemma ceilDiv_one [Nontrivial α] (b : β) : b ⌈/⌉ (1 : α) = b :=
                                    /-
                                      α : Type u_2
                                      β : Type u_3
                                      inst✝⁴ : OrderedSemiring α
                                      inst✝³ : OrderedAddCommMonoid β
                                      inst✝² : MulActionWithZero α β
                                      inst✝¹ : CeilDiv α β
                                      inst✝ : Nontrivial α
                                      b c : β
                                      ⊢ Iff (LE.le (CeilDiv.ceilDiv b 1) c) (LE.le b c)
                                    -/
  eq_of_forall_ge_iff <| fun c ↦ by simp [zero_lt_one' α]
                                    /-
                                      🎉 no goals
                                    -/


@[simp] lemma smul_ceilDiv [PosSMulMono α β] [PosSMulReflectLE α β] (ha : 0 < a) (b : β) :
    a • b ⌈/⌉ a = b :=
                            /-
                              α : Type u_2
                              β : Type u_3
                              inst✝⁵ : OrderedSemiring α
                              inst✝⁴ : OrderedAddCommMonoid β
                              inst✝³ : MulActionWithZero α β
                              inst✝² : CeilDiv α β
                              a : α
                              inst✝¹ : PosSMulMono α β
                              inst✝ : PosSMulReflectLE α β
                              ha : LT.lt 0 a
                              b : β
                              ⊢ ∀ (c : β), Iff (LE.le (CeilDiv.ceilDiv (HSMul.hSMul a b) a) c) (LE.le b c)
                            -/
  eq_of_forall_ge_iff <| by simp [smul_le_smul_iff_of_pos_left, ha]
                            /-
                              🎉 no goals
                            -/


lemma gc_floorDiv_mul (ha : 0 < a) : GaloisConnection (a * ·) (· ⌊/⌋ a) := gc_floorDiv_smul ha

lemma le_floorDiv_iff_mul_le (ha : 0 < a) : c ≤ b ⌊/⌋ a ↔ a • c ≤ b := le_floorDiv_iff_smul_le ha


lemma gc_mul_ceilDiv (ha : 0 < a) : GaloisConnection (· ⌈/⌉ a) (a * ·) := gc_smul_ceilDiv ha

lemma ceilDiv_le_iff_le_mul (ha : 0 < a) : b ⌈/⌉ a ≤ c ↔ b ≤ a * c := ceilDiv_le_iff_le_smul ha


instance instFloorDiv : FloorDiv ℕ ℕ where
  floorDiv := HDiv.hDiv
                         /-
                           ι : Type u_1
                           α : Type u_2
                           β : Type u_3
                           a : Nat
                           ha : LT.lt 0 a
                           ⊢ GaloisConnection (fun x => HSMul.hSMul a x) fun x => HDiv.hDiv x a
                         -/
  floorDiv_gc a ha := by simpa [mul_comm] using Nat.galoisConnection_mul_div ha
                         /-
                           🎉 no goals
                         -/
                               /-
                                 ι : Type u_1
                                 α : Type u_2
                                 β : Type u_3
                                 a : Nat
                                 ha : LE.le a 0
                                 b : Nat
                                 ⊢ Eq (HDiv.hDiv b a) 0
                               -/
  floorDiv_nonpos a ha b := by rw [ha.antisymm <| zero_le _, Nat.div_zero]
                               /-
                                 🎉 no goals
                               -/
  zero_floorDiv := Nat.zero_div


instance instCeilDiv : CeilDiv ℕ ℕ where
  ceilDiv a b := (a + b - 1) / b
  ceilDiv_gc a ha b c := by
    /-
      ι : Type u_1
      α : Type u_2
      β : Type u_3
      a : Nat
      ha : LT.lt 0 a
      b c : Nat
      ⊢ Iff (LE.le ((fun x => (fun a b => HDiv.hDiv (HSub.hSub (HAdd.hAdd a b) 1) b) …
    -/
    simp [div_le_iff_le_mul_add_pred ha, add_assoc, tsub_add_cancel_of_le <| succ_le_iff.2 ha]
    /-
      🎉 no goals
    -/
                              /-
                                ι : Type u_1
                                α : Type u_2
                                β : Type u_3
                                a : Nat
                                ha : LE.le a 0
                                b : Nat
                                ⊢ Eq ((fun a b => HDiv.hDiv (HSub.hSub (HAdd.hAdd a b) 1) b) b a) 0
                              -/
  ceilDiv_nonpos a ha b := by simp_rw [ha.antisymm <| zero_le _, Nat.div_zero]
                              /-
                                🎉 no goals
                              -/
                       /-
                         ι : Type u_1
                         α : Type u_2
                         β : Type u_3
                         a : Nat
                         ⊢ Eq ((fun a b => HDiv.hDiv (HSub.hSub (HAdd.hAdd a b) 1) b) 0 a) 0
                       -/
                                   /-
                                     🎉 no goals
                                   -/
  zero_ceilDiv a := by cases a <;> simp [Nat.div_eq_zero_iff]
                                   /-
                                     🎉 no goals
                                   -/


@[simp] lemma floorDiv_eq_div (a b : ℕ) : a ⌊/⌋ b = a / b := rfl

lemma ceilDiv_eq_add_pred_div (a b : ℕ) : a ⌈/⌉ b = (a + b - 1) / b := rfl


instance instFloorDiv : FloorDiv α (∀ i, π i) where
  floorDiv f a i := f i ⌊/⌋ a
  floorDiv_gc _a ha _f _g := forall_congr' fun _i ↦ gc_floorDiv_smul ha _ _
                               /-
                                 ι : Type u_1
                                 α : Type u_2
                                 β : Type u_3
                                 π : ι → Type u_4
                                 inst✝³ : OrderedAddCommMonoid α
                                 inst✝² : (i : ι) → OrderedAddCommMonoid (π i)
                                 inst✝¹ : (i : ι) → SMulZeroClass α (π i)
                                 inst✝ : (i : ι) → FloorDiv α (π i)
                                 a : α
                                 ha : LE.le a 0
                                 f : (i : ι) → π i
                                 ⊢ Eq ((fun f a i => FloorDiv.floorDiv (f i) a) f a) 0
                               -/
  floorDiv_nonpos a ha f := by ext i; exact floorDiv_of_nonpos ha _
                                      /-
                                        🎉 no goals
                                      -/
                        /-
                          ι : Type u_1
                          α : Type u_2
                          β : Type u_3
                          π : ι → Type u_4
                          inst✝³ : OrderedAddCommMonoid α
                          inst✝² : (i : ι) → OrderedAddCommMonoid (π i)
                          inst✝¹ : (i : ι) → SMulZeroClass α (π i)
                          inst✝ : (i : ι) → FloorDiv α (π i)
                          a : α
                          ⊢ Eq ((fun f a i => FloorDiv.floorDiv (f i) a) 0 a) 0
                        -/
  zero_floorDiv a := by ext i; exact zero_floorDiv a
                               /-
                                 🎉 no goals
                               -/


lemma floorDiv_def (f : ∀ i, π i) (a : α) : f ⌊/⌋ a = fun i ↦ f i ⌊/⌋ a := rfl

@[simp] lemma floorDiv_apply (f : ∀ i, π i) (a : α) (i : ι) : (f ⌊/⌋ a) i = f i ⌊/⌋ a := rfl


instance instCeilDiv : CeilDiv α (∀ i, π i) where
  ceilDiv f a i := f i ⌈/⌉ a
  ceilDiv_gc _a ha _f _g := forall_congr' fun _i ↦ gc_smul_ceilDiv ha _ _
                              /-
                                ι : Type u_1
                                α : Type u_2
                                β : Type u_3
                                π : ι → Type u_4
                                inst✝³ : OrderedAddCommMonoid α
                                inst✝² : (i : ι) → OrderedAddCommMonoid (π i)
                                inst✝¹ : (i : ι) → SMulZeroClass α (π i)
                                inst✝ : (i : ι) → CeilDiv α (π i)
                                a : α
                                ha : LE.le a 0
                                f : (i : ι) → π i
                                ⊢ Eq ((fun f a i => CeilDiv.ceilDiv (f i) a) f a) 0
                              -/
  ceilDiv_nonpos a ha f := by ext i; exact ceilDiv_of_nonpos ha _
                                     /-
                                       🎉 no goals
                                     -/
                       /-
                         ι : Type u_1
                         α : Type u_2
                         β : Type u_3
                         π : ι → Type u_4
                         inst✝³ : OrderedAddCommMonoid α
                         inst✝² : (i : ι) → OrderedAddCommMonoid (π i)
                         inst✝¹ : (i : ι) → SMulZeroClass α (π i)
                         inst✝ : (i : ι) → CeilDiv α (π i)
                         a : α
                         ⊢ Eq ((fun f a i => CeilDiv.ceilDiv (f i) a) 0 a) 0
                       -/
  zero_ceilDiv a := by ext; exact zero_ceilDiv _
                            /-
                              🎉 no goals
                            -/


lemma ceilDiv_def (f : ∀ i, π i) (a : α) : f ⌈/⌉ a = fun i ↦ f i ⌈/⌉ a := rfl

@[simp] lemma ceilDiv_apply (f : ∀ i, π i) (a : α) (i : ι) : (f ⌈/⌉ a) i = f i ⌈/⌉ a := rfl


noncomputable instance instFloorDiv : FloorDiv α (ι →₀ β) where
  floorDiv f a := f.mapRange (· ⌊/⌋ a) <| zero_floorDiv _
  floorDiv_gc _a ha f _g := forall_congr' fun i ↦ by
    /-
      ι : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝³ : OrderedAddCommMonoid α
      inst✝² : OrderedAddCommMonoid β
      inst✝¹ : SMulZeroClass α β
      inst✝ : FloorDiv α β
      f✝ : Finsupp ι β
      a _a : α
      ha : LT.lt 0 _a
      f _g : Finsupp ι β
      i : ι
      ⊢ Iff (LE.le (((fun x => HSMul.hSMul _a x) f) i) (_g i)) (LE.le (f i) (((fun x …
    -/
    simpa only [coe_smul, Pi.smul_apply, mapRange_apply] using gc_floorDiv_smul ha (f i) _
    /-
      🎉 no goals
    -/
                               /-
                                 ι : Type u_1
                                 α : Type u_2
                                 β : Type u_3
                                 inst✝³ : OrderedAddCommMonoid α
                                 inst✝² : OrderedAddCommMonoid β
                                 inst✝¹ : SMulZeroClass α β
                                 inst✝ : FloorDiv α β
                                 f✝ : Finsupp ι β
                                 a✝ a : α
                                 ha : LE.le a 0
                                 f : Finsupp ι β
                                 ⊢ Eq ((fun f a => Finsupp.mapRange (fun x => FloorDiv.floorDiv x a) ⋯ f) f a) 0
                               -/
  floorDiv_nonpos a ha f := by ext i; exact floorDiv_of_nonpos ha _
                                      /-
                                        🎉 no goals
                                      -/
                        /-
                          ι : Type u_1
                          α : Type u_2
                          β : Type u_3
                          inst✝³ : OrderedAddCommMonoid α
                          inst✝² : OrderedAddCommMonoid β
                          inst✝¹ : SMulZeroClass α β
                          inst✝ : FloorDiv α β
                          f : Finsupp ι β
                          a✝ a : α
                          ⊢ Eq ((fun f a => Finsupp.mapRange (fun x => FloorDiv.floorDiv x a) ⋯ f) 0 a) 0
                        -/
  zero_floorDiv a := by ext; exact zero_floorDiv _
                             /-
                               🎉 no goals
                             -/


lemma floorDiv_def (f : ι →₀ β) (a : α) : f ⌊/⌋ a = f.mapRange (· ⌊/⌋ a) (zero_floorDiv _) := rfl

@[norm_cast] lemma coe_floorDiv (f : ι →₀ β) (a : α) : f ⌊/⌋ a = fun i ↦ f i ⌊/⌋ a := rfl

@[simp] lemma floorDiv_apply (f : ι →₀ β) (a : α) (i : ι) : (f ⌊/⌋ a) i = f i ⌊/⌋ a := rfl


lemma support_floorDiv_subset : (f ⌊/⌋ a).support ⊆ f.support := by
  /-
    ι : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : OrderedAddCommMonoid α
    inst✝² : OrderedAddCommMonoid β
    inst✝¹ : SMulZeroClass α β
    inst✝ : FloorDiv α β
    f : Finsupp ι β
    a : α
    ⊢ HasSubset.Subset (FloorDiv.floorDiv f a).support f.support
  -/
  simp +contextual [Finset.subset_iff, not_imp_not]
  /-
    🎉 no goals
  -/


noncomputable instance instCeilDiv : CeilDiv α (ι →₀ β) where
  ceilDiv f a := f.mapRange (· ⌈/⌉ a) <| zero_ceilDiv _
  ceilDiv_gc _a ha f _g := forall_congr' fun i ↦ by
    /-
      ι : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝³ : OrderedAddCommMonoid α
      inst✝² : OrderedAddCommMonoid β
      inst✝¹ : SMulZeroClass α β
      inst✝ : CeilDiv α β
      f✝ : Finsupp ι β
      a _a : α
      ha : LT.lt 0 _a
      f _g : Finsupp ι β
      i : ι
      ⊢ Iff (LE.le (((fun x => (fun f a => Finsupp.mapRange (fun x => CeilDiv.ceilDi …
    -/
    simpa only [coe_smul, Pi.smul_apply, mapRange_apply] using gc_smul_ceilDiv ha (f i) _
    /-
      🎉 no goals
    -/
                              /-
                                ι : Type u_1
                                α : Type u_2
                                β : Type u_3
                                inst✝³ : OrderedAddCommMonoid α
                                inst✝² : OrderedAddCommMonoid β
                                inst✝¹ : SMulZeroClass α β
                                inst✝ : CeilDiv α β
                                f✝ : Finsupp ι β
                                a✝ a : α
                                ha : LE.le a 0
                                f : Finsupp ι β
                                ⊢ Eq ((fun f a => Finsupp.mapRange (fun x => CeilDiv.ceilDiv x a) ⋯ f) f a) 0
                              -/
  ceilDiv_nonpos a ha f := by ext i; exact ceilDiv_of_nonpos ha _
                                     /-
                                       🎉 no goals
                                     -/
                       /-
                         ι : Type u_1
                         α : Type u_2
                         β : Type u_3
                         inst✝³ : OrderedAddCommMonoid α
                         inst✝² : OrderedAddCommMonoid β
                         inst✝¹ : SMulZeroClass α β
                         inst✝ : CeilDiv α β
                         f : Finsupp ι β
                         a✝ a : α
                         ⊢ Eq ((fun f a => Finsupp.mapRange (fun x => CeilDiv.ceilDiv x a) ⋯ f) 0 a) 0
                       -/
  zero_ceilDiv a := by ext; exact zero_ceilDiv _
                            /-
                              🎉 no goals
                            -/


lemma ceilDiv_def (f : ι →₀ β) (a : α) : f ⌈/⌉ a = f.mapRange (· ⌈/⌉ a) (zero_ceilDiv _) := rfl

@[norm_cast] lemma coe_ceilDiv_def (f : ι →₀ β) (a : α) : f ⌈/⌉ a = fun i ↦ f i ⌈/⌉ a := rfl

@[simp] lemma ceilDiv_apply (f : ι →₀ β) (a : α) (i : ι) : (f ⌈/⌉ a) i = f i ⌈/⌉ a := rfl


lemma support_ceilDiv_subset : (f ⌈/⌉ a).support ⊆ f.support := by
  /-
    ι : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : OrderedAddCommMonoid α
    inst✝² : OrderedAddCommMonoid β
    inst✝¹ : SMulZeroClass α β
    inst✝ : CeilDiv α β
    f : Finsupp ι β
    a : α
    ⊢ HasSubset.Subset (CeilDiv.ceilDiv f a).support f.support
  -/
  simp +contextual [Finset.subset_iff, not_imp_not]
  /-
    🎉 no goals
  -/


