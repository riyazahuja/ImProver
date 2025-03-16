/-- This instance uses data fields from `Subtype.partialOrder` to help type-class inference.
The `Set.Ici` data fields are definitionally equal, but that requires unfolding semireducible
definitions, so type-class inference won't see this. -/
instance orderBot [Preorder α] {a : α} : OrderBot { x : α // a ≤ x } :=
  { Set.Ici.orderBot with }


theorem bot_eq [Preorder α] {a : α} : (⊥ : { x : α // a ≤ x }) = ⟨a, le_rfl⟩ :=
  rfl


instance noMaxOrder [PartialOrder α] [NoMaxOrder α] {a : α} : NoMaxOrder { x : α // a ≤ x } :=
                             /-
                               α : Type u_1
                               inst✝¹ : PartialOrder α
                               inst✝ : NoMaxOrder α
                               a : α
                               ⊢ NoMaxOrder ↑(Set.Ici a)
                             -/
  show NoMaxOrder (Ici a) by infer_instance
                             /-
                               🎉 no goals
                             -/


instance semilatticeSup [SemilatticeSup α] {a : α} : SemilatticeSup { x : α // a ≤ x } :=
  Set.Ici.semilatticeSup


instance semilatticeInf [SemilatticeInf α] {a : α} : SemilatticeInf { x : α // a ≤ x } :=
  Set.Ici.semilatticeInf


instance distribLattice [DistribLattice α] {a : α} : DistribLattice { x : α // a ≤ x } :=
  Set.Ici.distribLattice


instance instDenselyOrdered [Preorder α] [DenselyOrdered α] {a : α} :
    DenselyOrdered { x : α // a ≤ x } :=
  show DenselyOrdered (Ici a) from Set.instDenselyOrdered


/-- If `sSup ∅ ≤ a` then `{x : α // a ≤ x}` is a `ConditionallyCompleteLinearOrder`. -/
protected noncomputable abbrev conditionallyCompleteLinearOrder [ConditionallyCompleteLinearOrder α]
    {a : α} : ConditionallyCompleteLinearOrder { x : α // a ≤ x } :=
  { @ordConnectedSubsetConditionallyCompleteLinearOrder α (Set.Ici a) _ ⟨⟨a, le_rfl⟩⟩ _ with }


/-- If `sSup ∅ ≤ a` then `{x : α // a ≤ x}` is a `ConditionallyCompleteLinearOrderBot`.

This instance uses data fields from `Subtype.linearOrder` to help type-class inference.
The `Set.Ici` data fields are definitionally equal, but that requires unfolding semireducible
definitions, so type-class inference won't see this. -/
protected noncomputable abbrev conditionallyCompleteLinearOrderBot
    [ConditionallyCompleteLinearOrder α] (a : α) :
    ConditionallyCompleteLinearOrderBot { x : α // a ≤ x } :=
  { Nonneg.orderBot, Nonneg.conditionallyCompleteLinearOrder with
    csSup_empty := by
      /-
        α : Type u_1
        inst✝ : ConditionallyCompleteLinearOrder α
        a : α
        ⊢ Eq (SupSet.sSup EmptyCollection.emptyCollection) Bot.bot
      -/
      rw [@subset_sSup_def α (Set.Ici a) _ _ ⟨⟨a, le_rfl⟩⟩]; simp [bot_eq] }
                                                             /-
                                                               🎉 no goals
                                                             -/


instance inhabited [Preorder α] {a : α} : Inhabited { x : α // a ≤ x } :=
  ⟨⟨a, le_rfl⟩⟩


instance zero [Zero α] [Preorder α] : Zero { x : α // 0 ≤ x } :=
  ⟨⟨0, le_rfl⟩⟩


@[simp, norm_cast]
protected theorem coe_zero [Zero α] [Preorder α] : ((0 : { x : α // 0 ≤ x }) : α) = 0 :=
  rfl


@[simp]
theorem mk_eq_zero [Zero α] [Preorder α] {x : α} (hx : 0 ≤ x) :
    (⟨x, hx⟩ : { x : α // 0 ≤ x }) = 0 ↔ x = 0 :=
  Subtype.ext_iff


instance add [AddZeroClass α] [Preorder α] [AddLeftMono α] : Add { x : α // 0 ≤ x } :=
  ⟨fun x y => ⟨x + y, add_nonneg x.2 y.2⟩⟩


@[simp]
theorem mk_add_mk [AddZeroClass α] [Preorder α] [AddLeftMono α] {x y : α}
    (hx : 0 ≤ x) (hy : 0 ≤ y) :
    (⟨x, hx⟩ : { x : α // 0 ≤ x }) + ⟨y, hy⟩ = ⟨x + y, add_nonneg hx hy⟩ :=
  rfl


@[simp, norm_cast]
protected theorem coe_add [AddZeroClass α] [Preorder α] [AddLeftMono α]
    (a b : { x : α // 0 ≤ x }) : ((a + b : { x : α // 0 ≤ x }) : α) = a + b :=
  rfl


instance nsmul [AddMonoid α] [Preorder α] [AddLeftMono α] : SMul ℕ { x : α // 0 ≤ x } :=
  ⟨fun n x => ⟨n • (x : α), nsmul_nonneg x.prop n⟩⟩


@[simp]
theorem nsmul_mk [AddMonoid α] [Preorder α] [AddLeftMono α] (n : ℕ) {x : α}
    (hx : 0 ≤ x) : (n • (⟨x, hx⟩ : { x : α // 0 ≤ x })) = ⟨n • x, nsmul_nonneg hx n⟩ :=
  rfl


@[simp, norm_cast]
protected theorem coe_nsmul [AddMonoid α] [Preorder α] [AddLeftMono α]
    (n : ℕ) (a : { x : α // 0 ≤ x }) : ((n • a : { x : α // 0 ≤ x }) : α) = n • (a : α) :=
  rfl


instance one : One { x : α // 0 ≤ x } where
  one := ⟨1, zero_le_one⟩


@[simp, norm_cast]
protected theorem coe_one : ((1 : { x : α // 0 ≤ x }) : α) = 1 :=
  rfl


@[simp]
theorem mk_eq_one {x : α} (hx : 0 ≤ x) :
    (⟨x, hx⟩ : { x : α // 0 ≤ x }) = 1 ↔ x = 1 :=
  Subtype.ext_iff


instance mul : Mul { x : α // 0 ≤ x } where
  mul x y := ⟨x * y, mul_nonneg x.2 y.2⟩


@[simp, norm_cast]
protected theorem coe_mul (a b : { x : α // 0 ≤ x }) :
    ((a * b : { x : α // 0 ≤ x }) : α) = a * b :=
  rfl


@[simp]
theorem mk_mul_mk {x y : α} (hx : 0 ≤ x) (hy : 0 ≤ y) :
    (⟨x, hx⟩ : { x : α // 0 ≤ x }) * ⟨y, hy⟩ = ⟨x * y, mul_nonneg hx hy⟩ :=
  rfl


instance addMonoid : AddMonoid { x : α // 0 ≤ x } :=
  Subtype.coe_injective.addMonoid _ Nonneg.coe_zero (fun _ _ => rfl) fun _ _ => rfl


/-- Coercion `{x : α // 0 ≤ x} → α` as an `AddMonoidHom`. -/
def coeAddMonoidHom : { x : α // 0 ≤ x } →+ α :=
  { toFun := ((↑) : { x : α // 0 ≤ x } → α)
    map_zero' := Nonneg.coe_zero
    map_add' := Nonneg.coe_add }


@[norm_cast]
theorem nsmul_coe (n : ℕ) (r : { x : α // 0 ≤ x }) :
    ↑(n • r) = n • (r : α) :=
  Nonneg.coeAddMonoidHom.map_nsmul _ _


instance addCommMonoid : AddCommMonoid { x : α // 0 ≤ x } :=
  Subtype.coe_injective.addCommMonoid _ Nonneg.coe_zero (fun _ _ => rfl) (fun _ _ => rfl)


instance natCast : NatCast { x : α // 0 ≤ x } :=
  ⟨fun n => ⟨n, Nat.cast_nonneg' n⟩⟩


@[simp, norm_cast]
protected theorem coe_natCast (n : ℕ) : ((↑n : { x : α // 0 ≤ x }) : α) = n :=
  rfl


@[deprecated (since := "2024-04-17")]
alias coe_nat_cast := Nonneg.coe_natCast


@[simp]
theorem mk_natCast (n : ℕ) : (⟨n, n.cast_nonneg'⟩ : { x : α // 0 ≤ x }) = n :=
  rfl


@[deprecated (since := "2024-04-17")]
alias mk_nat_cast := mk_natCast


instance addMonoidWithOne : AddMonoidWithOne { x : α // 0 ≤ x } :=
  { Nonneg.one (α := α) with
    toNatCast := Nonneg.natCast
                       /-
                         α : Type u_1
                         inst✝³ : AddMonoidWithOne α
                         inst✝² : PartialOrder α
                         inst✝¹ : AddLeftMono α
                         inst✝ : ZeroLEOneClass α
                         ⊢ Eq (NatCast.natCast 0) 0
                       -/
    natCast_zero := by ext; simp
                            /-
                              🎉 no goals
                            -/
                                /-
                                  α : Type u_1
                                  inst✝³ : AddMonoidWithOne α
                                  inst✝² : PartialOrder α
                                  inst✝¹ : AddLeftMono α
                                  inst✝ : ZeroLEOneClass α
                                  x✝ : Nat
                                  ⊢ Eq (NatCast.natCast (HAdd.hAdd x✝ 1)) (HAdd.hAdd (NatCast.natCast x✝) 1)
                                -/
    natCast_succ := fun _ => by ext; simp }
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
theorem pow_nonneg {a : α} (H : 0 ≤ a) : ∀ n : ℕ, 0 ≤ a ^ n
  | 0 => by
    /-
      α : Type u_1
      inst✝³ : MonoidWithZero α
      inst✝² : Preorder α
      inst✝¹ : ZeroLEOneClass α
      inst✝ : PosMulMono α
      a : α
      H : LE.le 0 a
      ⊢ LE.le 0 (HPow.hPow a 0)
    -/
    rw [pow_zero]
    /-
      α : Type u_1
      inst✝³ : MonoidWithZero α
      inst✝² : Preorder α
      inst✝¹ : ZeroLEOneClass α
      inst✝ : PosMulMono α
      a : α
      H : LE.le 0 a
      ⊢ LE.le 0 1
    -/
    exact zero_le_one
    /-
      🎉 no goals
    -/
  | n + 1 => by
    /-
      α : Type u_1
      inst✝³ : MonoidWithZero α
      inst✝² : Preorder α
      inst✝¹ : ZeroLEOneClass α
      inst✝ : PosMulMono α
      a : α
      H : LE.le 0 a
      n : Nat
      ⊢ LE.le 0 (HPow.hPow a (HAdd.hAdd n 1))
    -/
    rw [pow_succ]
    /-
      α : Type u_1
      inst✝³ : MonoidWithZero α
      inst✝² : Preorder α
      inst✝¹ : ZeroLEOneClass α
      inst✝ : PosMulMono α
      a : α
      H : LE.le 0 a
      n : Nat
      ⊢ LE.le 0 (HMul.hMul (HPow.hPow a n) a)
    -/
    exact mul_nonneg (pow_nonneg H _) H
    /-
      🎉 no goals
    -/


instance pow : Pow { x : α // 0 ≤ x } ℕ where
  pow x n := ⟨(x : α) ^ n, pow_nonneg x.2 n⟩


@[simp, norm_cast]
protected theorem coe_pow (a : { x : α // 0 ≤ x }) (n : ℕ) :
    (↑(a ^ n) : α) = (a : α) ^ n :=
  rfl


@[simp]
theorem mk_pow {x : α} (hx : 0 ≤ x) (n : ℕ) :
    (⟨x, hx⟩ : { x : α // 0 ≤ x }) ^ n = ⟨x ^ n, pow_nonneg hx n⟩ :=
  rfl


instance semiring : Semiring { x : α // 0 ≤ x } :=
  Subtype.coe_injective.semiring _ Nonneg.coe_zero Nonneg.coe_one
    (fun _ _ => rfl) (fun _ _=> rfl) (fun _ _ => rfl)
    (fun _ _ => rfl) fun _ => rfl


                                                                  /-
                                                                    α : Type u_1
                                                                    inst✝⁴ : Semiring α
                                                                    inst✝³ : PartialOrder α
                                                                    inst✝² : ZeroLEOneClass α
                                                                    inst✝¹ : AddLeftMono α
                                                                    inst✝ : PosMulMono α
                                                                    ⊢ MonoidWithZero (Subtype fun x => LE.le 0 x)
                                                                  -/
instance monoidWithZero : MonoidWithZero { x : α // 0 ≤ x } := by infer_instance
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


/-- Coercion `{x : α // 0 ≤ x} → α` as a `RingHom`. -/
def coeRingHom : { x : α // 0 ≤ x } →+* α :=
  { toFun := ((↑) : { x : α // 0 ≤ x } → α)
    map_one' := Nonneg.coe_one
    map_mul' := Nonneg.coe_mul
    map_zero' := Nonneg.coe_zero,
    map_add' := Nonneg.coe_add }


instance commSemiring : CommSemiring { x : α // 0 ≤ x } :=
  Subtype.coe_injective.commSemiring _ Nonneg.coe_zero Nonneg.coe_one
    (fun _ _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl)
    (fun _ _ => rfl) fun _ => rfl


instance commMonoidWithZero : CommMonoidWithZero { x : α // 0 ≤ x } := inferInstance


/-- The function `a ↦ max a 0` of type `α → {x : α // 0 ≤ x}`. -/
def toNonneg (a : α) : { x : α // 0 ≤ x } :=
  ⟨max a 0, le_max_right _ _⟩


@[simp]
theorem coe_toNonneg {a : α} : (toNonneg a : α) = max a 0 :=
  rfl


@[simp]
                                                                           /-
                                                                             α : Type u_1
                                                                             inst✝¹ : Zero α
                                                                             inst✝ : LinearOrder α
                                                                             a : α
                                                                             h : LE.le 0 a
                                                                             ⊢ Eq (Nonneg.toNonneg a) ⟨a, h⟩
                                                                           -/
theorem toNonneg_of_nonneg {a : α} (h : 0 ≤ a) : toNonneg a = ⟨a, h⟩ := by simp [toNonneg, h]
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


@[simp]
theorem toNonneg_coe {a : { x : α // 0 ≤ x }} : toNonneg (a : α) = a :=
  toNonneg_of_nonneg a.2


@[simp]
theorem toNonneg_le {a : α} {b : { x : α // 0 ≤ x }} : toNonneg a ≤ b ↔ a ≤ b := by
  /-
    α : Type u_1
    inst✝¹ : Zero α
    inst✝ : LinearOrder α
    a : α
    b : Subtype fun x => LE.le 0 x
    ⊢ Iff (LE.le (Nonneg.toNonneg a) b) (LE.le a ↑b)
  -/
  cases' b with b hb
  /-
    case mk
    α : Type u_1
    inst✝¹ : Zero α
    inst✝ : LinearOrder α
    a b : α
    hb : LE.le 0 b
    ⊢ Iff (LE.le (Nonneg.toNonneg a) ⟨b, hb⟩) (LE.le a ↑⟨b, hb⟩)
  -/
  simp [toNonneg, hb]
  /-
    🎉 no goals
  -/


@[simp]
theorem toNonneg_lt {a : { x : α // 0 ≤ x }} {b : α} : a < toNonneg b ↔ ↑a < b := by
  /-
    α : Type u_1
    inst✝¹ : Zero α
    inst✝ : LinearOrder α
    a : Subtype fun x => LE.le 0 x
    b : α
    ⊢ Iff (LT.lt a (Nonneg.toNonneg b)) (LT.lt (↑a) b)
  -/
  cases' a with a ha
  /-
    case mk
    α : Type u_1
    inst✝¹ : Zero α
    inst✝ : LinearOrder α
    b a : α
    ha : LE.le 0 a
    ⊢ Iff (LT.lt ⟨a, ha⟩ (Nonneg.toNonneg b)) (LT.lt (↑⟨a, ha⟩) b)
  -/
  simp [toNonneg, ha.not_lt]
  /-
    🎉 no goals
  -/


instance sub [Sub α] : Sub { x : α // 0 ≤ x } :=
  ⟨fun x y => toNonneg (x - y)⟩


@[simp]
theorem mk_sub_mk [Sub α] {x y : α} (hx : 0 ≤ x) (hy : 0 ≤ y) :
    (⟨x, hx⟩ : { x : α // 0 ≤ x }) - ⟨y, hy⟩ = toNonneg (x - y) :=
  rfl


