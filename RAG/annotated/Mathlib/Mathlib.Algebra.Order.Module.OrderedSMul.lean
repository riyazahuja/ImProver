/-- The ordered scalar product property is when an ordered additive commutative monoid
with a partial order has a scalar multiplication which is compatible with the order. Note that this
is different from `IsOrderedSMul`, which uses `≤`, has no semiring assumption, and has no positivity
constraint on the defining conditions.
-/
class OrderedSMul (R M : Type*) [OrderedSemiring R] [OrderedAddCommMonoid M] [SMulWithZero R M] :
  Prop where
  /-- Scalar multiplication by positive elements preserves the order. -/
  protected smul_lt_smul_of_pos : ∀ {a b : M}, ∀ {c : R}, a < b → 0 < c → c • a < c • b
  /-- If `c • a < c • b` for some positive `c`, then `a < b`. -/
  protected lt_of_smul_lt_smul_of_pos : ∀ {a b : M}, ∀ {c : R}, c • a < c • b → 0 < c → a < b


instance OrderedSMul.toPosSMulStrictMono : PosSMulStrictMono R M where
  elim _a ha _b₁ _b₂ hb := OrderedSMul.smul_lt_smul_of_pos hb ha


instance OrderedSMul.toPosSMulReflectLT : PosSMulReflectLT R M :=
  PosSMulReflectLT.of_pos fun _a ha _b₁ _b₂ h ↦ OrderedSMul.lt_of_smul_lt_smul_of_pos h ha


instance OrderDual.instOrderedSMul : OrderedSMul R Mᵒᵈ where
  smul_lt_smul_of_pos := OrderedSMul.smul_lt_smul_of_pos (M := M)
  lt_of_smul_lt_smul_of_pos := OrderedSMul.lt_of_smul_lt_smul_of_pos (M := M)


/-- To prove that a linear ordered monoid is an ordered module, it suffices to verify only the first
axiom of `OrderedSMul`. -/
theorem OrderedSMul.mk'' [OrderedSemiring 𝕜] [LinearOrderedAddCommMonoid M] [SMulWithZero 𝕜 M]
    (h : ∀ ⦃c : 𝕜⦄, 0 < c → StrictMono fun a : M => c • a) : OrderedSMul 𝕜 M :=
  { smul_lt_smul_of_pos := fun hab hc => h hc hab
    lt_of_smul_lt_smul_of_pos := fun hab hc => (h hc).lt_iff_lt.1 hab }


instance Nat.orderedSMul [LinearOrderedCancelAddCommMonoid M] : OrderedSMul ℕ M :=
  OrderedSMul.mk'' fun n hn a b hab => by
    cases n with
    | zero => cases hn
    | succ n =>
      induction n with
      | zero => dsimp; rwa [one_nsmul, one_nsmul]
      | succ n ih => simp only [succ_nsmul _ n.succ, _root_.add_lt_add (ih n.succ_pos) hab]


instance Int.orderedSMul [LinearOrderedAddCommGroup M] : OrderedSMul ℤ M :=
  OrderedSMul.mk'' fun n hn => by
    /-
      ι : Type u_1
      𝕜 : Type u_2
      R : Type u_3
      M : Type u_4
      N : Type u_5
      inst✝ : LinearOrderedAddCommGroup M
      n : Int
      hn : LT.lt 0 n
      ⊢ StrictMono fun a => HSMul.hSMul n a
    -/
    cases n
      /-
        case ofNat
        ι : Type u_1
        𝕜 : Type u_2
        R : Type u_3
        M : Type u_4
        N : Type u_5
        inst✝ : LinearOrderedAddCommGroup M
        a✝ : Nat
        hn : LT.lt 0 (Int.ofNat a✝)
        ⊢ StrictMono fun a => HSMul.hSMul (Int.ofNat a✝) a
      -/
    · simp only [Int.ofNat_eq_coe, Int.natCast_pos, natCast_zsmul] at hn ⊢
      /-
        case ofNat
        ι : Type u_1
        𝕜 : Type u_2
        R : Type u_3
        M : Type u_4
        N : Type u_5
        inst✝ : LinearOrderedAddCommGroup M
        a✝ : Nat
        hn : LT.lt 0 a✝
        ⊢ StrictMono fun a => HSMul.hSMul a✝ a
      -/
      exact strictMono_smul_left_of_pos hn
      /-
        🎉 no goals
      -/
      /-
        case negSucc
        ι : Type u_1
        𝕜 : Type u_2
        R : Type u_3
        M : Type u_4
        N : Type u_5
        inst✝ : LinearOrderedAddCommGroup M
        a✝ : Nat
        hn : LT.lt 0 (Int.negSucc a✝)
        ⊢ StrictMono fun a => HSMul.hSMul (Int.negSucc a✝) a
      -/
    · cases (Int.negSucc_not_pos _).1 hn
      /-
        🎉 no goals
      -/


instance LinearOrderedSemiring.toOrderedSMul : OrderedSMul R R :=
  OrderedSMul.mk'' fun _ => strictMono_mul_left_of_pos


/-- To prove that a vector space over a linear ordered field is ordered, it suffices to verify only
the first axiom of `OrderedSMul`. -/
theorem OrderedSMul.mk' (h : ∀ ⦃a b : M⦄ ⦃c : 𝕜⦄, a < b → 0 < c → c • a ≤ c • b) :
    OrderedSMul 𝕜 M := by
  have hlt' : ∀ (a b : M) (c : 𝕜), a < b → 0 < c → c • a < c • b := by
    refine fun a b c hab hc => (h hab hc).lt_of_ne ?_
    rw [Ne, hc.ne'.isUnit.smul_left_cancel]
    exact hab.ne
  /-
    𝕜 : Type u_2
    M : Type u_4
    inst✝² : LinearOrderedSemifield 𝕜
    inst✝¹ : OrderedAddCommMonoid M
    inst✝ : MulActionWithZero 𝕜 M
    h : ∀ ⦃a b : M⦄ ⦃c : 𝕜⦄, LT.lt a b → LT.lt 0 c → LE.le (HSMul.hSMul c a) (HSMu …
    hlt' : ∀ (a b : M) (c : 𝕜), LT.lt a b → LT.lt 0 c → LT.lt (HSMul.hSMul c a) (H …
    ⊢ OrderedSMul 𝕜 M
  -/
  refine ⟨fun {a b c} => hlt' a b c, fun {a b c hab hc} => ?_⟩
  /-
    𝕜 : Type u_2
    M : Type u_4
    inst✝² : LinearOrderedSemifield 𝕜
    inst✝¹ : OrderedAddCommMonoid M
    inst✝ : MulActionWithZero 𝕜 M
    h : ∀ ⦃a b : M⦄ ⦃c : 𝕜⦄, LT.lt a b → LT.lt 0 c → LE.le (HSMul.hSMul c a) (HSMu …
    hlt' : ∀ (a b : M) (c : 𝕜), LT.lt a b → LT.lt 0 c → LT.lt (HSMul.hSMul c a) (H …
    a b : M
    c : 𝕜
    hab : LT.lt (HSMul.hSMul c a) (HSMul.hSMul c b)
    hc : LT.lt 0 c
    ⊢ LT.lt a b
  -/
  obtain ⟨c, rfl⟩ := hc.ne'.isUnit
  /-
    case intro
    𝕜 : Type u_2
    M : Type u_4
    inst✝² : LinearOrderedSemifield 𝕜
    inst✝¹ : OrderedAddCommMonoid M
    inst✝ : MulActionWithZero 𝕜 M
    h : ∀ ⦃a b : M⦄ ⦃c : 𝕜⦄, LT.lt a b → LT.lt 0 c → LE.le (HSMul.hSMul c a) (HSMu …
    hlt' : ∀ (a b : M) (c : 𝕜), LT.lt a b → LT.lt 0 c → LT.lt (HSMul.hSMul c a) (H …
    a b : M
    c : Units 𝕜
    hab : LT.lt (HSMul.hSMul (↑c) a) (HSMul.hSMul (↑c) b)
    hc : LT.lt 0 ↑c
    ⊢ LT.lt a b
  -/
  rw [← inv_smul_smul c a, ← inv_smul_smul c b]
  /-
    case intro
    𝕜 : Type u_2
    M : Type u_4
    inst✝² : LinearOrderedSemifield 𝕜
    inst✝¹ : OrderedAddCommMonoid M
    inst✝ : MulActionWithZero 𝕜 M
    h : ∀ ⦃a b : M⦄ ⦃c : 𝕜⦄, LT.lt a b → LT.lt 0 c → LE.le (HSMul.hSMul c a) (HSMu …
    hlt' : ∀ (a b : M) (c : 𝕜), LT.lt a b → LT.lt 0 c → LT.lt (HSMul.hSMul c a) (H …
    a b : M
    c : Units 𝕜
    hab : LT.lt (HSMul.hSMul (↑c) a) (HSMul.hSMul (↑c) b)
    hc : LT.lt 0 ↑c
    ⊢ LT.lt (HSMul.hSMul (Inv.inv c) (HSMul.hSMul c a)) (HSMul.hSMul (Inv.inv c) ( …
  -/
  refine hlt' _ _ _ hab (pos_of_mul_pos_right ?_ hc.le)
  /-
    case intro
    𝕜 : Type u_2
    M : Type u_4
    inst✝² : LinearOrderedSemifield 𝕜
    inst✝¹ : OrderedAddCommMonoid M
    inst✝ : MulActionWithZero 𝕜 M
    h : ∀ ⦃a b : M⦄ ⦃c : 𝕜⦄, LT.lt a b → LT.lt 0 c → LE.le (HSMul.hSMul c a) (HSMu …
    hlt' : ∀ (a b : M) (c : 𝕜), LT.lt a b → LT.lt 0 c → LT.lt (HSMul.hSMul c a) (H …
    a b : M
    c : Units 𝕜
    hab : LT.lt (HSMul.hSMul (↑c) a) (HSMul.hSMul (↑c) b)
    hc : LT.lt 0 ↑c
    ⊢ LT.lt 0 (HMul.hMul ↑c ↑(Inv.inv c))
  -/
  simp only [c.mul_inv, zero_lt_one]
  /-
    🎉 no goals
  -/


instance [OrderedSMul 𝕜 M] [OrderedSMul 𝕜 N] : OrderedSMul 𝕜 (M × N) :=
  OrderedSMul.mk' fun _ _ _ h hc =>
    ⟨smul_le_smul_of_nonneg_left h.1.1 hc.le, smul_le_smul_of_nonneg_left h.1.2 hc.le⟩


instance Pi.orderedSMul {M : ι → Type*} [∀ i, OrderedAddCommMonoid (M i)]
    [∀ i, MulActionWithZero 𝕜 (M i)] [∀ i, OrderedSMul 𝕜 (M i)] : OrderedSMul 𝕜 (∀ i, M i) :=
  OrderedSMul.mk' fun _ _ _ h hc i => smul_le_smul_of_nonneg_left (h.le i) hc.le


lemma inf_eq_half_smul_add_sub_abs_sub (x y : β) : x ⊓ y = (⅟2 : α) • (x + y - |y - x|) := by
  rw [← two_nsmul_inf_eq_add_sub_abs_sub x y, two_smul, ← two_smul α,
    smul_smul, invOf_mul_self, one_smul]


lemma sup_eq_half_smul_add_add_abs_sub (x y : β) : x ⊔ y = (⅟2 : α) • (x + y + |y - x|) := by
  rw [← two_nsmul_sup_eq_add_add_abs_sub x y, two_smul, ← two_smul α,
    smul_smul, invOf_mul_self, one_smul]


lemma inf_eq_half_smul_add_sub_abs_sub' (x y : β) : x ⊓ y = (2⁻¹ : α) • (x + y - |y - x|) := by
  /-
    α : Type u_6
    β : Type u_7
    inst✝⁵ : DivisionSemiring α
    inst✝⁴ : NeZero 2
    inst✝³ : Lattice β
    inst✝² : AddCommGroup β
    inst✝¹ : Module α β
    inst✝ : AddLeftMono β
    x y : β
    ⊢ Eq (Min.min x y) (HSMul.hSMul (Inv.inv 2) (HSub.hSub (HAdd.hAdd x y) (abs (H …
  -/
  letI := invertibleOfNonzero (two_ne_zero' α)
  /-
    α : Type u_6
    β : Type u_7
    inst✝⁵ : DivisionSemiring α
    inst✝⁴ : NeZero 2
    inst✝³ : Lattice β
    inst✝² : AddCommGroup β
    inst✝¹ : Module α β
    inst✝ : AddLeftMono β
    x y : β
    this : Invertible 2 := invertibleOfNonzero ⋯
    ⊢ Eq (Min.min x y) (HSMul.hSMul (Inv.inv 2) (HSub.hSub (HAdd.hAdd x y) (abs (H …
  -/
  exact inf_eq_half_smul_add_sub_abs_sub α x y
  /-
    🎉 no goals
  -/


lemma sup_eq_half_smul_add_add_abs_sub' (x y : β) : x ⊔ y = (2⁻¹ : α) • (x + y + |y - x|) := by
  /-
    α : Type u_6
    β : Type u_7
    inst✝⁵ : DivisionSemiring α
    inst✝⁴ : NeZero 2
    inst✝³ : Lattice β
    inst✝² : AddCommGroup β
    inst✝¹ : Module α β
    inst✝ : AddLeftMono β
    x y : β
    ⊢ Eq (Max.max x y) (HSMul.hSMul (Inv.inv 2) (HAdd.hAdd (HAdd.hAdd x y) (abs (H …
  -/
  letI := invertibleOfNonzero (two_ne_zero' α)
  /-
    α : Type u_6
    β : Type u_7
    inst✝⁵ : DivisionSemiring α
    inst✝⁴ : NeZero 2
    inst✝³ : Lattice β
    inst✝² : AddCommGroup β
    inst✝¹ : Module α β
    inst✝ : AddLeftMono β
    x y : β
    this : Invertible 2 := invertibleOfNonzero ⋯
    ⊢ Eq (Max.max x y) (HSMul.hSMul (Inv.inv 2) (HAdd.hAdd (HAdd.hAdd x y) (abs (H …
  -/
  exact sup_eq_half_smul_add_add_abs_sub α x y
  /-
    🎉 no goals
  -/


