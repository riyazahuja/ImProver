/-- An idempotent semiring is a semiring with the additional property that addition is idempotent.
-/
class IdemSemiring (α : Type u) extends Semiring α, SemilatticeSup α where
  protected sup := (· + ·)
  protected add_eq_sup : ∀ a b : α, a + b = a ⊔ b := by
    intros
    rfl
  /-- The bottom element of an idempotent semiring: `0` by default -/
  protected bot : α := 0
  protected bot_le : ∀ a, bot ≤ a


/-- An idempotent commutative semiring is a commutative semiring with the additional property that
addition is idempotent. -/
class IdemCommSemiring (α : Type u) extends CommSemiring α, IdemSemiring α


/-- Notation typeclass for the Kleene star `∗`. -/
class KStar (α : Type*) where
  /-- The Kleene star operator on a Kleene algebra -/
  protected kstar : α → α


@[inherit_doc] scoped[Computability] postfix:1024 "∗" => KStar.kstar


/-- A Kleene Algebra is an idempotent semiring with an additional unary operator `kstar` (for Kleene
star) that satisfies the following properties:
* `1 + a * a∗ ≤ a∗`
* `1 + a∗ * a ≤ a∗`
* If `a * c + b ≤ c`, then `a∗ * b ≤ c`
* If `c * a + b ≤ c`, then `b * a∗ ≤ c`
-/
class KleeneAlgebra (α : Type*) extends IdemSemiring α, KStar α where
  protected one_le_kstar : ∀ a : α, 1 ≤ a∗
  protected mul_kstar_le_kstar : ∀ a : α, a * a∗ ≤ a∗
  protected kstar_mul_le_kstar : ∀ a : α, a∗ * a ≤ a∗
  protected mul_kstar_le_self : ∀ a b : α, b * a ≤ b → b * a∗ ≤ b
  protected kstar_mul_le_self : ∀ a b : α, a * b ≤ b → a∗ * b ≤ b

-- See note [lower instance priority]

instance (priority := 100) IdemSemiring.toOrderBot [IdemSemiring α] : OrderBot α :=
  { ‹IdemSemiring α› with }

-- See note [reducible non-instances]

/-- Construct an idempotent semiring from an idempotent addition. -/
abbrev IdemSemiring.ofSemiring [Semiring α] (h : ∀ a : α, a + a = a) : IdemSemiring α :=
  { ‹Semiring α› with
    le := fun a b ↦ a + b = b
    le_refl := h
    le_trans := fun a b c hab hbc ↦ by
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        π : ι → Type u_4
        inst✝ : Semiring α
        h : ∀ (a : α), Eq (HAdd.hAdd a a) a
        a b c : α
        hab : LE.le a b
        hbc : LE.le b c
        ⊢ LE.le a c
      -/
      simp only
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        π : ι → Type u_4
        inst✝ : Semiring α
        h : ∀ (a : α), Eq (HAdd.hAdd a a) a
        a b c : α
        hab : LE.le a b
        hbc : LE.le b c
        ⊢ Eq (HAdd.hAdd a c) c
      -/
      rw [← hbc, ← add_assoc, hab]
      /-
        🎉 no goals
      -/
                                        /-
                                          α : Type u_1
                                          β : Type u_2
                                          ι : Type u_3
                                          π : ι → Type u_4
                                          inst✝ : Semiring α
                                          h : ∀ (a : α), Eq (HAdd.hAdd a a) a
                                          a b : α
                                          hab : LE.le a b
                                          hba : LE.le b a
                                          ⊢ Eq a b
                                        -/
    le_antisymm := fun a b hab hba ↦ by rwa [← hba, add_comm]
                                        /-
                                          🎉 no goals
                                        -/
    sup := (· + ·)
    le_sup_left := fun a b ↦ by
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        π : ι → Type u_4
        inst✝ : Semiring α
        h : ∀ (a : α), Eq (HAdd.hAdd a a) a
        a b : α
        ⊢ LE.le a ((fun x1 x2 => HAdd.hAdd x1 x2) a b)
      -/
      simp only
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        π : ι → Type u_4
        inst✝ : Semiring α
        h : ∀ (a : α), Eq (HAdd.hAdd a a) a
        a b : α
        ⊢ Eq (HAdd.hAdd a (HAdd.hAdd a b)) (HAdd.hAdd a b)
      -/
      rw [← add_assoc, h]
      /-
        🎉 no goals
      -/
    le_sup_right := fun a b ↦ by
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        π : ι → Type u_4
        inst✝ : Semiring α
        h : ∀ (a : α), Eq (HAdd.hAdd a a) a
        a b : α
        ⊢ LE.le b ((fun x1 x2 => HAdd.hAdd x1 x2) a b)
      -/
      simp only
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        π : ι → Type u_4
        inst✝ : Semiring α
        h : ∀ (a : α), Eq (HAdd.hAdd a a) a
        a b : α
        ⊢ Eq (HAdd.hAdd b (HAdd.hAdd a b)) (HAdd.hAdd a b)
      -/
      rw [add_comm, add_assoc, h]
      /-
        🎉 no goals
      -/
    sup_le := fun a b c hab hbc ↦ by
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        π : ι → Type u_4
        inst✝ : Semiring α
        h : ∀ (a : α), Eq (HAdd.hAdd a a) a
        a b c : α
        hab : LE.le a c
        hbc : LE.le b c
        ⊢ LE.le ((fun x1 x2 => HAdd.hAdd x1 x2) a b) c
      -/
      simp only
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        π : ι → Type u_4
        inst✝ : Semiring α
        h : ∀ (a : α), Eq (HAdd.hAdd a a) a
        a b c : α
        hab : LE.le a c
        hbc : LE.le b c
        ⊢ Eq (HAdd.hAdd (HAdd.hAdd a b) c) c
      -/
      rwa [add_assoc, hbc]
      /-
        🎉 no goals
      -/
    bot := 0
    bot_le := zero_add }


theorem add_eq_sup (a b : α) : a + b = a ⊔ b :=
  IdemSemiring.add_eq_sup _ _

-- Porting note: This simp theorem often leads to timeout when `α` has rich structure.
--               So, this theorem should be scoped.

                                           /-
                                             α : Type u_1
                                             inst✝ : IdemSemiring α
                                             a : α
                                             ⊢ Eq (HAdd.hAdd a a) a
                                           -/
theorem add_idem (a : α) : a + a = a := by simp
                                           /-
                                             🎉 no goals
                                           -/


theorem nsmul_eq_self : ∀ {n : ℕ} (_ : n ≠ 0) (a : α), n • a = a
  | 0, h => (h rfl).elim
  | 1, _ => one_nsmul
                           /-
                             α : Type u_1
                             inst✝ : IdemSemiring α
                             n : Nat
                             x✝ : Ne (HAdd.hAdd n 2) 0
                             a : α
                             ⊢ Eq (HSMul.hSMul (HAdd.hAdd n 2) a) a
                           -/
  | n + 2, _ => fun a ↦ by rw [succ_nsmul, nsmul_eq_self n.succ_ne_zero, add_idem]
                           /-
                             🎉 no goals
                           -/


                                                     /-
                                                       α : Type u_1
                                                       inst✝ : IdemSemiring α
                                                       a b : α
                                                       ⊢ Iff (Eq (HAdd.hAdd a b) a) (LE.le b a)
                                                     -/
theorem add_eq_left_iff_le : a + b = a ↔ b ≤ a := by simp
                                                     /-
                                                       🎉 no goals
                                                     -/


                                                      /-
                                                        α : Type u_1
                                                        inst✝ : IdemSemiring α
                                                        a b : α
                                                        ⊢ Iff (Eq (HAdd.hAdd a b) b) (LE.le a b)
                                                      -/
theorem add_eq_right_iff_le : a + b = b ↔ a ≤ b := by simp
                                                      /-
                                                        🎉 no goals
                                                      -/


alias ⟨_, LE.le.add_eq_left⟩ := add_eq_left_iff_le


alias ⟨_, LE.le.add_eq_right⟩ := add_eq_right_iff_le


                                                     /-
                                                       α : Type u_1
                                                       inst✝ : IdemSemiring α
                                                       a b c : α
                                                       ⊢ Iff (LE.le (HAdd.hAdd a b) c) (And (LE.le a c) (LE.le b c))
                                                     -/
theorem add_le_iff : a + b ≤ c ↔ a ≤ c ∧ b ≤ c := by simp
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem add_le (ha : a ≤ c) (hb : b ≤ c) : a + b ≤ c :=
  add_le_iff.2 ⟨ha, hb⟩

-- See note [lower instance priority]

instance (priority := 100) IdemSemiring.toCanonicallyOrderedAddCommMonoid :
    CanonicallyOrderedAddCommMonoid α :=
  { ‹IdemSemiring α› with
    add_le_add_left := fun a b hbc c ↦ by
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        π : ι → Type u_4
        inst✝ : IdemSemiring α
        a✝ b✝ c✝ a b : α
        hbc : LE.le a b
        c : α
        ⊢ LE.le (HAdd.hAdd c a) (HAdd.hAdd c b)
      -/
      simp_rw [add_eq_sup]
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        π : ι → Type u_4
        inst✝ : IdemSemiring α
        a✝ b✝ c✝ a b : α
        hbc : LE.le a b
        c : α
        ⊢ LE.le (Max.max c a) (Max.max c b)
      -/
      exact sup_le_sup_left hbc _
      /-
        🎉 no goals
      -/
    exists_add_of_le := fun h ↦ ⟨_, h.add_eq_right.symm⟩
                                                         /-
                                                           α : Type u_1
                                                           β : Type u_2
                                                           ι : Type u_3
                                                           π : ι → Type u_4
                                                           inst✝ : IdemSemiring α
                                                           a✝ b✝ c a b : α
                                                           ⊢ Eq (HAdd.hAdd a (HAdd.hAdd a b)) (HAdd.hAdd a b)
                                                         -/
    le_self_add := fun a b ↦ add_eq_right_iff_le.1 <| by rw [← add_assoc, add_idem] }
                                                         /-
                                                           🎉 no goals
                                                         -/

-- See note [lower instance priority]

instance (priority := 100) IdemSemiring.toMulLeftMono : MulLeftMono α :=
                                              /-
                                                α : Type u_1
                                                β : Type u_2
                                                ι : Type u_3
                                                π : ι → Type u_4
                                                inst✝ : IdemSemiring α
                                                a✝ b✝ c✝ a b c : α
                                                hbc : LE.le b c
                                                ⊢ Eq (HAdd.hAdd (HMul.hMul a c) (HMul.hMul a b)) (HMul.hMul a c)
                                              -/
  ⟨fun a b c hbc ↦ add_eq_left_iff_le.1 <| by rw [← mul_add, hbc.add_eq_left]⟩
                                              /-
                                                🎉 no goals
                                              -/

-- See note [lower instance priority]

instance (priority := 100) IdemSemiring.toMulRightMono : MulRightMono α :=
                                              /-
                                                α : Type u_1
                                                β : Type u_2
                                                ι : Type u_3
                                                π : ι → Type u_4
                                                inst✝ : IdemSemiring α
                                                a✝ b✝ c✝ a b c : α
                                                hbc : LE.le b c
                                                ⊢ Eq (HAdd.hAdd (Function.swap (fun x1 x2 => HMul.hMul x1 x2) a c) (Function.s …
                                              -/
  ⟨fun a b c hbc ↦ add_eq_left_iff_le.1 <| by rw [← add_mul, hbc.add_eq_left]⟩
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
theorem one_le_kstar : 1 ≤ a∗ :=
  KleeneAlgebra.one_le_kstar _


theorem mul_kstar_le_kstar : a * a∗ ≤ a∗ :=
  KleeneAlgebra.mul_kstar_le_kstar _


theorem kstar_mul_le_kstar : a∗ * a ≤ a∗ :=
  KleeneAlgebra.kstar_mul_le_kstar _


theorem mul_kstar_le_self : b * a ≤ b → b * a∗ ≤ b :=
  KleeneAlgebra.mul_kstar_le_self _ _


theorem kstar_mul_le_self : a * b ≤ b → a∗ * b ≤ b :=
  KleeneAlgebra.kstar_mul_le_self _ _


theorem mul_kstar_le (hb : b ≤ c) (ha : c * a ≤ c) : b * a∗ ≤ c :=
  (mul_le_mul_right' hb _).trans <| mul_kstar_le_self ha


theorem kstar_mul_le (hb : b ≤ c) (ha : a * c ≤ c) : a∗ * b ≤ c :=
  (mul_le_mul_left' hb _).trans <| kstar_mul_le_self ha


theorem kstar_le_of_mul_le_left (hb : 1 ≤ b) : b * a ≤ b → a∗ ≤ b := by
  /-
    α : Type u_1
    inst✝ : KleeneAlgebra α
    a b : α
    hb : LE.le 1 b
    ⊢ LE.le (HMul.hMul b a) b → LE.le (KStar.kstar a) b
  -/
  simpa using mul_kstar_le hb
  /-
    🎉 no goals
  -/


theorem kstar_le_of_mul_le_right (hb : 1 ≤ b) : a * b ≤ b → a∗ ≤ b := by
  /-
    α : Type u_1
    inst✝ : KleeneAlgebra α
    a b : α
    hb : LE.le 1 b
    ⊢ LE.le (HMul.hMul a b) b → LE.le (KStar.kstar a) b
  -/
  simpa using kstar_mul_le hb
  /-
    🎉 no goals
  -/


@[simp]
theorem le_kstar : a ≤ a∗ :=
  le_trans (le_mul_of_one_le_left' one_le_kstar) kstar_mul_le_kstar


@[mono]
theorem kstar_mono : Monotone (KStar.kstar : α → α) :=
  fun _ _ h ↦
    kstar_le_of_mul_le_left one_le_kstar <| kstar_mul_le (h.trans le_kstar) <| mul_kstar_le_kstar


@[simp]
theorem kstar_eq_one : a∗ = 1 ↔ a ≤ 1 :=
  ⟨le_kstar.trans_eq,
                                                                           /-
                                                                             α : Type u_1
                                                                             inst✝ : KleeneAlgebra α
                                                                             a : α
                                                                             h : LE.le a 1
                                                                             ⊢ LE.le (HMul.hMul 1 a) 1
                                                                           -/
    fun h ↦ one_le_kstar.antisymm' <| kstar_le_of_mul_le_left le_rfl <| by rwa [one_mul]⟩
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


@[simp] lemma kstar_zero : (0 : α)∗ = 1 := kstar_eq_one.2 (zero_le _)


@[simp]
theorem kstar_one : (1 : α)∗ = 1 :=
  kstar_eq_one.2 le_rfl


@[simp]
theorem kstar_mul_kstar (a : α) : a∗ * a∗ = a∗ :=
  (mul_kstar_le le_rfl <| kstar_mul_le_kstar).antisymm <| le_mul_of_one_le_left' one_le_kstar


@[simp]
theorem kstar_eq_self : a∗ = a ↔ a * a = a ∧ 1 ≤ a :=
               /-
                 α : Type u_1
                 inst✝ : KleeneAlgebra α
                 a : α
                 h : Eq (KStar.kstar a) a
                 ⊢ Eq (HMul.hMul a a) a
               -/
  ⟨fun h ↦ ⟨by rw [← h, kstar_mul_kstar], one_le_kstar.trans_eq h⟩,
               /-
                 🎉 no goals
               -/
    fun h ↦ (kstar_le_of_mul_le_left h.2 h.1.le).antisymm le_kstar⟩


@[simp]
theorem kstar_idem (a : α) : a∗∗ = a∗ :=
  kstar_eq_self.2 ⟨kstar_mul_kstar _, one_le_kstar⟩


@[simp]
theorem pow_le_kstar : ∀ {n : ℕ}, a ^ n ≤ a∗
  | 0 => (pow_zero _).trans_le one_le_kstar
  | n + 1 => by
    /-
      α : Type u_1
      inst✝ : KleeneAlgebra α
      a : α
      n : Nat
      ⊢ LE.le (HPow.hPow a (HAdd.hAdd n 1)) (KStar.kstar a)
    -/
    rw [pow_succ']
    /-
      α : Type u_1
      inst✝ : KleeneAlgebra α
      a : α
      n : Nat
      ⊢ LE.le (HMul.hMul a (HPow.hPow a n)) (KStar.kstar a)
    -/
    exact (mul_le_mul_left' pow_le_kstar _).trans mul_kstar_le_kstar
    /-
      🎉 no goals
    -/


instance instIdemSemiring [IdemSemiring α] [IdemSemiring β] : IdemSemiring (α × β) :=
  { Prod.instSemiring, Prod.instSemilatticeSup _ _, Prod.instOrderBot _ _ with
    add_eq_sup := fun _ _ ↦ Prod.ext (add_eq_sup _ _) (add_eq_sup _ _) }


instance [IdemCommSemiring α] [IdemCommSemiring β] : IdemCommSemiring (α × β) :=
  { Prod.instCommSemiring, Prod.instIdemSemiring with }


instance : KleeneAlgebra (α × β) :=
  { Prod.instIdemSemiring with
    kstar := fun a ↦ (a.1∗, a.2∗)
    one_le_kstar := fun _ ↦ ⟨one_le_kstar, one_le_kstar⟩
    mul_kstar_le_kstar := fun _ ↦ ⟨mul_kstar_le_kstar, mul_kstar_le_kstar⟩
    kstar_mul_le_kstar := fun _ ↦ ⟨kstar_mul_le_kstar, kstar_mul_le_kstar⟩
    mul_kstar_le_self := fun _ _ ↦ And.imp mul_kstar_le_self mul_kstar_le_self
    kstar_mul_le_self := fun _ _ ↦ And.imp kstar_mul_le_self kstar_mul_le_self }


theorem kstar_def (a : α × β) : a∗ = (a.1∗, a.2∗) :=
  rfl


@[simp]
theorem fst_kstar (a : α × β) : a∗.1 = a.1∗ :=
  rfl


@[simp]
theorem snd_kstar (a : α × β) : a∗.2 = a.2∗ :=
  rfl


instance instIdemSemiring [∀ i, IdemSemiring (π i)] : IdemSemiring (∀ i, π i) :=
  { Pi.semiring, Pi.instSemilatticeSup, Pi.instOrderBot with
    add_eq_sup := fun _ _ ↦ funext fun _ ↦ add_eq_sup _ _ }


instance [∀ i, IdemCommSemiring (π i)] : IdemCommSemiring (∀ i, π i) :=
  { Pi.commSemiring, Pi.instIdemSemiring with }


instance : KleeneAlgebra (∀ i, π i) :=
  { Pi.instIdemSemiring with
    kstar := fun a i ↦ (a i)∗
    one_le_kstar := fun _ _ ↦ one_le_kstar
    mul_kstar_le_kstar := fun _ _ ↦ mul_kstar_le_kstar
    kstar_mul_le_kstar := fun _ _ ↦ kstar_mul_le_kstar
    mul_kstar_le_self := fun _ _ h _ ↦ mul_kstar_le_self <| h _
    kstar_mul_le_self := fun _ _ h _ ↦ kstar_mul_le_self <| h _ }


theorem kstar_def (a : ∀ i, π i) : a∗ = fun i ↦ (a i)∗ :=
  rfl


@[simp]
theorem kstar_apply (a : ∀ i, π i) (i : ι) : a∗ i = (a i)∗ :=
  rfl


/-- Pullback an `IdemSemiring` instance along an injective function. -/
protected abbrev idemSemiring [IdemSemiring α] [Zero β] [One β] [Add β] [Mul β] [Pow β ℕ] [SMul ℕ β]
    [NatCast β] [Max β] [Bot β] (f : β → α) (hf : Injective f) (zero : f 0 = 0) (one : f 1 = 1)
    (add : ∀ x y, f (x + y) = f x + f y) (mul : ∀ x y, f (x * y) = f x * f y)
    (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x) (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n)
    (natCast : ∀ n : ℕ, f n = n) (sup : ∀ a b, f (a ⊔ b) = f a ⊔ f b) (bot : f ⊥ = ⊥) :
    IdemSemiring β :=
  { hf.semiring f zero one add mul nsmul npow natCast, hf.semilatticeSup _ sup,
    ‹Bot β› with
                                     /-
                                       α : Type u_1
                                       β : Type u_2
                                       ι : Type u_3
                                       π : ι → Type u_4
                                       inst✝⁹ : IdemSemiring α
                                       inst✝⁸ : Zero β
                                       inst✝⁷ : One β
                                       inst✝⁶ : Add β
                                       inst✝⁵ : Mul β
                                       inst✝⁴ : Pow β Nat
                                       inst✝³ : SMul Nat β
                                       inst✝² : NatCast β
                                       inst✝¹ : Max β
                                       inst✝ : Bot β
                                       f : β → α
                                       hf : Function.Injective f
                                       zero : Eq (f 0) 0
                                       one : Eq (f 1) 1
                                       add : ∀ (x y : β), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
                                       mul : ∀ (x y : β), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                                       nsmul : ∀ (n : Nat) (x : β), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
                                       npow : ∀ (x : β) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                                       natCast : ∀ (n : Nat), Eq (f ↑n) ↑n
                                       sup : ∀ (a b : β), Eq (f (Max.max a b)) (Max.max (f a) (f b))
                                       bot : Eq (f Bot.bot) Bot.bot
                                       a b : β
                                       ⊢ Eq (f (HAdd.hAdd a b)) (f (Max.max a b))
                                     -/
    add_eq_sup := fun a b ↦ hf <| by rw [sup, add, add_eq_sup]
                                     /-
                                       🎉 no goals
                                     -/
    bot := ⊥
    bot_le := fun a ↦ bot.trans_le <| @bot_le _ _ _ <| f a }

-- See note [reducible non-instances]

/-- Pullback an `IdemCommSemiring` instance along an injective function. -/
protected abbrev idemCommSemiring [IdemCommSemiring α] [Zero β] [One β] [Add β] [Mul β] [Pow β ℕ]
    [SMul ℕ β] [NatCast β] [Max β] [Bot β] (f : β → α) (hf : Injective f) (zero : f 0 = 0)
    (one : f 1 = 1) (add : ∀ x y, f (x + y) = f x + f y) (mul : ∀ x y, f (x * y) = f x * f y)
    (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x) (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n)
    (natCast : ∀ n : ℕ, f n = n) (sup : ∀ a b, f (a ⊔ b) = f a ⊔ f b) (bot : f ⊥ = ⊥) :
    IdemCommSemiring β :=
  { hf.commSemiring f zero one add mul nsmul npow natCast,
    hf.idemSemiring f zero one add mul nsmul npow natCast sup bot with }

-- See note [reducible non-instances]

/-- Pullback a `KleeneAlgebra` instance along an injective function. -/
protected abbrev kleeneAlgebra [KleeneAlgebra α] [Zero β] [One β] [Add β] [Mul β] [Pow β ℕ]
    [SMul ℕ β] [NatCast β] [Max β] [Bot β] [KStar β] (f : β → α) (hf : Injective f) (zero : f 0 = 0)
    (one : f 1 = 1) (add : ∀ x y, f (x + y) = f x + f y) (mul : ∀ x y, f (x * y) = f x * f y)
    (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x) (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n)
    (natCast : ∀ n : ℕ, f n = n) (sup : ∀ a b, f (a ⊔ b) = f a ⊔ f b) (bot : f ⊥ = ⊥)
    (kstar : ∀ a, f a∗ = (f a)∗) : KleeneAlgebra β :=
  { hf.idemSemiring f zero one add mul nsmul npow natCast sup bot,
    ‹KStar β› with
    one_le_kstar := fun a ↦ one.trans_le <| by
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        π : ι → Type u_4
        inst✝¹⁰ : KleeneAlgebra α
        inst✝⁹ : Zero β
        inst✝⁸ : One β
        inst✝⁷ : Add β
        inst✝⁶ : Mul β
        inst✝⁵ : Pow β Nat
        inst✝⁴ : SMul Nat β
        inst✝³ : NatCast β
        inst✝² : Max β
        inst✝¹ : Bot β
        inst✝ : KStar β
        f : β → α
        hf : Function.Injective f
        zero : Eq (f 0) 0
        one : Eq (f 1) 1
        add : ∀ (x y : β), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
        mul : ∀ (x y : β), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
        nsmul : ∀ (n : Nat) (x : β), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
        npow : ∀ (x : β) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
        natCast : ∀ (n : Nat), Eq (f ↑n) ↑n
        sup : ∀ (a b : β), Eq (f (Max.max a b)) (Max.max (f a) (f b))
        bot : Eq (f Bot.bot) Bot.bot
        kstar : ∀ (a : β), Eq (f (KStar.kstar a)) (KStar.kstar (f a))
        a : β
        ⊢ LE.le 1 (f (KStar.kstar a))
      -/
      rw [kstar]
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        π : ι → Type u_4
        inst✝¹⁰ : KleeneAlgebra α
        inst✝⁹ : Zero β
        inst✝⁸ : One β
        inst✝⁷ : Add β
        inst✝⁶ : Mul β
        inst✝⁵ : Pow β Nat
        inst✝⁴ : SMul Nat β
        inst✝³ : NatCast β
        inst✝² : Max β
        inst✝¹ : Bot β
        inst✝ : KStar β
        f : β → α
        hf : Function.Injective f
        zero : Eq (f 0) 0
        one : Eq (f 1) 1
        add : ∀ (x y : β), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
        mul : ∀ (x y : β), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
        nsmul : ∀ (n : Nat) (x : β), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
        npow : ∀ (x : β) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
        natCast : ∀ (n : Nat), Eq (f ↑n) ↑n
        sup : ∀ (a b : β), Eq (f (Max.max a b)) (Max.max (f a) (f b))
        bot : Eq (f Bot.bot) Bot.bot
        kstar : ∀ (a : β), Eq (f (KStar.kstar a)) (KStar.kstar (f a))
        a : β
        ⊢ LE.le 1 (KStar.kstar (f a))
      -/
      exact one_le_kstar
      /-
        🎉 no goals
      -/
    mul_kstar_le_kstar := fun a ↦ by
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        π : ι → Type u_4
        inst✝¹⁰ : KleeneAlgebra α
        inst✝⁹ : Zero β
        inst✝⁸ : One β
        inst✝⁷ : Add β
        inst✝⁶ : Mul β
        inst✝⁵ : Pow β Nat
        inst✝⁴ : SMul Nat β
        inst✝³ : NatCast β
        inst✝² : Max β
        inst✝¹ : Bot β
        inst✝ : KStar β
        f : β → α
        hf : Function.Injective f
        zero : Eq (f 0) 0
        one : Eq (f 1) 1
        add : ∀ (x y : β), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
        mul : ∀ (x y : β), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
        nsmul : ∀ (n : Nat) (x : β), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
        npow : ∀ (x : β) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
        natCast : ∀ (n : Nat), Eq (f ↑n) ↑n
        sup : ∀ (a b : β), Eq (f (Max.max a b)) (Max.max (f a) (f b))
        bot : Eq (f Bot.bot) Bot.bot
        kstar : ∀ (a : β), Eq (f (KStar.kstar a)) (KStar.kstar (f a))
        a : β
        ⊢ LE.le (HMul.hMul a (KStar.kstar a)) (KStar.kstar a)
      -/
      change f _ ≤ _
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        π : ι → Type u_4
        inst✝¹⁰ : KleeneAlgebra α
        inst✝⁹ : Zero β
        inst✝⁸ : One β
        inst✝⁷ : Add β
        inst✝⁶ : Mul β
        inst✝⁵ : Pow β Nat
        inst✝⁴ : SMul Nat β
        inst✝³ : NatCast β
        inst✝² : Max β
        inst✝¹ : Bot β
        inst✝ : KStar β
        f : β → α
        hf : Function.Injective f
        zero : Eq (f 0) 0
        one : Eq (f 1) 1
        add : ∀ (x y : β), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
        mul : ∀ (x y : β), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
        nsmul : ∀ (n : Nat) (x : β), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
        npow : ∀ (x : β) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
        natCast : ∀ (n : Nat), Eq (f ↑n) ↑n
        sup : ∀ (a b : β), Eq (f (Max.max a b)) (Max.max (f a) (f b))
        bot : Eq (f Bot.bot) Bot.bot
        kstar : ∀ (a : β), Eq (f (KStar.kstar a)) (KStar.kstar (f a))
        a : β
        ⊢ LE.le (f (HMul.hMul a (KStar.kstar a))) (f (KStar.kstar a))
      -/
      rw [mul, kstar]
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        π : ι → Type u_4
        inst✝¹⁰ : KleeneAlgebra α
        inst✝⁹ : Zero β
        inst✝⁸ : One β
        inst✝⁷ : Add β
        inst✝⁶ : Mul β
        inst✝⁵ : Pow β Nat
        inst✝⁴ : SMul Nat β
        inst✝³ : NatCast β
        inst✝² : Max β
        inst✝¹ : Bot β
        inst✝ : KStar β
        f : β → α
        hf : Function.Injective f
        zero : Eq (f 0) 0
        one : Eq (f 1) 1
        add : ∀ (x y : β), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
        mul : ∀ (x y : β), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
        nsmul : ∀ (n : Nat) (x : β), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
        npow : ∀ (x : β) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
        natCast : ∀ (n : Nat), Eq (f ↑n) ↑n
        sup : ∀ (a b : β), Eq (f (Max.max a b)) (Max.max (f a) (f b))
        bot : Eq (f Bot.bot) Bot.bot
        kstar : ∀ (a : β), Eq (f (KStar.kstar a)) (KStar.kstar (f a))
        a : β
        ⊢ LE.le (HMul.hMul (f a) (KStar.kstar (f a))) (KStar.kstar (f a))
      -/
      exact mul_kstar_le_kstar
      /-
        🎉 no goals
      -/
    kstar_mul_le_kstar := fun a ↦ by
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        π : ι → Type u_4
        inst✝¹⁰ : KleeneAlgebra α
        inst✝⁹ : Zero β
        inst✝⁸ : One β
        inst✝⁷ : Add β
        inst✝⁶ : Mul β
        inst✝⁵ : Pow β Nat
        inst✝⁴ : SMul Nat β
        inst✝³ : NatCast β
        inst✝² : Max β
        inst✝¹ : Bot β
        inst✝ : KStar β
        f : β → α
        hf : Function.Injective f
        zero : Eq (f 0) 0
        one : Eq (f 1) 1
        add : ∀ (x y : β), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
        mul : ∀ (x y : β), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
        nsmul : ∀ (n : Nat) (x : β), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
        npow : ∀ (x : β) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
        natCast : ∀ (n : Nat), Eq (f ↑n) ↑n
        sup : ∀ (a b : β), Eq (f (Max.max a b)) (Max.max (f a) (f b))
        bot : Eq (f Bot.bot) Bot.bot
        kstar : ∀ (a : β), Eq (f (KStar.kstar a)) (KStar.kstar (f a))
        a : β
        ⊢ LE.le (HMul.hMul (KStar.kstar a) a) (KStar.kstar a)
      -/
      change f _ ≤ _
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        π : ι → Type u_4
        inst✝¹⁰ : KleeneAlgebra α
        inst✝⁹ : Zero β
        inst✝⁸ : One β
        inst✝⁷ : Add β
        inst✝⁶ : Mul β
        inst✝⁵ : Pow β Nat
        inst✝⁴ : SMul Nat β
        inst✝³ : NatCast β
        inst✝² : Max β
        inst✝¹ : Bot β
        inst✝ : KStar β
        f : β → α
        hf : Function.Injective f
        zero : Eq (f 0) 0
        one : Eq (f 1) 1
        add : ∀ (x y : β), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
        mul : ∀ (x y : β), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
        nsmul : ∀ (n : Nat) (x : β), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
        npow : ∀ (x : β) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
        natCast : ∀ (n : Nat), Eq (f ↑n) ↑n
        sup : ∀ (a b : β), Eq (f (Max.max a b)) (Max.max (f a) (f b))
        bot : Eq (f Bot.bot) Bot.bot
        kstar : ∀ (a : β), Eq (f (KStar.kstar a)) (KStar.kstar (f a))
        a : β
        ⊢ LE.le (f (HMul.hMul (KStar.kstar a) a)) (f (KStar.kstar a))
      -/
      rw [mul, kstar]
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        π : ι → Type u_4
        inst✝¹⁰ : KleeneAlgebra α
        inst✝⁹ : Zero β
        inst✝⁸ : One β
        inst✝⁷ : Add β
        inst✝⁶ : Mul β
        inst✝⁵ : Pow β Nat
        inst✝⁴ : SMul Nat β
        inst✝³ : NatCast β
        inst✝² : Max β
        inst✝¹ : Bot β
        inst✝ : KStar β
        f : β → α
        hf : Function.Injective f
        zero : Eq (f 0) 0
        one : Eq (f 1) 1
        add : ∀ (x y : β), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
        mul : ∀ (x y : β), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
        nsmul : ∀ (n : Nat) (x : β), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
        npow : ∀ (x : β) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
        natCast : ∀ (n : Nat), Eq (f ↑n) ↑n
        sup : ∀ (a b : β), Eq (f (Max.max a b)) (Max.max (f a) (f b))
        bot : Eq (f Bot.bot) Bot.bot
        kstar : ∀ (a : β), Eq (f (KStar.kstar a)) (KStar.kstar (f a))
        a : β
        ⊢ LE.le (HMul.hMul (KStar.kstar (f a)) (f a)) (KStar.kstar (f a))
      -/
      exact kstar_mul_le_kstar
      /-
        🎉 no goals
      -/
    mul_kstar_le_self := fun a b (h : f _ ≤ _) ↦ by
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        π : ι → Type u_4
        inst✝¹⁰ : KleeneAlgebra α
        inst✝⁹ : Zero β
        inst✝⁸ : One β
        inst✝⁷ : Add β
        inst✝⁶ : Mul β
        inst✝⁵ : Pow β Nat
        inst✝⁴ : SMul Nat β
        inst✝³ : NatCast β
        inst✝² : Max β
        inst✝¹ : Bot β
        inst✝ : KStar β
        f : β → α
        hf : Function.Injective f
        zero : Eq (f 0) 0
        one : Eq (f 1) 1
        add : ∀ (x y : β), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
        mul : ∀ (x y : β), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
        nsmul : ∀ (n : Nat) (x : β), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
        npow : ∀ (x : β) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
        natCast : ∀ (n : Nat), Eq (f ↑n) ↑n
        sup : ∀ (a b : β), Eq (f (Max.max a b)) (Max.max (f a) (f b))
        bot : Eq (f Bot.bot) Bot.bot
        kstar : ∀ (a : β), Eq (f (KStar.kstar a)) (KStar.kstar (f a))
        a b : β
        h : LE.le (f (HMul.hMul b a)) (f b)
        ⊢ LE.le (HMul.hMul b (KStar.kstar a)) b
      -/
      change f _ ≤ _
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        π : ι → Type u_4
        inst✝¹⁰ : KleeneAlgebra α
        inst✝⁹ : Zero β
        inst✝⁸ : One β
        inst✝⁷ : Add β
        inst✝⁶ : Mul β
        inst✝⁵ : Pow β Nat
        inst✝⁴ : SMul Nat β
        inst✝³ : NatCast β
        inst✝² : Max β
        inst✝¹ : Bot β
        inst✝ : KStar β
        f : β → α
        hf : Function.Injective f
        zero : Eq (f 0) 0
        one : Eq (f 1) 1
        add : ∀ (x y : β), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
        mul : ∀ (x y : β), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
        nsmul : ∀ (n : Nat) (x : β), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
        npow : ∀ (x : β) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
        natCast : ∀ (n : Nat), Eq (f ↑n) ↑n
        sup : ∀ (a b : β), Eq (f (Max.max a b)) (Max.max (f a) (f b))
        bot : Eq (f Bot.bot) Bot.bot
        kstar : ∀ (a : β), Eq (f (KStar.kstar a)) (KStar.kstar (f a))
        a b : β
        h : LE.le (f (HMul.hMul b a)) (f b)
        ⊢ LE.le (f (HMul.hMul b (KStar.kstar a))) (f b)
      -/
      rw [mul, kstar]
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        π : ι → Type u_4
        inst✝¹⁰ : KleeneAlgebra α
        inst✝⁹ : Zero β
        inst✝⁸ : One β
        inst✝⁷ : Add β
        inst✝⁶ : Mul β
        inst✝⁵ : Pow β Nat
        inst✝⁴ : SMul Nat β
        inst✝³ : NatCast β
        inst✝² : Max β
        inst✝¹ : Bot β
        inst✝ : KStar β
        f : β → α
        hf : Function.Injective f
        zero : Eq (f 0) 0
        one : Eq (f 1) 1
        add : ∀ (x y : β), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
        mul : ∀ (x y : β), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
        nsmul : ∀ (n : Nat) (x : β), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
        npow : ∀ (x : β) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
        natCast : ∀ (n : Nat), Eq (f ↑n) ↑n
        sup : ∀ (a b : β), Eq (f (Max.max a b)) (Max.max (f a) (f b))
        bot : Eq (f Bot.bot) Bot.bot
        kstar : ∀ (a : β), Eq (f (KStar.kstar a)) (KStar.kstar (f a))
        a b : β
        h : LE.le (f (HMul.hMul b a)) (f b)
        ⊢ LE.le (HMul.hMul (f b) (KStar.kstar (f a))) (f b)
      -/
      rw [mul] at h
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        π : ι → Type u_4
        inst✝¹⁰ : KleeneAlgebra α
        inst✝⁹ : Zero β
        inst✝⁸ : One β
        inst✝⁷ : Add β
        inst✝⁶ : Mul β
        inst✝⁵ : Pow β Nat
        inst✝⁴ : SMul Nat β
        inst✝³ : NatCast β
        inst✝² : Max β
        inst✝¹ : Bot β
        inst✝ : KStar β
        f : β → α
        hf : Function.Injective f
        zero : Eq (f 0) 0
        one : Eq (f 1) 1
        add : ∀ (x y : β), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
        mul : ∀ (x y : β), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
        nsmul : ∀ (n : Nat) (x : β), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
        npow : ∀ (x : β) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
        natCast : ∀ (n : Nat), Eq (f ↑n) ↑n
        sup : ∀ (a b : β), Eq (f (Max.max a b)) (Max.max (f a) (f b))
        bot : Eq (f Bot.bot) Bot.bot
        kstar : ∀ (a : β), Eq (f (KStar.kstar a)) (KStar.kstar (f a))
        a b : β
        h : LE.le (HMul.hMul (f b) (f a)) (f b)
        ⊢ LE.le (HMul.hMul (f b) (KStar.kstar (f a))) (f b)
      -/
      exact mul_kstar_le_self h
      /-
        🎉 no goals
      -/
    kstar_mul_le_self := fun a b (h : f _ ≤ _) ↦ by
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        π : ι → Type u_4
        inst✝¹⁰ : KleeneAlgebra α
        inst✝⁹ : Zero β
        inst✝⁸ : One β
        inst✝⁷ : Add β
        inst✝⁶ : Mul β
        inst✝⁵ : Pow β Nat
        inst✝⁴ : SMul Nat β
        inst✝³ : NatCast β
        inst✝² : Max β
        inst✝¹ : Bot β
        inst✝ : KStar β
        f : β → α
        hf : Function.Injective f
        zero : Eq (f 0) 0
        one : Eq (f 1) 1
        add : ∀ (x y : β), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
        mul : ∀ (x y : β), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
        nsmul : ∀ (n : Nat) (x : β), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
        npow : ∀ (x : β) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
        natCast : ∀ (n : Nat), Eq (f ↑n) ↑n
        sup : ∀ (a b : β), Eq (f (Max.max a b)) (Max.max (f a) (f b))
        bot : Eq (f Bot.bot) Bot.bot
        kstar : ∀ (a : β), Eq (f (KStar.kstar a)) (KStar.kstar (f a))
        a b : β
        h : LE.le (f (HMul.hMul a b)) (f b)
        ⊢ LE.le (HMul.hMul (KStar.kstar a) b) b
      -/
      change f _ ≤ _
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        π : ι → Type u_4
        inst✝¹⁰ : KleeneAlgebra α
        inst✝⁹ : Zero β
        inst✝⁸ : One β
        inst✝⁷ : Add β
        inst✝⁶ : Mul β
        inst✝⁵ : Pow β Nat
        inst✝⁴ : SMul Nat β
        inst✝³ : NatCast β
        inst✝² : Max β
        inst✝¹ : Bot β
        inst✝ : KStar β
        f : β → α
        hf : Function.Injective f
        zero : Eq (f 0) 0
        one : Eq (f 1) 1
        add : ∀ (x y : β), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
        mul : ∀ (x y : β), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
        nsmul : ∀ (n : Nat) (x : β), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
        npow : ∀ (x : β) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
        natCast : ∀ (n : Nat), Eq (f ↑n) ↑n
        sup : ∀ (a b : β), Eq (f (Max.max a b)) (Max.max (f a) (f b))
        bot : Eq (f Bot.bot) Bot.bot
        kstar : ∀ (a : β), Eq (f (KStar.kstar a)) (KStar.kstar (f a))
        a b : β
        h : LE.le (f (HMul.hMul a b)) (f b)
        ⊢ LE.le (f (HMul.hMul (KStar.kstar a) b)) (f b)
      -/
      rw [mul, kstar]
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        π : ι → Type u_4
        inst✝¹⁰ : KleeneAlgebra α
        inst✝⁹ : Zero β
        inst✝⁸ : One β
        inst✝⁷ : Add β
        inst✝⁶ : Mul β
        inst✝⁵ : Pow β Nat
        inst✝⁴ : SMul Nat β
        inst✝³ : NatCast β
        inst✝² : Max β
        inst✝¹ : Bot β
        inst✝ : KStar β
        f : β → α
        hf : Function.Injective f
        zero : Eq (f 0) 0
        one : Eq (f 1) 1
        add : ∀ (x y : β), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
        mul : ∀ (x y : β), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
        nsmul : ∀ (n : Nat) (x : β), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
        npow : ∀ (x : β) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
        natCast : ∀ (n : Nat), Eq (f ↑n) ↑n
        sup : ∀ (a b : β), Eq (f (Max.max a b)) (Max.max (f a) (f b))
        bot : Eq (f Bot.bot) Bot.bot
        kstar : ∀ (a : β), Eq (f (KStar.kstar a)) (KStar.kstar (f a))
        a b : β
        h : LE.le (f (HMul.hMul a b)) (f b)
        ⊢ LE.le (HMul.hMul (KStar.kstar (f a)) (f b)) (f b)
      -/
      rw [mul] at h
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        π : ι → Type u_4
        inst✝¹⁰ : KleeneAlgebra α
        inst✝⁹ : Zero β
        inst✝⁸ : One β
        inst✝⁷ : Add β
        inst✝⁶ : Mul β
        inst✝⁵ : Pow β Nat
        inst✝⁴ : SMul Nat β
        inst✝³ : NatCast β
        inst✝² : Max β
        inst✝¹ : Bot β
        inst✝ : KStar β
        f : β → α
        hf : Function.Injective f
        zero : Eq (f 0) 0
        one : Eq (f 1) 1
        add : ∀ (x y : β), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
        mul : ∀ (x y : β), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
        nsmul : ∀ (n : Nat) (x : β), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
        npow : ∀ (x : β) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
        natCast : ∀ (n : Nat), Eq (f ↑n) ↑n
        sup : ∀ (a b : β), Eq (f (Max.max a b)) (Max.max (f a) (f b))
        bot : Eq (f Bot.bot) Bot.bot
        kstar : ∀ (a : β), Eq (f (KStar.kstar a)) (KStar.kstar (f a))
        a b : β
        h : LE.le (HMul.hMul (f a) (f b)) (f b)
        ⊢ LE.le (HMul.hMul (KStar.kstar (f a)) (f b)) (f b)
      -/
      exact kstar_mul_le_self h }
      /-
        🎉 no goals
      -/


