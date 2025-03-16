/-- See if the term is `a ⊂ b` and the goal is `a ⊆ b`. -/
@[gcongr_forward] def exactSubsetOfSSubset : Mathlib.Tactic.GCongr.ForwardExt where
  eval h goal := do goal.assignIfDefeq (← Lean.Meta.mkAppM ``subset_of_ssubset #[h])


/-- A `SemilatticeSup` is a join-semilattice, that is, a partial order
  with a join (a.k.a. lub / least upper bound, sup / supremum) operation
  `⊔` which is the least element larger than both factors. -/
class SemilatticeSup (α : Type u) extends PartialOrder α where
  /-- The binary supremum, used to derive `Max α` -/
  sup : α → α → α
  /-- The supremum is an upper bound on the first argument -/
  protected le_sup_left : ∀ a b : α, a ≤ sup a b
  /-- The supremum is an upper bound on the second argument -/
  protected le_sup_right : ∀ a b : α, b ≤ sup a b
  /-- The supremum is the *least* upper bound -/
  protected sup_le : ∀ a b c : α, a ≤ c → b ≤ c → sup a b ≤ c


instance SemilatticeSup.toMax [SemilatticeSup α] : Max α where max a b := SemilatticeSup.sup a b


/--
A type with a commutative, associative and idempotent binary `sup` operation has the structure of a
join-semilattice.

The partial order is defined so that `a ≤ b` unfolds to `a ⊔ b = b`; cf. `sup_eq_right`.
-/
def SemilatticeSup.mk' {α : Type*} [Max α] (sup_comm : ∀ a b : α, a ⊔ b = b ⊔ a)
    (sup_assoc : ∀ a b c : α, a ⊔ b ⊔ c = a ⊔ (b ⊔ c)) (sup_idem : ∀ a : α, a ⊔ a = a) :
    SemilatticeSup α where
  sup := (· ⊔ ·)
  le a b := a ⊔ b = b
  le_refl := sup_idem
                               /-
                                 α✝ : Type u
                                 β : Type v
                                 α : Type u_1
                                 inst✝ : Max α
                                 sup_comm : ∀ (a b : α), Eq (Max.max a b) (Max.max b a)
                                 sup_assoc : ∀ (a b c : α), Eq (Max.max (Max.max a b) c) (Max.max a (Max.max b  …
                                 sup_idem : ∀ (a : α), Eq (Max.max a a) a
                                 a b c : α
                                 hab : LE.le a b
                                 hbc : LE.le b c
                                 ⊢ LE.le a c
                               -/
  le_trans a b c hab hbc := by dsimp; rw [← hbc, ← sup_assoc, hab]
                                      /-
                                        🎉 no goals
                                      -/
                                /-
                                  α✝ : Type u
                                  β : Type v
                                  α : Type u_1
                                  inst✝ : Max α
                                  sup_comm : ∀ (a b : α), Eq (Max.max a b) (Max.max b a)
                                  sup_assoc : ∀ (a b c : α), Eq (Max.max (Max.max a b) c) (Max.max a (Max.max b  …
                                  sup_idem : ∀ (a : α), Eq (Max.max a a) a
                                  a b : α
                                  hab : LE.le a b
                                  hba : LE.le b a
                                  ⊢ Eq a b
                                -/
  le_antisymm a b hab hba := by rwa [← hba, sup_comm]
                                /-
                                  🎉 no goals
                                -/
                        /-
                          α✝ : Type u
                          β : Type v
                          α : Type u_1
                          inst✝ : Max α
                          sup_comm : ∀ (a b : α), Eq (Max.max a b) (Max.max b a)
                          sup_assoc : ∀ (a b c : α), Eq (Max.max (Max.max a b) c) (Max.max a (Max.max b  …
                          sup_idem : ∀ (a : α), Eq (Max.max a a) a
                          a b : α
                          ⊢ LE.le a ((fun x1 x2 => Max.max x1 x2) a b)
                        -/
  le_sup_left a b := by dsimp; rw [← sup_assoc, sup_idem]
                               /-
                                 🎉 no goals
                               -/
                         /-
                           α✝ : Type u
                           β : Type v
                           α : Type u_1
                           inst✝ : Max α
                           sup_comm : ∀ (a b : α), Eq (Max.max a b) (Max.max b a)
                           sup_assoc : ∀ (a b c : α), Eq (Max.max (Max.max a b) c) (Max.max a (Max.max b  …
                           sup_idem : ∀ (a : α), Eq (Max.max a a) a
                           a b : α
                           ⊢ LE.le b ((fun x1 x2 => Max.max x1 x2) a b)
                         -/
  le_sup_right a b := by dsimp; rw [sup_comm, sup_assoc, sup_idem]
                                /-
                                  🎉 no goals
                                -/
                             /-
                               α✝ : Type u
                               β : Type v
                               α : Type u_1
                               inst✝ : Max α
                               sup_comm : ∀ (a b : α), Eq (Max.max a b) (Max.max b a)
                               sup_assoc : ∀ (a b c : α), Eq (Max.max (Max.max a b) c) (Max.max a (Max.max b  …
                               sup_idem : ∀ (a : α), Eq (Max.max a a) a
                               a b c : α
                               hac : LE.le a c
                               hbc : LE.le b c
                               ⊢ LE.le ((fun x1 x2 => Max.max x1 x2) a b) c
                             -/
  sup_le a b c hac hbc := by dsimp; rwa [sup_assoc, hbc]
                                    /-
                                      🎉 no goals
                                    -/


@[simp]
theorem le_sup_left : a ≤ a ⊔ b :=
  SemilatticeSup.le_sup_left a b


@[deprecated (since := "2024-06-04")] alias le_sup_left' := le_sup_left


@[simp]
theorem le_sup_right : b ≤ a ⊔ b :=
  SemilatticeSup.le_sup_right a b


@[deprecated (since := "2024-06-04")] alias le_sup_right' := le_sup_right


theorem le_sup_of_le_left (h : c ≤ a) : c ≤ a ⊔ b :=
  le_trans h le_sup_left


theorem le_sup_of_le_right (h : c ≤ b) : c ≤ a ⊔ b :=
  le_trans h le_sup_right


theorem lt_sup_of_lt_left (h : c < a) : c < a ⊔ b :=
  h.trans_le le_sup_left


theorem lt_sup_of_lt_right (h : c < b) : c < a ⊔ b :=
  h.trans_le le_sup_right


theorem sup_le : a ≤ c → b ≤ c → a ⊔ b ≤ c :=
  SemilatticeSup.sup_le a b c


@[simp]
theorem sup_le_iff : a ⊔ b ≤ c ↔ a ≤ c ∧ b ≤ c :=
  ⟨fun h : a ⊔ b ≤ c => ⟨le_trans le_sup_left h, le_trans le_sup_right h⟩,
   fun ⟨h₁, h₂⟩ => sup_le h₁ h₂⟩


@[simp]
theorem sup_eq_left : a ⊔ b = a ↔ b ≤ a :=
                              /-
                                α : Type u
                                inst✝ : SemilatticeSup α
                                a b : α
                                ⊢ Iff (And (LE.le (Max.max a b) a) (LE.le a (Max.max a b))) (LE.le b a)
                              -/
  le_antisymm_iff.trans <| by simp [le_rfl]
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem sup_eq_right : a ⊔ b = b ↔ a ≤ b :=
                              /-
                                α : Type u
                                inst✝ : SemilatticeSup α
                                a b : α
                                ⊢ Iff (And (LE.le (Max.max a b) b) (LE.le b (Max.max a b))) (LE.le a b)
                              -/
  le_antisymm_iff.trans <| by simp [le_rfl]
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem left_eq_sup : a = a ⊔ b ↔ b ≤ a :=
  eq_comm.trans sup_eq_left


@[simp]
theorem right_eq_sup : b = a ⊔ b ↔ a ≤ b :=
  eq_comm.trans sup_eq_right


alias ⟨_, sup_of_le_left⟩ := sup_eq_left


alias ⟨le_of_sup_eq, sup_of_le_right⟩ := sup_eq_right


@[simp]
theorem left_lt_sup : a < a ⊔ b ↔ ¬b ≤ a :=
  le_sup_left.lt_iff_ne.trans <| not_congr left_eq_sup


@[simp]
theorem right_lt_sup : b < a ⊔ b ↔ ¬a ≤ b :=
  le_sup_right.lt_iff_ne.trans <| not_congr right_eq_sup


theorem left_or_right_lt_sup (h : a ≠ b) : a < a ⊔ b ∨ b < a ⊔ b :=
  h.not_le_or_not_le.symm.imp left_lt_sup.2 right_lt_sup.2


theorem le_iff_exists_sup : a ≤ b ↔ ∃ c, b = a ⊔ c := by
  /-
    α : Type u
    inst✝ : SemilatticeSup α
    a b : α
    ⊢ Iff (LE.le a b) (Exists fun c => Eq b (Max.max a c))
  -/
  constructor
    /-
      case mp
      α : Type u
      inst✝ : SemilatticeSup α
      a b : α
      ⊢ LE.le a b → Exists fun c => Eq b (Max.max a c)
    -/
  · intro h
    /-
      case mp
      α : Type u
      inst✝ : SemilatticeSup α
      a b : α
      h : LE.le a b
      ⊢ Exists fun c => Eq b (Max.max a c)
    -/
    exact ⟨b, (sup_eq_right.mpr h).symm⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u
      inst✝ : SemilatticeSup α
      a b : α
      ⊢ (Exists fun c => Eq b (Max.max a c)) → LE.le a b
    -/
  · rintro ⟨c, rfl : _ = _ ⊔ _⟩
    /-
      case mpr.intro
      α : Type u
      inst✝ : SemilatticeSup α
      a c : α
      ⊢ LE.le a (Max.max a c)
    -/
    exact le_sup_left
    /-
      🎉 no goals
    -/


@[gcongr]
theorem sup_le_sup (h₁ : a ≤ b) (h₂ : c ≤ d) : a ⊔ c ≤ b ⊔ d :=
  sup_le (le_sup_of_le_left h₁) (le_sup_of_le_right h₂)


@[gcongr]
theorem sup_le_sup_left (h₁ : a ≤ b) (c) : c ⊔ a ≤ c ⊔ b :=
  sup_le_sup le_rfl h₁


@[gcongr]
theorem sup_le_sup_right (h₁ : a ≤ b) (c) : a ⊔ c ≤ b ⊔ c :=
  sup_le_sup h₁ le_rfl


                                           /-
                                             α : Type u
                                             inst✝ : SemilatticeSup α
                                             a : α
                                             ⊢ Eq (Max.max a a) a
                                           -/
theorem sup_idem (a : α) : a ⊔ a = a := by simp
                                           /-
                                             🎉 no goals
                                           -/


instance : Std.IdempotentOp (α := α) (· ⊔ ·) := ⟨sup_idem⟩


                                                 /-
                                                   α : Type u
                                                   inst✝ : SemilatticeSup α
                                                   a b : α
                                                   ⊢ Eq (Max.max a b) (Max.max b a)
                                                 -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
theorem sup_comm (a b : α) : a ⊔ b = b ⊔ a := by apply le_antisymm <;> simp
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


instance : Std.Commutative (α := α) (· ⊔ ·) := ⟨sup_comm⟩


theorem sup_assoc (a b c : α) : a ⊔ b ⊔ c = a ⊔ (b ⊔ c) :=
                                  /-
                                    α : Type u
                                    inst✝ : SemilatticeSup α
                                    a b c x : α
                                    ⊢ Iff (LE.le (Max.max (Max.max a b) c) x) (LE.le (Max.max a (Max.max b c)) x)
                                  -/
  eq_of_forall_ge_iff fun x => by simp only [sup_le_iff]; rw [and_assoc]
                                                          /-
                                                            🎉 no goals
                                                          -/


instance : Std.Associative (α := α) (· ⊔ ·) := ⟨sup_assoc⟩


theorem sup_left_right_swap (a b c : α) : a ⊔ b ⊔ c = c ⊔ b ⊔ a := by
  /-
    α : Type u
    inst✝ : SemilatticeSup α
    a b c : α
    ⊢ Eq (Max.max (Max.max a b) c) (Max.max (Max.max c b) a)
  -/
  rw [sup_comm, sup_comm a, sup_assoc]
  /-
    🎉 no goals
  -/


                                                            /-
                                                              α : Type u
                                                              inst✝ : SemilatticeSup α
                                                              a b : α
                                                              ⊢ Eq (Max.max a (Max.max a b)) (Max.max a b)
                                                            -/
theorem sup_left_idem (a b : α) : a ⊔ (a ⊔ b) = a ⊔ b := by simp
                                                            /-
                                                              🎉 no goals
                                                            -/


                                                           /-
                                                             α : Type u
                                                             inst✝ : SemilatticeSup α
                                                             a b : α
                                                             ⊢ Eq (Max.max (Max.max a b) b) (Max.max a b)
                                                           -/
theorem sup_right_idem (a b : α) : a ⊔ b ⊔ b = a ⊔ b := by simp
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem sup_left_comm (a b c : α) : a ⊔ (b ⊔ c) = b ⊔ (a ⊔ c) := by
  /-
    α : Type u
    inst✝ : SemilatticeSup α
    a b c : α
    ⊢ Eq (Max.max a (Max.max b c)) (Max.max b (Max.max a c))
  -/
  rw [← sup_assoc, ← sup_assoc, @sup_comm α _ a]
  /-
    🎉 no goals
  -/


theorem sup_right_comm (a b c : α) : a ⊔ b ⊔ c = a ⊔ c ⊔ b := by
  /-
    α : Type u
    inst✝ : SemilatticeSup α
    a b c : α
    ⊢ Eq (Max.max (Max.max a b) c) (Max.max (Max.max a c) b)
  -/
  rw [sup_assoc, sup_assoc, sup_comm b]
  /-
    🎉 no goals
  -/


theorem sup_sup_sup_comm (a b c d : α) : a ⊔ b ⊔ (c ⊔ d) = a ⊔ c ⊔ (b ⊔ d) := by
  /-
    α : Type u
    inst✝ : SemilatticeSup α
    a b c d : α
    ⊢ Eq (Max.max (Max.max a b) (Max.max c d)) (Max.max (Max.max a c) (Max.max b d))
  -/
  rw [sup_assoc, sup_left_comm b, ← sup_assoc]
  /-
    🎉 no goals
  -/


theorem sup_sup_distrib_left (a b c : α) : a ⊔ (b ⊔ c) = a ⊔ b ⊔ (a ⊔ c) := by
  /-
    α : Type u
    inst✝ : SemilatticeSup α
    a b c : α
    ⊢ Eq (Max.max a (Max.max b c)) (Max.max (Max.max a b) (Max.max a c))
  -/
  rw [sup_sup_sup_comm, sup_idem]
  /-
    🎉 no goals
  -/


theorem sup_sup_distrib_right (a b c : α) : a ⊔ b ⊔ c = a ⊔ c ⊔ (b ⊔ c) := by
  /-
    α : Type u
    inst✝ : SemilatticeSup α
    a b c : α
    ⊢ Eq (Max.max (Max.max a b) c) (Max.max (Max.max a c) (Max.max b c))
  -/
  rw [sup_sup_sup_comm, sup_idem]
  /-
    🎉 no goals
  -/


theorem sup_congr_left (hb : b ≤ a ⊔ c) (hc : c ≤ a ⊔ b) : a ⊔ b = a ⊔ c :=
  (sup_le le_sup_left hb).antisymm <| sup_le le_sup_left hc


theorem sup_congr_right (ha : a ≤ b ⊔ c) (hb : b ≤ a ⊔ c) : a ⊔ c = b ⊔ c :=
  (sup_le ha le_sup_right).antisymm <| sup_le hb le_sup_right


theorem sup_eq_sup_iff_left : a ⊔ b = a ⊔ c ↔ b ≤ a ⊔ c ∧ c ≤ a ⊔ b :=
  ⟨fun h => ⟨h ▸ le_sup_right, h.symm ▸ le_sup_right⟩, fun h => sup_congr_left h.1 h.2⟩


theorem sup_eq_sup_iff_right : a ⊔ c = b ⊔ c ↔ a ≤ b ⊔ c ∧ b ≤ a ⊔ c :=
  ⟨fun h => ⟨h ▸ le_sup_left, h.symm ▸ le_sup_left⟩, fun h => sup_congr_right h.1 h.2⟩


theorem Ne.lt_sup_or_lt_sup (hab : a ≠ b) : a < a ⊔ b ∨ b < a ⊔ b :=
  hab.symm.not_le_or_not_le.imp left_lt_sup.2 right_lt_sup.2


/-- If `f` is monotone, `g` is antitone, and `f ≤ g`, then for all `a`, `b` we have `f a ≤ g b`. -/
theorem Monotone.forall_le_of_antitone {β : Type*} [Preorder β] {f g : α → β} (hf : Monotone f)
    (hg : Antitone g) (h : f ≤ g) (m n : α) : f m ≤ g n :=
  calc
    f m ≤ f (m ⊔ n) := hf le_sup_left
    _ ≤ g (m ⊔ n) := h _
    _ ≤ g n := hg le_sup_right


theorem SemilatticeSup.ext_sup {α} {A B : SemilatticeSup α}
    (H : ∀ x y : α, (haveI := A; x ≤ y) ↔ x ≤ y)
    (x y : α) :
    (haveI := A; x ⊔ y) = x ⊔ y :=
                                  /-
                                    α : Type u_1
                                    A B : SemilatticeSup α
                                    H : ∀ (x y : α), Iff (LE.le x y) (LE.le x y)
                                    x y c : α
                                    ⊢ Iff (LE.le (Max.max x y) c) (LE.le (Max.max x y) c)
                                  -/
  eq_of_forall_ge_iff fun c => by simp only [sup_le_iff]; rw [← H, @sup_le_iff α A, H, H]
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem SemilatticeSup.ext {α} {A B : SemilatticeSup α}
    (H : ∀ x y : α, (haveI := A; x ≤ y) ↔ x ≤ y) :
    A = B := by
  /-
    α : Type u_1
    A B : SemilatticeSup α
    H : ∀ (x y : α), Iff (LE.le x y) (LE.le x y)
    ⊢ Eq A B
  -/
  cases A
  /-
    case mk
    α : Type u_1
    B : SemilatticeSup α
    toPartialOrder✝ : PartialOrder α
    sup✝ : α → α → α
    le_sup_left✝ : ∀ (a b : α), LE.le a (sup✝ a b)
    le_sup_right✝ : ∀ (a b : α), LE.le b (sup✝ a b)
    sup_le✝ : ∀ (a b c : α), LE.le a c → LE.le b c → LE.le (sup✝ a b) c
    H : ∀ (x y : α), Iff (LE.le x y) (LE.le x y)
    ⊢ Eq (SemilatticeSup.mk sup✝ le_sup_left✝ le_sup_right✝ sup_le✝) B
  -/
  cases B
  /-
    case mk.mk
    α : Type u_1
    toPartialOrder✝¹ : PartialOrder α
    sup✝¹ : α → α → α
    le_sup_left✝¹ : ∀ (a b : α), LE.le a (sup✝¹ a b)
    le_sup_right✝¹ : ∀ (a b : α), LE.le b (sup✝¹ a b)
    sup_le✝¹ : ∀ (a b c : α), LE.le a c → LE.le b c → LE.le (sup✝¹ a b) c
    toPartialOrder✝ : PartialOrder α
    sup✝ : α → α → α
    le_sup_left✝ : ∀ (a b : α), LE.le a (sup✝ a b)
    le_sup_right✝ : ∀ (a b : α), LE.le b (sup✝ a b)
    sup_le✝ : ∀ (a b c : α), LE.le a c → LE.le b c → LE.le (sup✝ a b) c
    H : ∀ (x y : α), Iff (LE.le x y) (LE.le x y)
    ⊢ Eq (SemilatticeSup.mk sup✝¹ le_sup_left✝¹ le_sup_right✝¹ sup_le✝¹) (Semilatt …
  -/
  cases PartialOrder.ext H
  /-
    case mk.mk.refl
    α : Type u_1
    toPartialOrder✝ : PartialOrder α
    sup✝¹ : α → α → α
    le_sup_left✝¹ : ∀ (a b : α), LE.le a (sup✝¹ a b)
    le_sup_right✝¹ : ∀ (a b : α), LE.le b (sup✝¹ a b)
    sup_le✝¹ : ∀ (a b c : α), LE.le a c → LE.le b c → LE.le (sup✝¹ a b) c
    sup✝ : α → α → α
    le_sup_left✝ : ∀ (a b : α), LE.le a (sup✝ a b)
    le_sup_right✝ : ∀ (a b : α), LE.le b (sup✝ a b)
    sup_le✝ : ∀ (a b c : α), LE.le a c → LE.le b c → LE.le (sup✝ a b) c
    H : ∀ (x y : α), Iff (LE.le x y) (LE.le x y)
    ⊢ Eq (SemilatticeSup.mk sup✝¹ le_sup_left✝¹ le_sup_right✝¹ sup_le✝¹) (Semilatt …
  -/
  congr
  /-
    case mk.mk.refl.e_sup
    α : Type u_1
    toPartialOrder✝ : PartialOrder α
    sup✝¹ : α → α → α
    le_sup_left✝¹ : ∀ (a b : α), LE.le a (sup✝¹ a b)
    le_sup_right✝¹ : ∀ (a b : α), LE.le b (sup✝¹ a b)
    sup_le✝¹ : ∀ (a b c : α), LE.le a c → LE.le b c → LE.le (sup✝¹ a b) c
    sup✝ : α → α → α
    le_sup_left✝ : ∀ (a b : α), LE.le a (sup✝ a b)
    le_sup_right✝ : ∀ (a b : α), LE.le b (sup✝ a b)
    sup_le✝ : ∀ (a b c : α), LE.le a c → LE.le b c → LE.le (sup✝ a b) c
    H : ∀ (x y : α), Iff (LE.le x y) (LE.le x y)
    ⊢ Eq sup✝¹ sup✝
  -/
  ext; apply SemilatticeSup.ext_sup H
       /-
         🎉 no goals
       -/


theorem ite_le_sup (s s' : α) (P : Prop) [Decidable P] : ite P s s' ≤ s ⊔ s' :=
  if h : P then (if_pos h).trans_le le_sup_left else (if_neg h).trans_le le_sup_right


/-- A `SemilatticeInf` is a meet-semilattice, that is, a partial order
  with a meet (a.k.a. glb / greatest lower bound, inf / infimum) operation
  `⊓` which is the greatest element smaller than both factors. -/
class SemilatticeInf (α : Type u) extends PartialOrder α where
  /-- The binary infimum, used to derive `Min α` -/
  inf : α → α → α
  /-- The infimum is a lower bound on the first argument -/
  protected inf_le_left : ∀ a b : α, inf a b ≤ a
  /-- The infimum is a lower bound on the second argument -/
  protected inf_le_right : ∀ a b : α, inf a b ≤ b
  /-- The infimum is the *greatest* lower bound -/
  protected le_inf : ∀ a b c : α, a ≤ b → a ≤ c → a ≤ inf b c


instance SemilatticeInf.toMin [SemilatticeInf α] : Min α where min a b := SemilatticeInf.inf a b


instance OrderDual.instSemilatticeSup (α) [SemilatticeInf α] : SemilatticeSup αᵒᵈ where
  __ := inferInstanceAs (PartialOrder αᵒᵈ)
  __ := inferInstanceAs (Max αᵒᵈ)
  le_sup_left := @SemilatticeInf.inf_le_left α _
  le_sup_right := @SemilatticeInf.inf_le_right α _
  sup_le := fun _ _ _ hca hcb => @SemilatticeInf.le_inf α _ _ _ _ hca hcb


instance OrderDual.instSemilatticeInf (α) [SemilatticeSup α] : SemilatticeInf αᵒᵈ where
  __ := inferInstanceAs (PartialOrder αᵒᵈ)
  __ := inferInstanceAs (Min αᵒᵈ)
  inf_le_left := @le_sup_left α _
  inf_le_right := @le_sup_right α _
  le_inf := fun _ _ _ hca hcb => @sup_le α _ _ _ _ hca hcb


theorem SemilatticeSup.dual_dual (α : Type*) [H : SemilatticeSup α] :
    OrderDual.instSemilatticeSup αᵒᵈ = H :=
  SemilatticeSup.ext fun _ _ => Iff.rfl


@[simp]
theorem inf_le_left : a ⊓ b ≤ a :=
  SemilatticeInf.inf_le_left a b


@[deprecated (since := "2024-06-04")] alias inf_le_left' := inf_le_left


@[simp]
theorem inf_le_right : a ⊓ b ≤ b :=
  SemilatticeInf.inf_le_right a b


@[deprecated (since := "2024-06-04")] alias inf_le_right' := inf_le_right


theorem le_inf : a ≤ b → a ≤ c → a ≤ b ⊓ c :=
  SemilatticeInf.le_inf a b c


theorem inf_le_of_left_le (h : a ≤ c) : a ⊓ b ≤ c :=
  le_trans inf_le_left h


theorem inf_le_of_right_le (h : b ≤ c) : a ⊓ b ≤ c :=
  le_trans inf_le_right h


theorem inf_lt_of_left_lt (h : a < c) : a ⊓ b < c :=
  lt_of_le_of_lt inf_le_left h


theorem inf_lt_of_right_lt (h : b < c) : a ⊓ b < c :=
  lt_of_le_of_lt inf_le_right h


@[simp]
theorem le_inf_iff : a ≤ b ⊓ c ↔ a ≤ b ∧ a ≤ c :=
  @sup_le_iff αᵒᵈ _ _ _ _


@[simp]
theorem inf_eq_left : a ⊓ b = a ↔ a ≤ b :=
                              /-
                                α : Type u
                                inst✝ : SemilatticeInf α
                                a b : α
                                ⊢ Iff (And (LE.le (Min.min a b) a) (LE.le a (Min.min a b))) (LE.le a b)
                              -/
  le_antisymm_iff.trans <| by simp [le_rfl]
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem inf_eq_right : a ⊓ b = b ↔ b ≤ a :=
                              /-
                                α : Type u
                                inst✝ : SemilatticeInf α
                                a b : α
                                ⊢ Iff (And (LE.le (Min.min a b) b) (LE.le b (Min.min a b))) (LE.le b a)
                              -/
  le_antisymm_iff.trans <| by simp [le_rfl]
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem left_eq_inf : a = a ⊓ b ↔ a ≤ b :=
  eq_comm.trans inf_eq_left


@[simp]
theorem right_eq_inf : b = a ⊓ b ↔ b ≤ a :=
  eq_comm.trans inf_eq_right


alias ⟨le_of_inf_eq, inf_of_le_left⟩ := inf_eq_left


alias ⟨_, inf_of_le_right⟩ := inf_eq_right


@[simp]
theorem inf_lt_left : a ⊓ b < a ↔ ¬a ≤ b :=
  @left_lt_sup αᵒᵈ _ _ _


@[simp]
theorem inf_lt_right : a ⊓ b < b ↔ ¬b ≤ a :=
  @right_lt_sup αᵒᵈ _ _ _


theorem inf_lt_left_or_right (h : a ≠ b) : a ⊓ b < a ∨ a ⊓ b < b :=
  @left_or_right_lt_sup αᵒᵈ _ _ _ h


@[gcongr]
theorem inf_le_inf (h₁ : a ≤ b) (h₂ : c ≤ d) : a ⊓ c ≤ b ⊓ d :=
  @sup_le_sup αᵒᵈ _ _ _ _ _ h₁ h₂


@[gcongr]
theorem inf_le_inf_right (a : α) {b c : α} (h : b ≤ c) : b ⊓ a ≤ c ⊓ a :=
  inf_le_inf h le_rfl


@[gcongr]
theorem inf_le_inf_left (a : α) {b c : α} (h : b ≤ c) : a ⊓ b ≤ a ⊓ c :=
  inf_le_inf le_rfl h


                                           /-
                                             α : Type u
                                             inst✝ : SemilatticeInf α
                                             a : α
                                             ⊢ Eq (Min.min a a) a
                                           -/
theorem inf_idem (a : α) : a ⊓ a = a := by simp
                                           /-
                                             🎉 no goals
                                           -/


instance : Std.IdempotentOp (α := α) (· ⊓ ·) := ⟨inf_idem⟩


theorem inf_comm (a b : α) : a ⊓ b = b ⊓ a := @sup_comm αᵒᵈ _ _ _


instance : Std.Commutative (α := α) (· ⊓ ·) := ⟨inf_comm⟩


theorem inf_assoc (a b c : α) : a ⊓ b ⊓ c = a ⊓ (b ⊓ c) := @sup_assoc αᵒᵈ _ _ _ _


instance : Std.Associative (α := α) (· ⊓ ·) := ⟨inf_assoc⟩


theorem inf_left_right_swap (a b c : α) : a ⊓ b ⊓ c = c ⊓ b ⊓ a :=
  @sup_left_right_swap αᵒᵈ _ _ _ _


                                                            /-
                                                              α : Type u
                                                              inst✝ : SemilatticeInf α
                                                              a b : α
                                                              ⊢ Eq (Min.min a (Min.min a b)) (Min.min a b)
                                                            -/
theorem inf_left_idem (a b : α) : a ⊓ (a ⊓ b) = a ⊓ b := by simp
                                                            /-
                                                              🎉 no goals
                                                            -/


                                                           /-
                                                             α : Type u
                                                             inst✝ : SemilatticeInf α
                                                             a b : α
                                                             ⊢ Eq (Min.min (Min.min a b) b) (Min.min a b)
                                                           -/
theorem inf_right_idem (a b : α) : a ⊓ b ⊓ b = a ⊓ b := by simp
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem inf_left_comm (a b c : α) : a ⊓ (b ⊓ c) = b ⊓ (a ⊓ c) :=
  @sup_left_comm αᵒᵈ _ a b c


theorem inf_right_comm (a b c : α) : a ⊓ b ⊓ c = a ⊓ c ⊓ b :=
  @sup_right_comm αᵒᵈ _ a b c


theorem inf_inf_inf_comm (a b c d : α) : a ⊓ b ⊓ (c ⊓ d) = a ⊓ c ⊓ (b ⊓ d) :=
  @sup_sup_sup_comm αᵒᵈ _ _ _ _ _


theorem inf_inf_distrib_left (a b c : α) : a ⊓ (b ⊓ c) = a ⊓ b ⊓ (a ⊓ c) :=
  @sup_sup_distrib_left αᵒᵈ _ _ _ _


theorem inf_inf_distrib_right (a b c : α) : a ⊓ b ⊓ c = a ⊓ c ⊓ (b ⊓ c) :=
  @sup_sup_distrib_right αᵒᵈ _ _ _ _


theorem inf_congr_left (hb : a ⊓ c ≤ b) (hc : a ⊓ b ≤ c) : a ⊓ b = a ⊓ c :=
  @sup_congr_left αᵒᵈ _ _ _ _ hb hc


theorem inf_congr_right (h1 : b ⊓ c ≤ a) (h2 : a ⊓ c ≤ b) : a ⊓ c = b ⊓ c :=
  @sup_congr_right αᵒᵈ _ _ _ _ h1 h2


theorem inf_eq_inf_iff_left : a ⊓ b = a ⊓ c ↔ a ⊓ c ≤ b ∧ a ⊓ b ≤ c :=
  @sup_eq_sup_iff_left αᵒᵈ _ _ _ _


theorem inf_eq_inf_iff_right : a ⊓ c = b ⊓ c ↔ b ⊓ c ≤ a ∧ a ⊓ c ≤ b :=
  @sup_eq_sup_iff_right αᵒᵈ _ _ _ _


theorem Ne.inf_lt_or_inf_lt : a ≠ b → a ⊓ b < a ∨ a ⊓ b < b :=
  @Ne.lt_sup_or_lt_sup αᵒᵈ _ _ _


theorem SemilatticeInf.ext_inf {α} {A B : SemilatticeInf α}
    (H : ∀ x y : α, (haveI := A; x ≤ y) ↔ x ≤ y)
    (x y : α) :
    (haveI := A; x ⊓ y) = x ⊓ y :=
                                  /-
                                    α : Type u_1
                                    A B : SemilatticeInf α
                                    H : ∀ (x y : α), Iff (LE.le x y) (LE.le x y)
                                    x y c : α
                                    ⊢ Iff (LE.le c (Min.min x y)) (LE.le c (Min.min x y))
                                  -/
  eq_of_forall_le_iff fun c => by simp only [le_inf_iff]; rw [← H, @le_inf_iff α A, H, H]
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem SemilatticeInf.ext {α} {A B : SemilatticeInf α}
    (H : ∀ x y : α, (haveI := A; x ≤ y) ↔ x ≤ y) :
    A = B := by
  /-
    α : Type u_1
    A B : SemilatticeInf α
    H : ∀ (x y : α), Iff (LE.le x y) (LE.le x y)
    ⊢ Eq A B
  -/
  cases A
  /-
    case mk
    α : Type u_1
    B : SemilatticeInf α
    toPartialOrder✝ : PartialOrder α
    inf✝ : α → α → α
    inf_le_left✝ : ∀ (a b : α), LE.le (inf✝ a b) a
    inf_le_right✝ : ∀ (a b : α), LE.le (inf✝ a b) b
    le_inf✝ : ∀ (a b c : α), LE.le a b → LE.le a c → LE.le a (inf✝ b c)
    H : ∀ (x y : α), Iff (LE.le x y) (LE.le x y)
    ⊢ Eq (SemilatticeInf.mk inf✝ inf_le_left✝ inf_le_right✝ le_inf✝) B
  -/
  cases B
  /-
    case mk.mk
    α : Type u_1
    toPartialOrder✝¹ : PartialOrder α
    inf✝¹ : α → α → α
    inf_le_left✝¹ : ∀ (a b : α), LE.le (inf✝¹ a b) a
    inf_le_right✝¹ : ∀ (a b : α), LE.le (inf✝¹ a b) b
    le_inf✝¹ : ∀ (a b c : α), LE.le a b → LE.le a c → LE.le a (inf✝¹ b c)
    toPartialOrder✝ : PartialOrder α
    inf✝ : α → α → α
    inf_le_left✝ : ∀ (a b : α), LE.le (inf✝ a b) a
    inf_le_right✝ : ∀ (a b : α), LE.le (inf✝ a b) b
    le_inf✝ : ∀ (a b c : α), LE.le a b → LE.le a c → LE.le a (inf✝ b c)
    H : ∀ (x y : α), Iff (LE.le x y) (LE.le x y)
    ⊢ Eq (SemilatticeInf.mk inf✝¹ inf_le_left✝¹ inf_le_right✝¹ le_inf✝¹) (Semilatt …
  -/
  cases PartialOrder.ext H
  /-
    case mk.mk.refl
    α : Type u_1
    toPartialOrder✝ : PartialOrder α
    inf✝¹ : α → α → α
    inf_le_left✝¹ : ∀ (a b : α), LE.le (inf✝¹ a b) a
    inf_le_right✝¹ : ∀ (a b : α), LE.le (inf✝¹ a b) b
    le_inf✝¹ : ∀ (a b c : α), LE.le a b → LE.le a c → LE.le a (inf✝¹ b c)
    inf✝ : α → α → α
    inf_le_left✝ : ∀ (a b : α), LE.le (inf✝ a b) a
    inf_le_right✝ : ∀ (a b : α), LE.le (inf✝ a b) b
    le_inf✝ : ∀ (a b c : α), LE.le a b → LE.le a c → LE.le a (inf✝ b c)
    H : ∀ (x y : α), Iff (LE.le x y) (LE.le x y)
    ⊢ Eq (SemilatticeInf.mk inf✝¹ inf_le_left✝¹ inf_le_right✝¹ le_inf✝¹) (Semilatt …
  -/
  congr
  /-
    case mk.mk.refl.e_inf
    α : Type u_1
    toPartialOrder✝ : PartialOrder α
    inf✝¹ : α → α → α
    inf_le_left✝¹ : ∀ (a b : α), LE.le (inf✝¹ a b) a
    inf_le_right✝¹ : ∀ (a b : α), LE.le (inf✝¹ a b) b
    le_inf✝¹ : ∀ (a b c : α), LE.le a b → LE.le a c → LE.le a (inf✝¹ b c)
    inf✝ : α → α → α
    inf_le_left✝ : ∀ (a b : α), LE.le (inf✝ a b) a
    inf_le_right✝ : ∀ (a b : α), LE.le (inf✝ a b) b
    le_inf✝ : ∀ (a b c : α), LE.le a b → LE.le a c → LE.le a (inf✝ b c)
    H : ∀ (x y : α), Iff (LE.le x y) (LE.le x y)
    ⊢ Eq inf✝¹ inf✝
  -/
  ext; apply SemilatticeInf.ext_inf H
       /-
         🎉 no goals
       -/


theorem SemilatticeInf.dual_dual (α : Type*) [H : SemilatticeInf α] :
    OrderDual.instSemilatticeInf αᵒᵈ = H :=
  SemilatticeInf.ext fun _ _ => Iff.rfl


theorem inf_le_ite (s s' : α) (P : Prop) [Decidable P] : s ⊓ s' ≤ ite P s s' :=
  @ite_le_sup αᵒᵈ _ _ _ _ _


/--
A type with a commutative, associative and idempotent binary `inf` operation has the structure of a
meet-semilattice.

The partial order is defined so that `a ≤ b` unfolds to `b ⊓ a = a`; cf. `inf_eq_right`.
-/
def SemilatticeInf.mk' {α : Type*} [Min α] (inf_comm : ∀ a b : α, a ⊓ b = b ⊓ a)
    (inf_assoc : ∀ a b c : α, a ⊓ b ⊓ c = a ⊓ (b ⊓ c)) (inf_idem : ∀ a : α, a ⊓ a = a) :
    SemilatticeInf α := by
  /-
    α✝ : Type u
    β : Type v
    α : Type u_1
    inst✝ : Min α
    inf_comm : ∀ (a b : α), Eq (Min.min a b) (Min.min b a)
    inf_assoc : ∀ (a b c : α), Eq (Min.min (Min.min a b) c) (Min.min a (Min.min b  …
    inf_idem : ∀ (a : α), Eq (Min.min a a) a
    ⊢ SemilatticeInf α
  -/
  haveI : SemilatticeSup αᵒᵈ := SemilatticeSup.mk' inf_comm inf_assoc inf_idem
  /-
    α✝ : Type u
    β : Type v
    α : Type u_1
    inst✝ : Min α
    inf_comm : ∀ (a b : α), Eq (Min.min a b) (Min.min b a)
    inf_assoc : ∀ (a b c : α), Eq (Min.min (Min.min a b) c) (Min.min a (Min.min b  …
    inf_idem : ∀ (a : α), Eq (Min.min a a) a
    this : SemilatticeSup (OrderDual α)
    ⊢ SemilatticeInf α
  -/
  haveI i := OrderDual.instSemilatticeInf αᵒᵈ
  /-
    α✝ : Type u
    β : Type v
    α : Type u_1
    inst✝ : Min α
    inf_comm : ∀ (a b : α), Eq (Min.min a b) (Min.min b a)
    inf_assoc : ∀ (a b c : α), Eq (Min.min (Min.min a b) c) (Min.min a (Min.min b  …
    inf_idem : ∀ (a : α), Eq (Min.min a a) a
    this : SemilatticeSup (OrderDual α)
    i : SemilatticeInf (OrderDual (OrderDual α))
    ⊢ SemilatticeInf α
  -/
  exact i
  /-
    🎉 no goals
  -/


/-- A lattice is a join-semilattice which is also a meet-semilattice. -/
class Lattice (α : Type u) extends SemilatticeSup α, SemilatticeInf α


instance OrderDual.instLattice (α) [Lattice α] : Lattice αᵒᵈ where
  __ := OrderDual.instSemilatticeSup α
  __ := OrderDual.instSemilatticeInf α


/-- The partial orders from `SemilatticeSup_mk'` and `SemilatticeInf_mk'` agree
if `sup` and `inf` satisfy the lattice absorption laws `sup_inf_self` (`a ⊔ a ⊓ b = a`)
and `inf_sup_self` (`a ⊓ (a ⊔ b) = a`). -/
theorem semilatticeSup_mk'_partialOrder_eq_semilatticeInf_mk'_partialOrder
    {α : Type*} [Max α] [Min α]
    (sup_comm : ∀ a b : α, a ⊔ b = b ⊔ a) (sup_assoc : ∀ a b c : α, a ⊔ b ⊔ c = a ⊔ (b ⊔ c))
    (sup_idem : ∀ a : α, a ⊔ a = a) (inf_comm : ∀ a b : α, a ⊓ b = b ⊓ a)
    (inf_assoc : ∀ a b c : α, a ⊓ b ⊓ c = a ⊓ (b ⊓ c)) (inf_idem : ∀ a : α, a ⊓ a = a)
    (sup_inf_self : ∀ a b : α, a ⊔ a ⊓ b = a) (inf_sup_self : ∀ a b : α, a ⊓ (a ⊔ b) = a) :
    @SemilatticeSup.toPartialOrder _ (SemilatticeSup.mk' sup_comm sup_assoc sup_idem) =
      @SemilatticeInf.toPartialOrder _ (SemilatticeInf.mk' inf_comm inf_assoc inf_idem) :=
  PartialOrder.ext fun a b =>
    show a ⊔ b = b ↔ b ⊓ a = a from
                   /-
                     α : Type u_1
                     inst✝¹ : Max α
                     inst✝ : Min α
                     sup_comm : ∀ (a b : α), Eq (Max.max a b) (Max.max b a)
                     sup_assoc : ∀ (a b c : α), Eq (Max.max (Max.max a b) c) (Max.max a (Max.max b  …
                     sup_idem : ∀ (a : α), Eq (Max.max a a) a
                     inf_comm : ∀ (a b : α), Eq (Min.min a b) (Min.min b a)
                     inf_assoc : ∀ (a b c : α), Eq (Min.min (Min.min a b) c) (Min.min a (Min.min b  …
                     inf_idem : ∀ (a : α), Eq (Min.min a a) a
                     sup_inf_self : ∀ (a b : α), Eq (Max.max a (Min.min a b)) a
                     inf_sup_self : ∀ (a b : α), Eq (Min.min a (Max.max a b)) a
                     a b : α
                     h : Eq (Max.max a b) b
                     ⊢ Eq (Min.min b a) a
                   -/
                   /-
                     🎉 no goals
                   -/
      ⟨fun h => by rw [← h, inf_comm, inf_sup_self], fun h => by rw [← h, sup_comm, sup_inf_self]⟩
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


/-- A type with a pair of commutative and associative binary operations which satisfy two absorption
laws relating the two operations has the structure of a lattice.

The partial order is defined so that `a ≤ b` unfolds to `a ⊔ b = b`; cf. `sup_eq_right`.
-/
def Lattice.mk' {α : Type*} [Max α] [Min α] (sup_comm : ∀ a b : α, a ⊔ b = b ⊔ a)
    (sup_assoc : ∀ a b c : α, a ⊔ b ⊔ c = a ⊔ (b ⊔ c)) (inf_comm : ∀ a b : α, a ⊓ b = b ⊓ a)
    (inf_assoc : ∀ a b c : α, a ⊓ b ⊓ c = a ⊓ (b ⊓ c)) (sup_inf_self : ∀ a b : α, a ⊔ a ⊓ b = a)
    (inf_sup_self : ∀ a b : α, a ⊓ (a ⊔ b) = a) : Lattice α :=
  have sup_idem : ∀ b : α, b ⊔ b = b := fun b =>
    calc
                                    /-
                                      α✝ : Type u
                                      β : Type v
                                      α : Type u_1
                                      inst✝¹ : Max α
                                      inst✝ : Min α
                                      sup_comm : ∀ (a b : α), Eq (Max.max a b) (Max.max b a)
                                      sup_assoc : ∀ (a b c : α), Eq (Max.max (Max.max a b) c) (Max.max a (Max.max b  …
                                      inf_comm : ∀ (a b : α), Eq (Min.min a b) (Min.min b a)
                                      inf_assoc : ∀ (a b c : α), Eq (Min.min (Min.min a b) c) (Min.min a (Min.min b  …
                                      sup_inf_self : ∀ (a b : α), Eq (Max.max a (Min.min a b)) a
                                      inf_sup_self : ∀ (a b : α), Eq (Min.min a (Max.max a b)) a
                                      b : α
                                      ⊢ Eq (Max.max b b) (Max.max b (Min.min b (Max.max b b)))
                                    -/
      b ⊔ b = b ⊔ b ⊓ (b ⊔ b) := by rw [inf_sup_self]
                                    /-
                                      🎉 no goals
                                    -/
                  /-
                    α✝ : Type u
                    β : Type v
                    α : Type u_1
                    inst✝¹ : Max α
                    inst✝ : Min α
                    sup_comm : ∀ (a b : α), Eq (Max.max a b) (Max.max b a)
                    sup_assoc : ∀ (a b c : α), Eq (Max.max (Max.max a b) c) (Max.max a (Max.max b  …
                    inf_comm : ∀ (a b : α), Eq (Min.min a b) (Min.min b a)
                    inf_assoc : ∀ (a b c : α), Eq (Min.min (Min.min a b) c) (Min.min a (Min.min b  …
                    sup_inf_self : ∀ (a b : α), Eq (Max.max a (Min.min a b)) a
                    inf_sup_self : ∀ (a b : α), Eq (Min.min a (Max.max a b)) a
                    b : α
                    ⊢ Eq (Max.max b (Min.min b (Max.max b b))) b
                  -/
      _ = b := by rw [sup_inf_self]
                  /-
                    🎉 no goals
                  -/

  have inf_idem : ∀ b : α, b ⊓ b = b := fun b =>
    calc
                                    /-
                                      α✝ : Type u
                                      β : Type v
                                      α : Type u_1
                                      inst✝¹ : Max α
                                      inst✝ : Min α
                                      sup_comm : ∀ (a b : α), Eq (Max.max a b) (Max.max b a)
                                      sup_assoc : ∀ (a b c : α), Eq (Max.max (Max.max a b) c) (Max.max a (Max.max b  …
                                      inf_comm : ∀ (a b : α), Eq (Min.min a b) (Min.min b a)
                                      inf_assoc : ∀ (a b c : α), Eq (Min.min (Min.min a b) c) (Min.min a (Min.min b  …
                                      sup_inf_self : ∀ (a b : α), Eq (Max.max a (Min.min a b)) a
                                      inf_sup_self : ∀ (a b : α), Eq (Min.min a (Max.max a b)) a
                                      sup_idem : ∀ (b : α), Eq (Max.max b b) b
                                      b : α
                                      ⊢ Eq (Min.min b b) (Min.min b (Max.max b (Min.min b b)))
                                    -/
      b ⊓ b = b ⊓ (b ⊔ b ⊓ b) := by rw [sup_inf_self]
                                    /-
                                      🎉 no goals
                                    -/
                  /-
                    α✝ : Type u
                    β : Type v
                    α : Type u_1
                    inst✝¹ : Max α
                    inst✝ : Min α
                    sup_comm : ∀ (a b : α), Eq (Max.max a b) (Max.max b a)
                    sup_assoc : ∀ (a b c : α), Eq (Max.max (Max.max a b) c) (Max.max a (Max.max b  …
                    inf_comm : ∀ (a b : α), Eq (Min.min a b) (Min.min b a)
                    inf_assoc : ∀ (a b c : α), Eq (Min.min (Min.min a b) c) (Min.min a (Min.min b  …
                    sup_inf_self : ∀ (a b : α), Eq (Max.max a (Min.min a b)) a
                    inf_sup_self : ∀ (a b : α), Eq (Min.min a (Max.max a b)) a
                    sup_idem : ∀ (b : α), Eq (Max.max b b) b
                    b : α
                    ⊢ Eq (Min.min b (Max.max b (Min.min b b))) b
                  -/
      _ = b := by rw [inf_sup_self]
                  /-
                    🎉 no goals
                  -/

  let semilatt_inf_inst := SemilatticeInf.mk' inf_comm inf_assoc inf_idem
  let semilatt_sup_inst := SemilatticeSup.mk' sup_comm sup_assoc sup_idem
  have partial_order_eq : @SemilatticeSup.toPartialOrder _ semilatt_sup_inst =
                          @SemilatticeInf.toPartialOrder _ semilatt_inf_inst :=
    semilatticeSup_mk'_partialOrder_eq_semilatticeInf_mk'_partialOrder _ _ _ _ _ _
      sup_inf_self inf_sup_self
  { semilatt_sup_inst, semilatt_inf_inst with
    inf_le_left := fun a b => by
      /-
        α✝ : Type u
        β : Type v
        α : Type u_1
        inst✝¹ : Max α
        inst✝ : Min α
        sup_comm : ∀ (a b : α), Eq (Max.max a b) (Max.max b a)
        sup_assoc : ∀ (a b c : α), Eq (Max.max (Max.max a b) c) (Max.max a (Max.max b  …
        inf_comm : ∀ (a b : α), Eq (Min.min a b) (Min.min b a)
        inf_assoc : ∀ (a b c : α), Eq (Min.min (Min.min a b) c) (Min.min a (Min.min b  …
        sup_inf_self : ∀ (a b : α), Eq (Max.max a (Min.min a b)) a
        inf_sup_self : ∀ (a b : α), Eq (Min.min a (Max.max a b)) a
        sup_idem : ∀ (b : α), Eq (Max.max b b) b
        inf_idem : ∀ (b : α), Eq (Min.min b b) b
        semilatt_inf_inst : SemilatticeInf α := SemilatticeInf.mk' inf_comm inf_assoc  …
        semilatt_sup_inst : SemilatticeSup α := SemilatticeSup.mk' sup_comm sup_assoc  …
        partial_order_eq : Eq SemilatticeSup.toPartialOrder SemilatticeInf.toPartialOr …
        a b : α
        ⊢ LE.le (SemilatticeInf.inf a b) a
      -/
      rw [partial_order_eq]
      /-
        α✝ : Type u
        β : Type v
        α : Type u_1
        inst✝¹ : Max α
        inst✝ : Min α
        sup_comm : ∀ (a b : α), Eq (Max.max a b) (Max.max b a)
        sup_assoc : ∀ (a b c : α), Eq (Max.max (Max.max a b) c) (Max.max a (Max.max b  …
        inf_comm : ∀ (a b : α), Eq (Min.min a b) (Min.min b a)
        inf_assoc : ∀ (a b c : α), Eq (Min.min (Min.min a b) c) (Min.min a (Min.min b  …
        sup_inf_self : ∀ (a b : α), Eq (Max.max a (Min.min a b)) a
        inf_sup_self : ∀ (a b : α), Eq (Min.min a (Max.max a b)) a
        sup_idem : ∀ (b : α), Eq (Max.max b b) b
        inf_idem : ∀ (b : α), Eq (Min.min b b) b
        semilatt_inf_inst : SemilatticeInf α := SemilatticeInf.mk' inf_comm inf_assoc  …
        semilatt_sup_inst : SemilatticeSup α := SemilatticeSup.mk' sup_comm sup_assoc  …
        partial_order_eq : Eq SemilatticeSup.toPartialOrder SemilatticeInf.toPartialOr …
        a b : α
        ⊢ LE.le (SemilatticeInf.inf a b) a
      -/
      apply inf_le_left,
      /-
        🎉 no goals
      -/
    inf_le_right := fun a b => by
      /-
        α✝ : Type u
        β : Type v
        α : Type u_1
        inst✝¹ : Max α
        inst✝ : Min α
        sup_comm : ∀ (a b : α), Eq (Max.max a b) (Max.max b a)
        sup_assoc : ∀ (a b c : α), Eq (Max.max (Max.max a b) c) (Max.max a (Max.max b  …
        inf_comm : ∀ (a b : α), Eq (Min.min a b) (Min.min b a)
        inf_assoc : ∀ (a b c : α), Eq (Min.min (Min.min a b) c) (Min.min a (Min.min b  …
        sup_inf_self : ∀ (a b : α), Eq (Max.max a (Min.min a b)) a
        inf_sup_self : ∀ (a b : α), Eq (Min.min a (Max.max a b)) a
        sup_idem : ∀ (b : α), Eq (Max.max b b) b
        inf_idem : ∀ (b : α), Eq (Min.min b b) b
        semilatt_inf_inst : SemilatticeInf α := SemilatticeInf.mk' inf_comm inf_assoc  …
        semilatt_sup_inst : SemilatticeSup α := SemilatticeSup.mk' sup_comm sup_assoc  …
        partial_order_eq : Eq SemilatticeSup.toPartialOrder SemilatticeInf.toPartialOr …
        a b : α
        ⊢ LE.le (SemilatticeInf.inf a b) b
      -/
      rw [partial_order_eq]
      /-
        α✝ : Type u
        β : Type v
        α : Type u_1
        inst✝¹ : Max α
        inst✝ : Min α
        sup_comm : ∀ (a b : α), Eq (Max.max a b) (Max.max b a)
        sup_assoc : ∀ (a b c : α), Eq (Max.max (Max.max a b) c) (Max.max a (Max.max b  …
        inf_comm : ∀ (a b : α), Eq (Min.min a b) (Min.min b a)
        inf_assoc : ∀ (a b c : α), Eq (Min.min (Min.min a b) c) (Min.min a (Min.min b  …
        sup_inf_self : ∀ (a b : α), Eq (Max.max a (Min.min a b)) a
        inf_sup_self : ∀ (a b : α), Eq (Min.min a (Max.max a b)) a
        sup_idem : ∀ (b : α), Eq (Max.max b b) b
        inf_idem : ∀ (b : α), Eq (Min.min b b) b
        semilatt_inf_inst : SemilatticeInf α := SemilatticeInf.mk' inf_comm inf_assoc  …
        semilatt_sup_inst : SemilatticeSup α := SemilatticeSup.mk' sup_comm sup_assoc  …
        partial_order_eq : Eq SemilatticeSup.toPartialOrder SemilatticeInf.toPartialOr …
        a b : α
        ⊢ LE.le (SemilatticeInf.inf a b) b
      -/
      apply inf_le_right,
      /-
        🎉 no goals
      -/
    le_inf := fun a b c => by
      /-
        α✝ : Type u
        β : Type v
        α : Type u_1
        inst✝¹ : Max α
        inst✝ : Min α
        sup_comm : ∀ (a b : α), Eq (Max.max a b) (Max.max b a)
        sup_assoc : ∀ (a b c : α), Eq (Max.max (Max.max a b) c) (Max.max a (Max.max b  …
        inf_comm : ∀ (a b : α), Eq (Min.min a b) (Min.min b a)
        inf_assoc : ∀ (a b c : α), Eq (Min.min (Min.min a b) c) (Min.min a (Min.min b  …
        sup_inf_self : ∀ (a b : α), Eq (Max.max a (Min.min a b)) a
        inf_sup_self : ∀ (a b : α), Eq (Min.min a (Max.max a b)) a
        sup_idem : ∀ (b : α), Eq (Max.max b b) b
        inf_idem : ∀ (b : α), Eq (Min.min b b) b
        semilatt_inf_inst : SemilatticeInf α := SemilatticeInf.mk' inf_comm inf_assoc  …
        semilatt_sup_inst : SemilatticeSup α := SemilatticeSup.mk' sup_comm sup_assoc  …
        partial_order_eq : Eq SemilatticeSup.toPartialOrder SemilatticeInf.toPartialOr …
        a b c : α
        ⊢ LE.le a b → LE.le a c → LE.le a (SemilatticeInf.inf b c)
      -/
      rw [partial_order_eq]
      /-
        α✝ : Type u
        β : Type v
        α : Type u_1
        inst✝¹ : Max α
        inst✝ : Min α
        sup_comm : ∀ (a b : α), Eq (Max.max a b) (Max.max b a)
        sup_assoc : ∀ (a b c : α), Eq (Max.max (Max.max a b) c) (Max.max a (Max.max b  …
        inf_comm : ∀ (a b : α), Eq (Min.min a b) (Min.min b a)
        inf_assoc : ∀ (a b c : α), Eq (Min.min (Min.min a b) c) (Min.min a (Min.min b  …
        sup_inf_self : ∀ (a b : α), Eq (Max.max a (Min.min a b)) a
        inf_sup_self : ∀ (a b : α), Eq (Min.min a (Max.max a b)) a
        sup_idem : ∀ (b : α), Eq (Max.max b b) b
        inf_idem : ∀ (b : α), Eq (Min.min b b) b
        semilatt_inf_inst : SemilatticeInf α := SemilatticeInf.mk' inf_comm inf_assoc  …
        semilatt_sup_inst : SemilatticeSup α := SemilatticeSup.mk' sup_comm sup_assoc  …
        partial_order_eq : Eq SemilatticeSup.toPartialOrder SemilatticeInf.toPartialOr …
        a b c : α
        ⊢ LE.le a b → LE.le a c → LE.le a (SemilatticeInf.inf b c)
      -/
      apply le_inf }
      /-
        🎉 no goals
      -/


theorem inf_le_sup : a ⊓ b ≤ a ⊔ b :=
  inf_le_left.trans le_sup_left


                                                 /-
                                                   α : Type u
                                                   inst✝ : Lattice α
                                                   a b : α
                                                   ⊢ Iff (LE.le (Max.max a b) (Min.min a b)) (Eq a b)
                                                 -/
theorem sup_le_inf : a ⊔ b ≤ a ⊓ b ↔ a = b := by simp [le_antisymm_iff, and_comm]
                                                 /-
                                                   🎉 no goals
                                                 -/


                                                       /-
                                                         α : Type u
                                                         inst✝ : Lattice α
                                                         a b : α
                                                         ⊢ Iff (Eq (Min.min a b) (Max.max a b)) (Eq a b)
                                                       -/
@[simp] lemma inf_eq_sup : a ⊓ b = a ⊔ b ↔ a = b := by rw [← inf_le_sup.ge_iff_eq, sup_le_inf]
                                                       /-
                                                         🎉 no goals
                                                       -/

@[simp] lemma sup_eq_inf : a ⊔ b = a ⊓ b ↔ a = b := eq_comm.trans inf_eq_sup

                                                       /-
                                                         α : Type u
                                                         inst✝ : Lattice α
                                                         a b : α
                                                         ⊢ Iff (LT.lt (Min.min a b) (Max.max a b)) (Ne a b)
                                                       -/
@[simp] lemma inf_lt_sup : a ⊓ b < a ⊔ b ↔ a ≠ b := by rw [inf_le_sup.lt_iff_ne, Ne, inf_eq_sup]
                                                       /-
                                                         🎉 no goals
                                                       -/


lemma inf_eq_and_sup_eq_iff : a ⊓ b = c ∧ a ⊔ b = c ↔ a = c ∧ b = c := by
  /-
    α : Type u
    inst✝ : Lattice α
    a b c : α
    ⊢ Iff (And (Eq (Min.min a b) c) (Eq (Max.max a b) c)) (And (Eq a c) (Eq b c))
  -/
  refine ⟨fun h ↦ ?_, ?_⟩
    /-
      case refine_1
      α : Type u
      inst✝ : Lattice α
      a b c : α
      h : And (Eq (Min.min a b) c) (Eq (Max.max a b) c)
      ⊢ And (Eq a c) (Eq b c)
    -/
  · obtain rfl := sup_eq_inf.1 (h.2.trans h.1.symm)
    /-
      case refine_1
      α : Type u
      inst✝ : Lattice α
      a c : α
      h : And (Eq (Min.min a a) c) (Eq (Max.max a a) c)
      ⊢ And (Eq a c) (Eq a c)
    -/
    simpa using h
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u
      inst✝ : Lattice α
      a b c : α
      ⊢ And (Eq a c) (Eq b c) → And (Eq (Min.min a b) c) (Eq (Max.max a b) c)
    -/
  · rintro ⟨rfl, rfl⟩
    /-
      case refine_2.intro
      α : Type u
      inst✝ : Lattice α
      b : α
      ⊢ And (Eq (Min.min b b) b) (Eq (Max.max b b) b)
    -/
    exact ⟨inf_idem _, sup_idem _⟩
    /-
      🎉 no goals
    -/


theorem sup_inf_le : a ⊔ b ⊓ c ≤ (a ⊔ b) ⊓ (a ⊔ c) :=
  le_inf (sup_le_sup_left inf_le_left _) (sup_le_sup_left inf_le_right _)


theorem le_inf_sup : a ⊓ b ⊔ a ⊓ c ≤ a ⊓ (b ⊔ c) :=
  sup_le (inf_le_inf_left _ le_sup_left) (inf_le_inf_left _ le_sup_right)


                                             /-
                                               α : Type u
                                               inst✝ : Lattice α
                                               a b : α
                                               ⊢ Eq (Min.min a (Max.max a b)) a
                                             -/
theorem inf_sup_self : a ⊓ (a ⊔ b) = a := by simp
                                             /-
                                               🎉 no goals
                                             -/


                                           /-
                                             α : Type u
                                             inst✝ : Lattice α
                                             a b : α
                                             ⊢ Eq (Max.max a (Min.min a b)) a
                                           -/
theorem sup_inf_self : a ⊔ a ⊓ b = a := by simp
                                           /-
                                             🎉 no goals
                                           -/


                                                        /-
                                                          α : Type u
                                                          inst✝ : Lattice α
                                                          a b : α
                                                          ⊢ Iff (Eq (Max.max a b) b) (Eq (Min.min a b) a)
                                                        -/
theorem sup_eq_iff_inf_eq : a ⊔ b = b ↔ a ⊓ b = a := by rw [sup_eq_right, ← inf_eq_left]
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem Lattice.ext {α} {A B : Lattice α} (H : ∀ x y : α, (haveI := A; x ≤ y) ↔ x ≤ y) :
    A = B := by
  /-
    α : Type u_1
    A B : Lattice α
    H : ∀ (x y : α), Iff (LE.le x y) (LE.le x y)
    ⊢ Eq A B
  -/
  cases A
  /-
    case mk
    α : Type u_1
    B : Lattice α
    toSemilatticeSup✝ : SemilatticeSup α
    inf✝ : α → α → α
    inf_le_left✝ : ∀ (a b : α), LE.le (inf✝ a b) a
    inf_le_right✝ : ∀ (a b : α), LE.le (inf✝ a b) b
    le_inf✝ : ∀ (a b c : α), LE.le a b → LE.le a c → LE.le a (inf✝ b c)
    H : ∀ (x y : α), Iff (LE.le x y) (LE.le x y)
    ⊢ Eq (Lattice.mk inf✝ inf_le_left✝ inf_le_right✝ le_inf✝) B
  -/
  cases B
  /-
    case mk.mk
    α : Type u_1
    toSemilatticeSup✝¹ : SemilatticeSup α
    inf✝¹ : α → α → α
    inf_le_left✝¹ : ∀ (a b : α), LE.le (inf✝¹ a b) a
    inf_le_right✝¹ : ∀ (a b : α), LE.le (inf✝¹ a b) b
    le_inf✝¹ : ∀ (a b c : α), LE.le a b → LE.le a c → LE.le a (inf✝¹ b c)
    toSemilatticeSup✝ : SemilatticeSup α
    inf✝ : α → α → α
    inf_le_left✝ : ∀ (a b : α), LE.le (inf✝ a b) a
    inf_le_right✝ : ∀ (a b : α), LE.le (inf✝ a b) b
    le_inf✝ : ∀ (a b c : α), LE.le a b → LE.le a c → LE.le a (inf✝ b c)
    H : ∀ (x y : α), Iff (LE.le x y) (LE.le x y)
    ⊢ Eq (Lattice.mk inf✝¹ inf_le_left✝¹ inf_le_right✝¹ le_inf✝¹) (Lattice.mk inf✝ …
  -/
  cases SemilatticeSup.ext H
  /-
    case mk.mk.refl
    α : Type u_1
    toSemilatticeSup✝ : SemilatticeSup α
    inf✝¹ : α → α → α
    inf_le_left✝¹ : ∀ (a b : α), LE.le (inf✝¹ a b) a
    inf_le_right✝¹ : ∀ (a b : α), LE.le (inf✝¹ a b) b
    le_inf✝¹ : ∀ (a b c : α), LE.le a b → LE.le a c → LE.le a (inf✝¹ b c)
    inf✝ : α → α → α
    inf_le_left✝ : ∀ (a b : α), LE.le (inf✝ a b) a
    inf_le_right✝ : ∀ (a b : α), LE.le (inf✝ a b) b
    le_inf✝ : ∀ (a b c : α), LE.le a b → LE.le a c → LE.le a (inf✝ b c)
    H : ∀ (x y : α), Iff (LE.le x y) (LE.le x y)
    ⊢ Eq (Lattice.mk inf✝¹ inf_le_left✝¹ inf_le_right✝¹ le_inf✝¹) (Lattice.mk inf✝ …
  -/
  cases SemilatticeInf.ext H
  /-
    case mk.mk.refl.refl
    α : Type u_1
    toSemilatticeSup✝ : SemilatticeSup α
    inf✝ : α → α → α
    inf_le_left✝¹ : ∀ (a b : α), LE.le (inf✝ a b) a
    inf_le_right✝¹ : ∀ (a b : α), LE.le (inf✝ a b) b
    le_inf✝¹ : ∀ (a b c : α), LE.le a b → LE.le a c → LE.le a (inf✝ b c)
    inf_le_left✝ : ∀ (a b : α), LE.le (inf✝ a b) a
    inf_le_right✝ : ∀ (a b : α), LE.le (inf✝ a b) b
    le_inf✝ : ∀ (a b c : α), LE.le a b → LE.le a c → LE.le a (inf✝ b c)
    H : ∀ (x y : α), Iff (LE.le x y) (LE.le x y)
    ⊢ Eq (Lattice.mk inf✝ inf_le_left✝¹ inf_le_right✝¹ le_inf✝¹) (Lattice.mk inf✝  …
  -/
  congr
  /-
    🎉 no goals
  -/


/-- A distributive lattice is a lattice that satisfies any of four
equivalent distributive properties (of `sup` over `inf` or `inf` over `sup`,
on the left or right).

The definition here chooses `le_sup_inf`: `(x ⊔ y) ⊓ (x ⊔ z) ≤ x ⊔ (y ⊓ z)`. To prove distributivity
from the dual law, use `DistribLattice.of_inf_sup_le`.

A classic example of a distributive lattice
is the lattice of subsets of a set, and in fact this example is
generic in the sense that every distributive lattice is realizable
as a sublattice of a powerset lattice. -/
class DistribLattice (α) extends Lattice α where
  /-- The infimum distributes over the supremum -/
  protected le_sup_inf : ∀ x y z : α, (x ⊔ y) ⊓ (x ⊔ z) ≤ x ⊔ y ⊓ z


theorem le_sup_inf : ∀ {x y z : α}, (x ⊔ y) ⊓ (x ⊔ z) ≤ x ⊔ y ⊓ z :=
  fun {x y z} => DistribLattice.le_sup_inf x y z


theorem sup_inf_left (a b c : α) : a ⊔ b ⊓ c = (a ⊔ b) ⊓ (a ⊔ c) :=
  le_antisymm sup_inf_le le_sup_inf


theorem sup_inf_right (a b c : α) : a ⊓ b ⊔ c = (a ⊔ c) ⊓ (b ⊔ c) := by
  /-
    α : Type u
    inst✝ : DistribLattice α
    a b c : α
    ⊢ Eq (Max.max (Min.min a b) c) (Min.min (Max.max a c) (Max.max b c))
  -/
  simp only [sup_inf_left, sup_comm _ c, eq_self_iff_true]
  /-
    🎉 no goals
  -/


theorem inf_sup_left (a b c : α) : a ⊓ (b ⊔ c) = a ⊓ b ⊔ a ⊓ c :=
  calc
                                              /-
                                                α : Type u
                                                inst✝ : DistribLattice α
                                                a b c : α
                                                ⊢ Eq (Min.min a (Max.max b c)) (Min.min (Min.min a (Max.max a c)) (Max.max b c))
                                              -/
    a ⊓ (b ⊔ c) = a ⊓ (a ⊔ c) ⊓ (b ⊔ c) := by rw [inf_sup_self]
                                              /-
                                                🎉 no goals
                                              -/
                              /-
                                α : Type u
                                inst✝ : DistribLattice α
                                a b c : α
                                ⊢ Eq (Min.min (Min.min a (Max.max a c)) (Max.max b c)) (Min.min a (Max.max (Mi …
                              -/
    _ = a ⊓ (a ⊓ b ⊔ c) := by simp only [inf_assoc, sup_inf_right, eq_self_iff_true]
                              /-
                                🎉 no goals
                              -/
                                        /-
                                          α : Type u
                                          inst✝ : DistribLattice α
                                          a b c : α
                                          ⊢ Eq (Min.min a (Max.max (Min.min a b) c)) (Min.min (Max.max a (Min.min a b))  …
                                        -/
    _ = (a ⊔ a ⊓ b) ⊓ (a ⊓ b ⊔ c) := by rw [sup_inf_self]
                                        /-
                                          🎉 no goals
                                        -/
                                        /-
                                          α : Type u
                                          inst✝ : DistribLattice α
                                          a b c : α
                                          ⊢ Eq (Min.min (Max.max a (Min.min a b)) (Max.max (Min.min a b) c)) (Min.min (M …
                                        -/
    _ = (a ⊓ b ⊔ a) ⊓ (a ⊓ b ⊔ c) := by rw [sup_comm]
                                        /-
                                          🎉 no goals
                                        -/
                            /-
                              α : Type u
                              inst✝ : DistribLattice α
                              a b c : α
                              ⊢ Eq (Min.min (Max.max (Min.min a b) a) (Max.max (Min.min a b) c)) (Max.max (M …
                            -/
    _ = a ⊓ b ⊔ a ⊓ c := by rw [sup_inf_left]
                            /-
                              🎉 no goals
                            -/


instance OrderDual.instDistribLattice (α : Type*) [DistribLattice α] : DistribLattice αᵒᵈ where
  __ := inferInstanceAs (Lattice αᵒᵈ)
  le_sup_inf _ _ _ := (inf_sup_left _ _ _).le


theorem inf_sup_right (a b c : α) : (a ⊔ b) ⊓ c = a ⊓ c ⊔ b ⊓ c := by
  /-
    α : Type u
    inst✝ : DistribLattice α
    a b c : α
    ⊢ Eq (Min.min (Max.max a b) c) (Max.max (Min.min a c) (Min.min b c))
  -/
  simp only [inf_sup_left, inf_comm _ c, eq_self_iff_true]
  /-
    🎉 no goals
  -/


theorem le_of_inf_le_sup_le (h₁ : x ⊓ z ≤ y ⊓ z) (h₂ : x ⊔ z ≤ y ⊔ z) : x ≤ y :=
  calc
    x ≤ y ⊓ z ⊔ x := le_sup_right
                                /-
                                  α : Type u
                                  inst✝ : DistribLattice α
                                  x y z : α
                                  h₁ : LE.le (Min.min x z) (Min.min y z)
                                  h₂ : LE.le (Max.max x z) (Max.max y z)
                                  ⊢ Eq (Max.max (Min.min y z) x) (Min.min (Max.max y x) (Max.max x z))
                                -/
    _ = (y ⊔ x) ⊓ (x ⊔ z) := by rw [sup_inf_right, sup_comm x]
                                /-
                                  🎉 no goals
                                -/
    _ ≤ (y ⊔ x) ⊓ (y ⊔ z) := inf_le_inf_left _ h₂
                        /-
                          α : Type u
                          inst✝ : DistribLattice α
                          x y z : α
                          h₁ : LE.le (Min.min x z) (Min.min y z)
                          h₂ : LE.le (Max.max x z) (Max.max y z)
                          ⊢ Eq (Min.min (Max.max y x) (Max.max y z)) (Max.max y (Min.min x z))
                        -/
    _ = y ⊔ x ⊓ z := by rw [← sup_inf_left]
                        /-
                          🎉 no goals
                        -/
    _ ≤ y ⊔ y ⊓ z := sup_le_sup_left h₁ _
    _ ≤ _ := sup_le (le_refl y) inf_le_left


theorem eq_of_inf_eq_sup_eq {a b c : α} (h₁ : b ⊓ a = c ⊓ a) (h₂ : b ⊔ a = c ⊔ a) : b = c :=
  le_antisymm (le_of_inf_le_sup_le (le_of_eq h₁) (le_of_eq h₂))
    (le_of_inf_le_sup_le (le_of_eq h₁.symm) (le_of_eq h₂.symm))


/-- Prove distributivity of an existing lattice from the dual distributive law. -/
abbrev DistribLattice.ofInfSupLe
    [Lattice α] (inf_sup_le : ∀ a b c : α, a ⊓ (b ⊔ c) ≤ a ⊓ b ⊔ a ⊓ c) : DistribLattice α where
  le_sup_inf := (@OrderDual.instDistribLattice αᵒᵈ {inferInstanceAs (Lattice αᵒᵈ) with
      le_sup_inf := inf_sup_le}).le_sup_inf


instance (priority := 100) LinearOrder.toLattice {α : Type u} [o : LinearOrder α] : Lattice α where
  __ := o
  le_sup_left := le_max_left; le_sup_right := le_max_right; sup_le _ _ _ := max_le
  inf_le_left := min_le_left; inf_le_right := min_le_right; le_inf _ _ _ := le_min


@[deprecated "is syntactical" (since := "2024-11-13"), nolint synTaut]
theorem sup_eq_max : a ⊔ b = max a b :=
  rfl


@[deprecated "is syntactical" (since := "2024-11-13"), nolint synTaut]
theorem inf_eq_min : a ⊓ b = min a b :=
  rfl


theorem sup_ind (a b : α) {p : α → Prop} (ha : p a) (hb : p b) : p (a ⊔ b) :=
                                                /-
                                                  α : Type u
                                                  inst✝ : LinearOrder α
                                                  a b : α
                                                  p : α → Prop
                                                  ha : p a
                                                  hb : p b
                                                  h : LE.le a b
                                                  ⊢ p (Max.max a b)
                                                -/
  (IsTotal.total a b).elim (fun h : a ≤ b => by rwa [sup_eq_right.2 h]) fun h => by
                                                /-
                                                  🎉 no goals
                                                -/
  /-
    α : Type u
    inst✝ : LinearOrder α
    a b : α
    p : α → Prop
    ha : p a
    hb : p b
    h : LE.le b a
    ⊢ p (Max.max a b)
  -/
  rwa [sup_eq_left.2 h]
  /-
    🎉 no goals
  -/


@[simp]
theorem le_sup_iff : a ≤ b ⊔ c ↔ a ≤ b ∨ a ≤ c := by
  exact ⟨fun h =>
    (le_total c b).imp
      (fun bc => by rwa [sup_eq_left.2 bc] at h)
      (fun bc => by rwa [sup_eq_right.2 bc] at h),
    fun h => h.elim le_sup_of_le_left le_sup_of_le_right⟩


@[simp]
theorem lt_sup_iff : a < b ⊔ c ↔ a < b ∨ a < c := by
  exact ⟨fun h =>
    (le_total c b).imp
      (fun bc => by rwa [sup_eq_left.2 bc] at h)
      (fun bc => by rwa [sup_eq_right.2 bc] at h),
    fun h => h.elim lt_sup_of_lt_left lt_sup_of_lt_right⟩


@[simp]
theorem sup_lt_iff : b ⊔ c < a ↔ b < a ∧ c < a :=
  ⟨fun h => ⟨le_sup_left.trans_lt h, le_sup_right.trans_lt h⟩,
   fun h => sup_ind (p := (· < a)) b c h.1 h.2⟩


theorem inf_ind (a b : α) {p : α → Prop} : p a → p b → p (a ⊓ b) :=
  @sup_ind αᵒᵈ _ _ _ _


@[simp]
theorem inf_le_iff : b ⊓ c ≤ a ↔ b ≤ a ∨ c ≤ a :=
  @le_sup_iff αᵒᵈ _ _ _ _


@[simp]
theorem inf_lt_iff : b ⊓ c < a ↔ b < a ∨ c < a :=
  @lt_sup_iff αᵒᵈ _ _ _ _


@[simp]
theorem lt_inf_iff : a < b ⊓ c ↔ a < b ∧ a < c :=
  @sup_lt_iff αᵒᵈ _ _ _ _


theorem max_max_max_comm : max (max a b) (max c d) = max (max a c) (max b d) :=
  sup_sup_sup_comm _ _ _ _


theorem min_min_min_comm : min (min a b) (min c d) = min (min a c) (min b d) :=
  inf_inf_inf_comm _ _ _ _


theorem sup_eq_maxDefault [SemilatticeSup α] [DecidableRel ((· ≤ ·) : α → α → Prop)]
    [IsTotal α (· ≤ ·)] :
    (· ⊔ ·) = (maxDefault : α → α → α) := by
  /-
    α : Type u
    inst✝² : SemilatticeSup α
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : IsTotal α fun x1 x2 => LE.le x1 x2
    ⊢ Eq (fun x1 x2 => Max.max x1 x2) maxDefault
  -/
  ext x y
  /-
    case h.h
    α : Type u
    inst✝² : SemilatticeSup α
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : IsTotal α fun x1 x2 => LE.le x1 x2
    x y : α
    ⊢ Eq (Max.max x y) (maxDefault x y)
  -/
  unfold maxDefault
  /-
    case h.h
    α : Type u
    inst✝² : SemilatticeSup α
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : IsTotal α fun x1 x2 => LE.le x1 x2
    x y : α
    ⊢ Eq (Max.max x y) (ite (LE.le x y) y x)
  -/
  split_ifs with h'
  /-
    case pos
    α : Type u
    inst✝² : SemilatticeSup α
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : IsTotal α fun x1 x2 => LE.le x1 x2
    x y : α
    h' : LE.le x y
    ⊢ Eq (Max.max x y) y
  -/
  exacts [sup_of_le_right h', sup_of_le_left <| (total_of (· ≤ ·) x y).resolve_left h']
  /-
    🎉 no goals
  -/


theorem inf_eq_minDefault [SemilatticeInf α] [DecidableRel ((· ≤ ·) : α → α → Prop)]
    [IsTotal α (· ≤ ·)] :
    (· ⊓ ·) = (minDefault : α → α → α) := by
  /-
    α : Type u
    inst✝² : SemilatticeInf α
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : IsTotal α fun x1 x2 => LE.le x1 x2
    ⊢ Eq (fun x1 x2 => Min.min x1 x2) minDefault
  -/
  ext x y
  /-
    case h.h
    α : Type u
    inst✝² : SemilatticeInf α
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : IsTotal α fun x1 x2 => LE.le x1 x2
    x y : α
    ⊢ Eq (Min.min x y) (minDefault x y)
  -/
  unfold minDefault
  /-
    case h.h
    α : Type u
    inst✝² : SemilatticeInf α
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : IsTotal α fun x1 x2 => LE.le x1 x2
    x y : α
    ⊢ Eq (Min.min x y) (ite (LE.le x y) x y)
  -/
  split_ifs with h'
  /-
    case pos
    α : Type u
    inst✝² : SemilatticeInf α
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : IsTotal α fun x1 x2 => LE.le x1 x2
    x y : α
    h' : LE.le x y
    ⊢ Eq (Min.min x y) x
  -/
  exacts [inf_of_le_left h', inf_of_le_right <| (total_of (· ≤ ·) x y).resolve_left h']
  /-
    🎉 no goals
  -/


/-- A lattice with total order is a linear order.

See note [reducible non-instances]. -/
abbrev Lattice.toLinearOrder (α : Type u) [Lattice α] [DecidableEq α]
    [DecidableRel ((· ≤ ·) : α → α → Prop)] [DecidableRel ((· < ·) : α → α → Prop)]
    [IsTotal α (· ≤ ·)] : LinearOrder α where
  __ := ‹Lattice α›
  decidableLE := ‹_›
  decidableEq := ‹_›
  decidableLT := ‹_›
  le_total := total_of (· ≤ ·)
                /-
                  α✝ : Type u
                  β : Type v
                  α : Type u
                  inst✝⁴ : Lattice α
                  inst✝³ : DecidableEq α
                  inst✝² : DecidableRel fun x1 x2 => LE.le x1 x2
                  inst✝¹ : DecidableRel fun x1 x2 => LT.lt x1 x2
                  inst✝ : IsTotal α fun x1 x2 => LE.le x1 x2
                  ⊢ ∀ (a b : α), Eq (Max.max a b) (ite (LE.le a b) b a)
                -/
                /-
                  α✝ : Type u
                  β : Type v
                  α : Type u
                  inst✝⁴ : Lattice α
                  inst✝³ : DecidableEq α
                  inst✝² : DecidableRel fun x1 x2 => LE.le x1 x2
                  inst✝¹ : DecidableRel fun x1 x2 => LT.lt x1 x2
                  inst✝ : IsTotal α fun x1 x2 => LE.le x1 x2
                  ⊢ ∀ (a b : α), Eq (Min.min a b) (ite (LE.le a b) a b)
                -/
  max_def := by exact congr_fun₂ sup_eq_maxDefault
                /-
                  🎉 no goals
                -/
                /-
                  🎉 no goals
                -/
  min_def := by exact congr_fun₂ inf_eq_minDefault

-- see Note [lower instance priority]

instance (priority := 100) {α : Type u} [LinearOrder α] : DistribLattice α where
  __ := inferInstanceAs (Lattice α)
  le_sup_inf _ b c :=
    match le_total b c with
    | Or.inl h => inf_le_of_left_le <| sup_le_sup_left (le_inf (le_refl b) h) _
    | Or.inr h => inf_le_of_right_le <| sup_le_sup_left (le_inf h (le_refl c)) _


instance : DistribLattice ℕ := inferInstance

instance : Lattice ℤ := inferInstance


@[simp]
theorem ofDual_inf [Max α] (a b : αᵒᵈ) : ofDual (a ⊓ b) = ofDual a ⊔ ofDual b :=
  rfl


@[simp]
theorem ofDual_sup [Min α] (a b : αᵒᵈ) : ofDual (a ⊔ b) = ofDual a ⊓ ofDual b :=
  rfl


@[simp]
theorem toDual_inf [Min α] (a b : α) : toDual (a ⊓ b) = toDual a ⊔ toDual b :=
  rfl


@[simp]
theorem toDual_sup [Max α] (a b : α) : toDual (a ⊔ b) = toDual a ⊓ toDual b :=
  rfl


@[simp]
theorem ofDual_min (a b : αᵒᵈ) : ofDual (min a b) = max (ofDual a) (ofDual b) :=
  rfl


@[simp]
theorem ofDual_max (a b : αᵒᵈ) : ofDual (max a b) = min (ofDual a) (ofDual b) :=
  rfl


@[simp]
theorem toDual_min (a b : α) : toDual (min a b) = max (toDual a) (toDual b) :=
  rfl


@[simp]
theorem toDual_max (a b : α) : toDual (max a b) = min (toDual a) (toDual b) :=
  rfl


instance [∀ i, Max (α' i)] : Max (∀ i, α' i) :=
  ⟨fun f g i => f i ⊔ g i⟩


@[simp]
theorem sup_apply [∀ i, Max (α' i)] (f g : ∀ i, α' i) (i : ι) : (f ⊔ g) i = f i ⊔ g i :=
  rfl


theorem sup_def [∀ i, Max (α' i)] (f g : ∀ i, α' i) : f ⊔ g = fun i => f i ⊔ g i :=
  rfl


instance [∀ i, Min (α' i)] : Min (∀ i, α' i) :=
  ⟨fun f g i => f i ⊓ g i⟩


@[simp]
theorem inf_apply [∀ i, Min (α' i)] (f g : ∀ i, α' i) (i : ι) : (f ⊓ g) i = f i ⊓ g i :=
  rfl


theorem inf_def [∀ i, Min (α' i)] (f g : ∀ i, α' i) : f ⊓ g = fun i => f i ⊓ g i :=
  rfl


instance instSemilatticeSup [∀ i, SemilatticeSup (α' i)] : SemilatticeSup (∀ i, α' i) where
  le_sup_left _ _ _ := le_sup_left
  le_sup_right _ _ _ := le_sup_right
  sup_le _ _ _ ac bc i := sup_le (ac i) (bc i)


instance instSemilatticeInf [∀ i, SemilatticeInf (α' i)] : SemilatticeInf (∀ i, α' i) where
  inf_le_left _ _ _ := inf_le_left
  inf_le_right _ _ _ := inf_le_right
  le_inf _ _ _ ac bc i := le_inf (ac i) (bc i)


instance instLattice [∀ i, Lattice (α' i)] : Lattice (∀ i, α' i) where
  __ := inferInstanceAs (SemilatticeSup (∀ i, α' i))
  __ := inferInstanceAs (SemilatticeInf (∀ i, α' i))


instance instDistribLattice [∀ i, DistribLattice (α' i)] : DistribLattice (∀ i, α' i) where
  le_sup_inf _ _ _ _ := le_sup_inf


theorem update_sup [∀ i, SemilatticeSup (π i)] (f : ∀ i, π i) (i : ι) (a b : π i) :
    update f i (a ⊔ b) = update f i a ⊔ update f i b :=
                     /-
                       ι : Type u_1
                       π : ι → Type u_2
                       inst✝¹ : DecidableEq ι
                       inst✝ : (i : ι) → SemilatticeSup (π i)
                       f : (i : ι) → π i
                       i : ι
                       a b : π i
                       j : ι
                       ⊢ Eq (Function.update f i (Max.max a b) j) (Max.max (Function.update f i a) (F …
                     -/
                                                          /-
                                                            🎉 no goals
                                                          -/
  funext fun j => by obtain rfl | hji := eq_or_ne j i <;> simp [update_of_ne, *]
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem update_inf [∀ i, SemilatticeInf (π i)] (f : ∀ i, π i) (i : ι) (a b : π i) :
    update f i (a ⊓ b) = update f i a ⊓ update f i b :=
                     /-
                       ι : Type u_1
                       π : ι → Type u_2
                       inst✝¹ : DecidableEq ι
                       inst✝ : (i : ι) → SemilatticeInf (π i)
                       f : (i : ι) → π i
                       i : ι
                       a b : π i
                       j : ι
                       ⊢ Eq (Function.update f i (Min.min a b) j) (Min.min (Function.update f i a) (F …
                     -/
                                                          /-
                                                            🎉 no goals
                                                          -/
  funext fun j => by obtain rfl | hji := eq_or_ne j i <;> simp [update_of_ne, *]
                                                          /-
                                                            🎉 no goals
                                                          -/


/-- Pointwise supremum of two monotone functions is a monotone function. -/
protected theorem sup [Preorder α] [SemilatticeSup β] {f g : α → β} (hf : Monotone f)
    (hg : Monotone g) :
    Monotone (f ⊔ g) := fun _ _ h => sup_le_sup (hf h) (hg h)


/-- Pointwise infimum of two monotone functions is a monotone function. -/
protected theorem inf [Preorder α] [SemilatticeInf β] {f g : α → β} (hf : Monotone f)
    (hg : Monotone g) :
    Monotone (f ⊓ g) := fun _ _ h => inf_le_inf (hf h) (hg h)


/-- Pointwise maximum of two monotone functions is a monotone function. -/
protected theorem max [Preorder α] [LinearOrder β] {f g : α → β} (hf : Monotone f)
    (hg : Monotone g) :
    Monotone fun x => max (f x) (g x) :=
  hf.sup hg


/-- Pointwise minimum of two monotone functions is a monotone function. -/
protected theorem min [Preorder α] [LinearOrder β] {f g : α → β} (hf : Monotone f)
    (hg : Monotone g) :
    Monotone fun x => min (f x) (g x) :=
  hf.inf hg


theorem le_map_sup [SemilatticeSup α] [SemilatticeSup β] {f : α → β} (h : Monotone f) (x y : α) :
    f x ⊔ f y ≤ f (x ⊔ y) :=
  sup_le (h le_sup_left) (h le_sup_right)


theorem map_inf_le [SemilatticeInf α] [SemilatticeInf β] {f : α → β} (h : Monotone f) (x y : α) :
    f (x ⊓ y) ≤ f x ⊓ f y :=
  le_inf (h inf_le_left) (h inf_le_right)


theorem of_map_inf_le_left [SemilatticeInf α] [Preorder β] {f : α → β}
    (h : ∀ x y, f (x ⊓ y) ≤ f x) : Monotone f := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : SemilatticeInf α
    inst✝ : Preorder β
    f : α → β
    h : ∀ (x y : α), LE.le (f (Min.min x y)) (f x)
    ⊢ Monotone f
  -/
  intro x y hxy
  /-
    α : Type u
    β : Type v
    inst✝¹ : SemilatticeInf α
    inst✝ : Preorder β
    f : α → β
    h : ∀ (x y : α), LE.le (f (Min.min x y)) (f x)
    x y : α
    hxy : LE.le x y
    ⊢ LE.le (f x) (f y)
  -/
  rw [← inf_eq_right.2 hxy]
  /-
    α : Type u
    β : Type v
    inst✝¹ : SemilatticeInf α
    inst✝ : Preorder β
    f : α → β
    h : ∀ (x y : α), LE.le (f (Min.min x y)) (f x)
    x y : α
    hxy : LE.le x y
    ⊢ LE.le (f (Min.min y x)) (f y)
  -/
  apply h
  /-
    🎉 no goals
  -/


theorem of_map_inf_le [SemilatticeInf α] [SemilatticeInf β] {f : α → β}
    (h : ∀ x y, f (x ⊓ y) ≤ f x ⊓ f y) : Monotone f :=
  of_map_inf_le_left fun x y ↦ (h x y).trans inf_le_left


theorem of_map_inf [SemilatticeInf α] [SemilatticeInf β] {f : α → β}
    (h : ∀ x y, f (x ⊓ y) = f x ⊓ f y) : Monotone f :=
  of_map_inf_le fun x y ↦ (h x y).le


theorem of_left_le_map_sup [SemilatticeSup α] [Preorder β] {f : α → β}
    (h : ∀ x y, f x ≤ f (x ⊔ y)) : Monotone f :=
  monotone_dual_iff.1 <| of_map_inf_le_left h


theorem of_le_map_sup [SemilatticeSup α] [SemilatticeSup β] {f : α → β}
    (h : ∀ x y, f x ⊔ f y ≤ f (x ⊔ y)) : Monotone f :=
  monotone_dual_iff.mp <| of_map_inf_le h


theorem of_map_sup [SemilatticeSup α] [SemilatticeSup β] {f : α → β}
    (h : ∀ x y, f (x ⊔ y) = f x ⊔ f y) : Monotone f :=
  (@of_map_inf (OrderDual α) (OrderDual β) _ _ _ h).dual


theorem map_sup [SemilatticeSup β] {f : α → β} (hf : Monotone f) (x y : α) :
    f (x ⊔ y) = f x ⊔ f y :=
                                                /-
                                                  α : Type u
                                                  β : Type v
                                                  inst✝¹ : LinearOrder α
                                                  inst✝ : SemilatticeSup β
                                                  f : α → β
                                                  hf : Monotone f
                                                  x y : α
                                                  h : LE.le x y
                                                  ⊢ Eq (f (Max.max x y)) (Max.max (f x) (f y))
                                                -/
  (IsTotal.total x y).elim (fun h : x ≤ y => by simp only [h, hf h, sup_of_le_right]) fun h => by
                                                /-
                                                  🎉 no goals
                                                -/
    /-
      α : Type u
      β : Type v
      inst✝¹ : LinearOrder α
      inst✝ : SemilatticeSup β
      f : α → β
      hf : Monotone f
      x y : α
      h : LE.le y x
      ⊢ Eq (f (Max.max x y)) (Max.max (f x) (f y))
    -/
    simp only [h, hf h, sup_of_le_left]
    /-
      🎉 no goals
    -/


theorem map_inf [SemilatticeInf β] {f : α → β} (hf : Monotone f) (x y : α) :
    f (x ⊓ y) = f x ⊓ f y :=
  hf.dual.map_sup _ _


/-- Pointwise supremum of two monotone functions is a monotone function. -/
protected theorem sup [Preorder α] [SemilatticeSup β] {f g : α → β} {s : Set α}
    (hf : MonotoneOn f s) (hg : MonotoneOn g s) : MonotoneOn (f ⊔ g) s :=
  fun _ hx _ hy h => sup_le_sup (hf hx hy h) (hg hx hy h)


/-- Pointwise infimum of two monotone functions is a monotone function. -/
protected theorem inf [Preorder α] [SemilatticeInf β] {f g : α → β} {s : Set α}
    (hf : MonotoneOn f s) (hg : MonotoneOn g s) : MonotoneOn (f ⊓ g) s :=
  (hf.dual.sup hg.dual).dual


/-- Pointwise maximum of two monotone functions is a monotone function. -/
protected theorem max [Preorder α] [LinearOrder β] {f g : α → β} {s : Set α} (hf : MonotoneOn f s)
    (hg : MonotoneOn g s) : MonotoneOn (fun x => max (f x) (g x)) s :=
  hf.sup hg


/-- Pointwise minimum of two monotone functions is a monotone function. -/
protected theorem min [Preorder α] [LinearOrder β] {f g : α → β} {s : Set α} (hf : MonotoneOn f s)
    (hg : MonotoneOn g s) : MonotoneOn (fun x => min (f x) (g x)) s :=
  hf.inf hg


theorem of_map_inf [SemilatticeInf α] [SemilatticeInf β]
    (h : ∀ x ∈ s, ∀ y ∈ s, f (x ⊓ y) = f x ⊓ f y) : MonotoneOn f s := fun x hx y hy hxy =>
                      /-
                        α : Type u
                        β : Type v
                        f : α → β
                        s : Set α
                        inst✝¹ : SemilatticeInf α
                        inst✝ : SemilatticeInf β
                        h : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y → Eq (f (Min …
                        x : α
                        hx : Membership.mem s x
                        y : α
                        hy : Membership.mem s y
                        hxy : LE.le x y
                        ⊢ Eq (Min.min (f x) (f y)) (f x)
                      -/
  inf_eq_left.1 <| by rw [← h _ hx _ hy, inf_eq_left.2 hxy]
                      /-
                        🎉 no goals
                      -/


theorem of_map_sup [SemilatticeSup α] [SemilatticeSup β]
    (h : ∀ x ∈ s, ∀ y ∈ s, f (x ⊔ y) = f x ⊔ f y) : MonotoneOn f s :=
  (@of_map_inf αᵒᵈ βᵒᵈ _ _ _ _ h).dual


theorem map_sup [SemilatticeSup β] (hf : MonotoneOn f s) (hx : x ∈ s) (hy : y ∈ s) :
    f (x ⊔ y) = f x ⊔ f y := by
  /-
    α : Type u
    β : Type v
    f : α → β
    s : Set α
    x y : α
    inst✝¹ : LinearOrder α
    inst✝ : SemilatticeSup β
    hf : MonotoneOn f s
    hx : Membership.mem s x
    hy : Membership.mem s y
    ⊢ Eq (f (Max.max x y)) (Max.max (f x) (f y))
  -/
  cases le_total x y <;> have := hf ?_ ?_ ‹_› <;>
    first
    | assumption
    | simp only [*, sup_of_le_left, sup_of_le_right]


theorem map_inf [SemilatticeInf β] (hf : MonotoneOn f s) (hx : x ∈ s) (hy : y ∈ s) :
    f (x ⊓ y) = f x ⊓ f y :=
  hf.dual.map_sup hx hy


/-- Pointwise supremum of two monotone functions is a monotone function. -/
protected theorem sup [Preorder α] [SemilatticeSup β] {f g : α → β} (hf : Antitone f)
    (hg : Antitone g) :
    Antitone (f ⊔ g) := fun _ _ h => sup_le_sup (hf h) (hg h)


/-- Pointwise infimum of two monotone functions is a monotone function. -/
protected theorem inf [Preorder α] [SemilatticeInf β] {f g : α → β} (hf : Antitone f)
    (hg : Antitone g) :
    Antitone (f ⊓ g) := fun _ _ h => inf_le_inf (hf h) (hg h)


/-- Pointwise maximum of two monotone functions is a monotone function. -/
protected theorem max [Preorder α] [LinearOrder β] {f g : α → β} (hf : Antitone f)
    (hg : Antitone g) :
    Antitone fun x => max (f x) (g x) :=
  hf.sup hg


/-- Pointwise minimum of two monotone functions is a monotone function. -/
protected theorem min [Preorder α] [LinearOrder β] {f g : α → β} (hf : Antitone f)
    (hg : Antitone g) :
    Antitone fun x => min (f x) (g x) :=
  hf.inf hg


theorem map_sup_le [SemilatticeSup α] [SemilatticeInf β] {f : α → β} (h : Antitone f) (x y : α) :
    f (x ⊔ y) ≤ f x ⊓ f y :=
  h.dual_right.le_map_sup x y


theorem le_map_inf [SemilatticeInf α] [SemilatticeSup β] {f : α → β} (h : Antitone f) (x y : α) :
    f x ⊔ f y ≤ f (x ⊓ y) :=
  h.dual_right.map_inf_le x y


theorem map_sup [SemilatticeInf β] {f : α → β} (hf : Antitone f) (x y : α) :
    f (x ⊔ y) = f x ⊓ f y :=
  hf.dual_right.map_sup x y


theorem map_inf [SemilatticeSup β] {f : α → β} (hf : Antitone f) (x y : α) :
    f (x ⊓ y) = f x ⊔ f y :=
  hf.dual_right.map_inf x y


/-- Pointwise supremum of two antitone functions is an antitone function. -/
protected theorem sup [Preorder α] [SemilatticeSup β] {f g : α → β} {s : Set α}
    (hf : AntitoneOn f s) (hg : AntitoneOn g s) : AntitoneOn (f ⊔ g) s :=
  fun _ hx _ hy h => sup_le_sup (hf hx hy h) (hg hx hy h)


/-- Pointwise infimum of two antitone functions is an antitone function. -/
protected theorem inf [Preorder α] [SemilatticeInf β] {f g : α → β} {s : Set α}
    (hf : AntitoneOn f s) (hg : AntitoneOn g s) : AntitoneOn (f ⊓ g) s :=
  (hf.dual.sup hg.dual).dual


/-- Pointwise maximum of two antitone functions is an antitone function. -/
protected theorem max [Preorder α] [LinearOrder β] {f g : α → β} {s : Set α} (hf : AntitoneOn f s)
    (hg : AntitoneOn g s) : AntitoneOn (fun x => max (f x) (g x)) s :=
  hf.sup hg


/-- Pointwise minimum of two antitone functions is an antitone function. -/
protected theorem min [Preorder α] [LinearOrder β] {f g : α → β} {s : Set α} (hf : AntitoneOn f s)
    (hg : AntitoneOn g s) : AntitoneOn (fun x => min (f x) (g x)) s :=
  hf.inf hg


theorem of_map_inf [SemilatticeInf α] [SemilatticeSup β]
    (h : ∀ x ∈ s, ∀ y ∈ s, f (x ⊓ y) = f x ⊔ f y) : AntitoneOn f s := fun x hx y hy hxy =>
                      /-
                        α : Type u
                        β : Type v
                        f : α → β
                        s : Set α
                        inst✝¹ : SemilatticeInf α
                        inst✝ : SemilatticeSup β
                        h : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y → Eq (f (Min …
                        x : α
                        hx : Membership.mem s x
                        y : α
                        hy : Membership.mem s y
                        hxy : LE.le x y
                        ⊢ Eq (Max.max (f x) (f y)) (f x)
                      -/
  sup_eq_left.1 <| by rw [← h _ hx _ hy, inf_eq_left.2 hxy]
                      /-
                        🎉 no goals
                      -/


theorem of_map_sup [SemilatticeSup α] [SemilatticeInf β]
    (h : ∀ x ∈ s, ∀ y ∈ s, f (x ⊔ y) = f x ⊓ f y) : AntitoneOn f s :=
  (@of_map_inf αᵒᵈ βᵒᵈ _ _ _ _ h).dual


theorem map_sup [SemilatticeInf β] (hf : AntitoneOn f s) (hx : x ∈ s) (hy : y ∈ s) :
    f (x ⊔ y) = f x ⊓ f y := by
  /-
    α : Type u
    β : Type v
    f : α → β
    s : Set α
    x y : α
    inst✝¹ : LinearOrder α
    inst✝ : SemilatticeInf β
    hf : AntitoneOn f s
    hx : Membership.mem s x
    hy : Membership.mem s y
    ⊢ Eq (f (Max.max x y)) (Min.min (f x) (f y))
  -/
  cases le_total x y <;> have := hf ?_ ?_ ‹_› <;>
    first
    | assumption
    | simp only [*, sup_of_le_left, sup_of_le_right, inf_of_le_left, inf_of_le_right]


theorem map_inf [SemilatticeSup β] (hf : AntitoneOn f s) (hx : x ∈ s) (hy : y ∈ s) :
    f (x ⊓ y) = f x ⊔ f y :=
  hf.dual.map_sup hx hy


instance [Max α] [Max β] : Max (α × β) :=
  ⟨fun p q => ⟨p.1 ⊔ q.1, p.2 ⊔ q.2⟩⟩


instance [Min α] [Min β] : Min (α × β) :=
  ⟨fun p q => ⟨p.1 ⊓ q.1, p.2 ⊓ q.2⟩⟩


@[simp]
theorem mk_sup_mk [Max α] [Max β] (a₁ a₂ : α) (b₁ b₂ : β) :
    (a₁, b₁) ⊔ (a₂, b₂) = (a₁ ⊔ a₂, b₁ ⊔ b₂) :=
  rfl


@[simp]
theorem mk_inf_mk [Min α] [Min β] (a₁ a₂ : α) (b₁ b₂ : β) :
    (a₁, b₁) ⊓ (a₂, b₂) = (a₁ ⊓ a₂, b₁ ⊓ b₂) :=
  rfl


@[simp]
theorem fst_sup [Max α] [Max β] (p q : α × β) : (p ⊔ q).fst = p.fst ⊔ q.fst :=
  rfl


@[simp]
theorem fst_inf [Min α] [Min β] (p q : α × β) : (p ⊓ q).fst = p.fst ⊓ q.fst :=
  rfl


@[simp]
theorem snd_sup [Max α] [Max β] (p q : α × β) : (p ⊔ q).snd = p.snd ⊔ q.snd :=
  rfl


@[simp]
theorem snd_inf [Min α] [Min β] (p q : α × β) : (p ⊓ q).snd = p.snd ⊓ q.snd :=
  rfl


@[simp]
theorem swap_sup [Max α] [Max β] (p q : α × β) : (p ⊔ q).swap = p.swap ⊔ q.swap :=
  rfl


@[simp]
theorem swap_inf [Min α] [Min β] (p q : α × β) : (p ⊓ q).swap = p.swap ⊓ q.swap :=
  rfl


theorem sup_def [Max α] [Max β] (p q : α × β) : p ⊔ q = (p.fst ⊔ q.fst, p.snd ⊔ q.snd) :=
  rfl


theorem inf_def [Min α] [Min β] (p q : α × β) : p ⊓ q = (p.fst ⊓ q.fst, p.snd ⊓ q.snd) :=
  rfl


instance instSemilatticeSup [SemilatticeSup α] [SemilatticeSup β] : SemilatticeSup (α × β) where
  __ := inferInstanceAs (PartialOrder (α × β))
  sup a b := ⟨a.1 ⊔ b.1, a.2 ⊔ b.2⟩
  sup_le _ _ _ h₁ h₂ := ⟨sup_le h₁.1 h₂.1, sup_le h₁.2 h₂.2⟩
  le_sup_left _ _ := ⟨le_sup_left, le_sup_left⟩
  le_sup_right _ _ := ⟨le_sup_right, le_sup_right⟩


instance instSemilatticeInf [SemilatticeInf α] [SemilatticeInf β] : SemilatticeInf (α × β) where
  __ := inferInstanceAs (PartialOrder (α × β))
  inf a b := ⟨a.1 ⊓ b.1, a.2 ⊓ b.2⟩
  le_inf _ _ _ h₁ h₂ := ⟨le_inf h₁.1 h₂.1, le_inf h₁.2 h₂.2⟩
  inf_le_left _ _ := ⟨inf_le_left, inf_le_left⟩
  inf_le_right _ _ := ⟨inf_le_right, inf_le_right⟩


instance instLattice [Lattice α] [Lattice β] : Lattice (α × β) where
  __ := inferInstanceAs (SemilatticeSup (α × β))
  __ := inferInstanceAs (SemilatticeInf (α × β))


instance instDistribLattice [DistribLattice α] [DistribLattice β] : DistribLattice (α × β) where
  __ := inferInstanceAs (Lattice (α × β))
  le_sup_inf _ _ _ := ⟨le_sup_inf, le_sup_inf⟩


/-- A subtype forms a `⊔`-semilattice if `⊔` preserves the property.
See note [reducible non-instances]. -/
protected abbrev semilatticeSup [SemilatticeSup α] {P : α → Prop}
    (Psup : ∀ ⦃x y⦄, P x → P y → P (x ⊔ y)) :
    SemilatticeSup { x : α // P x } where
  sup x y := ⟨x.1 ⊔ y.1, Psup x.2 y.2⟩
  le_sup_left _ _ := le_sup_left
  le_sup_right _ _ := le_sup_right
  sup_le _ _ _ h1 h2 := sup_le h1 h2


/-- A subtype forms a `⊓`-semilattice if `⊓` preserves the property.
See note [reducible non-instances]. -/
protected abbrev semilatticeInf [SemilatticeInf α] {P : α → Prop}
    (Pinf : ∀ ⦃x y⦄, P x → P y → P (x ⊓ y)) :
    SemilatticeInf { x : α // P x } where
  inf x y := ⟨x.1 ⊓ y.1, Pinf x.2 y.2⟩
  inf_le_left _ _ := inf_le_left
  inf_le_right _ _ := inf_le_right
  le_inf _ _ _ h1 h2 := le_inf h1 h2


/-- A subtype forms a lattice if `⊔` and `⊓` preserve the property.
See note [reducible non-instances]. -/
protected abbrev lattice [Lattice α] {P : α → Prop} (Psup : ∀ ⦃x y⦄, P x → P y → P (x ⊔ y))
    (Pinf : ∀ ⦃x y⦄, P x → P y → P (x ⊓ y)) : Lattice { x : α // P x } where
  __ := Subtype.semilatticeInf Pinf
  __ := Subtype.semilatticeSup Psup


@[simp, norm_cast]
theorem coe_sup [SemilatticeSup α] {P : α → Prop}
    (Psup : ∀ ⦃x y⦄, P x → P y → P (x ⊔ y)) (x y : Subtype P) :
    (haveI := Subtype.semilatticeSup Psup; (x ⊔ y : Subtype P) : α) = (x ⊔ y : α) :=
  rfl


@[simp, norm_cast]
theorem coe_inf [SemilatticeInf α] {P : α → Prop}
    (Pinf : ∀ ⦃x y⦄, P x → P y → P (x ⊓ y)) (x y : Subtype P) :
    (haveI := Subtype.semilatticeInf Pinf; (x ⊓ y : Subtype P) : α) = (x ⊓ y : α) :=
  rfl


@[simp]
theorem mk_sup_mk [SemilatticeSup α] {P : α → Prop}
    (Psup : ∀ ⦃x y⦄, P x → P y → P (x ⊔ y)) {x y : α} (hx : P x) (hy : P y) :
    (haveI := Subtype.semilatticeSup Psup; (⟨x, hx⟩ ⊔ ⟨y, hy⟩ : Subtype P)) =
      ⟨x ⊔ y, Psup hx hy⟩ :=
  rfl


@[simp]
theorem mk_inf_mk [SemilatticeInf α] {P : α → Prop}
    (Pinf : ∀ ⦃x y⦄, P x → P y → P (x ⊓ y)) {x y : α} (hx : P x) (hy : P y) :
    (haveI := Subtype.semilatticeInf Pinf; (⟨x, hx⟩ ⊓ ⟨y, hy⟩ : Subtype P)) =
      ⟨x ⊓ y, Pinf hx hy⟩ :=
  rfl


/-- A type endowed with `⊔` is a `SemilatticeSup`, if it admits an injective map that
preserves `⊔` to a `SemilatticeSup`.
See note [reducible non-instances]. -/
protected abbrev Function.Injective.semilatticeSup [Max α] [SemilatticeSup β] (f : α → β)
    (hf_inj : Function.Injective f) (map_sup : ∀ a b, f (a ⊔ b) = f a ⊔ f b) :
    SemilatticeSup α where
  __ := PartialOrder.lift f hf_inj
  sup a b := max a b
  le_sup_left a b := by
    /-
      α : Type u
      β : Type v
      inst✝¹ : Max α
      inst✝ : SemilatticeSup β
      f : α → β
      hf_inj : Function.Injective f
      map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
      a b : α
      ⊢ LE.le a ((fun a b => Max.max a b) a b)
    -/
    change f a ≤ f (a ⊔ b)
    /-
      α : Type u
      β : Type v
      inst✝¹ : Max α
      inst✝ : SemilatticeSup β
      f : α → β
      hf_inj : Function.Injective f
      map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
      a b : α
      ⊢ LE.le (f a) (f (Max.max a b))
    -/
    rw [map_sup]
    /-
      α : Type u
      β : Type v
      inst✝¹ : Max α
      inst✝ : SemilatticeSup β
      f : α → β
      hf_inj : Function.Injective f
      map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
      a b : α
      ⊢ LE.le (f a) (Max.max (f a) (f b))
    -/
    exact le_sup_left
    /-
      🎉 no goals
    -/
  le_sup_right a b := by
    /-
      α : Type u
      β : Type v
      inst✝¹ : Max α
      inst✝ : SemilatticeSup β
      f : α → β
      hf_inj : Function.Injective f
      map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
      a b : α
      ⊢ LE.le b ((fun a b => Max.max a b) a b)
    -/
    change f b ≤ f (a ⊔ b)
    /-
      α : Type u
      β : Type v
      inst✝¹ : Max α
      inst✝ : SemilatticeSup β
      f : α → β
      hf_inj : Function.Injective f
      map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
      a b : α
      ⊢ LE.le (f b) (f (Max.max a b))
    -/
    rw [map_sup]
    /-
      α : Type u
      β : Type v
      inst✝¹ : Max α
      inst✝ : SemilatticeSup β
      f : α → β
      hf_inj : Function.Injective f
      map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
      a b : α
      ⊢ LE.le (f b) (Max.max (f a) (f b))
    -/
    exact le_sup_right
    /-
      🎉 no goals
    -/
  sup_le a b c ha hb := by
    /-
      α : Type u
      β : Type v
      inst✝¹ : Max α
      inst✝ : SemilatticeSup β
      f : α → β
      hf_inj : Function.Injective f
      map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
      a b c : α
      ha : LE.le a c
      hb : LE.le b c
      ⊢ LE.le ((fun a b => Max.max a b) a b) c
    -/
    change f (a ⊔ b) ≤ f c
    /-
      α : Type u
      β : Type v
      inst✝¹ : Max α
      inst✝ : SemilatticeSup β
      f : α → β
      hf_inj : Function.Injective f
      map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
      a b c : α
      ha : LE.le a c
      hb : LE.le b c
      ⊢ LE.le (f (Max.max a b)) (f c)
    -/
    rw [map_sup]
    /-
      α : Type u
      β : Type v
      inst✝¹ : Max α
      inst✝ : SemilatticeSup β
      f : α → β
      hf_inj : Function.Injective f
      map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
      a b c : α
      ha : LE.le a c
      hb : LE.le b c
      ⊢ LE.le (Max.max (f a) (f b)) (f c)
    -/
    exact sup_le ha hb
    /-
      🎉 no goals
    -/


/-- A type endowed with `⊓` is a `SemilatticeInf`, if it admits an injective map that
preserves `⊓` to a `SemilatticeInf`.
See note [reducible non-instances]. -/
protected abbrev Function.Injective.semilatticeInf [Min α] [SemilatticeInf β] (f : α → β)
    (hf_inj : Function.Injective f) (map_inf : ∀ a b, f (a ⊓ b) = f a ⊓ f b) :
    SemilatticeInf α where
  __ := PartialOrder.lift f hf_inj
  inf a b := min a b
  inf_le_left a b := by
    /-
      α : Type u
      β : Type v
      inst✝¹ : Min α
      inst✝ : SemilatticeInf β
      f : α → β
      hf_inj : Function.Injective f
      map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
      a b : α
      ⊢ LE.le ((fun a b => Min.min a b) a b) a
    -/
    change f (a ⊓ b) ≤ f a
    /-
      α : Type u
      β : Type v
      inst✝¹ : Min α
      inst✝ : SemilatticeInf β
      f : α → β
      hf_inj : Function.Injective f
      map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
      a b : α
      ⊢ LE.le (f (Min.min a b)) (f a)
    -/
    rw [map_inf]
    /-
      α : Type u
      β : Type v
      inst✝¹ : Min α
      inst✝ : SemilatticeInf β
      f : α → β
      hf_inj : Function.Injective f
      map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
      a b : α
      ⊢ LE.le (Min.min (f a) (f b)) (f a)
    -/
    exact inf_le_left
    /-
      🎉 no goals
    -/
  inf_le_right a b := by
    /-
      α : Type u
      β : Type v
      inst✝¹ : Min α
      inst✝ : SemilatticeInf β
      f : α → β
      hf_inj : Function.Injective f
      map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
      a b : α
      ⊢ LE.le ((fun a b => Min.min a b) a b) b
    -/
    change f (a ⊓ b) ≤ f b
    /-
      α : Type u
      β : Type v
      inst✝¹ : Min α
      inst✝ : SemilatticeInf β
      f : α → β
      hf_inj : Function.Injective f
      map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
      a b : α
      ⊢ LE.le (f (Min.min a b)) (f b)
    -/
    rw [map_inf]
    /-
      α : Type u
      β : Type v
      inst✝¹ : Min α
      inst✝ : SemilatticeInf β
      f : α → β
      hf_inj : Function.Injective f
      map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
      a b : α
      ⊢ LE.le (Min.min (f a) (f b)) (f b)
    -/
    exact inf_le_right
    /-
      🎉 no goals
    -/
  le_inf a b c ha hb := by
    /-
      α : Type u
      β : Type v
      inst✝¹ : Min α
      inst✝ : SemilatticeInf β
      f : α → β
      hf_inj : Function.Injective f
      map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
      a b c : α
      ha : LE.le a b
      hb : LE.le a c
      ⊢ LE.le a ((fun a b => Min.min a b) b c)
    -/
    change f a ≤ f (b ⊓ c)
    /-
      α : Type u
      β : Type v
      inst✝¹ : Min α
      inst✝ : SemilatticeInf β
      f : α → β
      hf_inj : Function.Injective f
      map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
      a b c : α
      ha : LE.le a b
      hb : LE.le a c
      ⊢ LE.le (f a) (f (Min.min b c))
    -/
    rw [map_inf]
    /-
      α : Type u
      β : Type v
      inst✝¹ : Min α
      inst✝ : SemilatticeInf β
      f : α → β
      hf_inj : Function.Injective f
      map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
      a b c : α
      ha : LE.le a b
      hb : LE.le a c
      ⊢ LE.le (f a) (Min.min (f b) (f c))
    -/
    exact le_inf ha hb
    /-
      🎉 no goals
    -/


/-- A type endowed with `⊔` and `⊓` is a `Lattice`, if it admits an injective map that
preserves `⊔` and `⊓` to a `Lattice`.
See note [reducible non-instances]. -/
protected abbrev Function.Injective.lattice [Max α] [Min α] [Lattice β] (f : α → β)
    (hf_inj : Function.Injective f)
    (map_sup : ∀ a b, f (a ⊔ b) = f a ⊔ f b) (map_inf : ∀ a b, f (a ⊓ b) = f a ⊓ f b) :
    Lattice α where
  __ := hf_inj.semilatticeSup f map_sup
  __ := hf_inj.semilatticeInf f map_inf


/-- A type endowed with `⊔` and `⊓` is a `DistribLattice`, if it admits an injective map that
preserves `⊔` and `⊓` to a `DistribLattice`.
See note [reducible non-instances]. -/
protected abbrev Function.Injective.distribLattice [Max α] [Min α] [DistribLattice β] (f : α → β)
    (hf_inj : Function.Injective f) (map_sup : ∀ a b, f (a ⊔ b) = f a ⊔ f b)
    (map_inf : ∀ a b, f (a ⊓ b) = f a ⊓ f b) :
    DistribLattice α where
  __ := hf_inj.lattice f map_sup map_inf
  le_sup_inf a b c := by
    /-
      α : Type u
      β : Type v
      inst✝² : Max α
      inst✝¹ : Min α
      inst✝ : DistribLattice β
      f : α → β
      hf_inj : Function.Injective f
      map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
      map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
      a b c : α
      ⊢ LE.le (Min.min (Max.max a b) (Max.max a c)) (Max.max a (Min.min b c))
    -/
    change f ((a ⊔ b) ⊓ (a ⊔ c)) ≤ f (a ⊔ b ⊓ c)
    /-
      α : Type u
      β : Type v
      inst✝² : Max α
      inst✝¹ : Min α
      inst✝ : DistribLattice β
      f : α → β
      hf_inj : Function.Injective f
      map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
      map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
      a b c : α
      ⊢ LE.le (f (Min.min (Max.max a b) (Max.max a c))) (f (Max.max a (Min.min b c)))
    -/
    rw [map_inf, map_sup, map_sup, map_sup, map_inf]
    /-
      α : Type u
      β : Type v
      inst✝² : Max α
      inst✝¹ : Min α
      inst✝ : DistribLattice β
      f : α → β
      hf_inj : Function.Injective f
      map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
      map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
      a b c : α
      ⊢ LE.le (Min.min (Max.max (f a) (f b)) (Max.max (f a) (f c))) (Max.max (f a) ( …
    -/
    exact le_sup_inf
    /-
      🎉 no goals
    -/


instance [SemilatticeSup α] : SemilatticeSup (ULift.{v} α) :=
  ULift.down_injective.semilatticeSup _ down_sup


instance [SemilatticeInf α] : SemilatticeInf (ULift.{v} α) :=
  ULift.down_injective.semilatticeInf _ down_inf


instance [Lattice α] : Lattice (ULift.{v} α) :=
  ULift.down_injective.lattice _ down_sup down_inf


instance [DistribLattice α] : DistribLattice (ULift.{v} α) :=
  ULift.down_injective.distribLattice _ down_sup down_inf


instance [LinearOrder α] : LinearOrder (ULift.{v} α) :=
  LinearOrder.liftWithOrd ULift.down ULift.down_injective down_sup down_inf
    fun _x _y => (down_compare _ _).symm


instance Bool.instDistribLattice : DistribLattice Bool :=
  inferInstance

