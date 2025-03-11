/-- Like `cmp`, but uses a `≤` on the type instead of `<`. Given two elements `x` and `y`, returns a
three-way comparison result `Ordering`. -/
def cmpLE {α} [LE α] [DecidableRel (α := α) (· ≤ ·)] (x y : α) : Ordering :=
  if x ≤ y then if y ≤ x then Ordering.eq else Ordering.lt else Ordering.gt


theorem cmpLE_swap {α} [LE α] [IsTotal α (· ≤ ·)] [DecidableRel (α := α) (· ≤ ·)] (x y : α) :
    (cmpLE x y).swap = cmpLE y x := by
  /-
    α : Type u_3
    inst✝² : LE α
    inst✝¹ : IsTotal α fun x1 x2 => LE.le x1 x2
    inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
    x y : α
    ⊢ Eq (cmpLE x y).swap (cmpLE y x)
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
  by_cases xy : x ≤ y <;> by_cases yx : y ≤ x <;> simp [cmpLE, *, Ordering.swap]
  /-
    case neg
    α : Type u_3
    inst✝² : LE α
    inst✝¹ : IsTotal α fun x1 x2 => LE.le x1 x2
    inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
    x y : α
    xy : Not (LE.le x y)
    yx : Not (LE.le y x)
    ⊢ False
  -/
  cases not_or_intro xy yx (total_of _ _ _)
  /-
    🎉 no goals
  -/


theorem cmpLE_eq_cmp {α} [Preorder α] [IsTotal α (· ≤ ·)] [DecidableRel (α := α) (· ≤ ·)]
    [DecidableRel (α := α) (· < ·)] (x y : α) : cmpLE x y = cmp x y := by
  /-
    α : Type u_3
    inst✝³ : Preorder α
    inst✝² : IsTotal α fun x1 x2 => LE.le x1 x2
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : DecidableRel fun x1 x2 => LT.lt x1 x2
    x y : α
    ⊢ Eq (cmpLE x y) (cmp x y)
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
  by_cases xy : x ≤ y <;> by_cases yx : y ≤ x <;> simp [cmpLE, lt_iff_le_not_le, *, cmp, cmpUsing]
  /-
    case neg
    α : Type u_3
    inst✝³ : Preorder α
    inst✝² : IsTotal α fun x1 x2 => LE.le x1 x2
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : DecidableRel fun x1 x2 => LT.lt x1 x2
    x y : α
    xy : Not (LE.le x y)
    yx : Not (LE.le y x)
    ⊢ False
  -/
  cases not_or_intro xy yx (total_of _ _ _)
  /-
    🎉 no goals
  -/


theorem compares_swap [LT α] {a b : α} {o : Ordering} : o.swap.Compares a b ↔ o.Compares b a := by
  /-
    α : Type u_1
    inst✝ : LT α
    a b : α
    o : Ordering
    ⊢ Iff (o.swap.Compares a b) (o.Compares b a)
  -/
  cases o
    /-
      case lt
      α : Type u_1
      inst✝ : LT α
      a b : α
      ⊢ Iff (Ordering.lt.swap.Compares a b) (Ordering.lt.Compares b a)
    -/
  · exact Iff.rfl
    /-
      🎉 no goals
    -/
    /-
      case eq
      α : Type u_1
      inst✝ : LT α
      a b : α
      ⊢ Iff (Ordering.eq.swap.Compares a b) (Ordering.eq.Compares b a)
    -/
  · exact eq_comm
    /-
      🎉 no goals
    -/
    /-
      case gt
      α : Type u_1
      inst✝ : LT α
      a b : α
      ⊢ Iff (Ordering.gt.swap.Compares a b) (Ordering.gt.Compares b a)
    -/
  · exact Iff.rfl
    /-
      🎉 no goals
    -/


alias ⟨Compares.of_swap, Compares.swap⟩ := compares_swap


theorem swap_eq_iff_eq_swap {o o' : Ordering} : o.swap = o' ↔ o = o'.swap := by
  /-
    o o' : Ordering
    ⊢ Iff (Eq o.swap o') (Eq o o'.swap)
  -/
  rw [← swap_inj, swap_swap]
  /-
    🎉 no goals
  -/


theorem Compares.eq_lt [Preorder α] : ∀ {o} {a b : α}, Compares o a b → (o = lt ↔ a < b)
  | lt, _, _, h => ⟨fun _ => h, fun _ => rfl⟩
                                /-
                                  α : Type u_1
                                  inst✝ : Preorder α
                                  a b : α
                                  h✝ : Ordering.eq.Compares a b
                                  h : Eq Ordering.eq Ordering.lt
                                  ⊢ LT.lt a b
                                -/
  | eq, a, b, h => ⟨fun h => by injection h, fun h' => (ne_of_lt h' h).elim⟩
                                /-
                                  🎉 no goals
                                -/
                                /-
                                  α : Type u_1
                                  inst✝ : Preorder α
                                  a b : α
                                  h✝ : Ordering.gt.Compares a b
                                  h : Eq Ordering.gt Ordering.lt
                                  ⊢ LT.lt a b
                                -/
  | gt, a, b, h => ⟨fun h => by injection h, fun h' => (lt_asymm h h').elim⟩
                                /-
                                  🎉 no goals
                                -/


theorem Compares.ne_lt [Preorder α] : ∀ {o} {a b : α}, Compares o a b → (o ≠ lt ↔ b ≤ a)
  | lt, _, _, h => ⟨absurd rfl, fun h' => (not_le_of_lt h h').elim⟩
                                                       /-
                                                         α : Type u_1
                                                         inst✝ : Preorder α
                                                         x✝² x✝¹ : α
                                                         h✝ : Ordering.eq.Compares x✝² x✝¹
                                                         x✝ : LE.le x✝¹ x✝²
                                                         h : Eq Ordering.eq Ordering.lt
                                                         ⊢ False
                                                       -/
  | eq, _, _, h => ⟨fun _ => ge_of_eq h, fun _ h => by injection h⟩
                                                       /-
                                                         🎉 no goals
                                                       -/
                                                       /-
                                                         α : Type u_1
                                                         inst✝ : Preorder α
                                                         x✝² x✝¹ : α
                                                         h✝ : Ordering.gt.Compares x✝² x✝¹
                                                         x✝ : LE.le x✝¹ x✝²
                                                         h : Eq Ordering.gt Ordering.lt
                                                         ⊢ False
                                                       -/
  | gt, _, _, h => ⟨fun _ => le_of_lt h, fun _ h => by injection h⟩
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem Compares.eq_eq [Preorder α] : ∀ {o} {a b : α}, Compares o a b → (o = eq ↔ a = b)
                                /-
                                  α : Type u_1
                                  inst✝ : Preorder α
                                  a b : α
                                  h✝ : Ordering.lt.Compares a b
                                  h : Eq Ordering.lt Ordering.eq
                                  ⊢ Eq a b
                                -/
  | lt, a, b, h => ⟨fun h => by injection h, fun h' => (ne_of_lt h h').elim⟩
                                /-
                                  🎉 no goals
                                -/
  | eq, _, _, h => ⟨fun _ => h, fun _ => rfl⟩
                                /-
                                  α : Type u_1
                                  inst✝ : Preorder α
                                  a b : α
                                  h✝ : Ordering.gt.Compares a b
                                  h : Eq Ordering.gt Ordering.eq
                                  ⊢ Eq a b
                                -/
  | gt, a, b, h => ⟨fun h => by injection h, fun h' => (ne_of_gt h h').elim⟩
                                /-
                                  🎉 no goals
                                -/


theorem Compares.eq_gt [Preorder α] {o} {a b : α} (h : Compares o a b) : o = gt ↔ b < a :=
  swap_eq_iff_eq_swap.symm.trans h.swap.eq_lt


theorem Compares.ne_gt [Preorder α] {o} {a b : α} (h : Compares o a b) : o ≠ gt ↔ a ≤ b :=
  (not_congr swap_eq_iff_eq_swap.symm).trans h.swap.ne_lt


theorem Compares.le_total [Preorder α] {a b : α} : ∀ {o}, Compares o a b → a ≤ b ∨ b ≤ a
  | lt, h => Or.inl (le_of_lt h)
  | eq, h => Or.inl (le_of_eq h)
  | gt, h => Or.inr (le_of_lt h)


theorem Compares.le_antisymm [Preorder α] {a b : α} : ∀ {o}, Compares o a b → a ≤ b → b ≤ a → a = b
  | lt, h, _, hba => (not_le_of_lt h hba).elim
  | eq, h, _, _ => h
  | gt, h, hab, _ => (not_le_of_lt h hab).elim


theorem Compares.inj [Preorder α] {o₁} :
    ∀ {o₂} {a b : α}, Compares o₁ a b → Compares o₂ a b → o₁ = o₂
  | lt, _, _, h₁, h₂ => h₁.eq_lt.2 h₂
  | eq, _, _, h₁, h₂ => h₁.eq_eq.2 h₂
  | gt, _, _, h₁, h₂ => h₁.eq_gt.2 h₂

-- Porting note: mathlib3 proof uses `change ... at hab`

theorem compares_iff_of_compares_impl [LinearOrder α] [Preorder β] {a b : α} {a' b' : β}
    (h : ∀ {o}, Compares o a b → Compares o a' b') (o) : Compares o a b ↔ Compares o a' b' := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : Preorder β
    a b : α
    a' b' : β
    h : ∀ {o : Ordering}, o.Compares a b → o.Compares a' b'
    o : Ordering
    ⊢ Iff (o.Compares a b) (o.Compares a' b')
  -/
  refine ⟨h, fun ho => ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : Preorder β
    a b : α
    a' b' : β
    h : ∀ {o : Ordering}, o.Compares a b → o.Compares a' b'
    o : Ordering
    ho : o.Compares a' b'
    ⊢ o.Compares a b
  -/
  rcases lt_trichotomy a b with hab | hab | hab
    /-
      case inl
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrder α
      inst✝ : Preorder β
      a b : α
      a' b' : β
      h : ∀ {o : Ordering}, o.Compares a b → o.Compares a' b'
      o : Ordering
      ho : o.Compares a' b'
      hab : LT.lt a b
      ⊢ o.Compares a b
    -/
  · have hab : Compares Ordering.lt a b := hab
    /-
      case inl
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrder α
      inst✝ : Preorder β
      a b : α
      a' b' : β
      h : ∀ {o : Ordering}, o.Compares a b → o.Compares a' b'
      o : Ordering
      ho : o.Compares a' b'
      hab✝ : LT.lt a b
      hab : Ordering.lt.Compares a b
      ⊢ o.Compares a b
    -/
    rwa [ho.inj (h hab)]
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrder α
      inst✝ : Preorder β
      a b : α
      a' b' : β
      h : ∀ {o : Ordering}, o.Compares a b → o.Compares a' b'
      o : Ordering
      ho : o.Compares a' b'
      hab : Eq a b
      ⊢ o.Compares a b
    -/
  · have hab : Compares Ordering.eq a b := hab
    /-
      case inr.inl
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrder α
      inst✝ : Preorder β
      a b : α
      a' b' : β
      h : ∀ {o : Ordering}, o.Compares a b → o.Compares a' b'
      o : Ordering
      ho : o.Compares a' b'
      hab✝ : Eq a b
      hab : Ordering.eq.Compares a b
      ⊢ o.Compares a b
    -/
    rwa [ho.inj (h hab)]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrder α
      inst✝ : Preorder β
      a b : α
      a' b' : β
      h : ∀ {o : Ordering}, o.Compares a b → o.Compares a' b'
      o : Ordering
      ho : o.Compares a' b'
      hab : LT.lt b a
      ⊢ o.Compares a b
    -/
  · have hab : Compares Ordering.gt a b := hab
    /-
      case inr.inr
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrder α
      inst✝ : Preorder β
      a b : α
      a' b' : β
      h : ∀ {o : Ordering}, o.Compares a b → o.Compares a' b'
      o : Ordering
      ho : o.Compares a' b'
      hab✝ : LT.lt b a
      hab : Ordering.gt.Compares a b
      ⊢ o.Compares a b
    -/
    rwa [ho.inj (h hab)]
    /-
      🎉 no goals
    -/


set_option linter.deprecated false in
@[deprecated swap_then (since := "2024-09-13")]
theorem swap_orElse (o₁ o₂) : (orElse o₁ o₂).swap = orElse o₁.swap o₂.swap := swap_then ..


set_option linter.deprecated false in
@[deprecated then_eq_lt (since := "2024-09-13")]
theorem orElse_eq_lt (o₁ o₂) : orElse o₁ o₂ = lt ↔ o₁ = lt ∨ o₁ = eq ∧ o₂ = lt := then_eq_lt ..


@[simp]
theorem toDual_compares_toDual [LT α] {a b : α} {o : Ordering} :
    Compares o (toDual a) (toDual b) ↔ Compares o b a := by
  /-
    α : Type u_1
    inst✝ : LT α
    a b : α
    o : Ordering
    ⊢ Iff (o.Compares (OrderDual.toDual a) (OrderDual.toDual b)) (o.Compares b a)
  -/
  cases o
  /-
    case lt
    α : Type u_1
    inst✝ : LT α
    a b : α
    ⊢ Iff (Ordering.lt.Compares (OrderDual.toDual a) (OrderDual.toDual b)) (Orderi …
  -/
  exacts [Iff.rfl, eq_comm, Iff.rfl]
  /-
    🎉 no goals
  -/


@[simp]
theorem ofDual_compares_ofDual [LT α] {a b : αᵒᵈ} {o : Ordering} :
    Compares o (ofDual a) (ofDual b) ↔ Compares o b a := by
  /-
    α : Type u_1
    inst✝ : LT α
    a b : OrderDual α
    o : Ordering
    ⊢ Iff (o.Compares (OrderDual.ofDual a) (OrderDual.ofDual b)) (o.Compares b a)
  -/
  cases o
  /-
    case lt
    α : Type u_1
    inst✝ : LT α
    a b : OrderDual α
    ⊢ Iff (Ordering.lt.Compares (OrderDual.ofDual a) (OrderDual.ofDual b)) (Orderi …
  -/
  exacts [Iff.rfl, eq_comm, Iff.rfl]
  /-
    🎉 no goals
  -/


theorem cmp_compares [LinearOrder α] (a b : α) : (cmp a b).Compares a b := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b : α
    ⊢ (cmp a b).Compares a b
  -/
                                            /-
                                              🎉 no goals
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
  obtain h | h | h := lt_trichotomy a b <;> simp [cmp, cmpUsing, h, h.not_lt]
                                            /-
                                              🎉 no goals
                                            -/


theorem Ordering.Compares.cmp_eq [LinearOrder α] {a b : α} {o : Ordering} (h : o.Compares a b) :
    cmp a b = o :=
  (cmp_compares a b).inj h


@[simp]
theorem cmp_swap [Preorder α] [DecidableRel (α := α) (· < ·)] (a b : α) :
    (cmp a b).swap = cmp b a := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : DecidableRel fun x1 x2 => LT.lt x1 x2
    a b : α
    ⊢ Eq (cmp a b).swap (cmp b a)
  -/
  unfold cmp cmpUsing
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : DecidableRel fun x1 x2 => LT.lt x1 x2
    a b : α
    ⊢ Eq (ite ((fun x1 x2 => LT.lt x1 x2) a b) Ordering.lt (ite ((fun x1 x2 => LT. …
  -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
  by_cases h : a < b <;> by_cases h₂ : b < a <;> simp [h, h₂, Ordering.swap]
                                                 /-
                                                   🎉 no goals
                                                 -/
  /-
    case pos
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : DecidableRel fun x1 x2 => LT.lt x1 x2
    a b : α
    h : LT.lt a b
    h₂ : LT.lt b a
    ⊢ False
  -/
  exact lt_asymm h h₂
  /-
    🎉 no goals
  -/


@[simp]
theorem cmpLE_toDual [LE α] [DecidableRel (α := α) (· ≤ ·)] (x y : α) :
    cmpLE (toDual x) (toDual y) = cmpLE y x :=
  rfl


@[simp]
theorem cmpLE_ofDual [LE α] [DecidableRel (α := α) (· ≤ ·)] (x y : αᵒᵈ) :
    cmpLE (ofDual x) (ofDual y) = cmpLE y x :=
  rfl


@[simp]
theorem cmp_toDual [LT α] [DecidableRel (α := α) (· < ·)] (x y : α) :
    cmp (toDual x) (toDual y) = cmp y x :=
  rfl


@[simp]
theorem cmp_ofDual [LT α] [DecidableRel (α := α) (· < ·)] (x y : αᵒᵈ) :
    cmp (ofDual x) (ofDual y) = cmp y x :=
  rfl


/-- Generate a linear order structure from a preorder and `cmp` function. -/
def linearOrderOfCompares [Preorder α] (cmp : α → α → Ordering)
    (h : ∀ a b, (cmp a b).Compares a b) : LinearOrder α :=
  let H : DecidableRel (α := α) (· ≤ ·) := fun a b => decidable_of_iff _ (h a b).ne_gt
  { inferInstanceAs (Preorder α) with
    le_antisymm := fun a b => (h a b).le_antisymm,
    le_total := fun a b => (h a b).le_total,
    toMin := minOfLe,
    toMax := maxOfLe,
    decidableLE := H,
    decidableLT := fun a b => decidable_of_iff _ (h a b).eq_lt,
    decidableEq := fun a b => decidable_of_iff _ (h a b).eq_eq }


@[simp]
theorem cmp_eq_lt_iff : cmp x y = Ordering.lt ↔ x < y :=
  Ordering.Compares.eq_lt (cmp_compares x y)


@[simp]
theorem cmp_eq_eq_iff : cmp x y = Ordering.eq ↔ x = y :=
  Ordering.Compares.eq_eq (cmp_compares x y)


@[simp]
theorem cmp_eq_gt_iff : cmp x y = Ordering.gt ↔ y < x :=
  Ordering.Compares.eq_gt (cmp_compares x y)


@[simp]
                                                     /-
                                                       α : Type u_1
                                                       inst✝ : LinearOrder α
                                                       x : α
                                                       ⊢ Eq (cmp x x) Ordering.eq
                                                     -/
theorem cmp_self_eq_eq : cmp x x = Ordering.eq := by rw [cmp_eq_eq_iff]
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem cmp_eq_cmp_symm : cmp x y = cmp x' y' ↔ cmp y x = cmp y' x' :=
               /-
                 α : Type u_1
                 inst✝¹ : LinearOrder α
                 x y : α
                 β : Type u_3
                 inst✝ : LinearOrder β
                 x' y' : β
                 h : Eq (cmp x y) (cmp x' y')
                 ⊢ Eq (cmp y x) (cmp y' x')
               -/
  ⟨fun h => by rwa [← cmp_swap x', ← cmp_swap, swap_inj],
               /-
                 🎉 no goals
               -/
               /-
                 α : Type u_1
                 inst✝¹ : LinearOrder α
                 x y : α
                 β : Type u_3
                 inst✝ : LinearOrder β
                 x' y' : β
                 h : Eq (cmp y x) (cmp y' x')
                 ⊢ Eq (cmp x y) (cmp x' y')
               -/
   fun h => by rwa [← cmp_swap y', ← cmp_swap, swap_inj]⟩
               /-
                 🎉 no goals
               -/


theorem lt_iff_lt_of_cmp_eq_cmp (h : cmp x y = cmp x' y') : x < y ↔ x' < y' := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    x y : α
    β : Type u_3
    inst✝ : LinearOrder β
    x' y' : β
    h : Eq (cmp x y) (cmp x' y')
    ⊢ Iff (LT.lt x y) (LT.lt x' y')
  -/
  rw [← cmp_eq_lt_iff, ← cmp_eq_lt_iff, h]
  /-
    🎉 no goals
  -/


theorem le_iff_le_of_cmp_eq_cmp (h : cmp x y = cmp x' y') : x ≤ y ↔ x' ≤ y' := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    x y : α
    β : Type u_3
    inst✝ : LinearOrder β
    x' y' : β
    h : Eq (cmp x y) (cmp x' y')
    ⊢ Iff (LE.le x y) (LE.le x' y')
  -/
  rw [← not_lt, ← not_lt]
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    x y : α
    β : Type u_3
    inst✝ : LinearOrder β
    x' y' : β
    h : Eq (cmp x y) (cmp x' y')
    ⊢ Iff (Not (LT.lt y x)) (Not (LT.lt y' x'))
  -/
  apply not_congr
  /-
    case h
    α : Type u_1
    inst✝¹ : LinearOrder α
    x y : α
    β : Type u_3
    inst✝ : LinearOrder β
    x' y' : β
    h : Eq (cmp x y) (cmp x' y')
    ⊢ Iff (LT.lt y x) (LT.lt y' x')
  -/
  apply lt_iff_lt_of_cmp_eq_cmp
  /-
    case h.h
    α : Type u_1
    inst✝¹ : LinearOrder α
    x y : α
    β : Type u_3
    inst✝ : LinearOrder β
    x' y' : β
    h : Eq (cmp x y) (cmp x' y')
    ⊢ Eq (cmp y x) (cmp y' x')
  -/
  rwa [cmp_eq_cmp_symm]
  /-
    🎉 no goals
  -/


theorem eq_iff_eq_of_cmp_eq_cmp (h : cmp x y = cmp x' y') : x = y ↔ x' = y' := by
  rw [le_antisymm_iff, le_antisymm_iff, le_iff_le_of_cmp_eq_cmp h,
      le_iff_le_of_cmp_eq_cmp (cmp_eq_cmp_symm.1 h)]


theorem LT.lt.cmp_eq_lt (h : x < y) : cmp x y = Ordering.lt :=
  (cmp_eq_lt_iff _ _).2 h


theorem LT.lt.cmp_eq_gt (h : x < y) : cmp y x = Ordering.gt :=
  (cmp_eq_gt_iff _ _).2 h


theorem Eq.cmp_eq_eq (h : x = y) : cmp x y = Ordering.eq :=
  (cmp_eq_eq_iff _ _).2 h


theorem Eq.cmp_eq_eq' (h : x = y) : cmp y x = Ordering.eq :=
  h.symm.cmp_eq_eq

