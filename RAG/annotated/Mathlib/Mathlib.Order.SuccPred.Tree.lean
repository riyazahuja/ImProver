/--
The unique atom less than an element in an `OrderBot` with archimedean predecessor.
-/
def findAtom (r : α) : α :=
  Order.pred^[Nat.find (bot_le (a := r)).exists_pred_iterate - 1] r


@[simp]
lemma findAtom_le (r : α) : findAtom r ≤ r :=
  Order.pred_iterate_le _ _


@[simp]
lemma findAtom_bot : findAtom (⊥ : α) = ⊥ := by
  /-
    α : Type u_1
    inst✝⁴ : PartialOrder α
    inst✝³ : PredOrder α
    inst✝² : IsPredArchimedean α
    inst✝¹ : OrderBot α
    inst✝ : DecidableEq α
    ⊢ Eq (IsPredArchimedean.findAtom Bot.bot) Bot.bot
  -/
  apply Function.iterate_fixed
  /-
    case h
    α : Type u_1
    inst✝⁴ : PartialOrder α
    inst✝³ : PredOrder α
    inst✝² : IsPredArchimedean α
    inst✝¹ : OrderBot α
    inst✝ : DecidableEq α
    ⊢ Eq (Order.pred Bot.bot) Bot.bot
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma pred_findAtom (r : α) : Order.pred (findAtom r) = ⊥ := by
  /-
    α : Type u_1
    inst✝⁴ : PartialOrder α
    inst✝³ : PredOrder α
    inst✝² : IsPredArchimedean α
    inst✝¹ : OrderBot α
    inst✝ : DecidableEq α
    r : α
    ⊢ Eq (Order.pred (IsPredArchimedean.findAtom r)) Bot.bot
  -/
  unfold findAtom
  /-
    α : Type u_1
    inst✝⁴ : PartialOrder α
    inst✝³ : PredOrder α
    inst✝² : IsPredArchimedean α
    inst✝¹ : OrderBot α
    inst✝ : DecidableEq α
    r : α
    ⊢ Eq (Order.pred (Nat.iterate Order.pred (HSub.hSub (Nat.find ⋯) 1) r)) Bot.bot
  -/
  generalize h : Nat.find (bot_le (a := r)).exists_pred_iterate = n
  /-
    α : Type u_1
    inst✝⁴ : PartialOrder α
    inst✝³ : PredOrder α
    inst✝² : IsPredArchimedean α
    inst✝¹ : OrderBot α
    inst✝ : DecidableEq α
    r : α
    n : Nat
    h : Eq (Nat.find ⋯) n
    ⊢ Eq (Order.pred (Nat.iterate Order.pred (HSub.hSub n 1) r)) Bot.bot
  -/
  cases n
  · have : Order.pred^[0] r = ⊥ := by
      rw [← h]
      apply Nat.find_spec (bot_le (a := r)).exists_pred_iterate
    /-
      case zero
      α : Type u_1
      inst✝⁴ : PartialOrder α
      inst✝³ : PredOrder α
      inst✝² : IsPredArchimedean α
      inst✝¹ : OrderBot α
      inst✝ : DecidableEq α
      r : α
      h : Eq (Nat.find ⋯) 0
      this : Eq (Nat.iterate Order.pred 0 r) Bot.bot
      ⊢ Eq (Order.pred (Nat.iterate Order.pred (HSub.hSub 0 1) r)) Bot.bot
    -/
    simp only [Function.iterate_zero, id_eq] at this
    /-
      case zero
      α : Type u_1
      inst✝⁴ : PartialOrder α
      inst✝³ : PredOrder α
      inst✝² : IsPredArchimedean α
      inst✝¹ : OrderBot α
      inst✝ : DecidableEq α
      r : α
      h : Eq (Nat.find ⋯) 0
      this : Eq r Bot.bot
      ⊢ Eq (Order.pred (Nat.iterate Order.pred (HSub.hSub 0 1) r)) Bot.bot
    -/
    simp [this]
    /-
      🎉 no goals
    -/
    /-
      case succ
      α : Type u_1
      inst✝⁴ : PartialOrder α
      inst✝³ : PredOrder α
      inst✝² : IsPredArchimedean α
      inst✝¹ : OrderBot α
      inst✝ : DecidableEq α
      r : α
      n✝ : Nat
      h : Eq (Nat.find ⋯) (HAdd.hAdd n✝ 1)
      ⊢ Eq (Order.pred (Nat.iterate Order.pred (HSub.hSub (HAdd.hAdd n✝ 1) 1) r)) Bo …
    -/
  · simp only [Nat.add_sub_cancel_right, ← Function.iterate_succ_apply', Nat.succ_eq_add_one]
    /-
      case succ
      α : Type u_1
      inst✝⁴ : PartialOrder α
      inst✝³ : PredOrder α
      inst✝² : IsPredArchimedean α
      inst✝¹ : OrderBot α
      inst✝ : DecidableEq α
      r : α
      n✝ : Nat
      h : Eq (Nat.find ⋯) (HAdd.hAdd n✝ 1)
      ⊢ Eq (Nat.iterate Order.pred (HAdd.hAdd n✝ 1) r) Bot.bot
    -/
    rw [← h]
    /-
      case succ
      α : Type u_1
      inst✝⁴ : PartialOrder α
      inst✝³ : PredOrder α
      inst✝² : IsPredArchimedean α
      inst✝¹ : OrderBot α
      inst✝ : DecidableEq α
      r : α
      n✝ : Nat
      h : Eq (Nat.find ⋯) (HAdd.hAdd n✝ 1)
      ⊢ Eq (Nat.iterate Order.pred (Nat.find ⋯) r) Bot.bot
    -/
    apply Nat.find_spec (bot_le (a := r)).exists_pred_iterate
    /-
      🎉 no goals
    -/


@[simp]
lemma findAtom_eq_bot {r : α} :
    findAtom r = ⊥ ↔ r = ⊥ where
  mp h := by
    /-
      α : Type u_1
      inst✝⁴ : PartialOrder α
      inst✝³ : PredOrder α
      inst✝² : IsPredArchimedean α
      inst✝¹ : OrderBot α
      inst✝ : DecidableEq α
      r : α
      h : Eq (IsPredArchimedean.findAtom r) Bot.bot
      ⊢ Eq r Bot.bot
    -/
    unfold findAtom at h
    /-
      α : Type u_1
      inst✝⁴ : PartialOrder α
      inst✝³ : PredOrder α
      inst✝² : IsPredArchimedean α
      inst✝¹ : OrderBot α
      inst✝ : DecidableEq α
      r : α
      h : Eq (Nat.iterate Order.pred (HSub.hSub (Nat.find ⋯) 1) r) Bot.bot
      ⊢ Eq r Bot.bot
    -/
    have := Nat.find_min' (bot_le (a := r)).exists_pred_iterate h
    /-
      α : Type u_1
      inst✝⁴ : PartialOrder α
      inst✝³ : PredOrder α
      inst✝² : IsPredArchimedean α
      inst✝¹ : OrderBot α
      inst✝ : DecidableEq α
      r : α
      h : Eq (Nat.iterate Order.pred (HSub.hSub (Nat.find ⋯) 1) r) Bot.bot
      this : LE.le (Nat.find ⋯) (HSub.hSub (Nat.find ⋯) 1)
      ⊢ Eq r Bot.bot
    -/
    replace : Nat.find (bot_le (a := r)).exists_pred_iterate = 0 := by omega
    /-
      α : Type u_1
      inst✝⁴ : PartialOrder α
      inst✝³ : PredOrder α
      inst✝² : IsPredArchimedean α
      inst✝¹ : OrderBot α
      inst✝ : DecidableEq α
      r : α
      h : Eq (Nat.iterate Order.pred (HSub.hSub (Nat.find ⋯) 1) r) Bot.bot
      this : Eq (Nat.find ⋯) 0
      ⊢ Eq r Bot.bot
    -/
    simpa [this] using h
    /-
      🎉 no goals
    -/
              /-
                α : Type u_1
                inst✝⁴ : PartialOrder α
                inst✝³ : PredOrder α
                inst✝² : IsPredArchimedean α
                inst✝¹ : OrderBot α
                inst✝ : DecidableEq α
                r : α
                h : Eq r Bot.bot
                ⊢ Eq (IsPredArchimedean.findAtom r) Bot.bot
              -/
  mpr h := by simp [h]
              /-
                🎉 no goals
              -/


lemma findAtom_ne_bot {r : α} :
    findAtom r ≠ ⊥ ↔ r ≠ ⊥ := findAtom_eq_bot.not


lemma isAtom_findAtom {r : α} (hr : r ≠ ⊥) :
    IsAtom (findAtom r) := by
  /-
    α : Type u_1
    inst✝⁴ : PartialOrder α
    inst✝³ : PredOrder α
    inst✝² : IsPredArchimedean α
    inst✝¹ : OrderBot α
    inst✝ : DecidableEq α
    r : α
    hr : Ne r Bot.bot
    ⊢ IsAtom (IsPredArchimedean.findAtom r)
  -/
  constructor
    /-
      case left
      α : Type u_1
      inst✝⁴ : PartialOrder α
      inst✝³ : PredOrder α
      inst✝² : IsPredArchimedean α
      inst✝¹ : OrderBot α
      inst✝ : DecidableEq α
      r : α
      hr : Ne r Bot.bot
      ⊢ Ne (IsPredArchimedean.findAtom r) Bot.bot
    -/
  · simp [hr]
    /-
      🎉 no goals
    -/
    /-
      case right
      α : Type u_1
      inst✝⁴ : PartialOrder α
      inst✝³ : PredOrder α
      inst✝² : IsPredArchimedean α
      inst✝¹ : OrderBot α
      inst✝ : DecidableEq α
      r : α
      hr : Ne r Bot.bot
      ⊢ ∀ (b : α), LT.lt b (IsPredArchimedean.findAtom r) → Eq b Bot.bot
    -/
  · intro b hb
    /-
      case right
      α : Type u_1
      inst✝⁴ : PartialOrder α
      inst✝³ : PredOrder α
      inst✝² : IsPredArchimedean α
      inst✝¹ : OrderBot α
      inst✝ : DecidableEq α
      r : α
      hr : Ne r Bot.bot
      b : α
      hb : LT.lt b (IsPredArchimedean.findAtom r)
      ⊢ Eq b Bot.bot
    -/
    apply Order.le_pred_of_lt at hb
    /-
      case right
      α : Type u_1
      inst✝⁴ : PartialOrder α
      inst✝³ : PredOrder α
      inst✝² : IsPredArchimedean α
      inst✝¹ : OrderBot α
      inst✝ : DecidableEq α
      r : α
      hr : Ne r Bot.bot
      b : α
      hb : LE.le b (Order.pred (IsPredArchimedean.findAtom r))
      ⊢ Eq b Bot.bot
    -/
    simpa using hb
    /-
      🎉 no goals
    -/


@[simp]
lemma isAtom_findAtom_iff {r : α} :
    IsAtom (findAtom r) ↔ r ≠ ⊥ where
  mpr := isAtom_findAtom
                /-
                  α : Type u_1
                  inst✝⁴ : PartialOrder α
                  inst✝³ : PredOrder α
                  inst✝² : IsPredArchimedean α
                  inst✝¹ : OrderBot α
                  inst✝ : DecidableEq α
                  r : α
                  h : IsAtom (IsPredArchimedean.findAtom r)
                  nh : Eq r Bot.bot
                  ⊢ False
                -/
  mp h nh := by simp only [nh, findAtom_bot] at h; exact h.1 rfl
                                                   /-
                                                     🎉 no goals
                                                   -/


instance instIsAtomic : IsAtomic α where
  eq_bot_or_exists_atom_le b := by classical
    rw [or_iff_not_imp_left]
    intro hb
    use findAtom b, isAtom_findAtom hb, findAtom_le b


/--
The type of rooted trees.
-/
structure RootedTree where
  /-- The type representing the elements in the tree. -/
  α : Type*
  /-- The type should be a `SemilatticeInf`,
    where `inf` is the least common ancestor in the tree. -/
  [semilatticeInf : SemilatticeInf α]
  /-- The type should have a bottom, the root. -/
  [orderBot : OrderBot α]
  /-- The type should have a predecessor for every element, its parent. -/
  [predOrder : PredOrder α]
  /-- The predecessor relationship should be archimedean. -/
  [isPredArchimedean : IsPredArchimedean α]


instance : CoeSort RootedTree Type* := ⟨RootedTree.α⟩


/--
A subtree is represented by its root, therefore this is a type synonym.
-/
def SubRootedTree (t : RootedTree) : Type* := t


/--
The root of a `SubRootedTree`.
-/
def SubRootedTree.root {t : RootedTree} (v : SubRootedTree t) : t := v


/--
The `SubRootedTree` rooted at a given node.
-/
def RootedTree.subtree (t : RootedTree) (r : t) : SubRootedTree t := r


@[simp]
lemma RootedTree.root_subtree (t : RootedTree) (r : t) : (t.subtree r).root = r := rfl


@[simp]
lemma RootedTree.subtree_root (t : RootedTree) (v : SubRootedTree t) : t.subtree v.root = v := rfl


@[ext]
lemma SubRootedTree.ext {t : RootedTree} {v₁ v₂ : SubRootedTree t}
    (h : v₁.root = v₂.root) : v₁ = v₂ := h


instance (t : RootedTree) : SetLike (SubRootedTree t) t where
  coe v := Set.Ici v.root
  coe_injective' a₁ a₂ h := by
    /-
      α : Type u_1
      inst✝² : PartialOrder α
      inst✝¹ : PredOrder α
      inst✝ : IsPredArchimedean α
      t : RootedTree
      a₁ a₂ : SubRootedTree t
      h : Eq ((fun v => Set.Ici v.root) a₁) ((fun v => Set.Ici v.root) a₂)
      ⊢ Eq a₁ a₂
    -/
    simpa only [Set.Ici_inj, ← SubRootedTree.ext_iff] using h
    /-
      🎉 no goals
    -/


lemma SubRootedTree.mem_iff {t : RootedTree} {r : SubRootedTree t} {v : t} :
    v ∈ r ↔ r.root ≤ v := Iff.rfl


/--
The coercion from a `SubRootedTree` to a `RootedTree`.
-/
@[coe, reducible]
noncomputable def SubRootedTree.coeTree {t : RootedTree} (r : SubRootedTree t) : RootedTree where
  α := Set.Ici r.root


noncomputable instance (t : RootedTree) : CoeOut (SubRootedTree t) RootedTree :=
  ⟨SubRootedTree.coeTree⟩


@[simp]
lemma SubRootedTree.bot_mem_iff {t : RootedTree} (r : SubRootedTree t) :
    ⊥ ∈ r ↔ r.root = ⊥ := by
  /-
    t : RootedTree
    r : SubRootedTree t
    ⊢ Iff (Membership.mem r Bot.bot) (Eq r.root Bot.bot)
  -/
  simp [mem_iff]
  /-
    🎉 no goals
  -/


/--
All of the immediate subtrees of a given rooted tree, that is subtrees which are rooted at a direct
child of the root (or, order theoretically, at an atom).
-/
def RootedTree.subtrees (t : RootedTree) : Set (SubRootedTree t) :=
  {x | IsAtom x.root}


lemma SubRootedTree.root_ne_bot_of_mem_subtrees (r : SubRootedTree t) (hr : r ∈ t.subtrees) :
    r.root ≠ ⊥ := by
  /-
    t : RootedTree
    r : SubRootedTree t
    hr : Membership.mem t.subtrees r
    ⊢ Ne r.root Bot.bot
  -/
  simp only [RootedTree.subtrees, Set.mem_setOf_eq] at hr
  /-
    t : RootedTree
    r : SubRootedTree t
    hr : IsAtom r.root
    ⊢ Ne r.root Bot.bot
  -/
  exact hr.1
  /-
    🎉 no goals
  -/


lemma RootedTree.mem_subtrees_disjoint_iff {t₁ t₂ : SubRootedTree t}
    (ht₁ : t₁ ∈ t.subtrees) (ht₂ : t₂ ∈ t.subtrees) (v₁ v₂ : t) (h₁ : v₁ ∈ t₁)
    (h₂ : v₂ ∈ t₂) :
    Disjoint v₁ v₂ ↔ t₁ ≠ t₂ where
  mp h := by
    /-
      t : RootedTree
      t₁ t₂ : SubRootedTree t
      ht₁ : Membership.mem t.subtrees t₁
      ht₂ : Membership.mem t.subtrees t₂
      v₁ v₂ : ↑t
      h₁ : Membership.mem t₁ v₁
      h₂ : Membership.mem t₂ v₂
      h : Disjoint v₁ v₂
      ⊢ Ne t₁ t₂
    -/
    intro nh
    have : t₁.root ≤ (v₁ : t) ⊓ (v₂ : t) := by
      simp only [le_inf_iff]
      exact ⟨h₁, nh ▸ h₂⟩
    /-
      t : RootedTree
      t₁ t₂ : SubRootedTree t
      ht₁ : Membership.mem t.subtrees t₁
      ht₂ : Membership.mem t.subtrees t₂
      v₁ v₂ : ↑t
      h₁ : Membership.mem t₁ v₁
      h₂ : Membership.mem t₂ v₂
      h : Disjoint v₁ v₂
      nh : Eq t₁ t₂
      this : LE.le t₁.root (Min.min v₁ v₂)
      ⊢ False
    -/
    rw [h.eq_bot] at this
    /-
      t : RootedTree
      t₁ t₂ : SubRootedTree t
      ht₁ : Membership.mem t.subtrees t₁
      ht₂ : Membership.mem t.subtrees t₂
      v₁ v₂ : ↑t
      h₁ : Membership.mem t₁ v₁
      h₂ : Membership.mem t₂ v₂
      h : Disjoint v₁ v₂
      nh : Eq t₁ t₂
      this : LE.le t₁.root Bot.bot
      ⊢ False
    -/
    simp only [le_bot_iff] at this
    /-
      t : RootedTree
      t₁ t₂ : SubRootedTree t
      ht₁ : Membership.mem t.subtrees t₁
      ht₂ : Membership.mem t.subtrees t₂
      v₁ v₂ : ↑t
      h₁ : Membership.mem t₁ v₁
      h₂ : Membership.mem t₂ v₂
      h : Disjoint v₁ v₂
      nh : Eq t₁ t₂
      this : Eq t₁.root Bot.bot
      ⊢ False
    -/
    exact t₁.root_ne_bot_of_mem_subtrees ht₁ this
    /-
      🎉 no goals
    -/
  mpr h := by
    /-
      t : RootedTree
      t₁ t₂ : SubRootedTree t
      ht₁ : Membership.mem t.subtrees t₁
      ht₂ : Membership.mem t.subtrees t₂
      v₁ v₂ : ↑t
      h₁ : Membership.mem t₁ v₁
      h₂ : Membership.mem t₂ v₂
      h : Ne t₁ t₂
      ⊢ Disjoint v₁ v₂
    -/
    rw [SubRootedTree.mem_iff] at h₁ h₂
    /-
      t : RootedTree
      t₁ t₂ : SubRootedTree t
      ht₁ : Membership.mem t.subtrees t₁
      ht₂ : Membership.mem t.subtrees t₂
      v₁ v₂ : ↑t
      h₁ : LE.le t₁.root v₁
      h₂ : LE.le t₂.root v₂
      h : Ne t₁ t₂
      ⊢ Disjoint v₁ v₂
    -/
    contrapose! h
    /-
      t : RootedTree
      t₁ t₂ : SubRootedTree t
      ht₁ : Membership.mem t.subtrees t₁
      ht₂ : Membership.mem t.subtrees t₂
      v₁ v₂ : ↑t
      h₁ : LE.le t₁.root v₁
      h₂ : LE.le t₂.root v₂
      h : Not (Disjoint v₁ v₂)
      ⊢ Eq t₁ t₂
    -/
    rw [disjoint_iff, ← ne_eq, ← bot_lt_iff_ne_bot] at h
    /-
      t : RootedTree
      t₁ t₂ : SubRootedTree t
      ht₁ : Membership.mem t.subtrees t₁
      ht₂ : Membership.mem t.subtrees t₂
      v₁ v₂ : ↑t
      h₁ : LE.le t₁.root v₁
      h₂ : LE.le t₂.root v₂
      h : LT.lt Bot.bot (Min.min v₁ v₂)
      ⊢ Eq t₁ t₂
    -/
    rcases lt_or_le_of_directed (by simp : v₁ ⊓ v₂ ≤ v₁) h₁ with oh | oh
      /-
        case inl
        t : RootedTree
        t₁ t₂ : SubRootedTree t
        ht₁ : Membership.mem t.subtrees t₁
        ht₂ : Membership.mem t.subtrees t₂
        v₁ v₂ : ↑t
        h₁ : LE.le t₁.root v₁
        h₂ : LE.le t₂.root v₂
        h : LT.lt Bot.bot (Min.min v₁ v₂)
        oh : LT.lt (Min.min v₁ v₂) t₁.root
        ⊢ Eq t₁ t₂
      -/
    · simp_all [RootedTree.subtrees, IsAtom.lt_iff]
      /-
        🎉 no goals
      -/
    /-
      case inr
      t : RootedTree
      t₁ t₂ : SubRootedTree t
      ht₁ : Membership.mem t.subtrees t₁
      ht₂ : Membership.mem t.subtrees t₂
      v₁ v₂ : ↑t
      h₁ : LE.le t₁.root v₁
      h₂ : LE.le t₂.root v₂
      h : LT.lt Bot.bot (Min.min v₁ v₂)
      oh : LE.le t₁.root (Min.min v₁ v₂)
      ⊢ Eq t₁ t₂
    -/
    rw [le_inf_iff] at oh
    /-
      case inr
      t : RootedTree
      t₁ t₂ : SubRootedTree t
      ht₁ : Membership.mem t.subtrees t₁
      ht₂ : Membership.mem t.subtrees t₂
      v₁ v₂ : ↑t
      h₁ : LE.le t₁.root v₁
      h₂ : LE.le t₂.root v₂
      h : LT.lt Bot.bot (Min.min v₁ v₂)
      oh : And (LE.le t₁.root v₁) (LE.le t₁.root v₂)
      ⊢ Eq t₁ t₂
    -/
    ext
    simpa only [ht₂.le_iff_eq ht₁.1, ht₁.le_iff_eq ht₂.1, eq_comm, or_self] using
      le_total_of_directed oh.2 h₂


lemma RootedTree.subtrees_disjoint : t.subtrees.PairwiseDisjoint ((↑) : _ → Set t) := by
  /-
    t : RootedTree
    ⊢ t.subtrees.PairwiseDisjoint SetLike.coe
  -/
  intro t₁ ht₁ t₂ ht₂ h
  /-
    t : RootedTree
    t₁ : SubRootedTree t
    ht₁ : Membership.mem t.subtrees t₁
    t₂ : SubRootedTree t
    ht₂ : Membership.mem t.subtrees t₂
    h : Ne t₁ t₂
    ⊢ Function.onFun Disjoint SetLike.coe t₁ t₂
  -/
  rw [Function.onFun_apply, Set.disjoint_left]
  /-
    t : RootedTree
    t₁ : SubRootedTree t
    ht₁ : Membership.mem t.subtrees t₁
    t₂ : SubRootedTree t
    ht₂ : Membership.mem t.subtrees t₂
    h : Ne t₁ t₂
    ⊢ ∀ ⦃a : ↑t⦄, Membership.mem (↑t₁) a → Not (Membership.mem (↑t₂) a)
  -/
  intro a ha hb
  /-
    t : RootedTree
    t₁ : SubRootedTree t
    ht₁ : Membership.mem t.subtrees t₁
    t₂ : SubRootedTree t
    ht₂ : Membership.mem t.subtrees t₂
    h : Ne t₁ t₂
    a : ↑t
    ha : Membership.mem (↑t₁) a
    hb : Membership.mem (↑t₂) a
    ⊢ False
  -/
  rw [← mem_subtrees_disjoint_iff ht₁ ht₂ a a ha hb, disjoint_self] at h
  /-
    t : RootedTree
    t₁ : SubRootedTree t
    ht₁ : Membership.mem t.subtrees t₁
    t₂ : SubRootedTree t
    ht₂ : Membership.mem t.subtrees t₂
    a : ↑t
    h : Eq a Bot.bot
    ha : Membership.mem (↑t₁) a
    hb : Membership.mem (↑t₂) a
    ⊢ False
  -/
  subst h
  /-
    t : RootedTree
    t₁ : SubRootedTree t
    ht₁ : Membership.mem t.subtrees t₁
    t₂ : SubRootedTree t
    ht₂ : Membership.mem t.subtrees t₂
    ha : Membership.mem (↑t₁) Bot.bot
    hb : Membership.mem (↑t₂) Bot.bot
    ⊢ False
  -/
  simp only [SetLike.mem_coe, SubRootedTree.bot_mem_iff] at ha
  /-
    t : RootedTree
    t₁ : SubRootedTree t
    ht₁ : Membership.mem t.subtrees t₁
    t₂ : SubRootedTree t
    ht₂ : Membership.mem t.subtrees t₂
    hb : Membership.mem (↑t₂) Bot.bot
    ha : Eq t₁.root Bot.bot
    ⊢ False
  -/
  exact t₁.root_ne_bot_of_mem_subtrees ht₁ ha
  /-
    🎉 no goals
  -/


/--
The immediate subtree of `t` containing `v`, or all of `t` if `v` is the root.
-/
def RootedTree.subtreeOf (t : RootedTree) [DecidableEq t] (v : t) : SubRootedTree t :=
  t.subtree (IsPredArchimedean.findAtom v)


@[simp]
lemma RootedTree.mem_subtreeOf [DecidableEq t] {v : t} :
    v ∈ t.subtreeOf v := by
  /-
    t : RootedTree
    inst✝ : DecidableEq ↑t
    v : ↑t
    ⊢ Membership.mem (t.subtreeOf v) v
  -/
  simp [SubRootedTree.mem_iff, RootedTree.subtreeOf]
  /-
    🎉 no goals
  -/


lemma RootedTree.subtreeOf_mem_subtrees [DecidableEq t] {v : t} (hr : v ≠ ⊥) :
    t.subtreeOf v ∈ t.subtrees := by
  /-
    t : RootedTree
    inst✝ : DecidableEq ↑t
    v : ↑t
    hr : Ne v Bot.bot
    ⊢ Membership.mem t.subtrees (t.subtreeOf v)
  -/
  simpa [RootedTree.subtrees, RootedTree.subtreeOf]
  /-
    🎉 no goals
  -/

