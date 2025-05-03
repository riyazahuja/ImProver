/-- A binary tree with values stored in non-leaf nodes. -/
inductive Tree.{u} (α : Type u) : Type u
  | nil : Tree α
  | node : α → Tree α → Tree α → Tree α
  deriving DecidableEq, Repr -- Porting note: Removed `has_reflect`, added `Repr`.


instance : Inhabited (Tree α) :=
  ⟨nil⟩


/-- Apply a function to each value in the tree.  This is the `map` function for the `Tree` functor.
-/
def map {β} (f : α → β) : Tree α → Tree β
  | nil => nil
  | node a l r => node (f a) (map f l) (map f r)


/-- The number of internal nodes (i.e. not including leaves) of a binary tree -/
@[simp]
def numNodes : Tree α → ℕ
  | nil => 0
  | node _ a b => a.numNodes + b.numNodes + 1


/-- The number of leaves of a binary tree -/
@[simp]
def numLeaves : Tree α → ℕ
  | nil => 1
  | node _ a b => a.numLeaves + b.numLeaves


/-- The height - length of the longest path from the root - of a binary tree -/
@[simp]
def height : Tree α → ℕ
  | nil => 0
  | node _ a b => max a.height b.height + 1


theorem numLeaves_eq_numNodes_succ (x : Tree α) : x.numLeaves = x.numNodes + 1 := by
  /-
    α : Type u
    x : Tree α
    ⊢ Eq x.numLeaves (HAdd.hAdd x.numNodes 1)
  -/
                  /-
                    🎉 no goals
                  -/
  induction x <;> simp [*, Nat.add_comm, Nat.add_assoc, Nat.add_left_comm]
                  /-
                    🎉 no goals
                  -/


theorem numLeaves_pos (x : Tree α) : 0 < x.numLeaves := by
  /-
    α : Type u
    x : Tree α
    ⊢ LT.lt 0 x.numLeaves
  -/
  rw [numLeaves_eq_numNodes_succ]
  /-
    α : Type u
    x : Tree α
    ⊢ LT.lt 0 (HAdd.hAdd x.numNodes 1)
  -/
  exact x.numNodes.zero_lt_succ
  /-
    🎉 no goals
  -/


theorem height_le_numNodes : ∀ x : Tree α, x.height ≤ x.numNodes
  | nil => Nat.le_refl _
  | node _ a b => Nat.succ_le_succ <|
    Nat.max_le.2 ⟨Nat.le_trans a.height_le_numNodes <| a.numNodes.le_add_right _,
      Nat.le_trans b.height_le_numNodes <| b.numNodes.le_add_left _⟩


/-- The left child of the tree, or `nil` if the tree is `nil` -/
@[simp]
def left : Tree α → Tree α
  | nil => nil
  | node _ l _r => l


/-- The right child of the tree, or `nil` if the tree is `nil` -/
@[simp]
def right : Tree α → Tree α
  | nil => nil
  | node _ _l r => r


/-- A node with `Unit` data -/
scoped infixr:65 " △ " => Tree.node ()

-- Porting note: workaround for https://github.com/leanprover/lean4/issues/2049

compile_inductive% Tree


@[elab_as_elim]
def unitRecOn {motive : Tree Unit → Sort*} (t : Tree Unit) (base : motive nil)
    (ind : ∀ x y, motive x → motive y → motive (x △ y)) : motive t :=
  t.recOn base fun _u ↦ ind


theorem left_node_right_eq_self : ∀ {x : Tree Unit} (_hx : x ≠ nil), x.left △ x.right = x
                 /-
                   h : Ne Tree.nil Tree.nil
                   ⊢ Eq (Tree.node Unit.unit Tree.nil.left Tree.nil.right) Tree.nil
                 -/
  | nil, h => by trivial
                 /-
                   🎉 no goals
                 -/
  | node _ _ _, _ => rfl  -- Porting note: `a △ b` no longer works in pattern matching


