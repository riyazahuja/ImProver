theorem not_le_delta {s} (H : 1 ≤ s) : ¬s ≤ delta * 0 :=
  not_le_of_gt H


theorem delta_lt_false {a b : ℕ} (h₁ : delta * a < b) (h₂ : delta * b < a) : False :=
                                               /-
                                                 a b : Nat
                                                 h₁ : LT.lt (HMul.hMul Ordnode.delta a) b
                                                 h₂ : LT.lt (HMul.hMul Ordnode.delta b) a
                                                 ⊢ LT.lt 0 Ordnode.delta
                                               -/
  not_le_of_lt (lt_trans ((mul_lt_mul_left (by decide)).2 h₁) h₂) <| by
                                               /-
                                                 🎉 no goals
                                               -/
    /-
      a b : Nat
      h₁ : LT.lt (HMul.hMul Ordnode.delta a) b
      h₂ : LT.lt (HMul.hMul Ordnode.delta b) a
      ⊢ LE.le a (HMul.hMul Ordnode.delta (HMul.hMul Ordnode.delta a))
    -/
    simpa [mul_assoc] using Nat.mul_le_mul_right a (by decide : 1 ≤ delta * delta)
    /-
      🎉 no goals
    -/


/-- O(n). Computes the actual number of elements in the set, ignoring the cached `size` field. -/
def realSize : Ordnode α → ℕ
  | nil => 0
  | node _ l _ r => realSize l + realSize r + 1


/-- The `Sized` property asserts that all the `size` fields in nodes match the actual size of the
respective subtrees. -/
def Sized : Ordnode α → Prop
  | nil => True
  | node s l _ r => s = size l + size r + 1 ∧ Sized l ∧ Sized r


theorem Sized.node' {l x r} (hl : @Sized α l) (hr : Sized r) : Sized (node' l x r) :=
  ⟨rfl, hl, hr⟩


theorem Sized.eq_node' {s l x r} (h : @Sized α (node s l x r)) : node s l x r = .node' l x r := by
  /-
    α : Type u_1
    s : Nat
    l : Ordnode α
    x : α
    r : Ordnode α
    h : (Ordnode.node s l x r).Sized
    ⊢ Eq (Ordnode.node s l x r) (l.node' x r)
  -/
  rw [h.1]
  /-
    🎉 no goals
  -/


theorem Sized.size_eq {s l x r} (H : Sized (@node α s l x r)) :
    size (@node α s l x r) = size l + size r + 1 :=
  H.1


@[elab_as_elim]
theorem Sized.induction {t} (hl : @Sized α t) {C : Ordnode α → Prop} (H0 : C nil)
    (H1 : ∀ l x r, C l → C r → C (.node' l x r)) : C t := by
  induction t with
  | nil => exact H0
  | node _ _ _ _ t_ih_l t_ih_r =>
    rw [hl.eq_node']
    exact H1 _ _ _ (t_ih_l hl.2.1) (t_ih_r hl.2.2)


theorem size_eq_realSize : ∀ {t : Ordnode α}, Sized t → size t = realSize t
  | nil, _ => rfl
  | node s l x r, ⟨h₁, h₂, h₃⟩ => by
    /-
      α : Type u_1
      s : Nat
      l : Ordnode α
      x : α
      r : Ordnode α
      h₁ : Eq s (HAdd.hAdd (HAdd.hAdd l.size r.size) 1)
      h₂ : l.Sized
      h₃ : r.Sized
      ⊢ Eq (Ordnode.node s l x r).size (Ordnode.node s l x r).realSize
    -/
    rw [size, h₁, size_eq_realSize h₂, size_eq_realSize h₃]; rfl
                                                             /-
                                                               🎉 no goals
                                                             -/


@[simp]
theorem Sized.size_eq_zero {t : Ordnode α} (ht : Sized t) : size t = 0 ↔ t = nil := by
  /-
    α : Type u_1
    t : Ordnode α
    ht : t.Sized
    ⊢ Iff (Eq t.size 0) (Eq t Ordnode.nil)
  -/
  cases t <;> [simp;simp [ht.1]]
  /-
    🎉 no goals
  -/


theorem Sized.pos {s l x r} (h : Sized (@node α s l x r)) : 0 < s := by
  /-
    α : Type u_1
    s : Nat
    l : Ordnode α
    x : α
    r : Ordnode α
    h : (Ordnode.node s l x r).Sized
    ⊢ LT.lt 0 s
  -/
  rw [h.1]; apply Nat.le_add_left
            /-
              🎉 no goals
            -/


theorem dual_dual : ∀ t : Ordnode α, dual (dual t) = t
  | nil => rfl
                       /-
                         α : Type u_1
                         s : Nat
                         l : Ordnode α
                         x : α
                         r : Ordnode α
                         ⊢ Eq (Ordnode.node s l x r).dual.dual (Ordnode.node s l x r)
                       -/
  | node s l x r => by rw [dual, dual, dual_dual l, dual_dual r]
                       /-
                         🎉 no goals
                       -/


@[simp]
                                                                 /-
                                                                   α : Type u_1
                                                                   t : Ordnode α
                                                                   ⊢ Eq t.dual.size t.size
                                                                 -/
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
theorem size_dual (t : Ordnode α) : size (dual t) = size t := by cases t <;> rfl
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


/-- The `BalancedSz l r` asserts that a hypothetical tree with children of sizes `l` and `r` is
balanced: either `l ≤ δ * r` and `r ≤ δ * r`, or the tree is trivial with a singleton on one side
and nothing on the other. -/
def BalancedSz (l r : ℕ) : Prop :=
  l + r ≤ 1 ∨ l ≤ delta * r ∧ r ≤ delta * l


instance BalancedSz.dec : DecidableRel BalancedSz := fun _ _ => inferInstanceAs (Decidable (_ ∨ _))


/-- The `Balanced t` asserts that the tree `t` satisfies the balance invariants
(at every level). -/
def Balanced : Ordnode α → Prop
  | nil => True
  | node _ l _ r => BalancedSz (size l) (size r) ∧ Balanced l ∧ Balanced r


instance Balanced.dec : DecidablePred (@Balanced α)
  | nil => by
    /-
      α : Type u_1
      ⊢ Decidable Ordnode.nil.Balanced
    -/
    unfold Balanced
    /-
      α : Type u_1
      ⊢ Decidable True
    -/
    infer_instance
    /-
      🎉 no goals
    -/
  | node _ l _ r => by
    /-
      α : Type u_1
      size✝ : Nat
      l : Ordnode α
      x✝ : α
      r : Ordnode α
      ⊢ Decidable (Ordnode.node size✝ l x✝ r).Balanced
    -/
    unfold Balanced
    /-
      α : Type u_1
      size✝ : Nat
      l : Ordnode α
      x✝ : α
      r : Ordnode α
      ⊢ Decidable (And (Ordnode.BalancedSz l.size r.size) (And l.Balanced r.Balanced))
    -/
    haveI := Balanced.dec l
    /-
      α : Type u_1
      size✝ : Nat
      l : Ordnode α
      x✝ : α
      r : Ordnode α
      this : Decidable l.Balanced
      ⊢ Decidable (And (Ordnode.BalancedSz l.size r.size) (And l.Balanced r.Balanced))
    -/
    haveI := Balanced.dec r
    /-
      α : Type u_1
      size✝ : Nat
      l : Ordnode α
      x✝ : α
      r : Ordnode α
      this✝ : Decidable l.Balanced
      this : Decidable r.Balanced
      ⊢ Decidable (And (Ordnode.BalancedSz l.size r.size) (And l.Balanced r.Balanced))
    -/
    infer_instance
    /-
      🎉 no goals
    -/


@[symm]
theorem BalancedSz.symm {l r : ℕ} : BalancedSz l r → BalancedSz r l :=
             /-
               l r : Nat
               ⊢ LE.le (HAdd.hAdd l r) 1 → LE.le (HAdd.hAdd r l) 1
             -/
  Or.imp (by rw [add_comm]; exact id) And.symm
                            /-
                              🎉 no goals
                            -/


theorem balancedSz_zero {l : ℕ} : BalancedSz l 0 ↔ l ≤ 1 := by
  /-
    l : Nat
    ⊢ Iff (Ordnode.BalancedSz l 0) (LE.le l 1)
  -/
  simp +contextual [BalancedSz]
  /-
    🎉 no goals
  -/


theorem balancedSz_up {l r₁ r₂ : ℕ} (h₁ : r₁ ≤ r₂) (h₂ : l + r₂ ≤ 1 ∨ r₂ ≤ delta * l)
    (H : BalancedSz l r₁) : BalancedSz l r₂ := by
  /-
    l r₁ r₂ : Nat
    h₁ : LE.le r₁ r₂
    h₂ : Or (LE.le (HAdd.hAdd l r₂) 1) (LE.le r₂ (HMul.hMul Ordnode.delta l))
    H : Ordnode.BalancedSz l r₁
    ⊢ Ordnode.BalancedSz l r₂
  -/
  refine or_iff_not_imp_left.2 fun h => ?_
  /-
    l r₁ r₂ : Nat
    h₁ : LE.le r₁ r₂
    h₂ : Or (LE.le (HAdd.hAdd l r₂) 1) (LE.le r₂ (HMul.hMul Ordnode.delta l))
    H : Ordnode.BalancedSz l r₁
    h : Not (LE.le (HAdd.hAdd l r₂) 1)
    ⊢ And (LE.le l (HMul.hMul Ordnode.delta r₂)) (LE.le r₂ (HMul.hMul Ordnode.delt …
  -/
  refine ⟨?_, h₂.resolve_left h⟩
  cases H with
  | inl H =>
    cases r₂
    · cases h (le_trans (Nat.add_le_add_left (Nat.zero_le _) _) H)
    · exact le_trans (le_trans (Nat.le_add_right _ _) H) (Nat.le_add_left 1 _)
  | inr H =>
    exact le_trans H.1 (Nat.mul_le_mul_left _ h₁)


theorem balancedSz_down {l r₁ r₂ : ℕ} (h₁ : r₁ ≤ r₂) (h₂ : l + r₂ ≤ 1 ∨ l ≤ delta * r₁)
    (H : BalancedSz l r₂) : BalancedSz l r₁ :=
  have : l + r₂ ≤ 1 → BalancedSz l r₁ := fun H => Or.inl (le_trans (Nat.add_le_add_left h₁ _) H)
  Or.casesOn H this fun H => Or.casesOn h₂ this fun h₂ => Or.inr ⟨h₂, le_trans h₁ H.2⟩


theorem Balanced.dual : ∀ {t : Ordnode α}, Balanced t → Balanced (dual t)
  | nil, _ => ⟨⟩
                                     /-
                                       α : Type u_1
                                       size✝ : Nat
                                       l : Ordnode α
                                       x✝ : α
                                       r : Ordnode α
                                       b : Ordnode.BalancedSz l.size r.size
                                       bl : l.Balanced
                                       br : r.Balanced
                                       ⊢ Ordnode.BalancedSz r.dual.size l.dual.size
                                     -/
  | node _ l _ r, ⟨b, bl, br⟩ => ⟨by rw [size_dual, size_dual]; exact b.symm, br.dual, bl.dual⟩
                                                                /-
                                                                  🎉 no goals
                                                                -/


/-- Build a tree from three nodes, left associated (ignores the invariants). -/
def node3L (l : Ordnode α) (x : α) (m : Ordnode α) (y : α) (r : Ordnode α) : Ordnode α :=
  node' (node' l x m) y r


/-- Build a tree from three nodes, right associated (ignores the invariants). -/
def node3R (l : Ordnode α) (x : α) (m : Ordnode α) (y : α) (r : Ordnode α) : Ordnode α :=
  node' l x (node' m y r)


/-- Build a tree from three nodes, with `a () b -> (a ()) b` and `a (b c) d -> ((a b) (c d))`. -/
def node4L : Ordnode α → α → Ordnode α → α → Ordnode α → Ordnode α
  | l, x, node _ ml y mr, z, r => node' (node' l x ml) y (node' mr z r)
  | l, x, nil, z, r => node3L l x nil z r

-- should not happen

/-- Build a tree from three nodes, with `a () b -> a (() b)` and `a (b c) d -> ((a b) (c d))`. -/
def node4R : Ordnode α → α → Ordnode α → α → Ordnode α → Ordnode α
  | l, x, node _ ml y mr, z, r => node' (node' l x ml) y (node' mr z r)
  | l, x, nil, z, r => node3R l x nil z r

-- should not happen

/-- Concatenate two nodes, performing a left rotation `x (y z) -> ((x y) z)`
if balance is upset. -/
def rotateL : Ordnode α → α → Ordnode α → Ordnode α
  | l, x, node _ m y r => if size m < ratio * size r then node3L l x m y r else node4L l x m y r
  | l, x, nil => node' l x nil

-- Porting note (https://github.com/leanprover-community/mathlib4/pull/11467): during the port we marked these lemmas with `@[eqns]`
-- to emulate the old Lean 3 behaviour.


theorem rotateL_node (l : Ordnode α) (x : α) (sz : ℕ) (m : Ordnode α) (y : α) (r : Ordnode α) :
    rotateL l x (node sz m y r) =
      if size m < ratio * size r then node3L l x m y r else node4L l x m y r :=
  rfl


theorem rotateL_nil (l : Ordnode α) (x : α) : rotateL l x nil = node' l x nil :=
  rfl

-- should not happen

/-- Concatenate two nodes, performing a right rotation `(x y) z -> (x (y z))`
if balance is upset. -/
def rotateR : Ordnode α → α → Ordnode α → Ordnode α
  | node _ l x m, y, r => if size m < ratio * size l then node3R l x m y r else node4R l x m y r
  | nil, y, r => node' nil y r

-- Porting note (https://github.com/leanprover-community/mathlib4/pull/11467): during the port we marked these lemmas with `@[eqns]`
-- to emulate the old Lean 3 behaviour.


theorem rotateR_node (sz : ℕ) (l : Ordnode α) (x : α) (m : Ordnode α) (y : α) (r : Ordnode α) :
    rotateR (node sz l x m) y r =
      if size m < ratio * size l then node3R l x m y r else node4R l x m y r :=
  rfl


theorem rotateR_nil (y : α) (r : Ordnode α) : rotateR nil y r = node' nil y r :=
  rfl

-- should not happen

/-- A left balance operation. This will rebalance a concatenation, assuming the original nodes are
not too far from balanced. -/
def balanceL' (l : Ordnode α) (x : α) (r : Ordnode α) : Ordnode α :=
  if size l + size r ≤ 1 then node' l x r
  else if size l > delta * size r then rotateR l x r else node' l x r


/-- A right balance operation. This will rebalance a concatenation, assuming the original nodes are
not too far from balanced. -/
def balanceR' (l : Ordnode α) (x : α) (r : Ordnode α) : Ordnode α :=
  if size l + size r ≤ 1 then node' l x r
  else if size r > delta * size l then rotateL l x r else node' l x r


/-- The full balance operation. This is the same as `balance`, but with less manual inlining.
It is somewhat easier to work with this version in proofs. -/
def balance' (l : Ordnode α) (x : α) (r : Ordnode α) : Ordnode α :=
  if size l + size r ≤ 1 then node' l x r
  else
    if size r > delta * size l then rotateL l x r
    else if size l > delta * size r then rotateR l x r else node' l x r


theorem dual_node' (l : Ordnode α) (x : α) (r : Ordnode α) :
                                                         /-
                                                           α : Type u_1
                                                           l : Ordnode α
                                                           x : α
                                                           r : Ordnode α
                                                           ⊢ Eq (l.node' x r).dual (r.dual.node' x l.dual)
                                                         -/
    dual (node' l x r) = node' (dual r) x (dual l) := by simp [node', add_comm]
                                                         /-
                                                           🎉 no goals
                                                         -/


theorem dual_node3L (l : Ordnode α) (x : α) (m : Ordnode α) (y : α) (r : Ordnode α) :
    dual (node3L l x m y r) = node3R (dual r) y (dual m) x (dual l) := by
  /-
    α : Type u_1
    l : Ordnode α
    x : α
    m : Ordnode α
    y : α
    r : Ordnode α
    ⊢ Eq (l.node3L x m y r).dual (r.dual.node3R y m.dual x l.dual)
  -/
  simp [node3L, node3R, dual_node', add_comm]
  /-
    🎉 no goals
  -/


theorem dual_node3R (l : Ordnode α) (x : α) (m : Ordnode α) (y : α) (r : Ordnode α) :
    dual (node3R l x m y r) = node3L (dual r) y (dual m) x (dual l) := by
  /-
    α : Type u_1
    l : Ordnode α
    x : α
    m : Ordnode α
    y : α
    r : Ordnode α
    ⊢ Eq (l.node3R x m y r).dual (r.dual.node3L y m.dual x l.dual)
  -/
  simp [node3L, node3R, dual_node', add_comm]
  /-
    🎉 no goals
  -/


theorem dual_node4L (l : Ordnode α) (x : α) (m : Ordnode α) (y : α) (r : Ordnode α) :
    dual (node4L l x m y r) = node4R (dual r) y (dual m) x (dual l) := by
  /-
    α : Type u_1
    l : Ordnode α
    x : α
    m : Ordnode α
    y : α
    r : Ordnode α
    ⊢ Eq (l.node4L x m y r).dual (r.dual.node4R y m.dual x l.dual)
  -/
              /-
                🎉 no goals
              -/
  cases m <;> simp [node4L, node4R, node3R, dual_node3L, dual_node', add_comm]
              /-
                🎉 no goals
              -/


theorem dual_node4R (l : Ordnode α) (x : α) (m : Ordnode α) (y : α) (r : Ordnode α) :
    dual (node4R l x m y r) = node4L (dual r) y (dual m) x (dual l) := by
  /-
    α : Type u_1
    l : Ordnode α
    x : α
    m : Ordnode α
    y : α
    r : Ordnode α
    ⊢ Eq (l.node4R x m y r).dual (r.dual.node4L y m.dual x l.dual)
  -/
              /-
                🎉 no goals
              -/
  cases m <;> simp [node4L, node4R, node3L, dual_node3R, dual_node', add_comm]
              /-
                🎉 no goals
              -/


theorem dual_rotateL (l : Ordnode α) (x : α) (r : Ordnode α) :
    dual (rotateL l x r) = rotateR (dual r) x (dual l) := by
  /-
    α : Type u_1
    l : Ordnode α
    x : α
    r : Ordnode α
    ⊢ Eq (l.rotateL x r).dual (r.dual.rotateR x l.dual)
  -/
              /-
                🎉 no goals
              -/
  cases r <;> simp [rotateL, rotateR, dual_node']; split_ifs <;>
    /-
      case pos
      α : Type u_1
      l : Ordnode α
      x : α
      size✝ : Nat
      l✝ : Ordnode α
      x✝ : α
      r✝ : Ordnode α
      h✝ : LT.lt l✝.size (HMul.hMul Ordnode.ratio r✝.size)
      ⊢ Eq (l.node3L x l✝ x✝ r✝).dual (r✝.dual.node3R x✝ l✝.dual x l.dual)
    -/
    /-
      🎉 no goals
    -/
    simp [dual_node3L, dual_node4L, node3R, add_comm]
    /-
      🎉 no goals
    -/


theorem dual_rotateR (l : Ordnode α) (x : α) (r : Ordnode α) :
    dual (rotateR l x r) = rotateL (dual r) x (dual l) := by
  /-
    α : Type u_1
    l : Ordnode α
    x : α
    r : Ordnode α
    ⊢ Eq (l.rotateR x r).dual (r.dual.rotateL x l.dual)
  -/
  rw [← dual_dual (rotateL _ _ _), dual_rotateL, dual_dual, dual_dual]
  /-
    🎉 no goals
  -/


theorem dual_balance' (l : Ordnode α) (x : α) (r : Ordnode α) :
    dual (balance' l x r) = balance' (dual r) x (dual l) := by
  /-
    α : Type u_1
    l : Ordnode α
    x : α
    r : Ordnode α
    ⊢ Eq (l.balance' x r).dual (r.dual.balance' x l.dual)
  -/
  simp [balance', add_comm]; split_ifs with h h_1 h_2 <;>
    /-
      case pos
      α : Type u_1
      l : Ordnode α
      x : α
      r : Ordnode α
      h : LE.le (HAdd.hAdd l.size r.size) 1
      ⊢ Eq (l.node' x r).dual (r.dual.node' x l.dual)
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
    simp [dual_node', dual_rotateL, dual_rotateR, add_comm]
    /-
      🎉 no goals
    -/
  /-
    case pos
    α : Type u_1
    l : Ordnode α
    x : α
    r : Ordnode α
    h : Not (LE.le (HAdd.hAdd l.size r.size) 1)
    h_1 : LT.lt (HMul.hMul Ordnode.delta l.size) r.size
    h_2 : LT.lt (HMul.hMul Ordnode.delta r.size) l.size
    ⊢ Eq (r.dual.rotateR x l.dual) (r.dual.rotateL x l.dual)
  -/
  cases delta_lt_false h_1 h_2
  /-
    🎉 no goals
  -/


theorem dual_balanceL (l : Ordnode α) (x : α) (r : Ordnode α) :
    dual (balanceL l x r) = balanceR (dual r) x (dual l) := by
  /-
    α : Type u_1
    l : Ordnode α
    x : α
    r : Ordnode α
    ⊢ Eq (l.balanceL x r).dual (r.dual.balanceR x l.dual)
  -/
  unfold balanceL balanceR
  /-
    α : Type u_1
    l : Ordnode α
    x : α
    r : Ordnode α
    ⊢ Eq (Ordnode.casesOn (id r) (Ordnode.casesOn (id l) (Ordnode.singleton x) fun …
  -/
  cases' r with rs rl rx rr
    /-
      case nil
      α : Type u_1
      l : Ordnode α
      x : α
      ⊢ Eq (Ordnode.casesOn (id Ordnode.nil) (Ordnode.casesOn (id l) (Ordnode.single …
    -/
  · cases' l with ls ll lx lr; · rfl
                                 /-
                                   🎉 no goals
                                 -/
    /-
      case nil.node
      α : Type u_1
      x : α
      ls : Nat
      ll : Ordnode α
      lx : α
      lr : Ordnode α
      ⊢ Eq (Ordnode.casesOn (id Ordnode.nil) (Ordnode.casesOn (id (Ordnode.node ls l …
    -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
    cases' ll with lls lll llx llr <;> cases' lr with lrs lrl lrx lrr <;> dsimp only [dual, id] <;>
      /-
        case nil.node.nil.node
        α : Type u_1
        x : α
        ls : Nat
        lx : α
        lrs : Nat
        lrl : Ordnode α
        lrx : α
        lrr : Ordnode α
        ⊢ Eq (Ordnode.node 3 (Ordnode.node 1 Ordnode.nil x Ordnode.nil) lrx (Ordnode.n …
      -/
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
      try rfl
    /-
      case nil.node.node.node
      α : Type u_1
      x : α
      ls : Nat
      lx : α
      lls : Nat
      lll : Ordnode α
      llx : α
      llr : Ordnode α
      lrs : Nat
      lrl : Ordnode α
      lrx : α
      lrr : Ordnode α
      ⊢ Eq (ite (LT.lt lrs (HMul.hMul Ordnode.ratio lls)) (Ordnode.node (HAdd.hAdd l …
    -/
                         /-
                           🎉 no goals
                         -/
    split_ifs with h <;> repeat simp [h, add_comm]
                         /-
                           🎉 no goals
                         -/
    /-
      case node
      α : Type u_1
      l : Ordnode α
      x : α
      rs : Nat
      rl : Ordnode α
      rx : α
      rr : Ordnode α
      ⊢ Eq (Ordnode.casesOn (id (Ordnode.node rs rl rx rr)) (Ordnode.casesOn (id l)  …
    -/
  · cases' l with ls ll lx lr; · rfl
                                 /-
                                   🎉 no goals
                                 -/
    /-
      case node.node
      α : Type u_1
      x : α
      rs : Nat
      rl : Ordnode α
      rx : α
      rr : Ordnode α
      ls : Nat
      ll : Ordnode α
      lx : α
      lr : Ordnode α
      ⊢ Eq (Ordnode.casesOn (id (Ordnode.node rs rl rx rr)) (Ordnode.casesOn (id (Or …
    -/
    dsimp only [dual, id]
    /-
      case node.node
      α : Type u_1
      x : α
      rs : Nat
      rl : Ordnode α
      rx : α
      rr : Ordnode α
      ls : Nat
      ll : Ordnode α
      lx : α
      lr : Ordnode α
      ⊢ Eq (ite (GT.gt ls (HMul.hMul Ordnode.delta rs)) (Ordnode.rec Ordnode.nil (fu …
    -/
    split_ifs; swap; · simp [add_comm]
                       /-
                         🎉 no goals
                       -/
    /-
      case pos
      α : Type u_1
      x : α
      rs : Nat
      rl : Ordnode α
      rx : α
      rr : Ordnode α
      ls : Nat
      ll : Ordnode α
      lx : α
      lr : Ordnode α
      h✝ : GT.gt ls (HMul.hMul Ordnode.delta rs)
      ⊢ Eq (Ordnode.rec Ordnode.nil (fun size l x_1 r l_ih r_ih => Ordnode.rec Ordno …
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
    cases' ll with lls lll llx llr <;> cases' lr with lrs lrl lrx lrr <;> try rfl
    /-
      case pos.node.node
      α : Type u_1
      x : α
      rs : Nat
      rl : Ordnode α
      rx : α
      rr : Ordnode α
      ls : Nat
      lx : α
      h✝ : GT.gt ls (HMul.hMul Ordnode.delta rs)
      lls : Nat
      lll : Ordnode α
      llx : α
      llr : Ordnode α
      lrs : Nat
      lrl : Ordnode α
      lrx : α
      lrr : Ordnode α
      ⊢ Eq (Ordnode.rec Ordnode.nil (fun size l x_1 r l_ih r_ih => Ordnode.rec Ordno …
    -/
    dsimp only [dual, id]
    /-
      case pos.node.node
      α : Type u_1
      x : α
      rs : Nat
      rl : Ordnode α
      rx : α
      rr : Ordnode α
      ls : Nat
      lx : α
      h✝ : GT.gt ls (HMul.hMul Ordnode.delta rs)
      lls : Nat
      lll : Ordnode α
      llx : α
      llr : Ordnode α
      lrs : Nat
      lrl : Ordnode α
      lrx : α
      lrr : Ordnode α
      ⊢ Eq (ite (LT.lt lrs (HMul.hMul Ordnode.ratio lls)) (Ordnode.node (HAdd.hAdd ( …
    -/
                         /-
                           🎉 no goals
                         -/
    split_ifs with h <;> simp [h, add_comm]
                         /-
                           🎉 no goals
                         -/


theorem dual_balanceR (l : Ordnode α) (x : α) (r : Ordnode α) :
    dual (balanceR l x r) = balanceL (dual r) x (dual l) := by
  /-
    α : Type u_1
    l : Ordnode α
    x : α
    r : Ordnode α
    ⊢ Eq (l.balanceR x r).dual (r.dual.balanceL x l.dual)
  -/
  rw [← dual_dual (balanceL _ _ _), dual_balanceL, dual_dual, dual_dual]
  /-
    🎉 no goals
  -/


theorem Sized.node3L {l x m y r} (hl : @Sized α l) (hm : Sized m) (hr : Sized r) :
    Sized (node3L l x m y r) :=
  (hl.node' hm).node' hr


theorem Sized.node3R {l x m y r} (hl : @Sized α l) (hm : Sized m) (hr : Sized r) :
    Sized (node3R l x m y r) :=
  hl.node' (hm.node' hr)


theorem Sized.node4L {l x m y r} (hl : @Sized α l) (hm : Sized m) (hr : Sized r) :
    Sized (node4L l x m y r) := by
  /-
    α : Type u_1
    l : Ordnode α
    x : α
    m : Ordnode α
    y : α
    r : Ordnode α
    hl : l.Sized
    hm : m.Sized
    hr : r.Sized
    ⊢ (l.node4L x m y r).Sized
  -/
  cases m <;> [exact (hl.node' hm).node' hr; exact (hl.node' hm.2.1).node' (hm.2.2.node' hr)]
  /-
    🎉 no goals
  -/


theorem node3L_size {l x m y r} : size (@node3L α l x m y r) = size l + size m + size r + 2 := by
  /-
    α : Type u_1
    l : Ordnode α
    x : α
    m : Ordnode α
    y : α
    r : Ordnode α
    ⊢ Eq (l.node3L x m y r).size (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd l.size m.size) r …
  -/
  dsimp [node3L, node', size]; rw [add_right_comm _ 1]
                               /-
                                 🎉 no goals
                               -/


theorem node3R_size {l x m y r} : size (@node3R α l x m y r) = size l + size m + size r + 2 := by
  /-
    α : Type u_1
    l : Ordnode α
    x : α
    m : Ordnode α
    y : α
    r : Ordnode α
    ⊢ Eq (l.node3R x m y r).size (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd l.size m.size) r …
  -/
  dsimp [node3R, node', size]; rw [← add_assoc, ← add_assoc]
                               /-
                                 🎉 no goals
                               -/


theorem node4L_size {l x m y r} (hm : Sized m) :
    size (@node4L α l x m y r) = size l + size m + size r + 2 := by
  /-
    α : Type u_1
    l : Ordnode α
    x : α
    m : Ordnode α
    y : α
    r : Ordnode α
    hm : m.Sized
    ⊢ Eq (l.node4L x m y r).size (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd l.size m.size) r …
  -/
  cases m <;> simp [node4L, node3L, node'] <;> [abel; (simp [size, hm.1]; abel)]
  /-
    🎉 no goals
  -/


theorem Sized.dual : ∀ {t : Ordnode α}, Sized t → Sized (dual t)
  | nil, _ => ⟨⟩
                                       /-
                                         α : Type u_1
                                         l : Ordnode α
                                         x✝ : α
                                         r : Ordnode α
                                         sl : l.Sized
                                         sr : r.Sized
                                         ⊢ Eq (HAdd.hAdd (HAdd.hAdd l.size r.size) 1) (HAdd.hAdd (HAdd.hAdd r.dual.size …
                                       -/
  | node _ l _ r, ⟨rfl, sl, sr⟩ => ⟨by simp [size_dual, add_comm], Sized.dual sr, Sized.dual sl⟩
                                       /-
                                         🎉 no goals
                                       -/


theorem Sized.dual_iff {t : Ordnode α} : Sized (.dual t) ↔ Sized t :=
               /-
                 α : Type u_1
                 t : Ordnode α
                 h : t.dual.Sized
                 ⊢ t.Sized
               -/
  ⟨fun h => by rw [← dual_dual t]; exact h.dual, Sized.dual⟩
                                   /-
                                     🎉 no goals
                                   -/


theorem Sized.rotateL {l x r} (hl : @Sized α l) (hr : Sized r) : Sized (rotateL l x r) := by
  /-
    α : Type u_1
    l : Ordnode α
    x : α
    r : Ordnode α
    hl : l.Sized
    hr : r.Sized
    ⊢ (l.rotateL x r).Sized
  -/
  cases r; · exact hl.node' hr
             /-
               🎉 no goals
             -/
  /-
    case node
    α : Type u_1
    l : Ordnode α
    x : α
    hl : l.Sized
    size✝ : Nat
    l✝ : Ordnode α
    x✝ : α
    r✝ : Ordnode α
    hr : (Ordnode.node size✝ l✝ x✝ r✝).Sized
    ⊢ (l.rotateL x (Ordnode.node size✝ l✝ x✝ r✝)).Sized
  -/
  rw [Ordnode.rotateL_node]; split_ifs
    /-
      case pos
      α : Type u_1
      l : Ordnode α
      x : α
      hl : l.Sized
      size✝ : Nat
      l✝ : Ordnode α
      x✝ : α
      r✝ : Ordnode α
      hr : (Ordnode.node size✝ l✝ x✝ r✝).Sized
      h✝ : LT.lt l✝.size (HMul.hMul Ordnode.ratio r✝.size)
      ⊢ (l.node3L x l✝ x✝ r✝).Sized
    -/
  · exact hl.node3L hr.2.1 hr.2.2
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      l : Ordnode α
      x : α
      hl : l.Sized
      size✝ : Nat
      l✝ : Ordnode α
      x✝ : α
      r✝ : Ordnode α
      hr : (Ordnode.node size✝ l✝ x✝ r✝).Sized
      h✝ : Not (LT.lt l✝.size (HMul.hMul Ordnode.ratio r✝.size))
      ⊢ (l.node4L x l✝ x✝ r✝).Sized
    -/
  · exact hl.node4L hr.2.1 hr.2.2
    /-
      🎉 no goals
    -/


theorem Sized.rotateR {l x r} (hl : @Sized α l) (hr : Sized r) : Sized (rotateR l x r) :=
                         /-
                           α : Type u_1
                           l : Ordnode α
                           x : α
                           r : Ordnode α
                           hl : l.Sized
                           hr : r.Sized
                           ⊢ (l.rotateR x r).dual.Sized
                         -/
  Sized.dual_iff.1 <| by rw [dual_rotateR]; exact hr.dual.rotateL hl.dual
                                            /-
                                              🎉 no goals
                                            -/


theorem Sized.rotateL_size {l x r} (hm : Sized r) :
    size (@Ordnode.rotateL α l x r) = size l + size r + 1 := by
  /-
    α : Type u_1
    l : Ordnode α
    x : α
    r : Ordnode α
    hm : r.Sized
    ⊢ Eq (l.rotateL x r).size (HAdd.hAdd (HAdd.hAdd l.size r.size) 1)
  -/
              /-
                🎉 no goals
              -/
  cases r <;> simp [Ordnode.rotateL]
  /-
    case node
    α : Type u_1
    l : Ordnode α
    x : α
    size✝ : Nat
    l✝ : Ordnode α
    x✝ : α
    r✝ : Ordnode α
    hm : (Ordnode.node size✝ l✝ x✝ r✝).Sized
    ⊢ Eq (ite (LT.lt l✝.size (HMul.hMul Ordnode.ratio r✝.size)) (l.node3L x l✝ x✝  …
  -/
  simp only [hm.1]
  /-
    case node
    α : Type u_1
    l : Ordnode α
    x : α
    size✝ : Nat
    l✝ : Ordnode α
    x✝ : α
    r✝ : Ordnode α
    hm : (Ordnode.node size✝ l✝ x✝ r✝).Sized
    ⊢ Eq (ite (LT.lt l✝.size (HMul.hMul Ordnode.ratio r✝.size)) (l.node3L x l✝ x✝  …
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
  split_ifs <;> simp [node3L_size, node4L_size hm.2.1] <;> abel
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem Sized.rotateR_size {l x r} (hl : Sized l) :
    size (@Ordnode.rotateR α l x r) = size l + size r + 1 := by
  /-
    α : Type u_1
    l : Ordnode α
    x : α
    r : Ordnode α
    hl : l.Sized
    ⊢ Eq (l.rotateR x r).size (HAdd.hAdd (HAdd.hAdd l.size r.size) 1)
  -/
  rw [← size_dual, dual_rotateR, hl.dual.rotateL_size, size_dual, size_dual, add_comm (size l)]
  /-
    🎉 no goals
  -/


theorem Sized.balance' {l x r} (hl : @Sized α l) (hr : Sized r) : Sized (balance' l x r) := by
  /-
    α : Type u_1
    l : Ordnode α
    x : α
    r : Ordnode α
    hl : l.Sized
    hr : r.Sized
    ⊢ (l.balance' x r).Sized
  -/
  unfold Ordnode.balance'; split_ifs
    /-
      case pos
      α : Type u_1
      l : Ordnode α
      x : α
      r : Ordnode α
      hl : l.Sized
      hr : r.Sized
      h✝ : LE.le (HAdd.hAdd l.size r.size) 1
      ⊢ (l.node' x r).Sized
    -/
  · exact hl.node' hr
    /-
      🎉 no goals
    -/
    /-
      case pos
      α : Type u_1
      l : Ordnode α
      x : α
      r : Ordnode α
      hl : l.Sized
      hr : r.Sized
      h✝¹ : Not (LE.le (HAdd.hAdd l.size r.size) 1)
      h✝ : GT.gt r.size (HMul.hMul Ordnode.delta l.size)
      ⊢ (l.rotateL x r).Sized
    -/
  · exact hl.rotateL hr
    /-
      🎉 no goals
    -/
    /-
      case pos
      α : Type u_1
      l : Ordnode α
      x : α
      r : Ordnode α
      hl : l.Sized
      hr : r.Sized
      h✝² : Not (LE.le (HAdd.hAdd l.size r.size) 1)
      h✝¹ : Not (GT.gt r.size (HMul.hMul Ordnode.delta l.size))
      h✝ : GT.gt l.size (HMul.hMul Ordnode.delta r.size)
      ⊢ (l.rotateR x r).Sized
    -/
  · exact hl.rotateR hr
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      l : Ordnode α
      x : α
      r : Ordnode α
      hl : l.Sized
      hr : r.Sized
      h✝² : Not (LE.le (HAdd.hAdd l.size r.size) 1)
      h✝¹ : Not (GT.gt r.size (HMul.hMul Ordnode.delta l.size))
      h✝ : Not (GT.gt l.size (HMul.hMul Ordnode.delta r.size))
      ⊢ (l.node' x r).Sized
    -/
  · exact hl.node' hr
    /-
      🎉 no goals
    -/


theorem size_balance' {l x r} (hl : @Sized α l) (hr : Sized r) :
    size (@balance' α l x r) = size l + size r + 1 := by
  /-
    α : Type u_1
    l : Ordnode α
    x : α
    r : Ordnode α
    hl : l.Sized
    hr : r.Sized
    ⊢ Eq (l.balance' x r).size (HAdd.hAdd (HAdd.hAdd l.size r.size) 1)
  -/
  unfold balance'; split_ifs
    /-
      case pos
      α : Type u_1
      l : Ordnode α
      x : α
      r : Ordnode α
      hl : l.Sized
      hr : r.Sized
      h✝ : LE.le (HAdd.hAdd l.size r.size) 1
      ⊢ Eq (l.node' x r).size (HAdd.hAdd (HAdd.hAdd l.size r.size) 1)
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case pos
      α : Type u_1
      l : Ordnode α
      x : α
      r : Ordnode α
      hl : l.Sized
      hr : r.Sized
      h✝¹ : Not (LE.le (HAdd.hAdd l.size r.size) 1)
      h✝ : GT.gt r.size (HMul.hMul Ordnode.delta l.size)
      ⊢ Eq (l.rotateL x r).size (HAdd.hAdd (HAdd.hAdd l.size r.size) 1)
    -/
  · exact hr.rotateL_size
    /-
      🎉 no goals
    -/
    /-
      case pos
      α : Type u_1
      l : Ordnode α
      x : α
      r : Ordnode α
      hl : l.Sized
      hr : r.Sized
      h✝² : Not (LE.le (HAdd.hAdd l.size r.size) 1)
      h✝¹ : Not (GT.gt r.size (HMul.hMul Ordnode.delta l.size))
      h✝ : GT.gt l.size (HMul.hMul Ordnode.delta r.size)
      ⊢ Eq (l.rotateR x r).size (HAdd.hAdd (HAdd.hAdd l.size r.size) 1)
    -/
  · exact hl.rotateR_size
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      l : Ordnode α
      x : α
      r : Ordnode α
      hl : l.Sized
      hr : r.Sized
      h✝² : Not (LE.le (HAdd.hAdd l.size r.size) 1)
      h✝¹ : Not (GT.gt r.size (HMul.hMul Ordnode.delta l.size))
      h✝ : Not (GT.gt l.size (HMul.hMul Ordnode.delta r.size))
      ⊢ Eq (l.node' x r).size (HAdd.hAdd (HAdd.hAdd l.size r.size) 1)
    -/
  · rfl
    /-
      🎉 no goals
    -/


theorem All.imp {P Q : α → Prop} (H : ∀ a, P a → Q a) : ∀ {t}, All P t → All Q t
  | nil, _ => ⟨⟩
  | node _ _ _ _, ⟨h₁, h₂, h₃⟩ => ⟨h₁.imp H, H _ h₂, h₃.imp H⟩


theorem Any.imp {P Q : α → Prop} (H : ∀ a, P a → Q a) : ∀ {t}, Any P t → Any Q t
  | nil => id
  | node _ _ _ _ => Or.imp (Any.imp H) <| Or.imp (H _) (Any.imp H)


theorem all_singleton {P : α → Prop} {x : α} : All P (singleton x) ↔ P x :=
  ⟨fun h => h.2.1, fun h => ⟨⟨⟩, h, ⟨⟩⟩⟩


theorem any_singleton {P : α → Prop} {x : α} : Any P (singleton x) ↔ P x :=
      /-
        α : Type u_1
        P : α → Prop
        x : α
        ⊢ Ordnode.Any P (Singleton.singleton x) → P x
      -/
  ⟨by rintro (⟨⟨⟩⟩ | h | ⟨⟨⟩⟩); exact h, fun h => Or.inr (Or.inl h)⟩
                                /-
                                  🎉 no goals
                                -/


theorem all_dual {P : α → Prop} : ∀ {t : Ordnode α}, All P (dual t) ↔ All P t
  | nil => Iff.rfl
  | node _ _l _x _r =>
    ⟨fun ⟨hr, hx, hl⟩ => ⟨all_dual.1 hl, hx, all_dual.1 hr⟩, fun ⟨hl, hx, hr⟩ =>
      ⟨all_dual.2 hr, hx, all_dual.2 hl⟩⟩


theorem all_iff_forall {P : α → Prop} : ∀ {t}, All P t ↔ ∀ x, Emem x t → P x
                                 /-
                                   α : Type u_1
                                   P : α → Prop
                                   ⊢ ∀ (x : α), Ordnode.Emem x Ordnode.nil → P x
                                 -/
  | nil => (iff_true_intro <| by rintro _ ⟨⟩).symm
                                 /-
                                   🎉 no goals
                                 -/
                       /-
                         α : Type u_1
                         P : α → Prop
                         size✝ : Nat
                         l : Ordnode α
                         x : α
                         r : Ordnode α
                         ⊢ Iff (Ordnode.All P (Ordnode.node size✝ l x r)) (∀ (x_1 : α), Ordnode.Emem x_ …
                       -/
  | node _ l x r => by simp [All, Emem, all_iff_forall, Any, or_imp, forall_and]
                       /-
                         🎉 no goals
                       -/


theorem any_iff_exists {P : α → Prop} : ∀ {t}, Any P t ↔ ∃ x, Emem x t ∧ P x
               /-
                 α : Type u_1
                 P : α → Prop
                 ⊢ Ordnode.Any P Ordnode.nil → Exists fun x => And (Ordnode.Emem x Ordnode.nil) …
               -/
               /-
                 🎉 no goals
               -/
  | nil => ⟨by rintro ⟨⟩, by rintro ⟨_, ⟨⟩, _⟩⟩
                             /-
                               🎉 no goals
                             -/
                       /-
                         α : Type u_1
                         P : α → Prop
                         size✝ : Nat
                         l : Ordnode α
                         x : α
                         r : Ordnode α
                         ⊢ Iff (Ordnode.Any P (Ordnode.node size✝ l x r)) (Exists fun x_1 => And (Ordno …
                       -/
  | node _ l x r => by simp only [Emem]; simp [Any, any_iff_exists, or_and_right, exists_or]
                                         /-
                                           🎉 no goals
                                         -/


theorem emem_iff_all {x : α} {t} : Emem x t ↔ ∀ P, All P t → P x :=
  ⟨fun h _ al => all_iff_forall.1 al _ h, fun H => H _ <| all_iff_forall.2 fun _ => id⟩


theorem all_node' {P l x r} : @All α P (node' l x r) ↔ All P l ∧ P x ∧ All P r :=
  Iff.rfl


theorem all_node3L {P l x m y r} :
    @All α P (node3L l x m y r) ↔ All P l ∧ P x ∧ All P m ∧ P y ∧ All P r := by
  /-
    α : Type u_1
    P : α → Prop
    l : Ordnode α
    x : α
    m : Ordnode α
    y : α
    r : Ordnode α
    ⊢ Iff (Ordnode.All P (l.node3L x m y r)) (And (Ordnode.All P l) (And (P x) (An …
  -/
  simp [node3L, all_node', and_assoc]
  /-
    🎉 no goals
  -/


theorem all_node3R {P l x m y r} :
    @All α P (node3R l x m y r) ↔ All P l ∧ P x ∧ All P m ∧ P y ∧ All P r :=
  Iff.rfl


theorem all_node4L {P l x m y r} :
    @All α P (node4L l x m y r) ↔ All P l ∧ P x ∧ All P m ∧ P y ∧ All P r := by
  /-
    α : Type u_1
    P : α → Prop
    l : Ordnode α
    x : α
    m : Ordnode α
    y : α
    r : Ordnode α
    ⊢ Iff (Ordnode.All P (l.node4L x m y r)) (And (Ordnode.All P l) (And (P x) (An …
  -/
              /-
                🎉 no goals
              -/
  cases m <;> simp [node4L, all_node', All, all_node3L, and_assoc]
              /-
                🎉 no goals
              -/


theorem all_node4R {P l x m y r} :
    @All α P (node4R l x m y r) ↔ All P l ∧ P x ∧ All P m ∧ P y ∧ All P r := by
  /-
    α : Type u_1
    P : α → Prop
    l : Ordnode α
    x : α
    m : Ordnode α
    y : α
    r : Ordnode α
    ⊢ Iff (Ordnode.All P (l.node4R x m y r)) (And (Ordnode.All P l) (And (P x) (An …
  -/
              /-
                🎉 no goals
              -/
  cases m <;> simp [node4R, all_node', All, all_node3R, and_assoc]
              /-
                🎉 no goals
              -/


theorem all_rotateL {P l x r} : @All α P (rotateL l x r) ↔ All P l ∧ P x ∧ All P r := by
  /-
    α : Type u_1
    P : α → Prop
    l : Ordnode α
    x : α
    r : Ordnode α
    ⊢ Iff (Ordnode.All P (l.rotateL x r)) (And (Ordnode.All P l) (And (P x) (Ordno …
  -/
              /-
                🎉 no goals
              -/
  cases r <;> simp [rotateL, all_node']; split_ifs <;>
    /-
      case pos
      α : Type u_1
      P : α → Prop
      l : Ordnode α
      x : α
      size✝ : Nat
      l✝ : Ordnode α
      x✝ : α
      r✝ : Ordnode α
      h✝ : LT.lt l✝.size (HMul.hMul Ordnode.ratio r✝.size)
      ⊢ Iff (Ordnode.All P (l.node3L x l✝ x✝ r✝)) (And (Ordnode.All P l) (And (P x)  …
    -/
    /-
      🎉 no goals
    -/
    simp [all_node3L, all_node4L, All, and_assoc]
    /-
      🎉 no goals
    -/


theorem all_rotateR {P l x r} : @All α P (rotateR l x r) ↔ All P l ∧ P x ∧ All P r := by
  /-
    α : Type u_1
    P : α → Prop
    l : Ordnode α
    x : α
    r : Ordnode α
    ⊢ Iff (Ordnode.All P (l.rotateR x r)) (And (Ordnode.All P l) (And (P x) (Ordno …
  -/
  rw [← all_dual, dual_rotateR, all_rotateL]; simp [all_dual, and_comm, and_left_comm, and_assoc]
                                              /-
                                                🎉 no goals
                                              -/


theorem all_balance' {P l x r} : @All α P (balance' l x r) ↔ All P l ∧ P x ∧ All P r := by
  /-
    α : Type u_1
    P : α → Prop
    l : Ordnode α
    x : α
    r : Ordnode α
    ⊢ Iff (Ordnode.All P (l.balance' x r)) (And (Ordnode.All P l) (And (P x) (Ordn …
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
  rw [balance']; split_ifs <;> simp [all_node', all_rotateL, all_rotateR]
                               /-
                                 🎉 no goals
                               -/


theorem foldr_cons_eq_toList : ∀ (t : Ordnode α) (r : List α), t.foldr List.cons r = toList t ++ r
  | nil, _ => rfl
  | node _ l x r, r' => by
    rw [foldr, foldr_cons_eq_toList l, foldr_cons_eq_toList r, ← List.cons_append,
                                                        /-
                                                          α : Type u_1
                                                          size✝ : Nat
                                                          l : Ordnode α
                                                          x : α
                                                          r : Ordnode α
                                                          r' : List α
                                                          ⊢ Eq (HAppend.hAppend (Ordnode.foldr List.cons l (List.cons x r.toList)) r') ( …
                                                        -/
        ← List.append_assoc, ← foldr_cons_eq_toList l]; rfl
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp]
theorem toList_nil : toList (@nil α) = [] :=
  rfl


@[simp]
theorem toList_node (s l x r) : toList (@node α s l x r) = toList l ++ x :: toList r := by
  /-
    α : Type u_1
    s : Nat
    l : Ordnode α
    x : α
    r : Ordnode α
    ⊢ Eq (Ordnode.node s l x r).toList (HAppend.hAppend l.toList (List.cons x r.to …
  -/
  rw [toList, foldr, foldr_cons_eq_toList]; rfl
                                            /-
                                              🎉 no goals
                                            -/


theorem emem_iff_mem_toList {x : α} {t} : Emem x t ↔ x ∈ toList t := by
  /-
    α : Type u_1
    x : α
    t : Ordnode α
    ⊢ Iff (Ordnode.Emem x t) (Membership.mem t.toList x)
  -/
                               /-
                                 🎉 no goals
                               -/
  unfold Emem; induction t <;> simp [Any, *, or_assoc]
                               /-
                                 🎉 no goals
                               -/


theorem length_toList' : ∀ t : Ordnode α, (toList t).length = t.realSize
  | nil => rfl
  | node _ l _ r => by
    rw [toList_node, List.length_append, List.length_cons, length_toList' l,
                           /-
                             α : Type u_1
                             size✝ : Nat
                             l : Ordnode α
                             x✝ : α
                             r : Ordnode α
                             ⊢ Eq (HAdd.hAdd l.realSize (HAdd.hAdd r.realSize 1)) (Ordnode.node size✝ l x✝  …
                           -/
        length_toList' r]; rfl
                           /-
                             🎉 no goals
                           -/


theorem length_toList {t : Ordnode α} (h : Sized t) : (toList t).length = t.size := by
  /-
    α : Type u_1
    t : Ordnode α
    h : t.Sized
    ⊢ Eq t.toList.length t.size
  -/
  rw [length_toList', size_eq_realSize h]
  /-
    🎉 no goals
  -/


theorem equiv_iff {t₁ t₂ : Ordnode α} (h₁ : Sized t₁) (h₂ : Sized t₂) :
    Equiv t₁ t₂ ↔ toList t₁ = toList t₂ :=
                                   /-
                                     α : Type u_1
                                     t₁ t₂ : Ordnode α
                                     h₁ : t₁.Sized
                                     h₂ : t₂.Sized
                                     h : Eq t₁.toList t₂.toList
                                     ⊢ Eq t₁.size t₂.size
                                   -/
  and_iff_right_of_imp fun h => by rw [← length_toList h₁, h, length_toList h₂]
                                   /-
                                     🎉 no goals
                                   -/


theorem pos_size_of_mem [LE α] [DecidableRel (α := α) (· ≤ ·)] {x : α} {t : Ordnode α} (h : Sized t)
                                       /-
                                         α : Type u_1
                                         inst✝¹ : LE α
                                         inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
                                         x : α
                                         t : Ordnode α
                                         h : t.Sized
                                         h_mem : Membership.mem t x
                                         ⊢ LT.lt 0 t.size
                                       -/
                                                  /-
                                                    🎉 no goals
                                                  -/
    (h_mem : x ∈ t) : 0 < size t := by cases t; · { contradiction }; · { simp [h.1] }
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


theorem findMin'_dual : ∀ (t) (x : α), findMin' (dual t) x = findMax' x t
  | nil, _ => rfl
  | node _ _ x r, _ => findMin'_dual r x


theorem findMax'_dual (t) (x : α) : findMax' x (dual t) = findMin' t x := by
  /-
    α : Type u_1
    t : Ordnode α
    x : α
    ⊢ Eq (Ordnode.findMax' x t.dual) (t.findMin' x)
  -/
  rw [← findMin'_dual, dual_dual]
  /-
    🎉 no goals
  -/


theorem findMin_dual : ∀ t : Ordnode α, findMin (dual t) = findMax t
  | nil => rfl
  | node _ _ _ _ => congr_arg some <| findMin'_dual _ _


theorem findMax_dual (t : Ordnode α) : findMax (dual t) = findMin t := by
  /-
    α : Type u_1
    t : Ordnode α
    ⊢ Eq t.dual.findMax t.findMin
  -/
  rw [← findMin_dual, dual_dual]
  /-
    🎉 no goals
  -/


theorem dual_eraseMin : ∀ t : Ordnode α, dual (eraseMin t) = eraseMax (dual t)
  | nil => rfl
  | node _ nil _ _ => rfl
  | node _ (node sz l' y r') x r => by
    /-
      α : Type u_1
      size✝ sz : Nat
      l' : Ordnode α
      y : α
      r' : Ordnode α
      x : α
      r : Ordnode α
      ⊢ Eq (Ordnode.node size✝ (Ordnode.node sz l' y r') x r).eraseMin.dual (Ordnode …
    -/
    rw [eraseMin, dual_balanceR, dual_eraseMin (node sz l' y r'), dual, dual, dual, eraseMax]
    /-
      🎉 no goals
    -/


theorem dual_eraseMax (t : Ordnode α) : dual (eraseMax t) = eraseMin (dual t) := by
  /-
    α : Type u_1
    t : Ordnode α
    ⊢ Eq t.eraseMax.dual t.dual.eraseMin
  -/
  rw [← dual_dual (eraseMin _), dual_eraseMin, dual_dual]
  /-
    🎉 no goals
  -/


theorem splitMin_eq :
    ∀ (s l) (x : α) (r), splitMin' l x r = (findMin' l x, eraseMin (node s l x r))
  | _, nil, _, _ => rfl
                                    /-
                                      α : Type u_1
                                      x✝ ls : Nat
                                      ll : Ordnode α
                                      lx : α
                                      lr : Ordnode α
                                      x : α
                                      r : Ordnode α
                                      ⊢ Eq ((Ordnode.node ls ll lx lr).splitMin' x r) { fst := (Ordnode.node ls ll l …
                                    -/
  | _, node ls ll lx lr, x, r => by rw [splitMin', splitMin_eq ls ll lx lr, findMin', eraseMin]
                                    /-
                                      🎉 no goals
                                    -/


theorem splitMax_eq :
    ∀ (s l) (x : α) (r), splitMax' l x r = (eraseMax (node s l x r), findMax' x r)
  | _, _, _, nil => rfl
                                    /-
                                      α : Type u_1
                                      x✝ : Nat
                                      l : Ordnode α
                                      x : α
                                      ls : Nat
                                      ll : Ordnode α
                                      lx : α
                                      lr : Ordnode α
                                      ⊢ Eq (l.splitMax' x (Ordnode.node ls ll lx lr)) { fst := (Ordnode.node x✝ l x  …
                                    -/
  | _, l, x, node ls ll lx lr => by rw [splitMax', splitMax_eq ls ll lx lr, findMax', eraseMax]
                                    /-
                                      🎉 no goals
                                    -/


@[elab_as_elim]
theorem findMin'_all {P : α → Prop} : ∀ (t) (x : α), All P t → P x → P (findMin' t x)
  | nil, _x, _, hx => hx
  | node _ ll lx _, _, ⟨h₁, h₂, _⟩, _ => findMin'_all ll lx h₁ h₂


@[elab_as_elim]
theorem findMax'_all {P : α → Prop} : ∀ (x : α) (t), P x → All P t → P (findMax' x t)
  | _x, nil, hx, _ => hx
  | _, node _ _ lx lr, _, ⟨_, h₂, h₃⟩ => findMax'_all lx lr h₂ h₃


@[simp]
                                                               /-
                                                                 α : Type u_1
                                                                 t : Ordnode α
                                                                 ⊢ Eq (t.merge Ordnode.nil) t
                                                               -/
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
theorem merge_nil_left (t : Ordnode α) : merge t nil = t := by cases t <;> rfl
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


@[simp]
theorem merge_nil_right (t : Ordnode α) : merge nil t = t :=
  rfl


@[simp]
theorem merge_node {ls ll lx lr rs rl rx rr} :
    merge (@node α ls ll lx lr) (node rs rl rx rr) =
      if delta * ls < rs then balanceL (merge (node ls ll lx lr) rl) rx rr
      else if delta * rs < ls then balanceR ll lx (merge lr (node rs rl rx rr))
      else glue (node ls ll lx lr) (node rs rl rx rr) :=
  rfl


theorem dual_insert [Preorder α] [IsTotal α (· ≤ ·)] [DecidableRel (α := α) (· ≤ ·)] (x : α) :
    ∀ t : Ordnode α, dual (Ordnode.insert x t) = @Ordnode.insert αᵒᵈ _ _ x (dual t)
  | nil => rfl
  | node _ l y r => by
    /-
      α : Type u_1
      inst✝² : Preorder α
      inst✝¹ : IsTotal α fun x1 x2 => LE.le x1 x2
      inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
      x : α
      size✝ : Nat
      l : Ordnode α
      y : α
      r : Ordnode α
      ⊢ Eq (Ordnode.insert x (Ordnode.node size✝ l y r)).dual (Ordnode.insert x (Ord …
    -/
    have : @cmpLE αᵒᵈ _ _ x y = cmpLE y x := rfl
    /-
      α : Type u_1
      inst✝² : Preorder α
      inst✝¹ : IsTotal α fun x1 x2 => LE.le x1 x2
      inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
      x : α
      size✝ : Nat
      l : Ordnode α
      y : α
      r : Ordnode α
      this : Eq (cmpLE x y) (cmpLE y x)
      ⊢ Eq (Ordnode.insert x (Ordnode.node size✝ l y r)).dual (Ordnode.insert x (Ord …
    -/
    rw [Ordnode.insert, dual, Ordnode.insert, this, ← cmpLE_swap x y]
    /-
      α : Type u_1
      inst✝² : Preorder α
      inst✝¹ : IsTotal α fun x1 x2 => LE.le x1 x2
      inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
      x : α
      size✝ : Nat
      l : Ordnode α
      y : α
      r : Ordnode α
      this : Eq (cmpLE x y) (cmpLE y x)
      ⊢ Eq (Ordnode.mem.match_1 (fun x => Ordnode α) (cmpLE x y) (fun _ => (Ordnode. …
    -/
    cases cmpLE x y <;>
      /-
        case lt
        α : Type u_1
        inst✝² : Preorder α
        inst✝¹ : IsTotal α fun x1 x2 => LE.le x1 x2
        inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
        x : α
        size✝ : Nat
        l : Ordnode α
        y : α
        r : Ordnode α
        this : Eq (cmpLE x y) (cmpLE y x)
        ⊢ Eq (Ordnode.mem.match_1 (fun x => Ordnode α) Ordering.lt (fun _ => (Ordnode. …
      -/
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
      simp [Ordering.swap, Ordnode.insert, dual_balanceL, dual_balanceR, dual_insert]
      /-
        🎉 no goals
      -/


theorem balance_eq_balance' {l x r} (hl : Balanced l) (hr : Balanced r) (sl : Sized l)
    (sr : Sized r) : @balance α l x r = balance' l x r := by
  /-
    α : Type u_1
    l : Ordnode α
    x : α
    r : Ordnode α
    hl : l.Balanced
    hr : r.Balanced
    sl : l.Sized
    sr : r.Sized
    ⊢ Eq (l.balance x r) (l.balance' x r)
  -/
  cases' l with ls ll lx lr
    /-
      case nil
      α : Type u_1
      x : α
      r : Ordnode α
      hr : r.Balanced
      sr : r.Sized
      hl : Ordnode.nil.Balanced
      sl : Ordnode.nil.Sized
      ⊢ Eq (Ordnode.nil.balance x r) (Ordnode.nil.balance' x r)
    -/
  · cases' r with rs rl rx rr
      /-
        case nil.nil
        α : Type u_1
        x : α
        hl : Ordnode.nil.Balanced
        sl : Ordnode.nil.Sized
        hr : Ordnode.nil.Balanced
        sr : Ordnode.nil.Sized
        ⊢ Eq (Ordnode.nil.balance x Ordnode.nil) (Ordnode.nil.balance' x Ordnode.nil)
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case nil.node
        α : Type u_1
        x : α
        hl : Ordnode.nil.Balanced
        sl : Ordnode.nil.Sized
        rs : Nat
        rl : Ordnode α
        rx : α
        rr : Ordnode α
        hr : (Ordnode.node rs rl rx rr).Balanced
        sr : (Ordnode.node rs rl rx rr).Sized
        ⊢ Eq (Ordnode.nil.balance x (Ordnode.node rs rl rx rr)) (Ordnode.nil.balance'  …
      -/
    · rw [sr.eq_node'] at hr ⊢
      /-
        case nil.node
        α : Type u_1
        x : α
        hl : Ordnode.nil.Balanced
        sl : Ordnode.nil.Sized
        rs : Nat
        rl : Ordnode α
        rx : α
        rr : Ordnode α
        hr : (rl.node' rx rr).Balanced
        sr : (Ordnode.node rs rl rx rr).Sized
        ⊢ Eq (Ordnode.nil.balance x (rl.node' rx rr)) (Ordnode.nil.balance' x (rl.node …
      -/
      cases' rl with rls rll rlx rlr <;> cases' rr with rrs rrl rrx rrr <;>
        /-
          case nil.node.nil.nil
          α : Type u_1
          x : α
          hl : Ordnode.nil.Balanced
          sl : Ordnode.nil.Sized
          rs : Nat
          rx : α
          hr : (Ordnode.nil.node' rx Ordnode.nil).Balanced
          sr : (Ordnode.node rs Ordnode.nil rx Ordnode.nil).Sized
          ⊢ Eq (Ordnode.nil.balance x (Ordnode.nil.node' rx Ordnode.nil)) (Ordnode.nil.b …
        -/
        dsimp [balance, balance']
        /-
          case nil.node.nil.nil
          α : Type u_1
          x : α
          hl : Ordnode.nil.Balanced
          sl : Ordnode.nil.Sized
          rs : Nat
          rx : α
          hr : (Ordnode.nil.node' rx Ordnode.nil).Balanced
          sr : (Ordnode.node rs Ordnode.nil rx Ordnode.nil).Sized
          ⊢ Eq (Ordnode.node 2 Ordnode.nil x (Ordnode.nil.node' rx Ordnode.nil)) (Ordnod …
        -/
      · rfl
        /-
          🎉 no goals
        -/
      · have : size rrl = 0 ∧ size rrr = 0 := by
          have := balancedSz_zero.1 hr.1.symm
          rwa [size, sr.2.2.1, Nat.succ_le_succ_iff, Nat.le_zero, add_eq_zero] at this
        /-
          case nil.node.nil.node
          α : Type u_1
          x : α
          hl : Ordnode.nil.Balanced
          sl : Ordnode.nil.Sized
          rs : Nat
          rx : α
          rrs : Nat
          rrl : Ordnode α
          rrx : α
          rrr : Ordnode α
          hr : (Ordnode.nil.node' rx (Ordnode.node rrs rrl rrx rrr)).Balanced
          sr : (Ordnode.node rs Ordnode.nil rx (Ordnode.node rrs rrl rrx rrr)).Sized
          this : And (Eq rrl.size 0) (Eq rrr.size 0)
          ⊢ Eq (Ordnode.node 3 (Ordnode.singleton x) rx (Ordnode.node rrs rrl rrx rrr))  …
        -/
        cases sr.2.2.2.1.size_eq_zero.1 this.1
        /-
          case nil.node.nil.node.refl
          α : Type u_1
          x : α
          hl : Ordnode.nil.Balanced
          sl : Ordnode.nil.Sized
          rs : Nat
          rx : α
          rrs : Nat
          rrx : α
          rrr : Ordnode α
          hr : (Ordnode.nil.node' rx (Ordnode.node rrs Ordnode.nil rrx rrr)).Balanced
          sr : (Ordnode.node rs Ordnode.nil rx (Ordnode.node rrs Ordnode.nil rrx rrr)).S …
          this : And (Eq Ordnode.nil.size 0) (Eq rrr.size 0)
          ⊢ Eq (Ordnode.node 3 (Ordnode.singleton x) rx (Ordnode.node rrs Ordnode.nil rr …
        -/
        cases sr.2.2.2.2.size_eq_zero.1 this.2
        /-
          case nil.node.nil.node.refl.refl
          α : Type u_1
          x : α
          hl : Ordnode.nil.Balanced
          sl : Ordnode.nil.Sized
          rs : Nat
          rx : α
          rrs : Nat
          rrx : α
          hr : (Ordnode.nil.node' rx (Ordnode.node rrs Ordnode.nil rrx Ordnode.nil)).Bal …
          sr : (Ordnode.node rs Ordnode.nil rx (Ordnode.node rrs Ordnode.nil rrx Ordnode …
          this : And (Eq Ordnode.nil.size 0) (Eq Ordnode.nil.size 0)
          ⊢ Eq (Ordnode.node 3 (Ordnode.singleton x) rx (Ordnode.node rrs Ordnode.nil rr …
        -/
        obtain rfl : rrs = 1 := sr.2.2.1
        /-
          case nil.node.nil.node.refl.refl
          α : Type u_1
          x : α
          hl : Ordnode.nil.Balanced
          sl : Ordnode.nil.Sized
          rs : Nat
          rx rrx : α
          this : And (Eq Ordnode.nil.size 0) (Eq Ordnode.nil.size 0)
          hr : (Ordnode.nil.node' rx (Ordnode.node 1 Ordnode.nil rrx Ordnode.nil)).Balan …
          sr : (Ordnode.node rs Ordnode.nil rx (Ordnode.node 1 Ordnode.nil rrx Ordnode.n …
          ⊢ Eq (Ordnode.node 3 (Ordnode.singleton x) rx (Ordnode.node 1 Ordnode.nil rrx  …
        -/
        rw [if_neg, if_pos, rotateL_node, if_pos]; · rfl
                                                     /-
                                                       🎉 no goals
                                                     -/
        /-
          case nil.node.nil.node.refl.refl.hc
          α : Type u_1
          x : α
          hl : Ordnode.nil.Balanced
          sl : Ordnode.nil.Sized
          rs : Nat
          rx rrx : α
          this : And (Eq Ordnode.nil.size 0) (Eq Ordnode.nil.size 0)
          hr : (Ordnode.nil.node' rx (Ordnode.node 1 Ordnode.nil rrx Ordnode.nil)).Balan …
          sr : (Ordnode.node rs Ordnode.nil rx (Ordnode.node 1 Ordnode.nil rrx Ordnode.n …
          ⊢ LT.lt Ordnode.nil.size (HMul.hMul Ordnode.ratio (Ordnode.node 1 Ordnode.nil  …
        -/
        all_goals dsimp only [size]; decide
        /-
          🎉 no goals
        -/
      · have : size rll = 0 ∧ size rlr = 0 := by
          have := balancedSz_zero.1 hr.1
          rwa [size, sr.2.1.1, Nat.succ_le_succ_iff, Nat.le_zero, add_eq_zero] at this
        /-
          case nil.node.node.nil
          α : Type u_1
          x : α
          hl : Ordnode.nil.Balanced
          sl : Ordnode.nil.Sized
          rs : Nat
          rx : α
          rls : Nat
          rll : Ordnode α
          rlx : α
          rlr : Ordnode α
          hr : ((Ordnode.node rls rll rlx rlr).node' rx Ordnode.nil).Balanced
          sr : (Ordnode.node rs (Ordnode.node rls rll rlx rlr) rx Ordnode.nil).Sized
          this : And (Eq rll.size 0) (Eq rlr.size 0)
          ⊢ Eq (Ordnode.node 3 (Ordnode.singleton x) rlx (Ordnode.singleton rx)) (ite (L …
        -/
        cases sr.2.1.2.1.size_eq_zero.1 this.1
        /-
          case nil.node.node.nil.refl
          α : Type u_1
          x : α
          hl : Ordnode.nil.Balanced
          sl : Ordnode.nil.Sized
          rs : Nat
          rx : α
          rls : Nat
          rlx : α
          rlr : Ordnode α
          hr : ((Ordnode.node rls Ordnode.nil rlx rlr).node' rx Ordnode.nil).Balanced
          sr : (Ordnode.node rs (Ordnode.node rls Ordnode.nil rlx rlr) rx Ordnode.nil).S …
          this : And (Eq Ordnode.nil.size 0) (Eq rlr.size 0)
          ⊢ Eq (Ordnode.node 3 (Ordnode.singleton x) rlx (Ordnode.singleton rx)) (ite (L …
        -/
        cases sr.2.1.2.2.size_eq_zero.1 this.2
        /-
          case nil.node.node.nil.refl.refl
          α : Type u_1
          x : α
          hl : Ordnode.nil.Balanced
          sl : Ordnode.nil.Sized
          rs : Nat
          rx : α
          rls : Nat
          rlx : α
          hr : ((Ordnode.node rls Ordnode.nil rlx Ordnode.nil).node' rx Ordnode.nil).Bal …
          sr : (Ordnode.node rs (Ordnode.node rls Ordnode.nil rlx Ordnode.nil) rx Ordnod …
          this : And (Eq Ordnode.nil.size 0) (Eq Ordnode.nil.size 0)
          ⊢ Eq (Ordnode.node 3 (Ordnode.singleton x) rlx (Ordnode.singleton rx)) (ite (L …
        -/
        obtain rfl : rls = 1 := sr.2.1.1
        /-
          case nil.node.node.nil.refl.refl
          α : Type u_1
          x : α
          hl : Ordnode.nil.Balanced
          sl : Ordnode.nil.Sized
          rs : Nat
          rx rlx : α
          this : And (Eq Ordnode.nil.size 0) (Eq Ordnode.nil.size 0)
          hr : ((Ordnode.node 1 Ordnode.nil rlx Ordnode.nil).node' rx Ordnode.nil).Balan …
          sr : (Ordnode.node rs (Ordnode.node 1 Ordnode.nil rlx Ordnode.nil) rx Ordnode. …
          ⊢ Eq (Ordnode.node 3 (Ordnode.singleton x) rlx (Ordnode.singleton rx)) (ite (L …
        -/
        rw [if_neg, if_pos, rotateL_node, if_neg]; · rfl
                                                     /-
                                                       🎉 no goals
                                                     -/
        /-
          case nil.node.node.nil.refl.refl.hnc
          α : Type u_1
          x : α
          hl : Ordnode.nil.Balanced
          sl : Ordnode.nil.Sized
          rs : Nat
          rx rlx : α
          this : And (Eq Ordnode.nil.size 0) (Eq Ordnode.nil.size 0)
          hr : ((Ordnode.node 1 Ordnode.nil rlx Ordnode.nil).node' rx Ordnode.nil).Balan …
          sr : (Ordnode.node rs (Ordnode.node 1 Ordnode.nil rlx Ordnode.nil) rx Ordnode. …
          ⊢ Not (LT.lt (Ordnode.node 1 Ordnode.nil rlx Ordnode.nil).size (HMul.hMul Ordn …
        -/
        all_goals dsimp only [size]; decide
        /-
          🎉 no goals
        -/
        /-
          case nil.node.node.node
          α : Type u_1
          x : α
          hl : Ordnode.nil.Balanced
          sl : Ordnode.nil.Sized
          rs : Nat
          rx : α
          rls : Nat
          rll : Ordnode α
          rlx : α
          rlr : Ordnode α
          rrs : Nat
          rrl : Ordnode α
          rrx : α
          rrr : Ordnode α
          hr : ((Ordnode.node rls rll rlx rlr).node' rx (Ordnode.node rrs rrl rrx rrr)). …
          sr : (Ordnode.node rs (Ordnode.node rls rll rlx rlr) rx (Ordnode.node rrs rrl  …
          ⊢ Eq (ite (LT.lt rls (HMul.hMul Ordnode.ratio rrs)) (Ordnode.node (HAdd.hAdd ( …
        -/
      · symm; rw [zero_add, if_neg, if_pos, rotateL]
          /-
            case nil.node.node.node
            α : Type u_1
            x : α
            hl : Ordnode.nil.Balanced
            sl : Ordnode.nil.Sized
            rs : Nat
            rx : α
            rls : Nat
            rll : Ordnode α
            rlx : α
            rlr : Ordnode α
            rrs : Nat
            rrl : Ordnode α
            rrx : α
            rrr : Ordnode α
            hr : ((Ordnode.node rls rll rlx rlr).node' rx (Ordnode.node rrs rrl rrx rrr)). …
            sr : (Ordnode.node rs (Ordnode.node rls rll rlx rlr) rx (Ordnode.node rrs rrl  …
            ⊢ Eq (ite (LT.lt (Ordnode.node rls rll rlx rlr).size (HMul.hMul Ordnode.ratio  …
          -/
        · dsimp only [size_node]; split_ifs
            /-
              case pos
              α : Type u_1
              x : α
              hl : Ordnode.nil.Balanced
              sl : Ordnode.nil.Sized
              rs : Nat
              rx : α
              rls : Nat
              rll : Ordnode α
              rlx : α
              rlr : Ordnode α
              rrs : Nat
              rrl : Ordnode α
              rrx : α
              rrr : Ordnode α
              hr : ((Ordnode.node rls rll rlx rlr).node' rx (Ordnode.node rrs rrl rrx rrr)). …
              sr : (Ordnode.node rs (Ordnode.node rls rll rlx rlr) rx (Ordnode.node rrs rrl  …
              h✝ : LT.lt rls (HMul.hMul Ordnode.ratio rrs)
              ⊢ Eq (Ordnode.nil.node3L x (Ordnode.node rls rll rlx rlr) rx (Ordnode.node rrs …
            -/
                                  /-
                                    🎉 no goals
                                  -/
          · simp [node3L, node']; abel
                                  /-
                                    🎉 no goals
                                  -/
            /-
              case neg
              α : Type u_1
              x : α
              hl : Ordnode.nil.Balanced
              sl : Ordnode.nil.Sized
              rs : Nat
              rx : α
              rls : Nat
              rll : Ordnode α
              rlx : α
              rlr : Ordnode α
              rrs : Nat
              rrl : Ordnode α
              rrx : α
              rrr : Ordnode α
              hr : ((Ordnode.node rls rll rlx rlr).node' rx (Ordnode.node rrs rrl rrx rrr)). …
              sr : (Ordnode.node rs (Ordnode.node rls rll rlx rlr) rx (Ordnode.node rrs rrl  …
              h✝ : Not (LT.lt rls (HMul.hMul Ordnode.ratio rrs))
              ⊢ Eq (Ordnode.nil.node4L x (Ordnode.node rls rll rlx rlr) rx (Ordnode.node rrs …
            -/
                                            /-
                                              🎉 no goals
                                            -/
          · simp [node4L, node', sr.2.1.1]; abel
                                            /-
                                              🎉 no goals
                                            -/
          /-
            case nil.node.node.node.hc
            α : Type u_1
            x : α
            hl : Ordnode.nil.Balanced
            sl : Ordnode.nil.Sized
            rs : Nat
            rx : α
            rls : Nat
            rll : Ordnode α
            rlx : α
            rlr : Ordnode α
            rrs : Nat
            rrl : Ordnode α
            rrx : α
            rrr : Ordnode α
            hr : ((Ordnode.node rls rll rlx rlr).node' rx (Ordnode.node rrs rrl rrx rrr)). …
            sr : (Ordnode.node rs (Ordnode.node rls rll rlx rlr) rx (Ordnode.node rrs rrl  …
            ⊢ GT.gt (HAdd.hAdd (HAdd.hAdd rls rrs) 1) 0
          -/
        · apply Nat.zero_lt_succ
          /-
            🎉 no goals
          -/
          /-
            case nil.node.node.node.hnc
            α : Type u_1
            x : α
            hl : Ordnode.nil.Balanced
            sl : Ordnode.nil.Sized
            rs : Nat
            rx : α
            rls : Nat
            rll : Ordnode α
            rlx : α
            rlr : Ordnode α
            rrs : Nat
            rrl : Ordnode α
            rrx : α
            rrr : Ordnode α
            hr : ((Ordnode.node rls rll rlx rlr).node' rx (Ordnode.node rrs rrl rrx rrr)). …
            sr : (Ordnode.node rs (Ordnode.node rls rll rlx rlr) rx (Ordnode.node rrs rrl  …
            ⊢ Not (LE.le (HAdd.hAdd (HAdd.hAdd rls rrs) 1) 1)
          -/
        · exact not_le_of_gt (Nat.succ_lt_succ (add_pos sr.2.1.pos sr.2.2.pos))
          /-
            🎉 no goals
          -/
    /-
      case node
      α : Type u_1
      x : α
      r : Ordnode α
      hr : r.Balanced
      sr : r.Sized
      ls : Nat
      ll : Ordnode α
      lx : α
      lr : Ordnode α
      hl : (Ordnode.node ls ll lx lr).Balanced
      sl : (Ordnode.node ls ll lx lr).Sized
      ⊢ Eq ((Ordnode.node ls ll lx lr).balance x r) ((Ordnode.node ls ll lx lr).bala …
    -/
  · cases' r with rs rl rx rr
      /-
        case node.nil
        α : Type u_1
        x : α
        ls : Nat
        ll : Ordnode α
        lx : α
        lr : Ordnode α
        hl : (Ordnode.node ls ll lx lr).Balanced
        sl : (Ordnode.node ls ll lx lr).Sized
        hr : Ordnode.nil.Balanced
        sr : Ordnode.nil.Sized
        ⊢ Eq ((Ordnode.node ls ll lx lr).balance x Ordnode.nil) ((Ordnode.node ls ll l …
      -/
    · rw [sl.eq_node'] at hl ⊢
      /-
        case node.nil
        α : Type u_1
        x : α
        ls : Nat
        ll : Ordnode α
        lx : α
        lr : Ordnode α
        hl : (ll.node' lx lr).Balanced
        sl : (Ordnode.node ls ll lx lr).Sized
        hr : Ordnode.nil.Balanced
        sr : Ordnode.nil.Sized
        ⊢ Eq ((ll.node' lx lr).balance x Ordnode.nil) ((ll.node' lx lr).balance' x Ord …
      -/
      cases' ll with lls lll llx llr <;> cases' lr with lrs lrl lrx lrr <;>
        /-
          case node.nil.nil.nil
          α : Type u_1
          x : α
          ls : Nat
          lx : α
          hr : Ordnode.nil.Balanced
          sr : Ordnode.nil.Sized
          hl : (Ordnode.nil.node' lx Ordnode.nil).Balanced
          sl : (Ordnode.node ls Ordnode.nil lx Ordnode.nil).Sized
          ⊢ Eq ((Ordnode.nil.node' lx Ordnode.nil).balance x Ordnode.nil) ((Ordnode.nil. …
        -/
        dsimp [balance, balance']
        /-
          case node.nil.nil.nil
          α : Type u_1
          x : α
          ls : Nat
          lx : α
          hr : Ordnode.nil.Balanced
          sr : Ordnode.nil.Sized
          hl : (Ordnode.nil.node' lx Ordnode.nil).Balanced
          sl : (Ordnode.node ls Ordnode.nil lx Ordnode.nil).Sized
          ⊢ Eq (Ordnode.node 2 (Ordnode.nil.node' lx Ordnode.nil) x Ordnode.nil) ((Ordno …
        -/
      · rfl
        /-
          🎉 no goals
        -/
      · have : size lrl = 0 ∧ size lrr = 0 := by
          have := balancedSz_zero.1 hl.1.symm
          rwa [size, sl.2.2.1, Nat.succ_le_succ_iff, Nat.le_zero, add_eq_zero] at this
        /-
          case node.nil.nil.node
          α : Type u_1
          x : α
          ls : Nat
          lx : α
          hr : Ordnode.nil.Balanced
          sr : Ordnode.nil.Sized
          lrs : Nat
          lrl : Ordnode α
          lrx : α
          lrr : Ordnode α
          hl : (Ordnode.nil.node' lx (Ordnode.node lrs lrl lrx lrr)).Balanced
          sl : (Ordnode.node ls Ordnode.nil lx (Ordnode.node lrs lrl lrx lrr)).Sized
          this : And (Eq lrl.size 0) (Eq lrr.size 0)
          ⊢ Eq (Ordnode.node 3 (Ordnode.singleton lx) lrx (Ordnode.singleton x)) (ite (L …
        -/
        cases sl.2.2.2.1.size_eq_zero.1 this.1
        /-
          case node.nil.nil.node.refl
          α : Type u_1
          x : α
          ls : Nat
          lx : α
          hr : Ordnode.nil.Balanced
          sr : Ordnode.nil.Sized
          lrs : Nat
          lrx : α
          lrr : Ordnode α
          hl : (Ordnode.nil.node' lx (Ordnode.node lrs Ordnode.nil lrx lrr)).Balanced
          sl : (Ordnode.node ls Ordnode.nil lx (Ordnode.node lrs Ordnode.nil lrx lrr)).S …
          this : And (Eq Ordnode.nil.size 0) (Eq lrr.size 0)
          ⊢ Eq (Ordnode.node 3 (Ordnode.singleton lx) lrx (Ordnode.singleton x)) (ite (L …
        -/
        cases sl.2.2.2.2.size_eq_zero.1 this.2
        /-
          case node.nil.nil.node.refl.refl
          α : Type u_1
          x : α
          ls : Nat
          lx : α
          hr : Ordnode.nil.Balanced
          sr : Ordnode.nil.Sized
          lrs : Nat
          lrx : α
          hl : (Ordnode.nil.node' lx (Ordnode.node lrs Ordnode.nil lrx Ordnode.nil)).Bal …
          sl : (Ordnode.node ls Ordnode.nil lx (Ordnode.node lrs Ordnode.nil lrx Ordnode …
          this : And (Eq Ordnode.nil.size 0) (Eq Ordnode.nil.size 0)
          ⊢ Eq (Ordnode.node 3 (Ordnode.singleton lx) lrx (Ordnode.singleton x)) (ite (L …
        -/
        obtain rfl : lrs = 1 := sl.2.2.1
        /-
          case node.nil.nil.node.refl.refl
          α : Type u_1
          x : α
          ls : Nat
          lx : α
          hr : Ordnode.nil.Balanced
          sr : Ordnode.nil.Sized
          lrx : α
          this : And (Eq Ordnode.nil.size 0) (Eq Ordnode.nil.size 0)
          hl : (Ordnode.nil.node' lx (Ordnode.node 1 Ordnode.nil lrx Ordnode.nil)).Balan …
          sl : (Ordnode.node ls Ordnode.nil lx (Ordnode.node 1 Ordnode.nil lrx Ordnode.n …
          ⊢ Eq (Ordnode.node 3 (Ordnode.singleton lx) lrx (Ordnode.singleton x)) (ite (L …
        -/
        rw [if_neg, if_pos, rotateR_node, if_neg]; · rfl
                                                     /-
                                                       🎉 no goals
                                                     -/
        /-
          case node.nil.nil.node.refl.refl.hnc
          α : Type u_1
          x : α
          ls : Nat
          lx : α
          hr : Ordnode.nil.Balanced
          sr : Ordnode.nil.Sized
          lrx : α
          this : And (Eq Ordnode.nil.size 0) (Eq Ordnode.nil.size 0)
          hl : (Ordnode.nil.node' lx (Ordnode.node 1 Ordnode.nil lrx Ordnode.nil)).Balan …
          sl : (Ordnode.node ls Ordnode.nil lx (Ordnode.node 1 Ordnode.nil lrx Ordnode.n …
          ⊢ Not (LT.lt (Ordnode.node 1 Ordnode.nil lrx Ordnode.nil).size (HMul.hMul Ordn …
        -/
        all_goals dsimp only [size]; decide
        /-
          🎉 no goals
        -/
      · have : size lll = 0 ∧ size llr = 0 := by
          have := balancedSz_zero.1 hl.1
          rwa [size, sl.2.1.1, Nat.succ_le_succ_iff, Nat.le_zero, add_eq_zero] at this
        /-
          case node.nil.node.nil
          α : Type u_1
          x : α
          ls : Nat
          lx : α
          hr : Ordnode.nil.Balanced
          sr : Ordnode.nil.Sized
          lls : Nat
          lll : Ordnode α
          llx : α
          llr : Ordnode α
          hl : ((Ordnode.node lls lll llx llr).node' lx Ordnode.nil).Balanced
          sl : (Ordnode.node ls (Ordnode.node lls lll llx llr) lx Ordnode.nil).Sized
          this : And (Eq lll.size 0) (Eq llr.size 0)
          ⊢ Eq (Ordnode.node 3 (Ordnode.node lls lll llx llr) lx (Ordnode.singleton x))  …
        -/
        cases sl.2.1.2.1.size_eq_zero.1 this.1
        /-
          case node.nil.node.nil.refl
          α : Type u_1
          x : α
          ls : Nat
          lx : α
          hr : Ordnode.nil.Balanced
          sr : Ordnode.nil.Sized
          lls : Nat
          llx : α
          llr : Ordnode α
          hl : ((Ordnode.node lls Ordnode.nil llx llr).node' lx Ordnode.nil).Balanced
          sl : (Ordnode.node ls (Ordnode.node lls Ordnode.nil llx llr) lx Ordnode.nil).S …
          this : And (Eq Ordnode.nil.size 0) (Eq llr.size 0)
          ⊢ Eq (Ordnode.node 3 (Ordnode.node lls Ordnode.nil llx llr) lx (Ordnode.single …
        -/
        cases sl.2.1.2.2.size_eq_zero.1 this.2
        /-
          case node.nil.node.nil.refl.refl
          α : Type u_1
          x : α
          ls : Nat
          lx : α
          hr : Ordnode.nil.Balanced
          sr : Ordnode.nil.Sized
          lls : Nat
          llx : α
          hl : ((Ordnode.node lls Ordnode.nil llx Ordnode.nil).node' lx Ordnode.nil).Bal …
          sl : (Ordnode.node ls (Ordnode.node lls Ordnode.nil llx Ordnode.nil) lx Ordnod …
          this : And (Eq Ordnode.nil.size 0) (Eq Ordnode.nil.size 0)
          ⊢ Eq (Ordnode.node 3 (Ordnode.node lls Ordnode.nil llx Ordnode.nil) lx (Ordnod …
        -/
        obtain rfl : lls = 1 := sl.2.1.1
        /-
          case node.nil.node.nil.refl.refl
          α : Type u_1
          x : α
          ls : Nat
          lx : α
          hr : Ordnode.nil.Balanced
          sr : Ordnode.nil.Sized
          llx : α
          this : And (Eq Ordnode.nil.size 0) (Eq Ordnode.nil.size 0)
          hl : ((Ordnode.node 1 Ordnode.nil llx Ordnode.nil).node' lx Ordnode.nil).Balan …
          sl : (Ordnode.node ls (Ordnode.node 1 Ordnode.nil llx Ordnode.nil) lx Ordnode. …
          ⊢ Eq (Ordnode.node 3 (Ordnode.node 1 Ordnode.nil llx Ordnode.nil) lx (Ordnode. …
        -/
        rw [if_neg, if_pos, rotateR_node, if_pos]; · rfl
                                                     /-
                                                       🎉 no goals
                                                     -/
        /-
          case node.nil.node.nil.refl.refl.hc
          α : Type u_1
          x : α
          ls : Nat
          lx : α
          hr : Ordnode.nil.Balanced
          sr : Ordnode.nil.Sized
          llx : α
          this : And (Eq Ordnode.nil.size 0) (Eq Ordnode.nil.size 0)
          hl : ((Ordnode.node 1 Ordnode.nil llx Ordnode.nil).node' lx Ordnode.nil).Balan …
          sl : (Ordnode.node ls (Ordnode.node 1 Ordnode.nil llx Ordnode.nil) lx Ordnode. …
          ⊢ LT.lt Ordnode.nil.size (HMul.hMul Ordnode.ratio (Ordnode.node 1 Ordnode.nil  …
        -/
        all_goals dsimp only [size]; decide
        /-
          🎉 no goals
        -/
        /-
          case node.nil.node.node
          α : Type u_1
          x : α
          ls : Nat
          lx : α
          hr : Ordnode.nil.Balanced
          sr : Ordnode.nil.Sized
          lls : Nat
          lll : Ordnode α
          llx : α
          llr : Ordnode α
          lrs : Nat
          lrl : Ordnode α
          lrx : α
          lrr : Ordnode α
          hl : ((Ordnode.node lls lll llx llr).node' lx (Ordnode.node lrs lrl lrx lrr)). …
          sl : (Ordnode.node ls (Ordnode.node lls lll llx llr) lx (Ordnode.node lrs lrl  …
          ⊢ Eq (ite (LT.lt lrs (HMul.hMul Ordnode.ratio lls)) (Ordnode.node (HAdd.hAdd ( …
        -/
      · symm; rw [if_neg, if_pos, rotateR]
          /-
            case node.nil.node.node
            α : Type u_1
            x : α
            ls : Nat
            lx : α
            hr : Ordnode.nil.Balanced
            sr : Ordnode.nil.Sized
            lls : Nat
            lll : Ordnode α
            llx : α
            llr : Ordnode α
            lrs : Nat
            lrl : Ordnode α
            lrx : α
            lrr : Ordnode α
            hl : ((Ordnode.node lls lll llx llr).node' lx (Ordnode.node lrs lrl lrx lrr)). …
            sl : (Ordnode.node ls (Ordnode.node lls lll llx llr) lx (Ordnode.node lrs lrl  …
            ⊢ Eq (ite (LT.lt (Ordnode.node lrs lrl lrx lrr).size (HMul.hMul Ordnode.ratio  …
          -/
        · dsimp only [size_node]; split_ifs
            /-
              case pos
              α : Type u_1
              x : α
              ls : Nat
              lx : α
              hr : Ordnode.nil.Balanced
              sr : Ordnode.nil.Sized
              lls : Nat
              lll : Ordnode α
              llx : α
              llr : Ordnode α
              lrs : Nat
              lrl : Ordnode α
              lrx : α
              lrr : Ordnode α
              hl : ((Ordnode.node lls lll llx llr).node' lx (Ordnode.node lrs lrl lrx lrr)). …
              sl : (Ordnode.node ls (Ordnode.node lls lll llx llr) lx (Ordnode.node lrs lrl  …
              h✝ : LT.lt lrs (HMul.hMul Ordnode.ratio lls)
              ⊢ Eq ((Ordnode.node lls lll llx llr).node3R lx (Ordnode.node lrs lrl lrx lrr)  …
            -/
                                  /-
                                    🎉 no goals
                                  -/
          · simp [node3R, node']; abel
                                  /-
                                    🎉 no goals
                                  -/
            /-
              case neg
              α : Type u_1
              x : α
              ls : Nat
              lx : α
              hr : Ordnode.nil.Balanced
              sr : Ordnode.nil.Sized
              lls : Nat
              lll : Ordnode α
              llx : α
              llr : Ordnode α
              lrs : Nat
              lrl : Ordnode α
              lrx : α
              lrr : Ordnode α
              hl : ((Ordnode.node lls lll llx llr).node' lx (Ordnode.node lrs lrl lrx lrr)). …
              sl : (Ordnode.node ls (Ordnode.node lls lll llx llr) lx (Ordnode.node lrs lrl  …
              h✝ : Not (LT.lt lrs (HMul.hMul Ordnode.ratio lls))
              ⊢ Eq ((Ordnode.node lls lll llx llr).node4R lx (Ordnode.node lrs lrl lrx lrr)  …
            -/
                                            /-
                                              🎉 no goals
                                            -/
          · simp [node4R, node', sl.2.2.1]; abel
                                            /-
                                              🎉 no goals
                                            -/
          /-
            case node.nil.node.node.hc
            α : Type u_1
            x : α
            ls : Nat
            lx : α
            hr : Ordnode.nil.Balanced
            sr : Ordnode.nil.Sized
            lls : Nat
            lll : Ordnode α
            llx : α
            llr : Ordnode α
            lrs : Nat
            lrl : Ordnode α
            lrx : α
            lrr : Ordnode α
            hl : ((Ordnode.node lls lll llx llr).node' lx (Ordnode.node lrs lrl lrx lrr)). …
            sl : (Ordnode.node ls (Ordnode.node lls lll llx llr) lx (Ordnode.node lrs lrl  …
            ⊢ GT.gt (HAdd.hAdd (HAdd.hAdd lls lrs) 1) 0
          -/
        · apply Nat.zero_lt_succ
          /-
            🎉 no goals
          -/
          /-
            case node.nil.node.node.hnc
            α : Type u_1
            x : α
            ls : Nat
            lx : α
            hr : Ordnode.nil.Balanced
            sr : Ordnode.nil.Sized
            lls : Nat
            lll : Ordnode α
            llx : α
            llr : Ordnode α
            lrs : Nat
            lrl : Ordnode α
            lrx : α
            lrr : Ordnode α
            hl : ((Ordnode.node lls lll llx llr).node' lx (Ordnode.node lrs lrl lrx lrr)). …
            sl : (Ordnode.node ls (Ordnode.node lls lll llx llr) lx (Ordnode.node lrs lrl  …
            ⊢ Not (LE.le (HAdd.hAdd (HAdd.hAdd lls lrs) 1) 1)
          -/
        · exact not_le_of_gt (Nat.succ_lt_succ (add_pos sl.2.1.pos sl.2.2.pos))
          /-
            🎉 no goals
          -/
      /-
        case node.node
        α : Type u_1
        x : α
        ls : Nat
        ll : Ordnode α
        lx : α
        lr : Ordnode α
        hl : (Ordnode.node ls ll lx lr).Balanced
        sl : (Ordnode.node ls ll lx lr).Sized
        rs : Nat
        rl : Ordnode α
        rx : α
        rr : Ordnode α
        hr : (Ordnode.node rs rl rx rr).Balanced
        sr : (Ordnode.node rs rl rx rr).Sized
        ⊢ Eq ((Ordnode.node ls ll lx lr).balance x (Ordnode.node rs rl rx rr)) ((Ordno …
      -/
    · simp [balance, balance']
      /-
        case node.node
        α : Type u_1
        x : α
        ls : Nat
        ll : Ordnode α
        lx : α
        lr : Ordnode α
        hl : (Ordnode.node ls ll lx lr).Balanced
        sl : (Ordnode.node ls ll lx lr).Sized
        rs : Nat
        rl : Ordnode α
        rx : α
        rr : Ordnode α
        hr : (Ordnode.node rs rl rx rr).Balanced
        sr : (Ordnode.node rs rl rx rr).Sized
        ⊢ Eq (ite (LT.lt (HMul.hMul Ordnode.delta ls) rs) (Ordnode.rec Ordnode.nil (fu …
      -/
      symm; rw [if_neg]
        /-
          case node.node
          α : Type u_1
          x : α
          ls : Nat
          ll : Ordnode α
          lx : α
          lr : Ordnode α
          hl : (Ordnode.node ls ll lx lr).Balanced
          sl : (Ordnode.node ls ll lx lr).Sized
          rs : Nat
          rl : Ordnode α
          rx : α
          rr : Ordnode α
          hr : (Ordnode.node rs rl rx rr).Balanced
          sr : (Ordnode.node rs rl rx rr).Sized
          ⊢ Eq (ite (LT.lt (HMul.hMul Ordnode.delta ls) rs) ((Ordnode.node ls ll lx lr). …
        -/
      · split_ifs with h h_1
        · have rd : delta ≤ size rl + size rr := by
            have := lt_of_le_of_lt (Nat.mul_le_mul_left _ sl.pos) h
            rwa [sr.1, Nat.lt_succ_iff] at this
          /-
            case pos
            α : Type u_1
            x : α
            ls : Nat
            ll : Ordnode α
            lx : α
            lr : Ordnode α
            hl : (Ordnode.node ls ll lx lr).Balanced
            sl : (Ordnode.node ls ll lx lr).Sized
            rs : Nat
            rl : Ordnode α
            rx : α
            rr : Ordnode α
            hr : (Ordnode.node rs rl rx rr).Balanced
            sr : (Ordnode.node rs rl rx rr).Sized
            h : LT.lt (HMul.hMul Ordnode.delta ls) rs
            rd : LE.le Ordnode.delta (HAdd.hAdd rl.size rr.size)
            ⊢ Eq ((Ordnode.node ls ll lx lr).rotateL x (Ordnode.node rs rl rx rr)) (Ordnod …
          -/
          cases' rl with rls rll rlx rlr
            /-
              case pos.nil
              α : Type u_1
              x : α
              ls : Nat
              ll : Ordnode α
              lx : α
              lr : Ordnode α
              hl : (Ordnode.node ls ll lx lr).Balanced
              sl : (Ordnode.node ls ll lx lr).Sized
              rs : Nat
              rx : α
              rr : Ordnode α
              h : LT.lt (HMul.hMul Ordnode.delta ls) rs
              hr : (Ordnode.node rs Ordnode.nil rx rr).Balanced
              sr : (Ordnode.node rs Ordnode.nil rx rr).Sized
              rd : LE.le Ordnode.delta (HAdd.hAdd Ordnode.nil.size rr.size)
              ⊢ Eq ((Ordnode.node ls ll lx lr).rotateL x (Ordnode.node rs Ordnode.nil rx rr) …
            -/
          · rw [size, zero_add] at rd
            /-
              case pos.nil
              α : Type u_1
              x : α
              ls : Nat
              ll : Ordnode α
              lx : α
              lr : Ordnode α
              hl : (Ordnode.node ls ll lx lr).Balanced
              sl : (Ordnode.node ls ll lx lr).Sized
              rs : Nat
              rx : α
              rr : Ordnode α
              h : LT.lt (HMul.hMul Ordnode.delta ls) rs
              hr : (Ordnode.node rs Ordnode.nil rx rr).Balanced
              sr : (Ordnode.node rs Ordnode.nil rx rr).Sized
              rd : LE.le Ordnode.delta rr.size
              ⊢ Eq ((Ordnode.node ls ll lx lr).rotateL x (Ordnode.node rs Ordnode.nil rx rr) …
            -/
            exact absurd (le_trans rd (balancedSz_zero.1 hr.1.symm)) (by decide)
            /-
              🎉 no goals
            -/
          /-
            case pos.node
            α : Type u_1
            x : α
            ls : Nat
            ll : Ordnode α
            lx : α
            lr : Ordnode α
            hl : (Ordnode.node ls ll lx lr).Balanced
            sl : (Ordnode.node ls ll lx lr).Sized
            rs : Nat
            rx : α
            rr : Ordnode α
            h : LT.lt (HMul.hMul Ordnode.delta ls) rs
            rls : Nat
            rll : Ordnode α
            rlx : α
            rlr : Ordnode α
            hr : (Ordnode.node rs (Ordnode.node rls rll rlx rlr) rx rr).Balanced
            sr : (Ordnode.node rs (Ordnode.node rls rll rlx rlr) rx rr).Sized
            rd : LE.le Ordnode.delta (HAdd.hAdd (Ordnode.node rls rll rlx rlr).size rr.size)
            ⊢ Eq ((Ordnode.node ls ll lx lr).rotateL x (Ordnode.node rs (Ordnode.node rls  …
          -/
          cases' rr with rrs rrl rrx rrr
            /-
              case pos.node.nil
              α : Type u_1
              x : α
              ls : Nat
              ll : Ordnode α
              lx : α
              lr : Ordnode α
              hl : (Ordnode.node ls ll lx lr).Balanced
              sl : (Ordnode.node ls ll lx lr).Sized
              rs : Nat
              rx : α
              h : LT.lt (HMul.hMul Ordnode.delta ls) rs
              rls : Nat
              rll : Ordnode α
              rlx : α
              rlr : Ordnode α
              hr : (Ordnode.node rs (Ordnode.node rls rll rlx rlr) rx Ordnode.nil).Balanced
              sr : (Ordnode.node rs (Ordnode.node rls rll rlx rlr) rx Ordnode.nil).Sized
              rd : LE.le Ordnode.delta (HAdd.hAdd (Ordnode.node rls rll rlx rlr).size Ordnod …
              ⊢ Eq ((Ordnode.node ls ll lx lr).rotateL x (Ordnode.node rs (Ordnode.node rls  …
            -/
          · exact absurd (le_trans rd (balancedSz_zero.1 hr.1)) (by decide)
            /-
              🎉 no goals
            -/
          /-
            case pos.node.node
            α : Type u_1
            x : α
            ls : Nat
            ll : Ordnode α
            lx : α
            lr : Ordnode α
            hl : (Ordnode.node ls ll lx lr).Balanced
            sl : (Ordnode.node ls ll lx lr).Sized
            rs : Nat
            rx : α
            h : LT.lt (HMul.hMul Ordnode.delta ls) rs
            rls : Nat
            rll : Ordnode α
            rlx : α
            rlr : Ordnode α
            rrs : Nat
            rrl : Ordnode α
            rrx : α
            rrr : Ordnode α
            hr : (Ordnode.node rs (Ordnode.node rls rll rlx rlr) rx (Ordnode.node rrs rrl  …
            sr : (Ordnode.node rs (Ordnode.node rls rll rlx rlr) rx (Ordnode.node rrs rrl  …
            rd : LE.le Ordnode.delta (HAdd.hAdd (Ordnode.node rls rll rlx rlr).size (Ordno …
            ⊢ Eq ((Ordnode.node ls ll lx lr).rotateL x (Ordnode.node rs (Ordnode.node rls  …
          -/
          dsimp [rotateL]; split_ifs
            /-
              case pos
              α : Type u_1
              x : α
              ls : Nat
              ll : Ordnode α
              lx : α
              lr : Ordnode α
              hl : (Ordnode.node ls ll lx lr).Balanced
              sl : (Ordnode.node ls ll lx lr).Sized
              rs : Nat
              rx : α
              h : LT.lt (HMul.hMul Ordnode.delta ls) rs
              rls : Nat
              rll : Ordnode α
              rlx : α
              rlr : Ordnode α
              rrs : Nat
              rrl : Ordnode α
              rrx : α
              rrr : Ordnode α
              hr : (Ordnode.node rs (Ordnode.node rls rll rlx rlr) rx (Ordnode.node rrs rrl  …
              sr : (Ordnode.node rs (Ordnode.node rls rll rlx rlr) rx (Ordnode.node rrs rrl  …
              rd : LE.le Ordnode.delta (HAdd.hAdd (Ordnode.node rls rll rlx rlr).size (Ordno …
              h✝ : LT.lt rls (HMul.hMul Ordnode.ratio rrs)
              ⊢ Eq ((Ordnode.node ls ll lx lr).node3L x (Ordnode.node rls rll rlx rlr) rx (O …
            -/
                                        /-
                                          🎉 no goals
                                        -/
          · simp [node3L, node', sr.1]; abel
                                        /-
                                          🎉 no goals
                                        -/
            /-
              case neg
              α : Type u_1
              x : α
              ls : Nat
              ll : Ordnode α
              lx : α
              lr : Ordnode α
              hl : (Ordnode.node ls ll lx lr).Balanced
              sl : (Ordnode.node ls ll lx lr).Sized
              rs : Nat
              rx : α
              h : LT.lt (HMul.hMul Ordnode.delta ls) rs
              rls : Nat
              rll : Ordnode α
              rlx : α
              rlr : Ordnode α
              rrs : Nat
              rrl : Ordnode α
              rrx : α
              rrr : Ordnode α
              hr : (Ordnode.node rs (Ordnode.node rls rll rlx rlr) rx (Ordnode.node rrs rrl  …
              sr : (Ordnode.node rs (Ordnode.node rls rll rlx rlr) rx (Ordnode.node rrs rrl  …
              rd : LE.le Ordnode.delta (HAdd.hAdd (Ordnode.node rls rll rlx rlr).size (Ordno …
              h✝ : Not (LT.lt rls (HMul.hMul Ordnode.ratio rrs))
              ⊢ Eq ((Ordnode.node ls ll lx lr).node4L x (Ordnode.node rls rll rlx rlr) rx (O …
            -/
                                                  /-
                                                    🎉 no goals
                                                  -/
          · simp [node4L, node', sr.1, sr.2.1.1]; abel
                                                  /-
                                                    🎉 no goals
                                                  -/
        · have ld : delta ≤ size ll + size lr := by
            have := lt_of_le_of_lt (Nat.mul_le_mul_left _ sr.pos) h_1
            rwa [sl.1, Nat.lt_succ_iff] at this
          /-
            case pos
            α : Type u_1
            x : α
            ls : Nat
            ll : Ordnode α
            lx : α
            lr : Ordnode α
            hl : (Ordnode.node ls ll lx lr).Balanced
            sl : (Ordnode.node ls ll lx lr).Sized
            rs : Nat
            rl : Ordnode α
            rx : α
            rr : Ordnode α
            hr : (Ordnode.node rs rl rx rr).Balanced
            sr : (Ordnode.node rs rl rx rr).Sized
            h : Not (LT.lt (HMul.hMul Ordnode.delta ls) rs)
            h_1 : LT.lt (HMul.hMul Ordnode.delta rs) ls
            ld : LE.le Ordnode.delta (HAdd.hAdd ll.size lr.size)
            ⊢ Eq ((Ordnode.node ls ll lx lr).rotateR x (Ordnode.node rs rl rx rr)) (Ordnod …
          -/
          cases' ll with lls lll llx llr
            /-
              case pos.nil
              α : Type u_1
              x : α
              ls : Nat
              lx : α
              lr : Ordnode α
              rs : Nat
              rl : Ordnode α
              rx : α
              rr : Ordnode α
              hr : (Ordnode.node rs rl rx rr).Balanced
              sr : (Ordnode.node rs rl rx rr).Sized
              h : Not (LT.lt (HMul.hMul Ordnode.delta ls) rs)
              h_1 : LT.lt (HMul.hMul Ordnode.delta rs) ls
              hl : (Ordnode.node ls Ordnode.nil lx lr).Balanced
              sl : (Ordnode.node ls Ordnode.nil lx lr).Sized
              ld : LE.le Ordnode.delta (HAdd.hAdd Ordnode.nil.size lr.size)
              ⊢ Eq ((Ordnode.node ls Ordnode.nil lx lr).rotateR x (Ordnode.node rs rl rx rr) …
            -/
          · rw [size, zero_add] at ld
            /-
              case pos.nil
              α : Type u_1
              x : α
              ls : Nat
              lx : α
              lr : Ordnode α
              rs : Nat
              rl : Ordnode α
              rx : α
              rr : Ordnode α
              hr : (Ordnode.node rs rl rx rr).Balanced
              sr : (Ordnode.node rs rl rx rr).Sized
              h : Not (LT.lt (HMul.hMul Ordnode.delta ls) rs)
              h_1 : LT.lt (HMul.hMul Ordnode.delta rs) ls
              hl : (Ordnode.node ls Ordnode.nil lx lr).Balanced
              sl : (Ordnode.node ls Ordnode.nil lx lr).Sized
              ld : LE.le Ordnode.delta lr.size
              ⊢ Eq ((Ordnode.node ls Ordnode.nil lx lr).rotateR x (Ordnode.node rs rl rx rr) …
            -/
            exact absurd (le_trans ld (balancedSz_zero.1 hl.1.symm)) (by decide)
            /-
              🎉 no goals
            -/
          /-
            case pos.node
            α : Type u_1
            x : α
            ls : Nat
            lx : α
            lr : Ordnode α
            rs : Nat
            rl : Ordnode α
            rx : α
            rr : Ordnode α
            hr : (Ordnode.node rs rl rx rr).Balanced
            sr : (Ordnode.node rs rl rx rr).Sized
            h : Not (LT.lt (HMul.hMul Ordnode.delta ls) rs)
            h_1 : LT.lt (HMul.hMul Ordnode.delta rs) ls
            lls : Nat
            lll : Ordnode α
            llx : α
            llr : Ordnode α
            hl : (Ordnode.node ls (Ordnode.node lls lll llx llr) lx lr).Balanced
            sl : (Ordnode.node ls (Ordnode.node lls lll llx llr) lx lr).Sized
            ld : LE.le Ordnode.delta (HAdd.hAdd (Ordnode.node lls lll llx llr).size lr.size)
            ⊢ Eq ((Ordnode.node ls (Ordnode.node lls lll llx llr) lx lr).rotateR x (Ordnod …
          -/
          cases' lr with lrs lrl lrx lrr
            /-
              case pos.node.nil
              α : Type u_1
              x : α
              ls : Nat
              lx : α
              rs : Nat
              rl : Ordnode α
              rx : α
              rr : Ordnode α
              hr : (Ordnode.node rs rl rx rr).Balanced
              sr : (Ordnode.node rs rl rx rr).Sized
              h : Not (LT.lt (HMul.hMul Ordnode.delta ls) rs)
              h_1 : LT.lt (HMul.hMul Ordnode.delta rs) ls
              lls : Nat
              lll : Ordnode α
              llx : α
              llr : Ordnode α
              hl : (Ordnode.node ls (Ordnode.node lls lll llx llr) lx Ordnode.nil).Balanced
              sl : (Ordnode.node ls (Ordnode.node lls lll llx llr) lx Ordnode.nil).Sized
              ld : LE.le Ordnode.delta (HAdd.hAdd (Ordnode.node lls lll llx llr).size Ordnod …
              ⊢ Eq ((Ordnode.node ls (Ordnode.node lls lll llx llr) lx Ordnode.nil).rotateR  …
            -/
          · exact absurd (le_trans ld (balancedSz_zero.1 hl.1)) (by decide)
            /-
              🎉 no goals
            -/
          /-
            case pos.node.node
            α : Type u_1
            x : α
            ls : Nat
            lx : α
            rs : Nat
            rl : Ordnode α
            rx : α
            rr : Ordnode α
            hr : (Ordnode.node rs rl rx rr).Balanced
            sr : (Ordnode.node rs rl rx rr).Sized
            h : Not (LT.lt (HMul.hMul Ordnode.delta ls) rs)
            h_1 : LT.lt (HMul.hMul Ordnode.delta rs) ls
            lls : Nat
            lll : Ordnode α
            llx : α
            llr : Ordnode α
            lrs : Nat
            lrl : Ordnode α
            lrx : α
            lrr : Ordnode α
            hl : (Ordnode.node ls (Ordnode.node lls lll llx llr) lx (Ordnode.node lrs lrl  …
            sl : (Ordnode.node ls (Ordnode.node lls lll llx llr) lx (Ordnode.node lrs lrl  …
            ld : LE.le Ordnode.delta (HAdd.hAdd (Ordnode.node lls lll llx llr).size (Ordno …
            ⊢ Eq ((Ordnode.node ls (Ordnode.node lls lll llx llr) lx (Ordnode.node lrs lrl …
          -/
          dsimp [rotateR]; split_ifs
            /-
              case pos
              α : Type u_1
              x : α
              ls : Nat
              lx : α
              rs : Nat
              rl : Ordnode α
              rx : α
              rr : Ordnode α
              hr : (Ordnode.node rs rl rx rr).Balanced
              sr : (Ordnode.node rs rl rx rr).Sized
              h : Not (LT.lt (HMul.hMul Ordnode.delta ls) rs)
              h_1 : LT.lt (HMul.hMul Ordnode.delta rs) ls
              lls : Nat
              lll : Ordnode α
              llx : α
              llr : Ordnode α
              lrs : Nat
              lrl : Ordnode α
              lrx : α
              lrr : Ordnode α
              hl : (Ordnode.node ls (Ordnode.node lls lll llx llr) lx (Ordnode.node lrs lrl  …
              sl : (Ordnode.node ls (Ordnode.node lls lll llx llr) lx (Ordnode.node lrs lrl  …
              ld : LE.le Ordnode.delta (HAdd.hAdd (Ordnode.node lls lll llx llr).size (Ordno …
              h✝ : LT.lt lrs (HMul.hMul Ordnode.ratio lls)
              ⊢ Eq ((Ordnode.node lls lll llx llr).node3R lx (Ordnode.node lrs lrl lrx lrr)  …
            -/
                                        /-
                                          🎉 no goals
                                        -/
          · simp [node3R, node', sl.1]; abel
                                        /-
                                          🎉 no goals
                                        -/
            /-
              case neg
              α : Type u_1
              x : α
              ls : Nat
              lx : α
              rs : Nat
              rl : Ordnode α
              rx : α
              rr : Ordnode α
              hr : (Ordnode.node rs rl rx rr).Balanced
              sr : (Ordnode.node rs rl rx rr).Sized
              h : Not (LT.lt (HMul.hMul Ordnode.delta ls) rs)
              h_1 : LT.lt (HMul.hMul Ordnode.delta rs) ls
              lls : Nat
              lll : Ordnode α
              llx : α
              llr : Ordnode α
              lrs : Nat
              lrl : Ordnode α
              lrx : α
              lrr : Ordnode α
              hl : (Ordnode.node ls (Ordnode.node lls lll llx llr) lx (Ordnode.node lrs lrl  …
              sl : (Ordnode.node ls (Ordnode.node lls lll llx llr) lx (Ordnode.node lrs lrl  …
              ld : LE.le Ordnode.delta (HAdd.hAdd (Ordnode.node lls lll llx llr).size (Ordno …
              h✝ : Not (LT.lt lrs (HMul.hMul Ordnode.ratio lls))
              ⊢ Eq ((Ordnode.node lls lll llx llr).node4R lx (Ordnode.node lrs lrl lrx lrr)  …
            -/
                                                  /-
                                                    🎉 no goals
                                                  -/
          · simp [node4R, node', sl.1, sl.2.2.1]; abel
                                                  /-
                                                    🎉 no goals
                                                  -/
          /-
            case neg
            α : Type u_1
            x : α
            ls : Nat
            ll : Ordnode α
            lx : α
            lr : Ordnode α
            hl : (Ordnode.node ls ll lx lr).Balanced
            sl : (Ordnode.node ls ll lx lr).Sized
            rs : Nat
            rl : Ordnode α
            rx : α
            rr : Ordnode α
            hr : (Ordnode.node rs rl rx rr).Balanced
            sr : (Ordnode.node rs rl rx rr).Sized
            h : Not (LT.lt (HMul.hMul Ordnode.delta ls) rs)
            h_1 : Not (LT.lt (HMul.hMul Ordnode.delta rs) ls)
            ⊢ Eq ((Ordnode.node ls ll lx lr).node' x (Ordnode.node rs rl rx rr)) (Ordnode. …
          -/
        · simp [node']
          /-
            🎉 no goals
          -/
        /-
          case node.node.hnc
          α : Type u_1
          x : α
          ls : Nat
          ll : Ordnode α
          lx : α
          lr : Ordnode α
          hl : (Ordnode.node ls ll lx lr).Balanced
          sl : (Ordnode.node ls ll lx lr).Sized
          rs : Nat
          rl : Ordnode α
          rx : α
          rr : Ordnode α
          hr : (Ordnode.node rs rl rx rr).Balanced
          sr : (Ordnode.node rs rl rx rr).Sized
          ⊢ Not (LE.le (HAdd.hAdd ls rs) 1)
        -/
      · exact not_le_of_gt (add_le_add (Nat.succ_le_of_lt sl.pos) (Nat.succ_le_of_lt sr.pos))
        /-
          🎉 no goals
        -/


theorem balanceL_eq_balance {l x r} (sl : Sized l) (sr : Sized r) (H1 : size l = 0 → size r ≤ 1)
    (H2 : 1 ≤ size l → 1 ≤ size r → size r ≤ delta * size l) :
    @balanceL α l x r = balance l x r := by
  /-
    α : Type u_1
    l : Ordnode α
    x : α
    r : Ordnode α
    sl : l.Sized
    sr : r.Sized
    H1 : Eq l.size 0 → LE.le r.size 1
    H2 : LE.le 1 l.size → LE.le 1 r.size → LE.le r.size (HMul.hMul Ordnode.delta l …
    ⊢ Eq (l.balanceL x r) (l.balance x r)
  -/
  cases' r with rs rl rx rr
    /-
      case nil
      α : Type u_1
      l : Ordnode α
      x : α
      sl : l.Sized
      sr : Ordnode.nil.Sized
      H1 : Eq l.size 0 → LE.le Ordnode.nil.size 1
      H2 : LE.le 1 l.size → LE.le 1 Ordnode.nil.size → LE.le Ordnode.nil.size (HMul. …
      ⊢ Eq (l.balanceL x Ordnode.nil) (l.balance x Ordnode.nil)
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case node
      α : Type u_1
      l : Ordnode α
      x : α
      sl : l.Sized
      rs : Nat
      rl : Ordnode α
      rx : α
      rr : Ordnode α
      sr : (Ordnode.node rs rl rx rr).Sized
      H1 : Eq l.size 0 → LE.le (Ordnode.node rs rl rx rr).size 1
      H2 : LE.le 1 l.size → LE.le 1 (Ordnode.node rs rl rx rr).size → LE.le (Ordnode …
      ⊢ Eq (l.balanceL x (Ordnode.node rs rl rx rr)) (l.balance x (Ordnode.node rs r …
    -/
  · cases' l with ls ll lx lr
    · have : size rl = 0 ∧ size rr = 0 := by
        have := H1 rfl
        rwa [size, sr.1, Nat.succ_le_succ_iff, Nat.le_zero, add_eq_zero] at this
      /-
        case node.nil
        α : Type u_1
        x : α
        rs : Nat
        rl : Ordnode α
        rx : α
        rr : Ordnode α
        sr : (Ordnode.node rs rl rx rr).Sized
        sl : Ordnode.nil.Sized
        H1 : Eq Ordnode.nil.size 0 → LE.le (Ordnode.node rs rl rx rr).size 1
        H2 : LE.le 1 Ordnode.nil.size → LE.le 1 (Ordnode.node rs rl rx rr).size → LE.l …
        this : And (Eq rl.size 0) (Eq rr.size 0)
        ⊢ Eq (Ordnode.nil.balanceL x (Ordnode.node rs rl rx rr)) (Ordnode.nil.balance  …
      -/
      cases sr.2.1.size_eq_zero.1 this.1
      /-
        case node.nil.refl
        α : Type u_1
        x : α
        rs : Nat
        rx : α
        rr : Ordnode α
        sl : Ordnode.nil.Sized
        sr : (Ordnode.node rs Ordnode.nil rx rr).Sized
        H1 : Eq Ordnode.nil.size 0 → LE.le (Ordnode.node rs Ordnode.nil rx rr).size 1
        H2 : LE.le 1 Ordnode.nil.size → LE.le 1 (Ordnode.node rs Ordnode.nil rx rr).si …
        this : And (Eq Ordnode.nil.size 0) (Eq rr.size 0)
        ⊢ Eq (Ordnode.nil.balanceL x (Ordnode.node rs Ordnode.nil rx rr)) (Ordnode.nil …
      -/
      cases sr.2.2.size_eq_zero.1 this.2
      /-
        case node.nil.refl.refl
        α : Type u_1
        x : α
        rs : Nat
        rx : α
        sl : Ordnode.nil.Sized
        sr : (Ordnode.node rs Ordnode.nil rx Ordnode.nil).Sized
        H1 : Eq Ordnode.nil.size 0 → LE.le (Ordnode.node rs Ordnode.nil rx Ordnode.nil …
        H2 : LE.le 1 Ordnode.nil.size → LE.le 1 (Ordnode.node rs Ordnode.nil rx Ordnod …
        this : And (Eq Ordnode.nil.size 0) (Eq Ordnode.nil.size 0)
        ⊢ Eq (Ordnode.nil.balanceL x (Ordnode.node rs Ordnode.nil rx Ordnode.nil)) (Or …
      -/
      rw [sr.eq_node']; rfl
                        /-
                          🎉 no goals
                        -/
      /-
        case node.node
        α : Type u_1
        x : α
        rs : Nat
        rl : Ordnode α
        rx : α
        rr : Ordnode α
        sr : (Ordnode.node rs rl rx rr).Sized
        ls : Nat
        ll : Ordnode α
        lx : α
        lr : Ordnode α
        sl : (Ordnode.node ls ll lx lr).Sized
        H1 : Eq (Ordnode.node ls ll lx lr).size 0 → LE.le (Ordnode.node rs rl rx rr).s …
        H2 : LE.le 1 (Ordnode.node ls ll lx lr).size → LE.le 1 (Ordnode.node rs rl rx  …
        ⊢ Eq ((Ordnode.node ls ll lx lr).balanceL x (Ordnode.node rs rl rx rr)) ((Ordn …
      -/
    · replace H2 : ¬rs > delta * ls := not_lt_of_le (H2 sl.pos sr.pos)
      /-
        case node.node
        α : Type u_1
        x : α
        rs : Nat
        rl : Ordnode α
        rx : α
        rr : Ordnode α
        sr : (Ordnode.node rs rl rx rr).Sized
        ls : Nat
        ll : Ordnode α
        lx : α
        lr : Ordnode α
        sl : (Ordnode.node ls ll lx lr).Sized
        H1 : Eq (Ordnode.node ls ll lx lr).size 0 → LE.le (Ordnode.node rs rl rx rr).s …
        H2 : Not (GT.gt rs (HMul.hMul Ordnode.delta ls))
        ⊢ Eq ((Ordnode.node ls ll lx lr).balanceL x (Ordnode.node rs rl rx rr)) ((Ordn …
      -/
                                                  /-
                                                    🎉 no goals
                                                  -/
      simp [balanceL, balance, H2]; split_ifs <;> simp [add_comm]
                                                  /-
                                                    🎉 no goals
                                                  -/


/-- `Raised n m` means `m` is either equal or one up from `n`. -/
def Raised (n m : ℕ) : Prop :=
  m = n ∨ m = n + 1


theorem raised_iff {n m} : Raised n m ↔ n ≤ m ∧ m ≤ n + 1 := by
  /-
    n m : Nat
    ⊢ Iff (Ordnode.Raised n m) (And (LE.le n m) (LE.le m (HAdd.hAdd n 1)))
  -/
  constructor
    /-
      case mp
      n m : Nat
      ⊢ Ordnode.Raised n m → And (LE.le n m) (LE.le m (HAdd.hAdd n 1))
    -/
  · rintro (rfl | rfl)
      /-
        case mp.inl
        m : Nat
        ⊢ And (LE.le m m) (LE.le m (HAdd.hAdd m 1))
      -/
    · exact ⟨le_rfl, Nat.le_succ _⟩
      /-
        🎉 no goals
      -/
      /-
        case mp.inr
        n : Nat
        ⊢ And (LE.le n (HAdd.hAdd n 1)) (LE.le (HAdd.hAdd n 1) (HAdd.hAdd n 1))
      -/
    · exact ⟨Nat.le_succ _, le_rfl⟩
      /-
        🎉 no goals
      -/
    /-
      case mpr
      n m : Nat
      ⊢ And (LE.le n m) (LE.le m (HAdd.hAdd n 1)) → Ordnode.Raised n m
    -/
  · rintro ⟨h₁, h₂⟩
    /-
      case mpr.intro
      n m : Nat
      h₁ : LE.le n m
      h₂ : LE.le m (HAdd.hAdd n 1)
      ⊢ Ordnode.Raised n m
    -/
    rcases eq_or_lt_of_le h₁ with (rfl | h₁)
      /-
        case mpr.intro.inl
        n : Nat
        h₁ : LE.le n n
        h₂ : LE.le n (HAdd.hAdd n 1)
        ⊢ Ordnode.Raised n n
      -/
    · exact Or.inl rfl
      /-
        🎉 no goals
      -/
      /-
        case mpr.intro.inr
        n m : Nat
        h₁✝ : LE.le n m
        h₂ : LE.le m (HAdd.hAdd n 1)
        h₁ : LT.lt n m
        ⊢ Ordnode.Raised n m
      -/
    · exact Or.inr (le_antisymm h₂ h₁)
      /-
        🎉 no goals
      -/


theorem Raised.dist_le {n m} (H : Raised n m) : Nat.dist n m ≤ 1 := by
  /-
    n m : Nat
    H : Ordnode.Raised n m
    ⊢ LE.le (n.dist m) 1
  -/
  cases' raised_iff.1 H with H1 H2; rwa [Nat.dist_eq_sub_of_le H1, tsub_le_iff_left]
                                    /-
                                      🎉 no goals
                                    -/


theorem Raised.dist_le' {n m} (H : Raised n m) : Nat.dist m n ≤ 1 := by
  /-
    n m : Nat
    H : Ordnode.Raised n m
    ⊢ LE.le (m.dist n) 1
  -/
  rw [Nat.dist_comm]; exact H.dist_le
                      /-
                        🎉 no goals
                      -/


theorem Raised.add_left (k) {n m} (H : Raised n m) : Raised (k + n) (k + m) := by
  /-
    k n m : Nat
    H : Ordnode.Raised n m
    ⊢ Ordnode.Raised (HAdd.hAdd k n) (HAdd.hAdd k m)
  -/
  rcases H with (rfl | rfl)
    /-
      case inl
      k m : Nat
      ⊢ Ordnode.Raised (HAdd.hAdd k m) (HAdd.hAdd k m)
    -/
  · exact Or.inl rfl
    /-
      🎉 no goals
    -/
    /-
      case inr
      k n : Nat
      ⊢ Ordnode.Raised (HAdd.hAdd k n) (HAdd.hAdd k (HAdd.hAdd n 1))
    -/
  · exact Or.inr rfl
    /-
      🎉 no goals
    -/


theorem Raised.add_right (k) {n m} (H : Raised n m) : Raised (n + k) (m + k) := by
  /-
    k n m : Nat
    H : Ordnode.Raised n m
    ⊢ Ordnode.Raised (HAdd.hAdd n k) (HAdd.hAdd m k)
  -/
  rw [add_comm, add_comm m]; exact H.add_left _
                             /-
                               🎉 no goals
                             -/


theorem Raised.right {l x₁ x₂ r₁ r₂} (H : Raised (size r₁) (size r₂)) :
    Raised (size (@node' α l x₁ r₁)) (size (@node' α l x₂ r₂)) := by
  /-
    α : Type u_1
    l : Ordnode α
    x₁ x₂ : α
    r₁ r₂ : Ordnode α
    H : Ordnode.Raised r₁.size r₂.size
    ⊢ Ordnode.Raised (l.node' x₁ r₁).size (l.node' x₂ r₂).size
  -/
  rw [node', size_node, size_node]; generalize size r₂ = m at H ⊢
  /-
    α : Type u_1
    l : Ordnode α
    x₁ x₂ : α
    r₁ r₂ : Ordnode α
    m : Nat
    H : Ordnode.Raised r₁.size m
    ⊢ Ordnode.Raised (HAdd.hAdd (HAdd.hAdd l.size r₁.size) 1) (HAdd.hAdd (HAdd.hAd …
  -/
  rcases H with (rfl | rfl)
    /-
      case inl
      α : Type u_1
      l : Ordnode α
      x₁ x₂ : α
      r₁ r₂ : Ordnode α
      ⊢ Ordnode.Raised (HAdd.hAdd (HAdd.hAdd l.size r₁.size) 1) (HAdd.hAdd (HAdd.hAd …
    -/
  · exact Or.inl rfl
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      l : Ordnode α
      x₁ x₂ : α
      r₁ r₂ : Ordnode α
      ⊢ Ordnode.Raised (HAdd.hAdd (HAdd.hAdd l.size r₁.size) 1) (HAdd.hAdd (HAdd.hAd …
    -/
  · exact Or.inr rfl
    /-
      🎉 no goals
    -/


theorem balanceL_eq_balance' {l x r} (hl : Balanced l) (hr : Balanced r) (sl : Sized l)
    (sr : Sized r)
    (H :
      (∃ l', Raised l' (size l) ∧ BalancedSz l' (size r)) ∨
        ∃ r', Raised (size r) r' ∧ BalancedSz (size l) r') :
    @balanceL α l x r = balance' l x r := by
  /-
    α : Type u_1
    l : Ordnode α
    x : α
    r : Ordnode α
    hl : l.Balanced
    hr : r.Balanced
    sl : l.Sized
    sr : r.Sized
    H : Or (Exists fun l' => And (Ordnode.Raised l' l.size) (Ordnode.BalancedSz l' …
    ⊢ Eq (l.balanceL x r) (l.balance' x r)
  -/
  rw [← balance_eq_balance' hl hr sl sr, balanceL_eq_balance sl sr]
    /-
      case H1
      α : Type u_1
      l : Ordnode α
      x : α
      r : Ordnode α
      hl : l.Balanced
      hr : r.Balanced
      sl : l.Sized
      sr : r.Sized
      H : Or (Exists fun l' => And (Ordnode.Raised l' l.size) (Ordnode.BalancedSz l' …
      ⊢ Eq l.size 0 → LE.le r.size 1
    -/
  · intro l0; rw [l0] at H
    /-
      case H1
      α : Type u_1
      l : Ordnode α
      x : α
      r : Ordnode α
      hl : l.Balanced
      hr : r.Balanced
      sl : l.Sized
      sr : r.Sized
      H : Or (Exists fun l' => And (Ordnode.Raised l' 0) (Ordnode.BalancedSz l' r.si …
      l0 : Eq l.size 0
      ⊢ LE.le r.size 1
    -/
    rcases H with (⟨_, ⟨⟨⟩⟩ | ⟨⟨⟩⟩, H⟩ | ⟨r', e, H⟩)
      /-
        case H1.inl.intro.intro.inl.refl
        α : Type u_1
        l : Ordnode α
        x : α
        r : Ordnode α
        hl : l.Balanced
        hr : r.Balanced
        sl : l.Sized
        sr : r.Sized
        l0 : Eq l.size 0
        H : Ordnode.BalancedSz 0 r.size
        ⊢ LE.le r.size 1
      -/
    · exact balancedSz_zero.1 H.symm
      /-
        🎉 no goals
      -/
    /-
      case H1.inr.intro.intro
      α : Type u_1
      l : Ordnode α
      x : α
      r : Ordnode α
      hl : l.Balanced
      hr : r.Balanced
      sl : l.Sized
      sr : r.Sized
      l0 : Eq l.size 0
      r' : Nat
      e : Ordnode.Raised r.size r'
      H : Ordnode.BalancedSz 0 r'
      ⊢ LE.le r.size 1
    -/
    exact le_trans (raised_iff.1 e).1 (balancedSz_zero.1 H.symm)
    /-
      🎉 no goals
    -/
    /-
      case H2
      α : Type u_1
      l : Ordnode α
      x : α
      r : Ordnode α
      hl : l.Balanced
      hr : r.Balanced
      sl : l.Sized
      sr : r.Sized
      H : Or (Exists fun l' => And (Ordnode.Raised l' l.size) (Ordnode.BalancedSz l' …
      ⊢ LE.le 1 l.size → LE.le 1 r.size → LE.le r.size (HMul.hMul Ordnode.delta l.si …
    -/
  · intro l1 _
    /-
      case H2
      α : Type u_1
      l : Ordnode α
      x : α
      r : Ordnode α
      hl : l.Balanced
      hr : r.Balanced
      sl : l.Sized
      sr : r.Sized
      H : Or (Exists fun l' => And (Ordnode.Raised l' l.size) (Ordnode.BalancedSz l' …
      l1 : LE.le 1 l.size
      a✝ : LE.le 1 r.size
      ⊢ LE.le r.size (HMul.hMul Ordnode.delta l.size)
    -/
    rcases H with (⟨l', e, H | ⟨_, H₂⟩⟩ | ⟨r', e, H | ⟨_, H₂⟩⟩)
      /-
        case H2.inl.intro.intro.inl
        α : Type u_1
        l : Ordnode α
        x : α
        r : Ordnode α
        hl : l.Balanced
        hr : r.Balanced
        sl : l.Sized
        sr : r.Sized
        l1 : LE.le 1 l.size
        a✝ : LE.le 1 r.size
        l' : Nat
        e : Ordnode.Raised l' l.size
        H : LE.le (HAdd.hAdd l' r.size) 1
        ⊢ LE.le r.size (HMul.hMul Ordnode.delta l.size)
      -/
    · exact le_trans (le_trans (Nat.le_add_left _ _) H) (mul_pos (by decide) l1 : (0 : ℕ) < _)
      /-
        🎉 no goals
      -/
      /-
        case H2.inl.intro.intro.inr.intro
        α : Type u_1
        l : Ordnode α
        x : α
        r : Ordnode α
        hl : l.Balanced
        hr : r.Balanced
        sl : l.Sized
        sr : r.Sized
        l1 : LE.le 1 l.size
        a✝ : LE.le 1 r.size
        l' : Nat
        e : Ordnode.Raised l' l.size
        left✝ : LE.le l' (HMul.hMul Ordnode.delta r.size)
        H₂ : LE.le r.size (HMul.hMul Ordnode.delta l')
        ⊢ LE.le r.size (HMul.hMul Ordnode.delta l.size)
      -/
    · exact le_trans H₂ (Nat.mul_le_mul_left _ (raised_iff.1 e).1)
      /-
        🎉 no goals
      -/
      /-
        case H2.inr.intro.intro.inl
        α : Type u_1
        l : Ordnode α
        x : α
        r : Ordnode α
        hl : l.Balanced
        hr : r.Balanced
        sl : l.Sized
        sr : r.Sized
        l1 : LE.le 1 l.size
        a✝ : LE.le 1 r.size
        r' : Nat
        e : Ordnode.Raised r.size r'
        H : LE.le (HAdd.hAdd l.size r') 1
        ⊢ LE.le r.size (HMul.hMul Ordnode.delta l.size)
      -/
    · cases raised_iff.1 e; unfold delta; omega
                                          /-
                                            🎉 no goals
                                          -/
      /-
        case H2.inr.intro.intro.inr.intro
        α : Type u_1
        l : Ordnode α
        x : α
        r : Ordnode α
        hl : l.Balanced
        hr : r.Balanced
        sl : l.Sized
        sr : r.Sized
        l1 : LE.le 1 l.size
        a✝ : LE.le 1 r.size
        r' : Nat
        e : Ordnode.Raised r.size r'
        left✝ : LE.le l.size (HMul.hMul Ordnode.delta r')
        H₂ : LE.le r' (HMul.hMul Ordnode.delta l.size)
        ⊢ LE.le r.size (HMul.hMul Ordnode.delta l.size)
      -/
    · exact le_trans (raised_iff.1 e).1 H₂
      /-
        🎉 no goals
      -/


theorem balance_sz_dual {l r}
    (H : (∃ l', Raised (@size α l) l' ∧ BalancedSz l' (@size α r)) ∨
        ∃ r', Raised r' (size r) ∧ BalancedSz (size l) r') :
    (∃ l', Raised l' (size (dual r)) ∧ BalancedSz l' (size (dual l))) ∨
      ∃ r', Raised (size (dual l)) r' ∧ BalancedSz (size (dual r)) r' := by
  /-
    α : Type u_1
    l r : Ordnode α
    H : Or (Exists fun l' => And (Ordnode.Raised l.size l') (Ordnode.BalancedSz l' …
    ⊢ Or (Exists fun l' => And (Ordnode.Raised l' r.dual.size) (Ordnode.BalancedSz …
  -/
  rw [size_dual, size_dual]
  exact
    H.symm.imp (Exists.imp fun _ => And.imp_right BalancedSz.symm)
      (Exists.imp fun _ => And.imp_right BalancedSz.symm)


theorem size_balanceL {l x r} (hl : Balanced l) (hr : Balanced r) (sl : Sized l) (sr : Sized r)
    (H : (∃ l', Raised l' (size l) ∧ BalancedSz l' (size r)) ∨
        ∃ r', Raised (size r) r' ∧ BalancedSz (size l) r') :
    size (@balanceL α l x r) = size l + size r + 1 := by
  /-
    α : Type u_1
    l : Ordnode α
    x : α
    r : Ordnode α
    hl : l.Balanced
    hr : r.Balanced
    sl : l.Sized
    sr : r.Sized
    H : Or (Exists fun l' => And (Ordnode.Raised l' l.size) (Ordnode.BalancedSz l' …
    ⊢ Eq (l.balanceL x r).size (HAdd.hAdd (HAdd.hAdd l.size r.size) 1)
  -/
  rw [balanceL_eq_balance' hl hr sl sr H, size_balance' sl sr]
  /-
    🎉 no goals
  -/


theorem all_balanceL {P l x r} (hl : Balanced l) (hr : Balanced r) (sl : Sized l) (sr : Sized r)
    (H :
      (∃ l', Raised l' (size l) ∧ BalancedSz l' (size r)) ∨
        ∃ r', Raised (size r) r' ∧ BalancedSz (size l) r') :
    All P (@balanceL α l x r) ↔ All P l ∧ P x ∧ All P r := by
  /-
    α : Type u_1
    P : α → Prop
    l : Ordnode α
    x : α
    r : Ordnode α
    hl : l.Balanced
    hr : r.Balanced
    sl : l.Sized
    sr : r.Sized
    H : Or (Exists fun l' => And (Ordnode.Raised l' l.size) (Ordnode.BalancedSz l' …
    ⊢ Iff (Ordnode.All P (l.balanceL x r)) (And (Ordnode.All P l) (And (P x) (Ordn …
  -/
  rw [balanceL_eq_balance' hl hr sl sr H, all_balance']
  /-
    🎉 no goals
  -/


theorem balanceR_eq_balance' {l x r} (hl : Balanced l) (hr : Balanced r) (sl : Sized l)
    (sr : Sized r)
    (H : (∃ l', Raised (size l) l' ∧ BalancedSz l' (size r)) ∨
        ∃ r', Raised r' (size r) ∧ BalancedSz (size l) r') :
    @balanceR α l x r = balance' l x r := by
  rw [← dual_dual (balanceR l x r), dual_balanceR,
    balanceL_eq_balance' hr.dual hl.dual sr.dual sl.dual (balance_sz_dual H), ← dual_balance',
    dual_dual]


theorem size_balanceR {l x r} (hl : Balanced l) (hr : Balanced r) (sl : Sized l) (sr : Sized r)
    (H : (∃ l', Raised (size l) l' ∧ BalancedSz l' (size r)) ∨
        ∃ r', Raised r' (size r) ∧ BalancedSz (size l) r') :
    size (@balanceR α l x r) = size l + size r + 1 := by
  /-
    α : Type u_1
    l : Ordnode α
    x : α
    r : Ordnode α
    hl : l.Balanced
    hr : r.Balanced
    sl : l.Sized
    sr : r.Sized
    H : Or (Exists fun l' => And (Ordnode.Raised l.size l') (Ordnode.BalancedSz l' …
    ⊢ Eq (l.balanceR x r).size (HAdd.hAdd (HAdd.hAdd l.size r.size) 1)
  -/
  rw [balanceR_eq_balance' hl hr sl sr H, size_balance' sl sr]
  /-
    🎉 no goals
  -/


theorem all_balanceR {P l x r} (hl : Balanced l) (hr : Balanced r) (sl : Sized l) (sr : Sized r)
    (H :
      (∃ l', Raised (size l) l' ∧ BalancedSz l' (size r)) ∨
        ∃ r', Raised r' (size r) ∧ BalancedSz (size l) r') :
    All P (@balanceR α l x r) ↔ All P l ∧ P x ∧ All P r := by
  /-
    α : Type u_1
    P : α → Prop
    l : Ordnode α
    x : α
    r : Ordnode α
    hl : l.Balanced
    hr : r.Balanced
    sl : l.Sized
    sr : r.Sized
    H : Or (Exists fun l' => And (Ordnode.Raised l.size l') (Ordnode.BalancedSz l' …
    ⊢ Iff (Ordnode.All P (l.balanceR x r)) (And (Ordnode.All P l) (And (P x) (Ordn …
  -/
  rw [balanceR_eq_balance' hl hr sl sr H, all_balance']
  /-
    🎉 no goals
  -/


/-- `Bounded t lo hi` says that every element `x ∈ t` is in the range `lo < x < hi`, and also this
property holds recursively in subtrees, making the full tree a BST. The bounds can be set to
`lo = ⊥` and `hi = ⊤` if we care only about the internal ordering constraints. -/
def Bounded : Ordnode α → WithBot α → WithTop α → Prop
  | nil, some a, some b => a < b
  | nil, _, _ => True
  | node _ l x r, o₁, o₂ => Bounded l o₁ x ∧ Bounded r (↑x) o₂


theorem Bounded.dual :
    ∀ {t : Ordnode α} {o₁ o₂}, Bounded t o₁ o₂ → @Bounded αᵒᵈ _ (dual t) o₂ o₁
                         /-
                           α : Type u_1
                           inst✝ : Preorder α
                           o₁ : WithBot α
                           o₂ : WithTop α
                           h : Ordnode.nil.Bounded o₁ o₂
                           ⊢ Ordnode.nil.dual.Bounded o₂ o₁
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
  | nil, o₁, o₂, h => by cases o₁ <;> cases o₂ <;> trivial
                                                   /-
                                                     🎉 no goals
                                                   -/
  | node _ _ _ _, _, _, ⟨ol, Or⟩ => ⟨Or.dual, ol.dual⟩


theorem Bounded.dual_iff {t : Ordnode α} {o₁ o₂} :
    Bounded t o₁ o₂ ↔ @Bounded αᵒᵈ _ (.dual t) o₂ o₁ :=
  ⟨Bounded.dual, fun h => by
    /-
      α : Type u_1
      inst✝ : Preorder α
      t : Ordnode α
      o₁ : WithBot α
      o₂ : WithTop α
      h : t.dual.Bounded o₂ o₁
      ⊢ t.Bounded o₁ o₂
    -/
    have := Bounded.dual h; rwa [dual_dual, OrderDual.Preorder.dual_dual] at this⟩
                            /-
                              🎉 no goals
                            -/


theorem Bounded.weak_left : ∀ {t : Ordnode α} {o₁ o₂}, Bounded t o₁ o₂ → Bounded t ⊥ o₂
                         /-
                           α : Type u_1
                           inst✝ : Preorder α
                           o₁ : WithBot α
                           o₂ : WithTop α
                           h : Ordnode.nil.Bounded o₁ o₂
                           ⊢ Ordnode.nil.Bounded Bot.bot o₂
                         -/
                                      /-
                                        🎉 no goals
                                      -/
  | nil, o₁, o₂, h => by cases o₂ <;> trivial
                                      /-
                                        🎉 no goals
                                      -/
  | node _ _ _ _, _, _, ⟨ol, Or⟩ => ⟨ol.weak_left, Or⟩


theorem Bounded.weak_right : ∀ {t : Ordnode α} {o₁ o₂}, Bounded t o₁ o₂ → Bounded t o₁ ⊤
                         /-
                           α : Type u_1
                           inst✝ : Preorder α
                           o₁ : WithBot α
                           o₂ : WithTop α
                           h : Ordnode.nil.Bounded o₁ o₂
                           ⊢ Ordnode.nil.Bounded o₁ Top.top
                         -/
                                      /-
                                        🎉 no goals
                                      -/
  | nil, o₁, o₂, h => by cases o₁ <;> trivial
                                      /-
                                        🎉 no goals
                                      -/
  | node _ _ _ _, _, _, ⟨ol, Or⟩ => ⟨ol, Or.weak_right⟩


theorem Bounded.weak {t : Ordnode α} {o₁ o₂} (h : Bounded t o₁ o₂) : Bounded t ⊥ ⊤ :=
  h.weak_left.weak_right


theorem Bounded.mono_left {x y : α} (xy : x ≤ y) :
    ∀ {t : Ordnode α} {o}, Bounded t y o → Bounded t x o
  | nil, none, _ => ⟨⟩
  | nil, some _, h => lt_of_le_of_lt xy h
  | node _ _ _ _, _o, ⟨ol, or⟩ => ⟨ol.mono_left xy, or⟩


theorem Bounded.mono_right {x y : α} (xy : x ≤ y) :
    ∀ {t : Ordnode α} {o}, Bounded t o x → Bounded t o y
  | nil, none, _ => ⟨⟩
  | nil, some _, h => lt_of_lt_of_le h xy
  | node _ _ _ _, _o, ⟨ol, or⟩ => ⟨ol, or.mono_right xy⟩


theorem Bounded.to_lt : ∀ {t : Ordnode α} {x y : α}, Bounded t x y → x < y
  | nil, _, _, h => h
  | node _ _ _ _, _, _, ⟨h₁, h₂⟩ => lt_trans h₁.to_lt h₂.to_lt


theorem Bounded.to_nil {t : Ordnode α} : ∀ {o₁ o₂}, Bounded t o₁ o₂ → Bounded nil o₁ o₂
  | none, _, _ => ⟨⟩
  | some _, none, _ => ⟨⟩
  | some _, some _, h => h.to_lt


theorem Bounded.trans_left {t₁ t₂ : Ordnode α} {x : α} :
    ∀ {o₁ o₂}, Bounded t₁ o₁ x → Bounded t₂ x o₂ → Bounded t₂ o₁ o₂
  | none, _, _, h₂ => h₂.weak_left
  | some _, _, h₁, h₂ => h₂.mono_left (le_of_lt h₁.to_lt)


theorem Bounded.trans_right {t₁ t₂ : Ordnode α} {x : α} :
    ∀ {o₁ o₂}, Bounded t₁ o₁ x → Bounded t₂ x o₂ → Bounded t₁ o₁ o₂
  | _, none, h₁, _ => h₁.weak_right
  | _, some _, h₁, h₂ => h₁.mono_right (le_of_lt h₂.to_lt)


theorem Bounded.mem_lt : ∀ {t o} {x : α}, Bounded t o x → All (· < x) t
  | nil, _, _, _ => ⟨⟩
  | node _ _ _ _, _, _, ⟨h₁, h₂⟩ =>
    ⟨h₁.mem_lt.imp fun _ h => lt_trans h h₂.to_lt, h₂.to_lt, h₂.mem_lt⟩


theorem Bounded.mem_gt : ∀ {t o} {x : α}, Bounded t x o → All (· > x) t
  | nil, _, _, _ => ⟨⟩
  | node _ _ _ _, _, _, ⟨h₁, h₂⟩ => ⟨h₁.mem_gt, h₁.to_lt, h₂.mem_gt.imp fun _ => lt_trans h₁.to_lt⟩


theorem Bounded.of_lt :
    ∀ {t o₁ o₂} {x : α}, Bounded t o₁ o₂ → Bounded nil o₁ x → All (· < x) t → Bounded t o₁ x
  | nil, _, _, _, _, hn, _ => hn
  | node _ _ _ _, _, _, _, ⟨h₁, h₂⟩, _, ⟨_, al₂, al₃⟩ => ⟨h₁, h₂.of_lt al₂ al₃⟩


theorem Bounded.of_gt :
    ∀ {t o₁ o₂} {x : α}, Bounded t o₁ o₂ → Bounded nil x o₂ → All (· > x) t → Bounded t x o₂
  | nil, _, _, _, _, hn, _ => hn
  | node _ _ _ _, _, _, _, ⟨h₁, h₂⟩, _, ⟨al₁, al₂, _⟩ => ⟨h₁.of_gt al₂ al₁, h₂⟩


theorem Bounded.to_sep {t₁ t₂ o₁ o₂} {x : α}
    (h₁ : Bounded t₁ o₁ (x : WithTop α)) (h₂ : Bounded t₂ (x : WithBot α) o₂) :
    t₁.All fun y => t₂.All fun z : α => y < z := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    t₁ t₂ : Ordnode α
    o₁ : WithBot α
    o₂ : WithTop α
    x : α
    h₁ : t₁.Bounded o₁ ↑x
    h₂ : t₂.Bounded (↑x) o₂
    ⊢ Ordnode.All (fun y => Ordnode.All (fun z => LT.lt y z) t₂) t₁
  -/
  refine h₁.mem_lt.imp fun y yx => ?_
  /-
    α : Type u_1
    inst✝ : Preorder α
    t₁ t₂ : Ordnode α
    o₁ : WithBot α
    o₂ : WithTop α
    x : α
    h₁ : t₁.Bounded o₁ ↑x
    h₂ : t₂.Bounded (↑x) o₂
    y : α
    yx : LT.lt y x
    ⊢ Ordnode.All (fun z => LT.lt y z) t₂
  -/
  exact h₂.mem_gt.imp fun z xz => lt_trans yx xz
  /-
    🎉 no goals
  -/


/-- The validity predicate for an `Ordnode` subtree. This asserts that the `size` fields are
correct, the tree is balanced, and the elements of the tree are organized according to the
ordering. This version of `Valid` also puts all elements in the tree in the interval `(lo, hi)`. -/
structure Valid' (lo : WithBot α) (t : Ordnode α) (hi : WithTop α) : Prop where
  ord : t.Bounded lo hi
  sz : t.Sized
  bal : t.Balanced


/-- The validity predicate for an `Ordnode` subtree. This asserts that the `size` fields are
correct, the tree is balanced, and the elements of the tree are organized according to the
ordering. -/
def Valid (t : Ordnode α) : Prop :=
  Valid' ⊥ t ⊤


theorem Valid'.mono_left {x y : α} (xy : x ≤ y) {t : Ordnode α} {o} (h : Valid' y t o) :
    Valid' x t o :=
  ⟨h.1.mono_left xy, h.2, h.3⟩


theorem Valid'.mono_right {x y : α} (xy : x ≤ y) {t : Ordnode α} {o} (h : Valid' o t x) :
    Valid' o t y :=
  ⟨h.1.mono_right xy, h.2, h.3⟩


theorem Valid'.trans_left {t₁ t₂ : Ordnode α} {x : α} {o₁ o₂} (h : Bounded t₁ o₁ x)
    (H : Valid' x t₂ o₂) : Valid' o₁ t₂ o₂ :=
  ⟨h.trans_left H.1, H.2, H.3⟩


theorem Valid'.trans_right {t₁ t₂ : Ordnode α} {x : α} {o₁ o₂} (H : Valid' o₁ t₁ x)
    (h : Bounded t₂ x o₂) : Valid' o₁ t₁ o₂ :=
  ⟨H.1.trans_right h, H.2, H.3⟩


theorem Valid'.of_lt {t : Ordnode α} {x : α} {o₁ o₂} (H : Valid' o₁ t o₂) (h₁ : Bounded nil o₁ x)
    (h₂ : All (· < x) t) : Valid' o₁ t x :=
  ⟨H.1.of_lt h₁ h₂, H.2, H.3⟩


theorem Valid'.of_gt {t : Ordnode α} {x : α} {o₁ o₂} (H : Valid' o₁ t o₂) (h₁ : Bounded nil x o₂)
    (h₂ : All (· > x) t) : Valid' x t o₂ :=
  ⟨H.1.of_gt h₁ h₂, H.2, H.3⟩


theorem Valid'.valid {t o₁ o₂} (h : @Valid' α _ o₁ t o₂) : Valid t :=
  ⟨h.1.weak, h.2, h.3⟩


theorem valid'_nil {o₁ o₂} (h : Bounded nil o₁ o₂) : Valid' o₁ (@nil α) o₂ :=
  ⟨h, ⟨⟩, ⟨⟩⟩


theorem valid_nil : Valid (@nil α) :=
  valid'_nil ⟨⟩


theorem Valid'.node {s l} {x : α} {r o₁ o₂} (hl : Valid' o₁ l x) (hr : Valid' x r o₂)
    (H : BalancedSz (size l) (size r)) (hs : s = size l + size r + 1) :
    Valid' o₁ (@node α s l x r) o₂ :=
  ⟨⟨hl.1, hr.1⟩, ⟨hs, hl.2, hr.2⟩, ⟨H, hl.3, hr.3⟩⟩


theorem Valid'.dual : ∀ {t : Ordnode α} {o₁ o₂}, Valid' o₁ t o₂ → @Valid' αᵒᵈ _ o₂ (dual t) o₁
  | .nil, _, _, h => valid'_nil h.1.dual
  | .node _ l _ r, _, _, ⟨⟨ol, Or⟩, ⟨rfl, sl, sr⟩, ⟨b, bl, br⟩⟩ =>
    let ⟨ol', sl', bl'⟩ := Valid'.dual ⟨ol, sl, bl⟩
    let ⟨or', sr', br'⟩ := Valid'.dual ⟨Or, sr, br⟩
                     /-
                       α : Type u_1
                       inst✝ : Preorder α
                       l : Ordnode α
                       x✝ : α
                       r : Ordnode α
                       lo✝ : WithBot α
                       hi✝ : WithTop α
                       ol : l.Bounded lo✝ ↑x✝
                       Or : r.Bounded (↑x✝) hi✝
                       sl : l.Sized
                       sr : r.Sized
                       b : Ordnode.BalancedSz l.size r.size
                       bl : l.Balanced
                       br : r.Balanced
                       ol' : l.dual.Bounded (↑x✝) lo✝
                       sl' : l.dual.Sized
                       bl' : l.dual.Balanced
                       or' : r.dual.Bounded hi✝ ↑x✝
                       sr' : r.dual.Sized
                       br' : r.dual.Balanced
                       ⊢ Eq (HAdd.hAdd (HAdd.hAdd l.size r.size) 1) (HAdd.hAdd (HAdd.hAdd r.dual.size …
                     -/
    ⟨⟨or', ol'⟩, ⟨by simp [size_dual, add_comm], sr', sl'⟩,
                     /-
                       🎉 no goals
                     -/
          /-
            α : Type u_1
            inst✝ : Preorder α
            l : Ordnode α
            x✝ : α
            r : Ordnode α
            lo✝ : WithBot α
            hi✝ : WithTop α
            ol : l.Bounded lo✝ ↑x✝
            Or : r.Bounded (↑x✝) hi✝
            sl : l.Sized
            sr : r.Sized
            b : Ordnode.BalancedSz l.size r.size
            bl : l.Balanced
            br : r.Balanced
            ol' : l.dual.Bounded (↑x✝) lo✝
            sl' : l.dual.Sized
            bl' : l.dual.Balanced
            or' : r.dual.Bounded hi✝ ↑x✝
            sr' : r.dual.Sized
            br' : r.dual.Balanced
            ⊢ Ordnode.BalancedSz r.dual.size l.dual.size
          -/
      ⟨by rw [size_dual, size_dual]; exact b.symm, br', bl'⟩⟩
                                     /-
                                       🎉 no goals
                                     -/


theorem Valid'.dual_iff {t : Ordnode α} {o₁ o₂} : Valid' o₁ t o₂ ↔ @Valid' αᵒᵈ _ o₂ (.dual t) o₁ :=
  ⟨Valid'.dual, fun h => by
    /-
      α : Type u_1
      inst✝ : Preorder α
      t : Ordnode α
      o₁ : WithBot α
      o₂ : WithTop α
      h : Ordnode.Valid' o₂ t.dual o₁
      ⊢ Ordnode.Valid' o₁ t o₂
    -/
    have := Valid'.dual h; rwa [dual_dual, OrderDual.Preorder.dual_dual] at this⟩
                           /-
                             🎉 no goals
                           -/


theorem Valid.dual {t : Ordnode α} : Valid t → @Valid αᵒᵈ _ (.dual t) :=
  Valid'.dual


theorem Valid.dual_iff {t : Ordnode α} : Valid t ↔ @Valid αᵒᵈ _ (.dual t) :=
  Valid'.dual_iff


theorem Valid'.left {s l x r o₁ o₂} (H : Valid' o₁ (@Ordnode.node α s l x r) o₂) : Valid' o₁ l x :=
  ⟨H.1.1, H.2.2.1, H.3.2.1⟩


theorem Valid'.right {s l x r o₁ o₂} (H : Valid' o₁ (@Ordnode.node α s l x r) o₂) : Valid' x r o₂ :=
  ⟨H.1.2, H.2.2.2, H.3.2.2⟩


nonrec theorem Valid.left {s l x r} (H : Valid (@node α s l x r)) : Valid l :=
  H.left.valid


nonrec theorem Valid.right {s l x r} (H : Valid (@node α s l x r)) : Valid r :=
  H.right.valid


theorem Valid.size_eq {s l x r} (H : Valid (@node α s l x r)) :
    size (@node α s l x r) = size l + size r + 1 :=
  H.2.1


theorem Valid'.node' {l} {x : α} {r o₁ o₂} (hl : Valid' o₁ l x) (hr : Valid' x r o₂)
    (H : BalancedSz (size l) (size r)) : Valid' o₁ (@node' α l x r) o₂ :=
  hl.node hr H rfl


theorem valid'_singleton {x : α} {o₁ o₂} (h₁ : Bounded nil o₁ x) (h₂ : Bounded nil x o₂) :
    Valid' o₁ (singleton x : Ordnode α) o₂ :=
  (valid'_nil h₁).node (valid'_nil h₂) (Or.inl zero_le_one) rfl


theorem valid_singleton {x : α} : Valid (singleton x : Ordnode α) :=
  valid'_singleton ⟨⟩ ⟨⟩


theorem Valid'.node3L {l} {x : α} {m} {y : α} {r o₁ o₂} (hl : Valid' o₁ l x) (hm : Valid' x m y)
    (hr : Valid' y r o₂) (H1 : BalancedSz (size l) (size m))
    (H2 : BalancedSz (size l + size m + 1) (size r)) : Valid' o₁ (@node3L α l x m y r) o₂ :=
  (hl.node' hm H1).node' hr H2


theorem Valid'.node3R {l} {x : α} {m} {y : α} {r o₁ o₂} (hl : Valid' o₁ l x) (hm : Valid' x m y)
    (hr : Valid' y r o₂) (H1 : BalancedSz (size l) (size m + size r + 1))
    (H2 : BalancedSz (size m) (size r)) : Valid' o₁ (@node3R α l x m y r) o₂ :=
  hl.node' (hm.node' hr H2) H1


theorem Valid'.node4L_lemma₁ {a b c d : ℕ} (lr₂ : 3 * (b + c + 1 + d) ≤ 16 * a + 9)
                                                                      /-
                                                                        a b c d : Nat
                                                                        lr₂ : LE.le (HMul.hMul 3 (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd b c) 1) d)) (HAdd.hA …
                                                                        mr₂ : LE.le (HAdd.hAdd (HAdd.hAdd b c) 1) (HMul.hMul 3 d)
                                                                        mm₁ : LE.le b (HMul.hMul 3 c)
                                                                        ⊢ LT.lt b (HAdd.hAdd (HMul.hMul 3 a) 1)
                                                                      -/
    (mr₂ : b + c + 1 ≤ 3 * d) (mm₁ : b ≤ 3 * c) : b < 3 * a + 1 := by omega
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


                                                                                     /-
                                                                                       b c d : Nat
                                                                                       mr₂ : LE.le (HAdd.hAdd (HAdd.hAdd b c) 1) (HMul.hMul 3 d)
                                                                                       ⊢ LE.le c (HMul.hMul 3 d)
                                                                                     -/
theorem Valid'.node4L_lemma₂ {b c d : ℕ} (mr₂ : b + c + 1 ≤ 3 * d) : c ≤ 3 * d := by omega
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


theorem Valid'.node4L_lemma₃ {b c d : ℕ} (mr₁ : 2 * d ≤ b + c + 1) (mm₁ : b ≤ 3 * c) :
                    /-
                      b c d : Nat
                      mr₁ : LE.le (HMul.hMul 2 d) (HAdd.hAdd (HAdd.hAdd b c) 1)
                      mm₁ : LE.le b (HMul.hMul 3 c)
                      ⊢ LE.le d (HMul.hMul 3 c)
                    -/
    d ≤ 3 * c := by omega
                    /-
                      🎉 no goals
                    -/


theorem Valid'.node4L_lemma₄ {a b c d : ℕ} (lr₁ : 3 * a ≤ b + c + 1 + d) (mr₂ : b + c + 1 ≤ 3 * d)
                                                          /-
                                                            a b c d : Nat
                                                            lr₁ : LE.le (HMul.hMul 3 a) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd b c) 1) d)
                                                            mr₂ : LE.le (HAdd.hAdd (HAdd.hAdd b c) 1) (HMul.hMul 3 d)
                                                            mm₁ : LE.le b (HMul.hMul 3 c)
                                                            ⊢ LE.le (HAdd.hAdd (HAdd.hAdd a b) 1) (HMul.hMul 3 (HAdd.hAdd (HAdd.hAdd c d)  …
                                                          -/
    (mm₁ : b ≤ 3 * c) : a + b + 1 ≤ 3 * (c + d + 1) := by omega
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem Valid'.node4L_lemma₅ {a b c d : ℕ} (lr₂ : 3 * (b + c + 1 + d) ≤ 16 * a + 9)
                                                                                    /-
                                                                                      a b c d : Nat
                                                                                      lr₂ : LE.le (HMul.hMul 3 (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd b c) 1) d)) (HAdd.hA …
                                                                                      mr₁ : LE.le (HMul.hMul 2 d) (HAdd.hAdd (HAdd.hAdd b c) 1)
                                                                                      mm₂ : LE.le c (HMul.hMul 3 b)
                                                                                      ⊢ LE.le (HAdd.hAdd (HAdd.hAdd c d) 1) (HMul.hMul 3 (HAdd.hAdd (HAdd.hAdd a b)  …
                                                                                    -/
    (mr₁ : 2 * d ≤ b + c + 1) (mm₂ : c ≤ 3 * b) : c + d + 1 ≤ 3 * (a + b + 1) := by omega
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


theorem Valid'.node4L {l} {x : α} {m} {y : α} {r o₁ o₂} (hl : Valid' o₁ l x) (hm : Valid' x m y)
    (hr : Valid' (↑y) r o₂) (Hm : 0 < size m)
    (H : size l = 0 ∧ size m = 1 ∧ size r ≤ 1 ∨
        0 < size l ∧
          ratio * size r ≤ size m ∧
            delta * size l ≤ size m + size r ∧
              3 * (size m + size r) ≤ 16 * size l + 9 ∧ size m ≤ delta * size r) :
    Valid' o₁ (@node4L α l x m y r) o₂ := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    l : Ordnode α
    x : α
    m : Ordnode α
    y : α
    r : Ordnode α
    o₁ : WithBot α
    o₂ : WithTop α
    hl : Ordnode.Valid' o₁ l ↑x
    hm : Ordnode.Valid' (↑x) m ↑y
    hr : Ordnode.Valid' (↑y) r o₂
    Hm : LT.lt 0 m.size
    H : Or (And (Eq l.size 0) (And (Eq m.size 1) (LE.le r.size 1))) (And (LT.lt 0  …
    ⊢ Ordnode.Valid' o₁ (l.node4L x m y r) o₂
  -/
  cases' m with s ml z mr; · cases Hm
                             /-
                               🎉 no goals
                             -/
  suffices
    BalancedSz (size l) (size ml) ∧
      BalancedSz (size mr) (size r) ∧ BalancedSz (size l + size ml + 1) (size mr + size r + 1) from
    Valid'.node' (hl.node' hm.left this.1) (hm.right.node' hr this.2.1) this.2.2
  /-
    case node
    α : Type u_1
    inst✝ : Preorder α
    l : Ordnode α
    x y : α
    r : Ordnode α
    o₁ : WithBot α
    o₂ : WithTop α
    hl : Ordnode.Valid' o₁ l ↑x
    hr : Ordnode.Valid' (↑y) r o₂
    s : Nat
    ml : Ordnode α
    z : α
    mr : Ordnode α
    hm : Ordnode.Valid' (↑x) (Ordnode.node s ml z mr) ↑y
    Hm : LT.lt 0 (Ordnode.node s ml z mr).size
    H : Or (And (Eq l.size 0) (And (Eq (Ordnode.node s ml z mr).size 1) (LE.le r.s …
    ⊢ And (Ordnode.BalancedSz l.size ml.size) (And (Ordnode.BalancedSz mr.size r.s …
  -/
  rcases H with (⟨l0, m1, r0⟩ | ⟨l0, mr₁, lr₁, lr₂, mr₂⟩)
    /-
      case node.inl.intro.intro
      α : Type u_1
      inst✝ : Preorder α
      l : Ordnode α
      x y : α
      r : Ordnode α
      o₁ : WithBot α
      o₂ : WithTop α
      hl : Ordnode.Valid' o₁ l ↑x
      hr : Ordnode.Valid' (↑y) r o₂
      s : Nat
      ml : Ordnode α
      z : α
      mr : Ordnode α
      hm : Ordnode.Valid' (↑x) (Ordnode.node s ml z mr) ↑y
      Hm : LT.lt 0 (Ordnode.node s ml z mr).size
      l0 : Eq l.size 0
      m1 : Eq (Ordnode.node s ml z mr).size 1
      r0 : LE.le r.size 1
      ⊢ And (Ordnode.BalancedSz l.size ml.size) (And (Ordnode.BalancedSz mr.size r.s …
    -/
  · rw [hm.2.size_eq, Nat.succ_inj', add_eq_zero] at m1
    /-
      case node.inl.intro.intro
      α : Type u_1
      inst✝ : Preorder α
      l : Ordnode α
      x y : α
      r : Ordnode α
      o₁ : WithBot α
      o₂ : WithTop α
      hl : Ordnode.Valid' o₁ l ↑x
      hr : Ordnode.Valid' (↑y) r o₂
      s : Nat
      ml : Ordnode α
      z : α
      mr : Ordnode α
      hm : Ordnode.Valid' (↑x) (Ordnode.node s ml z mr) ↑y
      Hm : LT.lt 0 (Ordnode.node s ml z mr).size
      l0 : Eq l.size 0
      m1 : And (Eq ml.size 0) (Eq mr.size 0)
      r0 : LE.le r.size 1
      ⊢ And (Ordnode.BalancedSz l.size ml.size) (And (Ordnode.BalancedSz mr.size r.s …
    -/
    rw [l0, m1.1, m1.2]; revert r0; rcases size r with (_ | _ | _) <;>
      [decide; decide; (intro r0; unfold BalancedSz delta; omega)]
    /-
      case node.inr.intro.intro.intro.intro
      α : Type u_1
      inst✝ : Preorder α
      l : Ordnode α
      x y : α
      r : Ordnode α
      o₁ : WithBot α
      o₂ : WithTop α
      hl : Ordnode.Valid' o₁ l ↑x
      hr : Ordnode.Valid' (↑y) r o₂
      s : Nat
      ml : Ordnode α
      z : α
      mr : Ordnode α
      hm : Ordnode.Valid' (↑x) (Ordnode.node s ml z mr) ↑y
      Hm : LT.lt 0 (Ordnode.node s ml z mr).size
      l0 : LT.lt 0 l.size
      mr₁ : LE.le (HMul.hMul Ordnode.ratio r.size) (Ordnode.node s ml z mr).size
      lr₁ : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd (Ordnode.node s ml z m …
      lr₂ : LE.le (HMul.hMul 3 (HAdd.hAdd (Ordnode.node s ml z mr).size r.size)) (HA …
      mr₂ : LE.le (Ordnode.node s ml z mr).size (HMul.hMul Ordnode.delta r.size)
      ⊢ And (Ordnode.BalancedSz l.size ml.size) (And (Ordnode.BalancedSz mr.size r.s …
    -/
  · rcases Nat.eq_zero_or_pos (size r) with r0 | r0
      /-
        case node.inr.intro.intro.intro.intro.inl
        α : Type u_1
        inst✝ : Preorder α
        l : Ordnode α
        x y : α
        r : Ordnode α
        o₁ : WithBot α
        o₂ : WithTop α
        hl : Ordnode.Valid' o₁ l ↑x
        hr : Ordnode.Valid' (↑y) r o₂
        s : Nat
        ml : Ordnode α
        z : α
        mr : Ordnode α
        hm : Ordnode.Valid' (↑x) (Ordnode.node s ml z mr) ↑y
        Hm : LT.lt 0 (Ordnode.node s ml z mr).size
        l0 : LT.lt 0 l.size
        mr₁ : LE.le (HMul.hMul Ordnode.ratio r.size) (Ordnode.node s ml z mr).size
        lr₁ : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd (Ordnode.node s ml z m …
        lr₂ : LE.le (HMul.hMul 3 (HAdd.hAdd (Ordnode.node s ml z mr).size r.size)) (HA …
        mr₂ : LE.le (Ordnode.node s ml z mr).size (HMul.hMul Ordnode.delta r.size)
        r0 : Eq r.size 0
        ⊢ And (Ordnode.BalancedSz l.size ml.size) (And (Ordnode.BalancedSz mr.size r.s …
      -/
    · rw [r0] at mr₂; cases not_le_of_lt Hm mr₂
                      /-
                        🎉 no goals
                      -/
    /-
      case node.inr.intro.intro.intro.intro.inr
      α : Type u_1
      inst✝ : Preorder α
      l : Ordnode α
      x y : α
      r : Ordnode α
      o₁ : WithBot α
      o₂ : WithTop α
      hl : Ordnode.Valid' o₁ l ↑x
      hr : Ordnode.Valid' (↑y) r o₂
      s : Nat
      ml : Ordnode α
      z : α
      mr : Ordnode α
      hm : Ordnode.Valid' (↑x) (Ordnode.node s ml z mr) ↑y
      Hm : LT.lt 0 (Ordnode.node s ml z mr).size
      l0 : LT.lt 0 l.size
      mr₁ : LE.le (HMul.hMul Ordnode.ratio r.size) (Ordnode.node s ml z mr).size
      lr₁ : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd (Ordnode.node s ml z m …
      lr₂ : LE.le (HMul.hMul 3 (HAdd.hAdd (Ordnode.node s ml z mr).size r.size)) (HA …
      mr₂ : LE.le (Ordnode.node s ml z mr).size (HMul.hMul Ordnode.delta r.size)
      r0 : GT.gt r.size 0
      ⊢ And (Ordnode.BalancedSz l.size ml.size) (And (Ordnode.BalancedSz mr.size r.s …
    -/
    rw [hm.2.size_eq] at lr₁ lr₂ mr₁ mr₂
    /-
      case node.inr.intro.intro.intro.intro.inr
      α : Type u_1
      inst✝ : Preorder α
      l : Ordnode α
      x y : α
      r : Ordnode α
      o₁ : WithBot α
      o₂ : WithTop α
      hl : Ordnode.Valid' o₁ l ↑x
      hr : Ordnode.Valid' (↑y) r o₂
      s : Nat
      ml : Ordnode α
      z : α
      mr : Ordnode α
      hm : Ordnode.Valid' (↑x) (Ordnode.node s ml z mr) ↑y
      Hm : LT.lt 0 (Ordnode.node s ml z mr).size
      l0 : LT.lt 0 l.size
      mr₁ : LE.le (HMul.hMul Ordnode.ratio r.size) (HAdd.hAdd (HAdd.hAdd ml.size mr. …
      lr₁ : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd  …
      lr₂ : LE.le (HMul.hMul 3 (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1)  …
      mr₂ : LE.le (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1) (HMul.hMul Ordnode.delta …
      r0 : GT.gt r.size 0
      ⊢ And (Ordnode.BalancedSz l.size ml.size) (And (Ordnode.BalancedSz mr.size r.s …
    -/
    by_cases mm : size ml + size mr ≤ 1
    · have r1 :=
        le_antisymm
          ((mul_le_mul_left (by decide)).1 (le_trans mr₁ (Nat.succ_le_succ mm) : _ ≤ ratio * 1)) r0
      /-
        case pos
        α : Type u_1
        inst✝ : Preorder α
        l : Ordnode α
        x y : α
        r : Ordnode α
        o₁ : WithBot α
        o₂ : WithTop α
        hl : Ordnode.Valid' o₁ l ↑x
        hr : Ordnode.Valid' (↑y) r o₂
        s : Nat
        ml : Ordnode α
        z : α
        mr : Ordnode α
        hm : Ordnode.Valid' (↑x) (Ordnode.node s ml z mr) ↑y
        Hm : LT.lt 0 (Ordnode.node s ml z mr).size
        l0 : LT.lt 0 l.size
        mr₁ : LE.le (HMul.hMul Ordnode.ratio r.size) (HAdd.hAdd (HAdd.hAdd ml.size mr. …
        lr₁ : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd  …
        lr₂ : LE.le (HMul.hMul 3 (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1)  …
        mr₂ : LE.le (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1) (HMul.hMul Ordnode.delta …
        r0 : GT.gt r.size 0
        mm : LE.le (HAdd.hAdd ml.size mr.size) 1
        r1 : Eq r.size 1
        ⊢ And (Ordnode.BalancedSz l.size ml.size) (And (Ordnode.BalancedSz mr.size r.s …
      -/
      rw [r1, add_assoc] at lr₁
      have l1 :=
        le_antisymm
          ((mul_le_mul_left (by decide)).1 (le_trans lr₁ (add_le_add_right mm 2) : _ ≤ delta * 1))
          l0
      /-
        case pos
        α : Type u_1
        inst✝ : Preorder α
        l : Ordnode α
        x y : α
        r : Ordnode α
        o₁ : WithBot α
        o₂ : WithTop α
        hl : Ordnode.Valid' o₁ l ↑x
        hr : Ordnode.Valid' (↑y) r o₂
        s : Nat
        ml : Ordnode α
        z : α
        mr : Ordnode α
        hm : Ordnode.Valid' (↑x) (Ordnode.node s ml z mr) ↑y
        Hm : LT.lt 0 (Ordnode.node s ml z mr).size
        l0 : LT.lt 0 l.size
        mr₁ : LE.le (HMul.hMul Ordnode.ratio r.size) (HAdd.hAdd (HAdd.hAdd ml.size mr. …
        lr₁ : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd (HAdd.hAdd ml.size mr. …
        lr₂ : LE.le (HMul.hMul 3 (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1)  …
        mr₂ : LE.le (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1) (HMul.hMul Ordnode.delta …
        r0 : GT.gt r.size 0
        mm : LE.le (HAdd.hAdd ml.size mr.size) 1
        r1 : Eq r.size 1
        l1 : Eq l.size 1
        ⊢ And (Ordnode.BalancedSz l.size ml.size) (And (Ordnode.BalancedSz mr.size r.s …
      -/
      rw [l1, r1]
      /-
        case pos
        α : Type u_1
        inst✝ : Preorder α
        l : Ordnode α
        x y : α
        r : Ordnode α
        o₁ : WithBot α
        o₂ : WithTop α
        hl : Ordnode.Valid' o₁ l ↑x
        hr : Ordnode.Valid' (↑y) r o₂
        s : Nat
        ml : Ordnode α
        z : α
        mr : Ordnode α
        hm : Ordnode.Valid' (↑x) (Ordnode.node s ml z mr) ↑y
        Hm : LT.lt 0 (Ordnode.node s ml z mr).size
        l0 : LT.lt 0 l.size
        mr₁ : LE.le (HMul.hMul Ordnode.ratio r.size) (HAdd.hAdd (HAdd.hAdd ml.size mr. …
        lr₁ : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd (HAdd.hAdd ml.size mr. …
        lr₂ : LE.le (HMul.hMul 3 (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1)  …
        mr₂ : LE.le (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1) (HMul.hMul Ordnode.delta …
        r0 : GT.gt r.size 0
        mm : LE.le (HAdd.hAdd ml.size mr.size) 1
        r1 : Eq r.size 1
        l1 : Eq l.size 1
        ⊢ And (Ordnode.BalancedSz 1 ml.size) (And (Ordnode.BalancedSz mr.size 1) (Ordn …
      -/
      revert mm; cases size ml <;> cases size mr <;> intro mm
        /-
          case pos.zero.zero
          α : Type u_1
          inst✝ : Preorder α
          l : Ordnode α
          x y : α
          r : Ordnode α
          o₁ : WithBot α
          o₂ : WithTop α
          hl : Ordnode.Valid' o₁ l ↑x
          hr : Ordnode.Valid' (↑y) r o₂
          s : Nat
          ml : Ordnode α
          z : α
          mr : Ordnode α
          hm : Ordnode.Valid' (↑x) (Ordnode.node s ml z mr) ↑y
          Hm : LT.lt 0 (Ordnode.node s ml z mr).size
          l0 : LT.lt 0 l.size
          mr₁ : LE.le (HMul.hMul Ordnode.ratio r.size) (HAdd.hAdd (HAdd.hAdd ml.size mr. …
          lr₁ : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd (HAdd.hAdd ml.size mr. …
          lr₂ : LE.le (HMul.hMul 3 (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1)  …
          mr₂ : LE.le (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1) (HMul.hMul Ordnode.delta …
          r0 : GT.gt r.size 0
          r1 : Eq r.size 1
          l1 : Eq l.size 1
          mm : LE.le (HAdd.hAdd 0 0) 1
          ⊢ And (Ordnode.BalancedSz 1 0) (And (Ordnode.BalancedSz 0 1) (Ordnode.Balanced …
        -/
      · decide
        /-
          🎉 no goals
        -/
        /-
          case pos.zero.succ
          α : Type u_1
          inst✝ : Preorder α
          l : Ordnode α
          x y : α
          r : Ordnode α
          o₁ : WithBot α
          o₂ : WithTop α
          hl : Ordnode.Valid' o₁ l ↑x
          hr : Ordnode.Valid' (↑y) r o₂
          s : Nat
          ml : Ordnode α
          z : α
          mr : Ordnode α
          hm : Ordnode.Valid' (↑x) (Ordnode.node s ml z mr) ↑y
          Hm : LT.lt 0 (Ordnode.node s ml z mr).size
          l0 : LT.lt 0 l.size
          mr₁ : LE.le (HMul.hMul Ordnode.ratio r.size) (HAdd.hAdd (HAdd.hAdd ml.size mr. …
          lr₁ : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd (HAdd.hAdd ml.size mr. …
          lr₂ : LE.le (HMul.hMul 3 (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1)  …
          mr₂ : LE.le (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1) (HMul.hMul Ordnode.delta …
          r0 : GT.gt r.size 0
          r1 : Eq r.size 1
          l1 : Eq l.size 1
          n✝ : Nat
          mm : LE.le (HAdd.hAdd 0 (HAdd.hAdd n✝ 1)) 1
          ⊢ And (Ordnode.BalancedSz 1 0) (And (Ordnode.BalancedSz (HAdd.hAdd n✝ 1) 1) (O …
        -/
      · rw [zero_add] at mm; rcases mm with (_ | ⟨⟨⟩⟩)
        /-
          case pos.zero.succ.refl
          α : Type u_1
          inst✝ : Preorder α
          l : Ordnode α
          x y : α
          r : Ordnode α
          o₁ : WithBot α
          o₂ : WithTop α
          hl : Ordnode.Valid' o₁ l ↑x
          hr : Ordnode.Valid' (↑y) r o₂
          s : Nat
          ml : Ordnode α
          z : α
          mr : Ordnode α
          hm : Ordnode.Valid' (↑x) (Ordnode.node s ml z mr) ↑y
          Hm : LT.lt 0 (Ordnode.node s ml z mr).size
          l0 : LT.lt 0 l.size
          mr₁ : LE.le (HMul.hMul Ordnode.ratio r.size) (HAdd.hAdd (HAdd.hAdd ml.size mr. …
          lr₁ : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd (HAdd.hAdd ml.size mr. …
          lr₂ : LE.le (HMul.hMul 3 (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1)  …
          mr₂ : LE.le (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1) (HMul.hMul Ordnode.delta …
          r0 : GT.gt r.size 0
          r1 : Eq r.size 1
          l1 : Eq l.size 1
          ⊢ And (Ordnode.BalancedSz 1 0) (And (Ordnode.BalancedSz (HAdd.hAdd 0 1) 1) (Or …
        -/
        decide
        /-
          🎉 no goals
        -/
        /-
          case pos.succ.zero
          α : Type u_1
          inst✝ : Preorder α
          l : Ordnode α
          x y : α
          r : Ordnode α
          o₁ : WithBot α
          o₂ : WithTop α
          hl : Ordnode.Valid' o₁ l ↑x
          hr : Ordnode.Valid' (↑y) r o₂
          s : Nat
          ml : Ordnode α
          z : α
          mr : Ordnode α
          hm : Ordnode.Valid' (↑x) (Ordnode.node s ml z mr) ↑y
          Hm : LT.lt 0 (Ordnode.node s ml z mr).size
          l0 : LT.lt 0 l.size
          mr₁ : LE.le (HMul.hMul Ordnode.ratio r.size) (HAdd.hAdd (HAdd.hAdd ml.size mr. …
          lr₁ : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd (HAdd.hAdd ml.size mr. …
          lr₂ : LE.le (HMul.hMul 3 (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1)  …
          mr₂ : LE.le (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1) (HMul.hMul Ordnode.delta …
          r0 : GT.gt r.size 0
          r1 : Eq r.size 1
          l1 : Eq l.size 1
          n✝ : Nat
          mm : LE.le (HAdd.hAdd (HAdd.hAdd n✝ 1) 0) 1
          ⊢ And (Ordnode.BalancedSz 1 (HAdd.hAdd n✝ 1)) (And (Ordnode.BalancedSz 0 1) (O …
        -/
      · rcases mm with (_ | ⟨⟨⟩⟩); decide
                                   /-
                                     🎉 no goals
                                   -/
        /-
          case pos.succ.succ
          α : Type u_1
          inst✝ : Preorder α
          l : Ordnode α
          x y : α
          r : Ordnode α
          o₁ : WithBot α
          o₂ : WithTop α
          hl : Ordnode.Valid' o₁ l ↑x
          hr : Ordnode.Valid' (↑y) r o₂
          s : Nat
          ml : Ordnode α
          z : α
          mr : Ordnode α
          hm : Ordnode.Valid' (↑x) (Ordnode.node s ml z mr) ↑y
          Hm : LT.lt 0 (Ordnode.node s ml z mr).size
          l0 : LT.lt 0 l.size
          mr₁ : LE.le (HMul.hMul Ordnode.ratio r.size) (HAdd.hAdd (HAdd.hAdd ml.size mr. …
          lr₁ : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd (HAdd.hAdd ml.size mr. …
          lr₂ : LE.le (HMul.hMul 3 (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1)  …
          mr₂ : LE.le (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1) (HMul.hMul Ordnode.delta …
          r0 : GT.gt r.size 0
          r1 : Eq r.size 1
          l1 : Eq l.size 1
          n✝¹ n✝ : Nat
          mm : LE.le (HAdd.hAdd (HAdd.hAdd n✝¹ 1) (HAdd.hAdd n✝ 1)) 1
          ⊢ And (Ordnode.BalancedSz 1 (HAdd.hAdd n✝¹ 1)) (And (Ordnode.BalancedSz (HAdd. …
        -/
      · rw [Nat.succ_add] at mm; rcases mm with (_ | ⟨⟨⟩⟩)
                                 /-
                                   🎉 no goals
                                 -/
    /-
      case neg
      α : Type u_1
      inst✝ : Preorder α
      l : Ordnode α
      x y : α
      r : Ordnode α
      o₁ : WithBot α
      o₂ : WithTop α
      hl : Ordnode.Valid' o₁ l ↑x
      hr : Ordnode.Valid' (↑y) r o₂
      s : Nat
      ml : Ordnode α
      z : α
      mr : Ordnode α
      hm : Ordnode.Valid' (↑x) (Ordnode.node s ml z mr) ↑y
      Hm : LT.lt 0 (Ordnode.node s ml z mr).size
      l0 : LT.lt 0 l.size
      mr₁ : LE.le (HMul.hMul Ordnode.ratio r.size) (HAdd.hAdd (HAdd.hAdd ml.size mr. …
      lr₁ : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd  …
      lr₂ : LE.le (HMul.hMul 3 (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1)  …
      mr₂ : LE.le (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1) (HMul.hMul Ordnode.delta …
      r0 : GT.gt r.size 0
      mm : Not (LE.le (HAdd.hAdd ml.size mr.size) 1)
      ⊢ And (Ordnode.BalancedSz l.size ml.size) (And (Ordnode.BalancedSz mr.size r.s …
    -/
    rcases hm.3.1.resolve_left mm with ⟨mm₁, mm₂⟩
    /-
      case neg.intro
      α : Type u_1
      inst✝ : Preorder α
      l : Ordnode α
      x y : α
      r : Ordnode α
      o₁ : WithBot α
      o₂ : WithTop α
      hl : Ordnode.Valid' o₁ l ↑x
      hr : Ordnode.Valid' (↑y) r o₂
      s : Nat
      ml : Ordnode α
      z : α
      mr : Ordnode α
      hm : Ordnode.Valid' (↑x) (Ordnode.node s ml z mr) ↑y
      Hm : LT.lt 0 (Ordnode.node s ml z mr).size
      l0 : LT.lt 0 l.size
      mr₁ : LE.le (HMul.hMul Ordnode.ratio r.size) (HAdd.hAdd (HAdd.hAdd ml.size mr. …
      lr₁ : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd  …
      lr₂ : LE.le (HMul.hMul 3 (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1)  …
      mr₂ : LE.le (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1) (HMul.hMul Ordnode.delta …
      r0 : GT.gt r.size 0
      mm : Not (LE.le (HAdd.hAdd ml.size mr.size) 1)
      mm₁ : LE.le ml.size (HMul.hMul Ordnode.delta mr.size)
      mm₂ : LE.le mr.size (HMul.hMul Ordnode.delta ml.size)
      ⊢ And (Ordnode.BalancedSz l.size ml.size) (And (Ordnode.BalancedSz mr.size r.s …
    -/
    rcases Nat.eq_zero_or_pos (size ml) with ml0 | ml0
      /-
        case neg.intro.inl
        α : Type u_1
        inst✝ : Preorder α
        l : Ordnode α
        x y : α
        r : Ordnode α
        o₁ : WithBot α
        o₂ : WithTop α
        hl : Ordnode.Valid' o₁ l ↑x
        hr : Ordnode.Valid' (↑y) r o₂
        s : Nat
        ml : Ordnode α
        z : α
        mr : Ordnode α
        hm : Ordnode.Valid' (↑x) (Ordnode.node s ml z mr) ↑y
        Hm : LT.lt 0 (Ordnode.node s ml z mr).size
        l0 : LT.lt 0 l.size
        mr₁ : LE.le (HMul.hMul Ordnode.ratio r.size) (HAdd.hAdd (HAdd.hAdd ml.size mr. …
        lr₁ : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd  …
        lr₂ : LE.le (HMul.hMul 3 (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1)  …
        mr₂ : LE.le (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1) (HMul.hMul Ordnode.delta …
        r0 : GT.gt r.size 0
        mm : Not (LE.le (HAdd.hAdd ml.size mr.size) 1)
        mm₁ : LE.le ml.size (HMul.hMul Ordnode.delta mr.size)
        mm₂ : LE.le mr.size (HMul.hMul Ordnode.delta ml.size)
        ml0 : Eq ml.size 0
        ⊢ And (Ordnode.BalancedSz l.size ml.size) (And (Ordnode.BalancedSz mr.size r.s …
      -/
    · rw [ml0, mul_zero, Nat.le_zero] at mm₂
      /-
        case neg.intro.inl
        α : Type u_1
        inst✝ : Preorder α
        l : Ordnode α
        x y : α
        r : Ordnode α
        o₁ : WithBot α
        o₂ : WithTop α
        hl : Ordnode.Valid' o₁ l ↑x
        hr : Ordnode.Valid' (↑y) r o₂
        s : Nat
        ml : Ordnode α
        z : α
        mr : Ordnode α
        hm : Ordnode.Valid' (↑x) (Ordnode.node s ml z mr) ↑y
        Hm : LT.lt 0 (Ordnode.node s ml z mr).size
        l0 : LT.lt 0 l.size
        mr₁ : LE.le (HMul.hMul Ordnode.ratio r.size) (HAdd.hAdd (HAdd.hAdd ml.size mr. …
        lr₁ : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd  …
        lr₂ : LE.le (HMul.hMul 3 (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1)  …
        mr₂ : LE.le (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1) (HMul.hMul Ordnode.delta …
        r0 : GT.gt r.size 0
        mm : Not (LE.le (HAdd.hAdd ml.size mr.size) 1)
        mm₁ : LE.le ml.size (HMul.hMul Ordnode.delta mr.size)
        mm₂ : Eq mr.size 0
        ml0 : Eq ml.size 0
        ⊢ And (Ordnode.BalancedSz l.size ml.size) (And (Ordnode.BalancedSz mr.size r.s …
      -/
      rw [ml0, mm₂] at mm; cases mm (by decide)
                           /-
                             🎉 no goals
                           -/
    have : 2 * size l ≤ size ml + size mr + 1 := by
      have := Nat.mul_le_mul_left ratio lr₁
      rw [mul_left_comm, mul_add] at this
      have := le_trans this (add_le_add_left mr₁ _)
      rw [← Nat.succ_mul] at this
      exact (mul_le_mul_left (by decide)).1 this
    /-
      case neg.intro.inr
      α : Type u_1
      inst✝ : Preorder α
      l : Ordnode α
      x y : α
      r : Ordnode α
      o₁ : WithBot α
      o₂ : WithTop α
      hl : Ordnode.Valid' o₁ l ↑x
      hr : Ordnode.Valid' (↑y) r o₂
      s : Nat
      ml : Ordnode α
      z : α
      mr : Ordnode α
      hm : Ordnode.Valid' (↑x) (Ordnode.node s ml z mr) ↑y
      Hm : LT.lt 0 (Ordnode.node s ml z mr).size
      l0 : LT.lt 0 l.size
      mr₁ : LE.le (HMul.hMul Ordnode.ratio r.size) (HAdd.hAdd (HAdd.hAdd ml.size mr. …
      lr₁ : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd  …
      lr₂ : LE.le (HMul.hMul 3 (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1)  …
      mr₂ : LE.le (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1) (HMul.hMul Ordnode.delta …
      r0 : GT.gt r.size 0
      mm : Not (LE.le (HAdd.hAdd ml.size mr.size) 1)
      mm₁ : LE.le ml.size (HMul.hMul Ordnode.delta mr.size)
      mm₂ : LE.le mr.size (HMul.hMul Ordnode.delta ml.size)
      ml0 : GT.gt ml.size 0
      this : LE.le (HMul.hMul 2 l.size) (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1)
      ⊢ And (Ordnode.BalancedSz l.size ml.size) (And (Ordnode.BalancedSz mr.size r.s …
    -/
    refine ⟨Or.inr ⟨?_, ?_⟩, Or.inr ⟨?_, ?_⟩, Or.inr ⟨?_, ?_⟩⟩
      /-
        case neg.intro.inr.refine_1
        α : Type u_1
        inst✝ : Preorder α
        l : Ordnode α
        x y : α
        r : Ordnode α
        o₁ : WithBot α
        o₂ : WithTop α
        hl : Ordnode.Valid' o₁ l ↑x
        hr : Ordnode.Valid' (↑y) r o₂
        s : Nat
        ml : Ordnode α
        z : α
        mr : Ordnode α
        hm : Ordnode.Valid' (↑x) (Ordnode.node s ml z mr) ↑y
        Hm : LT.lt 0 (Ordnode.node s ml z mr).size
        l0 : LT.lt 0 l.size
        mr₁ : LE.le (HMul.hMul Ordnode.ratio r.size) (HAdd.hAdd (HAdd.hAdd ml.size mr. …
        lr₁ : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd  …
        lr₂ : LE.le (HMul.hMul 3 (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1)  …
        mr₂ : LE.le (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1) (HMul.hMul Ordnode.delta …
        r0 : GT.gt r.size 0
        mm : Not (LE.le (HAdd.hAdd ml.size mr.size) 1)
        mm₁ : LE.le ml.size (HMul.hMul Ordnode.delta mr.size)
        mm₂ : LE.le mr.size (HMul.hMul Ordnode.delta ml.size)
        ml0 : GT.gt ml.size 0
        this : LE.le (HMul.hMul 2 l.size) (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1)
        ⊢ LE.le l.size (HMul.hMul Ordnode.delta ml.size)
      -/
    · refine (mul_le_mul_left (by decide)).1 (le_trans this ?_)
      /-
        case neg.intro.inr.refine_1
        α : Type u_1
        inst✝ : Preorder α
        l : Ordnode α
        x y : α
        r : Ordnode α
        o₁ : WithBot α
        o₂ : WithTop α
        hl : Ordnode.Valid' o₁ l ↑x
        hr : Ordnode.Valid' (↑y) r o₂
        s : Nat
        ml : Ordnode α
        z : α
        mr : Ordnode α
        hm : Ordnode.Valid' (↑x) (Ordnode.node s ml z mr) ↑y
        Hm : LT.lt 0 (Ordnode.node s ml z mr).size
        l0 : LT.lt 0 l.size
        mr₁ : LE.le (HMul.hMul Ordnode.ratio r.size) (HAdd.hAdd (HAdd.hAdd ml.size mr. …
        lr₁ : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd  …
        lr₂ : LE.le (HMul.hMul 3 (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1)  …
        mr₂ : LE.le (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1) (HMul.hMul Ordnode.delta …
        r0 : GT.gt r.size 0
        mm : Not (LE.le (HAdd.hAdd ml.size mr.size) 1)
        mm₁ : LE.le ml.size (HMul.hMul Ordnode.delta mr.size)
        mm₂ : LE.le mr.size (HMul.hMul Ordnode.delta ml.size)
        ml0 : GT.gt ml.size 0
        this : LE.le (HMul.hMul 2 l.size) (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1)
        ⊢ LE.le (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1) (HMul.hMul 2 (HMul.hMul Ordn …
      -/
      rw [two_mul, Nat.succ_le_iff]
      /-
        case neg.intro.inr.refine_1
        α : Type u_1
        inst✝ : Preorder α
        l : Ordnode α
        x y : α
        r : Ordnode α
        o₁ : WithBot α
        o₂ : WithTop α
        hl : Ordnode.Valid' o₁ l ↑x
        hr : Ordnode.Valid' (↑y) r o₂
        s : Nat
        ml : Ordnode α
        z : α
        mr : Ordnode α
        hm : Ordnode.Valid' (↑x) (Ordnode.node s ml z mr) ↑y
        Hm : LT.lt 0 (Ordnode.node s ml z mr).size
        l0 : LT.lt 0 l.size
        mr₁ : LE.le (HMul.hMul Ordnode.ratio r.size) (HAdd.hAdd (HAdd.hAdd ml.size mr. …
        lr₁ : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd  …
        lr₂ : LE.le (HMul.hMul 3 (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1)  …
        mr₂ : LE.le (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1) (HMul.hMul Ordnode.delta …
        r0 : GT.gt r.size 0
        mm : Not (LE.le (HAdd.hAdd ml.size mr.size) 1)
        mm₁ : LE.le ml.size (HMul.hMul Ordnode.delta mr.size)
        mm₂ : LE.le mr.size (HMul.hMul Ordnode.delta ml.size)
        ml0 : GT.gt ml.size 0
        this : LE.le (HMul.hMul 2 l.size) (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1)
        ⊢ LT.lt (HAdd.hAdd ml.size mr.size) (HAdd.hAdd (HMul.hMul Ordnode.delta ml.siz …
      -/
      refine add_lt_add_of_lt_of_le ?_ mm₂
      /-
        case neg.intro.inr.refine_1
        α : Type u_1
        inst✝ : Preorder α
        l : Ordnode α
        x y : α
        r : Ordnode α
        o₁ : WithBot α
        o₂ : WithTop α
        hl : Ordnode.Valid' o₁ l ↑x
        hr : Ordnode.Valid' (↑y) r o₂
        s : Nat
        ml : Ordnode α
        z : α
        mr : Ordnode α
        hm : Ordnode.Valid' (↑x) (Ordnode.node s ml z mr) ↑y
        Hm : LT.lt 0 (Ordnode.node s ml z mr).size
        l0 : LT.lt 0 l.size
        mr₁ : LE.le (HMul.hMul Ordnode.ratio r.size) (HAdd.hAdd (HAdd.hAdd ml.size mr. …
        lr₁ : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd  …
        lr₂ : LE.le (HMul.hMul 3 (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1)  …
        mr₂ : LE.le (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1) (HMul.hMul Ordnode.delta …
        r0 : GT.gt r.size 0
        mm : Not (LE.le (HAdd.hAdd ml.size mr.size) 1)
        mm₁ : LE.le ml.size (HMul.hMul Ordnode.delta mr.size)
        mm₂ : LE.le mr.size (HMul.hMul Ordnode.delta ml.size)
        ml0 : GT.gt ml.size 0
        this : LE.le (HMul.hMul 2 l.size) (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1)
        ⊢ LT.lt ml.size (HMul.hMul Ordnode.delta ml.size)
      -/
      simpa using (mul_lt_mul_right ml0).2 (by decide : 1 < 3)
      /-
        🎉 no goals
      -/
      /-
        case neg.intro.inr.refine_2
        α : Type u_1
        inst✝ : Preorder α
        l : Ordnode α
        x y : α
        r : Ordnode α
        o₁ : WithBot α
        o₂ : WithTop α
        hl : Ordnode.Valid' o₁ l ↑x
        hr : Ordnode.Valid' (↑y) r o₂
        s : Nat
        ml : Ordnode α
        z : α
        mr : Ordnode α
        hm : Ordnode.Valid' (↑x) (Ordnode.node s ml z mr) ↑y
        Hm : LT.lt 0 (Ordnode.node s ml z mr).size
        l0 : LT.lt 0 l.size
        mr₁ : LE.le (HMul.hMul Ordnode.ratio r.size) (HAdd.hAdd (HAdd.hAdd ml.size mr. …
        lr₁ : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd  …
        lr₂ : LE.le (HMul.hMul 3 (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1)  …
        mr₂ : LE.le (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1) (HMul.hMul Ordnode.delta …
        r0 : GT.gt r.size 0
        mm : Not (LE.le (HAdd.hAdd ml.size mr.size) 1)
        mm₁ : LE.le ml.size (HMul.hMul Ordnode.delta mr.size)
        mm₂ : LE.le mr.size (HMul.hMul Ordnode.delta ml.size)
        ml0 : GT.gt ml.size 0
        this : LE.le (HMul.hMul 2 l.size) (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1)
        ⊢ LE.le ml.size (HMul.hMul Ordnode.delta l.size)
      -/
    · exact Nat.le_of_lt_succ (Valid'.node4L_lemma₁ lr₂ mr₂ mm₁)
      /-
        🎉 no goals
      -/
      /-
        case neg.intro.inr.refine_3
        α : Type u_1
        inst✝ : Preorder α
        l : Ordnode α
        x y : α
        r : Ordnode α
        o₁ : WithBot α
        o₂ : WithTop α
        hl : Ordnode.Valid' o₁ l ↑x
        hr : Ordnode.Valid' (↑y) r o₂
        s : Nat
        ml : Ordnode α
        z : α
        mr : Ordnode α
        hm : Ordnode.Valid' (↑x) (Ordnode.node s ml z mr) ↑y
        Hm : LT.lt 0 (Ordnode.node s ml z mr).size
        l0 : LT.lt 0 l.size
        mr₁ : LE.le (HMul.hMul Ordnode.ratio r.size) (HAdd.hAdd (HAdd.hAdd ml.size mr. …
        lr₁ : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd  …
        lr₂ : LE.le (HMul.hMul 3 (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1)  …
        mr₂ : LE.le (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1) (HMul.hMul Ordnode.delta …
        r0 : GT.gt r.size 0
        mm : Not (LE.le (HAdd.hAdd ml.size mr.size) 1)
        mm₁ : LE.le ml.size (HMul.hMul Ordnode.delta mr.size)
        mm₂ : LE.le mr.size (HMul.hMul Ordnode.delta ml.size)
        ml0 : GT.gt ml.size 0
        this : LE.le (HMul.hMul 2 l.size) (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1)
        ⊢ LE.le mr.size (HMul.hMul Ordnode.delta r.size)
      -/
    · exact Valid'.node4L_lemma₂ mr₂
      /-
        🎉 no goals
      -/
      /-
        case neg.intro.inr.refine_4
        α : Type u_1
        inst✝ : Preorder α
        l : Ordnode α
        x y : α
        r : Ordnode α
        o₁ : WithBot α
        o₂ : WithTop α
        hl : Ordnode.Valid' o₁ l ↑x
        hr : Ordnode.Valid' (↑y) r o₂
        s : Nat
        ml : Ordnode α
        z : α
        mr : Ordnode α
        hm : Ordnode.Valid' (↑x) (Ordnode.node s ml z mr) ↑y
        Hm : LT.lt 0 (Ordnode.node s ml z mr).size
        l0 : LT.lt 0 l.size
        mr₁ : LE.le (HMul.hMul Ordnode.ratio r.size) (HAdd.hAdd (HAdd.hAdd ml.size mr. …
        lr₁ : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd  …
        lr₂ : LE.le (HMul.hMul 3 (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1)  …
        mr₂ : LE.le (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1) (HMul.hMul Ordnode.delta …
        r0 : GT.gt r.size 0
        mm : Not (LE.le (HAdd.hAdd ml.size mr.size) 1)
        mm₁ : LE.le ml.size (HMul.hMul Ordnode.delta mr.size)
        mm₂ : LE.le mr.size (HMul.hMul Ordnode.delta ml.size)
        ml0 : GT.gt ml.size 0
        this : LE.le (HMul.hMul 2 l.size) (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1)
        ⊢ LE.le r.size (HMul.hMul Ordnode.delta mr.size)
      -/
    · exact Valid'.node4L_lemma₃ mr₁ mm₁
      /-
        🎉 no goals
      -/
      /-
        case neg.intro.inr.refine_5
        α : Type u_1
        inst✝ : Preorder α
        l : Ordnode α
        x y : α
        r : Ordnode α
        o₁ : WithBot α
        o₂ : WithTop α
        hl : Ordnode.Valid' o₁ l ↑x
        hr : Ordnode.Valid' (↑y) r o₂
        s : Nat
        ml : Ordnode α
        z : α
        mr : Ordnode α
        hm : Ordnode.Valid' (↑x) (Ordnode.node s ml z mr) ↑y
        Hm : LT.lt 0 (Ordnode.node s ml z mr).size
        l0 : LT.lt 0 l.size
        mr₁ : LE.le (HMul.hMul Ordnode.ratio r.size) (HAdd.hAdd (HAdd.hAdd ml.size mr. …
        lr₁ : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd  …
        lr₂ : LE.le (HMul.hMul 3 (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1)  …
        mr₂ : LE.le (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1) (HMul.hMul Ordnode.delta …
        r0 : GT.gt r.size 0
        mm : Not (LE.le (HAdd.hAdd ml.size mr.size) 1)
        mm₁ : LE.le ml.size (HMul.hMul Ordnode.delta mr.size)
        mm₂ : LE.le mr.size (HMul.hMul Ordnode.delta ml.size)
        ml0 : GT.gt ml.size 0
        this : LE.le (HMul.hMul 2 l.size) (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1)
        ⊢ LE.le (HAdd.hAdd (HAdd.hAdd l.size ml.size) 1) (HMul.hMul Ordnode.delta (HAd …
      -/
    · exact Valid'.node4L_lemma₄ lr₁ mr₂ mm₁
      /-
        🎉 no goals
      -/
      /-
        case neg.intro.inr.refine_6
        α : Type u_1
        inst✝ : Preorder α
        l : Ordnode α
        x y : α
        r : Ordnode α
        o₁ : WithBot α
        o₂ : WithTop α
        hl : Ordnode.Valid' o₁ l ↑x
        hr : Ordnode.Valid' (↑y) r o₂
        s : Nat
        ml : Ordnode α
        z : α
        mr : Ordnode α
        hm : Ordnode.Valid' (↑x) (Ordnode.node s ml z mr) ↑y
        Hm : LT.lt 0 (Ordnode.node s ml z mr).size
        l0 : LT.lt 0 l.size
        mr₁ : LE.le (HMul.hMul Ordnode.ratio r.size) (HAdd.hAdd (HAdd.hAdd ml.size mr. …
        lr₁ : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd  …
        lr₂ : LE.le (HMul.hMul 3 (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1)  …
        mr₂ : LE.le (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1) (HMul.hMul Ordnode.delta …
        r0 : GT.gt r.size 0
        mm : Not (LE.le (HAdd.hAdd ml.size mr.size) 1)
        mm₁ : LE.le ml.size (HMul.hMul Ordnode.delta mr.size)
        mm₂ : LE.le mr.size (HMul.hMul Ordnode.delta ml.size)
        ml0 : GT.gt ml.size 0
        this : LE.le (HMul.hMul 2 l.size) (HAdd.hAdd (HAdd.hAdd ml.size mr.size) 1)
        ⊢ LE.le (HAdd.hAdd (HAdd.hAdd mr.size r.size) 1) (HMul.hMul Ordnode.delta (HAd …
      -/
    · exact Valid'.node4L_lemma₅ lr₂ mr₁ mm₂
      /-
        🎉 no goals
      -/


theorem Valid'.rotateL_lemma₁ {a b c : ℕ} (H2 : 3 * a ≤ b + c) (hb₂ : c ≤ 3 * b) : a ≤ 3 * b := by
  /-
    a b c : Nat
    H2 : LE.le (HMul.hMul 3 a) (HAdd.hAdd b c)
    hb₂ : LE.le c (HMul.hMul 3 b)
    ⊢ LE.le a (HMul.hMul 3 b)
  -/
  omega
  /-
    🎉 no goals
  -/


theorem Valid'.rotateL_lemma₂ {a b c : ℕ} (H3 : 2 * (b + c) ≤ 9 * a + 3) (h : b < 2 * c) :
                        /-
                          a b c : Nat
                          H3 : LE.le (HMul.hMul 2 (HAdd.hAdd b c)) (HAdd.hAdd (HMul.hMul 9 a) 3)
                          h : LT.lt b (HMul.hMul 2 c)
                          ⊢ LT.lt b (HAdd.hAdd (HMul.hMul 3 a) 1)
                        -/
    b < 3 * a + 1 := by omega
                        /-
                          🎉 no goals
                        -/


theorem Valid'.rotateL_lemma₃ {a b c : ℕ} (H2 : 3 * a ≤ b + c) (h : b < 2 * c) : a + b < 3 * c := by
  /-
    a b c : Nat
    H2 : LE.le (HMul.hMul 3 a) (HAdd.hAdd b c)
    h : LT.lt b (HMul.hMul 2 c)
    ⊢ LT.lt (HAdd.hAdd a b) (HMul.hMul 3 c)
  -/
  omega
  /-
    🎉 no goals
  -/


theorem Valid'.rotateL_lemma₄ {a b : ℕ} (H3 : 2 * b ≤ 9 * a + 3) : 3 * b ≤ 16 * a + 9 := by
  /-
    a b : Nat
    H3 : LE.le (HMul.hMul 2 b) (HAdd.hAdd (HMul.hMul 9 a) 3)
    ⊢ LE.le (HMul.hMul 3 b) (HAdd.hAdd (HMul.hMul 16 a) 9)
  -/
  omega
  /-
    🎉 no goals
  -/


theorem Valid'.rotateL {l} {x : α} {r o₁ o₂} (hl : Valid' o₁ l x) (hr : Valid' x r o₂)
    (H1 : ¬size l + size r ≤ 1) (H2 : delta * size l < size r)
    (H3 : 2 * size r ≤ 9 * size l + 5 ∨ size r ≤ 3) : Valid' o₁ (@rotateL α l x r) o₂ := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    l : Ordnode α
    x : α
    r : Ordnode α
    o₁ : WithBot α
    o₂ : WithTop α
    hl : Ordnode.Valid' o₁ l ↑x
    hr : Ordnode.Valid' (↑x) r o₂
    H1 : Not (LE.le (HAdd.hAdd l.size r.size) 1)
    H2 : LT.lt (HMul.hMul Ordnode.delta l.size) r.size
    H3 : Or (LE.le (HMul.hMul 2 r.size) (HAdd.hAdd (HMul.hMul 9 l.size) 5)) (LE.le …
    ⊢ Ordnode.Valid' o₁ (l.rotateL x r) o₂
  -/
  cases' r with rs rl rx rr; · cases H2
                               /-
                                 🎉 no goals
                               -/
  /-
    case node
    α : Type u_1
    inst✝ : Preorder α
    l : Ordnode α
    x : α
    o₁ : WithBot α
    o₂ : WithTop α
    hl : Ordnode.Valid' o₁ l ↑x
    rs : Nat
    rl : Ordnode α
    rx : α
    rr : Ordnode α
    hr : Ordnode.Valid' (↑x) (Ordnode.node rs rl rx rr) o₂
    H1 : Not (LE.le (HAdd.hAdd l.size (Ordnode.node rs rl rx rr).size) 1)
    H2 : LT.lt (HMul.hMul Ordnode.delta l.size) (Ordnode.node rs rl rx rr).size
    H3 : Or (LE.le (HMul.hMul 2 (Ordnode.node rs rl rx rr).size) (HAdd.hAdd (HMul. …
    ⊢ Ordnode.Valid' o₁ (l.rotateL x (Ordnode.node rs rl rx rr)) o₂
  -/
  rw [hr.2.size_eq, Nat.lt_succ_iff] at H2
  /-
    case node
    α : Type u_1
    inst✝ : Preorder α
    l : Ordnode α
    x : α
    o₁ : WithBot α
    o₂ : WithTop α
    hl : Ordnode.Valid' o₁ l ↑x
    rs : Nat
    rl : Ordnode α
    rx : α
    rr : Ordnode α
    hr : Ordnode.Valid' (↑x) (Ordnode.node rs rl rx rr) o₂
    H1 : Not (LE.le (HAdd.hAdd l.size (Ordnode.node rs rl rx rr).size) 1)
    H2 : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd rl.size rr.size)
    H3 : Or (LE.le (HMul.hMul 2 (Ordnode.node rs rl rx rr).size) (HAdd.hAdd (HMul. …
    ⊢ Ordnode.Valid' o₁ (l.rotateL x (Ordnode.node rs rl rx rr)) o₂
  -/
  rw [hr.2.size_eq] at H3
  replace H3 : 2 * (size rl + size rr) ≤ 9 * size l + 3 ∨ size rl + size rr ≤ 2 :=
    H3.imp (@Nat.le_of_add_le_add_right _ 2 _) Nat.le_of_succ_le_succ
  have H3_0 : size l = 0 → size rl + size rr ≤ 2 := by
    intro l0; rw [l0] at H3
    exact
      (or_iff_right_of_imp fun h => (mul_le_mul_left (by decide)).1 (le_trans h (by decide))).1 H3
  have H3p : size l > 0 → 2 * (size rl + size rr) ≤ 9 * size l + 3 := fun l0 : 1 ≤ size l =>
    (or_iff_left_of_imp <| by omega).1 H3
  /-
    case node
    α : Type u_1
    inst✝ : Preorder α
    l : Ordnode α
    x : α
    o₁ : WithBot α
    o₂ : WithTop α
    hl : Ordnode.Valid' o₁ l ↑x
    rs : Nat
    rl : Ordnode α
    rx : α
    rr : Ordnode α
    hr : Ordnode.Valid' (↑x) (Ordnode.node rs rl rx rr) o₂
    H1 : Not (LE.le (HAdd.hAdd l.size (Ordnode.node rs rl rx rr).size) 1)
    H2 : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd rl.size rr.size)
    H3 : Or (LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.hAdd (HMul.hMul …
    H3_0 : Eq l.size 0 → LE.le (HAdd.hAdd rl.size rr.size) 2
    H3p : GT.gt l.size 0 → LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.h …
    ⊢ Ordnode.Valid' o₁ (l.rotateL x (Ordnode.node rs rl rx rr)) o₂
  -/
  have ablem : ∀ {a b : ℕ}, 1 ≤ a → a + b ≤ 2 → b ≤ 1 := by omega
  have hlp : size l > 0 → ¬size rl + size rr ≤ 1 := fun l0 hb =>
    absurd (le_trans (le_trans (Nat.mul_le_mul_left _ l0) H2) hb) (by decide)
  /-
    case node
    α : Type u_1
    inst✝ : Preorder α
    l : Ordnode α
    x : α
    o₁ : WithBot α
    o₂ : WithTop α
    hl : Ordnode.Valid' o₁ l ↑x
    rs : Nat
    rl : Ordnode α
    rx : α
    rr : Ordnode α
    hr : Ordnode.Valid' (↑x) (Ordnode.node rs rl rx rr) o₂
    H1 : Not (LE.le (HAdd.hAdd l.size (Ordnode.node rs rl rx rr).size) 1)
    H2 : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd rl.size rr.size)
    H3 : Or (LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.hAdd (HMul.hMul …
    H3_0 : Eq l.size 0 → LE.le (HAdd.hAdd rl.size rr.size) 2
    H3p : GT.gt l.size 0 → LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.h …
    ablem : ∀ {a b : Nat}, LE.le 1 a → LE.le (HAdd.hAdd a b) 2 → LE.le b 1
    hlp : GT.gt l.size 0 → Not (LE.le (HAdd.hAdd rl.size rr.size) 1)
    ⊢ Ordnode.Valid' o₁ (l.rotateL x (Ordnode.node rs rl rx rr)) o₂
  -/
  rw [Ordnode.rotateL_node]; split_ifs with h
  · have rr0 : size rr > 0 :=
      (mul_lt_mul_left (by decide)).1 (lt_of_le_of_lt (Nat.zero_le _) h : ratio * 0 < _)
    suffices BalancedSz (size l) (size rl) ∧ BalancedSz (size l + size rl + 1) (size rr) by
      exact hl.node3L hr.left hr.right this.1 this.2
    /-
      case pos
      α : Type u_1
      inst✝ : Preorder α
      l : Ordnode α
      x : α
      o₁ : WithBot α
      o₂ : WithTop α
      hl : Ordnode.Valid' o₁ l ↑x
      rs : Nat
      rl : Ordnode α
      rx : α
      rr : Ordnode α
      hr : Ordnode.Valid' (↑x) (Ordnode.node rs rl rx rr) o₂
      H1 : Not (LE.le (HAdd.hAdd l.size (Ordnode.node rs rl rx rr).size) 1)
      H2 : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd rl.size rr.size)
      H3 : Or (LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.hAdd (HMul.hMul …
      H3_0 : Eq l.size 0 → LE.le (HAdd.hAdd rl.size rr.size) 2
      H3p : GT.gt l.size 0 → LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.h …
      ablem : ∀ {a b : Nat}, LE.le 1 a → LE.le (HAdd.hAdd a b) 2 → LE.le b 1
      hlp : GT.gt l.size 0 → Not (LE.le (HAdd.hAdd rl.size rr.size) 1)
      h : LT.lt rl.size (HMul.hMul Ordnode.ratio rr.size)
      rr0 : GT.gt rr.size 0
      ⊢ And (Ordnode.BalancedSz l.size rl.size) (Ordnode.BalancedSz (HAdd.hAdd (HAdd …
    -/
    rcases Nat.eq_zero_or_pos (size l) with l0 | l0
      /-
        case pos.inl
        α : Type u_1
        inst✝ : Preorder α
        l : Ordnode α
        x : α
        o₁ : WithBot α
        o₂ : WithTop α
        hl : Ordnode.Valid' o₁ l ↑x
        rs : Nat
        rl : Ordnode α
        rx : α
        rr : Ordnode α
        hr : Ordnode.Valid' (↑x) (Ordnode.node rs rl rx rr) o₂
        H1 : Not (LE.le (HAdd.hAdd l.size (Ordnode.node rs rl rx rr).size) 1)
        H2 : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd rl.size rr.size)
        H3 : Or (LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.hAdd (HMul.hMul …
        H3_0 : Eq l.size 0 → LE.le (HAdd.hAdd rl.size rr.size) 2
        H3p : GT.gt l.size 0 → LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.h …
        ablem : ∀ {a b : Nat}, LE.le 1 a → LE.le (HAdd.hAdd a b) 2 → LE.le b 1
        hlp : GT.gt l.size 0 → Not (LE.le (HAdd.hAdd rl.size rr.size) 1)
        h : LT.lt rl.size (HMul.hMul Ordnode.ratio rr.size)
        rr0 : GT.gt rr.size 0
        l0 : Eq l.size 0
        ⊢ And (Ordnode.BalancedSz l.size rl.size) (Ordnode.BalancedSz (HAdd.hAdd (HAdd …
      -/
    · rw [l0]; replace H3 := H3_0 l0
      /-
        case pos.inl
        α : Type u_1
        inst✝ : Preorder α
        l : Ordnode α
        x : α
        o₁ : WithBot α
        o₂ : WithTop α
        hl : Ordnode.Valid' o₁ l ↑x
        rs : Nat
        rl : Ordnode α
        rx : α
        rr : Ordnode α
        hr : Ordnode.Valid' (↑x) (Ordnode.node rs rl rx rr) o₂
        H1 : Not (LE.le (HAdd.hAdd l.size (Ordnode.node rs rl rx rr).size) 1)
        H2 : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd rl.size rr.size)
        H3_0 : Eq l.size 0 → LE.le (HAdd.hAdd rl.size rr.size) 2
        H3p : GT.gt l.size 0 → LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.h …
        ablem : ∀ {a b : Nat}, LE.le 1 a → LE.le (HAdd.hAdd a b) 2 → LE.le b 1
        hlp : GT.gt l.size 0 → Not (LE.le (HAdd.hAdd rl.size rr.size) 1)
        h : LT.lt rl.size (HMul.hMul Ordnode.ratio rr.size)
        rr0 : GT.gt rr.size 0
        l0 : Eq l.size 0
        H3 : LE.le (HAdd.hAdd rl.size rr.size) 2
        ⊢ And (Ordnode.BalancedSz 0 rl.size) (Ordnode.BalancedSz (HAdd.hAdd (HAdd.hAdd …
      -/
      have := hr.3.1
      /-
        case pos.inl
        α : Type u_1
        inst✝ : Preorder α
        l : Ordnode α
        x : α
        o₁ : WithBot α
        o₂ : WithTop α
        hl : Ordnode.Valid' o₁ l ↑x
        rs : Nat
        rl : Ordnode α
        rx : α
        rr : Ordnode α
        hr : Ordnode.Valid' (↑x) (Ordnode.node rs rl rx rr) o₂
        H1 : Not (LE.le (HAdd.hAdd l.size (Ordnode.node rs rl rx rr).size) 1)
        H2 : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd rl.size rr.size)
        H3_0 : Eq l.size 0 → LE.le (HAdd.hAdd rl.size rr.size) 2
        H3p : GT.gt l.size 0 → LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.h …
        ablem : ∀ {a b : Nat}, LE.le 1 a → LE.le (HAdd.hAdd a b) 2 → LE.le b 1
        hlp : GT.gt l.size 0 → Not (LE.le (HAdd.hAdd rl.size rr.size) 1)
        h : LT.lt rl.size (HMul.hMul Ordnode.ratio rr.size)
        rr0 : GT.gt rr.size 0
        l0 : Eq l.size 0
        H3 : LE.le (HAdd.hAdd rl.size rr.size) 2
        this : Ordnode.BalancedSz rl.size rr.size
        ⊢ And (Ordnode.BalancedSz 0 rl.size) (Ordnode.BalancedSz (HAdd.hAdd (HAdd.hAdd …
      -/
      rcases Nat.eq_zero_or_pos (size rl) with rl0 | rl0
        /-
          case pos.inl.inl
          α : Type u_1
          inst✝ : Preorder α
          l : Ordnode α
          x : α
          o₁ : WithBot α
          o₂ : WithTop α
          hl : Ordnode.Valid' o₁ l ↑x
          rs : Nat
          rl : Ordnode α
          rx : α
          rr : Ordnode α
          hr : Ordnode.Valid' (↑x) (Ordnode.node rs rl rx rr) o₂
          H1 : Not (LE.le (HAdd.hAdd l.size (Ordnode.node rs rl rx rr).size) 1)
          H2 : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd rl.size rr.size)
          H3_0 : Eq l.size 0 → LE.le (HAdd.hAdd rl.size rr.size) 2
          H3p : GT.gt l.size 0 → LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.h …
          ablem : ∀ {a b : Nat}, LE.le 1 a → LE.le (HAdd.hAdd a b) 2 → LE.le b 1
          hlp : GT.gt l.size 0 → Not (LE.le (HAdd.hAdd rl.size rr.size) 1)
          h : LT.lt rl.size (HMul.hMul Ordnode.ratio rr.size)
          rr0 : GT.gt rr.size 0
          l0 : Eq l.size 0
          H3 : LE.le (HAdd.hAdd rl.size rr.size) 2
          this : Ordnode.BalancedSz rl.size rr.size
          rl0 : Eq rl.size 0
          ⊢ And (Ordnode.BalancedSz 0 rl.size) (Ordnode.BalancedSz (HAdd.hAdd (HAdd.hAdd …
        -/
      · rw [rl0] at this ⊢
        /-
          case pos.inl.inl
          α : Type u_1
          inst✝ : Preorder α
          l : Ordnode α
          x : α
          o₁ : WithBot α
          o₂ : WithTop α
          hl : Ordnode.Valid' o₁ l ↑x
          rs : Nat
          rl : Ordnode α
          rx : α
          rr : Ordnode α
          hr : Ordnode.Valid' (↑x) (Ordnode.node rs rl rx rr) o₂
          H1 : Not (LE.le (HAdd.hAdd l.size (Ordnode.node rs rl rx rr).size) 1)
          H2 : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd rl.size rr.size)
          H3_0 : Eq l.size 0 → LE.le (HAdd.hAdd rl.size rr.size) 2
          H3p : GT.gt l.size 0 → LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.h …
          ablem : ∀ {a b : Nat}, LE.le 1 a → LE.le (HAdd.hAdd a b) 2 → LE.le b 1
          hlp : GT.gt l.size 0 → Not (LE.le (HAdd.hAdd rl.size rr.size) 1)
          h : LT.lt rl.size (HMul.hMul Ordnode.ratio rr.size)
          rr0 : GT.gt rr.size 0
          l0 : Eq l.size 0
          H3 : LE.le (HAdd.hAdd rl.size rr.size) 2
          this : Ordnode.BalancedSz 0 rr.size
          rl0 : Eq rl.size 0
          ⊢ And (Ordnode.BalancedSz 0 0) (Ordnode.BalancedSz (HAdd.hAdd (HAdd.hAdd 0 0)  …
        -/
        rw [le_antisymm (balancedSz_zero.1 this.symm) rr0]
        /-
          case pos.inl.inl
          α : Type u_1
          inst✝ : Preorder α
          l : Ordnode α
          x : α
          o₁ : WithBot α
          o₂ : WithTop α
          hl : Ordnode.Valid' o₁ l ↑x
          rs : Nat
          rl : Ordnode α
          rx : α
          rr : Ordnode α
          hr : Ordnode.Valid' (↑x) (Ordnode.node rs rl rx rr) o₂
          H1 : Not (LE.le (HAdd.hAdd l.size (Ordnode.node rs rl rx rr).size) 1)
          H2 : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd rl.size rr.size)
          H3_0 : Eq l.size 0 → LE.le (HAdd.hAdd rl.size rr.size) 2
          H3p : GT.gt l.size 0 → LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.h …
          ablem : ∀ {a b : Nat}, LE.le 1 a → LE.le (HAdd.hAdd a b) 2 → LE.le b 1
          hlp : GT.gt l.size 0 → Not (LE.le (HAdd.hAdd rl.size rr.size) 1)
          h : LT.lt rl.size (HMul.hMul Ordnode.ratio rr.size)
          rr0 : GT.gt rr.size 0
          l0 : Eq l.size 0
          H3 : LE.le (HAdd.hAdd rl.size rr.size) 2
          this : Ordnode.BalancedSz 0 rr.size
          rl0 : Eq rl.size 0
          ⊢ And (Ordnode.BalancedSz 0 0) (Ordnode.BalancedSz (HAdd.hAdd (HAdd.hAdd 0 0)  …
        -/
        decide
        /-
          🎉 no goals
        -/
      /-
        case pos.inl.inr
        α : Type u_1
        inst✝ : Preorder α
        l : Ordnode α
        x : α
        o₁ : WithBot α
        o₂ : WithTop α
        hl : Ordnode.Valid' o₁ l ↑x
        rs : Nat
        rl : Ordnode α
        rx : α
        rr : Ordnode α
        hr : Ordnode.Valid' (↑x) (Ordnode.node rs rl rx rr) o₂
        H1 : Not (LE.le (HAdd.hAdd l.size (Ordnode.node rs rl rx rr).size) 1)
        H2 : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd rl.size rr.size)
        H3_0 : Eq l.size 0 → LE.le (HAdd.hAdd rl.size rr.size) 2
        H3p : GT.gt l.size 0 → LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.h …
        ablem : ∀ {a b : Nat}, LE.le 1 a → LE.le (HAdd.hAdd a b) 2 → LE.le b 1
        hlp : GT.gt l.size 0 → Not (LE.le (HAdd.hAdd rl.size rr.size) 1)
        h : LT.lt rl.size (HMul.hMul Ordnode.ratio rr.size)
        rr0 : GT.gt rr.size 0
        l0 : Eq l.size 0
        H3 : LE.le (HAdd.hAdd rl.size rr.size) 2
        this : Ordnode.BalancedSz rl.size rr.size
        rl0 : GT.gt rl.size 0
        ⊢ And (Ordnode.BalancedSz 0 rl.size) (Ordnode.BalancedSz (HAdd.hAdd (HAdd.hAdd …
      -/
      have rr1 : size rr = 1 := le_antisymm (ablem rl0 H3) rr0
      /-
        case pos.inl.inr
        α : Type u_1
        inst✝ : Preorder α
        l : Ordnode α
        x : α
        o₁ : WithBot α
        o₂ : WithTop α
        hl : Ordnode.Valid' o₁ l ↑x
        rs : Nat
        rl : Ordnode α
        rx : α
        rr : Ordnode α
        hr : Ordnode.Valid' (↑x) (Ordnode.node rs rl rx rr) o₂
        H1 : Not (LE.le (HAdd.hAdd l.size (Ordnode.node rs rl rx rr).size) 1)
        H2 : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd rl.size rr.size)
        H3_0 : Eq l.size 0 → LE.le (HAdd.hAdd rl.size rr.size) 2
        H3p : GT.gt l.size 0 → LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.h …
        ablem : ∀ {a b : Nat}, LE.le 1 a → LE.le (HAdd.hAdd a b) 2 → LE.le b 1
        hlp : GT.gt l.size 0 → Not (LE.le (HAdd.hAdd rl.size rr.size) 1)
        h : LT.lt rl.size (HMul.hMul Ordnode.ratio rr.size)
        rr0 : GT.gt rr.size 0
        l0 : Eq l.size 0
        H3 : LE.le (HAdd.hAdd rl.size rr.size) 2
        this : Ordnode.BalancedSz rl.size rr.size
        rl0 : GT.gt rl.size 0
        rr1 : Eq rr.size 1
        ⊢ And (Ordnode.BalancedSz 0 rl.size) (Ordnode.BalancedSz (HAdd.hAdd (HAdd.hAdd …
      -/
      rw [add_comm] at H3
      /-
        case pos.inl.inr
        α : Type u_1
        inst✝ : Preorder α
        l : Ordnode α
        x : α
        o₁ : WithBot α
        o₂ : WithTop α
        hl : Ordnode.Valid' o₁ l ↑x
        rs : Nat
        rl : Ordnode α
        rx : α
        rr : Ordnode α
        hr : Ordnode.Valid' (↑x) (Ordnode.node rs rl rx rr) o₂
        H1 : Not (LE.le (HAdd.hAdd l.size (Ordnode.node rs rl rx rr).size) 1)
        H2 : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd rl.size rr.size)
        H3_0 : Eq l.size 0 → LE.le (HAdd.hAdd rl.size rr.size) 2
        H3p : GT.gt l.size 0 → LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.h …
        ablem : ∀ {a b : Nat}, LE.le 1 a → LE.le (HAdd.hAdd a b) 2 → LE.le b 1
        hlp : GT.gt l.size 0 → Not (LE.le (HAdd.hAdd rl.size rr.size) 1)
        h : LT.lt rl.size (HMul.hMul Ordnode.ratio rr.size)
        rr0 : GT.gt rr.size 0
        l0 : Eq l.size 0
        H3 : LE.le (HAdd.hAdd rr.size rl.size) 2
        this : Ordnode.BalancedSz rl.size rr.size
        rl0 : GT.gt rl.size 0
        rr1 : Eq rr.size 1
        ⊢ And (Ordnode.BalancedSz 0 rl.size) (Ordnode.BalancedSz (HAdd.hAdd (HAdd.hAdd …
      -/
      rw [rr1, show size rl = 1 from le_antisymm (ablem rr0 H3) rl0]
      /-
        case pos.inl.inr
        α : Type u_1
        inst✝ : Preorder α
        l : Ordnode α
        x : α
        o₁ : WithBot α
        o₂ : WithTop α
        hl : Ordnode.Valid' o₁ l ↑x
        rs : Nat
        rl : Ordnode α
        rx : α
        rr : Ordnode α
        hr : Ordnode.Valid' (↑x) (Ordnode.node rs rl rx rr) o₂
        H1 : Not (LE.le (HAdd.hAdd l.size (Ordnode.node rs rl rx rr).size) 1)
        H2 : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd rl.size rr.size)
        H3_0 : Eq l.size 0 → LE.le (HAdd.hAdd rl.size rr.size) 2
        H3p : GT.gt l.size 0 → LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.h …
        ablem : ∀ {a b : Nat}, LE.le 1 a → LE.le (HAdd.hAdd a b) 2 → LE.le b 1
        hlp : GT.gt l.size 0 → Not (LE.le (HAdd.hAdd rl.size rr.size) 1)
        h : LT.lt rl.size (HMul.hMul Ordnode.ratio rr.size)
        rr0 : GT.gt rr.size 0
        l0 : Eq l.size 0
        H3 : LE.le (HAdd.hAdd rr.size rl.size) 2
        this : Ordnode.BalancedSz rl.size rr.size
        rl0 : GT.gt rl.size 0
        rr1 : Eq rr.size 1
        ⊢ And (Ordnode.BalancedSz 0 1) (Ordnode.BalancedSz (HAdd.hAdd (HAdd.hAdd 0 1)  …
      -/
      decide
      /-
        🎉 no goals
      -/
    /-
      case pos.inr
      α : Type u_1
      inst✝ : Preorder α
      l : Ordnode α
      x : α
      o₁ : WithBot α
      o₂ : WithTop α
      hl : Ordnode.Valid' o₁ l ↑x
      rs : Nat
      rl : Ordnode α
      rx : α
      rr : Ordnode α
      hr : Ordnode.Valid' (↑x) (Ordnode.node rs rl rx rr) o₂
      H1 : Not (LE.le (HAdd.hAdd l.size (Ordnode.node rs rl rx rr).size) 1)
      H2 : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd rl.size rr.size)
      H3 : Or (LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.hAdd (HMul.hMul …
      H3_0 : Eq l.size 0 → LE.le (HAdd.hAdd rl.size rr.size) 2
      H3p : GT.gt l.size 0 → LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.h …
      ablem : ∀ {a b : Nat}, LE.le 1 a → LE.le (HAdd.hAdd a b) 2 → LE.le b 1
      hlp : GT.gt l.size 0 → Not (LE.le (HAdd.hAdd rl.size rr.size) 1)
      h : LT.lt rl.size (HMul.hMul Ordnode.ratio rr.size)
      rr0 : GT.gt rr.size 0
      l0 : GT.gt l.size 0
      ⊢ And (Ordnode.BalancedSz l.size rl.size) (Ordnode.BalancedSz (HAdd.hAdd (HAdd …
    -/
    replace H3 := H3p l0
    /-
      case pos.inr
      α : Type u_1
      inst✝ : Preorder α
      l : Ordnode α
      x : α
      o₁ : WithBot α
      o₂ : WithTop α
      hl : Ordnode.Valid' o₁ l ↑x
      rs : Nat
      rl : Ordnode α
      rx : α
      rr : Ordnode α
      hr : Ordnode.Valid' (↑x) (Ordnode.node rs rl rx rr) o₂
      H1 : Not (LE.le (HAdd.hAdd l.size (Ordnode.node rs rl rx rr).size) 1)
      H2 : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd rl.size rr.size)
      H3_0 : Eq l.size 0 → LE.le (HAdd.hAdd rl.size rr.size) 2
      H3p : GT.gt l.size 0 → LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.h …
      ablem : ∀ {a b : Nat}, LE.le 1 a → LE.le (HAdd.hAdd a b) 2 → LE.le b 1
      hlp : GT.gt l.size 0 → Not (LE.le (HAdd.hAdd rl.size rr.size) 1)
      h : LT.lt rl.size (HMul.hMul Ordnode.ratio rr.size)
      rr0 : GT.gt rr.size 0
      l0 : GT.gt l.size 0
      H3 : LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.hAdd (HMul.hMul 9 l …
      ⊢ And (Ordnode.BalancedSz l.size rl.size) (Ordnode.BalancedSz (HAdd.hAdd (HAdd …
    -/
    rcases hr.3.1.resolve_left (hlp l0) with ⟨_, hb₂⟩
    /-
      case pos.inr.intro
      α : Type u_1
      inst✝ : Preorder α
      l : Ordnode α
      x : α
      o₁ : WithBot α
      o₂ : WithTop α
      hl : Ordnode.Valid' o₁ l ↑x
      rs : Nat
      rl : Ordnode α
      rx : α
      rr : Ordnode α
      hr : Ordnode.Valid' (↑x) (Ordnode.node rs rl rx rr) o₂
      H1 : Not (LE.le (HAdd.hAdd l.size (Ordnode.node rs rl rx rr).size) 1)
      H2 : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd rl.size rr.size)
      H3_0 : Eq l.size 0 → LE.le (HAdd.hAdd rl.size rr.size) 2
      H3p : GT.gt l.size 0 → LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.h …
      ablem : ∀ {a b : Nat}, LE.le 1 a → LE.le (HAdd.hAdd a b) 2 → LE.le b 1
      hlp : GT.gt l.size 0 → Not (LE.le (HAdd.hAdd rl.size rr.size) 1)
      h : LT.lt rl.size (HMul.hMul Ordnode.ratio rr.size)
      rr0 : GT.gt rr.size 0
      l0 : GT.gt l.size 0
      H3 : LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.hAdd (HMul.hMul 9 l …
      left✝ : LE.le rl.size (HMul.hMul Ordnode.delta rr.size)
      hb₂ : LE.le rr.size (HMul.hMul Ordnode.delta rl.size)
      ⊢ And (Ordnode.BalancedSz l.size rl.size) (Ordnode.BalancedSz (HAdd.hAdd (HAdd …
    -/
    refine ⟨Or.inr ⟨?_, ?_⟩, Or.inr ⟨?_, ?_⟩⟩
      /-
        case pos.inr.intro.refine_1
        α : Type u_1
        inst✝ : Preorder α
        l : Ordnode α
        x : α
        o₁ : WithBot α
        o₂ : WithTop α
        hl : Ordnode.Valid' o₁ l ↑x
        rs : Nat
        rl : Ordnode α
        rx : α
        rr : Ordnode α
        hr : Ordnode.Valid' (↑x) (Ordnode.node rs rl rx rr) o₂
        H1 : Not (LE.le (HAdd.hAdd l.size (Ordnode.node rs rl rx rr).size) 1)
        H2 : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd rl.size rr.size)
        H3_0 : Eq l.size 0 → LE.le (HAdd.hAdd rl.size rr.size) 2
        H3p : GT.gt l.size 0 → LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.h …
        ablem : ∀ {a b : Nat}, LE.le 1 a → LE.le (HAdd.hAdd a b) 2 → LE.le b 1
        hlp : GT.gt l.size 0 → Not (LE.le (HAdd.hAdd rl.size rr.size) 1)
        h : LT.lt rl.size (HMul.hMul Ordnode.ratio rr.size)
        rr0 : GT.gt rr.size 0
        l0 : GT.gt l.size 0
        H3 : LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.hAdd (HMul.hMul 9 l …
        left✝ : LE.le rl.size (HMul.hMul Ordnode.delta rr.size)
        hb₂ : LE.le rr.size (HMul.hMul Ordnode.delta rl.size)
        ⊢ LE.le l.size (HMul.hMul Ordnode.delta rl.size)
      -/
    · exact Valid'.rotateL_lemma₁ H2 hb₂
      /-
        🎉 no goals
      -/
      /-
        case pos.inr.intro.refine_2
        α : Type u_1
        inst✝ : Preorder α
        l : Ordnode α
        x : α
        o₁ : WithBot α
        o₂ : WithTop α
        hl : Ordnode.Valid' o₁ l ↑x
        rs : Nat
        rl : Ordnode α
        rx : α
        rr : Ordnode α
        hr : Ordnode.Valid' (↑x) (Ordnode.node rs rl rx rr) o₂
        H1 : Not (LE.le (HAdd.hAdd l.size (Ordnode.node rs rl rx rr).size) 1)
        H2 : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd rl.size rr.size)
        H3_0 : Eq l.size 0 → LE.le (HAdd.hAdd rl.size rr.size) 2
        H3p : GT.gt l.size 0 → LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.h …
        ablem : ∀ {a b : Nat}, LE.le 1 a → LE.le (HAdd.hAdd a b) 2 → LE.le b 1
        hlp : GT.gt l.size 0 → Not (LE.le (HAdd.hAdd rl.size rr.size) 1)
        h : LT.lt rl.size (HMul.hMul Ordnode.ratio rr.size)
        rr0 : GT.gt rr.size 0
        l0 : GT.gt l.size 0
        H3 : LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.hAdd (HMul.hMul 9 l …
        left✝ : LE.le rl.size (HMul.hMul Ordnode.delta rr.size)
        hb₂ : LE.le rr.size (HMul.hMul Ordnode.delta rl.size)
        ⊢ LE.le rl.size (HMul.hMul Ordnode.delta l.size)
      -/
    · exact Nat.le_of_lt_succ (Valid'.rotateL_lemma₂ H3 h)
      /-
        🎉 no goals
      -/
      /-
        case pos.inr.intro.refine_3
        α : Type u_1
        inst✝ : Preorder α
        l : Ordnode α
        x : α
        o₁ : WithBot α
        o₂ : WithTop α
        hl : Ordnode.Valid' o₁ l ↑x
        rs : Nat
        rl : Ordnode α
        rx : α
        rr : Ordnode α
        hr : Ordnode.Valid' (↑x) (Ordnode.node rs rl rx rr) o₂
        H1 : Not (LE.le (HAdd.hAdd l.size (Ordnode.node rs rl rx rr).size) 1)
        H2 : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd rl.size rr.size)
        H3_0 : Eq l.size 0 → LE.le (HAdd.hAdd rl.size rr.size) 2
        H3p : GT.gt l.size 0 → LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.h …
        ablem : ∀ {a b : Nat}, LE.le 1 a → LE.le (HAdd.hAdd a b) 2 → LE.le b 1
        hlp : GT.gt l.size 0 → Not (LE.le (HAdd.hAdd rl.size rr.size) 1)
        h : LT.lt rl.size (HMul.hMul Ordnode.ratio rr.size)
        rr0 : GT.gt rr.size 0
        l0 : GT.gt l.size 0
        H3 : LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.hAdd (HMul.hMul 9 l …
        left✝ : LE.le rl.size (HMul.hMul Ordnode.delta rr.size)
        hb₂ : LE.le rr.size (HMul.hMul Ordnode.delta rl.size)
        ⊢ LE.le (HAdd.hAdd (HAdd.hAdd l.size rl.size) 1) (HMul.hMul Ordnode.delta rr.s …
      -/
    · exact Valid'.rotateL_lemma₃ H2 h
      /-
        🎉 no goals
      -/
    · exact
        le_trans hb₂
          (Nat.mul_le_mul_left _ <| le_trans (Nat.le_add_left _ _) (Nat.le_add_right _ _))
    /-
      case neg
      α : Type u_1
      inst✝ : Preorder α
      l : Ordnode α
      x : α
      o₁ : WithBot α
      o₂ : WithTop α
      hl : Ordnode.Valid' o₁ l ↑x
      rs : Nat
      rl : Ordnode α
      rx : α
      rr : Ordnode α
      hr : Ordnode.Valid' (↑x) (Ordnode.node rs rl rx rr) o₂
      H1 : Not (LE.le (HAdd.hAdd l.size (Ordnode.node rs rl rx rr).size) 1)
      H2 : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd rl.size rr.size)
      H3 : Or (LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.hAdd (HMul.hMul …
      H3_0 : Eq l.size 0 → LE.le (HAdd.hAdd rl.size rr.size) 2
      H3p : GT.gt l.size 0 → LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.h …
      ablem : ∀ {a b : Nat}, LE.le 1 a → LE.le (HAdd.hAdd a b) 2 → LE.le b 1
      hlp : GT.gt l.size 0 → Not (LE.le (HAdd.hAdd rl.size rr.size) 1)
      h : Not (LT.lt rl.size (HMul.hMul Ordnode.ratio rr.size))
      ⊢ Ordnode.Valid' o₁ (l.node4L x rl rx rr) o₂
    -/
  · rcases Nat.eq_zero_or_pos (size rl) with rl0 | rl0
      /-
        case neg.inl
        α : Type u_1
        inst✝ : Preorder α
        l : Ordnode α
        x : α
        o₁ : WithBot α
        o₂ : WithTop α
        hl : Ordnode.Valid' o₁ l ↑x
        rs : Nat
        rl : Ordnode α
        rx : α
        rr : Ordnode α
        hr : Ordnode.Valid' (↑x) (Ordnode.node rs rl rx rr) o₂
        H1 : Not (LE.le (HAdd.hAdd l.size (Ordnode.node rs rl rx rr).size) 1)
        H2 : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd rl.size rr.size)
        H3 : Or (LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.hAdd (HMul.hMul …
        H3_0 : Eq l.size 0 → LE.le (HAdd.hAdd rl.size rr.size) 2
        H3p : GT.gt l.size 0 → LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.h …
        ablem : ∀ {a b : Nat}, LE.le 1 a → LE.le (HAdd.hAdd a b) 2 → LE.le b 1
        hlp : GT.gt l.size 0 → Not (LE.le (HAdd.hAdd rl.size rr.size) 1)
        h : Not (LT.lt rl.size (HMul.hMul Ordnode.ratio rr.size))
        rl0 : Eq rl.size 0
        ⊢ Ordnode.Valid' o₁ (l.node4L x rl rx rr) o₂
      -/
    · rw [rl0, not_lt, Nat.le_zero, Nat.mul_eq_zero] at h
      /-
        case neg.inl
        α : Type u_1
        inst✝ : Preorder α
        l : Ordnode α
        x : α
        o₁ : WithBot α
        o₂ : WithTop α
        hl : Ordnode.Valid' o₁ l ↑x
        rs : Nat
        rl : Ordnode α
        rx : α
        rr : Ordnode α
        hr : Ordnode.Valid' (↑x) (Ordnode.node rs rl rx rr) o₂
        H1 : Not (LE.le (HAdd.hAdd l.size (Ordnode.node rs rl rx rr).size) 1)
        H2 : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd rl.size rr.size)
        H3 : Or (LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.hAdd (HMul.hMul …
        H3_0 : Eq l.size 0 → LE.le (HAdd.hAdd rl.size rr.size) 2
        H3p : GT.gt l.size 0 → LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.h …
        ablem : ∀ {a b : Nat}, LE.le 1 a → LE.le (HAdd.hAdd a b) 2 → LE.le b 1
        hlp : GT.gt l.size 0 → Not (LE.le (HAdd.hAdd rl.size rr.size) 1)
        h : Or (Eq Ordnode.ratio 0) (Eq rr.size 0)
        rl0 : Eq rl.size 0
        ⊢ Ordnode.Valid' o₁ (l.node4L x rl rx rr) o₂
      -/
      replace h := h.resolve_left (by decide)
      /-
        case neg.inl
        α : Type u_1
        inst✝ : Preorder α
        l : Ordnode α
        x : α
        o₁ : WithBot α
        o₂ : WithTop α
        hl : Ordnode.Valid' o₁ l ↑x
        rs : Nat
        rl : Ordnode α
        rx : α
        rr : Ordnode α
        hr : Ordnode.Valid' (↑x) (Ordnode.node rs rl rx rr) o₂
        H1 : Not (LE.le (HAdd.hAdd l.size (Ordnode.node rs rl rx rr).size) 1)
        H2 : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd rl.size rr.size)
        H3 : Or (LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.hAdd (HMul.hMul …
        H3_0 : Eq l.size 0 → LE.le (HAdd.hAdd rl.size rr.size) 2
        H3p : GT.gt l.size 0 → LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.h …
        ablem : ∀ {a b : Nat}, LE.le 1 a → LE.le (HAdd.hAdd a b) 2 → LE.le b 1
        hlp : GT.gt l.size 0 → Not (LE.le (HAdd.hAdd rl.size rr.size) 1)
        rl0 : Eq rl.size 0
        h : Eq rr.size 0
        ⊢ Ordnode.Valid' o₁ (l.node4L x rl rx rr) o₂
      -/
      rw [rl0, h, Nat.le_zero, Nat.mul_eq_zero] at H2
      /-
        case neg.inl
        α : Type u_1
        inst✝ : Preorder α
        l : Ordnode α
        x : α
        o₁ : WithBot α
        o₂ : WithTop α
        hl : Ordnode.Valid' o₁ l ↑x
        rs : Nat
        rl : Ordnode α
        rx : α
        rr : Ordnode α
        hr : Ordnode.Valid' (↑x) (Ordnode.node rs rl rx rr) o₂
        H1 : Not (LE.le (HAdd.hAdd l.size (Ordnode.node rs rl rx rr).size) 1)
        H2 : Or (Eq Ordnode.delta 0) (Eq l.size 0)
        H3 : Or (LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.hAdd (HMul.hMul …
        H3_0 : Eq l.size 0 → LE.le (HAdd.hAdd rl.size rr.size) 2
        H3p : GT.gt l.size 0 → LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.h …
        ablem : ∀ {a b : Nat}, LE.le 1 a → LE.le (HAdd.hAdd a b) 2 → LE.le b 1
        hlp : GT.gt l.size 0 → Not (LE.le (HAdd.hAdd rl.size rr.size) 1)
        rl0 : Eq rl.size 0
        h : Eq rr.size 0
        ⊢ Ordnode.Valid' o₁ (l.node4L x rl rx rr) o₂
      -/
      rw [hr.2.size_eq, rl0, h, H2.resolve_left (by decide)] at H1
      /-
        case neg.inl
        α : Type u_1
        inst✝ : Preorder α
        l : Ordnode α
        x : α
        o₁ : WithBot α
        o₂ : WithTop α
        hl : Ordnode.Valid' o₁ l ↑x
        rs : Nat
        rl : Ordnode α
        rx : α
        rr : Ordnode α
        hr : Ordnode.Valid' (↑x) (Ordnode.node rs rl rx rr) o₂
        H1 : Not (LE.le (HAdd.hAdd 0 (HAdd.hAdd (HAdd.hAdd 0 0) 1)) 1)
        H2 : Or (Eq Ordnode.delta 0) (Eq l.size 0)
        H3 : Or (LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.hAdd (HMul.hMul …
        H3_0 : Eq l.size 0 → LE.le (HAdd.hAdd rl.size rr.size) 2
        H3p : GT.gt l.size 0 → LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.h …
        ablem : ∀ {a b : Nat}, LE.le 1 a → LE.le (HAdd.hAdd a b) 2 → LE.le b 1
        hlp : GT.gt l.size 0 → Not (LE.le (HAdd.hAdd rl.size rr.size) 1)
        rl0 : Eq rl.size 0
        h : Eq rr.size 0
        ⊢ Ordnode.Valid' o₁ (l.node4L x rl rx rr) o₂
      -/
      cases H1 (by decide)
      /-
        🎉 no goals
      -/
    /-
      case neg.inr
      α : Type u_1
      inst✝ : Preorder α
      l : Ordnode α
      x : α
      o₁ : WithBot α
      o₂ : WithTop α
      hl : Ordnode.Valid' o₁ l ↑x
      rs : Nat
      rl : Ordnode α
      rx : α
      rr : Ordnode α
      hr : Ordnode.Valid' (↑x) (Ordnode.node rs rl rx rr) o₂
      H1 : Not (LE.le (HAdd.hAdd l.size (Ordnode.node rs rl rx rr).size) 1)
      H2 : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd rl.size rr.size)
      H3 : Or (LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.hAdd (HMul.hMul …
      H3_0 : Eq l.size 0 → LE.le (HAdd.hAdd rl.size rr.size) 2
      H3p : GT.gt l.size 0 → LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.h …
      ablem : ∀ {a b : Nat}, LE.le 1 a → LE.le (HAdd.hAdd a b) 2 → LE.le b 1
      hlp : GT.gt l.size 0 → Not (LE.le (HAdd.hAdd rl.size rr.size) 1)
      h : Not (LT.lt rl.size (HMul.hMul Ordnode.ratio rr.size))
      rl0 : GT.gt rl.size 0
      ⊢ Ordnode.Valid' o₁ (l.node4L x rl rx rr) o₂
    -/
    refine hl.node4L hr.left hr.right rl0 ?_
    /-
      case neg.inr
      α : Type u_1
      inst✝ : Preorder α
      l : Ordnode α
      x : α
      o₁ : WithBot α
      o₂ : WithTop α
      hl : Ordnode.Valid' o₁ l ↑x
      rs : Nat
      rl : Ordnode α
      rx : α
      rr : Ordnode α
      hr : Ordnode.Valid' (↑x) (Ordnode.node rs rl rx rr) o₂
      H1 : Not (LE.le (HAdd.hAdd l.size (Ordnode.node rs rl rx rr).size) 1)
      H2 : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd rl.size rr.size)
      H3 : Or (LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.hAdd (HMul.hMul …
      H3_0 : Eq l.size 0 → LE.le (HAdd.hAdd rl.size rr.size) 2
      H3p : GT.gt l.size 0 → LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.h …
      ablem : ∀ {a b : Nat}, LE.le 1 a → LE.le (HAdd.hAdd a b) 2 → LE.le b 1
      hlp : GT.gt l.size 0 → Not (LE.le (HAdd.hAdd rl.size rr.size) 1)
      h : Not (LT.lt rl.size (HMul.hMul Ordnode.ratio rr.size))
      rl0 : GT.gt rl.size 0
      ⊢ Or (And (Eq l.size 0) (And (Eq rl.size 1) (LE.le rr.size 1))) (And (LT.lt 0  …
    -/
    rcases Nat.eq_zero_or_pos (size l) with l0 | l0
      /-
        case neg.inr.inl
        α : Type u_1
        inst✝ : Preorder α
        l : Ordnode α
        x : α
        o₁ : WithBot α
        o₂ : WithTop α
        hl : Ordnode.Valid' o₁ l ↑x
        rs : Nat
        rl : Ordnode α
        rx : α
        rr : Ordnode α
        hr : Ordnode.Valid' (↑x) (Ordnode.node rs rl rx rr) o₂
        H1 : Not (LE.le (HAdd.hAdd l.size (Ordnode.node rs rl rx rr).size) 1)
        H2 : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd rl.size rr.size)
        H3 : Or (LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.hAdd (HMul.hMul …
        H3_0 : Eq l.size 0 → LE.le (HAdd.hAdd rl.size rr.size) 2
        H3p : GT.gt l.size 0 → LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.h …
        ablem : ∀ {a b : Nat}, LE.le 1 a → LE.le (HAdd.hAdd a b) 2 → LE.le b 1
        hlp : GT.gt l.size 0 → Not (LE.le (HAdd.hAdd rl.size rr.size) 1)
        h : Not (LT.lt rl.size (HMul.hMul Ordnode.ratio rr.size))
        rl0 : GT.gt rl.size 0
        l0 : Eq l.size 0
        ⊢ Or (And (Eq l.size 0) (And (Eq rl.size 1) (LE.le rr.size 1))) (And (LT.lt 0  …
      -/
    · replace H3 := H3_0 l0
      /-
        case neg.inr.inl
        α : Type u_1
        inst✝ : Preorder α
        l : Ordnode α
        x : α
        o₁ : WithBot α
        o₂ : WithTop α
        hl : Ordnode.Valid' o₁ l ↑x
        rs : Nat
        rl : Ordnode α
        rx : α
        rr : Ordnode α
        hr : Ordnode.Valid' (↑x) (Ordnode.node rs rl rx rr) o₂
        H1 : Not (LE.le (HAdd.hAdd l.size (Ordnode.node rs rl rx rr).size) 1)
        H2 : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd rl.size rr.size)
        H3_0 : Eq l.size 0 → LE.le (HAdd.hAdd rl.size rr.size) 2
        H3p : GT.gt l.size 0 → LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.h …
        ablem : ∀ {a b : Nat}, LE.le 1 a → LE.le (HAdd.hAdd a b) 2 → LE.le b 1
        hlp : GT.gt l.size 0 → Not (LE.le (HAdd.hAdd rl.size rr.size) 1)
        h : Not (LT.lt rl.size (HMul.hMul Ordnode.ratio rr.size))
        rl0 : GT.gt rl.size 0
        l0 : Eq l.size 0
        H3 : LE.le (HAdd.hAdd rl.size rr.size) 2
        ⊢ Or (And (Eq l.size 0) (And (Eq rl.size 1) (LE.le rr.size 1))) (And (LT.lt 0  …
      -/
      rcases Nat.eq_zero_or_pos (size rr) with rr0 | rr0
        /-
          case neg.inr.inl.inl
          α : Type u_1
          inst✝ : Preorder α
          l : Ordnode α
          x : α
          o₁ : WithBot α
          o₂ : WithTop α
          hl : Ordnode.Valid' o₁ l ↑x
          rs : Nat
          rl : Ordnode α
          rx : α
          rr : Ordnode α
          hr : Ordnode.Valid' (↑x) (Ordnode.node rs rl rx rr) o₂
          H1 : Not (LE.le (HAdd.hAdd l.size (Ordnode.node rs rl rx rr).size) 1)
          H2 : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd rl.size rr.size)
          H3_0 : Eq l.size 0 → LE.le (HAdd.hAdd rl.size rr.size) 2
          H3p : GT.gt l.size 0 → LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.h …
          ablem : ∀ {a b : Nat}, LE.le 1 a → LE.le (HAdd.hAdd a b) 2 → LE.le b 1
          hlp : GT.gt l.size 0 → Not (LE.le (HAdd.hAdd rl.size rr.size) 1)
          h : Not (LT.lt rl.size (HMul.hMul Ordnode.ratio rr.size))
          rl0 : GT.gt rl.size 0
          l0 : Eq l.size 0
          H3 : LE.le (HAdd.hAdd rl.size rr.size) 2
          rr0 : Eq rr.size 0
          ⊢ Or (And (Eq l.size 0) (And (Eq rl.size 1) (LE.le rr.size 1))) (And (LT.lt 0  …
        -/
      · have := hr.3.1
        /-
          case neg.inr.inl.inl
          α : Type u_1
          inst✝ : Preorder α
          l : Ordnode α
          x : α
          o₁ : WithBot α
          o₂ : WithTop α
          hl : Ordnode.Valid' o₁ l ↑x
          rs : Nat
          rl : Ordnode α
          rx : α
          rr : Ordnode α
          hr : Ordnode.Valid' (↑x) (Ordnode.node rs rl rx rr) o₂
          H1 : Not (LE.le (HAdd.hAdd l.size (Ordnode.node rs rl rx rr).size) 1)
          H2 : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd rl.size rr.size)
          H3_0 : Eq l.size 0 → LE.le (HAdd.hAdd rl.size rr.size) 2
          H3p : GT.gt l.size 0 → LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.h …
          ablem : ∀ {a b : Nat}, LE.le 1 a → LE.le (HAdd.hAdd a b) 2 → LE.le b 1
          hlp : GT.gt l.size 0 → Not (LE.le (HAdd.hAdd rl.size rr.size) 1)
          h : Not (LT.lt rl.size (HMul.hMul Ordnode.ratio rr.size))
          rl0 : GT.gt rl.size 0
          l0 : Eq l.size 0
          H3 : LE.le (HAdd.hAdd rl.size rr.size) 2
          rr0 : Eq rr.size 0
          this : Ordnode.BalancedSz rl.size rr.size
          ⊢ Or (And (Eq l.size 0) (And (Eq rl.size 1) (LE.le rr.size 1))) (And (LT.lt 0  …
        -/
        rw [rr0] at this
        /-
          case neg.inr.inl.inl
          α : Type u_1
          inst✝ : Preorder α
          l : Ordnode α
          x : α
          o₁ : WithBot α
          o₂ : WithTop α
          hl : Ordnode.Valid' o₁ l ↑x
          rs : Nat
          rl : Ordnode α
          rx : α
          rr : Ordnode α
          hr : Ordnode.Valid' (↑x) (Ordnode.node rs rl rx rr) o₂
          H1 : Not (LE.le (HAdd.hAdd l.size (Ordnode.node rs rl rx rr).size) 1)
          H2 : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd rl.size rr.size)
          H3_0 : Eq l.size 0 → LE.le (HAdd.hAdd rl.size rr.size) 2
          H3p : GT.gt l.size 0 → LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.h …
          ablem : ∀ {a b : Nat}, LE.le 1 a → LE.le (HAdd.hAdd a b) 2 → LE.le b 1
          hlp : GT.gt l.size 0 → Not (LE.le (HAdd.hAdd rl.size rr.size) 1)
          h : Not (LT.lt rl.size (HMul.hMul Ordnode.ratio rr.size))
          rl0 : GT.gt rl.size 0
          l0 : Eq l.size 0
          H3 : LE.le (HAdd.hAdd rl.size rr.size) 2
          rr0 : Eq rr.size 0
          this : Ordnode.BalancedSz rl.size 0
          ⊢ Or (And (Eq l.size 0) (And (Eq rl.size 1) (LE.le rr.size 1))) (And (LT.lt 0  …
        -/
        exact Or.inl ⟨l0, le_antisymm (balancedSz_zero.1 this) rl0, rr0.symm ▸ zero_le_one⟩
        /-
          🎉 no goals
        -/
      /-
        case neg.inr.inl.inr
        α : Type u_1
        inst✝ : Preorder α
        l : Ordnode α
        x : α
        o₁ : WithBot α
        o₂ : WithTop α
        hl : Ordnode.Valid' o₁ l ↑x
        rs : Nat
        rl : Ordnode α
        rx : α
        rr : Ordnode α
        hr : Ordnode.Valid' (↑x) (Ordnode.node rs rl rx rr) o₂
        H1 : Not (LE.le (HAdd.hAdd l.size (Ordnode.node rs rl rx rr).size) 1)
        H2 : LE.le (HMul.hMul Ordnode.delta l.size) (HAdd.hAdd rl.size rr.size)
        H3_0 : Eq l.size 0 → LE.le (HAdd.hAdd rl.size rr.size) 2
        H3p : GT.gt l.size 0 → LE.le (HMul.hMul 2 (HAdd.hAdd rl.size rr.size)) (HAdd.h …
        ablem : ∀ {a b : Nat}, LE.le 1 a → LE.le (HAdd.hAdd a b) 2 → LE.le b 1
        hlp : GT.gt l.size 0 → Not (LE.le (HAdd.hAdd rl.size rr.size) 1)
        h : Not (LT.lt rl.size (HMul.hMul Ordnode.ratio rr.size))
        rl0 : GT.gt rl.size 0
        l0 : Eq l.size 0
        H3 : LE.le (HAdd.hAdd rl.size rr.size) 2
        rr0 : GT.gt rr.size 0
        ⊢ Or (And (Eq l.size 0) (And (Eq rl.size 1) (LE.le rr.size 1))) (And (LT.lt 0  …
      -/
      exact Or.inl ⟨l0, le_antisymm (ablem rr0 <| by rwa [add_comm]) rl0, ablem rl0 H3⟩
      /-
        🎉 no goals
      -/
    exact
      Or.inr ⟨l0, not_lt.1 h, H2, Valid'.rotateL_lemma₄ (H3p l0), (hr.3.1.resolve_left (hlp l0)).1⟩


theorem Valid'.rotateR {l} {x : α} {r o₁ o₂} (hl : Valid' o₁ l x) (hr : Valid' x r o₂)
    (H1 : ¬size l + size r ≤ 1) (H2 : delta * size r < size l)
    (H3 : 2 * size l ≤ 9 * size r + 5 ∨ size l ≤ 3) : Valid' o₁ (@rotateR α l x r) o₂ := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    l : Ordnode α
    x : α
    r : Ordnode α
    o₁ : WithBot α
    o₂ : WithTop α
    hl : Ordnode.Valid' o₁ l ↑x
    hr : Ordnode.Valid' (↑x) r o₂
    H1 : Not (LE.le (HAdd.hAdd l.size r.size) 1)
    H2 : LT.lt (HMul.hMul Ordnode.delta r.size) l.size
    H3 : Or (LE.le (HMul.hMul 2 l.size) (HAdd.hAdd (HMul.hMul 9 r.size) 5)) (LE.le …
    ⊢ Ordnode.Valid' o₁ (l.rotateR x r) o₂
  -/
  refine Valid'.dual_iff.2 ?_
  /-
    α : Type u_1
    inst✝ : Preorder α
    l : Ordnode α
    x : α
    r : Ordnode α
    o₁ : WithBot α
    o₂ : WithTop α
    hl : Ordnode.Valid' o₁ l ↑x
    hr : Ordnode.Valid' (↑x) r o₂
    H1 : Not (LE.le (HAdd.hAdd l.size r.size) 1)
    H2 : LT.lt (HMul.hMul Ordnode.delta r.size) l.size
    H3 : Or (LE.le (HMul.hMul 2 l.size) (HAdd.hAdd (HMul.hMul 9 r.size) 5)) (LE.le …
    ⊢ Ordnode.Valid' o₂ (l.rotateR x r).dual o₁
  -/
  rw [dual_rotateR]
  /-
    α : Type u_1
    inst✝ : Preorder α
    l : Ordnode α
    x : α
    r : Ordnode α
    o₁ : WithBot α
    o₂ : WithTop α
    hl : Ordnode.Valid' o₁ l ↑x
    hr : Ordnode.Valid' (↑x) r o₂
    H1 : Not (LE.le (HAdd.hAdd l.size r.size) 1)
    H2 : LT.lt (HMul.hMul Ordnode.delta r.size) l.size
    H3 : Or (LE.le (HMul.hMul 2 l.size) (HAdd.hAdd (HMul.hMul 9 r.size) 5)) (LE.le …
    ⊢ Ordnode.Valid' o₂ (r.dual.rotateL x l.dual) o₁
  -/
  refine hr.dual.rotateL hl.dual ?_ ?_ ?_
    /-
      case refine_1
      α : Type u_1
      inst✝ : Preorder α
      l : Ordnode α
      x : α
      r : Ordnode α
      o₁ : WithBot α
      o₂ : WithTop α
      hl : Ordnode.Valid' o₁ l ↑x
      hr : Ordnode.Valid' (↑x) r o₂
      H1 : Not (LE.le (HAdd.hAdd l.size r.size) 1)
      H2 : LT.lt (HMul.hMul Ordnode.delta r.size) l.size
      H3 : Or (LE.le (HMul.hMul 2 l.size) (HAdd.hAdd (HMul.hMul 9 r.size) 5)) (LE.le …
      ⊢ Not (LE.le (HAdd.hAdd r.dual.size l.dual.size) 1)
    -/
  · rwa [size_dual, size_dual, add_comm]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝ : Preorder α
      l : Ordnode α
      x : α
      r : Ordnode α
      o₁ : WithBot α
      o₂ : WithTop α
      hl : Ordnode.Valid' o₁ l ↑x
      hr : Ordnode.Valid' (↑x) r o₂
      H1 : Not (LE.le (HAdd.hAdd l.size r.size) 1)
      H2 : LT.lt (HMul.hMul Ordnode.delta r.size) l.size
      H3 : Or (LE.le (HMul.hMul 2 l.size) (HAdd.hAdd (HMul.hMul 9 r.size) 5)) (LE.le …
      ⊢ LT.lt (HMul.hMul Ordnode.delta r.dual.size) l.dual.size
    -/
  · rwa [size_dual, size_dual]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_1
      inst✝ : Preorder α
      l : Ordnode α
      x : α
      r : Ordnode α
      o₁ : WithBot α
      o₂ : WithTop α
      hl : Ordnode.Valid' o₁ l ↑x
      hr : Ordnode.Valid' (↑x) r o₂
      H1 : Not (LE.le (HAdd.hAdd l.size r.size) 1)
      H2 : LT.lt (HMul.hMul Ordnode.delta r.size) l.size
      H3 : Or (LE.le (HMul.hMul 2 l.size) (HAdd.hAdd (HMul.hMul 9 r.size) 5)) (LE.le …
      ⊢ Or (LE.le (HMul.hMul 2 l.dual.size) (HAdd.hAdd (HMul.hMul 9 r.dual.size) 5)) …
    -/
  · rwa [size_dual, size_dual]
    /-
      🎉 no goals
    -/


theorem Valid'.balance'_aux {l} {x : α} {r o₁ o₂} (hl : Valid' o₁ l x) (hr : Valid' x r o₂)
    (H₁ : 2 * @size α r ≤ 9 * size l + 5 ∨ size r ≤ 3)
    (H₂ : 2 * @size α l ≤ 9 * size r + 5 ∨ size l ≤ 3) : Valid' o₁ (@balance' α l x r) o₂ := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    l : Ordnode α
    x : α
    r : Ordnode α
    o₁ : WithBot α
    o₂ : WithTop α
    hl : Ordnode.Valid' o₁ l ↑x
    hr : Ordnode.Valid' (↑x) r o₂
    H₁ : Or (LE.le (HMul.hMul 2 r.size) (HAdd.hAdd (HMul.hMul 9 l.size) 5)) (LE.le …
    H₂ : Or (LE.le (HMul.hMul 2 l.size) (HAdd.hAdd (HMul.hMul 9 r.size) 5)) (LE.le …
    ⊢ Ordnode.Valid' o₁ (l.balance' x r) o₂
  -/
  rw [balance']; split_ifs with h h_1 h_2
    /-
      case pos
      α : Type u_1
      inst✝ : Preorder α
      l : Ordnode α
      x : α
      r : Ordnode α
      o₁ : WithBot α
      o₂ : WithTop α
      hl : Ordnode.Valid' o₁ l ↑x
      hr : Ordnode.Valid' (↑x) r o₂
      H₁ : Or (LE.le (HMul.hMul 2 r.size) (HAdd.hAdd (HMul.hMul 9 l.size) 5)) (LE.le …
      H₂ : Or (LE.le (HMul.hMul 2 l.size) (HAdd.hAdd (HMul.hMul 9 r.size) 5)) (LE.le …
      h : LE.le (HAdd.hAdd l.size r.size) 1
      ⊢ Ordnode.Valid' o₁ (l.node' x r) o₂
    -/
  · exact hl.node' hr (Or.inl h)
    /-
      🎉 no goals
    -/
    /-
      case pos
      α : Type u_1
      inst✝ : Preorder α
      l : Ordnode α
      x : α
      r : Ordnode α
      o₁ : WithBot α
      o₂ : WithTop α
      hl : Ordnode.Valid' o₁ l ↑x
      hr : Ordnode.Valid' (↑x) r o₂
      H₁ : Or (LE.le (HMul.hMul 2 r.size) (HAdd.hAdd (HMul.hMul 9 l.size) 5)) (LE.le …
      H₂ : Or (LE.le (HMul.hMul 2 l.size) (HAdd.hAdd (HMul.hMul 9 r.size) 5)) (LE.le …
      h : Not (LE.le (HAdd.hAdd l.size r.size) 1)
      h_1 : GT.gt r.size (HMul.hMul Ordnode.delta l.size)
      ⊢ Ordnode.Valid' o₁ (l.rotateL x r) o₂
    -/
  · exact hl.rotateL hr h h_1 H₁
    /-
      🎉 no goals
    -/
    /-
      case pos
      α : Type u_1
      inst✝ : Preorder α
      l : Ordnode α
      x : α
      r : Ordnode α
      o₁ : WithBot α
      o₂ : WithTop α
      hl : Ordnode.Valid' o₁ l ↑x
      hr : Ordnode.Valid' (↑x) r o₂
      H₁ : Or (LE.le (HMul.hMul 2 r.size) (HAdd.hAdd (HMul.hMul 9 l.size) 5)) (LE.le …
      H₂ : Or (LE.le (HMul.hMul 2 l.size) (HAdd.hAdd (HMul.hMul 9 r.size) 5)) (LE.le …
      h : Not (LE.le (HAdd.hAdd l.size r.size) 1)
      h_1 : Not (GT.gt r.size (HMul.hMul Ordnode.delta l.size))
      h_2 : GT.gt l.size (HMul.hMul Ordnode.delta r.size)
      ⊢ Ordnode.Valid' o₁ (l.rotateR x r) o₂
    -/
  · exact hl.rotateR hr h h_2 H₂
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : Preorder α
      l : Ordnode α
      x : α
      r : Ordnode α
      o₁ : WithBot α
      o₂ : WithTop α
      hl : Ordnode.Valid' o₁ l ↑x
      hr : Ordnode.Valid' (↑x) r o₂
      H₁ : Or (LE.le (HMul.hMul 2 r.size) (HAdd.hAdd (HMul.hMul 9 l.size) 5)) (LE.le …
      H₂ : Or (LE.le (HMul.hMul 2 l.size) (HAdd.hAdd (HMul.hMul 9 r.size) 5)) (LE.le …
      h : Not (LE.le (HAdd.hAdd l.size r.size) 1)
      h_1 : Not (GT.gt r.size (HMul.hMul Ordnode.delta l.size))
      h_2 : Not (GT.gt l.size (HMul.hMul Ordnode.delta r.size))
      ⊢ Ordnode.Valid' o₁ (l.node' x r) o₂
    -/
  · exact hl.node' hr (Or.inr ⟨not_lt.1 h_2, not_lt.1 h_1⟩)
    /-
      🎉 no goals
    -/


theorem Valid'.balance'_lemma {α l l' r r'} (H1 : BalancedSz l' r')
    (H2 : Nat.dist (@size α l) l' ≤ 1 ∧ size r = r' ∨ Nat.dist (size r) r' ≤ 1 ∧ size l = l') :
    2 * @size α r ≤ 9 * size l + 5 ∨ size r ≤ 3 := by
  /-
    α : Type u_2
    l : Ordnode α
    l' : Nat
    r : Ordnode α
    r' : Nat
    H1 : Ordnode.BalancedSz l' r'
    H2 : Or (And (LE.le (l.size.dist l') 1) (Eq r.size r')) (And (LE.le (r.size.di …
    ⊢ Or (LE.le (HMul.hMul 2 r.size) (HAdd.hAdd (HMul.hMul 9 l.size) 5)) (LE.le r. …
  -/
  suffices @size α r ≤ 3 * (size l + 1) by omega
  /-
    α : Type u_2
    l : Ordnode α
    l' : Nat
    r : Ordnode α
    r' : Nat
    H1 : Ordnode.BalancedSz l' r'
    H2 : Or (And (LE.le (l.size.dist l') 1) (Eq r.size r')) (And (LE.le (r.size.di …
    ⊢ LE.le r.size (HMul.hMul 3 (HAdd.hAdd l.size 1))
  -/
  rcases H2 with (⟨hl, rfl⟩ | ⟨hr, rfl⟩) <;> rcases H1 with (h | ⟨_, h₂⟩)
    /-
      case inl.intro.inl
      α : Type u_2
      l : Ordnode α
      l' : Nat
      r : Ordnode α
      hl : LE.le (l.size.dist l') 1
      h : LE.le (HAdd.hAdd l' r.size) 1
      ⊢ LE.le r.size (HMul.hMul 3 (HAdd.hAdd l.size 1))
    -/
  · exact le_trans (Nat.le_add_left _ _) (le_trans h (Nat.le_add_left _ _))
    /-
      🎉 no goals
    -/
  · exact
      le_trans h₂
        (Nat.mul_le_mul_left _ <| le_trans (Nat.dist_tri_right _ _) (Nat.add_le_add_left hl _))
  · exact
      le_trans (Nat.dist_tri_left' _ _)
        (le_trans (add_le_add hr (le_trans (Nat.le_add_left _ _) h)) (by omega))
    /-
      case inr.intro.inr.intro
      α : Type u_2
      l r : Ordnode α
      r' : Nat
      hr : LE.le (r.size.dist r') 1
      left✝ : LE.le l.size (HMul.hMul Ordnode.delta r')
      h₂ : LE.le r' (HMul.hMul Ordnode.delta l.size)
      ⊢ LE.le r.size (HMul.hMul 3 (HAdd.hAdd l.size 1))
    -/
  · rw [Nat.mul_succ]
    /-
      case inr.intro.inr.intro
      α : Type u_2
      l r : Ordnode α
      r' : Nat
      hr : LE.le (r.size.dist r') 1
      left✝ : LE.le l.size (HMul.hMul Ordnode.delta r')
      h₂ : LE.le r' (HMul.hMul Ordnode.delta l.size)
      ⊢ LE.le r.size (HAdd.hAdd (HMul.hMul 3 l.size) 3)
    -/
    exact le_trans (Nat.dist_tri_right' _ _) (add_le_add h₂ (le_trans hr (by decide)))
    /-
      🎉 no goals
    -/


theorem Valid'.balance' {l} {x : α} {r o₁ o₂} (hl : Valid' o₁ l x) (hr : Valid' x r o₂)
    (H : ∃ l' r', BalancedSz l' r' ∧
          (Nat.dist (size l) l' ≤ 1 ∧ size r = r' ∨ Nat.dist (size r) r' ≤ 1 ∧ size l = l')) :
    Valid' o₁ (@balance' α l x r) o₂ :=
  let ⟨_, _, H1, H2⟩ := H
  Valid'.balance'_aux hl hr (Valid'.balance'_lemma H1 H2) (Valid'.balance'_lemma H1.symm H2.symm)


theorem Valid'.balance {l} {x : α} {r o₁ o₂} (hl : Valid' o₁ l x) (hr : Valid' x r o₂)
    (H : ∃ l' r', BalancedSz l' r' ∧
          (Nat.dist (size l) l' ≤ 1 ∧ size r = r' ∨ Nat.dist (size r) r' ≤ 1 ∧ size l = l')) :
    Valid' o₁ (@balance α l x r) o₂ := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    l : Ordnode α
    x : α
    r : Ordnode α
    o₁ : WithBot α
    o₂ : WithTop α
    hl : Ordnode.Valid' o₁ l ↑x
    hr : Ordnode.Valid' (↑x) r o₂
    H : Exists fun l' => Exists fun r' => And (Ordnode.BalancedSz l' r') (Or (And  …
    ⊢ Ordnode.Valid' o₁ (l.balance x r) o₂
  -/
  rw [balance_eq_balance' hl.3 hr.3 hl.2 hr.2]; exact hl.balance' hr H
                                                /-
                                                  🎉 no goals
                                                -/


theorem Valid'.balanceL_aux {l} {x : α} {r o₁ o₂} (hl : Valid' o₁ l x) (hr : Valid' x r o₂)
    (H₁ : size l = 0 → size r ≤ 1) (H₂ : 1 ≤ size l → 1 ≤ size r → size r ≤ delta * size l)
    (H₃ : 2 * @size α l ≤ 9 * size r + 5 ∨ size l ≤ 3) : Valid' o₁ (@balanceL α l x r) o₂ := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    l : Ordnode α
    x : α
    r : Ordnode α
    o₁ : WithBot α
    o₂ : WithTop α
    hl : Ordnode.Valid' o₁ l ↑x
    hr : Ordnode.Valid' (↑x) r o₂
    H₁ : Eq l.size 0 → LE.le r.size 1
    H₂ : LE.le 1 l.size → LE.le 1 r.size → LE.le r.size (HMul.hMul Ordnode.delta l …
    H₃ : Or (LE.le (HMul.hMul 2 l.size) (HAdd.hAdd (HMul.hMul 9 r.size) 5)) (LE.le …
    ⊢ Ordnode.Valid' o₁ (l.balanceL x r) o₂
  -/
  rw [balanceL_eq_balance hl.2 hr.2 H₁ H₂, balance_eq_balance' hl.3 hr.3 hl.2 hr.2]
  /-
    α : Type u_1
    inst✝ : Preorder α
    l : Ordnode α
    x : α
    r : Ordnode α
    o₁ : WithBot α
    o₂ : WithTop α
    hl : Ordnode.Valid' o₁ l ↑x
    hr : Ordnode.Valid' (↑x) r o₂
    H₁ : Eq l.size 0 → LE.le r.size 1
    H₂ : LE.le 1 l.size → LE.le 1 r.size → LE.le r.size (HMul.hMul Ordnode.delta l …
    H₃ : Or (LE.le (HMul.hMul 2 l.size) (HAdd.hAdd (HMul.hMul 9 r.size) 5)) (LE.le …
    ⊢ Ordnode.Valid' o₁ (l.balance' x r) o₂
  -/
  refine hl.balance'_aux hr (Or.inl ?_) H₃
  /-
    α : Type u_1
    inst✝ : Preorder α
    l : Ordnode α
    x : α
    r : Ordnode α
    o₁ : WithBot α
    o₂ : WithTop α
    hl : Ordnode.Valid' o₁ l ↑x
    hr : Ordnode.Valid' (↑x) r o₂
    H₁ : Eq l.size 0 → LE.le r.size 1
    H₂ : LE.le 1 l.size → LE.le 1 r.size → LE.le r.size (HMul.hMul Ordnode.delta l …
    H₃ : Or (LE.le (HMul.hMul 2 l.size) (HAdd.hAdd (HMul.hMul 9 r.size) 5)) (LE.le …
    ⊢ LE.le (HMul.hMul 2 r.size) (HAdd.hAdd (HMul.hMul 9 l.size) 5)
  -/
  rcases Nat.eq_zero_or_pos (size r) with r0 | r0
    /-
      case inl
      α : Type u_1
      inst✝ : Preorder α
      l : Ordnode α
      x : α
      r : Ordnode α
      o₁ : WithBot α
      o₂ : WithTop α
      hl : Ordnode.Valid' o₁ l ↑x
      hr : Ordnode.Valid' (↑x) r o₂
      H₁ : Eq l.size 0 → LE.le r.size 1
      H₂ : LE.le 1 l.size → LE.le 1 r.size → LE.le r.size (HMul.hMul Ordnode.delta l …
      H₃ : Or (LE.le (HMul.hMul 2 l.size) (HAdd.hAdd (HMul.hMul 9 r.size) 5)) (LE.le …
      r0 : Eq r.size 0
      ⊢ LE.le (HMul.hMul 2 r.size) (HAdd.hAdd (HMul.hMul 9 l.size) 5)
    -/
  · rw [r0]; exact Nat.zero_le _
             /-
               🎉 no goals
             -/
  /-
    case inr
    α : Type u_1
    inst✝ : Preorder α
    l : Ordnode α
    x : α
    r : Ordnode α
    o₁ : WithBot α
    o₂ : WithTop α
    hl : Ordnode.Valid' o₁ l ↑x
    hr : Ordnode.Valid' (↑x) r o₂
    H₁ : Eq l.size 0 → LE.le r.size 1
    H₂ : LE.le 1 l.size → LE.le 1 r.size → LE.le r.size (HMul.hMul Ordnode.delta l …
    H₃ : Or (LE.le (HMul.hMul 2 l.size) (HAdd.hAdd (HMul.hMul 9 r.size) 5)) (LE.le …
    r0 : GT.gt r.size 0
    ⊢ LE.le (HMul.hMul 2 r.size) (HAdd.hAdd (HMul.hMul 9 l.size) 5)
  -/
  rcases Nat.eq_zero_or_pos (size l) with l0 | l0
    /-
      case inr.inl
      α : Type u_1
      inst✝ : Preorder α
      l : Ordnode α
      x : α
      r : Ordnode α
      o₁ : WithBot α
      o₂ : WithTop α
      hl : Ordnode.Valid' o₁ l ↑x
      hr : Ordnode.Valid' (↑x) r o₂
      H₁ : Eq l.size 0 → LE.le r.size 1
      H₂ : LE.le 1 l.size → LE.le 1 r.size → LE.le r.size (HMul.hMul Ordnode.delta l …
      H₃ : Or (LE.le (HMul.hMul 2 l.size) (HAdd.hAdd (HMul.hMul 9 r.size) 5)) (LE.le …
      r0 : GT.gt r.size 0
      l0 : Eq l.size 0
      ⊢ LE.le (HMul.hMul 2 r.size) (HAdd.hAdd (HMul.hMul 9 l.size) 5)
    -/
  · rw [l0]; exact le_trans (Nat.mul_le_mul_left _ (H₁ l0)) (by decide)
             /-
               🎉 no goals
             -/
  /-
    case inr.inr
    α : Type u_1
    inst✝ : Preorder α
    l : Ordnode α
    x : α
    r : Ordnode α
    o₁ : WithBot α
    o₂ : WithTop α
    hl : Ordnode.Valid' o₁ l ↑x
    hr : Ordnode.Valid' (↑x) r o₂
    H₁ : Eq l.size 0 → LE.le r.size 1
    H₂ : LE.le 1 l.size → LE.le 1 r.size → LE.le r.size (HMul.hMul Ordnode.delta l …
    H₃ : Or (LE.le (HMul.hMul 2 l.size) (HAdd.hAdd (HMul.hMul 9 r.size) 5)) (LE.le …
    r0 : GT.gt r.size 0
    l0 : GT.gt l.size 0
    ⊢ LE.le (HMul.hMul 2 r.size) (HAdd.hAdd (HMul.hMul 9 l.size) 5)
  -/
  replace H₂ : _ ≤ 3 * _ := H₂ l0 r0; omega
                                      /-
                                        🎉 no goals
                                      -/


theorem Valid'.balanceL {l} {x : α} {r o₁ o₂} (hl : Valid' o₁ l x) (hr : Valid' x r o₂)
    (H : (∃ l', Raised l' (size l) ∧ BalancedSz l' (size r)) ∨
        ∃ r', Raised (size r) r' ∧ BalancedSz (size l) r') :
    Valid' o₁ (@balanceL α l x r) o₂ := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    l : Ordnode α
    x : α
    r : Ordnode α
    o₁ : WithBot α
    o₂ : WithTop α
    hl : Ordnode.Valid' o₁ l ↑x
    hr : Ordnode.Valid' (↑x) r o₂
    H : Or (Exists fun l' => And (Ordnode.Raised l' l.size) (Ordnode.BalancedSz l' …
    ⊢ Ordnode.Valid' o₁ (l.balanceL x r) o₂
  -/
  rw [balanceL_eq_balance' hl.3 hr.3 hl.2 hr.2 H]
  /-
    α : Type u_1
    inst✝ : Preorder α
    l : Ordnode α
    x : α
    r : Ordnode α
    o₁ : WithBot α
    o₂ : WithTop α
    hl : Ordnode.Valid' o₁ l ↑x
    hr : Ordnode.Valid' (↑x) r o₂
    H : Or (Exists fun l' => And (Ordnode.Raised l' l.size) (Ordnode.BalancedSz l' …
    ⊢ Ordnode.Valid' o₁ (l.balance' x r) o₂
  -/
  refine hl.balance' hr ?_
  /-
    α : Type u_1
    inst✝ : Preorder α
    l : Ordnode α
    x : α
    r : Ordnode α
    o₁ : WithBot α
    o₂ : WithTop α
    hl : Ordnode.Valid' o₁ l ↑x
    hr : Ordnode.Valid' (↑x) r o₂
    H : Or (Exists fun l' => And (Ordnode.Raised l' l.size) (Ordnode.BalancedSz l' …
    ⊢ Exists fun l' => Exists fun r' => And (Ordnode.BalancedSz l' r') (Or (And (L …
  -/
  rcases H with (⟨l', e, H⟩ | ⟨r', e, H⟩)
    /-
      case inl.intro.intro
      α : Type u_1
      inst✝ : Preorder α
      l : Ordnode α
      x : α
      r : Ordnode α
      o₁ : WithBot α
      o₂ : WithTop α
      hl : Ordnode.Valid' o₁ l ↑x
      hr : Ordnode.Valid' (↑x) r o₂
      l' : Nat
      e : Ordnode.Raised l' l.size
      H : Ordnode.BalancedSz l' r.size
      ⊢ Exists fun l' => Exists fun r' => And (Ordnode.BalancedSz l' r') (Or (And (L …
    -/
  · exact ⟨_, _, H, Or.inl ⟨e.dist_le', rfl⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.intro
      α : Type u_1
      inst✝ : Preorder α
      l : Ordnode α
      x : α
      r : Ordnode α
      o₁ : WithBot α
      o₂ : WithTop α
      hl : Ordnode.Valid' o₁ l ↑x
      hr : Ordnode.Valid' (↑x) r o₂
      r' : Nat
      e : Ordnode.Raised r.size r'
      H : Ordnode.BalancedSz l.size r'
      ⊢ Exists fun l' => Exists fun r' => And (Ordnode.BalancedSz l' r') (Or (And (L …
    -/
  · exact ⟨_, _, H, Or.inr ⟨e.dist_le, rfl⟩⟩
    /-
      🎉 no goals
    -/


theorem Valid'.balanceR_aux {l} {x : α} {r o₁ o₂} (hl : Valid' o₁ l x) (hr : Valid' x r o₂)
    (H₁ : size r = 0 → size l ≤ 1) (H₂ : 1 ≤ size r → 1 ≤ size l → size l ≤ delta * size r)
    (H₃ : 2 * @size α r ≤ 9 * size l + 5 ∨ size r ≤ 3) : Valid' o₁ (@balanceR α l x r) o₂ := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    l : Ordnode α
    x : α
    r : Ordnode α
    o₁ : WithBot α
    o₂ : WithTop α
    hl : Ordnode.Valid' o₁ l ↑x
    hr : Ordnode.Valid' (↑x) r o₂
    H₁ : Eq r.size 0 → LE.le l.size 1
    H₂ : LE.le 1 r.size → LE.le 1 l.size → LE.le l.size (HMul.hMul Ordnode.delta r …
    H₃ : Or (LE.le (HMul.hMul 2 r.size) (HAdd.hAdd (HMul.hMul 9 l.size) 5)) (LE.le …
    ⊢ Ordnode.Valid' o₁ (l.balanceR x r) o₂
  -/
  rw [Valid'.dual_iff, dual_balanceR]
  /-
    α : Type u_1
    inst✝ : Preorder α
    l : Ordnode α
    x : α
    r : Ordnode α
    o₁ : WithBot α
    o₂ : WithTop α
    hl : Ordnode.Valid' o₁ l ↑x
    hr : Ordnode.Valid' (↑x) r o₂
    H₁ : Eq r.size 0 → LE.le l.size 1
    H₂ : LE.le 1 r.size → LE.le 1 l.size → LE.le l.size (HMul.hMul Ordnode.delta r …
    H₃ : Or (LE.le (HMul.hMul 2 r.size) (HAdd.hAdd (HMul.hMul 9 l.size) 5)) (LE.le …
    ⊢ Ordnode.Valid' o₂ (r.dual.balanceL x l.dual) o₁
  -/
  have := hr.dual.balanceL_aux hl.dual
  /-
    α : Type u_1
    inst✝ : Preorder α
    l : Ordnode α
    x : α
    r : Ordnode α
    o₁ : WithBot α
    o₂ : WithTop α
    hl : Ordnode.Valid' o₁ l ↑x
    hr : Ordnode.Valid' (↑x) r o₂
    H₁ : Eq r.size 0 → LE.le l.size 1
    H₂ : LE.le 1 r.size → LE.le 1 l.size → LE.le l.size (HMul.hMul Ordnode.delta r …
    H₃ : Or (LE.le (HMul.hMul 2 r.size) (HAdd.hAdd (HMul.hMul 9 l.size) 5)) (LE.le …
    this : (Eq r.dual.size 0 → LE.le l.dual.size 1) → (LE.le 1 r.dual.size → LE.le …
    ⊢ Ordnode.Valid' o₂ (r.dual.balanceL x l.dual) o₁
  -/
  rw [size_dual, size_dual] at this
  /-
    α : Type u_1
    inst✝ : Preorder α
    l : Ordnode α
    x : α
    r : Ordnode α
    o₁ : WithBot α
    o₂ : WithTop α
    hl : Ordnode.Valid' o₁ l ↑x
    hr : Ordnode.Valid' (↑x) r o₂
    H₁ : Eq r.size 0 → LE.le l.size 1
    H₂ : LE.le 1 r.size → LE.le 1 l.size → LE.le l.size (HMul.hMul Ordnode.delta r …
    H₃ : Or (LE.le (HMul.hMul 2 r.size) (HAdd.hAdd (HMul.hMul 9 l.size) 5)) (LE.le …
    this : (Eq r.size 0 → LE.le l.size 1) → (LE.le 1 r.size → LE.le 1 l.size → LE. …
    ⊢ Ordnode.Valid' o₂ (r.dual.balanceL x l.dual) o₁
  -/
  exact this H₁ H₂ H₃
  /-
    🎉 no goals
  -/


theorem Valid'.balanceR {l} {x : α} {r o₁ o₂} (hl : Valid' o₁ l x) (hr : Valid' x r o₂)
    (H : (∃ l', Raised (size l) l' ∧ BalancedSz l' (size r)) ∨
        ∃ r', Raised r' (size r) ∧ BalancedSz (size l) r') :
    Valid' o₁ (@balanceR α l x r) o₂ := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    l : Ordnode α
    x : α
    r : Ordnode α
    o₁ : WithBot α
    o₂ : WithTop α
    hl : Ordnode.Valid' o₁ l ↑x
    hr : Ordnode.Valid' (↑x) r o₂
    H : Or (Exists fun l' => And (Ordnode.Raised l.size l') (Ordnode.BalancedSz l' …
    ⊢ Ordnode.Valid' o₁ (l.balanceR x r) o₂
  -/
  rw [Valid'.dual_iff, dual_balanceR]; exact hr.dual.balanceL hl.dual (balance_sz_dual H)
                                       /-
                                         🎉 no goals
                                       -/


theorem Valid'.eraseMax_aux {s l x r o₁ o₂} (H : Valid' o₁ (.node s l x r) o₂) :
    Valid' o₁ (@eraseMax α (.node' l x r)) ↑(findMax' x r) ∧
      size (.node' l x r) = size (eraseMax (.node' l x r)) + 1 := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    s : Nat
    l : Ordnode α
    x : α
    r : Ordnode α
    o₁ : WithBot α
    o₂ : WithTop α
    H : Ordnode.Valid' o₁ (Ordnode.node s l x r) o₂
    ⊢ And (Ordnode.Valid' o₁ (l.node' x r).eraseMax ↑(Ordnode.findMax' x r)) (Eq ( …
  -/
  have := H.2.eq_node'; rw [this] at H; clear this
  /-
    α : Type u_1
    inst✝ : Preorder α
    s : Nat
    l : Ordnode α
    x : α
    r : Ordnode α
    o₁ : WithBot α
    o₂ : WithTop α
    H : Ordnode.Valid' o₁ (l.node' x r) o₂
    ⊢ And (Ordnode.Valid' o₁ (l.node' x r).eraseMax ↑(Ordnode.findMax' x r)) (Eq ( …
  -/
  induction' r with rs rl rx rr _ IHrr generalizing l x o₁
    /-
      case nil
      α : Type u_1
      inst✝ : Preorder α
      s : Nat
      o₂ : WithTop α
      l : Ordnode α
      x : α
      o₁ : WithBot α
      H : Ordnode.Valid' o₁ (l.node' x Ordnode.nil) o₂
      ⊢ And (Ordnode.Valid' o₁ (l.node' x Ordnode.nil).eraseMax ↑(Ordnode.findMax' x …
    -/
  · exact ⟨H.left, rfl⟩
    /-
      🎉 no goals
    -/
  /-
    case node
    α : Type u_1
    inst✝ : Preorder α
    s : Nat
    o₂ : WithTop α
    rs : Nat
    rl : Ordnode α
    rx : α
    rr : Ordnode α
    l_ih✝ : ∀ {l : Ordnode α} {x : α} {o₁ : WithBot α}, Ordnode.Valid' o₁ (l.node' …
    IHrr : ∀ {l : Ordnode α} {x : α} {o₁ : WithBot α}, Ordnode.Valid' o₁ (l.node'  …
    l : Ordnode α
    x : α
    o₁ : WithBot α
    H : Ordnode.Valid' o₁ (l.node' x (Ordnode.node rs rl rx rr)) o₂
    ⊢ And (Ordnode.Valid' o₁ (l.node' x (Ordnode.node rs rl rx rr)).eraseMax ↑(Ord …
  -/
  have := H.2.2.2.eq_node'; rw [this] at H ⊢
  /-
    case node
    α : Type u_1
    inst✝ : Preorder α
    s : Nat
    o₂ : WithTop α
    rs : Nat
    rl : Ordnode α
    rx : α
    rr : Ordnode α
    l_ih✝ : ∀ {l : Ordnode α} {x : α} {o₁ : WithBot α}, Ordnode.Valid' o₁ (l.node' …
    IHrr : ∀ {l : Ordnode α} {x : α} {o₁ : WithBot α}, Ordnode.Valid' o₁ (l.node'  …
    l : Ordnode α
    x : α
    o₁ : WithBot α
    H : Ordnode.Valid' o₁ (l.node' x (rl.node' rx rr)) o₂
    this : Eq (Ordnode.node rs rl rx rr) (rl.node' rx rr)
    ⊢ And (Ordnode.Valid' o₁ (l.node' x (rl.node' rx rr)).eraseMax ↑(Ordnode.findM …
  -/
  rcases IHrr H.right with ⟨h, e⟩
  /-
    case node.intro
    α : Type u_1
    inst✝ : Preorder α
    s : Nat
    o₂ : WithTop α
    rs : Nat
    rl : Ordnode α
    rx : α
    rr : Ordnode α
    l_ih✝ : ∀ {l : Ordnode α} {x : α} {o₁ : WithBot α}, Ordnode.Valid' o₁ (l.node' …
    IHrr : ∀ {l : Ordnode α} {x : α} {o₁ : WithBot α}, Ordnode.Valid' o₁ (l.node'  …
    l : Ordnode α
    x : α
    o₁ : WithBot α
    H : Ordnode.Valid' o₁ (l.node' x (rl.node' rx rr)) o₂
    this : Eq (Ordnode.node rs rl rx rr) (rl.node' rx rr)
    h : Ordnode.Valid' (↑x) (rl.node' rx rr).eraseMax ↑(Ordnode.findMax' rx rr)
    e : Eq (rl.node' rx rr).size (HAdd.hAdd (rl.node' rx rr).eraseMax.size 1)
    ⊢ And (Ordnode.Valid' o₁ (l.node' x (rl.node' rx rr)).eraseMax ↑(Ordnode.findM …
  -/
  refine ⟨Valid'.balanceL H.left h (Or.inr ⟨_, Or.inr e, H.3.1⟩), ?_⟩
  /-
    case node.intro
    α : Type u_1
    inst✝ : Preorder α
    s : Nat
    o₂ : WithTop α
    rs : Nat
    rl : Ordnode α
    rx : α
    rr : Ordnode α
    l_ih✝ : ∀ {l : Ordnode α} {x : α} {o₁ : WithBot α}, Ordnode.Valid' o₁ (l.node' …
    IHrr : ∀ {l : Ordnode α} {x : α} {o₁ : WithBot α}, Ordnode.Valid' o₁ (l.node'  …
    l : Ordnode α
    x : α
    o₁ : WithBot α
    H : Ordnode.Valid' o₁ (l.node' x (rl.node' rx rr)) o₂
    this : Eq (Ordnode.node rs rl rx rr) (rl.node' rx rr)
    h : Ordnode.Valid' (↑x) (rl.node' rx rr).eraseMax ↑(Ordnode.findMax' rx rr)
    e : Eq (rl.node' rx rr).size (HAdd.hAdd (rl.node' rx rr).eraseMax.size 1)
    ⊢ Eq (l.node' x (rl.node' rx rr)).size (HAdd.hAdd (l.node' x (rl.node' rx rr)) …
  -/
  rw [eraseMax, size_balanceL H.3.2.1 h.3 H.2.2.1 h.2 (Or.inr ⟨_, Or.inr e, H.3.1⟩)]
  /-
    case node.intro
    α : Type u_1
    inst✝ : Preorder α
    s : Nat
    o₂ : WithTop α
    rs : Nat
    rl : Ordnode α
    rx : α
    rr : Ordnode α
    l_ih✝ : ∀ {l : Ordnode α} {x : α} {o₁ : WithBot α}, Ordnode.Valid' o₁ (l.node' …
    IHrr : ∀ {l : Ordnode α} {x : α} {o₁ : WithBot α}, Ordnode.Valid' o₁ (l.node'  …
    l : Ordnode α
    x : α
    o₁ : WithBot α
    H : Ordnode.Valid' o₁ (l.node' x (rl.node' rx rr)) o₂
    this : Eq (Ordnode.node rs rl rx rr) (rl.node' rx rr)
    h : Ordnode.Valid' (↑x) (rl.node' rx rr).eraseMax ↑(Ordnode.findMax' rx rr)
    e : Eq (rl.node' rx rr).size (HAdd.hAdd (rl.node' rx rr).eraseMax.size 1)
    ⊢ Eq (l.node' x (rl.node' rx rr)).size (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd l.size …
  -/
  rw [size_node, e]; rfl
                     /-
                       🎉 no goals
                     -/


theorem Valid'.eraseMin_aux {s l} {x : α} {r o₁ o₂} (H : Valid' o₁ (.node s l x r) o₂) :
    Valid' ↑(findMin' l x) (@eraseMin α (.node' l x r)) o₂ ∧
      size (.node' l x r) = size (eraseMin (.node' l x r)) + 1 := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    s : Nat
    l : Ordnode α
    x : α
    r : Ordnode α
    o₁ : WithBot α
    o₂ : WithTop α
    H : Ordnode.Valid' o₁ (Ordnode.node s l x r) o₂
    ⊢ And (Ordnode.Valid' (↑(l.findMin' x)) (l.node' x r).eraseMin o₂) (Eq (l.node …
  -/
  have := H.dual.eraseMax_aux
  rwa [← dual_node', size_dual, ← dual_eraseMin, size_dual, ← Valid'.dual_iff, findMax'_dual]
    at this


theorem eraseMin.valid : ∀ {t}, @Valid α _ t → Valid (eraseMin t)
  | nil, _ => valid_nil
                          /-
                            α : Type u_1
                            inst✝ : Preorder α
                            size✝ : Nat
                            l : Ordnode α
                            x : α
                            r : Ordnode α
                            h : (Ordnode.node size✝ l x r).Valid
                            ⊢ (Ordnode.node size✝ l x r).eraseMin.Valid
                          -/
  | node _ l x r, h => by rw [h.2.eq_node']; exact h.eraseMin_aux.1.valid
                                             /-
                                               🎉 no goals
                                             -/


theorem eraseMax.valid {t} (h : @Valid α _ t) : Valid (eraseMax t) := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    t : Ordnode α
    h : t.Valid
    ⊢ t.eraseMax.Valid
  -/
  rw [Valid.dual_iff, dual_eraseMax]; exact eraseMin.valid h.dual
                                      /-
                                        🎉 no goals
                                      -/


theorem Valid'.glue_aux {l r o₁ o₂} (hl : Valid' o₁ l o₂) (hr : Valid' o₁ r o₂)
    (sep : l.All fun x => r.All fun y => x < y) (bal : BalancedSz (size l) (size r)) :
    Valid' o₁ (@glue α l r) o₂ ∧ size (glue l r) = size l + size r := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    l r : Ordnode α
    o₁ : WithBot α
    o₂ : WithTop α
    hl : Ordnode.Valid' o₁ l o₂
    hr : Ordnode.Valid' o₁ r o₂
    sep : Ordnode.All (fun x => Ordnode.All (fun y => LT.lt x y) r) l
    bal : Ordnode.BalancedSz l.size r.size
    ⊢ And (Ordnode.Valid' o₁ (l.glue r) o₂) (Eq (l.glue r).size (HAdd.hAdd l.size  …
  -/
  cases' l with ls ll lx lr; · exact ⟨hr, (zero_add _).symm⟩
                               /-
                                 🎉 no goals
                               -/
  /-
    case node
    α : Type u_1
    inst✝ : Preorder α
    r : Ordnode α
    o₁ : WithBot α
    o₂ : WithTop α
    hr : Ordnode.Valid' o₁ r o₂
    ls : Nat
    ll : Ordnode α
    lx : α
    lr : Ordnode α
    hl : Ordnode.Valid' o₁ (Ordnode.node ls ll lx lr) o₂
    sep : Ordnode.All (fun x => Ordnode.All (fun y => LT.lt x y) r) (Ordnode.node  …
    bal : Ordnode.BalancedSz (Ordnode.node ls ll lx lr).size r.size
    ⊢ And (Ordnode.Valid' o₁ ((Ordnode.node ls ll lx lr).glue r) o₂) (Eq ((Ordnode …
  -/
  cases' r with rs rl rx rr; · exact ⟨hl, rfl⟩
                               /-
                                 🎉 no goals
                               -/
  /-
    case node.node
    α : Type u_1
    inst✝ : Preorder α
    o₁ : WithBot α
    o₂ : WithTop α
    ls : Nat
    ll : Ordnode α
    lx : α
    lr : Ordnode α
    hl : Ordnode.Valid' o₁ (Ordnode.node ls ll lx lr) o₂
    rs : Nat
    rl : Ordnode α
    rx : α
    rr : Ordnode α
    hr : Ordnode.Valid' o₁ (Ordnode.node rs rl rx rr) o₂
    sep : Ordnode.All (fun x => Ordnode.All (fun y => LT.lt x y) (Ordnode.node rs  …
    bal : Ordnode.BalancedSz (Ordnode.node ls ll lx lr).size (Ordnode.node rs rl r …
    ⊢ And (Ordnode.Valid' o₁ ((Ordnode.node ls ll lx lr).glue (Ordnode.node rs rl  …
  -/
  dsimp [glue]; split_ifs
    /-
      case pos
      α : Type u_1
      inst✝ : Preorder α
      o₁ : WithBot α
      o₂ : WithTop α
      ls : Nat
      ll : Ordnode α
      lx : α
      lr : Ordnode α
      hl : Ordnode.Valid' o₁ (Ordnode.node ls ll lx lr) o₂
      rs : Nat
      rl : Ordnode α
      rx : α
      rr : Ordnode α
      hr : Ordnode.Valid' o₁ (Ordnode.node rs rl rx rr) o₂
      sep : Ordnode.All (fun x => Ordnode.All (fun y => LT.lt x y) (Ordnode.node rs  …
      bal : Ordnode.BalancedSz (Ordnode.node ls ll lx lr).size (Ordnode.node rs rl r …
      h✝ : GT.gt ls rs
      ⊢ And (Ordnode.Valid' o₁ ((ll.splitMax' lx lr).1.balanceR (ll.splitMax' lx lr) …
    -/
  · rw [splitMax_eq]
      /-
        case pos
        α : Type u_1
        inst✝ : Preorder α
        o₁ : WithBot α
        o₂ : WithTop α
        ls : Nat
        ll : Ordnode α
        lx : α
        lr : Ordnode α
        hl : Ordnode.Valid' o₁ (Ordnode.node ls ll lx lr) o₂
        rs : Nat
        rl : Ordnode α
        rx : α
        rr : Ordnode α
        hr : Ordnode.Valid' o₁ (Ordnode.node rs rl rx rr) o₂
        sep : Ordnode.All (fun x => Ordnode.All (fun y => LT.lt x y) (Ordnode.node rs  …
        bal : Ordnode.BalancedSz (Ordnode.node ls ll lx lr).size (Ordnode.node rs rl r …
        h✝ : GT.gt ls rs
        ⊢ And (Ordnode.Valid' o₁ ({ fst := (Ordnode.node ?pos.s✝ ll lx lr).eraseMax, s …
      -/
    · cases' Valid'.eraseMax_aux hl with v e
      suffices H : _ by
        refine ⟨Valid'.balanceR v (hr.of_gt ?_ ?_) H, ?_⟩
        · refine findMax'_all (P := fun a : α => Bounded nil (a : WithTop α) o₂)
            lx lr hl.1.2.to_nil (sep.2.2.imp ?_)
          exact fun x h => hr.1.2.to_nil.mono_left (le_of_lt h.2.1)
        · exact @findMax'_all _ (fun a => All (· > a) (.node rs rl rx rr)) lx lr sep.2.1 sep.2.2
        · rw [size_balanceR v.3 hr.3 v.2 hr.2 H, add_right_comm, ← e, hl.2.1]; rfl
      /-
        case pos.intro
        α : Type u_1
        inst✝ : Preorder α
        o₁ : WithBot α
        o₂ : WithTop α
        ls : Nat
        ll : Ordnode α
        lx : α
        lr : Ordnode α
        hl : Ordnode.Valid' o₁ (Ordnode.node ls ll lx lr) o₂
        rs : Nat
        rl : Ordnode α
        rx : α
        rr : Ordnode α
        hr : Ordnode.Valid' o₁ (Ordnode.node rs rl rx rr) o₂
        sep : Ordnode.All (fun x => Ordnode.All (fun y => LT.lt x y) (Ordnode.node rs  …
        bal : Ordnode.BalancedSz (Ordnode.node ls ll lx lr).size (Ordnode.node rs rl r …
        h✝ : GT.gt ls rs
        v : Ordnode.Valid' o₁ (ll.node' lx lr).eraseMax ↑(Ordnode.findMax' lx lr)
        e : Eq (ll.node' lx lr).size (HAdd.hAdd (ll.node' lx lr).eraseMax.size 1)
        ⊢ Or (Exists fun l' => And (Ordnode.Raised { fst := (Ordnode.node (HAdd.hAdd ( …
      -/
      refine Or.inl ⟨_, Or.inr e, ?_⟩
      /-
        case pos.intro
        α : Type u_1
        inst✝ : Preorder α
        o₁ : WithBot α
        o₂ : WithTop α
        ls : Nat
        ll : Ordnode α
        lx : α
        lr : Ordnode α
        hl : Ordnode.Valid' o₁ (Ordnode.node ls ll lx lr) o₂
        rs : Nat
        rl : Ordnode α
        rx : α
        rr : Ordnode α
        hr : Ordnode.Valid' o₁ (Ordnode.node rs rl rx rr) o₂
        sep : Ordnode.All (fun x => Ordnode.All (fun y => LT.lt x y) (Ordnode.node rs  …
        bal : Ordnode.BalancedSz (Ordnode.node ls ll lx lr).size (Ordnode.node rs rl r …
        h✝ : GT.gt ls rs
        v : Ordnode.Valid' o₁ (ll.node' lx lr).eraseMax ↑(Ordnode.findMax' lx lr)
        e : Eq (ll.node' lx lr).size (HAdd.hAdd (ll.node' lx lr).eraseMax.size 1)
        ⊢ Ordnode.BalancedSz (ll.node' lx lr).size (Ordnode.node rs rl rx rr).size
      -/
      rwa [hl.2.eq_node'] at bal
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_1
      inst✝ : Preorder α
      o₁ : WithBot α
      o₂ : WithTop α
      ls : Nat
      ll : Ordnode α
      lx : α
      lr : Ordnode α
      hl : Ordnode.Valid' o₁ (Ordnode.node ls ll lx lr) o₂
      rs : Nat
      rl : Ordnode α
      rx : α
      rr : Ordnode α
      hr : Ordnode.Valid' o₁ (Ordnode.node rs rl rx rr) o₂
      sep : Ordnode.All (fun x => Ordnode.All (fun y => LT.lt x y) (Ordnode.node rs  …
      bal : Ordnode.BalancedSz (Ordnode.node ls ll lx lr).size (Ordnode.node rs rl r …
      h✝ : Not (GT.gt ls rs)
      ⊢ And (Ordnode.Valid' o₁ ((Ordnode.node ls ll lx lr).balanceL (rl.splitMin' rx …
    -/
  · rw [splitMin_eq]
      /-
        case neg
        α : Type u_1
        inst✝ : Preorder α
        o₁ : WithBot α
        o₂ : WithTop α
        ls : Nat
        ll : Ordnode α
        lx : α
        lr : Ordnode α
        hl : Ordnode.Valid' o₁ (Ordnode.node ls ll lx lr) o₂
        rs : Nat
        rl : Ordnode α
        rx : α
        rr : Ordnode α
        hr : Ordnode.Valid' o₁ (Ordnode.node rs rl rx rr) o₂
        sep : Ordnode.All (fun x => Ordnode.All (fun y => LT.lt x y) (Ordnode.node rs  …
        bal : Ordnode.BalancedSz (Ordnode.node ls ll lx lr).size (Ordnode.node rs rl r …
        h✝ : Not (GT.gt ls rs)
        ⊢ And (Ordnode.Valid' o₁ ((Ordnode.node ls ll lx lr).balanceL { fst := rl.find …
      -/
    · cases' Valid'.eraseMin_aux hr with v e
      suffices H : _ by
        refine ⟨Valid'.balanceL (hl.of_lt ?_ ?_) v H, ?_⟩
        · refine @findMin'_all (P := fun a : α => Bounded nil o₁ (a : WithBot α))
            _ rl rx (sep.2.1.1.imp ?_) hr.1.1.to_nil
          exact fun y h => hl.1.1.to_nil.mono_right (le_of_lt h)
        · exact
            @findMin'_all _ (fun a => All (· < a) (.node ls ll lx lr)) rl rx
              (all_iff_forall.2 fun x hx => sep.imp fun y hy => all_iff_forall.1 hy.1 _ hx)
              (sep.imp fun y hy => hy.2.1)
        · rw [size_balanceL hl.3 v.3 hl.2 v.2 H, add_assoc, ← e, hr.2.1]; rfl
      /-
        case neg.intro
        α : Type u_1
        inst✝ : Preorder α
        o₁ : WithBot α
        o₂ : WithTop α
        ls : Nat
        ll : Ordnode α
        lx : α
        lr : Ordnode α
        hl : Ordnode.Valid' o₁ (Ordnode.node ls ll lx lr) o₂
        rs : Nat
        rl : Ordnode α
        rx : α
        rr : Ordnode α
        hr : Ordnode.Valid' o₁ (Ordnode.node rs rl rx rr) o₂
        sep : Ordnode.All (fun x => Ordnode.All (fun y => LT.lt x y) (Ordnode.node rs  …
        bal : Ordnode.BalancedSz (Ordnode.node ls ll lx lr).size (Ordnode.node rs rl r …
        h✝ : Not (GT.gt ls rs)
        v : Ordnode.Valid' (↑(rl.findMin' rx)) (rl.node' rx rr).eraseMin o₂
        e : Eq (rl.node' rx rr).size (HAdd.hAdd (rl.node' rx rr).eraseMin.size 1)
        ⊢ Or (Exists fun l' => And (Ordnode.Raised l' (Ordnode.node ls ll lx lr).size) …
      -/
      refine Or.inr ⟨_, Or.inr e, ?_⟩
      /-
        case neg.intro
        α : Type u_1
        inst✝ : Preorder α
        o₁ : WithBot α
        o₂ : WithTop α
        ls : Nat
        ll : Ordnode α
        lx : α
        lr : Ordnode α
        hl : Ordnode.Valid' o₁ (Ordnode.node ls ll lx lr) o₂
        rs : Nat
        rl : Ordnode α
        rx : α
        rr : Ordnode α
        hr : Ordnode.Valid' o₁ (Ordnode.node rs rl rx rr) o₂
        sep : Ordnode.All (fun x => Ordnode.All (fun y => LT.lt x y) (Ordnode.node rs  …
        bal : Ordnode.BalancedSz (Ordnode.node ls ll lx lr).size (Ordnode.node rs rl r …
        h✝ : Not (GT.gt ls rs)
        v : Ordnode.Valid' (↑(rl.findMin' rx)) (rl.node' rx rr).eraseMin o₂
        e : Eq (rl.node' rx rr).size (HAdd.hAdd (rl.node' rx rr).eraseMin.size 1)
        ⊢ Ordnode.BalancedSz (Ordnode.node ls ll lx lr).size (rl.node' rx rr).size
      -/
      rwa [hr.2.eq_node'] at bal
      /-
        🎉 no goals
      -/


theorem Valid'.glue {l} {x : α} {r o₁ o₂} (hl : Valid' o₁ l x) (hr : Valid' x r o₂) :
    BalancedSz (size l) (size r) →
      Valid' o₁ (@glue α l r) o₂ ∧ size (@glue α l r) = size l + size r :=
  Valid'.glue_aux (hl.trans_right hr.1) (hr.trans_left hl.1) (hl.1.to_sep hr.1)


theorem Valid'.merge_lemma {a b c : ℕ} (h₁ : 3 * a < b + c + 1) (h₂ : b ≤ 3 * c) :
                                  /-
                                    a b c : Nat
                                    h₁ : LT.lt (HMul.hMul 3 a) (HAdd.hAdd (HAdd.hAdd b c) 1)
                                    h₂ : LE.le b (HMul.hMul 3 c)
                                    ⊢ LE.le (HMul.hMul 2 (HAdd.hAdd a b)) (HAdd.hAdd (HMul.hMul 9 c) 5)
                                  -/
    2 * (a + b) ≤ 9 * c + 5 := by omega
                                  /-
                                    🎉 no goals
                                  -/


theorem Valid'.merge_aux₁ {o₁ o₂ ls ll lx lr rs rl rx rr t}
    (hl : Valid' o₁ (@Ordnode.node α ls ll lx lr) o₂) (hr : Valid' o₁ (.node rs rl rx rr) o₂)
    (h : delta * ls < rs) (v : Valid' o₁ t rx) (e : size t = ls + size rl) :
    Valid' o₁ (.balanceL t rx rr) o₂ ∧ size (.balanceL t rx rr) = ls + rs := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    o₁ : WithBot α
    o₂ : WithTop α
    ls : Nat
    ll : Ordnode α
    lx : α
    lr : Ordnode α
    rs : Nat
    rl : Ordnode α
    rx : α
    rr t : Ordnode α
    hl : Ordnode.Valid' o₁ (Ordnode.node ls ll lx lr) o₂
    hr : Ordnode.Valid' o₁ (Ordnode.node rs rl rx rr) o₂
    h : LT.lt (HMul.hMul Ordnode.delta ls) rs
    v : Ordnode.Valid' o₁ t ↑rx
    e : Eq t.size (HAdd.hAdd ls rl.size)
    ⊢ And (Ordnode.Valid' o₁ (t.balanceL rx rr) o₂) (Eq (t.balanceL rx rr).size (H …
  -/
  rw [hl.2.1] at e
  /-
    α : Type u_1
    inst✝ : Preorder α
    o₁ : WithBot α
    o₂ : WithTop α
    ls : Nat
    ll : Ordnode α
    lx : α
    lr : Ordnode α
    rs : Nat
    rl : Ordnode α
    rx : α
    rr t : Ordnode α
    hl : Ordnode.Valid' o₁ (Ordnode.node ls ll lx lr) o₂
    hr : Ordnode.Valid' o₁ (Ordnode.node rs rl rx rr) o₂
    h : LT.lt (HMul.hMul Ordnode.delta ls) rs
    v : Ordnode.Valid' o₁ t ↑rx
    e : Eq t.size (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ll.size lr.size) 1) rl.size)
    ⊢ And (Ordnode.Valid' o₁ (t.balanceL rx rr) o₂) (Eq (t.balanceL rx rr).size (H …
  -/
  rw [hl.2.1, hr.2.1, delta] at h
  /-
    α : Type u_1
    inst✝ : Preorder α
    o₁ : WithBot α
    o₂ : WithTop α
    ls : Nat
    ll : Ordnode α
    lx : α
    lr : Ordnode α
    rs : Nat
    rl : Ordnode α
    rx : α
    rr t : Ordnode α
    hl : Ordnode.Valid' o₁ (Ordnode.node ls ll lx lr) o₂
    hr : Ordnode.Valid' o₁ (Ordnode.node rs rl rx rr) o₂
    h : LT.lt (HMul.hMul 3 (HAdd.hAdd (HAdd.hAdd ll.size lr.size) 1)) (HAdd.hAdd ( …
    v : Ordnode.Valid' o₁ t ↑rx
    e : Eq t.size (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ll.size lr.size) 1) rl.size)
    ⊢ And (Ordnode.Valid' o₁ (t.balanceL rx rr) o₂) (Eq (t.balanceL rx rr).size (H …
  -/
  rcases hr.3.1 with (H | ⟨hr₁, hr₂⟩); · omega
                                         /-
                                           🎉 no goals
                                         -/
  suffices H₂ : _ by
    suffices H₁ : _ by
      refine ⟨Valid'.balanceL_aux v hr.right H₁ H₂ ?_, ?_⟩
      · rw [e]; exact Or.inl (Valid'.merge_lemma h hr₁)
      · rw [balanceL_eq_balance v.2 hr.2.2.2 H₁ H₂, balance_eq_balance' v.3 hr.3.2.2 v.2 hr.2.2.2,
          size_balance' v.2 hr.2.2.2, e, hl.2.1, hr.2.1]
        abel
    · rw [e, add_right_comm]; rintro ⟨⟩
  /-
    case inr.intro
    α : Type u_1
    inst✝ : Preorder α
    o₁ : WithBot α
    o₂ : WithTop α
    ls : Nat
    ll : Ordnode α
    lx : α
    lr : Ordnode α
    rs : Nat
    rl : Ordnode α
    rx : α
    rr t : Ordnode α
    hl : Ordnode.Valid' o₁ (Ordnode.node ls ll lx lr) o₂
    hr : Ordnode.Valid' o₁ (Ordnode.node rs rl rx rr) o₂
    h : LT.lt (HMul.hMul 3 (HAdd.hAdd (HAdd.hAdd ll.size lr.size) 1)) (HAdd.hAdd ( …
    v : Ordnode.Valid' o₁ t ↑rx
    e : Eq t.size (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ll.size lr.size) 1) rl.size)
    hr₁ : LE.le rl.size (HMul.hMul Ordnode.delta rr.size)
    hr₂ : LE.le rr.size (HMul.hMul Ordnode.delta rl.size)
    ⊢ LE.le 1 t.size → LE.le 1 rr.size → LE.le rr.size (HMul.hMul Ordnode.delta t. …
  -/
  intro _ _; rw [e]; unfold delta at hr₂ ⊢; omega
                                            /-
                                              🎉 no goals
                                            -/


theorem Valid'.merge_aux {l r o₁ o₂} (hl : Valid' o₁ l o₂) (hr : Valid' o₁ r o₂)
    (sep : l.All fun x => r.All fun y => x < y) :
    Valid' o₁ (@merge α l r) o₂ ∧ size (merge l r) = size l + size r := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    l r : Ordnode α
    o₁ : WithBot α
    o₂ : WithTop α
    hl : Ordnode.Valid' o₁ l o₂
    hr : Ordnode.Valid' o₁ r o₂
    sep : Ordnode.All (fun x => Ordnode.All (fun y => LT.lt x y) r) l
    ⊢ And (Ordnode.Valid' o₁ (l.merge r) o₂) (Eq (l.merge r).size (HAdd.hAdd l.siz …
  -/
  induction' l with ls ll lx lr _ IHlr generalizing o₁ o₂ r
    /-
      case nil
      α : Type u_1
      inst✝ : Preorder α
      r : Ordnode α
      o₁ : WithBot α
      o₂ : WithTop α
      hl : Ordnode.Valid' o₁ Ordnode.nil o₂
      hr : Ordnode.Valid' o₁ r o₂
      sep : Ordnode.All (fun x => Ordnode.All (fun y => LT.lt x y) r) Ordnode.nil
      ⊢ And (Ordnode.Valid' o₁ (Ordnode.nil.merge r) o₂) (Eq (Ordnode.nil.merge r).s …
    -/
  · exact ⟨hr, (zero_add _).symm⟩
    /-
      🎉 no goals
    -/
  /-
    case node
    α : Type u_1
    inst✝ : Preorder α
    ls : Nat
    ll : Ordnode α
    lx : α
    lr : Ordnode α
    l_ih✝ : ∀ {r : Ordnode α} {o₁ : WithBot α} {o₂ : WithTop α}, Ordnode.Valid' o₁ …
    IHlr : ∀ {r : Ordnode α} {o₁ : WithBot α} {o₂ : WithTop α}, Ordnode.Valid' o₁  …
    r : Ordnode α
    o₁ : WithBot α
    o₂ : WithTop α
    hl : Ordnode.Valid' o₁ (Ordnode.node ls ll lx lr) o₂
    hr : Ordnode.Valid' o₁ r o₂
    sep : Ordnode.All (fun x => Ordnode.All (fun y => LT.lt x y) r) (Ordnode.node  …
    ⊢ And (Ordnode.Valid' o₁ ((Ordnode.node ls ll lx lr).merge r) o₂) (Eq ((Ordnod …
  -/
  induction' r with rs rl rx rr IHrl _ generalizing o₁ o₂
    /-
      case node.nil
      α : Type u_1
      inst✝ : Preorder α
      ls : Nat
      ll : Ordnode α
      lx : α
      lr : Ordnode α
      l_ih✝ : ∀ {r : Ordnode α} {o₁ : WithBot α} {o₂ : WithTop α}, Ordnode.Valid' o₁ …
      IHlr : ∀ {r : Ordnode α} {o₁ : WithBot α} {o₂ : WithTop α}, Ordnode.Valid' o₁  …
      o₁ : WithBot α
      o₂ : WithTop α
      hl : Ordnode.Valid' o₁ (Ordnode.node ls ll lx lr) o₂
      hr : Ordnode.Valid' o₁ Ordnode.nil o₂
      sep : Ordnode.All (fun x => Ordnode.All (fun y => LT.lt x y) Ordnode.nil) (Ord …
      ⊢ And (Ordnode.Valid' o₁ ((Ordnode.node ls ll lx lr).merge Ordnode.nil) o₂) (E …
    -/
  · exact ⟨hl, rfl⟩
    /-
      🎉 no goals
    -/
  /-
    case node.node
    α : Type u_1
    inst✝ : Preorder α
    ls : Nat
    ll : Ordnode α
    lx : α
    lr : Ordnode α
    l_ih✝ : ∀ {r : Ordnode α} {o₁ : WithBot α} {o₂ : WithTop α}, Ordnode.Valid' o₁ …
    IHlr : ∀ {r : Ordnode α} {o₁ : WithBot α} {o₂ : WithTop α}, Ordnode.Valid' o₁  …
    rs : Nat
    rl : Ordnode α
    rx : α
    rr : Ordnode α
    IHrl : ∀ {o₁ : WithBot α} {o₂ : WithTop α}, Ordnode.Valid' o₁ (Ordnode.node ls …
    r_ih✝ : ∀ {o₁ : WithBot α} {o₂ : WithTop α}, Ordnode.Valid' o₁ (Ordnode.node l …
    o₁ : WithBot α
    o₂ : WithTop α
    hl : Ordnode.Valid' o₁ (Ordnode.node ls ll lx lr) o₂
    hr : Ordnode.Valid' o₁ (Ordnode.node rs rl rx rr) o₂
    sep : Ordnode.All (fun x => Ordnode.All (fun y => LT.lt x y) (Ordnode.node rs  …
    ⊢ And (Ordnode.Valid' o₁ ((Ordnode.node ls ll lx lr).merge (Ordnode.node rs rl …
  -/
  rw [merge_node]; split_ifs with h h_1
  · cases'
      IHrl (hl.of_lt hr.1.1.to_nil <| sep.imp fun x h => h.2.1) hr.left
        (sep.imp fun x h => h.1) with
      v e
    /-
      case pos.intro
      α : Type u_1
      inst✝ : Preorder α
      ls : Nat
      ll : Ordnode α
      lx : α
      lr : Ordnode α
      l_ih✝ : ∀ {r : Ordnode α} {o₁ : WithBot α} {o₂ : WithTop α}, Ordnode.Valid' o₁ …
      IHlr : ∀ {r : Ordnode α} {o₁ : WithBot α} {o₂ : WithTop α}, Ordnode.Valid' o₁  …
      rs : Nat
      rl : Ordnode α
      rx : α
      rr : Ordnode α
      IHrl : ∀ {o₁ : WithBot α} {o₂ : WithTop α}, Ordnode.Valid' o₁ (Ordnode.node ls …
      r_ih✝ : ∀ {o₁ : WithBot α} {o₂ : WithTop α}, Ordnode.Valid' o₁ (Ordnode.node l …
      o₁ : WithBot α
      o₂ : WithTop α
      hl : Ordnode.Valid' o₁ (Ordnode.node ls ll lx lr) o₂
      hr : Ordnode.Valid' o₁ (Ordnode.node rs rl rx rr) o₂
      sep : Ordnode.All (fun x => Ordnode.All (fun y => LT.lt x y) (Ordnode.node rs  …
      h : LT.lt (HMul.hMul Ordnode.delta ls) rs
      v : Ordnode.Valid' o₁ ((Ordnode.node ls ll lx lr).merge rl) ↑rx
      e : Eq ((Ordnode.node ls ll lx lr).merge rl).size (HAdd.hAdd (Ordnode.node ls  …
      ⊢ And (Ordnode.Valid' o₁ (((Ordnode.node ls ll lx lr).merge rl).balanceL rx rr …
    -/
    exact Valid'.merge_aux₁ hl hr h v e
    /-
      🎉 no goals
    -/
    /-
      case pos
      α : Type u_1
      inst✝ : Preorder α
      ls : Nat
      ll : Ordnode α
      lx : α
      lr : Ordnode α
      l_ih✝ : ∀ {r : Ordnode α} {o₁ : WithBot α} {o₂ : WithTop α}, Ordnode.Valid' o₁ …
      IHlr : ∀ {r : Ordnode α} {o₁ : WithBot α} {o₂ : WithTop α}, Ordnode.Valid' o₁  …
      rs : Nat
      rl : Ordnode α
      rx : α
      rr : Ordnode α
      IHrl : ∀ {o₁ : WithBot α} {o₂ : WithTop α}, Ordnode.Valid' o₁ (Ordnode.node ls …
      r_ih✝ : ∀ {o₁ : WithBot α} {o₂ : WithTop α}, Ordnode.Valid' o₁ (Ordnode.node l …
      o₁ : WithBot α
      o₂ : WithTop α
      hl : Ordnode.Valid' o₁ (Ordnode.node ls ll lx lr) o₂
      hr : Ordnode.Valid' o₁ (Ordnode.node rs rl rx rr) o₂
      sep : Ordnode.All (fun x => Ordnode.All (fun y => LT.lt x y) (Ordnode.node rs  …
      h : Not (LT.lt (HMul.hMul Ordnode.delta ls) rs)
      h_1 : LT.lt (HMul.hMul Ordnode.delta rs) ls
      ⊢ And (Ordnode.Valid' o₁ (ll.balanceR lx (lr.merge (Ordnode.node rs rl rx rr)) …
    -/
  · cases' IHlr hl.right (hr.of_gt hl.1.2.to_nil sep.2.1) sep.2.2 with v e
    /-
      case pos.intro
      α : Type u_1
      inst✝ : Preorder α
      ls : Nat
      ll : Ordnode α
      lx : α
      lr : Ordnode α
      l_ih✝ : ∀ {r : Ordnode α} {o₁ : WithBot α} {o₂ : WithTop α}, Ordnode.Valid' o₁ …
      IHlr : ∀ {r : Ordnode α} {o₁ : WithBot α} {o₂ : WithTop α}, Ordnode.Valid' o₁  …
      rs : Nat
      rl : Ordnode α
      rx : α
      rr : Ordnode α
      IHrl : ∀ {o₁ : WithBot α} {o₂ : WithTop α}, Ordnode.Valid' o₁ (Ordnode.node ls …
      r_ih✝ : ∀ {o₁ : WithBot α} {o₂ : WithTop α}, Ordnode.Valid' o₁ (Ordnode.node l …
      o₁ : WithBot α
      o₂ : WithTop α
      hl : Ordnode.Valid' o₁ (Ordnode.node ls ll lx lr) o₂
      hr : Ordnode.Valid' o₁ (Ordnode.node rs rl rx rr) o₂
      sep : Ordnode.All (fun x => Ordnode.All (fun y => LT.lt x y) (Ordnode.node rs  …
      h : Not (LT.lt (HMul.hMul Ordnode.delta ls) rs)
      h_1 : LT.lt (HMul.hMul Ordnode.delta rs) ls
      v : Ordnode.Valid' (↑lx) (lr.merge (Ordnode.node rs rl rx rr)) o₂
      e : Eq (lr.merge (Ordnode.node rs rl rx rr)).size (HAdd.hAdd lr.size (Ordnode. …
      ⊢ And (Ordnode.Valid' o₁ (ll.balanceR lx (lr.merge (Ordnode.node rs rl rx rr)) …
    -/
    have := Valid'.merge_aux₁ hr.dual hl.dual h_1 v.dual
    rw [size_dual, add_comm, size_dual, ← dual_balanceR, ← Valid'.dual_iff, size_dual,
      add_comm rs] at this
    /-
      case pos.intro
      α : Type u_1
      inst✝ : Preorder α
      ls : Nat
      ll : Ordnode α
      lx : α
      lr : Ordnode α
      l_ih✝ : ∀ {r : Ordnode α} {o₁ : WithBot α} {o₂ : WithTop α}, Ordnode.Valid' o₁ …
      IHlr : ∀ {r : Ordnode α} {o₁ : WithBot α} {o₂ : WithTop α}, Ordnode.Valid' o₁  …
      rs : Nat
      rl : Ordnode α
      rx : α
      rr : Ordnode α
      IHrl : ∀ {o₁ : WithBot α} {o₂ : WithTop α}, Ordnode.Valid' o₁ (Ordnode.node ls …
      r_ih✝ : ∀ {o₁ : WithBot α} {o₂ : WithTop α}, Ordnode.Valid' o₁ (Ordnode.node l …
      o₁ : WithBot α
      o₂ : WithTop α
      hl : Ordnode.Valid' o₁ (Ordnode.node ls ll lx lr) o₂
      hr : Ordnode.Valid' o₁ (Ordnode.node rs rl rx rr) o₂
      sep : Ordnode.All (fun x => Ordnode.All (fun y => LT.lt x y) (Ordnode.node rs  …
      h : Not (LT.lt (HMul.hMul Ordnode.delta ls) rs)
      h_1 : LT.lt (HMul.hMul Ordnode.delta rs) ls
      v : Ordnode.Valid' (↑lx) (lr.merge (Ordnode.node rs rl rx rr)) o₂
      e : Eq (lr.merge (Ordnode.node rs rl rx rr)).size (HAdd.hAdd lr.size (Ordnode. …
      this : Eq (lr.merge (Ordnode.node rs rl rx rr)).size (HAdd.hAdd lr.size rs) →  …
      ⊢ And (Ordnode.Valid' o₁ (ll.balanceR lx (lr.merge (Ordnode.node rs rl rx rr)) …
    -/
    exact this e
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : Preorder α
      ls : Nat
      ll : Ordnode α
      lx : α
      lr : Ordnode α
      l_ih✝ : ∀ {r : Ordnode α} {o₁ : WithBot α} {o₂ : WithTop α}, Ordnode.Valid' o₁ …
      IHlr : ∀ {r : Ordnode α} {o₁ : WithBot α} {o₂ : WithTop α}, Ordnode.Valid' o₁  …
      rs : Nat
      rl : Ordnode α
      rx : α
      rr : Ordnode α
      IHrl : ∀ {o₁ : WithBot α} {o₂ : WithTop α}, Ordnode.Valid' o₁ (Ordnode.node ls …
      r_ih✝ : ∀ {o₁ : WithBot α} {o₂ : WithTop α}, Ordnode.Valid' o₁ (Ordnode.node l …
      o₁ : WithBot α
      o₂ : WithTop α
      hl : Ordnode.Valid' o₁ (Ordnode.node ls ll lx lr) o₂
      hr : Ordnode.Valid' o₁ (Ordnode.node rs rl rx rr) o₂
      sep : Ordnode.All (fun x => Ordnode.All (fun y => LT.lt x y) (Ordnode.node rs  …
      h : Not (LT.lt (HMul.hMul Ordnode.delta ls) rs)
      h_1 : Not (LT.lt (HMul.hMul Ordnode.delta rs) ls)
      ⊢ And (Ordnode.Valid' o₁ ((Ordnode.node ls ll lx lr).glue (Ordnode.node rs rl  …
    -/
  · refine Valid'.glue_aux hl hr sep (Or.inr ⟨not_lt.1 h_1, not_lt.1 h⟩)
    /-
      🎉 no goals
    -/


theorem Valid.merge {l r} (hl : Valid l) (hr : Valid r)
    (sep : l.All fun x => r.All fun y => x < y) : Valid (@merge α l r) :=
  (Valid'.merge_aux hl hr sep).1


theorem insertWith.valid_aux [IsTotal α (· ≤ ·)] [DecidableRel (α := α) (· ≤ ·)] (f : α → α) (x : α)
    (hf : ∀ y, x ≤ y ∧ y ≤ x → x ≤ f y ∧ f y ≤ x) :
    ∀ {t o₁ o₂},
      Valid' o₁ t o₂ →
        Bounded nil o₁ x →
          Bounded nil x o₂ →
            Valid' o₁ (insertWith f x t) o₂ ∧ Raised (size t) (size (insertWith f x t))
  | nil, _, _, _, bl, br => ⟨valid'_singleton bl br, Or.inr rfl⟩
  | node sz l y r, o₁, o₂, h, bl, br => by
    /-
      α : Type u_1
      inst✝² : Preorder α
      inst✝¹ : IsTotal α fun x1 x2 => LE.le x1 x2
      inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
      f : α → α
      x : α
      hf : ∀ (y : α), And (LE.le x y) (LE.le y x) → And (LE.le x (f y)) (LE.le (f y) …
      sz : Nat
      l : Ordnode α
      y : α
      r : Ordnode α
      o₁ : WithBot α
      o₂ : WithTop α
      h : Ordnode.Valid' o₁ (Ordnode.node sz l y r) o₂
      bl : Ordnode.nil.Bounded o₁ ↑x
      br : Ordnode.nil.Bounded (↑x) o₂
      ⊢ And (Ordnode.Valid' o₁ (Ordnode.insertWith f x (Ordnode.node sz l y r)) o₂)  …
    -/
    rw [insertWith, cmpLE]
    /-
      α : Type u_1
      inst✝² : Preorder α
      inst✝¹ : IsTotal α fun x1 x2 => LE.le x1 x2
      inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
      f : α → α
      x : α
      hf : ∀ (y : α), And (LE.le x y) (LE.le y x) → And (LE.le x (f y)) (LE.le (f y) …
      sz : Nat
      l : Ordnode α
      y : α
      r : Ordnode α
      o₁ : WithBot α
      o₂ : WithTop α
      h : Ordnode.Valid' o₁ (Ordnode.node sz l y r) o₂
      bl : Ordnode.nil.Bounded o₁ ↑x
      br : Ordnode.nil.Bounded (↑x) o₂
      ⊢ And (Ordnode.Valid' o₁ (Ordnode.mem.match_1 (fun x => Ordnode α) (ite (LE.le …
    -/
    split_ifs with h_1 h_2 <;> dsimp only
      /-
        case pos
        α : Type u_1
        inst✝² : Preorder α
        inst✝¹ : IsTotal α fun x1 x2 => LE.le x1 x2
        inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
        f : α → α
        x : α
        hf : ∀ (y : α), And (LE.le x y) (LE.le y x) → And (LE.le x (f y)) (LE.le (f y) …
        sz : Nat
        l : Ordnode α
        y : α
        r : Ordnode α
        o₁ : WithBot α
        o₂ : WithTop α
        h : Ordnode.Valid' o₁ (Ordnode.node sz l y r) o₂
        bl : Ordnode.nil.Bounded o₁ ↑x
        br : Ordnode.nil.Bounded (↑x) o₂
        h_1 : LE.le x y
        h_2 : LE.le y x
        ⊢ And (Ordnode.Valid' o₁ (Ordnode.node sz l (f y) r) o₂) (Ordnode.Raised (Ordn …
      -/
    · rcases h with ⟨⟨lx, xr⟩, hs, hb⟩
      /-
        case pos.mk.intro
        α : Type u_1
        inst✝² : Preorder α
        inst✝¹ : IsTotal α fun x1 x2 => LE.le x1 x2
        inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
        f : α → α
        x : α
        hf : ∀ (y : α), And (LE.le x y) (LE.le y x) → And (LE.le x (f y)) (LE.le (f y) …
        sz : Nat
        l : Ordnode α
        y : α
        r : Ordnode α
        o₁ : WithBot α
        o₂ : WithTop α
        bl : Ordnode.nil.Bounded o₁ ↑x
        br : Ordnode.nil.Bounded (↑x) o₂
        h_1 : LE.le x y
        h_2 : LE.le y x
        hs : (Ordnode.node sz l y r).Sized
        hb : (Ordnode.node sz l y r).Balanced
        lx : l.Bounded o₁ ↑y
        xr : r.Bounded (↑y) o₂
        ⊢ And (Ordnode.Valid' o₁ (Ordnode.node sz l (f y) r) o₂) (Ordnode.Raised (Ordn …
      -/
      rcases hf _ ⟨h_1, h_2⟩ with ⟨xf, fx⟩
      refine
        ⟨⟨⟨lx.mono_right (le_trans h_2 xf), xr.mono_left (le_trans fx h_1)⟩, hs, hb⟩, Or.inl rfl⟩
      /-
        case neg
        α : Type u_1
        inst✝² : Preorder α
        inst✝¹ : IsTotal α fun x1 x2 => LE.le x1 x2
        inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
        f : α → α
        x : α
        hf : ∀ (y : α), And (LE.le x y) (LE.le y x) → And (LE.le x (f y)) (LE.le (f y) …
        sz : Nat
        l : Ordnode α
        y : α
        r : Ordnode α
        o₁ : WithBot α
        o₂ : WithTop α
        h : Ordnode.Valid' o₁ (Ordnode.node sz l y r) o₂
        bl : Ordnode.nil.Bounded o₁ ↑x
        br : Ordnode.nil.Bounded (↑x) o₂
        h_1 : LE.le x y
        h_2 : Not (LE.le y x)
        ⊢ And (Ordnode.Valid' o₁ ((Ordnode.insertWith f x l).balanceL y r) o₂) (Ordnod …
      -/
    · rcases insertWith.valid_aux f x hf h.left bl (lt_of_le_not_le h_1 h_2) with ⟨vl, e⟩
      suffices H : _ by
        refine ⟨vl.balanceL h.right H, ?_⟩
        rw [size_balanceL vl.3 h.3.2.2 vl.2 h.2.2.2 H, h.2.size_eq]
        exact (e.add_right _).add_right _
      /-
        case neg.intro
        α : Type u_1
        inst✝² : Preorder α
        inst✝¹ : IsTotal α fun x1 x2 => LE.le x1 x2
        inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
        f : α → α
        x : α
        hf : ∀ (y : α), And (LE.le x y) (LE.le y x) → And (LE.le x (f y)) (LE.le (f y) …
        sz : Nat
        l : Ordnode α
        y : α
        r : Ordnode α
        o₁ : WithBot α
        o₂ : WithTop α
        h : Ordnode.Valid' o₁ (Ordnode.node sz l y r) o₂
        bl : Ordnode.nil.Bounded o₁ ↑x
        br : Ordnode.nil.Bounded (↑x) o₂
        h_1 : LE.le x y
        h_2 : Not (LE.le y x)
        vl : Ordnode.Valid' o₁ (Ordnode.insertWith f x l) ↑y
        e : Ordnode.Raised l.size (Ordnode.insertWith f x l).size
        ⊢ Or (Exists fun l' => And (Ordnode.Raised l' (Ordnode.insertWith f x l).size) …
      -/
      exact Or.inl ⟨_, e, h.3.1⟩
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        inst✝² : Preorder α
        inst✝¹ : IsTotal α fun x1 x2 => LE.le x1 x2
        inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
        f : α → α
        x : α
        hf : ∀ (y : α), And (LE.le x y) (LE.le y x) → And (LE.le x (f y)) (LE.le (f y) …
        sz : Nat
        l : Ordnode α
        y : α
        r : Ordnode α
        o₁ : WithBot α
        o₂ : WithTop α
        h : Ordnode.Valid' o₁ (Ordnode.node sz l y r) o₂
        bl : Ordnode.nil.Bounded o₁ ↑x
        br : Ordnode.nil.Bounded (↑x) o₂
        h_1 : Not (LE.le x y)
        ⊢ And (Ordnode.Valid' o₁ (l.balanceR y (Ordnode.insertWith f x r)) o₂) (Ordnod …
      -/
    · have : y < x := lt_of_le_not_le ((total_of (· ≤ ·) _ _).resolve_left h_1) h_1
      /-
        case neg
        α : Type u_1
        inst✝² : Preorder α
        inst✝¹ : IsTotal α fun x1 x2 => LE.le x1 x2
        inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
        f : α → α
        x : α
        hf : ∀ (y : α), And (LE.le x y) (LE.le y x) → And (LE.le x (f y)) (LE.le (f y) …
        sz : Nat
        l : Ordnode α
        y : α
        r : Ordnode α
        o₁ : WithBot α
        o₂ : WithTop α
        h : Ordnode.Valid' o₁ (Ordnode.node sz l y r) o₂
        bl : Ordnode.nil.Bounded o₁ ↑x
        br : Ordnode.nil.Bounded (↑x) o₂
        h_1 : Not (LE.le x y)
        this : LT.lt y x
        ⊢ And (Ordnode.Valid' o₁ (l.balanceR y (Ordnode.insertWith f x r)) o₂) (Ordnod …
      -/
      rcases insertWith.valid_aux f x hf h.right this br with ⟨vr, e⟩
      suffices H : _ by
        refine ⟨h.left.balanceR vr H, ?_⟩
        rw [size_balanceR h.3.2.1 vr.3 h.2.2.1 vr.2 H, h.2.size_eq]
        exact (e.add_left _).add_right _
      /-
        case neg.intro
        α : Type u_1
        inst✝² : Preorder α
        inst✝¹ : IsTotal α fun x1 x2 => LE.le x1 x2
        inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
        f : α → α
        x : α
        hf : ∀ (y : α), And (LE.le x y) (LE.le y x) → And (LE.le x (f y)) (LE.le (f y) …
        sz : Nat
        l : Ordnode α
        y : α
        r : Ordnode α
        o₁ : WithBot α
        o₂ : WithTop α
        h : Ordnode.Valid' o₁ (Ordnode.node sz l y r) o₂
        bl : Ordnode.nil.Bounded o₁ ↑x
        br : Ordnode.nil.Bounded (↑x) o₂
        h_1 : Not (LE.le x y)
        this : LT.lt y x
        vr : Ordnode.Valid' (↑y) (Ordnode.insertWith f x r) o₂
        e : Ordnode.Raised r.size (Ordnode.insertWith f x r).size
        ⊢ Or (Exists fun l' => And (Ordnode.Raised l.size l') (Ordnode.BalancedSz l' ( …
      -/
      exact Or.inr ⟨_, e, h.3.1⟩
      /-
        🎉 no goals
      -/


theorem insertWith.valid [IsTotal α (· ≤ ·)] [DecidableRel (α := α) (· ≤ ·)] (f : α → α) (x : α)
    (hf : ∀ y, x ≤ y ∧ y ≤ x → x ≤ f y ∧ f y ≤ x) {t} (h : Valid t) : Valid (insertWith f x t) :=
  (insertWith.valid_aux _ _ hf h ⟨⟩ ⟨⟩).1


theorem insert_eq_insertWith [DecidableRel (α := α) (· ≤ ·)] (x : α) :
    ∀ t, Ordnode.insert x t = insertWith (fun _ => x) x t
  | nil => rfl
  | node _ l y r => by
    /-
      α : Type u_1
      inst✝¹ : Preorder α
      inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
      x : α
      size✝ : Nat
      l : Ordnode α
      y : α
      r : Ordnode α
      ⊢ Eq (Ordnode.insert x (Ordnode.node size✝ l y r)) (Ordnode.insertWith (fun x_ …
    -/
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
    unfold Ordnode.insert insertWith; cases cmpLE x y <;> simp [insert_eq_insertWith]
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem insert.valid [IsTotal α (· ≤ ·)] [DecidableRel (α := α) (· ≤ ·)] (x : α) {t} (h : Valid t) :
    Valid (Ordnode.insert x t) := by
  /-
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : IsTotal α fun x1 x2 => LE.le x1 x2
    inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
    x : α
    t : Ordnode α
    h : t.Valid
    ⊢ (Ordnode.insert x t).Valid
  -/
  rw [insert_eq_insertWith]; exact insertWith.valid _ _ (fun _ _ => ⟨le_rfl, le_rfl⟩) h
                             /-
                               🎉 no goals
                             -/


theorem insert'_eq_insertWith [DecidableRel (α := α) (· ≤ ·)] (x : α) :
    ∀ t, insert' x t = insertWith id x t
  | nil => rfl
  | node _ l y r => by
    /-
      α : Type u_1
      inst✝¹ : Preorder α
      inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
      x : α
      size✝ : Nat
      l : Ordnode α
      y : α
      r : Ordnode α
      ⊢ Eq (Ordnode.insert' x (Ordnode.node size✝ l y r)) (Ordnode.insertWith id x ( …
    -/
                                                   /-
                                                     🎉 no goals
                                                   -/
                                                   /-
                                                     🎉 no goals
                                                   -/
    unfold insert' insertWith; cases cmpLE x y <;> simp [insert'_eq_insertWith]
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem insert'.valid [IsTotal α (· ≤ ·)] [DecidableRel (α := α) (· ≤ ·)]
    (x : α) {t} (h : Valid t) : Valid (insert' x t) := by
  /-
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : IsTotal α fun x1 x2 => LE.le x1 x2
    inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
    x : α
    t : Ordnode α
    h : t.Valid
    ⊢ (Ordnode.insert' x t).Valid
  -/
  rw [insert'_eq_insertWith]; exact insertWith.valid _ _ (fun _ => id) h
                              /-
                                🎉 no goals
                              -/


theorem Valid'.map_aux {β} [Preorder β] {f : α → β} (f_strict_mono : StrictMono f) {t a₁ a₂}
    (h : Valid' a₁ t a₂) :
    Valid' (Option.map f a₁) (map f t) (Option.map f a₂) ∧ (map f t).size = t.size := by
  induction t generalizing a₁ a₂ with
  | nil =>
    simp only [map, size_nil, and_true]; apply valid'_nil
    cases a₁; · trivial
    cases a₂; · trivial
    simp only [Option.map, Bounded]
    exact f_strict_mono h.ord
  | node _ _ _ _ t_ih_l t_ih_r =>
    have t_ih_l' := t_ih_l h.left
    have t_ih_r' := t_ih_r h.right
    clear t_ih_l t_ih_r
    cases' t_ih_l' with t_l_valid t_l_size
    cases' t_ih_r' with t_r_valid t_r_size
    simp only [map, size_node, and_true]
    constructor
    · exact And.intro t_l_valid.ord t_r_valid.ord
    · constructor
      · rw [t_l_size, t_r_size]; exact h.sz.1
      · constructor
        · exact t_l_valid.sz
        · exact t_r_valid.sz
    · constructor
      · rw [t_l_size, t_r_size]; exact h.bal.1
      · constructor
        · exact t_l_valid.bal
        · exact t_r_valid.bal


theorem map.valid {β} [Preorder β] {f : α → β} (f_strict_mono : StrictMono f) {t} (h : Valid t) :
    Valid (map f t) :=
  (Valid'.map_aux f_strict_mono h).1


theorem Valid'.erase_aux [DecidableRel (α := α) (· ≤ ·)] (x : α) {t a₁ a₂} (h : Valid' a₁ t a₂) :
    Valid' a₁ (erase x t) a₂ ∧ Raised (erase x t).size t.size := by
  induction t generalizing a₁ a₂ with
  | nil =>
    simpa [erase, Raised]
  | node _ t_l t_x t_r t_ih_l t_ih_r =>
    simp only [erase, size_node]
    have t_ih_l' := t_ih_l h.left
    have t_ih_r' := t_ih_r h.right
    clear t_ih_l t_ih_r
    cases' t_ih_l' with t_l_valid t_l_size
    cases' t_ih_r' with t_r_valid t_r_size
    cases cmpLE x t_x <;> rw [h.sz.1]
    · suffices h_balanceable : _ by
        constructor
        · exact Valid'.balanceR t_l_valid h.right h_balanceable
        · rw [size_balanceR t_l_valid.bal h.right.bal t_l_valid.sz h.right.sz h_balanceable]
          repeat apply Raised.add_right
          exact t_l_size
      left; exists t_l.size; exact And.intro t_l_size h.bal.1
    · have h_glue := Valid'.glue h.left h.right h.bal.1
      cases' h_glue with h_glue_valid h_glue_sized
      constructor
      · exact h_glue_valid
      · right; rw [h_glue_sized]
    · suffices h_balanceable : _ by
        constructor
        · exact Valid'.balanceL h.left t_r_valid h_balanceable
        · rw [size_balanceL h.left.bal t_r_valid.bal h.left.sz t_r_valid.sz h_balanceable]
          apply Raised.add_right
          apply Raised.add_left
          exact t_r_size
      right; exists t_r.size; exact And.intro t_r_size h.bal.1


theorem erase.valid [DecidableRel (α := α) (· ≤ ·)] (x : α) {t} (h : Valid t) : Valid (erase x t) :=
  (Valid'.erase_aux x h).1


theorem size_erase_of_mem [DecidableRel (α := α) (· ≤ ·)] {x : α} {t a₁ a₂} (h : Valid' a₁ t a₂)
    (h_mem : x ∈ t) : size (erase x t) = size t - 1 := by
  induction t generalizing a₁ a₂ with
  | nil =>
    contradiction
  | node _ t_l t_x t_r t_ih_l t_ih_r =>
    have t_ih_l' := t_ih_l h.left
    have t_ih_r' := t_ih_r h.right
    clear t_ih_l t_ih_r
    dsimp only [Membership.mem, mem] at h_mem
    unfold erase
    revert h_mem; cases cmpLE x t_x <;> intro h_mem <;> dsimp only at h_mem ⊢
    · have t_ih_l := t_ih_l' h_mem
      clear t_ih_l' t_ih_r'
      have t_l_h := Valid'.erase_aux x h.left
      cases' t_l_h with t_l_valid t_l_size
      rw [size_balanceR t_l_valid.bal h.right.bal t_l_valid.sz h.right.sz
          (Or.inl (Exists.intro t_l.size (And.intro t_l_size h.bal.1)))]
      rw [t_ih_l, h.sz.1]
      have h_pos_t_l_size := pos_size_of_mem h.left.sz h_mem
      revert h_pos_t_l_size; cases' t_l.size with t_l_size <;> intro h_pos_t_l_size
      · cases h_pos_t_l_size
      · simp [Nat.add_right_comm]
    · rw [(Valid'.glue h.left h.right h.bal.1).2, h.sz.1]; rfl
    · have t_ih_r := t_ih_r' h_mem
      clear t_ih_l' t_ih_r'
      have t_r_h := Valid'.erase_aux x h.right
      cases' t_r_h with t_r_valid t_r_size
      rw [size_balanceL h.left.bal t_r_valid.bal h.left.sz t_r_valid.sz
          (Or.inr (Exists.intro t_r.size (And.intro t_r_size h.bal.1)))]
      rw [t_ih_r, h.sz.1]
      have h_pos_t_r_size := pos_size_of_mem h.right.sz h_mem
      revert h_pos_t_r_size; cases' t_r.size with t_r_size <;> intro h_pos_t_r_size
      · cases h_pos_t_r_size
      · simp [Nat.add_assoc]


/-- An `Ordset α` is a finite set of values, represented as a tree. The operations on this type
maintain that the tree is balanced and correctly stores subtree sizes at each level. The
correctness property of the tree is baked into the type, so all operations on this type are correct
by construction. -/
def Ordset (α : Type*) [Preorder α] :=
  { t : Ordnode α // t.Valid }


/-- O(1). The empty set. -/
nonrec def nil : Ordset α :=
  ⟨nil, ⟨⟩, ⟨⟩, ⟨⟩⟩


/-- O(1). Get the size of the set. -/
def size (s : Ordset α) : ℕ :=
  s.1.size


/-- O(1). Construct a singleton set containing value `a`. -/
protected def singleton (a : α) : Ordset α :=
  ⟨singleton a, valid_singleton⟩


instance instEmptyCollection : EmptyCollection (Ordset α) :=
  ⟨nil⟩


instance instInhabited : Inhabited (Ordset α) :=
  ⟨nil⟩


instance instSingleton : Singleton α (Ordset α) :=
  ⟨Ordset.singleton⟩


/-- O(1). Is the set empty? -/
def Empty (s : Ordset α) : Prop :=
  s = ∅


theorem empty_iff {s : Ordset α} : s = ∅ ↔ s.1.empty :=
               /-
                 α : Type u_1
                 inst✝ : Preorder α
                 s : Ordset α
                 h : Eq s EmptyCollection.emptyCollection
                 ⊢ Eq (↑s).empty Bool.true
               -/
  ⟨fun h => by cases h; exact rfl,
                        /-
                          🎉 no goals
                        -/
                /-
                  α : Type u_1
                  inst✝ : Preorder α
                  s : Ordset α
                  h : Eq (↑s).empty Bool.true
                  ⊢ Eq s EmptyCollection.emptyCollection
                -/
                /-
                  🎉 no goals
                -/
    fun h => by cases s with | mk s_val _ => cases s_val <;> [rfl; cases h]⟩


instance Empty.instDecidablePred : DecidablePred (@Empty α _) :=
  fun _ => decidable_of_iff' _ empty_iff


/-- O(log n). Insert an element into the set, preserving balance and the BST property.
  If an equivalent element is already in the set, this replaces it. -/
protected def insert [IsTotal α (· ≤ ·)] [DecidableRel (α := α) (· ≤ ·)] (x : α) (s : Ordset α) :
    Ordset α :=
  ⟨Ordnode.insert x s.1, insert.valid _ s.2⟩


instance instInsert [IsTotal α (· ≤ ·)] [DecidableRel (α := α) (· ≤ ·)] : Insert α (Ordset α) :=
  ⟨Ordset.insert⟩


/-- O(log n). Insert an element into the set, preserving balance and the BST property.
  If an equivalent element is already in the set, the set is returned as is. -/
nonrec def insert' [IsTotal α (· ≤ ·)] [DecidableRel (α := α) (· ≤ ·)] (x : α) (s : Ordset α) :
    Ordset α :=
  ⟨insert' x s.1, insert'.valid _ s.2⟩


/-- O(log n). Does the set contain the element `x`? That is,
  is there an element that is equivalent to `x` in the order? -/
def mem (x : α) (s : Ordset α) : Bool :=
  x ∈ s.val


/-- O(log n). Retrieve an element in the set that is equivalent to `x` in the order,
  if it exists. -/
def find (x : α) (s : Ordset α) : Option α :=
  Ordnode.find x s.val


instance instMembership : Membership α (Ordset α) :=
  ⟨fun s x => mem x s⟩


instance mem.decidable (x : α) (s : Ordset α) : Decidable (x ∈ s) :=
  instDecidableEqBool _ _


theorem pos_size_of_mem {x : α} {t : Ordset α} (h_mem : x ∈ t) : 0 < size t := by
  simp? [Membership.mem, mem] at h_mem says
    simp only [Membership.mem, mem, Bool.decide_eq_true] at h_mem
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
    x : α
    t : Ordset α
    h_mem : Eq (Ordnode.mem x ↑t) Bool.true
    ⊢ LT.lt 0 t.size
  -/
  apply Ordnode.pos_size_of_mem t.property.sz h_mem
  /-
    🎉 no goals
  -/


/-- O(log n). Remove an element from the set equivalent to `x`. Does nothing if there
is no such element. -/
def erase [DecidableRel (α := α) (· ≤ ·)] (x : α) (s : Ordset α) : Ordset α :=
  ⟨Ordnode.erase x s.val, Ordnode.erase.valid x s.property⟩


/-- O(n). Map a function across a tree, without changing the structure. -/
def map {β} [Preorder β] (f : α → β) (f_strict_mono : StrictMono f) (s : Ordset α) : Ordset β :=
  ⟨Ordnode.map f s.val, Ordnode.map.valid f_strict_mono s.property⟩


