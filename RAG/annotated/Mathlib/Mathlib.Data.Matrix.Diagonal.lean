/-- `diagonal d` is the square matrix such that `(diagonal d) i i = d i` and `(diagonal d) i j = 0`
if `i ≠ j`.

Note that bundled versions exist as:
* `Matrix.diagonalAddMonoidHom`
* `Matrix.diagonalLinearMap`
* `Matrix.diagonalRingHom`
* `Matrix.diagonalAlgHom`
-/
def diagonal [Zero α] (d : n → α) : Matrix n n α :=
  of fun i j => if i = j then d i else 0

-- TODO: set as an equation lemma for `diagonal`, see https://github.com/leanprover-community/mathlib4/pull/3024

theorem diagonal_apply [Zero α] (d : n → α) (i j) : diagonal d i j = if i = j then d i else 0 :=
  rfl


@[simp]
theorem diagonal_apply_eq [Zero α] (d : n → α) (i : n) : (diagonal d) i i = d i := by
  /-
    n : Type u_3
    α : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Zero α
    d : n → α
    i : n
    ⊢ Eq (Matrix.diagonal d i i) (d i)
  -/
  simp [diagonal]
  /-
    🎉 no goals
  -/


@[simp]
theorem diagonal_apply_ne [Zero α] (d : n → α) {i j : n} (h : i ≠ j) : (diagonal d) i j = 0 := by
  /-
    n : Type u_3
    α : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Zero α
    d : n → α
    i j : n
    h : Ne i j
    ⊢ Eq (Matrix.diagonal d i j) 0
  -/
  simp [diagonal, h]
  /-
    🎉 no goals
  -/


theorem diagonal_apply_ne' [Zero α] (d : n → α) {i j : n} (h : j ≠ i) : (diagonal d) i j = 0 :=
  diagonal_apply_ne d h.symm


@[simp]
theorem diagonal_eq_diagonal_iff [Zero α] {d₁ d₂ : n → α} :
    diagonal d₁ = diagonal d₂ ↔ ∀ i, d₁ i = d₂ i :=
                 /-
                   n : Type u_3
                   α : Type v
                   inst✝¹ : DecidableEq n
                   inst✝ : Zero α
                   d₁ d₂ : n → α
                   h : Eq (Matrix.diagonal d₁) (Matrix.diagonal d₂)
                   i : n
                   ⊢ Eq (d₁ i) (d₂ i)
                 -/
  ⟨fun h i => by simpa using congr_arg (fun m : Matrix n n α => m i i) h, fun h => by
                 /-
                   🎉 no goals
                 -/
    /-
      n : Type u_3
      α : Type v
      inst✝¹ : DecidableEq n
      inst✝ : Zero α
      d₁ d₂ : n → α
      h : ∀ (i : n), Eq (d₁ i) (d₂ i)
      ⊢ Eq (Matrix.diagonal d₁) (Matrix.diagonal d₂)
    -/
    rw [show d₁ = d₂ from funext h]⟩
    /-
      🎉 no goals
    -/


theorem diagonal_injective [Zero α] : Function.Injective (diagonal : (n → α) → Matrix n n α) :=
                                    /-
                                      n : Type u_3
                                      α : Type v
                                      inst✝¹ : DecidableEq n
                                      inst✝ : Zero α
                                      d₁ d₂ : n → α
                                      h : Eq (Matrix.diagonal d₁) (Matrix.diagonal d₂)
                                      i : n
                                      ⊢ Eq (d₁ i) (d₂ i)
                                    -/
  fun d₁ d₂ h => funext fun i => by simpa using Matrix.ext_iff.mpr h i i
                                    /-
                                      🎉 no goals
                                    -/


@[simp]
theorem diagonal_zero [Zero α] : (diagonal fun _ => 0 : Matrix n n α) = 0 := by
  /-
    n : Type u_3
    α : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Zero α
    ⊢ Eq (Matrix.diagonal fun x => 0) 0
  -/
  ext
  /-
    case a
    n : Type u_3
    α : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Zero α
    i✝ j✝ : n
    ⊢ Eq (Matrix.diagonal (fun x => 0) i✝ j✝) (0 i✝ j✝)
  -/
  simp [diagonal]
  /-
    🎉 no goals
  -/


@[simp]
theorem diagonal_transpose [Zero α] (v : n → α) : (diagonal v)ᵀ = diagonal v := by
  /-
    n : Type u_3
    α : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Zero α
    v : n → α
    ⊢ Eq (Matrix.diagonal v).transpose (Matrix.diagonal v)
  -/
  ext i j
  /-
    case a
    n : Type u_3
    α : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Zero α
    v : n → α
    i j : n
    ⊢ Eq ((Matrix.diagonal v).transpose i j) (Matrix.diagonal v i j)
  -/
  by_cases h : i = j
    /-
      case pos
      n : Type u_3
      α : Type v
      inst✝¹ : DecidableEq n
      inst✝ : Zero α
      v : n → α
      i j : n
      h : Eq i j
      ⊢ Eq ((Matrix.diagonal v).transpose i j) (Matrix.diagonal v i j)
    -/
  · simp [h, transpose]
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Type u_3
      α : Type v
      inst✝¹ : DecidableEq n
      inst✝ : Zero α
      v : n → α
      i j : n
      h : Not (Eq i j)
      ⊢ Eq ((Matrix.diagonal v).transpose i j) (Matrix.diagonal v i j)
    -/
  · simp [h, transpose, diagonal_apply_ne' _ h]
    /-
      🎉 no goals
    -/


@[simp]
theorem diagonal_add [AddZeroClass α] (d₁ d₂ : n → α) :
    diagonal d₁ + diagonal d₂ = diagonal fun i => d₁ i + d₂ i := by
  /-
    n : Type u_3
    α : Type v
    inst✝¹ : DecidableEq n
    inst✝ : AddZeroClass α
    d₁ d₂ : n → α
    ⊢ Eq (HAdd.hAdd (Matrix.diagonal d₁) (Matrix.diagonal d₂)) (Matrix.diagonal fu …
  -/
  ext i j
  /-
    case a
    n : Type u_3
    α : Type v
    inst✝¹ : DecidableEq n
    inst✝ : AddZeroClass α
    d₁ d₂ : n → α
    i j : n
    ⊢ Eq (HAdd.hAdd (Matrix.diagonal d₁) (Matrix.diagonal d₂) i j) (Matrix.diagona …
  -/
  by_cases h : i = j <;>
  /-
    case pos
    n : Type u_3
    α : Type v
    inst✝¹ : DecidableEq n
    inst✝ : AddZeroClass α
    d₁ d₂ : n → α
    i j : n
    h : Eq i j
    ⊢ Eq (HAdd.hAdd (Matrix.diagonal d₁) (Matrix.diagonal d₂) i j) (Matrix.diagona …
  -/
  /-
    🎉 no goals
  -/
  simp [h]
  /-
    🎉 no goals
  -/


@[simp]
theorem diagonal_smul [Zero α] [SMulZeroClass R α] (r : R) (d : n → α) :
    diagonal (r • d) = r • diagonal d := by
  /-
    n : Type u_3
    R : Type u_7
    α : Type v
    inst✝² : DecidableEq n
    inst✝¹ : Zero α
    inst✝ : SMulZeroClass R α
    r : R
    d : n → α
    ⊢ Eq (Matrix.diagonal (HSMul.hSMul r d)) (HSMul.hSMul r (Matrix.diagonal d))
  -/
  ext i j
  /-
    case a
    n : Type u_3
    R : Type u_7
    α : Type v
    inst✝² : DecidableEq n
    inst✝¹ : Zero α
    inst✝ : SMulZeroClass R α
    r : R
    d : n → α
    i j : n
    ⊢ Eq (Matrix.diagonal (HSMul.hSMul r d) i j) (HSMul.hSMul r (Matrix.diagonal d …
  -/
                         /-
                           🎉 no goals
                         -/
  by_cases h : i = j <;> simp [h]
                         /-
                           🎉 no goals
                         -/


@[simp]
theorem diagonal_neg [NegZeroClass α] (d : n → α) :
    -diagonal d = diagonal fun i => -d i := by
  /-
    n : Type u_3
    α : Type v
    inst✝¹ : DecidableEq n
    inst✝ : NegZeroClass α
    d : n → α
    ⊢ Eq (Neg.neg (Matrix.diagonal d)) (Matrix.diagonal fun i => Neg.neg (d i))
  -/
  ext i j
  /-
    case a
    n : Type u_3
    α : Type v
    inst✝¹ : DecidableEq n
    inst✝ : NegZeroClass α
    d : n → α
    i j : n
    ⊢ Eq (Neg.neg (Matrix.diagonal d) i j) (Matrix.diagonal (fun i => Neg.neg (d i …
  -/
  by_cases h : i = j <;>
  /-
    case pos
    n : Type u_3
    α : Type v
    inst✝¹ : DecidableEq n
    inst✝ : NegZeroClass α
    d : n → α
    i j : n
    h : Eq i j
    ⊢ Eq (Neg.neg (Matrix.diagonal d) i j) (Matrix.diagonal (fun i => Neg.neg (d i …
  -/
  /-
    🎉 no goals
  -/
  simp [h]
  /-
    🎉 no goals
  -/


@[simp]
theorem diagonal_sub [SubNegZeroMonoid α] (d₁ d₂ : n → α) :
    diagonal d₁ - diagonal d₂ = diagonal fun i => d₁ i - d₂ i := by
  /-
    n : Type u_3
    α : Type v
    inst✝¹ : DecidableEq n
    inst✝ : SubNegZeroMonoid α
    d₁ d₂ : n → α
    ⊢ Eq (HSub.hSub (Matrix.diagonal d₁) (Matrix.diagonal d₂)) (Matrix.diagonal fu …
  -/
  ext i j
  /-
    case a
    n : Type u_3
    α : Type v
    inst✝¹ : DecidableEq n
    inst✝ : SubNegZeroMonoid α
    d₁ d₂ : n → α
    i j : n
    ⊢ Eq (HSub.hSub (Matrix.diagonal d₁) (Matrix.diagonal d₂) i j) (Matrix.diagona …
  -/
  by_cases h : i = j <;>
  /-
    case pos
    n : Type u_3
    α : Type v
    inst✝¹ : DecidableEq n
    inst✝ : SubNegZeroMonoid α
    d₁ d₂ : n → α
    i j : n
    h : Eq i j
    ⊢ Eq (HSub.hSub (Matrix.diagonal d₁) (Matrix.diagonal d₂) i j) (Matrix.diagona …
  -/
  /-
    🎉 no goals
  -/
  simp [h]
  /-
    🎉 no goals
  -/


instance [Zero α] [NatCast α] : NatCast (Matrix n n α) where
  natCast m := diagonal fun _ => m


@[norm_cast]
theorem diagonal_natCast [Zero α] [NatCast α] (m : ℕ) : diagonal (fun _ : n => (m : α)) = m := rfl


@[norm_cast]
theorem diagonal_natCast' [Zero α] [NatCast α] (m : ℕ) : diagonal ((m : n → α)) = m := rfl

-- See note [no_index around OfNat.ofNat]

theorem diagonal_ofNat [Zero α] [NatCast α] (m : ℕ) [m.AtLeastTwo] :
    diagonal (fun _ : n => no_index (OfNat.ofNat m : α)) = OfNat.ofNat m := rfl

-- See note [no_index around OfNat.ofNat]

theorem diagonal_ofNat' [Zero α] [NatCast α] (m : ℕ) [m.AtLeastTwo] :
    diagonal (no_index (OfNat.ofNat m : n → α)) = OfNat.ofNat m := rfl


instance [Zero α] [IntCast α] : IntCast (Matrix n n α) where
  intCast m := diagonal fun _ => m


@[norm_cast]
theorem diagonal_intCast [Zero α] [IntCast α] (m : ℤ) : diagonal (fun _ : n => (m : α)) = m := rfl


@[norm_cast]
theorem diagonal_intCast' [Zero α] [IntCast α] (m : ℤ) : diagonal ((m : n → α)) = m := rfl


@[simp]
theorem diagonal_map [Zero α] [Zero β] {f : α → β} (h : f 0 = 0) {d : n → α} :
    (diagonal d).map f = diagonal fun m => f (d m) := by
  /-
    n : Type u_3
    α : Type v
    β : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Zero α
    inst✝ : Zero β
    f : α → β
    h : Eq (f 0) 0
    d : n → α
    ⊢ Eq ((Matrix.diagonal d).map f) (Matrix.diagonal fun m => f (d m))
  -/
  ext
  /-
    case a
    n : Type u_3
    α : Type v
    β : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Zero α
    inst✝ : Zero β
    f : α → β
    h : Eq (f 0) 0
    d : n → α
    i✝ j✝ : n
    ⊢ Eq ((Matrix.diagonal d).map f i✝ j✝) (Matrix.diagonal (fun m => f (d m)) i✝  …
  -/
  simp only [diagonal_apply, map_apply]
  /-
    case a
    n : Type u_3
    α : Type v
    β : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Zero α
    inst✝ : Zero β
    f : α → β
    h : Eq (f 0) 0
    d : n → α
    i✝ j✝ : n
    ⊢ Eq (f (ite (Eq i✝ j✝) (d i✝) 0)) (ite (Eq i✝ j✝) (f (d i✝)) 0)
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp [h]
                /-
                  🎉 no goals
                -/


protected theorem map_natCast [AddMonoidWithOne α] [AddMonoidWithOne β]
    {f : α → β} (h : f 0 = 0) (d : ℕ) :
    (d : Matrix n n α).map f = diagonal (fun _ => f d) :=
  diagonal_map h

-- See note [no_index around OfNat.ofNat]

protected theorem map_ofNat [AddMonoidWithOne α] [AddMonoidWithOne β]
    {f : α → β} (h : f 0 = 0) (d : ℕ) [d.AtLeastTwo] :
    (no_index (OfNat.ofNat d) : Matrix n n α).map f = diagonal (fun _ => f (OfNat.ofNat d)) :=
  diagonal_map h


protected theorem map_intCast [AddGroupWithOne α] [AddGroupWithOne β]
    {f : α → β} (h : f 0 = 0) (d : ℤ) :
    (d : Matrix n n α).map f = diagonal (fun _ => f d) :=
  diagonal_map h


theorem diagonal_unique [Unique m] [DecidableEq m] [Zero α] (d : m → α) :
    diagonal d = of fun _ _ => d default := by
  /-
    m : Type u_2
    α : Type v
    inst✝² : Unique m
    inst✝¹ : DecidableEq m
    inst✝ : Zero α
    d : m → α
    ⊢ Eq (Matrix.diagonal d) (Matrix.of fun x x => d Inhabited.default)
  -/
  ext i j
  /-
    case a
    m : Type u_2
    α : Type v
    inst✝² : Unique m
    inst✝¹ : DecidableEq m
    inst✝ : Zero α
    d : m → α
    i j : m
    ⊢ Eq (Matrix.diagonal d i j) (Matrix.of (fun x x => d Inhabited.default) i j)
  -/
  rw [Subsingleton.elim i default, Subsingleton.elim j default, diagonal_apply_eq _ _, of_apply]
  /-
    🎉 no goals
  -/


instance one : One (Matrix n n α) :=
  ⟨diagonal fun _ => 1⟩


@[simp]
theorem diagonal_one : (diagonal fun _ => 1 : Matrix n n α) = 1 :=
  rfl


theorem one_apply {i j} : (1 : Matrix n n α) i j = if i = j then 1 else 0 :=
  rfl


@[simp]
theorem one_apply_eq (i) : (1 : Matrix n n α) i i = 1 :=
  diagonal_apply_eq _ i


@[simp]
theorem one_apply_ne {i j} : i ≠ j → (1 : Matrix n n α) i j = 0 :=
  diagonal_apply_ne _


theorem one_apply_ne' {i j} : j ≠ i → (1 : Matrix n n α) i j = 0 :=
  diagonal_apply_ne' _


@[simp]
theorem map_one [Zero β] [One β] (f : α → β) (h₀ : f 0 = 0) (h₁ : f 1 = 1) :
    (1 : Matrix n n α).map f = (1 : Matrix n n β) := by
  /-
    n : Type u_3
    α : Type v
    β : Type w
    inst✝⁴ : DecidableEq n
    inst✝³ : Zero α
    inst✝² : One α
    inst✝¹ : Zero β
    inst✝ : One β
    f : α → β
    h₀ : Eq (f 0) 0
    h₁ : Eq (f 1) 1
    ⊢ Eq (Matrix.map 1 f) 1
  -/
  ext
  /-
    case a
    n : Type u_3
    α : Type v
    β : Type w
    inst✝⁴ : DecidableEq n
    inst✝³ : Zero α
    inst✝² : One α
    inst✝¹ : Zero β
    inst✝ : One β
    f : α → β
    h₀ : Eq (f 0) 0
    h₁ : Eq (f 1) 1
    i✝ j✝ : n
    ⊢ Eq (Matrix.map 1 f i✝ j✝) (1 i✝ j✝)
  -/
  simp only [one_apply, map_apply]
  /-
    case a
    n : Type u_3
    α : Type v
    β : Type w
    inst✝⁴ : DecidableEq n
    inst✝³ : Zero α
    inst✝² : One α
    inst✝¹ : Zero β
    inst✝ : One β
    f : α → β
    h₀ : Eq (f 0) 0
    h₁ : Eq (f 1) 1
    i✝ j✝ : n
    ⊢ Eq (f (ite (Eq i✝ j✝) 1 0)) (ite (Eq i✝ j✝) 1 0)
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp [h₀, h₁]
                /-
                  🎉 no goals
                -/

-- Porting note: added implicit argument `(f := fun_ => α)`, why is that needed?

theorem one_eq_pi_single {i j} : (1 : Matrix n n α) i j = Pi.single (f := fun _ => α) i 1 j := by
  /-
    n : Type u_3
    α : Type v
    inst✝² : DecidableEq n
    inst✝¹ : Zero α
    inst✝ : One α
    i j : n
    ⊢ Eq (1 i j) (Pi.single i 1 j)
  -/
  simp only [one_apply, Pi.single_apply, eq_comm]
  /-
    🎉 no goals
  -/


instance instAddMonoidWithOne [AddMonoidWithOne α] : AddMonoidWithOne (Matrix n n α) where
  natCast_zero := show diagonal _ = _ by
    /-
      l : Type u_1
      m : Type u_2
      n : Type u_3
      o : Type u_4
      m' : o → Type u_5
      n' : o → Type u_6
      R : Type u_7
      S : Type u_8
      α : Type v
      β : Type w
      γ : Type u_9
      inst✝¹ : DecidableEq n
      inst✝ : AddMonoidWithOne α
      ⊢ Eq (Matrix.diagonal fun x => ↑0) 0
    -/
    rw [Nat.cast_zero, diagonal_zero]
    /-
      🎉 no goals
    -/
  natCast_succ n := show diagonal _ = diagonal _ + _ by
    /-
      l : Type u_1
      m : Type u_2
      n✝ : Type u_3
      o : Type u_4
      m' : o → Type u_5
      n' : o → Type u_6
      R : Type u_7
      S : Type u_8
      α : Type v
      β : Type w
      γ : Type u_9
      inst✝¹ : DecidableEq n✝
      inst✝ : AddMonoidWithOne α
      n : Nat
      ⊢ Eq (Matrix.diagonal fun x => ↑(HAdd.hAdd n 1)) (HAdd.hAdd (Matrix.diagonal f …
    -/
    rw [Nat.cast_succ, ← diagonal_add, diagonal_one]
    /-
      🎉 no goals
    -/


instance instAddGroupWithOne [AddGroupWithOne α] : AddGroupWithOne (Matrix n n α) where
  intCast_ofNat n := show diagonal _ = diagonal _ by
    /-
      l : Type u_1
      m : Type u_2
      n✝ : Type u_3
      o : Type u_4
      m' : o → Type u_5
      n' : o → Type u_6
      R : Type u_7
      S : Type u_8
      α : Type v
      β : Type w
      γ : Type u_9
      inst✝¹ : DecidableEq n✝
      inst✝ : AddGroupWithOne α
      n : Nat
      ⊢ Eq (Matrix.diagonal fun x => ↑↑n) (Matrix.diagonal fun x => ↑n)
    -/
    rw [Int.cast_natCast]
    /-
      🎉 no goals
    -/
  intCast_negSucc n := show diagonal _ = -(diagonal _) by
    /-
      l : Type u_1
      m : Type u_2
      n✝ : Type u_3
      o : Type u_4
      m' : o → Type u_5
      n' : o → Type u_6
      R : Type u_7
      S : Type u_8
      α : Type v
      β : Type w
      γ : Type u_9
      inst✝¹ : DecidableEq n✝
      inst✝ : AddGroupWithOne α
      n : Nat
      ⊢ Eq (Matrix.diagonal fun x => ↑(Int.negSucc n)) (Neg.neg (Matrix.diagonal fun …
    -/
    rw [Int.cast_negSucc, diagonal_neg]
    /-
      🎉 no goals
    -/
  __ := addGroup
  __ := instAddMonoidWithOne


instance instAddCommMonoidWithOne [AddCommMonoidWithOne α] :
    AddCommMonoidWithOne (Matrix n n α) where
  __ := addCommMonoid
  __ := instAddMonoidWithOne


instance instAddCommGroupWithOne [AddCommGroupWithOne α] :
    AddCommGroupWithOne (Matrix n n α) where
  __ := addCommGroup
  __ := instAddGroupWithOne


/-- The diagonal of a square matrix. -/
-- @[simp] -- Porting note: simpNF does not like this.
def diag (A : Matrix n n α) (i : n) : α :=
  A i i

-- Porting note: new, because of removed `simp` above.
-- TODO: set as an equation lemma for `diag`, see https://github.com/leanprover-community/mathlib4/pull/3024

@[simp]
theorem diag_apply (A : Matrix n n α) (i) : diag A i = A i i :=
  rfl


@[simp]
theorem diag_diagonal [DecidableEq n] [Zero α] (a : n → α) : diag (diagonal a) = a :=
  funext <| @diagonal_apply_eq _ _ _ _ a


@[simp]
theorem diag_transpose (A : Matrix n n α) : diag Aᵀ = diag A :=
  rfl


@[simp]
theorem diag_zero [Zero α] : diag (0 : Matrix n n α) = 0 :=
  rfl


@[simp]
theorem diag_add [Add α] (A B : Matrix n n α) : diag (A + B) = diag A + diag B :=
  rfl


@[simp]
theorem diag_sub [Sub α] (A B : Matrix n n α) : diag (A - B) = diag A - diag B :=
  rfl


@[simp]
theorem diag_neg [Neg α] (A : Matrix n n α) : diag (-A) = -diag A :=
  rfl


@[simp]
theorem diag_smul [SMul R α] (r : R) (A : Matrix n n α) : diag (r • A) = r • diag A :=
  rfl


@[simp]
theorem diag_one [DecidableEq n] [Zero α] [One α] : diag (1 : Matrix n n α) = 1 :=
  diag_diagonal _


theorem diag_map {f : α → β} {A : Matrix n n α} : diag (A.map f) = f ∘ diag A :=
  rfl


@[simp]
theorem transpose_eq_diagonal [DecidableEq n] [Zero α] {M : Matrix n n α} {v : n → α} :
    Mᵀ = diagonal v ↔ M = diagonal v :=
  (Function.Involutive.eq_iff transpose_transpose).trans <|
       /-
         n : Type u_3
         α : Type v
         inst✝¹ : DecidableEq n
         inst✝ : Zero α
         M : Matrix n n α
         v : n → α
         ⊢ Iff (Eq M (Matrix.diagonal v).transpose) (Eq M (Matrix.diagonal v))
       -/
    by rw [diagonal_transpose]
       /-
         🎉 no goals
       -/


@[simp]
theorem transpose_one [DecidableEq n] [Zero α] [One α] : (1 : Matrix n n α)ᵀ = 1 :=
  diagonal_transpose _


@[simp]
theorem transpose_eq_one [DecidableEq n] [Zero α] [One α] {M : Matrix n n α} : Mᵀ = 1 ↔ M = 1 :=
  transpose_eq_diagonal


@[simp]
theorem transpose_natCast [DecidableEq n] [AddMonoidWithOne α] (d : ℕ) :
    (d : Matrix n n α)ᵀ = d :=
  diagonal_transpose _


@[simp]
theorem transpose_eq_natCast [DecidableEq n] [AddMonoidWithOne α] {M : Matrix n n α} {d : ℕ} :
    Mᵀ = d ↔ M = d :=
  transpose_eq_diagonal

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem transpose_ofNat [DecidableEq n] [AddMonoidWithOne α] (d : ℕ) [d.AtLeastTwo] :
    (no_index (OfNat.ofNat d) : Matrix n n α)ᵀ = OfNat.ofNat d :=
  transpose_natCast _

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem transpose_eq_ofNat [DecidableEq n] [AddMonoidWithOne α]
    {M : Matrix n n α} {d : ℕ} [d.AtLeastTwo] :
    Mᵀ = no_index (OfNat.ofNat d) ↔ M = OfNat.ofNat d :=
  transpose_eq_diagonal


@[simp]
theorem transpose_intCast [DecidableEq n] [AddGroupWithOne α] (d : ℤ) :
    (d : Matrix n n α)ᵀ = d :=
  diagonal_transpose _


@[simp]
theorem transpose_eq_intCast [DecidableEq n] [AddGroupWithOne α]
    {M : Matrix n n α} {d : ℤ} :
    Mᵀ = d ↔ M = d :=
  transpose_eq_diagonal


/-- Given a `(m × m)` diagonal matrix defined by a map `d : m → α`, if the reindexing map `e` is
  injective, then the resulting matrix is again diagonal. -/
theorem submatrix_diagonal [Zero α] [DecidableEq m] [DecidableEq l] (d : m → α) (e : l → m)
    (he : Function.Injective e) : (diagonal d).submatrix e e = diagonal (d ∘ e) :=
  ext fun i j => by
    /-
      l : Type u_1
      m : Type u_2
      α : Type v
      inst✝² : Zero α
      inst✝¹ : DecidableEq m
      inst✝ : DecidableEq l
      d : m → α
      e : l → m
      he : Function.Injective e
      i j : l
      ⊢ Eq ((Matrix.diagonal d).submatrix e e i j) (Matrix.diagonal (Function.comp d …
    -/
    rw [submatrix_apply]
    /-
      l : Type u_1
      m : Type u_2
      α : Type v
      inst✝² : Zero α
      inst✝¹ : DecidableEq m
      inst✝ : DecidableEq l
      d : m → α
      e : l → m
      he : Function.Injective e
      i j : l
      ⊢ Eq (Matrix.diagonal d (e i) (e j)) (Matrix.diagonal (Function.comp d e) i j)
    -/
    by_cases h : i = j
      /-
        case pos
        l : Type u_1
        m : Type u_2
        α : Type v
        inst✝² : Zero α
        inst✝¹ : DecidableEq m
        inst✝ : DecidableEq l
        d : m → α
        e : l → m
        he : Function.Injective e
        i j : l
        h : Eq i j
        ⊢ Eq (Matrix.diagonal d (e i) (e j)) (Matrix.diagonal (Function.comp d e) i j)
      -/
    · rw [h, diagonal_apply_eq, diagonal_apply_eq]
      /-
        case pos
        l : Type u_1
        m : Type u_2
        α : Type v
        inst✝² : Zero α
        inst✝¹ : DecidableEq m
        inst✝ : DecidableEq l
        d : m → α
        e : l → m
        he : Function.Injective e
        i j : l
        h : Eq i j
        ⊢ Eq (d (e j)) (Function.comp d e j)
      -/
      simp only [Function.comp_apply] -- Porting note: (simp) added this
      /-
        🎉 no goals
      -/
      /-
        case neg
        l : Type u_1
        m : Type u_2
        α : Type v
        inst✝² : Zero α
        inst✝¹ : DecidableEq m
        inst✝ : DecidableEq l
        d : m → α
        e : l → m
        he : Function.Injective e
        i j : l
        h : Not (Eq i j)
        ⊢ Eq (Matrix.diagonal d (e i) (e j)) (Matrix.diagonal (Function.comp d e) i j)
      -/
    · rw [diagonal_apply_ne _ h, diagonal_apply_ne _ (he.ne h)]
      /-
        🎉 no goals
      -/


theorem submatrix_one [Zero α] [One α] [DecidableEq m] [DecidableEq l] (e : l → m)
    (he : Function.Injective e) : (1 : Matrix m m α).submatrix e e = 1 :=
  submatrix_diagonal _ e he


theorem diag_submatrix (A : Matrix m m α) (e : l → m) : diag (A.submatrix e e) = A.diag ∘ e :=
  rfl


@[simp]
theorem submatrix_diagonal_embedding [Zero α] [DecidableEq m] [DecidableEq l] (d : m → α)
    (e : l ↪ m) : (diagonal d).submatrix e e = diagonal (d ∘ e) :=
  submatrix_diagonal d e e.injective


@[simp]
theorem submatrix_diagonal_equiv [Zero α] [DecidableEq m] [DecidableEq l] (d : m → α) (e : l ≃ m) :
    (diagonal d).submatrix e e = diagonal (d ∘ e) :=
  submatrix_diagonal d e e.injective


@[simp]
theorem submatrix_one_embedding [Zero α] [One α] [DecidableEq m] [DecidableEq l] (e : l ↪ m) :
    (1 : Matrix m m α).submatrix e e = 1 :=
  submatrix_one e e.injective


@[simp]
theorem submatrix_one_equiv [Zero α] [One α] [DecidableEq m] [DecidableEq l] (e : l ≃ m) :
    (1 : Matrix m m α).submatrix e e = 1 :=
  submatrix_one e e.injective


