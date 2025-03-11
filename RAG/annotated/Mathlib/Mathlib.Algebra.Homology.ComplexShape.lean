/-- A `c : ComplexShape ι` describes the shape of a chain complex,
with chain groups indexed by `ι`.
Typically `ι` will be `ℕ`, `ℤ`, or `Fin n`.

There is a relation `Rel : ι → ι → Prop`,
and we will only allow a non-zero differential from `i` to `j` when `Rel i j`.

There are axioms which imply `{ j // c.Rel i j }` and `{ i // c.Rel i j }` are subsingletons.
This means that the shape consists of some union of lines, rays, intervals, and circles.

Below we define `c.next` and `c.prev` which provide these related elements.
-/
@[ext]
structure ComplexShape (ι : Type*) where
  /-- Nonzero differentials `X i ⟶ X j` shall be allowed
    on homological complexes when `Rel i j` holds. -/
  Rel : ι → ι → Prop
  /-- There is at most one nonzero differential from `X i`. -/
  next_eq : ∀ {i j j'}, Rel i j → Rel i j' → j = j'
  /-- There is at most one nonzero differential to `X j`. -/
  prev_eq : ∀ {i i' j}, Rel i j → Rel i' j → i = i'


/-- The complex shape where only differentials from each `X.i` to itself are allowed.

This is mostly only useful so we can describe the relation of "related in `k` steps" below.
-/
@[simps]
def refl (ι : Type*) : ComplexShape ι where
  Rel i j := i = j
  next_eq w w' := w.symm.trans w'
  prev_eq w w' := w.trans w'.symm


/-- The reverse of a `ComplexShape`.
-/
@[simps]
def symm (c : ComplexShape ι) : ComplexShape ι where
  Rel i j := c.Rel j i
  next_eq w w' := c.prev_eq w w'
  prev_eq w w' := c.next_eq w w'


@[simp]
theorem symm_symm (c : ComplexShape ι) : c.symm.symm = c := rfl


theorem symm_bijective :
    Function.Bijective (ComplexShape.symm : ComplexShape ι → ComplexShape ι) :=
  Function.bijective_iff_has_inverse.mpr ⟨_, symm_symm, symm_symm⟩


/-- The "composition" of two `ComplexShape`s.

We need this to define "related in k steps" later.
-/
@[simp]
def trans (c₁ c₂ : ComplexShape ι) : ComplexShape ι where
  Rel := Relation.Comp c₁.Rel c₂.Rel
  next_eq w w' := by
    /-
      ι : Type u_1
      c₁ c₂ : ComplexShape ι
      i✝ j✝ j'✝ : ι
      w : Relation.Comp c₁.Rel c₂.Rel i✝ j✝
      w' : Relation.Comp c₁.Rel c₂.Rel i✝ j'✝
      ⊢ Eq j✝ j'✝
    -/
    obtain ⟨k, w₁, w₂⟩ := w
    /-
      case intro.intro
      ι : Type u_1
      c₁ c₂ : ComplexShape ι
      i✝ j✝ j'✝ : ι
      w' : Relation.Comp c₁.Rel c₂.Rel i✝ j'✝
      k : ι
      w₁ : c₁.Rel i✝ k
      w₂ : c₂.Rel k j✝
      ⊢ Eq j✝ j'✝
    -/
    obtain ⟨k', w₁', w₂'⟩ := w'
    /-
      case intro.intro.intro.intro
      ι : Type u_1
      c₁ c₂ : ComplexShape ι
      i✝ j✝ j'✝ k : ι
      w₁ : c₁.Rel i✝ k
      w₂ : c₂.Rel k j✝
      k' : ι
      w₁' : c₁.Rel i✝ k'
      w₂' : c₂.Rel k' j'✝
      ⊢ Eq j✝ j'✝
    -/
    rw [c₁.next_eq w₁ w₁'] at w₂
    /-
      case intro.intro.intro.intro
      ι : Type u_1
      c₁ c₂ : ComplexShape ι
      i✝ j✝ j'✝ k : ι
      w₁ : c₁.Rel i✝ k
      k' : ι
      w₂ : c₂.Rel k' j✝
      w₁' : c₁.Rel i✝ k'
      w₂' : c₂.Rel k' j'✝
      ⊢ Eq j✝ j'✝
    -/
    exact c₂.next_eq w₂ w₂'
    /-
      🎉 no goals
    -/
  prev_eq w w' := by
    /-
      ι : Type u_1
      c₁ c₂ : ComplexShape ι
      i✝ i'✝ j✝ : ι
      w : Relation.Comp c₁.Rel c₂.Rel i✝ j✝
      w' : Relation.Comp c₁.Rel c₂.Rel i'✝ j✝
      ⊢ Eq i✝ i'✝
    -/
    obtain ⟨k, w₁, w₂⟩ := w
    /-
      case intro.intro
      ι : Type u_1
      c₁ c₂ : ComplexShape ι
      i✝ i'✝ j✝ : ι
      w' : Relation.Comp c₁.Rel c₂.Rel i'✝ j✝
      k : ι
      w₁ : c₁.Rel i✝ k
      w₂ : c₂.Rel k j✝
      ⊢ Eq i✝ i'✝
    -/
    obtain ⟨k', w₁', w₂'⟩ := w'
    /-
      case intro.intro.intro.intro
      ι : Type u_1
      c₁ c₂ : ComplexShape ι
      i✝ i'✝ j✝ k : ι
      w₁ : c₁.Rel i✝ k
      w₂ : c₂.Rel k j✝
      k' : ι
      w₁' : c₁.Rel i'✝ k'
      w₂' : c₂.Rel k' j✝
      ⊢ Eq i✝ i'✝
    -/
    rw [c₂.prev_eq w₂ w₂'] at w₁
    /-
      case intro.intro.intro.intro
      ι : Type u_1
      c₁ c₂ : ComplexShape ι
      i✝ i'✝ j✝ k : ι
      w₂ : c₂.Rel k j✝
      k' : ι
      w₁ : c₁.Rel i✝ k'
      w₁' : c₁.Rel i'✝ k'
      w₂' : c₂.Rel k' j✝
      ⊢ Eq i✝ i'✝
    -/
    exact c₁.prev_eq w₁ w₁'
    /-
      🎉 no goals
    -/


instance subsingleton_next (c : ComplexShape ι) (i : ι) : Subsingleton { j // c.Rel i j } := by
  /-
    ι : Type u_1
    c : ComplexShape ι
    i : ι
    ⊢ Subsingleton (Subtype fun j => c.Rel i j)
  -/
  constructor
  /-
    case allEq
    ι : Type u_1
    c : ComplexShape ι
    i : ι
    ⊢ ∀ (a b : Subtype fun j => c.Rel i j), Eq a b
  -/
  rintro ⟨j, rij⟩ ⟨k, rik⟩
  /-
    case allEq.mk.mk
    ι : Type u_1
    c : ComplexShape ι
    i j : ι
    rij : c.Rel i j
    k : ι
    rik : c.Rel i k
    ⊢ Eq ⟨j, rij⟩ ⟨k, rik⟩
  -/
  congr
  /-
    case allEq.mk.mk.e_val
    ι : Type u_1
    c : ComplexShape ι
    i j : ι
    rij : c.Rel i j
    k : ι
    rik : c.Rel i k
    ⊢ Eq j k
  -/
  exact c.next_eq rij rik
  /-
    🎉 no goals
  -/


instance subsingleton_prev (c : ComplexShape ι) (j : ι) : Subsingleton { i // c.Rel i j } := by
  /-
    ι : Type u_1
    c : ComplexShape ι
    j : ι
    ⊢ Subsingleton (Subtype fun i => c.Rel i j)
  -/
  constructor
  /-
    case allEq
    ι : Type u_1
    c : ComplexShape ι
    j : ι
    ⊢ ∀ (a b : Subtype fun i => c.Rel i j), Eq a b
  -/
  rintro ⟨i, rik⟩ ⟨j, rjk⟩
  /-
    case allEq.mk.mk
    ι : Type u_1
    c : ComplexShape ι
    j✝ i : ι
    rik : c.Rel i j✝
    j : ι
    rjk : c.Rel j j✝
    ⊢ Eq ⟨i, rik⟩ ⟨j, rjk⟩
  -/
  congr
  /-
    case allEq.mk.mk.e_val
    ι : Type u_1
    c : ComplexShape ι
    j✝ i : ι
    rik : c.Rel i j✝
    j : ι
    rjk : c.Rel j j✝
    ⊢ Eq i j
  -/
  exact c.prev_eq rik rjk
  /-
    🎉 no goals
  -/


open Classical in
/-- An arbitrary choice of index `j` such that `Rel i j`, if such exists.
Returns `i` otherwise.
-/
def next (c : ComplexShape ι) (i : ι) : ι :=
  if h : ∃ j, c.Rel i j then h.choose else i


open Classical in
/-- An arbitrary choice of index `i` such that `Rel i j`, if such exists.
Returns `j` otherwise.
-/
def prev (c : ComplexShape ι) (j : ι) : ι :=
  if h : ∃ i, c.Rel i j then h.choose else j


theorem next_eq' (c : ComplexShape ι) {i j : ι} (h : c.Rel i j) : c.next i = j := by
  /-
    ι : Type u_1
    c : ComplexShape ι
    i j : ι
    h : c.Rel i j
    ⊢ Eq (c.next i) j
  -/
  apply c.next_eq _ h
  /-
    ι : Type u_1
    c : ComplexShape ι
    i j : ι
    h : c.Rel i j
    ⊢ c.Rel i (c.next i)
  -/
  rw [next]
  /-
    ι : Type u_1
    c : ComplexShape ι
    i j : ι
    h : c.Rel i j
    ⊢ c.Rel i (dite (Exists fun j => c.Rel i j) (fun h => h.choose) fun h => i)
  -/
  rw [dif_pos]
  /-
    ι : Type u_1
    c : ComplexShape ι
    i j : ι
    h : c.Rel i j
    ⊢ c.Rel i (Exists.choose ?hc)
  -/
  exact Exists.choose_spec ⟨j, h⟩
  /-
    🎉 no goals
  -/


theorem prev_eq' (c : ComplexShape ι) {i j : ι} (h : c.Rel i j) : c.prev j = i := by
  /-
    ι : Type u_1
    c : ComplexShape ι
    i j : ι
    h : c.Rel i j
    ⊢ Eq (c.prev j) i
  -/
  apply c.prev_eq _ h
  /-
    ι : Type u_1
    c : ComplexShape ι
    i j : ι
    h : c.Rel i j
    ⊢ c.Rel (c.prev j) j
  -/
  rw [prev, dif_pos]
  /-
    ι : Type u_1
    c : ComplexShape ι
    i j : ι
    h : c.Rel i j
    ⊢ c.Rel (Exists.choose ?hc) j
  -/
  exact Exists.choose_spec (⟨i, h⟩ : ∃ k, c.Rel k j)
  /-
    🎉 no goals
  -/


lemma next_eq_self' (c : ComplexShape ι) (j : ι) (hj : ∀ k, ¬ c.Rel j k) :
    c.next j = j :=
              /-
                ι : Type u_1
                c : ComplexShape ι
                j : ι
                hj : ∀ (k : ι), Not (c.Rel j k)
                ⊢ Not (Exists fun j_1 => c.Rel j j_1)
              -/
  dif_neg (by simpa using hj)
              /-
                🎉 no goals
              -/


lemma prev_eq_self' (c : ComplexShape ι) (j : ι) (hj : ∀ i, ¬ c.Rel i j) :
    c.prev j = j :=
              /-
                ι : Type u_1
                c : ComplexShape ι
                j : ι
                hj : ∀ (i : ι), Not (c.Rel i j)
                ⊢ Not (Exists fun i => c.Rel i j)
              -/
  dif_neg (by simpa using hj)
              /-
                🎉 no goals
              -/


lemma next_eq_self (c : ComplexShape ι) (j : ι) (hj : ¬ c.Rel j (c.next j)) :
    c.next j = j :=
                                         /-
                                           ι : Type u_1
                                           c : ComplexShape ι
                                           j : ι
                                           hj : Not (c.Rel j (c.next j))
                                           k : ι
                                           hk' : c.Rel j k
                                           ⊢ c.Rel j (c.next j)
                                         -/
  c.next_eq_self' j (fun k hk' => hj (by simpa only [c.next_eq' hk'] using hk'))
                                         /-
                                           🎉 no goals
                                         -/


lemma prev_eq_self (c : ComplexShape ι) (j : ι) (hj : ¬ c.Rel (c.prev j) j) :
    c.prev j = j :=
                                         /-
                                           ι : Type u_1
                                           c : ComplexShape ι
                                           j : ι
                                           hj : Not (c.Rel (c.prev j) j)
                                           k : ι
                                           hk' : c.Rel k j
                                           ⊢ c.Rel (c.prev j) j
                                         -/
  c.prev_eq_self' j (fun k hk' => hj (by simpa only [c.prev_eq' hk'] using hk'))
                                         /-
                                           🎉 no goals
                                         -/


/-- The `ComplexShape` allowing differentials from `X i` to `X (i+a)`.
(For example when `a = 1`, a cohomology theory indexed by `ℕ` or `ℤ`)
-/
@[simps]
def up' {α : Type*} [AddRightCancelSemigroup α] (a : α) : ComplexShape α where
  Rel i j := i + a = j
  next_eq hi hj := hi.symm.trans hj
  prev_eq hi hj := add_right_cancel (hi.trans hj.symm)


/-- The `ComplexShape` allowing differentials from `X (j+a)` to `X j`.
(For example when `a = 1`, a homology theory indexed by `ℕ` or `ℤ`)
-/
@[simps]
def down' {α : Type*} [AddRightCancelSemigroup α] (a : α) : ComplexShape α where
  Rel i j := j + a = i
  next_eq hi hj := add_right_cancel (hi.trans hj.symm)
  prev_eq hi hj := hi.symm.trans hj


theorem down'_mk {α : Type*} [AddRightCancelSemigroup α] (a : α) (i j : α) (h : j + a = i) :
    (down' a).Rel i j := h


/-- The `ComplexShape` appropriate for cohomology, so `d : X i ⟶ X j` only when `j = i + 1`.
-/
@[simps!]
def up (α : Type*) [AddRightCancelSemigroup α] [One α] : ComplexShape α :=
  up' 1


/-- The `ComplexShape` appropriate for homology, so `d : X i ⟶ X j` only when `i = j + 1`.
-/
@[simps!]
def down (α : Type*) [AddRightCancelSemigroup α] [One α] : ComplexShape α :=
  down' 1


theorem down_mk {α : Type*} [AddRightCancelSemigroup α] [One α] (i j : α) (h : j + 1 = i) :
    (down α).Rel i j :=
  down'_mk (1 : α) i j h


instance (a : α) : DecidableRel (ComplexShape.up' a).Rel :=
                /-
                  α : Type u_1
                  inst✝¹ : AddRightCancelSemigroup α
                  inst✝ : DecidableEq α
                  a x✝¹ x✝ : α
                  ⊢ Decidable ((ComplexShape.up' a).Rel x✝¹ x✝)
                -/
  fun _ _ => by dsimp; infer_instance
                       /-
                         🎉 no goals
                       -/


instance (a : α) : DecidableRel (ComplexShape.down' a).Rel :=
                /-
                  α : Type u_1
                  inst✝¹ : AddRightCancelSemigroup α
                  inst✝ : DecidableEq α
                  a x✝¹ x✝ : α
                  ⊢ Decidable ((ComplexShape.down' a).Rel x✝¹ x✝)
                -/
  fun _ _ => by dsimp; infer_instance
                       /-
                         🎉 no goals
                       -/


instance : DecidableRel (ComplexShape.up α).Rel := by
  /-
    α : Type u_1
    inst✝² : AddRightCancelSemigroup α
    inst✝¹ : DecidableEq α
    inst✝ : One α
    ⊢ DecidableRel (ComplexShape.up α).Rel
  -/
  dsimp [ComplexShape.up]; infer_instance
                           /-
                             🎉 no goals
                           -/


instance : DecidableRel (ComplexShape.down α).Rel := by
  /-
    α : Type u_1
    inst✝² : AddRightCancelSemigroup α
    inst✝¹ : DecidableEq α
    inst✝ : One α
    ⊢ DecidableRel (ComplexShape.down α).Rel
  -/
  dsimp [ComplexShape.down]; infer_instance
                             /-
                               🎉 no goals
                             -/


