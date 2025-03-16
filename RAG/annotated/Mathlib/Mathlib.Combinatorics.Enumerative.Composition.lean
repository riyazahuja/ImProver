/-- A composition of `n` is a list of positive integers summing to `n`. -/
@[ext]
structure Composition (n : ℕ) where
  /-- List of positive integers summing to `n`-/
  blocks : List ℕ
  /-- Proof of positivity for `blocks`-/
  blocks_pos : ∀ {i}, i ∈ blocks → 0 < i
  /-- Proof that `blocks` sums to `n`-/
  blocks_sum : blocks.sum = n


/-- Combinatorial viewpoint on a composition of `n`, by seeing it as non-empty blocks of
consecutive integers in `{0, ..., n-1}`. We register every block by its left end-point, yielding
a finset containing `0`. As this does not make sense for `n = 0`, we add `n` to this finset, and
get a finset of `{0, ..., n}` containing `0` and `n`. This is the data in the structure
`CompositionAsSet n`. -/
@[ext]
structure CompositionAsSet (n : ℕ) where
  /-- Combinatorial viewpoint on a composition of `n` as consecutive integers `{0, ..., n-1}`-/
  boundaries : Finset (Fin n.succ)
  /-- Proof that `0` is a member of `boundaries`-/
  zero_mem : (0 : Fin n.succ) ∈ boundaries
  /-- Last element of the composition -/
  getLast_mem : Fin.last n ∈ boundaries


instance {n : ℕ} : Inhabited (CompositionAsSet n) :=
  ⟨⟨Finset.univ, Finset.mem_univ _, Finset.mem_univ _⟩⟩


instance (n : ℕ) : ToString (Composition n) :=
  ⟨fun c => toString c.blocks⟩


/-- The length of a composition, i.e., the number of blocks in the composition. -/
abbrev length : ℕ :=
  c.blocks.length


theorem blocks_length : c.blocks.length = c.length :=
  rfl


/-- The blocks of a composition, seen as a function on `Fin c.length`. When composing analytic
functions using compositions, this is the main player. -/
def blocksFun : Fin c.length → ℕ := c.blocks.get


theorem ofFn_blocksFun : ofFn c.blocksFun = c.blocks :=
  ofFn_get _


theorem sum_blocksFun : ∑ i, c.blocksFun i = n := by
  /-
    n : Nat
    c : Composition n
    ⊢ Eq (Finset.univ.sum fun i => c.blocksFun i) n
  -/
  conv_rhs => rw [← c.blocks_sum, ← ofFn_blocksFun, sum_ofFn]
  /-
    🎉 no goals
  -/


theorem blocksFun_mem_blocks (i : Fin c.length) : c.blocksFun i ∈ c.blocks :=
  get_mem _ _


@[simp]
theorem one_le_blocks {i : ℕ} (h : i ∈ c.blocks) : 1 ≤ i :=
  c.blocks_pos h


@[simp]
theorem one_le_blocks' {i : ℕ} (h : i < c.length) : 1 ≤ c.blocks[i] :=
  c.one_le_blocks (get_mem (blocks c) _)


@[simp]
theorem blocks_pos' (i : ℕ) (h : i < c.length) : 0 < c.blocks[i] :=
  c.one_le_blocks' h


theorem one_le_blocksFun (i : Fin c.length) : 1 ≤ c.blocksFun i :=
  c.one_le_blocks (c.blocksFun_mem_blocks i)


theorem blocksFun_le {n} (c : Composition n) (i : Fin c.length) :
    c.blocksFun i ≤ n := by
  /-
    n : Nat
    c : Composition n
    i : Fin c.length
    ⊢ LE.le (c.blocksFun i) n
  -/
  have := c.blocks_sum
  /-
    n : Nat
    c : Composition n
    i : Fin c.length
    this : Eq c.blocks.sum n
    ⊢ LE.le (c.blocksFun i) n
  -/
  have := List.le_sum_of_mem (c.blocksFun_mem_blocks i)
  /-
    n : Nat
    c : Composition n
    i : Fin c.length
    this✝ : Eq c.blocks.sum n
    this : LE.le (c.blocksFun i) c.blocks.sum
    ⊢ LE.le (c.blocksFun i) n
  -/
  simp_all
  /-
    🎉 no goals
  -/


theorem length_le : c.length ≤ n := by
  /-
    n : Nat
    c : Composition n
    ⊢ LE.le c.length n
  -/
  conv_rhs => rw [← c.blocks_sum]
  /-
    n : Nat
    c : Composition n
    ⊢ LE.le c.length c.blocks.sum
  -/
  exact length_le_sum_of_one_le _ fun i hi => c.one_le_blocks hi
  /-
    🎉 no goals
  -/


theorem length_pos_of_pos (h : 0 < n) : 0 < c.length := by
  /-
    n : Nat
    c : Composition n
    h : LT.lt 0 n
    ⊢ LT.lt 0 c.length
  -/
  apply length_pos_of_sum_pos
  /-
    case h
    n : Nat
    c : Composition n
    h : LT.lt 0 n
    ⊢ LT.lt 0 c.blocks.sum
  -/
  convert h
  /-
    case h.e'_4
    n : Nat
    c : Composition n
    h : LT.lt 0 n
    ⊢ Eq c.blocks.sum n
  -/
  exact c.blocks_sum
  /-
    🎉 no goals
  -/


/-- The sum of the sizes of the blocks in a composition up to `i`. -/
def sizeUpTo (i : ℕ) : ℕ :=
  (c.blocks.take i).sum


@[simp]
                                               /-
                                                 n : Nat
                                                 c : Composition n
                                                 ⊢ Eq (c.sizeUpTo 0) 0
                                               -/
theorem sizeUpTo_zero : c.sizeUpTo 0 = 0 := by simp [sizeUpTo]
                                               /-
                                                 🎉 no goals
                                               -/


theorem sizeUpTo_ofLength_le (i : ℕ) (h : c.length ≤ i) : c.sizeUpTo i = n := by
  /-
    n : Nat
    c : Composition n
    i : Nat
    h : LE.le c.length i
    ⊢ Eq (c.sizeUpTo i) n
  -/
  dsimp [sizeUpTo]
  /-
    n : Nat
    c : Composition n
    i : Nat
    h : LE.le c.length i
    ⊢ Eq (List.take i c.blocks).sum n
  -/
  convert c.blocks_sum
  /-
    case h.e'_2.h.e'_4
    n : Nat
    c : Composition n
    i : Nat
    h : LE.le c.length i
    ⊢ Eq (List.take i c.blocks) c.blocks
  -/
  exact take_of_length_le h
  /-
    🎉 no goals
  -/


@[simp]
theorem sizeUpTo_length : c.sizeUpTo c.length = n :=
  c.sizeUpTo_ofLength_le c.length le_rfl


theorem sizeUpTo_le (i : ℕ) : c.sizeUpTo i ≤ n := by
  /-
    n : Nat
    c : Composition n
    i : Nat
    ⊢ LE.le (c.sizeUpTo i) n
  -/
  conv_rhs => rw [← c.blocks_sum, ← sum_take_add_sum_drop _ i]
  /-
    n : Nat
    c : Composition n
    i : Nat
    ⊢ LE.le (c.sizeUpTo i) (HAdd.hAdd (List.take i c.blocks).sum (List.drop i c.bl …
  -/
  exact Nat.le_add_right _ _
  /-
    🎉 no goals
  -/


theorem sizeUpTo_succ {i : ℕ} (h : i < c.length) :
    c.sizeUpTo (i + 1) = c.sizeUpTo i + c.blocks[i] := by
  /-
    n : Nat
    c : Composition n
    i : Nat
    h : LT.lt i c.length
    ⊢ Eq (c.sizeUpTo (HAdd.hAdd i 1)) (HAdd.hAdd (c.sizeUpTo i) (GetElem.getElem c …
  -/
  simp only [sizeUpTo]
  /-
    n : Nat
    c : Composition n
    i : Nat
    h : LT.lt i c.length
    ⊢ Eq (List.take (HAdd.hAdd i 1) c.blocks).sum (HAdd.hAdd (List.take i c.blocks …
  -/
  rw [sum_take_succ _ _ h]
  /-
    🎉 no goals
  -/


theorem sizeUpTo_succ' (i : Fin c.length) :
    c.sizeUpTo ((i : ℕ) + 1) = c.sizeUpTo i + c.blocksFun i :=
  c.sizeUpTo_succ i.2


theorem sizeUpTo_strict_mono {i : ℕ} (h : i < c.length) : c.sizeUpTo i < c.sizeUpTo (i + 1) := by
  /-
    n : Nat
    c : Composition n
    i : Nat
    h : LT.lt i c.length
    ⊢ LT.lt (c.sizeUpTo i) (c.sizeUpTo (HAdd.hAdd i 1))
  -/
  rw [c.sizeUpTo_succ h]
  /-
    n : Nat
    c : Composition n
    i : Nat
    h : LT.lt i c.length
    ⊢ LT.lt (c.sizeUpTo i) (HAdd.hAdd (c.sizeUpTo i) (GetElem.getElem c.blocks i h))
  -/
  simp
  /-
    🎉 no goals
  -/


theorem monotone_sizeUpTo : Monotone c.sizeUpTo :=
  monotone_sum_take _


/-- The `i`-th boundary of a composition, i.e., the leftmost point of the `i`-th block. We include
a virtual point at the right of the last block, to make for a nice equiv with
`CompositionAsSet n`. -/
def boundary : Fin (c.length + 1) ↪o Fin (n + 1) :=
  (OrderEmbedding.ofStrictMono fun i => ⟨c.sizeUpTo i, Nat.lt_succ_of_le (c.sizeUpTo_le i)⟩) <|
    Fin.strictMono_iff_lt_succ.2 fun ⟨_, hi⟩ => c.sizeUpTo_strict_mono hi


@[simp]
                                               /-
                                                 n : Nat
                                                 c : Composition n
                                                 ⊢ Eq (c.boundary 0) 0
                                               -/
theorem boundary_zero : c.boundary 0 = 0 := by simp [boundary, Fin.ext_iff]
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
theorem boundary_last : c.boundary (Fin.last c.length) = Fin.last n := by
  /-
    n : Nat
    c : Composition n
    ⊢ Eq (c.boundary (Fin.last c.length)) (Fin.last n)
  -/
  simp [boundary, Fin.ext_iff]
  /-
    🎉 no goals
  -/


/-- The boundaries of a composition, i.e., the leftmost point of all the blocks. We include
a virtual point at the right of the last block, to make for a nice equiv with
`CompositionAsSet n`. -/
def boundaries : Finset (Fin (n + 1)) :=
  Finset.univ.map c.boundary.toEmbedding


                                                                                /-
                                                                                  n : Nat
                                                                                  c : Composition n
                                                                                  ⊢ Eq c.boundaries.card (HAdd.hAdd c.length 1)
                                                                                -/
theorem card_boundaries_eq_succ_length : c.boundaries.card = c.length + 1 := by simp [boundaries]
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


/-- To `c : Composition n`, one can associate a `CompositionAsSet n` by registering the leftmost
point of each block, and adding a virtual point at the right of the last block. -/
def toCompositionAsSet : CompositionAsSet n where
  boundaries := c.boundaries
  zero_mem := by
    /-
      n : Nat
      c : Composition n
      ⊢ Membership.mem c.boundaries 0
    -/
    simp only [boundaries, Finset.mem_univ, exists_prop_of_true, Finset.mem_map]
    /-
      n : Nat
      c : Composition n
      ⊢ Exists fun a => And True (Eq (c.boundary.toEmbedding a) 0)
    -/
    exact ⟨0, And.intro True.intro rfl⟩
    /-
      🎉 no goals
    -/
  getLast_mem := by
    /-
      n : Nat
      c : Composition n
      ⊢ Membership.mem c.boundaries (Fin.last n)
    -/
    simp only [boundaries, Finset.mem_univ, exists_prop_of_true, Finset.mem_map]
    /-
      n : Nat
      c : Composition n
      ⊢ Exists fun a => And True (Eq (c.boundary.toEmbedding a) (Fin.last n))
    -/
    exact ⟨Fin.last c.length, And.intro True.intro c.boundary_last⟩
    /-
      🎉 no goals
    -/


/-- The canonical increasing bijection between `Fin (c.length + 1)` and `c.boundaries` is
exactly `c.boundary`. -/
theorem orderEmbOfFin_boundaries :
    c.boundaries.orderEmbOfFin c.card_boundaries_eq_succ_length = c.boundary := by
  /-
    n : Nat
    c : Composition n
    ⊢ Eq (c.boundaries.orderEmbOfFin ⋯) c.boundary
  -/
  refine (Finset.orderEmbOfFin_unique' _ ?_).symm
  /-
    n : Nat
    c : Composition n
    ⊢ ∀ (x : Fin (HAdd.hAdd c.length 1)), Membership.mem c.boundaries (c.boundary x)
  -/
  exact fun i => (Finset.mem_map' _).2 (Finset.mem_univ _)
  /-
    🎉 no goals
  -/


/-- Embedding the `i`-th block of a composition (identified with `Fin (c.blocksFun i)`) into
`Fin n` at the relevant position. -/
def embedding (i : Fin c.length) : Fin (c.blocksFun i) ↪o Fin n :=
  (Fin.natAddOrderEmb <| c.sizeUpTo i).trans <| Fin.castLEOrderEmb <|
    calc
      c.sizeUpTo i + c.blocksFun i = c.sizeUpTo (i + 1) := (c.sizeUpTo_succ i.2).symm
      _ ≤ c.sizeUpTo c.length := monotone_sum_take _ i.2
      _ = n := c.sizeUpTo_length


@[simp]
theorem coe_embedding (i : Fin c.length) (j : Fin (c.blocksFun i)) :
    (c.embedding i j : ℕ) = c.sizeUpTo i + j :=
  rfl


/-- `index_exists` asserts there is some `i` with `j < c.sizeUpTo (i+1)`.
In the next definition `index` we use `Nat.find` to produce the minimal such index.
-/
theorem index_exists {j : ℕ} (h : j < n) : ∃ i : ℕ, j < c.sizeUpTo (i + 1) ∧ i < c.length := by
  /-
    n : Nat
    c : Composition n
    j : Nat
    h : LT.lt j n
    ⊢ Exists fun i => And (LT.lt j (c.sizeUpTo (HAdd.hAdd i 1))) (LT.lt i c.length)
  -/
  have n_pos : 0 < n := lt_of_le_of_lt (zero_le j) h
  /-
    n : Nat
    c : Composition n
    j : Nat
    h : LT.lt j n
    n_pos : LT.lt 0 n
    ⊢ Exists fun i => And (LT.lt j (c.sizeUpTo (HAdd.hAdd i 1))) (LT.lt i c.length)
  -/
  have : 0 < c.blocks.sum := by rwa [← c.blocks_sum] at n_pos
  /-
    n : Nat
    c : Composition n
    j : Nat
    h : LT.lt j n
    n_pos : LT.lt 0 n
    this : LT.lt 0 c.blocks.sum
    ⊢ Exists fun i => And (LT.lt j (c.sizeUpTo (HAdd.hAdd i 1))) (LT.lt i c.length)
  -/
  have length_pos : 0 < c.blocks.length := length_pos_of_sum_pos (blocks c) this
  /-
    n : Nat
    c : Composition n
    j : Nat
    h : LT.lt j n
    n_pos : LT.lt 0 n
    this : LT.lt 0 c.blocks.sum
    length_pos : LT.lt 0 c.blocks.length
    ⊢ Exists fun i => And (LT.lt j (c.sizeUpTo (HAdd.hAdd i 1))) (LT.lt i c.length)
  -/
  refine ⟨c.length - 1, ?_, Nat.pred_lt (ne_of_gt length_pos)⟩
  /-
    n : Nat
    c : Composition n
    j : Nat
    h : LT.lt j n
    n_pos : LT.lt 0 n
    this : LT.lt 0 c.blocks.sum
    length_pos : LT.lt 0 c.blocks.length
    ⊢ LT.lt j (c.sizeUpTo (HAdd.hAdd (HSub.hSub c.length 1) 1))
  -/
  have : c.length - 1 + 1 = c.length := Nat.succ_pred_eq_of_pos length_pos
  /-
    n : Nat
    c : Composition n
    j : Nat
    h : LT.lt j n
    n_pos : LT.lt 0 n
    this✝ : LT.lt 0 c.blocks.sum
    length_pos : LT.lt 0 c.blocks.length
    this : Eq (HAdd.hAdd (HSub.hSub c.length 1) 1) c.length
    ⊢ LT.lt j (c.sizeUpTo (HAdd.hAdd (HSub.hSub c.length 1) 1))
  -/
  simp [this, h]
  /-
    🎉 no goals
  -/


/-- `c.index j` is the index of the block in the composition `c` containing `j`. -/
def index (j : Fin n) : Fin c.length :=
  ⟨Nat.find (c.index_exists j.2), (Nat.find_spec (c.index_exists j.2)).2⟩


theorem lt_sizeUpTo_index_succ (j : Fin n) : (j : ℕ) < c.sizeUpTo (c.index j).succ :=
  (Nat.find_spec (c.index_exists j.2)).1


theorem sizeUpTo_index_le (j : Fin n) : c.sizeUpTo (c.index j) ≤ j := by
  /-
    n : Nat
    c : Composition n
    j : Fin n
    ⊢ LE.le (c.sizeUpTo ↑(c.index j)) ↑j
  -/
  by_contra H
  /-
    n : Nat
    c : Composition n
    j : Fin n
    H : Not (LE.le (c.sizeUpTo ↑(c.index j)) ↑j)
    ⊢ False
  -/
  set i := c.index j
  /-
    n : Nat
    c : Composition n
    j : Fin n
    i : Fin c.length := c.index j
    H : Not (LE.le (c.sizeUpTo ↑i) ↑j)
    ⊢ False
  -/
  push_neg at H
  have i_pos : (0 : ℕ) < i := by
    by_contra! i_pos
    revert H
    simp [nonpos_iff_eq_zero.1 i_pos, c.sizeUpTo_zero]
  /-
    n : Nat
    c : Composition n
    j : Fin n
    i : Fin c.length := c.index j
    H : LT.lt (↑j) (c.sizeUpTo ↑i)
    i_pos : LT.lt 0 ↑i
    ⊢ False
  -/
  let i₁ := (i : ℕ).pred
  /-
    n : Nat
    c : Composition n
    j : Fin n
    i : Fin c.length := c.index j
    H : LT.lt (↑j) (c.sizeUpTo ↑i)
    i_pos : LT.lt 0 ↑i
    i₁ : Nat := (↑i).pred
    ⊢ False
  -/
  have i₁_lt_i : i₁ < i := Nat.pred_lt (ne_of_gt i_pos)
  /-
    n : Nat
    c : Composition n
    j : Fin n
    i : Fin c.length := c.index j
    H : LT.lt (↑j) (c.sizeUpTo ↑i)
    i_pos : LT.lt 0 ↑i
    i₁ : Nat := (↑i).pred
    i₁_lt_i : LT.lt i₁ ↑i
    ⊢ False
  -/
  have i₁_succ : i₁ + 1 = i := Nat.succ_pred_eq_of_pos i_pos
  /-
    n : Nat
    c : Composition n
    j : Fin n
    i : Fin c.length := c.index j
    H : LT.lt (↑j) (c.sizeUpTo ↑i)
    i_pos : LT.lt 0 ↑i
    i₁ : Nat := (↑i).pred
    i₁_lt_i : LT.lt i₁ ↑i
    i₁_succ : Eq (HAdd.hAdd i₁ 1) ↑i
    ⊢ False
  -/
  have := Nat.find_min (c.index_exists j.2) i₁_lt_i
  /-
    n : Nat
    c : Composition n
    j : Fin n
    i : Fin c.length := c.index j
    H : LT.lt (↑j) (c.sizeUpTo ↑i)
    i_pos : LT.lt 0 ↑i
    i₁ : Nat := (↑i).pred
    i₁_lt_i : LT.lt i₁ ↑i
    i₁_succ : Eq (HAdd.hAdd i₁ 1) ↑i
    this : Not (And (LT.lt (↑j) (c.sizeUpTo (HAdd.hAdd i₁ 1))) (LT.lt i₁ c.length))
    ⊢ False
  -/
  simp [lt_trans i₁_lt_i (c.index j).2, i₁_succ] at this
  /-
    n : Nat
    c : Composition n
    j : Fin n
    i : Fin c.length := c.index j
    H : LT.lt (↑j) (c.sizeUpTo ↑i)
    i_pos : LT.lt 0 ↑i
    i₁ : Nat := (↑i).pred
    i₁_lt_i : LT.lt i₁ ↑i
    i₁_succ : Eq (HAdd.hAdd i₁ 1) ↑i
    this : LE.le (c.sizeUpTo ↑i) ↑j
    ⊢ False
  -/
  exact Nat.lt_le_asymm H this
  /-
    🎉 no goals
  -/


/-- Mapping an element `j` of `Fin n` to the element in the block containing it, identified with
`Fin (c.blocksFun (c.index j))` through the canonical increasing bijection. -/
def invEmbedding (j : Fin n) : Fin (c.blocksFun (c.index j)) :=
  ⟨j - c.sizeUpTo (c.index j), by
    /-
      n : Nat
      c : Composition n
      j : Fin n
      ⊢ LT.lt (HSub.hSub (↑j) (c.sizeUpTo ↑(c.index j))) (c.blocksFun (c.index j))
    -/
    rw [tsub_lt_iff_right, add_comm, ← sizeUpTo_succ']
      /-
        n : Nat
        c : Composition n
        j : Fin n
        ⊢ LT.lt (↑j) (c.sizeUpTo (HAdd.hAdd (↑(c.index j)) 1))
      -/
    · exact lt_sizeUpTo_index_succ _ _
      /-
        🎉 no goals
      -/
      /-
        n : Nat
        c : Composition n
        j : Fin n
        ⊢ LE.le (c.sizeUpTo ↑(c.index j)) ↑j
      -/
    · exact sizeUpTo_index_le _ _⟩
      /-
        🎉 no goals
      -/


@[simp]
theorem coe_invEmbedding (j : Fin n) : (c.invEmbedding j : ℕ) = j - c.sizeUpTo (c.index j) :=
  rfl


theorem embedding_comp_inv (j : Fin n) : c.embedding (c.index j) (c.invEmbedding j) = j := by
  /-
    n : Nat
    c : Composition n
    j : Fin n
    ⊢ Eq ((c.embedding (c.index j)) (c.invEmbedding j)) j
  -/
  rw [Fin.ext_iff]
  /-
    n : Nat
    c : Composition n
    j : Fin n
    ⊢ Eq ↑((c.embedding (c.index j)) (c.invEmbedding j)) ↑j
  -/
  apply add_tsub_cancel_of_le (c.sizeUpTo_index_le j)
  /-
    🎉 no goals
  -/


theorem mem_range_embedding_iff {j : Fin n} {i : Fin c.length} :
    j ∈ Set.range (c.embedding i) ↔ c.sizeUpTo i ≤ j ∧ (j : ℕ) < c.sizeUpTo (i : ℕ).succ := by
  /-
    n : Nat
    c : Composition n
    j : Fin n
    i : Fin c.length
    ⊢ Iff (Membership.mem (Set.range ⇑(c.embedding i)) j) (And (LE.le (c.sizeUpTo  …
  -/
  constructor
    /-
      case mp
      n : Nat
      c : Composition n
      j : Fin n
      i : Fin c.length
      ⊢ Membership.mem (Set.range ⇑(c.embedding i)) j → And (LE.le (c.sizeUpTo ↑i) ↑ …
    -/
  · intro h
    /-
      case mp
      n : Nat
      c : Composition n
      j : Fin n
      i : Fin c.length
      h : Membership.mem (Set.range ⇑(c.embedding i)) j
      ⊢ And (LE.le (c.sizeUpTo ↑i) ↑j) (LT.lt (↑j) (c.sizeUpTo (↑i).succ))
    -/
    rcases Set.mem_range.2 h with ⟨k, hk⟩
    /-
      case mp.intro
      n : Nat
      c : Composition n
      j : Fin n
      i : Fin c.length
      h : Membership.mem (Set.range ⇑(c.embedding i)) j
      k : Fin (c.blocksFun i)
      hk : Eq ((c.embedding i) k) j
      ⊢ And (LE.le (c.sizeUpTo ↑i) ↑j) (LT.lt (↑j) (c.sizeUpTo (↑i).succ))
    -/
    rw [Fin.ext_iff] at hk
    /-
      case mp.intro
      n : Nat
      c : Composition n
      j : Fin n
      i : Fin c.length
      h : Membership.mem (Set.range ⇑(c.embedding i)) j
      k : Fin (c.blocksFun i)
      hk : Eq ↑((c.embedding i) k) ↑j
      ⊢ And (LE.le (c.sizeUpTo ↑i) ↑j) (LT.lt (↑j) (c.sizeUpTo (↑i).succ))
    -/
    dsimp at hk
    /-
      case mp.intro
      n : Nat
      c : Composition n
      j : Fin n
      i : Fin c.length
      h : Membership.mem (Set.range ⇑(c.embedding i)) j
      k : Fin (c.blocksFun i)
      hk : Eq (HAdd.hAdd (c.sizeUpTo ↑i) ↑k) ↑j
      ⊢ And (LE.le (c.sizeUpTo ↑i) ↑j) (LT.lt (↑j) (c.sizeUpTo (↑i).succ))
    -/
    rw [← hk]
    /-
      case mp.intro
      n : Nat
      c : Composition n
      j : Fin n
      i : Fin c.length
      h : Membership.mem (Set.range ⇑(c.embedding i)) j
      k : Fin (c.blocksFun i)
      hk : Eq (HAdd.hAdd (c.sizeUpTo ↑i) ↑k) ↑j
      ⊢ And (LE.le (c.sizeUpTo ↑i) (HAdd.hAdd (c.sizeUpTo ↑i) ↑k)) (LT.lt (HAdd.hAdd …
    -/
    simp [sizeUpTo_succ', k.is_lt]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      n : Nat
      c : Composition n
      j : Fin n
      i : Fin c.length
      ⊢ And (LE.le (c.sizeUpTo ↑i) ↑j) (LT.lt (↑j) (c.sizeUpTo (↑i).succ)) → Members …
    -/
  · intro h
    /-
      case mpr
      n : Nat
      c : Composition n
      j : Fin n
      i : Fin c.length
      h : And (LE.le (c.sizeUpTo ↑i) ↑j) (LT.lt (↑j) (c.sizeUpTo (↑i).succ))
      ⊢ Membership.mem (Set.range ⇑(c.embedding i)) j
    -/
    apply Set.mem_range.2
    /-
      case mpr
      n : Nat
      c : Composition n
      j : Fin n
      i : Fin c.length
      h : And (LE.le (c.sizeUpTo ↑i) ↑j) (LT.lt (↑j) (c.sizeUpTo (↑i).succ))
      ⊢ Exists fun y => Eq ((c.embedding i) y) j
    -/
    refine ⟨⟨j - c.sizeUpTo i, ?_⟩, ?_⟩
      /-
        case mpr.refine_1
        n : Nat
        c : Composition n
        j : Fin n
        i : Fin c.length
        h : And (LE.le (c.sizeUpTo ↑i) ↑j) (LT.lt (↑j) (c.sizeUpTo (↑i).succ))
        ⊢ LT.lt (HSub.hSub (↑j) (c.sizeUpTo ↑i)) (c.blocksFun i)
      -/
    · rw [tsub_lt_iff_left, ← sizeUpTo_succ']
        /-
          case mpr.refine_1
          n : Nat
          c : Composition n
          j : Fin n
          i : Fin c.length
          h : And (LE.le (c.sizeUpTo ↑i) ↑j) (LT.lt (↑j) (c.sizeUpTo (↑i).succ))
          ⊢ LT.lt (↑j) (c.sizeUpTo (HAdd.hAdd (↑i) 1))
        -/
      · exact h.2
        /-
          🎉 no goals
        -/
        /-
          case mpr.refine_1
          n : Nat
          c : Composition n
          j : Fin n
          i : Fin c.length
          h : And (LE.le (c.sizeUpTo ↑i) ↑j) (LT.lt (↑j) (c.sizeUpTo (↑i).succ))
          ⊢ LE.le (c.sizeUpTo ↑i) ↑j
        -/
      · exact h.1
        /-
          🎉 no goals
        -/
      /-
        case mpr.refine_2
        n : Nat
        c : Composition n
        j : Fin n
        i : Fin c.length
        h : And (LE.le (c.sizeUpTo ↑i) ↑j) (LT.lt (↑j) (c.sizeUpTo (↑i).succ))
        ⊢ Eq ((c.embedding i) ⟨HSub.hSub (↑j) (c.sizeUpTo ↑i), ⋯⟩) j
      -/
    · rw [Fin.ext_iff]
      /-
        case mpr.refine_2
        n : Nat
        c : Composition n
        j : Fin n
        i : Fin c.length
        h : And (LE.le (c.sizeUpTo ↑i) ↑j) (LT.lt (↑j) (c.sizeUpTo (↑i).succ))
        ⊢ Eq ↑((c.embedding i) ⟨HSub.hSub (↑j) (c.sizeUpTo ↑i), ⋯⟩) ↑j
      -/
      exact add_tsub_cancel_of_le h.1
      /-
        🎉 no goals
      -/


/-- The embeddings of different blocks of a composition are disjoint. -/
theorem disjoint_range {i₁ i₂ : Fin c.length} (h : i₁ ≠ i₂) :
    Disjoint (Set.range (c.embedding i₁)) (Set.range (c.embedding i₂)) := by
  classical
    wlog h' : i₁ < i₂
    · exact (this c h.symm (h.lt_or_lt.resolve_left h')).symm
    by_contra d
    obtain ⟨x, hx₁, hx₂⟩ :
      ∃ x : Fin n, x ∈ Set.range (c.embedding i₁) ∧ x ∈ Set.range (c.embedding i₂) :=
      Set.not_disjoint_iff.1 d
    have A : (i₁ : ℕ).succ ≤ i₂ := Nat.succ_le_of_lt h'
    apply lt_irrefl (x : ℕ)
    calc
      (x : ℕ) < c.sizeUpTo (i₁ : ℕ).succ := (c.mem_range_embedding_iff.1 hx₁).2
      _ ≤ c.sizeUpTo (i₂ : ℕ) := monotone_sum_take _ A
      _ ≤ x := (c.mem_range_embedding_iff.1 hx₂).1


theorem mem_range_embedding (j : Fin n) : j ∈ Set.range (c.embedding (c.index j)) := by
  have : c.embedding (c.index j) (c.invEmbedding j) ∈ Set.range (c.embedding (c.index j)) :=
    Set.mem_range_self _
  /-
    n : Nat
    c : Composition n
    j : Fin n
    this : Membership.mem (Set.range ⇑(c.embedding (c.index j))) ((c.embedding (c. …
    ⊢ Membership.mem (Set.range ⇑(c.embedding (c.index j))) j
  -/
  rwa [c.embedding_comp_inv j] at this
  /-
    🎉 no goals
  -/


theorem mem_range_embedding_iff' {j : Fin n} {i : Fin c.length} :
    j ∈ Set.range (c.embedding i) ↔ i = c.index j := by
  /-
    n : Nat
    c : Composition n
    j : Fin n
    i : Fin c.length
    ⊢ Iff (Membership.mem (Set.range ⇑(c.embedding i)) j) (Eq i (c.index j))
  -/
  constructor
    /-
      case mp
      n : Nat
      c : Composition n
      j : Fin n
      i : Fin c.length
      ⊢ Membership.mem (Set.range ⇑(c.embedding i)) j → Eq i (c.index j)
    -/
  · rw [← not_imp_not]
    /-
      case mp
      n : Nat
      c : Composition n
      j : Fin n
      i : Fin c.length
      ⊢ Not (Eq i (c.index j)) → Not (Membership.mem (Set.range ⇑(c.embedding i)) j)
    -/
    intro h
    /-
      case mp
      n : Nat
      c : Composition n
      j : Fin n
      i : Fin c.length
      h : Not (Eq i (c.index j))
      ⊢ Not (Membership.mem (Set.range ⇑(c.embedding i)) j)
    -/
    exact Set.disjoint_right.1 (c.disjoint_range h) (c.mem_range_embedding j)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      n : Nat
      c : Composition n
      j : Fin n
      i : Fin c.length
      ⊢ Eq i (c.index j) → Membership.mem (Set.range ⇑(c.embedding i)) j
    -/
  · intro h
    /-
      case mpr
      n : Nat
      c : Composition n
      j : Fin n
      i : Fin c.length
      h : Eq i (c.index j)
      ⊢ Membership.mem (Set.range ⇑(c.embedding i)) j
    -/
    rw [h]
    /-
      case mpr
      n : Nat
      c : Composition n
      j : Fin n
      i : Fin c.length
      h : Eq i (c.index j)
      ⊢ Membership.mem (Set.range ⇑(c.embedding (c.index j))) j
    -/
    exact c.mem_range_embedding j
    /-
      🎉 no goals
    -/


theorem index_embedding (i : Fin c.length) (j : Fin (c.blocksFun i)) :
    c.index (c.embedding i j) = i := by
  /-
    n : Nat
    c : Composition n
    i : Fin c.length
    j : Fin (c.blocksFun i)
    ⊢ Eq (c.index ((c.embedding i) j)) i
  -/
  symm
  /-
    n : Nat
    c : Composition n
    i : Fin c.length
    j : Fin (c.blocksFun i)
    ⊢ Eq i (c.index ((c.embedding i) j))
  -/
  rw [← mem_range_embedding_iff']
  /-
    n : Nat
    c : Composition n
    i : Fin c.length
    j : Fin (c.blocksFun i)
    ⊢ Membership.mem (Set.range ⇑(c.embedding i)) ((c.embedding i) j)
  -/
  apply Set.mem_range_self
  /-
    🎉 no goals
  -/


theorem invEmbedding_comp (i : Fin c.length) (j : Fin (c.blocksFun i)) :
    (c.invEmbedding (c.embedding i j) : ℕ) = j := by
  /-
    n : Nat
    c : Composition n
    i : Fin c.length
    j : Fin (c.blocksFun i)
    ⊢ Eq ↑(c.invEmbedding ((c.embedding i) j)) ↑j
  -/
  simp_rw [coe_invEmbedding, index_embedding, coe_embedding, add_tsub_cancel_left]
  /-
    🎉 no goals
  -/


/-- Equivalence between the disjoint union of the blocks (each of them seen as
`Fin (c.blocksFun i)`) with `Fin n`. -/
def blocksFinEquiv : (Σi : Fin c.length, Fin (c.blocksFun i)) ≃ Fin n where
  toFun x := c.embedding x.1 x.2
  invFun j := ⟨c.index j, c.invEmbedding j⟩
  left_inv x := by
    /-
      n : Nat
      c : Composition n
      x : Sigma fun i => Fin (c.blocksFun i)
      ⊢ Eq ((fun j => ⟨c.index j, c.invEmbedding j⟩) ((fun x => (c.embedding x.fst)  …
    -/
    rcases x with ⟨i, y⟩
    /-
      case mk
      n : Nat
      c : Composition n
      i : Fin c.length
      y : Fin (c.blocksFun i)
      ⊢ Eq ((fun j => ⟨c.index j, c.invEmbedding j⟩) ((fun x => (c.embedding x.fst)  …
    -/
    dsimp
    /-
      case mk
      n : Nat
      c : Composition n
      i : Fin c.length
      y : Fin (c.blocksFun i)
      ⊢ Eq ⟨c.index ((c.embedding i) y), c.invEmbedding ((c.embedding i) y)⟩ ⟨i, y⟩
    -/
    congr; · exact c.index_embedding _ _
             /-
               🎉 no goals
             -/
    /-
      case mk.h.e_4
      n : Nat
      c : Composition n
      i : Fin c.length
      y : Fin (c.blocksFun i)
      ⊢ HEq (c.invEmbedding ((c.embedding i) y)) y
    -/
    rw [Fin.heq_ext_iff]
      /-
        case mk.h.e_4
        n : Nat
        c : Composition n
        i : Fin c.length
        y : Fin (c.blocksFun i)
        ⊢ Eq ↑(c.invEmbedding ((c.embedding i) y)) ↑y
      -/
    · exact c.invEmbedding_comp _ _
      /-
        🎉 no goals
      -/
      /-
        case mk.h.e_4.h
        n : Nat
        c : Composition n
        i : Fin c.length
        y : Fin (c.blocksFun i)
        ⊢ Eq (c.blocksFun (c.index ((c.embedding i) y))) (c.blocksFun i)
      -/
    · rw [c.index_embedding]
      /-
        🎉 no goals
      -/
  right_inv j := c.embedding_comp_inv j


theorem blocksFun_congr {n₁ n₂ : ℕ} (c₁ : Composition n₁) (c₂ : Composition n₂) (i₁ : Fin c₁.length)
    (i₂ : Fin c₂.length) (hn : n₁ = n₂) (hc : c₁.blocks = c₂.blocks) (hi : (i₁ : ℕ) = i₂) :
    c₁.blocksFun i₁ = c₂.blocksFun i₂ := by
  /-
    n₁ n₂ : Nat
    c₁ : Composition n₁
    c₂ : Composition n₂
    i₁ : Fin c₁.length
    i₂ : Fin c₂.length
    hn : Eq n₁ n₂
    hc : Eq c₁.blocks c₂.blocks
    hi : Eq ↑i₁ ↑i₂
    ⊢ Eq (c₁.blocksFun i₁) (c₂.blocksFun i₂)
  -/
  cases hn
  /-
    case refl
    n₁ : Nat
    c₁ : Composition n₁
    i₁ : Fin c₁.length
    c₂ : Composition n₁
    i₂ : Fin c₂.length
    hc : Eq c₁.blocks c₂.blocks
    hi : Eq ↑i₁ ↑i₂
    ⊢ Eq (c₁.blocksFun i₁) (c₂.blocksFun i₂)
  -/
  rw [← Composition.ext_iff] at hc
  /-
    case refl
    n₁ : Nat
    c₁ : Composition n₁
    i₁ : Fin c₁.length
    c₂ : Composition n₁
    i₂ : Fin c₂.length
    hc : Eq c₁ c₂
    hi : Eq ↑i₁ ↑i₂
    ⊢ Eq (c₁.blocksFun i₁) (c₂.blocksFun i₂)
  -/
  cases hc
  /-
    case refl.refl
    n₁ : Nat
    c₁ : Composition n₁
    i₁ i₂ : Fin c₁.length
    hi : Eq ↑i₁ ↑i₂
    ⊢ Eq (c₁.blocksFun i₁) (c₁.blocksFun i₂)
  -/
  congr
  /-
    case refl.refl.e_a
    n₁ : Nat
    c₁ : Composition n₁
    i₁ i₂ : Fin c₁.length
    hi : Eq ↑i₁ ↑i₂
    ⊢ Eq i₁ i₂
  -/
  rwa [Fin.ext_iff]
  /-
    🎉 no goals
  -/


/-- Two compositions (possibly of different integers) coincide if and only if they have the
same sequence of blocks. -/
theorem sigma_eq_iff_blocks_eq {c : Σn, Composition n} {c' : Σn, Composition n} :
    c = c' ↔ c.2.blocks = c'.2.blocks := by
  /-
    c c' : Sigma fun n => Composition n
    ⊢ Iff (Eq c c') (Eq c.snd.blocks c'.snd.blocks)
  -/
  refine ⟨fun H => by rw [H], fun H => ?_⟩
  /-
    c c' : Sigma fun n => Composition n
    H : Eq c.snd.blocks c'.snd.blocks
    ⊢ Eq c c'
  -/
  rcases c with ⟨n, c⟩
  /-
    case mk
    c' : Sigma fun n => Composition n
    n : Nat
    c : Composition n
    H : Eq ⟨n, c⟩.snd.blocks c'.snd.blocks
    ⊢ Eq ⟨n, c⟩ c'
  -/
  rcases c' with ⟨n', c'⟩
  /-
    case mk.mk
    n : Nat
    c : Composition n
    n' : Nat
    c' : Composition n'
    H : Eq ⟨n, c⟩.snd.blocks ⟨n', c'⟩.snd.blocks
    ⊢ Eq ⟨n, c⟩ ⟨n', c'⟩
  -/
  have : n = n' := by rw [← c.blocks_sum, ← c'.blocks_sum, H]
  /-
    case mk.mk
    n : Nat
    c : Composition n
    n' : Nat
    c' : Composition n'
    H : Eq ⟨n, c⟩.snd.blocks ⟨n', c'⟩.snd.blocks
    this : Eq n n'
    ⊢ Eq ⟨n, c⟩ ⟨n', c'⟩
  -/
  induction this
  /-
    case mk.mk.refl
    n : Nat
    c : Composition n
    n' : Nat
    c' : Composition n
    H : Eq ⟨n, c⟩.snd.blocks ⟨n, c'⟩.snd.blocks
    ⊢ Eq ⟨n, c⟩ ⟨n, c'⟩
  -/
  congr
  /-
    case mk.mk.refl.e_snd
    n : Nat
    c : Composition n
    n' : Nat
    c' : Composition n
    H : Eq ⟨n, c⟩.snd.blocks ⟨n, c'⟩.snd.blocks
    ⊢ Eq c c'
  -/
  ext1
  /-
    case mk.mk.refl.e_snd.blocks
    n : Nat
    c : Composition n
    n' : Nat
    c' : Composition n
    H : Eq ⟨n, c⟩.snd.blocks ⟨n, c'⟩.snd.blocks
    ⊢ Eq c.blocks c'.blocks
  -/
  exact H
  /-
    🎉 no goals
  -/


/-- The composition made of blocks all of size `1`. -/
def ones (n : ℕ) : Composition n :=
                                         /-
                                           n✝ : Nat
                                           c : Composition n✝
                                           n i : Nat
                                           hi : Membership.mem (List.replicate n 1) i
                                           ⊢ LT.lt 0 i
                                         -/
                                         /-
                                           🎉 no goals
                                         -/
  ⟨replicate n (1 : ℕ), fun {i} hi => by simp [List.eq_of_mem_replicate hi], by simp⟩
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


instance {n : ℕ} : Inhabited (Composition n) :=
  ⟨Composition.ones n⟩


@[simp]
theorem ones_length (n : ℕ) : (ones n).length = n :=
  List.length_replicate n 1


@[simp]
theorem ones_blocks (n : ℕ) : (ones n).blocks = replicate n (1 : ℕ) :=
  rfl


@[simp]
theorem ones_blocksFun (n : ℕ) (i : Fin (ones n).length) : (ones n).blocksFun i = 1 := by
  /-
    n : Nat
    i : Fin (Composition.ones n).length
    ⊢ Eq ((Composition.ones n).blocksFun i) 1
  -/
  simp only [blocksFun, ones, get_eq_getElem, getElem_replicate]
  /-
    🎉 no goals
  -/


@[simp]
theorem ones_sizeUpTo (n : ℕ) (i : ℕ) : (ones n).sizeUpTo i = min i n := by
  /-
    n i : Nat
    ⊢ Eq ((Composition.ones n).sizeUpTo i) (Min.min i n)
  -/
  simp [sizeUpTo, ones_blocks, take_replicate]
  /-
    🎉 no goals
  -/


@[simp]
theorem ones_embedding (i : Fin (ones n).length) (h : 0 < (ones n).blocksFun i) :
    (ones n).embedding i ⟨0, h⟩ = ⟨i, lt_of_lt_of_le i.2 (ones n).length_le⟩ := by
  /-
    n : Nat
    i : Fin (Composition.ones n).length
    h : LT.lt 0 ((Composition.ones n).blocksFun i)
    ⊢ Eq (((Composition.ones n).embedding i) ⟨0, h⟩) ⟨↑i, ⋯⟩
  -/
  ext
  /-
    case h
    n : Nat
    i : Fin (Composition.ones n).length
    h : LT.lt 0 ((Composition.ones n).blocksFun i)
    ⊢ Eq ↑(((Composition.ones n).embedding i) ⟨0, h⟩) ↑⟨↑i, ⋯⟩
  -/
  simpa using i.2.le
  /-
    🎉 no goals
  -/


theorem eq_ones_iff {c : Composition n} : c = ones n ↔ ∀ i ∈ c.blocks, i = 1 := by
  /-
    n : Nat
    c : Composition n
    ⊢ Iff (Eq c (Composition.ones n)) (∀ (i : Nat), Membership.mem c.blocks i → Eq …
  -/
  constructor
    /-
      case mp
      n : Nat
      c : Composition n
      ⊢ Eq c (Composition.ones n) → ∀ (i : Nat), Membership.mem c.blocks i → Eq i 1
    -/
  · rintro rfl
    /-
      case mp
      n : Nat
      ⊢ ∀ (i : Nat), Membership.mem (Composition.ones n).blocks i → Eq i 1
    -/
    exact fun i => eq_of_mem_replicate
    /-
      🎉 no goals
    -/
    /-
      case mpr
      n : Nat
      c : Composition n
      ⊢ (∀ (i : Nat), Membership.mem c.blocks i → Eq i 1) → Eq c (Composition.ones n)
    -/
  · intro H
    /-
      case mpr
      n : Nat
      c : Composition n
      H : ∀ (i : Nat), Membership.mem c.blocks i → Eq i 1
      ⊢ Eq c (Composition.ones n)
    -/
    ext1
    /-
      case mpr.blocks
      n : Nat
      c : Composition n
      H : ∀ (i : Nat), Membership.mem c.blocks i → Eq i 1
      ⊢ Eq c.blocks (Composition.ones n).blocks
    -/
    have A : c.blocks = replicate c.blocks.length 1 := eq_replicate_of_mem H
    have : c.blocks.length = n := by
      conv_rhs => rw [← c.blocks_sum, A]
      simp
    /-
      case mpr.blocks
      n : Nat
      c : Composition n
      H : ∀ (i : Nat), Membership.mem c.blocks i → Eq i 1
      A : Eq c.blocks (List.replicate c.blocks.length 1)
      this : Eq c.blocks.length n
      ⊢ Eq c.blocks (Composition.ones n).blocks
    -/
    rw [A, this, ones_blocks]
    /-
      🎉 no goals
    -/


theorem ne_ones_iff {c : Composition n} : c ≠ ones n ↔ ∃ i ∈ c.blocks, 1 < i := by
  /-
    n : Nat
    c : Composition n
    ⊢ Iff (Ne c (Composition.ones n)) (Exists fun i => And (Membership.mem c.block …
  -/
  refine (not_congr eq_ones_iff).trans ?_
  /-
    n : Nat
    c : Composition n
    ⊢ Iff (Not (∀ (i : Nat), Membership.mem c.blocks i → Eq i 1)) (Exists fun i => …
  -/
  have : ∀ j ∈ c.blocks, j = 1 ↔ j ≤ 1 := fun j hj => by simp [le_antisymm_iff, c.one_le_blocks hj]
  /-
    n : Nat
    c : Composition n
    this : ∀ (j : Nat), Membership.mem c.blocks j → Iff (Eq j 1) (LE.le j 1)
    ⊢ Iff (Not (∀ (i : Nat), Membership.mem c.blocks i → Eq i 1)) (Exists fun i => …
  -/
  simp +contextual [this]
  /-
    🎉 no goals
  -/


theorem eq_ones_iff_length {c : Composition n} : c = ones n ↔ c.length = n := by
  /-
    n : Nat
    c : Composition n
    ⊢ Iff (Eq c (Composition.ones n)) (Eq c.length n)
  -/
  constructor
    /-
      case mp
      n : Nat
      c : Composition n
      ⊢ Eq c (Composition.ones n) → Eq c.length n
    -/
  · rintro rfl
    /-
      case mp
      n : Nat
      ⊢ Eq (Composition.ones n).length n
    -/
    exact ones_length n
    /-
      🎉 no goals
    -/
    /-
      case mpr
      n : Nat
      c : Composition n
      ⊢ Eq c.length n → Eq c (Composition.ones n)
    -/
  · contrapose
    /-
      case mpr
      n : Nat
      c : Composition n
      ⊢ Not (Eq c (Composition.ones n)) → Not (Eq c.length n)
    -/
    intro H length_n
    /-
      case mpr
      n : Nat
      c : Composition n
      H : Not (Eq c (Composition.ones n))
      length_n : Eq c.length n
      ⊢ False
    -/
    apply lt_irrefl n
    calc
      n = ∑ i : Fin c.length, 1 := by simp [length_n]
      _ < ∑ i : Fin c.length, c.blocksFun i := by
        {
        obtain ⟨i, hi, i_blocks⟩ : ∃ i ∈ c.blocks, 1 < i := ne_ones_iff.1 H
        rw [← ofFn_blocksFun, mem_ofFn c.blocksFun, Set.mem_range] at hi
        obtain ⟨j : Fin c.length, hj : c.blocksFun j = i⟩ := hi
        rw [← hj] at i_blocks
        exact Finset.sum_lt_sum (fun i _ => one_le_blocksFun c i) ⟨j, Finset.mem_univ _, i_blocks⟩
        }
      _ = n := c.sum_blocksFun


theorem eq_ones_iff_le_length {c : Composition n} : c = ones n ↔ n ≤ c.length := by
  /-
    n : Nat
    c : Composition n
    ⊢ Iff (Eq c (Composition.ones n)) (LE.le n c.length)
  -/
  simp [eq_ones_iff_length, le_antisymm_iff, c.length_le]
  /-
    🎉 no goals
  -/


/-- The composition made of a single block of size `n`. -/
def single (n : ℕ) (h : 0 < n) : Composition n :=
           /-
             n✝ : Nat
             c : Composition n✝
             n : Nat
             h : LT.lt 0 n
             ⊢ ∀ {i : Nat}, Membership.mem (List.cons n List.nil) i → LT.lt 0 i
           -/
           /-
             🎉 no goals
           -/
  ⟨[n], by simp [h], by simp⟩
                        /-
                          🎉 no goals
                        -/


@[simp]
theorem single_length {n : ℕ} (h : 0 < n) : (single n h).length = 1 :=
  rfl


@[simp]
theorem single_blocks {n : ℕ} (h : 0 < n) : (single n h).blocks = [n] :=
  rfl


@[simp]
theorem single_blocksFun {n : ℕ} (h : 0 < n) (i : Fin (single n h).length) :
                                       /-
                                         n : Nat
                                         h : LT.lt 0 n
                                         i : Fin (Composition.single n h).length
                                         ⊢ Eq ((Composition.single n h).blocksFun i) n
                                       -/
    (single n h).blocksFun i = n := by simp [blocksFun, single, blocks, i.2]
                                       /-
                                         🎉 no goals
                                       -/


@[simp]
theorem single_embedding {n : ℕ} (h : 0 < n) (i : Fin n) :
    ((single n h).embedding (0 : Fin 1)) i = i := by
  /-
    n : Nat
    h : LT.lt 0 n
    i : Fin n
    ⊢ Eq (((Composition.single n h).embedding 0) i) i
  -/
  ext
  /-
    case h
    n : Nat
    h : LT.lt 0 n
    i : Fin n
    ⊢ Eq ↑(((Composition.single n h).embedding 0) i) ↑i
  -/
  simp
  /-
    🎉 no goals
  -/


theorem eq_single_iff_length {n : ℕ} (h : 0 < n) {c : Composition n} :
    c = single n h ↔ c.length = 1 := by
  /-
    n : Nat
    h : LT.lt 0 n
    c : Composition n
    ⊢ Iff (Eq c (Composition.single n h)) (Eq c.length 1)
  -/
  constructor
    /-
      case mp
      n : Nat
      h : LT.lt 0 n
      c : Composition n
      ⊢ Eq c (Composition.single n h) → Eq c.length 1
    -/
  · intro H
    /-
      case mp
      n : Nat
      h : LT.lt 0 n
      c : Composition n
      H : Eq c (Composition.single n h)
      ⊢ Eq c.length 1
    -/
    rw [H]
    /-
      case mp
      n : Nat
      h : LT.lt 0 n
      c : Composition n
      H : Eq c (Composition.single n h)
      ⊢ Eq (Composition.single n h).length 1
    -/
    exact single_length h
    /-
      🎉 no goals
    -/
    /-
      case mpr
      n : Nat
      h : LT.lt 0 n
      c : Composition n
      ⊢ Eq c.length 1 → Eq c (Composition.single n h)
    -/
  · intro H
    /-
      case mpr
      n : Nat
      h : LT.lt 0 n
      c : Composition n
      H : Eq c.length 1
      ⊢ Eq c (Composition.single n h)
    -/
    ext1
    /-
      case mpr.blocks
      n : Nat
      h : LT.lt 0 n
      c : Composition n
      H : Eq c.length 1
      ⊢ Eq c.blocks (Composition.single n h).blocks
    -/
    have A : c.blocks.length = 1 := H ▸ c.blocks_length
    /-
      case mpr.blocks
      n : Nat
      h : LT.lt 0 n
      c : Composition n
      H : Eq c.length 1
      A : Eq c.blocks.length 1
      ⊢ Eq c.blocks (Composition.single n h).blocks
    -/
    have B : c.blocks.sum = n := c.blocks_sum
    /-
      case mpr.blocks
      n : Nat
      h : LT.lt 0 n
      c : Composition n
      H : Eq c.length 1
      A : Eq c.blocks.length 1
      B : Eq c.blocks.sum n
      ⊢ Eq c.blocks (Composition.single n h).blocks
    -/
    rw [eq_cons_of_length_one A] at B ⊢
    /-
      case mpr.blocks
      n : Nat
      h : LT.lt 0 n
      c : Composition n
      H : Eq c.length 1
      A : Eq c.blocks.length 1
      B : Eq (List.cons (c.blocks.get ⟨0, ⋯⟩) List.nil).sum n
      ⊢ Eq (List.cons (c.blocks.get ⟨0, ⋯⟩) List.nil) (Composition.single n h).blocks
    -/
    simpa [single_blocks] using B
    /-
      🎉 no goals
    -/


theorem ne_single_iff {n : ℕ} (hn : 0 < n) {c : Composition n} :
    c ≠ single n hn ↔ ∀ i, c.blocksFun i < n := by
  /-
    n : Nat
    hn : LT.lt 0 n
    c : Composition n
    ⊢ Iff (Ne c (Composition.single n hn)) (∀ (i : Fin c.length), LT.lt (c.blocksF …
  -/
  rw [← not_iff_not]
  /-
    n : Nat
    hn : LT.lt 0 n
    c : Composition n
    ⊢ Iff (Not (Ne c (Composition.single n hn))) (Not (∀ (i : Fin c.length), LT.lt …
  -/
  push_neg
  /-
    n : Nat
    hn : LT.lt 0 n
    c : Composition n
    ⊢ Iff (Eq c (Composition.single n hn)) (Exists fun i => LE.le n (c.blocksFun i))
  -/
  constructor
    /-
      case mp
      n : Nat
      hn : LT.lt 0 n
      c : Composition n
      ⊢ Eq c (Composition.single n hn) → Exists fun i => LE.le n (c.blocksFun i)
    -/
  · rintro rfl
    /-
      case mp
      n : Nat
      hn : LT.lt 0 n
      ⊢ Exists fun i => LE.le n ((Composition.single n hn).blocksFun i)
    -/
    exact ⟨⟨0, by simp⟩, by simp⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      n : Nat
      hn : LT.lt 0 n
      c : Composition n
      ⊢ (Exists fun i => LE.le n (c.blocksFun i)) → Eq c (Composition.single n hn)
    -/
  · rintro ⟨i, hi⟩
    /-
      case mpr.intro
      n : Nat
      hn : LT.lt 0 n
      c : Composition n
      i : Fin c.length
      hi : LE.le n (c.blocksFun i)
      ⊢ Eq c (Composition.single n hn)
    -/
    rw [eq_single_iff_length]
    have : ∀ j : Fin c.length, j = i := by
      intro j
      by_contra ji
      apply lt_irrefl (∑ k, c.blocksFun k)
      calc
        ∑ k, c.blocksFun k ≤ c.blocksFun i := by simp only [c.sum_blocksFun, hi]
        _ < ∑ k, c.blocksFun k :=
          Finset.single_lt_sum ji (Finset.mem_univ _) (Finset.mem_univ _) (c.one_le_blocksFun j)
            fun _ _ _ => zero_le _

    /-
      case mpr.intro
      n : Nat
      hn : LT.lt 0 n
      c : Composition n
      i : Fin c.length
      hi : LE.le n (c.blocksFun i)
      this : ∀ (j : Fin c.length), Eq j i
      ⊢ Eq c.length 1
    -/
    simpa using Fintype.card_eq_one_of_forall_eq this
    /-
      🎉 no goals
    -/


/-- Auxiliary for `List.splitWrtComposition`. -/
def splitWrtCompositionAux : List α → List ℕ → List (List α)
  | _, [] => []
  | l, n::ns =>
    let (l₁, l₂) := l.splitAt n
    l₁::splitWrtCompositionAux l₂ ns


/-- Given a list of length `n` and a composition `[i₁, ..., iₖ]` of `n`, split `l` into a list of
`k` lists corresponding to the blocks of the composition, of respective lengths `i₁`, ..., `iₖ`.
This makes sense mostly when `n = l.length`, but this is not necessary for the definition. -/
def splitWrtComposition (l : List α) (c : Composition n) : List (List α) :=
  splitWrtCompositionAux l c.blocks

-- Porting note: can't refer to subeqn in Lean 4 this way, and seems to definitionally simp
--attribute [local simp] splitWrtCompositionAux.equations._eqn_1


@[local simp]
theorem splitWrtCompositionAux_cons (l : List α) (n ns) :
    l.splitWrtCompositionAux (n::ns) = take n l::(drop n l).splitWrtCompositionAux ns := by
  /-
    α : Type u_1
    l : List α
    n : Nat
    ns : List Nat
    ⊢ Eq (l.splitWrtCompositionAux (List.cons n ns)) (List.cons (List.take n l) (( …
  -/
  simp [splitWrtCompositionAux]
  /-
    🎉 no goals
  -/


theorem length_splitWrtCompositionAux (l : List α) (ns) :
    length (l.splitWrtCompositionAux ns) = ns.length := by
    /-
      α : Type u_1
      l : List α
      ns : List Nat
      ⊢ Eq (l.splitWrtCompositionAux ns).length ns.length
    -/
    induction ns generalizing l
      /-
        case nil
        α : Type u_1
        l : List α
        ⊢ Eq (l.splitWrtCompositionAux List.nil).length List.nil.length
      -/
    · simp [splitWrtCompositionAux, *]
      /-
        🎉 no goals
      -/
      /-
        case cons
        α : Type u_1
        head✝ : Nat
        tail✝ : List Nat
        tail_ih✝ : ∀ (l : List α), Eq (l.splitWrtCompositionAux tail✝).length tail✝.le …
        l : List α
        ⊢ Eq (l.splitWrtCompositionAux (List.cons head✝ tail✝)).length (List.cons head …
      -/
    · simp [*]
      /-
        🎉 no goals
      -/


/-- When one splits a list along a composition `c`, the number of sublists thus created is
`c.length`. -/
@[simp]
theorem length_splitWrtComposition (l : List α) (c : Composition n) :
    length (l.splitWrtComposition c) = c.length :=
  length_splitWrtCompositionAux _ _



theorem map_length_splitWrtCompositionAux {ns : List ℕ} :
    ∀ {l : List α}, ns.sum ≤ l.length → map length (l.splitWrtCompositionAux ns) = ns := by
  /-
    α : Type u_1
    ns : List Nat
    ⊢ ∀ {l : List α}, LE.le ns.sum l.length → Eq (List.map List.length (l.splitWrt …
  -/
  induction' ns with n ns IH <;> intro l h <;> simp at h
    /-
      case nil
      α : Type u_1
      l : List α
      h : True
      ⊢ Eq (List.map List.length (l.splitWrtCompositionAux List.nil)) List.nil
    -/
  · simp [splitWrtCompositionAux]
    /-
      🎉 no goals
    -/
  /-
    case cons
    α : Type u_1
    n : Nat
    ns : List Nat
    IH : ∀ {l : List α}, LE.le ns.sum l.length → Eq (List.map List.length (l.split …
    l : List α
    h : LE.le (HAdd.hAdd n ns.sum) l.length
    ⊢ Eq (List.map List.length (l.splitWrtCompositionAux (List.cons n ns))) (List. …
  -/
  have := le_trans (Nat.le_add_right _ _) h
  /-
    case cons
    α : Type u_1
    n : Nat
    ns : List Nat
    IH : ∀ {l : List α}, LE.le ns.sum l.length → Eq (List.map List.length (l.split …
    l : List α
    h : LE.le (HAdd.hAdd n ns.sum) l.length
    this : LE.le n l.length
    ⊢ Eq (List.map List.length (l.splitWrtCompositionAux (List.cons n ns))) (List. …
  -/
  simp only [splitWrtCompositionAux_cons, this]; dsimp
  /-
    case cons
    α : Type u_1
    n : Nat
    ns : List Nat
    IH : ∀ {l : List α}, LE.le ns.sum l.length → Eq (List.map List.length (l.split …
    l : List α
    h : LE.le (HAdd.hAdd n ns.sum) l.length
    this : LE.le n l.length
    ⊢ Eq (List.cons (List.take n l).length (List.map List.length ((List.drop n l). …
  -/
  rw [length_take, IH] <;> simp [length_drop]
    /-
      case cons
      α : Type u_1
      n : Nat
      ns : List Nat
      IH : ∀ {l : List α}, LE.le ns.sum l.length → Eq (List.map List.length (l.split …
      l : List α
      h : LE.le (HAdd.hAdd n ns.sum) l.length
      this : LE.le n l.length
      ⊢ LE.le n l.length
    -/
  · assumption
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      n : Nat
      ns : List Nat
      IH : ∀ {l : List α}, LE.le ns.sum l.length → Eq (List.map List.length (l.split …
      l : List α
      h : LE.le (HAdd.hAdd n ns.sum) l.length
      this : LE.le n l.length
      ⊢ LE.le ns.sum (HSub.hSub l.length n)
    -/
  · exact le_tsub_of_add_le_left h
    /-
      🎉 no goals
    -/


/-- When one splits a list along a composition `c`, the lengths of the sublists thus created are
given by the block sizes in `c`. -/
theorem map_length_splitWrtComposition (l : List α) (c : Composition l.length) :
    map length (l.splitWrtComposition c) = c.blocks :=
  map_length_splitWrtCompositionAux (le_of_eq c.blocks_sum)


theorem length_pos_of_mem_splitWrtComposition {l l' : List α} {c : Composition l.length}
    (h : l' ∈ l.splitWrtComposition c) : 0 < length l' := by
  have : l'.length ∈ (l.splitWrtComposition c).map List.length :=
    List.mem_map_of_mem List.length h
  /-
    α : Type u_1
    l l' : List α
    c : Composition l.length
    h : Membership.mem (l.splitWrtComposition c) l'
    this : Membership.mem (List.map List.length (l.splitWrtComposition c)) l'.length
    ⊢ LT.lt 0 l'.length
  -/
  rw [map_length_splitWrtComposition] at this
  /-
    α : Type u_1
    l l' : List α
    c : Composition l.length
    h : Membership.mem (l.splitWrtComposition c) l'
    this : Membership.mem c.blocks l'.length
    ⊢ LT.lt 0 l'.length
  -/
  exact c.blocks_pos this
  /-
    🎉 no goals
  -/


theorem sum_take_map_length_splitWrtComposition (l : List α) (c : Composition l.length) (i : ℕ) :
    (((l.splitWrtComposition c).map length).take i).sum = c.sizeUpTo i := by
  /-
    α : Type u_1
    l : List α
    c : Composition l.length
    i : Nat
    ⊢ Eq (List.take i (List.map List.length (l.splitWrtComposition c))).sum (c.siz …
  -/
  congr
  /-
    case e_a.e_a
    α : Type u_1
    l : List α
    c : Composition l.length
    i : Nat
    ⊢ Eq (List.map List.length (l.splitWrtComposition c)) c.blocks
  -/
  exact map_length_splitWrtComposition l c
  /-
    🎉 no goals
  -/


theorem getElem_splitWrtCompositionAux (l : List α) (ns : List ℕ) {i : ℕ}
    (hi : i < (l.splitWrtCompositionAux ns).length) :
    (l.splitWrtCompositionAux ns)[i] =
      (l.take (ns.take (i + 1)).sum).drop (ns.take i).sum := by
  /-
    α : Type u_1
    l : List α
    ns : List Nat
    i : Nat
    hi : LT.lt i (l.splitWrtCompositionAux ns).length
    ⊢ Eq (GetElem.getElem (l.splitWrtCompositionAux ns) i hi) (List.drop (List.tak …
  -/
  induction' ns with n ns IH generalizing l i
    /-
      case nil
      α : Type u_1
      l : List α
      i : Nat
      hi : LT.lt i (l.splitWrtCompositionAux List.nil).length
      ⊢ Eq (GetElem.getElem (l.splitWrtCompositionAux List.nil) i hi) (List.drop (Li …
    -/
  · cases hi
    /-
      🎉 no goals
    -/
  /-
    case cons
    α : Type u_1
    n : Nat
    ns : List Nat
    IH : ∀ (l : List α) {i : Nat} (hi : LT.lt i (l.splitWrtCompositionAux ns).leng …
    l : List α
    i : Nat
    hi : LT.lt i (l.splitWrtCompositionAux (List.cons n ns)).length
    ⊢ Eq (GetElem.getElem (l.splitWrtCompositionAux (List.cons n ns)) i hi) (List. …
  -/
  cases' i with i
    /-
      case cons.zero
      α : Type u_1
      n : Nat
      ns : List Nat
      IH : ∀ (l : List α) {i : Nat} (hi : LT.lt i (l.splitWrtCompositionAux ns).leng …
      l : List α
      hi : LT.lt 0 (l.splitWrtCompositionAux (List.cons n ns)).length
      ⊢ Eq (GetElem.getElem (l.splitWrtCompositionAux (List.cons n ns)) 0 hi) (List. …
    -/
  · rw [Nat.add_zero, List.take_zero, sum_nil]
    /-
      case cons.zero
      α : Type u_1
      n : Nat
      ns : List Nat
      IH : ∀ (l : List α) {i : Nat} (hi : LT.lt i (l.splitWrtCompositionAux ns).leng …
      l : List α
      hi : LT.lt 0 (l.splitWrtCompositionAux (List.cons n ns)).length
      ⊢ Eq (GetElem.getElem (l.splitWrtCompositionAux (List.cons n ns)) 0 hi) (List. …
    -/
    simp
    /-
      🎉 no goals
    -/
  · simp only [splitWrtCompositionAux, getElem_cons_succ, IH, take,
        sum_cons, Nat.add_eq, add_zero, splitAt_eq, drop_take, drop_drop]
    /-
      case cons.succ
      α : Type u_1
      n : Nat
      ns : List Nat
      IH : ∀ (l : List α) {i : Nat} (hi : LT.lt i (l.splitWrtCompositionAux ns).leng …
      l : List α
      i : Nat
      hi : LT.lt (HAdd.hAdd i 1) (l.splitWrtCompositionAux (List.cons n ns)).length
      ⊢ Eq (List.take (HSub.hSub (List.take (HAdd.hAdd i 1) ns).sum (List.take i ns) …
    -/
    rw [Nat.add_sub_add_left]
    /-
      🎉 no goals
    -/


/-- The `i`-th sublist in the splitting of a list `l` along a composition `c`, is the slice of `l`
between the indices `c.sizeUpTo i` and `c.sizeUpTo (i+1)`, i.e., the indices in the `i`-th
block of the composition. -/
theorem getElem_splitWrtComposition' (l : List α) (c : Composition n) {i : ℕ}
    (hi : i < (l.splitWrtComposition c).length) :
    (l.splitWrtComposition c)[i] = (l.take (c.sizeUpTo (i + 1))).drop (c.sizeUpTo i) :=
  getElem_splitWrtCompositionAux _ _ hi

-- Porting note: restatement of `get_splitWrtComposition`

theorem getElem_splitWrtComposition (l : List α) (c : Composition n)
    (i : Nat) (h : i < (l.splitWrtComposition c).length) :
    (l.splitWrtComposition c)[i] = (l.take (c.sizeUpTo (i + 1))).drop (c.sizeUpTo i) :=
  getElem_splitWrtComposition' _ _ h


@[deprecated getElem_splitWrtCompositionAux (since := "2024-06-12")]
theorem get_splitWrtCompositionAux (l : List α) (ns : List ℕ) {i : ℕ} (hi) :
    (l.splitWrtCompositionAux ns).get ⟨i, hi⟩  =
      (l.take (ns.take (i + 1)).sum).drop (ns.take i).sum := by
  /-
    α : Type u_1
    l : List α
    ns : List Nat
    i : Nat
    hi : LT.lt i (l.splitWrtCompositionAux ns).length
    ⊢ Eq ((l.splitWrtCompositionAux ns).get ⟨i, hi⟩) (List.drop (List.take i ns).s …
  -/
  simp [getElem_splitWrtCompositionAux]
  /-
    🎉 no goals
  -/


/-- The `i`-th sublist in the splitting of a list `l` along a composition `c`, is the slice of `l`
between the indices `c.sizeUpTo i` and `c.sizeUpTo (i+1)`, i.e., the indices in the `i`-th
block of the composition. -/
@[deprecated getElem_splitWrtComposition' (since := "2024-06-12")]
theorem get_splitWrtComposition' (l : List α) (c : Composition n) {i : ℕ}
    (hi : i < (l.splitWrtComposition c).length) :
    (l.splitWrtComposition c).get ⟨i, hi⟩ = (l.take (c.sizeUpTo (i + 1))).drop (c.sizeUpTo i) := by
  /-
    n : Nat
    α : Type u_1
    l : List α
    c : Composition n
    i : Nat
    hi : LT.lt i (l.splitWrtComposition c).length
    ⊢ Eq ((l.splitWrtComposition c).get ⟨i, hi⟩) (List.drop (c.sizeUpTo i) (List.t …
  -/
  simp [getElem_splitWrtComposition']
  /-
    🎉 no goals
  -/

-- Porting note: restatement of `get_splitWrtComposition`

@[deprecated getElem_splitWrtComposition (since := "2024-06-12")]
theorem get_splitWrtComposition (l : List α) (c : Composition n)
    (i : Fin (l.splitWrtComposition c).length) :
    get (l.splitWrtComposition c) i = (l.take (c.sizeUpTo (i + 1))).drop (c.sizeUpTo i) := by
  /-
    n : Nat
    α : Type u_1
    l : List α
    c : Composition n
    i : Fin (l.splitWrtComposition c).length
    ⊢ Eq ((l.splitWrtComposition c).get i) (List.drop (c.sizeUpTo ↑i) (List.take ( …
  -/
  simp [getElem_splitWrtComposition]
  /-
    🎉 no goals
  -/


theorem flatten_splitWrtCompositionAux {ns : List ℕ} :
    ∀ {l : List α}, ns.sum = l.length → (l.splitWrtCompositionAux ns).flatten = l := by
  /-
    α : Type u_1
    ns : List Nat
    ⊢ ∀ {l : List α}, Eq ns.sum l.length → Eq (l.splitWrtCompositionAux ns).flatte …
  -/
  induction' ns with n ns IH <;> intro l h <;> simp at h
    /-
      case nil
      α : Type u_1
      l : List α
      h : Eq 0 l.length
      ⊢ Eq (l.splitWrtCompositionAux List.nil).flatten l
    -/
  · exact (length_eq_zero.1 h.symm).symm
    /-
      🎉 no goals
    -/
  /-
    case cons
    α : Type u_1
    n : Nat
    ns : List Nat
    IH : ∀ {l : List α}, Eq ns.sum l.length → Eq (l.splitWrtCompositionAux ns).fla …
    l : List α
    h : Eq (HAdd.hAdd n ns.sum) l.length
    ⊢ Eq (l.splitWrtCompositionAux (List.cons n ns)).flatten l
  -/
  simp only [splitWrtCompositionAux_cons]; dsimp
  /-
    case cons
    α : Type u_1
    n : Nat
    ns : List Nat
    IH : ∀ {l : List α}, Eq ns.sum l.length → Eq (l.splitWrtCompositionAux ns).fla …
    l : List α
    h : Eq (HAdd.hAdd n ns.sum) l.length
    ⊢ Eq (HAppend.hAppend (List.take n l) ((List.drop n l).splitWrtCompositionAux  …
  -/
  rw [IH]
    /-
      case cons
      α : Type u_1
      n : Nat
      ns : List Nat
      IH : ∀ {l : List α}, Eq ns.sum l.length → Eq (l.splitWrtCompositionAux ns).fla …
      l : List α
      h : Eq (HAdd.hAdd n ns.sum) l.length
      ⊢ Eq (HAppend.hAppend (List.take n l) (List.drop n l)) l
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      n : Nat
      ns : List Nat
      IH : ∀ {l : List α}, Eq ns.sum l.length → Eq (l.splitWrtCompositionAux ns).fla …
      l : List α
      h : Eq (HAdd.hAdd n ns.sum) l.length
      ⊢ Eq ns.sum (List.drop n l).length
    -/
  · rw [length_drop, ← h, add_tsub_cancel_left]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-15")]
alias join_splitWrtCompositionAux := flatten_splitWrtCompositionAux


/-- If one splits a list along a composition, and then flattens the sublists, one gets back the
original list. -/
@[simp]
theorem flatten_splitWrtComposition (l : List α) (c : Composition l.length) :
    (l.splitWrtComposition c).flatten = l :=
  flatten_splitWrtCompositionAux c.blocks_sum


@[deprecated (since := "2024-10-15")] alias join_splitWrtComposition := flatten_splitWrtComposition


/-- If one joins a list of lists and then splits the flattening along the right composition,
one gets back the original list of lists. -/
@[simp]
theorem splitWrtComposition_flatten (L : List (List α)) (c : Composition L.flatten.length)
    (h : map length L = c.blocks) : splitWrtComposition (flatten L) c = L := by
  simp only [eq_self_iff_true, and_self_iff, eq_iff_flatten_eq, flatten_splitWrtComposition,
    map_length_splitWrtComposition, h]


@[deprecated (since := "2024-10-15")]
alias splitWrtComposition_join := splitWrtComposition_flatten


/-- Bijection between compositions of `n` and subsets of `{0, ..., n-2}`, defined by
considering the restriction of the subset to `{1, ..., n-1}` and shifting to the left by one. -/
def compositionAsSetEquiv (n : ℕ) : CompositionAsSet n ≃ Finset (Fin (n - 1)) where
  toFun c :=
    { i : Fin (n - 1) |
        (⟨1 + (i : ℕ), by
              /-
                n✝ n : Nat
                c : CompositionAsSet n
                i : Fin (HSub.hSub n 1)
                ⊢ LT.lt (HAdd.hAdd 1 ↑i) n.succ
              -/
              apply (add_lt_add_left i.is_lt 1).trans_le
              /-
                n✝ n : Nat
                c : CompositionAsSet n
                i : Fin (HSub.hSub n 1)
                ⊢ LE.le (HAdd.hAdd 1 (HSub.hSub n 1)) n.succ
              -/
              rw [Nat.succ_eq_add_one, add_comm]
              /-
                n✝ n : Nat
                c : CompositionAsSet n
                i : Fin (HSub.hSub n 1)
                ⊢ LE.le (HAdd.hAdd (HSub.hSub n 1) 1) (HAdd.hAdd n 1)
              -/
              exact add_le_add (Nat.sub_le n 1) (le_refl 1)⟩ :
              /-
                🎉 no goals
              -/
            Fin n.succ) ∈
          c.boundaries }.toFinset
  invFun s :=
    { boundaries :=
        { i : Fin n.succ |
            i = 0 ∨ i = Fin.last n ∨ ∃ (j : Fin (n - 1)) (_hj : j ∈ s), (i : ℕ) = j + 1 }.toFinset
                     /-
                       n✝ n : Nat
                       s : Finset (Fin (HSub.hSub n 1))
                       ⊢ Membership.mem (setOf fun i => Or (Eq i 0) (Or (Eq i (Fin.last n)) (Exists f …
                     -/
      zero_mem := by simp
                     /-
                       🎉 no goals
                     -/
                        /-
                          n✝ n : Nat
                          s : Finset (Fin (HSub.hSub n 1))
                          ⊢ Membership.mem (setOf fun i => Or (Eq i 0) (Or (Eq i (Fin.last n)) (Exists f …
                        -/
      getLast_mem := by simp }
                        /-
                          🎉 no goals
                        -/
  left_inv := by
    /-
      n✝ n : Nat
      ⊢ Function.LeftInverse (fun s => { boundaries := (setOf fun i => Or (Eq i 0) ( …
    -/
    intro c
    /-
      n✝ n : Nat
      c : CompositionAsSet n
      ⊢ Eq ((fun s => { boundaries := (setOf fun i => Or (Eq i 0) (Or (Eq i (Fin.las …
    -/
    ext i
    simp only [add_comm, Set.toFinset_setOf, Finset.mem_univ,
     forall_true_left, Finset.mem_filter, true_and, exists_prop]
    /-
      case boundaries.h
      n✝ n : Nat
      c : CompositionAsSet n
      i : Fin n.succ
      ⊢ Iff (Or (Eq i 0) (Or (Eq i (Fin.last n)) (Exists fun j => And (Membership.me …
    -/
    constructor
      /-
        case boundaries.h.mp
        n✝ n : Nat
        c : CompositionAsSet n
        i : Fin n.succ
        ⊢ Or (Eq i 0) (Or (Eq i (Fin.last n)) (Exists fun j => And (Membership.mem c.b …
      -/
    · rintro (rfl | rfl | ⟨j, hj1, hj2⟩)
        /-
          case boundaries.h.mp.inl
          n✝ n : Nat
          c : CompositionAsSet n
          ⊢ Membership.mem c.boundaries 0
        -/
      · exact c.zero_mem
        /-
          🎉 no goals
        -/
        /-
          case boundaries.h.mp.inr.inl
          n✝ n : Nat
          c : CompositionAsSet n
          ⊢ Membership.mem c.boundaries (Fin.last n)
        -/
      · exact c.getLast_mem
        /-
          🎉 no goals
        -/
        /-
          case boundaries.h.mp.inr.inr.intro.intro
          n✝ n : Nat
          c : CompositionAsSet n
          i : Fin n.succ
          j : Fin (HSub.hSub n 1)
          hj1 : Membership.mem c.boundaries ⟨HAdd.hAdd (↑j) 1, ⋯⟩
          hj2 : Eq (↑i) (HAdd.hAdd (↑j) 1)
          ⊢ Membership.mem c.boundaries i
        -/
      · convert hj1
        /-
          🎉 no goals
        -/
      /-
        case boundaries.h.mpr
        n✝ n : Nat
        c : CompositionAsSet n
        i : Fin n.succ
        ⊢ Membership.mem c.boundaries i → Or (Eq i 0) (Or (Eq i (Fin.last n)) (Exists  …
      -/
    · simp only [or_iff_not_imp_left]
      /-
        case boundaries.h.mpr
        n✝ n : Nat
        c : CompositionAsSet n
        i : Fin n.succ
        ⊢ Membership.mem c.boundaries i → Not (Eq i 0) → Not (Eq i (Fin.last n)) → Exi …
      -/
      intro i_mem i_ne_zero i_ne_last
      simp? [Fin.ext_iff] at i_ne_zero i_ne_last says
        simp only [Nat.succ_eq_add_one, Fin.ext_iff, Fin.val_zero, Fin.val_last]
          at i_ne_zero i_ne_last
      have A : (1 + (i - 1) : ℕ) = (i : ℕ) := by
        rw [add_comm]
        exact Nat.succ_pred_eq_of_pos (pos_iff_ne_zero.mpr i_ne_zero)
      /-
        case boundaries.h.mpr
        n✝ n : Nat
        c : CompositionAsSet n
        i : Fin n.succ
        i_mem : Membership.mem c.boundaries i
        i_ne_zero : Not (Eq (↑i) 0)
        i_ne_last : Not (Eq (↑i) n)
        A : Eq (HAdd.hAdd 1 (HSub.hSub (↑i) 1)) ↑i
        ⊢ Exists fun j => And (Membership.mem c.boundaries ⟨HAdd.hAdd (↑j) 1, ⋯⟩) (Eq  …
      -/
      refine ⟨⟨i - 1, ?_⟩, ?_, ?_⟩
        /-
          case boundaries.h.mpr.refine_1
          n✝ n : Nat
          c : CompositionAsSet n
          i : Fin n.succ
          i_mem : Membership.mem c.boundaries i
          i_ne_zero : Not (Eq (↑i) 0)
          i_ne_last : Not (Eq (↑i) n)
          A : Eq (HAdd.hAdd 1 (HSub.hSub (↑i) 1)) ↑i
          ⊢ LT.lt (HSub.hSub (↑i) 1) (HSub.hSub n 1)
        -/
      · have : (i : ℕ) < n + 1 := i.2
        simp? [Nat.lt_succ_iff_lt_or_eq, i_ne_last] at this says
          simp only [Nat.succ_eq_add_one, Nat.lt_succ_iff_lt_or_eq, i_ne_last, or_false] at this
        /-
          case boundaries.h.mpr.refine_1
          n✝ n : Nat
          c : CompositionAsSet n
          i : Fin n.succ
          i_mem : Membership.mem c.boundaries i
          i_ne_zero : Not (Eq (↑i) 0)
          i_ne_last : Not (Eq (↑i) n)
          A : Eq (HAdd.hAdd 1 (HSub.hSub (↑i) 1)) ↑i
          this : LT.lt (↑i) n
          ⊢ LT.lt (HSub.hSub (↑i) 1) (HSub.hSub n 1)
        -/
        exact Nat.pred_lt_pred i_ne_zero this
        /-
          🎉 no goals
        -/
        /-
          case boundaries.h.mpr.refine_2
          n✝ n : Nat
          c : CompositionAsSet n
          i : Fin n.succ
          i_mem : Membership.mem c.boundaries i
          i_ne_zero : Not (Eq (↑i) 0)
          i_ne_last : Not (Eq (↑i) n)
          A : Eq (HAdd.hAdd 1 (HSub.hSub (↑i) 1)) ↑i
          ⊢ Membership.mem c.boundaries ⟨HAdd.hAdd (↑⟨HSub.hSub (↑i) 1, ⋯⟩) 1, ⋯⟩
        -/
      · convert i_mem
        /-
          case h.e'_5.h.e'_2
          n✝ n : Nat
          c : CompositionAsSet n
          i : Fin n.succ
          i_mem : Membership.mem c.boundaries i
          i_ne_zero : Not (Eq (↑i) 0)
          i_ne_last : Not (Eq (↑i) n)
          A : Eq (HAdd.hAdd 1 (HSub.hSub (↑i) 1)) ↑i
          ⊢ Eq (HAdd.hAdd (↑⟨HSub.hSub (↑i) 1, ⋯⟩) 1) ↑i
        -/
        simp only
        /-
          case h.e'_5.h.e'_2
          n✝ n : Nat
          c : CompositionAsSet n
          i : Fin n.succ
          i_mem : Membership.mem c.boundaries i
          i_ne_zero : Not (Eq (↑i) 0)
          i_ne_last : Not (Eq (↑i) n)
          A : Eq (HAdd.hAdd 1 (HSub.hSub (↑i) 1)) ↑i
          ⊢ Eq (HAdd.hAdd (HSub.hSub (↑i) 1) 1) ↑i
        -/
        rwa [add_comm]
        /-
          🎉 no goals
        -/
        /-
          case boundaries.h.mpr.refine_3
          n✝ n : Nat
          c : CompositionAsSet n
          i : Fin n.succ
          i_mem : Membership.mem c.boundaries i
          i_ne_zero : Not (Eq (↑i) 0)
          i_ne_last : Not (Eq (↑i) n)
          A : Eq (HAdd.hAdd 1 (HSub.hSub (↑i) 1)) ↑i
          ⊢ Eq (↑i) (HAdd.hAdd (↑⟨HSub.hSub (↑i) 1, ⋯⟩) 1)
        -/
      · simp only
        /-
          case boundaries.h.mpr.refine_3
          n✝ n : Nat
          c : CompositionAsSet n
          i : Fin n.succ
          i_mem : Membership.mem c.boundaries i
          i_ne_zero : Not (Eq (↑i) 0)
          i_ne_last : Not (Eq (↑i) n)
          A : Eq (HAdd.hAdd 1 (HSub.hSub (↑i) 1)) ↑i
          ⊢ Eq (↑i) (HAdd.hAdd (HSub.hSub (↑i) 1) 1)
        -/
        symm
        /-
          case boundaries.h.mpr.refine_3
          n✝ n : Nat
          c : CompositionAsSet n
          i : Fin n.succ
          i_mem : Membership.mem c.boundaries i
          i_ne_zero : Not (Eq (↑i) 0)
          i_ne_last : Not (Eq (↑i) n)
          A : Eq (HAdd.hAdd 1 (HSub.hSub (↑i) 1)) ↑i
          ⊢ Eq (HAdd.hAdd (HSub.hSub (↑i) 1) 1) ↑i
        -/
        rwa [add_comm]
        /-
          🎉 no goals
        -/
  right_inv := by
    /-
      n✝ n : Nat
      ⊢ Function.RightInverse (fun s => { boundaries := (setOf fun i => Or (Eq i 0)  …
    -/
    intro s
    /-
      n✝ n : Nat
      s : Finset (Fin (HSub.hSub n 1))
      ⊢ Eq ((fun c => (setOf fun i => Membership.mem c.boundaries ⟨HAdd.hAdd 1 ↑i, ⋯ …
    -/
    ext i
    have : 1 + (i : ℕ) ≠ n := by
      apply ne_of_lt
      convert add_lt_add_left i.is_lt 1
      rw [add_comm]
      apply (Nat.succ_pred_eq_of_pos _).symm
      exact (zero_le i.val).trans_lt (i.2.trans_le (Nat.sub_le n 1))
    simp only [add_comm, Fin.ext_iff, Fin.val_zero, Fin.val_last, exists_prop, Set.toFinset_setOf,
      Finset.mem_univ, forall_true_left, Finset.mem_filter, add_eq_zero, and_false,
      add_left_inj, false_or, true_and, reduceCtorEq]
    /-
      case h
      n✝ n : Nat
      s : Finset (Fin (HSub.hSub n 1))
      i : Fin (HSub.hSub n 1)
      this : Ne (HAdd.hAdd 1 ↑i) n
      ⊢ Iff (Or (Eq (HAdd.hAdd (↑i) 1) n) (Exists fun j => And (Membership.mem s j)  …
    -/
    erw [Set.mem_setOf_eq]
    /-
      case h
      n✝ n : Nat
      s : Finset (Fin (HSub.hSub n 1))
      i : Fin (HSub.hSub n 1)
      this : Ne (HAdd.hAdd 1 ↑i) n
      ⊢ Iff (Or (Eq (HAdd.hAdd (↑i) 1) n) (Exists fun j => And (Membership.mem s j)  …
    -/
    simp only [Finset.mem_val]
    /-
      case h
      n✝ n : Nat
      s : Finset (Fin (HSub.hSub n 1))
      i : Fin (HSub.hSub n 1)
      this : Ne (HAdd.hAdd 1 ↑i) n
      ⊢ Iff (Or (Eq (HAdd.hAdd (↑i) 1) n) (Exists fun j => And (Membership.mem s j)  …
    -/
    constructor
      /-
        case h.mp
        n✝ n : Nat
        s : Finset (Fin (HSub.hSub n 1))
        i : Fin (HSub.hSub n 1)
        this : Ne (HAdd.hAdd 1 ↑i) n
        ⊢ Or (Eq (HAdd.hAdd (↑i) 1) n) (Exists fun j => And (Membership.mem s j) (Eq ↑ …
      -/
    · intro h
      /-
        case h.mp
        n✝ n : Nat
        s : Finset (Fin (HSub.hSub n 1))
        i : Fin (HSub.hSub n 1)
        this : Ne (HAdd.hAdd 1 ↑i) n
        h : Or (Eq (HAdd.hAdd (↑i) 1) n) (Exists fun j => And (Membership.mem s j) (Eq …
        ⊢ Membership.mem s i
      -/
      cases' h with n h
        /-
          case h.mp.inl
          n✝¹ n✝ : Nat
          s : Finset (Fin (HSub.hSub n✝ 1))
          i : Fin (HSub.hSub n✝ 1)
          this : Ne (HAdd.hAdd 1 ↑i) n✝
          n : Eq (HAdd.hAdd (↑i) 1) n✝
          ⊢ Membership.mem s i
        -/
      · rw [add_comm] at this
        /-
          case h.mp.inl
          n✝¹ n✝ : Nat
          s : Finset (Fin (HSub.hSub n✝ 1))
          i : Fin (HSub.hSub n✝ 1)
          this : Ne (HAdd.hAdd (↑i) 1) n✝
          n : Eq (HAdd.hAdd (↑i) 1) n✝
          ⊢ Membership.mem s i
        -/
        contradiction
        /-
          🎉 no goals
        -/
        /-
          case h.mp.inr
          n✝ n : Nat
          s : Finset (Fin (HSub.hSub n 1))
          i : Fin (HSub.hSub n 1)
          this : Ne (HAdd.hAdd 1 ↑i) n
          h : Exists fun j => And (Membership.mem s j) (Eq ↑i ↑j)
          ⊢ Membership.mem s i
        -/
      · cases' h with w h; cases' h with h₁ h₂
        /-
          case h.mp.inr.intro.intro
          n✝ n : Nat
          s : Finset (Fin (HSub.hSub n 1))
          i : Fin (HSub.hSub n 1)
          this : Ne (HAdd.hAdd 1 ↑i) n
          w : Fin (HSub.hSub n 1)
          h₁ : Membership.mem s w
          h₂ : Eq ↑i ↑w
          ⊢ Membership.mem s i
        -/
        rw [← Fin.ext_iff] at h₂
        /-
          case h.mp.inr.intro.intro
          n✝ n : Nat
          s : Finset (Fin (HSub.hSub n 1))
          i : Fin (HSub.hSub n 1)
          this : Ne (HAdd.hAdd 1 ↑i) n
          w : Fin (HSub.hSub n 1)
          h₁ : Membership.mem s w
          h₂ : Eq i w
          ⊢ Membership.mem s i
        -/
        rwa [h₂]
        /-
          🎉 no goals
        -/
      /-
        case h.mpr
        n✝ n : Nat
        s : Finset (Fin (HSub.hSub n 1))
        i : Fin (HSub.hSub n 1)
        this : Ne (HAdd.hAdd 1 ↑i) n
        ⊢ Membership.mem s i → Or (Eq (HAdd.hAdd (↑i) 1) n) (Exists fun j => And (Memb …
      -/
    · intro h
      /-
        case h.mpr
        n✝ n : Nat
        s : Finset (Fin (HSub.hSub n 1))
        i : Fin (HSub.hSub n 1)
        this : Ne (HAdd.hAdd 1 ↑i) n
        h : Membership.mem s i
        ⊢ Or (Eq (HAdd.hAdd (↑i) 1) n) (Exists fun j => And (Membership.mem s j) (Eq ↑ …
      -/
      apply Or.inr
      /-
        case h.mpr.h
        n✝ n : Nat
        s : Finset (Fin (HSub.hSub n 1))
        i : Fin (HSub.hSub n 1)
        this : Ne (HAdd.hAdd 1 ↑i) n
        h : Membership.mem s i
        ⊢ Exists fun j => And (Membership.mem s j) (Eq ↑i ↑j)
      -/
      use i, h
      /-
        🎉 no goals
      -/


instance compositionAsSetFintype (n : ℕ) : Fintype (CompositionAsSet n) :=
  Fintype.ofEquiv _ (compositionAsSetEquiv n).symm


theorem compositionAsSet_card (n : ℕ) : Fintype.card (CompositionAsSet n) = 2 ^ (n - 1) := by
  /-
    n : Nat
    ⊢ Eq (Fintype.card (CompositionAsSet n)) (HPow.hPow 2 (HSub.hSub n 1))
  -/
  have : Fintype.card (Finset (Fin (n - 1))) = 2 ^ (n - 1) := by simp
  /-
    n : Nat
    this : Eq (Fintype.card (Finset (Fin (HSub.hSub n 1)))) (HPow.hPow 2 (HSub.hSu …
    ⊢ Eq (Fintype.card (CompositionAsSet n)) (HPow.hPow 2 (HSub.hSub n 1))
  -/
  rw [← this]
  /-
    n : Nat
    this : Eq (Fintype.card (Finset (Fin (HSub.hSub n 1)))) (HPow.hPow 2 (HSub.hSu …
    ⊢ Eq (Fintype.card (CompositionAsSet n)) (Fintype.card (Finset (Fin (HSub.hSub …
  -/
  exact Fintype.card_congr (compositionAsSetEquiv n)
  /-
    🎉 no goals
  -/


theorem boundaries_nonempty : c.boundaries.Nonempty :=
  ⟨0, c.zero_mem⟩


theorem card_boundaries_pos : 0 < Finset.card c.boundaries :=
  Finset.card_pos.mpr c.boundaries_nonempty


/-- Number of blocks in a `CompositionAsSet`. -/
def length : ℕ :=
  Finset.card c.boundaries - 1


theorem card_boundaries_eq_succ_length : c.boundaries.card = c.length + 1 :=
  (tsub_eq_iff_eq_add_of_le (Nat.succ_le_of_lt c.card_boundaries_pos)).mp rfl


theorem length_lt_card_boundaries : c.length < c.boundaries.card := by
  /-
    n : Nat
    c : CompositionAsSet n
    ⊢ LT.lt c.length c.boundaries.card
  -/
  rw [c.card_boundaries_eq_succ_length]
  /-
    n : Nat
    c : CompositionAsSet n
    ⊢ LT.lt c.length (HAdd.hAdd c.length 1)
  -/
  exact Nat.lt_add_one _
  /-
    🎉 no goals
  -/


theorem lt_length (i : Fin c.length) : (i : ℕ) + 1 < c.boundaries.card :=
  lt_tsub_iff_right.mp i.2


theorem lt_length' (i : Fin c.length) : (i : ℕ) < c.boundaries.card :=
  lt_of_le_of_lt (Nat.le_succ i) (c.lt_length i)


/-- Canonical increasing bijection from `Fin c.boundaries.card` to `c.boundaries`. -/
def boundary : Fin c.boundaries.card ↪o Fin (n + 1) :=
  c.boundaries.orderEmbOfFin rfl


@[simp]
theorem boundary_zero : (c.boundary ⟨0, c.card_boundaries_pos⟩ : Fin (n + 1)) = 0 := by
  /-
    n : Nat
    c : CompositionAsSet n
    ⊢ Eq (c.boundary ⟨0, ⋯⟩) 0
  -/
  rw [boundary, Finset.orderEmbOfFin_zero rfl c.card_boundaries_pos]
  /-
    n : Nat
    c : CompositionAsSet n
    ⊢ Eq (c.boundaries.min' ⋯) 0
  -/
  exact le_antisymm (Finset.min'_le _ _ c.zero_mem) (Fin.zero_le _)
  /-
    🎉 no goals
  -/


@[simp]
theorem boundary_length : c.boundary ⟨c.length, c.length_lt_card_boundaries⟩ = Fin.last n := by
  /-
    n : Nat
    c : CompositionAsSet n
    ⊢ Eq (c.boundary ⟨c.length, ⋯⟩) (Fin.last n)
  -/
  convert Finset.orderEmbOfFin_last rfl c.card_boundaries_pos
  /-
    case h.e'_3
    n : Nat
    c : CompositionAsSet n
    ⊢ Eq (Fin.last n) (c.boundaries.max' ⋯)
  -/
  exact le_antisymm (Finset.le_max' _ _ c.getLast_mem) (Fin.le_last _)
  /-
    🎉 no goals
  -/


/-- Size of the `i`-th block in a `CompositionAsSet`, seen as a function on `Fin c.length`. -/
def blocksFun (i : Fin c.length) : ℕ :=
  c.boundary ⟨(i : ℕ) + 1, c.lt_length i⟩ - c.boundary ⟨i, c.lt_length' i⟩


theorem blocksFun_pos (i : Fin c.length) : 0 < c.blocksFun i :=
  haveI : (⟨i, c.lt_length' i⟩ : Fin c.boundaries.card) < ⟨i + 1, c.lt_length i⟩ :=
    Nat.lt_succ_self _
  lt_tsub_iff_left.mpr ((c.boundaries.orderEmbOfFin rfl).strictMono this)


/-- List of the sizes of the blocks in a `CompositionAsSet`. -/
def blocks (c : CompositionAsSet n) : List ℕ :=
  ofFn c.blocksFun


@[simp]
theorem blocks_length : c.blocks.length = c.length :=
  length_ofFn _


theorem blocks_partial_sum {i : ℕ} (h : i < c.boundaries.card) :
    (c.blocks.take i).sum = c.boundary ⟨i, h⟩ := by
  /-
    n : Nat
    c : CompositionAsSet n
    i : Nat
    h : LT.lt i c.boundaries.card
    ⊢ Eq (List.take i c.blocks).sum ↑(c.boundary ⟨i, h⟩)
  -/
  induction' i with i IH
    /-
      case zero
      n : Nat
      c : CompositionAsSet n
      h : LT.lt 0 c.boundaries.card
      ⊢ Eq (List.take 0 c.blocks).sum ↑(c.boundary ⟨0, h⟩)
    -/
  · simp
    /-
      🎉 no goals
    -/
  have A : i < c.blocks.length := by
    rw [c.card_boundaries_eq_succ_length] at h
    simp [blocks, Nat.lt_of_succ_lt_succ h]
  /-
    case succ
    n : Nat
    c : CompositionAsSet n
    i : Nat
    IH : ∀ (h : LT.lt i c.boundaries.card), Eq (List.take i c.blocks).sum ↑(c.boun …
    h : LT.lt (HAdd.hAdd i 1) c.boundaries.card
    A : LT.lt i c.blocks.length
    ⊢ Eq (List.take (HAdd.hAdd i 1) c.blocks).sum ↑(c.boundary ⟨HAdd.hAdd i 1, h⟩)
  -/
  have B : i < c.boundaries.card := lt_of_lt_of_le A (by simp [blocks, length, Nat.sub_le])
  /-
    case succ
    n : Nat
    c : CompositionAsSet n
    i : Nat
    IH : ∀ (h : LT.lt i c.boundaries.card), Eq (List.take i c.blocks).sum ↑(c.boun …
    h : LT.lt (HAdd.hAdd i 1) c.boundaries.card
    A : LT.lt i c.blocks.length
    B : LT.lt i c.boundaries.card
    ⊢ Eq (List.take (HAdd.hAdd i 1) c.blocks).sum ↑(c.boundary ⟨HAdd.hAdd i 1, h⟩)
  -/
  rw [sum_take_succ _ _ A, IH B]
  /-
    case succ
    n : Nat
    c : CompositionAsSet n
    i : Nat
    IH : ∀ (h : LT.lt i c.boundaries.card), Eq (List.take i c.blocks).sum ↑(c.boun …
    h : LT.lt (HAdd.hAdd i 1) c.boundaries.card
    A : LT.lt i c.blocks.length
    B : LT.lt i c.boundaries.card
    ⊢ Eq (HAdd.hAdd (↑(c.boundary ⟨i, B⟩)) (GetElem.getElem c.blocks i A)) ↑(c.bou …
  -/
  simp [blocks, blocksFun, get_ofFn]
  /-
    🎉 no goals
  -/


theorem mem_boundaries_iff_exists_blocks_sum_take_eq {j : Fin (n + 1)} :
    j ∈ c.boundaries ↔ ∃ i < c.boundaries.card, (c.blocks.take i).sum = j := by
  /-
    n : Nat
    c : CompositionAsSet n
    j : Fin (HAdd.hAdd n 1)
    ⊢ Iff (Membership.mem c.boundaries j) (Exists fun i => And (LT.lt i c.boundari …
  -/
  constructor
    /-
      case mp
      n : Nat
      c : CompositionAsSet n
      j : Fin (HAdd.hAdd n 1)
      ⊢ Membership.mem c.boundaries j → Exists fun i => And (LT.lt i c.boundaries.ca …
    -/
  · intro hj
    /-
      case mp
      n : Nat
      c : CompositionAsSet n
      j : Fin (HAdd.hAdd n 1)
      hj : Membership.mem c.boundaries j
      ⊢ Exists fun i => And (LT.lt i c.boundaries.card) (Eq (List.take i c.blocks).s …
    -/
    rcases (c.boundaries.orderIsoOfFin rfl).surjective ⟨j, hj⟩ with ⟨i, hi⟩
    /-
      case mp.intro
      n : Nat
      c : CompositionAsSet n
      j : Fin (HAdd.hAdd n 1)
      hj : Membership.mem c.boundaries j
      i : Fin c.boundaries.card
      hi : Eq ((c.boundaries.orderIsoOfFin ⋯) i) ⟨j, hj⟩
      ⊢ Exists fun i => And (LT.lt i c.boundaries.card) (Eq (List.take i c.blocks).s …
    -/
    rw [Subtype.ext_iff, Subtype.coe_mk] at hi
    /-
      case mp.intro
      n : Nat
      c : CompositionAsSet n
      j : Fin (HAdd.hAdd n 1)
      hj : Membership.mem c.boundaries j
      i : Fin c.boundaries.card
      hi : Eq ↑((c.boundaries.orderIsoOfFin ⋯) i) ↑⟨j, hj⟩
      ⊢ Exists fun i => And (LT.lt i c.boundaries.card) (Eq (List.take i c.blocks).s …
    -/
    refine ⟨i.1, i.2, ?_⟩
    /-
      case mp.intro
      n : Nat
      c : CompositionAsSet n
      j : Fin (HAdd.hAdd n 1)
      hj : Membership.mem c.boundaries j
      i : Fin c.boundaries.card
      hi : Eq ↑((c.boundaries.orderIsoOfFin ⋯) i) ↑⟨j, hj⟩
      ⊢ Eq (List.take (↑i) c.blocks).sum ↑j
    -/
    dsimp at hi
    /-
      case mp.intro
      n : Nat
      c : CompositionAsSet n
      j : Fin (HAdd.hAdd n 1)
      hj : Membership.mem c.boundaries j
      i : Fin c.boundaries.card
      hi : Eq ((c.boundaries.orderEmbOfFin ⋯) i) j
      ⊢ Eq (List.take (↑i) c.blocks).sum ↑j
    -/
    rw [← hi, c.blocks_partial_sum i.2]
    /-
      case mp.intro
      n : Nat
      c : CompositionAsSet n
      j : Fin (HAdd.hAdd n 1)
      hj : Membership.mem c.boundaries j
      i : Fin c.boundaries.card
      hi : Eq ((c.boundaries.orderEmbOfFin ⋯) i) j
      ⊢ Eq ↑(c.boundary ⟨↑i, ⋯⟩) ↑((c.boundaries.orderEmbOfFin ⋯) i)
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case mpr
      n : Nat
      c : CompositionAsSet n
      j : Fin (HAdd.hAdd n 1)
      ⊢ (Exists fun i => And (LT.lt i c.boundaries.card) (Eq (List.take i c.blocks). …
    -/
  · rintro ⟨i, hi, H⟩
    /-
      case mpr.intro.intro
      n : Nat
      c : CompositionAsSet n
      j : Fin (HAdd.hAdd n 1)
      i : Nat
      hi : LT.lt i c.boundaries.card
      H : Eq (List.take i c.blocks).sum ↑j
      ⊢ Membership.mem c.boundaries j
    -/
    convert (c.boundaries.orderIsoOfFin rfl ⟨i, hi⟩).2
    /-
      case h.e'_5
      n : Nat
      c : CompositionAsSet n
      j : Fin (HAdd.hAdd n 1)
      i : Nat
      hi : LT.lt i c.boundaries.card
      H : Eq (List.take i c.blocks).sum ↑j
      ⊢ Eq j ↑((c.boundaries.orderIsoOfFin ⋯) ⟨i, hi⟩)
    -/
    have : c.boundary ⟨i, hi⟩ = j := by rwa [Fin.ext_iff, ← c.blocks_partial_sum hi]
    /-
      case h.e'_5
      n : Nat
      c : CompositionAsSet n
      j : Fin (HAdd.hAdd n 1)
      i : Nat
      hi : LT.lt i c.boundaries.card
      H : Eq (List.take i c.blocks).sum ↑j
      this : Eq (c.boundary ⟨i, hi⟩) j
      ⊢ Eq j ↑((c.boundaries.orderIsoOfFin ⋯) ⟨i, hi⟩)
    -/
    exact this.symm
    /-
      🎉 no goals
    -/


theorem blocks_sum : c.blocks.sum = n := by
  /-
    n : Nat
    c : CompositionAsSet n
    ⊢ Eq c.blocks.sum n
  -/
  have : c.blocks.take c.length = c.blocks := take_of_length_le (by simp [blocks])
  /-
    n : Nat
    c : CompositionAsSet n
    this : Eq (List.take c.length c.blocks) c.blocks
    ⊢ Eq c.blocks.sum n
  -/
  rw [← this, c.blocks_partial_sum c.length_lt_card_boundaries, c.boundary_length]
  /-
    n : Nat
    c : CompositionAsSet n
    this : Eq (List.take c.length c.blocks) c.blocks
    ⊢ Eq (↑(Fin.last n)) n
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Associating a `Composition n` to a `CompositionAsSet n`, by registering the sizes of the
blocks as a list of positive integers. -/
def toComposition : Composition n where
  blocks := c.blocks
                   /-
                     n : Nat
                     c : CompositionAsSet n
                     ⊢ ∀ {i : Nat}, Membership.mem c.blocks i → LT.lt 0 i
                   -/
  blocks_pos := by simp only [blocks, forall_mem_ofFn_iff, blocksFun_pos c, forall_true_iff]
                   /-
                     🎉 no goals
                   -/
  blocks_sum := c.blocks_sum


@[simp]
theorem Composition.toCompositionAsSet_length (c : Composition n) :
    c.toCompositionAsSet.length = c.length := by
  /-
    n : Nat
    c : Composition n
    ⊢ Eq c.toCompositionAsSet.length c.length
  -/
  simp [Composition.toCompositionAsSet, CompositionAsSet.length, c.card_boundaries_eq_succ_length]
  /-
    🎉 no goals
  -/


@[simp]
theorem CompositionAsSet.toComposition_length (c : CompositionAsSet n) :
    c.toComposition.length = c.length := by
  /-
    n : Nat
    c : CompositionAsSet n
    ⊢ Eq c.toComposition.length c.length
  -/
  simp [CompositionAsSet.toComposition, Composition.length, Composition.blocks]
  /-
    🎉 no goals
  -/


@[simp]
theorem Composition.toCompositionAsSet_blocks (c : Composition n) :
    c.toCompositionAsSet.blocks = c.blocks := by
  /-
    n : Nat
    c : Composition n
    ⊢ Eq c.toCompositionAsSet.blocks c.blocks
  -/
  let d := c.toCompositionAsSet
  /-
    n : Nat
    c : Composition n
    d : CompositionAsSet n := c.toCompositionAsSet
    ⊢ Eq c.toCompositionAsSet.blocks c.blocks
  -/
  change d.blocks = c.blocks
  /-
    n : Nat
    c : Composition n
    d : CompositionAsSet n := c.toCompositionAsSet
    ⊢ Eq d.blocks c.blocks
  -/
  have length_eq : d.blocks.length = c.blocks.length := by simp [d, blocks_length]
  suffices H : ∀ i ≤ d.blocks.length, (d.blocks.take i).sum = (c.blocks.take i).sum from
    eq_of_sum_take_eq length_eq H
  /-
    n : Nat
    c : Composition n
    d : CompositionAsSet n := c.toCompositionAsSet
    length_eq : Eq d.blocks.length c.blocks.length
    ⊢ ∀ (i : Nat), LE.le i d.blocks.length → Eq (List.take i d.blocks).sum (List.t …
  -/
  intro i hi
  have i_lt : i < d.boundaries.card := by
    -- Porting note: relied on `convert` unfolding definitions, switched to using a `simpa`
    simpa [CompositionAsSet.blocks, length_ofFn,
      d.card_boundaries_eq_succ_length] using Nat.lt_succ_iff.2 hi
  /-
    n : Nat
    c : Composition n
    d : CompositionAsSet n := c.toCompositionAsSet
    length_eq : Eq d.blocks.length c.blocks.length
    i : Nat
    hi : LE.le i d.blocks.length
    i_lt : LT.lt i d.boundaries.card
    ⊢ Eq (List.take i d.blocks).sum (List.take i c.blocks).sum
  -/
  have i_lt' : i < c.boundaries.card := i_lt
  /-
    n : Nat
    c : Composition n
    d : CompositionAsSet n := c.toCompositionAsSet
    length_eq : Eq d.blocks.length c.blocks.length
    i : Nat
    hi : LE.le i d.blocks.length
    i_lt : LT.lt i d.boundaries.card
    i_lt' : LT.lt i c.boundaries.card
    ⊢ Eq (List.take i d.blocks).sum (List.take i c.blocks).sum
  -/
  have i_lt'' : i < c.length + 1 := by rwa [c.card_boundaries_eq_succ_length] at i_lt'
  have A :
    d.boundaries.orderEmbOfFin rfl ⟨i, i_lt⟩ =
      c.boundaries.orderEmbOfFin c.card_boundaries_eq_succ_length ⟨i, i_lt''⟩ :=
    rfl
  /-
    n : Nat
    c : Composition n
    d : CompositionAsSet n := c.toCompositionAsSet
    length_eq : Eq d.blocks.length c.blocks.length
    i : Nat
    hi : LE.le i d.blocks.length
    i_lt : LT.lt i d.boundaries.card
    i_lt' : LT.lt i c.boundaries.card
    i_lt'' : LT.lt i (HAdd.hAdd c.length 1)
    A : Eq ((d.boundaries.orderEmbOfFin ⋯) ⟨i, i_lt⟩) ((c.boundaries.orderEmbOfFin …
    ⊢ Eq (List.take i d.blocks).sum (List.take i c.blocks).sum
  -/
  have B : c.sizeUpTo i = c.boundary ⟨i, i_lt''⟩ := rfl
  rw [d.blocks_partial_sum i_lt, CompositionAsSet.boundary, ← Composition.sizeUpTo, B, A,
    c.orderEmbOfFin_boundaries]


@[simp]
theorem CompositionAsSet.toComposition_blocks (c : CompositionAsSet n) :
    c.toComposition.blocks = c.blocks :=
  rfl


@[simp]
theorem CompositionAsSet.toComposition_boundaries (c : CompositionAsSet n) :
    c.toComposition.boundaries = c.boundaries := by
  /-
    n : Nat
    c : CompositionAsSet n
    ⊢ Eq c.toComposition.boundaries c.boundaries
  -/
  ext j
  /-
    case h
    n : Nat
    c : CompositionAsSet n
    j : Fin (HAdd.hAdd n 1)
    ⊢ Iff (Membership.mem c.toComposition.boundaries j) (Membership.mem c.boundari …
  -/
  simp only [c.mem_boundaries_iff_exists_blocks_sum_take_eq, Composition.boundaries, Finset.mem_map]
  /-
    case h
    n : Nat
    c : CompositionAsSet n
    j : Fin (HAdd.hAdd n 1)
    ⊢ Iff (Exists fun a => And (Membership.mem Finset.univ a) (Eq (c.toComposition …
  -/
  constructor
    /-
      case h.mp
      n : Nat
      c : CompositionAsSet n
      j : Fin (HAdd.hAdd n 1)
      ⊢ (Exists fun a => And (Membership.mem Finset.univ a) (Eq (c.toComposition.bou …
    -/
  · rintro ⟨i, _, hi⟩
    /-
      case h.mp.intro.intro
      n : Nat
      c : CompositionAsSet n
      j : Fin (HAdd.hAdd n 1)
      i : Fin (HAdd.hAdd c.toComposition.length 1)
      left✝ : Membership.mem Finset.univ i
      hi : Eq (c.toComposition.boundary.toEmbedding i) j
      ⊢ Exists fun i => And (LT.lt i c.boundaries.card) (Eq (List.take i c.blocks).s …
    -/
    refine ⟨i.1, ?_, ?_⟩
      /-
        case h.mp.intro.intro.refine_1
        n : Nat
        c : CompositionAsSet n
        j : Fin (HAdd.hAdd n 1)
        i : Fin (HAdd.hAdd c.toComposition.length 1)
        left✝ : Membership.mem Finset.univ i
        hi : Eq (c.toComposition.boundary.toEmbedding i) j
        ⊢ LT.lt (↑i) c.boundaries.card
      -/
    · simpa [c.card_boundaries_eq_succ_length] using i.2
      /-
        🎉 no goals
      -/
      /-
        case h.mp.intro.intro.refine_2
        n : Nat
        c : CompositionAsSet n
        j : Fin (HAdd.hAdd n 1)
        i : Fin (HAdd.hAdd c.toComposition.length 1)
        left✝ : Membership.mem Finset.univ i
        hi : Eq (c.toComposition.boundary.toEmbedding i) j
        ⊢ Eq (List.take (↑i) c.blocks).sum ↑j
      -/
    · simp [Composition.boundary, Composition.sizeUpTo, ← hi]
      /-
        🎉 no goals
      -/
    /-
      case h.mpr
      n : Nat
      c : CompositionAsSet n
      j : Fin (HAdd.hAdd n 1)
      ⊢ (Exists fun i => And (LT.lt i c.boundaries.card) (Eq (List.take i c.blocks). …
    -/
  · rintro ⟨i, i_lt, hi⟩
    /-
      case h.mpr.intro.intro
      n : Nat
      c : CompositionAsSet n
      j : Fin (HAdd.hAdd n 1)
      i : Nat
      i_lt : LT.lt i c.boundaries.card
      hi : Eq (List.take i c.blocks).sum ↑j
      ⊢ Exists fun a => And (Membership.mem Finset.univ a) (Eq (c.toComposition.boun …
    -/
    refine ⟨i, by simp, ?_⟩
    /-
      case h.mpr.intro.intro
      n : Nat
      c : CompositionAsSet n
      j : Fin (HAdd.hAdd n 1)
      i : Nat
      i_lt : LT.lt i c.boundaries.card
      hi : Eq (List.take i c.blocks).sum ↑j
      ⊢ Eq (c.toComposition.boundary.toEmbedding ↑i) j
    -/
    rw [c.card_boundaries_eq_succ_length] at i_lt
    /-
      case h.mpr.intro.intro
      n : Nat
      c : CompositionAsSet n
      j : Fin (HAdd.hAdd n 1)
      i : Nat
      i_lt : LT.lt i (HAdd.hAdd c.length 1)
      hi : Eq (List.take i c.blocks).sum ↑j
      ⊢ Eq (c.toComposition.boundary.toEmbedding ↑i) j
    -/
    simp [Composition.boundary, Nat.mod_eq_of_lt i_lt, Composition.sizeUpTo, hi]
    /-
      🎉 no goals
    -/


@[simp]
theorem Composition.toCompositionAsSet_boundaries (c : Composition n) :
    c.toCompositionAsSet.boundaries = c.boundaries :=
  rfl


/-- Equivalence between `Composition n` and `CompositionAsSet n`. -/
def compositionEquiv (n : ℕ) : Composition n ≃ CompositionAsSet n where
  toFun c := c.toCompositionAsSet
  invFun c := c.toComposition
  left_inv c := by
    /-
      n✝ n : Nat
      c : Composition n
      ⊢ Eq ((fun c => c.toComposition) ((fun c => c.toCompositionAsSet) c)) c
    -/
    ext1
    /-
      case blocks
      n✝ n : Nat
      c : Composition n
      ⊢ Eq ((fun c => c.toComposition) ((fun c => c.toCompositionAsSet) c)).blocks c …
    -/
    exact c.toCompositionAsSet_blocks
    /-
      🎉 no goals
    -/
  right_inv c := by
    /-
      n✝ n : Nat
      c : CompositionAsSet n
      ⊢ Eq ((fun c => c.toCompositionAsSet) ((fun c => c.toComposition) c)) c
    -/
    ext1
    /-
      case boundaries
      n✝ n : Nat
      c : CompositionAsSet n
      ⊢ Eq ((fun c => c.toCompositionAsSet) ((fun c => c.toComposition) c)).boundari …
    -/
    exact c.toComposition_boundaries
    /-
      🎉 no goals
    -/


instance compositionFintype (n : ℕ) : Fintype (Composition n) :=
  Fintype.ofEquiv _ (compositionEquiv n).symm


theorem composition_card (n : ℕ) : Fintype.card (Composition n) = 2 ^ (n - 1) := by
  /-
    n : Nat
    ⊢ Eq (Fintype.card (Composition n)) (HPow.hPow 2 (HSub.hSub n 1))
  -/
  rw [← compositionAsSet_card n]
  /-
    n : Nat
    ⊢ Eq (Fintype.card (Composition n)) (Fintype.card (CompositionAsSet n))
  -/
  exact Fintype.card_congr (compositionEquiv n)
  /-
    🎉 no goals
  -/

