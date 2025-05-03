/-- A semistandard Young tableau is a filling of the cells of a Young diagram by natural
numbers, such that the entries in each row are weakly increasing (left to right), and the entries
in each column are strictly increasing (top to bottom).

Here, a semistandard Young tableau is represented as an unrestricted function `ℕ → ℕ → ℕ` that, for
reasons of extensionality, is required to vanish outside `μ`. -/
structure SemistandardYoungTableau (μ : YoungDiagram) where
  /-- `entry i j` is value of the `(i, j)` entry of the SSYT `μ`. -/
  entry : ℕ → ℕ → ℕ
  /-- The entries in each row are weakly increasing (left to right). -/
  row_weak' : ∀ {i j1 j2 : ℕ}, j1 < j2 → (i, j2) ∈ μ → entry i j1 ≤ entry i j2
  /-- The entries in each column are strictly increasing (top to bottom). -/
  col_strict' : ∀ {i1 i2 j : ℕ}, i1 < i2 → (i2, j) ∈ μ → entry i1 j < entry i2 j
  /-- `entry` is required to be zero for all pairs `(i, j) ∉ μ`. -/
  zeros' : ∀ {i j}, (i, j) ∉ μ → entry i j = 0


instance instFunLike {μ : YoungDiagram} : FunLike (SemistandardYoungTableau μ) ℕ (ℕ → ℕ) where
  coe := SemistandardYoungTableau.entry
  coe_injective' T T' h := by
    /-
      μ : YoungDiagram
      T T' : SemistandardYoungTableau μ
      h : Eq T.entry T'.entry
      ⊢ Eq T T'
    -/
    cases T
    /-
      case mk
      μ : YoungDiagram
      T' : SemistandardYoungTableau μ
      entry✝ : Nat → Nat → Nat
      row_weak'✝ : ∀ {i j1 j2 : Nat}, LT.lt j1 j2 → Membership.mem μ { fst := i, snd …
      col_strict'✝ : ∀ {i1 i2 j : Nat}, LT.lt i1 i2 → Membership.mem μ { fst := i2,  …
      zeros'✝ : ∀ {i j : Nat}, Not (Membership.mem μ { fst := i, snd := j }) → Eq (e …
      h : Eq { entry := entry✝, row_weak' := row_weak'✝, col_strict' := col_strict'✝ …
      ⊢ Eq { entry := entry✝, row_weak' := row_weak'✝, col_strict' := col_strict'✝,  …
    -/
    cases T'
    /-
      case mk.mk
      μ : YoungDiagram
      entry✝¹ : Nat → Nat → Nat
      row_weak'✝¹ : ∀ {i j1 j2 : Nat}, LT.lt j1 j2 → Membership.mem μ { fst := i, sn …
      col_strict'✝¹ : ∀ {i1 i2 j : Nat}, LT.lt i1 i2 → Membership.mem μ { fst := i2, …
      zeros'✝¹ : ∀ {i j : Nat}, Not (Membership.mem μ { fst := i, snd := j }) → Eq ( …
      entry✝ : Nat → Nat → Nat
      row_weak'✝ : ∀ {i j1 j2 : Nat}, LT.lt j1 j2 → Membership.mem μ { fst := i, snd …
      col_strict'✝ : ∀ {i1 i2 j : Nat}, LT.lt i1 i2 → Membership.mem μ { fst := i2,  …
      zeros'✝ : ∀ {i j : Nat}, Not (Membership.mem μ { fst := i, snd := j }) → Eq (e …
      h : Eq { entry := entry✝¹, row_weak' := row_weak'✝¹, col_strict' := col_strict …
      ⊢ Eq { entry := entry✝¹, row_weak' := row_weak'✝¹, col_strict' := col_strict'✝ …
    -/
    congr
    /-
      🎉 no goals
    -/


@[simp]
theorem to_fun_eq_coe {μ : YoungDiagram} {T : SemistandardYoungTableau μ} :
    T.entry = (T : ℕ → ℕ → ℕ) :=
  rfl


@[ext]
theorem ext {μ : YoungDiagram} {T T' : SemistandardYoungTableau μ} (h : ∀ i j, T i j = T' i j) :
    T = T' :=
  DFunLike.ext T T' fun _ ↦ by
    /-
      μ : YoungDiagram
      T T' : SemistandardYoungTableau μ
      h : ∀ (i j : Nat), Eq (T i j) (T' i j)
      x✝ : Nat
      ⊢ Eq (T x✝) (T' x✝)
    -/
    funext
    /-
      case h
      μ : YoungDiagram
      T T' : SemistandardYoungTableau μ
      h : ∀ (i j : Nat), Eq (T i j) (T' i j)
      x✝¹ x✝ : Nat
      ⊢ Eq (T x✝¹ x✝) (T' x✝¹ x✝)
    -/
    apply h
    /-
      🎉 no goals
    -/


/-- Copy of an `SemistandardYoungTableau μ` with a new `entry` equal to the old one. Useful to fix
definitional equalities. -/
protected def copy {μ : YoungDiagram} (T : SemistandardYoungTableau μ) (entry' : ℕ → ℕ → ℕ)
    (h : entry' = T) : SemistandardYoungTableau μ where
  entry := entry'
  row_weak' := h.symm ▸ T.row_weak'
  col_strict' := h.symm ▸ T.col_strict'
  zeros' := h.symm ▸ T.zeros'


@[simp]
theorem coe_copy {μ : YoungDiagram} (T : SemistandardYoungTableau μ) (entry' : ℕ → ℕ → ℕ)
    (h : entry' = T) : ⇑(T.copy entry' h) = entry' :=
  rfl


theorem copy_eq {μ : YoungDiagram} (T : SemistandardYoungTableau μ) (entry' : ℕ → ℕ → ℕ)
    (h : entry' = T) : T.copy entry' h = T :=
  DFunLike.ext' h


theorem row_weak {μ : YoungDiagram} (T : SemistandardYoungTableau μ) {i j1 j2 : ℕ} (hj : j1 < j2)
    (hcell : (i, j2) ∈ μ) : T i j1 ≤ T i j2 :=
  T.row_weak' hj hcell


theorem col_strict {μ : YoungDiagram} (T : SemistandardYoungTableau μ) {i1 i2 j : ℕ} (hi : i1 < i2)
    (hcell : (i2, j) ∈ μ) : T i1 j < T i2 j :=
  T.col_strict' hi hcell


theorem zeros {μ : YoungDiagram} (T : SemistandardYoungTableau μ) {i j : ℕ}
    (not_cell : (i, j) ∉ μ) : T i j = 0 :=
  T.zeros' not_cell


theorem row_weak_of_le {μ : YoungDiagram} (T : SemistandardYoungTableau μ) {i j1 j2 : ℕ}
    (hj : j1 ≤ j2) (cell : (i, j2) ∈ μ) : T i j1 ≤ T i j2 := by
  /-
    μ : YoungDiagram
    T : SemistandardYoungTableau μ
    i j1 j2 : Nat
    hj : LE.le j1 j2
    cell : Membership.mem μ { fst := i, snd := j2 }
    ⊢ LE.le (T i j1) (T i j2)
  -/
  cases' eq_or_lt_of_le hj with h h
    /-
      case inl
      μ : YoungDiagram
      T : SemistandardYoungTableau μ
      i j1 j2 : Nat
      hj : LE.le j1 j2
      cell : Membership.mem μ { fst := i, snd := j2 }
      h : Eq j1 j2
      ⊢ LE.le (T i j1) (T i j2)
    -/
  · rw [h]
    /-
      🎉 no goals
    -/
    /-
      case inr
      μ : YoungDiagram
      T : SemistandardYoungTableau μ
      i j1 j2 : Nat
      hj : LE.le j1 j2
      cell : Membership.mem μ { fst := i, snd := j2 }
      h : LT.lt j1 j2
      ⊢ LE.le (T i j1) (T i j2)
    -/
  · exact T.row_weak h cell
    /-
      🎉 no goals
    -/


theorem col_weak {μ : YoungDiagram} (T : SemistandardYoungTableau μ) {i1 i2 j : ℕ} (hi : i1 ≤ i2)
    (cell : (i2, j) ∈ μ) : T i1 j ≤ T i2 j := by
  /-
    μ : YoungDiagram
    T : SemistandardYoungTableau μ
    i1 i2 j : Nat
    hi : LE.le i1 i2
    cell : Membership.mem μ { fst := i2, snd := j }
    ⊢ LE.le (T i1 j) (T i2 j)
  -/
  cases' eq_or_lt_of_le hi with h h
    /-
      case inl
      μ : YoungDiagram
      T : SemistandardYoungTableau μ
      i1 i2 j : Nat
      hi : LE.le i1 i2
      cell : Membership.mem μ { fst := i2, snd := j }
      h : Eq i1 i2
      ⊢ LE.le (T i1 j) (T i2 j)
    -/
  · rw [h]
    /-
      🎉 no goals
    -/
    /-
      case inr
      μ : YoungDiagram
      T : SemistandardYoungTableau μ
      i1 i2 j : Nat
      hi : LE.le i1 i2
      cell : Membership.mem μ { fst := i2, snd := j }
      h : LT.lt i1 i2
      ⊢ LE.le (T i1 j) (T i2 j)
    -/
  · exact le_of_lt (T.col_strict h cell)
    /-
      🎉 no goals
    -/


/-- The "highest weight" SSYT of a given shape has all i's in row i, for each i. -/
def highestWeight (μ : YoungDiagram) : SemistandardYoungTableau μ where
  entry i j := if (i, j) ∈ μ then i else 0
  row_weak' hj hcell := by
    /-
      μ : YoungDiagram
      i✝ j1✝ j2✝ : Nat
      hj : LT.lt j1✝ j2✝
      hcell : Membership.mem μ { fst := i✝, snd := j2✝ }
      ⊢ LE.le ((fun i j => ite (Membership.mem μ { fst := i, snd := j }) i 0) i✝ j1✝ …
    -/
    simp only
    /-
      μ : YoungDiagram
      i✝ j1✝ j2✝ : Nat
      hj : LT.lt j1✝ j2✝
      hcell : Membership.mem μ { fst := i✝, snd := j2✝ }
      ⊢ LE.le (ite (Membership.mem μ { fst := i✝, snd := j1✝ }) i✝ 0) (ite (Membersh …
    -/
    rw [if_pos hcell, if_pos (μ.up_left_mem (by rfl) (le_of_lt hj) hcell)]
    /-
      🎉 no goals
    -/
  col_strict' hi hcell := by
    /-
      μ : YoungDiagram
      i1✝ i2✝ j✝ : Nat
      hi : LT.lt i1✝ i2✝
      hcell : Membership.mem μ { fst := i2✝, snd := j✝ }
      ⊢ LT.lt ((fun i j => ite (Membership.mem μ { fst := i, snd := j }) i 0) i1✝ j✝ …
    -/
    simp only
    /-
      μ : YoungDiagram
      i1✝ i2✝ j✝ : Nat
      hi : LT.lt i1✝ i2✝
      hcell : Membership.mem μ { fst := i2✝, snd := j✝ }
      ⊢ LT.lt (ite (Membership.mem μ { fst := i1✝, snd := j✝ }) i1✝ 0) (ite (Members …
    -/
    rwa [if_pos hcell, if_pos (μ.up_left_mem (le_of_lt hi) (by rfl) hcell)]
    /-
      🎉 no goals
    -/
  zeros' not_cell := if_neg not_cell


@[simp]
theorem highestWeight_apply {μ : YoungDiagram} {i j : ℕ} :
    highestWeight μ i j = if (i, j) ∈ μ then i else 0 :=
  rfl


instance {μ : YoungDiagram} : Inhabited (SemistandardYoungTableau μ) :=
  ⟨highestWeight μ⟩


