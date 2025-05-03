/--
The **Krull dimension** of a preorder `α` is the supremum of the rightmost index of all relation
series of `α` ordered by `<`. If there is no series `a₀ < a₁ < ... < aₙ` in `α`, then its Krull
dimension is defined to be negative infinity; if the length of all series `a₀ < a₁ < ... < aₙ` is
unbounded, its Krull dimension is defined to be positive infinity.
-/
noncomputable def krullDim (α : Type*) [Preorder α] : WithBot ℕ∞ :=
  ⨆ (p : LTSeries α), p.length


/--
The **height** of an element `a` in a preorder `α` is the supremum of the rightmost index of all
relation series of `α` ordered by `<` and ending below or at `a`.
-/
noncomputable def height {α : Type*} [Preorder α] (a : α) : ℕ∞ :=
  ⨆ (p : LTSeries α) (_ : p.last ≤ a), p.length


/--
The **coheight** of an element `a` in a preorder `α` is the supremum of the rightmost index of all
relation series of `α` ordered by `<` and beginning with `a`.

The definition of `coheight` is via the `height` in the dual order, in order to easily transfer
theorems between `height` and `coheight`. See `coheight_eq` for the definition with a
series ordered by `<` and beginning with `a`.
-/
noncomputable def coheight {α : Type*} [Preorder α] (a : α) : ℕ∞ := height (α := αᵒᵈ) a


@[simp] lemma height_toDual (x : α) : height (OrderDual.toDual x) = coheight x := rfl

@[simp] lemma height_ofDual (x : αᵒᵈ) : height (OrderDual.ofDual x) = coheight x := rfl

@[simp] lemma coheight_toDual (x : α) : coheight (OrderDual.toDual x) = height x := rfl

@[simp] lemma coheight_ofDual (x : αᵒᵈ) : coheight (OrderDual.ofDual x) = height x := rfl


/--
The **coheight** of an element `a` in a preorder `α` is the supremum of the rightmost index of all
relation series of `α` ordered by `<` and beginning with `a`.

This is not the definition of `coheight`. The definition of `coheight` is via the `height` in the
dual order, in order to easily transfer theorems between `height` and `coheight`.
-/
lemma coheight_eq (a : α) :
    coheight a = ⨆ (p : LTSeries α) (_ : a ≤ p.head), (p.length : ℕ∞) := by
  apply Equiv.iSup_congr ⟨RelSeries.reverse, RelSeries.reverse, RelSeries.reverse_reverse,
    RelSeries.reverse_reverse⟩
  /-
    α : Type u_1
    inst✝ : Preorder α
    a : α
    ⊢ ∀ (x : RelSeries fun x1 x2 => LT.lt x1 x2), Eq (iSup fun x_1 => ↑({ toFun := …
  -/
  congr! 1
  /-
    🎉 no goals
  -/


lemma height_le_iff {a : α} {n : ℕ∞} :
    height a ≤ n ↔ ∀ ⦃p : LTSeries α⦄, p.last ≤ a → p.length ≤ n := by
 /-
   α : Type u_1
   inst✝ : Preorder α
   a : α
   n : ENat
   ⊢ Iff (LE.le (Order.height a) n) (∀ ⦃p : LTSeries α⦄, LE.le (RelSeries.last p) …
 -/
 rw [height, iSup₂_le_iff]
 /-
   🎉 no goals
 -/


lemma coheight_le_iff {a : α} {n : ℕ∞} :
    coheight a ≤ n ↔ ∀ ⦃p : LTSeries α⦄, a ≤ p.head → p.length ≤ n := by
 /-
   α : Type u_1
   inst✝ : Preorder α
   a : α
   n : ENat
   ⊢ Iff (LE.le (Order.coheight a) n) (∀ ⦃p : LTSeries α⦄, LE.le a (RelSeries.hea …
 -/
 rw [coheight_eq, iSup₂_le_iff]
 /-
   🎉 no goals
 -/


lemma height_le {a : α} {n : ℕ∞} (h : ∀ (p : LTSeries α), p.last = a → p.length ≤ n) :
    height a ≤ n := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    a : α
    n : ENat
    h : ∀ (p : LTSeries α), Eq (RelSeries.last p) a → LE.le (↑p.length) n
    ⊢ LE.le (Order.height a) n
  -/
  apply height_le_iff.mpr
  /-
    α : Type u_1
    inst✝ : Preorder α
    a : α
    n : ENat
    h : ∀ (p : LTSeries α), Eq (RelSeries.last p) a → LE.le (↑p.length) n
    ⊢ ∀ ⦃p : LTSeries α⦄, LE.le (RelSeries.last p) a → LE.le (↑p.length) n
  -/
  intro p hlast
  /-
    α : Type u_1
    inst✝ : Preorder α
    a : α
    n : ENat
    h : ∀ (p : LTSeries α), Eq (RelSeries.last p) a → LE.le (↑p.length) n
    p : LTSeries α
    hlast : LE.le (RelSeries.last p) a
    ⊢ LE.le (↑p.length) n
  -/
  wlog hlenpos : p.length ≠ 0
    /-
      case inr
      α : Type u_1
      inst✝ : Preorder α
      a : α
      n : ENat
      h : ∀ (p : LTSeries α), Eq (RelSeries.last p) a → LE.le (↑p.length) n
      p : LTSeries α
      hlast : LE.le (RelSeries.last p) a
      this : ∀ {α : Type u_1} [inst : Preorder α] {a : α} {n : ENat}, (∀ (p : LTSeri …
      hlenpos : Not (Ne p.length 0)
      ⊢ LE.le (↑p.length) n
    -/
  · simp_all
    /-
      🎉 no goals
    -/
  -- We replace the last element in the series with `a`
  /-
    α✝ : Type u_1
    inst✝¹ : Preorder α✝
    α : Type u_1
    inst✝ : Preorder α
    a : α
    n : ENat
    h : ∀ (p : LTSeries α), Eq (RelSeries.last p) a → LE.le (↑p.length) n
    p : LTSeries α
    hlast : LE.le (RelSeries.last p) a
    hlenpos : Ne p.length 0
    ⊢ LE.le (↑p.length) n
  -/
  let p' := p.eraseLast.snoc a (lt_of_lt_of_le (p.eraseLast_last_rel_last (by simp_all)) hlast)
  /-
    α✝ : Type u_1
    inst✝¹ : Preorder α✝
    α : Type u_1
    inst✝ : Preorder α
    a : α
    n : ENat
    h : ∀ (p : LTSeries α), Eq (RelSeries.last p) a → LE.le (↑p.length) n
    p : LTSeries α
    hlast : LE.le (RelSeries.last p) a
    hlenpos : Ne p.length 0
    p' : RelSeries fun x1 x2 => LT.lt x1 x2 := (RelSeries.eraseLast p).snoc a ⋯
    ⊢ LE.le (↑p.length) n
  -/
  rw [show p.length = p'.length by simp [p']; omega]
  /-
    α✝ : Type u_1
    inst✝¹ : Preorder α✝
    α : Type u_1
    inst✝ : Preorder α
    a : α
    n : ENat
    h : ∀ (p : LTSeries α), Eq (RelSeries.last p) a → LE.le (↑p.length) n
    p : LTSeries α
    hlast : LE.le (RelSeries.last p) a
    hlenpos : Ne p.length 0
    p' : RelSeries fun x1 x2 => LT.lt x1 x2 := (RelSeries.eraseLast p).snoc a ⋯
    ⊢ LE.le (↑p'.length) n
  -/
  apply h
  /-
    case a
    α✝ : Type u_1
    inst✝¹ : Preorder α✝
    α : Type u_1
    inst✝ : Preorder α
    a : α
    n : ENat
    h : ∀ (p : LTSeries α), Eq (RelSeries.last p) a → LE.le (↑p.length) n
    p : LTSeries α
    hlast : LE.le (RelSeries.last p) a
    hlenpos : Ne p.length 0
    p' : RelSeries fun x1 x2 => LT.lt x1 x2 := (RelSeries.eraseLast p).snoc a ⋯
    ⊢ Eq p'.last a
  -/
  simp [p']
  /-
    🎉 no goals
  -/


/--
Variant of `height_le_iff` ranging only over those series that end exactly on `a`.
-/
lemma height_le_iff' {a : α} {n : ℕ∞} :
    height a ≤ n ↔ ∀ ⦃p : LTSeries α⦄, p.last = a → p.length ≤ n := by
 /-
   α : Type u_1
   inst✝ : Preorder α
   a : α
   n : ENat
   ⊢ Iff (LE.le (Order.height a) n) (∀ ⦃p : LTSeries α⦄, Eq (RelSeries.last p) a  …
 -/
 constructor
   /-
     case mp
     α : Type u_1
     inst✝ : Preorder α
     a : α
     n : ENat
     ⊢ LE.le (Order.height a) n → ∀ ⦃p : LTSeries α⦄, Eq (RelSeries.last p) a → LE. …
   -/
 · rw [height_le_iff]
   /-
     case mp
     α : Type u_1
     inst✝ : Preorder α
     a : α
     n : ENat
     ⊢ (∀ ⦃p : LTSeries α⦄, LE.le (RelSeries.last p) a → LE.le (↑p.length) n) → ∀ ⦃ …
   -/
   exact fun h p hlast => h (le_of_eq hlast)
   /-
     🎉 no goals
   -/
   /-
     case mpr
     α : Type u_1
     inst✝ : Preorder α
     a : α
     n : ENat
     ⊢ (∀ ⦃p : LTSeries α⦄, Eq (RelSeries.last p) a → LE.le (↑p.length) n) → LE.le  …
   -/
 · exact height_le
   /-
     🎉 no goals
   -/


/--
Alternative definition of height, with the supremum ranging only over those series that end at `a`.
-/
lemma height_eq_iSup_last_eq (a : α) :
    height a = ⨆ (p : LTSeries α) (_ : p.last = a), ↑(p.length) := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    a : α
    ⊢ Eq (Order.height a) (iSup fun p => iSup fun x => ↑p.length)
  -/
  apply eq_of_forall_ge_iff
  /-
    case H
    α : Type u_1
    inst✝ : Preorder α
    a : α
    ⊢ ∀ (c : ENat), Iff (LE.le (Order.height a) c) (LE.le (iSup fun p => iSup fun  …
  -/
  intro n
  /-
    case H
    α : Type u_1
    inst✝ : Preorder α
    a : α
    n : ENat
    ⊢ Iff (LE.le (Order.height a) n) (LE.le (iSup fun p => iSup fun x => ↑p.length …
  -/
  rw [height_le_iff', iSup₂_le_iff]
  /-
    🎉 no goals
  -/


/--
Alternative definition of coheight, with the supremum only ranging over those series
that begin at `a`.
-/
lemma coheight_eq_iSup_head_eq (a : α) :
    coheight a = ⨆ (p : LTSeries α) (_ : p.head = a), ↑(p.length) := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    a : α
    ⊢ Eq (Order.coheight a) (iSup fun p => iSup fun x => ↑p.length)
  -/
  show height (α := αᵒᵈ) a = ⨆ (p : LTSeries α) (_ : p.head = a), ↑(p.length)
  /-
    α : Type u_1
    inst✝ : Preorder α
    a : α
    ⊢ Eq (Order.height a) (iSup fun p => iSup fun x => ↑p.length)
  -/
  rw [height_eq_iSup_last_eq]
  apply Equiv.iSup_congr ⟨RelSeries.reverse, RelSeries.reverse, RelSeries.reverse_reverse,
    RelSeries.reverse_reverse⟩
  /-
    α : Type u_1
    inst✝ : Preorder α
    a : α
    ⊢ ∀ (x : RelSeries fun x1 x2 => LT.lt x1 x2), Eq (iSup fun x_1 => ↑({ toFun := …
  -/
  simp
  /-
    🎉 no goals
  -/


/--
Variant of `coheight_le_iff` ranging only over those series that begin exactly on `a`.
-/
lemma coheight_le_iff' {a : α} {n : ℕ∞} :
    coheight a ≤ n ↔ ∀ ⦃p : LTSeries α⦄, p.head = a → p.length ≤ n := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    a : α
    n : ENat
    ⊢ Iff (LE.le (Order.coheight a) n) (∀ ⦃p : LTSeries α⦄, Eq (RelSeries.head p)  …
  -/
  rw [coheight_eq_iSup_head_eq, iSup₂_le_iff]
  /-
    🎉 no goals
  -/


lemma coheight_le {a : α} {n : ℕ∞} (h : ∀ (p : LTSeries α), p.head = a → p.length ≤ n) :
    coheight a ≤ n :=
  coheight_le_iff'.mpr h


lemma length_le_height {p : LTSeries α} {x : α} (hlast : p.last ≤ x) :
    p.length ≤ height x := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    p : LTSeries α
    x : α
    hlast : LE.le (RelSeries.last p) x
    ⊢ LE.le (↑p.length) (Order.height x)
  -/
  by_cases hlen0 : p.length ≠ 0
  · let p' := p.eraseLast.snoc x (by
      apply lt_of_lt_of_le
      · apply p.step ⟨p.length - 1, by omega⟩
      · convert hlast
        simp only [Fin.succ_mk, Nat.succ_eq_add_one, RelSeries.last, Fin.last]
        congr; omega)
    suffices p'.length ≤ height x by
      simp [p'] at this
      convert this
      norm_cast
      omega
    /-
      case pos
      α : Type u_1
      inst✝ : Preorder α
      p : LTSeries α
      x : α
      hlast : LE.le (RelSeries.last p) x
      hlen0 : Ne p.length 0
      p' : RelSeries fun x1 x2 => LT.lt x1 x2 := (RelSeries.eraseLast p).snoc x ⋯
      ⊢ LE.le (↑p'.length) (Order.height x)
    -/
    refine le_iSup₂_of_le p' ?_ le_rfl
    /-
      case pos
      α : Type u_1
      inst✝ : Preorder α
      p : LTSeries α
      x : α
      hlast : LE.le (RelSeries.last p) x
      hlen0 : Ne p.length 0
      p' : RelSeries fun x1 x2 => LT.lt x1 x2 := (RelSeries.eraseLast p).snoc x ⋯
      ⊢ LE.le p'.last x
    -/
    simp [p']
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : Preorder α
      p : LTSeries α
      x : α
      hlast : LE.le (RelSeries.last p) x
      hlen0 : Not (Ne p.length 0)
      ⊢ LE.le (↑p.length) (Order.height x)
    -/
  · simp_all
    /-
      🎉 no goals
    -/


lemma length_le_coheight {x : α} {p : LTSeries α} (hhead : x ≤ p.head) :
    p.length ≤ coheight x :=
                                                   /-
                                                     α : Type u_1
                                                     inst✝ : Preorder α
                                                     x : α
                                                     p : LTSeries α
                                                     hhead : LE.le x (RelSeries.head p)
                                                     ⊢ LE.le (RelSeries.reverse p).last x
                                                   -/
  length_le_height (α := αᵒᵈ) (p := p.reverse) (by simpa)
                                                   /-
                                                     🎉 no goals
                                                   -/


/--
The height of the last element in a series is larger or equal to the length of the series.
-/
lemma length_le_height_last {p : LTSeries α} : p.length ≤ height p.last :=
  length_le_height le_rfl


/--
The coheight of the first element in a series is larger or equal to the length of the series.
-/
lemma length_le_coheight_head {p : LTSeries α} : p.length ≤ coheight p.head :=
  length_le_coheight le_rfl


/--
The height of an element in a series is larger or equal to its index in the series.
-/
lemma index_le_height (p : LTSeries α) (i : Fin (p.length + 1)) : i ≤ height (p i) :=
  length_le_height_last (p := p.take i)


/--
The coheight of an element in a series is larger or equal to its reverse index in the series.
-/
lemma rev_index_le_coheight (p : LTSeries α) (i : Fin (p.length + 1)) : i.rev ≤ coheight (p i) := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    p : LTSeries α
    i : Fin (HAdd.hAdd p.length 1)
    ⊢ LE.le (↑↑i.rev) (Order.coheight (p.toFun i))
  -/
  simpa using index_le_height (α := αᵒᵈ) p.reverse i.rev
  /-
    🎉 no goals
  -/


/--
In a maximally long series, i.e one as long as the height of the last element, the height of each
element is its index in the series.
-/
lemma height_eq_index_of_length_eq_height_last {p : LTSeries α} (h : p.length = height p.last)
    (i : Fin (p.length + 1)) : height (p i) = i := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    p : LTSeries α
    h : Eq (↑p.length) (Order.height (RelSeries.last p))
    i : Fin (HAdd.hAdd p.length 1)
    ⊢ Eq (Order.height (p.toFun i)) ↑↑i
  -/
  refine le_antisymm (height_le ?_) (index_le_height p i)
  /-
    α : Type u_1
    inst✝ : Preorder α
    p : LTSeries α
    h : Eq (↑p.length) (Order.height (RelSeries.last p))
    i : Fin (HAdd.hAdd p.length 1)
    ⊢ ∀ (p_1 : LTSeries α), Eq (RelSeries.last p_1) (p.toFun i) → LE.le ↑p_1.lengt …
  -/
  intro p' hp'
  /-
    α : Type u_1
    inst✝ : Preorder α
    p : LTSeries α
    h : Eq (↑p.length) (Order.height (RelSeries.last p))
    i : Fin (HAdd.hAdd p.length 1)
    p' : LTSeries α
    hp' : Eq (RelSeries.last p') (p.toFun i)
    ⊢ LE.le ↑p'.length ↑↑i
  -/
  have hp'' := length_le_height_last (p := p'.smash (p.drop i) (by simpa))
  /-
    α : Type u_1
    inst✝ : Preorder α
    p : LTSeries α
    h : Eq (↑p.length) (Order.height (RelSeries.last p))
    i : Fin (HAdd.hAdd p.length 1)
    p' : LTSeries α
    hp' : Eq (RelSeries.last p') (p.toFun i)
    hp'' : LE.le (↑(RelSeries.smash p' (RelSeries.drop p i) ⋯).length) (Order.heig …
    ⊢ LE.le ↑p'.length ↑↑i
  -/
  simp [← h] at hp''; clear h
  /-
    α : Type u_1
    inst✝ : Preorder α
    p : LTSeries α
    i : Fin (HAdd.hAdd p.length 1)
    p' : LTSeries α
    hp' : Eq (RelSeries.last p') (p.toFun i)
    hp'' : LE.le (HAdd.hAdd (↑p'.length) (HSub.hSub ↑p.length ↑↑i)) ↑p.length
    ⊢ LE.le ↑p'.length ↑↑i
  -/
  norm_cast at *
  /-
    α : Type u_1
    inst✝ : Preorder α
    p : LTSeries α
    i : Fin (HAdd.hAdd p.length 1)
    p' : LTSeries α
    hp' : Eq (RelSeries.last p') (p.toFun i)
    hp'' : LE.le (HAdd.hAdd p'.length (HSub.hSub p.length ↑i)) p.length
    ⊢ LE.le p'.length ↑i
  -/
  omega
  /-
    🎉 no goals
  -/


/--
In a maximally long series, i.e one as long as the coheight of the first element, the coheight of
each element is its reverse index in the series.
-/
lemma coheight_eq_index_of_length_eq_head_coheight {p : LTSeries α} (h : p.length = coheight p.head)
    (i : Fin (p.length + 1)) : coheight (p i) = i.rev := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    p : LTSeries α
    h : Eq (↑p.length) (Order.coheight (RelSeries.head p))
    i : Fin (HAdd.hAdd p.length 1)
    ⊢ Eq (Order.coheight (p.toFun i)) ↑↑i.rev
  -/
  simpa using height_eq_index_of_length_eq_height_last (α := αᵒᵈ) (p := p.reverse) (by simpa) i.rev
  /-
    🎉 no goals
  -/


lemma height_mono : Monotone (α := α) height :=
  fun _ _ hab ↦ biSup_mono (fun _ hla => hla.trans hab)


@[gcongr] protected lemma _root_.GCongr.height_le_height (a b : α) (hab : a ≤ b) :
    height a ≤ height b := height_mono hab


lemma coheight_anti : Antitone (α := α) coheight :=
  (height_mono (α := αᵒᵈ)).dual_left


@[gcongr] protected lemma _root_.GCongr.coheight_le_coheight (a b : α) (hba : b ≤ a) :
    coheight a ≤ coheight b := coheight_anti hba


private lemma height_add_const (a : α) (n : ℕ∞) :
    height a + n = ⨆ (p : LTSeries α) (_ : p.last = a), p.length + n := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    a : α
    n : ENat
    ⊢ Eq (HAdd.hAdd (Order.height a) n) (iSup fun p => iSup fun x => HAdd.hAdd (↑p …
  -/
  have hne : Nonempty { p : LTSeries α // p.last = a } := ⟨RelSeries.singleton _ a, rfl⟩
  /-
    α : Type u_1
    inst✝ : Preorder α
    a : α
    n : ENat
    hne : Nonempty (Subtype fun p => Eq (RelSeries.last p) a)
    ⊢ Eq (HAdd.hAdd (Order.height a) n) (iSup fun p => iSup fun x => HAdd.hAdd (↑p …
  -/
  rw [height_eq_iSup_last_eq, iSup_subtype', iSup_subtype', ENat.iSup_add]
  /-
    🎉 no goals
  -/

/- For elements of finite height, `height` is strictly monotone. -/

@[gcongr] lemma height_strictMono {x y : α} (hxy : x < y) (hfin : height x < ⊤) :
    height x < height y := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    x y : α
    hxy : LT.lt x y
    hfin : LT.lt (Order.height x) Top.top
    ⊢ LT.lt (Order.height x) (Order.height y)
  -/
  rw [← ENat.add_one_le_iff hfin.ne, height_add_const, iSup₂_le_iff]
  /-
    α : Type u_1
    inst✝ : Preorder α
    x y : α
    hxy : LT.lt x y
    hfin : LT.lt (Order.height x) Top.top
    ⊢ ∀ (i : LTSeries α), Eq (RelSeries.last i) x → LE.le (HAdd.hAdd (↑i.length) 1 …
  -/
  intro p hlast
  /-
    α : Type u_1
    inst✝ : Preorder α
    x y : α
    hxy : LT.lt x y
    hfin : LT.lt (Order.height x) Top.top
    p : LTSeries α
    hlast : Eq (RelSeries.last p) x
    ⊢ LE.le (HAdd.hAdd (↑p.length) 1) (Order.height y)
  -/
  have := length_le_height_last (p := p.snoc y (by simp [*]))
  /-
    α : Type u_1
    inst✝ : Preorder α
    x y : α
    hxy : LT.lt x y
    hfin : LT.lt (Order.height x) Top.top
    p : LTSeries α
    hlast : Eq (RelSeries.last p) x
    this : LE.le (↑(RelSeries.snoc p y ⋯).length) (Order.height (RelSeries.snoc p  …
    ⊢ LE.le (HAdd.hAdd (↑p.length) 1) (Order.height y)
  -/
  simpa using this
  /-
    🎉 no goals
  -/

/- For elements of finite height, `coheight` is strictly antitone. -/

@[gcongr] lemma coheight_strictAnti {x y : α} (hyx : y < x) (hfin : coheight x < ⊤) :
    coheight x < coheight y :=
  height_strictMono (α := αᵒᵈ) hyx hfin


lemma height_le_height_apply_of_strictMono (f : α → β) (hf : StrictMono f) (x : α) :
    height x ≤ height (f x) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : α → β
    hf : StrictMono f
    x : α
    ⊢ LE.le (Order.height x) (Order.height (f x))
  -/
  simp only [height_eq_iSup_last_eq]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : α → β
    hf : StrictMono f
    x : α
    ⊢ LE.le (iSup fun p => iSup fun x => ↑p.length) (iSup fun p => iSup fun x => ↑ …
  -/
  apply iSup₂_le
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : α → β
    hf : StrictMono f
    x : α
    ⊢ ∀ (i : LTSeries α), Eq (RelSeries.last i) x → LE.le (↑i.length) (iSup fun p  …
  -/
  intro p hlast
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : α → β
    hf : StrictMono f
    x : α
    p : LTSeries α
    hlast : Eq (RelSeries.last p) x
    ⊢ LE.le (↑p.length) (iSup fun p => iSup fun x => ↑p.length)
  -/
  apply le_iSup₂_of_le (p.map f hf) (by simp [hlast]) (by simp)
  /-
    🎉 no goals
  -/


lemma coheight_le_coheight_apply_of_strictMono (f : α → β) (hf : StrictMono f) (x : α) :
    coheight x ≤ coheight (f x) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : α → β
    hf : StrictMono f
    x : α
    ⊢ LE.le (Order.coheight x) (Order.coheight (f x))
  -/
  apply height_le_height_apply_of_strictMono (α := αᵒᵈ)
  /-
    case hf
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : α → β
    hf : StrictMono f
    x : α
    ⊢ StrictMono f
  -/
  exact fun _ _ h ↦ hf h
  /-
    🎉 no goals
  -/


@[simp]
lemma height_orderIso (f : α ≃o β) (x : α) : height (f x) = height x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : OrderIso α β
    x : α
    ⊢ Eq (Order.height (f x)) (Order.height x)
  -/
  apply le_antisymm
    /-
      case a
      α : Type u_1
      β : Type u_2
      inst✝¹ : Preorder α
      inst✝ : Preorder β
      f : OrderIso α β
      x : α
      ⊢ LE.le (Order.height (f x)) (Order.height x)
    -/
  · simpa using height_le_height_apply_of_strictMono _ f.symm.strictMono (f x)
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u_1
      β : Type u_2
      inst✝¹ : Preorder α
      inst✝ : Preorder β
      f : OrderIso α β
      x : α
      ⊢ LE.le (Order.height x) (Order.height (f x))
    -/
  · exact height_le_height_apply_of_strictMono _ f.strictMono x
    /-
      🎉 no goals
    -/


lemma coheight_orderIso (f : α ≃o β) (x : α) : coheight (f x) = coheight x :=
  height_orderIso (α := αᵒᵈ) f.dual x


private lemma exists_eq_iSup_of_iSup_eq_coe {α : Type*} [Nonempty α] {f : α → ℕ∞} {n : ℕ}
    (h : (⨆ x, f x) = n) : ∃ x, f x = n := by
  /-
    α : Type u_3
    inst✝ : Nonempty α
    f : α → ENat
    n : Nat
    h : Eq (iSup fun x => f x) ↑n
    ⊢ Exists fun x => Eq (f x) ↑n
  -/
  obtain ⟨x, hx⟩ := ENat.sSup_mem_of_nonempty_of_lt_top (h ▸ ENat.coe_lt_top _)
  /-
    case intro
    α : Type u_3
    inst✝ : Nonempty α
    f : α → ENat
    n : Nat
    h : Eq (iSup fun x => f x) ↑n
    x : α
    hx : Eq ((fun x => f x) x) (SupSet.sSup (Set.range fun x => f x))
    ⊢ Exists fun x => Eq (f x) ↑n
  -/
  use x
  /-
    case h
    α : Type u_3
    inst✝ : Nonempty α
    f : α → ENat
    n : Nat
    h : Eq (iSup fun x => f x) ↑n
    x : α
    hx : Eq ((fun x => f x) x) (SupSet.sSup (Set.range fun x => f x))
    ⊢ Eq (f x) ↑n
  -/
  simpa [hx] using h
  /-
    🎉 no goals
  -/


/-- There exist a series ending in a element for any length up to the element’s height.  -/
lemma exists_series_of_le_height (a : α) {n : ℕ} (h : n ≤ height a) :
    ∃ p : LTSeries α, p.last = a ∧ p.length = n := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    a : α
    n : Nat
    h : LE.le (↑n) (Order.height a)
    ⊢ Exists fun p => And (Eq (RelSeries.last p) a) (Eq p.length n)
  -/
  have hne : Nonempty { p : LTSeries α // p.last = a } := ⟨RelSeries.singleton _ a, rfl⟩
  cases ha : height a with
  | top =>
    clear h
    rw [height_eq_iSup_last_eq, iSup_subtype', ENat.iSup_coe_eq_top, bddAbove_def] at ha
    contrapose! ha
    use n
    rintro m ⟨⟨p, rfl⟩, hp⟩
    simp only at hp
    by_contra! hnm
    apply ha (p.drop ⟨m-n, by omega⟩) (by simp) (by simp; omega)
  | coe m =>
    rw [ha, Nat.cast_le] at h
    rw [height_eq_iSup_last_eq, iSup_subtype'] at ha
    obtain ⟨⟨p, hlast⟩, hlen⟩ := exists_eq_iSup_of_iSup_eq_coe ha
    simp only [Nat.cast_inj] at hlen
    use p.drop ⟨m-n, by omega⟩
    constructor
    · simp [hlast]
    · simp [hlen]; omega


lemma exists_series_of_le_coheight (a : α) {n : ℕ} (h : n ≤ coheight a) :
    ∃ p : LTSeries α, p.head = a ∧ p.length = n := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    a : α
    n : Nat
    h : LE.le (↑n) (Order.coheight a)
    ⊢ Exists fun p => And (Eq (RelSeries.head p) a) (Eq p.length n)
  -/
  obtain ⟨p, hp, hl⟩ := exists_series_of_le_height (α := αᵒᵈ) a h
  /-
    case intro.intro
    α : Type u_1
    inst✝ : Preorder α
    a : α
    n : Nat
    h : LE.le (↑n) (Order.coheight a)
    p : LTSeries (OrderDual α)
    hp : Eq (RelSeries.last p) a
    hl : Eq p.length n
    ⊢ Exists fun p => And (Eq (RelSeries.head p) a) (Eq p.length n)
  -/
  exact ⟨p.reverse, by simpa, by simpa⟩
  /-
    🎉 no goals
  -/


/-- For an element of finite height there exists a series ending in that element of that height. -/
lemma exists_series_of_height_eq_coe (a : α) {n : ℕ} (h : height a = n) :
    ∃ p : LTSeries α, p.last = a ∧ p.length = n :=
  exists_series_of_le_height a (le_of_eq h.symm)


lemma exists_series_of_coheight_eq_coe (a : α) {n : ℕ} (h : coheight a = n) :
    ∃ p : LTSeries α, p.head = a ∧ p.length = n :=
  exists_series_of_le_coheight a (le_of_eq h.symm)


/-- Another characterization of height, based on the supremum of the heights of elements below. -/
lemma height_eq_iSup_lt_height (x : α) : height x = ⨆ y < x, height y + 1 := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    x : α
    ⊢ Eq (Order.height x) (iSup fun y => iSup fun h => HAdd.hAdd (Order.height y) 1)
  -/
  apply le_antisymm
    /-
      case a
      α : Type u_1
      inst✝ : Preorder α
      x : α
      ⊢ LE.le (Order.height x) (iSup fun y => iSup fun h => HAdd.hAdd (Order.height  …
    -/
  · apply height_le
    /-
      case a.h
      α : Type u_1
      inst✝ : Preorder α
      x : α
      ⊢ ∀ (p : LTSeries α), Eq (RelSeries.last p) x → LE.le (↑p.length) (iSup fun y  …
    -/
    intro p hp
    cases hlen : p.length with
    | zero => simp
    | succ n =>
      apply le_iSup_of_le p.eraseLast.last
      apply le_iSup_of_le (by rw [← hp]; apply RelSeries.eraseLast_last_rel_last _ (by omega))
      rw [height_add_const]
      apply le_iSup₂_of_le p.eraseLast (by rfl) (by simp [hlen])
    /-
      case a
      α : Type u_1
      inst✝ : Preorder α
      x : α
      ⊢ LE.le (iSup fun y => iSup fun h => HAdd.hAdd (Order.height y) 1) (Order.heig …
    -/
  · apply iSup₂_le; intro y hyx
    /-
      case a.h
      α : Type u_1
      inst✝ : Preorder α
      x y : α
      hyx : LT.lt y x
      ⊢ LE.le (HAdd.hAdd (Order.height y) 1) (Order.height x)
    -/
    rw [height_add_const]
    /-
      case a.h
      α : Type u_1
      inst✝ : Preorder α
      x y : α
      hyx : LT.lt y x
      ⊢ LE.le (iSup fun p => iSup fun x => HAdd.hAdd (↑p.length) 1) (Order.height x)
    -/
    apply iSup₂_le; intro p hp
    /-
      case a.h.h
      α : Type u_1
      inst✝ : Preorder α
      x y : α
      hyx : LT.lt y x
      p : LTSeries α
      hp : Eq (RelSeries.last p) y
      ⊢ LE.le (HAdd.hAdd (↑p.length) 1) (Order.height x)
    -/
    apply le_iSup₂_of_le (p.snoc x (hp ▸ hyx)) (by simp) (by simp)
    /-
      🎉 no goals
    -/


/--
Another characterization of coheight, based on the supremum of the coheights of elements above.
-/
lemma coheight_eq_iSup_gt_coheight (x : α) : coheight x = ⨆ y > x, coheight y + 1 :=
  height_eq_iSup_lt_height (α := αᵒᵈ) x


lemma height_le_coe_iff {x : α} {n : ℕ} : height x ≤ n ↔ ∀ y < x, height y < n := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    x : α
    n : Nat
    ⊢ Iff (LE.le (Order.height x) ↑n) (∀ (y : α), LT.lt y x → LT.lt (Order.height  …
  -/
  conv_lhs => rw [height_eq_iSup_lt_height, iSup₂_le_iff]
  /-
    α : Type u_1
    inst✝ : Preorder α
    x : α
    n : Nat
    ⊢ Iff (∀ (i : α), LT.lt i x → LE.le (HAdd.hAdd (Order.height i) 1) ↑n) (∀ (y : …
  -/
  congr! 2 with y _
  /-
    case a.h.h'.a
    α : Type u_1
    inst✝ : Preorder α
    x : α
    n : Nat
    y : α
    a✝ : LT.lt y x
    ⊢ Iff (LE.le (HAdd.hAdd (Order.height y) 1) ↑n) (LT.lt (Order.height y) ↑n)
  -/
  cases height y
    /-
      case a.h.h'.a.top
      α : Type u_1
      inst✝ : Preorder α
      x : α
      n : Nat
      y : α
      a✝ : LT.lt y x
      ⊢ Iff (LE.le (HAdd.hAdd Top.top 1) ↑n) (LT.lt Top.top ↑n)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case a.h.h'.a.coe
      α : Type u_1
      inst✝ : Preorder α
      x : α
      n : Nat
      y : α
      a✝¹ : LT.lt y x
      a✝ : Nat
      ⊢ Iff (LE.le (HAdd.hAdd (↑a✝) 1) ↑n) (LT.lt ↑a✝ ↑n)
    -/
  · norm_cast
    /-
      🎉 no goals
    -/


lemma coheight_le_coe_iff {x : α} {n : ℕ} : coheight x ≤ n ↔ ∀ y > x, coheight y < n :=
  height_le_coe_iff (α := αᵒᵈ)


/--
The height of an element is infinite iff there exist series of arbitrary length ending in that
element.
-/
lemma height_eq_top_iff {x : α} :
    height x = ⊤ ↔ ∀ n, ∃ p : LTSeries α, p.last = x ∧ p.length = n where
  mp h n := by
    /-
      α : Type u_1
      inst✝ : Preorder α
      x : α
      h : Eq (Order.height x) Top.top
      n : Nat
      ⊢ Exists fun p => And (Eq (RelSeries.last p) x) (Eq p.length n)
    -/
    apply exists_series_of_le_height x (n := n)
    /-
      α : Type u_1
      inst✝ : Preorder α
      x : α
      h : Eq (Order.height x) Top.top
      n : Nat
      ⊢ LE.le (↑n) (Order.height x)
    -/
    simp [h]
    /-
      🎉 no goals
    -/
  mpr h := by
    /-
      α : Type u_1
      inst✝ : Preorder α
      x : α
      h : ∀ (n : Nat), Exists fun p => And (Eq (RelSeries.last p) x) (Eq p.length n)
      ⊢ Eq (Order.height x) Top.top
    -/
    rw [height_eq_iSup_last_eq, iSup_subtype', ENat.iSup_coe_eq_top, bddAbove_def]
    /-
      α : Type u_1
      inst✝ : Preorder α
      x : α
      h : ∀ (n : Nat), Exists fun p => And (Eq (RelSeries.last p) x) (Eq p.length n)
      ⊢ Not (Exists fun x_1 => ∀ (y : Nat), Membership.mem (Set.range fun x_2 => (↑x …
    -/
    push_neg
    /-
      α : Type u_1
      inst✝ : Preorder α
      x : α
      h : ∀ (n : Nat), Exists fun p => And (Eq (RelSeries.last p) x) (Eq p.length n)
      ⊢ ∀ (x_1 : Nat), Exists fun y => And (Membership.mem (Set.range fun x_2 => (↑x …
    -/
    intro n
    /-
      α : Type u_1
      inst✝ : Preorder α
      x : α
      h : ∀ (n : Nat), Exists fun p => And (Eq (RelSeries.last p) x) (Eq p.length n)
      n : Nat
      ⊢ Exists fun y => And (Membership.mem (Set.range fun x_1 => (↑x_1).length) y)  …
    -/
    obtain ⟨p, hlast, hp⟩ := h (n+1)
    /-
      case intro.intro
      α : Type u_1
      inst✝ : Preorder α
      x : α
      h : ∀ (n : Nat), Exists fun p => And (Eq (RelSeries.last p) x) (Eq p.length n)
      n : Nat
      p : LTSeries α
      hlast : Eq (RelSeries.last p) x
      hp : Eq p.length (HAdd.hAdd n 1)
      ⊢ Exists fun y => And (Membership.mem (Set.range fun x_1 => (↑x_1).length) y)  …
    -/
    exact ⟨p.length, ⟨⟨⟨p, hlast⟩, by simp [hp]⟩, by simp [hp]⟩⟩
    /-
      🎉 no goals
    -/


/--
The coheight of an element is infinite iff there exist series of arbitrary length ending in that
element.
-/
lemma coheight_eq_top_iff {x : α} :
    coheight x = ⊤ ↔ ∀ n, ∃ p : LTSeries α, p.head = x ∧ p.length = n := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    x : α
    ⊢ Iff (Eq (Order.coheight x) Top.top) (∀ (n : Nat), Exists fun p => And (Eq (R …
  -/
  convert height_eq_top_iff (α := αᵒᵈ) (x := x) using 2 with n
  /-
    case h.e'_2.h.a
    α : Type u_1
    inst✝ : Preorder α
    x : α
    n : Nat
    ⊢ Iff (Exists fun p => And (Eq (RelSeries.head p) x) (Eq p.length n)) (Exists  …
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
  constructor <;> (intro ⟨p, hp, hl⟩; use p.reverse; constructor <;> simpa)
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


/-- The elements of height zero are the minimal elements. -/
@[simp] lemma height_eq_zero {x : α} : height x = 0 ↔ IsMin x := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    x : α
    ⊢ Iff (Eq (Order.height x) 0) (IsMin x)
  -/
  simpa [isMin_iff_forall_not_lt] using height_le_coe_iff (x := x) (n := 0)
  /-
    🎉 no goals
  -/


protected alias ⟨_, IsMin.height_eq_zero⟩ := height_eq_zero


/-- The elements of coheight zero are the maximal elements. -/
@[simp] lemma coheight_eq_zero {x : α} : coheight x = 0 ↔ IsMax x :=
  height_eq_zero (α := αᵒᵈ)


protected alias ⟨_, IsMax.coheight_eq_zero⟩ := coheight_eq_zero


                                                                                          /-
                                                                                            α : Type u_3
                                                                                            inst✝¹ : Preorder α
                                                                                            inst✝ : OrderBot α
                                                                                            ⊢ Eq (Order.height Bot.bot) 0
                                                                                          -/
@[simp] lemma height_bot (α : Type*) [Preorder α] [OrderBot α] : height (⊥ : α) = 0 := by simp
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


                                                                                              /-
                                                                                                α : Type u_3
                                                                                                inst✝¹ : Preorder α
                                                                                                inst✝ : OrderTop α
                                                                                                ⊢ Eq (Order.coheight Top.top) 0
                                                                                              -/
@[simp] lemma coheight_top (α : Type*) [Preorder α] [OrderTop α] : coheight (⊤ : α) = 0 := by simp
                                                                                              /-
                                                                                                🎉 no goals
                                                                                              -/


lemma coe_lt_height_iff {x : α} {n : ℕ} (hfin : height x < ⊤) :
    n < height x ↔ ∃ y < x, height y = n where
  mp h := by
    /-
      α : Type u_1
      inst✝ : Preorder α
      x : α
      n : Nat
      hfin : LT.lt (Order.height x) Top.top
      h : LT.lt (↑n) (Order.height x)
      ⊢ Exists fun y => And (LT.lt y x) (Eq (Order.height y) ↑n)
    -/
    obtain ⟨m, hx : height x = m⟩ := Option.ne_none_iff_exists'.mp hfin.ne_top
    /-
      case intro
      α : Type u_1
      inst✝ : Preorder α
      x : α
      n : Nat
      hfin : LT.lt (Order.height x) Top.top
      h : LT.lt (↑n) (Order.height x)
      m : Nat
      hx : Eq (Order.height x) ↑m
      ⊢ Exists fun y => And (LT.lt y x) (Eq (Order.height y) ↑n)
    -/
    rw [hx] at h; norm_cast at h
    /-
      case intro
      α : Type u_1
      inst✝ : Preorder α
      x : α
      n : Nat
      hfin : LT.lt (Order.height x) Top.top
      m : Nat
      hx : Eq (Order.height x) ↑m
      h : LT.lt n m
      ⊢ Exists fun y => And (LT.lt y x) (Eq (Order.height y) ↑n)
    -/
    obtain ⟨p, hp, hlen⟩ := exists_series_of_height_eq_coe x hx
    /-
      case intro.intro.intro
      α : Type u_1
      inst✝ : Preorder α
      x : α
      n : Nat
      hfin : LT.lt (Order.height x) Top.top
      m : Nat
      hx : Eq (Order.height x) ↑m
      h : LT.lt n m
      p : LTSeries α
      hp : Eq (RelSeries.last p) x
      hlen : Eq p.length m
      ⊢ Exists fun y => And (LT.lt y x) (Eq (Order.height y) ↑n)
    -/
    use p ⟨n, by omega⟩
    /-
      case h
      α : Type u_1
      inst✝ : Preorder α
      x : α
      n : Nat
      hfin : LT.lt (Order.height x) Top.top
      m : Nat
      hx : Eq (Order.height x) ↑m
      h : LT.lt n m
      p : LTSeries α
      hp : Eq (RelSeries.last p) x
      hlen : Eq p.length m
      ⊢ And (LT.lt (p.toFun ⟨n, ⋯⟩) x) (Eq (Order.height (p.toFun ⟨n, ⋯⟩)) ↑n)
    -/
    constructor
      /-
        case h.left
        α : Type u_1
        inst✝ : Preorder α
        x : α
        n : Nat
        hfin : LT.lt (Order.height x) Top.top
        m : Nat
        hx : Eq (Order.height x) ↑m
        h : LT.lt n m
        p : LTSeries α
        hp : Eq (RelSeries.last p) x
        hlen : Eq p.length m
        ⊢ LT.lt (p.toFun ⟨n, ⋯⟩) x
      -/
    · rw [← hp]
      /-
        case h.left
        α : Type u_1
        inst✝ : Preorder α
        x : α
        n : Nat
        hfin : LT.lt (Order.height x) Top.top
        m : Nat
        hx : Eq (Order.height x) ↑m
        h : LT.lt n m
        p : LTSeries α
        hp : Eq (RelSeries.last p) x
        hlen : Eq p.length m
        ⊢ LT.lt (p.toFun ⟨n, ⋯⟩) (RelSeries.last p)
      -/
      apply LTSeries.strictMono
      /-
        case h.left.a
        α : Type u_1
        inst✝ : Preorder α
        x : α
        n : Nat
        hfin : LT.lt (Order.height x) Top.top
        m : Nat
        hx : Eq (Order.height x) ↑m
        h : LT.lt n m
        p : LTSeries α
        hp : Eq (RelSeries.last p) x
        hlen : Eq p.length m
        ⊢ LT.lt ⟨n, ⋯⟩ (Fin.last p.length)
      -/
      simp [Fin.last]; omega
                       /-
                         🎉 no goals
                       -/
      /-
        case h.right
        α : Type u_1
        inst✝ : Preorder α
        x : α
        n : Nat
        hfin : LT.lt (Order.height x) Top.top
        m : Nat
        hx : Eq (Order.height x) ↑m
        h : LT.lt n m
        p : LTSeries α
        hp : Eq (RelSeries.last p) x
        hlen : Eq p.length m
        ⊢ Eq (Order.height (p.toFun ⟨n, ⋯⟩)) ↑n
      -/
    · exact height_eq_index_of_length_eq_height_last (by simp [hlen, hp, hx]) ⟨n, by omega⟩
      /-
        🎉 no goals
      -/
  mpr := fun ⟨y, hyx, hy⟩ =>
    hy ▸ height_strictMono hyx (lt_of_le_of_lt (height_mono hyx.le) hfin)


lemma coe_lt_coheight_iff {x : α} {n : ℕ} (hfin : coheight x < ⊤):
    n < coheight x ↔ ∃ y > x, coheight y = n :=
  coe_lt_height_iff (α := αᵒᵈ) hfin


lemma height_eq_coe_add_one_iff {x : α} {n : ℕ} :
    height x = n + 1 ↔ height x < ⊤ ∧ (∃ y < x, height y = n) ∧ (∀ y < x, height y ≤ n) := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    x : α
    n : Nat
    ⊢ Iff (Eq (Order.height x) (HAdd.hAdd (↑n) 1)) (And (LT.lt (Order.height x) To …
  -/
  wlog hfin : height x < ⊤
    /-
      case inr
      α : Type u_1
      inst✝ : Preorder α
      x : α
      n : Nat
      this : ∀ {α : Type u_1} [inst : Preorder α] {x : α} {n : Nat}, LT.lt (Order.he …
      hfin : Not (LT.lt (Order.height x) Top.top)
      ⊢ Iff (Eq (Order.height x) (HAdd.hAdd (↑n) 1)) (And (LT.lt (Order.height x) To …
    -/
  · simp_all
    /-
      case inr
      α : Type u_1
      inst✝ : Preorder α
      x : α
      n : Nat
      this : ∀ {α : Type u_1} [inst : Preorder α] {x : α} {n : Nat}, LT.lt (Order.he …
      hfin : Eq (Order.height x) Top.top
      ⊢ Not (Eq Top.top (HAdd.hAdd (↑n) 1))
    -/
    exact ne_of_beq_false rfl
    /-
      🎉 no goals
    -/
  /-
    α✝ : Type u_1
    inst✝¹ : Preorder α✝
    α : Type u_1
    inst✝ : Preorder α
    x : α
    n : Nat
    hfin : LT.lt (Order.height x) Top.top
    ⊢ Iff (Eq (Order.height x) (HAdd.hAdd (↑n) 1)) (And (LT.lt (Order.height x) To …
  -/
  simp only [hfin, true_and]
  /-
    α✝ : Type u_1
    inst✝¹ : Preorder α✝
    α : Type u_1
    inst✝ : Preorder α
    x : α
    n : Nat
    hfin : LT.lt (Order.height x) Top.top
    ⊢ Iff (Eq (Order.height x) (HAdd.hAdd (↑n) 1)) (And (Exists fun y => And (LT.l …
  -/
  trans n < height x ∧ height x ≤ n + 1
    /-
      α✝ : Type u_1
      inst✝¹ : Preorder α✝
      α : Type u_1
      inst✝ : Preorder α
      x : α
      n : Nat
      hfin : LT.lt (Order.height x) Top.top
      ⊢ Iff (Eq (Order.height x) (HAdd.hAdd (↑n) 1)) (And (LT.lt (↑n) (Order.height  …
    -/
  · rw [le_antisymm_iff, and_comm]
    /-
      α✝ : Type u_1
      inst✝¹ : Preorder α✝
      α : Type u_1
      inst✝ : Preorder α
      x : α
      n : Nat
      hfin : LT.lt (Order.height x) Top.top
      ⊢ Iff (And (LE.le (HAdd.hAdd (↑n) 1) (Order.height x)) (LE.le (Order.height x) …
    -/
    simp [hfin, ENat.lt_add_one_iff, ENat.add_one_le_iff]
    /-
      🎉 no goals
    -/
    /-
      α✝ : Type u_1
      inst✝¹ : Preorder α✝
      α : Type u_1
      inst✝ : Preorder α
      x : α
      n : Nat
      hfin : LT.lt (Order.height x) Top.top
      ⊢ Iff (And (LT.lt (↑n) (Order.height x)) (LE.le (Order.height x) (HAdd.hAdd (↑ …
    -/
  · congr! 1
      /-
        case a.h.e'_1.a
        α✝ : Type u_1
        inst✝¹ : Preorder α✝
        α : Type u_1
        inst✝ : Preorder α
        x : α
        n : Nat
        hfin : LT.lt (Order.height x) Top.top
        ⊢ Iff (LT.lt (↑n) (Order.height x)) (Exists fun y => And (LT.lt y x) (Eq (Orde …
      -/
    · exact coe_lt_height_iff hfin
      /-
        🎉 no goals
      -/
      /-
        case a.h.e'_2.a
        α✝ : Type u_1
        inst✝¹ : Preorder α✝
        α : Type u_1
        inst✝ : Preorder α
        x : α
        n : Nat
        hfin : LT.lt (Order.height x) Top.top
        ⊢ Iff (LE.le (Order.height x) (HAdd.hAdd (↑n) 1)) (∀ (y : α), LT.lt y x → LE.l …
      -/
    · simpa [hfin, ENat.lt_add_one_iff] using height_le_coe_iff (x := x) (n := n+1)
      /-
        🎉 no goals
      -/


lemma coheight_eq_coe_add_one_iff {x : α} {n : ℕ} :
    coheight x = n + 1 ↔
      coheight x < ⊤ ∧ (∃ y > x, coheight y = n) ∧ (∀ y > x, coheight y ≤ n) :=
  height_eq_coe_add_one_iff (α := αᵒᵈ)


lemma height_eq_coe_iff {x : α} {n : ℕ} :
    height x = n ↔
      height x < ⊤ ∧ (n = 0 ∨ ∃ y < x, height y = n - 1) ∧ (∀ y < x, height y < n) := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    x : α
    n : Nat
    ⊢ Iff (Eq (Order.height x) ↑n) (And (LT.lt (Order.height x) Top.top) (And (Or  …
  -/
  wlog hfin : height x < ⊤
    /-
      case inr
      α : Type u_1
      inst✝ : Preorder α
      x : α
      n : Nat
      this : ∀ {α : Type u_1} [inst : Preorder α] {x : α} {n : Nat}, LT.lt (Order.he …
      hfin : Not (LT.lt (Order.height x) Top.top)
      ⊢ Iff (Eq (Order.height x) ↑n) (And (LT.lt (Order.height x) Top.top) (And (Or  …
    -/
  · simp_all
    /-
      🎉 no goals
    -/
  /-
    α✝ : Type u_1
    inst✝¹ : Preorder α✝
    α : Type u_1
    inst✝ : Preorder α
    x : α
    n : Nat
    hfin : LT.lt (Order.height x) Top.top
    ⊢ Iff (Eq (Order.height x) ↑n) (And (LT.lt (Order.height x) Top.top) (And (Or  …
  -/
  simp only [hfin, true_and]
  /-
    α✝ : Type u_1
    inst✝¹ : Preorder α✝
    α : Type u_1
    inst✝ : Preorder α
    x : α
    n : Nat
    hfin : LT.lt (Order.height x) Top.top
    ⊢ Iff (Eq (Order.height x) ↑n) (And (Or (Eq n 0) (Exists fun y => And (LT.lt y …
  -/
  cases n
  /-
    case zero
    α✝ : Type u_1
    inst✝¹ : Preorder α✝
    α : Type u_1
    inst✝ : Preorder α
    x : α
    hfin : LT.lt (Order.height x) Top.top
    ⊢ Iff (Eq (Order.height x) ↑0) (And (Or (Eq 0 0) (Exists fun y => And (LT.lt y …
  -/
  case zero => simp [isMin_iff_forall_not_lt]
  case succ n =>
    simp only [Nat.cast_add, Nat.cast_one, add_eq_zero, one_ne_zero, and_false, false_or]
    rw [height_eq_coe_add_one_iff]
    simp only [hfin, true_and]
    congr! 3
    rename_i y _
    cases height y <;> simp; norm_cast; omega


lemma coheight_eq_coe_iff {x : α} {n : ℕ} :
    coheight x = n ↔
      coheight x < ⊤ ∧ (n = 0 ∨ ∃ y > x, coheight y = n - 1) ∧ (∀ y > x, coheight y < n) :=
  height_eq_coe_iff (α := αᵒᵈ)


/-- The elements of finite height `n` are the minimal elements among those of height `≥ n`. -/
lemma height_eq_coe_iff_minimal_le_height {a : α} {n : ℕ} :
    height a = n ↔ Minimal (fun y => n ≤ height y) a := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    a : α
    n : Nat
    ⊢ Iff (Eq (Order.height a) ↑n) (Minimal (fun y => LE.le (↑n) (Order.height y)) …
  -/
  by_cases hfin : height a < ⊤
  · cases hn : n with
    | zero => simp
    | succ => simp [minimal_iff_forall_lt, height_eq_coe_add_one_iff, ENat.add_one_le_iff,
        coe_lt_height_iff, *]
  · suffices ∃ x < a, ↑n ≤ height x by
      simp_all [minimal_iff_forall_lt]
    /-
      case neg
      α : Type u_1
      inst✝ : Preorder α
      a : α
      n : Nat
      hfin : Not (LT.lt (Order.height a) Top.top)
      ⊢ Exists fun x => And (LT.lt x a) (LE.le (↑n) (Order.height x))
    -/
    simp only [not_lt, top_le_iff, height_eq_top_iff] at hfin
    /-
      case neg
      α : Type u_1
      inst✝ : Preorder α
      a : α
      n : Nat
      hfin : ∀ (n : Nat), Exists fun p => And (Eq (RelSeries.last p) a) (Eq p.length …
      ⊢ Exists fun x => And (LT.lt x a) (LE.le (↑n) (Order.height x))
    -/
    obtain ⟨p, rfl, hp⟩ := hfin (n+1)
    /-
      case neg.intro.intro
      α : Type u_1
      inst✝ : Preorder α
      n : Nat
      p : LTSeries α
      hp : Eq p.length (HAdd.hAdd n 1)
      hfin : ∀ (n : Nat), Exists fun p_1 => And (Eq (RelSeries.last p_1) (RelSeries. …
      ⊢ Exists fun x => And (LT.lt x (RelSeries.last p)) (LE.le (↑n) (Order.height x))
    -/
    use p.eraseLast.last, RelSeries.eraseLast_last_rel_last _ (by omega)
    /-
      case right
      α : Type u_1
      inst✝ : Preorder α
      n : Nat
      p : LTSeries α
      hp : Eq p.length (HAdd.hAdd n 1)
      hfin : ∀ (n : Nat), Exists fun p_1 => And (Eq (RelSeries.last p_1) (RelSeries. …
      ⊢ LE.le (↑n) (Order.height (RelSeries.eraseLast p).last)
    -/
    simpa [hp] using length_le_height_last (p := p.eraseLast)
    /-
      🎉 no goals
    -/


/-- The elements of finite coheight `n` are the maximal elements among those of coheight `≥ n`. -/
lemma coheight_eq_coe_iff_maximal_le_coheight {a : α} {n : ℕ} :
    coheight a = n ↔ Maximal (fun y => n ≤ coheight y) a :=
  height_eq_coe_iff_minimal_le_height (α := αᵒᵈ)


lemma LTSeries.length_le_krullDim (p : LTSeries α) : p.length ≤ krullDim α := le_sSup ⟨_, rfl⟩


lemma krullDim_eq_bot_iff : krullDim α = ⊥ ↔ IsEmpty α := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    ⊢ Iff (Eq (Order.krullDim α) Bot.bot) (IsEmpty α)
  -/
  rw [eq_bot_iff, krullDim, iSup_le_iff]
  /-
    α : Type u_1
    inst✝ : Preorder α
    ⊢ Iff (∀ (i : LTSeries α), LE.le (↑i.length) Bot.bot) (IsEmpty α)
  -/
  simp only [le_bot_iff, WithBot.natCast_ne_bot, isEmpty_iff]
  /-
    α : Type u_1
    inst✝ : Preorder α
    ⊢ Iff (LTSeries α → False) (α → False)
  -/
  exact ⟨fun H x ↦ H ⟨0, fun _ ↦ x, by simp⟩, (· <| · 1)⟩
  /-
    🎉 no goals
  -/


lemma krullDim_nonneg_iff : 0 ≤ krullDim α ↔ Nonempty α := by
  rw [← not_iff_not, not_le, not_nonempty_iff, ← krullDim_eq_bot_iff, ← WithBot.lt_coe_bot,
    bot_eq_zero, WithBot.coe_zero]


lemma krullDim_eq_bot [IsEmpty α] : krullDim α = ⊥ := krullDim_eq_bot_iff.mpr ‹_›


@[deprecated (since := "2024-12-22")] alias krullDim_eq_bot_of_isEmpty := krullDim_eq_bot


lemma krullDim_nonneg [Nonempty α] : 0 ≤ krullDim α := krullDim_nonneg_iff.mpr ‹_›


@[deprecated (since := "2024-12-22")] alias krullDim_nonneg_of_nonempty := krullDim_nonneg


lemma krullDim_nonpos_iff_forall_isMax : krullDim α ≤ 0 ↔ ∀ x : α, IsMax x := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    ⊢ Iff (LE.le (Order.krullDim α) 0) (∀ (x : α), IsMax x)
  -/
  simp only [krullDim, iSup_le_iff, isMax_iff_forall_not_lt]
  refine ⟨fun H x y h ↦ (H ⟨1, ![x, y],
    fun i ↦ by obtain rfl := Subsingleton.elim i 0; simpa⟩).not_lt (by simp), ?_⟩
    /-
      α : Type u_1
      inst✝ : Preorder α
      ⊢ (∀ (x b : α), Not (LT.lt x b)) → ∀ (i : LTSeries α), LE.le (↑i.length) 0
    -/
  · rintro H ⟨_ | n, l, h⟩
      /-
        case mk.zero
        α : Type u_1
        inst✝ : Preorder α
        H : ∀ (x b : α), Not (LT.lt x b)
        l : Fin (HAdd.hAdd 0 1) → α
        h : ∀ (i : Fin 0), LT.lt (l i.castSucc) (l i.succ)
        ⊢ LE.le (↑{ length := 0, toFun := l, step := h }.length) 0
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case mk.succ
        α : Type u_1
        inst✝ : Preorder α
        H : ∀ (x b : α), Not (LT.lt x b)
        n : Nat
        l : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1) → α
        h : ∀ (i : Fin (HAdd.hAdd n 1)), LT.lt (l i.castSucc) (l i.succ)
        ⊢ LE.le (↑{ length := HAdd.hAdd n 1, toFun := l, step := h }.length) 0
      -/
    · cases H (l 0) (l 1) (h 0)
      /-
        🎉 no goals
      -/


lemma krullDim_nonpos_iff_forall_isMin : krullDim α ≤ 0 ↔ ∀ x : α, IsMin x := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    ⊢ Iff (LE.le (Order.krullDim α) 0) (∀ (x : α), IsMin x)
  -/
  simp only [krullDim_nonpos_iff_forall_isMax, IsMax, IsMin]
  /-
    α : Type u_1
    inst✝ : Preorder α
    ⊢ Iff (∀ (x : α) ⦃b : α⦄, LE.le x b → LE.le b x) (∀ (x : α) ⦃b : α⦄, LE.le b x …
  -/
  exact forall_swap
  /-
    🎉 no goals
  -/


lemma krullDim_le_one_iff : krullDim α ≤ 1 ↔ ∀ x : α, IsMin x ∨ IsMax x := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    ⊢ Iff (LE.le (Order.krullDim α) 1) (∀ (x : α), Or (IsMin x) (IsMax x))
  -/
  rw [← not_iff_not]
  /-
    α : Type u_1
    inst✝ : Preorder α
    ⊢ Iff (Not (LE.le (Order.krullDim α) 1)) (Not (∀ (x : α), Or (IsMin x) (IsMax  …
  -/
  simp_rw [isMax_iff_forall_not_lt, isMin_iff_forall_not_lt, krullDim, iSup_le_iff]
  /-
    α : Type u_1
    inst✝ : Preorder α
    ⊢ Iff (Not (∀ (i : LTSeries α), LE.le (↑i.length) 1)) (Not (∀ (x : α), Or (∀ ( …
  -/
  push_neg
  /-
    α : Type u_1
    inst✝ : Preorder α
    ⊢ Iff (Exists fun i => LT.lt 1 ↑i.length) (Exists fun x => And (Exists fun b = …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      inst✝ : Preorder α
      ⊢ (Exists fun i => LT.lt 1 ↑i.length) → Exists fun x => And (Exists fun b => L …
    -/
  · rintro ⟨⟨_ | _ | n, l, hl⟩, hl'⟩
    /-
      case mp.intro.mk.zero
      α : Type u_1
      inst✝ : Preorder α
      l : Fin (HAdd.hAdd 0 1) → α
      hl : ∀ (i : Fin 0), LT.lt (l i.castSucc) (l i.succ)
      hl' : LT.lt 1 ↑{ length := 0, toFun := l, step := hl }.length
      ⊢ Exists fun x => And (Exists fun b => LT.lt b x) (Exists fun b => LT.lt x b)
    -/
    iterate 2 · cases hl'.not_le (by simp)
    /-
      case mp.intro.mk.succ.succ
      α : Type u_1
      inst✝ : Preorder α
      n : Nat
      l : Fin (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd n 1) 1) 1) → α
      hl : ∀ (i : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)), LT.lt (l i.castSucc) (l i.succ)
      hl' : LT.lt 1 ↑{ length := HAdd.hAdd (HAdd.hAdd n 1) 1, toFun := l, step := hl …
      ⊢ Exists fun x => And (Exists fun b => LT.lt b x) (Exists fun b => LT.lt x b)
    -/
    exact ⟨l 1, ⟨l 0, hl 0⟩, l 2, hl 1⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      inst✝ : Preorder α
      ⊢ (Exists fun x => And (Exists fun b => LT.lt b x) (Exists fun b => LT.lt x b) …
    -/
  · rintro ⟨x, ⟨y, hxy⟩, z, hzx⟩
    /-
      case mpr.intro.intro.intro.intro
      α : Type u_1
      inst✝ : Preorder α
      x y : α
      hxy : LT.lt y x
      z : α
      hzx : LT.lt x z
      ⊢ Exists fun i => LT.lt 1 ↑i.length
    -/
    exact ⟨⟨2, ![y, x, z], fun i ↦ by fin_cases i <;> simpa⟩, by simp⟩
    /-
      🎉 no goals
    -/


lemma krullDim_le_one_iff_forall_isMax {α : Type*} [PartialOrder α] [OrderBot α] :
    krullDim α ≤ 1 ↔ ∀ x : α, x ≠ ⊥ → IsMax x := by
  /-
    α : Type u_3
    inst✝¹ : PartialOrder α
    inst✝ : OrderBot α
    ⊢ Iff (LE.le (Order.krullDim α) 1) (∀ (x : α), Ne x Bot.bot → IsMax x)
  -/
  simp [krullDim_le_one_iff, ← or_iff_not_imp_left]
  /-
    🎉 no goals
  -/


lemma krullDim_le_one_iff_forall_isMin {α : Type*} [PartialOrder α] [OrderTop α] :
    krullDim α ≤ 1 ↔ ∀ x : α, x ≠ ⊤ → IsMin x := by
  /-
    α : Type u_3
    inst✝¹ : PartialOrder α
    inst✝ : OrderTop α
    ⊢ Iff (LE.le (Order.krullDim α) 1) (∀ (x : α), Ne x Top.top → IsMin x)
  -/
  simp [krullDim_le_one_iff, ← or_iff_not_imp_right]
  /-
    🎉 no goals
  -/


lemma krullDim_pos_iff : 0 < krullDim α ↔ ∃ x y : α, x < y := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    ⊢ Iff (LT.lt 0 (Order.krullDim α)) (Exists fun x => Exists fun y => LT.lt x y)
  -/
  rw [← not_iff_not]
  /-
    α : Type u_1
    inst✝ : Preorder α
    ⊢ Iff (Not (LT.lt 0 (Order.krullDim α))) (Not (Exists fun x => Exists fun y => …
  -/
  push_neg
  /-
    α : Type u_1
    inst✝ : Preorder α
    ⊢ Iff (LE.le (Order.krullDim α) 0) (∀ (x y : α), Not (LT.lt x y))
  -/
  simp_rw [← isMax_iff_forall_not_lt, ← krullDim_nonpos_iff_forall_isMax]
  /-
    🎉 no goals
  -/


lemma one_le_krullDim_iff : 1 ≤ krullDim α ↔ ∃ x y : α, x < y := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    ⊢ Iff (LE.le 1 (Order.krullDim α)) (Exists fun x => Exists fun y => LT.lt x y)
  -/
  rw [← krullDim_pos_iff, ← Nat.cast_zero, ← WithBot.add_one_le_iff, Nat.cast_zero, zero_add]
  /-
    🎉 no goals
  -/


lemma krullDim_nonpos_of_subsingleton [Subsingleton α] : krullDim α ≤ 0 := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : Subsingleton α
    ⊢ LE.le (Order.krullDim α) 0
  -/
  rw [krullDim_nonpos_iff_forall_isMax]
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : Subsingleton α
    ⊢ ∀ (x : α), IsMax x
  -/
  exact fun x y h ↦ (Subsingleton.elim x y).ge
  /-
    🎉 no goals
  -/


lemma krullDim_eq_zero_of_unique [Unique α] : krullDim α = 0 :=
  le_antisymm krullDim_nonpos_of_subsingleton krullDim_nonneg


lemma krullDim_eq_length_of_finiteDimensionalOrder [FiniteDimensionalOrder α] :
    krullDim α = (LTSeries.longestOf α).length :=
  le_antisymm
    (iSup_le <| fun _ ↦ WithBot.coe_le_coe.mpr <| WithTop.coe_le_coe.mpr <|
      RelSeries.length_le_length_longestOf _ _) <|
    le_iSup (fun (i : LTSeries _) ↦ (i.length : WithBot (WithTop ℕ))) <| LTSeries.longestOf _


lemma krullDim_eq_top [InfiniteDimensionalOrder α] :
    krullDim α = ⊤ :=
  le_antisymm le_top <| le_iSup_iff.mpr <| fun m hm ↦ match m, hm with
  | ⊥, hm => False.elim <| by
    /-
      α : Type u_1
      inst✝¹ : Preorder α
      inst✝ : InfiniteDimensionalOrder α
      m : WithBot ENat
      hm✝ : ∀ (i : LTSeries α), LE.le (↑i.length) m
      hm : ∀ (i : LTSeries α), LE.le (↑i.length) Bot.bot
      ⊢ False
    -/
    haveI : Inhabited α := ⟨LTSeries.withLength _ 0 0⟩
    /-
      α : Type u_1
      inst✝¹ : Preorder α
      inst✝ : InfiniteDimensionalOrder α
      m : WithBot ENat
      hm✝ : ∀ (i : LTSeries α), LE.le (↑i.length) m
      hm : ∀ (i : LTSeries α), LE.le (↑i.length) Bot.bot
      this : Inhabited α
      ⊢ False
    -/
    exact not_le_of_lt (WithBot.bot_lt_coe _ : ⊥ < (0 : WithBot (WithTop ℕ))) <| hm default
    /-
      🎉 no goals
    -/
  | some ⊤, _ => le_refl _
  | some (some m), hm => by
    /-
      α : Type u_1
      inst✝¹ : Preorder α
      inst✝ : InfiniteDimensionalOrder α
      m✝ : WithBot ENat
      hm✝ : ∀ (i : LTSeries α), LE.le (↑i.length) m✝
      m : Nat
      hm : ∀ (i : LTSeries α), LE.le (↑i.length) (Option.some (Option.some m))
      ⊢ LE.le Top.top (Option.some (Option.some m))
    -/
    refine (not_lt_of_le (hm (LTSeries.withLength _ (m + 1))) ?_).elim
    /-
      α : Type u_1
      inst✝¹ : Preorder α
      inst✝ : InfiniteDimensionalOrder α
      m✝ : WithBot ENat
      hm✝ : ∀ (i : LTSeries α), LE.le (↑i.length) m✝
      m : Nat
      hm : ∀ (i : LTSeries α), LE.le (↑i.length) (Option.some (Option.some m))
      ⊢ LT.lt (Option.some (Option.some m)) ↑(LTSeries.withLength α (HAdd.hAdd m 1)) …
    -/
    erw [WithBot.coe_lt_coe, WithTop.coe_lt_coe]
    /-
      α : Type u_1
      inst✝¹ : Preorder α
      inst✝ : InfiniteDimensionalOrder α
      m✝ : WithBot ENat
      hm✝ : ∀ (i : LTSeries α), LE.le (↑i.length) m✝
      m : Nat
      hm : ∀ (i : LTSeries α), LE.le (↑i.length) (Option.some (Option.some m))
      ⊢ LT.lt m ↑(LTSeries.withLength α (HAdd.hAdd m 1)).length
    -/
    simp
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-12-22")]
alias krullDim_eq_top_of_infiniteDimensionalOrder := krullDim_eq_top


lemma krullDim_eq_top_iff : krullDim α = ⊤ ↔ InfiniteDimensionalOrder α := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    ⊢ Iff (Eq (Order.krullDim α) Top.top) (InfiniteDimensionalOrder α)
  -/
  refine ⟨fun h ↦ ?_, fun _ ↦ krullDim_eq_top⟩
  /-
    α : Type u_1
    inst✝ : Preorder α
    h : Eq (Order.krullDim α) Top.top
    ⊢ InfiniteDimensionalOrder α
  -/
  cases isEmpty_or_nonempty α
    /-
      case inl
      α : Type u_1
      inst✝ : Preorder α
      h : Eq (Order.krullDim α) Top.top
      h✝ : IsEmpty α
      ⊢ InfiniteDimensionalOrder α
    -/
  · simp [krullDim_eq_bot] at h
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    inst✝ : Preorder α
    h : Eq (Order.krullDim α) Top.top
    h✝ : Nonempty α
    ⊢ InfiniteDimensionalOrder α
  -/
  cases finiteDimensionalOrder_or_infiniteDimensionalOrder α
    /-
      case inr.inl
      α : Type u_1
      inst✝ : Preorder α
      h : Eq (Order.krullDim α) Top.top
      h✝¹ : Nonempty α
      h✝ : FiniteDimensionalOrder α
      ⊢ InfiniteDimensionalOrder α
    -/
  · rw [krullDim_eq_length_of_finiteDimensionalOrder] at h
    /-
      case inr.inl
      α : Type u_1
      inst✝ : Preorder α
      h✝¹ : Nonempty α
      h✝ : FiniteDimensionalOrder α
      h : Eq (↑(LTSeries.longestOf α).length) Top.top
      ⊢ InfiniteDimensionalOrder α
    -/
    cases h
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      α : Type u_1
      inst✝ : Preorder α
      h : Eq (Order.krullDim α) Top.top
      h✝¹ : Nonempty α
      h✝ : InfiniteDimensionalOrder α
      ⊢ InfiniteDimensionalOrder α
    -/
  · infer_instance
    /-
      🎉 no goals
    -/


lemma le_krullDim_iff {n : ℕ} : n ≤ krullDim α ↔ ∃ l : LTSeries α, l.length = n := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    n : Nat
    ⊢ Iff (LE.le (↑n) (Order.krullDim α)) (Exists fun l => Eq l.length n)
  -/
  cases isEmpty_or_nonempty α
    /-
      case inl
      α : Type u_1
      inst✝ : Preorder α
      n : Nat
      h✝ : IsEmpty α
      ⊢ Iff (LE.le (↑n) (Order.krullDim α)) (Exists fun l => Eq l.length n)
    -/
  · simp [krullDim_eq_bot]
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    inst✝ : Preorder α
    n : Nat
    h✝ : Nonempty α
    ⊢ Iff (LE.le (↑n) (Order.krullDim α)) (Exists fun l => Eq l.length n)
  -/
  cases finiteDimensionalOrder_or_infiniteDimensionalOrder α
    /-
      case inr.inl
      α : Type u_1
      inst✝ : Preorder α
      n : Nat
      h✝¹ : Nonempty α
      h✝ : FiniteDimensionalOrder α
      ⊢ Iff (LE.le (↑n) (Order.krullDim α)) (Exists fun l => Eq l.length n)
    -/
  · rw [krullDim_eq_length_of_finiteDimensionalOrder, Nat.cast_le]
    /-
      case inr.inl
      α : Type u_1
      inst✝ : Preorder α
      n : Nat
      h✝¹ : Nonempty α
      h✝ : FiniteDimensionalOrder α
      ⊢ Iff (LE.le n (LTSeries.longestOf α).length) (Exists fun l => Eq l.length n)
    -/
    constructor
      /-
        case inr.inl.mp
        α : Type u_1
        inst✝ : Preorder α
        n : Nat
        h✝¹ : Nonempty α
        h✝ : FiniteDimensionalOrder α
        ⊢ LE.le n (LTSeries.longestOf α).length → Exists fun l => Eq l.length n
      -/
    · exact fun H ↦ ⟨(LTSeries.longestOf α).take ⟨_, Nat.lt_succ.mpr H⟩, rfl⟩
      /-
        🎉 no goals
      -/
      /-
        case inr.inl.mpr
        α : Type u_1
        inst✝ : Preorder α
        n : Nat
        h✝¹ : Nonempty α
        h✝ : FiniteDimensionalOrder α
        ⊢ (Exists fun l => Eq l.length n) → LE.le n (LTSeries.longestOf α).length
      -/
    · exact fun ⟨l, hl⟩ ↦ hl ▸ l.longestOf_is_longest
      /-
        🎉 no goals
      -/
    /-
      case inr.inr
      α : Type u_1
      inst✝ : Preorder α
      n : Nat
      h✝¹ : Nonempty α
      h✝ : InfiniteDimensionalOrder α
      ⊢ Iff (LE.le (↑n) (Order.krullDim α)) (Exists fun l => Eq l.length n)
    -/
  · simpa [krullDim_eq_top] using Rel.InfiniteDimensional.exists_relSeries_with_length n
    /-
      🎉 no goals
    -/


/-- A definition of krullDim for nonempty `α` that avoids `WithBot` -/
lemma krullDim_eq_iSup_length [Nonempty α] :
    krullDim α = ⨆ (p : LTSeries α), (p.length : ℕ∞) := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : Nonempty α
    ⊢ Eq (Order.krullDim α) ↑(iSup fun p => ↑p.length)
  -/
  unfold krullDim
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : Nonempty α
    ⊢ Eq (iSup fun p => ↑p.length) ↑(iSup fun p => ↑p.length)
  -/
  rw [WithBot.coe_iSup (OrderTop.bddAbove _)]
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : Nonempty α
    ⊢ Eq (iSup fun p => ↑p.length) (iSup fun i => ↑↑i.length)
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma krullDim_lt_coe_iff {n : ℕ} : krullDim α < n ↔ ∀ l : LTSeries α, l.length < n := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    n : Nat
    ⊢ Iff (LT.lt (Order.krullDim α) ↑n) (∀ (l : LTSeries α), LT.lt l.length n)
  -/
  rw [krullDim, ← WithBot.coe_natCast]
  /-
    α : Type u_1
    inst✝ : Preorder α
    n : Nat
    ⊢ Iff (LT.lt (iSup fun p => ↑p.length) ↑↑n) (∀ (l : LTSeries α), LT.lt l.lengt …
  -/
  cases' n with n
    /-
      case zero
      α : Type u_1
      inst✝ : Preorder α
      ⊢ Iff (LT.lt (iSup fun p => ↑p.length) ↑↑0) (∀ (l : LTSeries α), LT.lt l.lengt …
    -/
  · rw [ENat.coe_zero, ← bot_eq_zero, WithBot.lt_coe_bot]
    /-
      case zero
      α : Type u_1
      inst✝ : Preorder α
      ⊢ Iff (Eq (iSup fun p => ↑p.length) Bot.bot) (∀ (l : LTSeries α), LT.lt l.leng …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      α : Type u_1
      inst✝ : Preorder α
      n : Nat
      ⊢ Iff (LT.lt (iSup fun p => ↑p.length) ↑↑(HAdd.hAdd n 1)) (∀ (l : LTSeries α), …
    -/
  · simp [WithBot.lt_add_one_iff, WithBot.coe_natCast, Nat.lt_succ]
    /-
      🎉 no goals
    -/


lemma krullDim_le_of_strictMono (f : α → β) (hf : StrictMono f) : krullDim α ≤ krullDim β :=
  iSup_le fun p ↦ le_sSup ⟨p.map f hf, rfl⟩


lemma krullDim_le_of_strictComono_and_surj
    (f : α → β) (hf : ∀ ⦃a b⦄, f a < f b → a < b) (hf' : Function.Surjective f) :
    krullDim β ≤ krullDim α :=
  iSup_le fun p ↦ le_sSup ⟨p.comap _ hf hf', rfl⟩


lemma krullDim_eq_of_orderIso (f : α ≃o β) : krullDim α = krullDim β :=
  le_antisymm (krullDim_le_of_strictMono _ f.strictMono) <|
    krullDim_le_of_strictMono _ f.symm.strictMono


@[simp] lemma krullDim_orderDual : krullDim αᵒᵈ = krullDim α :=
  le_antisymm (iSup_le fun i ↦ le_sSup ⟨i.reverse, rfl⟩) <|
    iSup_le fun i ↦ le_sSup ⟨i.reverse, rfl⟩


lemma height_le_krullDim (a : α) : height a ≤ krullDim α := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    a : α
    ⊢ LE.le (↑(Order.height a)) (Order.krullDim α)
  -/
  have : Nonempty α := ⟨a⟩
  /-
    α : Type u_1
    inst✝ : Preorder α
    a : α
    this : Nonempty α
    ⊢ LE.le (↑(Order.height a)) (Order.krullDim α)
  -/
  rw [krullDim_eq_iSup_length]
  /-
    α : Type u_1
    inst✝ : Preorder α
    a : α
    this : Nonempty α
    ⊢ LE.le ↑(Order.height a) ↑(iSup fun p => ↑p.length)
  -/
  simp only [WithBot.coe_le_coe, iSup_le_iff]
  /-
    α : Type u_1
    inst✝ : Preorder α
    a : α
    this : Nonempty α
    ⊢ LE.le (Order.height a) (iSup fun p => ↑p.length)
  -/
  exact height_le fun p _ ↦ le_iSup_of_le p le_rfl
  /-
    🎉 no goals
  -/


lemma coheight_le_krullDim (a : α) : coheight a ≤ krullDim α := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    a : α
    ⊢ LE.le (↑(Order.coheight a)) (Order.krullDim α)
  -/
  simpa using height_le_krullDim (α := αᵒᵈ) a
  /-
    🎉 no goals
  -/


/--
The Krull dimension is the supremum of the elements' heights.

This version of the lemma assumes that `α` is nonempty. In this case, the coercion from `ℕ∞` to
`WithBot ℕ∞` is on the outside fo the right-hand side, which is usually more convenient.

If `α` were empty, then `krullDim α = ⊥`. See `krullDim_eq_iSup_height` for the more general
version, with the coercion under the supremum.
-/
lemma krullDim_eq_iSup_height_of_nonempty [Nonempty α] : krullDim α = ↑(⨆ (a : α), height a) := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : Nonempty α
    ⊢ Eq (Order.krullDim α) ↑(iSup fun a => Order.height a)
  -/
  apply le_antisymm
    /-
      case a
      α : Type u_1
      inst✝¹ : Preorder α
      inst✝ : Nonempty α
      ⊢ LE.le (Order.krullDim α) ↑(iSup fun a => Order.height a)
    -/
  · apply iSup_le
    /-
      case a.h
      α : Type u_1
      inst✝¹ : Preorder α
      inst✝ : Nonempty α
      ⊢ ∀ (i : LTSeries α), LE.le ↑i.length ↑(iSup fun a => Order.height a)
    -/
    intro p
    suffices p.length ≤ ⨆ (a : α), height a by
      exact (WithBot.unbot'_le_iff fun _ => this).mp this
    /-
      case a.h
      α : Type u_1
      inst✝¹ : Preorder α
      inst✝ : Nonempty α
      p : LTSeries α
      ⊢ LE.le (↑p.length) (iSup fun a => Order.height a)
    -/
    apply le_iSup_of_le p.last (length_le_height_last (p := p))
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u_1
      inst✝¹ : Preorder α
      inst✝ : Nonempty α
      ⊢ LE.le (↑(iSup fun a => Order.height a)) (Order.krullDim α)
    -/
  · rw [WithBot.coe_iSup (by bddDefault)]
    /-
      case a
      α : Type u_1
      inst✝¹ : Preorder α
      inst✝ : Nonempty α
      ⊢ LE.le (iSup fun i => ↑(Order.height i)) (Order.krullDim α)
    -/
    apply iSup_le
    /-
      case a.h
      α : Type u_1
      inst✝¹ : Preorder α
      inst✝ : Nonempty α
      ⊢ ∀ (i : α), LE.le (↑(Order.height i)) (Order.krullDim α)
    -/
    apply height_le_krullDim
    /-
      🎉 no goals
    -/


/--
The Krull dimension is the supremum of the elements' coheights.

This version of the lemma assumes that `α` is nonempty. In this case, the coercion from `ℕ∞` to
`WithBot ℕ∞` is on the outside of the right-hand side, which is usually more convenient.

If `α` were empty, then `krullDim α = ⊥`. See `krullDim_eq_iSup_coheight` for the more general
version, with the coercion under the supremum.
-/
lemma krullDim_eq_iSup_coheight_of_nonempty [Nonempty α] :
    krullDim α = ↑(⨆ (a : α), coheight a) := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : Nonempty α
    ⊢ Eq (Order.krullDim α) ↑(iSup fun a => Order.coheight a)
  -/
  simpa using krullDim_eq_iSup_height_of_nonempty (α := αᵒᵈ)
  /-
    🎉 no goals
  -/


/--
The Krull dimension is the supremum of the elements' height plus coheight.
-/
lemma krullDim_eq_iSup_height_add_coheight_of_nonempty [Nonempty α] :
    krullDim α = ↑(⨆ (a : α), height a + coheight a) := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : Nonempty α
    ⊢ Eq (Order.krullDim α) ↑(iSup fun a => HAdd.hAdd (Order.height a) (Order.cohe …
  -/
  apply le_antisymm
    /-
      case a
      α : Type u_1
      inst✝¹ : Preorder α
      inst✝ : Nonempty α
      ⊢ LE.le (Order.krullDim α) ↑(iSup fun a => HAdd.hAdd (Order.height a) (Order.c …
    -/
  · rw [krullDim_eq_iSup_height_of_nonempty, WithBot.coe_le_coe]
    /-
      case a
      α : Type u_1
      inst✝¹ : Preorder α
      inst✝ : Nonempty α
      ⊢ LE.le (iSup fun a => Order.height a) (iSup fun a => HAdd.hAdd (Order.height  …
    -/
    apply ciSup_mono (by bddDefault) (by simp)
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u_1
      inst✝¹ : Preorder α
      inst✝ : Nonempty α
      ⊢ LE.le (↑(iSup fun a => HAdd.hAdd (Order.height a) (Order.coheight a))) (Orde …
    -/
  · wlog hnottop : krullDim α < ⊤
      /-
        case a.inr
        α : Type u_1
        inst✝¹ : Preorder α
        inst✝ : Nonempty α
        this : ∀ {α : Type u_1} [inst : Preorder α] [inst_1 : Nonempty α], LT.lt (Orde …
        hnottop : Not (LT.lt (Order.krullDim α) Top.top)
        ⊢ LE.le (↑(iSup fun a => HAdd.hAdd (Order.height a) (Order.coheight a))) (Orde …
      -/
    · simp_all
      /-
        🎉 no goals
      -/
    /-
      α✝ : Type u_1
      inst✝² : Preorder α✝
      α : Type u_1
      inst✝¹ : Preorder α
      inst✝ : Nonempty α
      hnottop : LT.lt (Order.krullDim α) Top.top
      ⊢ LE.le (↑(iSup fun a => HAdd.hAdd (Order.height a) (Order.coheight a))) (Orde …
    -/
    rw [krullDim_eq_iSup_length, WithBot.coe_le_coe]
    /-
      α✝ : Type u_1
      inst✝² : Preorder α✝
      α : Type u_1
      inst✝¹ : Preorder α
      inst✝ : Nonempty α
      hnottop : LT.lt (Order.krullDim α) Top.top
      ⊢ LE.le (iSup fun a => HAdd.hAdd (Order.height a) (Order.coheight a)) (iSup fu …
    -/
    apply iSup_le
    /-
      case h
      α✝ : Type u_1
      inst✝² : Preorder α✝
      α : Type u_1
      inst✝¹ : Preorder α
      inst✝ : Nonempty α
      hnottop : LT.lt (Order.krullDim α) Top.top
      ⊢ ∀ (i : α), LE.le (HAdd.hAdd (Order.height i) (Order.coheight i)) (iSup fun p …
    -/
    intro a
    /-
      case h
      α✝ : Type u_1
      inst✝² : Preorder α✝
      α : Type u_1
      inst✝¹ : Preorder α
      inst✝ : Nonempty α
      hnottop : LT.lt (Order.krullDim α) Top.top
      a : α
      ⊢ LE.le (HAdd.hAdd (Order.height a) (Order.coheight a)) (iSup fun p => ↑p.leng …
    -/
    have : height a < ⊤ := WithBot.coe_lt_coe.mp (lt_of_le_of_lt (height_le_krullDim a) hnottop)
    /-
      case h
      α✝ : Type u_1
      inst✝² : Preorder α✝
      α : Type u_1
      inst✝¹ : Preorder α
      inst✝ : Nonempty α
      hnottop : LT.lt (Order.krullDim α) Top.top
      a : α
      this : LT.lt (Order.height a) Top.top
      ⊢ LE.le (HAdd.hAdd (Order.height a) (Order.coheight a)) (iSup fun p => ↑p.leng …
    -/
    have : coheight a < ⊤ := WithBot.coe_lt_coe.mp (lt_of_le_of_lt (coheight_le_krullDim a) hnottop)
    cases hh : height a with
    | top => simp_all
    | coe n =>
      cases hch : coheight a with
      | top => simp_all
      | coe m =>
        obtain ⟨p₁, hlast, hlen₁⟩ := exists_series_of_height_eq_coe a hh
        obtain ⟨p₂, hhead, hlen₂⟩ := exists_series_of_coheight_eq_coe a hch
        apply le_iSup_of_le ((p₁.smash p₂) (by simp [*])) (by simp [*])


/--
The Krull dimension is the supremum of the elements' heights.

If `α` is `Nonempty`, then `krullDim_eq_iSup_height_of_nonempty`, with the coercion from
`ℕ∞` to `WithBot ℕ∞` outside the supremum, can be more convenient.
-/
lemma krullDim_eq_iSup_height : krullDim α = ⨆ (a : α), ↑(height a) := by
  cases isEmpty_or_nonempty α with
  | inl h => rw [krullDim_eq_bot, ciSup_of_empty]
  | inr h => rw [krullDim_eq_iSup_height_of_nonempty, WithBot.coe_iSup (OrderTop.bddAbove _)]


/--
The Krull dimension is the supremum of the elements' coheights.

If `α` is `Nonempty`, then `krullDim_eq_iSup_coheight_of_nonempty`, with the coercion from
`ℕ∞` to `WithBot ℕ∞` outside the supremum, can be more convenient.
-/
lemma krullDim_eq_iSup_coheight : krullDim α = ⨆ (a : α), ↑(coheight a) := by
  cases isEmpty_or_nonempty α with
  | inl h => rw [krullDim_eq_bot, ciSup_of_empty]
  | inr h => rw [krullDim_eq_iSup_coheight_of_nonempty, WithBot.coe_iSup (OrderTop.bddAbove _)]


@[simp] -- not as useful as a simp lemma as it looks, due to the coe on the left
lemma height_top_eq_krullDim [OrderTop α] : height (⊤ : α) = krullDim α := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : OrderTop α
    ⊢ Eq (↑(Order.height Top.top)) (Order.krullDim α)
  -/
  rw [krullDim_eq_iSup_length]
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : OrderTop α
    ⊢ Eq ↑(Order.height Top.top) ↑(iSup fun p => ↑p.length)
  -/
  simp only [WithBot.coe_inj]
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : OrderTop α
    ⊢ Eq (Order.height Top.top) (iSup fun p => ↑p.length)
  -/
  apply le_antisymm
    /-
      case a
      α : Type u_1
      inst✝¹ : Preorder α
      inst✝ : OrderTop α
      ⊢ LE.le (Order.height Top.top) (iSup fun p => ↑p.length)
    -/
  · exact height_le fun p _ ↦ le_iSup_of_le p le_rfl
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u_1
      inst✝¹ : Preorder α
      inst✝ : OrderTop α
      ⊢ LE.le (iSup fun p => ↑p.length) (Order.height Top.top)
    -/
  · exact iSup_le fun _ => length_le_height le_top
    /-
      🎉 no goals
    -/


@[simp] -- not as useful as a simp lemma as it looks, due to the coe on the left
lemma coheight_bot_eq_krullDim [OrderBot α] : coheight (⊥ : α) = krullDim α := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : OrderBot α
    ⊢ Eq (↑(Order.coheight Bot.bot)) (Order.krullDim α)
  -/
  rw [← krullDim_orderDual]
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : OrderBot α
    ⊢ Eq (↑(Order.coheight Bot.bot)) (Order.krullDim (OrderDual α))
  -/
  exact height_top_eq_krullDim (α := αᵒᵈ)
  /-
    🎉 no goals
  -/


@[simp] lemma height_nat (n : ℕ) : height n = n := by
  induction n using Nat.strongRecOn with | ind n ih =>
  apply le_antisymm
  · apply height_le_coe_iff.mpr
    simp +contextual only [ih, Nat.cast_lt, implies_true]
  · exact length_le_height_last (p := LTSeries.range n)


@[simp] lemma coheight_of_noMaxOrder [NoMaxOrder α] (a : α) : coheight a = ⊤ := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : NoMaxOrder α
    a : α
    ⊢ Eq (Order.coheight a) Top.top
  -/
  obtain ⟨f, hstrictmono⟩ := Nat.exists_strictMono ↑(Set.Ioi a)
  /-
    case intro
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : NoMaxOrder α
    a : α
    f : Nat → ↑(Set.Ioi a)
    hstrictmono : StrictMono f
    ⊢ Eq (Order.coheight a) Top.top
  -/
  apply coheight_eq_top_iff.mpr
  /-
    case intro
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : NoMaxOrder α
    a : α
    f : Nat → ↑(Set.Ioi a)
    hstrictmono : StrictMono f
    ⊢ ∀ (n : Nat), Exists fun p => And (Eq (RelSeries.head p) a) (Eq p.length n)
  -/
  intro m
  /-
    case intro
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : NoMaxOrder α
    a : α
    f : Nat → ↑(Set.Ioi a)
    hstrictmono : StrictMono f
    m : Nat
    ⊢ Exists fun p => And (Eq (RelSeries.head p) a) (Eq p.length m)
  -/
  use {length := m, toFun := fun i => if i = 0 then a else f i, step := ?step }
  /-
    case h
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : NoMaxOrder α
    a : α
    f : Nat → ↑(Set.Ioi a)
    hstrictmono : StrictMono f
    m : Nat
    ⊢ And (Eq { length := m, toFun := fun i => ite (Eq i 0) a ↑(f ↑i), step := ?st …
  -/
  case h => simp [RelSeries.head]
  case step =>
    intro ⟨i, hi⟩
    by_cases hzero : i = 0
    · subst i
      exact (f 1).prop
    · suffices f i < f (i + 1) by simp [Fin.ext_iff, hzero, this]
      apply hstrictmono
      omega


@[simp] lemma height_of_noMinOrder [NoMinOrder α] (a : α) : height a = ⊤ :=
  -- Implementation note: Here it's a bit easier to define the coheight variant first
  coheight_of_noMaxOrder (α := αᵒᵈ) a


@[simp] lemma krullDim_of_noMaxOrder [Nonempty α] [NoMaxOrder α] : krullDim α = ⊤ := by
  /-
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : Nonempty α
    inst✝ : NoMaxOrder α
    ⊢ Eq (Order.krullDim α) Top.top
  -/
  simp [krullDim_eq_iSup_coheight, coheight_of_noMaxOrder]
  /-
    🎉 no goals
  -/


@[simp] lemma krullDim_of_noMinOrder [Nonempty α] [NoMinOrder α] : krullDim α = ⊤ := by
  /-
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : Nonempty α
    inst✝ : NoMinOrder α
    ⊢ Eq (Order.krullDim α) Top.top
  -/
  simp [krullDim_eq_iSup_height, height_of_noMinOrder]
  /-
    🎉 no goals
  -/


lemma coheight_nat (n : ℕ) : coheight n = ⊤ := coheight_of_noMaxOrder ..


lemma krullDim_nat : krullDim ℕ = ⊤ := krullDim_of_noMaxOrder ..


lemma height_int (n : ℤ) : height n = ⊤ := height_of_noMinOrder ..


lemma coheight_int (n : ℤ) : coheight n = ⊤ := coheight_of_noMaxOrder ..


lemma krullDim_int : krullDim ℤ = ⊤ := krullDim_of_noMaxOrder ..


@[simp] lemma height_coe_withBot (x : α) : height (x : WithBot α) = height x + 1 := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    x : α
    ⊢ Eq (Order.height ↑x) (HAdd.hAdd (Order.height x) 1)
  -/
  apply le_antisymm
    /-
      case a
      α : Type u_1
      inst✝ : Preorder α
      x : α
      ⊢ LE.le (Order.height ↑x) (HAdd.hAdd (Order.height x) 1)
    -/
  · apply height_le
    /-
      case a.h
      α : Type u_1
      inst✝ : Preorder α
      x : α
      ⊢ ∀ (p : LTSeries (WithBot α)), Eq (RelSeries.last p) ↑x → LE.le (↑p.length) ( …
    -/
    intro p hlast
    /-
      case a.h
      α : Type u_1
      inst✝ : Preorder α
      x : α
      p : LTSeries (WithBot α)
      hlast : Eq (RelSeries.last p) ↑x
      ⊢ LE.le (↑p.length) (HAdd.hAdd (Order.height x) 1)
    -/
    wlog hlenpos : p.length ≠ 0
      /-
        case a.h.inr
        α : Type u_1
        inst✝ : Preorder α
        x : α
        p : LTSeries (WithBot α)
        hlast : Eq (RelSeries.last p) ↑x
        this : ∀ {α : Type u_1} [inst : Preorder α] (x : α) (p : LTSeries (WithBot α)) …
        hlenpos : Not (Ne p.length 0)
        ⊢ LE.le (↑p.length) (HAdd.hAdd (Order.height x) 1)
      -/
    · simp_all
      /-
        🎉 no goals
      -/
    -- essentially p' := (p.drop 1).map unbot
    let p' : LTSeries α := {
      length := p.length - 1
      toFun := fun ⟨i, hi⟩ => (p ⟨i+1, by omega⟩).unbot (by
        apply LT.lt.ne_bot (a := p.head)
        apply p.strictMono
        exact compare_gt_iff_gt.mp rfl)
      step := fun i => by simpa [WithBot.unbot_lt_iff] using p.step ⟨i + 1, by omega⟩ }
    have hlast' : p'.last = x := by
      simp only [p', RelSeries.last, Fin.val_last, WithBot.unbot_eq_iff, ← hlast, Fin.last]
      congr
      omega
    suffices p'.length ≤ height p'.last by
      simpa [p', hlast'] using this
    /-
      α✝ : Type u_1
      inst✝¹ : Preorder α✝
      α : Type u_1
      inst✝ : Preorder α
      x : α
      p : LTSeries (WithBot α)
      hlast : Eq (RelSeries.last p) ↑x
      hlenpos : Ne p.length 0
      p' : LTSeries α := { length := HSub.hSub p.length 1, toFun := fun x => Order.h …
      hlast' : Eq (RelSeries.last p') x
      ⊢ LE.le (↑p'.length) (Order.height (RelSeries.last p'))
    -/
    apply length_le_height_last
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u_1
      inst✝ : Preorder α
      x : α
      ⊢ LE.le (HAdd.hAdd (Order.height x) 1) (Order.height ↑x)
    -/
  · rw [height_add_const]
    /-
      case a
      α : Type u_1
      inst✝ : Preorder α
      x : α
      ⊢ LE.le (iSup fun p => iSup fun x => HAdd.hAdd (↑p.length) 1) (Order.height ↑x)
    -/
    apply iSup₂_le
    /-
      case a.h
      α : Type u_1
      inst✝ : Preorder α
      x : α
      ⊢ ∀ (i : LTSeries α), Eq (RelSeries.last i) x → LE.le (HAdd.hAdd (↑i.length) 1 …
    -/
    intro p hlast
    /-
      case a.h
      α : Type u_1
      inst✝ : Preorder α
      x : α
      p : LTSeries α
      hlast : Eq (RelSeries.last p) x
      ⊢ LE.le (HAdd.hAdd (↑p.length) 1) (Order.height ↑x)
    -/
    let p' := (p.map _ WithBot.coe_strictMono).cons ⊥ (by simp)
    /-
      case a.h
      α : Type u_1
      inst✝ : Preorder α
      x : α
      p : LTSeries α
      hlast : Eq (RelSeries.last p) x
      p' : RelSeries fun x1 x2 => LT.lt x1 x2 := RelSeries.cons (p.map (fun a => ↑a) …
      ⊢ LE.le (HAdd.hAdd (↑p.length) 1) (Order.height ↑x)
    -/
    apply le_iSup₂_of_le p' (by simp [p', hlast]) (by simp [p'])
    /-
      🎉 no goals
    -/


@[simp] lemma coheight_coe_withTop (x : α) : coheight (x : WithTop α) = coheight x + 1 :=
  height_coe_withBot (α := αᵒᵈ) x


@[simp] lemma height_coe_withTop (x : α) : height (x : WithTop α) = height x := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    x : α
    ⊢ Eq (Order.height ↑x) (Order.height x)
  -/
  apply le_antisymm
    /-
      case a
      α : Type u_1
      inst✝ : Preorder α
      x : α
      ⊢ LE.le (Order.height ↑x) (Order.height x)
    -/
  · apply height_le
    /-
      case a.h
      α : Type u_1
      inst✝ : Preorder α
      x : α
      ⊢ ∀ (p : LTSeries (WithTop α)), Eq (RelSeries.last p) ↑x → LE.le (↑p.length) ( …
    -/
    intro p hlast
    -- essentially p' := p.map untop
    let p' : LTSeries α := {
      length := p.length
      toFun := fun i => (p i).untop (by
        apply WithTop.lt_top_iff_ne_top.mp
        apply lt_of_le_of_lt
        · exact p.monotone (Fin.le_last _)
        · rw [RelSeries.last] at hlast
          simp [hlast])
      step := fun i => by simpa only [WithTop.untop_lt_iff, WithTop.coe_untop] using p.step i }
    have hlast' : p'.last = x := by
      simp only [p', RelSeries.last, Fin.val_last, WithTop.untop_eq_iff, ← hlast]
    suffices p'.length ≤ height p'.last by
      rw [hlast'] at this
      simpa [p'] using this
    /-
      case a.h
      α : Type u_1
      inst✝ : Preorder α
      x : α
      p : LTSeries (WithTop α)
      hlast : Eq (RelSeries.last p) ↑x
      p' : LTSeries α := { length := p.length, toFun := fun i => (p.toFun i).untop ⋯ …
      hlast' : Eq (RelSeries.last p') x
      ⊢ LE.le (↑p'.length) (Order.height (RelSeries.last p'))
    -/
    apply length_le_height_last
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u_1
      inst✝ : Preorder α
      x : α
      ⊢ LE.le (Order.height x) (Order.height ↑x)
    -/
  · apply height_le
    /-
      case a.h
      α : Type u_1
      inst✝ : Preorder α
      x : α
      ⊢ ∀ (p : LTSeries α), Eq (RelSeries.last p) x → LE.le (↑p.length) (Order.heigh …
    -/
    intro p hlast
    /-
      case a.h
      α : Type u_1
      inst✝ : Preorder α
      x : α
      p : LTSeries α
      hlast : Eq (RelSeries.last p) x
      ⊢ LE.le (↑p.length) (Order.height ↑x)
    -/
    let p' := p.map _ WithTop.coe_strictMono
    /-
      case a.h
      α : Type u_1
      inst✝ : Preorder α
      x : α
      p : LTSeries α
      hlast : Eq (RelSeries.last p) x
      p' : LTSeries (WithTop α) := p.map (fun a => ↑a) ⋯
      ⊢ LE.le (↑p.length) (Order.height ↑x)
    -/
    apply le_iSup₂_of_le p' (by simp [p', hlast]) (by simp [p'])
    /-
      🎉 no goals
    -/


@[simp] lemma coheight_coe_withBot (x : α) : coheight (x : WithBot α) = coheight x :=
  height_coe_withTop (α := αᵒᵈ) x


@[simp] lemma krullDim_WithTop [Nonempty α] : krullDim (WithTop α) = krullDim α + 1 := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : Nonempty α
    ⊢ Eq (Order.krullDim (WithTop α)) (HAdd.hAdd (Order.krullDim α) 1)
  -/
  rw [← height_top_eq_krullDim, krullDim_eq_iSup_height_of_nonempty, height_eq_iSup_lt_height]
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : Nonempty α
    ⊢ Eq (↑(iSup fun y => iSup fun h => HAdd.hAdd (Order.height y) 1)) (HAdd.hAdd  …
  -/
  norm_cast
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : Nonempty α
    ⊢ Eq (iSup fun y => iSup fun x => HAdd.hAdd (Order.height y) 1) (HAdd.hAdd (iS …
  -/
  simp_rw [WithTop.lt_top_iff_ne_top]
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : Nonempty α
    ⊢ Eq (iSup fun y => iSup fun x => HAdd.hAdd (Order.height y) 1) (HAdd.hAdd (iS …
  -/
  rw [ENat.iSup_add, iSup_subtype']
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : Nonempty α
    ⊢ Eq (iSup fun x => HAdd.hAdd (Order.height ↑x) 1) (iSup fun i => HAdd.hAdd (O …
  -/
  symm
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : Nonempty α
    ⊢ Eq (iSup fun i => HAdd.hAdd (Order.height i) 1) (iSup fun x => HAdd.hAdd (Or …
  -/
  apply Equiv.withTopSubtypeNe.symm.iSup_congr
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : Nonempty α
    ⊢ ∀ (x : α), Eq (HAdd.hAdd (Order.height ↑(Equiv.withTopSubtypeNe.symm x)) 1)  …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp] lemma krullDim_withBot [Nonempty α] : krullDim (WithBot α) = krullDim α + 1 := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : Nonempty α
    ⊢ Eq (Order.krullDim (WithBot α)) (HAdd.hAdd (Order.krullDim α) 1)
  -/
  conv_lhs => rw [← krullDim_orderDual]
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : Nonempty α
    ⊢ Eq (Order.krullDim (OrderDual (WithBot α))) (HAdd.hAdd (Order.krullDim α) 1)
  -/
  conv_rhs => rw [← krullDim_orderDual]
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : Nonempty α
    ⊢ Eq (Order.krullDim (OrderDual (WithBot α))) (HAdd.hAdd (Order.krullDim (Orde …
  -/
  exact krullDim_WithTop (α := αᵒᵈ)
  /-
    🎉 no goals
  -/


@[simp]
lemma krullDim_enat : krullDim ℕ∞ = ⊤ := by
  /-
    ⊢ Eq (Order.krullDim ENat) Top.top
  -/
  show (krullDim (WithTop ℕ) = ⊤)
  /-
    ⊢ Eq (Order.krullDim (WithTop Nat)) Top.top
  -/
  simp only [krullDim_WithTop, krullDim_nat]
  /-
    ⊢ Eq (HAdd.hAdd Top.top 1) Top.top
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma height_enat (n : ℕ∞) : height n = n := by
  cases n with
  | top => simp only [← WithBot.coe_eq_coe, height_top_eq_krullDim, krullDim_enat, WithBot.coe_top]
  | coe n => exact (height_coe_withTop _).trans (height_nat _)


@[simp]
lemma coheight_coe_enat (n : ℕ) : coheight (n : ℕ∞) = ⊤ := by
  /-
    n : Nat
    ⊢ Eq (Order.coheight ↑n) Top.top
  -/
  apply (coheight_coe_withTop _).trans
  /-
    n : Nat
    ⊢ Eq (HAdd.hAdd (Order.coheight n) 1) Top.top
  -/
  simp only [Nat.cast_id, coheight_nat, top_add]
  /-
    🎉 no goals
  -/


