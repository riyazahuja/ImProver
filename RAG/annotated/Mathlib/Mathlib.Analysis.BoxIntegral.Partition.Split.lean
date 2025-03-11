open scoped Classical in
/-- Given a box `I` and `x ∈ (I.lower i, I.upper i)`, the hyperplane `{y : ι → ℝ | y i = x}` splits
`I` into two boxes. `BoxIntegral.Box.splitLower I i x` is the box `I ∩ {y | y i ≤ x}`
(if it is nonempty). As usual, we represent a box that may be empty as
`WithBot (BoxIntegral.Box ι)`. -/
def splitLower (I : Box ι) (i : ι) (x : ℝ) : WithBot (Box ι) :=
  mk' I.lower (update I.upper i (min x (I.upper i)))


@[simp]
theorem coe_splitLower : (splitLower I i x : Set (ι → ℝ)) = ↑I ∩ { y | y i ≤ x } := by
  /-
    ι : Type u_1
    I : BoxIntegral.Box ι
    i : ι
    x : Real
    ⊢ Eq (↑(I.splitLower i x)) (Inter.inter (↑I) (setOf fun y => LE.le (y i) x))
  -/
  rw [splitLower, coe_mk']
  /-
    ι : Type u_1
    I : BoxIntegral.Box ι
    i : ι
    x : Real
    ⊢ Eq (Set.univ.pi fun i_1 => Set.Ioc (I.lower i_1) (Function.update I.upper i  …
  -/
  ext y
  simp only [mem_univ_pi, mem_Ioc, mem_inter_iff, mem_coe, mem_setOf_eq, forall_and, ← Pi.le_def,
    le_update_iff, le_min_iff, and_assoc, and_forall_ne (p := fun j => y j ≤ upper I j) i, mem_def]
  /-
    case h
    ι : Type u_1
    I : BoxIntegral.Box ι
    i : ι
    x : Real
    y : ι → Real
    ⊢ Iff (And (∀ (x : ι), LT.lt (I.lower x) (y x)) (And (LE.le (y i) x) (LE.le y  …
  -/
  rw [and_comm (a := y i ≤ x)]
  /-
    🎉 no goals
  -/


theorem splitLower_le : I.splitLower i x ≤ I :=
                                /-
                                  ι : Type u_1
                                  I : BoxIntegral.Box ι
                                  i : ι
                                  x : Real
                                  ⊢ HasSubset.Subset ↑(I.splitLower i x) ↑↑I
                                -/
  withBotCoe_subset_iff.1 <| by simp
                                /-
                                  🎉 no goals
                                -/


@[simp]
theorem splitLower_eq_bot {i x} : I.splitLower i x = ⊥ ↔ x ≤ I.lower i := by
  classical
  rw [splitLower, mk'_eq_bot, exists_update_iff I.upper fun j y => y ≤ I.lower j]
  simp [(I.lower_lt_upper _).not_le]


@[simp]
theorem splitLower_eq_self : I.splitLower i x = I ↔ I.upper i ≤ x := by
  /-
    ι : Type u_1
    I : BoxIntegral.Box ι
    i : ι
    x : Real
    ⊢ Iff (Eq (I.splitLower i x) ↑I) (LE.le (I.upper i) x)
  -/
  simp [splitLower, update_eq_iff]
  /-
    🎉 no goals
  -/


theorem splitLower_def [DecidableEq ι] {i x} (h : x ∈ Ioo (I.lower i) (I.upper i))
    (h' : ∀ j, I.lower j < update I.upper i x j :=
      (forall_update_iff I.upper fun j y => I.lower j < y).2
        ⟨h.1, fun _ _ => I.lower_lt_upper _⟩) :
    I.splitLower i x = (⟨I.lower, update I.upper i x, h'⟩ : Box ι) := by
  simp (config := { unfoldPartialApp := true }) only [splitLower, mk'_eq_coe, min_eq_left h.2.le,
    update, and_self]


open scoped Classical in
/-- Given a box `I` and `x ∈ (I.lower i, I.upper i)`, the hyperplane `{y : ι → ℝ | y i = x}` splits
`I` into two boxes. `BoxIntegral.Box.splitUpper I i x` is the box `I ∩ {y | x < y i}`
(if it is nonempty). As usual, we represent a box that may be empty as
`WithBot (BoxIntegral.Box ι)`. -/
def splitUpper (I : Box ι) (i : ι) (x : ℝ) : WithBot (Box ι) :=
  mk' (update I.lower i (max x (I.lower i))) I.upper


@[simp]
theorem coe_splitUpper : (splitUpper I i x : Set (ι → ℝ)) = ↑I ∩ { y | x < y i } := by
  classical
  rw [splitUpper, coe_mk']
  ext y
  simp only [mem_univ_pi, mem_Ioc, mem_inter_iff, mem_coe, mem_setOf_eq, forall_and,
    forall_update_iff I.lower fun j z => z < y j, max_lt_iff, and_assoc (a := x < y i),
    and_forall_ne (p := fun j => lower I j < y j) i, mem_def]
  exact and_comm


theorem splitUpper_le : I.splitUpper i x ≤ I :=
                                /-
                                  ι : Type u_1
                                  I : BoxIntegral.Box ι
                                  i : ι
                                  x : Real
                                  ⊢ HasSubset.Subset ↑(I.splitUpper i x) ↑↑I
                                -/
  withBotCoe_subset_iff.1 <| by simp
                                /-
                                  🎉 no goals
                                -/


@[simp]
theorem splitUpper_eq_bot {i x} : I.splitUpper i x = ⊥ ↔ I.upper i ≤ x := by
  classical
  rw [splitUpper, mk'_eq_bot, exists_update_iff I.lower fun j y => I.upper j ≤ y]
  simp [(I.lower_lt_upper _).not_le]


@[simp]
theorem splitUpper_eq_self : I.splitUpper i x = I ↔ x ≤ I.lower i := by
  /-
    ι : Type u_1
    I : BoxIntegral.Box ι
    i : ι
    x : Real
    ⊢ Iff (Eq (I.splitUpper i x) ↑I) (LE.le x (I.lower i))
  -/
  simp [splitUpper, update_eq_iff]
  /-
    🎉 no goals
  -/


theorem splitUpper_def [DecidableEq ι] {i x} (h : x ∈ Ioo (I.lower i) (I.upper i))
    (h' : ∀ j, update I.lower i x j < I.upper j :=
      (forall_update_iff I.lower fun j y => y < I.upper j).2
        ⟨h.2, fun _ _ => I.lower_lt_upper _⟩) :
    I.splitUpper i x = (⟨update I.lower i x, I.upper, h'⟩ : Box ι) := by
  simp (config := { unfoldPartialApp := true }) only [splitUpper, mk'_eq_coe, max_eq_left h.1.le,
    update, and_self]


theorem disjoint_splitLower_splitUpper (I : Box ι) (i : ι) (x : ℝ) :
    Disjoint (I.splitLower i x) (I.splitUpper i x) := by
  /-
    ι : Type u_1
    I : BoxIntegral.Box ι
    i : ι
    x : Real
    ⊢ Disjoint (I.splitLower i x) (I.splitUpper i x)
  -/
  rw [← disjoint_withBotCoe, coe_splitLower, coe_splitUpper]
  /-
    ι : Type u_1
    I : BoxIntegral.Box ι
    i : ι
    x : Real
    ⊢ Disjoint (Inter.inter (↑I) (setOf fun y => LE.le (y i) x)) (Inter.inter (↑I) …
  -/
  refine (Disjoint.inf_left' _ ?_).inf_right' _
  /-
    ι : Type u_1
    I : BoxIntegral.Box ι
    i : ι
    x : Real
    ⊢ Disjoint (setOf fun y => LE.le (y i) x) (setOf fun y => LT.lt x (y i))
  -/
  rw [Set.disjoint_left]
  /-
    ι : Type u_1
    I : BoxIntegral.Box ι
    i : ι
    x : Real
    ⊢ ∀ ⦃a : ι → Real⦄, Membership.mem (setOf fun y => LE.le (y i) x) a → Not (Mem …
  -/
  exact fun y (hle : y i ≤ x) hlt => not_lt_of_le hle hlt
  /-
    🎉 no goals
  -/


theorem splitLower_ne_splitUpper (I : Box ι) (i : ι) (x : ℝ) :
    I.splitLower i x ≠ I.splitUpper i x := by
  /-
    ι : Type u_1
    I : BoxIntegral.Box ι
    i : ι
    x : Real
    ⊢ Ne (I.splitLower i x) (I.splitUpper i x)
  -/
  cases' le_or_lt x (I.lower i) with h
    /-
      case inl
      ι : Type u_1
      I : BoxIntegral.Box ι
      i : ι
      x : Real
      h : LE.le x (I.lower i)
      ⊢ Ne (I.splitLower i x) (I.splitUpper i x)
    -/
  · rw [splitUpper_eq_self.2 h, splitLower_eq_bot.2 h]
    /-
      case inl
      ι : Type u_1
      I : BoxIntegral.Box ι
      i : ι
      x : Real
      h : LE.le x (I.lower i)
      ⊢ Ne Bot.bot ↑I
    -/
    exact WithBot.bot_ne_coe
    /-
      🎉 no goals
    -/
    /-
      case inr
      ι : Type u_1
      I : BoxIntegral.Box ι
      i : ι
      x : Real
      h✝ : LT.lt (I.lower i) x
      ⊢ Ne (I.splitLower i x) (I.splitUpper i x)
    -/
  · refine (disjoint_splitLower_splitUpper I i x).ne ?_
    /-
      case inr
      ι : Type u_1
      I : BoxIntegral.Box ι
      i : ι
      x : Real
      h✝ : LT.lt (I.lower i) x
      ⊢ Ne (I.splitLower i x) Bot.bot
    -/
    rwa [Ne, splitLower_eq_bot, not_le]
    /-
      🎉 no goals
    -/


open scoped Classical in
/-- The partition of `I : Box ι` into the boxes `I ∩ {y | y ≤ x i}` and `I ∩ {y | x i < y}`.
One of these boxes can be empty, then this partition is just the single-box partition `⊤`. -/
def split (I : Box ι) (i : ι) (x : ℝ) : Prepartition I :=
  ofWithBot {I.splitLower i x, I.splitUpper i x}
    (by
      /-
        ι : Type u_1
        M : Type u_2
        n : Nat
        I✝ J : BoxIntegral.Box ι
        i✝ : ι
        x✝ : Real
        I : BoxIntegral.Box ι
        i : ι
        x : Real
        ⊢ ∀ (J : WithBot (BoxIntegral.Box ι)), Membership.mem (Insert.insert (I.splitL …
      -/
      simp only [Finset.mem_insert, Finset.mem_singleton]
      /-
        ι : Type u_1
        M : Type u_2
        n : Nat
        I✝ J : BoxIntegral.Box ι
        i✝ : ι
        x✝ : Real
        I : BoxIntegral.Box ι
        i : ι
        x : Real
        ⊢ ∀ (J : WithBot (BoxIntegral.Box ι)), Or (Eq J (I.splitLower i x)) (Eq J (I.s …
      -/
      rintro J (rfl | rfl)
      /-
        case inl
        ι : Type u_1
        M : Type u_2
        n : Nat
        I✝ J : BoxIntegral.Box ι
        i✝ : ι
        x✝ : Real
        I : BoxIntegral.Box ι
        i : ι
        x : Real
        ⊢ LE.le (I.splitLower i x) ↑I
      -/
      exacts [Box.splitLower_le, Box.splitUpper_le])
      /-
        🎉 no goals
      -/
    (by
      simp only [Finset.coe_insert, Finset.coe_singleton, true_and, Set.mem_singleton_iff,
        pairwise_insert_of_symmetric symmetric_disjoint, pairwise_singleton]
      /-
        ι : Type u_1
        M : Type u_2
        n : Nat
        I✝ J : BoxIntegral.Box ι
        i✝ : ι
        x✝ : Real
        I : BoxIntegral.Box ι
        i : ι
        x : Real
        ⊢ ∀ (b : WithBot (BoxIntegral.Box ι)), Eq b (I.splitUpper i x) → Ne (I.splitLo …
      -/
      rintro J rfl -
      /-
        ι : Type u_1
        M : Type u_2
        n : Nat
        I✝ J : BoxIntegral.Box ι
        i✝ : ι
        x✝ : Real
        I : BoxIntegral.Box ι
        i : ι
        x : Real
        ⊢ Disjoint (I.splitLower i x) (I.splitUpper i x)
      -/
      exact I.disjoint_splitLower_splitUpper i x)
      /-
        🎉 no goals
      -/


@[simp]
theorem mem_split_iff : J ∈ split I i x ↔ ↑J = I.splitLower i x ∨ ↑J = I.splitUpper i x := by
  /-
    ι : Type u_1
    I J : BoxIntegral.Box ι
    i : ι
    x : Real
    ⊢ Iff (Membership.mem (BoxIntegral.Prepartition.split I i x) J) (Or (Eq (↑J) ( …
  -/
  simp [split]
  /-
    🎉 no goals
  -/


theorem mem_split_iff' : J ∈ split I i x ↔
    (J : Set (ι → ℝ)) = ↑I ∩ { y | y i ≤ x } ∨ (J : Set (ι → ℝ)) = ↑I ∩ { y | x < y i } := by
  /-
    ι : Type u_1
    I J : BoxIntegral.Box ι
    i : ι
    x : Real
    ⊢ Iff (Membership.mem (BoxIntegral.Prepartition.split I i x) J) (Or (Eq (↑J) ( …
  -/
  simp [mem_split_iff, ← Box.withBotCoe_inj]
  /-
    🎉 no goals
  -/


@[simp]
theorem iUnion_split (I : Box ι) (i : ι) (x : ℝ) : (split I i x).iUnion = I := by
  /-
    ι : Type u_1
    I : BoxIntegral.Box ι
    i : ι
    x : Real
    ⊢ Eq (BoxIntegral.Prepartition.split I i x).iUnion ↑I
  -/
  simp [split, ← inter_union_distrib_left, ← setOf_or, le_or_lt]
  /-
    🎉 no goals
  -/


theorem isPartitionSplit (I : Box ι) (i : ι) (x : ℝ) : IsPartition (split I i x) :=
  isPartition_iff_iUnion_eq.2 <| iUnion_split I i x

-- Porting note: In the type, changed `Option.elim` to `Option.elim'`

theorem sum_split_boxes {M : Type*} [AddCommMonoid M] (I : Box ι) (i : ι) (x : ℝ) (f : Box ι → M) :
    (∑ J ∈ (split I i x).boxes, f J) =
      (I.splitLower i x).elim' 0 f + (I.splitUpper i x).elim' 0 f := by
  classical
  rw [split, sum_ofWithBot, Finset.sum_pair (I.splitLower_ne_splitUpper i x)]


/-- If `x ∉ (I.lower i, I.upper i)`, then the hyperplane `{y | y i = x}` does not split `I`. -/
theorem split_of_not_mem_Ioo (h : x ∉ Ioo (I.lower i) (I.upper i)) : split I i x = ⊤ := by
  /-
    ι : Type u_1
    I : BoxIntegral.Box ι
    i : ι
    x : Real
    h : Not (Membership.mem (Set.Ioo (I.lower i) (I.upper i)) x)
    ⊢ Eq (BoxIntegral.Prepartition.split I i x) Top.top
  -/
  refine ((isPartitionTop I).eq_of_boxes_subset fun J hJ => ?_).symm
  /-
    ι : Type u_1
    I : BoxIntegral.Box ι
    i : ι
    x : Real
    h : Not (Membership.mem (Set.Ioo (I.lower i) (I.upper i)) x)
    J : BoxIntegral.Box ι
    hJ : Membership.mem Top.top.boxes J
    ⊢ Membership.mem (BoxIntegral.Prepartition.split I i x).boxes J
  -/
  rcases mem_top.1 hJ with rfl; clear hJ
  /-
    ι : Type u_1
    i : ι
    x : Real
    J : BoxIntegral.Box ι
    h : Not (Membership.mem (Set.Ioo (J.lower i) (J.upper i)) x)
    ⊢ Membership.mem (BoxIntegral.Prepartition.split J i x).boxes J
  -/
  rw [mem_boxes, mem_split_iff]
  /-
    ι : Type u_1
    i : ι
    x : Real
    J : BoxIntegral.Box ι
    h : Not (Membership.mem (Set.Ioo (J.lower i) (J.upper i)) x)
    ⊢ Or (Eq (↑J) (J.splitLower i x)) (Eq (↑J) (J.splitUpper i x))
  -/
  rw [mem_Ioo, not_and_or, not_lt, not_lt] at h
  /-
    ι : Type u_1
    i : ι
    x : Real
    J : BoxIntegral.Box ι
    h : Or (LE.le x (J.lower i)) (LE.le (J.upper i) x)
    ⊢ Or (Eq (↑J) (J.splitLower i x)) (Eq (↑J) (J.splitUpper i x))
  -/
  cases h <;> [right; left]
    /-
      case inl.h
      ι : Type u_1
      i : ι
      x : Real
      J : BoxIntegral.Box ι
      h✝ : LE.le x (J.lower i)
      ⊢ Eq (↑J) (J.splitUpper i x)
    -/
  · rwa [eq_comm, Box.splitUpper_eq_self]
    /-
      🎉 no goals
    -/
    /-
      case inr.h
      ι : Type u_1
      i : ι
      x : Real
      J : BoxIntegral.Box ι
      h✝ : LE.le (J.upper i) x
      ⊢ Eq (↑J) (J.splitLower i x)
    -/
  · rwa [eq_comm, Box.splitLower_eq_self]
    /-
      🎉 no goals
    -/


theorem coe_eq_of_mem_split_of_mem_le {y : ι → ℝ} (h₁ : J ∈ split I i x) (h₂ : y ∈ J)
    (h₃ : y i ≤ x) : (J : Set (ι → ℝ)) = ↑I ∩ { y | y i ≤ x } := by
  /-
    ι : Type u_1
    I J : BoxIntegral.Box ι
    i : ι
    x : Real
    y : ι → Real
    h₁ : Membership.mem (BoxIntegral.Prepartition.split I i x) J
    h₂ : Membership.mem J y
    h₃ : LE.le (y i) x
    ⊢ Eq (↑J) (Inter.inter (↑I) (setOf fun y => LE.le (y i) x))
  -/
  refine (mem_split_iff'.1 h₁).resolve_right fun H => ?_
  /-
    ι : Type u_1
    I J : BoxIntegral.Box ι
    i : ι
    x : Real
    y : ι → Real
    h₁ : Membership.mem (BoxIntegral.Prepartition.split I i x) J
    h₂ : Membership.mem J y
    h₃ : LE.le (y i) x
    H : Eq (↑J) (Inter.inter (↑I) (setOf fun y => LT.lt x (y i)))
    ⊢ False
  -/
  rw [← Box.mem_coe, H] at h₂
  /-
    ι : Type u_1
    I J : BoxIntegral.Box ι
    i : ι
    x : Real
    y : ι → Real
    h₁ : Membership.mem (BoxIntegral.Prepartition.split I i x) J
    h₂ : Membership.mem (Inter.inter (↑I) (setOf fun y => LT.lt x (y i))) y
    h₃ : LE.le (y i) x
    H : Eq (↑J) (Inter.inter (↑I) (setOf fun y => LT.lt x (y i)))
    ⊢ False
  -/
  exact h₃.not_lt h₂.2
  /-
    🎉 no goals
  -/


theorem coe_eq_of_mem_split_of_lt_mem {y : ι → ℝ} (h₁ : J ∈ split I i x) (h₂ : y ∈ J)
    (h₃ : x < y i) : (J : Set (ι → ℝ)) = ↑I ∩ { y | x < y i } := by
  /-
    ι : Type u_1
    I J : BoxIntegral.Box ι
    i : ι
    x : Real
    y : ι → Real
    h₁ : Membership.mem (BoxIntegral.Prepartition.split I i x) J
    h₂ : Membership.mem J y
    h₃ : LT.lt x (y i)
    ⊢ Eq (↑J) (Inter.inter (↑I) (setOf fun y => LT.lt x (y i)))
  -/
  refine (mem_split_iff'.1 h₁).resolve_left fun H => ?_
  /-
    ι : Type u_1
    I J : BoxIntegral.Box ι
    i : ι
    x : Real
    y : ι → Real
    h₁ : Membership.mem (BoxIntegral.Prepartition.split I i x) J
    h₂ : Membership.mem J y
    h₃ : LT.lt x (y i)
    H : Eq (↑J) (Inter.inter (↑I) (setOf fun y => LE.le (y i) x))
    ⊢ False
  -/
  rw [← Box.mem_coe, H] at h₂
  /-
    ι : Type u_1
    I J : BoxIntegral.Box ι
    i : ι
    x : Real
    y : ι → Real
    h₁ : Membership.mem (BoxIntegral.Prepartition.split I i x) J
    h₂ : Membership.mem (Inter.inter (↑I) (setOf fun y => LE.le (y i) x)) y
    h₃ : LT.lt x (y i)
    H : Eq (↑J) (Inter.inter (↑I) (setOf fun y => LE.le (y i) x))
    ⊢ False
  -/
  exact h₃.not_le h₂.2
  /-
    🎉 no goals
  -/


@[simp]
theorem restrict_split (h : I ≤ J) (i : ι) (x : ℝ) : (split J i x).restrict I = split I i x := by
  /-
    ι : Type u_1
    I J : BoxIntegral.Box ι
    h : LE.le I J
    i : ι
    x : Real
    ⊢ Eq ((BoxIntegral.Prepartition.split J i x).restrict I) (BoxIntegral.Preparti …
  -/
  refine ((isPartitionSplit J i x).restrict h).eq_of_boxes_subset ?_
  /-
    ι : Type u_1
    I J : BoxIntegral.Box ι
    h : LE.le I J
    i : ι
    x : Real
    ⊢ HasSubset.Subset ((BoxIntegral.Prepartition.split J i x).restrict I).boxes ( …
  -/
  simp only [Finset.subset_iff, mem_boxes, mem_restrict', exists_prop, mem_split_iff']
  /-
    ι : Type u_1
    I J : BoxIntegral.Box ι
    h : LE.le I J
    i : ι
    x : Real
    ⊢ ∀ ⦃x_1 : BoxIntegral.Box ι⦄, (Exists fun J' => And (Or (Eq (↑J') (Inter.inte …
  -/
  have : ∀ s, (I ∩ s : Set (ι → ℝ)) ⊆ J := fun s => inter_subset_left.trans h
  /-
    ι : Type u_1
    I J : BoxIntegral.Box ι
    h : LE.le I J
    i : ι
    x : Real
    this : ∀ (s : Set (ι → Real)), HasSubset.Subset (Inter.inter (↑I) s) ↑J
    ⊢ ∀ ⦃x_1 : BoxIntegral.Box ι⦄, (Exists fun J' => And (Or (Eq (↑J') (Inter.inte …
  -/
  rintro J₁ ⟨J₂, H₂ | H₂, H₁⟩ <;> [left; right] <;>
    /-
      case intro.intro.inl.h
      ι : Type u_1
      I J : BoxIntegral.Box ι
      h : LE.le I J
      i : ι
      x : Real
      this : ∀ (s : Set (ι → Real)), HasSubset.Subset (Inter.inter (↑I) s) ↑J
      J₁ J₂ : BoxIntegral.Box ι
      H₁ : Eq (↑J₁) (Inter.inter ↑I ↑J₂)
      H₂ : Eq (↑J₂) (Inter.inter (↑J) (setOf fun y => LE.le (y i) x))
      ⊢ Eq (↑J₁) (Inter.inter (↑I) (setOf fun y => LE.le (y i) x))
    -/
    /-
      🎉 no goals
    -/
    simp [H₁, H₂, inter_left_comm (I : Set (ι → ℝ)), this]
    /-
      🎉 no goals
    -/


theorem inf_split (π : Prepartition I) (i : ι) (x : ℝ) :
    π ⊓ split I i x = π.biUnion fun J => split J i x :=
  biUnion_congr_of_le rfl fun _ hJ => restrict_split hJ i x


/-- Split a box along many hyperplanes `{y | y i = x}`; each hyperplane is given by the pair
`(i x)`. -/
def splitMany (I : Box ι) (s : Finset (ι × ℝ)) : Prepartition I :=
  s.inf fun p => split I p.1 p.2


@[simp]
theorem splitMany_empty (I : Box ι) : splitMany I ∅ = ⊤ :=
  Finset.inf_empty


open scoped Classical in
@[simp]
theorem splitMany_insert (I : Box ι) (s : Finset (ι × ℝ)) (p : ι × ℝ) :
    splitMany I (insert p s) = splitMany I s ⊓ split I p.1 p.2 := by
  /-
    ι : Type u_1
    I : BoxIntegral.Box ι
    s : Finset (Prod ι Real)
    p : Prod ι Real
    ⊢ Eq (BoxIntegral.Prepartition.splitMany I (Insert.insert p s)) (Min.min (BoxI …
  -/
  rw [splitMany, Finset.inf_insert, inf_comm, splitMany]
  /-
    🎉 no goals
  -/


theorem splitMany_le_split (I : Box ι) {s : Finset (ι × ℝ)} {p : ι × ℝ} (hp : p ∈ s) :
    splitMany I s ≤ split I p.1 p.2 :=
  Finset.inf_le hp


theorem isPartition_splitMany (I : Box ι) (s : Finset (ι × ℝ)) : IsPartition (splitMany I s) := by
  classical
  exact Finset.induction_on s (by simp only [splitMany_empty, isPartitionTop]) fun a s _ hs => by
    simpa only [splitMany_insert, inf_split] using hs.biUnion fun J _ => isPartitionSplit _ _ _


@[simp]
theorem iUnion_splitMany (I : Box ι) (s : Finset (ι × ℝ)) : (splitMany I s).iUnion = I :=
  (isPartition_splitMany I s).iUnion_eq


theorem inf_splitMany {I : Box ι} (π : Prepartition I) (s : Finset (ι × ℝ)) :
    π ⊓ splitMany I s = π.biUnion fun J => splitMany J s := by
  classical
  induction' s using Finset.induction_on with p s _ ihp
  · simp
  · simp_rw [splitMany_insert, ← inf_assoc, ihp, inf_split, biUnion_assoc]


open scoped Classical in
/-- Let `s : Finset (ι × ℝ)` be a set of hyperplanes `{x : ι → ℝ | x i = r}` in `ι → ℝ` encoded as
pairs `(i, r)`. Suppose that this set contains all faces of a box `J`. The hyperplanes of `s` split
a box `I` into subboxes. Let `Js` be one of them. If `J` and `Js` have nonempty intersection, then
`Js` is a subbox of `J`. -/
theorem not_disjoint_imp_le_of_subset_of_mem_splitMany {I J Js : Box ι} {s : Finset (ι × ℝ)}
    (H : ∀ i, {(i, J.lower i), (i, J.upper i)} ⊆ s) (HJs : Js ∈ splitMany I s)
    (Hn : ¬Disjoint (J : WithBot (Box ι)) Js) : Js ≤ J := by
  /-
    ι : Type u_1
    I J Js : BoxIntegral.Box ι
    s : Finset (Prod ι Real)
    H : ∀ (i : ι), HasSubset.Subset (Insert.insert { fst := i, snd := J.lower i }  …
    HJs : Membership.mem (BoxIntegral.Prepartition.splitMany I s) Js
    Hn : Not (Disjoint ↑J ↑Js)
    ⊢ LE.le Js J
  -/
  simp only [Finset.insert_subset_iff, Finset.singleton_subset_iff] at H
  /-
    ι : Type u_1
    I J Js : BoxIntegral.Box ι
    s : Finset (Prod ι Real)
    HJs : Membership.mem (BoxIntegral.Prepartition.splitMany I s) Js
    Hn : Not (Disjoint ↑J ↑Js)
    H : ∀ (i : ι), And (Membership.mem s { fst := i, snd := J.lower i }) (Membersh …
    ⊢ LE.le Js J
  -/
  rcases Box.not_disjoint_coe_iff_nonempty_inter.mp Hn with ⟨x, hx, hxs⟩
  /-
    case intro.intro
    ι : Type u_1
    I J Js : BoxIntegral.Box ι
    s : Finset (Prod ι Real)
    HJs : Membership.mem (BoxIntegral.Prepartition.splitMany I s) Js
    Hn : Not (Disjoint ↑J ↑Js)
    H : ∀ (i : ι), And (Membership.mem s { fst := i, snd := J.lower i }) (Membersh …
    x : ι → Real
    hx : Membership.mem (↑J) x
    hxs : Membership.mem (↑Js) x
    ⊢ LE.le Js J
  -/
  refine fun y hy i => ⟨?_, ?_⟩
    /-
      case intro.intro.refine_1
      ι : Type u_1
      I J Js : BoxIntegral.Box ι
      s : Finset (Prod ι Real)
      HJs : Membership.mem (BoxIntegral.Prepartition.splitMany I s) Js
      Hn : Not (Disjoint ↑J ↑Js)
      H : ∀ (i : ι), And (Membership.mem s { fst := i, snd := J.lower i }) (Membersh …
      x : ι → Real
      hx : Membership.mem (↑J) x
      hxs : Membership.mem (↑Js) x
      y : ι → Real
      hy : Membership.mem Js y
      i : ι
      ⊢ LT.lt (J.lower i) (y i)
    -/
  · rcases splitMany_le_split I (H i).1 HJs with ⟨Jl, Hmem : Jl ∈ split I i (J.lower i), Hle⟩
    /-
      case intro.intro.refine_1.intro.intro
      ι : Type u_1
      I J Js : BoxIntegral.Box ι
      s : Finset (Prod ι Real)
      HJs : Membership.mem (BoxIntegral.Prepartition.splitMany I s) Js
      Hn : Not (Disjoint ↑J ↑Js)
      H : ∀ (i : ι), And (Membership.mem s { fst := i, snd := J.lower i }) (Membersh …
      x : ι → Real
      hx : Membership.mem (↑J) x
      hxs : Membership.mem (↑Js) x
      y : ι → Real
      hy : Membership.mem Js y
      i : ι
      Jl : BoxIntegral.Box ι
      Hmem : Membership.mem (BoxIntegral.Prepartition.split I i (J.lower i)) Jl
      Hle : LE.le Js Jl
      ⊢ LT.lt (J.lower i) (y i)
    -/
    have := Hle hxs
    /-
      case intro.intro.refine_1.intro.intro
      ι : Type u_1
      I J Js : BoxIntegral.Box ι
      s : Finset (Prod ι Real)
      HJs : Membership.mem (BoxIntegral.Prepartition.splitMany I s) Js
      Hn : Not (Disjoint ↑J ↑Js)
      H : ∀ (i : ι), And (Membership.mem s { fst := i, snd := J.lower i }) (Membersh …
      x : ι → Real
      hx : Membership.mem (↑J) x
      hxs : Membership.mem (↑Js) x
      y : ι → Real
      hy : Membership.mem Js y
      i : ι
      Jl : BoxIntegral.Box ι
      Hmem : Membership.mem (BoxIntegral.Prepartition.split I i (J.lower i)) Jl
      Hle : LE.le Js Jl
      this : Membership.mem Jl x
      ⊢ LT.lt (J.lower i) (y i)
    -/
    rw [← Box.coe_subset_coe, coe_eq_of_mem_split_of_lt_mem Hmem this (hx i).1] at Hle
    /-
      case intro.intro.refine_1.intro.intro
      ι : Type u_1
      I J Js : BoxIntegral.Box ι
      s : Finset (Prod ι Real)
      HJs : Membership.mem (BoxIntegral.Prepartition.splitMany I s) Js
      Hn : Not (Disjoint ↑J ↑Js)
      H : ∀ (i : ι), And (Membership.mem s { fst := i, snd := J.lower i }) (Membersh …
      x : ι → Real
      hx : Membership.mem (↑J) x
      hxs : Membership.mem (↑Js) x
      y : ι → Real
      hy : Membership.mem Js y
      i : ι
      Jl : BoxIntegral.Box ι
      Hmem : Membership.mem (BoxIntegral.Prepartition.split I i (J.lower i)) Jl
      Hle : HasSubset.Subset (↑Js) (Inter.inter (↑I) (setOf fun y => LT.lt (J.lower  …
      this : Membership.mem Jl x
      ⊢ LT.lt (J.lower i) (y i)
    -/
    exact (Hle hy).2
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      ι : Type u_1
      I J Js : BoxIntegral.Box ι
      s : Finset (Prod ι Real)
      HJs : Membership.mem (BoxIntegral.Prepartition.splitMany I s) Js
      Hn : Not (Disjoint ↑J ↑Js)
      H : ∀ (i : ι), And (Membership.mem s { fst := i, snd := J.lower i }) (Membersh …
      x : ι → Real
      hx : Membership.mem (↑J) x
      hxs : Membership.mem (↑Js) x
      y : ι → Real
      hy : Membership.mem Js y
      i : ι
      ⊢ LE.le (y i) (J.upper i)
    -/
  · rcases splitMany_le_split I (H i).2 HJs with ⟨Jl, Hmem : Jl ∈ split I i (J.upper i), Hle⟩
    /-
      case intro.intro.refine_2.intro.intro
      ι : Type u_1
      I J Js : BoxIntegral.Box ι
      s : Finset (Prod ι Real)
      HJs : Membership.mem (BoxIntegral.Prepartition.splitMany I s) Js
      Hn : Not (Disjoint ↑J ↑Js)
      H : ∀ (i : ι), And (Membership.mem s { fst := i, snd := J.lower i }) (Membersh …
      x : ι → Real
      hx : Membership.mem (↑J) x
      hxs : Membership.mem (↑Js) x
      y : ι → Real
      hy : Membership.mem Js y
      i : ι
      Jl : BoxIntegral.Box ι
      Hmem : Membership.mem (BoxIntegral.Prepartition.split I i (J.upper i)) Jl
      Hle : LE.le Js Jl
      ⊢ LE.le (y i) (J.upper i)
    -/
    have := Hle hxs
    /-
      case intro.intro.refine_2.intro.intro
      ι : Type u_1
      I J Js : BoxIntegral.Box ι
      s : Finset (Prod ι Real)
      HJs : Membership.mem (BoxIntegral.Prepartition.splitMany I s) Js
      Hn : Not (Disjoint ↑J ↑Js)
      H : ∀ (i : ι), And (Membership.mem s { fst := i, snd := J.lower i }) (Membersh …
      x : ι → Real
      hx : Membership.mem (↑J) x
      hxs : Membership.mem (↑Js) x
      y : ι → Real
      hy : Membership.mem Js y
      i : ι
      Jl : BoxIntegral.Box ι
      Hmem : Membership.mem (BoxIntegral.Prepartition.split I i (J.upper i)) Jl
      Hle : LE.le Js Jl
      this : Membership.mem Jl x
      ⊢ LE.le (y i) (J.upper i)
    -/
    rw [← Box.coe_subset_coe, coe_eq_of_mem_split_of_mem_le Hmem this (hx i).2] at Hle
    /-
      case intro.intro.refine_2.intro.intro
      ι : Type u_1
      I J Js : BoxIntegral.Box ι
      s : Finset (Prod ι Real)
      HJs : Membership.mem (BoxIntegral.Prepartition.splitMany I s) Js
      Hn : Not (Disjoint ↑J ↑Js)
      H : ∀ (i : ι), And (Membership.mem s { fst := i, snd := J.lower i }) (Membersh …
      x : ι → Real
      hx : Membership.mem (↑J) x
      hxs : Membership.mem (↑Js) x
      y : ι → Real
      hy : Membership.mem Js y
      i : ι
      Jl : BoxIntegral.Box ι
      Hmem : Membership.mem (BoxIntegral.Prepartition.split I i (J.upper i)) Jl
      Hle : HasSubset.Subset (↑Js) (Inter.inter (↑I) (setOf fun y => LE.le (y i) (J. …
      this : Membership.mem Jl x
      ⊢ LE.le (y i) (J.upper i)
    -/
    exact (Hle hy).2
    /-
      🎉 no goals
    -/


/-- Let `s` be a finite set of boxes in `ℝⁿ = ι → ℝ`. Then there exists a finite set `t₀` of
hyperplanes (namely, the set of all hyperfaces of boxes in `s`) such that for any `t ⊇ t₀`
and any box `I` in `ℝⁿ` the following holds. The hyperplanes from `t` split `I` into subboxes.
Let `J'` be one of them, and let `J` be one of the boxes in `s`. If these boxes have a nonempty
intersection, then `J' ≤ J`. -/
theorem eventually_not_disjoint_imp_le_of_mem_splitMany (s : Finset (Box ι)) :
    ∀ᶠ t : Finset (ι × ℝ) in atTop, ∀ (I : Box ι), ∀ J ∈ s, ∀ J' ∈ splitMany I t,
      ¬Disjoint (J : WithBot (Box ι)) J' → J' ≤ J := by
  classical
  cases nonempty_fintype ι
  refine eventually_atTop.2
    ⟨s.biUnion fun J => Finset.univ.biUnion fun i => {(i, J.lower i), (i, J.upper i)},
      fun t ht I J hJ J' hJ' => not_disjoint_imp_le_of_subset_of_mem_splitMany (fun i => ?_) hJ'⟩
  exact fun p hp =>
    ht (Finset.mem_biUnion.2 ⟨J, hJ, Finset.mem_biUnion.2 ⟨i, Finset.mem_univ _, hp⟩⟩)


theorem eventually_splitMany_inf_eq_filter (π : Prepartition I) :
    ∀ᶠ t : Finset (ι × ℝ) in atTop,
      π ⊓ splitMany I t = (splitMany I t).filter fun J => ↑J ⊆ π.iUnion := by
  /-
    ι : Type u_1
    I : BoxIntegral.Box ι
    inst✝ : Finite ι
    π : BoxIntegral.Prepartition I
    ⊢ Filter.Eventually (fun t => Eq (Min.min π (BoxIntegral.Prepartition.splitMan …
  -/
  refine (eventually_not_disjoint_imp_le_of_mem_splitMany π.boxes).mono fun t ht => ?_
  /-
    ι : Type u_1
    I : BoxIntegral.Box ι
    inst✝ : Finite ι
    π : BoxIntegral.Prepartition I
    t : Finset (Prod ι Real)
    ht : ∀ (I_1 J : BoxIntegral.Box ι), Membership.mem π.boxes J → ∀ (J' : BoxInte …
    ⊢ Eq (Min.min π (BoxIntegral.Prepartition.splitMany I t)) ((BoxIntegral.Prepar …
  -/
  refine le_antisymm ((biUnion_le_iff _).2 fun J hJ => ?_) (le_inf (fun J hJ => ?_) (filter_le _ _))
    /-
      case refine_1
      ι : Type u_1
      I : BoxIntegral.Box ι
      inst✝ : Finite ι
      π : BoxIntegral.Prepartition I
      t : Finset (Prod ι Real)
      ht : ∀ (I_1 J : BoxIntegral.Box ι), Membership.mem π.boxes J → ∀ (J' : BoxInte …
      J : BoxIntegral.Box ι
      hJ : Membership.mem π J
      ⊢ LE.le ((BoxIntegral.Prepartition.splitMany I t).restrict J) (((BoxIntegral.P …
    -/
  · refine ofWithBot_mono ?_
    /-
      case refine_1
      ι : Type u_1
      I : BoxIntegral.Box ι
      inst✝ : Finite ι
      π : BoxIntegral.Prepartition I
      t : Finset (Prod ι Real)
      ht : ∀ (I_1 J : BoxIntegral.Box ι), Membership.mem π.boxes J → ∀ (J' : BoxInte …
      J : BoxIntegral.Box ι
      hJ : Membership.mem π J
      ⊢ ∀ (J_1 : WithBot (BoxIntegral.Box ι)), Membership.mem (Finset.image (fun J'  …
    -/
    simp only [Finset.mem_image, exists_prop, mem_boxes, mem_filter]
    /-
      case refine_1
      ι : Type u_1
      I : BoxIntegral.Box ι
      inst✝ : Finite ι
      π : BoxIntegral.Prepartition I
      t : Finset (Prod ι Real)
      ht : ∀ (I_1 J : BoxIntegral.Box ι), Membership.mem π.boxes J → ∀ (J' : BoxInte …
      J : BoxIntegral.Box ι
      hJ : Membership.mem π J
      ⊢ ∀ (J_1 : WithBot (BoxIntegral.Box ι)), (Exists fun a => And (Membership.mem  …
    -/
    rintro _ ⟨J₁, h₁, rfl⟩ hne
    /-
      case refine_1.intro.intro
      ι : Type u_1
      I : BoxIntegral.Box ι
      inst✝ : Finite ι
      π : BoxIntegral.Prepartition I
      t : Finset (Prod ι Real)
      ht : ∀ (I_1 J : BoxIntegral.Box ι), Membership.mem π.boxes J → ∀ (J' : BoxInte …
      J : BoxIntegral.Box ι
      hJ : Membership.mem π J
      J₁ : BoxIntegral.Box ι
      h₁ : Membership.mem (BoxIntegral.Prepartition.splitMany I t) J₁
      hne : Ne (Min.min ↑J ↑J₁) Bot.bot
      ⊢ Exists fun J' => And (Exists fun a => And (And (Membership.mem (BoxIntegral. …
    -/
    refine ⟨_, ⟨J₁, ⟨h₁, Subset.trans ?_ (π.subset_iUnion hJ)⟩, rfl⟩, le_rfl⟩
    /-
      case refine_1.intro.intro
      ι : Type u_1
      I : BoxIntegral.Box ι
      inst✝ : Finite ι
      π : BoxIntegral.Prepartition I
      t : Finset (Prod ι Real)
      ht : ∀ (I_1 J : BoxIntegral.Box ι), Membership.mem π.boxes J → ∀ (J' : BoxInte …
      J : BoxIntegral.Box ι
      hJ : Membership.mem π J
      J₁ : BoxIntegral.Box ι
      h₁ : Membership.mem (BoxIntegral.Prepartition.splitMany I t) J₁
      hne : Ne (Min.min ↑J ↑J₁) Bot.bot
      ⊢ HasSubset.Subset ↑J₁ ↑J
    -/
    exact ht I J hJ J₁ h₁ (mt disjoint_iff.1 hne)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u_1
      I : BoxIntegral.Box ι
      inst✝ : Finite ι
      π : BoxIntegral.Prepartition I
      t : Finset (Prod ι Real)
      ht : ∀ (I_1 J : BoxIntegral.Box ι), Membership.mem π.boxes J → ∀ (J' : BoxInte …
      J : BoxIntegral.Box ι
      hJ : Membership.mem ((BoxIntegral.Prepartition.splitMany I t).filter fun J =>  …
      ⊢ Exists fun I' => And (Membership.mem π I') (LE.le J I')
    -/
  · rw [mem_filter] at hJ
    /-
      case refine_2
      ι : Type u_1
      I : BoxIntegral.Box ι
      inst✝ : Finite ι
      π : BoxIntegral.Prepartition I
      t : Finset (Prod ι Real)
      ht : ∀ (I_1 J : BoxIntegral.Box ι), Membership.mem π.boxes J → ∀ (J' : BoxInte …
      J : BoxIntegral.Box ι
      hJ : And (Membership.mem (BoxIntegral.Prepartition.splitMany I t) J) (HasSubse …
      ⊢ Exists fun I' => And (Membership.mem π I') (LE.le J I')
    -/
    rcases Set.mem_iUnion₂.1 (hJ.2 J.upper_mem) with ⟨J', hJ', hmem⟩
    /-
      case refine_2.intro.intro
      ι : Type u_1
      I : BoxIntegral.Box ι
      inst✝ : Finite ι
      π : BoxIntegral.Prepartition I
      t : Finset (Prod ι Real)
      ht : ∀ (I_1 J : BoxIntegral.Box ι), Membership.mem π.boxes J → ∀ (J' : BoxInte …
      J : BoxIntegral.Box ι
      hJ : And (Membership.mem (BoxIntegral.Prepartition.splitMany I t) J) (HasSubse …
      J' : BoxIntegral.Box ι
      hJ' : Membership.mem π J'
      hmem : Membership.mem (↑J') J.upper
      ⊢ Exists fun I' => And (Membership.mem π I') (LE.le J I')
    -/
    refine ⟨J', hJ', ht I _ hJ' _ hJ.1 <| Box.not_disjoint_coe_iff_nonempty_inter.2 ?_⟩
    /-
      case refine_2.intro.intro
      ι : Type u_1
      I : BoxIntegral.Box ι
      inst✝ : Finite ι
      π : BoxIntegral.Prepartition I
      t : Finset (Prod ι Real)
      ht : ∀ (I_1 J : BoxIntegral.Box ι), Membership.mem π.boxes J → ∀ (J' : BoxInte …
      J : BoxIntegral.Box ι
      hJ : And (Membership.mem (BoxIntegral.Prepartition.splitMany I t) J) (HasSubse …
      J' : BoxIntegral.Box ι
      hJ' : Membership.mem π J'
      hmem : Membership.mem (↑J') J.upper
      ⊢ (Inter.inter ↑J' ↑J).Nonempty
    -/
    exact ⟨J.upper, hmem, J.upper_mem⟩
    /-
      🎉 no goals
    -/


theorem exists_splitMany_inf_eq_filter_of_finite (s : Set (Prepartition I)) (hs : s.Finite) :
    ∃ t : Finset (ι × ℝ),
      ∀ π ∈ s, π ⊓ splitMany I t = (splitMany I t).filter fun J => ↑J ⊆ π.iUnion :=
  haveI := fun π (_ : π ∈ s) => eventually_splitMany_inf_eq_filter π
  (hs.eventually_all.2 this).exists


/-- If `π` is a partition of `I`, then there exists a finite set `s` of hyperplanes such that
`splitMany I s ≤ π`. -/
theorem IsPartition.exists_splitMany_le {I : Box ι} {π : Prepartition I} (h : IsPartition π) :
    ∃ s, splitMany I s ≤ π := by
  /-
    ι : Type u_1
    inst✝ : Finite ι
    I : BoxIntegral.Box ι
    π : BoxIntegral.Prepartition I
    h : π.IsPartition
    ⊢ Exists fun s => LE.le (BoxIntegral.Prepartition.splitMany I s) π
  -/
  refine (eventually_splitMany_inf_eq_filter π).exists.imp fun s hs => ?_
  /-
    ι : Type u_1
    inst✝ : Finite ι
    I : BoxIntegral.Box ι
    π : BoxIntegral.Prepartition I
    h : π.IsPartition
    s : Finset (Prod ι Real)
    hs : Eq (Min.min π (BoxIntegral.Prepartition.splitMany I s)) ((BoxIntegral.Pre …
    ⊢ LE.le (BoxIntegral.Prepartition.splitMany I s) π
  -/
  rwa [h.iUnion_eq, filter_of_true, inf_eq_right] at hs
  /-
    case hp
    ι : Type u_1
    inst✝ : Finite ι
    I : BoxIntegral.Box ι
    π : BoxIntegral.Prepartition I
    h : π.IsPartition
    s : Finset (Prod ι Real)
    hs : Eq (Min.min π (BoxIntegral.Prepartition.splitMany I s)) ((BoxIntegral.Pre …
    ⊢ ∀ (J : BoxIntegral.Box ι), Membership.mem (BoxIntegral.Prepartition.splitMan …
  -/
  exact fun J hJ => le_of_mem _ hJ
  /-
    🎉 no goals
  -/


/-- For every prepartition `π` of `I` there exists a prepartition that covers exactly
`I \ π.iUnion`. -/
theorem exists_iUnion_eq_diff (π : Prepartition I) :
    ∃ π' : Prepartition I, π'.iUnion = ↑I \ π.iUnion := by
  /-
    ι : Type u_1
    I : BoxIntegral.Box ι
    inst✝ : Finite ι
    π : BoxIntegral.Prepartition I
    ⊢ Exists fun π' => Eq π'.iUnion (SDiff.sdiff (↑I) π.iUnion)
  -/
  rcases π.eventually_splitMany_inf_eq_filter.exists with ⟨s, hs⟩
  /-
    case intro
    ι : Type u_1
    I : BoxIntegral.Box ι
    inst✝ : Finite ι
    π : BoxIntegral.Prepartition I
    s : Finset (Prod ι Real)
    hs : Eq (Min.min π (BoxIntegral.Prepartition.splitMany I s)) ((BoxIntegral.Pre …
    ⊢ Exists fun π' => Eq π'.iUnion (SDiff.sdiff (↑I) π.iUnion)
  -/
  use (splitMany I s).filter fun J => ¬(J : Set (ι → ℝ)) ⊆ π.iUnion
  /-
    case h
    ι : Type u_1
    I : BoxIntegral.Box ι
    inst✝ : Finite ι
    π : BoxIntegral.Prepartition I
    s : Finset (Prod ι Real)
    hs : Eq (Min.min π (BoxIntegral.Prepartition.splitMany I s)) ((BoxIntegral.Pre …
    ⊢ Eq ((BoxIntegral.Prepartition.splitMany I s).filter fun J => Not (HasSubset. …
  -/
  simp [← hs]
  /-
    🎉 no goals
  -/


/-- If `π` is a prepartition of `I`, then `π.compl` is a prepartition of `I`
such that `π.compl.iUnion = I \ π.iUnion`. -/
def compl (π : Prepartition I) : Prepartition I :=
  π.exists_iUnion_eq_diff.choose


@[simp]
theorem iUnion_compl (π : Prepartition I) : π.compl.iUnion = ↑I \ π.iUnion :=
  π.exists_iUnion_eq_diff.choose_spec


/-- Since the definition of `BoxIntegral.Prepartition.compl` uses `Exists.choose`,
the result depends only on `π.iUnion`. -/
theorem compl_congr {π₁ π₂ : Prepartition I} (h : π₁.iUnion = π₂.iUnion) : π₁.compl = π₂.compl := by
  /-
    ι : Type u_1
    I : BoxIntegral.Box ι
    inst✝ : Finite ι
    π₁ π₂ : BoxIntegral.Prepartition I
    h : Eq π₁.iUnion π₂.iUnion
    ⊢ Eq π₁.compl π₂.compl
  -/
  dsimp only [compl]
  /-
    ι : Type u_1
    I : BoxIntegral.Box ι
    inst✝ : Finite ι
    π₁ π₂ : BoxIntegral.Prepartition I
    h : Eq π₁.iUnion π₂.iUnion
    ⊢ Eq ⋯.choose ⋯.choose
  -/
  congr 1
  /-
    case e_p
    ι : Type u_1
    I : BoxIntegral.Box ι
    inst✝ : Finite ι
    π₁ π₂ : BoxIntegral.Prepartition I
    h : Eq π₁.iUnion π₂.iUnion
    ⊢ Eq (fun π' => Eq π'.iUnion (SDiff.sdiff (↑I) π₁.iUnion)) fun π' => Eq π'.iUn …
  -/
  rw [h]
  /-
    🎉 no goals
  -/


theorem IsPartition.compl_eq_bot {π : Prepartition I} (h : IsPartition π) : π.compl = ⊥ := by
  /-
    ι : Type u_1
    I : BoxIntegral.Box ι
    inst✝ : Finite ι
    π : BoxIntegral.Prepartition I
    h : π.IsPartition
    ⊢ Eq π.compl Bot.bot
  -/
  rw [← iUnion_eq_empty, iUnion_compl, h.iUnion_eq, diff_self]
  /-
    🎉 no goals
  -/


@[simp]
theorem compl_top : (⊤ : Prepartition I).compl = ⊥ :=
  (isPartitionTop I).compl_eq_bot


