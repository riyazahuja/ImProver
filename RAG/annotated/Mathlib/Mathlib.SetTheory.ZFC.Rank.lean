/-- The ordinal rank of a pre-set -/
noncomputable def rank : PSet.{u} → Ordinal.{u}
  | ⟨_, A⟩ => ⨆ a, succ (rank (A a))


theorem rank_congr : ∀ {x y : PSet}, Equiv x y → rank x = rank y
  | ⟨_, _⟩, ⟨_, _⟩, ⟨αβ, βα⟩ => by
    /-
      α✝¹ : Type u_1
      A✝¹ : α✝¹ → PSet.{u_1}
      α✝ : Type u_1
      A✝ : α✝ → PSet.{u_1}
      αβ : ∀ (a : α✝¹), Exists fun b => (A✝¹ a).Equiv (A✝ b)
      βα : ∀ (b : α✝), Exists fun a => (A✝¹ a).Equiv (A✝ b)
      ⊢ Eq (PSet.mk α✝¹ A✝¹).rank (PSet.mk α✝ A✝).rank
    -/
    apply congr_arg sSup
    /-
      α✝¹ : Type u_1
      A✝¹ : α✝¹ → PSet.{u_1}
      α✝ : Type u_1
      A✝ : α✝ → PSet.{u_1}
      αβ : ∀ (a : α✝¹), Exists fun b => (A✝¹ a).Equiv (A✝ b)
      βα : ∀ (b : α✝), Exists fun a => (A✝¹ a).Equiv (A✝ b)
      ⊢ Eq (Set.range fun a => Order.succ (A✝¹ a).rank) (Set.range fun a => Order.su …
    -/
    ext
    /-
      case h
      α✝¹ : Type u_1
      A✝¹ : α✝¹ → PSet.{u_1}
      α✝ : Type u_1
      A✝ : α✝ → PSet.{u_1}
      αβ : ∀ (a : α✝¹), Exists fun b => (A✝¹ a).Equiv (A✝ b)
      βα : ∀ (b : α✝), Exists fun a => (A✝¹ a).Equiv (A✝ b)
      x✝ : Ordinal.{u_1}
      ⊢ Iff (Membership.mem (Set.range fun a => Order.succ (A✝¹ a).rank) x✝) (Member …
    -/
    constructor <;> simp <;> intro a h
      /-
        case h.mp
        α✝¹ : Type u_1
        A✝¹ : α✝¹ → PSet.{u_1}
        α✝ : Type u_1
        A✝ : α✝ → PSet.{u_1}
        αβ : ∀ (a : α✝¹), Exists fun b => (A✝¹ a).Equiv (A✝ b)
        βα : ∀ (b : α✝), Exists fun a => (A✝¹ a).Equiv (A✝ b)
        x✝ : Ordinal.{u_1}
        a : α✝¹
        h : Eq (Order.succ (A✝¹ a).rank) x✝
        ⊢ Exists fun y => Eq (Order.succ (A✝ y).rank) x✝
      -/
    · obtain ⟨b, h'⟩ := αβ a
      /-
        case h.mp.intro
        α✝¹ : Type u_1
        A✝¹ : α✝¹ → PSet.{u_1}
        α✝ : Type u_1
        A✝ : α✝ → PSet.{u_1}
        αβ : ∀ (a : α✝¹), Exists fun b => (A✝¹ a).Equiv (A✝ b)
        βα : ∀ (b : α✝), Exists fun a => (A✝¹ a).Equiv (A✝ b)
        x✝ : Ordinal.{u_1}
        a : α✝¹
        h : Eq (Order.succ (A✝¹ a).rank) x✝
        b : α✝
        h' : (A✝¹ a).Equiv (A✝ b)
        ⊢ Exists fun y => Eq (Order.succ (A✝ y).rank) x✝
      -/
      exists b
      /-
        case h.mp.intro
        α✝¹ : Type u_1
        A✝¹ : α✝¹ → PSet.{u_1}
        α✝ : Type u_1
        A✝ : α✝ → PSet.{u_1}
        αβ : ∀ (a : α✝¹), Exists fun b => (A✝¹ a).Equiv (A✝ b)
        βα : ∀ (b : α✝), Exists fun a => (A✝¹ a).Equiv (A✝ b)
        x✝ : Ordinal.{u_1}
        a : α✝¹
        h : Eq (Order.succ (A✝¹ a).rank) x✝
        b : α✝
        h' : (A✝¹ a).Equiv (A✝ b)
        ⊢ Eq (Order.succ (A✝ b).rank) x✝
      -/
      rw [← h, rank_congr h']
      /-
        🎉 no goals
      -/
      /-
        case h.mpr
        α✝¹ : Type u_1
        A✝¹ : α✝¹ → PSet.{u_1}
        α✝ : Type u_1
        A✝ : α✝ → PSet.{u_1}
        αβ : ∀ (a : α✝¹), Exists fun b => (A✝¹ a).Equiv (A✝ b)
        βα : ∀ (b : α✝), Exists fun a => (A✝¹ a).Equiv (A✝ b)
        x✝ : Ordinal.{u_1}
        a : α✝
        h : Eq (Order.succ (A✝ a).rank) x✝
        ⊢ Exists fun y => Eq (Order.succ (A✝¹ y).rank) x✝
      -/
    · obtain ⟨b, h'⟩ := βα a
      /-
        case h.mpr.intro
        α✝¹ : Type u_1
        A✝¹ : α✝¹ → PSet.{u_1}
        α✝ : Type u_1
        A✝ : α✝ → PSet.{u_1}
        αβ : ∀ (a : α✝¹), Exists fun b => (A✝¹ a).Equiv (A✝ b)
        βα : ∀ (b : α✝), Exists fun a => (A✝¹ a).Equiv (A✝ b)
        x✝ : Ordinal.{u_1}
        a : α✝
        h : Eq (Order.succ (A✝ a).rank) x✝
        b : α✝¹
        h' : (A✝¹ b).Equiv (A✝ a)
        ⊢ Exists fun y => Eq (Order.succ (A✝¹ y).rank) x✝
      -/
      exists b
      /-
        case h.mpr.intro
        α✝¹ : Type u_1
        A✝¹ : α✝¹ → PSet.{u_1}
        α✝ : Type u_1
        A✝ : α✝ → PSet.{u_1}
        αβ : ∀ (a : α✝¹), Exists fun b => (A✝¹ a).Equiv (A✝ b)
        βα : ∀ (b : α✝), Exists fun a => (A✝¹ a).Equiv (A✝ b)
        x✝ : Ordinal.{u_1}
        a : α✝
        h : Eq (Order.succ (A✝ a).rank) x✝
        b : α✝¹
        h' : (A✝¹ b).Equiv (A✝ a)
        ⊢ Eq (Order.succ (A✝¹ b).rank) x✝
      -/
      rw [← h, rank_congr h']
      /-
        🎉 no goals
      -/


theorem rank_lt_of_mem : ∀ {x y : PSet}, y ∈ x → rank y < rank x
  | ⟨_, _⟩, _, ⟨_, h⟩ => by
    /-
      α✝ : Type u_1
      A✝ : α✝ → PSet.{u_1}
      x✝ : PSet.{u_1}
      w✝ : (PSet.mk α✝ A✝).Type
      h : x✝.Equiv ((PSet.mk α✝ A✝).Func w✝)
      ⊢ LT.lt x✝.rank (PSet.mk α✝ A✝).rank
    -/
    rw [rank_congr h, ← succ_le_iff]
    /-
      α✝ : Type u_1
      A✝ : α✝ → PSet.{u_1}
      x✝ : PSet.{u_1}
      w✝ : (PSet.mk α✝ A✝).Type
      h : x✝.Equiv ((PSet.mk α✝ A✝).Func w✝)
      ⊢ LE.le (Order.succ ((PSet.mk α✝ A✝).Func w✝).rank) (PSet.mk α✝ A✝).rank
    -/
    apply Ordinal.le_iSup
    /-
      🎉 no goals
    -/


theorem rank_le_iff {o : Ordinal} : ∀ {x : PSet}, rank x ≤ o ↔ ∀ ⦃y⦄, y ∈ x → rank y < o
  | ⟨_, A⟩ => by
    /-
      o : Ordinal.{u_1}
      α✝ : Type u_1
      A : α✝ → PSet.{u_1}
      ⊢ Iff (LE.le (PSet.mk α✝ A).rank o) (∀ ⦃y : PSet.{u_1}⦄, Membership.mem (PSet. …
    -/
    refine ⟨fun h _ h' => (rank_lt_of_mem h').trans_le h, fun h ↦ Ordinal.iSup_le fun a ↦ ?_⟩
    /-
      o : Ordinal.{u_1}
      α✝ : Type u_1
      A : α✝ → PSet.{u_1}
      h : ∀ ⦃y : PSet.{u_1}⦄, Membership.mem (PSet.mk α✝ A) y → LT.lt y.rank o
      a : α✝
      ⊢ LE.le (Order.succ (A a).rank) o
    -/
    rw [succ_le_iff]
    /-
      o : Ordinal.{u_1}
      α✝ : Type u_1
      A : α✝ → PSet.{u_1}
      h : ∀ ⦃y : PSet.{u_1}⦄, Membership.mem (PSet.mk α✝ A) y → LT.lt y.rank o
      a : α✝
      ⊢ LT.lt (A a).rank o
    -/
    exact h (Mem.mk A a)
    /-
      🎉 no goals
    -/


theorem lt_rank_iff {o : Ordinal} {x : PSet} : o < rank x ↔ ∃ y ∈ x, o ≤ rank y := by
  /-
    o : Ordinal.{u_1}
    x : PSet.{u_1}
    ⊢ Iff (LT.lt o x.rank) (Exists fun y => And (Membership.mem x y) (LE.le o y.ra …
  -/
  rw [← not_iff_not, not_lt, rank_le_iff]
  /-
    o : Ordinal.{u_1}
    x : PSet.{u_1}
    ⊢ Iff (∀ ⦃y : PSet.{u_1}⦄, Membership.mem x y → LT.lt y.rank o) (Not (Exists f …
  -/
  simp
  /-
    🎉 no goals
  -/


@[gcongr] theorem rank_mono (h : x ⊆ y) : rank x ≤ rank y :=
  rank_le_iff.2 fun _ h₁ => rank_lt_of_mem (mem_of_subset h h₁)


@[simp]
                                      /-
                                        ⊢ Eq EmptyCollection.emptyCollection.rank 0
                                      -/
theorem rank_empty : rank ∅ = 0 := by simp [empty_def, rank]
                                      /-
                                        🎉 no goals
                                      -/


@[simp]
theorem rank_insert (x y : PSet) : rank (insert x y) = max (succ (rank x)) (rank y) := by
  /-
    x y : PSet.{u_1}
    ⊢ Eq (Insert.insert x y).rank (Max.max (Order.succ x.rank) y.rank)
  -/
  apply le_antisymm
    /-
      case a
      x y : PSet.{u_1}
      ⊢ LE.le (Insert.insert x y).rank (Max.max (Order.succ x.rank) y.rank)
    -/
  · simp_rw [rank_le_iff, mem_insert_iff]
    /-
      case a
      x y : PSet.{u_1}
      ⊢ ∀ ⦃y_1 : PSet.{u_1}⦄, Or (y_1.Equiv x) (Membership.mem y y_1) → LT.lt y_1.ra …
    -/
    rintro _ (h | h)
      /-
        case a.inl
        x y y✝ : PSet.{u_1}
        h : y✝.Equiv x
        ⊢ LT.lt y✝.rank (Max.max (Order.succ x.rank) y.rank)
      -/
    · simp [rank_congr h]
      /-
        🎉 no goals
      -/
      /-
        case a.inr
        x y y✝ : PSet.{u_1}
        h : Membership.mem y y✝
        ⊢ LT.lt y✝.rank (Max.max (Order.succ x.rank) y.rank)
      -/
    · simp [rank_lt_of_mem h]
      /-
        🎉 no goals
      -/
    /-
      case a
      x y : PSet.{u_1}
      ⊢ LE.le (Max.max (Order.succ x.rank) y.rank) (Insert.insert x y).rank
    -/
  · apply max_le
      /-
        case a.h₁
        x y : PSet.{u_1}
        ⊢ LE.le (Order.succ x.rank) (Insert.insert x y).rank
      -/
    · exact (rank_lt_of_mem (mem_insert x y)).succ_le
      /-
        🎉 no goals
      -/
      /-
        case a.h₂
        x y : PSet.{u_1}
        ⊢ LE.le y.rank (Insert.insert x y).rank
      -/
    · exact rank_mono (subset_iff.2 fun z => mem_insert_of_mem x)
      /-
        🎉 no goals
      -/


@[simp]
theorem rank_singleton (x : PSet) : rank {x} = succ (rank x) :=
                              /-
                                x : PSet.{u_1}
                                ⊢ Eq (Max.max (Order.succ x.rank) EmptyCollection.emptyCollection.rank) (Order …
                              -/
  (rank_insert _ _).trans (by simp)
                              /-
                                🎉 no goals
                              -/


theorem rank_pair (x y : PSet) : rank {x, y} = max (succ (rank x)) (succ (rank y)) := by
  /-
    x y : PSet.{u_1}
    ⊢ Eq (Insert.insert x (Singleton.singleton y)).rank (Max.max (Order.succ x.ran …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem rank_powerset (x : PSet) : rank (powerset x) = succ (rank x) := by
  /-
    x : PSet.{u_1}
    ⊢ Eq x.powerset.rank (Order.succ x.rank)
  -/
  apply le_antisymm
    /-
      case a
      x : PSet.{u_1}
      ⊢ LE.le x.powerset.rank (Order.succ x.rank)
    -/
  · simp_rw [rank_le_iff, mem_powerset, lt_succ_iff]
    /-
      case a
      x : PSet.{u_1}
      ⊢ ∀ ⦃y : PSet.{u_1}⦄, HasSubset.Subset y x → LE.le y.rank x.rank
    -/
    intro
    /-
      case a
      x y✝ : PSet.{u_1}
      ⊢ HasSubset.Subset y✝ x → LE.le y✝.rank x.rank
    -/
    exact rank_mono
    /-
      🎉 no goals
    -/
    /-
      case a
      x : PSet.{u_1}
      ⊢ LE.le (Order.succ x.rank) x.powerset.rank
    -/
  · rw [succ_le_iff]
    /-
      case a
      x : PSet.{u_1}
      ⊢ LT.lt x.rank x.powerset.rank
    -/
    apply rank_lt_of_mem
    /-
      case a.a
      x : PSet.{u_1}
      ⊢ Membership.mem x.powerset x
    -/
    simp
    /-
      🎉 no goals
    -/


/-- For the rank of `⋃₀ x`, we only have `rank (⋃₀ x) ≤ rank x ≤ rank (⋃₀ x) + 1`.

This inequality is split into `rank_sUnion_le` and `le_succ_rank_sUnion`. -/
theorem rank_sUnion_le (x : PSet) : rank (⋃₀ x) ≤ rank x := by
  /-
    x : PSet.{u_1}
    ⊢ LE.le x.sUnion.rank x.rank
  -/
  simp_rw [rank_le_iff, mem_sUnion]
  /-
    x : PSet.{u_1}
    ⊢ ∀ ⦃y : PSet.{u_1}⦄, (Exists fun z => And (Membership.mem x z) (Membership.me …
  -/
  intro _ ⟨_, _, _⟩
  /-
    x y✝ w✝ : PSet.{u_1}
    left✝ : Membership.mem x w✝
    right✝ : Membership.mem w✝ y✝
    ⊢ LT.lt y✝.rank x.rank
  -/
                                     /-
                                       🎉 no goals
                                     -/
  trans <;> apply rank_lt_of_mem <;> assumption
                                     /-
                                       🎉 no goals
                                     -/


theorem le_succ_rank_sUnion (x : PSet) : rank x ≤ succ (rank (⋃₀ x)) := by
  /-
    x : PSet.{u_1}
    ⊢ LE.le x.rank (Order.succ x.sUnion.rank)
  -/
  rw [← rank_powerset]
  /-
    x : PSet.{u_1}
    ⊢ LE.le x.rank x.sUnion.powerset.rank
  -/
  apply rank_mono
  /-
    case h
    x : PSet.{u_1}
    ⊢ HasSubset.Subset x x.sUnion.powerset
  -/
  rw [subset_iff]
  /-
    case h
    x : PSet.{u_1}
    ⊢ ∀ ⦃z : PSet.{u_1}⦄, Membership.mem x z → Membership.mem x.sUnion.powerset z
  -/
  intro z _
  /-
    case h
    x z : PSet.{u_1}
    a✝ : Membership.mem x z
    ⊢ Membership.mem x.sUnion.powerset z
  -/
  rw [mem_powerset, subset_iff]
  /-
    case h
    x z : PSet.{u_1}
    a✝ : Membership.mem x z
    ⊢ ∀ ⦃z_1 : PSet.{u_1}⦄, Membership.mem z z_1 → Membership.mem x.sUnion z_1
  -/
  intro _ _
  /-
    case h
    x z : PSet.{u_1}
    a✝¹ : Membership.mem x z
    z✝ : PSet.{u_1}
    a✝ : Membership.mem z z✝
    ⊢ Membership.mem x.sUnion z✝
  -/
  rw [mem_sUnion]
  /-
    case h
    x z : PSet.{u_1}
    a✝¹ : Membership.mem x z
    z✝ : PSet.{u_1}
    a✝ : Membership.mem z z✝
    ⊢ Exists fun z => And (Membership.mem x z) (Membership.mem z z✝)
  -/
  exists z
  /-
    🎉 no goals
  -/


/-- `PSet.rank` is equal to the `IsWellFounded.rank` over `∈`. -/
theorem rank_eq_wfRank : lift.{u + 1, u} (rank x) = IsWellFounded.rank (α := PSet) (· ∈ ·) x := by
  /-
    x : PSet.{u}
    ⊢ Eq (Ordinal.lift.{u + 1, u} x.rank) (IsWellFounded.rank (fun x1 x2 => Member …
  -/
  induction' x using mem_wf.induction with x ih
  /-
    case h
    x✝ x : PSet.{u}
    ih : ∀ (y : PSet.{u}), Membership.mem x y → Eq (Ordinal.lift.{u + 1, u} y.rank …
    ⊢ Eq (Ordinal.lift.{u + 1, u} x.rank) (IsWellFounded.rank (fun x1 x2 => Member …
  -/
  rw [IsWellFounded.rank_eq]
  /-
    case h
    x✝ x : PSet.{u}
    ih : ∀ (y : PSet.{u}), Membership.mem x y → Eq (Ordinal.lift.{u + 1, u} y.rank …
    ⊢ Eq (Ordinal.lift.{u + 1, u} x.rank) (iSup fun b => Order.succ (IsWellFounded …
  -/
  simp_rw [← fun y : { y // y ∈ x } => ih y y.2]
  /-
    case h
    x✝ x : PSet.{u}
    ih : ∀ (y : PSet.{u}), Membership.mem x y → Eq (Ordinal.lift.{u + 1, u} y.rank …
    ⊢ Eq (Ordinal.lift.{u + 1, u} x.rank) (iSup fun b => Order.succ (Ordinal.lift. …
  -/
  apply (le_of_forall_lt _).antisymm (Ordinal.iSup_le _) <;> intro h
    /-
      x✝ x : PSet.{u}
      ih : ∀ (y : PSet.{u}), Membership.mem x y → Eq (Ordinal.lift.{u + 1, u} y.rank …
      h : Ordinal.{u + 1}
      ⊢ LT.lt h (Ordinal.lift.{u + 1, u} x.rank) → LT.lt h (iSup fun b => Order.succ …
    -/
  · rw [lt_lift_iff]
    /-
      x✝ x : PSet.{u}
      ih : ∀ (y : PSet.{u}), Membership.mem x y → Eq (Ordinal.lift.{u + 1, u} y.rank …
      h : Ordinal.{u + 1}
      ⊢ (Exists fun a' => And (LT.lt a' x.rank) (Eq (Ordinal.lift.{u + 1, u} a') h)) …
    -/
    rintro ⟨o, h, rfl⟩
    /-
      case intro.intro
      x✝ x : PSet.{u}
      ih : ∀ (y : PSet.{u}), Membership.mem x y → Eq (Ordinal.lift.{u + 1, u} y.rank …
      o : Ordinal.{u}
      h : LT.lt o x.rank
      ⊢ LT.lt (Ordinal.lift.{u + 1, u} o) (iSup fun b => Order.succ (Ordinal.lift.{u …
    -/
    simpa [Ordinal.lt_iSup_iff] using lt_rank_iff.1 h
    /-
      🎉 no goals
    -/
    /-
      x✝ x : PSet.{u}
      ih : ∀ (y : PSet.{u}), Membership.mem x y → Eq (Ordinal.lift.{u + 1, u} y.rank …
      h : Subtype fun b => Membership.mem x b
      ⊢ LE.le (Order.succ (Ordinal.lift.{u + 1, u} (↑h).rank)) (Ordinal.lift.{u + 1, …
    -/
  · simpa using rank_lt_of_mem h.2
    /-
      🎉 no goals
    -/


/-- The ordinal rank of a ZFC set -/
noncomputable def rank : ZFSet.{u} → Ordinal.{u} :=
  Quotient.lift _ fun _ _ => PSet.rank_congr


theorem rank_lt_of_mem : y ∈ x → rank y < rank x :=
  Quotient.inductionOn₂ x y fun _ _ => PSet.rank_lt_of_mem


theorem rank_le_iff {o : Ordinal} : rank x ≤ o ↔ ∀ ⦃y⦄, y ∈ x → rank y < o :=
  ⟨fun h _ h' => (rank_lt_of_mem h').trans_le h,
    Quotient.inductionOn x fun _ h =>
      PSet.rank_le_iff.2 fun y h' => @h ⟦y⟧ h'⟩


theorem lt_rank_iff {o : Ordinal} : o < rank x ↔ ∃ y ∈ x, o ≤ rank y := by
  /-
    x : ZFSet.{u}
    o : Ordinal.{u}
    ⊢ Iff (LT.lt o x.rank) (Exists fun y => And (Membership.mem x y) (LE.le o y.ra …
  -/
  rw [← not_iff_not, not_lt, rank_le_iff]
  /-
    x : ZFSet.{u}
    o : Ordinal.{u}
    ⊢ Iff (∀ ⦃y : ZFSet.{u}⦄, Membership.mem x y → LT.lt y.rank o) (Not (Exists fu …
  -/
  simp
  /-
    🎉 no goals
  -/


@[gcongr] theorem rank_mono (h : x ⊆ y) : rank x ≤ rank y :=
  rank_le_iff.2 fun _ h₁ => rank_lt_of_mem (h h₁)


@[simp]
theorem rank_empty : rank ∅ = 0 := PSet.rank_empty


@[simp]
theorem rank_insert (x y : ZFSet) : rank (insert x y) = max (succ (rank x)) (rank y) :=
  Quotient.inductionOn₂ x y PSet.rank_insert


@[simp]
theorem rank_singleton (x : ZFSet) : rank {x} = succ (rank x) :=
                              /-
                                x : ZFSet.{u_1}
                                ⊢ Eq (Max.max (Order.succ x.rank) EmptyCollection.emptyCollection.rank) (Order …
                              -/
  (rank_insert _ _).trans (by simp)
                              /-
                                🎉 no goals
                              -/


theorem rank_pair (x y : ZFSet) : rank {x, y} = max (succ (rank x)) (succ (rank y)) := by
  /-
    x y : ZFSet.{u_1}
    ⊢ Eq (Insert.insert x (Singleton.singleton y)).rank (Max.max (Order.succ x.ran …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem rank_union (x y : ZFSet) : rank (x ∪ y) = max (rank x) (rank y) := by
  /-
    x y : ZFSet.{u_1}
    ⊢ Eq (Union.union x y).rank (Max.max x.rank y.rank)
  -/
  apply le_antisymm
    /-
      case a
      x y : ZFSet.{u_1}
      ⊢ LE.le (Union.union x y).rank (Max.max x.rank y.rank)
    -/
  · simp_rw [rank_le_iff, mem_union, lt_max_iff]
    /-
      case a
      x y : ZFSet.{u_1}
      ⊢ ∀ ⦃y_1 : ZFSet.{u_1}⦄, Or (Membership.mem x y_1) (Membership.mem y y_1) → Or …
    -/
    intro
    /-
      case a
      x y y✝ : ZFSet.{u_1}
      ⊢ Or (Membership.mem x y✝) (Membership.mem y y✝) → Or (LT.lt y✝.rank x.rank) ( …
    -/
                     /-
                       🎉 no goals
                     -/
    apply Or.imp <;> apply rank_lt_of_mem
                     /-
                       🎉 no goals
                     -/
    /-
      case a
      x y : ZFSet.{u_1}
      ⊢ LE.le (Max.max x.rank y.rank) (Union.union x y).rank
    -/
                                                       /-
                                                         🎉 no goals
                                                       -/
  · apply max_le <;> apply rank_mono <;> intro _ h <;> simp [h]
                                                       /-
                                                         🎉 no goals
                                                       -/


@[simp]
theorem rank_powerset (x : ZFSet) : rank (powerset x) = succ (rank x) :=
  Quotient.inductionOn x PSet.rank_powerset


/-- For the rank of `⋃₀ x`, we only have `rank (⋃₀ x) ≤ rank x ≤ rank (⋃₀ x) + 1`.

This inequality is split into `rank_sUnion_le` and `le_succ_rank_sUnion`. -/
theorem rank_sUnion_le (x : ZFSet) : rank (⋃₀ x) ≤ rank x := by
  /-
    x : ZFSet.{u_1}
    ⊢ LE.le x.sUnion.rank x.rank
  -/
  simp_rw [rank_le_iff, mem_sUnion]
  /-
    x : ZFSet.{u_1}
    ⊢ ∀ ⦃y : ZFSet.{u_1}⦄, (Exists fun z => And (Membership.mem x z) (Membership.m …
  -/
  intro _ ⟨_, _, _⟩
  /-
    x y✝ w✝ : ZFSet.{u_1}
    left✝ : Membership.mem x w✝
    right✝ : Membership.mem w✝ y✝
    ⊢ LT.lt y✝.rank x.rank
  -/
                                     /-
                                       🎉 no goals
                                     -/
  trans <;> apply rank_lt_of_mem <;> assumption
                                     /-
                                       🎉 no goals
                                     -/


theorem le_succ_rank_sUnion (x : ZFSet) : rank x ≤ succ (rank (⋃₀ x)) := by
  /-
    x : ZFSet.{u_1}
    ⊢ LE.le x.rank (Order.succ x.sUnion.rank)
  -/
  rw [← rank_powerset]
  /-
    x : ZFSet.{u_1}
    ⊢ LE.le x.rank x.sUnion.powerset.rank
  -/
  apply rank_mono
  /-
    case h
    x : ZFSet.{u_1}
    ⊢ HasSubset.Subset x x.sUnion.powerset
  -/
  intro z _
  /-
    case h
    x z : ZFSet.{u_1}
    a✝ : Membership.mem x z
    ⊢ Membership.mem x.sUnion.powerset z
  -/
  rw [mem_powerset]
  /-
    case h
    x z : ZFSet.{u_1}
    a✝ : Membership.mem x z
    ⊢ HasSubset.Subset z x.sUnion
  -/
  intro _ _
  /-
    case h
    x z : ZFSet.{u_1}
    a✝¹ : Membership.mem x z
    z✝ : ZFSet.{u_1}
    a✝ : Membership.mem z z✝
    ⊢ Membership.mem x.sUnion z✝
  -/
  rw [mem_sUnion]
  /-
    case h
    x z : ZFSet.{u_1}
    a✝¹ : Membership.mem x z
    z✝ : ZFSet.{u_1}
    a✝ : Membership.mem z z✝
    ⊢ Exists fun z => And (Membership.mem x z) (Membership.mem z z✝)
  -/
  exists z
  /-
    🎉 no goals
  -/


@[simp]
theorem rank_range {α : Type*} [Small.{u} α] (f : α → ZFSet.{u}) :
    rank (range f) = ⨆ i, succ (rank (f i)) := by
  /-
    α : Type u_1
    inst✝ : Small.{u, u_1} α
    f : α → ZFSet.{u}
    ⊢ Eq (ZFSet.range f).rank (iSup fun i => Order.succ (f i).rank)
  -/
  apply (Ordinal.iSup_le _).antisymm'
    /-
      α : Type u_1
      inst✝ : Small.{u, u_1} α
      f : α → ZFSet.{u}
      ⊢ LE.le (ZFSet.range f).rank (iSup fun i => Order.succ (f i).rank)
    -/
  · simpa [rank_le_iff, ← succ_le_iff] using Ordinal.le_iSup _
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      inst✝ : Small.{u, u_1} α
      f : α → ZFSet.{u}
      ⊢ ∀ (i : α), LE.le (Order.succ (f i).rank) (ZFSet.range f).rank
    -/
  · simp [rank_lt_of_mem]
    /-
      🎉 no goals
    -/


/-- `ZFSet.rank` is equal to the `IsWellFounded.rank` over `∈`. -/
theorem rank_eq_wfRank : lift.{u + 1, u} (rank x) = IsWellFounded.rank (α := ZFSet) (· ∈ ·) x := by
  /-
    x : ZFSet.{u}
    ⊢ Eq (Ordinal.lift.{u + 1, u} x.rank) (IsWellFounded.rank (fun x1 x2 => Member …
  -/
  induction' x using inductionOn with x ih
  /-
    case h
    x✝ x : ZFSet.{u}
    ih : ∀ (y : ZFSet.{u}), Membership.mem x y → Eq (Ordinal.lift.{u + 1, u} y.ran …
    ⊢ Eq (Ordinal.lift.{u + 1, u} x.rank) (IsWellFounded.rank (fun x1 x2 => Member …
  -/
  rw [IsWellFounded.rank_eq]
  /-
    case h
    x✝ x : ZFSet.{u}
    ih : ∀ (y : ZFSet.{u}), Membership.mem x y → Eq (Ordinal.lift.{u + 1, u} y.ran …
    ⊢ Eq (Ordinal.lift.{u + 1, u} x.rank) (iSup fun b => Order.succ (IsWellFounded …
  -/
  simp_rw [← fun y : { y // y ∈ x } => ih y y.2]
  /-
    case h
    x✝ x : ZFSet.{u}
    ih : ∀ (y : ZFSet.{u}), Membership.mem x y → Eq (Ordinal.lift.{u + 1, u} y.ran …
    ⊢ Eq (Ordinal.lift.{u + 1, u} x.rank) (iSup fun b => Order.succ (Ordinal.lift. …
  -/
  apply (le_of_forall_lt _).antisymm (Ordinal.iSup_le _) <;> intro h
    /-
      x✝ x : ZFSet.{u}
      ih : ∀ (y : ZFSet.{u}), Membership.mem x y → Eq (Ordinal.lift.{u + 1, u} y.ran …
      h : Ordinal.{u + 1}
      ⊢ LT.lt h (Ordinal.lift.{u + 1, u} x.rank) → LT.lt h (iSup fun b => Order.succ …
    -/
  · rw [lt_lift_iff]
    /-
      x✝ x : ZFSet.{u}
      ih : ∀ (y : ZFSet.{u}), Membership.mem x y → Eq (Ordinal.lift.{u + 1, u} y.ran …
      h : Ordinal.{u + 1}
      ⊢ (Exists fun a' => And (LT.lt a' x.rank) (Eq (Ordinal.lift.{u + 1, u} a') h)) …
    -/
    rintro ⟨o, h, rfl⟩
    /-
      case intro.intro
      x✝ x : ZFSet.{u}
      ih : ∀ (y : ZFSet.{u}), Membership.mem x y → Eq (Ordinal.lift.{u + 1, u} y.ran …
      o : Ordinal.{u}
      h : LT.lt o x.rank
      ⊢ LT.lt (Ordinal.lift.{u + 1, u} o) (iSup fun b => Order.succ (Ordinal.lift.{u …
    -/
    simpa [Ordinal.lt_iSup_iff] using lt_rank_iff.1 h
    /-
      🎉 no goals
    -/
    /-
      x✝ x : ZFSet.{u}
      ih : ∀ (y : ZFSet.{u}), Membership.mem x y → Eq (Ordinal.lift.{u + 1, u} y.ran …
      h : Subtype fun b => Membership.mem x b
      ⊢ LE.le (Order.succ (Ordinal.lift.{u + 1, u} (↑h).rank)) (Ordinal.lift.{u + 1, …
    -/
  · simpa using rank_lt_of_mem h.2
    /-
      🎉 no goals
    -/


