/-- `lowerCrossingTimeAux a f c N` is the first time `f` reached below `a` after time `c` before
time `N`. -/
noncomputable def lowerCrossingTimeAux [Preorder ι] [InfSet ι] (a : ℝ) (f : ι → Ω → ℝ) (c N : ι) :
    Ω → ι :=
  hitting f (Set.Iic a) c N


/-- `upperCrossingTime a b f N n` is the first time before time `N`, `f` reaches
above `b` after `f` reached below `a` for the `n - 1`-th time. -/
noncomputable def upperCrossingTime [Preorder ι] [OrderBot ι] [InfSet ι] (a b : ℝ) (f : ι → Ω → ℝ)
    (N : ι) : ℕ → Ω → ι
  | 0 => ⊥
  | n + 1 => fun ω =>
    hitting f (Set.Ici b) (lowerCrossingTimeAux a f (upperCrossingTime a b f N n ω) N ω) N ω


/-- `lowerCrossingTime a b f N n` is the first time before time `N`, `f` reaches
below `a` after `f` reached above `b` for the `n`-th time. -/
noncomputable def lowerCrossingTime [Preorder ι] [OrderBot ι] [InfSet ι] (a b : ℝ) (f : ι → Ω → ℝ)
    (N : ι) (n : ℕ) : Ω → ι := fun ω => hitting f (Set.Iic a) (upperCrossingTime a b f N n ω) N ω


@[simp]
theorem upperCrossingTime_zero : upperCrossingTime a b f N 0 = ⊥ :=
  rfl


@[simp]
theorem lowerCrossingTime_zero : lowerCrossingTime a b f N 0 = hitting f (Set.Iic a) ⊥ N :=
  rfl


theorem upperCrossingTime_succ : upperCrossingTime a b f N (n + 1) ω =
    hitting f (Set.Ici b) (lowerCrossingTimeAux a f (upperCrossingTime a b f N n ω) N ω) N ω := by
  /-
    Ω : Type u_1
    ι : Type u_2
    inst✝² : Preorder ι
    inst✝¹ : OrderBot ι
    inst✝ : InfSet ι
    a b : Real
    f : ι → Ω → Real
    N : ι
    n : Nat
    ω : Ω
    ⊢ Eq (MeasureTheory.upperCrossingTime a b f N (HAdd.hAdd n 1) ω) (MeasureTheor …
  -/
  rw [upperCrossingTime]
  /-
    🎉 no goals
  -/


theorem upperCrossingTime_succ_eq (ω : Ω) : upperCrossingTime a b f N (n + 1) ω =
    hitting f (Set.Ici b) (lowerCrossingTime a b f N n ω) N ω := by
  /-
    Ω : Type u_1
    ι : Type u_2
    inst✝² : Preorder ι
    inst✝¹ : OrderBot ι
    inst✝ : InfSet ι
    a b : Real
    f : ι → Ω → Real
    N : ι
    n : Nat
    ω : Ω
    ⊢ Eq (MeasureTheory.upperCrossingTime a b f N (HAdd.hAdd n 1) ω) (MeasureTheor …
  -/
  simp only [upperCrossingTime_succ]
  /-
    Ω : Type u_1
    ι : Type u_2
    inst✝² : Preorder ι
    inst✝¹ : OrderBot ι
    inst✝ : InfSet ι
    a b : Real
    f : ι → Ω → Real
    N : ι
    n : Nat
    ω : Ω
    ⊢ Eq (MeasureTheory.hitting f (Set.Ici b) (MeasureTheory.lowerCrossingTimeAux  …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem upperCrossingTime_le : upperCrossingTime a b f N n ω ≤ N := by
  /-
    Ω : Type u_1
    ι : Type u_2
    inst✝ : ConditionallyCompleteLinearOrderBot ι
    a b : Real
    f : ι → Ω → Real
    N : ι
    n : Nat
    ω : Ω
    ⊢ LE.le (MeasureTheory.upperCrossingTime a b f N n ω) N
  -/
  cases n
    /-
      case zero
      Ω : Type u_1
      ι : Type u_2
      inst✝ : ConditionallyCompleteLinearOrderBot ι
      a b : Real
      f : ι → Ω → Real
      N : ι
      ω : Ω
      ⊢ LE.le (MeasureTheory.upperCrossingTime a b f N 0 ω) N
    -/
  · simp only [upperCrossingTime_zero, Pi.bot_apply, bot_le]
    /-
      🎉 no goals
    -/
    /-
      case succ
      Ω : Type u_1
      ι : Type u_2
      inst✝ : ConditionallyCompleteLinearOrderBot ι
      a b : Real
      f : ι → Ω → Real
      N : ι
      ω : Ω
      n✝ : Nat
      ⊢ LE.le (MeasureTheory.upperCrossingTime a b f N (HAdd.hAdd n✝ 1) ω) N
    -/
  · simp only [upperCrossingTime_succ, hitting_le]
    /-
      🎉 no goals
    -/


@[simp]
theorem upperCrossingTime_zero' : upperCrossingTime a b f ⊥ n ω = ⊥ :=
  eq_bot_iff.2 upperCrossingTime_le


theorem lowerCrossingTime_le : lowerCrossingTime a b f N n ω ≤ N := by
  /-
    Ω : Type u_1
    ι : Type u_2
    inst✝ : ConditionallyCompleteLinearOrderBot ι
    a b : Real
    f : ι → Ω → Real
    N : ι
    n : Nat
    ω : Ω
    ⊢ LE.le (MeasureTheory.lowerCrossingTime a b f N n ω) N
  -/
  simp only [lowerCrossingTime, hitting_le ω]
  /-
    🎉 no goals
  -/


theorem upperCrossingTime_le_lowerCrossingTime :
    upperCrossingTime a b f N n ω ≤ lowerCrossingTime a b f N n ω := by
  /-
    Ω : Type u_1
    ι : Type u_2
    inst✝ : ConditionallyCompleteLinearOrderBot ι
    a b : Real
    f : ι → Ω → Real
    N : ι
    n : Nat
    ω : Ω
    ⊢ LE.le (MeasureTheory.upperCrossingTime a b f N n ω) (MeasureTheory.lowerCros …
  -/
  simp only [lowerCrossingTime, le_hitting upperCrossingTime_le ω]
  /-
    🎉 no goals
  -/


theorem lowerCrossingTime_le_upperCrossingTime_succ :
    lowerCrossingTime a b f N n ω ≤ upperCrossingTime a b f N (n + 1) ω := by
  /-
    Ω : Type u_1
    ι : Type u_2
    inst✝ : ConditionallyCompleteLinearOrderBot ι
    a b : Real
    f : ι → Ω → Real
    N : ι
    n : Nat
    ω : Ω
    ⊢ LE.le (MeasureTheory.lowerCrossingTime a b f N n ω) (MeasureTheory.upperCros …
  -/
  rw [upperCrossingTime_succ]
  /-
    Ω : Type u_1
    ι : Type u_2
    inst✝ : ConditionallyCompleteLinearOrderBot ι
    a b : Real
    f : ι → Ω → Real
    N : ι
    n : Nat
    ω : Ω
    ⊢ LE.le (MeasureTheory.lowerCrossingTime a b f N n ω) (MeasureTheory.hitting f …
  -/
  exact le_hitting lowerCrossingTime_le ω
  /-
    🎉 no goals
  -/


theorem lowerCrossingTime_mono (hnm : n ≤ m) :
    lowerCrossingTime a b f N n ω ≤ lowerCrossingTime a b f N m ω := by
  /-
    Ω : Type u_1
    ι : Type u_2
    inst✝ : ConditionallyCompleteLinearOrderBot ι
    a b : Real
    f : ι → Ω → Real
    N : ι
    n m : Nat
    ω : Ω
    hnm : LE.le n m
    ⊢ LE.le (MeasureTheory.lowerCrossingTime a b f N n ω) (MeasureTheory.lowerCros …
  -/
  suffices Monotone fun n => lowerCrossingTime a b f N n ω by exact this hnm
  exact monotone_nat_of_le_succ fun n =>
    le_trans lowerCrossingTime_le_upperCrossingTime_succ upperCrossingTime_le_lowerCrossingTime


theorem upperCrossingTime_mono (hnm : n ≤ m) :
    upperCrossingTime a b f N n ω ≤ upperCrossingTime a b f N m ω := by
  /-
    Ω : Type u_1
    ι : Type u_2
    inst✝ : ConditionallyCompleteLinearOrderBot ι
    a b : Real
    f : ι → Ω → Real
    N : ι
    n m : Nat
    ω : Ω
    hnm : LE.le n m
    ⊢ LE.le (MeasureTheory.upperCrossingTime a b f N n ω) (MeasureTheory.upperCros …
  -/
  suffices Monotone fun n => upperCrossingTime a b f N n ω by exact this hnm
  exact monotone_nat_of_le_succ fun n =>
    le_trans upperCrossingTime_le_lowerCrossingTime lowerCrossingTime_le_upperCrossingTime_succ


theorem stoppedValue_lowerCrossingTime (h : lowerCrossingTime a b f N n ω ≠ N) :
    stoppedValue f (lowerCrossingTime a b f N n) ω ≤ a := by
  /-
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N n : Nat
    ω : Ω
    h : Ne (MeasureTheory.lowerCrossingTime a b f N n ω) N
    ⊢ LE.le (MeasureTheory.stoppedValue f (MeasureTheory.lowerCrossingTime a b f N …
  -/
  obtain ⟨j, hj₁, hj₂⟩ := (hitting_le_iff_of_lt _ (lt_of_le_of_ne lowerCrossingTime_le h)).1 le_rfl
  /-
    case intro.intro
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N n : Nat
    ω : Ω
    h : Ne (MeasureTheory.lowerCrossingTime a b f N n ω) N
    j : Nat
    hj₁ : Membership.mem (Set.Icc (MeasureTheory.upperCrossingTime a b f N n ω) (M …
    hj₂ : Membership.mem (Set.Iic a) (f j ω)
    ⊢ LE.le (MeasureTheory.stoppedValue f (MeasureTheory.lowerCrossingTime a b f N …
  -/
  exact stoppedValue_hitting_mem ⟨j, ⟨hj₁.1, le_trans hj₁.2 lowerCrossingTime_le⟩, hj₂⟩
  /-
    🎉 no goals
  -/


theorem stoppedValue_upperCrossingTime (h : upperCrossingTime a b f N (n + 1) ω ≠ N) :
    b ≤ stoppedValue f (upperCrossingTime a b f N (n + 1)) ω := by
  /-
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N n : Nat
    ω : Ω
    h : Ne (MeasureTheory.upperCrossingTime a b f N (HAdd.hAdd n 1) ω) N
    ⊢ LE.le b (MeasureTheory.stoppedValue f (MeasureTheory.upperCrossingTime a b f …
  -/
  obtain ⟨j, hj₁, hj₂⟩ := (hitting_le_iff_of_lt _ (lt_of_le_of_ne upperCrossingTime_le h)).1 le_rfl
  /-
    case intro.intro
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N n : Nat
    ω : Ω
    h : Ne (MeasureTheory.upperCrossingTime a b f N (HAdd.hAdd n 1) ω) N
    j : Nat
    hj₁ : Membership.mem (Set.Icc (MeasureTheory.lowerCrossingTimeAux a f (Measure …
    hj₂ : Membership.mem (Set.Ici b) (f j ω)
    ⊢ LE.le b (MeasureTheory.stoppedValue f (MeasureTheory.upperCrossingTime a b f …
  -/
  exact stoppedValue_hitting_mem ⟨j, ⟨hj₁.1, le_trans hj₁.2 (hitting_le _)⟩, hj₂⟩
  /-
    🎉 no goals
  -/


theorem upperCrossingTime_lt_lowerCrossingTime (hab : a < b)
    (hn : lowerCrossingTime a b f N (n + 1) ω ≠ N) :
    upperCrossingTime a b f N (n + 1) ω < lowerCrossingTime a b f N (n + 1) ω := by
  refine lt_of_le_of_ne upperCrossingTime_le_lowerCrossingTime fun h =>
    not_le.2 hab <| le_trans ?_ (stoppedValue_lowerCrossingTime hn)
  /-
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N n : Nat
    ω : Ω
    hab : LT.lt a b
    hn : Ne (MeasureTheory.lowerCrossingTime a b f N (HAdd.hAdd n 1) ω) N
    h : Eq (MeasureTheory.upperCrossingTime a b f N (HAdd.hAdd n 1) ω) (MeasureThe …
    ⊢ LE.le b (MeasureTheory.stoppedValue f (MeasureTheory.lowerCrossingTime a b f …
  -/
  simp only [stoppedValue]
  /-
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N n : Nat
    ω : Ω
    hab : LT.lt a b
    hn : Ne (MeasureTheory.lowerCrossingTime a b f N (HAdd.hAdd n 1) ω) N
    h : Eq (MeasureTheory.upperCrossingTime a b f N (HAdd.hAdd n 1) ω) (MeasureThe …
    ⊢ LE.le b (f (MeasureTheory.lowerCrossingTime a b f N (HAdd.hAdd n 1) ω) ω)
  -/
  rw [← h]
  /-
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N n : Nat
    ω : Ω
    hab : LT.lt a b
    hn : Ne (MeasureTheory.lowerCrossingTime a b f N (HAdd.hAdd n 1) ω) N
    h : Eq (MeasureTheory.upperCrossingTime a b f N (HAdd.hAdd n 1) ω) (MeasureThe …
    ⊢ LE.le b (f (MeasureTheory.upperCrossingTime a b f N (HAdd.hAdd n 1) ω) ω)
  -/
  exact stoppedValue_upperCrossingTime (h.symm ▸ hn)
  /-
    🎉 no goals
  -/


theorem lowerCrossingTime_lt_upperCrossingTime (hab : a < b)
    (hn : upperCrossingTime a b f N (n + 1) ω ≠ N) :
    lowerCrossingTime a b f N n ω < upperCrossingTime a b f N (n + 1) ω := by
  refine lt_of_le_of_ne lowerCrossingTime_le_upperCrossingTime_succ fun h =>
    not_le.2 hab <| le_trans (stoppedValue_upperCrossingTime hn) ?_
  /-
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N n : Nat
    ω : Ω
    hab : LT.lt a b
    hn : Ne (MeasureTheory.upperCrossingTime a b f N (HAdd.hAdd n 1) ω) N
    h : Eq (MeasureTheory.lowerCrossingTime a b f N n ω) (MeasureTheory.upperCross …
    ⊢ LE.le (MeasureTheory.stoppedValue f (MeasureTheory.upperCrossingTime a b f N …
  -/
  simp only [stoppedValue]
  /-
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N n : Nat
    ω : Ω
    hab : LT.lt a b
    hn : Ne (MeasureTheory.upperCrossingTime a b f N (HAdd.hAdd n 1) ω) N
    h : Eq (MeasureTheory.lowerCrossingTime a b f N n ω) (MeasureTheory.upperCross …
    ⊢ LE.le (f (MeasureTheory.upperCrossingTime a b f N (HAdd.hAdd n 1) ω) ω) a
  -/
  rw [← h]
  /-
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N n : Nat
    ω : Ω
    hab : LT.lt a b
    hn : Ne (MeasureTheory.upperCrossingTime a b f N (HAdd.hAdd n 1) ω) N
    h : Eq (MeasureTheory.lowerCrossingTime a b f N n ω) (MeasureTheory.upperCross …
    ⊢ LE.le (f (MeasureTheory.lowerCrossingTime a b f N n ω) ω) a
  -/
  exact stoppedValue_lowerCrossingTime (h.symm ▸ hn)
  /-
    🎉 no goals
  -/


theorem upperCrossingTime_lt_succ (hab : a < b) (hn : upperCrossingTime a b f N (n + 1) ω ≠ N) :
    upperCrossingTime a b f N n ω < upperCrossingTime a b f N (n + 1) ω :=
  lt_of_le_of_lt upperCrossingTime_le_lowerCrossingTime
    (lowerCrossingTime_lt_upperCrossingTime hab hn)


theorem lowerCrossingTime_stabilize (hnm : n ≤ m) (hn : lowerCrossingTime a b f N n ω = N) :
    lowerCrossingTime a b f N m ω = N :=
  le_antisymm lowerCrossingTime_le (le_trans (le_of_eq hn.symm) (lowerCrossingTime_mono hnm))


theorem upperCrossingTime_stabilize (hnm : n ≤ m) (hn : upperCrossingTime a b f N n ω = N) :
    upperCrossingTime a b f N m ω = N :=
  le_antisymm upperCrossingTime_le (le_trans (le_of_eq hn.symm) (upperCrossingTime_mono hnm))


theorem lowerCrossingTime_stabilize' (hnm : n ≤ m) (hn : N ≤ lowerCrossingTime a b f N n ω) :
    lowerCrossingTime a b f N m ω = N :=
  lowerCrossingTime_stabilize hnm (le_antisymm lowerCrossingTime_le hn)


theorem upperCrossingTime_stabilize' (hnm : n ≤ m) (hn : N ≤ upperCrossingTime a b f N n ω) :
    upperCrossingTime a b f N m ω = N :=
  upperCrossingTime_stabilize hnm (le_antisymm upperCrossingTime_le hn)

-- `upperCrossingTime_bound_eq` provides an explicit bound

theorem exists_upperCrossingTime_eq (f : ℕ → Ω → ℝ) (N : ℕ) (ω : Ω) (hab : a < b) :
    ∃ n, upperCrossingTime a b f N n ω = N := by
  /-
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N : Nat
    ω : Ω
    hab : LT.lt a b
    ⊢ Exists fun n => Eq (MeasureTheory.upperCrossingTime a b f N n ω) N
  -/
  by_contra h; push_neg at h
  have : StrictMono fun n => upperCrossingTime a b f N n ω :=
    strictMono_nat_of_lt_succ fun n => upperCrossingTime_lt_succ hab (h _)
  obtain ⟨_, ⟨k, rfl⟩, hk⟩ :
      ∃ (m : _) (_ : m ∈ Set.range fun n => upperCrossingTime a b f N n ω), N < m :=
    ⟨upperCrossingTime a b f N (N + 1) ω, ⟨N + 1, rfl⟩,
      lt_of_lt_of_le N.lt_succ_self (StrictMono.id_le this (N + 1))⟩
  /-
    case intro.intro.intro
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N : Nat
    ω : Ω
    hab : LT.lt a b
    h : ∀ (n : Nat), Ne (MeasureTheory.upperCrossingTime a b f N n ω) N
    this : StrictMono fun n => MeasureTheory.upperCrossingTime a b f N n ω
    k : Nat
    hk : LT.lt N ((fun n => MeasureTheory.upperCrossingTime a b f N n ω) k)
    ⊢ False
  -/
  exact not_le.2 hk upperCrossingTime_le
  /-
    🎉 no goals
  -/


theorem upperCrossingTime_lt_bddAbove (hab : a < b) :
    BddAbove {n | upperCrossingTime a b f N n ω < N} := by
  /-
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N : Nat
    ω : Ω
    hab : LT.lt a b
    ⊢ BddAbove (setOf fun n => LT.lt (MeasureTheory.upperCrossingTime a b f N n ω) …
  -/
  obtain ⟨k, hk⟩ := exists_upperCrossingTime_eq f N ω hab
  /-
    case intro
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N : Nat
    ω : Ω
    hab : LT.lt a b
    k : Nat
    hk : Eq (MeasureTheory.upperCrossingTime a b f N k ω) N
    ⊢ BddAbove (setOf fun n => LT.lt (MeasureTheory.upperCrossingTime a b f N n ω) …
  -/
  refine ⟨k, fun n (hn : upperCrossingTime a b f N n ω < N) => ?_⟩
  /-
    case intro
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N : Nat
    ω : Ω
    hab : LT.lt a b
    k : Nat
    hk : Eq (MeasureTheory.upperCrossingTime a b f N k ω) N
    n : Nat
    hn : LT.lt (MeasureTheory.upperCrossingTime a b f N n ω) N
    ⊢ LE.le n k
  -/
  by_contra hn'
  /-
    case intro
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N : Nat
    ω : Ω
    hab : LT.lt a b
    k : Nat
    hk : Eq (MeasureTheory.upperCrossingTime a b f N k ω) N
    n : Nat
    hn : LT.lt (MeasureTheory.upperCrossingTime a b f N n ω) N
    hn' : Not (LE.le n k)
    ⊢ False
  -/
  exact hn.ne (upperCrossingTime_stabilize (not_le.1 hn').le hk)
  /-
    🎉 no goals
  -/


theorem upperCrossingTime_lt_nonempty (hN : 0 < N) :
    {n | upperCrossingTime a b f N n ω < N}.Nonempty :=
  ⟨0, hN⟩


theorem upperCrossingTime_bound_eq (f : ℕ → Ω → ℝ) (N : ℕ) (ω : Ω) (hab : a < b) :
    upperCrossingTime a b f N N ω = N := by
  /-
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N : Nat
    ω : Ω
    hab : LT.lt a b
    ⊢ Eq (MeasureTheory.upperCrossingTime a b f N N ω) N
  -/
  by_cases hN' : N < Nat.find (exists_upperCrossingTime_eq f N ω hab)
    /-
      case pos
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      N : Nat
      ω : Ω
      hab : LT.lt a b
      hN' : LT.lt N (Nat.find ⋯)
      ⊢ Eq (MeasureTheory.upperCrossingTime a b f N N ω) N
    -/
  · refine le_antisymm upperCrossingTime_le ?_
    have hmono : StrictMonoOn (fun n => upperCrossingTime a b f N n ω)
        (Set.Iic (Nat.find (exists_upperCrossingTime_eq f N ω hab)).pred) := by
      refine strictMonoOn_Iic_of_lt_succ fun m hm => upperCrossingTime_lt_succ hab ?_
      rw [Nat.lt_pred_iff] at hm
      convert Nat.find_min _ hm
    /-
      case pos
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      N : Nat
      ω : Ω
      hab : LT.lt a b
      hN' : LT.lt N (Nat.find ⋯)
      hmono : StrictMonoOn (fun n => MeasureTheory.upperCrossingTime a b f N n ω) (S …
      ⊢ LE.le N (MeasureTheory.upperCrossingTime a b f N N ω)
    -/
    convert StrictMonoOn.Iic_id_le hmono N (Nat.le_sub_one_of_lt hN')
    /-
      🎉 no goals
    -/
    /-
      case neg
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      N : Nat
      ω : Ω
      hab : LT.lt a b
      hN' : Not (LT.lt N (Nat.find ⋯))
      ⊢ Eq (MeasureTheory.upperCrossingTime a b f N N ω) N
    -/
  · rw [not_lt] at hN'
    /-
      case neg
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      N : Nat
      ω : Ω
      hab : LT.lt a b
      hN' : LE.le (Nat.find ⋯) N
      ⊢ Eq (MeasureTheory.upperCrossingTime a b f N N ω) N
    -/
    exact upperCrossingTime_stabilize hN' (Nat.find_spec (exists_upperCrossingTime_eq f N ω hab))
    /-
      🎉 no goals
    -/


theorem upperCrossingTime_eq_of_bound_le (hab : a < b) (hn : N ≤ n) :
    upperCrossingTime a b f N n ω = N :=
  le_antisymm upperCrossingTime_le
    (le_trans (upperCrossingTime_bound_eq f N ω hab).symm.le (upperCrossingTime_mono hn))


theorem Adapted.isStoppingTime_crossing (hf : Adapted ℱ f) :
    IsStoppingTime ℱ (upperCrossingTime a b f N n) ∧
      IsStoppingTime ℱ (lowerCrossingTime a b f N n) := by
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    a b : Real
    f : Nat → Ω → Real
    N n : Nat
    ℱ : MeasureTheory.Filtration Nat m0
    hf : MeasureTheory.Adapted ℱ f
    ⊢ And (MeasureTheory.IsStoppingTime ℱ (MeasureTheory.upperCrossingTime a b f N …
  -/
  induction' n with k ih
    /-
      case zero
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      a b : Real
      f : Nat → Ω → Real
      N n : Nat
      ℱ : MeasureTheory.Filtration Nat m0
      hf : MeasureTheory.Adapted ℱ f
      ⊢ And (MeasureTheory.IsStoppingTime ℱ (MeasureTheory.upperCrossingTime a b f N …
    -/
  · refine ⟨isStoppingTime_const _ 0, ?_⟩
    /-
      case zero
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      a b : Real
      f : Nat → Ω → Real
      N n : Nat
      ℱ : MeasureTheory.Filtration Nat m0
      hf : MeasureTheory.Adapted ℱ f
      ⊢ MeasureTheory.IsStoppingTime ℱ (MeasureTheory.lowerCrossingTime a b f N 0)
    -/
    simp [hitting_isStoppingTime hf measurableSet_Iic]
    /-
      🎉 no goals
    -/
    /-
      case succ
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      a b : Real
      f : Nat → Ω → Real
      N n : Nat
      ℱ : MeasureTheory.Filtration Nat m0
      hf : MeasureTheory.Adapted ℱ f
      k : Nat
      ih : And (MeasureTheory.IsStoppingTime ℱ (MeasureTheory.upperCrossingTime a b  …
      ⊢ And (MeasureTheory.IsStoppingTime ℱ (MeasureTheory.upperCrossingTime a b f N …
    -/
  · obtain ⟨_, ih₂⟩ := ih
    have : IsStoppingTime ℱ (upperCrossingTime a b f N (k + 1)) := by
      intro n
      simp_rw [upperCrossingTime_succ_eq]
      exact isStoppingTime_hitting_isStoppingTime ih₂ (fun _ => lowerCrossingTime_le)
        measurableSet_Ici hf _
    /-
      case succ.intro
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      a b : Real
      f : Nat → Ω → Real
      N n : Nat
      ℱ : MeasureTheory.Filtration Nat m0
      hf : MeasureTheory.Adapted ℱ f
      k : Nat
      left✝ : MeasureTheory.IsStoppingTime ℱ (MeasureTheory.upperCrossingTime a b f  …
      ih₂ : MeasureTheory.IsStoppingTime ℱ (MeasureTheory.lowerCrossingTime a b f N k)
      this : MeasureTheory.IsStoppingTime ℱ (MeasureTheory.upperCrossingTime a b f N …
      ⊢ And (MeasureTheory.IsStoppingTime ℱ (MeasureTheory.upperCrossingTime a b f N …
    -/
    refine ⟨this, ?_⟩
    /-
      case succ.intro
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      a b : Real
      f : Nat → Ω → Real
      N n : Nat
      ℱ : MeasureTheory.Filtration Nat m0
      hf : MeasureTheory.Adapted ℱ f
      k : Nat
      left✝ : MeasureTheory.IsStoppingTime ℱ (MeasureTheory.upperCrossingTime a b f  …
      ih₂ : MeasureTheory.IsStoppingTime ℱ (MeasureTheory.lowerCrossingTime a b f N k)
      this : MeasureTheory.IsStoppingTime ℱ (MeasureTheory.upperCrossingTime a b f N …
      ⊢ MeasureTheory.IsStoppingTime ℱ (MeasureTheory.lowerCrossingTime a b f N (HAd …
    -/
    intro n
    exact isStoppingTime_hitting_isStoppingTime this (fun _ => upperCrossingTime_le)
      measurableSet_Iic hf _


theorem Adapted.isStoppingTime_upperCrossingTime (hf : Adapted ℱ f) :
    IsStoppingTime ℱ (upperCrossingTime a b f N n) :=
  hf.isStoppingTime_crossing.1


theorem Adapted.isStoppingTime_lowerCrossingTime (hf : Adapted ℱ f) :
    IsStoppingTime ℱ (lowerCrossingTime a b f N n) :=
  hf.isStoppingTime_crossing.2


/-- `upcrossingStrat a b f N n` is 1 if `n` is between a consecutive pair of lower and upper
crossings and is 0 otherwise. `upcrossingStrat` is shifted by one index so that it is adapted
rather than predictable. -/
noncomputable def upcrossingStrat (a b : ℝ) (f : ℕ → Ω → ℝ) (N n : ℕ) (ω : Ω) : ℝ :=
  ∑ k ∈ Finset.range N,
    (Set.Ico (lowerCrossingTime a b f N k ω) (upperCrossingTime a b f N (k + 1) ω)).indicator 1 n


theorem upcrossingStrat_nonneg : 0 ≤ upcrossingStrat a b f N n ω :=
  Finset.sum_nonneg fun _ _ => Set.indicator_nonneg (fun _ _ => zero_le_one) _


theorem upcrossingStrat_le_one : upcrossingStrat a b f N n ω ≤ 1 := by
  /-
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N n : Nat
    ω : Ω
    ⊢ LE.le (MeasureTheory.upcrossingStrat a b f N n ω) 1
  -/
  rw [upcrossingStrat, ← Finset.indicator_biUnion_apply]
    /-
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      N n : Nat
      ω : Ω
      ⊢ LE.le ((Set.iUnion fun i => Set.iUnion fun h => Set.Ico (MeasureTheory.lower …
    -/
  · exact Set.indicator_le_self' (fun _ _ => zero_le_one) _
    /-
      🎉 no goals
    -/
  /-
    case h
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N n : Nat
    ω : Ω
    ⊢ (↑(Finset.range N)).PairwiseDisjoint fun k => Set.Ico (MeasureTheory.lowerCr …
  -/
  intro i _ j _ hij
  /-
    case h
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N n : Nat
    ω : Ω
    i : Nat
    a✝¹ : Membership.mem (↑(Finset.range N)) i
    j : Nat
    a✝ : Membership.mem (↑(Finset.range N)) j
    hij : Ne i j
    ⊢ Function.onFun Disjoint (fun k => Set.Ico (MeasureTheory.lowerCrossingTime a …
  -/
  simp only [Set.Ico_disjoint_Ico]
  /-
    case h
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N n : Nat
    ω : Ω
    i : Nat
    a✝¹ : Membership.mem (↑(Finset.range N)) i
    j : Nat
    a✝ : Membership.mem (↑(Finset.range N)) j
    hij : Ne i j
    ⊢ LE.le (Min.min (MeasureTheory.upperCrossingTime a b f N (HAdd.hAdd i 1) ω) ( …
  -/
  obtain hij' | hij' := lt_or_gt_of_ne hij
  · rw [min_eq_left (upperCrossingTime_mono (Nat.succ_le_succ hij'.le) :
      upperCrossingTime a b f N _ ω ≤ upperCrossingTime a b f N _ ω),
      max_eq_right (lowerCrossingTime_mono hij'.le :
        lowerCrossingTime a b f N _ _ ≤ lowerCrossingTime _ _ _ _ _ _)]
    refine le_trans upperCrossingTime_le_lowerCrossingTime
      (lowerCrossingTime_mono (Nat.succ_le_of_lt hij'))
    /-
      case h.inr
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      N n : Nat
      ω : Ω
      i : Nat
      a✝¹ : Membership.mem (↑(Finset.range N)) i
      j : Nat
      a✝ : Membership.mem (↑(Finset.range N)) j
      hij : Ne i j
      hij' : GT.gt i j
      ⊢ LE.le (Min.min (MeasureTheory.upperCrossingTime a b f N (HAdd.hAdd i 1) ω) ( …
    -/
  · rw [gt_iff_lt] at hij'
    rw [min_eq_right (upperCrossingTime_mono (Nat.succ_le_succ hij'.le) :
      upperCrossingTime a b f N _ ω ≤ upperCrossingTime a b f N _ ω),
      max_eq_left (lowerCrossingTime_mono hij'.le :
        lowerCrossingTime a b f N _ _ ≤ lowerCrossingTime _ _ _ _ _ _)]
    refine le_trans upperCrossingTime_le_lowerCrossingTime
      (lowerCrossingTime_mono (Nat.succ_le_of_lt hij'))


theorem Adapted.upcrossingStrat_adapted (hf : Adapted ℱ f) :
    Adapted ℱ (upcrossingStrat a b f N) := by
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    a b : Real
    f : Nat → Ω → Real
    N : Nat
    ℱ : MeasureTheory.Filtration Nat m0
    hf : MeasureTheory.Adapted ℱ f
    ⊢ MeasureTheory.Adapted ℱ (MeasureTheory.upcrossingStrat a b f N)
  -/
  intro n
  change StronglyMeasurable[ℱ n] fun ω =>
    ∑ k ∈ Finset.range N, ({n | lowerCrossingTime a b f N k ω ≤ n} ∩
      {n | n < upperCrossingTime a b f N (k + 1) ω}).indicator 1 n
  refine Finset.stronglyMeasurable_sum _ fun i _ =>
    stronglyMeasurable_const.indicator ((hf.isStoppingTime_lowerCrossingTime n).inter ?_)
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    a b : Real
    f : Nat → Ω → Real
    N : Nat
    ℱ : MeasureTheory.Filtration Nat m0
    hf : MeasureTheory.Adapted ℱ f
    n i : Nat
    x✝ : Membership.mem (Finset.range N) i
    ⊢ MeasurableSet fun ω => setOf (fun n => LT.lt n (MeasureTheory.upperCrossingT …
  -/
  simp_rw [← not_le]
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    a b : Real
    f : Nat → Ω → Real
    N : Nat
    ℱ : MeasureTheory.Filtration Nat m0
    hf : MeasureTheory.Adapted ℱ f
    n i : Nat
    x✝ : Membership.mem (Finset.range N) i
    ⊢ MeasurableSet fun ω => setOf (fun n => Not (LE.le (MeasureTheory.upperCrossi …
  -/
  exact (hf.isStoppingTime_upperCrossingTime n).compl
  /-
    🎉 no goals
  -/


theorem Submartingale.sum_upcrossingStrat_mul [IsFiniteMeasure μ] (hf : Submartingale f ℱ μ)
    (a b : ℝ) (N : ℕ) : Submartingale (fun n : ℕ =>
      ∑ k ∈ Finset.range n, upcrossingStrat a b f N k * (f (k + 1) - f k)) ℱ μ :=
  hf.sum_mul_sub hf.adapted.upcrossingStrat_adapted (fun _ _ => upcrossingStrat_le_one) fun _ _ =>
    upcrossingStrat_nonneg


theorem Submartingale.sum_sub_upcrossingStrat_mul [IsFiniteMeasure μ] (hf : Submartingale f ℱ μ)
    (a b : ℝ) (N : ℕ) : Submartingale (fun n : ℕ =>
      ∑ k ∈ Finset.range n, (1 - upcrossingStrat a b f N k) * (f (k + 1) - f k)) ℱ μ := by
  refine hf.sum_mul_sub (fun n => (adapted_const ℱ 1 n).sub (hf.adapted.upcrossingStrat_adapted n))
    (?_ : ∀ n ω, (1 - upcrossingStrat a b f N n) ω ≤ 1) ?_
    /-
      case refine_1
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      f : Nat → Ω → Real
      ℱ : MeasureTheory.Filtration Nat m0
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hf : MeasureTheory.Submartingale f ℱ μ
      a b : Real
      N : Nat
      ⊢ ∀ (n : Nat) (ω : Ω), LE.le (HSub.hSub 1 (MeasureTheory.upcrossingStrat a b f …
    -/
  · exact fun n ω => sub_le_self _ upcrossingStrat_nonneg
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      f : Nat → Ω → Real
      ℱ : MeasureTheory.Filtration Nat m0
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hf : MeasureTheory.Submartingale f ℱ μ
      a b : Real
      N : Nat
      ⊢ ∀ (n : Nat) (ω : Ω), LE.le 0 (HSub.hSub 1 (MeasureTheory.upcrossingStrat a b …
    -/
  · intro n ω
    /-
      case refine_2
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      f : Nat → Ω → Real
      ℱ : MeasureTheory.Filtration Nat m0
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hf : MeasureTheory.Submartingale f ℱ μ
      a b : Real
      N n : Nat
      ω : Ω
      ⊢ LE.le 0 (HSub.hSub 1 (MeasureTheory.upcrossingStrat a b f N n) ω)
    -/
    simp [upcrossingStrat_le_one]
    /-
      🎉 no goals
    -/


theorem Submartingale.sum_mul_upcrossingStrat_le [IsFiniteMeasure μ] (hf : Submartingale f ℱ μ) :
    μ[∑ k ∈ Finset.range n, upcrossingStrat a b f N k * (f (k + 1) - f k)] ≤ μ[f n] - μ[f 0] := by
  have h₁ : (0 : ℝ) ≤
      μ[∑ k ∈ Finset.range n, (1 - upcrossingStrat a b f N k) * (f (k + 1) - f k)] := by
    have := (hf.sum_sub_upcrossingStrat_mul a b N).setIntegral_le (zero_le n) MeasurableSet.univ
    rw [setIntegral_univ, setIntegral_univ] at this
    refine le_trans ?_ this
    simp only [Finset.range_zero, Finset.sum_empty, integral_zero', le_refl]
  have h₂ : μ[∑ k ∈ Finset.range n, (1 - upcrossingStrat a b f N k) * (f (k + 1) - f k)] =
    μ[∑ k ∈ Finset.range n, (f (k + 1) - f k)] -
      μ[∑ k ∈ Finset.range n, upcrossingStrat a b f N k * (f (k + 1) - f k)] := by
    simp only [sub_mul, one_mul, Finset.sum_sub_distrib, Pi.sub_apply, Finset.sum_apply,
      Pi.mul_apply]
    refine integral_sub (Integrable.sub (integrable_finset_sum _ fun i _ => hf.integrable _)
      (integrable_finset_sum _ fun i _ => hf.integrable _)) ?_
    convert (hf.sum_upcrossingStrat_mul a b N).integrable n using 1
    ext; simp
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    a b : Real
    f : Nat → Ω → Real
    N n : Nat
    ℱ : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : MeasureTheory.Submartingale f ℱ μ
    h₁ : LE.le 0 (MeasureTheory.integral μ fun x => (Finset.range n).sum (fun k => …
    h₂ : Eq (MeasureTheory.integral μ fun x => (Finset.range n).sum (fun k => HMul …
    ⊢ LE.le (MeasureTheory.integral μ fun x => (Finset.range n).sum (fun k => HMul …
  -/
  rw [h₂, sub_nonneg] at h₁
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    a b : Real
    f : Nat → Ω → Real
    N n : Nat
    ℱ : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : MeasureTheory.Submartingale f ℱ μ
    h₁ : LE.le (MeasureTheory.integral μ fun x => (Finset.range n).sum (fun k => H …
    h₂ : Eq (MeasureTheory.integral μ fun x => (Finset.range n).sum (fun k => HMul …
    ⊢ LE.le (MeasureTheory.integral μ fun x => (Finset.range n).sum (fun k => HMul …
  -/
  refine le_trans h₁ ?_
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    a b : Real
    f : Nat → Ω → Real
    N n : Nat
    ℱ : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : MeasureTheory.Submartingale f ℱ μ
    h₁ : LE.le (MeasureTheory.integral μ fun x => (Finset.range n).sum (fun k => H …
    h₂ : Eq (MeasureTheory.integral μ fun x => (Finset.range n).sum (fun k => HMul …
    ⊢ LE.le (MeasureTheory.integral μ fun x => (Finset.range n).sum (fun k => HSub …
  -/
  simp_rw [Finset.sum_range_sub, integral_sub' (hf.integrable _) (hf.integrable _), le_refl]
  /-
    🎉 no goals
  -/


/-- The number of upcrossings (strictly) before time `N`. -/
noncomputable def upcrossingsBefore [Preorder ι] [OrderBot ι] [InfSet ι] (a b : ℝ) (f : ι → Ω → ℝ)
    (N : ι) (ω : Ω) : ℕ :=
  sSup {n | upperCrossingTime a b f N n ω < N}


@[simp]
theorem upcrossingsBefore_bot [Preorder ι] [OrderBot ι] [InfSet ι] {a b : ℝ} {f : ι → Ω → ℝ}
                                                    /-
                                                      Ω : Type u_1
                                                      ι : Type u_2
                                                      inst✝² : Preorder ι
                                                      inst✝¹ : OrderBot ι
                                                      inst✝ : InfSet ι
                                                      a b : Real
                                                      f : ι → Ω → Real
                                                      ω : Ω
                                                      ⊢ Eq (MeasureTheory.upcrossingsBefore a b f Bot.bot ω) Bot.bot
                                                    -/
    {ω : Ω} : upcrossingsBefore a b f ⊥ ω = ⊥ := by simp [upcrossingsBefore]
                                                    /-
                                                      🎉 no goals
                                                    -/


                                                                       /-
                                                                         Ω : Type u_1
                                                                         a b : Real
                                                                         f : Nat → Ω → Real
                                                                         ω : Ω
                                                                         ⊢ Eq (MeasureTheory.upcrossingsBefore a b f 0 ω) 0
                                                                       -/
theorem upcrossingsBefore_zero : upcrossingsBefore a b f 0 ω = 0 := by simp [upcrossingsBefore]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[simp]
theorem upcrossingsBefore_zero' : upcrossingsBefore a b f 0 = 0 := by
  /-
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    ⊢ Eq (MeasureTheory.upcrossingsBefore a b f 0) 0
  -/
  ext ω; exact upcrossingsBefore_zero
         /-
           🎉 no goals
         -/


theorem upperCrossingTime_lt_of_le_upcrossingsBefore (hN : 0 < N) (hab : a < b)
    (hn : n ≤ upcrossingsBefore a b f N ω) : upperCrossingTime a b f N n ω < N :=
  haveI : upperCrossingTime a b f N (upcrossingsBefore a b f N ω) ω < N :=
    (upperCrossingTime_lt_nonempty hN).csSup_mem
      ((OrderBot.bddBelow _).finite_of_bddAbove (upperCrossingTime_lt_bddAbove hab))
  lt_of_le_of_lt (upperCrossingTime_mono hn) this


theorem upperCrossingTime_eq_of_upcrossingsBefore_lt (hab : a < b)
    (hn : upcrossingsBefore a b f N ω < n) : upperCrossingTime a b f N n ω = N := by
  /-
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N n : Nat
    ω : Ω
    hab : LT.lt a b
    hn : LT.lt (MeasureTheory.upcrossingsBefore a b f N ω) n
    ⊢ Eq (MeasureTheory.upperCrossingTime a b f N n ω) N
  -/
  refine le_antisymm upperCrossingTime_le (not_lt.1 ?_)
  /-
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N n : Nat
    ω : Ω
    hab : LT.lt a b
    hn : LT.lt (MeasureTheory.upcrossingsBefore a b f N ω) n
    ⊢ Not (LT.lt (MeasureTheory.upperCrossingTime a b f N n ω) N)
  -/
  convert not_mem_of_csSup_lt hn (upperCrossingTime_lt_bddAbove hab) using 1
  /-
    🎉 no goals
  -/


theorem upcrossingsBefore_le (f : ℕ → Ω → ℝ) (ω : Ω) (hab : a < b) :
    upcrossingsBefore a b f N ω ≤ N := by
  /-
    Ω : Type u_1
    a b : Real
    N : Nat
    f : Nat → Ω → Real
    ω : Ω
    hab : LT.lt a b
    ⊢ LE.le (MeasureTheory.upcrossingsBefore a b f N ω) N
  -/
  by_cases hN : N = 0
    /-
      case pos
      Ω : Type u_1
      a b : Real
      N : Nat
      f : Nat → Ω → Real
      ω : Ω
      hab : LT.lt a b
      hN : Eq N 0
      ⊢ LE.le (MeasureTheory.upcrossingsBefore a b f N ω) N
    -/
  · subst hN
    /-
      case pos
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      ω : Ω
      hab : LT.lt a b
      ⊢ LE.le (MeasureTheory.upcrossingsBefore a b f 0 ω) 0
    -/
    rw [upcrossingsBefore_zero]
    /-
      🎉 no goals
    -/
    /-
      case neg
      Ω : Type u_1
      a b : Real
      N : Nat
      f : Nat → Ω → Real
      ω : Ω
      hab : LT.lt a b
      hN : Not (Eq N 0)
      ⊢ LE.le (MeasureTheory.upcrossingsBefore a b f N ω) N
    -/
  · refine csSup_le ⟨0, zero_lt_iff.2 hN⟩ fun n (hn : _ < N) => ?_
    /-
      case neg
      Ω : Type u_1
      a b : Real
      N : Nat
      f : Nat → Ω → Real
      ω : Ω
      hab : LT.lt a b
      hN : Not (Eq N 0)
      n : Nat
      hn : LT.lt (MeasureTheory.upperCrossingTime a b f N n ω) N
      ⊢ LE.le n N
    -/
    by_contra hnN
    /-
      case neg
      Ω : Type u_1
      a b : Real
      N : Nat
      f : Nat → Ω → Real
      ω : Ω
      hab : LT.lt a b
      hN : Not (Eq N 0)
      n : Nat
      hn : LT.lt (MeasureTheory.upperCrossingTime a b f N n ω) N
      hnN : Not (LE.le n N)
      ⊢ False
    -/
    exact hn.ne (upperCrossingTime_eq_of_bound_le hab (not_le.1 hnN).le)
    /-
      🎉 no goals
    -/


theorem crossing_eq_crossing_of_lowerCrossingTime_lt {M : ℕ} (hNM : N ≤ M)
    (h : lowerCrossingTime a b f N n ω < N) :
    upperCrossingTime a b f M n ω = upperCrossingTime a b f N n ω ∧
      lowerCrossingTime a b f M n ω = lowerCrossingTime a b f N n ω := by
  have h' : upperCrossingTime a b f N n ω < N :=
    lt_of_le_of_lt upperCrossingTime_le_lowerCrossingTime h
  /-
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N n : Nat
    ω : Ω
    M : Nat
    hNM : LE.le N M
    h : LT.lt (MeasureTheory.lowerCrossingTime a b f N n ω) N
    h' : LT.lt (MeasureTheory.upperCrossingTime a b f N n ω) N
    ⊢ And (Eq (MeasureTheory.upperCrossingTime a b f M n ω) (MeasureTheory.upperCr …
  -/
  induction' n with k ih
  · simp only [upperCrossingTime_zero, bot_eq_zero', eq_self_iff_true,
      lowerCrossingTime_zero, true_and, eq_comm]
    /-
      case zero
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      N n : Nat
      ω : Ω
      M : Nat
      hNM : LE.le N M
      h : LT.lt (MeasureTheory.lowerCrossingTime a b f N 0 ω) N
      h' : LT.lt (MeasureTheory.upperCrossingTime a b f N 0 ω) N
      ⊢ Eq (MeasureTheory.hitting f (Set.Iic a) 0 N ω) (MeasureTheory.hitting f (Set …
    -/
    refine hitting_eq_hitting_of_exists hNM ?_
    /-
      case zero
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      N n : Nat
      ω : Ω
      M : Nat
      hNM : LE.le N M
      h : LT.lt (MeasureTheory.lowerCrossingTime a b f N 0 ω) N
      h' : LT.lt (MeasureTheory.upperCrossingTime a b f N 0 ω) N
      ⊢ Exists fun j => And (Membership.mem (Set.Icc 0 N) j) (Membership.mem (Set.Ii …
    -/
    rw [lowerCrossingTime, hitting_lt_iff] at h
      /-
        case zero
        Ω : Type u_1
        a b : Real
        f : Nat → Ω → Real
        N n : Nat
        ω : Ω
        M : Nat
        hNM : LE.le N M
        h : Exists fun j => And (Membership.mem (Set.Ico (MeasureTheory.upperCrossingT …
        h' : LT.lt (MeasureTheory.upperCrossingTime a b f N 0 ω) N
        ⊢ Exists fun j => And (Membership.mem (Set.Icc 0 N) j) (Membership.mem (Set.Ii …
      -/
    · obtain ⟨j, hj₁, hj₂⟩ := h
      /-
        case zero.intro.intro
        Ω : Type u_1
        a b : Real
        f : Nat → Ω → Real
        N n : Nat
        ω : Ω
        M : Nat
        hNM : LE.le N M
        h' : LT.lt (MeasureTheory.upperCrossingTime a b f N 0 ω) N
        j : Nat
        hj₁ : Membership.mem (Set.Ico (MeasureTheory.upperCrossingTime a b f N 0 ω) N) j
        hj₂ : Membership.mem (Set.Iic a) (f j ω)
        ⊢ Exists fun j => And (Membership.mem (Set.Icc 0 N) j) (Membership.mem (Set.Ii …
      -/
      exact ⟨j, ⟨hj₁.1, hj₁.2.le⟩, hj₂⟩
      /-
        🎉 no goals
      -/
      /-
        case zero.hi
        Ω : Type u_1
        a b : Real
        f : Nat → Ω → Real
        N n : Nat
        ω : Ω
        M : Nat
        hNM : LE.le N M
        h : LT.lt (MeasureTheory.hitting f (Set.Iic a) (MeasureTheory.upperCrossingTim …
        h' : LT.lt (MeasureTheory.upperCrossingTime a b f N 0 ω) N
        ⊢ LE.le N N
      -/
    · exact le_rfl
      /-
        🎉 no goals
      -/
  · specialize ih (lt_of_le_of_lt (lowerCrossingTime_mono (Nat.le_succ _)) h)
      (lt_of_le_of_lt (upperCrossingTime_mono (Nat.le_succ _)) h')
    have : upperCrossingTime a b f M k.succ ω = upperCrossingTime a b f N k.succ ω := by
      rw [upperCrossingTime_succ_eq, hitting_lt_iff] at h'
      · simp only [upperCrossingTime_succ_eq]
        obtain ⟨j, hj₁, hj₂⟩ := h'
        rw [eq_comm, ih.2]
        exact hitting_eq_hitting_of_exists hNM ⟨j, ⟨hj₁.1, hj₁.2.le⟩, hj₂⟩
      · exact le_rfl
    /-
      case succ
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      N n : Nat
      ω : Ω
      M : Nat
      hNM : LE.le N M
      k : Nat
      h : LT.lt (MeasureTheory.lowerCrossingTime a b f N (HAdd.hAdd k 1) ω) N
      h' : LT.lt (MeasureTheory.upperCrossingTime a b f N (HAdd.hAdd k 1) ω) N
      ih : And (Eq (MeasureTheory.upperCrossingTime a b f M k ω) (MeasureTheory.uppe …
      this : Eq (MeasureTheory.upperCrossingTime a b f M k.succ ω) (MeasureTheory.up …
      ⊢ And (Eq (MeasureTheory.upperCrossingTime a b f M (HAdd.hAdd k 1) ω) (Measure …
    -/
    refine ⟨this, ?_⟩
    /-
      case succ
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      N n : Nat
      ω : Ω
      M : Nat
      hNM : LE.le N M
      k : Nat
      h : LT.lt (MeasureTheory.lowerCrossingTime a b f N (HAdd.hAdd k 1) ω) N
      h' : LT.lt (MeasureTheory.upperCrossingTime a b f N (HAdd.hAdd k 1) ω) N
      ih : And (Eq (MeasureTheory.upperCrossingTime a b f M k ω) (MeasureTheory.uppe …
      this : Eq (MeasureTheory.upperCrossingTime a b f M k.succ ω) (MeasureTheory.up …
      ⊢ Eq (MeasureTheory.lowerCrossingTime a b f M (HAdd.hAdd k 1) ω) (MeasureTheor …
    -/
    simp only [lowerCrossingTime, eq_comm, this, Nat.succ_eq_add_one]
    /-
      case succ
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      N n : Nat
      ω : Ω
      M : Nat
      hNM : LE.le N M
      k : Nat
      h : LT.lt (MeasureTheory.lowerCrossingTime a b f N (HAdd.hAdd k 1) ω) N
      h' : LT.lt (MeasureTheory.upperCrossingTime a b f N (HAdd.hAdd k 1) ω) N
      ih : And (Eq (MeasureTheory.upperCrossingTime a b f M k ω) (MeasureTheory.uppe …
      this : Eq (MeasureTheory.upperCrossingTime a b f M k.succ ω) (MeasureTheory.up …
      ⊢ Eq (MeasureTheory.hitting f (Set.Iic a) (MeasureTheory.upperCrossingTime a b …
    -/
    refine hitting_eq_hitting_of_exists hNM ?_
    /-
      case succ
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      N n : Nat
      ω : Ω
      M : Nat
      hNM : LE.le N M
      k : Nat
      h : LT.lt (MeasureTheory.lowerCrossingTime a b f N (HAdd.hAdd k 1) ω) N
      h' : LT.lt (MeasureTheory.upperCrossingTime a b f N (HAdd.hAdd k 1) ω) N
      ih : And (Eq (MeasureTheory.upperCrossingTime a b f M k ω) (MeasureTheory.uppe …
      this : Eq (MeasureTheory.upperCrossingTime a b f M k.succ ω) (MeasureTheory.up …
      ⊢ Exists fun j => And (Membership.mem (Set.Icc (MeasureTheory.upperCrossingTim …
    -/
    rw [lowerCrossingTime, hitting_lt_iff _ le_rfl] at h
    /-
      case succ
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      N n : Nat
      ω : Ω
      M : Nat
      hNM : LE.le N M
      k : Nat
      h : Exists fun j => And (Membership.mem (Set.Ico (MeasureTheory.upperCrossingT …
      h' : LT.lt (MeasureTheory.upperCrossingTime a b f N (HAdd.hAdd k 1) ω) N
      ih : And (Eq (MeasureTheory.upperCrossingTime a b f M k ω) (MeasureTheory.uppe …
      this : Eq (MeasureTheory.upperCrossingTime a b f M k.succ ω) (MeasureTheory.up …
      ⊢ Exists fun j => And (Membership.mem (Set.Icc (MeasureTheory.upperCrossingTim …
    -/
    obtain ⟨j, hj₁, hj₂⟩ := h
    /-
      case succ.intro.intro
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      N n : Nat
      ω : Ω
      M : Nat
      hNM : LE.le N M
      k : Nat
      h' : LT.lt (MeasureTheory.upperCrossingTime a b f N (HAdd.hAdd k 1) ω) N
      ih : And (Eq (MeasureTheory.upperCrossingTime a b f M k ω) (MeasureTheory.uppe …
      this : Eq (MeasureTheory.upperCrossingTime a b f M k.succ ω) (MeasureTheory.up …
      j : Nat
      hj₁ : Membership.mem (Set.Ico (MeasureTheory.upperCrossingTime a b f N (HAdd.h …
      hj₂ : Membership.mem (Set.Iic a) (f j ω)
      ⊢ Exists fun j => And (Membership.mem (Set.Icc (MeasureTheory.upperCrossingTim …
    -/
    exact ⟨j, ⟨hj₁.1, hj₁.2.le⟩, hj₂⟩
    /-
      🎉 no goals
    -/


theorem crossing_eq_crossing_of_upperCrossingTime_lt {M : ℕ} (hNM : N ≤ M)
    (h : upperCrossingTime a b f N (n + 1) ω < N) :
    upperCrossingTime a b f M (n + 1) ω = upperCrossingTime a b f N (n + 1) ω ∧
      lowerCrossingTime a b f M n ω = lowerCrossingTime a b f N n ω := by
  have := (crossing_eq_crossing_of_lowerCrossingTime_lt hNM
    (lt_of_le_of_lt lowerCrossingTime_le_upperCrossingTime_succ h)).2
  /-
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N n : Nat
    ω : Ω
    M : Nat
    hNM : LE.le N M
    h : LT.lt (MeasureTheory.upperCrossingTime a b f N (HAdd.hAdd n 1) ω) N
    this : Eq (MeasureTheory.lowerCrossingTime a b f M n ω) (MeasureTheory.lowerCr …
    ⊢ And (Eq (MeasureTheory.upperCrossingTime a b f M (HAdd.hAdd n 1) ω) (Measure …
  -/
  refine ⟨?_, this⟩
  /-
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N n : Nat
    ω : Ω
    M : Nat
    hNM : LE.le N M
    h : LT.lt (MeasureTheory.upperCrossingTime a b f N (HAdd.hAdd n 1) ω) N
    this : Eq (MeasureTheory.lowerCrossingTime a b f M n ω) (MeasureTheory.lowerCr …
    ⊢ Eq (MeasureTheory.upperCrossingTime a b f M (HAdd.hAdd n 1) ω) (MeasureTheor …
  -/
  rw [upperCrossingTime_succ_eq, upperCrossingTime_succ_eq, eq_comm, this]
  /-
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N n : Nat
    ω : Ω
    M : Nat
    hNM : LE.le N M
    h : LT.lt (MeasureTheory.upperCrossingTime a b f N (HAdd.hAdd n 1) ω) N
    this : Eq (MeasureTheory.lowerCrossingTime a b f M n ω) (MeasureTheory.lowerCr …
    ⊢ Eq (MeasureTheory.hitting f (Set.Ici b) (MeasureTheory.lowerCrossingTime a b …
  -/
  refine hitting_eq_hitting_of_exists hNM ?_
  /-
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N n : Nat
    ω : Ω
    M : Nat
    hNM : LE.le N M
    h : LT.lt (MeasureTheory.upperCrossingTime a b f N (HAdd.hAdd n 1) ω) N
    this : Eq (MeasureTheory.lowerCrossingTime a b f M n ω) (MeasureTheory.lowerCr …
    ⊢ Exists fun j => And (Membership.mem (Set.Icc (MeasureTheory.lowerCrossingTim …
  -/
  rw [upperCrossingTime_succ_eq, hitting_lt_iff] at h
    /-
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      N n : Nat
      ω : Ω
      M : Nat
      hNM : LE.le N M
      h : Exists fun j => And (Membership.mem (Set.Ico (MeasureTheory.lowerCrossingT …
      this : Eq (MeasureTheory.lowerCrossingTime a b f M n ω) (MeasureTheory.lowerCr …
      ⊢ Exists fun j => And (Membership.mem (Set.Icc (MeasureTheory.lowerCrossingTim …
    -/
  · obtain ⟨j, hj₁, hj₂⟩ := h
    /-
      case intro.intro
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      N n : Nat
      ω : Ω
      M : Nat
      hNM : LE.le N M
      this : Eq (MeasureTheory.lowerCrossingTime a b f M n ω) (MeasureTheory.lowerCr …
      j : Nat
      hj₁ : Membership.mem (Set.Ico (MeasureTheory.lowerCrossingTime a b f N n ω) N) j
      hj₂ : Membership.mem (Set.Ici b) (f j ω)
      ⊢ Exists fun j => And (Membership.mem (Set.Icc (MeasureTheory.lowerCrossingTim …
    -/
    exact ⟨j, ⟨hj₁.1, hj₁.2.le⟩, hj₂⟩
    /-
      🎉 no goals
    -/
    /-
      case hi
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      N n : Nat
      ω : Ω
      M : Nat
      hNM : LE.le N M
      h : LT.lt (MeasureTheory.hitting f (Set.Ici b) (MeasureTheory.lowerCrossingTim …
      this : Eq (MeasureTheory.lowerCrossingTime a b f M n ω) (MeasureTheory.lowerCr …
      ⊢ LE.le N N
    -/
  · exact le_rfl
    /-
      🎉 no goals
    -/


theorem upperCrossingTime_eq_upperCrossingTime_of_lt {M : ℕ} (hNM : N ≤ M)
    (h : upperCrossingTime a b f N n ω < N) :
    upperCrossingTime a b f M n ω = upperCrossingTime a b f N n ω := by
  /-
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N n : Nat
    ω : Ω
    M : Nat
    hNM : LE.le N M
    h : LT.lt (MeasureTheory.upperCrossingTime a b f N n ω) N
    ⊢ Eq (MeasureTheory.upperCrossingTime a b f M n ω) (MeasureTheory.upperCrossin …
  -/
  cases n
    /-
      case zero
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      N : Nat
      ω : Ω
      M : Nat
      hNM : LE.le N M
      h : LT.lt (MeasureTheory.upperCrossingTime a b f N 0 ω) N
      ⊢ Eq (MeasureTheory.upperCrossingTime a b f M 0 ω) (MeasureTheory.upperCrossin …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      N : Nat
      ω : Ω
      M : Nat
      hNM : LE.le N M
      n✝ : Nat
      h : LT.lt (MeasureTheory.upperCrossingTime a b f N (HAdd.hAdd n✝ 1) ω) N
      ⊢ Eq (MeasureTheory.upperCrossingTime a b f M (HAdd.hAdd n✝ 1) ω) (MeasureTheo …
    -/
  · exact (crossing_eq_crossing_of_upperCrossingTime_lt hNM h).1
    /-
      🎉 no goals
    -/


theorem upcrossingsBefore_mono (hab : a < b) : Monotone fun N ω => upcrossingsBefore a b f N ω := by
  /-
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    hab : LT.lt a b
    ⊢ Monotone fun N ω => MeasureTheory.upcrossingsBefore a b f N ω
  -/
  intro N M hNM ω
  /-
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    hab : LT.lt a b
    N M : Nat
    hNM : LE.le N M
    ω : Ω
    ⊢ LE.le ((fun N ω => MeasureTheory.upcrossingsBefore a b f N ω) N ω) ((fun N ω …
  -/
  simp only [upcrossingsBefore]
  /-
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    hab : LT.lt a b
    N M : Nat
    hNM : LE.le N M
    ω : Ω
    ⊢ LE.le (SupSet.sSup (setOf fun n => LT.lt (MeasureTheory.upperCrossingTime a  …
  -/
  by_cases hemp : {n : ℕ | upperCrossingTime a b f N n ω < N}.Nonempty
    /-
      case pos
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      hab : LT.lt a b
      N M : Nat
      hNM : LE.le N M
      ω : Ω
      hemp : (setOf fun n => LT.lt (MeasureTheory.upperCrossingTime a b f N n ω) N). …
      ⊢ LE.le (SupSet.sSup (setOf fun n => LT.lt (MeasureTheory.upperCrossingTime a  …
    -/
  · refine csSup_le_csSup (upperCrossingTime_lt_bddAbove hab) hemp fun n hn => ?_
    /-
      case pos
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      hab : LT.lt a b
      N M : Nat
      hNM : LE.le N M
      ω : Ω
      hemp : (setOf fun n => LT.lt (MeasureTheory.upperCrossingTime a b f N n ω) N). …
      n : Nat
      hn : Membership.mem (setOf fun n => LT.lt (MeasureTheory.upperCrossingTime a b …
      ⊢ Membership.mem (setOf fun n => LT.lt (MeasureTheory.upperCrossingTime a b f  …
    -/
    rw [Set.mem_setOf_eq, upperCrossingTime_eq_upperCrossingTime_of_lt hNM hn]
    /-
      case pos
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      hab : LT.lt a b
      N M : Nat
      hNM : LE.le N M
      ω : Ω
      hemp : (setOf fun n => LT.lt (MeasureTheory.upperCrossingTime a b f N n ω) N). …
      n : Nat
      hn : Membership.mem (setOf fun n => LT.lt (MeasureTheory.upperCrossingTime a b …
      ⊢ LT.lt (MeasureTheory.upperCrossingTime a b f N n ω) M
    -/
    exact lt_of_lt_of_le hn hNM
    /-
      🎉 no goals
    -/
    /-
      case neg
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      hab : LT.lt a b
      N M : Nat
      hNM : LE.le N M
      ω : Ω
      hemp : Not (setOf fun n => LT.lt (MeasureTheory.upperCrossingTime a b f N n ω) …
      ⊢ LE.le (SupSet.sSup (setOf fun n => LT.lt (MeasureTheory.upperCrossingTime a  …
    -/
  · rw [Set.not_nonempty_iff_eq_empty] at hemp
    /-
      case neg
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      hab : LT.lt a b
      N M : Nat
      hNM : LE.le N M
      ω : Ω
      hemp : Eq (setOf fun n => LT.lt (MeasureTheory.upperCrossingTime a b f N n ω)  …
      ⊢ LE.le (SupSet.sSup (setOf fun n => LT.lt (MeasureTheory.upperCrossingTime a  …
    -/
    simp [hemp, csSup_empty, bot_eq_zero', zero_le']
    /-
      🎉 no goals
    -/


theorem upcrossingsBefore_lt_of_exists_upcrossing (hab : a < b) {N₁ N₂ : ℕ} (hN₁ : N ≤ N₁)
    (hN₁' : f N₁ ω < a) (hN₂ : N₁ ≤ N₂) (hN₂' : b < f N₂ ω) :
    upcrossingsBefore a b f N ω < upcrossingsBefore a b f (N₂ + 1) ω := by
  /-
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N : Nat
    ω : Ω
    hab : LT.lt a b
    N₁ N₂ : Nat
    hN₁ : LE.le N N₁
    hN₁' : LT.lt (f N₁ ω) a
    hN₂ : LE.le N₁ N₂
    hN₂' : LT.lt b (f N₂ ω)
    ⊢ LT.lt (MeasureTheory.upcrossingsBefore a b f N ω) (MeasureTheory.upcrossings …
  -/
  refine lt_of_lt_of_le (Nat.lt_succ_self _) (le_csSup (upperCrossingTime_lt_bddAbove hab) ?_)
  /-
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N : Nat
    ω : Ω
    hab : LT.lt a b
    N₁ N₂ : Nat
    hN₁ : LE.le N N₁
    hN₁' : LT.lt (f N₁ ω) a
    hN₂ : LE.le N₁ N₂
    hN₂' : LT.lt b (f N₂ ω)
    ⊢ Membership.mem (setOf fun n => LT.lt (MeasureTheory.upperCrossingTime a b f  …
  -/
  rw [Set.mem_setOf_eq, upperCrossingTime_succ_eq, hitting_lt_iff _ le_rfl]
  /-
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N : Nat
    ω : Ω
    hab : LT.lt a b
    N₁ N₂ : Nat
    hN₁ : LE.le N N₁
    hN₁' : LT.lt (f N₁ ω) a
    hN₂ : LE.le N₁ N₂
    hN₂' : LT.lt b (f N₂ ω)
    ⊢ Exists fun j => And (Membership.mem (Set.Ico (MeasureTheory.lowerCrossingTim …
  -/
  refine ⟨N₂, ⟨?_, Nat.lt_succ_self _⟩, hN₂'.le⟩
  /-
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N : Nat
    ω : Ω
    hab : LT.lt a b
    N₁ N₂ : Nat
    hN₁ : LE.le N N₁
    hN₁' : LT.lt (f N₁ ω) a
    hN₂ : LE.le N₁ N₂
    hN₂' : LT.lt b (f N₂ ω)
    ⊢ LE.le (MeasureTheory.lowerCrossingTime a b f (HAdd.hAdd N₂ 1) (MeasureTheory …
  -/
  rw [lowerCrossingTime, hitting_le_iff_of_lt _ (Nat.lt_succ_self _)]
  /-
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N : Nat
    ω : Ω
    hab : LT.lt a b
    N₁ N₂ : Nat
    hN₁ : LE.le N N₁
    hN₁' : LT.lt (f N₁ ω) a
    hN₂ : LE.le N₁ N₂
    hN₂' : LT.lt b (f N₂ ω)
    ⊢ Exists fun j => And (Membership.mem (Set.Icc (MeasureTheory.upperCrossingTim …
  -/
  refine ⟨N₁, ⟨le_trans ?_ hN₁, hN₂⟩, hN₁'.le⟩
  /-
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N : Nat
    ω : Ω
    hab : LT.lt a b
    N₁ N₂ : Nat
    hN₁ : LE.le N N₁
    hN₁' : LT.lt (f N₁ ω) a
    hN₂ : LE.le N₁ N₂
    hN₂' : LT.lt b (f N₂ ω)
    ⊢ LE.le (MeasureTheory.upperCrossingTime a b f (HAdd.hAdd N₂ 1) (MeasureTheory …
  -/
  by_cases hN : 0 < N
  · have : upperCrossingTime a b f N (upcrossingsBefore a b f N ω) ω < N :=
      Nat.sSup_mem (upperCrossingTime_lt_nonempty hN) (upperCrossingTime_lt_bddAbove hab)
    rw [upperCrossingTime_eq_upperCrossingTime_of_lt (hN₁.trans (hN₂.trans <| Nat.le_succ _))
      this]
    /-
      case pos
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      N : Nat
      ω : Ω
      hab : LT.lt a b
      N₁ N₂ : Nat
      hN₁ : LE.le N N₁
      hN₁' : LT.lt (f N₁ ω) a
      hN₂ : LE.le N₁ N₂
      hN₂' : LT.lt b (f N₂ ω)
      hN : LT.lt 0 N
      this : LT.lt (MeasureTheory.upperCrossingTime a b f N (MeasureTheory.upcrossin …
      ⊢ LE.le (MeasureTheory.upperCrossingTime a b f N (MeasureTheory.upcrossingsBef …
    -/
    exact this.le
    /-
      🎉 no goals
    -/
    /-
      case neg
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      N : Nat
      ω : Ω
      hab : LT.lt a b
      N₁ N₂ : Nat
      hN₁ : LE.le N N₁
      hN₁' : LT.lt (f N₁ ω) a
      hN₂ : LE.le N₁ N₂
      hN₂' : LT.lt b (f N₂ ω)
      hN : Not (LT.lt 0 N)
      ⊢ LE.le (MeasureTheory.upperCrossingTime a b f (HAdd.hAdd N₂ 1) (MeasureTheory …
    -/
  · rw [not_lt, Nat.le_zero] at hN
    /-
      case neg
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      N : Nat
      ω : Ω
      hab : LT.lt a b
      N₁ N₂ : Nat
      hN₁ : LE.le N N₁
      hN₁' : LT.lt (f N₁ ω) a
      hN₂ : LE.le N₁ N₂
      hN₂' : LT.lt b (f N₂ ω)
      hN : Eq N 0
      ⊢ LE.le (MeasureTheory.upperCrossingTime a b f (HAdd.hAdd N₂ 1) (MeasureTheory …
    -/
    rw [hN, upcrossingsBefore_zero, upperCrossingTime_zero, Pi.bot_apply, bot_eq_zero']
    /-
      🎉 no goals
    -/


theorem lowerCrossingTime_lt_of_lt_upcrossingsBefore (hN : 0 < N) (hab : a < b)
    (hn : n < upcrossingsBefore a b f N ω) : lowerCrossingTime a b f N n ω < N :=
  lt_of_le_of_lt lowerCrossingTime_le_upperCrossingTime_succ
    (upperCrossingTime_lt_of_le_upcrossingsBefore hN hab hn)


theorem le_sub_of_le_upcrossingsBefore (hN : 0 < N) (hab : a < b)
    (hn : n < upcrossingsBefore a b f N ω) :
    b - a ≤ stoppedValue f (upperCrossingTime a b f N (n + 1)) ω -
      stoppedValue f (lowerCrossingTime a b f N n) ω :=
  sub_le_sub
    (stoppedValue_upperCrossingTime (upperCrossingTime_lt_of_le_upcrossingsBefore hN hab hn).ne)
    (stoppedValue_lowerCrossingTime (lowerCrossingTime_lt_of_lt_upcrossingsBefore hN hab hn).ne)


theorem sub_eq_zero_of_upcrossingsBefore_lt (hab : a < b) (hn : upcrossingsBefore a b f N ω < n) :
    stoppedValue f (upperCrossingTime a b f N (n + 1)) ω -
      stoppedValue f (lowerCrossingTime a b f N n) ω = 0 := by
  have : N ≤ upperCrossingTime a b f N n ω := by
    rw [upcrossingsBefore] at hn
    rw [← not_lt]
    exact fun h => not_le.2 hn (le_csSup (upperCrossingTime_lt_bddAbove hab) h)
  simp [stoppedValue, upperCrossingTime_stabilize' (Nat.le_succ n) this,
    lowerCrossingTime_stabilize' le_rfl (le_trans this upperCrossingTime_le_lowerCrossingTime)]


theorem mul_upcrossingsBefore_le (hf : a ≤ f N ω) (hab : a < b) :
    (b - a) * upcrossingsBefore a b f N ω ≤
    ∑ k ∈ Finset.range N, upcrossingStrat a b f N k ω * (f (k + 1) - f k) ω := by
  classical
  by_cases hN : N = 0
  · simp [hN]
  simp_rw [upcrossingStrat, Finset.sum_mul, ←
    Set.indicator_mul_left _ _ (fun x ↦ (f (x + 1) - f x) ω), Pi.one_apply, Pi.sub_apply, one_mul]
  rw [Finset.sum_comm]
  have h₁ : ∀ k, ∑ n ∈ Finset.range N, (Set.Ico (lowerCrossingTime a b f N k ω)
      (upperCrossingTime a b f N (k + 1) ω)).indicator (fun m => f (m + 1) ω - f m ω) n =
      stoppedValue f (upperCrossingTime a b f N (k + 1)) ω -
        stoppedValue f (lowerCrossingTime a b f N k) ω := by
    intro k
    rw [Finset.sum_indicator_eq_sum_filter, (_ : Finset.filter (fun i => i ∈ Set.Ico
      (lowerCrossingTime a b f N k ω) (upperCrossingTime a b f N (k + 1) ω)) (Finset.range N) =
      Finset.Ico (lowerCrossingTime a b f N k ω) (upperCrossingTime a b f N (k + 1) ω)),
      Finset.sum_Ico_eq_add_neg _ lowerCrossingTime_le_upperCrossingTime_succ,
      Finset.sum_range_sub fun n => f n ω, Finset.sum_range_sub fun n => f n ω, neg_sub,
      sub_add_sub_cancel]
    · rfl
    · ext i
      simp only [Set.mem_Ico, Finset.mem_filter, Finset.mem_range, Finset.mem_Ico,
        and_iff_right_iff_imp, and_imp]
      exact fun _ h => lt_of_lt_of_le h upperCrossingTime_le
  simp_rw [h₁]
  have h₂ : ∑ _k ∈ Finset.range (upcrossingsBefore a b f N ω), (b - a) ≤
      ∑ k ∈ Finset.range N, (stoppedValue f (upperCrossingTime a b f N (k + 1)) ω -
        stoppedValue f (lowerCrossingTime a b f N k) ω) := by
    calc
      ∑ _k ∈ Finset.range (upcrossingsBefore a b f N ω), (b - a) ≤
          ∑ k ∈ Finset.range (upcrossingsBefore a b f N ω),
            (stoppedValue f (upperCrossingTime a b f N (k + 1)) ω -
              stoppedValue f (lowerCrossingTime a b f N k) ω) := by
        refine Finset.sum_le_sum fun i hi =>
          le_sub_of_le_upcrossingsBefore (zero_lt_iff.2 hN) hab ?_
        rwa [Finset.mem_range] at hi
      _ ≤ ∑ k ∈ Finset.range N, (stoppedValue f (upperCrossingTime a b f N (k + 1)) ω -
          stoppedValue f (lowerCrossingTime a b f N k) ω) := by
        refine Finset.sum_le_sum_of_subset_of_nonneg
          (Finset.range_subset.2 (upcrossingsBefore_le f ω hab)) fun i _ hi => ?_
        by_cases hi' : i = upcrossingsBefore a b f N ω
        · subst hi'
          simp only [stoppedValue]
          rw [upperCrossingTime_eq_of_upcrossingsBefore_lt hab (Nat.lt_succ_self _)]
          by_cases heq : lowerCrossingTime a b f N (upcrossingsBefore a b f N ω) ω = N
          · rw [heq, sub_self]
          · rw [sub_nonneg]
            exact le_trans (stoppedValue_lowerCrossingTime heq) hf
        · rw [sub_eq_zero_of_upcrossingsBefore_lt hab]
          rw [Finset.mem_range, not_lt] at hi
          exact lt_of_le_of_ne hi (Ne.symm hi')
  refine le_trans ?_ h₂
  rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_comm]


theorem integral_mul_upcrossingsBefore_le_integral [IsFiniteMeasure μ] (hf : Submartingale f ℱ μ)
    (hfN : ∀ ω, a ≤ f N ω) (hfzero : 0 ≤ f 0) (hab : a < b) :
    (b - a) * μ[upcrossingsBefore a b f N] ≤ μ[f N] :=
  calc
    (b - a) * μ[upcrossingsBefore a b f N] ≤
        μ[∑ k ∈ Finset.range N, upcrossingStrat a b f N k * (f (k + 1) - f k)] := by
      /-
        Ω : Type u_1
        m0 : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        a b : Real
        f : Nat → Ω → Real
        N : Nat
        ℱ : MeasureTheory.Filtration Nat m0
        inst✝ : MeasureTheory.IsFiniteMeasure μ
        hf : MeasureTheory.Submartingale f ℱ μ
        hfN : ∀ (ω : Ω), LE.le a (f N ω)
        hfzero : LE.le 0 (f 0)
        hab : LT.lt a b
        ⊢ LE.le (HMul.hMul (HSub.hSub b a) (MeasureTheory.integral μ fun x => ↑(Measur …
      -/
      rw [← integral_mul_left]
      /-
        Ω : Type u_1
        m0 : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        a b : Real
        f : Nat → Ω → Real
        N : Nat
        ℱ : MeasureTheory.Filtration Nat m0
        inst✝ : MeasureTheory.IsFiniteMeasure μ
        hf : MeasureTheory.Submartingale f ℱ μ
        hfN : ∀ (ω : Ω), LE.le a (f N ω)
        hfzero : LE.le 0 (f 0)
        hab : LT.lt a b
        ⊢ LE.le (MeasureTheory.integral μ fun a_1 => HMul.hMul (HSub.hSub b a) ↑(Measu …
      -/
      refine integral_mono_of_nonneg ?_ ((hf.sum_upcrossingStrat_mul a b N).integrable N) ?_
        /-
          case refine_1
          Ω : Type u_1
          m0 : MeasurableSpace Ω
          μ : MeasureTheory.Measure Ω
          a b : Real
          f : Nat → Ω → Real
          N : Nat
          ℱ : MeasureTheory.Filtration Nat m0
          inst✝ : MeasureTheory.IsFiniteMeasure μ
          hf : MeasureTheory.Submartingale f ℱ μ
          hfN : ∀ (ω : Ω), LE.le a (f N ω)
          hfzero : LE.le 0 (f 0)
          hab : LT.lt a b
          ⊢ (MeasureTheory.ae μ).EventuallyLE 0 fun a_1 => HMul.hMul (HSub.hSub b a) ↑(M …
        -/
      · exact Eventually.of_forall fun ω => mul_nonneg (sub_nonneg.2 hab.le) (Nat.cast_nonneg _)
        /-
          🎉 no goals
        -/
        /-
          case refine_2
          Ω : Type u_1
          m0 : MeasurableSpace Ω
          μ : MeasureTheory.Measure Ω
          a b : Real
          f : Nat → Ω → Real
          N : Nat
          ℱ : MeasureTheory.Filtration Nat m0
          inst✝ : MeasureTheory.IsFiniteMeasure μ
          hf : MeasureTheory.Submartingale f ℱ μ
          hfN : ∀ (ω : Ω), LE.le a (f N ω)
          hfzero : LE.le 0 (f 0)
          hab : LT.lt a b
          ⊢ (MeasureTheory.ae μ).EventuallyLE (fun a_1 => HMul.hMul (HSub.hSub b a) ↑(Me …
        -/
      · filter_upwards with ω
        /-
          case refine_2.h
          Ω : Type u_1
          m0 : MeasurableSpace Ω
          μ : MeasureTheory.Measure Ω
          a b : Real
          f : Nat → Ω → Real
          N : Nat
          ℱ : MeasureTheory.Filtration Nat m0
          inst✝ : MeasureTheory.IsFiniteMeasure μ
          hf : MeasureTheory.Submartingale f ℱ μ
          hfN : ∀ (ω : Ω), LE.le a (f N ω)
          hfzero : LE.le 0 (f 0)
          hab : LT.lt a b
          ω : Ω
          ⊢ LE.le (HMul.hMul (HSub.hSub b a) ↑(MeasureTheory.upcrossingsBefore a b f N ω …
        -/
        simpa using mul_upcrossingsBefore_le (hfN ω) hab
        /-
          🎉 no goals
        -/
    _ ≤ μ[f N] - μ[f 0] := hf.sum_mul_upcrossingStrat_le
    _ ≤ μ[f N] := (sub_le_self_iff _).2 (integral_nonneg hfzero)


theorem crossing_pos_eq (hab : a < b) :
    upperCrossingTime 0 (b - a) (fun n ω => (f n ω - a)⁺) N n = upperCrossingTime a b f N n ∧
      lowerCrossingTime 0 (b - a) (fun n ω => (f n ω - a)⁺) N n = lowerCrossingTime a b f N n := by
  /-
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N n : Nat
    hab : LT.lt a b
    ⊢ And (Eq (MeasureTheory.upperCrossingTime 0 (HSub.hSub b a) (fun n ω => PosPa …
  -/
  have hab' : 0 < b - a := sub_pos.2 hab
  have hf : ∀ ω i, b - a ≤ (f i ω - a)⁺ ↔ b ≤ f i ω := by
    intro i ω
    refine ⟨fun h => ?_, fun h => ?_⟩
    · rwa [← sub_le_sub_iff_right a, ←
        posPart_eq_of_posPart_pos (lt_of_lt_of_le hab' h)]
    · rw [← sub_le_sub_iff_right a] at h
      rwa [posPart_eq_self.2 (le_trans hab'.le h)]
  /-
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N n : Nat
    hab : LT.lt a b
    hab' : LT.lt 0 (HSub.hSub b a)
    hf : ∀ (ω : Ω) (i : Nat), Iff (LE.le (HSub.hSub b a) (PosPart.posPart (HSub.hS …
    ⊢ And (Eq (MeasureTheory.upperCrossingTime 0 (HSub.hSub b a) (fun n ω => PosPa …
  -/
  have hf' (ω i) : (f i ω - a)⁺ ≤ 0 ↔ f i ω ≤ a := by rw [posPart_nonpos, sub_nonpos]
  /-
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N n : Nat
    hab : LT.lt a b
    hab' : LT.lt 0 (HSub.hSub b a)
    hf : ∀ (ω : Ω) (i : Nat), Iff (LE.le (HSub.hSub b a) (PosPart.posPart (HSub.hS …
    hf' : ∀ (ω : Ω) (i : Nat), Iff (LE.le (PosPart.posPart (HSub.hSub (f i ω) a))  …
    ⊢ And (Eq (MeasureTheory.upperCrossingTime 0 (HSub.hSub b a) (fun n ω => PosPa …
  -/
  induction' n with k ih
    /-
      case zero
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      N n : Nat
      hab : LT.lt a b
      hab' : LT.lt 0 (HSub.hSub b a)
      hf : ∀ (ω : Ω) (i : Nat), Iff (LE.le (HSub.hSub b a) (PosPart.posPart (HSub.hS …
      hf' : ∀ (ω : Ω) (i : Nat), Iff (LE.le (PosPart.posPart (HSub.hSub (f i ω) a))  …
      ⊢ And (Eq (MeasureTheory.upperCrossingTime 0 (HSub.hSub b a) (fun n ω => PosPa …
    -/
  · refine ⟨rfl, ?_⟩
    simp (config := { unfoldPartialApp := true }) only [lowerCrossingTime_zero, hitting,
      Set.mem_Icc, Set.mem_Iic]
    /-
      case zero
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      N n : Nat
      hab : LT.lt a b
      hab' : LT.lt 0 (HSub.hSub b a)
      hf : ∀ (ω : Ω) (i : Nat), Iff (LE.le (HSub.hSub b a) (PosPart.posPart (HSub.hS …
      hf' : ∀ (ω : Ω) (i : Nat), Iff (LE.le (PosPart.posPart (HSub.hSub (f i ω) a))  …
      ⊢ Eq (fun x => ite (Exists fun j => And (Membership.mem (Set.Icc Bot.bot N) j) …
    -/
    ext ω
    /-
      case zero.h
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      N n : Nat
      hab : LT.lt a b
      hab' : LT.lt 0 (HSub.hSub b a)
      hf : ∀ (ω : Ω) (i : Nat), Iff (LE.le (HSub.hSub b a) (PosPart.posPart (HSub.hS …
      hf' : ∀ (ω : Ω) (i : Nat), Iff (LE.le (PosPart.posPart (HSub.hSub (f i ω) a))  …
      ω : Ω
      ⊢ Eq (ite (Exists fun j => And (Membership.mem (Set.Icc Bot.bot N) j) (Members …
    -/
    split_ifs with h₁ h₂ h₂
      /-
        case pos
        Ω : Type u_1
        a b : Real
        f : Nat → Ω → Real
        N n : Nat
        hab : LT.lt a b
        hab' : LT.lt 0 (HSub.hSub b a)
        hf : ∀ (ω : Ω) (i : Nat), Iff (LE.le (HSub.hSub b a) (PosPart.posPart (HSub.hS …
        hf' : ∀ (ω : Ω) (i : Nat), Iff (LE.le (PosPart.posPart (HSub.hSub (f i ω) a))  …
        ω : Ω
        h₁ : Exists fun j => And (Membership.mem (Set.Icc Bot.bot N) j) (Membership.me …
        h₂ : Exists fun j => And (Membership.mem (Set.Icc Bot.bot N) j) (Membership.me …
        ⊢ Eq (InfSet.sInf (Inter.inter (Set.Icc Bot.bot N) (setOf fun i => LE.le (PosP …
      -/
    · simp_rw [hf']
      /-
        🎉 no goals
      -/
      /-
        case neg
        Ω : Type u_1
        a b : Real
        f : Nat → Ω → Real
        N n : Nat
        hab : LT.lt a b
        hab' : LT.lt 0 (HSub.hSub b a)
        hf : ∀ (ω : Ω) (i : Nat), Iff (LE.le (HSub.hSub b a) (PosPart.posPart (HSub.hS …
        hf' : ∀ (ω : Ω) (i : Nat), Iff (LE.le (PosPart.posPart (HSub.hSub (f i ω) a))  …
        ω : Ω
        h₁ : Exists fun j => And (Membership.mem (Set.Icc Bot.bot N) j) (Membership.me …
        h₂ : Not (Exists fun j => And (Membership.mem (Set.Icc Bot.bot N) j) (Membersh …
        ⊢ Eq (InfSet.sInf (Inter.inter (Set.Icc Bot.bot N) (setOf fun i => LE.le (PosP …
      -/
    · simp_rw [Set.mem_Iic, ← hf' _ _] at h₂
      /-
        case neg
        Ω : Type u_1
        a b : Real
        f : Nat → Ω → Real
        N n : Nat
        hab : LT.lt a b
        hab' : LT.lt 0 (HSub.hSub b a)
        hf : ∀ (ω : Ω) (i : Nat), Iff (LE.le (HSub.hSub b a) (PosPart.posPart (HSub.hS …
        hf' : ∀ (ω : Ω) (i : Nat), Iff (LE.le (PosPart.posPart (HSub.hSub (f i ω) a))  …
        ω : Ω
        h₁ : Exists fun j => And (Membership.mem (Set.Icc Bot.bot N) j) (Membership.me …
        h₂ : Not (Exists fun j => And (Membership.mem (Set.Icc Bot.bot N) j) (LE.le (P …
        ⊢ Eq (InfSet.sInf (Inter.inter (Set.Icc Bot.bot N) (setOf fun i => LE.le (PosP …
      -/
      exact False.elim (h₂ h₁)
      /-
        🎉 no goals
      -/
      /-
        case pos
        Ω : Type u_1
        a b : Real
        f : Nat → Ω → Real
        N n : Nat
        hab : LT.lt a b
        hab' : LT.lt 0 (HSub.hSub b a)
        hf : ∀ (ω : Ω) (i : Nat), Iff (LE.le (HSub.hSub b a) (PosPart.posPart (HSub.hS …
        hf' : ∀ (ω : Ω) (i : Nat), Iff (LE.le (PosPart.posPart (HSub.hSub (f i ω) a))  …
        ω : Ω
        h₁ : Not (Exists fun j => And (Membership.mem (Set.Icc Bot.bot N) j) (Membersh …
        h₂ : Exists fun j => And (Membership.mem (Set.Icc Bot.bot N) j) (Membership.me …
        ⊢ Eq N (InfSet.sInf (Inter.inter (Set.Icc Bot.bot N) (setOf fun i => LE.le (f  …
      -/
    · simp_rw [Set.mem_Iic, hf' _ _] at h₁
      /-
        case pos
        Ω : Type u_1
        a b : Real
        f : Nat → Ω → Real
        N n : Nat
        hab : LT.lt a b
        hab' : LT.lt 0 (HSub.hSub b a)
        hf : ∀ (ω : Ω) (i : Nat), Iff (LE.le (HSub.hSub b a) (PosPart.posPart (HSub.hS …
        hf' : ∀ (ω : Ω) (i : Nat), Iff (LE.le (PosPart.posPart (HSub.hSub (f i ω) a))  …
        ω : Ω
        h₂ : Exists fun j => And (Membership.mem (Set.Icc Bot.bot N) j) (Membership.me …
        h₁ : Not (Exists fun j => And (Membership.mem (Set.Icc Bot.bot N) j) (LE.le (f …
        ⊢ Eq N (InfSet.sInf (Inter.inter (Set.Icc Bot.bot N) (setOf fun i => LE.le (f  …
      -/
      exact False.elim (h₁ h₂)
      /-
        🎉 no goals
      -/
      /-
        case neg
        Ω : Type u_1
        a b : Real
        f : Nat → Ω → Real
        N n : Nat
        hab : LT.lt a b
        hab' : LT.lt 0 (HSub.hSub b a)
        hf : ∀ (ω : Ω) (i : Nat), Iff (LE.le (HSub.hSub b a) (PosPart.posPart (HSub.hS …
        hf' : ∀ (ω : Ω) (i : Nat), Iff (LE.le (PosPart.posPart (HSub.hSub (f i ω) a))  …
        ω : Ω
        h₁ : Not (Exists fun j => And (Membership.mem (Set.Icc Bot.bot N) j) (Membersh …
        h₂ : Not (Exists fun j => And (Membership.mem (Set.Icc Bot.bot N) j) (Membersh …
        ⊢ Eq N N
      -/
    · rfl
      /-
        🎉 no goals
      -/
  · have : upperCrossingTime 0 (b - a) (fun n ω => (f n ω - a)⁺) N (k + 1) =
        upperCrossingTime a b f N (k + 1) := by
      ext ω
      simp only [upperCrossingTime_succ_eq, ← ih.2, hitting, Set.mem_Ici, tsub_le_iff_right]
      split_ifs with h₁ h₂ h₂
      · simp_rw [← sub_le_iff_le_add, hf ω]
      · refine False.elim (h₂ ?_)
        simp_all only [Set.mem_Ici, not_true_eq_false]
      · refine False.elim (h₁ ?_)
        simp_all only [Set.mem_Ici]
      · rfl
    /-
      case succ
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      N n : Nat
      hab : LT.lt a b
      hab' : LT.lt 0 (HSub.hSub b a)
      hf : ∀ (ω : Ω) (i : Nat), Iff (LE.le (HSub.hSub b a) (PosPart.posPart (HSub.hS …
      hf' : ∀ (ω : Ω) (i : Nat), Iff (LE.le (PosPart.posPart (HSub.hSub (f i ω) a))  …
      k : Nat
      ih : And (Eq (MeasureTheory.upperCrossingTime 0 (HSub.hSub b a) (fun n ω => Po …
      this : Eq (MeasureTheory.upperCrossingTime 0 (HSub.hSub b a) (fun n ω => PosPa …
      ⊢ And (Eq (MeasureTheory.upperCrossingTime 0 (HSub.hSub b a) (fun n ω => PosPa …
    -/
    refine ⟨this, ?_⟩
    /-
      case succ
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      N n : Nat
      hab : LT.lt a b
      hab' : LT.lt 0 (HSub.hSub b a)
      hf : ∀ (ω : Ω) (i : Nat), Iff (LE.le (HSub.hSub b a) (PosPart.posPart (HSub.hS …
      hf' : ∀ (ω : Ω) (i : Nat), Iff (LE.le (PosPart.posPart (HSub.hSub (f i ω) a))  …
      k : Nat
      ih : And (Eq (MeasureTheory.upperCrossingTime 0 (HSub.hSub b a) (fun n ω => Po …
      this : Eq (MeasureTheory.upperCrossingTime 0 (HSub.hSub b a) (fun n ω => PosPa …
      ⊢ Eq (MeasureTheory.lowerCrossingTime 0 (HSub.hSub b a) (fun n ω => PosPart.po …
    -/
    ext ω
    /-
      case succ.h
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      N n : Nat
      hab : LT.lt a b
      hab' : LT.lt 0 (HSub.hSub b a)
      hf : ∀ (ω : Ω) (i : Nat), Iff (LE.le (HSub.hSub b a) (PosPart.posPart (HSub.hS …
      hf' : ∀ (ω : Ω) (i : Nat), Iff (LE.le (PosPart.posPart (HSub.hSub (f i ω) a))  …
      k : Nat
      ih : And (Eq (MeasureTheory.upperCrossingTime 0 (HSub.hSub b a) (fun n ω => Po …
      this : Eq (MeasureTheory.upperCrossingTime 0 (HSub.hSub b a) (fun n ω => PosPa …
      ω : Ω
      ⊢ Eq (MeasureTheory.lowerCrossingTime 0 (HSub.hSub b a) (fun n ω => PosPart.po …
    -/
    simp only [lowerCrossingTime, this, hitting, Set.mem_Iic]
    /-
      case succ.h
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      N n : Nat
      hab : LT.lt a b
      hab' : LT.lt 0 (HSub.hSub b a)
      hf : ∀ (ω : Ω) (i : Nat), Iff (LE.le (HSub.hSub b a) (PosPart.posPart (HSub.hS …
      hf' : ∀ (ω : Ω) (i : Nat), Iff (LE.le (PosPart.posPart (HSub.hSub (f i ω) a))  …
      k : Nat
      ih : And (Eq (MeasureTheory.upperCrossingTime 0 (HSub.hSub b a) (fun n ω => Po …
      this : Eq (MeasureTheory.upperCrossingTime 0 (HSub.hSub b a) (fun n ω => PosPa …
      ω : Ω
      ⊢ Eq (ite (Exists fun j => And (Membership.mem (Set.Icc (MeasureTheory.upperCr …
    -/
    split_ifs with h₁ h₂ h₂
      /-
        case pos
        Ω : Type u_1
        a b : Real
        f : Nat → Ω → Real
        N n : Nat
        hab : LT.lt a b
        hab' : LT.lt 0 (HSub.hSub b a)
        hf : ∀ (ω : Ω) (i : Nat), Iff (LE.le (HSub.hSub b a) (PosPart.posPart (HSub.hS …
        hf' : ∀ (ω : Ω) (i : Nat), Iff (LE.le (PosPart.posPart (HSub.hSub (f i ω) a))  …
        k : Nat
        ih : And (Eq (MeasureTheory.upperCrossingTime 0 (HSub.hSub b a) (fun n ω => Po …
        this : Eq (MeasureTheory.upperCrossingTime 0 (HSub.hSub b a) (fun n ω => PosPa …
        ω : Ω
        h₁ : Exists fun j => And (Membership.mem (Set.Icc (MeasureTheory.upperCrossing …
        h₂ : Exists fun j => And (Membership.mem (Set.Icc (MeasureTheory.upperCrossing …
        ⊢ Eq (InfSet.sInf (Inter.inter (Set.Icc (MeasureTheory.upperCrossingTime a b f …
      -/
    · simp_rw [hf' ω]
      /-
        🎉 no goals
      -/
      /-
        case neg
        Ω : Type u_1
        a b : Real
        f : Nat → Ω → Real
        N n : Nat
        hab : LT.lt a b
        hab' : LT.lt 0 (HSub.hSub b a)
        hf : ∀ (ω : Ω) (i : Nat), Iff (LE.le (HSub.hSub b a) (PosPart.posPart (HSub.hS …
        hf' : ∀ (ω : Ω) (i : Nat), Iff (LE.le (PosPart.posPart (HSub.hSub (f i ω) a))  …
        k : Nat
        ih : And (Eq (MeasureTheory.upperCrossingTime 0 (HSub.hSub b a) (fun n ω => Po …
        this : Eq (MeasureTheory.upperCrossingTime 0 (HSub.hSub b a) (fun n ω => PosPa …
        ω : Ω
        h₁ : Exists fun j => And (Membership.mem (Set.Icc (MeasureTheory.upperCrossing …
        h₂ : Not (Exists fun j => And (Membership.mem (Set.Icc (MeasureTheory.upperCro …
        ⊢ Eq (InfSet.sInf (Inter.inter (Set.Icc (MeasureTheory.upperCrossingTime a b f …
      -/
    · refine False.elim (h₂ ?_)
      /-
        case neg
        Ω : Type u_1
        a b : Real
        f : Nat → Ω → Real
        N n : Nat
        hab : LT.lt a b
        hab' : LT.lt 0 (HSub.hSub b a)
        hf : ∀ (ω : Ω) (i : Nat), Iff (LE.le (HSub.hSub b a) (PosPart.posPart (HSub.hS …
        hf' : ∀ (ω : Ω) (i : Nat), Iff (LE.le (PosPart.posPart (HSub.hSub (f i ω) a))  …
        k : Nat
        ih : And (Eq (MeasureTheory.upperCrossingTime 0 (HSub.hSub b a) (fun n ω => Po …
        this : Eq (MeasureTheory.upperCrossingTime 0 (HSub.hSub b a) (fun n ω => PosPa …
        ω : Ω
        h₁ : Exists fun j => And (Membership.mem (Set.Icc (MeasureTheory.upperCrossing …
        h₂ : Not (Exists fun j => And (Membership.mem (Set.Icc (MeasureTheory.upperCro …
        ⊢ Exists fun j => And (Membership.mem (Set.Icc (MeasureTheory.upperCrossingTim …
      -/
      simp_all only [Set.mem_Iic, not_true_eq_false]
      /-
        🎉 no goals
      -/
      /-
        case pos
        Ω : Type u_1
        a b : Real
        f : Nat → Ω → Real
        N n : Nat
        hab : LT.lt a b
        hab' : LT.lt 0 (HSub.hSub b a)
        hf : ∀ (ω : Ω) (i : Nat), Iff (LE.le (HSub.hSub b a) (PosPart.posPart (HSub.hS …
        hf' : ∀ (ω : Ω) (i : Nat), Iff (LE.le (PosPart.posPart (HSub.hSub (f i ω) a))  …
        k : Nat
        ih : And (Eq (MeasureTheory.upperCrossingTime 0 (HSub.hSub b a) (fun n ω => Po …
        this : Eq (MeasureTheory.upperCrossingTime 0 (HSub.hSub b a) (fun n ω => PosPa …
        ω : Ω
        h₁ : Not (Exists fun j => And (Membership.mem (Set.Icc (MeasureTheory.upperCro …
        h₂ : Exists fun j => And (Membership.mem (Set.Icc (MeasureTheory.upperCrossing …
        ⊢ Eq N (InfSet.sInf (Inter.inter (Set.Icc (MeasureTheory.upperCrossingTime a b …
      -/
    · refine False.elim (h₁ ?_)
      /-
        case pos
        Ω : Type u_1
        a b : Real
        f : Nat → Ω → Real
        N n : Nat
        hab : LT.lt a b
        hab' : LT.lt 0 (HSub.hSub b a)
        hf : ∀ (ω : Ω) (i : Nat), Iff (LE.le (HSub.hSub b a) (PosPart.posPart (HSub.hS …
        hf' : ∀ (ω : Ω) (i : Nat), Iff (LE.le (PosPart.posPart (HSub.hSub (f i ω) a))  …
        k : Nat
        ih : And (Eq (MeasureTheory.upperCrossingTime 0 (HSub.hSub b a) (fun n ω => Po …
        this : Eq (MeasureTheory.upperCrossingTime 0 (HSub.hSub b a) (fun n ω => PosPa …
        ω : Ω
        h₁ : Not (Exists fun j => And (Membership.mem (Set.Icc (MeasureTheory.upperCro …
        h₂ : Exists fun j => And (Membership.mem (Set.Icc (MeasureTheory.upperCrossing …
        ⊢ Exists fun j => And (Membership.mem (Set.Icc (MeasureTheory.upperCrossingTim …
      -/
      simp_all only [Set.mem_Iic]
      /-
        🎉 no goals
      -/
      /-
        case neg
        Ω : Type u_1
        a b : Real
        f : Nat → Ω → Real
        N n : Nat
        hab : LT.lt a b
        hab' : LT.lt 0 (HSub.hSub b a)
        hf : ∀ (ω : Ω) (i : Nat), Iff (LE.le (HSub.hSub b a) (PosPart.posPart (HSub.hS …
        hf' : ∀ (ω : Ω) (i : Nat), Iff (LE.le (PosPart.posPart (HSub.hSub (f i ω) a))  …
        k : Nat
        ih : And (Eq (MeasureTheory.upperCrossingTime 0 (HSub.hSub b a) (fun n ω => Po …
        this : Eq (MeasureTheory.upperCrossingTime 0 (HSub.hSub b a) (fun n ω => PosPa …
        ω : Ω
        h₁ : Not (Exists fun j => And (Membership.mem (Set.Icc (MeasureTheory.upperCro …
        h₂ : Not (Exists fun j => And (Membership.mem (Set.Icc (MeasureTheory.upperCro …
        ⊢ Eq N N
      -/
    · rfl
      /-
        🎉 no goals
      -/


theorem upcrossingsBefore_pos_eq (hab : a < b) :
    upcrossingsBefore 0 (b - a) (fun n ω => (f n ω - a)⁺) N ω = upcrossingsBefore a b f N ω := by
  /-
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N : Nat
    ω : Ω
    hab : LT.lt a b
    ⊢ Eq (MeasureTheory.upcrossingsBefore 0 (HSub.hSub b a) (fun n ω => PosPart.po …
  -/
  simp_rw [upcrossingsBefore, (crossing_pos_eq hab).1]
  /-
    🎉 no goals
  -/


theorem mul_integral_upcrossingsBefore_le_integral_pos_part_aux [IsFiniteMeasure μ]
    (hf : Submartingale f ℱ μ) (hab : a < b) :
    (b - a) * μ[upcrossingsBefore a b f N] ≤ μ[fun ω => (f N ω - a)⁺] := by
  refine le_trans (le_of_eq ?_)
    (integral_mul_upcrossingsBefore_le_integral (hf.sub_martingale (martingale_const _ _ _)).pos
      (fun ω => posPart_nonneg _)
      (fun ω => posPart_nonneg _) (sub_pos.2 hab))
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    a b : Real
    f : Nat → Ω → Real
    N : Nat
    ℱ : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : MeasureTheory.Submartingale f ℱ μ
    hab : LT.lt a b
    ⊢ Eq (HMul.hMul (HSub.hSub b a) (MeasureTheory.integral μ fun x => ↑(MeasureTh …
  -/
  simp_rw [sub_zero, ← upcrossingsBefore_pos_eq hab]
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    a b : Real
    f : Nat → Ω → Real
    N : Nat
    ℱ : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : MeasureTheory.Submartingale f ℱ μ
    hab : LT.lt a b
    ⊢ Eq (HMul.hMul (HSub.hSub b a) (MeasureTheory.integral μ fun x => ↑(MeasureTh …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- **Doob's upcrossing estimate**: given a real valued discrete submartingale `f` and real
values `a` and `b`, we have `(b - a) * 𝔼[upcrossingsBefore a b f N] ≤ 𝔼[(f N - a)⁺]` where
`upcrossingsBefore a b f N` is the number of times the process `f` crossed from below `a` to above
`b` before the time `N`. -/
theorem Submartingale.mul_integral_upcrossingsBefore_le_integral_pos_part [IsFiniteMeasure μ]
    (a b : ℝ) (hf : Submartingale f ℱ μ) (N : ℕ) :
    (b - a) * μ[upcrossingsBefore a b f N] ≤ μ[fun ω => (f N ω - a)⁺] := by
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    f : Nat → Ω → Real
    ℱ : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    a b : Real
    hf : MeasureTheory.Submartingale f ℱ μ
    N : Nat
    ⊢ LE.le (HMul.hMul (HSub.hSub b a) (MeasureTheory.integral μ fun x => ↑(Measur …
  -/
  by_cases hab : a < b
    /-
      case pos
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      f : Nat → Ω → Real
      ℱ : MeasureTheory.Filtration Nat m0
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      a b : Real
      hf : MeasureTheory.Submartingale f ℱ μ
      N : Nat
      hab : LT.lt a b
      ⊢ LE.le (HMul.hMul (HSub.hSub b a) (MeasureTheory.integral μ fun x => ↑(Measur …
    -/
  · exact mul_integral_upcrossingsBefore_le_integral_pos_part_aux hf hab
    /-
      🎉 no goals
    -/
    /-
      case neg
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      f : Nat → Ω → Real
      ℱ : MeasureTheory.Filtration Nat m0
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      a b : Real
      hf : MeasureTheory.Submartingale f ℱ μ
      N : Nat
      hab : Not (LT.lt a b)
      ⊢ LE.le (HMul.hMul (HSub.hSub b a) (MeasureTheory.integral μ fun x => ↑(Measur …
    -/
  · rw [not_lt, ← sub_nonpos] at hab
    exact le_trans (mul_nonpos_of_nonpos_of_nonneg hab (by positivity))
      (integral_nonneg fun ω => posPart_nonneg _)


theorem upcrossingsBefore_eq_sum (hab : a < b) : upcrossingsBefore a b f N ω =
    ∑ i ∈ Finset.Ico 1 (N + 1), {n | upperCrossingTime a b f N n ω < N}.indicator 1 i := by
  /-
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    N : Nat
    ω : Ω
    hab : LT.lt a b
    ⊢ Eq (MeasureTheory.upcrossingsBefore a b f N ω) ((Finset.Ico 1 (HAdd.hAdd N 1 …
  -/
  by_cases hN : N = 0
    /-
      case pos
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      N : Nat
      ω : Ω
      hab : LT.lt a b
      hN : Eq N 0
      ⊢ Eq (MeasureTheory.upcrossingsBefore a b f N ω) ((Finset.Ico 1 (HAdd.hAdd N 1 …
    -/
  · simp [hN]
    /-
      🎉 no goals
    -/
  rw [← Finset.sum_Ico_consecutive _ (Nat.succ_le_succ zero_le')
    (Nat.succ_le_succ (upcrossingsBefore_le f ω hab))]
  have h₁ : ∀ k ∈ Finset.Ico 1 (upcrossingsBefore a b f N ω + 1),
      {n : ℕ | upperCrossingTime a b f N n ω < N}.indicator 1 k = 1 := by
    rintro k hk
    rw [Finset.mem_Ico] at hk
    rw [Set.indicator_of_mem]
    · rfl
    · exact upperCrossingTime_lt_of_le_upcrossingsBefore (zero_lt_iff.2 hN) hab
        (Nat.lt_succ_iff.1 hk.2)
  have h₂ : ∀ k ∈ Finset.Ico (upcrossingsBefore a b f N ω + 1) (N + 1),
      {n : ℕ | upperCrossingTime a b f N n ω < N}.indicator 1 k = 0 := by
    rintro k hk
    rw [Finset.mem_Ico, Nat.succ_le_iff] at hk
    rw [Set.indicator_of_not_mem]
    simp only [Set.mem_setOf_eq, not_lt]
    exact (upperCrossingTime_eq_of_upcrossingsBefore_lt hab hk.1).symm.le
  rw [Finset.sum_congr rfl h₁, Finset.sum_congr rfl h₂, Finset.sum_const, Finset.sum_const,
    smul_eq_mul, mul_one, smul_eq_mul, mul_zero, Nat.card_Ico, Nat.add_succ_sub_one,
    add_zero, add_zero]


theorem Adapted.measurable_upcrossingsBefore (hf : Adapted ℱ f) (hab : a < b) :
    Measurable (upcrossingsBefore a b f N) := by
  have : upcrossingsBefore a b f N = fun ω =>
      ∑ i ∈ Finset.Ico 1 (N + 1), {n | upperCrossingTime a b f N n ω < N}.indicator 1 i := by
    ext ω
    exact upcrossingsBefore_eq_sum hab
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    a b : Real
    f : Nat → Ω → Real
    N : Nat
    ℱ : MeasureTheory.Filtration Nat m0
    hf : MeasureTheory.Adapted ℱ f
    hab : LT.lt a b
    this : Eq (MeasureTheory.upcrossingsBefore a b f N) fun ω => (Finset.Ico 1 (HA …
    ⊢ Measurable (MeasureTheory.upcrossingsBefore a b f N)
  -/
  rw [this]
  exact Finset.measurable_sum _ fun i _ => Measurable.indicator measurable_const <|
    ℱ.le N _ (hf.isStoppingTime_upperCrossingTime.measurableSet_lt_of_pred N)


theorem Adapted.integrable_upcrossingsBefore [IsFiniteMeasure μ] (hf : Adapted ℱ f) (hab : a < b) :
    Integrable (fun ω => (upcrossingsBefore a b f N ω : ℝ)) μ :=
  haveI : ∀ᵐ ω ∂μ, ‖(upcrossingsBefore a b f N ω : ℝ)‖ ≤ N := by
    /-
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      a b : Real
      f : Nat → Ω → Real
      N : Nat
      ℱ : MeasureTheory.Filtration Nat m0
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hf : MeasureTheory.Adapted ℱ f
      hab : LT.lt a b
      ⊢ Filter.Eventually (fun ω => LE.le (Norm.norm ↑(MeasureTheory.upcrossingsBefo …
    -/
    filter_upwards with ω
    /-
      case h
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      a b : Real
      f : Nat → Ω → Real
      N : Nat
      ℱ : MeasureTheory.Filtration Nat m0
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hf : MeasureTheory.Adapted ℱ f
      hab : LT.lt a b
      ω : Ω
      ⊢ LE.le (Norm.norm ↑(MeasureTheory.upcrossingsBefore a b f N ω)) ↑N
    -/
    rw [Real.norm_eq_abs, Nat.abs_cast, Nat.cast_le]
    /-
      case h
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      a b : Real
      f : Nat → Ω → Real
      N : Nat
      ℱ : MeasureTheory.Filtration Nat m0
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hf : MeasureTheory.Adapted ℱ f
      hab : LT.lt a b
      ω : Ω
      ⊢ LE.le (MeasureTheory.upcrossingsBefore a b f N ω) N
    -/
    exact upcrossingsBefore_le _ _ hab
    /-
      🎉 no goals
    -/
  ⟨Measurable.aestronglyMeasurable (measurable_from_top.comp (hf.measurable_upcrossingsBefore hab)),
    hasFiniteIntegral_of_bounded this⟩


/-- The number of upcrossings of a realization of a stochastic process (`upcrossings` takes value
in `ℝ≥0∞` and so is allowed to be `∞`). -/
noncomputable def upcrossings [Preorder ι] [OrderBot ι] [InfSet ι] (a b : ℝ) (f : ι → Ω → ℝ)
    (ω : Ω) : ℝ≥0∞ :=
  ⨆ N, (upcrossingsBefore a b f N ω : ℝ≥0∞)


theorem Adapted.measurable_upcrossings (hf : Adapted ℱ f) (hab : a < b) :
    Measurable (upcrossings a b f) :=
  .iSup fun _ => measurable_from_top.comp (hf.measurable_upcrossingsBefore hab)


theorem upcrossings_lt_top_iff :
    upcrossings a b f ω < ∞ ↔ ∃ k, ∀ N, upcrossingsBefore a b f N ω ≤ k := by
  have : upcrossings a b f ω < ∞ ↔ ∃ k : ℝ≥0, upcrossings a b f ω ≤ k := by
    constructor
    · intro h
      lift upcrossings a b f ω to ℝ≥0 using h.ne with r hr
      exact ⟨r, le_rfl⟩
    · rintro ⟨k, hk⟩
      exact lt_of_le_of_lt hk ENNReal.coe_lt_top
  /-
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    ω : Ω
    this : Iff (LT.lt (MeasureTheory.upcrossings a b f ω) Top.top) (Exists fun k = …
    ⊢ Iff (LT.lt (MeasureTheory.upcrossings a b f ω) Top.top) (Exists fun k => ∀ ( …
  -/
  simp_rw [this, upcrossings, iSup_le_iff]
  /-
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    ω : Ω
    this : Iff (LT.lt (MeasureTheory.upcrossings a b f ω) Top.top) (Exists fun k = …
    ⊢ Iff (Exists fun k => ∀ (i : Nat), LE.le ↑(MeasureTheory.upcrossingsBefore a  …
  -/
  constructor <;> rintro ⟨k, hk⟩
    /-
      case mp.intro
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      ω : Ω
      this : Iff (LT.lt (MeasureTheory.upcrossings a b f ω) Top.top) (Exists fun k = …
      k : NNReal
      hk : ∀ (i : Nat), LE.le ↑(MeasureTheory.upcrossingsBefore a b f i ω) ↑k
      ⊢ Exists fun k => ∀ (N : Nat), LE.le (MeasureTheory.upcrossingsBefore a b f N  …
    -/
  · obtain ⟨m, hm⟩ := exists_nat_ge k
    /-
      case mp.intro.intro
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      ω : Ω
      this : Iff (LT.lt (MeasureTheory.upcrossings a b f ω) Top.top) (Exists fun k = …
      k : NNReal
      hk : ∀ (i : Nat), LE.le ↑(MeasureTheory.upcrossingsBefore a b f i ω) ↑k
      m : Nat
      hm : LE.le k ↑m
      ⊢ Exists fun k => ∀ (N : Nat), LE.le (MeasureTheory.upcrossingsBefore a b f N  …
    -/
    refine ⟨m, fun N => Nat.cast_le.1 ((hk N).trans ?_)⟩
    /-
      case mp.intro.intro
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      ω : Ω
      this : Iff (LT.lt (MeasureTheory.upcrossings a b f ω) Top.top) (Exists fun k = …
      k : NNReal
      hk : ∀ (i : Nat), LE.le ↑(MeasureTheory.upcrossingsBefore a b f i ω) ↑k
      m : Nat
      hm : LE.le k ↑m
      N : Nat
      ⊢ LE.le ↑k ↑m
    -/
    rwa [← ENNReal.coe_natCast, ENNReal.coe_le_coe]
    /-
      🎉 no goals
    -/
    /-
      case mpr.intro
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      ω : Ω
      this : Iff (LT.lt (MeasureTheory.upcrossings a b f ω) Top.top) (Exists fun k = …
      k : Nat
      hk : ∀ (N : Nat), LE.le (MeasureTheory.upcrossingsBefore a b f N ω) k
      ⊢ Exists fun k => ∀ (i : Nat), LE.le ↑(MeasureTheory.upcrossingsBefore a b f i …
    -/
  · refine ⟨k, fun N => ?_⟩
    /-
      case mpr.intro
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      ω : Ω
      this : Iff (LT.lt (MeasureTheory.upcrossings a b f ω) Top.top) (Exists fun k = …
      k : Nat
      hk : ∀ (N : Nat), LE.le (MeasureTheory.upcrossingsBefore a b f N ω) k
      N : Nat
      ⊢ LE.le ↑(MeasureTheory.upcrossingsBefore a b f N ω) ↑↑k
    -/
    simp only [ENNReal.coe_natCast, Nat.cast_le, hk N]
    /-
      🎉 no goals
    -/


/-- A variant of Doob's upcrossing estimate obtained by taking the supremum on both sides. -/
theorem Submartingale.mul_lintegral_upcrossings_le_lintegral_pos_part [IsFiniteMeasure μ] (a b : ℝ)
    (hf : Submartingale f ℱ μ) : ENNReal.ofReal (b - a) * ∫⁻ ω, upcrossings a b f ω ∂μ ≤
      ⨆ N, ∫⁻ ω, ENNReal.ofReal ((f N ω - a)⁺) ∂μ := by
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    f : Nat → Ω → Real
    ℱ : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    a b : Real
    hf : MeasureTheory.Submartingale f ℱ μ
    ⊢ LE.le (HMul.hMul (ENNReal.ofReal (HSub.hSub b a)) (MeasureTheory.lintegral μ …
  -/
  by_cases hab : a < b
    /-
      case pos
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      f : Nat → Ω → Real
      ℱ : MeasureTheory.Filtration Nat m0
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      a b : Real
      hf : MeasureTheory.Submartingale f ℱ μ
      hab : LT.lt a b
      ⊢ LE.le (HMul.hMul (ENNReal.ofReal (HSub.hSub b a)) (MeasureTheory.lintegral μ …
    -/
  · simp_rw [upcrossings]
    have : ∀ N, ∫⁻ ω, ENNReal.ofReal ((f N ω - a)⁺) ∂μ = ENNReal.ofReal (∫ ω, (f N ω - a)⁺ ∂μ) := by
      intro N
      rw [ofReal_integral_eq_lintegral_ofReal]
      · exact (hf.sub_martingale (martingale_const _ _ _)).pos.integrable _
      · exact Eventually.of_forall fun ω => posPart_nonneg _
    /-
      case pos
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      f : Nat → Ω → Real
      ℱ : MeasureTheory.Filtration Nat m0
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      a b : Real
      hf : MeasureTheory.Submartingale f ℱ μ
      hab : LT.lt a b
      this : ∀ (N : Nat), Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (Pos …
      ⊢ LE.le (HMul.hMul (ENNReal.ofReal (HSub.hSub b a)) (MeasureTheory.lintegral μ …
    -/
    rw [lintegral_iSup']
      /-
        case pos
        Ω : Type u_1
        m0 : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        f : Nat → Ω → Real
        ℱ : MeasureTheory.Filtration Nat m0
        inst✝ : MeasureTheory.IsFiniteMeasure μ
        a b : Real
        hf : MeasureTheory.Submartingale f ℱ μ
        hab : LT.lt a b
        this : ∀ (N : Nat), Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (Pos …
        ⊢ LE.le (HMul.hMul (ENNReal.ofReal (HSub.hSub b a)) (iSup fun n => MeasureTheo …
      -/
    · simp_rw [this, ENNReal.mul_iSup, iSup_le_iff]
      /-
        case pos
        Ω : Type u_1
        m0 : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        f : Nat → Ω → Real
        ℱ : MeasureTheory.Filtration Nat m0
        inst✝ : MeasureTheory.IsFiniteMeasure μ
        a b : Real
        hf : MeasureTheory.Submartingale f ℱ μ
        hab : LT.lt a b
        this : ∀ (N : Nat), Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (Pos …
        ⊢ ∀ (i : Nat), LE.le (HMul.hMul (ENNReal.ofReal (HSub.hSub b a)) (MeasureTheor …
      -/
      intro N
      rw [(by simp :
          ∫⁻ ω, upcrossingsBefore a b f N ω ∂μ = ∫⁻ ω, ↑(upcrossingsBefore a b f N ω : ℝ≥0) ∂μ),
        lintegral_coe_eq_integral, ← ENNReal.ofReal_mul (sub_pos.2 hab).le]
        /-
          case pos
          Ω : Type u_1
          m0 : MeasurableSpace Ω
          μ : MeasureTheory.Measure Ω
          f : Nat → Ω → Real
          ℱ : MeasureTheory.Filtration Nat m0
          inst✝ : MeasureTheory.IsFiniteMeasure μ
          a b : Real
          hf : MeasureTheory.Submartingale f ℱ μ
          hab : LT.lt a b
          this : ∀ (N : Nat), Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (Pos …
          N : Nat
          ⊢ LE.le (ENNReal.ofReal (HMul.hMul (HSub.hSub b a) (MeasureTheory.integral μ f …
        -/
      · simp_rw [NNReal.coe_natCast]
        exact (ENNReal.ofReal_le_ofReal
          (hf.mul_integral_upcrossingsBefore_le_integral_pos_part a b N)).trans
            (le_iSup (α := ℝ≥0∞) _ N)
        /-
          case pos.hfi
          Ω : Type u_1
          m0 : MeasurableSpace Ω
          μ : MeasureTheory.Measure Ω
          f : Nat → Ω → Real
          ℱ : MeasureTheory.Filtration Nat m0
          inst✝ : MeasureTheory.IsFiniteMeasure μ
          a b : Real
          hf : MeasureTheory.Submartingale f ℱ μ
          hab : LT.lt a b
          this : ∀ (N : Nat), Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (Pos …
          N : Nat
          ⊢ MeasureTheory.Integrable (fun x => ↑↑(MeasureTheory.upcrossingsBefore a b f  …
        -/
      · simp only [NNReal.coe_natCast, hf.adapted.integrable_upcrossingsBefore hab]
        /-
          🎉 no goals
        -/
    · exact fun n => measurable_from_top.comp_aemeasurable
        (hf.adapted.measurable_upcrossingsBefore hab).aemeasurable
      /-
        case pos.h_mono
        Ω : Type u_1
        m0 : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        f : Nat → Ω → Real
        ℱ : MeasureTheory.Filtration Nat m0
        inst✝ : MeasureTheory.IsFiniteMeasure μ
        a b : Real
        hf : MeasureTheory.Submartingale f ℱ μ
        hab : LT.lt a b
        this : ∀ (N : Nat), Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (Pos …
        ⊢ Filter.Eventually (fun x => Monotone fun n => ↑(MeasureTheory.upcrossingsBef …
      -/
    · filter_upwards with ω N M hNM
      /-
        case pos.h_mono.h
        Ω : Type u_1
        m0 : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        f : Nat → Ω → Real
        ℱ : MeasureTheory.Filtration Nat m0
        inst✝ : MeasureTheory.IsFiniteMeasure μ
        a b : Real
        hf : MeasureTheory.Submartingale f ℱ μ
        hab : LT.lt a b
        this : ∀ (N : Nat), Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (Pos …
        ω : Ω
        N M : Nat
        hNM : LE.le N M
        ⊢ LE.le ((fun n => ↑(MeasureTheory.upcrossingsBefore a b f n ω)) N) ((fun n => …
      -/
      rw [Nat.cast_le]
      /-
        case pos.h_mono.h
        Ω : Type u_1
        m0 : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        f : Nat → Ω → Real
        ℱ : MeasureTheory.Filtration Nat m0
        inst✝ : MeasureTheory.IsFiniteMeasure μ
        a b : Real
        hf : MeasureTheory.Submartingale f ℱ μ
        hab : LT.lt a b
        this : ∀ (N : Nat), Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (Pos …
        ω : Ω
        N M : Nat
        hNM : LE.le N M
        ⊢ LE.le (MeasureTheory.upcrossingsBefore a b f N ω) (MeasureTheory.upcrossings …
      -/
      exact upcrossingsBefore_mono hab hNM ω
      /-
        🎉 no goals
      -/
    /-
      case neg
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      f : Nat → Ω → Real
      ℱ : MeasureTheory.Filtration Nat m0
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      a b : Real
      hf : MeasureTheory.Submartingale f ℱ μ
      hab : Not (LT.lt a b)
      ⊢ LE.le (HMul.hMul (ENNReal.ofReal (HSub.hSub b a)) (MeasureTheory.lintegral μ …
    -/
  · rw [not_lt, ← sub_nonpos] at hab
    /-
      case neg
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      f : Nat → Ω → Real
      ℱ : MeasureTheory.Filtration Nat m0
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      a b : Real
      hf : MeasureTheory.Submartingale f ℱ μ
      hab : LE.le (HSub.hSub b a) 0
      ⊢ LE.le (HMul.hMul (ENNReal.ofReal (HSub.hSub b a)) (MeasureTheory.lintegral μ …
    -/
    rw [ENNReal.ofReal_of_nonpos hab, zero_mul]
    /-
      case neg
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      f : Nat → Ω → Real
      ℱ : MeasureTheory.Filtration Nat m0
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      a b : Real
      hf : MeasureTheory.Submartingale f ℱ μ
      hab : LE.le (HSub.hSub b a) 0
      ⊢ LE.le 0 (iSup fun N => MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (Po …
    -/
    exact zero_le _
    /-
      🎉 no goals
    -/


