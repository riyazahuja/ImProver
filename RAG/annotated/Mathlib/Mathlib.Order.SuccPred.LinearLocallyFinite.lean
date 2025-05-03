instance (priority := 100) isPredArchimedean_of_isSuccArchimedean [IsSuccArchimedean ι] :
    IsPredArchimedean ι where
  exists_pred_iterate_of_le {i j} hij := by
    /-
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : PredOrder ι
      inst✝ : IsSuccArchimedean ι
      i j : ι
      hij : LE.le i j
      ⊢ Exists fun n => Eq (Nat.iterate Order.pred n j) i
    -/
    have h_exists := exists_succ_iterate_of_le hij
    obtain ⟨n, hn_eq, hn_lt_ne⟩ : ∃ n, succ^[n] i = j ∧ ∀ m < n, succ^[m] i ≠ j :=
      ⟨Nat.find h_exists, Nat.find_spec h_exists, fun m hmn ↦ Nat.find_min h_exists hmn⟩
    /-
      case intro.intro
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : PredOrder ι
      inst✝ : IsSuccArchimedean ι
      i j : ι
      hij : LE.le i j
      h_exists : Exists fun n => Eq (Nat.iterate Order.succ n i) j
      n : Nat
      hn_eq : Eq (Nat.iterate Order.succ n i) j
      hn_lt_ne : ∀ (m : Nat), LT.lt m n → Ne (Nat.iterate Order.succ m i) j
      ⊢ Exists fun n => Eq (Nat.iterate Order.pred n j) i
    -/
    refine ⟨n, ?_⟩
    /-
      case intro.intro
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : PredOrder ι
      inst✝ : IsSuccArchimedean ι
      i j : ι
      hij : LE.le i j
      h_exists : Exists fun n => Eq (Nat.iterate Order.succ n i) j
      n : Nat
      hn_eq : Eq (Nat.iterate Order.succ n i) j
      hn_lt_ne : ∀ (m : Nat), LT.lt m n → Ne (Nat.iterate Order.succ m i) j
      ⊢ Eq (Nat.iterate Order.pred n j) i
    -/
    rw [← hn_eq]
    cases n with
    | zero => simp only [Function.iterate_zero, id]
    | succ n =>
      rw [pred_succ_iterate_of_not_isMax]
      rw [Nat.succ_sub_succ_eq_sub, tsub_zero]
      suffices succ^[n] i < succ^[n.succ] i from not_isMax_of_lt this
      refine lt_of_le_of_ne ?_ ?_
      · rw [Function.iterate_succ_apply']
        exact le_succ _
      · rw [hn_eq]
        exact hn_lt_ne _ (Nat.lt_succ_self n)


instance isSuccArchimedean_of_isPredArchimedean [IsPredArchimedean ι] : IsSuccArchimedean ι :=
  inferInstanceAs (IsSuccArchimedean ιᵒᵈᵒᵈ)


/-- In a linear `SuccOrder` that's also a `PredOrder`, `IsSuccArchimedean` and `IsPredArchimedean`
are equivalent. -/
theorem isSuccArchimedean_iff_isPredArchimedean : IsSuccArchimedean ι ↔ IsPredArchimedean ι where
  mp _ := isPredArchimedean_of_isSuccArchimedean
  mpr _ := isSuccArchimedean_of_isPredArchimedean


/-- Successor in a linear order. This defines a true successor only when `i` is isolated from above,
i.e. when `i` is not the greatest lower bound of `(i, ∞)`. -/
noncomputable def succFn (i : ι) : ι :=
  (exists_glb_Ioi i).choose


theorem succFn_spec (i : ι) : IsGLB (Set.Ioi i) (succFn i) :=
  (exists_glb_Ioi i).choose_spec


theorem le_succFn (i : ι) : i ≤ succFn i := by
  /-
    ι : Type u_1
    inst✝ : LinearOrder ι
    i : ι
    ⊢ LE.le i (LinearLocallyFiniteOrder.succFn i)
  -/
  rw [le_isGLB_iff (succFn_spec i), mem_lowerBounds]
  /-
    ι : Type u_1
    inst✝ : LinearOrder ι
    i : ι
    ⊢ ∀ (x : ι), Membership.mem (Set.Ioi i) x → LE.le i x
  -/
  exact fun x hx ↦ le_of_lt hx
  /-
    🎉 no goals
  -/


theorem isGLB_Ioc_of_isGLB_Ioi {i j k : ι} (hij_lt : i < j) (h : IsGLB (Set.Ioi i) k) :
    IsGLB (Set.Ioc i j) k := by
  /-
    ι : Type u_1
    inst✝ : LinearOrder ι
    i j k : ι
    hij_lt : LT.lt i j
    h : IsGLB (Set.Ioi i) k
    ⊢ IsGLB (Set.Ioc i j) k
  -/
  simp_rw [IsGLB, IsGreatest, mem_upperBounds, mem_lowerBounds] at h ⊢
  /-
    ι : Type u_1
    inst✝ : LinearOrder ι
    i j k : ι
    hij_lt : LT.lt i j
    h : And (∀ (x : ι), Membership.mem (Set.Ioi i) x → LE.le k x) (∀ (x : ι), (∀ ( …
    ⊢ And (∀ (x : ι), Membership.mem (Set.Ioc i j) x → LE.le k x) (∀ (x : ι), (∀ ( …
  -/
  refine ⟨fun x hx ↦ h.1 x hx.1, fun x hx ↦ h.2 x ?_⟩
  /-
    ι : Type u_1
    inst✝ : LinearOrder ι
    i j k : ι
    hij_lt : LT.lt i j
    h : And (∀ (x : ι), Membership.mem (Set.Ioi i) x → LE.le k x) (∀ (x : ι), (∀ ( …
    x : ι
    hx : ∀ (x_1 : ι), Membership.mem (Set.Ioc i j) x_1 → LE.le x x_1
    ⊢ ∀ (x_1 : ι), Membership.mem (Set.Ioi i) x_1 → LE.le x x_1
  -/
  intro y hy
  /-
    ι : Type u_1
    inst✝ : LinearOrder ι
    i j k : ι
    hij_lt : LT.lt i j
    h : And (∀ (x : ι), Membership.mem (Set.Ioi i) x → LE.le k x) (∀ (x : ι), (∀ ( …
    x : ι
    hx : ∀ (x_1 : ι), Membership.mem (Set.Ioc i j) x_1 → LE.le x x_1
    y : ι
    hy : Membership.mem (Set.Ioi i) y
    ⊢ LE.le x y
  -/
  rcases le_or_lt y j with h_le | h_lt
    /-
      case inl
      ι : Type u_1
      inst✝ : LinearOrder ι
      i j k : ι
      hij_lt : LT.lt i j
      h : And (∀ (x : ι), Membership.mem (Set.Ioi i) x → LE.le k x) (∀ (x : ι), (∀ ( …
      x : ι
      hx : ∀ (x_1 : ι), Membership.mem (Set.Ioc i j) x_1 → LE.le x x_1
      y : ι
      hy : Membership.mem (Set.Ioi i) y
      h_le : LE.le y j
      ⊢ LE.le x y
    -/
  · exact hx y ⟨hy, h_le⟩
    /-
      🎉 no goals
    -/
    /-
      case inr
      ι : Type u_1
      inst✝ : LinearOrder ι
      i j k : ι
      hij_lt : LT.lt i j
      h : And (∀ (x : ι), Membership.mem (Set.Ioi i) x → LE.le k x) (∀ (x : ι), (∀ ( …
      x : ι
      hx : ∀ (x_1 : ι), Membership.mem (Set.Ioc i j) x_1 → LE.le x x_1
      y : ι
      hy : Membership.mem (Set.Ioi i) y
      h_lt : LT.lt j y
      ⊢ LE.le x y
    -/
  · exact le_trans (hx j ⟨hij_lt, le_rfl⟩) h_lt.le
    /-
      🎉 no goals
    -/


theorem isMax_of_succFn_le [LocallyFiniteOrder ι] (i : ι) (hi : succFn i ≤ i) : IsMax i := by
  /-
    ι : Type u_1
    inst✝¹ : LinearOrder ι
    inst✝ : LocallyFiniteOrder ι
    i : ι
    hi : LE.le (LinearLocallyFiniteOrder.succFn i) i
    ⊢ IsMax i
  -/
  refine fun j _ ↦ not_lt.mp fun hij_lt ↦ ?_
  /-
    ι : Type u_1
    inst✝¹ : LinearOrder ι
    inst✝ : LocallyFiniteOrder ι
    i : ι
    hi : LE.le (LinearLocallyFiniteOrder.succFn i) i
    j : ι
    x✝ : LE.le i j
    hij_lt : LT.lt i j
    ⊢ False
  -/
  have h_succFn_eq : succFn i = i := le_antisymm hi (le_succFn i)
  have h_glb : IsGLB (Finset.Ioc i j : Set ι) i := by
    rw [Finset.coe_Ioc]
    have h := succFn_spec i
    rw [h_succFn_eq] at h
    exact isGLB_Ioc_of_isGLB_Ioi hij_lt h
  have hi_mem : i ∈ Finset.Ioc i j := by
    refine Finset.isGLB_mem _ h_glb ?_
    exact ⟨_, Finset.mem_Ioc.mpr ⟨hij_lt, le_rfl⟩⟩
  /-
    ι : Type u_1
    inst✝¹ : LinearOrder ι
    inst✝ : LocallyFiniteOrder ι
    i : ι
    hi : LE.le (LinearLocallyFiniteOrder.succFn i) i
    j : ι
    x✝ : LE.le i j
    hij_lt : LT.lt i j
    h_succFn_eq : Eq (LinearLocallyFiniteOrder.succFn i) i
    h_glb : IsGLB (↑(Finset.Ioc i j)) i
    hi_mem : Membership.mem (Finset.Ioc i j) i
    ⊢ False
  -/
  rw [Finset.mem_Ioc] at hi_mem
  /-
    ι : Type u_1
    inst✝¹ : LinearOrder ι
    inst✝ : LocallyFiniteOrder ι
    i : ι
    hi : LE.le (LinearLocallyFiniteOrder.succFn i) i
    j : ι
    x✝ : LE.le i j
    hij_lt : LT.lt i j
    h_succFn_eq : Eq (LinearLocallyFiniteOrder.succFn i) i
    h_glb : IsGLB (↑(Finset.Ioc i j)) i
    hi_mem : And (LT.lt i i) (LE.le i j)
    ⊢ False
  -/
  exact lt_irrefl i hi_mem.1
  /-
    🎉 no goals
  -/


theorem succFn_le_of_lt (i j : ι) (hij : i < j) : succFn i ≤ j := by
  /-
    ι : Type u_1
    inst✝ : LinearOrder ι
    i j : ι
    hij : LT.lt i j
    ⊢ LE.le (LinearLocallyFiniteOrder.succFn i) j
  -/
  have h := succFn_spec i
  /-
    ι : Type u_1
    inst✝ : LinearOrder ι
    i j : ι
    hij : LT.lt i j
    h : IsGLB (Set.Ioi i) (LinearLocallyFiniteOrder.succFn i)
    ⊢ LE.le (LinearLocallyFiniteOrder.succFn i) j
  -/
  rw [IsGLB, IsGreatest, mem_lowerBounds] at h
  /-
    ι : Type u_1
    inst✝ : LinearOrder ι
    i j : ι
    hij : LT.lt i j
    h : And (∀ (x : ι), Membership.mem (Set.Ioi i) x → LE.le (LinearLocallyFiniteO …
    ⊢ LE.le (LinearLocallyFiniteOrder.succFn i) j
  -/
  exact h.1 j hij
  /-
    🎉 no goals
  -/


theorem le_of_lt_succFn (j i : ι) (hij : j < succFn i) : j ≤ i := by
  /-
    ι : Type u_1
    inst✝ : LinearOrder ι
    j i : ι
    hij : LT.lt j (LinearLocallyFiniteOrder.succFn i)
    ⊢ LE.le j i
  -/
  rw [lt_isGLB_iff (succFn_spec i)] at hij
  /-
    ι : Type u_1
    inst✝ : LinearOrder ι
    j i : ι
    hij : Exists fun c => And (Membership.mem (lowerBounds (Set.Ioi i)) c) (LT.lt  …
    ⊢ LE.le j i
  -/
  obtain ⟨k, hk_lb, hk⟩ := hij
  /-
    case intro.intro
    ι : Type u_1
    inst✝ : LinearOrder ι
    j i k : ι
    hk_lb : Membership.mem (lowerBounds (Set.Ioi i)) k
    hk : LT.lt j k
    ⊢ LE.le j i
  -/
  rw [mem_lowerBounds] at hk_lb
  /-
    case intro.intro
    ι : Type u_1
    inst✝ : LinearOrder ι
    j i k : ι
    hk_lb : ∀ (x : ι), Membership.mem (Set.Ioi i) x → LE.le k x
    hk : LT.lt j k
    ⊢ LE.le j i
  -/
  exact not_lt.mp fun hi_lt_j ↦ not_le.mpr hk (hk_lb j hi_lt_j)
  /-
    🎉 no goals
  -/


noncomputable instance (priority := 100) [LocallyFiniteOrder ι] : SuccOrder ι where
  succ := succFn
  le_succ := le_succFn
  max_of_succ_le h := isMax_of_succFn_le _ h
  succ_le_of_lt h := succFn_le_of_lt _ _ h


noncomputable instance (priority := 100) [LocallyFiniteOrder ι] : PredOrder ι :=
  inferInstanceAs (PredOrder ιᵒᵈᵒᵈ)


instance (priority := 100) [LocallyFiniteOrder ι] : IsSuccArchimedean ι where
  exists_succ_iterate_of_le := by
    /-
      ι : Type u_1
      inst✝¹ : LinearOrder ι
      inst✝ : LocallyFiniteOrder ι
      ⊢ ∀ {a b : ι}, LE.le a b → Exists fun n => Eq (Nat.iterate Order.succ n a) b
    -/
    intro i j hij
    /-
      ι : Type u_1
      inst✝¹ : LinearOrder ι
      inst✝ : LocallyFiniteOrder ι
      i j : ι
      hij : LE.le i j
      ⊢ Exists fun n => Eq (Nat.iterate Order.succ n i) j
    -/
    rw [le_iff_lt_or_eq] at hij
    /-
      ι : Type u_1
      inst✝¹ : LinearOrder ι
      inst✝ : LocallyFiniteOrder ι
      i j : ι
      hij : Or (LT.lt i j) (Eq i j)
      ⊢ Exists fun n => Eq (Nat.iterate Order.succ n i) j
    -/
    cases' hij with hij hij
    /-
      case inl
      ι : Type u_1
      inst✝¹ : LinearOrder ι
      inst✝ : LocallyFiniteOrder ι
      i j : ι
      hij : LT.lt i j
      ⊢ Exists fun n => Eq (Nat.iterate Order.succ n i) j
    -/
    swap
      /-
        case inr
        ι : Type u_1
        inst✝¹ : LinearOrder ι
        inst✝ : LocallyFiniteOrder ι
        i j : ι
        hij : Eq i j
        ⊢ Exists fun n => Eq (Nat.iterate Order.succ n i) j
      -/
    · refine ⟨0, ?_⟩
      /-
        case inr
        ι : Type u_1
        inst✝¹ : LinearOrder ι
        inst✝ : LocallyFiniteOrder ι
        i j : ι
        hij : Eq i j
        ⊢ Eq (Nat.iterate Order.succ 0 i) j
      -/
      simpa only [Function.iterate_zero, id] using hij
      /-
        🎉 no goals
      -/
    /-
      case inl
      ι : Type u_1
      inst✝¹ : LinearOrder ι
      inst✝ : LocallyFiniteOrder ι
      i j : ι
      hij : LT.lt i j
      ⊢ Exists fun n => Eq (Nat.iterate Order.succ n i) j
    -/
    by_contra! h
    have h_lt : ∀ n, succ^[n] i < j := by
      intro n
      induction' n with n hn
      · simpa only [Function.iterate_zero, id] using hij
      · refine lt_of_le_of_ne ?_ (h _)
        rw [Function.iterate_succ', Function.comp_apply]
        exact succ_le_of_lt hn
    have h_mem : ∀ n, succ^[n] i ∈ Finset.Icc i j :=
      fun n ↦ Finset.mem_Icc.mpr ⟨le_succ_iterate n i, (h_lt n).le⟩
    obtain ⟨n, m, hnm, h_eq⟩ : ∃ n m, n < m ∧ succ^[n] i = succ^[m] i := by
      let f : ℕ → Finset.Icc i j := fun n ↦ ⟨succ^[n] i, h_mem n⟩
      obtain ⟨n, m, hnm_ne, hfnm⟩ : ∃ n m, n ≠ m ∧ f n = f m :=
        Finite.exists_ne_map_eq_of_infinite f
      have hnm_eq : succ^[n] i = succ^[m] i := by simpa only [f, Subtype.mk_eq_mk] using hfnm
      rcases le_total n m with h_le | h_le
      · exact ⟨n, m, lt_of_le_of_ne h_le hnm_ne, hnm_eq⟩
      · exact ⟨m, n, lt_of_le_of_ne h_le hnm_ne.symm, hnm_eq.symm⟩
    /-
      case inl.intro.intro.intro
      ι : Type u_1
      inst✝¹ : LinearOrder ι
      inst✝ : LocallyFiniteOrder ι
      i j : ι
      hij : LT.lt i j
      h : ∀ (n : Nat), Ne (Nat.iterate Order.succ n i) j
      h_lt : ∀ (n : Nat), LT.lt (Nat.iterate Order.succ n i) j
      h_mem : ∀ (n : Nat), Membership.mem (Finset.Icc i j) (Nat.iterate Order.succ n …
      n m : Nat
      hnm : LT.lt n m
      h_eq : Eq (Nat.iterate Order.succ n i) (Nat.iterate Order.succ m i)
      ⊢ False
    -/
    have h_max : IsMax (succ^[n] i) := isMax_iterate_succ_of_eq_of_ne h_eq hnm.ne
    /-
      case inl.intro.intro.intro
      ι : Type u_1
      inst✝¹ : LinearOrder ι
      inst✝ : LocallyFiniteOrder ι
      i j : ι
      hij : LT.lt i j
      h : ∀ (n : Nat), Ne (Nat.iterate Order.succ n i) j
      h_lt : ∀ (n : Nat), LT.lt (Nat.iterate Order.succ n i) j
      h_mem : ∀ (n : Nat), Membership.mem (Finset.Icc i j) (Nat.iterate Order.succ n …
      n m : Nat
      hnm : LT.lt n m
      h_eq : Eq (Nat.iterate Order.succ n i) (Nat.iterate Order.succ m i)
      h_max : IsMax (Nat.iterate Order.succ n i)
      ⊢ False
    -/
    exact not_le.mpr (h_lt n) (h_max (h_lt n).le)
    /-
      🎉 no goals
    -/


instance (priority := 100) [LocallyFiniteOrder ι] : IsPredArchimedean ι :=
  inferInstance


/-- `toZ` numbers elements of `ι` according to their order, starting from `i0`. We prove in
`orderIsoRangeToZOfLinearSuccPredArch` that this defines an `OrderIso` between `ι` and
the range of `toZ`. -/
def toZ (i0 i : ι) : ℤ :=
  dite (i0 ≤ i) (fun hi ↦ Nat.find (exists_succ_iterate_of_le hi)) fun hi ↦
    -Nat.find (exists_pred_iterate_of_le (α := ι) (not_le.mp hi).le)


theorem toZ_of_ge (hi : i0 ≤ i) : toZ i0 i = Nat.find (exists_succ_iterate_of_le hi) :=
  dif_pos hi


theorem toZ_of_lt (hi : i < i0) :
    toZ i0 i = -Nat.find (exists_pred_iterate_of_le (α := ι) hi.le) :=
  dif_neg (not_le.mpr hi)


@[simp]
theorem toZ_of_eq : toZ i0 i0 = 0 := by
  /-
    ι : Type u_1
    inst✝³ : LinearOrder ι
    inst✝² : SuccOrder ι
    inst✝¹ : IsSuccArchimedean ι
    inst✝ : PredOrder ι
    i0 : ι
    ⊢ Eq (toZ i0 i0) 0
  -/
  rw [toZ_of_ge le_rfl]
  /-
    ι : Type u_1
    inst✝³ : LinearOrder ι
    inst✝² : SuccOrder ι
    inst✝¹ : IsSuccArchimedean ι
    inst✝ : PredOrder ι
    i0 : ι
    ⊢ Eq (↑(Nat.find ⋯)) 0
  -/
  norm_cast
  /-
    ι : Type u_1
    inst✝³ : LinearOrder ι
    inst✝² : SuccOrder ι
    inst✝¹ : IsSuccArchimedean ι
    inst✝ : PredOrder ι
    i0 : ι
    ⊢ Eq (Nat.find ⋯) 0
  -/
  refine le_antisymm (Nat.find_le ?_) (zero_le _)
  /-
    ι : Type u_1
    inst✝³ : LinearOrder ι
    inst✝² : SuccOrder ι
    inst✝¹ : IsSuccArchimedean ι
    inst✝ : PredOrder ι
    i0 : ι
    ⊢ Eq (Nat.iterate Order.succ 0 i0) i0
  -/
  rw [Function.iterate_zero, id]
  /-
    🎉 no goals
  -/


theorem iterate_succ_toZ (i : ι) (hi : i0 ≤ i) : succ^[(toZ i0 i).toNat] i0 = i := by
  /-
    ι : Type u_1
    inst✝³ : LinearOrder ι
    inst✝² : SuccOrder ι
    inst✝¹ : IsSuccArchimedean ι
    inst✝ : PredOrder ι
    i0 i : ι
    hi : LE.le i0 i
    ⊢ Eq (Nat.iterate Order.succ (toZ i0 i).toNat i0) i
  -/
  rw [toZ_of_ge hi, Int.toNat_natCast]
  /-
    ι : Type u_1
    inst✝³ : LinearOrder ι
    inst✝² : SuccOrder ι
    inst✝¹ : IsSuccArchimedean ι
    inst✝ : PredOrder ι
    i0 i : ι
    hi : LE.le i0 i
    ⊢ Eq (Nat.iterate Order.succ (Nat.find ⋯) i0) i
  -/
  exact Nat.find_spec (exists_succ_iterate_of_le hi)
  /-
    🎉 no goals
  -/


theorem iterate_pred_toZ (i : ι) (hi : i < i0) : pred^[(-toZ i0 i).toNat] i0 = i := by
  /-
    ι : Type u_1
    inst✝³ : LinearOrder ι
    inst✝² : SuccOrder ι
    inst✝¹ : IsSuccArchimedean ι
    inst✝ : PredOrder ι
    i0 i : ι
    hi : LT.lt i i0
    ⊢ Eq (Nat.iterate Order.pred (Neg.neg (toZ i0 i)).toNat i0) i
  -/
  rw [toZ_of_lt hi, neg_neg, Int.toNat_natCast]
  /-
    ι : Type u_1
    inst✝³ : LinearOrder ι
    inst✝² : SuccOrder ι
    inst✝¹ : IsSuccArchimedean ι
    inst✝ : PredOrder ι
    i0 i : ι
    hi : LT.lt i i0
    ⊢ Eq (Nat.iterate Order.pred (Nat.find ⋯) i0) i
  -/
  exact Nat.find_spec (exists_pred_iterate_of_le hi.le)
  /-
    🎉 no goals
  -/


                                                    /-
                                                      ι : Type u_1
                                                      inst✝³ : LinearOrder ι
                                                      inst✝² : SuccOrder ι
                                                      inst✝¹ : IsSuccArchimedean ι
                                                      inst✝ : PredOrder ι
                                                      i0 i : ι
                                                      hi : LE.le i0 i
                                                      ⊢ LE.le 0 (toZ i0 i)
                                                    -/
lemma toZ_nonneg (hi : i0 ≤ i) : 0 ≤ toZ i0 i := by rw [toZ_of_ge hi]; exact Int.natCast_nonneg _
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


theorem toZ_neg (hi : i < i0) : toZ i0 i < 0 := by
  /-
    ι : Type u_1
    inst✝³ : LinearOrder ι
    inst✝² : SuccOrder ι
    inst✝¹ : IsSuccArchimedean ι
    inst✝ : PredOrder ι
    i0 i : ι
    hi : LT.lt i i0
    ⊢ LT.lt (toZ i0 i) 0
  -/
  refine lt_of_le_of_ne ?_ ?_
    /-
      case refine_1
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 i : ι
      hi : LT.lt i i0
      ⊢ LE.le (toZ i0 i) 0
    -/
  · rw [toZ_of_lt hi]
    /-
      case refine_1
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 i : ι
      hi : LT.lt i i0
      ⊢ LE.le (Neg.neg ↑(Nat.find ⋯)) 0
    -/
    omega
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 i : ι
      hi : LT.lt i i0
      ⊢ Ne (toZ i0 i) 0
    -/
  · by_contra h
    /-
      case refine_2
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 i : ι
      hi : LT.lt i i0
      h : Eq (toZ i0 i) 0
      ⊢ False
    -/
    have h_eq := iterate_pred_toZ i hi
    /-
      case refine_2
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 i : ι
      hi : LT.lt i i0
      h : Eq (toZ i0 i) 0
      h_eq : Eq (Nat.iterate Order.pred (Neg.neg (toZ i0 i)).toNat i0) i
      ⊢ False
    -/
    rw [← h_eq, h] at hi
    /-
      case refine_2
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 i : ι
      hi : LT.lt (Nat.iterate Order.pred (-0).toNat i0) i0
      h : Eq (toZ i0 i) 0
      h_eq : Eq (Nat.iterate Order.pred (Neg.neg (toZ i0 i)).toNat i0) i
      ⊢ False
    -/
    simp only [neg_zero, Int.toNat_zero, Function.iterate_zero, id, lt_self_iff_false] at hi
    /-
      🎉 no goals
    -/


theorem toZ_iterate_succ_le (n : ℕ) : toZ i0 (succ^[n] i0) ≤ n := by
  /-
    ι : Type u_1
    inst✝³ : LinearOrder ι
    inst✝² : SuccOrder ι
    inst✝¹ : IsSuccArchimedean ι
    inst✝ : PredOrder ι
    i0 : ι
    n : Nat
    ⊢ LE.le (toZ i0 (Nat.iterate Order.succ n i0)) ↑n
  -/
  rw [toZ_of_ge (le_succ_iterate _ _)]
  /-
    ι : Type u_1
    inst✝³ : LinearOrder ι
    inst✝² : SuccOrder ι
    inst✝¹ : IsSuccArchimedean ι
    inst✝ : PredOrder ι
    i0 : ι
    n : Nat
    ⊢ LE.le ↑(Nat.find ⋯) ↑n
  -/
  norm_cast
  /-
    ι : Type u_1
    inst✝³ : LinearOrder ι
    inst✝² : SuccOrder ι
    inst✝¹ : IsSuccArchimedean ι
    inst✝ : PredOrder ι
    i0 : ι
    n : Nat
    ⊢ LE.le (Nat.find ⋯) n
  -/
  exact Nat.find_min' _ rfl
  /-
    🎉 no goals
  -/


theorem toZ_iterate_pred_ge (n : ℕ) : -(n : ℤ) ≤ toZ i0 (pred^[n] i0) := by
  /-
    ι : Type u_1
    inst✝³ : LinearOrder ι
    inst✝² : SuccOrder ι
    inst✝¹ : IsSuccArchimedean ι
    inst✝ : PredOrder ι
    i0 : ι
    n : Nat
    ⊢ LE.le (Neg.neg ↑n) (toZ i0 (Nat.iterate Order.pred n i0))
  -/
  rcases le_or_lt i0 (pred^[n] i0) with h | h
    /-
      case inl
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 : ι
      n : Nat
      h : LE.le i0 (Nat.iterate Order.pred n i0)
      ⊢ LE.le (Neg.neg ↑n) (toZ i0 (Nat.iterate Order.pred n i0))
    -/
  · have h_eq : pred^[n] i0 = i0 := le_antisymm (pred_iterate_le _ _) h
    /-
      case inl
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 : ι
      n : Nat
      h : LE.le i0 (Nat.iterate Order.pred n i0)
      h_eq : Eq (Nat.iterate Order.pred n i0) i0
      ⊢ LE.le (Neg.neg ↑n) (toZ i0 (Nat.iterate Order.pred n i0))
    -/
    rw [h_eq, toZ_of_eq]
    /-
      case inl
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 : ι
      n : Nat
      h : LE.le i0 (Nat.iterate Order.pred n i0)
      h_eq : Eq (Nat.iterate Order.pred n i0) i0
      ⊢ LE.le (Neg.neg ↑n) 0
    -/
    omega
    /-
      🎉 no goals
    -/
    /-
      case inr
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 : ι
      n : Nat
      h : LT.lt (Nat.iterate Order.pred n i0) i0
      ⊢ LE.le (Neg.neg ↑n) (toZ i0 (Nat.iterate Order.pred n i0))
    -/
  · rw [toZ_of_lt h]
    /-
      case inr
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 : ι
      n : Nat
      h : LT.lt (Nat.iterate Order.pred n i0) i0
      ⊢ LE.le (Neg.neg ↑n) (Neg.neg ↑(Nat.find ⋯))
    -/
    refine Int.neg_le_neg ?_
    /-
      case inr
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 : ι
      n : Nat
      h : LT.lt (Nat.iterate Order.pred n i0) i0
      ⊢ LE.le ↑(Nat.find ⋯) ↑n
    -/
    norm_cast
    /-
      case inr
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 : ι
      n : Nat
      h : LT.lt (Nat.iterate Order.pred n i0) i0
      ⊢ LE.le (Nat.find ⋯) n
    -/
    exact Nat.find_min' _ rfl
    /-
      🎉 no goals
    -/


theorem toZ_iterate_succ_of_not_isMax (n : ℕ) (hn : ¬IsMax (succ^[n] i0)) :
    toZ i0 (succ^[n] i0) = n := by
  /-
    ι : Type u_1
    inst✝³ : LinearOrder ι
    inst✝² : SuccOrder ι
    inst✝¹ : IsSuccArchimedean ι
    inst✝ : PredOrder ι
    i0 : ι
    n : Nat
    hn : Not (IsMax (Nat.iterate Order.succ n i0))
    ⊢ Eq (toZ i0 (Nat.iterate Order.succ n i0)) ↑n
  -/
  let m := (toZ i0 (succ^[n] i0)).toNat
  /-
    ι : Type u_1
    inst✝³ : LinearOrder ι
    inst✝² : SuccOrder ι
    inst✝¹ : IsSuccArchimedean ι
    inst✝ : PredOrder ι
    i0 : ι
    n : Nat
    hn : Not (IsMax (Nat.iterate Order.succ n i0))
    m : Nat := (toZ i0 (Nat.iterate Order.succ n i0)).toNat
    ⊢ Eq (toZ i0 (Nat.iterate Order.succ n i0)) ↑n
  -/
  have h_eq : succ^[m] i0 = succ^[n] i0 := iterate_succ_toZ _ (le_succ_iterate _ _)
  /-
    ι : Type u_1
    inst✝³ : LinearOrder ι
    inst✝² : SuccOrder ι
    inst✝¹ : IsSuccArchimedean ι
    inst✝ : PredOrder ι
    i0 : ι
    n : Nat
    hn : Not (IsMax (Nat.iterate Order.succ n i0))
    m : Nat := (toZ i0 (Nat.iterate Order.succ n i0)).toNat
    h_eq : Eq (Nat.iterate Order.succ m i0) (Nat.iterate Order.succ n i0)
    ⊢ Eq (toZ i0 (Nat.iterate Order.succ n i0)) ↑n
  -/
  by_cases hmn : m = n
    /-
      case pos
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 : ι
      n : Nat
      hn : Not (IsMax (Nat.iterate Order.succ n i0))
      m : Nat := (toZ i0 (Nat.iterate Order.succ n i0)).toNat
      h_eq : Eq (Nat.iterate Order.succ m i0) (Nat.iterate Order.succ n i0)
      hmn : Eq m n
      ⊢ Eq (toZ i0 (Nat.iterate Order.succ n i0)) ↑n
    -/
  · nth_rw 2 [← hmn]
    /-
      case pos
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 : ι
      n : Nat
      hn : Not (IsMax (Nat.iterate Order.succ n i0))
      m : Nat := (toZ i0 (Nat.iterate Order.succ n i0)).toNat
      h_eq : Eq (Nat.iterate Order.succ m i0) (Nat.iterate Order.succ n i0)
      hmn : Eq m n
      ⊢ Eq (toZ i0 (Nat.iterate Order.succ n i0)) ↑m
    -/
    rw [Int.toNat_eq_max, toZ_of_ge (le_succ_iterate _ _), max_eq_left]
    /-
      case pos
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 : ι
      n : Nat
      hn : Not (IsMax (Nat.iterate Order.succ n i0))
      m : Nat := (toZ i0 (Nat.iterate Order.succ n i0)).toNat
      h_eq : Eq (Nat.iterate Order.succ m i0) (Nat.iterate Order.succ n i0)
      hmn : Eq m n
      ⊢ LE.le 0 ↑(Nat.find ⋯)
    -/
    exact Int.natCast_nonneg _
    /-
      🎉 no goals
    -/
  /-
    case neg
    ι : Type u_1
    inst✝³ : LinearOrder ι
    inst✝² : SuccOrder ι
    inst✝¹ : IsSuccArchimedean ι
    inst✝ : PredOrder ι
    i0 : ι
    n : Nat
    hn : Not (IsMax (Nat.iterate Order.succ n i0))
    m : Nat := (toZ i0 (Nat.iterate Order.succ n i0)).toNat
    h_eq : Eq (Nat.iterate Order.succ m i0) (Nat.iterate Order.succ n i0)
    hmn : Not (Eq m n)
    ⊢ Eq (toZ i0 (Nat.iterate Order.succ n i0)) ↑n
  -/
  suffices IsMax (succ^[n] i0) from absurd this hn
  /-
    case neg
    ι : Type u_1
    inst✝³ : LinearOrder ι
    inst✝² : SuccOrder ι
    inst✝¹ : IsSuccArchimedean ι
    inst✝ : PredOrder ι
    i0 : ι
    n : Nat
    hn : Not (IsMax (Nat.iterate Order.succ n i0))
    m : Nat := (toZ i0 (Nat.iterate Order.succ n i0)).toNat
    h_eq : Eq (Nat.iterate Order.succ m i0) (Nat.iterate Order.succ n i0)
    hmn : Not (Eq m n)
    ⊢ IsMax (Nat.iterate Order.succ n i0)
  -/
  exact isMax_iterate_succ_of_eq_of_ne h_eq.symm (Ne.symm hmn)
  /-
    🎉 no goals
  -/


theorem toZ_iterate_pred_of_not_isMin (n : ℕ) (hn : ¬IsMin (pred^[n] i0)) :
    toZ i0 (pred^[n] i0) = -n := by
  /-
    ι : Type u_1
    inst✝³ : LinearOrder ι
    inst✝² : SuccOrder ι
    inst✝¹ : IsSuccArchimedean ι
    inst✝ : PredOrder ι
    i0 : ι
    n : Nat
    hn : Not (IsMin (Nat.iterate Order.pred n i0))
    ⊢ Eq (toZ i0 (Nat.iterate Order.pred n i0)) (Neg.neg ↑n)
  -/
  cases' n with n
    /-
      case zero
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 : ι
      hn : Not (IsMin (Nat.iterate Order.pred 0 i0))
      ⊢ Eq (toZ i0 (Nat.iterate Order.pred 0 i0)) (Neg.neg ↑0)
    -/
  · simp
    /-
      🎉 no goals
    -/
  have : pred^[n.succ] i0 < i0 := by
    refine lt_of_le_of_ne (pred_iterate_le _ _) fun h_pred_iterate_eq ↦ hn ?_
    have h_pred_eq_pred : pred^[n.succ] i0 = pred^[0] i0 := by
      rwa [Function.iterate_zero, id]
    exact isMin_iterate_pred_of_eq_of_ne h_pred_eq_pred (Nat.succ_ne_zero n)
  /-
    case succ
    ι : Type u_1
    inst✝³ : LinearOrder ι
    inst✝² : SuccOrder ι
    inst✝¹ : IsSuccArchimedean ι
    inst✝ : PredOrder ι
    i0 : ι
    n : Nat
    hn : Not (IsMin (Nat.iterate Order.pred (HAdd.hAdd n 1) i0))
    this : LT.lt (Nat.iterate Order.pred n.succ i0) i0
    ⊢ Eq (toZ i0 (Nat.iterate Order.pred (HAdd.hAdd n 1) i0)) (Neg.neg ↑(HAdd.hAdd …
  -/
  let m := (-toZ i0 (pred^[n.succ] i0)).toNat
  /-
    case succ
    ι : Type u_1
    inst✝³ : LinearOrder ι
    inst✝² : SuccOrder ι
    inst✝¹ : IsSuccArchimedean ι
    inst✝ : PredOrder ι
    i0 : ι
    n : Nat
    hn : Not (IsMin (Nat.iterate Order.pred (HAdd.hAdd n 1) i0))
    this : LT.lt (Nat.iterate Order.pred n.succ i0) i0
    m : Nat := (Neg.neg (toZ i0 (Nat.iterate Order.pred n.succ i0))).toNat
    ⊢ Eq (toZ i0 (Nat.iterate Order.pred (HAdd.hAdd n 1) i0)) (Neg.neg ↑(HAdd.hAdd …
  -/
  have h_eq : pred^[m] i0 = pred^[n.succ] i0 := iterate_pred_toZ _ this
  /-
    case succ
    ι : Type u_1
    inst✝³ : LinearOrder ι
    inst✝² : SuccOrder ι
    inst✝¹ : IsSuccArchimedean ι
    inst✝ : PredOrder ι
    i0 : ι
    n : Nat
    hn : Not (IsMin (Nat.iterate Order.pred (HAdd.hAdd n 1) i0))
    this : LT.lt (Nat.iterate Order.pred n.succ i0) i0
    m : Nat := (Neg.neg (toZ i0 (Nat.iterate Order.pred n.succ i0))).toNat
    h_eq : Eq (Nat.iterate Order.pred m i0) (Nat.iterate Order.pred n.succ i0)
    ⊢ Eq (toZ i0 (Nat.iterate Order.pred (HAdd.hAdd n 1) i0)) (Neg.neg ↑(HAdd.hAdd …
  -/
  by_cases hmn : m = n + 1
    /-
      case pos
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 : ι
      n : Nat
      hn : Not (IsMin (Nat.iterate Order.pred (HAdd.hAdd n 1) i0))
      this : LT.lt (Nat.iterate Order.pred n.succ i0) i0
      m : Nat := (Neg.neg (toZ i0 (Nat.iterate Order.pred n.succ i0))).toNat
      h_eq : Eq (Nat.iterate Order.pred m i0) (Nat.iterate Order.pred n.succ i0)
      hmn : Eq m (HAdd.hAdd n 1)
      ⊢ Eq (toZ i0 (Nat.iterate Order.pred (HAdd.hAdd n 1) i0)) (Neg.neg ↑(HAdd.hAdd …
    -/
  · nth_rw 2 [← hmn]
    /-
      case pos
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 : ι
      n : Nat
      hn : Not (IsMin (Nat.iterate Order.pred (HAdd.hAdd n 1) i0))
      this : LT.lt (Nat.iterate Order.pred n.succ i0) i0
      m : Nat := (Neg.neg (toZ i0 (Nat.iterate Order.pred n.succ i0))).toNat
      h_eq : Eq (Nat.iterate Order.pred m i0) (Nat.iterate Order.pred n.succ i0)
      hmn : Eq m (HAdd.hAdd n 1)
      ⊢ Eq (toZ i0 (Nat.iterate Order.pred (HAdd.hAdd n 1) i0)) (Neg.neg ↑m)
    -/
    rw [Int.toNat_eq_max, toZ_of_lt this, max_eq_left, neg_neg]
    /-
      case pos
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 : ι
      n : Nat
      hn : Not (IsMin (Nat.iterate Order.pred (HAdd.hAdd n 1) i0))
      this : LT.lt (Nat.iterate Order.pred n.succ i0) i0
      m : Nat := (Neg.neg (toZ i0 (Nat.iterate Order.pred n.succ i0))).toNat
      h_eq : Eq (Nat.iterate Order.pred m i0) (Nat.iterate Order.pred n.succ i0)
      hmn : Eq m (HAdd.hAdd n 1)
      ⊢ LE.le 0 (Neg.neg (Neg.neg ↑(Nat.find ⋯)))
    -/
    rw [neg_neg]
    /-
      case pos
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 : ι
      n : Nat
      hn : Not (IsMin (Nat.iterate Order.pred (HAdd.hAdd n 1) i0))
      this : LT.lt (Nat.iterate Order.pred n.succ i0) i0
      m : Nat := (Neg.neg (toZ i0 (Nat.iterate Order.pred n.succ i0))).toNat
      h_eq : Eq (Nat.iterate Order.pred m i0) (Nat.iterate Order.pred n.succ i0)
      hmn : Eq m (HAdd.hAdd n 1)
      ⊢ LE.le 0 ↑(Nat.find ⋯)
    -/
    exact Int.natCast_nonneg _
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 : ι
      n : Nat
      hn : Not (IsMin (Nat.iterate Order.pred (HAdd.hAdd n 1) i0))
      this : LT.lt (Nat.iterate Order.pred n.succ i0) i0
      m : Nat := (Neg.neg (toZ i0 (Nat.iterate Order.pred n.succ i0))).toNat
      h_eq : Eq (Nat.iterate Order.pred m i0) (Nat.iterate Order.pred n.succ i0)
      hmn : Not (Eq m (HAdd.hAdd n 1))
      ⊢ Eq (toZ i0 (Nat.iterate Order.pred (HAdd.hAdd n 1) i0)) (Neg.neg ↑(HAdd.hAdd …
    -/
  · suffices IsMin (pred^[n.succ] i0) from absurd this hn
    /-
      case neg
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 : ι
      n : Nat
      hn : Not (IsMin (Nat.iterate Order.pred (HAdd.hAdd n 1) i0))
      this : LT.lt (Nat.iterate Order.pred n.succ i0) i0
      m : Nat := (Neg.neg (toZ i0 (Nat.iterate Order.pred n.succ i0))).toNat
      h_eq : Eq (Nat.iterate Order.pred m i0) (Nat.iterate Order.pred n.succ i0)
      hmn : Not (Eq m (HAdd.hAdd n 1))
      ⊢ IsMin (Nat.iterate Order.pred n.succ i0)
    -/
    exact isMin_iterate_pred_of_eq_of_ne h_eq.symm (Ne.symm hmn)
    /-
      🎉 no goals
    -/


theorem le_of_toZ_le {j : ι} (h_le : toZ i0 i ≤ toZ i0 j) : i ≤ j := by
  /-
    ι : Type u_1
    inst✝³ : LinearOrder ι
    inst✝² : SuccOrder ι
    inst✝¹ : IsSuccArchimedean ι
    inst✝ : PredOrder ι
    i0 i j : ι
    h_le : LE.le (toZ i0 i) (toZ i0 j)
    ⊢ LE.le i j
  -/
  rcases le_or_lt i0 i with hi | hi <;> rcases le_or_lt i0 j with hj | hj
    /-
      case inl.inl
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 i j : ι
      h_le : LE.le (toZ i0 i) (toZ i0 j)
      hi : LE.le i0 i
      hj : LE.le i0 j
      ⊢ LE.le i j
    -/
  · rw [← iterate_succ_toZ i hi, ← iterate_succ_toZ j hj]
    /-
      case inl.inl
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 i j : ι
      h_le : LE.le (toZ i0 i) (toZ i0 j)
      hi : LE.le i0 i
      hj : LE.le i0 j
      ⊢ LE.le (Nat.iterate Order.succ (toZ i0 i).toNat i0) (Nat.iterate Order.succ ( …
    -/
    exact Monotone.monotone_iterate_of_le_map succ_mono (le_succ _) (Int.toNat_le_toNat h_le)
    /-
      🎉 no goals
    -/
    /-
      case inl.inr
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 i j : ι
      h_le : LE.le (toZ i0 i) (toZ i0 j)
      hi : LE.le i0 i
      hj : LT.lt j i0
      ⊢ LE.le i j
    -/
  · exact absurd ((toZ_neg hj).trans_le (toZ_nonneg hi)) (not_lt.mpr h_le)
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 i j : ι
      h_le : LE.le (toZ i0 i) (toZ i0 j)
      hi : LT.lt i i0
      hj : LE.le i0 j
      ⊢ LE.le i j
    -/
  · exact hi.le.trans hj
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 i j : ι
      h_le : LE.le (toZ i0 i) (toZ i0 j)
      hi : LT.lt i i0
      hj : LT.lt j i0
      ⊢ LE.le i j
    -/
  · rw [← iterate_pred_toZ i hi, ← iterate_pred_toZ j hj]
    /-
      case inr.inr
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 i j : ι
      h_le : LE.le (toZ i0 i) (toZ i0 j)
      hi : LT.lt i i0
      hj : LT.lt j i0
      ⊢ LE.le (Nat.iterate Order.pred (Neg.neg (toZ i0 i)).toNat i0) (Nat.iterate Or …
    -/
    refine Monotone.antitone_iterate_of_map_le pred_mono (pred_le _) (Int.toNat_le_toNat ?_)
    /-
      case inr.inr
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 i j : ι
      h_le : LE.le (toZ i0 i) (toZ i0 j)
      hi : LT.lt i i0
      hj : LT.lt j i0
      ⊢ LE.le (Neg.neg (toZ i0 j)) (Neg.neg (toZ i0 i))
    -/
    exact Int.neg_le_neg h_le
    /-
      🎉 no goals
    -/


theorem toZ_mono {i j : ι} (h_le : i ≤ j) : toZ i0 i ≤ toZ i0 j := by
  /-
    ι : Type u_1
    inst✝³ : LinearOrder ι
    inst✝² : SuccOrder ι
    inst✝¹ : IsSuccArchimedean ι
    inst✝ : PredOrder ι
    i0 i j : ι
    h_le : LE.le i j
    ⊢ LE.le (toZ i0 i) (toZ i0 j)
  -/
  by_cases hi_max : IsMax i
    /-
      case pos
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 i j : ι
      h_le : LE.le i j
      hi_max : IsMax i
      ⊢ LE.le (toZ i0 i) (toZ i0 j)
    -/
  · rw [le_antisymm h_le (hi_max h_le)]
    /-
      🎉 no goals
    -/
  /-
    case neg
    ι : Type u_1
    inst✝³ : LinearOrder ι
    inst✝² : SuccOrder ι
    inst✝¹ : IsSuccArchimedean ι
    inst✝ : PredOrder ι
    i0 i j : ι
    h_le : LE.le i j
    hi_max : Not (IsMax i)
    ⊢ LE.le (toZ i0 i) (toZ i0 j)
  -/
  by_cases hj_min : IsMin j
    /-
      case pos
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 i j : ι
      h_le : LE.le i j
      hi_max : Not (IsMax i)
      hj_min : IsMin j
      ⊢ LE.le (toZ i0 i) (toZ i0 j)
    -/
  · rw [le_antisymm h_le (hj_min h_le)]
    /-
      🎉 no goals
    -/
  /-
    case neg
    ι : Type u_1
    inst✝³ : LinearOrder ι
    inst✝² : SuccOrder ι
    inst✝¹ : IsSuccArchimedean ι
    inst✝ : PredOrder ι
    i0 i j : ι
    h_le : LE.le i j
    hi_max : Not (IsMax i)
    hj_min : Not (IsMin j)
    ⊢ LE.le (toZ i0 i) (toZ i0 j)
  -/
  rcases le_or_lt i0 i with hi | hi <;> rcases le_or_lt i0 j with hj | hj
    /-
      case neg.inl.inl
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 i j : ι
      h_le : LE.le i j
      hi_max : Not (IsMax i)
      hj_min : Not (IsMin j)
      hi : LE.le i0 i
      hj : LE.le i0 j
      ⊢ LE.le (toZ i0 i) (toZ i0 j)
    -/
  · let m := Nat.find (exists_succ_iterate_of_le h_le)
    /-
      case neg.inl.inl
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 i j : ι
      h_le : LE.le i j
      hi_max : Not (IsMax i)
      hj_min : Not (IsMin j)
      hi : LE.le i0 i
      hj : LE.le i0 j
      m : Nat := Nat.find ⋯
      ⊢ LE.le (toZ i0 i) (toZ i0 j)
    -/
    have hm : succ^[m] i = j := Nat.find_spec (exists_succ_iterate_of_le h_le)
    have hj_eq : j = succ^[(toZ i0 i).toNat + m] i0 := by
      rw [← hm, add_comm]
      nth_rw 1 [← iterate_succ_toZ i hi]
      rw [Function.iterate_add]
      rfl
    /-
      case neg.inl.inl
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 i j : ι
      h_le : LE.le i j
      hi_max : Not (IsMax i)
      hj_min : Not (IsMin j)
      hi : LE.le i0 i
      hj : LE.le i0 j
      m : Nat := Nat.find ⋯
      hm : Eq (Nat.iterate Order.succ m i) j
      hj_eq : Eq j (Nat.iterate Order.succ (HAdd.hAdd (toZ i0 i).toNat m) i0)
      ⊢ LE.le (toZ i0 i) (toZ i0 j)
    -/
    by_contra h
    /-
      case neg.inl.inl
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 i j : ι
      h_le : LE.le i j
      hi_max : Not (IsMax i)
      hj_min : Not (IsMin j)
      hi : LE.le i0 i
      hj : LE.le i0 j
      m : Nat := Nat.find ⋯
      hm : Eq (Nat.iterate Order.succ m i) j
      hj_eq : Eq j (Nat.iterate Order.succ (HAdd.hAdd (toZ i0 i).toNat m) i0)
      h : Not (LE.le (toZ i0 i) (toZ i0 j))
      ⊢ False
    -/
    by_cases hm0 : m = 0
      /-
        case pos
        ι : Type u_1
        inst✝³ : LinearOrder ι
        inst✝² : SuccOrder ι
        inst✝¹ : IsSuccArchimedean ι
        inst✝ : PredOrder ι
        i0 i j : ι
        h_le : LE.le i j
        hi_max : Not (IsMax i)
        hj_min : Not (IsMin j)
        hi : LE.le i0 i
        hj : LE.le i0 j
        m : Nat := Nat.find ⋯
        hm : Eq (Nat.iterate Order.succ m i) j
        hj_eq : Eq j (Nat.iterate Order.succ (HAdd.hAdd (toZ i0 i).toNat m) i0)
        h : Not (LE.le (toZ i0 i) (toZ i0 j))
        hm0 : Eq m 0
        ⊢ False
      -/
    · rw [hm0, Function.iterate_zero, id] at hm
      /-
        case pos
        ι : Type u_1
        inst✝³ : LinearOrder ι
        inst✝² : SuccOrder ι
        inst✝¹ : IsSuccArchimedean ι
        inst✝ : PredOrder ι
        i0 i j : ι
        h_le : LE.le i j
        hi_max : Not (IsMax i)
        hj_min : Not (IsMin j)
        hi : LE.le i0 i
        hj : LE.le i0 j
        m : Nat := Nat.find ⋯
        hm : Eq i j
        hj_eq : Eq j (Nat.iterate Order.succ (HAdd.hAdd (toZ i0 i).toNat m) i0)
        h : Not (LE.le (toZ i0 i) (toZ i0 j))
        hm0 : Eq m 0
        ⊢ False
      -/
      rw [hm] at h
      /-
        case pos
        ι : Type u_1
        inst✝³ : LinearOrder ι
        inst✝² : SuccOrder ι
        inst✝¹ : IsSuccArchimedean ι
        inst✝ : PredOrder ι
        i0 i j : ι
        h_le : LE.le i j
        hi_max : Not (IsMax i)
        hj_min : Not (IsMin j)
        hi : LE.le i0 i
        hj : LE.le i0 j
        m : Nat := Nat.find ⋯
        hm : Eq i j
        hj_eq : Eq j (Nat.iterate Order.succ (HAdd.hAdd (toZ i0 i).toNat m) i0)
        h : Not (LE.le (toZ i0 j) (toZ i0 j))
        hm0 : Eq m 0
        ⊢ False
      -/
      exact h (le_of_eq rfl)
      /-
        🎉 no goals
      -/
    /-
      case neg
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 i j : ι
      h_le : LE.le i j
      hi_max : Not (IsMax i)
      hj_min : Not (IsMin j)
      hi : LE.le i0 i
      hj : LE.le i0 j
      m : Nat := Nat.find ⋯
      hm : Eq (Nat.iterate Order.succ m i) j
      hj_eq : Eq j (Nat.iterate Order.succ (HAdd.hAdd (toZ i0 i).toNat m) i0)
      h : Not (LE.le (toZ i0 i) (toZ i0 j))
      hm0 : Not (Eq m 0)
      ⊢ False
    -/
    refine hi_max (max_of_succ_le (le_trans ?_ (@le_of_toZ_le _ _ _ _ _ i0 j i ?_)))
    · have h_succ_le : succ^[(toZ i0 i).toNat + 1] i0 ≤ j := by
        rw [hj_eq]
        refine Monotone.monotone_iterate_of_le_map succ_mono (le_succ i0) (add_le_add_left ?_ _)
        exact Nat.one_le_iff_ne_zero.mpr hm0
      /-
        case neg.refine_1
        ι : Type u_1
        inst✝³ : LinearOrder ι
        inst✝² : SuccOrder ι
        inst✝¹ : IsSuccArchimedean ι
        inst✝ : PredOrder ι
        i0 i j : ι
        h_le : LE.le i j
        hi_max : Not (IsMax i)
        hj_min : Not (IsMin j)
        hi : LE.le i0 i
        hj : LE.le i0 j
        m : Nat := Nat.find ⋯
        hm : Eq (Nat.iterate Order.succ m i) j
        hj_eq : Eq j (Nat.iterate Order.succ (HAdd.hAdd (toZ i0 i).toNat m) i0)
        h : Not (LE.le (toZ i0 i) (toZ i0 j))
        hm0 : Not (Eq m 0)
        h_succ_le : LE.le (Nat.iterate Order.succ (HAdd.hAdd (toZ i0 i).toNat 1) i0) j
        ⊢ LE.le (Order.succ i) j
      -/
      rwa [Function.iterate_succ', Function.comp_apply, iterate_succ_toZ i hi] at h_succ_le
      /-
        🎉 no goals
      -/
      /-
        case neg.refine_2
        ι : Type u_1
        inst✝³ : LinearOrder ι
        inst✝² : SuccOrder ι
        inst✝¹ : IsSuccArchimedean ι
        inst✝ : PredOrder ι
        i0 i j : ι
        h_le : LE.le i j
        hi_max : Not (IsMax i)
        hj_min : Not (IsMin j)
        hi : LE.le i0 i
        hj : LE.le i0 j
        m : Nat := Nat.find ⋯
        hm : Eq (Nat.iterate Order.succ m i) j
        hj_eq : Eq j (Nat.iterate Order.succ (HAdd.hAdd (toZ i0 i).toNat m) i0)
        h : Not (LE.le (toZ i0 i) (toZ i0 j))
        hm0 : Not (Eq m 0)
        ⊢ LE.le (toZ i0 j) (toZ i0 i)
      -/
    · exact le_of_not_le h
      /-
        🎉 no goals
      -/
    /-
      case neg.inl.inr
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 i j : ι
      h_le : LE.le i j
      hi_max : Not (IsMax i)
      hj_min : Not (IsMin j)
      hi : LE.le i0 i
      hj : LT.lt j i0
      ⊢ LE.le (toZ i0 i) (toZ i0 j)
    -/
  · exact absurd h_le (not_le.mpr (hj.trans_le hi))
    /-
      🎉 no goals
    -/
    /-
      case neg.inr.inl
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 i j : ι
      h_le : LE.le i j
      hi_max : Not (IsMax i)
      hj_min : Not (IsMin j)
      hi : LT.lt i i0
      hj : LE.le i0 j
      ⊢ LE.le (toZ i0 i) (toZ i0 j)
    -/
  · exact (toZ_neg hi).le.trans (toZ_nonneg hj)
    /-
      🎉 no goals
    -/
    /-
      case neg.inr.inr
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 i j : ι
      h_le : LE.le i j
      hi_max : Not (IsMax i)
      hj_min : Not (IsMin j)
      hi : LT.lt i i0
      hj : LT.lt j i0
      ⊢ LE.le (toZ i0 i) (toZ i0 j)
    -/
  · let m := Nat.find (exists_pred_iterate_of_le (α := ι) h_le)
    /-
      case neg.inr.inr
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 i j : ι
      h_le : LE.le i j
      hi_max : Not (IsMax i)
      hj_min : Not (IsMin j)
      hi : LT.lt i i0
      hj : LT.lt j i0
      m : Nat := Nat.find ⋯
      ⊢ LE.le (toZ i0 i) (toZ i0 j)
    -/
    have hm : pred^[m] j = i := Nat.find_spec (exists_pred_iterate_of_le (α := ι) h_le)
    have hj_eq : i = pred^[(-toZ i0 j).toNat + m] i0 := by
      rw [← hm, add_comm]
      nth_rw 1 [← iterate_pred_toZ j hj]
      rw [Function.iterate_add]
      rfl
    /-
      case neg.inr.inr
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 i j : ι
      h_le : LE.le i j
      hi_max : Not (IsMax i)
      hj_min : Not (IsMin j)
      hi : LT.lt i i0
      hj : LT.lt j i0
      m : Nat := Nat.find ⋯
      hm : Eq (Nat.iterate Order.pred m j) i
      hj_eq : Eq i (Nat.iterate Order.pred (HAdd.hAdd (Neg.neg (toZ i0 j)).toNat m)  …
      ⊢ LE.le (toZ i0 i) (toZ i0 j)
    -/
    by_contra h
    /-
      case neg.inr.inr
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 i j : ι
      h_le : LE.le i j
      hi_max : Not (IsMax i)
      hj_min : Not (IsMin j)
      hi : LT.lt i i0
      hj : LT.lt j i0
      m : Nat := Nat.find ⋯
      hm : Eq (Nat.iterate Order.pred m j) i
      hj_eq : Eq i (Nat.iterate Order.pred (HAdd.hAdd (Neg.neg (toZ i0 j)).toNat m)  …
      h : Not (LE.le (toZ i0 i) (toZ i0 j))
      ⊢ False
    -/
    by_cases hm0 : m = 0
      /-
        case pos
        ι : Type u_1
        inst✝³ : LinearOrder ι
        inst✝² : SuccOrder ι
        inst✝¹ : IsSuccArchimedean ι
        inst✝ : PredOrder ι
        i0 i j : ι
        h_le : LE.le i j
        hi_max : Not (IsMax i)
        hj_min : Not (IsMin j)
        hi : LT.lt i i0
        hj : LT.lt j i0
        m : Nat := Nat.find ⋯
        hm : Eq (Nat.iterate Order.pred m j) i
        hj_eq : Eq i (Nat.iterate Order.pred (HAdd.hAdd (Neg.neg (toZ i0 j)).toNat m)  …
        h : Not (LE.le (toZ i0 i) (toZ i0 j))
        hm0 : Eq m 0
        ⊢ False
      -/
    · rw [hm0, Function.iterate_zero, id] at hm
      /-
        case pos
        ι : Type u_1
        inst✝³ : LinearOrder ι
        inst✝² : SuccOrder ι
        inst✝¹ : IsSuccArchimedean ι
        inst✝ : PredOrder ι
        i0 i j : ι
        h_le : LE.le i j
        hi_max : Not (IsMax i)
        hj_min : Not (IsMin j)
        hi : LT.lt i i0
        hj : LT.lt j i0
        m : Nat := Nat.find ⋯
        hm : Eq j i
        hj_eq : Eq i (Nat.iterate Order.pred (HAdd.hAdd (Neg.neg (toZ i0 j)).toNat m)  …
        h : Not (LE.le (toZ i0 i) (toZ i0 j))
        hm0 : Eq m 0
        ⊢ False
      -/
      rw [hm] at h
      /-
        case pos
        ι : Type u_1
        inst✝³ : LinearOrder ι
        inst✝² : SuccOrder ι
        inst✝¹ : IsSuccArchimedean ι
        inst✝ : PredOrder ι
        i0 i j : ι
        h_le : LE.le i j
        hi_max : Not (IsMax i)
        hj_min : Not (IsMin j)
        hi : LT.lt i i0
        hj : LT.lt j i0
        m : Nat := Nat.find ⋯
        hm : Eq j i
        hj_eq : Eq i (Nat.iterate Order.pred (HAdd.hAdd (Neg.neg (toZ i0 j)).toNat m)  …
        h : Not (LE.le (toZ i0 i) (toZ i0 i))
        hm0 : Eq m 0
        ⊢ False
      -/
      exact h (le_of_eq rfl)
      /-
        🎉 no goals
      -/
    /-
      case neg
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 i j : ι
      h_le : LE.le i j
      hi_max : Not (IsMax i)
      hj_min : Not (IsMin j)
      hi : LT.lt i i0
      hj : LT.lt j i0
      m : Nat := Nat.find ⋯
      hm : Eq (Nat.iterate Order.pred m j) i
      hj_eq : Eq i (Nat.iterate Order.pred (HAdd.hAdd (Neg.neg (toZ i0 j)).toNat m)  …
      h : Not (LE.le (toZ i0 i) (toZ i0 j))
      hm0 : Not (Eq m 0)
      ⊢ False
    -/
    refine hj_min (min_of_le_pred ?_)
    /-
      case neg
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : IsSuccArchimedean ι
      inst✝ : PredOrder ι
      i0 i j : ι
      h_le : LE.le i j
      hi_max : Not (IsMax i)
      hj_min : Not (IsMin j)
      hi : LT.lt i i0
      hj : LT.lt j i0
      m : Nat := Nat.find ⋯
      hm : Eq (Nat.iterate Order.pred m j) i
      hj_eq : Eq i (Nat.iterate Order.pred (HAdd.hAdd (Neg.neg (toZ i0 j)).toNat m)  …
      h : Not (LE.le (toZ i0 i) (toZ i0 j))
      hm0 : Not (Eq m 0)
      ⊢ LE.le j (Order.pred j)
    -/
    refine (@le_of_toZ_le _ _ _ _ _ i0 j i ?_).trans ?_
      /-
        case neg.refine_1
        ι : Type u_1
        inst✝³ : LinearOrder ι
        inst✝² : SuccOrder ι
        inst✝¹ : IsSuccArchimedean ι
        inst✝ : PredOrder ι
        i0 i j : ι
        h_le : LE.le i j
        hi_max : Not (IsMax i)
        hj_min : Not (IsMin j)
        hi : LT.lt i i0
        hj : LT.lt j i0
        m : Nat := Nat.find ⋯
        hm : Eq (Nat.iterate Order.pred m j) i
        hj_eq : Eq i (Nat.iterate Order.pred (HAdd.hAdd (Neg.neg (toZ i0 j)).toNat m)  …
        h : Not (LE.le (toZ i0 i) (toZ i0 j))
        hm0 : Not (Eq m 0)
        ⊢ LE.le (toZ i0 j) (toZ i0 i)
      -/
    · exact le_of_not_le h
      /-
        🎉 no goals
      -/
    · have h_le_pred : i ≤ pred^[(-toZ i0 j).toNat + 1] i0 := by
        rw [hj_eq]
        refine Monotone.antitone_iterate_of_map_le pred_mono (pred_le i0) (add_le_add_left ?_ _)
        exact Nat.one_le_iff_ne_zero.mpr hm0
      /-
        case neg.refine_2
        ι : Type u_1
        inst✝³ : LinearOrder ι
        inst✝² : SuccOrder ι
        inst✝¹ : IsSuccArchimedean ι
        inst✝ : PredOrder ι
        i0 i j : ι
        h_le : LE.le i j
        hi_max : Not (IsMax i)
        hj_min : Not (IsMin j)
        hi : LT.lt i i0
        hj : LT.lt j i0
        m : Nat := Nat.find ⋯
        hm : Eq (Nat.iterate Order.pred m j) i
        hj_eq : Eq i (Nat.iterate Order.pred (HAdd.hAdd (Neg.neg (toZ i0 j)).toNat m)  …
        h : Not (LE.le (toZ i0 i) (toZ i0 j))
        hm0 : Not (Eq m 0)
        h_le_pred : LE.le i (Nat.iterate Order.pred (HAdd.hAdd (Neg.neg (toZ i0 j)).to …
        ⊢ LE.le i (Order.pred j)
      -/
      rwa [Function.iterate_succ', Function.comp_apply, iterate_pred_toZ j hj] at h_le_pred
      /-
        🎉 no goals
      -/


theorem toZ_le_iff (i j : ι) : toZ i0 i ≤ toZ i0 j ↔ i ≤ j :=
  ⟨le_of_toZ_le, toZ_mono⟩


theorem toZ_iterate_succ [NoMaxOrder ι] (n : ℕ) : toZ i0 (succ^[n] i0) = n :=
  toZ_iterate_succ_of_not_isMax n (not_isMax _)


theorem toZ_iterate_pred [NoMinOrder ι] (n : ℕ) : toZ i0 (pred^[n] i0) = -n :=
  toZ_iterate_pred_of_not_isMin n (not_isMin _)


theorem injective_toZ : Function.Injective (toZ i0) :=
  fun _ _ h ↦ le_antisymm (le_of_toZ_le h.le) (le_of_toZ_le h.symm.le)


/-- `toZ` defines an `OrderIso` between `ι` and its range. -/
noncomputable def orderIsoRangeToZOfLinearSuccPredArch [hι : Nonempty ι] :
    ι ≃o Set.range (toZ hι.some) where
  toEquiv := Equiv.ofInjective _ injective_toZ
                     /-
                       ι : Type u_1
                       inst✝³ : LinearOrder ι
                       inst✝² : SuccOrder ι
                       inst✝¹ : PredOrder ι
                       inst✝ : IsSuccArchimedean ι
                       hι : Nonempty ι
                       ⊢ ∀ {a b : ι}, Iff (LE.le ((Equiv.ofInjective (toZ hι.some) ⋯) a) ((Equiv.ofIn …
                     -/
  map_rel_iff' := by intro i j; exact toZ_le_iff i j
                                /-
                                  🎉 no goals
                                -/


instance (priority := 100) countable_of_linear_succ_pred_arch : Countable ι := by
  /-
    ι : Type u_1
    inst✝³ : LinearOrder ι
    inst✝² : SuccOrder ι
    inst✝¹ : PredOrder ι
    inst✝ : IsSuccArchimedean ι
    ⊢ Countable ι
  -/
  cases' isEmpty_or_nonempty ι with _ hι
    /-
      case inl
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : PredOrder ι
      inst✝ : IsSuccArchimedean ι
      h✝ : IsEmpty ι
      ⊢ Countable ι
    -/
  · infer_instance
    /-
      🎉 no goals
    -/
    /-
      case inr
      ι : Type u_1
      inst✝³ : LinearOrder ι
      inst✝² : SuccOrder ι
      inst✝¹ : PredOrder ι
      inst✝ : IsSuccArchimedean ι
      hι : Nonempty ι
      ⊢ Countable ι
    -/
  · exact Countable.of_equiv _ orderIsoRangeToZOfLinearSuccPredArch.symm.toEquiv
    /-
      🎉 no goals
    -/


/-- If the order has neither bot nor top, `toZ` defines an `OrderIso` between `ι` and `ℤ`. -/
noncomputable def orderIsoIntOfLinearSuccPredArch [NoMaxOrder ι] [NoMinOrder ι] [hι : Nonempty ι] :
    ι ≃o ℤ where
  toFun := toZ hι.some
  invFun n := if 0 ≤ n then succ^[n.toNat] hι.some else pred^[(-n).toNat] hι.some
  left_inv i := by
    /-
      ι : Type u_1
      inst✝⁵ : LinearOrder ι
      inst✝⁴ : SuccOrder ι
      inst✝³ : PredOrder ι
      inst✝² : IsSuccArchimedean ι
      inst✝¹ : NoMaxOrder ι
      inst✝ : NoMinOrder ι
      hι : Nonempty ι
      i : ι
      ⊢ Eq ((fun n => ite (LE.le 0 n) (Nat.iterate Order.succ n.toNat hι.some) (Nat. …
    -/
    rcases le_or_lt hι.some i with hi | hi
      /-
        case inl
        ι : Type u_1
        inst✝⁵ : LinearOrder ι
        inst✝⁴ : SuccOrder ι
        inst✝³ : PredOrder ι
        inst✝² : IsSuccArchimedean ι
        inst✝¹ : NoMaxOrder ι
        inst✝ : NoMinOrder ι
        hι : Nonempty ι
        i : ι
        hi : LE.le hι.some i
        ⊢ Eq ((fun n => ite (LE.le 0 n) (Nat.iterate Order.succ n.toNat hι.some) (Nat. …
      -/
    · have h_nonneg : 0 ≤ toZ hι.some i := toZ_nonneg hi
      /-
        case inl
        ι : Type u_1
        inst✝⁵ : LinearOrder ι
        inst✝⁴ : SuccOrder ι
        inst✝³ : PredOrder ι
        inst✝² : IsSuccArchimedean ι
        inst✝¹ : NoMaxOrder ι
        inst✝ : NoMinOrder ι
        hι : Nonempty ι
        i : ι
        hi : LE.le hι.some i
        h_nonneg : LE.le 0 (toZ hι.some i)
        ⊢ Eq ((fun n => ite (LE.le 0 n) (Nat.iterate Order.succ n.toNat hι.some) (Nat. …
      -/
      simp_rw [if_pos h_nonneg]
      /-
        case inl
        ι : Type u_1
        inst✝⁵ : LinearOrder ι
        inst✝⁴ : SuccOrder ι
        inst✝³ : PredOrder ι
        inst✝² : IsSuccArchimedean ι
        inst✝¹ : NoMaxOrder ι
        inst✝ : NoMinOrder ι
        hι : Nonempty ι
        i : ι
        hi : LE.le hι.some i
        h_nonneg : LE.le 0 (toZ hι.some i)
        ⊢ Eq (Nat.iterate Order.succ (toZ hι.some i).toNat hι.some) i
      -/
      exact iterate_succ_toZ i hi
      /-
        🎉 no goals
      -/
      /-
        case inr
        ι : Type u_1
        inst✝⁵ : LinearOrder ι
        inst✝⁴ : SuccOrder ι
        inst✝³ : PredOrder ι
        inst✝² : IsSuccArchimedean ι
        inst✝¹ : NoMaxOrder ι
        inst✝ : NoMinOrder ι
        hι : Nonempty ι
        i : ι
        hi : LT.lt i hι.some
        ⊢ Eq ((fun n => ite (LE.le 0 n) (Nat.iterate Order.succ n.toNat hι.some) (Nat. …
      -/
    · have h_neg : toZ hι.some i < 0 := toZ_neg hi
      /-
        case inr
        ι : Type u_1
        inst✝⁵ : LinearOrder ι
        inst✝⁴ : SuccOrder ι
        inst✝³ : PredOrder ι
        inst✝² : IsSuccArchimedean ι
        inst✝¹ : NoMaxOrder ι
        inst✝ : NoMinOrder ι
        hι : Nonempty ι
        i : ι
        hi : LT.lt i hι.some
        h_neg : LT.lt (toZ hι.some i) 0
        ⊢ Eq ((fun n => ite (LE.le 0 n) (Nat.iterate Order.succ n.toNat hι.some) (Nat. …
      -/
      simp_rw [if_neg (not_le.mpr h_neg)]
      /-
        case inr
        ι : Type u_1
        inst✝⁵ : LinearOrder ι
        inst✝⁴ : SuccOrder ι
        inst✝³ : PredOrder ι
        inst✝² : IsSuccArchimedean ι
        inst✝¹ : NoMaxOrder ι
        inst✝ : NoMinOrder ι
        hι : Nonempty ι
        i : ι
        hi : LT.lt i hι.some
        h_neg : LT.lt (toZ hι.some i) 0
        ⊢ Eq (Nat.iterate Order.pred (Neg.neg (toZ hι.some i)).toNat hι.some) i
      -/
      exact iterate_pred_toZ i hi
      /-
        🎉 no goals
      -/
  right_inv n := by
    /-
      ι : Type u_1
      inst✝⁵ : LinearOrder ι
      inst✝⁴ : SuccOrder ι
      inst✝³ : PredOrder ι
      inst✝² : IsSuccArchimedean ι
      inst✝¹ : NoMaxOrder ι
      inst✝ : NoMinOrder ι
      hι : Nonempty ι
      n : Int
      ⊢ Eq (toZ hι.some ((fun n => ite (LE.le 0 n) (Nat.iterate Order.succ n.toNat h …
    -/
    rcases le_or_lt 0 n with hn | hn
      /-
        case inl
        ι : Type u_1
        inst✝⁵ : LinearOrder ι
        inst✝⁴ : SuccOrder ι
        inst✝³ : PredOrder ι
        inst✝² : IsSuccArchimedean ι
        inst✝¹ : NoMaxOrder ι
        inst✝ : NoMinOrder ι
        hι : Nonempty ι
        n : Int
        hn : LE.le 0 n
        ⊢ Eq (toZ hι.some ((fun n => ite (LE.le 0 n) (Nat.iterate Order.succ n.toNat h …
      -/
    · simp_rw [if_pos hn]
      /-
        case inl
        ι : Type u_1
        inst✝⁵ : LinearOrder ι
        inst✝⁴ : SuccOrder ι
        inst✝³ : PredOrder ι
        inst✝² : IsSuccArchimedean ι
        inst✝¹ : NoMaxOrder ι
        inst✝ : NoMinOrder ι
        hι : Nonempty ι
        n : Int
        hn : LE.le 0 n
        ⊢ Eq (toZ hι.some (Nat.iterate Order.succ n.toNat hι.some)) n
      -/
      rw [toZ_iterate_succ]
      /-
        case inl
        ι : Type u_1
        inst✝⁵ : LinearOrder ι
        inst✝⁴ : SuccOrder ι
        inst✝³ : PredOrder ι
        inst✝² : IsSuccArchimedean ι
        inst✝¹ : NoMaxOrder ι
        inst✝ : NoMinOrder ι
        hι : Nonempty ι
        n : Int
        hn : LE.le 0 n
        ⊢ Eq (↑n.toNat) n
      -/
      exact Int.toNat_of_nonneg hn
      /-
        🎉 no goals
      -/
      /-
        case inr
        ι : Type u_1
        inst✝⁵ : LinearOrder ι
        inst✝⁴ : SuccOrder ι
        inst✝³ : PredOrder ι
        inst✝² : IsSuccArchimedean ι
        inst✝¹ : NoMaxOrder ι
        inst✝ : NoMinOrder ι
        hι : Nonempty ι
        n : Int
        hn : LT.lt n 0
        ⊢ Eq (toZ hι.some ((fun n => ite (LE.le 0 n) (Nat.iterate Order.succ n.toNat h …
      -/
    · simp_rw [if_neg (not_le.mpr hn)]
      /-
        case inr
        ι : Type u_1
        inst✝⁵ : LinearOrder ι
        inst✝⁴ : SuccOrder ι
        inst✝³ : PredOrder ι
        inst✝² : IsSuccArchimedean ι
        inst✝¹ : NoMaxOrder ι
        inst✝ : NoMinOrder ι
        hι : Nonempty ι
        n : Int
        hn : LT.lt n 0
        ⊢ Eq (toZ hι.some (Nat.iterate Order.pred (Neg.neg n).toNat hι.some)) n
      -/
      rw [toZ_iterate_pred]
      /-
        case inr
        ι : Type u_1
        inst✝⁵ : LinearOrder ι
        inst✝⁴ : SuccOrder ι
        inst✝³ : PredOrder ι
        inst✝² : IsSuccArchimedean ι
        inst✝¹ : NoMaxOrder ι
        inst✝ : NoMinOrder ι
        hι : Nonempty ι
        n : Int
        hn : LT.lt n 0
        ⊢ Eq (Neg.neg ↑(Neg.neg n).toNat) n
      -/
      simp only [hn.le, Int.toNat_of_nonneg, Int.neg_nonneg_of_nonpos, Int.neg_neg]
      /-
        🎉 no goals
      -/
                     /-
                       ι : Type u_1
                       inst✝⁵ : LinearOrder ι
                       inst✝⁴ : SuccOrder ι
                       inst✝³ : PredOrder ι
                       inst✝² : IsSuccArchimedean ι
                       inst✝¹ : NoMaxOrder ι
                       inst✝ : NoMinOrder ι
                       hι : Nonempty ι
                       ⊢ ∀ {a b : ι}, Iff (LE.le ({ toFun := toZ hι.some, invFun := fun n => ite (LE. …
                     -/
  map_rel_iff' := by intro i j; exact toZ_le_iff i j
                                /-
                                  🎉 no goals
                                -/


/-- If the order has a bot but no top, `toZ` defines an `OrderIso` between `ι` and `ℕ`. -/
def orderIsoNatOfLinearSuccPredArch [NoMaxOrder ι] [OrderBot ι] : ι ≃o ℕ where
  toFun i := (toZ ⊥ i).toNat
  invFun n := succ^[n] ⊥
  left_inv i := by
    /-
      ι : Type u_1
      inst✝⁵ : LinearOrder ι
      inst✝⁴ : SuccOrder ι
      inst✝³ : PredOrder ι
      inst✝² : IsSuccArchimedean ι
      inst✝¹ : NoMaxOrder ι
      inst✝ : OrderBot ι
      i : ι
      ⊢ Eq ((fun n => Nat.iterate Order.succ n Bot.bot) ((fun i => (toZ Bot.bot i).t …
    -/
    dsimp only
    /-
      ι : Type u_1
      inst✝⁵ : LinearOrder ι
      inst✝⁴ : SuccOrder ι
      inst✝³ : PredOrder ι
      inst✝² : IsSuccArchimedean ι
      inst✝¹ : NoMaxOrder ι
      inst✝ : OrderBot ι
      i : ι
      ⊢ Eq (Nat.iterate Order.succ (toZ Bot.bot i).toNat Bot.bot) i
    -/
    exact iterate_succ_toZ i bot_le
    /-
      🎉 no goals
    -/
  right_inv n := by
    /-
      ι : Type u_1
      inst✝⁵ : LinearOrder ι
      inst✝⁴ : SuccOrder ι
      inst✝³ : PredOrder ι
      inst✝² : IsSuccArchimedean ι
      inst✝¹ : NoMaxOrder ι
      inst✝ : OrderBot ι
      n : Nat
      ⊢ Eq ((fun i => (toZ Bot.bot i).toNat) ((fun n => Nat.iterate Order.succ n Bot …
    -/
    dsimp only
    /-
      ι : Type u_1
      inst✝⁵ : LinearOrder ι
      inst✝⁴ : SuccOrder ι
      inst✝³ : PredOrder ι
      inst✝² : IsSuccArchimedean ι
      inst✝¹ : NoMaxOrder ι
      inst✝ : OrderBot ι
      n : Nat
      ⊢ Eq (toZ Bot.bot (Nat.iterate Order.succ n Bot.bot)).toNat n
    -/
    rw [toZ_iterate_succ]
    /-
      ι : Type u_1
      inst✝⁵ : LinearOrder ι
      inst✝⁴ : SuccOrder ι
      inst✝³ : PredOrder ι
      inst✝² : IsSuccArchimedean ι
      inst✝¹ : NoMaxOrder ι
      inst✝ : OrderBot ι
      n : Nat
      ⊢ Eq (↑n).toNat n
    -/
    exact Int.toNat_natCast n
    /-
      🎉 no goals
    -/
  map_rel_iff' := by
    /-
      ι : Type u_1
      inst✝⁵ : LinearOrder ι
      inst✝⁴ : SuccOrder ι
      inst✝³ : PredOrder ι
      inst✝² : IsSuccArchimedean ι
      inst✝¹ : NoMaxOrder ι
      inst✝ : OrderBot ι
      ⊢ ∀ {a b : ι}, Iff (LE.le ({ toFun := fun i => (toZ Bot.bot i).toNat, invFun : …
    -/
    intro i j
    /-
      ι : Type u_1
      inst✝⁵ : LinearOrder ι
      inst✝⁴ : SuccOrder ι
      inst✝³ : PredOrder ι
      inst✝² : IsSuccArchimedean ι
      inst✝¹ : NoMaxOrder ι
      inst✝ : OrderBot ι
      i j : ι
      ⊢ Iff (LE.le ({ toFun := fun i => (toZ Bot.bot i).toNat, invFun := fun n => Na …
    -/
    simp only [Equiv.coe_fn_mk, Int.toNat_le]
    /-
      ι : Type u_1
      inst✝⁵ : LinearOrder ι
      inst✝⁴ : SuccOrder ι
      inst✝³ : PredOrder ι
      inst✝² : IsSuccArchimedean ι
      inst✝¹ : NoMaxOrder ι
      inst✝ : OrderBot ι
      i j : ι
      ⊢ Iff (LE.le (toZ Bot.bot i) ↑(toZ Bot.bot j).toNat) (LE.le i j)
    -/
    rw [← @toZ_le_iff ι _ _ _ _ ⊥, Int.toNat_of_nonneg (toZ_nonneg bot_le)]
    /-
      🎉 no goals
    -/


/-- If the order has both a bot and a top, `toZ` gives an `OrderIso` between `ι` and
`Finset.range n` for some `n`. -/
def orderIsoRangeOfLinearSuccPredArch [OrderBot ι] [OrderTop ι] :
    ι ≃o Finset.range ((toZ ⊥ (⊤ : ι)).toNat + 1) where
  toFun i :=
    ⟨(toZ ⊥ i).toNat,
      Finset.mem_range_succ_iff.mpr (Int.toNat_le_toNat ((toZ_le_iff _ _).mpr le_top))⟩
  invFun n := succ^[n] ⊥
  left_inv i := iterate_succ_toZ i bot_le
  right_inv n := by
    /-
      ι : Type u_1
      inst✝⁵ : LinearOrder ι
      inst✝⁴ : SuccOrder ι
      inst✝³ : PredOrder ι
      inst✝² : IsSuccArchimedean ι
      inst✝¹ : OrderBot ι
      inst✝ : OrderTop ι
      n : Subtype fun x => Membership.mem (Finset.range (HAdd.hAdd (toZ Bot.bot Top. …
      ⊢ Eq ((fun i => ⟨(toZ Bot.bot i).toNat, ⋯⟩) ((fun n => Nat.iterate Order.succ  …
    -/
    ext1
    /-
      case a
      ι : Type u_1
      inst✝⁵ : LinearOrder ι
      inst✝⁴ : SuccOrder ι
      inst✝³ : PredOrder ι
      inst✝² : IsSuccArchimedean ι
      inst✝¹ : OrderBot ι
      inst✝ : OrderTop ι
      n : Subtype fun x => Membership.mem (Finset.range (HAdd.hAdd (toZ Bot.bot Top. …
      ⊢ Eq ↑((fun i => ⟨(toZ Bot.bot i).toNat, ⋯⟩) ((fun n => Nat.iterate Order.succ …
    -/
    simp only [Subtype.coe_mk]
    /-
      case a
      ι : Type u_1
      inst✝⁵ : LinearOrder ι
      inst✝⁴ : SuccOrder ι
      inst✝³ : PredOrder ι
      inst✝² : IsSuccArchimedean ι
      inst✝¹ : OrderBot ι
      inst✝ : OrderTop ι
      n : Subtype fun x => Membership.mem (Finset.range (HAdd.hAdd (toZ Bot.bot Top. …
      ⊢ Eq (toZ Bot.bot (Nat.iterate Order.succ (↑n) Bot.bot)).toNat ↑n
    -/
    refine le_antisymm ?_ ?_
      /-
        case a.refine_1
        ι : Type u_1
        inst✝⁵ : LinearOrder ι
        inst✝⁴ : SuccOrder ι
        inst✝³ : PredOrder ι
        inst✝² : IsSuccArchimedean ι
        inst✝¹ : OrderBot ι
        inst✝ : OrderTop ι
        n : Subtype fun x => Membership.mem (Finset.range (HAdd.hAdd (toZ Bot.bot Top. …
        ⊢ LE.le (toZ Bot.bot (Nat.iterate Order.succ (↑n) Bot.bot)).toNat ↑n
      -/
    · rw [Int.toNat_le]
      /-
        case a.refine_1
        ι : Type u_1
        inst✝⁵ : LinearOrder ι
        inst✝⁴ : SuccOrder ι
        inst✝³ : PredOrder ι
        inst✝² : IsSuccArchimedean ι
        inst✝¹ : OrderBot ι
        inst✝ : OrderTop ι
        n : Subtype fun x => Membership.mem (Finset.range (HAdd.hAdd (toZ Bot.bot Top. …
        ⊢ LE.le (toZ Bot.bot (Nat.iterate Order.succ (↑n) Bot.bot)) ↑↑n
      -/
      exact toZ_iterate_succ_le _
      /-
        🎉 no goals
      -/
    /-
      case a.refine_2
      ι : Type u_1
      inst✝⁵ : LinearOrder ι
      inst✝⁴ : SuccOrder ι
      inst✝³ : PredOrder ι
      inst✝² : IsSuccArchimedean ι
      inst✝¹ : OrderBot ι
      inst✝ : OrderTop ι
      n : Subtype fun x => Membership.mem (Finset.range (HAdd.hAdd (toZ Bot.bot Top. …
      ⊢ LE.le (↑n) (toZ Bot.bot (Nat.iterate Order.succ (↑n) Bot.bot)).toNat
    -/
    by_cases hn_max : IsMax (succ^[↑n] (⊥ : ι))
      /-
        case pos
        ι : Type u_1
        inst✝⁵ : LinearOrder ι
        inst✝⁴ : SuccOrder ι
        inst✝³ : PredOrder ι
        inst✝² : IsSuccArchimedean ι
        inst✝¹ : OrderBot ι
        inst✝ : OrderTop ι
        n : Subtype fun x => Membership.mem (Finset.range (HAdd.hAdd (toZ Bot.bot Top. …
        hn_max : IsMax (Nat.iterate Order.succ (↑n) Bot.bot)
        ⊢ LE.le (↑n) (toZ Bot.bot (Nat.iterate Order.succ (↑n) Bot.bot)).toNat
      -/
    · rw [← isTop_iff_isMax, isTop_iff_eq_top] at hn_max
      /-
        case pos
        ι : Type u_1
        inst✝⁵ : LinearOrder ι
        inst✝⁴ : SuccOrder ι
        inst✝³ : PredOrder ι
        inst✝² : IsSuccArchimedean ι
        inst✝¹ : OrderBot ι
        inst✝ : OrderTop ι
        n : Subtype fun x => Membership.mem (Finset.range (HAdd.hAdd (toZ Bot.bot Top. …
        hn_max : Eq (Nat.iterate Order.succ (↑n) Bot.bot) Top.top
        ⊢ LE.le (↑n) (toZ Bot.bot (Nat.iterate Order.succ (↑n) Bot.bot)).toNat
      -/
      rw [hn_max]
      /-
        case pos
        ι : Type u_1
        inst✝⁵ : LinearOrder ι
        inst✝⁴ : SuccOrder ι
        inst✝³ : PredOrder ι
        inst✝² : IsSuccArchimedean ι
        inst✝¹ : OrderBot ι
        inst✝ : OrderTop ι
        n : Subtype fun x => Membership.mem (Finset.range (HAdd.hAdd (toZ Bot.bot Top. …
        hn_max : Eq (Nat.iterate Order.succ (↑n) Bot.bot) Top.top
        ⊢ LE.le (↑n) (toZ Bot.bot Top.top).toNat
      -/
      exact Nat.lt_succ_iff.mp (Finset.mem_range.mp n.prop)
      /-
        🎉 no goals
      -/
      /-
        case neg
        ι : Type u_1
        inst✝⁵ : LinearOrder ι
        inst✝⁴ : SuccOrder ι
        inst✝³ : PredOrder ι
        inst✝² : IsSuccArchimedean ι
        inst✝¹ : OrderBot ι
        inst✝ : OrderTop ι
        n : Subtype fun x => Membership.mem (Finset.range (HAdd.hAdd (toZ Bot.bot Top. …
        hn_max : Not (IsMax (Nat.iterate Order.succ (↑n) Bot.bot))
        ⊢ LE.le (↑n) (toZ Bot.bot (Nat.iterate Order.succ (↑n) Bot.bot)).toNat
      -/
    · rw [toZ_iterate_succ_of_not_isMax _ hn_max]
      /-
        case neg
        ι : Type u_1
        inst✝⁵ : LinearOrder ι
        inst✝⁴ : SuccOrder ι
        inst✝³ : PredOrder ι
        inst✝² : IsSuccArchimedean ι
        inst✝¹ : OrderBot ι
        inst✝ : OrderTop ι
        n : Subtype fun x => Membership.mem (Finset.range (HAdd.hAdd (toZ Bot.bot Top. …
        hn_max : Not (IsMax (Nat.iterate Order.succ (↑n) Bot.bot))
        ⊢ LE.le (↑n) (↑↑n).toNat
      -/
      simp only [Int.toNat_natCast, le_refl]
      /-
        🎉 no goals
      -/
  map_rel_iff' := by
    /-
      ι : Type u_1
      inst✝⁵ : LinearOrder ι
      inst✝⁴ : SuccOrder ι
      inst✝³ : PredOrder ι
      inst✝² : IsSuccArchimedean ι
      inst✝¹ : OrderBot ι
      inst✝ : OrderTop ι
      ⊢ ∀ {a b : ι}, Iff (LE.le ({ toFun := fun i => ⟨(toZ Bot.bot i).toNat, ⋯⟩, inv …
    -/
    intro i j
    /-
      ι : Type u_1
      inst✝⁵ : LinearOrder ι
      inst✝⁴ : SuccOrder ι
      inst✝³ : PredOrder ι
      inst✝² : IsSuccArchimedean ι
      inst✝¹ : OrderBot ι
      inst✝ : OrderTop ι
      i j : ι
      ⊢ Iff (LE.le ({ toFun := fun i => ⟨(toZ Bot.bot i).toNat, ⋯⟩, invFun := fun n  …
    -/
    simp only [Equiv.coe_fn_mk, Subtype.mk_le_mk, Int.toNat_le]
    /-
      ι : Type u_1
      inst✝⁵ : LinearOrder ι
      inst✝⁴ : SuccOrder ι
      inst✝³ : PredOrder ι
      inst✝² : IsSuccArchimedean ι
      inst✝¹ : OrderBot ι
      inst✝ : OrderTop ι
      i j : ι
      ⊢ Iff (LE.le (toZ Bot.bot i) ↑(toZ Bot.bot j).toNat) (LE.le i j)
    -/
    rw [← @toZ_le_iff ι _ _ _ _ ⊥, Int.toNat_of_nonneg (toZ_nonneg bot_le)]
    /-
      🎉 no goals
    -/


